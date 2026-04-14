import os
import signal

from vllm.config import ParallelConfig
from vllm.logger import logger
from vllm.transformers_utils.config import \
    maybe_register_config_serialize_by_value
from vllm.utils.system_utils import decorate_logs, set_process_title
from vllm.v1.engine.core import DPEngineCoreProc, EngineCoreProc

from vllm_ascend import envs
from vllm_ascend.utils import vllm_version_is
from vllm.config import VllmConfig
from vllm.v1.kv_cache_interface import KVCacheConfig

import time
import vllm
from vllm.v1.core.kv_cache_utils import generate_scheduler_kv_cache_config
from vllm_ascend.patch.platform.patch_kv_cache_coordinator import USE_MULTI_GROUPS_KV_CACHE
from vllm_ascend.patch.platform.patch_kv_cache_utils import get_kv_cache_configs_with_multi_groups as get_kv_cache_configs


def run_engine_core(*args, dp_rank: int = 0, local_dp_rank: int = 0, **kwargs):
    """Launch EngineCore busy loop in background process."""

    if os.getenv("SHM_BARRIER", "true").lower() in ("true", "1"):
        from vllm.distributed.device_communicators.shm_broadcast import \
            MessageQueue  # noqa

    # Signal handler used for graceful termination.
    # SystemExit exception is only raised once to allow this and worker
    # processes to terminate without error
    shutdown_requested = False

    # Ensure we can serialize transformer config after spawning
    maybe_register_config_serialize_by_value()

    def signal_handler(signum, frame):
        nonlocal shutdown_requested
        if not shutdown_requested:
            shutdown_requested = True
            raise SystemExit()

    # Either SIGTERM or SIGINT will terminate the engine_core
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    engine_core: EngineCoreProc | None = None
    try:
        parallel_config: ParallelConfig = kwargs["vllm_config"].parallel_config
        if parallel_config.data_parallel_size > 1 or dp_rank > 0:
            set_process_title("EngineCore", f"DP{dp_rank}")
            decorate_logs()
            # Set data parallel rank for this engine process.
            parallel_config.data_parallel_rank = dp_rank
            parallel_config.data_parallel_rank_local = local_dp_rank
            if envs.VLLM_ASCEND_BALANCE_SCHEDULING and vllm_version_is(
                    "0.13.0"):
                from vllm_ascend.patch.platform.patch_balance_schedule import \
                    BalanceDPEngineCoreProc
                engine_core = BalanceDPEngineCoreProc(*args, **kwargs)
            else:
                engine_core = DPEngineCoreProc(*args, **kwargs)
        else:
            set_process_title("EngineCore")
            decorate_logs()
            engine_core = EngineCoreProc(*args, **kwargs)

        engine_core.run_busy_loop()

    except SystemExit:
        logger.debug("EngineCore exiting.")
        raise
    except Exception as e:
        if engine_core is None:
            logger.exception("EngineCore failed to start.")
        else:
            logger.exception("EngineCore encountered a fatal error.")
            engine_core._send_engine_dead()
        raise e
    finally:
        if engine_core is not None:
            engine_core.shutdown()

def _initialize_kv_caches_with_multi_groups(
    self, vllm_config: VllmConfig
) -> tuple[int, int, KVCacheConfig]:
    start = time.time()

    # Get all kv cache needed by the model
    kv_cache_specs = self.model_executor.get_kv_cache_specs()

    has_kv_cache = False
    has_kv_cache = any(kv_cache_spec for kv_cache_spec in kv_cache_specs)

    if has_kv_cache:
        if os.environ.get("VLLM_ELASTIC_EP_SCALE_UP_LAUNCH") == "1":
            dp_group = getattr(self, "dp_group", None)
            assert dp_group is not None
            self.available_gpu_memory_for_kv_cache = (
                ParallelConfig.sync_kv_cache_memory_size(dp_group, -1)
            )
            available_gpu_memory = [self.available_gpu_memory_for_kv_cache] * len(
                kv_cache_specs
            )
        else:
            # Profiles the peak memory usage of the model to determine how
            # much memory can be allocated for kv cache.
            available_gpu_memory = self.model_executor.determine_available_memory()
            self.available_gpu_memory_for_kv_cache = available_gpu_memory[0]
    else:
        # Attention free models don't need memory for kv cache
        available_gpu_memory = [0] * len(kv_cache_specs)

    assert len(kv_cache_specs) == len(available_gpu_memory)

    kv_cache_configs = get_kv_cache_configs(
        vllm_config, kv_cache_specs, available_gpu_memory
    )
    scheduler_kv_cache_config = generate_scheduler_kv_cache_config(kv_cache_configs)
    num_gpu_blocks = scheduler_kv_cache_config.num_blocks
    num_cpu_blocks = 0

    # Initialize kv cache and warmup the execution
    self.model_executor.initialize_from_config(kv_cache_configs)

    elapsed = time.time() - start
    logger.info_once(
        "init engine (profile, create kv cache, warmup model) took %.2f seconds",
        elapsed,
        scope="local",
    )
    return num_gpu_blocks, num_cpu_blocks, scheduler_kv_cache_config

if USE_MULTI_GROUPS_KV_CACHE:
    EngineCoreProc._initialize_kv_caches = _initialize_kv_caches_with_multi_groups
    vllm.v1.engine.core.get_kv_cache_configs = get_kv_cache_configs

EngineCoreProc.run_engine_core = run_engine_core
