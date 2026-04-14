import os

from vllm.config import ParallelConfig
from vllm.logger import logger
from vllm.v1.engine.core import DPEngineCoreProc, EngineCoreProc

from vllm.config import VllmConfig
from vllm.v1.kv_cache_interface import KVCacheConfig

import time
import vllm
from vllm.v1.core.kv_cache_utils import generate_scheduler_kv_cache_config
from vllm_ascend.patch.platform.patch_kv_cache_coordinator import USE_MULTI_GROUPS_KV_CACHE
from vllm_ascend.patch.platform.patch_kv_cache_utils import get_kv_cache_configs_with_multi_groups as get_kv_cache_configs


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