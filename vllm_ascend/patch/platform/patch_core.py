import os
import time

from vllm.config import ParallelConfig
from vllm.logger import logger

from vllm.config import VllmConfig
from vllm.v1.kv_cache_interface import KVCacheConfig

from vllm.v1.core.kv_cache_utils import generate_scheduler_kv_cache_config
from vllm_ascend.patch.platform.patch_kv_cache_coordinator import USE_MULTI_GROUPS_KV_CACHE
from vllm_ascend.patch.platform.patch_kv_cache_utils import get_kv_cache_configs_with_multi_groups as get_kv_cache_configs
from vllm.v1.engine.core import EngineCoreProc
from vllm.tracing import instrument
import vllm.envs as envs

@instrument(span_name="Prepare model")
def _initialize_kv_caches_with_multi_groups(self, vllm_config: VllmConfig) -> KVCacheConfig:
    start = time.time()

    # Get all kv cache needed by the model
    kv_cache_specs = self.model_executor.get_kv_cache_specs()

    has_kv_cache = False
    has_kv_cache = any(kv_cache_spec for kv_cache_spec in kv_cache_specs)
    if has_kv_cache:
        if envs.VLLM_ELASTIC_EP_SCALE_UP_LAUNCH:
            # NOTE(yongji): should already be set
            # during _eep_scale_up_before_kv_init
            assert self.available_gpu_memory_for_kv_cache > 0
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

    # Track max_model_len before KV cache config to detect auto-fit changes
    max_model_len_before = vllm_config.model_config.max_model_len

    kv_cache_configs = get_kv_cache_configs(
        vllm_config, kv_cache_specs, available_gpu_memory
    )

    # If auto-fit reduced max_model_len, sync the new value to workers.
    # This is needed because workers were spawned before memory profiling
    # and have the original (larger) max_model_len cached.
    max_model_len_after = vllm_config.model_config.max_model_len
    if max_model_len_after != max_model_len_before:
        self.collective_rpc("update_max_model_len", args=(max_model_len_after,))

    scheduler_kv_cache_config = generate_scheduler_kv_cache_config(kv_cache_configs)
    vllm_config.cache_config.num_gpu_blocks = scheduler_kv_cache_config.num_blocks
    kv_cache_groups = scheduler_kv_cache_config.kv_cache_groups
    if kv_cache_groups:
        vllm_config.cache_config.block_size = min(
            g.kv_cache_spec.block_size for g in kv_cache_groups
        )

    vllm_config.validate_block_size()

    # Initialize kv cache and warmup the execution
    self.model_executor.initialize_from_config(kv_cache_configs)

    elapsed = time.time() - start
    logger.info_once(
        "init engine (profile, create kv cache, warmup model) took %.2f seconds",
        elapsed,
        scope="local",
    )
    return scheduler_kv_cache_config

if USE_MULTI_GROUPS_KV_CACHE:
    EngineCoreProc._initialize_kv_caches = _initialize_kv_caches_with_multi_groups