# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from math import lcm

from vllm.utils.hashing import get_hash_fn_by_name
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine.core import EngineCore


if not hasattr(Scheduler, "_vllm_ascend_orig_init"):
    Scheduler._vllm_ascend_orig_init = Scheduler.__init__  # type: ignore[attr-defined]


if not hasattr(EngineCore, "_vllm_ascend_orig_init"):
    EngineCore._vllm_ascend_orig_init = EngineCore.__init__  # type: ignore[attr-defined]


def _resolve_scheduler_block_size(vllm_config, kv_cache_config, block_size: int) -> int:
    if len(kv_cache_config.kv_cache_groups) <= 1:
        return block_size

    dcp_world_size = vllm_config.parallel_config.decode_context_parallel_size
    pcp_world_size = vllm_config.parallel_config.prefill_context_parallel_size
    if dcp_world_size != 1 or pcp_world_size != 1:
        raise ValueError(
            "Hybrid KV cache groups with multiple block sizes do not support "
            "context parallelism (dcp_world_size/pcp_world_size > 1)."
        )

    group_block_sizes = [
        group.kv_cache_spec.block_size for group in kv_cache_config.kv_cache_groups
    ]
    return lcm(*group_block_sizes)


def _scheduler_init_with_hybrid_block_size(self, *args, **kwargs):
    if "block_size" in kwargs:
        block_size = kwargs["block_size"]
        vllm_config = kwargs["vllm_config"]
        kv_cache_config = kwargs["kv_cache_config"]
        kwargs["block_size"] = _resolve_scheduler_block_size(
            vllm_config,
            kv_cache_config,
            block_size,
        )
        return self._vllm_ascend_orig_init(*args, **kwargs)

    if len(args) >= 4:
        vllm_config = args[0]
        kv_cache_config = args[1]
        block_size = args[3]
        args = (
            *args[:3],
            _resolve_scheduler_block_size(vllm_config, kv_cache_config, block_size),
            *args[4:],
        )
    return self._vllm_ascend_orig_init(*args, **kwargs)


def _init_with_resolved_hash_block_size(self, *args, **kwargs):
    self._vllm_ascend_orig_init(*args, **kwargs)

    if self.request_block_hasher is None:
        return

    coordinator = getattr(self.scheduler.kv_cache_manager, "coordinator", None)
    hash_block_size = getattr(coordinator, "hash_block_size", None)
    if hash_block_size is None:
        return

    caching_hash_fn = get_hash_fn_by_name(
        self.vllm_config.cache_config.prefix_caching_hash_algo
    )
    init_none_hash(caching_hash_fn)
    self.request_block_hasher = get_request_block_hasher(
        hash_block_size,
        caching_hash_fn,
    )


Scheduler.__init__ = _scheduler_init_with_hybrid_block_size
EngineCore.__init__ = _init_with_resolved_hash_block_size
