# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.utils.hashing import get_hash_fn_by_name
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.engine.core import EngineCore


if not hasattr(EngineCore, "_vllm_ascend_orig_init"):
    EngineCore._vllm_ascend_orig_init = EngineCore.__init__  # type: ignore[attr-defined]


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


EngineCore.__init__ = _init_with_resolved_hash_block_size
