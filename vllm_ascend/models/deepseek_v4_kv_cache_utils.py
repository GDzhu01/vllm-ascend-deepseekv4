import torch


DEFAULT_DEEPSEEK_SVF_BLOCK_SIZE = 128
DEFAULT_DEEPSEEK_V4_INDEXER_HEAD_SIZE = 128
DEFAULT_DEEPSEEK_V4_INDEXER_SCALE_DIM = 1


def _get_dtype_size(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def get_deepseek_svf_block_size(configured_block_size: int | None) -> int:
    return configured_block_size or DEFAULT_DEEPSEEK_SVF_BLOCK_SIZE


def get_deepseek_svf_alignment(
    *,
    head_dim: int,
    rope_head_dim: int = 0,
) -> int:
    return head_dim + rope_head_dim


def _get_deepseek_v4_indexer_page_size_bytes(
    target_block_size: int,
    *,
    indexer_head_size: int,
    indexer_dtype: torch.dtype,
    indexer_scale_dim: int,
    indexer_scale_dtype: torch.dtype,
) -> int:
    return target_block_size * (
        indexer_head_size * _get_dtype_size(indexer_dtype)
        + indexer_scale_dim * _get_dtype_size(indexer_scale_dtype)
    )


def _get_deepseek_v4_mla_page_size_bytes(
    target_block_size: int,
    *,
    kv_head_size: int,
    kv_dtype: torch.dtype,
) -> int:
    return target_block_size * kv_head_size * _get_dtype_size(kv_dtype)


def get_deepseek_v4_state_cache_layout(
    target_block_size: int,
    *,
    compress_ratio: int,
    state_dim: int,
    state_dtype: torch.dtype,
    kv_head_size: int,
    kv_dtype: torch.dtype,
    indexer_head_size: int = DEFAULT_DEEPSEEK_V4_INDEXER_HEAD_SIZE,
    indexer_dtype: torch.dtype = torch.int8,
    indexer_scale_dim: int = DEFAULT_DEEPSEEK_V4_INDEXER_SCALE_DIM,
    indexer_scale_dtype: torch.dtype = torch.float16,
) -> tuple[int, int]:
    if compress_ratio == 4:
        reference_page_size_bytes = _get_deepseek_v4_indexer_page_size_bytes(
            target_block_size,
            indexer_head_size=indexer_head_size,
            indexer_dtype=indexer_dtype,
            indexer_scale_dim=indexer_scale_dim,
            indexer_scale_dtype=indexer_scale_dtype,
        )
    elif compress_ratio == 128:
        reference_page_size_bytes = _get_deepseek_v4_mla_page_size_bytes(
            target_block_size,
            kv_head_size=kv_head_size,
            kv_dtype=kv_dtype,
        )
    else:
        raise ValueError(
            f"Only support compress_ratio in [4, 128]. Got {compress_ratio}."
        )

    state_token_size_bytes = state_dim * _get_dtype_size(state_dtype)
    state_block_size = max(reference_page_size_bytes // state_token_size_bytes, 1)
    return state_block_size, reference_page_size_bytes
