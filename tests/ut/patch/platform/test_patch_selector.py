# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import patch

import torch

from vllm_ascend.patch.platform import patch_selector


def test_get_attn_backend_supports_legacy_block_size_argument():
    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(
            user_specified_block_size=False,
            block_size=256,
        ),
        attention_config=SimpleNamespace(backend="ascend"),
    )

    with patch("vllm.config.get_current_vllm_config", return_value=vllm_config):
        with patch(
            "vllm_ascend.patch.platform.patch_selector._cached_get_attn_backend",
            return_value="backend",
        ) as mock_cached:
            result = patch_selector.get_attn_backend(
                0,
                torch.bfloat16,
                None,
                128,
                use_mla=True,
                use_sparse=True,
                use_compress=True,
            )

    assert result == "backend"
    selector_config = mock_cached.call_args.kwargs["attn_selector_config"]
    assert selector_config.block_size == 128
    assert selector_config.use_mla is True
    assert selector_config.use_sparse is True
    assert selector_config.use_compress is True


def test_get_attn_backend_uses_user_block_size_for_new_signature_calls():
    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(
            user_specified_block_size=True,
            block_size=64,
        ),
        attention_config=SimpleNamespace(backend="ascend"),
    )

    with patch("vllm.config.get_current_vllm_config", return_value=vllm_config):
        with patch(
            "vllm_ascend.patch.platform.patch_selector._cached_get_attn_backend",
            return_value="backend",
        ) as mock_cached:
            result = patch_selector.get_attn_backend(
                0,
                torch.bfloat16,
                None,
                use_mla=False,
                use_sparse=False,
                use_per_head_quant_scales=True,
                num_heads=32,
            )

    assert result == "backend"
    selector_config = mock_cached.call_args.kwargs["attn_selector_config"]
    assert selector_config.block_size == 64
    assert selector_config.use_per_head_quant_scales is True
    assert mock_cached.call_args.kwargs["num_heads"] == 32
