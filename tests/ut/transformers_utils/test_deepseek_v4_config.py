# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from vllm_ascend.transformers_utils.configs.deepseek_v4 import DeepseekV4Config


def _write_config(tmp_path, **overrides):
    config = {
        "model_type": "deepseek_v4",
        "num_hidden_layers": 4,
        "compress_ratios": [1, 1, 4, 128],
        "rope_theta": 12345,
        "compress_rope_theta": 67890,
        "swiglu_limit": 12.5,
        "rope_scaling": {
            "type": "yarn",
            "factor": 16,
            "beta_fast": 32,
            "beta_slow": 1,
            "original_max_position_embeddings": 65536,
        },
    }
    config.update(overrides)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path


def test_missing_compress_ratios_defaults_to_uncompressed_layers():
    config = DeepseekV4Config(swiglu_limit=10.0)

    assert config.compress_ratios == [0] * config.num_hidden_layers
    assert config.rope_theta == 10000.0
    assert config.compress_rope_theta == 160000.0
    assert config.get_layer_compress_ratio(0) == 1
    assert config.get_layer_rope_theta(0) == 10000.0


def test_compress_parameters_are_loaded_from_metadata(tmp_path):
    _write_config(tmp_path, compress_ratios=[0, 1, 4, 128])

    config = DeepseekV4Config.from_pretrained(tmp_path)

    assert config.compress_ratios == [0, 1, 4, 128]
    assert config.get_layer_compress_ratio(0) == 1
    assert config.get_layer_compress_ratio(1) == 1
    assert config.get_layer_compress_ratio(3) == 128
    assert config.get_layer_rope_theta(0) == 12345
    assert config.get_layer_rope_theta(1) == 12345
    assert config.get_layer_rope_theta(2) == 67890
    assert config.get_layer_rope_theta(3) == 67890
    assert config.swiglu_limit == 12.5
    assert config.get_rope_groups_for_compress_ratio(0) == ["default"]
    assert config.get_rope_groups_for_compress_ratio(1) == ["default"]
    assert config.get_rope_groups_for_compress_ratio(4) == ["default", "c4"]
    assert config.get_rope_groups_for_compress_ratio(128) == ["default", "c128"]


def test_invalid_compress_ratio_is_rejected(tmp_path):
    _write_config(tmp_path, compress_ratios=[1, 7, 4, 128])

    with pytest.raises(ValueError, match="compress_ratios contains unsupported values"):
        DeepseekV4Config.from_pretrained(tmp_path)


def test_short_compress_ratio_list_is_padded_with_zero(tmp_path):
    _write_config(tmp_path, compress_ratios=[1, 4, 128])

    with pytest.warns(UserWarning, match="remaining layers will be padded with 0"):
        config = DeepseekV4Config.from_pretrained(tmp_path)

    assert config.compress_ratios == [1, 4, 128, 0]
    assert config.mtp_compress_ratios == []
    assert config.get_layer_compress_ratio(3) == 1
    assert config.get_layer_rope_theta(3) == 12345


def test_long_compress_ratio_list_preserves_mtp_entries(tmp_path):
    _write_config(tmp_path, compress_ratios=[1, 4, 128, 0, 4, 128])

    with pytest.warns(UserWarning, match="preserved as mtp_compress_ratios"):
        config = DeepseekV4Config.from_pretrained(tmp_path)

    assert config.compress_ratios == [1, 4, 128, 0, 4, 128]
    assert config.mtp_compress_ratios == [4, 128]
