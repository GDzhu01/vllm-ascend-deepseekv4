import torch

from vllm_ascend.attention.dsa_v1 import _make_2d_slot_mapping


def test_make_2d_slot_mapping_preserves_padding_slots():
    slot_mapping = torch.tensor([0, 127, 128, 255, -1], dtype=torch.int32)

    two_dim_slot_mapping = _make_2d_slot_mapping(slot_mapping, block_size=128)

    expected = torch.tensor(
        [
            [0, 0],
            [0, 127],
            [1, 0],
            [1, 127],
            [-1, -1],
        ],
        dtype=torch.int32,
    )
    assert torch.equal(two_dim_slot_mapping, expected)
