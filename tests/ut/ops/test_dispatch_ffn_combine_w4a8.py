from pathlib import Path


def test_w4a8_fused_epilogue_uses_single_ub_stage():
    repo_root = Path(__file__).parents[3]
    header = repo_root / "csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/dispatch_ffn_combine_w4_a8.h"

    assert "constexpr uint32_t ubStages = 1;" in header.read_text()
