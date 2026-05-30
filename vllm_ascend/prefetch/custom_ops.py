import torch

from vllm.utils.torch_utils import direct_register_custom_op


def _prefetch_sync_impl(
    hidden_states: torch.Tensor,
    layer_idx: int,
) -> None:
    from vllm_ascend.prefetch.manager import sync_prefetch
    sync_prefetch()


def _prefetch_sync_fake(
    hidden_states: torch.Tensor,
    layer_idx: int,
) -> None:
    pass


def _prefetch_after_attn_impl(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    max_weight_size: int,
    layer_idx: int,
    weight_name: str,
) -> None:
    from vllm_ascend.prefetch.manager import (
        prefetch_weight_sync, prefetch_weight, _is_graph_capturing,
    )
    from vllm_ascend.envs import VLLM_PREFETCH
    if VLLM_PREFETCH and weight is not None and weight.numel() > 0:
        if _is_graph_capturing():
            prefetch_weight_sync(weight, max_weight_size, weight_name=weight_name)
        else:
            prefetch_weight(weight, max_weight_size, weight_name=weight_name)


def _prefetch_after_attn_fake(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    max_weight_size: int,
    layer_idx: int,
    weight_name: str,
) -> None:
    pass


def _prefetch_after_mlp_impl(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    max_weight_size: int,
    layer_idx: int,
    weight_name: str,
) -> None:
    from vllm_ascend.prefetch.manager import (
        prefetch_weight_sync, prefetch_weight, _is_graph_capturing,
    )
    from vllm_ascend.envs import VLLM_PREFETCH
    if VLLM_PREFETCH and weight is not None and weight.numel() > 0:
        if _is_graph_capturing():
            prefetch_weight_sync(weight, max_weight_size, weight_name=weight_name)
        else:
            prefetch_weight(weight, max_weight_size, weight_name=weight_name)


def _prefetch_after_mlp_fake(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    max_weight_size: int,
    layer_idx: int,
    weight_name: str,
) -> None:
    pass


def _prefetch_moe_experts_impl(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    expert_ids: torch.Tensor,
    max_size_per_expert: int,
    layer_idx: int,
    weight_name: str,
) -> None:
    from vllm_ascend.prefetch.manager import prefetch_weight_sync, _is_graph_capturing
    from vllm_ascend.envs import VLLM_PREFETCH
    if not VLLM_PREFETCH or weight is None or weight.numel() == 0:
        return
    if torch.compiler.is_compiling():
        return
    if _is_graph_capturing():
        return
    expert_ids = expert_ids.flatten()
    n = expert_ids.numel()
    seen = set()
    for i in range(n):
        eid = expert_ids[i].item()
        if eid < 0 or eid in seen:
            continue
        seen.add(eid)
        ew = weight[eid]
        prefetch_weight_sync(
            ew, max_size_per_expert,
            weight_name=f"{weight_name}.e{eid}",
        )


def _prefetch_moe_experts_fake(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    expert_ids: torch.Tensor,
    max_size_per_expert: int,
    layer_idx: int,
    weight_name: str,
) -> None:
    pass


def register_prefetch_ops():
    direct_register_custom_op(
        op_name="prefetch_sync",
        op_func=_prefetch_sync_impl,
        mutates_args=["hidden_states"],
        fake_impl=_prefetch_sync_fake,
    )

    direct_register_custom_op(
        op_name="prefetch_after_attn",
        op_func=_prefetch_after_attn_impl,
        mutates_args=["hidden_states"],
        fake_impl=_prefetch_after_attn_fake,
    )

    direct_register_custom_op(
        op_name="prefetch_after_mlp",
        op_func=_prefetch_after_mlp_impl,
        mutates_args=["hidden_states"],
        fake_impl=_prefetch_after_mlp_fake,
    )

    direct_register_custom_op(
        op_name="prefetch_moe_experts",
        op_func=_prefetch_moe_experts_impl,
        mutates_args=["hidden_states"],
        fake_impl=_prefetch_moe_experts_fake,
    )
