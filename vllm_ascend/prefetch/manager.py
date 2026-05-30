import torch
import torch_npu

_PREFETCH_STREAM: torch.npu.Stream | None = None
_PREFETCH_EVENT: torch.npu.Event | None = None


def _is_graph_capturing() -> bool:
    try:
        return torch.npu.is_current_stream_capturing()
    except (AttributeError, RuntimeError):
        return False


def get_prefetch_stream() -> torch.npu.Stream:
    global _PREFETCH_STREAM
    if _PREFETCH_STREAM is None:
        _PREFETCH_STREAM = torch_npu.npu.Stream()
        from vllm_ascend.envs import VLLM_PREFETCH_LOG
        if VLLM_PREFETCH_LOG:
            print("Created NPU prefetch stream")
    return _PREFETCH_STREAM


def get_prefetch_event() -> torch.npu.Event:
    global _PREFETCH_EVENT
    if _PREFETCH_EVENT is None:
        _PREFETCH_EVENT = torch_npu.npu.Event()
        from vllm_ascend.envs import VLLM_PREFETCH_LOG
        if VLLM_PREFETCH_LOG:
            print("Created NPU prefetch event")
    return _PREFETCH_EVENT


def prefetch_weight(
    weight: torch.Tensor,
    max_weight_size: int = 0,
    weight_name: str = "",
    *,
    enabled: bool = True,
) -> None:
    if not enabled or weight is None or weight.numel() == 0:
        return

    if _is_graph_capturing():
        return

    compute_stream = torch_npu.npu.current_stream()
    prefetch_stream = get_prefetch_stream()
    prefetch_event = get_prefetch_event()

    weight_size = weight.element_size() * weight.numel()
    if 0 < max_weight_size < weight_size:
        return

    from vllm_ascend.envs import VLLM_PREFETCH_LOG
    if VLLM_PREFETCH_LOG:
        print(
            f"[prefetch] {weight_name or hex(id(weight))} "
            f"| size={weight_size} bytes"
        )

    compute_stream.record_event(prefetch_event)
    prefetch_stream.wait_event(prefetch_event)

    with torch_npu.npu.stream(prefetch_stream):
        torch_npu.npu_prefetch(weight, weight, weight_size, 0)


def sync_prefetch():
    if _is_graph_capturing():
        return

    compute_stream = torch_npu.npu.current_stream()
    prefetch_stream = get_prefetch_stream()
    compute_stream.wait_stream(prefetch_stream)

    from vllm_ascend.envs import VLLM_PREFETCH_LOG
    if VLLM_PREFETCH_LOG:
        print("[prefetch] sync done")


def prefetch_weight_sync(
    weight: torch.Tensor,
    max_weight_size: int = 0,
    weight_name: str = "",
    *,
    enabled: bool = True,
) -> None:
    if not enabled or weight is None or weight.numel() == 0:
        return

    if torch.compiler.is_compiling():
        return

    weight_size = weight.element_size() * weight.numel()
    if 0 < max_weight_size < weight_size:
        return

    from vllm_ascend.envs import VLLM_PREFETCH_LOG
    if VLLM_PREFETCH_LOG:
        print(
            f"[prefetch-sync] {weight_name or hex(id(weight))} "
            f"| size={weight_size} bytes"
        )

    torch_npu.npu_prefetch(weight, weight, weight_size, 0)
