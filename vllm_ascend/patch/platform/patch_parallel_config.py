import torch
from torch.distributed import ProcessGroup, ReduceOp
from torch.distributed.distributed_c10d import (
    Backend,
    PrefixStore,
    _get_default_timeout,
)
from torch.distributed.rendezvous import rendezvous

import vllm
from vllm.utils.network_utils import get_tcp_uri
from vllm.distributed.utils import init_gloo_process_group

@staticmethod
def has_unfinished_dp(dp_group: ProcessGroup, has_unfinished: bool) -> bool:
    tensor = torch.tensor([has_unfinished], dtype=torch.int32, device="npu",)
    # dp rank 0: has_unfinished_seqs=True
    # dp rank 1: has_unfinished_seqs=False
    # aggregated: has_unfinished_seqs=True
    # so this is an OR operation, i.e. MAX in integers
    torch.distributed.all_reduce(tensor, op=ReduceOp.MAX, group=dp_group)
    aggregated_has_unfinished = bool(tensor.item())
    return aggregated_has_unfinished


@staticmethod
def sync_kv_cache_memory_size(dp_group: ProcessGroup, kv_cache_memory: int) -> int:
    if kv_cache_memory == -1:
        kv_cache_memory = torch.iinfo(torch.int64).max
    tensor = torch.tensor([kv_cache_memory], dtype=torch.int64, device="npu")
    # we cannot use broadcast for stateless dp group since it depends
    # on global rank
    torch.distributed.all_reduce(tensor, op=ReduceOp.MIN, group=dp_group)
    return tensor.item()

def stateless_init_torch_distributed_process_group(
    host: str, port: int, rank: int, world_size: int, backend: str
) -> ProcessGroup:
    """
    A replacement for `torch.distributed.init_process_group` that does not
    pollute the global state. The created ProcessGroup object can be used for
    some operations such as `allreduce`, because it does not depend on the
    global rank. However, some operations such as `broadcast` cannot be used
    because it depends on the global rank.

    # TODO: ask for help from PyTorch team if we need the `broadcast` operation.

    This function is useful when we are not sure about the total number of
    processes in the process group. For example, we may have process
    1, 2, ..., 8 who want to communicate, and process 9 might be the same
    process as process 1, or it might be a different process; process 10
    might be the same process as process 5, or it might be a different process.
    In this case, how can we reliably form a communication channel within
    process 9 and 10, without affecting the communication channel within
    process 1, 2, ..., 8?

    One possible solution is to figure out if process 9 and 10 are the same
    as process 1 and 5 beforehand, and then form a communication channel
    based on the information, adjusting the ranks and world_size etc. However,
    figuring out the information is not always easy, and it will interfere
    with the main communication channel.

    Our solution is to always form a communication channel with process 1, 2,
    ..., 8, and then use this function to form another communication channel
    with process 9 and 10. This way, regardless of whether process 9 and 10
    are the same as process 1 and 5, the main communication channel is
    always formed with process 1, 2, ..., 8, and the additional communication
    channel is formed with process 9 and 10.
    """
    init_method = get_tcp_uri(host, port)
    backend = Backend(backend)  # it is basically string
    timeout = _get_default_timeout(backend)

    store, rank, world_size = next(
        rendezvous(init_method, rank, world_size, timeout=timeout)
    )
    store.set_timeout(timeout)

    group_rank = rank
    group_size = world_size

    # Use a PrefixStore to avoid accidental overrides of keys used by
    # different systems (e.g. RPC) in case the store is multi-tenant.
    prefix_store = PrefixStore(init_method, store)
    try:
        from vllm.platforms import current_platform
        return current_platform.stateless_init_device_torch_dist_pg(
            backend=backend,
            prefix_store=prefix_store,
            group_rank=group_rank,
            group_size=group_size,
            timeout=timeout,
        )
    except NotImplementedError:
        # If platform doesn't implement stateless_init_device_torch_dist_pg, it
        # will raise a NotImplementedError. In this case, we fall back to gloo.
        return init_gloo_process_group(
            prefix_store=prefix_store,
            group_rank=group_rank,
            group_size=group_size,
            timeout=timeout,
        )


vllm.config.parallel.ParallelConfig.has_unfinished_dp = has_unfinished_dp
vllm.config.parallel.ParallelConfig.sync_kv_cache_memory_size = sync_kv_cache_memory_size
vllm.distributed.utils.stateless_init_torch_distributed_process_group = stateless_init_torch_distributed_process_group
