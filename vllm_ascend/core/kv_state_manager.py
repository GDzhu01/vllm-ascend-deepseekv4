from vllm.v1.request import Request


class KVStateManager:
    def __init__(
        self,
        max_num_seqs: int,
    ):
        self.max_num_seqs = max_num_seqs
        # same as kv_cache, keep state 0 empty for padding in block_table
        self.states_pool: set[int] = set(range(1, max_num_seqs + 1))
        self.req_to_state_id: dict[str, int] = {}

    def allocate_slots(
        self,
        request: Request = None,
    ) -> int:
        if len(self.states_pool) == 0:
            return None
        request_id = request.request_id
        if request_id in self.req_to_state_id:
            raise ValueError(f"Request {request_id} already allocated, only allocate state while prefill.")
        state_to_allocate = self.states_pool.pop()
        self.req_to_state_id[request_id] = state_to_allocate
        return state_to_allocate

    def free(
            self,
            request: Request = None,
        ) -> None:
        request_id = request.request_id
        if request_id not in self.req_to_state_id:
            raise ValueError(f"Request {request_id} not allocated a state, unable to free.")
        state_to_free = self.req_to_state_id.pop(request_id)
        self.states_pool.add(state_to_free)
