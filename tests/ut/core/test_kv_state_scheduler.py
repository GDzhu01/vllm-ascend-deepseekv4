# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace
from unittest.mock import MagicMock

from vllm.v1.core.sched.output import SchedulerOutput

from vllm_ascend.core.kv_state_scheduler import AsyncKVStateScheduler


def test_async_kv_state_scheduler_marks_structured_decode_request():
    scheduler = object.__new__(AsyncKVStateScheduler)
    scheduler.num_spec_tokens = 2
    scheduler._spec_token_placeholders = [-1, -1]
    scheduler.finished_req_ids = {"finished"}
    scheduler._free_encoder_inputs = MagicMock()

    prefill_req = SimpleNamespace(
        num_computed_tokens=0,
        num_tokens=5,
        num_output_placeholders=0,
        use_structured_output=True,
        has_encoder_inputs=False,
        spec_token_ids=[],
        is_prefill_chunk=False,
    )
    decode_req = SimpleNamespace(
        num_computed_tokens=5,
        num_tokens=5,
        num_output_placeholders=0,
        use_structured_output=True,
        has_encoder_inputs=False,
        spec_token_ids=[],
        is_prefill_chunk=False,
    )
    scheduler.requests = {
        "prefill": prefill_req,
        "decode": decode_req,
    }

    scheduler_output = SchedulerOutput.make_empty()
    scheduler_output.num_scheduled_tokens = {
        "prefill": 3,
        "decode": 1,
    }
    scheduler_output.total_num_scheduled_tokens = 4

    scheduler._update_after_schedule(scheduler_output)

    assert scheduler_output.has_structured_output_requests
    assert not scheduler_output.pending_structured_output_tokens
    assert prefill_req.is_prefill_chunk
    assert prefill_req.num_output_placeholders == 0
    assert not decode_req.is_prefill_chunk
    assert decode_req.num_output_placeholders == 1
    assert decode_req.spec_token_ids is scheduler._spec_token_placeholders
    assert scheduler.finished_req_ids == set()
