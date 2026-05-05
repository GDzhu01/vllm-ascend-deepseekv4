#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.reasoning import ReasoningParserManager
from vllm.renderers.registry import RENDERER_REGISTRY
from vllm.tokenizers.registry import TokenizerRegistry
from vllm.tool_parsers import ToolParserManager

import vllm_ascend.patch.platform.patch_deepseek_v4_agentic as patch

TC_START = "<｜DSML｜tool_calls>"
TC_END = "</｜DSML｜tool_calls>"
INV_START = '<｜DSML｜invoke name="'
INV_END = "</｜DSML｜invoke>"
PARAM_START = '<｜DSML｜parameter name="'
PARAM_END = "</｜DSML｜parameter>"


class FakeHfTokenizer:
    vocab_size = 100

    def get_added_vocab(self) -> dict[str, int]:
        return {"</think>": 100}

    def get_vocab(self) -> dict[str, int]:
        return {"<think>": 99, "</think>": 100}

    def encode(
        self,
        text: str,
        add_special_tokens: bool = False,
        **kwargs,
    ) -> list[int]:
        self.last_encode = (text, add_special_tokens, kwargs)
        return [len(text)]


def _tokenizer():
    return patch.get_deepseek_v4_tokenizer(FakeHfTokenizer())


def _parser():
    tokenizer = MagicMock()
    tokenizer.get_vocab.return_value = {}
    return patch.DeepSeekV4ToolParser(tokenizer)


def _reasoning_parser(**chat_template_kwargs):
    return patch.DeepSeekV4ReasoningParser(
        FakeHfTokenizer(),
        chat_template_kwargs=chat_template_kwargs,
    )


def _request(tools=None):
    request = MagicMock()
    request.tools = tools or []
    request.tool_choice = "auto"
    return request


def _tool(name: str, properties: dict[str, dict]):
    return SimpleNamespace(
        function=SimpleNamespace(
            name=name,
            parameters={
                "type": "object",
                "properties": properties,
            },
        )
    )


def _model_output(function_name: str, params: list[tuple[str, str, str]]) -> str:
    param_str = "".join(
        f'{PARAM_START}{name}" string="{is_string}">{value}{PARAM_END}'
        for name, is_string, value in params
    )
    return f"{TC_START}{INV_START}{function_name}\">{param_str}{INV_END}{TC_END}"


def test_deepseek_v4_registries_are_available():
    assert TokenizerRegistry.load_tokenizer_cls("deepseek_v4").__name__ == (
        "DeepseekV4Tokenizer"
    )
    assert RENDERER_REGISTRY.load_renderer_cls("deepseek_v4").__name__ == (
        "DeepseekV4Renderer"
    )
    assert ToolParserManager.get_tool_parser("deepseek_v4").__name__ == (
        "DeepSeekV4ToolParser"
    )
    assert ReasoningParserManager.get_reasoning_parser("deepseek_v4").__name__ == (
        "DeepSeekV4ReasoningParser"
    )


def test_tokenizer_renders_thinking_prompt_and_escapes_arguments_schema():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "echo_args",
                "description": "Echo arguments",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "arguments": {"type": "string"},
                    },
                    "required": ["arguments"],
                },
            },
        }
    ]

    prompt = _tokenizer().apply_chat_template(
        [{"role": "user", "content": "Echo this"}],
        tools=tools,
        tokenize=False,
        enable_thinking=True,
        reasoning_effort="high",
    )

    assert prompt.endswith("<｜User｜>Echo this<｜Assistant｜><think>")
    assert "<｜DSML｜tool_calls>" in prompt
    assert "__vllm_param_arguments__" in prompt
    assert '"required": ["__vllm_param_arguments__"]' in prompt
    assert '"arguments": {"type": "string"}' not in prompt


@pytest.mark.parametrize("reasoning_effort", ["high", None])
def test_tokenizer_noop_reasoning_effort_values_do_not_add_max_prompt(
    reasoning_effort,
):
    prompt = _tokenizer().apply_chat_template(
        [{"role": "user", "content": "Hello"}],
        tokenize=False,
        enable_thinking=True,
        reasoning_effort=reasoning_effort,
    )

    assert prompt == "<｜begin▁of▁sentence｜><｜User｜>Hello<｜Assistant｜><think>"
    assert "Reasoning Effort: Absolute maximum" not in prompt


def test_tokenizer_max_reasoning_effort_adds_max_prompt():
    prompt = _tokenizer().apply_chat_template(
        [{"role": "user", "content": "Hello"}],
        tokenize=False,
        enable_thinking=True,
        reasoning_effort="max",
    )

    assert prompt.startswith(
        "<｜begin▁of▁sentence｜>Reasoning Effort: Absolute maximum"
    )
    assert prompt.endswith("<｜User｜>Hello<｜Assistant｜><think>")


def test_tokenizer_thinking_false_overrides_enable_thinking_and_max_effort():
    prompt = _tokenizer().apply_chat_template(
        [{"role": "user", "content": "Hello"}],
        tokenize=False,
        thinking=False,
        enable_thinking=True,
        reasoning_effort="max",
    )

    assert prompt == "<｜begin▁of▁sentence｜><｜User｜>Hello<｜Assistant｜></think>"
    assert "Reasoning Effort: Absolute maximum" not in prompt


@pytest.mark.parametrize("reasoning_effort", ["high", "max"])
def test_chat_completion_request_accepts_reasoning_effort_values(reasoning_effort):
    request = ChatCompletionRequest.model_validate(
        {
            "messages": [{"role": "user", "content": "Hello"}],
            "reasoning_effort": reasoning_effort,
        }
    )

    assert request.reasoning_effort == reasoning_effort


@pytest.mark.parametrize("reasoning_effort", ["none", "low", "medium", "xhigh"])
def test_chat_completion_request_rejects_unsupported_reasoning_effort_values(
    reasoning_effort,
):
    with pytest.raises(ValueError):
        ChatCompletionRequest.model_validate(
            {
                "messages": [{"role": "user", "content": "Hello"}],
                "reasoning_effort": reasoning_effort,
            }
        )


def test_tokenizer_escapes_arguments_history_tool_call_name():
    prompt = _tokenizer().apply_chat_template(
        [
            {"role": "user", "content": "Echo this"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "echo_args",
                            "arguments": '{"arguments": "hello"}',
                        },
                    }
                ],
            },
        ],
        tokenize=False,
    )

    assert 'parameter name="__vllm_param_arguments__" string="true">hello' in prompt
    assert 'parameter name="arguments"' not in prompt


def test_reasoning_parser_counts_generated_thinking_tokens_without_start_token():
    parser = _reasoning_parser(enable_thinking=True)

    assert parser.count_reasoning_tokens([11, 12, 100, 13]) == 2
    assert parser.count_reasoning_tokens([11, 12, 13]) == 3


def test_reasoning_parser_counts_explicit_thinking_span_tokens():
    parser = _reasoning_parser(enable_thinking=True)

    assert parser.count_reasoning_tokens([99, 11, 12, 100, 13]) == 2


def test_reasoning_parser_counts_zero_tokens_when_thinking_is_disabled():
    parser = _reasoning_parser(thinking=False)

    assert parser.count_reasoning_tokens([11, 12, 100, 13]) == 0


def test_parser_preserves_typed_arguments():
    parser = _parser()
    request = _request(
        [
            _tool(
                "plan_trip",
                {
                    "days": {"type": "integer"},
                    "flexible": {"type": "boolean"},
                    "cities": {"type": "array"},
                    "notes": {"type": "string"},
                },
            )
        ]
    )
    model_output = _model_output(
        "plan_trip",
        [
            ("days", "false", "3"),
            ("flexible", "false", "false"),
            ("cities", "false", '["Beijing", "Shanghai"]'),
            ("notes", "true", "window seat"),
        ],
    )

    result = parser.extract_tool_calls(model_output, request)

    assert result.tools_called
    assert json.loads(result.tool_calls[0].function.arguments) == {
        "days": 3,
        "flexible": False,
        "cities": ["Beijing", "Shanghai"],
        "notes": "window seat",
    }


def test_parser_repairs_wrappers_and_unescapes_arguments_name():
    parser = _parser()
    request = _request([_tool("echo_args", {"arguments": {"type": "string"}})])
    model_output = _model_output(
        "echo_args",
        [("__vllm_param_arguments__", "true", "hello")],
    )

    result = parser.extract_tool_calls(model_output, request)

    assert result.tools_called
    assert json.loads(result.tool_calls[0].function.arguments) == {"arguments": "hello"}

    request = _request([_tool("get_weather", {"location": {"type": "string"}})])
    model_output = _model_output(
        "get_weather",
        [("input", "true", '{"location": "Beijing"}')],
    )

    result = parser.extract_tool_calls(model_output, request)

    assert result.tools_called
    assert json.loads(result.tool_calls[0].function.arguments) == {
        "location": "Beijing"
    }


def test_streaming_split_start_token_does_not_leak_dsml_markers():
    parser = _parser()
    request = _request()
    full_text = "I will check." + _model_output(
        "search",
        [("query", "true", "vllm")],
    )
    previous_text = ""
    deltas = []

    for start in range(len(full_text)):
        delta_text = full_text[start : start + 1]
        current_text = previous_text + delta_text
        delta = parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=delta_text,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[1],
            request=request,
        )
        previous_text = current_text
        if delta is not None:
            deltas.append(delta)
            deltas.extend(parser.drain_pending_tool_call_deltas())

    content = "".join(delta.content or "" for delta in deltas)
    tool_deltas = [
        tool_call
        for delta in deltas
        for tool_call in (delta.tool_calls or [])
    ]
    argument_pieces = [
        tool_call.function.arguments
        for tool_call in tool_deltas
        if tool_call.function and tool_call.function.arguments is not None
    ]

    assert content == "I will check."
    assert "DSML" not in content
    assert sum(tool_call.id is not None for tool_call in tool_deltas) == 1
    assert sum(tool_call.type is not None for tool_call in tool_deltas) == 1
    assert sum(bool(tool_call.function.name) for tool_call in tool_deltas) == 1
    assert json.loads("".join(argument_pieces)) == {"query": "vllm"}


def test_streaming_tool_arguments_emit_before_invoke_is_complete():
    parser = _parser()
    request = _request()
    prefix = (
        f"{TC_START}{INV_START}search\">"
        f'{PARAM_START}query" string="true">'
    )
    suffix = f"{PARAM_END}{INV_END}{TC_END}"
    previous_text = ""
    deltas = []

    for delta_text in [prefix, "partial value", suffix]:
        current_text = previous_text + delta_text
        delta = parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=delta_text,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[1],
            request=request,
        )
        previous_text = current_text
        if delta is not None:
            deltas.append(delta)
            deltas.extend(parser.drain_pending_tool_call_deltas())

    tool_deltas = [
        tool_call
        for delta in deltas
        for tool_call in (delta.tool_calls or [])
    ]
    argument_pieces = [
        tool_call.function.arguments
        for tool_call in tool_deltas
        if tool_call.function and tool_call.function.arguments is not None
    ]

    assert '"partial value' in "".join(argument_pieces[:-1])
    assert json.loads("".join(argument_pieces)) == {"query": "partial value"}


def test_streaming_content_and_tool_call_in_same_delta_are_both_emitted():
    parser = _parser()
    request = _request()
    delta_text = "I will check." + (
        f"{TC_START}{INV_START}search\">"
        f'{PARAM_START}query" string="true">vllm'
        f"{PARAM_END}{INV_END}{TC_END}"
    )

    delta = parser.extract_tool_calls_streaming(
        previous_text="",
        current_text=delta_text,
        delta_text=delta_text,
        previous_token_ids=[],
        current_token_ids=[],
        delta_token_ids=[1],
        request=request,
    )
    deltas = [delta]
    deltas.extend(parser.drain_pending_tool_call_deltas())

    tool_deltas = [
        tool_call
        for item in deltas
        if item is not None
        for tool_call in (item.tool_calls or [])
    ]
    argument_pieces = [
        tool_call.function.arguments
        for tool_call in tool_deltas
        if tool_call.function and tool_call.function.arguments is not None
    ]

    assert deltas[0].content == "I will check."
    assert json.loads("".join(argument_pieces)) == {"query": "vllm"}


def test_streaming_repairs_wrapped_input_arguments():
    parser = _parser()
    request = _request([_tool("get_weather", {"location": {"type": "string"}})])
    full_text = (
        f"{TC_START}{INV_START}get_weather\">"
        f'{PARAM_START}input" string="true">{{"location": "Beijing"}}'
        f"{PARAM_END}{INV_END}{TC_END}"
    )
    previous_text = ""
    deltas = []

    for start in range(0, len(full_text), 7):
        delta_text = full_text[start : start + 7]
        current_text = previous_text + delta_text
        delta = parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=delta_text,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[1],
            request=request,
        )
        previous_text = current_text
        if delta is not None:
            deltas.append(delta)
            deltas.extend(parser.drain_pending_tool_call_deltas())

    argument_pieces = [
        tool_call.function.arguments
        for delta in deltas
        for tool_call in (delta.tool_calls or [])
        if tool_call.function and tool_call.function.arguments is not None
    ]

    assert json.loads("".join(argument_pieces)) == {"location": "Beijing"}


def test_streaming_parser_state_is_isolated_between_choices():
    request = _request()
    parser0 = _parser()
    parser1 = _parser()

    first_part = (
        f"{TC_START}{INV_START}search\">"
        f'{PARAM_START}query" string="true">abc'
    )
    first_delta = parser0.extract_tool_calls_streaming(
        previous_text="",
        current_text=first_part,
        delta_text=first_part,
        previous_token_ids=[],
        current_token_ids=[],
        delta_token_ids=[1],
        request=request,
    )
    first_deltas = [first_delta]
    first_deltas.extend(parser0.drain_pending_tool_call_deltas())

    second_delta = parser1.extract_tool_calls_streaming(
        previous_text="",
        current_text="plain choice",
        delta_text="plain choice",
        previous_token_ids=[],
        current_token_ids=[],
        delta_token_ids=[1],
        request=request,
    )

    final_part = f"def{PARAM_END}{INV_END}{TC_END}"
    final_delta = parser0.extract_tool_calls_streaming(
        previous_text=first_part,
        current_text=first_part + final_part,
        delta_text=final_part,
        previous_token_ids=[],
        current_token_ids=[],
        delta_token_ids=[1],
        request=request,
    )
    first_deltas.append(final_delta)
    first_deltas.extend(parser0.drain_pending_tool_call_deltas())

    content = "".join(delta.content or "" for delta in first_deltas if delta is not None)
    tool_deltas = [
        tool_call
        for delta in first_deltas
        if delta is not None
        for tool_call in (delta.tool_calls or [])
    ]
    argument_pieces = [
        tool_call.function.arguments
        for tool_call in tool_deltas
        if tool_call.function and tool_call.function.arguments is not None
    ]

    assert second_delta.content == "plain choice"
    assert "DSML" not in content
    assert json.loads("".join(argument_pieces)) == {"query": "abcdef"}
