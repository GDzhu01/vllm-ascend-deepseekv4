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
#
# Backport DeepSeek V4 tokenizer, tool-call parser, and reasoning parser
# registration from upstream vLLM.
#

from __future__ import annotations

import copy
import json
import sys
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import regex as re
from transformers import PreTrainedTokenizerFast
from vllm.config import VllmConfig
from vllm.entrypoints.chat_utils import (
    ChatCompletionMessageParam,
    ConversationMessage,
    parse_chat_messages,
    parse_chat_messages_async,
)
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.logger import init_logger
from vllm.reasoning import ReasoningParserManager
from vllm.renderers.base import BaseRenderer
from vllm.renderers.inputs import DictPrompt
from vllm.renderers.inputs.preprocess import parse_dec_only_prompt
from vllm.renderers.params import ChatParams
from vllm.renderers.registry import RENDERER_REGISTRY
from vllm.tokenizers import cached_get_tokenizer
from vllm.tokenizers.hf import HfTokenizer, get_cached_tokenizer
from vllm.tokenizers.protocol import TokenizerLike
from vllm.tokenizers.registry import TokenizerRegistry
from vllm.tool_parsers import ToolParserManager
from vllm.tool_parsers.abstract_tool_parser import ToolParser

logger = init_logger(__name__)

bos_token: str = "<｜begin▁of▁sentence｜>"
eos_token: str = "<｜end▁of▁sentence｜>"
thinking_start_token: str = "<think>"
thinking_end_token: str = "</think>"
dsml_token: str = "｜DSML｜"

USER_SP_TOKEN = "<｜User｜>"
ASSISTANT_SP_TOKEN = "<｜Assistant｜>"
LATEST_REMINDER_SP_TOKEN = "<｜latest_reminder｜>"

DS_TASK_SP_TOKENS = {
    "action": "<｜action｜>",
    "query": "<｜query｜>",
    "authority": "<｜authority｜>",
    "domain": "<｜domain｜>",
    "title": "<｜title｜>",
    "read_url": "<｜read_url｜>",
}
VALID_TASKS = set(DS_TASK_SP_TOKENS.keys())

system_msg_template: str = "{content}"
user_msg_template: str = "{content}"
latest_reminder_msg_template: str = "{content}"
assistant_msg_template: str = "{reasoning}{content}{tool_calls}" + eos_token
assistant_msg_wo_eos_template: str = "{reasoning}{content}{tool_calls}"
thinking_template: str = "{reasoning}"

response_format_template: str = (
    "## Response Format:\n\nYou MUST strictly adhere to the following schema to reply:\n{schema}"
)
tool_call_template: str = (
    '<{dsml_token}invoke name="{name}">\n{arguments}\n</{dsml_token}invoke>'
)
tool_calls_template = (
    "<{dsml_token}{tc_block_name}>\n{tool_calls}\n</{dsml_token}{tc_block_name}>"
)
tool_calls_block_name: str = "tool_calls"
ESCAPED_ARGUMENTS_PARAM_NAME = "__vllm_param_arguments__"
REASONING_EFFORT_MAX_VALUES = frozenset({"max", "xhigh"})
REASONING_EFFORT_NOOP_VALUES = frozenset({"none", "low", "medium", "high", None})

tool_output_template: str = "<tool_result>{content}</tool_result>"

REASONING_EFFORT_MAX = (
    "Reasoning Effort: Absolute maximum with no shortcuts permitted.\n"
    "You MUST be very thorough in your thinking and comprehensively decompose "
    "the problem to resolve the root cause, rigorously stress-testing your "
    "logic against all potential paths, edge cases, and adversarial scenarios.\n"
    "Explicitly write out your entire deliberation process, documenting every "
    "intermediate step, considered alternative, and rejected hypothesis to "
    "ensure absolutely no assumption is left unchecked.\n\n"
)

TOOLS_TEMPLATE = (
    "## Tools\n\n"
    "You have access to a set of tools to help answer the user's question. "
    'You can invoke tools by writing a "<{dsml_token}tool_calls>" block like '
    "the following:\n\n"
    "<{dsml_token}tool_calls>\n"
    '<{dsml_token}invoke name="$TOOL_NAME">\n'
    '<{dsml_token}parameter name="$PARAMETER_NAME" string="true|false">'
    "$PARAMETER_VALUE</{dsml_token}parameter>\n"
    "...\n"
    "</{dsml_token}invoke>\n"
    '<{dsml_token}invoke name="$TOOL_NAME2">\n'
    "...\n"
    "</{dsml_token}invoke>\n"
    "</{dsml_token}tool_calls>\n\n"
    'String parameters should be specified as is and set `string="true"`. '
    "For all other types (numbers, booleans, arrays, objects), pass the value "
    'in JSON format and set `string="false"`.\n\n'
    "If thinking_mode is enabled (triggered by {thinking_start_token}), you "
    "MUST output your complete reasoning inside "
    "{thinking_start_token}...{thinking_end_token} BEFORE any tool calls or "
    "final response.\n\n"
    "Otherwise, output directly after {thinking_end_token} with tool calls or "
    "final response.\n\n"
    "### Available Tool Schemas\n\n"
    "{tool_schemas}\n\n"
    "You MUST strictly follow the above defined tool name and parameter "
    "schemas to invoke tool calls.\n"
)


def _patch_chat_completion_reasoning_effort() -> None:
    reasoning_effort_annotation = (
        Literal["none", "low", "medium", "high", "max", "xhigh"] | None
    )
    ChatCompletionRequest.__annotations__["reasoning_effort"] = (
        reasoning_effort_annotation
    )
    ChatCompletionRequest.model_fields["reasoning_effort"].annotation = (
        reasoning_effort_annotation
    )
    ChatCompletionRequest.model_rebuild(force=True)


def to_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return json.dumps(value, ensure_ascii=True)


def tools_from_openai_format(tools):
    return [tool["function"] for tool in tools]


def _escape_param_name(name: str) -> str:
    if name == "arguments":
        return ESCAPED_ARGUMENTS_PARAM_NAME
    return name


def _unescape_param_name(name: str) -> str:
    if name == ESCAPED_ARGUMENTS_PARAM_NAME:
        return "arguments"
    return name


def _escape_tool_schema(tool: dict[str, Any]) -> dict[str, Any]:
    escaped_tool = copy.deepcopy(tool)
    parameters = escaped_tool.get("parameters")
    if not isinstance(parameters, dict):
        return escaped_tool

    properties = parameters.get("properties")
    if isinstance(properties, dict):
        parameters["properties"] = {
            _escape_param_name(key): value for key, value in properties.items()
        }

    required = parameters.get("required")
    if isinstance(required, list):
        parameters["required"] = [
            _escape_param_name(name) if isinstance(name, str) else name
            for name in required
        ]

    return escaped_tool


def tool_calls_from_openai_format(tool_calls):
    return [
        {
            "name": tool_call["function"]["name"],
            "arguments": tool_call["function"]["arguments"],
        }
        for tool_call in tool_calls
    ]


def encode_arguments_to_dsml(tool_call: dict[str, Any]) -> str:
    p_dsml_template = (
        '<{dsml_token}parameter name="{key}" string="{is_str}">{value}</{dsml_token}parameter>'
    )
    p_dsml_strs = []

    arguments = _normalize_tool_call_arguments(tool_call["arguments"])
    if not isinstance(arguments, dict):
        return ""

    for key, value in arguments.items():
        p_dsml_strs.append(
            p_dsml_template.format(
                dsml_token=dsml_token,
                key=_escape_param_name(key),
                is_str="true" if isinstance(value, str) else "false",
                value=value if isinstance(value, str) else to_json(value),
            )
        )

    return "\n".join(p_dsml_strs)


def _normalize_tool_call_arguments(arguments: Any) -> dict[str, Any] | None:
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return None

    if not isinstance(arguments, dict):
        return None

    if set(arguments.keys()) == {"input"}:
        inner = arguments["input"]
        if isinstance(inner, str):
            try:
                inner = json.loads(inner)
            except json.JSONDecodeError:
                return arguments
        if isinstance(inner, dict):
            arguments = inner

    if set(arguments.keys()) == {"arguments"}:
        inner = arguments["arguments"]
        if isinstance(inner, str):
            try:
                inner = json.loads(inner)
            except json.JSONDecodeError:
                return arguments
            if isinstance(inner, dict):
                return inner

    return arguments


def decode_dsml_to_arguments(
    tool_name: str, tool_args: dict[str, tuple[str, str]]
) -> dict[str, str]:
    def _decode_value(key: str, value: str, string: str):
        if string == "true":
            value = to_json(value)
        return f"{to_json(key)}: {value}"

    tool_args_json = (
        "{"
        + ", ".join(
            [
                _decode_value(_unescape_param_name(k), v, string=is_str)
                for k, (v, is_str) in tool_args.items()
            ]
        )
        + "}"
    )
    return dict(name=tool_name, arguments=tool_args_json)


def render_tools(tools: list[dict[str, str | dict[str, Any]]]) -> str:
    tools_json = [to_json(_escape_tool_schema(t)) for t in tools]

    return TOOLS_TEMPLATE.format(
        tool_schemas="\n".join(tools_json),
        dsml_token=dsml_token,
        thinking_start_token=thinking_start_token,
        thinking_end_token=thinking_end_token,
    )


def find_last_user_index(messages: list[dict[str, Any]]) -> int:
    last_user_index = -1
    for idx in range(len(messages) - 1, -1, -1):
        if messages[idx].get("role") in ["user", "developer"]:
            last_user_index = idx
            break
    return last_user_index


def render_message(
    index: int,
    messages: list[dict[str, Any]],
    thinking_mode: str,
    drop_thinking: bool = True,
    reasoning_effort: str | None = None,
) -> str:
    assert 0 <= index < len(messages)
    assert thinking_mode in ["chat", "thinking"], f"Invalid thinking_mode `{thinking_mode}`"

    prompt = ""
    msg = messages[index]
    last_user_idx = find_last_user_index(messages)

    role = msg.get("role")
    content = msg.get("content")
    tools = msg.get("tools")
    response_format = msg.get("response_format")
    tool_calls = msg.get("tool_calls")
    reasoning = msg.get("reasoning")
    wo_eos = msg.get("wo_eos", False)

    if tools:
        tools = tools_from_openai_format(tools)
    if tool_calls:
        tool_calls = tool_calls_from_openai_format(tool_calls)

    assert reasoning_effort in REASONING_EFFORT_MAX_VALUES | REASONING_EFFORT_NOOP_VALUES, (
        f"Invalid reasoning effort: {reasoning_effort}"
    )
    if (
        index == 0
        and thinking_mode == "thinking"
        and reasoning_effort in REASONING_EFFORT_MAX_VALUES
    ):
        prompt += REASONING_EFFORT_MAX

    if role == "system":
        prompt += system_msg_template.format(content=content or "")
        if tools:
            prompt += "\n\n" + render_tools(tools)
        if response_format:
            prompt += "\n\n" + response_format_template.format(
                schema=to_json(response_format)
            )

    elif role == "developer":
        assert content, f"Invalid message for role `{role}`: {msg}"

        content_developer = USER_SP_TOKEN + content
        if tools:
            content_developer += "\n\n" + render_tools(tools)
        if response_format:
            content_developer += "\n\n" + response_format_template.format(
                schema=to_json(response_format)
            )
        prompt += user_msg_template.format(content=content_developer)

    elif role == "user":
        prompt += USER_SP_TOKEN
        content_blocks = msg.get("content_blocks")
        if content_blocks:
            parts = []
            for block in content_blocks:
                block_type = block.get("type")
                if block_type == "text":
                    parts.append(block.get("text", ""))
                elif block_type == "tool_result":
                    tool_content = block.get("content", "")
                    if isinstance(tool_content, list):
                        text_parts = []
                        for item in tool_content:
                            if item.get("type") == "text":
                                text_parts.append(item.get("text", ""))
                            else:
                                text_parts.append(f"[Unsupported {item.get('type')}]")
                        tool_content = "\n\n".join(text_parts)
                    parts.append(tool_output_template.format(content=tool_content))
                else:
                    parts.append(f"[Unsupported {block_type}]")
            prompt += "\n\n".join(parts)
        else:
            prompt += content or ""

    elif role == "latest_reminder":
        prompt += LATEST_REMINDER_SP_TOKEN + latest_reminder_msg_template.format(
            content=content
        )

    elif role == "tool":
        raise NotImplementedError(
            "deepseek_v4 merges tool messages into user; please preprocess with merge_tool_messages()"
        )

    elif role == "assistant":
        thinking_part = ""
        tool_call_content = ""

        if tool_calls:
            tool_call_items = [
                tool_call_template.format(
                    dsml_token=dsml_token,
                    name=tool_call.get("name"),
                    arguments=encode_arguments_to_dsml(tool_call),
                )
                for tool_call in tool_calls
            ]
            tool_call_content += "\n\n" + tool_calls_template.format(
                dsml_token=dsml_token,
                tool_calls="\n".join(tool_call_items),
                tc_block_name=tool_calls_block_name,
            )

        summary_content = content or ""
        reasoning = reasoning or ""
        prev_has_task = index - 1 >= 0 and messages[index - 1].get("task") is not None

        if thinking_mode == "thinking" and not prev_has_task:
            if not drop_thinking or index > last_user_idx:
                thinking_part = (
                    thinking_template.format(reasoning=reasoning) + thinking_end_token
                )

        if wo_eos:
            prompt += assistant_msg_wo_eos_template.format(
                reasoning=thinking_part,
                content=summary_content,
                tool_calls=tool_call_content,
            )
        else:
            prompt += assistant_msg_template.format(
                reasoning=thinking_part,
                content=summary_content,
                tool_calls=tool_call_content,
            )
    else:
        raise NotImplementedError(f"Unknown role: {role}")

    if index + 1 < len(messages) and messages[index + 1].get("role") not in [
        "assistant",
        "latest_reminder",
    ]:
        return prompt

    task = messages[index].get("task")
    if task is not None:
        assert task in VALID_TASKS, (
            f"Invalid task: '{task}'. Valid tasks are: {list(VALID_TASKS)}"
        )
        task_sp_token = DS_TASK_SP_TOKENS[task]
        if task != "action":
            prompt += task_sp_token
        else:
            prompt += ASSISTANT_SP_TOKEN
            prompt += (
                thinking_end_token
                if thinking_mode != "thinking"
                else thinking_start_token
            )
            prompt += task_sp_token

    elif messages[index].get("role") in ["user", "developer"]:
        prompt += ASSISTANT_SP_TOKEN
        if thinking_mode == "thinking" and (
            not drop_thinking or index >= last_user_idx
        ):
            prompt += thinking_start_token
        else:
            prompt += thinking_end_token

    return prompt


def merge_tool_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []

    for message in messages:
        message = copy.deepcopy(message)
        role = message.get("role")

        if role == "tool":
            tool_block = {
                "type": "tool_result",
                "tool_use_id": message.get("tool_call_id", ""),
                "content": message.get("content", ""),
            }
            if (
                merged
                and merged[-1].get("role") == "user"
                and "content_blocks" in merged[-1]
            ):
                merged[-1]["content_blocks"].append(tool_block)
            else:
                merged.append({"role": "user", "content_blocks": [tool_block]})
        elif role == "user":
            text_block = {"type": "text", "text": message.get("content", "")}
            if (
                merged
                and merged[-1].get("role") == "user"
                and "content_blocks" in merged[-1]
                and merged[-1].get("task") is None
            ):
                merged[-1]["content_blocks"].append(text_block)
            else:
                new_message = {
                    "role": "user",
                    "content": message.get("content", ""),
                    "content_blocks": [text_block],
                }
                for key in ("task", "wo_eos", "mask"):
                    if key in message:
                        new_message[key] = message[key]
                merged.append(new_message)
        else:
            merged.append(message)

    return merged


def sort_tool_results_by_call_order(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    last_tool_call_order: dict[str, int] = {}

    for message in messages:
        role = message.get("role")
        if role == "assistant" and message.get("tool_calls"):
            last_tool_call_order = {}
            for idx, tool_call in enumerate(message["tool_calls"]):
                tool_call_id = tool_call.get("id") or tool_call.get("function", {}).get(
                    "id", ""
                )
                if tool_call_id:
                    last_tool_call_order[tool_call_id] = idx

        elif role == "user" and message.get("content_blocks"):
            tool_blocks = [
                block
                for block in message["content_blocks"]
                if block.get("type") == "tool_result"
            ]
            if len(tool_blocks) > 1 and last_tool_call_order:
                sorted_blocks = sorted(
                    tool_blocks,
                    key=lambda block: last_tool_call_order.get(
                        block.get("tool_use_id", ""), 0
                    ),
                )
                sorted_idx = 0
                new_blocks = []
                for block in message["content_blocks"]:
                    if block.get("type") == "tool_result":
                        new_blocks.append(sorted_blocks[sorted_idx])
                        sorted_idx += 1
                    else:
                        new_blocks.append(block)
                message["content_blocks"] = new_blocks

    return messages


def encode_messages(
    messages: list[dict[str, Any]],
    thinking_mode: str,
    context: list[dict[str, Any]] | None = None,
    drop_thinking: bool = True,
    add_default_bos_token: bool = True,
    reasoning_effort: str | None = None,
) -> str:
    context = context if context else []
    reasoning_effort = _normalize_reasoning_effort(reasoning_effort)

    messages = merge_tool_messages(messages)
    messages = sort_tool_results_by_call_order(context + messages)[len(context) :]
    if context:
        context = merge_tool_messages(context)
        context = sort_tool_results_by_call_order(context)

    full_messages = context + messages
    prompt = bos_token if add_default_bos_token and len(context) == 0 else ""

    effective_drop_thinking = drop_thinking
    if any(message.get("tools") for message in full_messages):
        effective_drop_thinking = False

    if thinking_mode == "thinking" and effective_drop_thinking:
        full_messages = _drop_thinking_messages(full_messages)
        num_to_render = len(full_messages) - len(_drop_thinking_messages(context))
        context_len = len(full_messages) - num_to_render
    else:
        num_to_render = len(messages)
        context_len = len(context)

    for idx in range(num_to_render):
        prompt += render_message(
            idx + context_len,
            full_messages,
            thinking_mode=thinking_mode,
            drop_thinking=effective_drop_thinking,
            reasoning_effort=reasoning_effort,
        )

    return prompt


def _drop_thinking_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    last_user_idx = find_last_user_index(messages)
    result = []
    keep_roles = {"user", "system", "tool", "latest_reminder", "direct_search_results"}

    for idx, message in enumerate(messages):
        role = message.get("role")
        if role in keep_roles or idx >= last_user_idx:
            result.append(message)
        elif role == "assistant":
            message = copy.copy(message)
            message.pop("reasoning", None)
            result.append(message)

    return result


def _normalize_reasoning_effort(reasoning_effort: str | None) -> str | None:
    if reasoning_effort in REASONING_EFFORT_MAX_VALUES:
        return "max"
    if reasoning_effort in REASONING_EFFORT_NOOP_VALUES:
        return None
    return None


def get_deepseek_v4_tokenizer(tokenizer: HfTokenizer) -> HfTokenizer:
    dsv4_tokenizer = copy.copy(tokenizer)

    added_vocab = tokenizer.get_added_vocab()
    added_vocab_size = len(added_vocab)
    tokenizer_vocab_size = tokenizer.vocab_size

    class _DeepseekV4Tokenizer(tokenizer.__class__):  # type: ignore
        def apply_chat_template(
            self,
            messages: list[Any],
            tools: list[dict[str, Any]] | None = None,
            **kwargs,
        ) -> str | list[int]:
            thinking = kwargs.get("thinking", False) or kwargs.get(
                "enable_thinking", False
            )
            thinking_mode = "thinking" if thinking else "chat"

            conversation = kwargs.get("conversation", messages)
            rendered_messages = conversation.copy()
            if tools is not None and len(tools) > 0:
                rendered_messages.insert(0, {"role": "system"})
                rendered_messages[0]["tools"] = tools

            reasoning_effort = _normalize_reasoning_effort(
                kwargs.get("reasoning_effort")
            )

            prompt_str = encode_messages(
                rendered_messages,
                thinking_mode=thinking_mode,
                drop_thinking=kwargs.get("drop_thinking", True),
                reasoning_effort=reasoning_effort,
            )

            if kwargs.get("tokenize", True):
                tokenizer_kwargs = {
                    key: kwargs[key]
                    for key in ("truncation", "max_length")
                    if key in kwargs
                }
                return self.encode(
                    prompt_str,
                    add_special_tokens=False,
                    **tokenizer_kwargs,
                )

            return prompt_str

        def num_special_tokens_to_add(self) -> int:
            return len(self.encode(""))

        def __len__(self) -> int:
            return tokenizer_vocab_size + added_vocab_size

        def get_added_vocab(self) -> dict[str, int]:
            return added_vocab.copy()

        def __reduce__(self):
            return get_deepseek_v4_tokenizer, (tokenizer,)

    _DeepseekV4Tokenizer.__name__ = f"DSV4{tokenizer.__class__.__name__}"

    dsv4_tokenizer.__class__ = _DeepseekV4Tokenizer
    return dsv4_tokenizer


class DeepseekV4Tokenizer(TokenizerLike):
    @classmethod
    def from_pretrained(
        cls,
        path_or_repo_id: str | Path,
        *args,
        **kwargs,
    ) -> HfTokenizer:
        tokenizer = PreTrainedTokenizerFast.from_pretrained(
            path_or_repo_id, *args, **kwargs
        )
        return get_cached_tokenizer(get_deepseek_v4_tokenizer(tokenizer))


class DeepseekV4Renderer(BaseRenderer[HfTokenizer]):
    @classmethod
    def from_config(
        cls,
        config: VllmConfig,
        tokenizer_kwargs: dict[str, Any],
    ) -> DeepseekV4Renderer:
        if config.model_config.skip_tokenizer_init:
            tokenizer = None
        else:
            tokenizer = cached_get_tokenizer(
                tokenizer_cls=DeepseekV4Tokenizer,
                **tokenizer_kwargs,
            )

        return cls(config, tokenizer)

    def render_messages(
        self,
        messages: list[ChatCompletionMessageParam],
        params: ChatParams,
    ) -> tuple[list[ConversationMessage], DictPrompt]:
        tokenizer = self.get_tokenizer()
        conversation, mm_data, mm_uuids = parse_chat_messages(
            messages,
            self.model_config,
            content_format="string",
            media_io_kwargs=params.media_io_kwargs,
            mm_processor_kwargs=params.mm_processor_kwargs,
        )

        prompt_raw = tokenizer.apply_chat_template(
            conversation=conversation,
            messages=messages,
            **params.get_apply_chat_template_kwargs(),
        )

        prompt = parse_dec_only_prompt(prompt_raw)
        if mm_data is not None:
            prompt["multi_modal_data"] = mm_data
        if mm_uuids is not None:
            prompt["multi_modal_uuids"] = mm_uuids

        return conversation, prompt

    async def render_messages_async(
        self,
        messages: list[ChatCompletionMessageParam],
        params: ChatParams,
    ) -> tuple[list[ConversationMessage], DictPrompt]:
        tokenizer = self.get_tokenizer()
        conversation, mm_data, mm_uuids = await parse_chat_messages_async(
            messages,
            self.model_config,
            content_format="string",
            media_io_kwargs=params.media_io_kwargs,
            mm_processor_kwargs=params.mm_processor_kwargs,
        )

        prompt_raw = tokenizer.apply_chat_template(
            conversation=conversation,
            messages=messages,
            **params.get_apply_chat_template_kwargs(),
        )

        prompt = parse_dec_only_prompt(prompt_raw)
        if mm_data is not None:
            prompt["multi_modal_data"] = mm_data
        if mm_uuids is not None:
            prompt["multi_modal_uuids"] = mm_uuids

        return conversation, prompt


def _partial_tag_overlap(text: str, tag: str) -> int:
    max_overlap = min(len(text), len(tag) - 1)
    for overlap in range(max_overlap, 0, -1):
        if text.endswith(tag[:overlap]):
            return overlap
    return 0


class DeepSeekV4ToolParser(ToolParser):
    tool_call_start_token: str = "<｜DSML｜tool_calls>"
    tool_call_end_token: str = "</｜DSML｜tool_calls>"

    def __init__(self, tokenizer: TokenizerLike):
        super().__init__(tokenizer)

        self.current_tool_id: str | None = None  # type: ignore[assignment]
        self.current_tool_index: int = 0
        self._sent_content_idx: int = 0

        self.tool_call_complete_regex = re.compile(
            re.escape(self.tool_call_start_token)
            + r"(.*?)"
            + re.escape(self.tool_call_end_token),
            re.DOTALL,
        )
        self.invoke_complete_regex = re.compile(
            r'<｜DSML｜invoke\s+name="([^"]+)"\s*>(.*?)</｜DSML｜invoke>', re.DOTALL
        )
        self.parameter_complete_regex = re.compile(
            r'<｜DSML｜parameter\s+name="([^"]+)"\s+string="(true|false)"\s*>(.*?)</｜DSML｜parameter>',
            re.DOTALL,
        )

        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the ToolParser constructor during construction."
            )

        logger.debug("vLLM Successfully import tool parser %s !", self.__class__.__name__)

    def adjust_request(self, request: ChatCompletionRequest) -> ChatCompletionRequest:
        request = super().adjust_request(request)
        if request.tools and request.tool_choice != "none":
            request.skip_special_tokens = False
        return request

    def _generate_tool_call_id(self) -> str:
        return f"call_{uuid.uuid4().hex[:24]}"

    @staticmethod
    def _function_name(tool) -> str | None:
        if isinstance(tool, dict):
            function = tool.get("function")
            if isinstance(function, dict):
                return function.get("name")
            return getattr(function, "name", None)
        return getattr(getattr(tool, "function", None), "name", None)

    @staticmethod
    def _function_parameters(tool):
        if isinstance(tool, dict):
            function = tool.get("function")
            if isinstance(function, dict):
                return function.get("parameters")
            return getattr(function, "parameters", None)
        return getattr(getattr(tool, "function", None), "parameters", None)

    @staticmethod
    def _convert_param_value_checked(value: str, param_type: str) -> Any:
        if value.lower() == "null":
            return None

        param_type = param_type.lower()
        if param_type in ["string", "str", "text"]:
            return value
        if param_type in ["integer", "int"]:
            return int(value)
        if param_type in ["number", "float"]:
            val = float(value)
            return val if val != int(val) else int(val)
        if param_type in ["boolean", "bool"]:
            value = value.strip()
            if value.lower() not in ["false", "0", "true", "1"]:
                raise ValueError("Invalid boolean value")
            return value.lower() in ["true", "1"]
        if param_type in ["object", "array"]:
            return json.loads(value)
        return json.loads(value)

    def _convert_param_value(self, value: str, param_type: str | list[str]) -> Any:
        if not isinstance(param_type, list):
            param_type = [param_type]
        for current_type in param_type:
            try:
                return self._convert_param_value_checked(value, current_type)
            except Exception:
                continue
        return value

    def _extract_param_name(self, param_name: str) -> str:
        if param_name == ESCAPED_ARGUMENTS_PARAM_NAME:
            return "arguments"
        return param_name

    def _get_param_config(
        self,
        request: ChatCompletionRequest | None,
        function_name: str | None,
    ) -> dict[str, dict]:
        if not request or not request.tools or not function_name:
            return {}

        for tool in request.tools:
            if self._function_name(tool) != function_name:
                continue
            params = self._function_parameters(tool)
            if isinstance(params, dict):
                properties = params.get("properties")
                if isinstance(properties, dict):
                    return properties
            return {}

        return {}

    def _coerce_param_value(
        self,
        value: str,
        *,
        string_attr: str,
        param_type,
    ):
        if string_attr == "true":
            return value
        if param_type:
            return self._convert_param_value(value, param_type)
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value

    @staticmethod
    def _repair_param_dict(
        param_dict: dict,
        param_config: dict[str, dict],
    ) -> dict:
        allowed = set(param_config.keys())
        for wrapper in ("arguments", "input"):
            if set(param_dict.keys()) != {wrapper} or wrapper in allowed:
                continue
            inner = param_dict[wrapper]
            if isinstance(inner, str):
                try:
                    inner = json.loads(inner)
                except json.JSONDecodeError:
                    return param_dict
            if isinstance(inner, dict) and set(inner.keys()).issubset(allowed):
                return inner
        return param_dict

    def _parse_invoke_params(
        self,
        invoke_str: str,
        request: ChatCompletionRequest | None = None,
        function_name: str | None = None,
    ) -> dict:
        param_config = self._get_param_config(request, function_name)
        param_dict = {}

        for param_name, string_attr, param_val in self.parameter_complete_regex.findall(
            invoke_str
        ):
            original_param_name = param_name
            param_name = self._extract_param_name(param_name)
            param_type = None
            if (
                original_param_name == ESCAPED_ARGUMENTS_PARAM_NAME
                and "arguments" in param_config
            ):
                param_type = param_config["arguments"].get("type")
            elif param_name in param_config and isinstance(param_config[param_name], dict):
                param_type = param_config[param_name].get("type")

            param_dict[param_name] = self._coerce_param_value(
                param_val,
                string_attr=string_attr,
                param_type=param_type,
            )

        return self._repair_param_dict(param_dict, param_config)

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        if self.tool_call_start_token not in model_output:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        try:
            tool_calls = []
            for tool_call_match in self.tool_call_complete_regex.findall(model_output):
                for invoke_name, invoke_content in self.invoke_complete_regex.findall(
                    tool_call_match
                ):
                    param_dict = self._parse_invoke_params(
                        invoke_content,
                        request=request,
                        function_name=invoke_name,
                    )
                    tool_calls.append(
                        ToolCall(
                            type="function",
                            function=FunctionCall(
                                name=invoke_name,
                                arguments=json.dumps(param_dict, ensure_ascii=False),
                            ),
                        )
                    )

            if not tool_calls:
                return ExtractedToolCallInformation(
                    tools_called=False, tool_calls=[], content=model_output
                )

            first_tool_idx = model_output.find(self.tool_call_start_token)
            content = model_output[:first_tool_idx] if first_tool_idx > 0 else None

            return ExtractedToolCallInformation(
                tools_called=True, tool_calls=tool_calls, content=content
            )

        except Exception:
            logger.exception("Error extracting DeepSeek V4 tool calls")
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

    def _reset_streaming_state(self):
        self.current_tool_index = 0
        self._sent_content_idx = 0
        self.prev_tool_call_arr.clear()
        self.streamed_args_for_tool.clear()

    def _extract_delta_tool_calls(
        self,
        current_text: str,
        request: ChatCompletionRequest | None,
    ) -> list[DeltaToolCall]:
        complete_invokes = self.invoke_complete_regex.findall(current_text)
        delta_tool_calls: list[DeltaToolCall] = []

        while len(complete_invokes) > self.current_tool_index:
            invoke_name, invoke_body = complete_invokes[self.current_tool_index]
            param_dict = self._parse_invoke_params(
                invoke_body,
                request=request,
                function_name=invoke_name,
            )
            args_json = json.dumps(param_dict, ensure_ascii=False)
            idx = self.current_tool_index
            self.current_tool_index += 1

            self.prev_tool_call_arr.append({"name": invoke_name, "arguments": param_dict})
            self.streamed_args_for_tool.append(args_json)

            delta_tool_calls.append(
                DeltaToolCall(
                    index=idx,
                    id=self._generate_tool_call_id(),
                    function=DeltaFunctionCall(
                        name=invoke_name,
                        arguments=args_json,
                    ),
                    type="function",
                )
            )

        return delta_tool_calls

    def _extract_content(self, current_text: str) -> str | None:
        if self.tool_call_start_token not in current_text:
            overlap = _partial_tag_overlap(current_text, self.tool_call_start_token)
            sendable_idx = len(current_text) - overlap
        else:
            sendable_idx = current_text.index(self.tool_call_start_token)

        if sendable_idx > self._sent_content_idx:
            content = current_text[self._sent_content_idx : sendable_idx]
            self._sent_content_idx = sendable_idx
            return content
        return None

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        if not previous_text:
            self._reset_streaming_state()

        content = self._extract_content(current_text)
        delta_tool_calls = self._extract_delta_tool_calls(current_text, request)

        if (
            not delta_text
            and self.tool_call_start_token not in current_text
            and self._sent_content_idx < len(current_text)
        ):
            held_content = current_text[self._sent_content_idx :]
            self._sent_content_idx = len(current_text)
            content = (content or "") + held_content

        if delta_tool_calls or content:
            return DeltaMessage(content=content, tool_calls=delta_tool_calls)

        if not delta_text and delta_token_ids and self.prev_tool_call_arr:
            return DeltaMessage(content="")

        return None


_patch_chat_completion_reasoning_effort()
TokenizerRegistry.register(
    "deepseek_v4",
    "vllm_ascend.patch.platform.patch_deepseek_v4_agentic",
    "DeepseekV4Tokenizer",
)
RENDERER_REGISTRY.register(
    "deepseek_v4",
    "vllm_ascend.patch.platform.patch_deepseek_v4_agentic",
    "DeepseekV4Renderer",
)
ToolParserManager.register_lazy_module(
    "deepseek_v4",
    "vllm_ascend.patch.platform.patch_deepseek_v4_agentic",
    "DeepSeekV4ToolParser",
)
ReasoningParserManager.register_lazy_module(
    "deepseek_v4",
    "vllm.reasoning.deepseek_v3_reasoning_parser",
    "DeepSeekV3ReasoningParser",
)

sys.modules.setdefault("vllm.tokenizers.deepseek_v4", sys.modules[__name__])
sys.modules.setdefault("vllm.tokenizers.deepseek_v4_encoding", sys.modules[__name__])
sys.modules.setdefault("vllm.tool_parsers.deepseekv4_tool_parser", sys.modules[__name__])
