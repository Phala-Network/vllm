# SPDX-License-Identifier: Apache-2.0

import json
from unittest.mock import MagicMock

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.parser.abstract_parser import DelegatingParser
from vllm.parser.gemma4 import (
    CHANNEL_END,
    CHANNEL_START,
    TOOL_CALL_END,
    TOOL_CALL_START,
)
from vllm.parser.parser_manager import ParserManager
from vllm.tool_parsers.gemma4_engine_tool_parser import Gemma4EngineToolParser


def _tokenizer():
    vocab = {
        TOOL_CALL_START: 48,
        TOOL_CALL_END: 49,
        CHANNEL_START: 50,
        CHANNEL_END: 51,
    }
    decode_map = {value: key for key, value in vocab.items()}
    tokenizer = MagicMock()
    tokenizer.encode.return_value = [1, 2, 3]
    tokenizer.get_vocab.return_value = vocab
    tokenizer.decode.side_effect = lambda ids: decode_map.get(ids[0], f"tok{ids[0]}")
    return tokenizer


def _request():
    request = MagicMock(spec=ChatCompletionRequest)
    request.tools = []
    request.tool_choice = "auto"
    request.include_reasoning = True
    return request


def _collect_name_and_arguments(parser, chunks):
    previous_text = ""
    previous_token_ids = []
    name = None
    arguments = ""
    special = {
        TOOL_CALL_START: 48,
        TOOL_CALL_END: 49,
        CHANNEL_START: 50,
        CHANNEL_END: 51,
    }

    for chunk in chunks:
        current_text = previous_text + chunk
        found = []
        for token, token_id in special.items():
            start = 0
            while True:
                index = chunk.find(token, start)
                if index < 0:
                    break
                found.append((index, token_id))
                start = index + len(token)
        found.sort()
        delta_token_ids = [token_id for _, token_id in found] or [0]
        current_token_ids = previous_token_ids + delta_token_ids
        delta = parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=chunk,
            previous_token_ids=tuple(previous_token_ids),
            current_token_ids=tuple(current_token_ids),
            delta_token_ids=tuple(delta_token_ids),
            request=_request(),
        )
        if delta and delta.tool_calls:
            for tool_call in delta.tool_calls:
                function = tool_call.function
                if isinstance(function, dict):
                    name = function.get("name") or name
                    arguments += function.get("arguments", "") or ""
                else:
                    name = getattr(function, "name", None) or name
                    arguments += getattr(function, "arguments", "") or ""
        previous_text = current_text
        previous_token_ids = current_token_ids

    return name, arguments


def test_parenthesized_tool_call_non_streaming():
    parser = Gemma4EngineToolParser(_tokenizer(), tools=[])
    output = '<|tool_call>call:terminal(command:<|"|>ls -a<|"|>)<tool_call|>'
    result = parser.extract_tool_calls(output, _request())

    assert result.tools_called is True
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].function.name == "terminal"
    assert json.loads(result.tool_calls[0].function.arguments) == {"command": "ls -a"}


def test_parenthesized_tool_call_streaming():
    parser = Gemma4EngineToolParser(_tokenizer(), tools=[])
    name, arguments = _collect_name_and_arguments(
        parser,
        [
            "<|tool_call>",
            "call:terminal(",
            'command:<|"|>ls -a<|"|>)',
            "<tool_call|>",
        ],
    )

    assert name == "terminal"
    assert json.loads(arguments) == {"command": "ls -a"}


def test_shared_gemma4_engine_preserves_reasoning_and_tool_adapters():
    parser_cls = ParserManager.get_parser(
        tool_parser_name="gemma4",
        reasoning_parser_name="gemma4",
        enable_auto_tools=True,
    )

    assert parser_cls is not None
    assert issubclass(parser_cls, DelegatingParser)
    parser = parser_cls(
        _tokenizer(),
        tools=[],
        chat_template_kwargs={"enable_thinking": False},
    )
    assert parser.reasoning_parser is not None
    assert parser.tool_parser is not None
    assert parser.reasoning_parser._parser_engine._thinking_enabled is False
