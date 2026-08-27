# SPDX-License-Identifier: Apache-2.0
"""Focused regressions imported from open upstream protocol PRs.

These cases stay model- and request-path specific so a related upstream title
does not become a blanket backport into the Gemma4 production fork.
"""

import json
from unittest.mock import MagicMock

import pytest

from tests.parser.engine.streaming_helpers import (
    collect_function_name,
    collect_tool_arguments,
    simulate_tool_streaming,
)
from vllm.config import StructuredOutputsConfig
from vllm.entrypoints.chat_utils import _postprocess_messages
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.exceptions import VLLMValidationError
from vllm.parser.gemma4 import (
    CHANNEL_END,
    CHANNEL_START,
    TOOL_CALL_END,
    TOOL_CALL_START,
    Gemma4Parser,
    _parse_gemma4_args,
    _parse_gemma4_array,
)
from vllm.sampling_params import SamplingParams, StructuredOutputsParams
from vllm.tool_parsers.gemma4_engine_tool_parser import Gemma4EngineToolParser
from vllm.v1.structured_output.backend_xgrammar import (
    has_xgrammar_unsupported_json_features,
)


def _tokenizer():
    vocab = {
        TOOL_CALL_START: 48,
        TOOL_CALL_END: 49,
        CHANNEL_START: 50,
        CHANNEL_END: 51,
        "<|turn>": 52,
        "<|tool_response>": 53,
    }
    decode_map = {value: key for key, value in vocab.items()}
    tokenizer = MagicMock()
    tokenizer.encode.return_value = [1, 2, 3]
    tokenizer.get_vocab.return_value = vocab
    tokenizer.decode.side_effect = lambda ids, **_: "".join(
        decode_map.get(token_id, f"tok{token_id}") for token_id in ids
    )
    return tokenizer


def _request(*, tool_choice="auto", tools=None):
    request = MagicMock(spec=ChatCompletionRequest)
    request.tools = [] if tools is None else tools
    request.tool_choice = tool_choice
    request.include_reasoning = True
    return request


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('opts:{"mode": "fast"}', {"opts": {"mode": "fast"}}),
        ('data_refs=["ds_a"]', {"data_refs": ["ds_a"]}),
        ('a:"x,y",b:1', {"a": "x,y", "b": 1}),
        ('opts:{"pattern":"a}b"}', {"opts": {"pattern": "a}b"}}),
        ("name:'ds_152a4bfd'", {"name": "ds_152a4bfd"}),
        (
            'code:<|"|>x<|"|>,data_refs=["ds_a"],note:<|"|>hi<|"|>',
            {"code": "x", "data_refs": ["ds_a"], "note": "hi"},
        ),
        ('<|"|>a=b<|"|>:v', {"a=b": "v"}),
        ('"a:b":1', {"a:b": 1}),
        ('xs:["a]b","c"]', {"xs": ["a]b", "c"]}),
    ],
)
def test_pr51307_json_python_literals_in_tool_arguments(raw, expected):
    assert _parse_gemma4_args(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('"a,b","c"', ["a,b", "c"]),
        ("'DUPE_RESULTS','EVASION_RESULTS'", ["DUPE_RESULTS", "EVASION_RESULTS"]),
        ('"42"', ["42"]),
        (r"'a\nb'", ["a\nb"]),
        (r'"a\u00e9b"', ["a\u00e9b"]),
    ],
)
def test_pr51307_json_python_literals_in_tool_arrays(raw, expected):
    assert _parse_gemma4_array(raw) == expected


def test_pr51307_unterminated_fallback_quote_is_withheld_while_streaming():
    assert _parse_gemma4_array('"abc', partial=True) == []
    assert _parse_gemma4_args('name:"abc', partial=True) == {}


@pytest.mark.parametrize(
    "output",
    [
        '<|tool_call>:get_weather{location:<|"|>London<|"|>}<tool_call|>',
        "<|tool_call>:set_status{}<tool_call|>",
    ],
)
def test_pr53444_bare_tool_opener_is_parsed(output):
    parser = Gemma4EngineToolParser(_tokenizer(), tools=[])
    result = parser.extract_tool_calls(output, _request())

    assert result.tools_called is True
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].function.name in {"get_weather", "set_status"}


def test_pr53444_colon_in_plain_content_is_not_a_tool_call():
    parser = Gemma4EngineToolParser(_tokenizer(), tools=[])
    result = parser.extract_tool_calls("status: healthy", _request())

    assert result.tools_called is False
    assert result.content == "status: healthy"


def test_pr53444_documented_and_malformed_calls_remain_bounded():
    parser = Gemma4EngineToolParser(_tokenizer(), tools=[])
    result = parser.extract_tool_calls(
        "<|tool_call>:bad no brace<tool_call|>"
        '<|tool_call>call:get_weather{location:<|"|>London<|"|>}'
        "<tool_call|>",
        _request(),
    )

    assert result.tools_called is True
    assert [call.function.name for call in result.tool_calls] == [
        "bad no brace",
        "get_weather",
    ]


def _assistant_tool_call(arguments):
    return [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_0",
                    "type": "function",
                    "function": {"name": "write", "arguments": arguments},
                }
            ],
        }
    ]


@pytest.mark.parametrize(
    "arguments",
    [
        '{"cmd": "unterminated',
        [],
        "[]",
        "42",
        "true",
        '"hello"',
        "null",
    ],
)
def test_pr48922_bad_historical_tool_arguments_are_recoverable(arguments):
    messages = _assistant_tool_call(arguments)
    _postprocess_messages(messages)
    assert messages[0]["tool_calls"][0]["function"]["arguments"] == {}


def test_pr48922_valid_historical_tool_arguments_stay_objects():
    messages = _assistant_tool_call('{"path": "README.md"}')
    _postprocess_messages(messages)
    assert messages[0]["tool_calls"][0]["function"]["arguments"] == {
        "path": "README.md"
    }


class _StubModelConfig:
    is_diffusion = False


def _validate_structured_outputs(params):
    SamplingParams(structured_outputs=params)._validate_structured_outputs(
        _StubModelConfig(),
        StructuredOutputsConfig(backend="auto"),
        tokenizer=object(),
    )


@pytest.mark.parametrize(
    ("params", "message"),
    [
        (StructuredOutputsParams(regex=""), "regex cannot be an empty string"),
        (StructuredOutputsParams(regex="  "), "regex cannot be an empty string"),
        (
            StructuredOutputsParams(structural_tag=""),
            "structural_tag cannot be an empty string",
        ),
        (
            StructuredOutputsParams(structural_tag="\t"),
            "structural_tag cannot be an empty string",
        ),
        (
            StructuredOutputsParams(json={}),
            "json cannot be an empty JSON schema",
        ),
        (
            StructuredOutputsParams(json="{}"),
            "json cannot be an empty JSON schema",
        ),
    ],
)
def test_pr47450_pr52020_degenerate_constraints_rejected(params, message):
    with pytest.raises(VLLMValidationError, match=message):
        _validate_structured_outputs(params)


def test_nonempty_json_schema_remains_valid():
    _validate_structured_outputs(
        StructuredOutputsParams(
            json={
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
                "additionalProperties": False,
            }
        )
    )


@pytest.mark.parametrize(
    "schema",
    [
        {"type": ["integer", "null"], "multipleOf": 3},
        {"type": ["array", "null"], "uniqueItems": True},
        {"type": ["string", "null"], "format": "unknown-format"},
        {"type": ["object", "null"], "patternProperties": {"^x": {}}},
    ],
)
def test_pr48416_list_type_does_not_bypass_xgrammar_feature_gate(schema):
    assert has_xgrammar_unsupported_json_features(schema)


@pytest.mark.parametrize(
    "schema",
    [
        {"type": ["string", "null"]},
        {"type": ["string", "null"], "format": "email"},
        {"type": ["integer", "null"]},
        {"type": ["array", "null"], "items": {"type": "string"}},
    ],
)
def test_pr48416_supported_list_types_are_not_overflagged(schema):
    assert not has_xgrammar_unsupported_json_features(schema)


@pytest.mark.parametrize(
    "schema",
    [
        {"type": "integer", "multipleOf": 3},
        {"type": ["integer", "null"], "multipleOf": 3},
    ],
)
def test_pr48416_multiple_of_resolves_to_guidance(schema):
    params = SamplingParams(structured_outputs=StructuredOutputsParams(json=schema))
    params._validate_structured_outputs(
        _StubModelConfig(),
        StructuredOutputsConfig(backend="auto"),
        tokenizer=object(),
    )

    assert params.structured_outputs is not None
    assert params.structured_outputs._backend == "guidance"


def test_pr48416_supported_list_type_stays_on_xgrammar():
    params = SamplingParams(
        structured_outputs=StructuredOutputsParams(
            json={"type": ["integer", "null"], "const": 6}
        )
    )
    params._validate_structured_outputs(
        _StubModelConfig(),
        StructuredOutputsConfig(backend="auto"),
        tokenizer=object(),
    )

    assert params.structured_outputs is not None
    assert params.structured_outputs._backend == "xgrammar"


def test_pr47562_truncated_native_call_is_dropped_non_streaming():
    parser = Gemma4EngineToolParser(_tokenizer(), tools=[])
    result = parser.extract_tool_calls(
        '<|tool_call>call:get_weather{location:<|"|>Lon',
        _request(),
    )

    assert result.tools_called is False
    assert result.tool_calls == []
    assert not result.content


def test_pr50015_non_streaming_parse_is_seeded_from_open_reasoning_prompt():
    tokenizer = _tokenizer()
    parser = Gemma4Parser(tokenizer, chat_template_kwargs={"enable_thinking": True})
    request = _request()

    delta = parser.parse_delta(
        "still reasoning",
        [1001],
        request,
        prompt_token_ids=[50, 1001],
        finished=True,
    )

    assert delta is not None
    assert delta.reasoning == "still reasoning"
    assert delta.content is None
    assert not delta.tool_calls


def test_required_prose_is_not_promoted_to_an_upstream_parser_error():
    parser = Gemma4Parser(
        _tokenizer(),
        chat_template_kwargs={"enable_thinking": False},
    )
    request = _request(tool_choice="required")

    _, content, tool_calls = parser.parse("plain fallback answer", request)

    assert content == "plain fallback answer"
    assert not tool_calls


def _nested_gemma4_object(depth):
    return "value:" + "{nested:" * depth + "1" + "}" * depth


def _nested_object_leaf(parsed, depth):
    value = parsed["value"]
    for _ in range(depth):
        value = value["nested"]
    return value


def _nested_native_tool_call(depth):
    nested = "{nested:" * depth + "1" + "}" * depth
    return f"<|tool_call>call:deep_tool{{value:{nested}}}<tool_call|>"


def test_pr50944_python_parser_accepts_128_and_rejects_129_objects():
    parsed = _parse_gemma4_args(_nested_gemma4_object(128))
    assert _nested_object_leaf(parsed, 128) == 1

    with pytest.raises(ValueError, match="maximum nesting depth of 128"):
        _parse_gemma4_args(_nested_gemma4_object(129))

    with pytest.raises(ValueError, match="maximum nesting depth of 128"):
        _parse_gemma4_args(_nested_gemma4_object(129), partial=True)


def test_pr50944_python_parser_accepts_128_and_rejects_129_arrays():
    parsed = _parse_gemma4_args("value:" + "[" * 128 + "1" + "]" * 128)
    value = parsed["value"]
    for _ in range(128):
        assert isinstance(value, list)
        value = value[0]
    assert value == 1

    with pytest.raises(ValueError, match="maximum nesting depth of 128"):
        _parse_gemma4_args("value:" + "[" * 129 + "1" + "]" * 129)


def test_pr50944_python_parser_counts_mixed_container_depth():
    parsed = _parse_gemma4_args("value:" + "[{nested:" * 64 + "1" + "}]" * 64)
    value = parsed["value"]
    for _ in range(64):
        value = value[0]["nested"]
    assert value == 1

    with pytest.raises(ValueError, match="maximum nesting depth of 128"):
        _parse_gemma4_args("value:" + "[{nested:" * 64 + "[1]" + "}]" * 64)


def test_pr50944_python_parser_ignores_brackets_inside_strings():
    value = "{" * 129 + "[" * 129
    assert _parse_gemma4_args(f'value:<|"|>{value}<|"|>') == {"value": value}


def test_pr50944_python_parser_allows_many_sibling_containers():
    raw = ",".join(f"v{i}:{{}}" for i in range(256))
    parsed = _parse_gemma4_args(raw)
    assert parsed == {f"v{i}": {} for i in range(256)}


def test_pr50944_python_parser_drops_excessive_complete_call_non_streaming():
    parser = Gemma4Parser(
        _tokenizer(),
        chat_template_kwargs={"enable_thinking": False},
    )

    reasoning, content, tool_calls = parser.parse(
        _nested_native_tool_call(129), _request()
    )

    assert reasoning is None
    assert content is None
    assert not tool_calls


def test_pr50944_python_parser_drops_excessive_complete_call_streaming():
    parser = Gemma4Parser(
        _tokenizer(),
        chat_template_kwargs={"enable_thinking": False},
    )
    raw = _nested_native_tool_call(129)
    results = simulate_tool_streaming(parser, _request(), [raw])
    finish = parser.finish_streaming()
    if finish is not None:
        results.append((finish, raw))

    assert collect_function_name(results) is None
    assert collect_tool_arguments(results) == ""


def test_pr50944_python_parser_keeps_boundary_call_stream_parity():
    raw = _nested_native_tool_call(128)
    parser = Gemma4Parser(
        _tokenizer(),
        chat_template_kwargs={"enable_thinking": False},
    )
    _, _, tool_calls = parser.parse(raw, _request())
    assert len(tool_calls) == 1
    non_stream_args = json.loads(tool_calls[0].arguments)
    assert _nested_object_leaf(non_stream_args, 128) == 1

    parser = Gemma4Parser(
        _tokenizer(),
        chat_template_kwargs={"enable_thinking": False},
    )
    results = simulate_tool_streaming(parser, _request(), [raw])
    stream_args = json.loads(collect_tool_arguments(results))

    assert collect_function_name(results) == "deep_tool"
    assert stream_args == non_stream_args
