# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Barrier
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import vllm.v1.structured_output as module
from vllm.v1.structured_output.backend_types import StructuredOutputOptions


def make_manager(async_compile, vocab_size=256):
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(distributed_executor_backend="mp"),
        scheduler_config=SimpleNamespace(max_num_seqs=4),
        model_config=SimpleNamespace(
            skip_tokenizer_init=True,
            get_vocab_size=lambda: vocab_size,
            is_diffusion=False,
        ),
        structured_outputs_config=SimpleNamespace(
            enable_in_reasoning=False,
            disable_any_whitespace=False,
            disable_additional_properties=False,
        ),
        speculative_config=None,
        num_speculative_tokens=0,
    )
    manager = module.StructuredOutputManager(config)
    manager.tokenizer = None
    manager._use_async_grammar_compilation = async_compile
    manager.executor = ThreadPoolExecutor(max_workers=2)
    return manager


def make_request(backend, index, kind=StructuredOutputOptions.JSON, spec="{}"):
    return SimpleNamespace(
        request_id=f"{backend}-{index}",
        sampling_params=SimpleNamespace(
            structured_outputs=SimpleNamespace(_backend=backend),
            all_stop_token_ids={index},
        ),
        structured_output_request=SimpleNamespace(
            structured_output_key=(kind, spec), grammar=None, reasoning_ended=True
        ),
    )


def compiled(request):
    result = request.structured_output_request.grammar
    if isinstance(result, Future):
        result = result.result(timeout=30)
    request.structured_output_request.grammar = result
    return result


@pytest.mark.parametrize("order", [("guidance", "xgrammar"), ("xgrammar", "guidance")])
@pytest.mark.parametrize("async_compile", [False, True])
def test_request_backend_isolation(monkeypatch, order, async_compile):
    # The barrier makes both backends compile concurrently, without sleeps.
    barrier = Barrier(2) if async_compile else None
    instances = {}
    factories = {}
    for name in order:
        instance = Mock()

        def compile_grammar(kind, spec, *, stop_token_ids, name=name):
            if barrier is not None:
                barrier.wait(timeout=10)
            return name, spec, stop_token_ids

        instance.compile_grammar.side_effect = compile_grammar
        instances[name] = instance
        factories[name] = Mock(return_value=instance)
        monkeypatch.setattr(module, name.capitalize() + "Backend", factories[name])

    manager = make_manager(async_compile)
    requests = [make_request(name, i, spec=str(i)) for i, name in enumerate(order * 2)]
    try:
        for request in requests:
            manager.grammar_init(request)
        for i, request in enumerate(requests):
            assert compiled(request) == (order[i % 2], str(i), {i})
        for factory in factories.values():
            factory.assert_called_once()
    finally:
        manager.executor.shutdown(wait=True)
        manager.clear_backend()
    for instance in instances.values():
        instance.destroy.assert_called_once()
    manager.clear_backend()
    for instance in instances.values():
        instance.destroy.assert_called_once()


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(os.environ.get("VLLM_TEST_TOKENIZER", "gpt2"))


@pytest.mark.parametrize("order", [("guidance", "xgrammar"), ("xgrammar", "guidance")])
@pytest.mark.parametrize("async_compile", [False, True])
def test_real_mixed_grammars_and_bitmasks(tokenizer, order, async_compile):
    manager = make_manager(async_compile, len(tokenizer))
    manager.tokenizer = tokenizer
    requests = {
        "guidance": make_request(
            "guidance",
            0,
            spec='{"type":"object","properties":{"city":{"const":"Paris"}},'
            '"required":["city"],"additionalProperties":false}',
        ),
        "xgrammar": make_request(
            "xgrammar",
            1,
            StructuredOutputOptions.STRUCTURAL_TAG,
            '{"type":"structural_tag","format":{"type":"const_string","value":"tool:Paris"}}',
        ),
    }
    outputs = {"guidance": '{"city":"Paris"}', "xgrammar": "tool:Paris"}
    try:
        for name in order:
            requests[name].sampling_params.all_stop_token_ids = {tokenizer.eos_token_id}
            manager.grammar_init(requests[name])
        for request in requests.values():
            compiled(request)
        masks = manager.grammar_bitmask(requests, list(order), {})
        assert masks is not None
        # Independently allocated masks must match rows in the shared batch.
        for row, name in enumerate(order):
            backend = manager._backends[name]
            own_mask = backend.allocate_token_bitmask(1)
            grammar = requests[name].structured_output_request.grammar
            grammar.fill_bitmask(own_mask, 0)
            assert torch.equal(torch.from_numpy(masks[row]), own_mask[0])
            tokens = tokenizer.encode(outputs[name], add_special_tokens=False)
            assert grammar.validate_tokens(tokens) == tokens
            assert grammar.accept_tokens(requests[name].request_id, tokens)
    finally:
        manager.executor.shutdown(wait=True)
        manager.clear_backend()


def test_guidance_compilation_keeps_schema_local(monkeypatch):
    import vllm.v1.structured_output.backend_guidance as guidance

    barrier = Barrier(2)

    class Backend(guidance.GuidanceBackend):
        def __setattr__(self, name, value):
            super().__setattr__(name, value)
            # Force the old shared-field implementation to overwrite the first
            # schema before either caller can consume it.
            if name == "serialized_grammar":
                barrier.wait(timeout=10)

    backend = object.__new__(Backend)
    backend.disable_any_whitespace = False
    backend.disable_additional_properties = False
    backend.ll_tokenizer = object()
    backend.vocab_size = 256
    monkeypatch.setattr(guidance, "serialize_guidance_grammar", lambda *args: args[1])
    monkeypatch.setattr(
        guidance,
        "llguidance",
        SimpleNamespace(LLMatcher=lambda tokenizer, schema, **kw: schema),
    )
    monkeypatch.setattr(
        guidance,
        "GuidanceGrammar",
        lambda **kw: SimpleNamespace(schema=kw["ll_matcher"], check_error=lambda: None),
    )
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                backend.compile_grammar, StructuredOutputOptions.JSON, schema
            )
            for schema in ("schema-a", "schema-b")
        ]
        assert [f.result(timeout=15).schema for f in futures] == [
            "schema-a",
            "schema-b",
        ]
