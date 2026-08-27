# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time
from concurrent.futures import Future
from unittest.mock import MagicMock

import pytest
from transformers import AutoTokenizer

from vllm.config import DeviceConfig, StructuredOutputsConfig, VllmConfig
from vllm.config.model import ModelConfig
from vllm.config.parallel import ParallelConfig
from vllm.config.speculative import SpeculativeConfig
from vllm.sampling_params import SamplingParams, StructuredOutputsParams
from vllm.tokenizers import get_tokenizer
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager
from vllm.v1.structured_output.backend_guidance import GuidanceBackend
from vllm.v1.structured_output.backend_outlines import OutlinesGrammar
from vllm.v1.structured_output.backend_types import StructuredOutputOptions

TOKENIZER = "openai-community/gpt2"


@pytest.fixture(scope="module")
def mistral_tokenizer():
    return get_tokenizer(
        tokenizer_name="mistralai/Mistral-Small-3.2-24B-Instruct-2506",
        tokenizer_mode="mistral",
    )


def test_backend_guidance_rollback_terminated():
    # Test that the backend guidance successfully rollbacks from a
    # terminated state. This can happen with speculative decoding,
    # where the draft model proposes EOS and it is verified by the
    # guidance backend. In that case we are in a stopped state, but
    # it should be reverted in case EOS is not accepted by the target
    # model.
    structured_outputs_config = StructuredOutputsConfig(backend="guidance")
    vllm_config = VllmConfig(structured_outputs_config=structured_outputs_config)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)

    backend = GuidanceBackend(
        vllm_config,
        tokenizer=tokenizer,
        vocab_size=50257,
    )

    grammar = backend.compile_grammar(
        StructuredOutputOptions.JSON, '{"type": "object"}'
    )

    prompt = tokenizer.encode('{"a": "b"}')
    assert len(prompt) > 1
    dummy_wrong = tokenizer.encode('{"a"}')
    for token in prompt:
        assert grammar.accept_tokens("", [token])
    assert not grammar.is_terminated()
    assert grammar.accept_tokens("", [tokenizer.eos_token_id])
    assert grammar.is_terminated()
    # Giving any other token should also be accepted
    assert grammar.accept_tokens("", dummy_wrong)
    # Rollback is done from where state was terminated, so from '}' not EOS
    grammar.rollback(len(prompt) - 1)
    assert not grammar.is_terminated()
    assert grammar.validate_tokens([tokenizer.eos_token_id]) == []
    assert grammar.validate_tokens(dummy_wrong) != dummy_wrong
    assert grammar.accept_tokens("", prompt[1:])
    assert not grammar.is_terminated()
    assert grammar.accept_tokens("", [tokenizer.eos_token_id])
    assert grammar.is_terminated()
    # Rollback of <= 0 should not change the terminated state
    grammar.rollback(0)
    assert grammar.is_terminated()
    grammar.rollback(-1)
    assert grammar.is_terminated()


def test_grammar_bitmask_with_specdec():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    prompt = tokenizer.encode('{"a": "b"}')
    vllm_config = VllmConfig(
        model_config=ModelConfig(tokenizer=TOKENIZER),
        structured_outputs_config=StructuredOutputsConfig(backend="guidance"),
        speculative_config=SpeculativeConfig(model="[ngram]", num_speculative_tokens=3),
    )
    structured_output_manager = StructuredOutputManager(vllm_config)

    for i in range(1, 2):
        sampling_params = SamplingParams(
            structured_outputs=StructuredOutputsParams(
                json='{"type": "object"}',
            ),
        )
        sampling_params.structured_outputs._backend = "guidance"
        sampling_params.update_from_generation_config({}, tokenizer.eos_token_id)

        my_req_id = f"my_req_id_{i}"
        request = Request(
            my_req_id,
            prompt_token_ids=prompt[:i],
            sampling_params=sampling_params,
            pooling_params=None,
        )

        structured_output_manager.grammar_init(request)

        def grammar_bitmask(req: Request, tokens: list[int]) -> None:
            structured_output_manager.grammar_bitmask(
                requests={req.request_id: req},
                structured_output_request_ids={req.request_id: 0},
                scheduled_spec_decode_tokens={req.request_id: tokens},
            )
            # At this point, we rolled-back, so should not be terminated
            assert not req.structured_output_request.grammar.is_terminated()

        # The grammar might not yet be compiled, so we wait for it
        while not request.structured_output_request._check_grammar_completion():
            continue

        assert request.structured_output_request.grammar.accept_tokens(
            request.request_id, prompt[:i]
        )

        grammar_bitmask(request, prompt[i:] + [tokenizer.eos_token_id])
        grammar_bitmask(
            request, prompt[i:] + [tokenizer.eos_token_id] + prompt
        )  # EOS not the final token
        grammar_bitmask(request, prompt[i:])  # EOS not present
        grammar_bitmask(request, prompt[i:] + [tokenizer.eos_token_id])


@pytest.mark.parametrize("async_grammar", [True, False])
def test_grammar_init_async_and_sync(async_grammar):
    """Test grammar initialization works correctly in both async and sync modes.

    This test validates that the distributed_executor_backend config option
    correctly controls whether grammar compilation happens asynchronously
    (via executor.submit) or synchronously. When set to "external_launcher",
    grammar compilation is synchronous to avoid deadlocks.
    """
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    prompt = tokenizer.encode('{"a": "b"}')

    # Use "external_launcher" for sync mode, None for async mode
    executor_backend = None if async_grammar else "external_launcher"
    vllm_config = VllmConfig(
        model_config=ModelConfig(tokenizer=TOKENIZER),
        structured_outputs_config=StructuredOutputsConfig(backend="guidance"),
        parallel_config=ParallelConfig(distributed_executor_backend=executor_backend),
    )
    structured_output_manager = StructuredOutputManager(vllm_config)

    sampling_params = SamplingParams(
        structured_outputs=StructuredOutputsParams(
            json='{"type": "object"}',
        ),
    )
    sampling_params.structured_outputs._backend = "guidance"
    sampling_params.update_from_generation_config({}, tokenizer.eos_token_id)

    request = Request(
        "test_request",
        prompt_token_ids=prompt,
        sampling_params=sampling_params,
        pooling_params=None,
    )

    structured_output_manager.grammar_init(request)

    # Check the internal _grammar type immediately after init
    # Before _check_grammar_completion is called, async mode should have a Future
    raw_grammar = request.structured_output_request._grammar
    if async_grammar:
        assert isinstance(raw_grammar, Future), (
            "Async mode should store a Future before completion"
        )
    else:
        assert not isinstance(raw_grammar, Future), (
            "Sync mode should store the grammar directly, not a Future"
        )

    # Wait for grammar to be ready (handles both async and sync cases)
    start_time = time.time()
    while not request.structured_output_request._check_grammar_completion():
        if time.time() - start_time > 5:  # 5-second timeout
            pytest.fail("Grammar compilation timed out")
        time.sleep(0.01)

    # After completion, _grammar should no longer be a Future
    assert not isinstance(request.structured_output_request._grammar, Future)

    # Verify grammar is properly initialized and functional
    grammar = request.structured_output_request.grammar
    assert grammar is not None
    assert not grammar.is_terminated()

    # Verify the grammar can accept valid tokens
    assert grammar.accept_tokens(request.request_id, prompt)


def test_manager_compiles_with_each_request_selected_backend(monkeypatch):
    class FakeBackend:
        def __init__(self, name):
            self.name = name

        def compile_grammar(
            self,
            request_type,
            grammar_spec,
            stop_token_ids=None,
            so_params=None,
        ):
            return (self.name, request_type, grammar_spec, stop_token_ids)

        def destroy(self):
            pass

    monkeypatch.setattr(
        "vllm.v1.structured_output.XgrammarBackend",
        lambda *args, **kwargs: FakeBackend("xgrammar"),
    )
    monkeypatch.setattr(
        "vllm.v1.structured_output.GuidanceBackend",
        lambda *args, **kwargs: FakeBackend("guidance"),
    )

    manager = StructuredOutputManager.__new__(StructuredOutputManager)
    manager.backend = None
    manager.backends = {}
    manager.vllm_config = MagicMock()
    manager.vllm_config.model_config.get_vocab_size.return_value = 256
    manager.tokenizer = MagicMock()
    manager._grammar_bitmask = None
    manager._use_async_grammar_compilation = False

    def make_request(backend_name):
        request = MagicMock()
        request.request_id = backend_name
        request.structured_output_request.structured_output_key = (
            StructuredOutputOptions.JSON,
            '{"type":"object"}',
        )
        request.sampling_params.structured_outputs._backend = backend_name
        request.sampling_params.all_stop_token_ids = {1, 2}
        return request

    xgrammar_request = make_request("xgrammar")
    guidance_request = make_request("guidance")
    manager.grammar_init(xgrammar_request)
    manager.grammar_init(guidance_request)

    assert xgrammar_request.structured_output_request.grammar[0] == "xgrammar"
    assert guidance_request.structured_output_request.grammar[0] == "guidance"
    assert set(manager.backends) == {"xgrammar", "guidance"}


def test_outlines_accepts_stop_token_after_regex_completion():
    class FinishedGuide:
        def is_finished(self):
            return True

        def accepts_tokens(self, tokens):
            raise AssertionError("The completed guide must not receive stop tokens")

    grammar = OutlinesGrammar(
        vocab_size=256,
        guide=FinishedGuide(),
        stop_token_ids={1},
    )

    assert not grammar.is_terminated()
    assert grammar.accept_tokens("request", [1, 2])
    assert grammar.validate_tokens([1, 2]) == [1]
    assert grammar.is_terminated()


def test_outlines_accepts_json_tail_and_stop_token_in_one_step():
    class FinishingGuide:
        def __init__(self):
            self.state = 0
            self.history = []

        def accepts_tokens(self, tokens):
            state = self.state
            for token in tokens:
                if state == 0 and token == 10:
                    state = 1
                elif state == 1 and token == 11:
                    state = 2
                else:
                    return False
            return True

        def advance(self, token):
            assert self.accepts_tokens([token])
            self.history.append(self.state)
            self.state += 1

        def rollback_state(self, num_tokens):
            for _ in range(num_tokens):
                self.state = self.history.pop()

        def is_finished(self):
            return self.state == 2

    guide = FinishingGuide()
    grammar = OutlinesGrammar(
        vocab_size=256,
        guide=guide,
        stop_token_ids={1},
    )

    # Draft validation accepts a stop token only after the hypothetical JSON
    # prefix completes, and leaves the live DFA unchanged.
    assert grammar.validate_tokens([10, 11, 1]) == [10, 11, 1]
    assert guide.state == 0
    assert grammar.validate_tokens([10, 1]) == [10]
    assert guide.state == 0

    # Simulate grammar_bitmask(): temporarily advance the JSON tail, observe
    # termination for the bonus row, then roll the lookahead back.
    for token in [10, 11]:
        assert not grammar.is_terminated()
        assert grammar.accept_tokens("request", [token])
    assert not grammar.is_terminated()
    assert grammar._prev_finished
    grammar.rollback(2)
    assert guide.state == 0
    assert not grammar._prev_finished

    # The verifier may commit the JSON tail and bonus EOS as one token block.
    assert grammar.accept_tokens("request", [10, 11, 1])
    assert guide.state == 2
    assert grammar.num_processed_tokens == 2
    assert grammar.is_terminated()


def test_manager_mixes_real_xgrammar_and_outlines_bitmasks():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    vllm_config = VllmConfig(
        model_config=ModelConfig(tokenizer=TOKENIZER),
        structured_outputs_config=StructuredOutputsConfig(backend="auto"),
        device_config=DeviceConfig(device="cpu"),
    )
    manager = StructuredOutputManager(vllm_config)

    def make_request(request_id, schema, backend_name):
        sampling_params = SamplingParams(
            structured_outputs=StructuredOutputsParams(json=schema),
        )
        sampling_params.structured_outputs._backend = backend_name
        sampling_params.update_from_generation_config({}, tokenizer.eos_token_id)
        return Request(
            request_id,
            prompt_token_ids=tokenizer.encode("Return JSON."),
            sampling_params=sampling_params,
            pooling_params=None,
        )

    xgrammar_request = make_request(
        "xgrammar",
        '{"type":"object","properties":{"ok":{"type":"boolean"}}}',
        "xgrammar",
    )
    outlines_request = make_request(
        "outlines",
        (
            '{"type":"array","items":{"type":"integer"},'
            '"contains":{"const":7},"minContains":1}'
        ),
        "outlines",
    )

    manager.grammar_init(xgrammar_request)
    manager.grammar_init(outlines_request)
    for request in (xgrammar_request, outlines_request):
        start_time = time.time()
        while not request.structured_output_request._check_grammar_completion():
            if time.time() - start_time > 5:
                pytest.fail(f"Grammar compilation timed out for {request.request_id}")
            time.sleep(0.01)

    bitmask = manager.grammar_bitmask(
        requests={
            xgrammar_request.request_id: xgrammar_request,
            outlines_request.request_id: outlines_request,
        },
        structured_output_request_ids=[
            xgrammar_request.request_id,
            outlines_request.request_id,
        ],
        scheduled_spec_decode_tokens={},
    )

    assert bitmask is not None
    expected_mask_width = (manager.vllm_config.model_config.get_vocab_size() + 31) // 32
    assert bitmask.shape == (2, expected_mask_width)
    assert set(manager.backends) == {"xgrammar", "outlines"}

    outlines_grammar = outlines_request.structured_output_request.grammar
    for token in tokenizer.encode("[7]"):
        assert not outlines_grammar.is_terminated()
        assert outlines_grammar.accept_tokens(outlines_request.request_id, [token])
    assert not outlines_grammar.is_terminated()
    assert outlines_grammar.accept_tokens(
        outlines_request.request_id, [tokenizer.eos_token_id]
    )
    assert outlines_grammar.is_terminated()


@pytest.mark.parametrize(
    "request_type,grammar_spec",
    [
        pytest.param(
            StructuredOutputOptions.JSON,
            '{"type": "object"}',
            id="json",
        ),
        pytest.param(
            StructuredOutputOptions.GRAMMAR,
            'start: "hello" | "world"',
            id="lark",
        ),
    ],
)
def test_mistral_tokenizer_compile_grammar(
    mistral_tokenizer,
    request_type: StructuredOutputOptions,
    grammar_spec: str,
) -> None:
    vllm_config = VllmConfig(
        structured_outputs_config=StructuredOutputsConfig(backend="guidance"),
    )
    backend = GuidanceBackend(
        vllm_config,
        tokenizer=mistral_tokenizer,
        vocab_size=mistral_tokenizer.vocab_size,
    )
    assert backend.ll_tokenizer is mistral_tokenizer.llg_tokenizer

    grammar = backend.compile_grammar(request_type, grammar_spec)
    assert grammar is not None
    assert not grammar.is_terminated()
