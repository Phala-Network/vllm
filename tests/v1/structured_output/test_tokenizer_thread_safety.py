# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression coverage imported from open upstream PR #47509."""

import concurrent.futures as cf

import pytest
from transformers import AutoTokenizer

from vllm.config import StructuredOutputsConfig, VllmConfig
from vllm.config.model import ModelConfig
from vllm.tokenizers import cached_tokenizer_from_config
from vllm.tokenizers.hf import ThreadSafeHFTokenizerMixin
from vllm.v1.structured_output import StructuredOutputManager

pytestmark = pytest.mark.cpu_test

TOKENIZER = "gpt2"
_N_THREADS = 32
_N_ITERS = 200
_TEXT = "hello world this is a concurrency test " * 4


def _make_manager() -> StructuredOutputManager:
    vllm_config = VllmConfig(
        model_config=ModelConfig(tokenizer=TOKENIZER),
        structured_outputs_config=StructuredOutputsConfig(backend="guidance"),
    )
    return StructuredOutputManager(vllm_config)


def _hammer(tokenizer) -> list[str]:
    errors: list[str] = []

    def work(_: int) -> None:
        try:
            for j in range(_N_ITERS):
                if j % 2 == 0:
                    tokenizer(_TEXT, truncation=True, max_length=8)
                else:
                    tokenizer(_TEXT)
        except Exception as exc:  # noqa: BLE001 - collect worker failures
            errors.append(repr(exc))

    with cf.ThreadPoolExecutor(max_workers=_N_THREADS) as executor:
        list(executor.map(work, range(_N_THREADS)))
    return errors


def test_manager_tokenizer_is_thread_safe_wrapped():
    manager = _make_manager()
    assert isinstance(manager.tokenizer, ThreadSafeHFTokenizerMixin)


def test_manager_does_not_mutate_shared_cache():
    manager = _make_manager()
    shared = cached_tokenizer_from_config(model_config=manager.vllm_config.model_config)
    assert not isinstance(shared, ThreadSafeHFTokenizerMixin)
    assert manager.tokenizer is not shared


def test_manager_tokenizer_survives_concurrent_encode():
    manager = _make_manager()
    errors = _hammer(manager.tokenizer)
    assert errors == [], f"pooled tokenizer raced under concurrency: {errors[:3]}"


def test_raw_fast_tokenizer_is_thread_unsafe():
    raw = AutoTokenizer.from_pretrained(TOKENIZER)
    assert raw.is_fast
    for _ in range(3):
        errors = _hammer(raw)
        if errors:
            assert all("Already borrowed" in error for error in errors)
            return
    pytest.skip("could not reproduce the 'Already borrowed' race here")
