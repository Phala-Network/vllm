# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
from pydantic import ValidationError

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.completion.protocol import CompletionRequest


@pytest.mark.parametrize("field_name", ["prompt_logprobs", "top_logprobs"])
def test_chat_non_numeric_logprobs_rejected(field_name):
    with pytest.raises(ValidationError, match=f"`{field_name}` must be an integer"):
        ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "hello"}],
            **{field_name: "2"},
        )


@pytest.mark.parametrize("field_name", ["prompt_logprobs", "logprobs"])
def test_completion_non_numeric_logprobs_rejected(field_name):
    with pytest.raises(ValidationError, match=f"`{field_name}` must be an integer"):
        CompletionRequest(
            model="test-model",
            prompt="Test prompt",
            max_tokens=10,
            **{field_name: "2"},
        )
