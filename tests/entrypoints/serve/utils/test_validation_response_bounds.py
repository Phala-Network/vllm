# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
from types import SimpleNamespace

import pytest
from fastapi.exceptions import RequestValidationError

from vllm.entrypoints.serve.utils.server_utils import validation_exception_handler


def _fake_request() -> SimpleNamespace:
    return SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(args=SimpleNamespace(log_error_stack=False))
        ),
        state=SimpleNamespace(),
    )


@pytest.mark.asyncio
async def test_validation_input_not_echoed_and_error_count_capped():
    repeated_input = "x" * 2000
    errors = [
        {
            "type": "string_type",
            "loc": ("body", "input", index, "content"),
            "msg": "Input should be a valid string",
            "input": repeated_input,
        }
        for index in range(12000)
    ]

    response = await validation_exception_handler(
        _fake_request(), RequestValidationError(errors)
    )
    body = json.loads(response.body)
    message = body["error"]["message"]

    assert repeated_input not in message
    assert len(response.body) < 50_000
    assert "12000 validation errors" in message
    assert "more" in message
