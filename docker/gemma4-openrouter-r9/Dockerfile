ARG BASE_IMAGE=ghcr.io/phala-network/vllm-openai:main-aa9903490-cu130-ubuntu2404-gemma4-pth-toolfix-r8@sha256:d157dc04ae0a436d8e28d87e4a8b3f911e6bb22491cde4de32fd704f42899b52
FROM ${BASE_IMAGE}

ARG SOURCE_COMMIT

ENV VLLM_PACKAGE_ROOT=/usr/local/lib/python3.12/dist-packages/vllm

COPY vllm/parser/gemma4.py /tmp/vllm-r9/gemma4.py
COPY vllm/entrypoints/openai/chat_completion/protocol.py /tmp/vllm-r9/protocol.py
COPY tests/entrypoints/openai/chat_completion/test_chat_error.py /opt/vllm-r9-tests/test_chat_error.py

RUN test "$(sha256sum "$VLLM_PACKAGE_ROOT/parser/gemma4.py" | cut -d' ' -f1)" = "aa9f11849c9a97a919c6d151dc4ec8d600f89f5fd73964dd270077a71d82cd4a" \
    && test "$(sha256sum "$VLLM_PACKAGE_ROOT/entrypoints/openai/chat_completion/protocol.py" | cut -d' ' -f1)" = "cb756e3d18e9061a2b306f305e10bd71d43f01ad1b236d0e9cbbb8756cd504dc" \
    && test "$(sha256sum /tmp/vllm-r9/gemma4.py | cut -d' ' -f1)" = "962a387b621b5043086ab97c63216ba154efc89efe196d7d5d7465b5d0ebca0e" \
    && test "$(sha256sum /tmp/vllm-r9/protocol.py | cut -d' ' -f1)" = "8ced1da9c297f1aabd9f451c6b84b92724cf528a3d7d8e8aeacacf6d17906953" \
    && install -m 0644 /tmp/vllm-r9/gemma4.py "$VLLM_PACKAGE_ROOT/parser/gemma4.py" \
    && install -m 0644 /tmp/vllm-r9/protocol.py "$VLLM_PACKAGE_ROOT/entrypoints/openai/chat_completion/protocol.py" \
    && uv run --no-project --python /usr/bin/python3 python -m py_compile \
        "$VLLM_PACKAGE_ROOT/parser/gemma4.py" \
        "$VLLM_PACKAGE_ROOT/entrypoints/openai/chat_completion/protocol.py" \
    && test "$(sha256sum "$VLLM_PACKAGE_ROOT/parser/gemma4.py" | cut -d' ' -f1)" = "962a387b621b5043086ab97c63216ba154efc89efe196d7d5d7465b5d0ebca0e" \
    && test "$(sha256sum "$VLLM_PACKAGE_ROOT/entrypoints/openai/chat_completion/protocol.py" | cut -d' ' -f1)" = "8ced1da9c297f1aabd9f451c6b84b92724cf528a3d7d8e8aeacacf6d17906953" \
    && rm -rf /tmp/vllm-r9

RUN uv run --no-project --python /usr/bin/python3 python -m pytest -q \
    --confcutdir=/opt/vllm-r9-tests \
    /opt/vllm-r9-tests/test_chat_error.py::test_boolean_structured_outputs_hint_uses_response_format

LABEL org.opencontainers.image.source="https://github.com/zozyo/vllm" \
      org.opencontainers.image.revision="${SOURCE_COMMIT}" \
      org.opencontainers.image.description="Gemma4 OpenRouter structured_outputs boolean compatibility and reasoning attribution overlay on the r8 FA4/FP8 image" \
      io.phala.patch.scope="gemma4-openrouter-structured-outputs-bool,gemma4-thinking-default,gemma4-nonstreaming-preamble-reasoning"
