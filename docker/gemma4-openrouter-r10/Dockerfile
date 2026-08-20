ARG BASE_IMAGE=ghcr.io/phala-network/vllm-openai:main-aa9903490-cu130-ubuntu2404-gemma4-pth-toolfix-r9@sha256:901d87095866d597ba1427dbeac27e2581991c5534c1e6cd3e96ed5b8232ad74
FROM ${BASE_IMAGE}

ARG SOURCE_COMMIT

ENV VLLM_PACKAGE_ROOT=/usr/local/lib/python3.12/dist-packages/vllm

COPY vllm/v1/structured_output/__init__.py /tmp/vllm-r10/__init__.py
COPY vllm/v1/structured_output/backend_guidance.py /tmp/vllm-r10/backend_guidance.py
COPY tests/v1/structured_output/test_backend_guidance.py /opt/vllm-r10-tests/test_backend_guidance.py
COPY tests/v1/structured_output/test_validation.py /opt/vllm-r10-tests/test_validation.py

RUN test "$(sha256sum "$VLLM_PACKAGE_ROOT/v1/structured_output/__init__.py" | cut -d' ' -f1)" = "011e972604ed3a2a1e1db5421012060a0dc9133826ac0c58236eaca6a8756fc8" \
    && test "$(sha256sum "$VLLM_PACKAGE_ROOT/v1/structured_output/backend_guidance.py" | cut -d' ' -f1)" = "9bd293695af898d4b3ae1f71596baa6680ad57f556ea743ace163adfb21fea0b" \
    && test "$(sha256sum "$VLLM_PACKAGE_ROOT/parser/gemma4.py" | cut -d' ' -f1)" = "962a387b621b5043086ab97c63216ba154efc89efe196d7d5d7465b5d0ebca0e" \
    && test "$(sha256sum "$VLLM_PACKAGE_ROOT/entrypoints/openai/chat_completion/protocol.py" | cut -d' ' -f1)" = "8ced1da9c297f1aabd9f451c6b84b92724cf528a3d7d8e8aeacacf6d17906953" \
    && test "$(sha256sum /tmp/vllm-r10/__init__.py | cut -d' ' -f1)" = "0c77f58e1083ef67761024b7a89ab87e23e5cc363d32f3f09b2b7c47ffa507e8" \
    && test "$(sha256sum /tmp/vllm-r10/backend_guidance.py | cut -d' ' -f1)" = "534228422f4ac48218ca9a97b1780c7d7890ee335035614eb0ee4f4189538f64" \
    && install -m 0644 /tmp/vllm-r10/__init__.py "$VLLM_PACKAGE_ROOT/v1/structured_output/__init__.py" \
    && install -m 0644 /tmp/vllm-r10/backend_guidance.py "$VLLM_PACKAGE_ROOT/v1/structured_output/backend_guidance.py" \
    && uv run --no-project --python /usr/bin/python3 python -m py_compile \
        "$VLLM_PACKAGE_ROOT/v1/structured_output/__init__.py" \
        "$VLLM_PACKAGE_ROOT/v1/structured_output/backend_guidance.py" \
    && test "$(sha256sum "$VLLM_PACKAGE_ROOT/v1/structured_output/__init__.py" | cut -d' ' -f1)" = "0c77f58e1083ef67761024b7a89ab87e23e5cc363d32f3f09b2b7c47ffa507e8" \
    && test "$(sha256sum "$VLLM_PACKAGE_ROOT/v1/structured_output/backend_guidance.py" | cut -d' ' -f1)" = "534228422f4ac48218ca9a97b1780c7d7890ee335035614eb0ee4f4189538f64" \
    && rm -rf /tmp/vllm-r10

RUN uv run --no-project --python /usr/bin/python3 python -m pytest -q \
    --confcutdir=/opt/vllm-r10-tests \
    /opt/vllm-r10-tests/test_backend_guidance.py::test_manager_compiles_with_each_request_selected_backend \
    /opt/vllm-r10-tests/test_validation.py

LABEL org.opencontainers.image.source="https://github.com/zozyo/vllm" \
      org.opencontainers.image.revision="${SOURCE_COMMIT}" \
      org.opencontainers.image.description="Gemma4 OpenRouter mixed structured-output backend compatibility overlay on the r9 FA4/FP8 image" \
      io.phala.patch.scope="gemma4-openrouter-r9,mixed-structured-output-backends,guidance-schema-fallback"
