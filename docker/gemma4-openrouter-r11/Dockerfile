ARG BASE_IMAGE=ghcr.io/phala-network/vllm-openai:main-aa9903490-cu130-ubuntu2404-gemma4-pth-toolfix-r10@sha256:a5427aa261b560c444d2a9ed969ad8d633eb13ce9efec2526394c5f113516e2e
FROM ${BASE_IMAGE}

ARG SOURCE_COMMIT

ENV VLLM_PACKAGE_ROOT=/usr/local/lib/python3.12/dist-packages/vllm

COPY vllm/v1/structured_output/backend_outlines.py /tmp/vllm-r11/backend_outlines.py
COPY tests/v1/structured_output/test_backend_guidance.py /opt/vllm-r11-tests/test_backend_guidance.py

RUN test "$(sha256sum "$VLLM_PACKAGE_ROOT/v1/structured_output/backend_outlines.py" | cut -d' ' -f1)" = "957066467c3ba1112114283155c409878100f4a673b747f293b459dcd259c6c1" \
    && test "$(sha256sum "$VLLM_PACKAGE_ROOT/v1/structured_output/__init__.py" | cut -d' ' -f1)" = "0c77f58e1083ef67761024b7a89ab87e23e5cc363d32f3f09b2b7c47ffa507e8" \
    && test "$(sha256sum "$VLLM_PACKAGE_ROOT/v1/structured_output/backend_guidance.py" | cut -d' ' -f1)" = "534228422f4ac48218ca9a97b1780c7d7890ee335035614eb0ee4f4189538f64" \
    && test "$(sha256sum /tmp/vllm-r11/backend_outlines.py | cut -d' ' -f1)" = "00dedd94882c92175f49ef3c041c125265c5e49d8e05d498dc7c83f0b8e96a53" \
    && install -m 0644 /tmp/vllm-r11/backend_outlines.py "$VLLM_PACKAGE_ROOT/v1/structured_output/backend_outlines.py" \
    && uv run --no-project --python /usr/bin/python3 python -m py_compile \
        "$VLLM_PACKAGE_ROOT/v1/structured_output/backend_outlines.py" \
    && test "$(sha256sum "$VLLM_PACKAGE_ROOT/v1/structured_output/backend_outlines.py" | cut -d' ' -f1)" = "00dedd94882c92175f49ef3c041c125265c5e49d8e05d498dc7c83f0b8e96a53" \
    && rm -rf /tmp/vllm-r11

RUN uv run --no-project --python /usr/bin/python3 python -m pytest -q \
    --confcutdir=/opt/vllm-r11-tests \
    /opt/vllm-r11-tests/test_backend_guidance.py::test_manager_compiles_with_each_request_selected_backend \
    /opt/vllm-r11-tests/test_backend_guidance.py::test_outlines_accepts_stop_token_after_regex_completion

LABEL org.opencontainers.image.source="https://github.com/zozyo/vllm" \
      org.opencontainers.image.revision="${SOURCE_COMMIT}" \
      org.opencontainers.image.description="Gemma4 OpenRouter outlines termination compatibility overlay on the r10 FA4/FP8 image" \
      io.phala.patch.scope="gemma4-openrouter-r10,outlines-stop-token-termination"
