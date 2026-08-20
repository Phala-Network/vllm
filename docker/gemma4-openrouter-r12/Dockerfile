ARG BASE_IMAGE=ghcr.io/phala-network/vllm-openai:main-aa9903490-cu130-ubuntu2404-gemma4-pth-toolfix-r11@sha256:b7eb38c9e59214fc3dc96efa138018df5e91b747749af99c8ae71578550a9d66
FROM ${BASE_IMAGE}

ARG SOURCE_COMMIT

ENV VLLM_PACKAGE_ROOT=/usr/local/lib/python3.12/dist-packages/vllm \
    VLLM_STATIC_KV_SCALE_PATH=/opt/vllm/kv-scales/gemma4-static-kv-scales-mtp.json

COPY vllm/v1/structured_output/backend_outlines.py /tmp/vllm-r12/backend_outlines.py
COPY tests/v1/structured_output/test_backend_guidance.py /opt/vllm-r12-tests/test_backend_guidance.py
COPY docker/gemma4-openrouter-r12/gemma4-static-kv-scales-mtp.json /opt/vllm/kv-scales/gemma4-static-kv-scales-mtp.json

RUN test "$(sha256sum "$VLLM_PACKAGE_ROOT/v1/structured_output/backend_outlines.py" | cut -d' ' -f1)" = "00dedd94882c92175f49ef3c041c125265c5e49d8e05d498dc7c83f0b8e96a53" \
    && test "$(sha256sum /tmp/vllm-r12/backend_outlines.py | cut -d' ' -f1)" = "8d54fa3f8c67aa7bced53280229315f6abc8bbba96a5c5f94132b69e9e7d001b" \
    && test "$(sha256sum /opt/vllm-r12-tests/test_backend_guidance.py | cut -d' ' -f1)" = "97cf4e00d7c1314d00c69ecaf63cd87b76649bf15c23da84ba7d19179f0ab3d6" \
    && test "$(sha256sum /opt/vllm/kv-scales/gemma4-static-kv-scales-mtp.json | cut -d' ' -f1)" = "e268cb034a8d80edb36a9f6b69f502834e8053acf73255e95a002550f1ff8d02" \
    && install -m 0644 /tmp/vllm-r12/backend_outlines.py "$VLLM_PACKAGE_ROOT/v1/structured_output/backend_outlines.py" \
    && uv run --no-project --python /usr/bin/python3 python -m py_compile "$VLLM_PACKAGE_ROOT/v1/structured_output/backend_outlines.py" \
    && rm -rf /tmp/vllm-r12

LABEL org.opencontainers.image.source="https://github.com/zozyo/vllm" \
      org.opencontainers.image.revision="${SOURCE_COMMIT}" \
      org.opencontainers.image.description="Gemma4 MTP4 static KV scales and speculative Outlines termination fix" \
      io.phala.patch.scope="gemma4-openrouter-r11,mtp4-static-kv-scales,outlines-speculative-stop-token"
