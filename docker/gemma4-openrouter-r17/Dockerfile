ARG BASE_IMAGE=ghcr.io/phala-network/vllm-openai:main-aa9903490-cu130-ubuntu2404-gemma4-pth-toolfix-r16@sha256:0ade0eb9e905093c5ee27814bd0a98c9a5d82c5d708cb954fb401cd15aba01e9

FROM ${BASE_IMAGE}

ARG SOURCE_COMMIT
ENV VLLM_PACKAGE_ROOT=/usr/local/lib/python3.12/dist-packages/vllm
SHELL ["/bin/bash", "-c"]

COPY docker/gemma4-openrouter-r17/base.sha256 /tmp/vllm-r17/base.sha256
COPY docker/gemma4-openrouter-r17/patched.sha256 /tmp/vllm-r17/patched.sha256
COPY vllm/config/structured_outputs.py /tmp/vllm-r17/vllm/config/structured_outputs.py
COPY vllm/entrypoints/chat_utils.py /tmp/vllm-r17/vllm/entrypoints/chat_utils.py
COPY vllm/parser/gemma4.py /tmp/vllm-r17/vllm/parser/gemma4.py
COPY vllm/sampling_params.py /tmp/vllm-r17/vllm/sampling_params.py
COPY vllm/v1/structured_output/__init__.py /tmp/vllm-r17/vllm/v1/structured_output/__init__.py
COPY vllm/v1/structured_output/backend_guidance.py /tmp/vllm-r17/vllm/v1/structured_output/backend_guidance.py
COPY vllm/v1/structured_output/backend_lm_format_enforcer.py /tmp/vllm-r17/vllm/v1/structured_output/backend_lm_format_enforcer.py
COPY vllm/v1/structured_output/backend_outlines.py /tmp/vllm-r17/vllm/v1/structured_output/backend_outlines.py
COPY vllm/v1/structured_output/backend_types.py /tmp/vllm-r17/vllm/v1/structured_output/backend_types.py
COPY vllm/v1/structured_output/backend_xgrammar.py /tmp/vllm-r17/vllm/v1/structured_output/backend_xgrammar.py

RUN set -eux; \
    sha256sum --check --strict /tmp/vllm-r17/base.sha256; \
    sha256sum --check --strict /tmp/vllm-r17/patched.sha256; \
    find /tmp/vllm-r17/vllm -type f -name '*.py' -print0 \
      | while IFS= read -r -d '' source; do \
          relative="${source#/tmp/vllm-r17/vllm/}"; \
          install -D -m 0644 "$source" "$VLLM_PACKAGE_ROOT/$relative"; \
        done; \
    find "$VLLM_PACKAGE_ROOT" -type d -name __pycache__ -prune -exec rm -rf '{}' +; \
    uv venv --system-site-packages /tmp/vllm-r17-venv; \
    find "$VLLM_PACKAGE_ROOT" -type f -name '*.py' -newer /tmp/vllm-r17/base.sha256 -print0 \
      | xargs -0 /tmp/vllm-r17-venv/bin/python -m py_compile; \
    sed 's#/tmp/vllm-r17/vllm/#/usr/local/lib/python3.12/dist-packages/vllm/#' \
      /tmp/vllm-r17/patched.sha256 \
      | sha256sum --check --strict; \
    rm -rf /tmp/vllm-r17 /tmp/vllm-r17-venv

LABEL org.opencontainers.image.title="Phala Gemma4 vLLM r17 protocol repairs" \
      org.opencontainers.image.source="https://github.com/Phala-Network/vllm" \
      org.opencontainers.image.revision="${SOURCE_COMMIT}" \
      org.opencontainers.image.base.name="ghcr.io/phala-network/vllm-openai:main-aa9903490-cu130-ubuntu2404-gemma4-pth-toolfix-r16" \
      org.opencontainers.image.base.digest="sha256:0ade0eb9e905093c5ee27814bd0a98c9a5d82c5d708cb954fb401cd15aba01e9" \
      io.phala.patch.scope="gemma4-tool-parser,structured-output-validation,mixed-backends,tokenizer-thread-safety" \
      io.phala.upstream-pr-51307="9ea8cfbaef756006ed28ff0bffd24b9970d37003" \
      io.phala.upstream-pr-53444="6be086f4660dbace7742963572734998843450cd" \
      io.phala.upstream-pr-48922="d4deae84733337d6a152d5e81926120a78f65c5c" \
      io.phala.upstream-pr-47450="942b598a689a53fd869a8c7f3c36c5b5262a70dc" \
      io.phala.upstream-pr-52020="d27fb41a1b6587ed154664451cbee953c3c91432" \
      io.phala.upstream-pr-48416="af9d0fc8dd9219568717e0919f7a60f2af475f8f" \
      io.phala.upstream-pr-47562="c0928e80202a4594e5f0c87fc8a1c4c247242e49" \
      io.phala.upstream-pr-50015="5fa1b9250066cc72ede68eea96061441c6e6a05d" \
      io.phala.upstream-pr-50944="15a05461f6278bbdd34093d696ef90d213f3ae01" \
      io.phala.upstream-pr-47509="e26d57524bbc96f7c32bad727b126fadb6f2db27"
