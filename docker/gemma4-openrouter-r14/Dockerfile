ARG BASE_IMAGE=ghcr.io/phala-network/vllm-openai:main-aa9903490-cu130-ubuntu2404-gemma4-pth-toolfix-r13@sha256:a4b6403697a8d39da60c0ba4e326a71cd6fd1c7c94637330c8be6067fa435d0c

FROM ${BASE_IMAGE}

ARG SOURCE_COMMIT
ENV VLLM_PACKAGE_ROOT=/usr/local/lib/python3.12/dist-packages/vllm

COPY vllm/envs.py /tmp/vllm-r14/vllm/envs.py
COPY vllm/v1/structured_output/utils.py /tmp/vllm-r14/vllm/v1/structured_output/utils.py
COPY vllm/v1/structured_output/backend_outlines.py /tmp/vllm-r14/vllm/v1/structured_output/backend_outlines.py
COPY vllm/v1/structured_output/backend_xgrammar.py /tmp/vllm-r14/vllm/v1/structured_output/backend_xgrammar.py
COPY tests/v1/structured_output/test_regex_compilation_timeout.py /opt/vllm-r14-tests/tests/v1/structured_output/test_regex_compilation_timeout.py

RUN set -eux; \
    test "$(sha256sum "$VLLM_PACKAGE_ROOT/envs.py" | cut -d' ' -f1)" = "0aae5f063141e705b6594aa974205df27a63d519e4e6b6193dd6880f261cfa09"; \
    test "$(sha256sum "$VLLM_PACKAGE_ROOT/v1/structured_output/utils.py" | cut -d' ' -f1)" = "a4d0532c026ca5bba67ac3a45ac1602f9c825767844378fef8dca0ea4a5cdf7e"; \
    test "$(sha256sum "$VLLM_PACKAGE_ROOT/v1/structured_output/backend_outlines.py" | cut -d' ' -f1)" = "8d54fa3f8c67aa7bced53280229315f6abc8bbba96a5c5f94132b69e9e7d001b"; \
    test "$(sha256sum "$VLLM_PACKAGE_ROOT/v1/structured_output/backend_xgrammar.py" | cut -d' ' -f1)" = "e5b0c492195175ad1379b123a63fcd99c94aa1f2d9a8b4655d7e2d2e53f16de6"; \
    test "$(sha256sum /tmp/vllm-r14/vllm/envs.py | cut -d' ' -f1)" = "6fe2e1274504faf0a1a8330efeab413df165fa2d6fe00d253ba30f5460be0e62"; \
    test "$(sha256sum /tmp/vllm-r14/vllm/v1/structured_output/utils.py | cut -d' ' -f1)" = "08e7fdda102961d40cd1018d8ae0f9fe739e097263e2cf9b2b3dc238fde6fa11"; \
    test "$(sha256sum /tmp/vllm-r14/vllm/v1/structured_output/backend_outlines.py | cut -d' ' -f1)" = "f03b1fee6ef9cc7781722a00f4931bcd0cd403c6ec6d4bf418b8e8e47420c738"; \
    test "$(sha256sum /tmp/vllm-r14/vllm/v1/structured_output/backend_xgrammar.py | cut -d' ' -f1)" = "44af09cd1bcf3af5672845fa6c26e6ddfb69536f14a28b9066a4d89fe5c003f5"; \
    test "$(sha256sum /opt/vllm-r14-tests/tests/v1/structured_output/test_regex_compilation_timeout.py | cut -d' ' -f1)" = "28028f1f8e917dcc862570e2974db9257fea184c29b205f81d665c11637bc5a4"; \
    install -m 0644 /tmp/vllm-r14/vllm/envs.py "$VLLM_PACKAGE_ROOT/envs.py"; \
    install -m 0644 /tmp/vllm-r14/vllm/v1/structured_output/utils.py "$VLLM_PACKAGE_ROOT/v1/structured_output/utils.py"; \
    install -m 0644 /tmp/vllm-r14/vllm/v1/structured_output/backend_outlines.py "$VLLM_PACKAGE_ROOT/v1/structured_output/backend_outlines.py"; \
    install -m 0644 /tmp/vllm-r14/vllm/v1/structured_output/backend_xgrammar.py "$VLLM_PACKAGE_ROOT/v1/structured_output/backend_xgrammar.py"; \
    python3 -m py_compile \
      "$VLLM_PACKAGE_ROOT/envs.py" \
      "$VLLM_PACKAGE_ROOT/v1/structured_output/utils.py" \
      "$VLLM_PACKAGE_ROOT/v1/structured_output/backend_outlines.py" \
      "$VLLM_PACKAGE_ROOT/v1/structured_output/backend_xgrammar.py"; \
    cd /opt/vllm-r14-tests; \
    VLLM_TARGET_DEVICE=cpu python3 -m pytest --import-mode=importlib -q \
      tests/v1/structured_output/test_regex_compilation_timeout.py; \
    VLLM_TARGET_DEVICE=cpu VLLM_REGEX_COMPILATION_TIMEOUT_S=2 python3 -c 'from vllm.v1.structured_output.utils import _xgr_grammar_from_regex, compile_regex_with_timeout; result = compile_regex_with_timeout(_xgr_grammar_from_regex, r"[a-z]{1,8}"); assert isinstance(result, str) and result'; \
    rm -rf /tmp/vllm-r14

LABEL org.opencontainers.image.source="https://github.com/Phala-Network/vllm" \
      org.opencontainers.image.revision="${SOURCE_COMMIT}" \
      org.opencontainers.image.description="Gemma4 r14 killable regex compilation workers and IPC hardening" \
      io.phala.patch.scope="gemma4-openrouter-r13,regex-subprocess,regex-pipe-cleanup,tool-strict,mtp-structured-output"
