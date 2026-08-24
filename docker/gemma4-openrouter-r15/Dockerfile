ARG BASE_IMAGE=ghcr.io/phala-network/vllm-openai:main-aa9903490-cu130-ubuntu2404-gemma4-pth-toolfix-r14@sha256:676d980cf3d0ee003ca7cf763bebcadefcbb5dd0527b4d56c2fcc7b750baaff2

FROM ${BASE_IMAGE}

ARG SOURCE_COMMIT
ENV VLLM_PACKAGE_ROOT=/usr/local/lib/python3.12/dist-packages/vllm

COPY vllm/v1/structured_output/utils.py /tmp/vllm-r15/vllm/v1/structured_output/utils.py
COPY vllm/v1/structured_output/__init__.py /tmp/vllm-r15/vllm/v1/structured_output/__init__.py
COPY tests/__init__.py /opt/vllm-r15-tests/tests/__init__.py
COPY tests/v1/__init__.py /opt/vllm-r15-tests/tests/v1/__init__.py
COPY tests/v1/structured_output/__init__.py /opt/vllm-r15-tests/tests/v1/structured_output/__init__.py
COPY tests/v1/structured_output/test_regex_compilation_timeout.py /opt/vllm-r15-tests/tests/v1/structured_output/test_regex_compilation_timeout.py

RUN set -eux; \
    test "$(sha256sum "$VLLM_PACKAGE_ROOT/v1/structured_output/utils.py" | cut -d' ' -f1)" = "08e7fdda102961d40cd1018d8ae0f9fe739e097263e2cf9b2b3dc238fde6fa11"; \
    test "$(sha256sum "$VLLM_PACKAGE_ROOT/v1/structured_output/__init__.py" | cut -d' ' -f1)" = "1c6a322064b797a2e29101e83d4b94b002ded88e6a088057d13149babe7d6925"; \
    test "$(sha256sum /tmp/vllm-r15/vllm/v1/structured_output/utils.py | cut -d' ' -f1)" = "3b3de74cf438a0fceb5864fb4e3329a85a2d77428ff3741c3e0b39a89abd05f8"; \
    test "$(sha256sum /tmp/vllm-r15/vllm/v1/structured_output/__init__.py | cut -d' ' -f1)" = "274f3316403acbd8bbc56a2c1af5ef66ed60610ff5e500ea1570b6fc7f100093"; \
    test "$(sha256sum /opt/vllm-r15-tests/tests/v1/structured_output/test_regex_compilation_timeout.py | cut -d' ' -f1)" = "a39d1e760a0ec887c3b48ebc471c741efbcc02abc6f8f3e51a4c0b1d72e446be"; \
    install -m 0644 /tmp/vllm-r15/vllm/v1/structured_output/utils.py "$VLLM_PACKAGE_ROOT/v1/structured_output/utils.py"; \
    install -m 0644 /tmp/vllm-r15/vllm/v1/structured_output/__init__.py "$VLLM_PACKAGE_ROOT/v1/structured_output/__init__.py"; \
    python3 -m py_compile \
      "$VLLM_PACKAGE_ROOT/v1/structured_output/utils.py" \
      "$VLLM_PACKAGE_ROOT/v1/structured_output/__init__.py"; \
    cd /opt/vllm-r15-tests; \
    VLLM_TARGET_DEVICE=cpu python3 -m pytest --import-mode=importlib -q \
      tests/v1/structured_output/test_regex_compilation_timeout.py; \
    VLLM_TARGET_DEVICE=cpu VLLM_REGEX_COMPILATION_TIMEOUT_S=5 python3 -c 'from concurrent.futures import ThreadPoolExecutor; from vllm.v1.structured_output.utils import _xgr_grammar_from_regex, compile_regex_with_timeout, initialize_regex_compilation_forkserver; initialize_regex_compilation_forkserver(); executor = ThreadPoolExecutor(max_workers=1); result = executor.submit(compile_regex_with_timeout, _xgr_grammar_from_regex, r"[a-z]{1,8}").result(timeout=10); executor.shutdown(); assert isinstance(result, str) and result'; \
    rm -rf /tmp/vllm-r15

LABEL org.opencontainers.image.source="https://github.com/Phala-Network/vllm" \
      org.opencontainers.image.revision="${SOURCE_COMMIT}" \
      org.opencontainers.image.description="Gemma4 r15 thread-safe preloaded forkserver for killable regex compilation" \
      io.phala.patch.scope="gemma4-openrouter-r14,regex-forkserver-preload,regex-thread-safety,regex-pipe-cleanup,tool-strict,mtp-structured-output"
