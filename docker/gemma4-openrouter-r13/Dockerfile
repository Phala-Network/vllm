# syntax=docker/dockerfile:1.7
ARG BASE_IMAGE=ghcr.io/phala-network/vllm-openai:main-aa9903490-cu130-ubuntu2404-gemma4-pth-toolfix-r12@sha256:27f9382fdb8d59b5dab28f244569ae499ade293ca649f60849cb92f8cc258c41

FROM ${BASE_IMAGE} AS globaltimer-builder

USER root
ENV MAX_JOBS=1 \
    TORCH_CUDA_ARCH_LIST=9.0 \
    TORCH_EXTENSIONS_DIR=/opt/vllm/torch-extensions

RUN apt-get update \
    && apt-get install -y --no-install-recommends libcusparse-dev-12-9 \
    && rm -rf /var/lib/apt/lists/*

RUN python3 - <<'PY'
import os
import shutil
from pathlib import Path

from flashinfer.utils import get_globaltimer_kernel

kernel = get_globaltimer_kernel()
assert callable(kernel), "FlashInfer globaltimer precompile returned no kernel"
root = Path(os.environ["TORCH_EXTENSIONS_DIR"])
matches = list(root.rglob("flashinfer_globaltimer*.so"))
assert len(matches) == 1, matches
out = Path("/opt/vllm/prebuilt/flashinfer_globaltimer.so")
out.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(matches[0], out)
print(f"precompiled_globaltimer={matches[0]}")
PY

FROM ${BASE_IMAGE}

ARG SOURCE_COMMIT

ENV VLLM_PACKAGE_ROOT=/usr/local/lib/python3.12/dist-packages/vllm \
    FLASHINFER_PACKAGE_ROOT=/usr/local/lib/python3.12/dist-packages/flashinfer \
    TORCH_EXTENSIONS_DIR=/opt/vllm/torch-extensions \
    MALLOC_ARENA_MAX=8 \
    MALLOC_TRIM_THRESHOLD_=131072

COPY --from=globaltimer-builder /opt/vllm/prebuilt/flashinfer_globaltimer.so /usr/local/lib/python3.12/dist-packages/flashinfer_globaltimer.so

COPY vllm/utils/cache.py /tmp/vllm-r13/vllm/utils/cache.py
COPY vllm/config/multimodal.py /tmp/vllm-r13/vllm/config/multimodal.py
COPY vllm/multimodal/cache.py /tmp/vllm-r13/vllm/multimodal/cache.py
COPY vllm/renderers/online_renderer.py /tmp/vllm-r13/vllm/renderers/online_renderer.py
COPY vllm/v1/structured_output/__init__.py /tmp/vllm-r13/vllm/v1/structured_output/__init__.py
COPY vllm/v1/structured_output/backend_xgrammar.py /tmp/vllm-r13/vllm/v1/structured_output/backend_xgrammar.py
COPY tests/utils_/test_cache.py /opt/vllm-r13-tests/tests/utils_/test_cache.py
COPY tests/multimodal/test_cache.py /opt/vllm-r13-tests/tests/multimodal/test_cache.py
COPY tests/v1/spec_decode/test_mtp_structured_output.py /opt/vllm-r13-tests/tests/v1/spec_decode/test_mtp_structured_output.py
COPY tests/entrypoints/openai/test_render_parity.py /opt/vllm-r13-tests/tests/entrypoints/openai/test_render_parity.py
COPY docker/gemma4-openrouter-r13/patch_flashinfer_globaltimer.py /tmp/vllm-r13/patch_flashinfer_globaltimer.py
COPY docker/gemma4-openrouter-r13/test_mm_cache_rss.py /opt/vllm-r13-tests/test_mm_cache_rss.py

RUN set -eux; \
    test "$(sha256sum "$VLLM_PACKAGE_ROOT/utils/cache.py" | cut -d' ' -f1)" = "3fc9d9cfaf9bd86a992fb2c150b0466c91cf97547e91a858a6a96cbc2b367e18"; \
    test "$(sha256sum "$VLLM_PACKAGE_ROOT/config/multimodal.py" | cut -d' ' -f1)" = "310ed83a80f6ce4718bb0f02d3b07a2217061ef5eed756c784d26db939545add"; \
    test "$(sha256sum "$VLLM_PACKAGE_ROOT/multimodal/cache.py" | cut -d' ' -f1)" = "47343ccba0b59f4d47a0780668312e2f71abfa6f23375c7bf59ca251304ca1d1"; \
    test "$(sha256sum "$VLLM_PACKAGE_ROOT/renderers/online_renderer.py" | cut -d' ' -f1)" = "3b114527e647358cd470d806fca6de3c44b01cce11a3eff5e4ac41d2d7f00763"; \
    test "$(sha256sum "$VLLM_PACKAGE_ROOT/v1/structured_output/__init__.py" | cut -d' ' -f1)" = "0c77f58e1083ef67761024b7a89ab87e23e5cc363d32f3f09b2b7c47ffa507e8"; \
    test "$(sha256sum "$VLLM_PACKAGE_ROOT/v1/structured_output/backend_xgrammar.py" | cut -d' ' -f1)" = "cdaebca794c8aa919097ee250f1bc0fa87f8fe52a026eee4877605e18134752e"; \
    test "$(sha256sum "$FLASHINFER_PACKAGE_ROOT/utils.py" | cut -d' ' -f1)" = "912466a826fec0f0bdc15e7dd8caa02edb304374d2aade147f112ff09d058136"; \
    test "$(sha256sum /tmp/vllm-r13/vllm/utils/cache.py | cut -d' ' -f1)" = "397e993cebeeb44f37104e6f9a48ac7b4f3f878e0294fd8899e22f161de379c5"; \
    test "$(sha256sum /tmp/vllm-r13/vllm/config/multimodal.py | cut -d' ' -f1)" = "fbf34cab7761204219047795e4d950a8c8043e9b070efb89888704fc26dca2c4"; \
    test "$(sha256sum /tmp/vllm-r13/vllm/multimodal/cache.py | cut -d' ' -f1)" = "6d082a47edb023b717d789e1ac83edc2ba40b7d92a30d3d9e342a354413e5acc"; \
    test "$(sha256sum /tmp/vllm-r13/vllm/renderers/online_renderer.py | cut -d' ' -f1)" = "c7f75d35ac4db9c46744a729f1da081a1b43cc0d62e574e9ee335113a778eddb"; \
    test "$(sha256sum /tmp/vllm-r13/vllm/v1/structured_output/__init__.py | cut -d' ' -f1)" = "1c6a322064b797a2e29101e83d4b94b002ded88e6a088057d13149babe7d6925"; \
    test "$(sha256sum /tmp/vllm-r13/vllm/v1/structured_output/backend_xgrammar.py | cut -d' ' -f1)" = "e5b0c492195175ad1379b123a63fcd99c94aa1f2d9a8b4655d7e2d2e53f16de6"; \
    test "$(sha256sum /opt/vllm-r13-tests/tests/utils_/test_cache.py | cut -d' ' -f1)" = "276b76a6a666f0effd6c71fcb9cf120fca7ea12faa3942be94086b56350f13df"; \
    test "$(sha256sum /opt/vllm-r13-tests/tests/multimodal/test_cache.py | cut -d' ' -f1)" = "f6c69fc96c20984ec40498c5e2288f76759a2d7534be0ba968febcf77b766354"; \
    test "$(sha256sum /opt/vllm-r13-tests/tests/v1/spec_decode/test_mtp_structured_output.py | cut -d' ' -f1)" = "6c7c481161e4543d9128f85a3101f393020a468f4986542d7c9389165b7a8254"; \
    test "$(sha256sum /opt/vllm-r13-tests/tests/entrypoints/openai/test_render_parity.py | cut -d' ' -f1)" = "88a2809d42debd3d97443b785800f15ef639689a889984900e775fc13207bc8e"; \
    test "$(sha256sum /tmp/vllm-r13/patch_flashinfer_globaltimer.py | cut -d' ' -f1)" = "90a25bfa923a677769fc7c8373c9a593d9fcbfb539226bc3236cea40610bd0db"; \
    test "$(sha256sum /opt/vllm-r13-tests/test_mm_cache_rss.py | cut -d' ' -f1)" = "767c5fdaee44d7d5dc77f5e54ad9342bd4e7d59b5c12fdd91484c700520060c4"; \
    install -m 0644 /tmp/vllm-r13/vllm/utils/cache.py "$VLLM_PACKAGE_ROOT/utils/cache.py"; \
    install -m 0644 /tmp/vllm-r13/vllm/config/multimodal.py "$VLLM_PACKAGE_ROOT/config/multimodal.py"; \
    install -m 0644 /tmp/vllm-r13/vllm/multimodal/cache.py "$VLLM_PACKAGE_ROOT/multimodal/cache.py"; \
    install -m 0644 /tmp/vllm-r13/vllm/renderers/online_renderer.py "$VLLM_PACKAGE_ROOT/renderers/online_renderer.py"; \
    install -m 0644 /tmp/vllm-r13/vllm/v1/structured_output/__init__.py "$VLLM_PACKAGE_ROOT/v1/structured_output/__init__.py"; \
    install -m 0644 /tmp/vllm-r13/vllm/v1/structured_output/backend_xgrammar.py "$VLLM_PACKAGE_ROOT/v1/structured_output/backend_xgrammar.py"; \
    python3 /tmp/vllm-r13/patch_flashinfer_globaltimer.py "$FLASHINFER_PACKAGE_ROOT/utils.py"; \
    python3 -m py_compile \
      "$VLLM_PACKAGE_ROOT/utils/cache.py" \
      "$VLLM_PACKAGE_ROOT/config/multimodal.py" \
      "$VLLM_PACKAGE_ROOT/multimodal/cache.py" \
      "$VLLM_PACKAGE_ROOT/renderers/online_renderer.py" \
      "$VLLM_PACKAGE_ROOT/v1/structured_output/__init__.py" \
      "$VLLM_PACKAGE_ROOT/v1/structured_output/backend_xgrammar.py" \
      "$FLASHINFER_PACKAGE_ROOT/utils.py"; \
    test "$(sha256sum "$VLLM_PACKAGE_ROOT/_C_stable_libtorch.abi3.so" | cut -d' ' -f1)" = "e8d11cee75dc85fa1bde970655053a75dcdf2292619a5b7af5977345904d8f54"; \
    test ! -e /usr/local/cuda/include/cusparse.h; \
    ldd /usr/local/lib/python3.12/dist-packages/flashinfer_globaltimer.so | tee /tmp/globaltimer.ldd; \
    ! grep -q "not found" /tmp/globaltimer.ldd; \
    rm -rf /tmp/vllm-r13 /tmp/globaltimer.ldd

RUN python3 - <<'PY'
from unittest.mock import patch

import flashinfer.utils as utils

utils.get_globaltimer_kernel.cache_clear()
with patch(
    "torch.utils.cpp_extension.load_inline",
    side_effect=AssertionError("runtime attempted globaltimer JIT"),
):
    kernel = utils.get_globaltimer_kernel()
assert callable(kernel)
print("precompiled_globaltimer_direct_load=passed")
PY

LABEL org.opencontainers.image.source="https://github.com/zozyo/vllm" \
      org.opencontainers.image.revision="${SOURCE_COMMIT}" \
      org.opencontainers.image.description="Gemma4 r13 host-OOM hardening, H200 globaltimer precompile, tool and structured-output fixes" \
      io.phala.patch.scope="gemma4-openrouter-r12,mm-cache-trim,glibc-arena-bound,flashinfer-globaltimer-sm90,tool-strict,mtp-structured-output"
