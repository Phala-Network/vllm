ARG BASE_IMAGE=ghcr.io/phala-network/vllm-openai:main-aa9903490-cu130-ubuntu2404-gemma4-pth-toolfix-r7@sha256:42bb76d21f2806b79a2e0dfa2f9e26a6dbbe95e5232b1996b3641bd726d027b5
FROM ${BASE_IMAGE}

ARG SOURCE_COMMIT

ENV CUTE_INTERFACE=/usr/local/lib/python3.12/dist-packages/vllm/vllm_flash_attn/cute/interface.py \
    VLLM_STATIC_KV_SCALE_PATH=/opt/vllm/kv-scales/gemma4-static-kv-scales.json

COPY vllm/model_executor/layers/attention/attention.py /tmp/vllm-source/attention.py
COPY vllm/v1/attention/backends/flash_attn.py /tmp/vllm-source/flash_attn.py
COPY vllm/v1/worker/gpu_model_runner.py /tmp/vllm-source/gpu_model_runner.py
COPY vllm/vllm_flash_attn/flash_attn_interface.py /tmp/vllm-source/flash_attn_interface.py
COPY docker/gemma4-fa4-fp8/assets/cute_interface.py /tmp/vllm-source/cute_interface.py
COPY docker/gemma4-fa4-fp8/assets/gemma4-static-kv-scales.json /tmp/vllm-source/gemma4-static-kv-scales.json
COPY tests/test_static_kv_scales.py /opt/vllm-tests/test_static_kv_scales.py
COPY tests/vllm_flash_attn/test_flash_attn_interface.py /opt/vllm-tests/test_flash_attn_interface.py

RUN test "$(sha256sum "$CUTE_INTERFACE" | cut -d' ' -f1)" = "36be77b5b32066fa3d70c3d698edba2108e2f8fd1d91d74eb8a24e0972e39351" \
    && test "$(sha256sum /tmp/vllm-source/attention.py | cut -d' ' -f1)" = "3777c115cb706ca2911e64a17ed9a5abf72bc9f6926f34736b8416f1fb935664" \
    && test "$(sha256sum /tmp/vllm-source/flash_attn.py | cut -d' ' -f1)" = "2b6950c048f9a7177c64d7df298fbdeae4abf0fb1f80843c60aa9b41f76094ba" \
    && test "$(sha256sum /tmp/vllm-source/gpu_model_runner.py | cut -d' ' -f1)" = "f3f53457e51fce5f922b43fb491c4ec480cd63016922aab2995e7e8da99ecabc" \
    && test "$(sha256sum /tmp/vllm-source/flash_attn_interface.py | cut -d' ' -f1)" = "8196c80c3120467c16160e2bcd4b7f5edbba2112e362a777237e8c14ed490efe" \
    && test "$(sha256sum /opt/vllm-tests/test_static_kv_scales.py | cut -d' ' -f1)" = "b7f69e1dc2da9c87476923466f64647253adfd5d783a9afd26487dce0d7781c5" \
    && test "$(sha256sum /opt/vllm-tests/test_flash_attn_interface.py | cut -d' ' -f1)" = "563bea0ff2cbb0e402987a5186a5b098571ae210301f188a6a49854d1552644f" \
    && test "$(sha256sum /tmp/vllm-source/cute_interface.py | cut -d' ' -f1)" = "088cc05a287e52c22264f13412958393d40a0ec5dc5687d388fba5f0e05d106b" \
    && test "$(sha256sum /tmp/vllm-source/gemma4-static-kv-scales.json | cut -d' ' -f1)" = "199c8e529b0515fcc789c89a07f3b23b0c7221fcfc51332968054f5cf0b1bdc6" \
    && install -m 0644 /tmp/vllm-source/attention.py /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/attention/attention.py \
    && install -m 0644 /tmp/vllm-source/flash_attn.py /usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/flash_attn.py \
    && install -m 0644 /tmp/vllm-source/gpu_model_runner.py /usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py \
    && install -m 0644 /tmp/vllm-source/flash_attn_interface.py /usr/local/lib/python3.12/dist-packages/vllm/vllm_flash_attn/flash_attn_interface.py \
    && install -m 0644 /tmp/vllm-source/cute_interface.py "$CUTE_INTERFACE" \
    && install -D -m 0444 /tmp/vllm-source/gemma4-static-kv-scales.json "$VLLM_STATIC_KV_SCALE_PATH" \
    && python3 -m py_compile \
        /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/attention/attention.py \
        /usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/flash_attn.py \
        /usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py \
        /usr/local/lib/python3.12/dist-packages/vllm/vllm_flash_attn/flash_attn_interface.py \
        "$CUTE_INTERFACE" \
    && rm -rf /tmp/vllm-source \
    && test "$(sha256sum "$CUTE_INTERFACE" | cut -d' ' -f1)" = "088cc05a287e52c22264f13412958393d40a0ec5dc5687d388fba5f0e05d106b" \
    && test "$(sha256sum "$VLLM_STATIC_KV_SCALE_PATH" | cut -d' ' -f1)" = "199c8e529b0515fcc789c89a07f3b23b0c7221fcfc51332968054f5cf0b1bdc6"

RUN python3 -m pytest -q --confcutdir=/opt/vllm-tests \
    /opt/vllm-tests/test_static_kv_scales.py

LABEL org.opencontainers.image.source="https://github.com/zozyo/vllm" \
      org.opencontainers.image.revision="${SOURCE_COMMIT}" \
      org.opencontainers.image.description="Gemma4 uniform FA4, page64, calibrated per-head FP8 E4M3 KV, SM90 single-stage dequant" \
      io.phala.patch.scope="gemma4-pth-kv,page-alignment,dynamic-pth-scale,gemma4-tool-types,named-tool-rejection,incomplete-tool-withholding,const-schema-types,shared-engine-named-tool-rejection,tool-none-prompt-guard,static-kv-scales,fa4-fp8-single-stage"
