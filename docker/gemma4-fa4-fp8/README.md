# Gemma4 FA4 with calibrated FP8 KV

This overlay packages the use1-19 Gemma4 FA4 and FP8 E4M3 KV fixes on top of
the immutable Phala r7 image.

The RedHat FP8 checkpoint does not contain usable per-head K/V scales. The
included sidecar was calibrated across all 60 Gemma4 layers and uses 10%
headroom. Its SHA-256 is
`199c8e529b0515fcc789c89a07f3b23b0c7221fcfc51332968054f5cf0b1bdc6`.

The SM90 FA4 FP8 dequant path writes K/V tiles into one shared-memory stage.
The included CuTe file therefore selects one stage for FP8 KV dequant while
retaining the normal two-stage d256 pipeline for non-FP8 workloads. The
Dockerfile verifies both the original installed CuTe SHA and the replacement
SHA before installing the fix.

Build from the repository root:

```bash
docker build \
  --build-arg SOURCE_COMMIT="$(git rev-parse HEAD)" \
  -f docker/gemma4-fa4-fp8/Dockerfile \
  -t ghcr.io/phala-network/vllm-openai:main-aa9903490-cu130-ubuntu2404-gemma4-pth-toolfix-r8 \
  .
```

Run the FlashAttention interface regression with a GPU after the image build:

```bash
docker run --rm --gpus all --entrypoint python3 "$IMAGE" \
  -m pytest -q --confcutdir=/opt/vllm-tests \
  /opt/vllm-tests/test_flash_attn_interface.py
```

The validated serving configuration uses `--attention-backend FLASH_ATTN`,
`--kv-cache-dtype fp8_e4m3`, `--block-size 64`, and
`--max-model-len 262144`. MTP is disabled.
