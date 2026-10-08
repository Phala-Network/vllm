FROM vllm/vllm-openai:v0.31.0@sha256:c1c9f6fd5c109ba7f0546a59f5b2f15fb87f64c77782e90a27b648b42a8e67c3

# The published source contains complete v0.31.0 sources and the two patches.
# Build context artifacts are downloaded from the pinned Mooncake Release.
COPY phala/v0.31.0-phala1/original-module-sha256.txt /tmp/original-module-sha256.txt
RUN sha256sum --check /tmp/original-module-sha256.txt && rm /tmp/original-module-sha256.txt
COPY phala/v0.31.0-phala1/artifacts/ /tmp/phala-wheel/
RUN cd /tmp/phala-wheel && sha256sum --check SHA256SUMS \
    && UV_NO_CACHE=1 uv pip install --system --no-deps --reinstall \
       ./mooncake_transfer_engine_cuda13-0.3.13.post1+phala.1-cp312-cp312-manylinux_2_28_x86_64.whl \
    && rm -rf /tmp/phala-wheel
COPY vllm/v1/structured_output/__init__.py /usr/local/lib/python3.12/dist-packages/vllm/v1/structured_output/__init__.py
COPY vllm/v1/structured_output/backend_guidance.py /usr/local/lib/python3.12/dist-packages/vllm/v1/structured_output/backend_guidance.py

LABEL org.opencontainers.image.source="https://github.com/Phala-Network/vllm" \
      org.opencontainers.image.version="0.31.0-phala1"
