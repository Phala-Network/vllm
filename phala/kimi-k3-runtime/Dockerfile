FROM vllm/vllm-openai:v0.31.0@sha256:a4a4c0437bf7240089da5f08aa370c4aee17ae5290f7a3b468825ee26c4c3a6b
COPY artifacts/target/ /opt/runtime-model/target/
COPY artifacts/draft/ /opt/runtime-model/draft/
COPY manifest.json install_view.py cpu_check.py /opt/runtime-model/
COPY layout_audit.py patch_layout.py layout_fixture.py /opt/runtime-model/
RUN python3 /opt/runtime-model/install_view.py && python3 /opt/runtime-model/patch_layout.py && python3 /opt/runtime-model/layout_fixture.py
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_MODULES_CACHE=/tmp/kimi-k3-hf-modules
