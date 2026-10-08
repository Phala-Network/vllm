# Phala Kimi-K3 runtime

This recipe derives from official vLLM v0.31.0, pinned by digest in Dockerfile. It bakes revision-pinned RedHatAI Kimi-K3 tokenizer/processor metadata and adds a source-guarded registration-time Mooncake KV layout audit. Model weights remain external read-only mounts. No transfer algorithm is changed.

## Build

From this directory on a Linux Docker builder:

```sh
uv run --no-project --python 3.12 prepare.py
docker build --network=none -t ghcr.io/phala-network/vllm:0.31.0-kimi-k3-runtime-20261008 .
```

The frozen manifest verifies downloaded artifacts. Do not regenerate it for this release. Mount the target and draft weights read-only at the respective `weights_root` paths in manifest.json. Use `/opt/runtime-model/target` and `/opt/runtime-model/draft` as model paths; do not overlay those directories. The manifest pins both HF revisions and weight hashes. No weights are included in the build context.

## Checks

Run against the built image without GPUs, credentials or model mounts:

```sh
docker run --rm --network=none --entrypoint hf IMAGE --help
docker run --rm --network=none --entrypoint hf IMAGE download --help
docker run --rm --network=none --entrypoint python3 IMAGE /opt/runtime-model/cpu_check.py
docker run --rm --network=none --entrypoint python3 IMAGE /opt/runtime-model/layout_fixture.py
```

The CPU fixture checks layout identity and rejects incompatible shape/stride/storage mappings. Unknown audit layouts fail at registration. These checks do not establish model inference or throughput acceptance.

## Release provenance

This source directory retroactively archives the exact custom build inputs used on 2026-10-08. The historical image was built before this Git commit existed; it is not claimed to have been built by checking out this commit. `historical-build.json` records byte-level correspondence, pinned base, image ID and completed checks. The immutable GHCR manifest digest is recorded in the GitHub Release after publication. This Kimi image is distinct from `phala-v0.31.0-1` and does not include that release's structured-output/Mooncake wheel fixes.
