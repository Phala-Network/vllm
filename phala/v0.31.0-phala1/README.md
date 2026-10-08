# vLLM 0.31.0 Phala runtime 1

Candidate packaging for the official vLLM 0.31.0 image with two focused fixes:

- Mooncake 0.3.13.post1 TCP recovery after queue rejection, from upstream commit `74d26b56e535b02427e94c3a6087c31a4cfc4963`.
- Request-local structured-output backend selection, so JSON and named-tool grammars can safely use different backends. Guidance compilation also keeps its serialized grammar local to each call.

The complete source branch starts at official vLLM `db9527a46873454610df6dbedf79a36d6bf1a7f6`. The recipe uses the unchanged official image by digest, verifies both original Python modules before replacing them, and installs only the fixed CUDA13 Mooncake wheel with dependency resolution disabled. Model weights, compiled vLLM kernels, the Rust frontend, and serving arguments are inherited unchanged.

Obtain the CPython 3.12 wheel and its SHA256 manifest from the [Mooncake source Release](https://github.com/Phala-Network/Mooncake/releases/tag/phala-v0.3.13.post1-1), placing both files in this directory's ignored `artifacts/` subdirectory. Do not use an unverified replacement wheel.

```bash
docker build -f phala/v0.31.0-phala1/Dockerfile \
  -t ghcr.io/phala-network/vllm:0.31.0-phala1 .
```

Before publishing, the exact source commit must have a public tag and Release. Run final-image tests outside the source checkout without source mounts, recording imported module paths and hashes. Required gates include both `hf --help` and `hf download --help` through the direct image entrypoint; installed-module mixed-backend CPU tests; and package versions. GPU inference, cache producer/consumer checks and rollout adoption are separate runtime gates. This candidate is not an accepted runtime until its Release records the image digest and those receipts.

The shared Mooncake master/store can retain the original official image: the client-side pump fix does not change the wire protocol or require restarting shared cache services.
