# Gemma4 OpenRouter r9 overlay

This overlay is intentionally based on the immutable Gemma4 r8 FA4/FP8 image.
It replaces only the two Python modules changed by the OpenRouter structured
output and Gemma4 reasoning compatibility patch.

The build fails closed unless both source modules in the r8 image and both
replacement modules have the expected SHA-256 digests. It also compiles the
installed modules and runs the focused boolean `structured_outputs` regression
against the installed image package.

Build with the Git revision containing this recipe:

```bash
docker build \
  --build-arg SOURCE_COMMIT="$(git rev-parse HEAD)" \
  -f docker/gemma4-openrouter-r9/Dockerfile \
  -t ghcr.io/phala-network/vllm-openai:main-aa9903490-cu130-ubuntu2404-gemma4-pth-toolfix-r9 \
  .
```
