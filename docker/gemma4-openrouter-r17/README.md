# Gemma4 OpenRouter r17 overlay

This overlay starts from the immutable r16 image and packages the Tool Call and
Structured Output repairs validated with the production Gemma4 checkpoint:

- `RedHatAI/gemma-4-31B-it-FP8-block`;
- `max_model_len=262144`;
- FlashAttention 4 and calibrated FP8 E4M3 KV cache;
- Gemma4 MTP with four speculative tokens.

The overlay carries the applicable portions of upstream PRs #51307, #53444,
#48922, #47450, #52020, #48416, #47562, #50015, #50944, and #47509. The build
records their exact head commits. PR #49625 is excluded because r16 supports
concurrent per-request backends and passed a 32-request mixed xgrammar/guidance
test. PR #40097 is excluded because repetition detection truncated valid
repeated-value JSON during production-model regression testing.

The Dockerfile fails closed on any r16 base-file drift, verifies every patched
file before and after installation, and compiles the installed Python sources
through an isolated `uv` virtual environment. Source tests and production-model
API gates remain release evidence rather than runtime image content.
