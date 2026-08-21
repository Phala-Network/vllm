# Gemma4 OpenRouter r13 overlay

This image extends the immutable r12 image without changing the serving model,
262144 maximum context, FA4, FP8 E4M3 KV, MTP4, or prefix-cache behavior.

OOM hardening:

- backports merged vLLM PR #53016 so a single MM item larger than the processor
  cache is served uncached instead of aborting the engine;
- bounds glibc to eight arenas and lowers the trim threshold;
- calls `malloc_trim(0)` after P1 MM LRU eviction, throttled to once per 30
  seconds, so the 4 GiB logical cache cannot leave unbounded allocator-retained
  RSS under image churn.

Tool and Structured Output hardening:

- backports merged PR #52805 for XGrammar termination with MTP;
- carries approved PR #53046 to validate post-reasoning speculative tokens
  before advancing the grammar;
- strips the OpenAI server-side function `strict` directive before rendering
  tools into the model prompt.

FlashInfer `%globaltimer`:

- compiles only SM90 (`TORCH_CUDA_ARCH_LIST=9.0`) in a build stage with
  `libcusparse-dev-12-9` and `libcusolver-dev-12-9`;
- copies only the extension into the runtime image;
- keeps `cusparse.h` out of the final image and directly imports the prebuilt
  module before the JIT fallback.

PRs #52452, #52830, and #49461 remain unmerged and are intentionally excluded.
The image must pass source tests, CPU RSS churn, stable-ABI hash, prebuilt-load,
and no-production-container-change gates before publication or deployment.
