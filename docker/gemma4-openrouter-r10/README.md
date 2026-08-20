# Gemma4 OpenRouter r10 overlay

This overlay is pinned to the immutable r9 image and changes only vLLM's V1
structured-output manager and llguidance feature detection.

The V1 request validator can select `xgrammar`, `guidance`, or `outlines` for
each schema when the server uses `backend=auto`. Upstream EngineCore currently
caches only the backend selected by the first request. A later schema that
needs another backend is therefore compiled by the wrong implementation.

r10 keeps one backend instance per selected backend and compiles each request
with its validator-selected implementation. The shared token bitmask remains
compatible because these implementations use the same int32 vocabulary bitset
layout; the manager verifies the allocated shape before admitting a new
backend. The focused real-backend test mixes xgrammar and outlines grammars in
one manager and fills both rows of the same bitmask.

The overlay also treats `contains`, `minContains`, `maxContains`, `uniqueItems`,
`propertyNames`, and `patternProperties` as unsupported by the installed
llguidance matcher. Auto mode can then choose outlines when outlines supports
the schema. In the pinned dependency set, outlines supports the first three
tested OpenRouter corpus classes but still rejects `patternProperties`; r10
does not pretend that unsupported keyword is enforced.

Build-time checks fail closed on the exact r9 source hashes and the exact r10
replacement hashes. The image runs the mixed-backend regression and the
structured-output validation suite during build. The post-build gate must also
run the real xgrammar/outlines shared-bitmask test with the Hugging Face cache
available before deployment.
