# Gemma4 OpenRouter r11 overlay

r11 is pinned to the immutable r10 image and changes only the Outlines V1
structured-output backend.

r10 correctly routes schemas unsupported by xgrammar and llguidance to
outlines. Live Gemma4 requests then exposed an upstream termination mismatch:
after the Outlines regex reaches its accepting state, vLLM delays the grammar's
terminated flag for one step so the model can emit EOS. The existing
`accept_tokens()` implementation still sends that EOS to the already-finished
Outlines guide, which rejects token `1` and turns a valid JSON completion into
an HTTP 500.

r11 passes the request stop-token IDs into `OutlinesGrammar`. During the
backend's documented delayed-termination step it accepts a leading stop token
without advancing the completed guide; validation returns the same one-token
prefix. Other tokens remain rejected.

The regression suite includes a dependency-free completed-guide test and a
real xgrammar/outlines mixed-manager test that generates `[7]`, accepts EOS,
and reaches the terminated state. The production configuration keeps
speculative decoding disabled.
