# Gemma4 OpenRouter r15 overlay

This overlay builds from the immutable r14 image and fixes the production
regression found by the `use2-19` canary: r14 used `fork` from vLLM's grammar
`ThreadPoolExecutor`, causing native XGrammar compilation children to exit with
code 1 even for a simple regex.

r15 uses a preloaded `forkserver` and warms it before vLLM creates the grammar
executor. The one-time startup cost is paid during Engine initialization;
request-time workers then start below the existing regex compilation deadline.
Timeout cleanup remains kill-and-join, and the pipe result channel still drains
large results before joining the child.

The image build verifies the immutable r14 inputs, runs the focused timeout
suite, and performs native XGrammar compilation from a worker thread. The
production canary additionally validates the Gemma4 tokenizer path, timeout
cleanup, service health, and RSS behavior.
