# Gemma4 OpenRouter r14 overlay

This overlay is built from the immutable Gemma4 r13 image and replaces only the
vLLM structured-output files needed to make regex compilation workers killable.

It backports vLLM PR #52119 and adds a pipe-based parent/child result channel.
The parent drains large results before joining the child, closes both IPC ends,
and always kills and joins a worker that crosses the request deadline. This
avoids the thread leak and host-memory OOM from the previous timeout path, as
well as the `multiprocessing.Queue.empty()` race and feeder-thread deadlock in
the upstream candidate.

The Dockerfile guards every source and destination file by SHA-256 and runs the
focused timeout suite plus a native XGrammar regex compile during the build.
