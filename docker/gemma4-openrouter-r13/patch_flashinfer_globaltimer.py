#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

EXPECTED_INPUT_SHA256 = (
    "912466a826fec0f0bdc15e7dd8caa02edb304374d2aade147f112ff09d058136"
)

OLD = '''@lru_cache(maxsize=1)
def get_globaltimer_kernel():
    """Lazily JIT-build the %globaltimer kernel."""

    _GLOBALTIMER_KERNEL_CU = r"""
'''

NEW = '''@lru_cache(maxsize=1)
def get_globaltimer_kernel():
    """Load the image-precompiled %globaltimer kernel, with JIT fallback."""

    try:
        from flashinfer_globaltimer import get_globaltimer_timestamp
    except ModuleNotFoundError as e:
        if e.name != "flashinfer_globaltimer":
            raise
    else:
        return get_globaltimer_timestamp

    _GLOBALTIMER_KERNEL_CU = r"""
'''


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: patch_flashinfer_globaltimer.py FLASHINFER_UTILS_PY")

    path = Path(sys.argv[1])
    raw = path.read_bytes()
    actual = sha256(raw)
    if actual != EXPECTED_INPUT_SHA256:
        raise SystemExit(
            f"refusing to patch unexpected FlashInfer utils.py: {actual}"
        )

    text = raw.decode("utf-8")
    if text.count(OLD) != 1:
        raise SystemExit("FlashInfer globaltimer patch point is not unique")
    updated = text.replace(OLD, NEW, 1).encode("utf-8")
    path.write_bytes(updated)
    print(f"flashinfer_utils_patched_sha256={sha256(updated)}")


if __name__ == "__main__":
    main()
