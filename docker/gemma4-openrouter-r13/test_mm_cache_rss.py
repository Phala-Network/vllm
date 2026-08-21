#!/usr/bin/env python3
from __future__ import annotations

import gc
import json
import os
from pathlib import Path

from vllm.config.multimodal import MultiModalConfig
from vllm.multimodal.cache import MultiModalReceiverCache, _MM_CACHE_MALLOC_TRIMMER
from vllm.multimodal.inputs import MultiModalKwargsItem
from vllm.utils.mem_constants import GiB_bytes, MiB_bytes


def rss_bytes() -> int:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("VmRSS is unavailable")


class StubModelConfig:
    def get_multimodal_config(self) -> MultiModalConfig:
        return MultiModalConfig(mm_processor_cache_gb=(48 * MiB_bytes) / GiB_bytes)


def insert(cache: MultiModalReceiverCache, index: int) -> None:
    item = MultiModalKwargsItem.dummy(nbytes=32 * MiB_bytes)
    assert cache.get_and_update_item(item, f"item-{index}") is item


def main() -> None:
    assert os.environ.get("MALLOC_ARENA_MAX") == "8"
    assert os.environ.get("MALLOC_TRIM_THRESHOLD_") == "131072"

    cache = MultiModalReceiverCache(StubModelConfig())
    original_interval = _MM_CACHE_MALLOC_TRIMMER.interval_s
    _MM_CACHE_MALLOC_TRIMMER.interval_s = 0.0
    try:
        for index in range(8):
            insert(cache, index)
        cache.clear_cache()
        gc.collect()
        _MM_CACHE_MALLOC_TRIMMER.maybe_trim()
        baseline = rss_bytes()
        peak = baseline

        for index in range(8, 104):
            insert(cache, index)
            peak = max(peak, rss_bytes())

        cache.clear_cache()
        gc.collect()
        _MM_CACHE_MALLOC_TRIMMER.maybe_trim()
        final = rss_bytes()
    finally:
        _MM_CACHE_MALLOC_TRIMMER.interval_s = original_interval

    result = {
        "baseline_rss_mib": round(baseline / MiB_bytes, 2),
        "peak_rss_mib": round(peak / MiB_bytes, 2),
        "final_rss_mib": round(final / MiB_bytes, 2),
        "peak_delta_mib": round((peak - baseline) / MiB_bytes, 2),
        "final_delta_mib": round((final - baseline) / MiB_bytes, 2),
        "churn_gib": 3.0,
    }
    print(json.dumps(result, sort_keys=True))

    assert peak - baseline < 512 * MiB_bytes, result
    assert final - baseline < 128 * MiB_bytes, result


if __name__ == "__main__":
    main()
