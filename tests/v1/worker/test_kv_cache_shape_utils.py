# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.worker.kv_cache_shape_utils import (
    get_padded_attention_kv_cache_shape,
    scale_padded_page_size,
)


def test_get_padded_attention_kv_cache_shape_expands_last_dimension():
    shape = get_padded_attention_kv_cache_shape(
        (7, 1, 16, 1032),
        num_blocks=7,
        padded_page_size_bytes=16 * 1040,
        dtype=torch.uint8,
    )

    assert shape == (7, 1, 16, 1040)


def test_scale_padded_page_size_for_kernel_block_size():
    assert (
        scale_padded_page_size(
            32 * 1040,
            block_size=32,
            target_block_size=16,
        )
        == 16 * 1040
    )


def test_scale_padded_page_size_rejects_non_divisible_scaling():
    with pytest.raises(ValueError, match="divisible"):
        scale_padded_page_size(
            65,
            block_size=10,
            target_block_size=3,
        )


def test_get_padded_attention_kv_cache_shape_rejects_shrink():
    with pytest.raises(ValueError, match="shrink"):
        get_padded_attention_kv_cache_shape(
            (7, 1, 16, 1032),
            num_blocks=7,
            padded_page_size_bytes=16 * 1024,
            dtype=torch.uint8,
        )


def test_get_padded_attention_kv_cache_shape_rejects_misaligned_page():
    with pytest.raises(ValueError, match="does not align"):
        get_padded_attention_kv_cache_shape(
            (7, 3, 16, 1032),
            num_blocks=7,
            padded_page_size_bytes=16 * 1040,
            dtype=torch.uint8,
        )
