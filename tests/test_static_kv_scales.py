# SPDX-License-Identifier: Apache-2.0

import json

import pytest
import torch

from vllm.model_executor.layers.attention.attention import (
    apply_static_kv_scales,
    record_kv_scale_calibration_sample,
)


class DummyAttention(torch.nn.Module):
    def __init__(
        self,
        layer_name: str = "model.layers.0.self_attn.attn",
        scales_are_parameters: bool = False,
    ):
        super().__init__()
        self.layer_name = layer_name
        self.kv_cache_dtype = "fp8_e4m3"
        self.num_kv_heads = 2
        if scales_are_parameters:
            self.register_parameter(
                "_k_scale", torch.nn.Parameter(torch.tensor(1.0), requires_grad=False)
            )
            self.register_parameter(
                "_v_scale", torch.nn.Parameter(torch.tensor(1.0), requires_grad=False)
            )
        else:
            self.register_buffer("_k_scale", torch.tensor(1.0, dtype=torch.float32))
            self.register_buffer("_v_scale", torch.tensor(1.0, dtype=torch.float32))
        self._k_scale_float = 1.0
        self._v_scale_float = 1.0
        self._k_scale_cpu = torch.tensor(1.0, dtype=torch.float32)
        self._v_scale_cpu = torch.tensor(1.0, dtype=torch.float32)


def test_apply_static_kv_scales_per_head(tmp_path, monkeypatch):
    path = tmp_path / "scales.json"
    path.write_text(
        json.dumps(
            {
                "schema": "vllm-static-kv-scales-v1",
                "layers": {
                    "model.layers.0.self_attn.attn": {
                        "k_scale": [0.01, 0.02],
                        "v_scale": [0.03, 0.04],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("VLLM_STATIC_KV_SCALE_PATH", str(path))
    layer = DummyAttention()

    assert apply_static_kv_scales(layer, layer.layer_name)
    torch.testing.assert_close(layer._k_scale, torch.tensor([0.01, 0.02]))
    torch.testing.assert_close(layer._v_scale, torch.tensor([0.03, 0.04]))
    assert layer._k_scale_float == pytest.approx(0.02)
    assert layer._v_scale_float == pytest.approx(0.04)


def test_apply_static_kv_scales_replaces_compressed_tensor_parameters(
    tmp_path, monkeypatch
):
    path = tmp_path / "scales.json"
    path.write_text(
        json.dumps(
            {
                "schema": "vllm-static-kv-scales-v1",
                "layers": {
                    "model.layers.0.self_attn.attn": {
                        "k_scale": [0.01, 0.02],
                        "v_scale": [0.03, 0.04],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("VLLM_STATIC_KV_SCALE_PATH", str(path))
    layer = DummyAttention(scales_are_parameters=True)

    assert apply_static_kv_scales(layer, layer.layer_name)
    assert "_k_scale" not in layer._parameters
    assert "_v_scale" not in layer._parameters
    assert "_k_scale" in layer._buffers
    assert "_v_scale" in layer._buffers
    torch.testing.assert_close(layer._k_scale, torch.tensor([0.01, 0.02]))
    torch.testing.assert_close(layer._v_scale, torch.tensor([0.03, 0.04]))


def test_apply_static_kv_scales_requires_every_quantized_layer(tmp_path, monkeypatch):
    path = tmp_path / "scales.json"
    path.write_text(
        json.dumps({"schema": "vllm-static-kv-scales-v1", "layers": {}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("VLLM_STATIC_KV_SCALE_PATH", str(path))
    layer = DummyAttention()

    with pytest.raises(KeyError, match="Missing static KV scales"):
        apply_static_kv_scales(layer, layer.layer_name)


def test_record_kv_scale_calibration_sample(tmp_path, monkeypatch):
    path = tmp_path / "amax.jsonl"
    monkeypatch.setenv("VLLM_KV_SCALE_CALIBRATION_PATH", str(path))
    layer = DummyAttention()
    key = torch.tensor(
        [
            [[-1.0, 2.0], [3.0, -4.0]],
            [[5.0, -6.0], [-7.0, 8.0]],
        ]
    )
    value = key * 0.5
    slot_mapping = torch.tensor([0, 1])

    record_kv_scale_calibration_sample(layer, key, value, slot_mapping)

    sample = json.loads(path.read_text(encoding="utf-8"))
    assert sample["schema"] == "vllm-kv-amax-sample-v1"
    assert sample["layer"] == layer.layer_name
    assert sample["tokens"] == 2
    assert sample["k_amax"] == [6.0, 8.0]
    assert sample["v_amax"] == [3.0, 4.0]
