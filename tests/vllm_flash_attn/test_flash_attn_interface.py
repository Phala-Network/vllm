import pytest

from vllm.vllm_flash_attn.flash_attn_interface import _get_fa4_fp8_kv_tile_mn


@pytest.mark.parametrize(
    ("fp8_kv_dequant", "head_dim", "page_size", "expected"),
    [
        (True, 256, 64, (128, 64)),
        (False, 256, 64, None),
        (True, 512, 64, None),
        (True, 192, 64, None),
        (True, 256, 128, None),
        (True, 256, None, None),
    ],
)
def test_get_fa4_fp8_kv_tile_mn(
    fp8_kv_dequant: bool,
    head_dim: int,
    page_size: int | None,
    expected: tuple[int, int] | None,
) -> None:
    assert (
        _get_fa4_fp8_kv_tile_mn(
            fp8_kv_dequant=fp8_kv_dequant,
            head_dim=head_dim,
            page_size=page_size,
        )
        == expected
    )
