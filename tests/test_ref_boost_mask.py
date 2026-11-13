"""ref_boost attention bias: an empty ref_boost_mask must not produce a bias.

LoadImage's MASK output is all zeros for an opaque image, so wiring it to
ref_boost_mask without painting a region boosts nothing. The pack still built a dense
all-zero (L, L) bias, which changes no output but pushes every block onto a masked
attention kernel (Sage, Flash and Comfy Kitchen fall back to masked SDPA).

Run standalone:  python tests/test_ref_boost_mask.py
Or via pytest:   pytest tests/test_ref_boost_mask.py
"""
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_pre_encode import _load_pack  # noqa: E402

TXT, GRID, TGT = 5, (4, 3), 12        # text tokens, ref patch grid, target tokens
REF = GRID[0] * GRID[1]
ROWS0 = TXT + REF


def _bias(mod, boost, mask):
    return mod._ref_attn_bias([boost], mask, TXT, [REF], TGT, [GRID], "cpu", torch.float32)


def test_empty_mask_gives_no_bias():
    mod = _load_pack()
    assert _bias(mod, 2.0, torch.zeros(1, 64, 64)) is None


def test_unit_boost_gives_no_bias():
    mod = _load_pack()
    assert _bias(mod, 1.0, None) is None


def test_no_mask_boosts_whole_reference():
    mod = _load_pack()
    bias = _bias(mod, 2.0, None)
    assert torch.allclose(bias[0, 0, ROWS0:, TXT:ROWS0], torch.full((TGT, REF), math.log(2.0)))
    assert not bias[0, 0, :ROWS0].any()                      # text/ref rows untouched
    assert not bias[0, 0, ROWS0:, :TXT].any()                # text columns untouched
    assert not bias[0, 0, ROWS0:, ROWS0:].any()              # target columns untouched


def test_painted_mask_boosts_only_its_region():
    mod = _load_pack()
    mask = torch.zeros(1, 64, 64)
    mask[:, :32] = 1.0                                       # top half of the reference
    bias = _bias(mod, 2.0, mask)
    boosted = bias[0, 0, ROWS0:, TXT:ROWS0].reshape(TGT, *GRID)
    assert torch.allclose(boosted[:, :2], torch.full((TGT, 2, GRID[1]), math.log(2.0)))
    assert not boosted[:, 2:].any()


if __name__ == "__main__":
    failures = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as e:                           # noqa: BLE001
                failures += 1
                print(f"FAIL {name}: {e!r}")
    sys.exit(1 if failures else 0)
