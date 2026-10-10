"""Equivalence tests for the maskless `boost_attention="split"` path.

`ref_boost` is an additive log(b) logit bias on [target rows x reference columns].
The original path materializes it as a dense (L, L) bias, which forces a masked kernel.
`split` rebuilds the same softmax from maskless calls to the active backend. These tests
pin that both paths agree (up to float tolerance) for every boost shape the pack builds,
using an exact fp64 attention as the "backend".

Self-contained: installs a minimal ComfyUI stub when the real one is not importable.

Run standalone:  python tests/test_split_boost.py
Or via pytest:   pytest tests/test_split_boost.py
"""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_pre_encode import _load_pack  # noqa: E402  (reuses the ComfyUI stub)

PACK = _load_pack()
TOL = 1e-6


def _backend(q, k, v, heads, mask=None, skip_reshape=False, skip_output_reshape=False, **_):
    """Exact attention with the comfy calling convention (q/k/v already (B, H, L, D))."""
    s = (q @ k.transpose(-1, -2)) / q.shape[-1] ** 0.5
    if mask is not None:
        s = s + mask
    out = s.softmax(-1) @ v
    if skip_output_reshape:
        return out
    b, h, l, d = out.shape
    return out.transpose(1, 2).reshape(b, l, h * d)


def _qkv(heads, L, d=16, kvheads=None, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(1, heads, L, d, generator=g, dtype=torch.float64)
    k = torch.randn(1, kvheads or heads, L, d, generator=g, dtype=torch.float64)
    v = torch.randn(1, kvheads or heads, L, d, generator=g, dtype=torch.float64)
    return q, k, v


def _compare(boosts, slens, txtlen=7, tgtlen=12, mask=None, grids=None, heads=3, kvheads=None):
    L = txtlen + sum(slens) + tgtlen
    grids = grids or [(sl, 1) for sl in slens]
    q, k, v = _qkv(heads, L, kvheads=kvheads)
    bias = PACK._ref_attn_bias(boosts, mask, txtlen, slens, tgtlen, grids, "cpu", torch.float64)
    sets, rows0 = PACK._boost_sets(boosts, mask, txtlen, slens, grids, "cpu")
    override = PACK._make_split_boost_override(sets, rows0, None)
    if kvheads:
        rep = heads // kvheads
        dense = _backend(q, k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1), heads,
                         mask=bias, skip_reshape=True)
    else:
        dense = _backend(q, k, v, heads, mask=bias, skip_reshape=True)
    split = override(_backend, q, k, v, heads, mask=None, skip_reshape=True)
    plain = _backend(q, k, v, heads, skip_reshape=True) if not kvheads else None
    assert split.shape == dense.shape
    err = (split - dense).abs().max().item()
    assert err < TOL, f"split differs from dense bias by {err}"
    if plain is not None:
        assert (dense - plain).abs().max().item() > 1e-3, "test case has no boost effect"
    return err


def test_single_ref_boost_above_one():
    _compare([4.0], [10])


def test_single_ref_boost_below_one():
    _compare([0.5], [10])


def test_two_refs_with_distinct_boosts():
    _compare([2.0, 3.0], [9, 11])


def test_two_refs_only_one_boosted():
    _compare([1.0, 2.0], [9, 11])


def test_boost_mask_restricts_last_ref():
    mask = torch.zeros(1, 4, 5)
    mask[:, 1:3, 1:4] = 1
    _compare([4.0], [20], mask=mask, grids=[(4, 5)])


def test_unexpanded_gqa_kv():
    _compare([3.0], [10], heads=4, kvheads=2)


def test_skip_output_reshape_layout():
    L = 7 + 10 + 12
    q, k, v = _qkv(2, L)
    sets, rows0 = PACK._boost_sets([4.0], None, 7, [10], [(10, 1)], "cpu")
    override = PACK._make_split_boost_override(sets, rows0, None)
    flat = override(_backend, q, k, v, 2, skip_reshape=True)
    hnd = override(_backend, q, k, v, 2, skip_reshape=True, skip_output_reshape=True)
    assert flat.shape == (1, L, 2 * 16) and hnd.shape == (1, 2, L, 16)
    assert torch.allclose(flat, hnd.transpose(1, 2).reshape(1, L, 32))


def test_masked_calls_pass_through_untouched():
    L = 7 + 10 + 12
    q, k, v = _qkv(2, L)
    sets, rows0 = PACK._boost_sets([4.0], None, 7, [10], [(10, 1)], "cpu")
    override = PACK._make_split_boost_override(sets, rows0, None)
    other = torch.zeros(1, 1, L, L, dtype=torch.float64)
    other[..., :3] = -1e9
    expected = _backend(q, k, v, 2, mask=other, skip_reshape=True)
    assert torch.equal(override(_backend, q, k, v, 2, mask=other, skip_reshape=True), expected)


def test_boost_sets_select_the_same_columns_as_the_dense_bias():
    mask = torch.zeros(1, 4, 5)
    mask[:, 0:2, 2:5] = 1
    txtlen, slens, tgtlen = 3, [6, 20], 4
    boosts = [2.0, 5.0]
    bias = PACK._ref_attn_bias(boosts, mask, txtlen, slens, tgtlen, [(2, 3), (4, 5)], "cpu", torch.float64)
    sets, rows0 = PACK._boost_sets(boosts, mask, txtlen, slens, [(2, 3), (4, 5)], "cpu")
    assert rows0 == txtlen + sum(slens)
    rebuilt = torch.zeros_like(bias)
    for b, cols in sets:
        rebuilt[:, :, rows0:, cols] = torch.log(torch.tensor(b, dtype=torch.float64))
    assert torch.allclose(rebuilt, bias)


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        fn()
        print(f"ok  {name}")
    print(f"{len(tests)} passed")
