"""Flash-attention compatibility of the ref_boost bias path (issue #6).

`--use-flash-attention` + any attention mask makes ComfyUI's flash backend bail out:
    [WARNING] Flash Attention failed, using default SDPA: Mask must not be set for Flash attention
per block per step, with a ~2x sampling slowdown (the SDPA fallback materializes the
(L,L) bias). The fix keeps every attention call maskless: the biased softmax is rebuilt
exactly from flash's logsumexp (see _make_flash_bias_override in __init__.py).

These tests pin:
  1. the override's math against an fp32 SDPA-with-additive-bias reference,
  2. the bias-mask parser (_bias_to_sets),
  3. an end-to-end tiny SingleStreamDiT forward: no flash warnings, boost still active,
     and numerical agreement with the old masked-SDPA path.

Needs CUDA + flash_attn. Run standalone:  python tests/test_flash_bias_attention.py
Or via pytest:                        pytest tests/test_flash_bias_attention.py
"""
import importlib.util
import logging
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PACK = os.path.dirname(_HERE)                         # .../custom_nodes/comfyui-krea2edit
_COMFY_ROOT = os.path.dirname(os.path.dirname(_PACK))  # .../ComfyUI

# Enable ComfyUI's flash-attention backend BEFORE comfy is imported (module-level
# switch in comfy.ldm.modules.attention). Only possible when comfy isn't in sys.modules
# yet; under a shared pytest session without flash, the warning-count assertions skip.
_COMFY_PRELOADED = "comfy" in sys.modules
if not _COMFY_PRELOADED:
    sys.path.insert(0, _COMFY_ROOT)
    sys.argv = [sys.argv[0], "--use-flash-attention"]
    import comfy.options  # noqa: E402

    comfy.options.enable_args_parsing(True)


import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402


def _load_pack():
    """Import the pack's __init__.py under a synthetic name (folder has a hyphen)."""
    spec = importlib.util.spec_from_file_location(
        "comfyui_krea2edit_under_test", os.path.join(_PACK, "__init__.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _need_flash():
    mod = _load_pack()
    assert mod._flash_attn_func is not None, "flash_attn is not installed in this environment"
    assert torch.cuda.is_available(), "no CUDA device"
    return mod


class _WarningCollector(logging.Handler):
    """Captures comfy's 'Flash Attention failed' warnings."""

    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        if "Flash Attention failed" in record.getMessage():
            self.records.append(record.getMessage())


def _sdpa_reference(q, k, v, bias):
    """fp32 additive-bias attention, output shaped like comfy's (B, L, H*D)."""
    out = F.scaled_dot_product_attention(q.float(), k.float(), v.float(), attn_mask=bias.float())
    return out.transpose(1, 2).reshape(q.shape[0], q.shape[2], q.shape[1] * q.shape[3])


def _run_override(mod, q, k, v, bias):
    """Dispatch through comfy's real optimized_attention_masked + wrap_attn override path."""
    import comfy.ldm.modules.attention as A

    to = {"optimized_attention_override": mod._make_flash_bias_override(None)}
    return A.optimized_attention_masked(q, k, v, q.shape[1], mask=bias,
                                        skip_reshape=True, transformer_options=to)


def _assert_close(actual, expected, label, atol=3e-2, rtol=3e-2):
    actual, expected = actual.float(), expected.float()
    diff = (actual - expected).abs().max().item()
    ok = torch.allclose(actual, expected, atol=atol, rtol=rtol)
    assert ok, f"{label}: max abs diff {diff:.4g} exceeds tol (atol={atol}, rtol={rtol})"


def test_bias_to_sets_parses_ref_bias_structure():
    mod = _need_flash()
    # [text 0:16 | ref_a 16:48 | ref_b 48:64 | target 64:96); non-contiguous mask on ref_b
    L = 96
    bias = torch.zeros(1, 1, L, L, dtype=torch.bfloat16)
    bias[:, :, 64:, 16:48] = torch.log(torch.tensor(2.0, dtype=torch.bfloat16))
    cols_b = torch.zeros(L, dtype=torch.bool)
    cols_b[[48, 50, 55, 63]] = True                      # sparse boost_mask region
    bias[:, :, 64:, cols_b] = torch.log(torch.tensor(0.5, dtype=torch.bfloat16))

    parsed = mod._bias_to_sets(bias)
    assert parsed is not None
    sets, rows = parsed
    factors = sorted(b for b, _ in sets)
    assert abs(factors[0] - 0.5) < 1e-2 and abs(factors[1] - 2.0) < 1e-2
    assert rows.shape == (L,) and not rows[:64].any() and rows[64:].all()

    assert mod._bias_to_sets(torch.zeros(1, 1, 4, 4)) is None          # no bias
    bool_mask = torch.zeros(1, 1, 4, 4, dtype=torch.bool)
    bool_mask[:, :, 2:, 0] = True
    assert mod._bias_to_sets(bool_mask) is None                        # bool masks are foreign


def test_flash_bias_matches_sdpa_reference():
    mod = _need_flash()
    dev, dtype = "cuda", torch.bfloat16
    torch.manual_seed(0)
    B, H, L, D = 2, 4, 256, 64
    q = torch.randn(B, H, L, D, device=dev, dtype=dtype)
    k = torch.randn(B, H, L, D, device=dev, dtype=dtype)
    v = torch.randn(B, H, L, D, device=dev, dtype=dtype)

    # [text 0:32 | ref_a 32:96 | ref_b 96:128 | target 128:256]; boost 2.5 on a, 0.4 on b
    bias = torch.zeros(1, 1, L, L, device=dev, dtype=dtype)
    bias[:, :, 128:, 32:96] = torch.log(torch.tensor(2.5, dtype=dtype))
    bias[:, :, 128:, 96:128] = torch.log(torch.tensor(0.4, dtype=dtype))

    out = _run_override(mod, q, k, v, bias)
    assert out.shape == (B, L, H * D)
    _assert_close(out, _sdpa_reference(q, k, v, bias), "two-set boost (b=2.5, b=0.4)")


def test_flash_bias_sparse_cols_and_mid_sequence_rows():
    mod = _need_flash()
    dev, dtype = "cuda", torch.bfloat16
    torch.manual_seed(1)
    B, H, L, D = 1, 8, 512, 32
    q = torch.randn(B, H, L, D, device=dev, dtype=dtype)
    k = torch.randn(B, H, L, D, device=dev, dtype=dtype)
    v = torch.randn(B, H, L, D, device=dev, dtype=dtype)

    cols = torch.rand(L, device=dev) < 0.3                # boost_mask-style sparse region
    bias = torch.zeros(1, 1, L, L, device=dev, dtype=dtype)
    bias[:, :, 100:400, cols] = torch.log(torch.tensor(3.0, dtype=dtype))  # rows mid-sequence

    out = _run_override(mod, q, k, v, bias)
    _assert_close(out, _sdpa_reference(q, k, v, bias), "sparse cols, mid-sequence rows")

    # unboosted rows must be EXACTLY the maskless flash result (no correction bleed)
    plain = _run_override(mod, q, k, v, None)
    _assert_close(out[:, :100], plain[:, :100], "unboosted rows untouched", atol=1e-3, rtol=1e-3)


def test_end_to_end_tiny_dit_no_flash_warnings_and_matches_masked_path():
    mod = _need_flash()
    import comfy.model_management as mm

    if not mm.flash_attention_enabled():
        print("SKIP  comfy imported without --use-flash-attention; standalone run needed")
        return
    import comfy.ldm.modules.attention as A

    assert A.optimized_attention_masked is A.attention_flash, "flash backend not active"

    from comfy.ldm.krea2.model import SingleStreamDiT

    torch.manual_seed(7)
    dev, dtype = "cuda", torch.bfloat16
    m = SingleStreamDiT(features=256, tdim=64, txtdim=64, heads=8, kvheads=4, multiplier=2,
                        layers=2, patch=2, channels=16, txtlayers=3, txtheads=4, txtkvheads=4,
                        device=dev, dtype=dtype, operations=torch.nn).eval()
    # RMSNorm/DoubleSharedModulation allocate params with torch.empty (weights come from
    # a checkpoint in production) — init everything so the tiny net stays finite in bf16.
    with torch.no_grad():
        for p in m.parameters():
            torch.nn.init.normal_(p, std=0.02)
        # amplify the attention value path so the ref-boost effect is well above the
        # flash-vs-SDPA agreement tolerance (otherwise "bias applied" vs "bias dropped"
        # is indistinguishable at bf16 quantum scale on a random tiny net)
        for blk in m.blocks:
            torch.nn.init.normal_(blk.attn.wv.weight, std=0.5)
            torch.nn.init.normal_(blk.attn.wo.weight, std=0.5)
    assert all(torch.isfinite(p).all() for p in m.parameters())
    x = torch.randn(1, 16, 32, 32, device=dev, dtype=dtype)
    src = torch.randn(1, 16, 32, 32, device=dev, dtype=dtype)
    ctx = torch.randn(1, 12, 3 * 64, device=dev, dtype=dtype)
    t = torch.tensor([1000.0], device=dev)

    with torch.no_grad():
        # 1) fixed path: flash-compatible override, no mask ever reaches a kernel
        coll = _WarningCollector()
        root = logging.getLogger()
        root.addHandler(coll)
        try:
            out_flash = mod.krea2_edit_forward(m, x, t, ctx, src, {}, ref_boost=2.5)
        finally:
            root.removeHandler(coll)
        assert not coll.records, f"flash warnings leaked: {coll.records[:1]}"

        # 2) old path: force the SDPA-with-mask fallback, expect the warning, compare
        saved, mod._flash_attn_func = mod._flash_attn_func, None
        try:
            coll2 = _WarningCollector()
            root.addHandler(coll2)
            try:
                out_sdpa = mod.krea2_edit_forward(m, x, t, ctx, src, {}, ref_boost=2.5)
            finally:
                root.removeHandler(coll2)
            assert coll2.records, "fallback run should have produced flash warnings (sanity)"
        finally:
            mod._flash_attn_func = saved

        # 3) boost must actually flow through: effect >> the flash-vs-SDPA gap above
        out_noboost = mod.krea2_edit_forward(m, x, t, ctx, src, {}, ref_boost=1.0)

    assert out_flash.shape == (1, 16, 32, 32)
    _assert_close(out_flash, out_sdpa, "flash-bias vs masked-SDPA end-to-end", atol=5e-2, rtol=5e-2)
    effect = (out_flash - out_noboost).abs().max().item()
    assert effect > 0.04, f"ref_boost had no measurable effect (max diff {effect:.2e})"


if __name__ == "__main__":
    failures = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except Exception as e:  # noqa: BLE001
                failures += 1
                print(f"FAIL  {name}: {type(e).__name__}: {e}")
    sys.exit(1 if failures else 0)
