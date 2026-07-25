"""Regression tests for the pixel path's VAE-encode timing.

`vae.encode` reaches ComfyUI's `load_models_gpu(..., memory_required=...)`, which calls
`free_memory()` with an empty `keep_loaded` — so it will partially unload whatever is
already resident on the device. When the first encode landed *inside* the sampling
wrapper, that victim was the diffusion model the sampler was in the middle of using, and
nothing re-expands it (`sampler_helpers` loads once, before the loop). The rest of the run
then streamed weights from CPU on every step.

`target_latent` moves the encode to node-execution time, before the sampler loads
anything. These tests pin the observable behavior: WHEN the encode happens, and that it
still happens exactly once either way.

Self-contained: installs a minimal ComfyUI stub when the real one is not importable, so
it runs both inside a ComfyUI checkout and standalone.

Run standalone:  python tests/test_pre_encode.py
Or via pytest:   pytest tests/test_pre_encode.py
"""
import importlib.util
import os
import sys
import types

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_PACK = os.path.dirname(_HERE)
_COMFY_ROOT = os.path.dirname(os.path.dirname(_PACK))   # .../ComfyUI, when installed there

if _COMFY_ROOT not in sys.path:
    sys.path.insert(0, _COMFY_ROOT)


def _install_comfy_stubs():
    """Only the surface the pack imports at module level. Skipped when real comfy is
    importable, so this never shadows a genuine ComfyUI checkout."""
    try:
        import comfy.patcher_extension  # noqa: F401
        return
    except Exception:                                            # noqa: BLE001
        pass

    class WrappersMP:
        DIFFUSION_MODEL = "diffusion_model"

    class WrapperExecutor:
        def __init__(self, original, class_obj, wrappers, idx):
            self.original, self.class_obj = original, class_obj
            self.wrappers, self.idx = list(wrappers), idx
            self.is_last = idx == len(wrappers)

        def execute(self, *args, **kwargs):
            if self.is_last:
                return self.original(*args, **kwargs)
            return self.wrappers[self.idx](self, *args, **kwargs)

    def add_wrapper_with_key(wrapper_type, key, wrapper, options):
        options.setdefault("wrappers", {}).setdefault(wrapper_type, {}).setdefault(key, []).append(wrapper)

    mods = {}
    for name in ("comfy", "comfy.patcher_extension", "comfy.utils", "comfy.ldm",
                 "comfy.ldm.common_dit", "comfy.ldm.flux", "comfy.ldm.flux.layers",
                 "comfy.ldm.flux.math", "comfy.ldm.modules", "comfy.ldm.modules.attention"):
        mods[name] = types.ModuleType(name)

    mods["comfy.patcher_extension"].WrappersMP = WrappersMP
    mods["comfy.patcher_extension"].WrapperExecutor = WrapperExecutor
    mods["comfy.patcher_extension"].add_wrapper_with_key = add_wrapper_with_key
    mods["comfy.utils"].common_upscale = lambda s, w, h, *a: s
    mods["comfy.ldm.common_dit"].pad_to_patch_size = lambda v, _p, **_k: v
    mods["comfy.ldm.flux.layers"].timestep_embedding = lambda t, d: torch.zeros(t.shape[0], d)
    mods["comfy.ldm.flux.math"].apply_rope = lambda q, k, _f: (q, k)
    mods["comfy.ldm.modules.attention"].optimized_attention_masked = lambda *a, **k: None
    mods["comfy.ldm.modules.attention"].attention_pytorch = lambda *a, **k: None

    for attr, parent, child in (("patcher_extension", "comfy", "comfy.patcher_extension"),
                                ("utils", "comfy", "comfy.utils"),
                                ("ldm", "comfy", "comfy.ldm"),
                                ("common_dit", "comfy.ldm", "comfy.ldm.common_dit"),
                                ("flux", "comfy.ldm", "comfy.ldm.flux"),
                                ("modules", "comfy.ldm", "comfy.ldm.modules"),
                                ("layers", "comfy.ldm.flux", "comfy.ldm.flux.layers"),
                                ("math", "comfy.ldm.flux", "comfy.ldm.flux.math"),
                                ("attention", "comfy.ldm.modules", "comfy.ldm.modules.attention")):
        setattr(mods[parent], attr, mods[child])
    sys.modules.update(mods)


def _load_pack():
    """Import the pack's __init__.py under a synthetic name (the folder has a hyphen)."""
    _install_comfy_stubs()
    spec = importlib.util.spec_from_file_location(
        "comfyui_krea2edit_pre_encode_test", os.path.join(_PACK, "__init__.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _FakeVAE:
    """Records every encode so tests can assert on call timing and count."""

    def __init__(self):
        self.calls = []

    def encode(self, pixels):        # pixels: (B, H, W, C)
        self.calls.append(tuple(pixels.shape))
        b, h, w, _ = pixels.shape
        return torch.zeros(b, 4, h // 8, w // 8)


class _FakeInner:
    def process_latent_in(self, x):
        return x


class _FakeModelPatcher:
    """Minimal stand-in for ModelPatcher — only what patch() touches."""

    def __init__(self):
        self.model = _FakeInner()
        self.model_options = {}

    def clone(self):
        c = _FakeModelPatcher()
        c.model = self.model      # share the inner model, like the real clone
        return c


def _latent(h=64, w=64):
    return {"samples": torch.zeros(1, 4, h, w)}


def _image(px=512):
    return torch.zeros(1, px, px, 3)


def _patch(mod, **kwargs):
    import comfy.patcher_extension as pe

    node = mod.Krea2EditModelPatch()
    (m,) = node.patch(_FakeModelPatcher(), _latent(), **kwargs)
    to = m.model_options["transformer_options"]
    return to["wrappers"][pe.WrappersMP.DIFFUSION_MODEL]["krea2_edit"][0]


def _run_steps(wrapper, n=3, h=64, w=64):
    """Drive the wrapper the way a sampler would: same latent shape, n times."""
    import comfy.patcher_extension as pe

    for _ in range(n):
        executor = pe.WrapperExecutor(
            original=lambda *a, **k: None, class_obj=object(), wrappers=[wrapper], idx=0,
        )
        executor.execute(torch.zeros(1, 4, h, w), None, None, None, {})


def _stub_forward(mod, seen):
    mod.krea2_edit_forward = lambda dm, x, t, ctx, src, to, **k: seen.append(src)


def test_target_latent_encodes_before_sampling():
    """The whole point: with target_latent wired, the encode is done by the time patch()
    returns — i.e. before the sampler loads the diffusion model."""
    mod = _load_pack()
    seen = []
    _stub_forward(mod, seen)
    vae = _FakeVAE()
    wrapper = _patch(mod, vae=vae, source_image=_image(), target_latent=_latent())

    assert len(vae.calls) == 1, "source should be encoded during patch(), not later"

    _run_steps(wrapper)
    assert len(vae.calls) == 1, "sampling must reuse the pre-encode, never re-encode"
    assert seen[0].shape == (1, 4, 64, 64)


def test_without_target_latent_encode_lands_in_the_wrapper():
    """Legacy path stays functional (and stays the slow one) — documents the contrast."""
    mod = _load_pack()
    seen = []
    _stub_forward(mod, seen)
    vae = _FakeVAE()
    wrapper = _patch(mod, vae=vae, source_image=_image())

    assert vae.calls == [], "nothing to encode yet without a known target resolution"

    _run_steps(wrapper)
    assert len(vae.calls) == 1, "encode happens on the first step, then caches"


def test_both_references_are_pre_encoded():
    mod = _load_pack()
    seen = []
    _stub_forward(mod, seen)
    vae = _FakeVAE()
    wrapper = _patch(mod, vae=vae, source_image=_image(), source_image_b=_image(),
                     target_latent=_latent())

    assert len(vae.calls) == 2, "scene and subject refs both pre-encoded"

    _run_steps(wrapper)
    assert len(vae.calls) == 2
    assert isinstance(seen[0], list) and len(seen[0]) == 2


def test_mismatched_target_latent_falls_back_correctly():
    """A target_latent that doesn't match what's actually sampled must not corrupt the
    source — the cache simply misses and the wrapper re-encodes at the real size."""
    mod = _load_pack()
    seen = []
    _stub_forward(mod, seen)
    vae = _FakeVAE()
    wrapper = _patch(mod, vae=vae, source_image=_image(), target_latent=_latent(32, 32))

    assert len(vae.calls) == 1                    # primed at the wrong size

    _run_steps(wrapper, n=2, h=64, w=64)
    assert len(vae.calls) == 2, "one re-encode at the real target size, then cached"
    assert seen[0].shape == (1, 4, 64, 64), "source must match the sampled grid"


def test_target_latent_without_pixel_path_is_inert():
    """No vae/source_image means the latent path — target_latent must change nothing."""
    mod = _load_pack()
    seen = []
    _stub_forward(mod, seen)
    wrapper = _patch(mod, target_latent=_latent())
    _run_steps(wrapper)
    assert seen[0].shape == (1, 4, 64, 64)


if __name__ == "__main__":
    failures = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except Exception as e:                               # noqa: BLE001
                failures += 1
                print(f"FAIL  {name}: {type(e).__name__}: {e}")
    sys.exit(1 if failures else 0)
