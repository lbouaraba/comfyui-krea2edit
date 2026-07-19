"""Regression test: the DIFFUSION_MODEL wrapper must accept the exact positional
call shape that comfy/ldm/krea2/model.py uses.

Upstream commit c9602625 inserted a new ``ref_latents`` positional argument between
``attention_mask`` and ``transformer_options`` in SingleStreamDiT.forward:

    .execute(x, timesteps, context, attention_mask, ref_latents, transformer_options, **kwargs)

WrapperExecutor forwards these positionally as ``wrapper(self, x, timesteps, context,
attention_mask, ref_latents, transformer_options)`` -> 7 positional args. A wrapper that
predates the change only has 6 slots and dies with:

    TypeError: wrapper() takes from 4 to 6 positional arguments but 7 were given

This test reproduces that call path with a stubbed forward so it stays fast and
model-free, and guards against the signature drifting out of sync again.

Run standalone:  python tests/test_wrapper_signature.py
Or via pytest:   pytest tests/test_wrapper_signature.py
"""
import importlib.util
import inspect
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PACK = os.path.dirname(_HERE)                       # .../custom_nodes/comfyui-krea2edit
_COMFY_ROOT = os.path.dirname(os.path.dirname(_PACK))  # .../ComfyUI

# ComfyUI root must be importable so the pack's ``import comfy.*`` lines resolve.
if _COMFY_ROOT not in sys.path:
    sys.path.insert(0, _COMFY_ROOT)


def _load_pack():
    """Import the pack's __init__.py under a synthetic name (folder has a hyphen)."""
    spec = importlib.util.spec_from_file_location(
        "comfyui_krea2edit_under_test", os.path.join(_PACK, "__init__.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _FakeInner:
    def process_latent_in(self, x):
        return x


class _FakeModelPatcher:
    """Minimal stand-in for comfy.model_patcher.ModelPatcher, only what patch() touches."""
    def __init__(self):
        self.model = _FakeInner()
        self.model_options = {}

    def clone(self):
        c = _FakeModelPatcher()
        c.model = self.model  # share inner model, like the real clone
        return c


def _get_wrapper(mod):
    import comfy.patcher_extension as pe

    node = mod.Krea2EditModelPatch()
    (m,) = node.patch(_FakeModelPatcher(), {"samples": object()})
    to = m.model_options["transformer_options"]
    return to["wrappers"][pe.WrappersMP.DIFFUSION_MODEL]["krea2_edit"][0]


def test_wrapper_accepts_upstream_positional_call():
    import comfy.patcher_extension as pe

    mod = _load_pack()
    wrapper = _get_wrapper(mod)

    # Stub the heavy forward: we only care that the wrapper's signature binds.
    sentinel = object()
    mod.krea2_edit_forward = lambda *a, **k: sentinel

    executor = pe.WrapperExecutor(
        original=lambda *a, **k: None,
        class_obj=object(),   # stands in for the SingleStreamDiT instance
        wrappers=[wrapper],
        idx=0,
    )

    # Exactly the shape model.py:281 uses: 6 positionals after (implicit) executor.
    result = executor.execute(
        None,   # x
        None,   # timesteps
        None,   # context
        None,   # attention_mask
        None,   # ref_latents  <-- the argument added by c9602625
        {},     # transformer_options
    )
    assert result is sentinel


def test_wrapper_signature_matches_forward():
    """Guard the ordering: wrapper params after `executor` must equal SingleStreamDiT.forward
    params after `self` (so transformer_options never binds to ref_latents by accident)."""
    from comfy.ldm.krea2.model import SingleStreamDiT

    mod = _load_pack()
    wrapper = _get_wrapper(mod)

    fwd = [p for p in inspect.signature(SingleStreamDiT.forward).parameters if p != "self"]
    wrp = [p for p in inspect.signature(wrapper).parameters if p != "executor"]
    # Compare the shared positional prefix (both end in **kwargs).
    assert wrp[:len(fwd)] == fwd, f"wrapper params {wrp} must mirror forward params {fwd}"


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
