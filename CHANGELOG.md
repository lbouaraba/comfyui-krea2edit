# Changelog — Krea 2 Identity Edit

Weights: https://huggingface.co/conradlocke/krea2-identity-edit
v1.2 updates the **nodes** (see below); they stay backward-compatible with v1/v1.1
weights via `fit_mode: crop`.

## v1.2.5 — 2026-07-29

### Added
- **`target_latent` input on the patch node** (#15 — thanks @ethanfel, first outside
  code contribution!): wire the same latent that feeds `KSampler.latent_image` and the
  pixel path VAE-encodes the source at node-execution time instead of on the first
  sampling step. Without it, that mid-sampling encode can make ComfyUI evict part of
  the resident diffusion model on VRAM-tight setups — every remaining step then
  streams weights from CPU. Optional; unwired behavior is unchanged. See the README
  section "Pixel path and VRAM".

### Fixed
- Pre-encode console messages now report resolutions in pixels, not latent units.

## v1.2.4 — 2026-07-29

Node-only update (no new weights) — recommended for everyone on v1.2 weights.

### Fixed
- **Vertical-outpaint "doubling" band largely fixed.** Two geometry defects made the
  reference band misregister against the target grid: content was squashed by up to
  15px by /16 flooring, and odd-gap refs sat 8px off their true center from integer
  offset flooring. Sources are now center-cropped so the fitted axis lands on the /16
  grid exactly, and RoPE offsets are placed at the true (fractional) center. Biggest
  effect on outpaint, style-guard and removal edits. Note: the v1.2 weights learned
  some of this artifact during training, so rare cases can persist — `fit_mode: crop`
  remains a workaround; a fully clean from-scratch training is in progress for v2.
- From v1.2.3 (pushed, never release-tagged): a newer workflow meeting an older node
  install now warns "update the node pack" instead of crashing with a TypeError (#5);
  README `grounding_px` trained range corrected to the v1.2 reality (384–768).

### Advisory
- **Prefer `euler` (or other ODE samplers) over `er_sde` for outpainting.** The SDE
  noise injection disrupts the reference-copy channel and ruins outpaint coherence.

## v1.2 — 2026-07-17 (recommended)

`krea2_identity_edit_v1_2.safetensors` — pair it with the v1.2 nodes in this repo.

### New

**In the model:**
- **Better face likeness** on restaged subjects.
- **Character reference sheets** — both *using* a sheet as reference and *creating*
  one from a character.
- **Head / face swap** (and eye / person replacement).
- **Outpainting.**
- **Inpainting.**
- **Try-on** — put a garment onto a person.
- **Better person removal.**
- **Higher fidelity across the board, from a 1024 pass** — v1.1 had no high-resolution
  adaptation; v1.2 does.

**In the nodes:**
- **`ref_boost` — a reference-fidelity dial.** Turn how hard an edit locks onto the
  reference's appearance up or down (1.0 = neutral, >1 = pull harder toward the
  reference). Best value is model-specific.
- **No more blurry/stretched results (new `fit` geometry).** Sources are resampled to
  the target grid at a training-matched offset — and the old "match the source aspect
  ratio" requirement is gone. Needs `vae` + `source_image` connected on the patch node.

### Thanks
Head / face / eye / person swap is trained on **stablellama**'s MIT-licensed
[`change_eye_face_head_person`](https://huggingface.co/datasets/stablellama/change_eye_face_head_person)
dataset — big thanks for making it available.

### Node changes (technical)
- `fit_mode` defaults to `fit` (training-matched); `crop` remains for v1/v1.1-legacy weights.
- Added `ref_boost` / `ref_boost_a` reference-fidelity dials.

## v1.1 — 2026-07-09

`krea2_identity_edit_v1_1.safetensors`

- **Substantially improved face likeness and image fidelity**
- **Much stronger edit locality** — camera, pose, and untouched elements stay
  fixed far more reliably
- Better two-person identity separation
- More reliable object remove / replace
- Better compound outfit-change compliance
- Corrected reference geometry handling (training refs are now center-cropped,
  matching the shipped workflows)

**Known limitations of v1.1:**
- *Person*-replacement ("replace the woman with an orangutan") is currently
  weaker than v1 — keep v1 for that use case until v1.2
- No high-resolution adaptation pass yet: at high resolutions (especially
  two-person edits) identities can bleed together — prefer ~1–1.5MP and upscale
- `grounding_px`: v1.1's trained range is 384–768 (1024 often still works).
  If you get duplicated/split compositions, lower `grounding_px`.

## v1 — 2026-07-07

`krea2_identity_edit_v1.safetensors` — initial release. Remains available for
workflow reproducibility and for person-replacement edits.
