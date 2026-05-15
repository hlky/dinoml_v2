# MiDaS, DPT, and GLPN Depth

## Coverage

- Diffusers: not covered.
- Transformers: DPT is covered under `src/transformers/models/dpt`; GLPN is covered under `src/transformers/models/glpn`.
- Third-party/UI: SD.Next routes DPT/GLPN through Transformers-backed depth preprocessors; Forge lists MiDaS depth/normal preprocessing.

## Runtime Contract

These produce depth or normal-like condition maps from an input image. The model graph is Transformers-owned for DPT/GLPN, while MiDaS-style UI behavior includes model selection, resize, inversion/normalization, and optional normal map generation.

`controlnet_aux.MidasDetector` defaults to `model_type="dpt_hybrid"` and `dpt_hybrid-midas-501f0c75.pt`. The local API also supports `dpt_large`, `midas_v21`, and `midas_v21_small`. Runtime converts RGB to -1..1, runs `MiDaSInference`, min/max normalizes depth to an 8-bit map, and can optionally return normals by Sobel gradients over raw depth with background suppression via `bg_th`.

## Operators

- Vision transformer or hybrid conv/transformer encoder depending on checkpoint.
- Image resize/normalize and depth output resizing.
- Optional normal-map reconstruction from depth gradients for MiDaS normal mode.

## DinoML Notes

Treat model inference and condition rendering separately. DPT/GLPN should reuse Transformers coverage; MiDaS-specific normal maps need their own postprocess contract.

## Sources

- `H:/controlnet_aux/src/controlnet_aux/midas/__init__.py`
- `H:/controlnet_aux/src/controlnet_aux/midas/api.py`
- `transformers/src/transformers/models/dpt`
- `transformers/src/transformers/models/glpn`
- `vladmandic/sdnext/modules/control/proc/dpt.py:25`
- `vladmandic/sdnext/modules/control/proc/glpn.py:17`
- `lllyasviel/stable-diffusion-webui-forge/extensions-builtin/forge_legacy_preprocessors/legacy_preprocessors/preprocessor.py:232`
