# Haoming02/sd-webui-forge-classic

## Source

- UI clone: `Haoming02/sd-webui-forge-classic`

## Summary

Forge Classic is valuable for upscaler/restoration handling and WebUI-compatible
extras APIs. It keeps the classic postprocessing surface while adding newer
upscaler utilities and Forge-style backend behavior.

## Model Support

- Main image/video models listed in README: SD1, SDXL, SD3, advanced SDXL,
  Flux.2 Klein, Z-Image, Wan 2.2, Qwen Image, Qwen Image Edit, and Flux
  Kontext.
- Video-specific support: Wan 2.2, including FirstLastFrameToVideo.
- Runtime/formats: Nunchaku SVDQ models and `--nunchaku` install path.
- Components: Flux.2 small decoder, Qwen2D VAE, SD1/SDXL/Wan VAE handling,
  LoRA, ControlNet rewrite, LLLite/Union ControlNet references.
- Scope note: despite the package name, this fork has expanded well beyond the
  original SD1/SDXL classic surface.

## Feature Surface

- Extras upscaling with one or two upscalers, target resize, crop, and cache.
- SD upscale script.
- Face restoration via GFPGAN and CodeFormer.
- API endpoints for upscalers, latent upscalers, and face restorers.
- Upscaler utilities for ATD/DAT/DRCT-style models and memory handling.

## Auxiliary Model Families

- ESRGAN/RealESRGAN-style upscalers.
- ATD, DAT, DRCT and related restoration networks.
- GFPGAN and CodeFormer face restoration.
- facexlib/RetinaFace style face helper behavior.
- Latent upscale modes and SD upscale.

## Packages and Loaders

- WebUI-compatible upscaler abstractions.
- Local upscaler utilities and architecture-specific memory handling.
- GFPGAN/CodeFormer dependencies inherited from WebUI family.

## Code Anchors

- `Haoming02/sd-webui-forge-classic/README.md:31`
  lists Flux.2 Klein support.
- `Haoming02/sd-webui-forge-classic/README.md:39`
  lists Z-Image support.
- `Haoming02/sd-webui-forge-classic/README.md:41`
  lists Wan 2.2 support.
- `Haoming02/sd-webui-forge-classic/README.md:50`
  lists advanced SDXL support.
- `Haoming02/sd-webui-forge-classic/README.md:57`
  lists Qwen Image/Edit support.
- `Haoming02/sd-webui-forge-classic/README.md:62`
  lists Flux Kontext support.
- `Haoming02/sd-webui-forge-classic/README.md:69`
  lists FirstLastFrameToVideo support for Wan 2.2.
- `Haoming02/sd-webui-forge-classic/README.md:70`
  lists Nunchaku SVDQ support.
- `Haoming02/sd-webui-forge-classic/README.md:79`
  lists Flux.2 small decoder and Qwen2D VAE support.
- `Haoming02/sd-webui-forge-classic/README.md:142`
  lists SD3 support.
- `Haoming02/sd-webui-forge-classic/README.md:279`
  documents `--nunchaku`.
- `Haoming02/sd-webui-forge-classic/modules/upscaler_utils.py:72`
  contains ATD/DAT/DRCT tiling/memory handling.
- `Haoming02/sd-webui-forge-classic/modules/gfpgan_model.py:25`
  defines GFPGAN v1.4 model path/download behavior.
- `Haoming02/sd-webui-forge-classic/modules/codeformer_model.py`
  defines CodeFormer face restoration.
- `Haoming02/sd-webui-forge-classic/modules/api/api.py:224`
  exposes `/sdapi/v1/upscalers`.
- `Haoming02/sd-webui-forge-classic/modules/api/api.py:225`
  exposes `/sdapi/v1/latent-upscale-modes`.
- `Haoming02/sd-webui-forge-classic/modules/api/api.py:228`
  exposes `/sdapi/v1/face-restorers`.
- `Haoming02/sd-webui-forge-classic/modules/api/models.py:134`
  exposes GFPGAN visibility in extras requests.
- `Haoming02/sd-webui-forge-classic/modules/api/models.py:135`
  exposes CodeFormer visibility in extras requests.
- `Haoming02/sd-webui-forge-classic/scripts/postprocessing_upscale.py:35`
  builds extras upscale UI.
- `Haoming02/sd-webui-forge-classic/scripts/postprocessing_upscale.py:122`
  caches upscaler results.
- `Haoming02/sd-webui-forge-classic/scripts/sd_upscale.py:26`
  builds SD upscale script controls.

## DinoML Gaps

- Recent model-family support from a WebUI-compatible surface: Flux.2, Z-Image,
  Qwen Image, Wan 2.2, SD3, Nunchaku, and special VAEs/decoders.
- API-level schema for postprocessing operations, including two-stage upscaler
  blending and face restoration knobs.
- Upscaler tile/cache/memory metadata that can survive provider selection.
- Legacy-compatible extras behavior likely expected by UI clients.

## Further Exploration Additions

- Lumina-Image-2.0, Neta-Lumina, and NetaYume-Lumina are supported model
  presets/families.
  Anchors: `Haoming02/sd-webui-forge-classic/README.md:74`,
  `Haoming02/sd-webui-forge-classic/modules_forge/presets.py:10`.
- Runtime quant formats include `fp4mixed`, `fp8mixed`, `mxfp8`, `nvfp4`, and
  `fp8_scaled`, plus int8 Triton matmul.
  Anchors: `Haoming02/sd-webui-forge-classic/README.md:78`,
  `Haoming02/sd-webui-forge-classic/README.md:99`.
- Runtime attention/math support includes SageAttention, FlashAttention, fp16
  accumulation, and `torch._scaled_mm`.
  Anchors: `Haoming02/sd-webui-forge-classic/README.md:97`,
  `Haoming02/sd-webui-forge-classic/README.md:360`.
- LLLite and Union ControlNet should both be represented in the reference.
  Anchors: `Haoming02/sd-webui-forge-classic/README.md:134`,
  `Haoming02/sd-webui-forge-classic/README.md:136`,
  `Haoming02/sd-webui-forge-classic/modules_forge/supported_controlnet.py:126`.
- Hypernetworks and CLIP Interrogator are present and should be represented for
  Classic.
  Anchors: `Haoming02/sd-webui-forge-classic/README.md:144`,
  `Haoming02/sd-webui-forge-classic/README.md:145`.
