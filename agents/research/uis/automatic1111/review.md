# AUTOMATIC1111/stable-diffusion-webui

## Source

- UI clone: `H:/uis/AUTOMATIC1111/stable-diffusion-webui`

## Summary

AUTOMATIC1111 is the baseline WebUI for "extras" workflows. Its strongest
auxiliary coverage is image upscaling, face restoration, interrogation/tagging,
high-res fixing, tiled SD upscale, and API surfaces for those features. It does
not ship the full ControlNet preprocessor suite in core, but its extension model
made those preprocessors a normal UI expectation.

## Model Support

- Main image diffusion: Stable Diffusion 1.x, Stable Diffusion 2.x, SDXL-class
  checkpoints through WebUI loading, plus Segmind SSD-1B.
- File formats: `.ckpt` and `.safetensors` checkpoints, including on-the-fly
  checkpoint reload and checkpoint merge workflows.
- Model components: VAE selection, textual inversion embeddings, hypernetworks,
  LoRA and extension-provided adapters.
- Generation modes: txt2img, img2img, inpaint, extension canvas workflows,
  hires fix, and SD upscale.
- Video/audio: no meaningful first-party video or audio model runtime in core;
  those are extension territory.

## Feature Surface

- Extras/postprocess tab with one or two upscalers, crop/resize modes, face
  restoration, and optional ordering between upscaling and restoration.
- Hires fix and SD upscale script for generation-time upscaling.
- Built-in upscaler model discovery and API enumeration.
- BLIP/CLIP interrogation and DeepDanbooru tagging.
- GFPGAN and CodeFormer face restoration.
- Extension points for scripts and model loaders.

## Auxiliary Model Families

- ESRGAN/RRDB.
- RealESRGAN/SRVGG, now loaded through Spandrel in recent code paths.
- DAT and HAT image restoration.
- SwinIR and ScuNET extension-builtin upscalers.
- LDSR diffusion upscaler.
- GFPGAN and CodeFormer face restoration.
- facexlib/RetinaFace face detection and alignment.
- BLIP, CLIP interrogator, and DeepDanbooru tagger.

## Packages and Loaders

- `spandrel` for generic image-to-image restoration loading.
- `facexlib` for face detection/alignment helpers.
- Local PyTorch architecture code remains for several legacy upscalers.

## Code Anchors

- `H:/uis/AUTOMATIC1111/stable-diffusion-webui/README.md:87`
  documents Stable Diffusion 2.0 support.
- `H:/uis/AUTOMATIC1111/stable-diffusion-webui/README.md:90`
  documents safetensors loading.
- `H:/uis/AUTOMATIC1111/stable-diffusion-webui/README.md:94`
  documents Segmind Stable Diffusion support.
- `H:/uis/AUTOMATIC1111/stable-diffusion-webui/README.md:175`
  lists upstream Stable Diffusion references.
- `H:/uis/AUTOMATIC1111/stable-diffusion-webui/modules/sd_models.py`
  handles checkpoint discovery/loading.
- `H:/uis/AUTOMATIC1111/stable-diffusion-webui/modules/sd_vae.py`
  handles VAE selection/loading.
- `H:/uis/AUTOMATIC1111/stable-diffusion-webui/extensions-builtin/Lora/`
  contains built-in LoRA network support.
- `H:/uis/AUTOMATIC1111/stable-diffusion-webui/modules/textual_inversion/`
  contains textual inversion embedding support.
- `H:/uis/AUTOMATIC1111/stable-diffusion-webui/modules/extras.py`
  implements checkpoint merge and extras workflows.
- `H:/uis/AUTOMATIC1111/stable-diffusion-webui/modules/modelloader.py:164`
  loads Spandrel models.
- `H:/uis/AUTOMATIC1111/stable-diffusion-webui/modules/esrgan_model.py:7`
  registers `UpscalerESRGAN`.
- `H:/uis/AUTOMATIC1111/stable-diffusion-webui/modules/realesrgan_model.py:39`
  loads RealESRGAN via Spandrel.
- `H:/uis/AUTOMATIC1111/stable-diffusion-webui/modules/dat_model.py:9`
  registers DAT upscalers.
- `H:/uis/AUTOMATIC1111/stable-diffusion-webui/modules/hat_model.py:10`
  registers HAT upscalers.
- `H:/uis/AUTOMATIC1111/stable-diffusion-webui/extensions-builtin/SwinIR/scripts/swinir_model.py:70`
  loads SwinIR.
- `H:/uis/AUTOMATIC1111/stable-diffusion-webui/extensions-builtin/ScuNET/scripts/scunet_model.py:64`
  loads ScuNET.
- `H:/uis/AUTOMATIC1111/stable-diffusion-webui/modules/gfpgan_model.py:23`
  registers GFPGAN restoration.
- `H:/uis/AUTOMATIC1111/stable-diffusion-webui/modules/codeformer_model.py:25`
  registers CodeFormer restoration.
- `H:/uis/AUTOMATIC1111/stable-diffusion-webui/modules/face_restoration_utils.py:43`
  wraps facexlib face helper behavior.
- `H:/uis/AUTOMATIC1111/stable-diffusion-webui/modules/interrogate.py`
  covers BLIP/CLIP interrogation.
- `H:/uis/AUTOMATIC1111/stable-diffusion-webui/modules/deepbooru.py`
  covers DeepDanbooru tagging.
- `H:/uis/AUTOMATIC1111/stable-diffusion-webui/scripts/postprocessing_upscale.py`
  implements extras upscaling behavior.
- `H:/uis/AUTOMATIC1111/stable-diffusion-webui/scripts/sd_upscale.py`
  implements tiled SD upscale.

## DinoML Gaps

- Main model compatibility layer for WebUI-style checkpoints, VAEs, LoRAs,
  textual inversion, and merge/reload behavior.
- Auxiliary image-to-image model contract distinct from diffusion pipelines:
  tiled image input, scale metadata, model scale inference, tile overlap, and
  postprocess blending.
- Face restoration contract: detector/alignment/crop/paste-back workflow plus
  restoration model execution.
- Caption/tagger utilities as UI-facing inference tools rather than diffusion
  model components.

## Further Exploration Additions

- Swin2SR is a supported/upstream-noted upscaler in addition to SwinIR.
  Anchors: `H:/uis/AUTOMATIC1111/stable-diffusion-webui/README.md:31`,
  `H:/uis/AUTOMATIC1111/stable-diffusion-webui/README.md:182`.
- LDSR should be tracked in the cross-UI reference as a diffusion upscaler.
  Anchor:
  `H:/uis/AUTOMATIC1111/stable-diffusion-webui/extensions-builtin/LDSR/scripts/ldsr_model.py:11`.
- Hypertile exposes U-Net/VAE attention tiling as a runtime/product feature.
  Anchors:
  `H:/uis/AUTOMATIC1111/stable-diffusion-webui/extensions-builtin/hypertile/scripts/hypertile_script.py:85`,
  `H:/uis/AUTOMATIC1111/stable-diffusion-webui/extensions-builtin/hypertile/hypertile.py:318`.
- Built-in LoRA support includes LyCORIS/OFT/BOFT-style variants.
  Anchors:
  `H:/uis/AUTOMATIC1111/stable-diffusion-webui/extensions-builtin/Lora/network_oft.py:13`,
  `H:/uis/AUTOMATIC1111/stable-diffusion-webui/extensions-builtin/Lora/preload.py:8`.
