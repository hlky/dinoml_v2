# vladmandic/sdnext

## Source

- UI clone: `vladmandic/sdnext`

## Summary

SD.Next is a large WebUI-family implementation with a more consolidated
postprocess and control architecture than AUTOMATIC1111. It is especially useful
for DinoML because it puts upscalers, VAE upscalers, Control processors,
captioning, detailers, and runtime compilation/memory policy into clear module
families.

## Model Support

- Main image/video scope: the README describes SD.Next as an image and video
  generation UI built on Stable Diffusion and supporting many advanced models.
- Diffusers-first pipelines for SDXL/APG, SD3, Flux, Wan, Hunyuan Video, and
  HiDream-style pipelines through local pipeline wrappers.
- Model components: checkpoints, VAEs, LoRAs, text encoders, ControlNet-family
  adapters, IP/vision conditioning, caption/tagger models, and upscalers.
- Main video support: Wan and Hunyuan Video appear through CFG-Zero pipeline
  wrappers; video-specific VAE tiling/slicing behavior is included.
- Runtime support: compile hooks, upscaler compilation, model switching, and
  backend/provider options.

## Feature Surface

- Postprocess upscalers, including Spandrel and legacy architecture paths.
- VAE/diffusion latent upscalers and AuraSR/SeedVR-style postprocess models.
- Control processor/unit system for ControlNet, T2I-Adapter, ControlNet-XS,
  LLLite, IP/vision-style conditioning, and preprocessors.
- YOLO/detailer API tests and endpoints.
- Caption APIs for BLIP, OpenCLIP, taggers, and VLM-style captioning.
- Compile hooks and upscaler compile paths.

## Auxiliary Model Families

- Spandrel image-to-image upscalers.
- ESRGAN/RRDB, RealESRGAN/SRVGG, SwinIR, ScuNET.
- AuraSR, SeedVR, diffusion latent upscalers.
- HED, TEED, DPT/GLPN, LeReS, OneFormer, Anyline and similar Control
  preprocessors.
- IP-Adapter/vision encoders and ControlNet-family adapters.
- YOLO detailer/detection.
- Captioning/tagging models through caption modules.

## Packages and Loaders

- `spandrel` for broad restoration architecture support.
- Local postprocess architecture implementations for ESRGAN, RealESRGAN,
  SwinIR, and ScuNET.
- Control modules provide their own processor abstractions.

## Code Anchors

- `vladmandic/sdnext/README.md:6` describes SD.Next image/video model
  scope.
- `vladmandic/sdnext/modules/cfgzero/flux_pipeline.py:147`
  implements Flux CFG-Zero pipeline wrapper.
- `vladmandic/sdnext/modules/cfgzero/sd3_pipeline.py:160`
  implements Stable Diffusion 3 CFG-Zero pipeline wrapper.
- `vladmandic/sdnext/modules/cfgzero/wan_t2v_pipeline.py:107`
  implements Wan text-to-video pipeline wrapper.
- `vladmandic/sdnext/modules/cfgzero/hunyuan_t2v_pipeline.py:157`
  implements Hunyuan Video pipeline wrapper.
- `vladmandic/sdnext/modules/cfgzero/hidream_pipeline.py`
  implements HiDream-family pipeline wrapper.
- `vladmandic/sdnext/modules/cmd_args.py:23`
  defines VAE directory support.
- `vladmandic/sdnext/modules/cmd_args.py:24`
  defines LoRA directory support.
- `vladmandic/sdnext/modules/upscaler_spandrel.py:44`
  handles Spandrel import/install/load path.
- `vladmandic/sdnext/modules/upscaler_vae.py`
  covers VAE/latent upscaler behavior.
- `vladmandic/sdnext/modules/postprocess/esrgan_model.py:120`
  registers ESRGAN.
- `vladmandic/sdnext/modules/postprocess/realesrgan_model.py:9`
  registers RealESRGAN.
- `vladmandic/sdnext/modules/postprocess/swinir_model.py:12`
  registers SwinIR.
- `vladmandic/sdnext/modules/postprocess/scunet_model.py:8`
  registers ScuNET.
- `vladmandic/sdnext/modules/postprocess/aurasr_model.py:14`
  registers AuraSR 4x.
- `vladmandic/sdnext/modules/postprocess/seedvr_model.py:24`
  registers SeedVR2 model options.
- `vladmandic/sdnext/modules/control/processor.py`
  centralizes Control processor registration.
- `vladmandic/sdnext/modules/control/proc/`
  contains individual processor implementations.
- `vladmandic/sdnext/modules/control/units/`
  contains control unit implementations.
- `vladmandic/sdnext/modules/caption/caption.py`
  centralizes captioning.
- `vladmandic/sdnext/test/test-detailer-api.py:3`
  names YOLO Detailer API tests.

## DinoML Gaps

- Diffusers-backed multi-family model support with WebUI-compatible switching,
  component discovery, LoRA/adapters, video pipelines, and provider controls.
- Unified postprocess model registry covering restoration, VAE upscalers, and
  video restoration helpers.
- Control preprocessing as an independent image-to-condition stage with model
  cache/lifecycle separate from diffusion.
- Detection/detailer workflows as structured mask/detection/crop/paste
  pipelines.

## Further Exploration Additions

- Stable Cascade/APG support should be tracked as image-model/runtime support.
  Anchors: `vladmandic/sdnext/scripts/apg.py:73`,
  `vladmandic/sdnext/modules/apg/pipeline_stable_cascade_prior_apg.py:73`.
- AnimateDiff/AnimateFace-style video generation is present as a script surface.
  Anchor: `vladmandic/sdnext/scripts/animatediff.py:17`.
- Nunchaku attention/offload runtime knobs are exposed in attention/UI paths.
  Anchors: `vladmandic/sdnext/modules/attention.py:201`,
  `vladmandic/sdnext/modules/ui_definitions.py:169`.
- Captioning is broader than a generic tagger row: README/UI paths mention
  150+ OpenCLIP, WaifuDiffusion, DeepDanbooru, and 25+ VLMs.
  Anchors: `vladmandic/sdnext/README.md:62`,
  `vladmandic/sdnext/modules/ui_caption.py:252`.
