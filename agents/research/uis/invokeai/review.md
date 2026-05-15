# invoke-ai/InvokeAI

## Source

- UI clone: `invoke-ai/InvokeAI`

## Summary

InvokeAI is the cleanest source for typed auxiliary model taxonomy. It models
Spandrel as a first-class model type, exposes preprocessors as invocations, and
has explicit model-manager configs for ControlNet, IP/vision encoders,
segmentation, and starter models.

## Model Support

- Main image models listed in README: SDXL, Flux.1 Dev/Schnell/Kontext/Krea,
  Flux Redux, Flux Fill, Flux.2 Klein 4B/9B, Z-Image Turbo/Base, Qwen Image,
  and Qwen Image Edit.
- Video: Wan is supported API-only.
- Formats: ckpt, diffusers, and some GGUF models.
- Typed model components: main models, VAEs, LoRAs, textual inversion,
  ControlNet, T2I-Adapter, Control LoRA, IP-Adapter/vision encoders, T5/text
  encoders, SigLIP, Qwen VL encoders, Spandrel, and unknown fallback configs.
- Scheduler/control support: explicit scheduler invocations and Z-Image
  ControlNet support.

## Feature Surface

- Typed model manager taxonomy and config factory.
- Spandrel image-to-image model category and tiled invocation.
- RealESRGAN/ESRGAN invocation.
- ControlNet preprocessor invocations: Canny, HED, PiDiNet, MLSD, lineart,
  normal maps, depth, PBR maps, DWPose/OpenPose-style pose.
- Segment Anything invocation supporting SAM/SAM2.
- GroundingDINO and bbox-to-mask style workflows.
- MediaPipe face detection.
- IP-Adapter, Flux Redux, CLIP/SigLIP/Qwen vision encoders.
- Tiled multi-diffusion with ControlNet-data cropping.
- Z-Image ControlNet mode support for spatial controls.

## Auxiliary Model Families

- Spandrel image-to-image/upscale models.
- ESRGAN/RealESRGAN.
- Depth Anything and other depth processors.
- DWPose/OpenPose via ONNX-style detector path.
- HED, MLSD, PiDiNet, lineart, NormalBae, PBR map processors.
- SAM/SAM2 and GroundingDINO.
- CLIP, SigLIP, Qwen vision encoders.
- MediaPipe face detector.

## Packages and Loaders

- `spandrel` for restoration/upscale model loading.
- `transformers` is used for some vision encoders, but the UI gap is the
  invocation/output contract and postprocess behavior.
- ONNX-style model use appears in DWPose/OpenPose paths.

## Code Anchors

- `invoke-ai/InvokeAI/README.md:64` starts supported model list.
- `invoke-ai/InvokeAI/README.md:68` lists Flux.1 support.
- `invoke-ai/InvokeAI/README.md:74` lists Flux.2 Klein support.
- `invoke-ai/InvokeAI/README.md:76` lists Z-Image support.
- `invoke-ai/InvokeAI/README.md:79` lists Qwen Image support.
- `invoke-ai/InvokeAI/README.md:83` lists Wan API-only support.
- `invoke-ai/InvokeAI/README.md:87` documents ckpt/diffusers/GGUF
  support.
- `invoke-ai/InvokeAI/invokeai/backend/model_manager/taxonomy.py`
  defines base model, model type, and format taxonomy.
- `invoke-ai/InvokeAI/invokeai/backend/model_manager/configs/main.py`
  configures main model identification.
- `invoke-ai/InvokeAI/invokeai/backend/model_manager/configs/lora.py`
  configures LoRA models.
- `invoke-ai/InvokeAI/invokeai/backend/model_manager/configs/textual_inversion.py:28`
  configures textual inversion.
- `invoke-ai/InvokeAI/invokeai/backend/model_manager/configs/vae.py:74`
  configures standalone VAE checkpoints.
- `invoke-ai/InvokeAI/invokeai/backend/model_manager/configs/vae.py:163`
  configures Flux.2 VAE checkpoints.
- `invoke-ai/InvokeAI/invokeai/backend/model_manager/configs/vae.py:201`
  configures Qwen Image VAE checkpoints.
- `invoke-ai/InvokeAI/invokeai/backend/model_manager/configs/t2i_adapter.py:27`
  configures T2I-Adapter models.
- `invoke-ai/InvokeAI/invokeai/backend/model_manager/taxonomy.py:65`
  defines `ModelType.SpandrelImageToImage`.
- `invoke-ai/InvokeAI/invokeai/backend/model_manager/configs/spandrel.py`
  configures Spandrel model handling.
- `invoke-ai/InvokeAI/invokeai/app/invocations/spandrel_image_to_image.py:166`
  implements tiled/autoscale Spandrel invocation.
- `invoke-ai/InvokeAI/invokeai/app/invocations/upscale.py:14`
  covers RealESRGAN/ESRGAN invocation data.
- `invoke-ai/InvokeAI/invokeai/app/invocations/dw_openpose.py`
  implements DWPose/OpenPose-style invocation.
- `invoke-ai/InvokeAI/invokeai/app/invocations/hed.py`
  implements HED edge detection invocation.
- `invoke-ai/InvokeAI/invokeai/app/invocations/mlsd.py:16`
  implements MLSD detection invocation.
- `invoke-ai/InvokeAI/invokeai/app/invocations/pidi.py:16`
  implements PiDiNet edge detection invocation.
- `invoke-ai/InvokeAI/invokeai/app/invocations/normal_bae.py`
  implements NormalBae invocation.
- `invoke-ai/InvokeAI/invokeai/app/invocations/segment_anything.py:58`
  implements SAM/SAM2 mask invocation.
- `invoke-ai/InvokeAI/invokeai/backend/model_manager/configs/controlnet.py:38`
  maps lineart/scribble preprocessor hints.
- `invoke-ai/InvokeAI/invokeai/app/invocations/z_image_control.py:25`
  defines Z-Image spatial control modes.
- `invoke-ai/InvokeAI/invokeai/app/invocations/tiled_multi_diffusion_denoise_latents.py:40`
  crops ControlNet data per tile.
- `invoke-ai/InvokeAI/invokeai/backend/model_manager/starter_models.py`
  lists starter/downloadable models.

## DinoML Gaps

- Typed model taxonomy and configs for all model categories, not just
  preprocessors or auxiliary models.
- Typed auxiliary model taxonomy and admission rules.
- Invocation schemas for preprocessors and segmentation outputs.
- Explicit mask/detection/conditioning artifacts, including bbox/point prompt
  inputs and per-tile ControlNet data transforms.

## Further Exploration Additions

- Base taxonomy includes newer families not fully reflected in the first pass:
  Flux2, CogView4, ZImage, QwenImage, Anima, and Cosmos Predict2.
  Anchor: `invoke-ai/InvokeAI/invokeai/backend/model_manager/taxonomy.py:49`.
- Model/component taxonomy includes ONNX, T5Encoder, Qwen3Encoder,
  QwenVLEncoder, SigLIP, TextLLM, BNB int8, BNB NF4, and GGUF quantized
  formats.
  Anchors: `invoke-ai/InvokeAI/invokeai/backend/model_manager/taxonomy.py:68`,
  `invoke-ai/InvokeAI/invokeai/backend/model_manager/taxonomy.py:86`,
  `invoke-ai/InvokeAI/invokeai/backend/model_manager/taxonomy.py:196`.
- Config factory and LoRA configs explicitly support Flux2, CogView4,
  QwenImage, ZImage, and Anima across diffusers, checkpoint, GGUF, VAE, and
  LoRA forms.
  Anchors: `invoke-ai/InvokeAI/invokeai/backend/model_manager/configs/factory.py:173`,
  `invoke-ai/InvokeAI/invokeai/backend/model_manager/configs/lora.py:681`.
- Anima has a distinct invocation suite with Qwen3 encoder, Wan/QwenImage VAE,
  LoRA application, 3D latent shape, and image latent conversion.
  Anchors:
  `invoke-ai/InvokeAI/invokeai/app/invocations/anima_model_loader.py:42`,
  `invoke-ai/InvokeAI/invokeai/app/invocations/anima_text_encoder.py:62`.
- CogView4 has loader, text encoder, denoise, and image-to-latents invocations.
  Anchors:
  `invoke-ai/InvokeAI/invokeai/app/invocations/cogview4_model_loader.py:36`,
  `invoke-ai/InvokeAI/invokeai/app/invocations/cogview4_denoise.py:40`.
- Text LLM/prompt expansion is a product surface with model type, loader,
  invocation, utility route, UI strings, and starter models.
  Anchors: `invoke-ai/InvokeAI/invokeai/backend/model_manager/taxonomy.py:86`,
  `invoke-ai/InvokeAI/invokeai/app/invocations/text_llm.py:22`,
  `invoke-ai/InvokeAI/invokeai/backend/model_manager/starter_models.py:880`.
- Qwen2.5-VL has standalone/quantized encoder support, including fp8-scaled
  starter models.
  Anchors: `invoke-ai/InvokeAI/invokeai/backend/model_manager/configs/qwen_vl_encoder.py:50`,
  `invoke-ai/InvokeAI/invokeai/backend/model_manager/starter_models.py:681`.
- Z-Image control support includes ControlNet config, invocation, and
  GGUF-aware denoise path.
  Anchors: `invoke-ai/InvokeAI/invokeai/backend/model_manager/configs/controlnet.py:250`,
  `invoke-ai/InvokeAI/invokeai/app/invocations/z_image_control.py:64`,
  `invoke-ai/InvokeAI/invokeai/app/invocations/z_image_denoise.py:428`.
