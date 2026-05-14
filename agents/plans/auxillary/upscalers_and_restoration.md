# Upscalers and Restoration

## Why This Matters

All surveyed UI families expose image upscaling and face restoration as normal
user workflows. These are not diffusion pipelines; they are image-to-image
auxiliary models plus tiling, autoscale, face crop/alignment, and paste-back
logic. DinoML will need a separate auxiliary runtime story for these if the UI
target includes common "extras" workflows.

## Model Families

- ESRGAN/RRDB and RealESRGAN/SRVGG.
- SwinIR and ScuNET.
- DAT, HAT, Compact, GRL, ATD, DRCT.
- RealPLKSR, Nomos, AnimeSharp, UltraMix and other downloadable Spandrel
  community models.
- GFPGAN and CodeFormer for face restoration.
- RetinaFace/facexlib, insightface, and FaceDetailer-style detector + mask
  workflows.

## Packages and Loaders

- `spandrel` is the common modern loader for arbitrary image-to-image/upscale
  architectures.
- `spandrel_extra_arches` extends architecture coverage in Comfy-style stacks.
- `facexlib` supplies face detection/alignment helpers for GFPGAN/CodeFormer.
- Some UIs still carry direct ESRGAN/RealESRGAN/SwinIR/ScuNET model code.

## Code Anchors

- `H:/uis/AUTOMATIC1111/stable-diffusion-webui/modules/modelloader.py:164`
  `load_spandrel_model`.
- `H:/uis/AUTOMATIC1111/stable-diffusion-webui/modules/esrgan_model.py:7`
  `UpscalerESRGAN`.
- `H:/uis/AUTOMATIC1111/stable-diffusion-webui/modules/realesrgan_model.py:39`
  RealESRGAN via Spandrel.
- `H:/uis/AUTOMATIC1111/stable-diffusion-webui/modules/dat_model.py:9`
  `UpscalerDAT`.
- `H:/uis/AUTOMATIC1111/stable-diffusion-webui/modules/hat_model.py:10`
  `UpscalerHAT`.
- `H:/uis/AUTOMATIC1111/stable-diffusion-webui/extensions-builtin/SwinIR/scripts/swinir_model.py:70`
  SwinIR loader.
- `H:/uis/AUTOMATIC1111/stable-diffusion-webui/extensions-builtin/ScuNET/scripts/scunet_model.py:64`
  ScuNET loader.
- `H:/uis/Comfy-Org/ComfyUI/comfy_extras/nodes_upscale_model.py:1`
  Spandrel-backed Comfy upscaler node.
- `H:/uis/invoke-ai/InvokeAI/invokeai/backend/model_manager/taxonomy.py:65`
  `ModelType.SpandrelImageToImage`.
- `H:/uis/invoke-ai/InvokeAI/invokeai/app/invocations/spandrel_image_to_image.py:166`
  tiled/autoscale Spandrel invocation.
- `H:/uis/invoke-ai/InvokeAI/invokeai/app/invocations/upscale.py:14`
  RealESRGAN/ESRGAN model URLs and invocation.
- `H:/uis/vladmandic/sdnext/modules/upscaler_spandrel.py:44`
  runtime `spandrel` install/load path.
- `H:/uis/Panchovix/stable-diffusion-webui-reForge/modules/compact_model.py:62`
  Compact model route.
- `H:/uis/Panchovix/stable-diffusion-webui-reForge/modules/grl_model.py:62`
  GRL model route.
- `H:/uis/Haoming02/sd-webui-forge-classic/modules/upscaler_utils.py:72`
  ATD/DAT/DRCT tiling/memory handling.
- `H:/uis/LykosAI/StabilityMatrix/StabilityMatrix.Core/Models/Api/Comfy/ComfyUpscaler.cs:9`
  StabilityMatrix upscaler model surface.
- `H:/uis/LykosAI/StabilityMatrix/StabilityMatrix.Core/Helper/RemoteModels.cs:22`
  downloadable upscaler catalog.

## Face Restoration Anchors

- `H:/uis/AUTOMATIC1111/stable-diffusion-webui/modules/gfpgan_model.py:23`
  `FaceRestorerGFPGAN`.
- `H:/uis/AUTOMATIC1111/stable-diffusion-webui/modules/codeformer_model.py:25`
  `FaceRestorerCodeFormer`.
- `H:/uis/AUTOMATIC1111/stable-diffusion-webui/modules/face_restoration_utils.py:43`
  facexlib RetinaFace/FaceRestoreHelper.
- `H:/uis/Haoming02/sd-webui-forge-classic/modules/gfpgan_model.py:25`
  GFPGAN v1.4 download path.
- `H:/uis/LykosAI/StabilityMatrix/StabilityMatrix.Core/Models/Api/Comfy/Nodes/ComfyNodeBuilder.cs:1160`
  Comfy FaceDetailer workflow construction.

## DinoML Gap

High. This needs model-family detection, tiled image execution, explicit
tile/overlap metadata, predictable image dtype/range handling, and face
crop/align/paste-back workflow contracts. It should not be hidden inside a
diffusion model path.

## Candidate First Slice

Start with one RRDB/RealESRGAN-compatible model loaded through a fixed reference
path, plus tiled inference. Record tile size, overlap, scale, input range,
output range, and padding/crop policy as artifact-visible metadata.

