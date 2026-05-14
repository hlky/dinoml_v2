# UI Coverage Map

## AUTOMATIC1111/stable-diffusion-webui

- Core extras: ESRGAN, RealESRGAN, SwinIR, ScuNET, DAT, HAT, LDSR, GFPGAN,
  CodeFormer, BLIP/CLIP interrogation, DeepDanbooru.
- Key files:
  - `H:/uis/AUTOMATIC1111/stable-diffusion-webui/modules/esrgan_model.py`
  - `H:/uis/AUTOMATIC1111/stable-diffusion-webui/modules/realesrgan_model.py`
  - `H:/uis/AUTOMATIC1111/stable-diffusion-webui/extensions-builtin/SwinIR/scripts/swinir_model.py`
  - `H:/uis/AUTOMATIC1111/stable-diffusion-webui/extensions-builtin/ScuNET/scripts/scunet_model.py`
  - `H:/uis/AUTOMATIC1111/stable-diffusion-webui/modules/gfpgan_model.py`
  - `H:/uis/AUTOMATIC1111/stable-diffusion-webui/modules/codeformer_model.py`
  - `H:/uis/AUTOMATIC1111/stable-diffusion-webui/modules/interrogate.py`
  - `H:/uis/AUTOMATIC1111/stable-diffusion-webui/modules/deepbooru.py`

## Forge, reForge, Forge Classic

- Adds or preserves the built-in ControlNet preprocessor suite under
  `extensions-builtin/forge_legacy_preprocessors`.
- reForge/classic broaden Spandrel architecture handling and include memory or
  tiling heuristics for large image restoration networks such as ATD, DAT, and
  DRCT.
- Key files:
  - `H:/uis/lllyasviel/stable-diffusion-webui-forge/extensions-builtin/forge_legacy_preprocessors/legacy_preprocessors/preprocessor.py`
  - `H:/uis/lllyasviel/stable-diffusion-webui-forge/extensions-builtin/forge_legacy_preprocessors/legacy_preprocessors/preprocessor_compiled.py`
  - `H:/uis/Panchovix/stable-diffusion-webui-reForge/extensions-builtin/sd_forge_controlnet/scripts/controlnet.py`
  - `H:/uis/Haoming02/sd-webui-forge-classic/modules/upscaler_utils.py`

## SD.Next

- Consolidates upscalers behind Spandrel and VAE/algorithmic upscalers.
- Has its own Control processor/unit system for ControlNet, T2I-Adapter,
  ControlNet-XS, LLLite, Anyline, HED, DPT/GLPN, OneFormer, LeReS, and TEED.
- Caption APIs cover OpenCLIP, BLIP, taggers, and VLMs.
- Key files:
  - `H:/uis/vladmandic/sdnext/modules/upscaler_spandrel.py`
  - `H:/uis/vladmandic/sdnext/modules/upscaler_vae.py`
  - `H:/uis/vladmandic/sdnext/modules/control/processor.py`
  - `H:/uis/vladmandic/sdnext/modules/control/proc/*`
  - `H:/uis/vladmandic/sdnext/modules/control/units/*`
  - `H:/uis/vladmandic/sdnext/modules/caption/caption.py`

## ComfyUI

- Built-in model folders already name the auxiliary surface:
  `upscale_models`, `latent_upscale_models`, `clip_vision`, `style_models`,
  `gligen`, `audio_encoders`, `background_removal`, `frame_interpolation`,
  `optical_flow`.
- Comfy has local implementations or nodes for Spandrel upscalers, ControlNet,
  CLIP Vision, GLIGEN, TAESD/TAEHV, audio encoders/VAEs, RIFE/FILM, RAFT,
  RT-DETR, SAM3, and many video/audio model families.
- Key files:
  - `H:/uis/Comfy-Org/ComfyUI/folder_paths.py`
  - `H:/uis/Comfy-Org/ComfyUI/nodes.py`
  - `H:/uis/Comfy-Org/ComfyUI/comfy_extras/nodes_upscale_model.py`
  - `H:/uis/Comfy-Org/ComfyUI/comfy_extras/nodes_frame_interpolation.py`
  - `H:/uis/Comfy-Org/ComfyUI/comfy_extras/nodes_void.py`
  - `H:/uis/Comfy-Org/ComfyUI/comfy_extras/nodes_sam3.py`
  - `H:/uis/Comfy-Org/ComfyUI/comfy/audio_encoders/*`
  - `H:/uis/Comfy-Org/ComfyUI/comfy/ldm/*`

## InvokeAI

- Tracks Spandrel as a typed model category and provides tiled Spandrel
  invocations.
- Includes RealESRGAN/ESRGAN invocation, Depth Anything, DWPose via ONNX,
  MediaPipe face detection, HED/MLSD/lineart preprocessors, SAM/SAM2,
  GroundingDINO, IP-Adapter, Flux Redux, and CLIP/SigLIP/Qwen vision encoders.
- Key files:
  - `H:/uis/invoke-ai/InvokeAI/invokeai/backend/model_manager/taxonomy.py`
  - `H:/uis/invoke-ai/InvokeAI/invokeai/backend/model_manager/configs/spandrel.py`
  - `H:/uis/invoke-ai/InvokeAI/invokeai/app/invocations/spandrel_image_to_image.py`
  - `H:/uis/invoke-ai/InvokeAI/invokeai/app/invocations/upscale.py`
  - `H:/uis/invoke-ai/InvokeAI/invokeai/app/invocations/dw_openpose.py`
  - `H:/uis/invoke-ai/InvokeAI/invokeai/app/invocations/segment_anything.py`
  - `H:/uis/invoke-ai/InvokeAI/invokeai/backend/model_manager/starter_models.py`

## Wan2GP

- The broadest gap source in this pass. It includes custom Wan and LTX video
  stacks, flow-match schedulers, camera/trajectory/VACE/MultiTalk conditioning,
  RAFT/DWPose/MiDaS/Depth Anything 2/3 preprocessors, SAM3/MatAnyone masks,
  FlashVSR, RIFE, MMAudio, TTS/voice/audio pipelines, custom qtypes, sparse
  attention, and Triton kernels.
- This is where DinoML's artifact-visible state discipline matters most:
  condition artifacts, temporal shape contracts, scheduler plans, VAE tiling,
  model residency, and packed qtype/provider selection should be explicit.

## SwarmUI and StabilityMatrix

- Useful as compatibility maps. They enumerate what a Comfy-backed UI expects
  to be installable and shareable: IP-Adapter, ControlNet preprocessors, frame
  interpolation, SAM2/SAM3, Ultralytics/YOLO, CLIPSeg, rembg, upscalers,
  video/audio model categories, and remote model catalogs.
- Key files:
  - `H:/uis/mcmonkeyprojects/SwarmUI/src/BuiltinExtensions/ComfyUIBackend/ComfyUISelfStartBackend.cs`
  - `H:/uis/mcmonkeyprojects/SwarmUI/src/Core/InstallableFeatures.cs`
  - `H:/uis/LykosAI/StabilityMatrix/StabilityMatrix.Core/Models/Api/Comfy/ComfyAuxPreprocessor.cs`
  - `H:/uis/LykosAI/StabilityMatrix/StabilityMatrix.Core/Helper/RemoteModels.cs`

