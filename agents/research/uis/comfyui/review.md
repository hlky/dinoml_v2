# Comfy-Org/ComfyUI

## Source

- UI clone: `Comfy-Org/ComfyUI`

## Summary

ComfyUI has the broadest auxiliary-model surface because it exposes model
categories as graph node types and folder namespaces. Its local categories name
many DinoML gaps directly: upscalers, latent upscalers, CLIP vision, GLIGEN,
audio encoders, background removal, frame interpolation, optical flow, and
segmentation.

## Model Support

- Main image models: SD 1.x/2.x, SDXL, SDXL Turbo, SD3/SD3.5, AuraFlow,
  HunyuanDiT, Flux, Flux Kontext, Flux 2, Qwen Image, Qwen Image Edit, Hunyuan
  Image 2.1, and API-node hosted closed-source models.
- Main video models: Stable Video Diffusion, Mochi, LTX-Video, Hunyuan Video,
  Hunyuan Video 1.5, Wan 2.1, Wan 2.2.
- Audio models: Stable Audio plus newer audio nodes/blueprints; Whisper encoder
  support appears as a local audio encoder.
- 3D/API models: Hunyuan3D and API-node hosted services.
- File formats/components: all-in-one `.ckpt`/`.safetensors` checkpoints,
  standalone diffusion models, VAEs, CLIP/text encoders, LoRAs, ControlNets,
  T2I-Adapters, CLIP Vision, style models, GLIGEN, and auxiliary folders.
- Adapters: LoRA, ControlNet, T2I-Adapter, IP/vision conditioning via graph
  nodes and model folders.

## Feature Surface

- Node graph execution with explicit model inputs and outputs.
- Separate folders for auxiliary model classes.
- Spandrel-based image upscaling.
- CLIP vision and style conditioning.
- GLIGEN positional/textbox conditioning.
- TAESD/TAEHV latent previews.
- Frame interpolation and optical flow nodes.
- BiRefNet background removal.
- SAM3 image/video segmentation blueprints.
- Audio encoders and audio/video model components.
- Quantized tensor layout support, including NVFP4 layout metadata.

## Auxiliary Model Families

- Spandrel image-to-image upscalers.
- CLIP Vision, SigLIP, and LLaVA-style vision encoders.
- Style models and GLIGEN.
- TAESD and TAEHV preview decoders.
- RIFE/FILM-style frame interpolation.
- RAFT-style optical flow.
- SAM3/SAM3.1 segmentation and tracking.
- RT-DETR/detection surfaces in extra nodes.
- BiRefNet background removal.
- Whisper audio encoder.
- Big audio/video model families through Comfy extra nodes and blueprints.
- NVFP4 quantized tensor layouts.

## Packages and Loaders

- `spandrel` and optional `spandrel_extra_arches` for upscalers.
- Comfy local loaders for CLIP vision, GLIGEN, background removal, audio
  encoders, model detection, and quantized layouts.

## Code Anchors

- `Comfy-Org/ComfyUI/README.md:73` lists SDXL/SDXL Turbo support.
- `Comfy-Org/ComfyUI/README.md:75` lists SD3/SD3.5 support.
- `Comfy-Org/ComfyUI/README.md:77` lists AuraFlow support.
- `Comfy-Org/ComfyUI/README.md:78` lists HunyuanDiT support.
- `Comfy-Org/ComfyUI/README.md:79` lists Flux support.
- `Comfy-Org/ComfyUI/README.md:82` lists Qwen Image support.
- `Comfy-Org/ComfyUI/README.md:84` lists Flux 2 support.
- `Comfy-Org/ComfyUI/README.md:92` starts video model support list.
- `Comfy-Org/ComfyUI/README.md:100` starts audio model support list.
- `Comfy-Org/ComfyUI/README.md:109` documents checkpoint/component
  loading.
- `Comfy-Org/ComfyUI/README.md:119` documents ControlNet/T2I-Adapter.
- `Comfy-Org/ComfyUI/folder_paths.py:30` defines `clip_vision`.
- `Comfy-Org/ComfyUI/folder_paths.py:39` defines `upscale_models`.
- `Comfy-Org/ComfyUI/folder_paths.py:41` defines
  `latent_upscale_models`.
- `Comfy-Org/ComfyUI/folder_paths.py:53` defines `audio_encoders`.
- `Comfy-Org/ComfyUI/folder_paths.py:55` defines
  `background_removal`.
- `Comfy-Org/ComfyUI/folder_paths.py:57` defines
  `frame_interpolation`.
- `Comfy-Org/ComfyUI/folder_paths.py:59` defines `optical_flow`.
- `Comfy-Org/ComfyUI/comfy_extras/nodes_upscale_model.py`
  implements Spandrel upscaler nodes.
- `Comfy-Org/ComfyUI/nodes.py:1017` loads CLIP vision models.
- `Comfy-Org/ComfyUI/nodes.py:1053` loads style models.
- `Comfy-Org/ComfyUI/nodes.py:1152` loads GLIGEN models.
- `Comfy-Org/ComfyUI/latent_preview.py:39` implements TAESD preview.
- `Comfy-Org/ComfyUI/latent_preview.py:47` implements TAEHV preview.
- `Comfy-Org/ComfyUI/comfy/bg_removal_model.py:11` references
  BiRefNet.
- `Comfy-Org/ComfyUI/comfy/audio_encoders/whisper.py:9`
  implements Whisper feature extraction.
- `Comfy-Org/ComfyUI/comfy/model_detection.py:774` detects SAM3/SAM3.1.
- `Comfy-Org/ComfyUI/comfy/quant_ops.py:134` implements
  `TensorCoreNVFP4Layout`.
- `Comfy-Org/ComfyUI/blueprints/Frame Interpolation.json`
  downloads FILM interpolation weights.
- `Comfy-Org/ComfyUI/blueprints/Remove Background (BiRefNet).json`
  downloads BiRefNet weights.
- `Comfy-Org/ComfyUI/blueprints/Image Segmentation (SAM3).json`
  shows SAM3 mask workflow.
- `Comfy-Org/ComfyUI/blueprints/Video Segmentation (SAM3).json`
  shows temporally consistent SAM3 video masks.

## DinoML Gaps

- General graph-runtime model taxonomy for image, video, audio, 3D/API,
  adapters, encoders, VAEs, and auxiliary models.
- Node-compatible auxiliary model contracts with typed inputs/outputs, not only
  pipeline calls.
- File/folder taxonomy for auxiliary model discovery.
- Explicit runtime support for image/video/audio helper models, segmentation
  workflows, quantized tensor layouts, and preview decoders.

## Further Exploration Additions

- Supported model families are broader than the first pass: Cosmos T2V/I2V/
  Predict2, Anima, Z-Image, Wan variants, Hunyuan3D, HiDream/HiDreamO1,
  Chroma/Radiance, ACE-Step, OmniGen2, Hunyuan Image/Video 1.5, Kandinsky 5,
  LongCat Image, RT-DETR v4, Ernie Image, SAM3/SAM3.1, and CogVideoX
  T2V/I2V/Inpaint are represented in supported model detection.
  Anchors include
  `Comfy-Org/ComfyUI/comfy/supported_models.py:954`,
  `Comfy-Org/ComfyUI/comfy/supported_models.py:1027`,
  `Comfy-Org/ComfyUI/comfy/supported_models.py:1098`,
  `Comfy-Org/ComfyUI/comfy/supported_models.py:1479`,
  `Comfy-Org/ComfyUI/comfy/supported_models.py:1931`.
- Folder taxonomy also includes `hypernetworks`, `photomaker`, `classifiers`,
  `model_patches`, and `custom_nodes`.
  Anchor: `Comfy-Org/ComfyUI/folder_paths.py:43`.
- Product/API endpoints cover model discovery, extensions, metadata, object
  info, jobs, and prompt submission.
  Anchors: `Comfy-Org/ComfyUI/server.py:330`,
  `Comfy-Org/ComfyUI/server.py:624`,
  `Comfy-Org/ComfyUI/server.py:916`.
- Media/API helpers support audio, video, and 3D uploads, plus job previews for
  image/video/audio/3D/text.
  Anchors: `Comfy-Org/ComfyUI/comfy_api_nodes/util/upload_helpers.py:113`,
  `Comfy-Org/ComfyUI/comfy_execution/jobs.py:23`.
- LTX 2.x synchronized audio/video workflows appear in blueprints.
  Anchors: `Comfy-Org/ComfyUI/blueprints/Canny to Video (LTX 2.0).json:659`,
  `Comfy-Org/ComfyUI/blueprints/First-Last-Frame to Video.json:2480`.
- Audio is broader than Stable Audio/Whisper: latent audio, audio VAE
  encode/decode, tiled decode, FLAC/MP3/Opus save, preview/load/record/trim.
  Anchor: `Comfy-Org/ComfyUI/comfy_extras/nodes_audio.py:15`.
- Runtime formats include fp8 e4m3/e5m2/e8m0, scaled fp8, and NVFP4/fp4
  machinery.
  Anchors: `Comfy-Org/ComfyUI/nodes.py:936`,
  `Comfy-Org/ComfyUI/comfy/cli_args.py:68`,
  `Comfy-Org/ComfyUI/comfy/quant_ops.py:134`.
