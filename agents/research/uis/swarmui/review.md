# mcmonkeyprojects/SwarmUI

## Source

- UI clone: `mcmonkeyprojects/SwarmUI`

## Summary

SwarmUI is primarily an orchestrating UI around ComfyUI backends, but it adds
important product-level expectations: model metadata, model downloader, backend
management, auto-workflow generation, detailer/segmentation helpers through
extra nodes, audio model documentation, TensorRT tooling, permissions, and API
surfaces. For DinoML, SwarmUI is evidence that UI clients need a high-level
model catalog and workflow synthesis layer in addition to raw model execution.

## Model Support

- Main image models: Stable Diffusion, Z-Image, Flux, Qwen Image and whatever
  the configured ComfyUI backend supports.
- Main video models: Wan, Hunyuan Video and backend-supported video models.
- Audio: ACE-Step is documented as native audio support; audio settings and
  web routes exist for generated audio.
- Model sets/catalog: Stable-Diffusion, VAE, LoRA, Embedding, ControlNet, Clip,
  ClipVision, model classes, compatibility classes, metadata and special
  formats.
- Runtime/extensions: installable ControlNet preprocessors, GIMM frame
  interpolator, TensorRT, GGUF, and Comfy backend add-ons.
- Formats: `.safetensors`, pickle-to-safetensors conversion, `.gguf`
  recognition, model metadata headers, special format metadata.

## Feature Surface

- ComfyUI backend auto-install and pass-through workflow support.
- Generate tab with auto-workflow generation and powertools.
- Model downloader and metadata database.
- Multiple model roots and model class sorting.
- ControlNet parameter permissions and control model refresh.
- Extra Comfy nodes maintained directly by Swarm.
- Mask utilities, unsampler, custom sampler, and internal load node.
- YOLOv8 face detection and segmentation syntax through Swarm nodes.
- Optional `rembg`, `spandrel`, `kornia`, `opencv-python-headless`, and
  `ultralytics`.
- Audio model support documentation, including ACE-Step.
- TensorRT model creation tooling.

## Auxiliary Model Families

- Comfy-backed upscalers and preprocessors, inherited from ComfyUI.
- YOLOv8 detection via `ultralytics`.
- `rembg` background removal.
- Spandrel upscalers through backend/package install.
- ACE-Step audio model support.
- TensorRT-compiled model variants.
- Swarm custom sampler/unsampler and mask operations.

## Packages and Loaders

- ComfyUI backend is the execution engine for most model nodes.
- `spandrel`, `rembg`, `kornia`, `opencv-python-headless`, and `ultralytics`
  are explicitly documented package dependencies/optional installs.
- C# model handlers scan metadata, model roots, and special formats.

## Code Anchors

- `mcmonkeyprojects/SwarmUI/README.md:7` states image/video model scope.
- `mcmonkeyprojects/SwarmUI/src/Core/Program.cs:499`
  registers Stable-Diffusion model set.
- `mcmonkeyprojects/SwarmUI/src/Core/Program.cs:501`
  registers VAE model set.
- `mcmonkeyprojects/SwarmUI/src/Core/Program.cs:503`
  registers LoRA model set.
- `mcmonkeyprojects/SwarmUI/src/Core/Program.cs:505`
  registers Embedding model set.
- `mcmonkeyprojects/SwarmUI/src/Core/Program.cs:507`
  registers ControlNet model set.
- `mcmonkeyprojects/SwarmUI/src/Core/Program.cs:509`
  registers Clip model set.
- `mcmonkeyprojects/SwarmUI/src/Core/Program.cs:511`
  registers ClipVision model set.
- `mcmonkeyprojects/SwarmUI/src/Core/Settings.cs:540`
  starts default VAE settings per model family.
- `mcmonkeyprojects/SwarmUI/src/Core/Settings.cs:566`
  defines default Mochi text-to-video VAE.
- `mcmonkeyprojects/SwarmUI/src/Core/InstallableFeatures.cs:29`
  registers ControlNet preprocessors.
- `mcmonkeyprojects/SwarmUI/src/Core/InstallableFeatures.cs:31`
  registers GIMM video frame interpolator.
- `mcmonkeyprojects/SwarmUI/src/Core/InstallableFeatures.cs:32`
  registers TensorRT support.
- `mcmonkeyprojects/SwarmUI/src/Core/InstallableFeatures.cs:35`
  registers GGUF support.
- `mcmonkeyprojects/SwarmUI/README.md:150` documents ComfyUI
  auto-install.
- `mcmonkeyprojects/SwarmUI/README.md:154` lists optional packages
  including `spandrel`, `rembg`, `kornia`, and OpenCV.
- `mcmonkeyprojects/SwarmUI/README.md:155` lists `ultralytics` for
  YOLOv8 face detection and `SwarmYoloDetection`.
- `mcmonkeyprojects/SwarmUI/docs/Audio Model Support.md:10` describes
  audio model categories.
- `mcmonkeyprojects/SwarmUI/docs/Audio Model Support.md:17` names
  ACE-Step support.
- `mcmonkeyprojects/SwarmUI/src/Accounts/Permissions.cs:68` defines
  control model refresh permission.
- `mcmonkeyprojects/SwarmUI/src/Accounts/Permissions.cs:70` defines
  TensorRT creation permission.
- `mcmonkeyprojects/SwarmUI/src/BuiltinExtensions/ComfyUIBackend/ExtraNodes/SwarmComfyCommon/SwarmMasks.py:3`
  implements mask helpers.
- `mcmonkeyprojects/SwarmUI/src/BuiltinExtensions/ComfyUIBackend/ExtraNodes/SwarmComfyCommon/SwarmUnsampler.py:24`
  describes reverse sampling/unsampling.
- `mcmonkeyprojects/SwarmUI/src/BuiltinExtensions/ComfyUIBackend/ExtraNodes/SwarmComfyCommon/SwarmKSampler.py:36`
  handles video latent frame iteration.
- `mcmonkeyprojects/SwarmUI/src/BuiltinExtensions/ComfyUIBackend/ExtraNodes/SwarmComfyCommon/SwarmInternalUtil.py:43`
  defines an internal "just load" model/clip/vae node.
- `mcmonkeyprojects/SwarmUI/src/Text2Image/T2IModelHandler.cs:594`
  detects `.gguf` model names.
- `mcmonkeyprojects/SwarmUI/src/Text2Image/T2IModelHandler.cs:698`
  stores usage hints in model metadata.
- `mcmonkeyprojects/SwarmUI/src/Text2Image/T2IModelHandler.cs:711`
  stores special model format metadata.

## DinoML Gaps

- Model catalog and compatibility surface for all model sets, not only
  auxiliary helpers.
- UI-facing model catalog: metadata, preview images, usage hints, special
  formats, multiple roots, and permissions.
- Backend orchestration contract for a DinoML engine to replace or augment
  Comfy-style execution.
- Product workflows for masks, YOLO/detailing, audio, TensorRT/provider
  variants, and generated workflows.

## Further Exploration Additions

- API surface is broader than generation/model routes: admin, backend, util,
  models, T2I, basic, grid, Comfy, image-batch APIs, WebSocket generation,
  model download/edit/list, workflow extraction, Comfy feature install, LoRA
  extraction, and TensorRT creation.
  Anchors: `mcmonkeyprojects/SwarmUI/docs/API.md:5`,
  `mcmonkeyprojects/SwarmUI/docs/APIRoutes/ModelsAPI.md:69`,
  `mcmonkeyprojects/SwarmUI/docs/APIRoutes/ComfyUIWebAPI.md:61`.
- Workflow generator supports modern families including Nunchaku formats, Wan,
  Nvidia Cosmos, Hunyuan Image Refiner, Hunyuan Video 1.5 SR, Flux2, OmniGen,
  Qwen, ZImage, Chroma, Ernie, HiDreamO1, Wan Phantom, LTXv2, Cosmos I2V, and
  Qwen Image Edit Plus.
  Anchors:
  `mcmonkeyprojects/SwarmUI/src/BuiltinExtensions/ComfyUIBackend/WorkflowGenerator.cs:391`,
  `mcmonkeyprojects/SwarmUI/src/BuiltinExtensions/ComfyUIBackend/WorkflowGenerator.cs:1026`,
  `mcmonkeyprojects/SwarmUI/src/BuiltinExtensions/ComfyUIBackend/WorkflowGenerator.cs:2336`.
- Common model catalog includes Z-Image Turbo FP8Mix, Flux.2 VAE, Cosmos VAE,
  LTX-2 Video/Audio VAE, Qwen Image VAE, and Hunyuan Image 2.1 VAE.
  Anchor: `mcmonkeyprojects/SwarmUI/src/Text2Image/CommonModels.cs:76`.
- ACE-Step exposes audio parameters as T2I controls: style, BPM, time
  signature, language, and key scale.
  Anchors: `mcmonkeyprojects/SwarmUI/src/Text2Image/T2IParamTypes.cs:423`,
  `mcmonkeyprojects/SwarmUI/src/Text2Image/T2IParamTypes.cs:436`.
- Text encoder settings include CLIP-L/G, T5-XXL, LLaVA, LLaMA, Qwen LLM, and
  Mistral LLM across SD3/Flux/Wan/Hunyuan/HiDream/OmniGen/Qwen/Flux2.
  Anchors: `mcmonkeyprojects/SwarmUI/src/Text2Image/T2IParamTypes.cs:691`,
  `mcmonkeyprojects/SwarmUI/src/Text2Image/T2IParamTypes.cs:712`.
- TensorRT engine support includes SDXL 0.9/1.0, SD3 Medium, SDXL Turbo, SDXL
  Refiner, and SVD engine classes, with reduced flexibility noted in docs.
  Anchors: `mcmonkeyprojects/SwarmUI/src/Text2Image/T2IModelClassSorter.cs:824`,
  `mcmonkeyprojects/SwarmUI/docs/Model Support.md:676`.
