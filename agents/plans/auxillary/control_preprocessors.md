# Control and Conditioning Preprocessors

## Why This Matters

ControlNet, T2I-Adapter, IP-Adapter, Flux control, regional guidance, and
inpaint/detailer workflows depend heavily on preprocessing models and
postprocessing rules. The generated conditioning image or tensor is often the
real UI feature, even when the downstream ControlNet weights are available from
`diffusers`.

## Model and Feature Families

- Edges and lines: Canny, HED, HED safe, TEED, PiDiNet, Anyline, lineart,
  lineart anime, manga line, scribble/sketch.
- Structure: MLSD straight-line detection, tile, blur, shuffle, color maps.
- Pose: OpenPose, DWPose, AnimalPose, DensePose, MeshGraphormer, MediaPipe face
  mesh.
- Depth and normals: MiDaS, LeReS, ZoeDepth, Depth Anything v1/v2/v3,
  NormalBae/BAE normals.
- Segmentation and matting: OneFormer, UniFormer semantic segmentation,
  AnimeFace segment, LaMa.
- Vision conditioning helpers: CLIP Vision, style models, GLIGEN, IP-Adapter
  reference image features.

## Packages

`controlnet_aux`, `onnxruntime`, `opencv-python`, `opencv-contrib-python`,
`mediapipe`, `insightface`, `pycocotools`, `timm`, and UI-vendored detector or
segmentation code.

## Local controlnet_aux Findings

The local clone at `H:/controlnet_aux` is a concrete source package for many of
these preprocessors, not just a dependency name. Detailed reports now live under
`control_preprocessors/`, including:

- `control_preprocessors/controlnet_aux_registry.md`: package exports and
  `Processor` registry aliases.
- `control_preprocessors/sam_automatic_masks.md`: Segment Anything automatic
  mask rendering via `SamDetector`.
- Updated detector reports for Canny, HED, TEED, Anyline, PiDiNet, MLSD,
  lineart, OpenPose/DWPose, MiDaS, ZoeDepth/LeReS, NormalBae, and MediaPipe
  face.

Several entries are mixed coverage: ZoeDepth/DPT/GLPN overlap Transformers
model families, while `controlnet_aux` owns the preprocessor wrapper,
checkpoint defaults, mode aliases, resize policy, and rendered condition image.

## Forge and reForge Anchors

- `H:/uis/lllyasviel/stable-diffusion-webui-forge/extensions-builtin/forge_legacy_preprocessors/scripts/legacy_preprocessors.py:40`
  `LegacyPreprocessor`.
- `H:/uis/lllyasviel/stable-diffusion-webui-forge/extensions-builtin/forge_legacy_preprocessors/scripts/legacy_preprocessors.py:114`
  iterates/registers legacy preprocessors.
- `H:/uis/lllyasviel/stable-diffusion-webui-forge/extensions-builtin/forge_legacy_preprocessors/legacy_preprocessors/preprocessor.py:119`
  HED.
- `H:/uis/lllyasviel/stable-diffusion-webui-forge/extensions-builtin/forge_legacy_preprocessors/legacy_preprocessors/preprocessor.py:175`
  MLSD.
- `H:/uis/lllyasviel/stable-diffusion-webui-forge/extensions-builtin/forge_legacy_preprocessors/legacy_preprocessors/preprocessor.py:199`
  Depth Anything.
- `H:/uis/lllyasviel/stable-diffusion-webui-forge/extensions-builtin/forge_legacy_preprocessors/legacy_preprocessors/preprocessor.py:219`
  Depth Anything v2.
- `H:/uis/lllyasviel/stable-diffusion-webui-forge/extensions-builtin/forge_legacy_preprocessors/legacy_preprocessors/preprocessor.py:232`
  MiDaS depth/normal.
- `H:/uis/lllyasviel/stable-diffusion-webui-forge/extensions-builtin/forge_legacy_preprocessors/legacy_preprocessors/preprocessor.py:263`
  LeReS.
- `H:/uis/lllyasviel/stable-diffusion-webui-forge/extensions-builtin/forge_legacy_preprocessors/legacy_preprocessors/preprocessor.py:308`
  OpenPose/DWPose route.
- `H:/uis/lllyasviel/stable-diffusion-webui-forge/extensions-builtin/forge_legacy_preprocessors/legacy_preprocessors/preprocessor.py:456`
  lineart.
- `H:/uis/lllyasviel/stable-diffusion-webui-forge/extensions-builtin/forge_legacy_preprocessors/legacy_preprocessors/preprocessor.py:577`
  ZoeDepth.
- `H:/uis/lllyasviel/stable-diffusion-webui-forge/extensions-builtin/forge_legacy_preprocessors/legacy_preprocessors/preprocessor.py:607`
  OneFormer.
- `H:/uis/lllyasviel/stable-diffusion-webui-forge/extensions-builtin/forge_legacy_preprocessors/legacy_preprocessors/preprocessor.py:700`
  DensePose.
- `H:/uis/lllyasviel/stable-diffusion-webui-forge/extensions-builtin/forge_legacy_preprocessors/legacy_preprocessors/preprocessor.py:715`
  TEED.
- `H:/uis/Panchovix/stable-diffusion-webui-reForge/extensions-builtin/sd_forge_controlnet/scripts/controlnet.py:310`
  ControlNet invokes selected preprocessors.

## SD.Next Anchors

- `H:/uis/vladmandic/sdnext/modules/control/processor.py:34`
  processor categories including TEED, Anyline, LeReS, OneFormer.
- `H:/uis/vladmandic/sdnext/modules/control/proc/anyline/__init__.py:10`
  Anyline via `controlnet_aux`.
- `H:/uis/vladmandic/sdnext/modules/control/proc/dpt.py:25`
  DPT depth using transformers.
- `H:/uis/vladmandic/sdnext/modules/control/proc/glpn.py:17`
  GLPN depth using transformers.
- `H:/uis/vladmandic/sdnext/modules/control/proc/hed.py:63`
  custom HED.
- `H:/uis/vladmandic/sdnext/modules/control/units/controlnet.py:16`
  ControlNet unit load surface.
- `H:/uis/vladmandic/sdnext/modules/control/units/t2iadapter.py:16`
  T2I adapter unit.
- `H:/uis/vladmandic/sdnext/modules/control/units/lite.py:19`
  LLLite control unit.

## InvokeAI and StabilityMatrix Anchors

- `H:/uis/invoke-ai/InvokeAI/invokeai/app/invocations/hed.py:12`
  HED invocation.
- `H:/uis/invoke-ai/InvokeAI/invokeai/app/invocations/mlsd.py:11`
  MLSD invocation.
- `H:/uis/invoke-ai/InvokeAI/invokeai/app/invocations/lineart.py:12`
  lineart invocation.
- `H:/uis/invoke-ai/InvokeAI/invokeai/app/invocations/depth_anything.py:21`
  Depth Anything pipeline invocation.
- `H:/uis/invoke-ai/InvokeAI/invokeai/app/invocations/dw_openpose.py:1`
  ONNX DWPose invocation.
- `H:/uis/invoke-ai/InvokeAI/invokeai/app/invocations/mediapipe_face.py:9`
  MediaPipe face detection.
- `H:/uis/LykosAI/StabilityMatrix/StabilityMatrix.Core/Models/Api/Comfy/ComfyAuxPreprocessor.cs:15`
  broad Comfy preprocessor enum.

## DinoML Gap

Very high. This should be modeled as image-to-condition compilation/runtime
surface, with explicit output image/tensor shape, channel count, dtype/range,
thresholds, resize policy, and postprocess. The model import itself is only one
piece.
