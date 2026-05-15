# Segmentation, Detection, and Mask Workflows

## Why This Matters

Modern UIs use segmentation and detection for object selection, regional
guidance, inpainting, background removal, detailers, and video mask tracking.
Even when SAM or GroundingDINO are available through `transformers`, the UI
feature includes prompts/boxes/points, NMS, mask refinement, compositing, and
stateful selection.

## Model Families

- SAM, SAM2, SAM3.
- GroundingDINO and Grounded-SAM workflows.
- RT-DETR and COCO detectors.
- YOLOv8 via Ultralytics.
- CLIPSeg.
- rembg/U2Net/ISNet background removal.
- MediaPipe face detection and face mesh.
- FaceDetailer detector + mask + inpaint loops.

## Code Anchors

- `Comfy-Org/ComfyUI/comfy_extras/nodes_sam3.py:88`
  SAM3 detect/video-track/mask node surface.
- `Comfy-Org/ComfyUI/comfy_extras/nodes_sam3.py:260`
  video tracking entrypoints.
- `Comfy-Org/ComfyUI/comfy_extras/nodes_rtdetr.py:146`
  RT-DETR/COCO detection node.
- `Comfy-Org/ComfyUI/comfy_extras/nodes_bg_removal.py:10`
  background-removal model loading.
- `invoke-ai/InvokeAI/invokeai/app/invocations/segment_anything.py:51`
  SAM/SAM2 invocation.
- `invoke-ai/InvokeAI/invokeai/app/invocations/grounding_dino.py:30`
  GroundingDINO invocation.
- `mcmonkeyprojects/SwarmUI/src/BuiltinExtensions/ComfyUIBackend/ExtraNodes/SwarmComfyCommon/SwarmClipSeg.py:4`
  CLIPSeg through transformers.
- `mcmonkeyprojects/SwarmUI/src/BuiltinExtensions/ComfyUIBackend/ExtraNodes/SwarmComfyExtra/SwarmYolo.py:5`
  `ultralytics.YOLO`.
- `mcmonkeyprojects/SwarmUI/src/BuiltinExtensions/ComfyUIBackend/ExtraNodes/SwarmComfyExtra/SwarmRemBg.py:5`
  `rembg`.
- `LykosAI/StabilityMatrix/StabilityMatrix.Core/Helper/RemoteModels.cs:198`
  Ultralytics and SAM model catalogs.
- `LykosAI/StabilityMatrix/StabilityMatrix.Core/Models/Api/Comfy/Nodes/ComfyNodeBuilder.cs:698`
  Ultralytics/SAM workflow node construction.
- `deepbeepmeep/Wan2GP/preprocessing/sam3/model/sam3_video_predictor.py`
  SAM3 video predictor implementation.
- `deepbeepmeep/Wan2GP/preprocessing/sam3/perflib/triton/nms.py`
  Triton NMS helper.

## DinoML Gap

High. Treat this as a workflow surface:

- detector outputs with boxes/classes/scores;
- mask prompts from points, boxes, text, or selected objects;
- NMS and mask selection;
- mask refinement and compositing;
- video tracking memory/state;
- conversion to inpaint masks, control images, and region maps.

The minimum useful DinoML slice is likely not "compile SAM"; it is a bounded
mask/detection artifact contract with one supported backend/model family.

