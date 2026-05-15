# Depth Anything

## Coverage

- Diffusers: not covered as a pipeline component.
- Transformers: covered by `src/transformers/models/depth_anything`.
- Third-party/UI: InvokeAI wraps the Transformers depth-estimation pipeline; Forge/SD.Next expose it as ControlNet preprocessing.

## Runtime Contract

InvokeAI's `DepthAnythingPipeline` wraps `transformers.pipeline(task="depth-estimation")` and returns a PIL depth image. The neural graph should be covered by the Transformers `DepthAnything` audit lane, while the UI contract adds resize, output normalization, and condition-image formatting.

## Operators

See the Transformers Depth Anything report lane for the model body. The auxiliary-specific requirements are image preprocessing, depth map postprocess, and conversion to ControlNet condition tensor/image.

## DinoML Notes

Do not duplicate the model implementation if Transformers support lands. Add an auxiliary wrapper contract that can call a compiled depth model and produce the exact UI depth image range and resolution.

## Sources

- `transformers/src/transformers/models/depth_anything`
- `invoke-ai/InvokeAI/invokeai/backend/image_util/depth_anything/depth_anything_pipeline.py`

