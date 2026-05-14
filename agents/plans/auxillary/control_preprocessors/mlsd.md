# MLSD

## Coverage

- Diffusers: not covered.
- Transformers: not covered.
- Third-party/UI: vendored in InvokeAI and exposed by ControlNet preprocessor UIs.

## Runtime Contract

MLSD detects straight line segments. The inspected tiny model uses a MobileNetV2-style NCHW backbone, feature pyramid merges, residual conv blocks, dilated conv, and bilinear upsampling. The raw tensor output is decoded into line detections by preprocessing code outside the network.

`controlnet_aux.MLSDdetector` loads the large variant by default (`mlsd_large_512_fp32.pth`, or `annotator/ckpts/mlsd_large_512_fp32.pth` for `lllyasviel/ControlNet`). Runtime defaults are `thr_v=0.1`, `thr_d=0.1`, `detect_resolution=512`, and `image_resolution=512`. Detected line coordinates from `pred_lines` are rasterized with `cv2.line` onto a black condition image.

## Operators

- Conv2d, BatchNorm2d, ReLU/ReLU6.
- Depthwise separable conv.
- Pad, MaxPool2d, bilinear interpolate.
- Concat and residual add.
- Postprocess line proposal decoding.

## DinoML Notes

Neural inference is feasible with common conv primitives, but end-to-end parity depends on the line decoder and thresholding. Keep MLSD as a structured detector output plus rendered condition image, not just an image tensor.

## Sources

- `H:/controlnet_aux/src/controlnet_aux/mlsd/__init__.py`
- `H:/uis/invoke-ai/InvokeAI/invokeai/backend/image_util/mlsd/models/mbv2_mlsd_tiny.py`
- `H:/uis/invoke-ai/InvokeAI/invokeai/backend/image_util/mlsd/models/mbv2_mlsd_large.py`
- `H:/uis/invoke-ai/InvokeAI/invokeai/backend/image_util/mlsd/__init__.py`
