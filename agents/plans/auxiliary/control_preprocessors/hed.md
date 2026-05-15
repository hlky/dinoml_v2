# HED

## Coverage

- Diffusers: not covered as a core model.
- Transformers: not covered.
- Third-party/UI: vendored from `controlnet_aux` in InvokeAI and Forge-style preprocessors.

## Runtime Contract

The InvokeAI implementation loads `lllyasviel/Annotators/ControlNetHED.pth` into `ControlNetHED_Apache2`. The model is a five-stage VGG-like NCHW conv stack with side projections at each stage. Outputs are resized to input resolution, averaged, passed through sigmoid, scaled to `uint8`, and optionally transformed into a scribble map through NMS, blur, and hard thresholding.

The local `controlnet_aux` package exposes the same shape as `HEDdetector`. It defaults to `ControlNetHED.pth`, accepts `safe` and `scribble` flags, and the registry aliases those into `softedge_hed`, `softedge_hedsafe`, `scribble_hed`, and `scribble_hedsafe`.

## Operators

- Conv2d, ReLU, MaxPool2d.
- Multi-scale resize/interpolate.
- Sigmoid, mean over side-output channel stack.
- Optional NMS, Gaussian blur, threshold.

## DinoML Notes

The neural part is simple conv inference, but the useful UI feature includes pre/postprocessing. Keep `safe` and `scribble` as explicit preprocessing options. First integration can run the model through a normal vision graph and leave NMS/blur on CPU.

## Sources

- `H:/controlnet_aux/src/controlnet_aux/hed/__init__.py`
- `H:/controlnet_aux/src/controlnet_aux/processor.py`
- `invoke-ai/InvokeAI/invokeai/backend/image_util/hed.py`
- `lllyasviel/stable-diffusion-webui-forge/extensions-builtin/forge_legacy_preprocessors/legacy_preprocessors/preprocessor.py:119`
