# NormalBae

## Coverage

- Diffusers: not covered.
- Transformers: not covered as a named family.
- Third-party/UI: vendored in InvokeAI under `normal_bae`.

## Runtime Contract

NormalBae produces surface-normal condition images. The local InvokeAI package wraps a normal map detector and includes an EfficientNet-like support tree. Output is a rendered normal RGB map, not a scalar depth map.

`controlnet_aux.NormalBaeDetector` defaults to `scannet.pt` and builds an `NNET` with `Encoder()` plus `Decoder(args)`. The wrapper uses ImageNet mean/std normalization, resizes to `detect_resolution`, runs the network, takes the final 3-channel normal prediction, maps `[-1, 1]` to `[0, 255]`, and resizes to `image_resolution`.

## Operators

- Conv/normalization/activation-heavy vision backbone.
- Image resize/normalize.
- Vector normalization and RGB normal-map rendering.

## DinoML Notes

Normal maps should be a separate output type in the preprocessor schema with vector range and channel interpretation. It should not be collapsed into generic grayscale depth.

## Sources

- `H:/controlnet_aux/src/controlnet_aux/normalbae/__init__.py`
- `H:/controlnet_aux/src/controlnet_aux/normalbae/nets/NNET.py`
- `H:/uis/invoke-ai/InvokeAI/invokeai/backend/image_util/normal_bae/__init__.py`
- `H:/uis/invoke-ai/InvokeAI/invokeai/backend/image_util/normal_bae/nets`
