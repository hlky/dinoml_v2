# ZoeDepth and LeReS

## Coverage

- Diffusers: not covered.
- Transformers: ZoeDepth is covered by `src/transformers/models/zoedepth`; LeReS is not covered in the inspected Transformers tree.
- Third-party/UI: Forge exposes both as ControlNet preprocessors.

## Runtime Contract

Both are depth preprocessors that produce a single-channel depth map rendered for ControlNet conditioning. ZoeDepth can reuse Transformers model coverage. LeReS remains a third-party detector path with model-specific preprocessing and depth normalization.

`controlnet_aux.ZoeDetector` defaults to `model_type="zoedepth"` and `ZoeD_M12_N.pt`, with a `ZoeDepthNK` option. It runs `model.infer`, normalizes depth by the 2nd and 85th percentiles, inverts the map (`1.0 - depth`), and optionally applies gamma correction.

`LeresDetector` defaults to `res101.pth` plus `latest_net_G.pth` for the Pix2Pix boosting path. `depth_leres++` is the registry alias with `boost=True`; `depth_leres` leaves boosting off. After estimation, the wrapper normalizes to 16-bit, converts to 8-bit, inverts, and applies optional threshold cutoffs `thr_a` and `thr_b`.

## Operators

- ZoeDepth: Transformers vision/depth model operators.
- LeReS: ResNeXt-style relative depth network, optional Pix2Pix boosting path, resize/normalize/threshold postprocess.

## DinoML Notes

ZoeDepth should compose the Transformers model report with an auxiliary output-image contract. LeReS can be documented as a third-party preprocessor now, but should stay external until the ResNeXt and Pix2Pix boosting graphs are admitted.

## Sources

- `H:/controlnet_aux/src/controlnet_aux/zoe/__init__.py`
- `H:/controlnet_aux/src/controlnet_aux/leres/__init__.py`
- `H:/controlnet_aux/src/controlnet_aux/processor.py`
- `transformers/src/transformers/models/zoedepth`
- `lllyasviel/stable-diffusion-webui-forge/extensions-builtin/forge_legacy_preprocessors/legacy_preprocessors/preprocessor.py:263`
- `lllyasviel/stable-diffusion-webui-forge/extensions-builtin/forge_legacy_preprocessors/legacy_preprocessors/preprocessor.py:577`
