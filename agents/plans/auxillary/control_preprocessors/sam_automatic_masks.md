# SAM Automatic Masks

## Coverage

- Diffusers: not covered as a core model; SAM masks can be used as control, inpaint, or segmentation-conditioning inputs.
- Transformers: not covered by the inspected local Transformers tree as the Meta SAM implementation used here.
- Third-party/package: `controlnet_aux` includes a `SamDetector` wrapper over Segment Anything implementations.

## Runtime Contract

`SamDetector.from_pretrained` loads a SAM-family checkpoint by `model_type` and wraps it in `SamAutomaticMaskGenerator`. Supported model types in the local package include `vit_h`, `vit_l`, `vit_b`, and `vit_t`. The README shows standard SAM weights from `ybelkada/segment-anything` and MobileSAM weights from `dhkim2810/MobileSAM`.

At call time the wrapper normalizes the input into HWC RGB, resizes to `detect_resolution` (default 512), generates masks, sorts annotations by area, and renders a random-color RGB segmentation map. The random color rendering is part of the generated condition image, so reproducibility needs an explicit seed or a deterministic renderer if this is admitted.

## Operators

- ViT image encoder, prompt encoder, and mask decoder inside SAM.
- Image normalization using SAM pixel mean/std and square padding.
- Automatic mask generation, mask sorting by area, and RGB mask rasterization.

## DinoML Notes

SAM should be represented as an external segmentation preprocessor before attempting native graph coverage. The automatic mask generator is more than one forward pass; crop policy, stability thresholds, NMS, and color rendering belong in the preprocessor manifest.

## Sources

- `H:/controlnet_aux/src/controlnet_aux/segment_anything/__init__.py`
- `H:/controlnet_aux/src/controlnet_aux/segment_anything/modeling/sam.py`
- `H:/controlnet_aux/README.md`
