# OneFormer and UniFormer Segmentation

## Coverage

- Diffusers: not covered.
- Transformers: OneFormer is covered by `src/transformers/models/oneformer`; UniFormer is not present as a named Transformers model in the inspected tree.
- Third-party/UI: Forge and SD.Next expose semantic segmentation preprocessors.

## Runtime Contract

These preprocessors map an input image to a class-colored segmentation condition image. The output palette, class mapping, and resize policy are part of parity.

## Operators

- OneFormer: Transformers segmentation model operators plus postprocess.
- UniFormer: likely third-party semantic segmentation model; needs exact source/package audit.
- Argmax/class selection, palette lookup, resize.

## DinoML Notes

End-to-end output is structured semantic labels plus a rendered condition image. Keep the class palette and label mapping artifact-visible.

## Sources

- `transformers/src/transformers/models/oneformer`
- `transformers/src/transformers/pipelines/image_segmentation.py`
- `lllyasviel/stable-diffusion-webui-forge/extensions-builtin/forge_legacy_preprocessors/legacy_preprocessors/preprocessor.py:607`

