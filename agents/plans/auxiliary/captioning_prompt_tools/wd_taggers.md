# Waifu Diffusion Taggers

## Coverage

- Diffusers: not covered.
- Transformers: not covered for the ONNX tagger path.
- Third-party/UI: SD.Next uses SmilingWolf WD tagger ONNX models.

## Runtime Contract

SD.Next downloads `model.onnx` and `selected_tags.csv` for a selected WD model, resizes images to 448x448, runs ONNX inference, then applies category-specific thresholds for general, character, and rating tags before formatting prompt text.

## Operators

- ONNX image classifier, depending on selected backbone: EVA02, ViT, ConvNeXt, SwinV2, MOAT, etc.
- Sigmoid/probability extraction.
- CSV tag metadata, thresholding, max-tags, sorting, escaping.

## DinoML Notes

Start as an ONNX external-provider or CPU-side auxiliary. Native DinoML support would require per-backbone audits; do not assume one operator shape across all WD tagger variants.

## Sources

- `vladmandic/sdnext/modules/caption/waifudiffusion.py`
- `vladmandic/sdnext/modules/caption/caption.py`

