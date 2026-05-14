# PPFormulaNet Source Notes

Audit target: Transformers model family `pp_formulanet`.

## Pinned source

- Local Transformers checkout: `X:/H/transformers`
- Commit inspected: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`
- Source directory: `X:/H/transformers/src/transformers/models/pp_formulanet`
- Generated runtime files inspected:
  - `configuration_pp_formulanet.py`
  - `modeling_pp_formulanet.py`
  - `image_processing_pp_formulanet.py`
  - `processing_pp_formulanet.py`
  - `__init__.py`
- Future edit authority: `modular_pp_formulanet.py`; generated files state they are generated from this modular file.

## Hugging Face files fetched

Fetched from `https://huggingface.co/{model_id}/resolve/main/{file}` on 2026-05-13:

- `PaddlePaddle/PP-FormulaNet-L_safetensors`
  - `config.json`
  - `generation_config.json`
  - `processor_config.json`
  - `tokenizer_config.json`
- `PaddlePaddle/PP-FormulaNet_plus-L_safetensors`
  - `config.json`
  - `generation_config.json`
  - `processor_config.json`
  - `tokenizer_config.json`
- `PaddlePaddle/PP-FormulaNet-S`
  - `config.json` only; this is a PaddleOCR deployment-style config with preprocessing/postprocessing and embedded tokenizer metadata, not a native Transformers `pp_formulanet` config.
- `PaddlePaddle/PP-FormulaNet_plus-M`
  - `config.json` only; same PaddleOCR deployment-style caveat.
- `PaddlePaddle/PP-FormulaNet_plus-S`
  - `config.json` only; same PaddleOCR deployment-style caveat.

The fetched PaddleOCR deployment configs are large because they embed tokenizer/dictionary metadata. They are useful as evidence of external deployment variants, but this report does not treat them as native `PPFormulaNetConfig` checkpoints.

## Key source facts used

- Processor emits only `pixel_values`; there is no text prompt or placeholder-token image stitch for the primary image-to-LaTeX path.
- Image processor defaults: crop margin, resize, thumbnail, center pad, rescale, normalize, output size `768x768`, mean/std `[0.7931] * 3` / `[0.1738] * 3`, channels-first tensor output.
- Vision encoder consumes `pixel_values` as `[B, 3, H, W]`, requires `H == W == vision_config.image_size`, applies patch Conv2d with `kernel=stride=patch_size`, then immediately uses `[B, Hpatch, Wpatch, C]` hidden maps.
- Vision encoder layers are SAM/SLANeXt-like: LayerNorm over channels-last maps, windowed or global dense self-attention with decomposed relative position bias, MLP, residuals.
- Vision neck converts channels-last maps to NCHW for Conv2d + channels-first LayerNorm + Conv2d + channels-first LayerNorm.
- Multi-modal projector consumes neck NCHW maps, applies two stride-2 Conv2d layers, flattens spatial positions to a sequence, and applies two Linear layers to decoder hidden width.
- Text side is MBart-like decoder only: token embedding with optional scale, learned positions with offset 2, causal self-attention, encoder-decoder cross-attention, GELU FFN, final LayerNorm, untied LM head.
- Generation uses encoder/projector output as `encoder_hidden_states`; pixel values should be supplied only on the first generation iteration when cache is enabled.
- `PPFormulaNetForConditionalGeneration.get_encoder()` returns `self.model.get_encoder()`, but `PPFormulaNetModel` in the inspected generated file does not define `get_encoder()`. Treat this as a source gap to verify before depending on generic encoder-decoder generation helpers.
