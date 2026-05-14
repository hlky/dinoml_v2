# Source Notes: pp_ocrv5_server_rec

## Pinned source

- Transformers checkout: `X:/H/transformers`
- Commit inspected: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`
- Model source directory: `src/transformers/models/pp_ocrv5_server_rec`
- Nested backbone source: `src/transformers/models/hgnet_v2`

The generated files in `pp_ocrv5_server_rec` state that future source edits should be made in
`modular_pp_ocrv5_server_rec.py`. For DinoML audit purposes, `modeling_pp_ocrv5_server_rec.py`
was also inspected because it contains the generated concrete implementation imported by users.

## Local source anchors

- `configuration_pp_ocrv5_server_rec.py`: config defaults, nested `hgnet_v2` backbone defaults,
  OCR head dimensions.
- `image_processing_pp_ocrv5_server_rec.py`: image resize/pad/normalize and CTC-style greedy
  text postprocess.
- `modeling_pp_ocrv5_server_rec.py`: concrete backbone wrapper, SVTR encoder, attention, MLP,
  classification head, and output ABI.
- `modular_pp_ocrv5_server_rec.py`: upstream modular source basis for generated files.
- `hgnet_v2/configuration_hgnet_v2.py` and `hgnet_v2/modeling_hgnet_v2.py`: nested CNN backbone
  config defaults and operator structure consumed by this recognizer.

## Hugging Face artifacts checked

- `https://huggingface.co/PaddlePaddle/PP-OCRv5_server_rec_safetensors/raw/main/config.json`
  - Transformers-native config: `model_type=pp_ocrv5_server_rec`, `hidden_size=120`,
    `depth=2`, `num_attention_heads=8`, `head_out_channels=18385`, nested
    `backbone_config.model_type=hgnet_v2`, `arch=L`.
- `https://huggingface.co/PaddlePaddle/PP-OCRv5_server_rec_safetensors/raw/main/preprocessor_config.json`
  - Image processor config: resize/pad height `48`, base width `320`, max width `3200`,
    character list length `18385`, first entry `blank`.
- `https://huggingface.co/PaddlePaddle/PP-OCRv5_server_rec/raw/main/config.json`
  - Paddle-style deployment metadata, not the Transformers model config. Useful for OCR pipeline
    provenance: `CTCLabelDecode`, `RecResizeImg`, TensorRT dynamic shape hints
    min/opt/max `1x3x48x160`, `1x3x48x320`, `8x3x48x3200`.
- `https://huggingface.co/PaddlePaddle/PP-OCRv5_server_rec/raw/main/preprocessor_config.json`
  - Not present at time of audit.

## Observed source facts

- Primary input ABI is `pixel_values` in NCHW image layout.
- The image processor emits `pixel_values`; it resizes every batch to the target width derived from
  the widest image in the batch, normalizes with ImageNet mean/std, and pads to width 320 only when
  the target width is below 320.
- The model body is CNN backbone (`HGNetV2Backbone`) -> average pool -> SVTR-style OCR encoder ->
  linear classifier -> softmax over vocabulary/classes.
- The recognizer head returns probabilities, not raw logits. Postprocess takes argmax over the last
  axis, removes adjacent duplicates, drops class id 0, indexes `character_list`, and reports the
  mean probability over retained positions.
- There is no autoregressive decode, beam search, tokenizer generation loop, or KV cache in the
  inspected in-library source.
- Attention is noncausal self-attention over the flattened spatial sequence inside the SVTR encoder.
- The SVTR encoder has a hard `squeeze(2)` after NCHW convs, so the height dimension at that point
  must be exactly 1.
