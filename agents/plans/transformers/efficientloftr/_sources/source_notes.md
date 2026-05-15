# EfficientLoFTR Source Notes

Scope: Transformers `efficientloftr` at local source checkout `transformers`, commit `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`.

## Local source files inspected

- `src/transformers/models/efficientloftr/configuration_efficientloftr.py`
- `src/transformers/models/efficientloftr/modeling_efficientloftr.py`
- `src/transformers/models/efficientloftr/modular_efficientloftr.py`
- `src/transformers/models/efficientloftr/image_processing_efficientloftr.py`
- `src/transformers/models/efficientloftr/image_processing_pil_efficientloftr.py`
- `src/transformers/models/efficientloftr/convert_efficientloftr_to_hf.py`
- `tests/models/efficientloftr/test_modeling_efficientloftr.py` for debug config shape constraints only.

Generated-file note: `modeling_efficientloftr.py`, `image_processing_efficientloftr.py`, and `image_processing_pil_efficientloftr.py` are generated from modular sources where applicable. `modular_efficientloftr.py` only owns the processor postprocess override; the modeling body is in the generated modeling file.

## HF configs and repo metadata checked

- Official native Transformers checkpoint: [zju-community/efficientloftr](https://huggingface.co/zju-community/efficientloftr), repo sha `face1a79050ffa3e9da28720d1cf93aaf2e8f421`, not gated, Apache-2.0, `model.safetensors`, 16,050,816 F32 parameters by HF repo metadata.
- Official `config.json`: [raw](https://huggingface.co/zju-community/efficientloftr/raw/main/config.json)
- Official `preprocessor_config.json`: [raw](https://huggingface.co/zju-community/efficientloftr/raw/main/preprocessor_config.json)
- Earlier native Transformers mirror: [stevenbucaille/efficientloftr](https://huggingface.co/stevenbucaille/efficientloftr), repo sha `9aec0b3da50cdf02656b5ac2162a670d0e2013af`, same model type and parameter count metadata, but its config has historical keys `stage_block_dims` / `stage_hidden_expansion` that current `EfficientLoFTRConfig` does not declare.
- Public non-native repositories found via HF model API and treated as out of scope for this native-source audit:
  - [xmanifold/efficient_loftr](https://huggingface.co/xmanifold/efficient_loftr): raw `eloftr_outdoor.ckpt`, no `config.json`.
  - [stevenbucaille/efficient_loftr_pth](https://huggingface.co/stevenbucaille/efficient_loftr_pth): raw `.pth`, no `config.json`.
  - [kornia/Efficient_LOFTR](https://huggingface.co/kornia/Efficient_LOFTR): raw `.ckpt`, no `config.json`.
  - [zahilaty/EfficientLoFTR-ONNX](https://huggingface.co/zahilaty/EfficientLoFTR-ONNX): ONNX artifact, empty API config.

## Key source observations

- Processor emits `pixel_values` as image pairs with shape `[B, 2, 3, H, W]`; default resize is `480x640`, rescale is `1/255`, grayscale conversion is enabled, and the model immediately selects one channel to produce `[2B, 1, H, W]`.
- Backbone is NCHW RepVGG-style stages with Conv2d + BatchNorm2d + activation, stage strides defaulting to `[2, 1, 2, 2]`, so default `480x640` produces a coarse map around `60x80`.
- Coarse transformer keeps image-like feature maps in NCHW outside attention. Each layer runs self aggregated attention on both images, then cross aggregated attention image0->image1 and image1->image0. Query aggregation is depthwise Conv2d with kernel/stride 4, KV aggregation is MaxPool2d with kernel/stride 4, then tensors permute to NHWC for LayerNorm and token attention.
- Attention is noncausal MHA by default: `hidden_size=256`, `num_attention_heads=8`, `num_key_value_heads=8`, `head_dim=32`, attention projection bias disabled. Current source sets KV head count equal to Q head count in config post-init, so GQA/MQA should be rejected unless source changes.
- 2D RoPE is generated from the coarse map after query aggregation. With default coarse `60x80` and aggregation `4`, attention token grid is `15x20`.
- Coarse matching computes all-pairs similarity between flattened coarse maps, divides by temperature, optionally applies dual softmax over both axes, thresholds, masks borders, enforces mutual nearest neighbors, and emits variable-length padded match indices/scores.
- Fine fusion upsamples coarse features through conv blocks and bilinear interpolation, unfolds local windows from both images, gathers windows by coarse match indices, runs two fine correlation stages, uses argmax for first-stage fine coordinates, and uses a 3x3 softmax + spatial expectation for second-stage refinement.
- Postprocess rescales normalized model keypoints to original image sizes using `target_sizes` `[B, 2, 2]`, filters by threshold and `matches > -1`, and returns variable-length per-pair records. No NMS is present.
