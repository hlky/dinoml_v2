# SwiftFormer config snapshot

Fetched from Hugging Face model repos on 2026-05-13. Source basis is
Transformers checkout `X:/H/transformers` at
`b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`.

| Model id | Architectures | Depths | Embed dims | MLP ratio | Labels | Dtype | Weights listed |
|---|---|---:|---:|---:|---:|---|---|
| `MBZUAI/swiftformer-xs` | `SwiftFormerForImageClassification` | `3/3/6/4` | `48/56/112/220` | 4 | 1000 from `id2label` | `float32` | `model.safetensors`, `pytorch_model.bin` |
| `MBZUAI/swiftformer-s` | `SwiftFormerForImageClassification` | `3/3/9/6` | `48/64/168/224` | 4 | 1000 from `id2label` | `float32` | `pytorch_model.bin` |
| `MBZUAI/swiftformer-l1` | `SwiftFormerForImageClassification` | `4/3/10/5` | `48/96/192/384` | 4 | 1000 from `id2label` | `float32` | `pytorch_model.bin` |
| `MBZUAI/swiftformer-l3` | `SwiftFormerForImageClassification` | `4/4/12/6` | `64/128/320/512` | 4 | 1000 from `id2label` | `float32` | `pytorch_model.bin` |

All four checkpoint `config.json` files include:

- `model_type: "swiftformer"`
- `downsamples: [true, true, true, true]`
- `down_patch_size: 3`, `down_stride: 2`, `down_pad: 1`
- `drop_path_rate: 0.0`
- `hidden_act: "gelu"`
- `batch_norm_eps: 1e-5`
- `use_layer_scale: true`
- `layer_scale_init_value: 1e-5`

Fields omitted from the fetched checkpoint configs but supplied by
`SwiftFormerConfig` defaults:

- `image_size: 224`
- `drop_mlp_rate: 0.0`
- `drop_conv_encoder_rate: 0.0`

All four `preprocessor_config.json` files include:

- `do_resize: true`
- `size: 224`
- `do_normalize: true`
- `image_mean: [0.485, 0.456, 0.406]`
- `image_std: [0.229, 0.224, 0.225]`

The preprocessor configs omit `do_rescale`, `rescale_factor`, and
`image_processor_type`; `AutoImageProcessor` maps `swiftformer` to
`ViTImageProcessor`, whose class defaults include resize to 224x224,
rescale by `1/255`, and ImageNet normalization.

No gated or 401 links were encountered. `model.safetensors.index.json` returned
404 for the sampled repos; `swiftformer-xs` has a single `model.safetensors`,
while the other sampled repos list only `pytorch_model.bin`.
