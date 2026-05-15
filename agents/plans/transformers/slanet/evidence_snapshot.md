# SLANet Evidence Snapshot

Source basis:

- Transformers checkout: `transformers`
- Commit: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`
- Primary generated source: `src/transformers/models/slanet/modeling_slanet.py`
- Authoritative source for future Transformers edits: `src/transformers/models/slanet/modular_slanet.py`

Representative public HF repos found through the HF model API:

| Model id | HF repo sha | Files observed | Notes |
|---|---:|---|---|
| `PaddlePaddle/SLANet_plus_safetensors` | `e44d12b3fe695792170dd46623075f63d8544461` | `config.json`, `preprocessor_config.json`, `inference.yml`, `model.safetensors` | Open, not gated. |
| `PaddlePaddle/SLANet_safetensors` | `1f7450d2a77bf1831125e9ce0c87d3e60bfd1629` | `config.json`, `preprocessor_config.json`, `inference.yml`, `model.safetensors` | Open, not gated. |

Config fields identical in both public checkpoints:

```json
{
  "model_type": "slanet",
  "post_conv_out_channels": 96,
  "out_channels": 50,
  "hidden_size": 256,
  "max_text_length": 500,
  "backbone_config": {
    "model_type": "pp_lcnet",
    "scale": 1,
    "out_features": ["stage2", "stage3", "stage4", "stage5"],
    "out_indices": [2, 3, 4, 5]
  }
}
```

Preprocessor fields identical in both public checkpoints:

```json
{
  "image_processor_type": "SLANeXtImageProcessor",
  "do_resize": true,
  "size": {"height": 488, "width": 488},
  "pad_size": {"height": 488, "width": 488},
  "do_normalize": true,
  "image_mean": [0.485, 0.456, 0.406],
  "image_std": [0.229, 0.224, 0.225],
  "do_pad": true
}
```

Safetensors header checks, both repos:

| Tensor | dtype | shape |
|---|---|---:|
| `backbone.vision_backbone.encoder.convolution.convolution.weight` | `F32` | `[16, 3, 3, 3]` |
| `backbone.post_csp_pan.channel_projector.0.convolution.weight` | `F32` | `[96, 64, 1, 1]` |
| `backbone.post_csp_pan.downsamples.0.depthwise_convolution.convolution.weight` | `F32` | `[96, 1, 5, 5]` |
| `head.structure_attention_cell.rnn.weight_ih` | `F32` | `[768, 146]` |
| `head.structure_generator.fc2.weight` | `F32` | `[50, 256]` |

`inference.yml` notes:

- `SLANet_plus_safetensors` TensorRT dynamic-shape examples: `[1,3,32,32]`, `[1,3,64,448]`, `[1,3,488,488]`.
- `SLANet_safetensors` TensorRT dynamic-shape examples: `[1,3,32,32]`, `[1,3,64,448]`, `[8,3,488,488]`.
- Paddle preprocessing lists BGR decode and `ToCHWImage`; HF processor source emits `pixel_values` as channel-first tensors but defaults to RGB conversion through the Transformers image backend.
