# Representative HF Config Sweep

Fetched from `https://huggingface.co/{model_id}/raw/main/config.json`,
`preprocessor_config.json`, and `https://huggingface.co/api/models/{model_id}` on
2026-05-13.

| Model id | Repo sha | Decoder | Hidden | Layers | Heads | Head dim | Experts | Part features | Out stage | Effective image/patch | Heatmap |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|---|---|
| [`usyd-community/vitpose-base-simple`](https://huggingface.co/usyd-community/vitpose-base-simple) | `a93ac0c...` | simple | 768 | 12 | 12 | 64 | 1 | 0 | `stage12` | 256x192 / 16x16 | 64x48 |
| [`usyd-community/vitpose-base`](https://huggingface.co/usyd-community/vitpose-base) | `95be299...` | classic | 768 | 12 | 12 | 64 | 1 | 0 | `stage12` | 256x192 / 16x16 | 64x48 |
| [`usyd-community/vitpose-base-coco-aic-mpii`](https://huggingface.co/usyd-community/vitpose-base-coco-aic-mpii) | `1c97abb...` | classic | 768 | 12 | 12 | 64 | 1 | 0 | `stage12` | 256x192 / 16x16 | 64x48 |
| [`usyd-community/vitpose-plus-small`](https://huggingface.co/usyd-community/vitpose-plus-small) | `0c30b65...` | classic | 384 | 12 | 12 | 32 | 6 | 96 | `stage12` | 256x192 / 16x16 | 64x48 |
| [`usyd-community/vitpose-plus-base`](https://huggingface.co/usyd-community/vitpose-plus-base) | `92be54d...` | classic | 768 | 12 | 12 | 64 | 6 | 192 | `stage12` | 256x192 / 16x16 | 64x48 |
| [`usyd-community/vitpose-plus-large`](https://huggingface.co/usyd-community/vitpose-plus-large) | `e211df3...` | classic | 1024 | 24 | 16 | 64 | 6 | 256 | `stage24` | 256x192 / 16x16 | 64x48 |
| [`usyd-community/vitpose-plus-huge`](https://huggingface.co/usyd-community/vitpose-plus-huge) | `9f36d7a...` | classic | 1280 | 32 | 16 | 80 | 6 | 320 | `stage32` | 256x192 / 16x16 | 64x48 |

Effective defaults in source when checkpoint config omits fields:

- `image_size=(256, 192)`, `patch_size=(16, 16)`, `num_channels=3`
- `hidden_size=768`, `num_hidden_layers=12`, `num_attention_heads=12`
- `mlp_ratio=4`, `hidden_act="gelu"`, `qkv_bias=true`
- `hidden_dropout_prob=0.0`, `attention_probs_dropout_prob=0.0`
- `layer_norm_eps=1e-12`, `initializer_range=0.02`
- `num_experts=1`, `part_features=256` at backbone-config default level; for non-plus converted checkpoints the config explicitly stores `part_features=0`, but the non-MoE path ignores it.
- top-level `scale_factor=4`, `use_simple_decoder=true` default; most non-simple checkpoint configs explicitly set `use_simple_decoder=false`.

All sampled processors match:

```json
{
  "image_processor_type": "VitPoseImageProcessor",
  "size": {"height": 256, "width": 192},
  "do_affine_transform": true,
  "normalize_factor": 200.0,
  "do_rescale": true,
  "rescale_factor": 0.00392156862745098,
  "do_normalize": true,
  "image_mean": [0.485, 0.456, 0.406],
  "image_std": [0.229, 0.224, 0.225]
}
```
