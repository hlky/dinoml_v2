# Pixio Config Snapshots

Audit date: 2026-05-13

Source checkout: `X:/H/transformers` at `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`.

## Public mirror: `LiheYoung/pixio-vith16`

Fetched from `https://huggingface.co/LiheYoung/pixio-vith16/raw/main/config.json`.

```json
{
  "apply_layernorm": true,
  "architectures": ["PixioModel"],
  "attention_probs_dropout_prob": 0.0,
  "drop_path_rate": 0.0,
  "dtype": "float32",
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.0,
  "hidden_size": 1280,
  "image_size": 256,
  "initializer_range": 0.02,
  "layer_norm_eps": 1e-06,
  "mlp_ratio": 4,
  "model_type": "pixio",
  "n_cls_tokens": 8,
  "num_attention_heads": 16,
  "num_channels": 3,
  "num_hidden_layers": 32,
  "out_features": ["stage32"],
  "out_indices": [32],
  "patch_size": 16,
  "qkv_bias": true,
  "reshape_hidden_states": true,
  "transformers_version": "5.0.0.dev0"
}
```

## Official gated repos

The following official repos were found via the Hugging Face plugin and are gated. Raw `config.json` requests returned access-restricted errors without accepted access:

| Model id | HF status | Repo metadata parameters | License tag |
|---|---:|---:|---|
| `facebook/pixio-vitb16` | gated | 85.9M | `fair-noncommercial-research-license` |
| `facebook/pixio-vitl16` | gated | 303.4M | `fair-noncommercial-research-license` |
| `facebook/pixio-vith16` | gated | 631.0M | `fair-noncommercial-research-license` |
| `facebook/pixio-vit1b16` | gated | 1361.5M | `fair-noncommercial-research-license` |
| `facebook/pixio-vit5b16` | gated | 5440.9M | `fair-noncommercial-research-license` |

## Converter size map

Source: `src/transformers/models/pixio/convert_pixio_to_pytorch.py`.

| Converter name | hidden_size | layers | heads | Inferred head_dim | Shared defaults |
|---|---:|---:|---:|---:|---|
| `pixio_vitb16` | 768 | 12 | 12 | 64 | patch 16, image 256, 8 CLS tokens, MLP ratio 4 |
| `pixio_vitl16` | 1024 | 24 | 16 | 64 | patch 16, image 256, 8 CLS tokens, MLP ratio 4 |
| `pixio_vith16` | 1280 | 32 | 16 | 80 | patch 16, image 256, 8 CLS tokens, MLP ratio 4 |
| `pixio_vit1b16` | 1536 | 48 | 24 | 64 | patch 16, image 256, 8 CLS tokens, MLP ratio 4 |
| `pixio_vit5b16` | 3072 | 48 | 32 | 96 | patch 16, image 256, 8 CLS tokens, MLP ratio 4 |

Note: the converter records original checkpoint QKV as one packed `qkv` matrix split in Q, K, V order. Its target names appear older than the generated `modeling_pixio.py` names in this checkout, so the audit uses it for source checkpoint layout and size-map context only.
