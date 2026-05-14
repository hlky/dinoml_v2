# Representative HF Config Snapshots

These are trimmed snapshots of fetched Hugging Face configs that carry a nested `vitdet` backbone. They are not full repo mirrors.

## `hustvl/vitmatte-base-composition-1k`

Source: https://huggingface.co/hustvl/vitmatte-base-composition-1k/raw/main/config.json

```json
{
  "architectures": ["VitMatteForImageMatting"],
  "model_type": "vitmatte",
  "hidden_size": 768,
  "torch_dtype": "float32",
  "backbone_config": {
    "image_size": 512,
    "model_type": "vitdet",
    "num_channels": 4,
    "out_features": ["stage12"],
    "out_indices": [12],
    "residual_block_indices": [2, 5, 8, 11],
    "use_relative_position_embeddings": true,
    "window_block_indices": [0, 1, 3, 4, 6, 7, 9, 10],
    "window_size": 14
  }
}
```

Effective VitDet defaults from `VitDetConfig` for omitted backbone fields include `hidden_size=768`, `num_hidden_layers=12`, `num_attention_heads=12`, `mlp_ratio=4`, `patch_size=16`, `qkv_bias=true`, `use_absolute_position_embeddings=true`, and `layer_norm_eps=1e-6`.

## `hustvl/vitmatte-small-composition-1k`

Source: https://huggingface.co/hustvl/vitmatte-small-composition-1k/raw/main/config.json

```json
{
  "architectures": ["VitMatteForImageMatting"],
  "model_type": "vitmatte",
  "hidden_size": 384,
  "torch_dtype": "float32",
  "backbone_config": {
    "hidden_size": 384,
    "image_size": 512,
    "model_type": "vitdet",
    "num_attention_heads": 6,
    "num_channels": 4,
    "out_features": ["stage12"],
    "out_indices": [12],
    "residual_block_indices": [2, 5, 8, 11],
    "use_relative_position_embeddings": true,
    "window_block_indices": [0, 1, 3, 4, 6, 7, 9, 10],
    "window_size": 14
  }
}
```

Effective VitDet defaults from `VitDetConfig` for omitted backbone fields include `num_hidden_layers=12`, `mlp_ratio=4`, `patch_size=16`, `qkv_bias=true`, `use_absolute_position_embeddings=true`, and `layer_norm_eps=1e-6`.

## Gated or missing config URLs

- https://huggingface.co/google/vitdet-base-patch16-224
- https://huggingface.co/google/vitdet-large-patch16-224
- https://huggingface.co/hustvl/vitmatte-large-composition-1k

Access to gated repos would resolve whether public standalone base/large checkpoint configs add fields beyond the source defaults. The source config class and accessible nested VitMatte configs are sufficient for operator/layout auditing.

