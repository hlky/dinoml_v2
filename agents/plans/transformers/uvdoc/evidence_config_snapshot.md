# UVDoc Config Evidence Snapshot

Source: `https://huggingface.co/PaddlePaddle/UVDoc_safetensors/resolve/main/config.json`

Fetched during audit on 2026-05-13.

Key fields:

```json
{
  "model_type": "uvdoc",
  "kernel_size": 5,
  "backbone_config": {
    "model_type": "uvdoc_backbone",
    "resnet_head": [[3, 32], [32, 32]],
    "resnet_configs": [
      [[32, 32, 1, false], [32, 32, 3, false], [32, 32, 3, false]],
      [[32, 64, 1, true], [64, 64, 3, false], [64, 64, 3, false], [64, 64, 3, false]],
      [[64, 128, 1, true], [128, 128, 3, false], [128, 128, 3, false], [128, 128, 3, false], [128, 128, 3, false], [128, 128, 3, false]]
    ],
    "stage_configs": [
      [[128, 1]],
      [[128, 2]],
      [[128, 5]],
      [[128, 8], [128, 3], [128, 2]],
      [[128, 12], [128, 7], [128, 4]],
      [[128, 18], [128, 12], [128, 6]]
    ],
    "out_features": ["stage1", "stage2", "stage3", "stage4", "stage5", "stage6"],
    "out_indices": [1, 2, 3, 4, 5, 6]
  },
  "bridge_connector": [128, 128],
  "out_point_positions2D": [[128, 32], [32, 2]],
  "dilation_values": [[1], [2], [5], [8, 3, 2], [12, 7, 4], [18, 12, 6]],
  "padding_mode": "reflect",
  "hidden_act": "prelu"
}
```

Preprocessor source: `https://huggingface.co/PaddlePaddle/UVDoc_safetensors/resolve/main/preprocessor_config.json`

```json
{
  "data_format": "channels_first",
  "do_rescale": true,
  "do_resize": true,
  "image_processor_type": "UVDocImageProcessor",
  "resample": 2,
  "rescale_factor": 0.00392156862745098,
  "size": {"height": 712, "width": 488}
}
```
