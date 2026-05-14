# PP-LCNetV3 Source Default Config Snapshot

Source basis: `X:/H/transformers/src/transformers/models/pp_lcnet_v3/configuration_pp_lcnet_v3.py` at commit `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`.

```json
{
  "model_type": "pp_lcnet_v3",
  "scale": 1.0,
  "stem_channels": 16,
  "stem_stride": 2,
  "reduction": 4,
  "divisor": 8,
  "hidden_act": "hardswish",
  "conv_symmetric_num": 4,
  "block_configs": [
    [[3, 16, 32, 1, false]],
    [[3, 32, 64, 2, false], [3, 64, 64, 1, false]],
    [[3, 64, 128, 2, false], [3, 128, 128, 1, false]],
    [[3, 128, 256, 2, false], [5, 256, 256, 1, false], [5, 256, 256, 1, false], [5, 256, 256, 1, false], [5, 256, 256, 1, false]],
    [[5, 256, 512, 2, true], [5, 512, 512, 1, true], [5, 512, 512, 1, false], [5, 512, 512, 1, false]]
  ],
  "stage_names": ["stem", "stage1", "stage2", "stage3", "stage4", "stage5"],
  "depths": [1, 2, 2, 5, 4]
}
```

Notes:
- `configuration_pp_lcnet_v3.py` is generated from `modular_pp_lcnet_v3.py`; future source edits should target the modular file.
- The installed local Python package could not import this checkout because its installed `huggingface_hub` lacks `is_offline_mode`, so this snapshot is source-derived rather than produced by `PPLCNetV3Config().to_dict()`.
