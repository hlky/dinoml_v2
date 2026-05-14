# PP-LCNet Representative Config Snapshots

Fetched from Hugging Face Hub on 2026-05-13. Repos were public and not gated.

## PaddlePaddle/PP-LCNet_x1_0_doc_ori_safetensors

- Repo SHA: `b2716954d2e321dd4436400b37584fab641027ab`
- Hub metadata: Apache-2.0, `library_name=PaddleOCR`, `pipeline_tag=image-to-text`, `usedStorage=13539168`

```json
{
  "model_type": "pp_lcnet",
  "scale": 1.0,
  "reduction": 4,
  "hidden_dropout_prob": 0.2,
  "class_expand": 1280,
  "hidden_act": "hardswish",
  "id2label": {"0": "0", "1": "90", "2": "180", "3": "270"}
}
```

```json
{
  "image_processor_type": "PPLCNetImageProcessor",
  "do_resize": true,
  "resize_short": 256,
  "do_center_crop": true,
  "crop_size": 224,
  "do_rescale": true,
  "rescale_factor": 0.00392156862745098,
  "do_normalize": true,
  "image_mean": [0.406, 0.456, 0.485],
  "image_std": [0.225, 0.224, 0.229],
  "image_mode": "BGR",
  "channel_first": false
}
```

## PaddlePaddle/PP-LCNet_x1_0_table_cls_safetensors

- Repo SHA: `db46a3a25c8b3c88f86539fd28aceecc2a1b3ee1`
- Hub metadata: Apache-2.0, `library_name=PaddleOCR`, `pipeline_tag=image-to-text`, `usedStorage=6759480`

```json
{
  "model_type": "pp_lcnet",
  "scale": 1.0,
  "reduction": 4,
  "hidden_dropout_prob": 0.2,
  "class_expand": 1280,
  "hidden_act": "hardswish",
  "id2label": {"0": "wired_table", "1": "wireless_table"}
}
```

Processor settings match `PP-LCNet_x1_0_doc_ori_safetensors`: short edge resize to 256, center crop 224, rescale, normalize, RGB-to-BGR.

## PaddlePaddle/PP-LCNet_x0_25_textline_ori_safetensors

- Repo SHA: `1e6737131dedda1e87f7c01de171c20e81789c49`
- Hub metadata: Apache-2.0, `library_name=PaddleOCR`, `pipeline_tag=image-to-text`, `usedStorage=2005776`

```json
{
  "model_type": "pp_lcnet",
  "scale": 0.25,
  "reduction": 4,
  "hidden_dropout_prob": 0.2,
  "class_expand": 1280,
  "hidden_act": "hardswish",
  "block_configs": [
    [[3, 16, 32, 1, false]],
    [[3, 32, 64, [2, 1], false], [3, 64, 64, 1, false]],
    [[3, 64, 128, [2, 1], false], [3, 128, 128, 1, false]],
    [[3, 128, 256, [2, 1], false], [5, 256, 256, 1, false], [5, 256, 256, 1, false], [5, 256, 256, 1, false], [5, 256, 256, 1, false], [5, 256, 256, 1, false]],
    [[5, 256, 512, [2, 1], true], [5, 512, 512, 1, true]]
  ],
  "id2label": {"0": "0_degree", "1": "180_degree"}
}
```

```json
{
  "image_processor_type": "PPLCNetImageProcessor",
  "do_resize": true,
  "size": {"width": 160, "height": 80},
  "resize_short": null,
  "do_center_crop": false,
  "do_rescale": true,
  "rescale_factor": 0.00392156862745098,
  "do_normalize": true,
  "image_mean": [0.406, 0.456, 0.485],
  "image_std": [0.225, 0.224, 0.229],
  "image_mode": "BGR",
  "channel_first": false
}
```

## PaddlePaddle/PP-LCNet_x1_0_textline_ori_safetensors

Config matches the x0.25 text-line structure except `scale=1.0`; processor settings match x0.25 text-line.

```json
{
  "model_type": "pp_lcnet",
  "scale": 1.0,
  "reduction": 4,
  "hidden_dropout_prob": 0.2,
  "class_expand": 1280,
  "hidden_act": "hardswish",
  "block_configs": "same as x0.25 textline, including [2, 1] strides",
  "id2label": {"0": "0_degree", "1": "180_degree"}
}
```
