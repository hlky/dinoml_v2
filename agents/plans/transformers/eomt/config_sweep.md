# EoMT Representative Config Sweep

All values below come from public Hugging Face `config.json` or `preprocessor_config.json` files read on 2026-05-13. `num_labels` is inferred from `id2label` entry count; model output class width is `num_labels + 1` because source appends a null/no-object class.

| Model id | Task flavor | hidden | layers | heads | head dim | image | patch | patch grid | input seq before queries | queries | final query blocks | labels | MLP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `tue-mps/coco_panoptic_eomt_small_640_2x` | panoptic | 384 | 12 | 6 | 64 | 640 | 16 | 40x40 | 1605 | 200 | 3 | 133 | GELU MLP |
| `tue-mps/coco_panoptic_eomt_base_640_2x` | panoptic | 768 | 12 | 12 | 64 | 640 | 16 | 40x40 | 1605 | 200 | 3 | 133 | GELU MLP |
| `tue-mps/coco_panoptic_eomt_large_640` | panoptic | 1024 | 24 | 16 | 64 | 640 | 16 | 40x40 | 1605 | 200 | 4 | 133 | GELU MLP |
| `tue-mps/coco_instance_eomt_large_640` | instance | 1024 | 24 | 16 | 64 | 640 | 16 | 40x40 | 1605 | 200 | 4 | 80 | GELU MLP |
| `tue-mps/ade20k_semantic_eomt_large_512` | semantic | 1024 | 24 | 16 | 64 | 512 | 16 | 32x32 | 1029 | 100 | 4 | 150 | GELU MLP |
| `tue-mps/cityscapes_semantic_eomt_large_1024` | semantic | 1024 | 24 | 16 | 64 | 1024 | 16 | 64x64 | 4101 | 100 | 4 | 19 | GELU MLP |
| `tue-mps/coco_panoptic_eomt_large_1280` | panoptic | 1024 | 24 | 16 | 64 | 1280 | 16 | 80x80 | 6405 | 200 | 4 | 133 | GELU MLP |
| `tue-mps/coco_panoptic_eomt_giant_640` | panoptic | 1536 | 40 | 24 | 64 | 640 | 16 | 40x40 | 1605 | 200 | 5 | 133 | SwiGLU |
| `tue-mps/coco_panoptic_eomt_7b_640` | panoptic | 4096 | 32 | 32 | 128 | 640 | 16 | 40x40 | 1605 | 200 | 5 | 133 | SwiGLU |

Processor variants:

| Model id | size | `do_split_image` | `do_pad` | output layout |
|---|---|---:|---:|---|
| `tue-mps/coco_panoptic_eomt_large_640` | shortest=640, longest=640 | false | true | channels_first |
| `tue-mps/coco_instance_eomt_large_640` | shortest=640, longest=640 | false | true | channels_first |
| `tue-mps/ade20k_semantic_eomt_large_512` | shortest=512, longest=null | true | false | channels_first |
| `tue-mps/coco_panoptic_eomt_giant_640` | shortest=640, longest=640 | false | true | channels_first |
