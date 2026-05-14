# Vivit HF config summaries

Fetched from Hugging Face raw URLs on 2026-05-13. These are selected operator-relevant fields, not full config copies.

## Official Google checkpoints

Links:

- [google/vivit-b-16x2 config](https://huggingface.co/google/vivit-b-16x2/blob/main/config.json)
- [google/vivit-b-16x2 preprocessor](https://huggingface.co/google/vivit-b-16x2/blob/main/preprocessor_config.json)
- [google/vivit-b-16x2-kinetics400 config](https://huggingface.co/google/vivit-b-16x2-kinetics400/blob/main/config.json)
- [google/vivit-b-16x2-kinetics400 preprocessor](https://huggingface.co/google/vivit-b-16x2-kinetics400/blob/main/preprocessor_config.json)

| Model id | Source type | Architectures | Hidden/layers/heads | Frames/image | Tubelet | Labels | Processor notable fields |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `google/vivit-b-16x2` | official config | `VivitForVideoClassification` | 768 / 12 / 12 | `num_frames=32`, `image_size=224` | `[2,16,16]` | 400 Kinetics labels | shortest edge 256, crop 224, rescale 1/127.5, `do_normalize=false`, historical `do_zero_centering=true` |
| `google/vivit-b-16x2-kinetics400` | official config | `ViViTForVideoClassification` historical casing | 768 / 12 / 12 | historical `video_size=[32,224,224]`; omits current `num_frames` and `image_size` fields | `[2,16,16]` | 400 placeholder labels | shortest edge 224, crop 224, rescale 1/127.5, `do_normalize=true`, mean/std `[0.5,0.5,0.5]` |

Current `VivitConfig` supplies omitted `num_frames=32` and `image_size=224`. Current `VivitImageProcessor` reads `offset`, not the historical `do_zero_centering` key.

## Open community fine-tunes used only for variation sweep

These are not official architecture sources; they are useful for seeing config drift around heads and legacy fields.

| Model id | Link | Hidden/layers/heads | Frames/image | Tubelet | Labels | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| `NiiCole/vivit-b-16x2-kinetics400-finetuned-ucf101-subset` | [config](https://huggingface.co/NiiCole/vivit-b-16x2-kinetics400-finetuned-ucf101-subset/blob/main/config.json) | 768 / 12 / 12 | `num_frames=32`, `image_size=224`, legacy `video_size=[32,224,224]` | `[2,16,16]` | 10 | same body, smaller classifier |
| `prathameshdalal/vivit-b-16x2-kinetics400-UCF-Crime` | [config](https://huggingface.co/prathameshdalal/vivit-b-16x2-kinetics400-UCF-Crime/blob/main/config.json) | 768 / 12 / 12 | `num_frames=32`, `image_size=224`, legacy `video_size=[32,224,224]` | `[2,16,16]` | 14 | same body, smaller classifier |
| `Arekku21/vivit-b-16x2-kinetics400-finetuned-MSL` | [config](https://huggingface.co/Arekku21/vivit-b-16x2-kinetics400-finetuned-MSL/blob/main/config.json) | 768 / 12 / 12 | `num_frames=32`, `image_size=224`, legacy `video_size=[32,224,224]` | `[2,16,16]` | 3 | same body, smaller classifier |

`juliendenize/COMEDIAN-ViViT-tiny` appeared in model search metadata but `config.json` returned 404, so it is not used for runtime facts.
