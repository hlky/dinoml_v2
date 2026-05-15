# VitPose Source Notes

Source basis:

- Transformers checkout: `transformers`
- Commit: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`
- Family source:
  - `src/transformers/models/vitpose/configuration_vitpose.py`
  - `src/transformers/models/vitpose/modeling_vitpose.py`
  - `src/transformers/models/vitpose/image_processing_vitpose.py`
  - `src/transformers/models/vitpose/image_processing_pil_vitpose.py`
  - `src/transformers/models/vitpose/convert_vitpose_to_hf.py`
  - `src/transformers/models/vitpose_backbone/configuration_vitpose_backbone.py`
  - `src/transformers/models/vitpose_backbone/modeling_vitpose_backbone.py`

Representative Hugging Face configs inspected on 2026-05-13:

| Model id | Repo sha | Gated | Config notes |
|---|---:|---:|---|
| [`usyd-community/vitpose-base-simple`](https://huggingface.co/usyd-community/vitpose-base-simple) | `a93ac0c67e0b7e2c55287d21d4c460c8f3c54d45` | false | base defaults, `use_simple_decoder=true`, output `stage12` |
| [`usyd-community/vitpose-base`](https://huggingface.co/usyd-community/vitpose-base) | `95be2991424e646950d656bb7fc15ec9be119700` | false | base defaults, classic deconv head, output `stage12` |
| [`usyd-community/vitpose-base-coco-aic-mpii`](https://huggingface.co/usyd-community/vitpose-base-coco-aic-mpii) | `1c97abb299207947767d028bf349c543ae06a49f` | false | base defaults, classic deconv head, multi-dataset checkpoint name but config has `num_experts` omitted/effective `1` |
| [`usyd-community/vitpose-plus-small`](https://huggingface.co/usyd-community/vitpose-plus-small) | `0c30b6534bb621af0162b481176742577264e36e` | false | `hidden_size=384`, `num_experts=6`, `part_features=96`, output `stage12` |
| [`usyd-community/vitpose-plus-base`](https://huggingface.co/usyd-community/vitpose-plus-base) | `92be54d7a29e42fad47b6e2ca01dd9e685a61e0d` | false | base defaults plus `num_experts=6`, `part_features=192`, output `stage12` |
| [`usyd-community/vitpose-plus-large`](https://huggingface.co/usyd-community/vitpose-plus-large) | `e211df377e49c89dba508ab3e83ddef13c0832b4` | false | `hidden_size=1024`, `layers=24`, `heads=16`, `num_experts=6`, `part_features` omitted/effective `256`, output `stage24` |
| [`usyd-community/vitpose-plus-huge`](https://huggingface.co/usyd-community/vitpose-plus-huge) | `9f36d7aec1800d23e97f10c2e74393aee92aa53f` | false | `hidden_size=1280`, `layers=32`, `heads=16`, `num_experts=6`, `part_features=320`, output `stage32` |

All sampled `preprocessor_config.json` files:

- `image_processor_type="VitPoseImageProcessor"`
- `size={"height": 256, "width": 192}`
- `do_affine_transform=true`
- `normalize_factor=200.0`
- `do_rescale=true`, `rescale_factor=1/255`
- `do_normalize=true`, ImageNet mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`

Source observations:

- The top-level pose model loads a backbone via `load_backbone(config)`, consumes the last selected `feature_maps` entry, permutes `[B, S, C] -> [B, C, S]`, then reshapes to `[B, C, image_h // patch_h, image_w // patch_w]`.
- The custom VitPose backbone is ViT-like but uses patch embedding `Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size, padding=2)`. This is not the usual padding-free ViT patchify rewrite.
- The backbone adds position embeddings as `patch + position_embeddings[:, 1:] + position_embeddings[:, :1]`; there is no runtime interpolation path in the inspected source.
- Attention is noncausal full self-attention over image patch tokens. `_supports_sdpa` and `_supports_flash_attn` are true, but the source fallback uses ordinary Q/K/V linear projections and scaled dot-product softmax with no mask.
- `VitPose+` uses a naive MoE MLP when `num_experts > 1`: one shared `fc1`, a shared `fc2` producing `hidden_size - part_features`, and one expert linear per dataset producing `part_features`, selected by `dataset_index`.
- Simple head: `ReLU -> bilinear Upsample(scale_factor=config.scale_factor, align_corners=False) -> Conv2d(C, num_labels, 3x3, padding=1)`.
- Classic head: two `ConvTranspose2d(..., kernel=4, stride=2, padding=1, bias=False) -> BatchNorm2d -> ReLU` blocks, then `Conv2d(256, num_labels, 1x1)`.
- Optional `flip_pairs` are applied inside the head by swapping left/right keypoint channels and horizontal flipping heatmaps. Combined target support exists in helper signature but the pose head emits standard gaussian heatmaps.
- Postprocess is processor-owned CPU/Numpy logic: argmax over heatmaps, DARK refinement with SciPy gaussian filter/log/Hessian inverse, and affine inverse coordinate mapping from heatmap coordinates to original image box coordinates.
- No sampled official configs were gated. No model execution or imports were performed for this audit.
