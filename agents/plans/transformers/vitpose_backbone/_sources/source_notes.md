# Source notes: vitpose_backbone

Audit date: 2026-05-13.

Scope: `vitpose_backbone` only. Files were written only under `agents/plans/transformers/vitpose_backbone/`.

## Local source basis

- Transformers checkout: `transformers`
- Commit inspected: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`
- Primary source:
  - `src/transformers/models/vitpose_backbone/configuration_vitpose_backbone.py`
  - `src/transformers/models/vitpose_backbone/modeling_vitpose_backbone.py`
- Related consumer/source context:
  - `src/transformers/models/vitpose/configuration_vitpose.py`
  - `src/transformers/models/vitpose/modeling_vitpose.py`
  - `src/transformers/models/vitpose/image_processing_vitpose.py`
  - `src/transformers/backbone_utils.py`

## Key line anchors

- `modeling_vitpose_backbone.py:17`: source docstring names the two differences from ViT: patch embedding `padding=2` and MoE MLP.
- `modeling_vitpose_backbone.py:61`: `nn.Conv2d(..., kernel_size=patch_size, stride=patch_size, padding=2)`.
- `modeling_vitpose_backbone.py:85` and `:92`: position table `[1,num_patches+1,C]`, added as `pos[:,1:] + pos[:,:1]`.
- `modeling_vitpose_backbone.py:146-160`: separate Q/K/V Linear layers and `[B,S,H,D] -> [B,H,S,D]` transpose.
- `modeling_vitpose_backbone.py:162-171`: attention dispatch through `ALL_ATTENTION_FUNCTIONS`, eager fallback receives `attention_mask=None`.
- `modeling_vitpose_backbone.py:219-237`: naive MoE computes each expert and masks by `indices == i`.
- `modeling_vitpose_backbone.py:241-266`: MoE MLP shared projection plus expert tail concat.
- `modeling_vitpose_backbone.py:301-307`: missing `dataset_index` rejects when `num_experts > 1`.
- `modeling_vitpose_backbone.py:429-433`: selected stages get final LayerNorm before entering `feature_maps`.
- `configuration_vitpose_backbone.py:47-61`: source defaults for image/patch size, hidden dimensions, MoE, activation, qkv bias.
- `configuration_vitpose_backbone.py:66-68`: `stage_names` are generated from layer count and output features/indices are aligned.
- `modeling_vitpose.py:261-266`: parent pose model reshapes selected sequence feature to `[B,C,H_patch,W_patch]`.
- `backbone_utils.py:65-70` and `:102-117`: default final-stage selection, negative index normalization, duplicate/order validation.

## Representative config snapshot summary

Fetched via public Hugging Face `resolve/main/config.json` and `preprocessor_config.json` URLs; no model weights were downloaded and no code was executed.

- `usyd-community/vitpose-base-simple`
  - Raw JSON backbone fields: `model_type=vitpose_backbone`, `out_indices=[12]`, `out_features=["stage12"]`, `part_features=0`.
  - Effective source defaults: `hidden_size=768`, `num_hidden_layers=12`, `num_attention_heads=12`, `num_experts=1`.
  - Parent head: simple decoder.
- `usyd-community/vitpose-base`
  - Same backbone fields/effective defaults as base-simple.
  - Parent head: classic decoder.
- `usyd-community/vitpose-base-coco-aic-mpii`
  - Same backbone fields/effective defaults as base.
- `usyd-community/vitpose-plus-small`
  - Raw JSON backbone fields: `hidden_size=384`, `num_experts=6`, `part_features=96`, `out_indices=[12]`.
  - Effective defaults: `num_hidden_layers=12`, `num_attention_heads=12`.
- `usyd-community/vitpose-plus-base`
  - Raw JSON backbone fields: `num_experts=6`, `part_features=192`, `out_indices=[12]`.
  - Effective defaults: `hidden_size=768`, `num_hidden_layers=12`, `num_attention_heads=12`.
- `usyd-community/vitpose-plus-large`
  - Raw JSON backbone fields: `hidden_size=1024`, `num_attention_heads=16`, `num_hidden_layers=24`, `num_experts=6`, `out_indices=[24]`.
  - Effective default: `part_features=256`.
- `usyd-community/vitpose-plus-huge`
  - Raw JSON backbone fields: `hidden_size=1280`, `num_attention_heads=16`, `num_hidden_layers=32`, `num_experts=6`, `part_features=320`, `out_indices=[32]`.
- `hf-internal-testing/tiny-random-VitPoseForPoseEstimation`
  - Raw JSON backbone fields: `hidden_size=16`, `num_attention_heads=2`, `num_experts=2`, `part_features=10`, `out_indices=[12]`.
  - Effective default: `num_hidden_layers=12`.

Preprocessor configs checked for base-simple, plus-small, and plus-large all reported:

```text
image_processor_type = VitPoseImageProcessor
size = {height: 256, width: 192}
do_affine_transform = true
do_rescale = true
rescale_factor = 0.00392156862745098
do_normalize = true
image_mean = [0.485, 0.456, 0.406]
image_std = [0.229, 0.224, 0.225]
normalize_factor = 200.0
```

## Gated/config gaps to keep visible

- Raw checkpoint configs omit many effective defaults; integration should serialize resolved config values in DinoML artifacts.
- `plus-large` omits `part_features`; the current source default of `256` matches converter intent but should be treated as source-derived, not raw-config-derived.
- `VitPoseBackbonePatchEmbeddings` computes `num_patches` from integer division of `image_size` by `patch_size`, while the actual Conv2d uses `padding=2`; non-default shapes should be guarded so convolution grid and position length agree.
- MoE path has no learned router. `dataset_index` is an input ABI and must be validated before optimized expert dispatch.
- Parent `feature_maps` naming can mislead: backbone outputs sequences `[B,S,C]`; image-like `[B,C,Hp,Wp]` is created in `VitPoseForPoseEstimation`.
- `output_attentions=True` may require backend-specific attention probability materialization; first backbone parity can defer it with a clear rejection.
