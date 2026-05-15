# timm_backbone source notes

Audit target: `timm_backbone`

Transformers checkout: `transformers`

Pinned commit inspected: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`

## Local source files

- `src/transformers/models/timm_backbone/configuration_timm_backbone.py`
  - `TimmBackboneConfig` is `@strict`, `model_type = "timm_backbone"`.
  - Source-declared fields: `backbone`, `num_channels`, `features_only`, `_out_indices`, `freeze_batch_norm_2d`, `output_stride`.
  - `__post_init__` defaults `out_indices` to `[-1]`.
- `src/transformers/models/timm_backbone/modeling_timm_backbone.py`
  - `TimmBackbone.__init__` requires the `timm` backend and raises if `config.backbone is None`.
  - Neural body is delegated through `timm.create_model(config.backbone, pretrained=..., features_only=..., in_chans=..., out_indices=..., output_stride=..., **kwargs)`.
  - Forward accepts `pixel_values` and optional output flags, rejects `output_attentions=True`, and returns `BackboneOutput(feature_maps=tuple(...), hidden_states=..., attentions=None)`.
- `src/transformers/backbone_utils.py`
  - `BackboneMixin._init_timm_backbone` derives `stage_names`, `num_features`, `out_features`, and `out_indices` from `backbone.feature_info`.
  - `BackboneConfigMixin` validates `out_features`/`out_indices` ordering, uniqueness, and stage membership.
  - `consolidate_backbone_kwargs_to_config` routes non-Hub backbone strings to `TimmBackboneConfig`; Hub repo ids are loaded as normal HF configs.
- `src/transformers/models/auto/auto_factory.py`
  - `AutoBackbone.from_pretrained` treats a non-Hub string such as `resnet18` as a timm backbone name.
  - For timm fallback, `out_features` and `output_loading_info=True` are rejected; `from_pretrained` always passes `pretrained=True`.
- `src/transformers/dependency_versions_table.py`
  - timm dependency floor at this commit: `timm>=1.0.23`.

## Source links

- [modeling_timm_backbone.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/timm_backbone/modeling_timm_backbone.py)
- [configuration_timm_backbone.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/timm_backbone/configuration_timm_backbone.py)
- [backbone_utils.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/backbone_utils.py)
- [auto_factory.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/auto/auto_factory.py)
- [dependency_versions_table.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/dependency_versions_table.py)

## Representative Hub configs checked

Open Hub configs:

- [Noah-Wang/eva02-ai-art-detector config.json](https://huggingface.co/Noah-Wang/eva02-ai-art-detector/raw/main/config.json)
  - Tags include `timm_backbone`; `library_name` is `timm`.
  - Config has `model_type: "timm_backbone"` but uses timm-style keys such as `architecture`, `num_classes`, `num_features`, `global_pool`, and `pretrained_cfg`.
  - It does not include current in-library `backbone`.
  - No `preprocessor_config.json` found at main revision.
- [Noah-Wang/eva02-ai-art-detector-prod config.json](https://huggingface.co/Noah-Wang/eva02-ai-art-detector-prod/raw/main/config.json)
  - Same pattern as above, with `num_classes: 9`.
  - No `preprocessor_config.json` found at main revision.
- [scizzum/model_10_22_run config.json](https://huggingface.co/scizzum/model_10_22_run/raw/main/config.json)
  - `architectures: ["TimmBackbone"]`, `model_type: "timm_backbone"`, `timm_model_name: "eva_giant_patch14_224.clip_ft_in1k"`.
  - Contains EVA-like structural fields (`image_size`, `patch_size`, `hidden_size`, `num_hidden_layers`, `num_attention_heads`) but still omits current source `backbone`.
  - [preprocessor_config.json](https://huggingface.co/scizzum/model_10_22_run/raw/main/preprocessor_config.json) exists: resize/rescale/normalize to 224 with CLIP-style mean/std.
- [omlab/omdet-turbo-swin-tiny-hf config.json](https://huggingface.co/omlab/omdet-turbo-swin-tiny-hf/raw/main/config.json)
  - Composite model config, not `model_type=timm_backbone`.
  - Uses `use_timm_backbone: true`, `backbone: "swin_tiny_patch4_window7_224"`, and `backbone_kwargs` including `out_indices: [1,2,3]`, `img_size: 640`, `always_partition: true`.
  - In current source, `OmDetTurboConfig` converts these to `TimmBackboneConfig` plus separately forwarded `timm_kwargs`.
  - [preprocessor_config.json](https://huggingface.co/omlab/omdet-turbo-swin-tiny-hf/raw/main/preprocessor_config.json) uses `DetrImageProcessor`, size 640x640, ImageNet mean/std in 0-255 scale, `do_rescale=false`.
- [facebook/detr-resnet-50 config.json](https://huggingface.co/facebook/detr-resnet-50/raw/main/config.json)
  - Historical composite DETR config with `backbone: "resnet50"` but no explicit `use_timm_backbone`; current config source defaults `use_timm_backbone=True` unless routed otherwise.
  - Preprocessor uses `DetrImageProcessor`, shortest edge 800 / longest edge 1333, ImageNet mean/std.
- [facebook/maskformer-swin-base-ade config.json](https://huggingface.co/facebook/maskformer-swin-base-ade/raw/main/config.json)
  - Composite config has a native Transformers `SwinConfig` under `backbone_config`; this is a useful contrast case and should not be admitted as `timm_backbone`.
  - Preprocessor uses legacy `MaskFormerFeatureExtractor`, size 640, `size_divisibility: 32`.

## Key audit inferences

- `timm_backbone` itself owns wrapper dispatch, feature-map ABI, output filtering, batch-norm freeze toggling, and AutoBackbone routing. It does not own the neural operator surface for ResNet, Swin, EVA, ConvNeXt, EfficientNet, or other timm bodies.
- Current source does not provide a family image processor. Preprocessing comes from the parent model processor or from timm/HF repo metadata.
- Config fields such as `architecture`, `timm_model_name`, `patch_size`, `hidden_size`, or `num_hidden_layers` seen in Hub timm-style configs are not read by current `TimmBackboneConfig`/`TimmBackbone` unless separately converted to `backbone`/kwargs.
- Admission should require exact delegated body allowlists and an extracted feature-info ABI snapshot: stage names, output indices/features, channels, strides/reductions, layout, and input preprocessing contract.
