# hgnet_v2 source notes

Audit date: 2026-05-13

## Local source basis

- Transformers checkout: `X:/H/transformers`
- Commit: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`
- Primary source: `src/transformers/models/hgnet_v2/modular_hgnet_v2.py`
- Generated runtime files: `src/transformers/models/hgnet_v2/modeling_hgnet_v2.py`, `src/transformers/models/hgnet_v2/configuration_hgnet_v2.py`
- The generated files state they are produced from `modular_hgnet_v2.py`; report treats modular as authoritative for future source edits and generated modeling as runtime parity.

## Source observations

- `HGNetV2PreTrainedModel.main_input_name = "pixel_values"` and `input_modalities = ("image",)`.
- `HGNetV2Backbone.has_attentions = False`.
- `HGNetV2ConvLayer` is `Conv2d(bias=False) -> BatchNorm2d -> ACT2FN/Identity -> optional HGNetV2LearnableAffineBlock`.
- Conv padding is `(kernel_size - 1) // 2`; for stem 2x2 convs that is zero, so the preceding explicit `F.pad(..., (0,1,0,1))` is required for parity.
- `HGNetV2ConvLayerLight` is pointwise `Conv1x1 -> BN` with no activation, then depthwise spatial conv with activation.
- `HGNetV2Embeddings` uses `MaxPool2d(kernel_size=2, stride=1, ceil_mode=True)` and `torch.cat(..., dim=1)`.
- `HGNetV2BasicLayer` concatenates original input plus each sublayer output along channel axis, then applies two 1x1 aggregation conv layers. Residual add is enabled only for non-first blocks in a stage.
- `HGNetV2Stage` downsampling uses depthwise 3x3 conv with `activation=None`.
- `HGNetV2Encoder` can collect hidden states before each stage and after the final stage.
- `HGNetV2Backbone` returns selected feature maps from hidden states indexed by `stage_names`.
- `HGNetV2ForImageClassification` adds `AdaptiveAvgPool2d((1,1))`, `Flatten`, and optional `Linear(config.hidden_sizes[-1], config.num_labels)`.

## Backbone utility observations

- `BackboneConfigMixin` defaults to the last stage when both `out_features` and `out_indices` are omitted.
- It validates that `out_features` are ordered, unique, and a subset of `stage_names`, and that `out_indices` match them.
- `BackboneMixin.channels` uses `num_features`, which HGNetV2 sets to `[embedding_size] + hidden_sizes`; this is metadata and can diverge from actual conv channels if config is malformed.

## Processor observations

- No `image_processing_hgnet_v2.py` exists.
- `ustc-community/hgnet-v2` uses `RTDetrImageProcessor`.
- RTDetrImageProcessor default model inputs are `["pixel_values", "pixel_mask"]`, but with `do_pad=false` the inspected standalone preprocessor emits only stacked `pixel_values`.
- Default standalone preprocessor config: resize to 640x640, rescale by `1/255`, do not normalize, do not pad.
- Detection postprocessing in RTDetrImageProcessor converts relative center boxes to xyxy, applies optional target-size scaling, selects scores/top-k, and threshold-filters. This belongs to detector integration, not HGNetV2Backbone.

## Representative config notes

### https://huggingface.co/ustc-community/hgnet-v2

- Native `model_type: "hgnet_v2"`, `architectures: ["HGNetV2Backbone"]`.
- `stem_channels: [3,32,64]`.
- `stage_out_channels: [128,512,1024,2048]`.
- `stage_num_blocks: [1,2,5,2]`.
- `stage_numb_of_layers: [6,6,6,6]`.
- `use_learnable_affine_block: false`.
- `out_features: ["stage2","stage3","stage4"]`.
- Includes historical fields not read by native source for graph construction: `layer_type`, `downsample_in_bottleneck`, `downsample_in_first_stage`.

### https://huggingface.co/ustc-community/dfine-small-coco

- Top-level `model_type: "d_fine"` with nested `backbone_config.model_type: "hgnet_v2"`.
- Smaller backbone: `stem_channels: [3,16,16]`, `stage_out_channels: [64,256,512,1024]`, `stage_num_blocks: [1,1,2,1]`, `stage_numb_of_layers: [3,3,3,3]`.
- `use_learnable_affine_block: true`.
- Consumes `out_features: ["stage2","stage3","stage4"]` with `feat_strides: [8,16,32]`.

### https://huggingface.co/ustc-community/dfine-large-coco

- Top-level `model_type: "d_fine"` with nested HGNetV2 backbone.
- Large/default-like backbone: `stem_channels: [3,32,48]`, `stage_out_channels: [128,512,1024,2048]`, `stage_num_blocks: [1,1,3,1]`, `stage_numb_of_layers: [6,6,6,6]`.
- `use_learnable_affine_block: false`.
- Consumes `out_features: ["stage2","stage3","stage4"]`.

### https://huggingface.co/Intellindust/DEIMv2_HGNetv2_ATTO_COCO

- External-style config, not native Transformers `HGNetV2Config`.
- Has `HGNetv2.name: "Atto"`, `use_lab: true`, and `return_idx: [2]`.
- No `preprocessor_config.json` found at main during audit.
- Treat as variation/gating signal only; native DinoML loader should not accept this as a direct hgnet_v2 config without a separate translator/audit.

### https://huggingface.co/Intellindust/DEIMv2_HGNetv2_N_COCO

- External-style config, not native Transformers `HGNetV2Config`.
- Has `HGNetv2.name: "B0"`, `use_lab: true`, and `return_idx: [2,3]`.
- No `preprocessor_config.json` found at main during audit.
- Treat as variation/gating signal only.

## Gated or unavailable-source notes

- No inspected Hugging Face repo returned a 401/403 gated-access response.
- The only availability gap observed was missing `preprocessor_config.json` on the two Intellindust DEIMv2 repos. Those repos are not native Transformers hgnet_v2 checkpoints, so this gap should be handled by a DEIMv2/composed-model audit rather than by widening hgnet_v2 admission.

## Related local composed-family references

- `src/transformers/models/d_fine/configuration_d_fine.py` uses `backbone_config` via AutoConfig with default config type `hgnet_v2`.
- `src/transformers/models/deimv2/configuration_deimv2.py` uses `backbone_config` similarly and consumes feature maps in a hybrid encoder.
- `src/transformers/models/pp_ocrv5_server_det/modular_pp_ocrv5_server_det.py` defaults to HGNetV2-style `out_features: ["stage1","stage2","stage3","stage4"]` and consumes feature maps in the detection head.
- `src/transformers/models/pp_ocrv5_server_rec/modular_pp_ocrv5_server_rec.py` consumes the last HGNetV2 feature map for recognition.
- `src/transformers/models/pp_doclayout_v2` and `pp_doclayout_v3` default to HGNetV2 backbones but own layout detection/reading-order heads separately.
