# Transformers audit: timm_backbone

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: wrapper family; representative configs listed below
Config source: TimmBackboneConfig plus representative Hub config.json/preprocessor_config.json files
Source files inspected:
  - src/transformers/models/timm_backbone/configuration_timm_backbone.py
  - src/transformers/models/timm_backbone/modeling_timm_backbone.py
  - src/transformers/backbone_utils.py
  - src/transformers/models/auto/auto_factory.py
  - src/transformers/dependency_versions_table.py
Any missing files or assumptions:
  - No family-local image processor exists.
  - No timm model source was audited as part of this family report.
  - No imports, model execution, DinoML tests, or DinoML code edits were run.
```

Primary source links:

- [modeling_timm_backbone.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/timm_backbone/modeling_timm_backbone.py)
- [configuration_timm_backbone.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/timm_backbone/configuration_timm_backbone.py)
- [backbone_utils.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/backbone_utils.py)
- [auto_factory.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/auto/auto_factory.py)

Representative configs inspected:

- [Noah-Wang/eva02-ai-art-detector](https://huggingface.co/Noah-Wang/eva02-ai-art-detector/raw/main/config.json)
- [Noah-Wang/eva02-ai-art-detector-prod](https://huggingface.co/Noah-Wang/eva02-ai-art-detector-prod/raw/main/config.json)
- [scizzum/model_10_22_run](https://huggingface.co/scizzum/model_10_22_run/raw/main/config.json) and [preprocessor](https://huggingface.co/scizzum/model_10_22_run/raw/main/preprocessor_config.json)
- [omlab/omdet-turbo-swin-tiny-hf](https://huggingface.co/omlab/omdet-turbo-swin-tiny-hf/raw/main/config.json) and [preprocessor](https://huggingface.co/omlab/omdet-turbo-swin-tiny-hf/raw/main/preprocessor_config.json)
- [facebook/detr-resnet-50](https://huggingface.co/facebook/detr-resnet-50/raw/main/config.json) and [preprocessor](https://huggingface.co/facebook/detr-resnet-50/raw/main/preprocessor_config.json)
- [facebook/maskformer-swin-base-ade](https://huggingface.co/facebook/maskformer-swin-base-ade/raw/main/config.json), as a contrast case where `backbone_config` is native Swin rather than timm.

## 2. High-level architecture

`timm_backbone` is not a fixed neural architecture. It is a Transformers wrapper around an external [timm](https://github.com/huggingface/pytorch-image-models) model created by `timm.create_model(...)`.

Dataflow:

```text
image processor or parent model preprocessing
-> pixel_values [B,C,H,W]
-> TimmBackbone wrapper
-> timm.create_model delegated body
-> timm feature_info-selected feature maps
-> BackboneOutput(feature_maps, optional hidden_states, attentions=None)
-> parent head, decoder, FPN, detector, segmenter, depth head, or caller
```

Wrapper-owned stages:

- Parse `TimmBackboneConfig`.
- Require `timm` backend.
- Instantiate the delegated timm model with `backbone`, `features_only`, `in_chans`, `out_indices`, `output_stride`, and extra kwargs.
- Derive `stage_names`, `num_features`, `out_features`, and `out_indices` from `backbone.feature_info`.
- Apply optional timm batch-norm freezing.
- Normalize outputs into Transformers `BackboneOutput`.

Delegated stages:

- All convolutions, attention blocks, MLPs, norms, pooling, classifier heads, positional embeddings, stochastic-depth branches, window partitioning, patch embedding, and feature extraction semantics belong to the selected timm model body.
- DinoML should not infer operator coverage from this wrapper alone.

First useful DinoML runtime target:

```text
metadata-only / wrapper-dispatch parity for an allowlisted timm body, returning image-like feature maps.
```

This report does not own a ResNet/Swin/EVA/ConvNeXt operator audit. Those must be separate body-specific reports or routed to fallback.

## 3. Important config dimensions

Source-declared `TimmBackboneConfig` fields:

| Field | Source default | Runtime significance |
| --- | ---: | --- |
| `model_type` | `"timm_backbone"` | AutoConfig/AutoBackbone routing key. |
| `backbone` | `None` | Required timm model name; `TimmBackbone.__init__` raises when absent. |
| `num_channels` | `3` | Passed as `in_chans` unless caller supplies `in_chans` kwarg. |
| `features_only` | `True` | Passed to timm. Source comment says this is not possible for some transformer architectures. |
| `out_indices` | `[-1]` after post-init | Passed to timm; wrapper later aligns with `feature_info`. |
| `out_features` | derived | Derived from timm `feature_info.module_name()`, not known from config alone. |
| `stage_names` | derived | Derived from `feature_info.info[*]["module"]`. |
| `freeze_batch_norm_2d` | `False` | Calls timm BN freeze helper when true. |
| `output_stride` | `None` | Passed through to timm; support is delegated and model-dependent. |
| `return_dict`, `output_hidden_states`, `output_attentions` | inherited | `output_attentions=True` is rejected. |

Representative checkpoint/config sweep:

| Source | Scope | Delegated body selector | Input/preproc signal | Output feature request | Operator-significant notes |
| --- | --- | --- | --- | --- | --- |
| Source default `TimmBackboneConfig(backbone="resnet18")` | In-library wrapper example/test | `backbone="resnet18"` | `[B,3,H,W]`, no family processor | default `[-1]` | Current source path; full op surface is timm ResNet. |
| `Noah-Wang/eva02-ai-art-detector` | Hub timm-style image classification config | `architecture="eva02_base_patch14_448"` | `pretrained_cfg.input_size=[3,448,448]`, CLIP mean/std | not current-source compatible | `model_type=timm_backbone` but omits source-required `backbone`; config uses timm keys. |
| `Noah-Wang/eva02-ai-art-detector-prod` | Hub timm-style image classification config | `architecture="eva02_base_patch14_448"` | same 448 preprocessing metadata | not current-source compatible | `num_classes=9`; same current-source `backbone` gap. |
| `scizzum/model_10_22_run` | Hub config advertising `TimmBackbone` | `timm_model_name="eva_giant_patch14_224.clip_ft_in1k"` | preprocessor resizes to 224, CLIP mean/std | not current-source compatible without mapping | Has `hidden_size=1408`, `layers=40`, `heads=16`; current source does not read these fields. |
| `omlab/omdet-turbo-swin-tiny-hf` | Composite model using timm backbone | `backbone="swin_tiny_patch4_window7_224"` plus `img_size=640`, `always_partition=True` | `DetrImageProcessor`, 640x640, 0-255 mean/std, `do_rescale=false` | `out_indices=[1,2,3]` | Current `OmDetTurboConfig` explicitly forwards extra timm kwargs outside `TimmBackboneConfig`. |
| `facebook/detr-resnet-50` | Historical composite DETR config | `backbone="resnet50"` | shortest-edge 800 / longest-edge 1333 | current config defaults choose timm unless overridden | Parent model report must own DETR head/postprocess; wrapper only owns backbone dispatch. |
| `facebook/maskformer-swin-base-ade` | Contrast, native HF backbone | `backbone_config.model_type="swin"` | size 640, divisibility 32 | native Swin `out_indices` | Not a timm_backbone admission case. |

## 3a. Family variation traps

- `timm_backbone` is a wrapper family. `backbone="resnet18"`, `backbone="swin_tiny_patch4_window7_224"`, and `backbone="eva_giant_patch14_224.clip_ft_in1k"` imply completely different operator surfaces.
- Hub configs tagged `timm_backbone` may be timm-library configs, not valid current Transformers `TimmBackboneConfig` JSONs. Fields like `architecture` and `timm_model_name` are not read by current source as `backbone`.
- `features_only=True` is the normal wrapper path, but the source comment notes this is currently not possible for transformer architectures. Some delegated timm transformer bodies may fail or return non-image-like structures.
- `out_indices` are validated only after `feature_info` exists. Stage count, stage names, channel widths, and reduction/stride metadata are delegated.
- `out_features` cannot be specified through `AutoBackbone.from_pretrained` timm fallback; the current source rejects it there.
- Parent configs may pass extra timm kwargs through side channels. Example: `OmDetTurboConfig` extracts `img_size` and `always_partition` into `config.timm_kwargs` because `TimmBackboneConfig` does not forward arbitrary config attributes by itself.
- Layout is source NCHW: `pixel_values` are passed directly to timm as `[B,C,H,W]`. Channel-last/NHWC is only a guarded optimization opportunity inside an allowlisted body.
- Attention support is unknown per delegated body, but wrapper-level `output_attentions=True` always raises.
- Hidden-state behavior is wrapper-specific: requesting hidden states temporarily changes `self._backbone.return_layers` to all stages and then restores selected layers.
- BatchNorm freezing is a wrapper toggle, but only affects timm modules with supported BN classes.
- `output_stride` may imply dilation/stride replacement for CNNs; support is timm-body-specific and must be admitted per body.
- Weight naming and load behavior are timm-owned for `from_pretrained("resnet18")`; current tests explicitly skip ordinary HF save/load and safetensors expectations for `TimmBackbone`.

## 4. Operator coverage checklist

Wrapper-owned required operators and contracts:

### Dispatch / metadata

- `timm.create_model` boundary with exact kwargs:
  - `model_name = config.backbone`
  - `pretrained`
  - `features_only`
  - `in_chans`
  - `out_indices`
  - `output_stride`
  - parent-supplied extra kwargs
- `feature_info` extraction:
  - `stage_names = [stage["module"] for stage in feature_info.info]`
  - `num_features = [stage["num_chs"] ...]`
  - selected `out_indices = feature_info.out_indices`
  - selected `out_features = feature_info.module_name()`
- Output ABI:
  - `feature_maps`: tuple of selected tensors.
  - `hidden_states`: tuple of all feature-info stages when requested.
  - `attentions`: always `None`.

### Tensor/layout ops

- Input contract: `pixel_values` is rank-4 `[B,C,H,W]`, `C == num_channels/in_chans`.
- Feature-map ABI should preserve timm output layout, usually `[B,C_i,H_i,W_i]` for CNN-like bodies. Do not assume this for every timm transformer body without a body audit.
- Tuple packing/unpacking and filtered output handling.
- No general `permute`/layout conversion in wrapper source.

### Neural network primitives

- None owned by wrapper.
- Delegated model bodies may require conv, depthwise conv, patch embedding, LayerNorm, BatchNorm, GELU/SILU/ReLU, window attention, RoPE/relative bias, pooling, classifier heads, etc. These are unsupported by this report unless an exact timm body is allowlisted and audited.

### Attention primitives

- Wrapper has no attention output ABI. It rejects `output_attentions=True`.
- Delegated timm vision transformers may contain attention, but DinoML must audit them separately.

### Preprocessing-coupled ops

- No family-local image processor.
- Parent processors or timm metadata define resize/crop/rescale/normalize/pad behavior.

### Generation/cache/state ops

- Not applicable. This is an image backbone wrapper, not an autoregressive model.

### Quantized/packed weight metadata ops

- None in wrapper source.
- Any timm-specific checkpoint storage, classifier head metadata, or external quantization is outside this report.

## 5. Layer/block breakdown

Wrapper forward path:

```text
TimmBackbone.forward(pixel_values, output_attentions, output_hidden_states, return_dict, **kwargs):
  resolve return_dict/output flags
  if output_attentions:
    raise ValueError
  if output_hidden_states:
    save selected return_layers
    set return_layers to all feature_info layers
    hidden_states = delegated_timm_backbone(pixel_values, **kwargs)
    restore selected return_layers
    feature_maps = tuple(hidden_states[i] for i in out_indices)
  else:
    feature_maps = delegated_timm_backbone(pixel_values, **kwargs)
    hidden_states = None
  return BackboneOutput(feature_maps=tuple(feature_maps), hidden_states=..., attentions=None)
```

Initialization path:

```text
TimmBackbone.__init__(config, **kwargs):
  require timm
  require config.backbone is not None
  out_indices = config.out_indices or (-1,)
  in_chans = kwargs.pop("in_chans", config.num_channels)
  backbone = timm.create_model(config.backbone, features_only=config.features_only, ...)
  BackboneMixin initializes feature-info ABI from backbone.feature_info
  optionally freeze BN
  record selected/all return_layers
```

No repeated neural block is defined in Transformers source. Any block such as ResNet bottleneck, Swin block, EVA transformer block, ConvNeXt block, or EfficientNet MBConv belongs to the delegated timm body.

## 6. Attention requirements

No wrapper-level attention is required.

Required for this family:

- Reject `output_attentions=True` with a clear error.
- Preserve `attentions=None` in returned `BackboneOutput`.

Delegated-body caveat:

- Some timm bodies contain dense MHA, window attention, relative-position bias, absolute positional embeddings, or RoPE-like custom math. Those are not admitted by a generic `timm_backbone` integration.
- If a parent model requires a timm Swin/EVA/ViT body, create a separate exact-body audit and define its attention/cache-free encoder ABI.

KV cache, causal masks, packed varlen decode, generation attention, and cross-attention are not applicable to the wrapper.

## 7. Position encoding and custom math

The wrapper defines no position encoding.

Potential delegated examples, not admitted here:

- Swin-style window relative position bias.
- ViT/EVA absolute position embeddings and interpolation.
- CNN stride/dilation spatial indexing.

DinoML first integration should not implement any of these under the `timm_backbone` family name. The safe snippet for this family is only the dispatch guard:

```python
def admit_timm_backbone(config, body_allowlist):
    body = config.backbone
    if body not in body_allowlist:
        raise UnsupportedTimmBackbone(body)
    return body_allowlist[body].feature_abi
```

## 8. Preprocessing and input packing

Wrapper input:

```text
pixel_values: float tensor [B, C, H, W]
```

The wrapper performs no image resize, crop, rescale, normalization, padding, channel conversion, mask creation, or packed patch construction.

Representative preprocessing sources:

- `scizzum/model_10_22_run`: `preprocessor_config.json` resizes to 224, rescales, normalizes with CLIP-style mean/std, and emits standard image tensor input.
- `omlab/omdet-turbo-swin-tiny-hf`: `DetrImageProcessor` config uses fixed 640x640 resize, `do_rescale=false`, and 0-255 ImageNet mean/std. Parent processor also owns detection annotation conversion for training/eval.
- `facebook/detr-resnet-50`: `DetrImageProcessor` uses shortest-edge/longest-edge resize and ImageNet normalization; parent DETR owns pixel masks and postprocessing.
- `facebook/maskformer-swin-base-ade`: legacy `MaskFormerFeatureExtractor` uses resize, normalization, and size divisibility; native Swin backbone, not timm.

Layout constraints:

- Initial graph translation should remain NCHW faithful.
- NHWC/channel-last is a body-specific optimization only when the whole delegated region and its parent consumers are controlled.
- If a layout pass rewrites a selected body, it must also rewrite axis-sensitive parents: concat axes for multi-scale features, normalization axes, pooling axes, FPN inputs, mask resizing, and any detector/segmenter spatial metadata.

## 9. Graph rewrite / lowering opportunities

### Rewrite: exact delegated body specialization

Source pattern:

```text
TimmBackbone(backbone=<body>, features_only=True, out_indices=<indices>)
```

Replacement:

```text
BodySpecificBackbone_<body>(pixel_values) -> feature_map tuple with same feature_info ABI
```

Preconditions:

- Exact timm model name is allowlisted.
- timm version/source commit for that body is pinned.
- Feature-info snapshot is captured: stage names, indices, channels, reductions/strides, output tensor ranks/layouts.
- Parent processor contract is known.
- Parent consumer accepts the same selected feature maps.

Failure cases:

- Unknown body name.
- Body lacks `features_only` support.
- Config omits `backbone` and only has legacy `architecture`/`timm_model_name`.
- Extra kwargs affect topology but are not captured in the specialization key.
- Returned feature layout is not image-like when parent expects image maps.

Parity test sketch:

```text
For each allowlisted body:
  run timm/Transformers wrapper and DinoML body on fixed image tensors
  compare each selected feature_map shape, dtype, layout, and values
  repeat with output_hidden_states=True and verify all-stage tuple ABI
```

### Rewrite: Conv patch/stem lowering inside an allowlisted body

This is not wrapper-generic. For exact CNN/ViT patch stems, use body audit guards such as:

- `Conv2d` with static weights and known NCHW input.
- Non-overlap patch conv only when `kernel_size == stride`, `padding == 0`, `dilation == 1`, `groups == 1`, and image dimensions are divisible.
- Preserve timm flatten order and positional embedding assumptions.

### Rewrite: feature tuple materialization elision

Source pattern:

```text
feature_maps = tuple(timm_outputs)
parent consumes only one or a fixed subset
```

Replacement:

```text
produce only selected feature buffers needed by parent
```

Preconditions:

- Parent consumer does not request hidden states.
- `out_indices` are static and validated.
- No hooks or debug output require full hidden-state tuple.

Failure cases:

- `output_hidden_states=True`.
- Parent head dynamically indexes feature tuple.
- Body requires intermediate feature capture hooks that alter execution.

## 10. Kernel fusion candidates

Highest priority:

- Wrapper admission and ABI capture. This prevents unsupported delegated bodies from entering the compiler under a misleading generic `timm_backbone` label.
- Exact-body feature-map parity for one small allowlisted body, likely `resnet18` or `resnet50`, because current tests use these and the output ABI is easy to validate.

Medium priority:

- NCHW Conv/BN/activation fusion for allowlisted CNN bodies.
- Static feature-map tuple pruning for parent heads that consume fixed `out_indices`.
- Parent-specific multi-scale feature normalization/projection fusion after the feature ABI is stable.

Lower priority:

- Vision-transformer body fusions such as QKV + attention or MLP activation for EVA/ViT/Swin timm bodies. These require separate source audits.
- NHWC/channel-last body conversion. Useful for conv throughput, but it must be guarded and axis-rewritten through the parent consumer.
- Frozen BatchNorm folding. Only safe for eval, known BN modules, and body-specific weights.

## 11. Runtime staging plan

Stage 1: wrapper metadata parser

- Parse `TimmBackboneConfig` and reject configs without `backbone`.
- Reject legacy Hub timm-style configs unless a converter maps `architecture` or `timm_model_name` to `backbone`.
- Record source timm dependency/version and all kwargs as an artifact-visible delegated-body key.

Stage 2: feature ABI manifest

- For allowlisted bodies only, capture feature-info ABI:
  - stage names
  - selected `out_indices` and `out_features`
  - channel counts
  - reductions/strides if available from timm `feature_info`
  - tensor rank/layout
  - parent preprocessing contract

Stage 3: fallback routing

- Route unknown timm bodies to PyTorch/timm fallback or reject at compile time.
- Do not lower delegated operators under the generic wrapper.

Stage 4: first exact-body lowering

- Choose one body, preferably `resnet18`/`resnet50`, and run a separate body audit.
- Implement only that body's operators and feature-map outputs.

Stage 5: parent composition

- Compose with one parent that uses a timm backbone, such as DETR-style multi-scale consumption or OmDet-Turbo Swin, only after both parent and body ABIs are audited.

Stage 6: layout/fusion optimization

- Add guarded channel-last or fused conv/norm/activation paths per allowlisted body.

Stub initially:

- `output_hidden_states=True` may be deferred if first parent does not need all stages.
- BatchNorm freeze toggling can be represented as a load-time weight transform/fold or rejected until exact-body support exists.

## 12. Parity and validation plan

Wrapper-level tests:

- Config parsing: valid `backbone`, missing `backbone`, invalid `out_indices`, duplicate/out-of-order indices, mismatched `out_features` and `stage_names`.
- AutoBackbone routing:
  - non-Hub string, e.g. `resnet18`, becomes timm fallback.
  - Hub model id remains normal HF config path.
  - `out_features` with timm fallback is rejected.
- Forward ABI:
  - `feature_maps` tuple length equals selected `out_indices`.
  - `output_attentions=True` rejects.
  - `return_dict=False` returns tuple-compatible structure.
  - `output_hidden_states=True` returns all-stage hidden states and selected feature maps.

Exact-body parity tests after allowlisting:

- Random NCHW image tensors for shape/value parity against Transformers+timm.
- Representative processor output tensor parity for real images.
- Each selected feature map compared independently.
- Parent head smoke test with fixed feature maps.

Suggested tolerances:

- fp32 body parity: `rtol=1e-4`, `atol=1e-4` initially, adjust by body.
- fp16/bf16 optimized paths: body-specific tolerances after source/fallback comparison.

## 13. Performance probes

- Processor throughput separately from backbone throughput.
- Per-body backbone latency/throughput for `[B,C,H,W]` sweeps.
- Feature-output count sweep: final-only `[-1]` versus multi-scale `[1,2,3]` or `[1,2,3,4]`.
- Layout comparison: faithful NCHW versus guarded NHWC/channel-last for exact bodies.
- Parent composition probe: detector/segmenter head with precomputed feature maps versus end-to-end.
- Memory probe: peak activation memory when `output_hidden_states=True` versus selected feature maps only.
- Batch-size sweep for common image sizes: 224, 384, 448, 640, and DETR long-edge resize shapes.
- Fallback overhead probe: PyTorch/timm delegated execution versus DinoML exact-body lowering.

## 14. Skip/defer list

Safe to defer for first integration:

- All unknown timm bodies.
- Generic timm operator lowering.
- `features_only=False` classifier/logit path.
- `output_attentions=True`, since source rejects it.
- Training, gradients, stochastic depth/dropout behavior beyond eval parity.
- Generic save/load parity for timm checkpoints and safetensors.
- Device-map/meta-init support; source tests skip several ordinary HF loading paths.
- Channel-last conversion until exact body and parent consumers are audited.
- Legacy Hub timm-style configs unless a converter is explicitly designed.

Must not be silently deferred if admitting a body:

- Exact preprocessing and NCHW input contract.
- Feature-info ABI.
- Extra timm kwargs that affect topology, e.g. `img_size`, `always_partition`, `output_stride`, `in_chans`.
- Parent head expectations for feature count, channel widths, and spatial strides.

## 15. Final implementation checklist

- [ ] Add `timm_backbone` config parser with strict required `backbone`.
- [ ] Add compile admission rule: reject unknown delegated timm bodies.
- [ ] Add artifact-visible delegated-body key: timm model name, timm version/source, kwargs, `features_only`, `in_chans`, `out_indices`, `output_stride`.
- [ ] Add feature-info ABI manifest schema: stage names, selected features, indices, channels, reductions/strides, ranks, layouts.
- [ ] Add config-gap handling for Hub timm-style `architecture` / `timm_model_name` configs.
- [ ] Decide fallback policy: PyTorch/timm fallback or compile-time rejection.
- [ ] Pick first exact delegated body and create a separate body audit.
- [ ] Add wrapper parity tests for output tuple/dict behavior and attention rejection.
- [ ] Add body parity tests against Transformers+timm for each selected feature map.
- [ ] Add parent-composition test only after parent report and body report agree on feature-map ABI.
- [ ] Add performance probes separating preprocessing, backbone, and parent head.
