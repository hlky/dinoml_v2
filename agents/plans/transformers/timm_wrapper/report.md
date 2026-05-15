# DinoML Transformers Audit: timm_wrapper

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: representative timm/* Hub checkpoints, not one fixed architecture
Config source: HF config.json files that contain timm pretrained_cfg; sampled repos have no separate preprocessor_config.json
Source files inspected:
- transformers/src/transformers/models/timm_wrapper/configuration_timm_wrapper.py
- transformers/src/transformers/models/timm_wrapper/modeling_timm_wrapper.py
- transformers/src/transformers/models/timm_wrapper/image_processing_timm_wrapper.py
- transformers/src/transformers/configuration_utils.py, timm config detection path
- transformers/src/transformers/models/auto/image_processing_auto.py, timm config.json image-processor fallback
- transformers/src/transformers/conversion_mapping.py, timm_model prefix mapping
Any missing files or assumptions:
- This report audits the Transformers wrapper contract. The actual neural graph is owned by the external `timm` package selected by `config.architecture` and `config.model_args`.
- No native Transformers implementation exists for the selected timm body. DinoML should not admit arbitrary `timm_wrapper` checkpoints without an explicit architecture/operator allowlist.
- Sampled official `timm/*` repos returned 404 for `preprocessor_config.json`; preprocessing is derived from `config.json.pretrained_cfg`.
```

Snapshots were written under `agents/plans/transformers/timm_wrapper/_sources/`.

Representative HF configs inspected:

- [timm/resnet50.a1_in1k](https://huggingface.co/timm/resnet50.a1_in1k)
- [timm/efficientnet_b0.ra_in1k](https://huggingface.co/timm/efficientnet_b0.ra_in1k)
- [timm/convnext_base.fb_in22k_ft_in1k](https://huggingface.co/timm/convnext_base.fb_in22k_ft_in1k)
- [timm/vit_base_patch16_224.augreg2_in21k_ft_in1k](https://huggingface.co/timm/vit_base_patch16_224.augreg2_in21k_ft_in1k)
- [timm/swin_base_patch4_window7_224.ms_in22k_ft_in1k](https://huggingface.co/timm/swin_base_patch4_window7_224.ms_in22k_ft_in1k)

## 2. High-level architecture

`timm_wrapper` is an image-only adapter around an external timm model. It exposes two primary runtime contracts:

- `TimmWrapperModel`: feature extraction and optional pooled feature output.
- `TimmWrapperForImageClassification`: image classification logits from the timm model head.

Dataflow:

```text
image preprocessing from config.json.pretrained_cfg -> NCHW pixel_values -> timm.create_model(config.architecture, model_args) -> features/head -> pooled output or logits
```

Stage decomposition:

- CPU/data pipeline: timm `resolve_data_config` and `create_transform`, image conversion, resize/crop/interpolation, tensor conversion, normalization.
- GPU/runtime: external timm model forward. Transformers only casts `pixel_values` to model device/dtype before dispatch.
- Classification head: owned by timm. `TimmWrapperForImageClassification` creates it with `num_classes=config.num_labels`.
- Feature extraction: `TimmWrapperModel` creates timm with `num_classes=0`; if `features_only=True` is in `model_args`, output is whatever timm `forward` returns for that feature extractor.

Inference target for first DinoML integration: classify and feature-extract from a small allowlist of timm architectures whose bodies are already audited or separately admitted, such as ResNet, ConvNeXt, EfficientNet, ViT, and Swin. The wrapper itself should be treated as dispatch and metadata, not as a model family with fixed kernels.

## 3. Important config dimensions

Wrapper config fields:

| Field | Source | Meaning |
|---|---:|---|
| `architecture` | config class default `resnet50`; checkpoint value usually explicit | Name passed to `timm.create_model` |
| `model_args` | optional config field | Arbitrary kwargs passed to timm, including `features_only`, custom depth, output controls, or architecture-specific knobs |
| `do_pooling` | default `True` | `TimmWrapperModel` calls `forward_head(last_hidden_state)` when true and not `features_only` |
| `num_classes` | timm config root/pretrained_cfg | Converted into Transformers `num_labels`; root and nested `pretrained_cfg.num_classes` are removed/normalized on config load |
| `num_features` | sampled config.json | Feature width advertised by timm config, not validated by wrapper source |
| `pretrained_cfg.input_size` | config.json | Processor input convention as `[C, H, W]` |
| `pretrained_cfg.interpolation` | config.json | timm transform interpolation, commonly bicubic |
| `pretrained_cfg.crop_pct` | config.json | Resize/crop policy input to timm transforms |
| `pretrained_cfg.mean/std` | config.json | Channel normalization constants |
| cache support | source | None; no generation or KV cache |
| attention support | source | Wrapper does not expose attentions and raises if `output_attentions=True` |

Representative checkpoint sweep:

| Checkpoint | `architecture` | `num_features` | `num_classes` | Input size | Pool/classifier metadata | Operator implication |
|---|---|---:|---:|---|---|---|
| `timm/resnet50.a1_in1k` | `resnet50` | 2048 | 1000 | `[3,224,224]` | pool `[7,7]`, classifier `fc` | Conv/BN/ReLU/residual bottleneck; external timm-owned |
| `timm/efficientnet_b0.ra_in1k` | `efficientnet_b0` | 1280 | 1000 | `[3,224,224]` | pool `[7,7]`, classifier `classifier` | depthwise separable conv, SE/SiLU-style blocks depending on timm body |
| `timm/convnext_base.fb_in22k_ft_in1k` | `convnext_base` | 1024 | 1000 | `[3,224,224]` | pool `[7,7]`, classifier `head.fc` | depthwise 7x7 conv, LayerNorm/MLP-style ConvNeXt body |
| `timm/vit_base_patch16_224.augreg2_in21k_ft_in1k` | `vit_base_patch16_224` | 768 | 1000 | `[3,224,224]` | `global_pool=token`, classifier `head` | patch embedding + encoder MHA; external timm implementation |
| `timm/swin_base_patch4_window7_224.ms_in22k_ft_in1k` | `swin_base_patch4_window7_224` | 1024 | 1000 | `[3,224,224]` | `global_pool=avg`, classifier `head.fc` | windowed attention + patch merging; external timm implementation |

## 3a. Family variation traps

- `timm_wrapper` is dynamic: `architecture` selects a timm registry entry, so operator structure can change from CNN to ViT to Swin to MLP-like or custom hybrids.
- `model_args` is an arbitrary pass-through to `timm.create_model`; shape and operator coverage can change without new Transformers source.
- `features_only=True` changes `TimmWrapperModel` from `forward_features`/`forward_head` semantics to `timm_model.forward(...)` returning a feature list or tuple.
- `output_hidden_states=True` is allowed only if the selected timm model implements `forward_intermediates`; otherwise the source raises.
- `output_attentions=True` is always rejected by the wrapper, even if a timm architecture internally uses attention.
- `config.json` for timm repos often omits `model_type`; Transformers injects `model_type="timm_wrapper"` when a config dict contains `pretrained_cfg`.
- Image processor config is the model `config.json`; sampled repos lack `preprocessor_config.json`.
- Source tensors are NCHW (`pixel_values` from timm transforms). NHWC should be a guarded provider/layout optimization inside an admitted timm body, never a blind wrapper-level translation.
- Weight keys may be unprefixed timm names; Transformers adds `timm_model.` through `load_state_dict` and conversion mapping.
- Label maps may be populated from timm ImageNet metadata if `label_names` and custom labels are absent; this is metadata, not a runtime op.

## 4. Operator coverage checklist

The wrapper-level required operators are small:

Tensor/layout ops:

- Image batch tensor input `pixel_values` in NCHW `[B, C, H, W]`.
- Device/dtype cast: `pixel_values.to(self.device, self.dtype)`.
- Tuple/list output packaging for feature maps and hidden states.
- Optional prefix rewrite for weights: add `timm_model.` to unprefixed state dict keys.

Neural network primitives:

- No fixed wrapper-owned primitives beyond delegated timm model.
- Linear classifier initialization is wrapper-aware for newly initialized timm classifier heads, but inference should load checkpoint weights.

Attention primitives:

- No wrapper-level attention API. `output_attentions=True` must reject.
- Attention requirements belong to the admitted external timm architecture, for example ViT MHA or Swin window attention.

Generation/cache ops:

- None. `use_cache` is accepted in `TimmWrapperModel.forward` only as an unused signature compatibility argument.

Preprocessing-coupled ops:

- timm transform pipeline from `pretrained_cfg`: resize/crop/interpolation, RGB/PIL conversion for non-tensor inputs, tensor conversion, channel normalization.
- `return_tensors` must be `"pt"`; other return tensor types raise.

External dependency/provider admission:

- `timm.create_model(config.architecture, pretrained=False, num_classes=..., **model_args)`.
- Require a DinoML allowlist for `(architecture, model_args)` before graph lowering.
- Reject or route to PyTorch/timm fallback when the architecture is unknown, `model_args` contains unsupported dynamic behavior, or timm implementation uses operators not admitted by DinoML.

Representative architecture-owned operators if admitted:

- ResNet: Conv2d, BatchNorm2d, ReLU, MaxPool, AdaptiveAvgPool, residual adds, Linear.
- EfficientNet: Conv2d, depthwise Conv2d, BatchNorm2d, SiLU, squeeze-excitation pooling and channel MLP/conv, DropPath disabled in eval, Linear.
- ConvNeXt: depthwise Conv2d, LayerNorm/channel layout transitions, pointwise Linear/Conv, GELU, residual adds, GlobalResponseNorm for V2 variants if selected.
- ViT: patch Conv2d/Linear, class token, absolute position embedding/interpolation, LayerNorm, QKV Linear, dense MHA, GELU MLP, Linear head.
- Swin: patch embedding, window partition/reverse, shifted-window attention, relative position bias, mask add, patch merging, LayerNorm, MLP, pooling/head.

## 5. Layer/block breakdown

Wrapper creation:

```text
config = TimmWrapperConfig(...)
extra_init_kwargs = config.model_args or {}
timm_model = timm.create_model(config.architecture, pretrained=False, num_classes=..., **extra_init_kwargs)
```

`TimmWrapperModel`, `features_only=False`:

```text
pixel_values: [B,C,H,W] -> cast to model device/dtype
if output_hidden_states:
  last_hidden_state, hidden_states = timm_model.forward_intermediates(pixel_values, indices=optional_list)
else:
  last_hidden_state = timm_model.forward_features(pixel_values)
if do_pooling:
  pooler_output = timm_model.forward_head(last_hidden_state)  # classification head omitted because num_classes=0
else:
  pooler_output = None
return last_hidden_state, pooler_output, hidden_states
```

`TimmWrapperModel`, `features_only=True`:

```text
pixel_values: [B,C,H,W] -> cast
last_hidden_state = timm_model.forward(pixel_values, **kwargs)
hidden_states = last_hidden_state if output_hidden_states else None
pooler_output = None
```

`TimmWrapperForImageClassification`:

```text
pixel_values: [B,C,H,W] -> cast
if output_hidden_states:
  last_hidden_state, hidden_states = timm_model.forward_intermediates(pixel_values, indices=optional_list)
  logits = timm_model.forward_head(last_hidden_state)  # [B,num_labels]
else:
  logits = timm_model(pixel_values)  # [B,num_labels]
return logits, hidden_states
```

Projection shapes and biases are not fixed by Transformers source. They come from the selected timm architecture and loaded weights.

## 6. Attention requirements

No attention is required by the wrapper itself. `output_attentions=True` is rejected in both model classes.

For architectures that internally use attention, the attention contract must come from a separate timm-body audit or an allowlisted architecture mapping. Examples:

- ViT-like: noncausal encoder self-attention, dense MHA, no KV cache, no generation decode.
- Swin-like: noncausal local/window self-attention, shifted-window masks, relative position bias, no KV cache.

FlashAttention/SDPA compatibility cannot be inferred from `timm_wrapper` source. DinoML should not enable fused attention based only on `model_type="timm_wrapper"`; it must key off admitted architecture and verified timm implementation details.

## 7. Position encoding and custom math

No wrapper-specific position encoding exists. Position encoding, relative bias, RoPE, absolute embeddings, convolutional position encodings, and custom math are external timm-body responsibilities.

Wrapper-specific config detection can be summarized as:

```python
def is_timm_wrapper_config(config_dict):
    return "pretrained_cfg" in config_dict
```

Wrapper-specific forward dispatch can be summarized as:

```python
def timm_wrapper_features(pixel_values, timm_model, output_hidden_states, do_pooling):
    pixel_values = pixel_values.to(device=timm_model_device, dtype=timm_model_dtype)
    if output_hidden_states:
        x, hs = timm_model.forward_intermediates(pixel_values)
    else:
        x, hs = timm_model.forward_features(pixel_values), None
    pooled = timm_model.forward_head(x) if do_pooling else None
    return x, pooled, hs
```

The snippet is wrapper behavior only; actual timm position math is not reproduced here.

## 8. Preprocessing and input packing

`TimmWrapperImageProcessor` reads `pretrained_cfg` from the same `config.json` used by the model. It constructs:

```text
data_config = timm.data.resolve_data_config(pretrained_cfg, model=None, verbose=False)
val_transforms = timm.data.create_transform(**data_config, is_training=False)
```

Observed sampled preprocessing contracts:

| Checkpoint family | Input size | Interpolation | Crop pct | Mean/std |
|---|---|---|---:|---|
| ResNet50 | `[3,224,224]` | bicubic | 0.95 | ImageNet `[0.485,0.456,0.406]` / `[0.229,0.224,0.225]` |
| EfficientNet-B0 | `[3,224,224]` | bicubic | 0.875 | ImageNet |
| ConvNeXt-base | `[3,224,224]` | bicubic | 0.875 | ImageNet |
| ViT-base patch16 | `[3,224,224]` | bicubic | 0.9 | `[0.5,0.5,0.5]` / `[0.5,0.5,0.5]` |
| Swin-base | `[3,224,224]` | bicubic | 0.9 | ImageNet |

Processor details:

- Output is `BatchFeature({"pixel_values": images}, tensor_type="pt")`.
- `return_tensors` must be `"pt"`.
- If input is already a torch tensor, the processor applies `val_transforms` directly and adds a batch dimension for a single CHW tensor.
- If input is not a torch tensor, it flattens image lists, converts each image to PIL, applies transforms, and stacks.
- Some timm transform versions use `ToTensor`; when present, torch tensor input is converted to CPU numpy first.

No token packing, attention masks, position IDs, or multimodal placeholder tokens exist.

## 9. Graph rewrite / lowering opportunities

### Rewrite: wrapper dispatch admission

Source pattern:

```text
timm.create_model(config.architecture, pretrained=False, num_classes=..., **model_args)
```

Replacement:

```text
LookupDinoMLTimmArchitecture(architecture, normalized_model_args) -> admitted graph template/provider fallback
```

Preconditions:

- `architecture` exactly matches an allowlisted timm implementation version.
- `model_args` is empty or matches an allowlisted normalized dict.
- Loaded weight names and shapes match the template.
- `features_only`, `output_hidden_states`, and `do_pooling` output ABI is tested for that architecture.

Failure cases:

- Unknown architecture in current timm version.
- `model_args` changes depth, input channels, output stride, feature indices, dynamic image sizes, or head behavior without a matching template.
- Timm body uses unsupported custom ops.

Parity test sketch:

- Compare wrapper outputs against PyTorch/timm for `TimmWrapperModel` and `TimmWrapperForImageClassification` on fixed preprocessed `pixel_values`.

### Rewrite: weight prefix normalization

Source pattern:

```text
state_dict = {f"timm_model.{k}" if "timm_model." not in k else k: v for k, v in state_dict.items()}
```

Replacement:

```text
Normalize checkpoint keys once at load time.
```

Preconditions:

- Key does not already start with `timm_model.`.
- No unrelated parameter name contains `timm_model.` mid-string.

Failure cases:

- Mixed checkpoint where some keys are prefixed and others are intentionally not for a non-timm submodule.

### Rewrite: preprocessing specialization

Source pattern:

```text
PIL/tensor -> timm val_transforms -> normalized NCHW pixel_values
```

Replacement:

```text
Decode/resize/crop/normalize data pipeline -> pinned NCHW or provider-preferred NHWC staging -> graph input
```

Preconditions:

- `pretrained_cfg` transform chain is resolved and frozen.
- Interpolation, crop mode, crop pct, mean/std, input size, and tensor channel order are represented explicitly.
- Any NHWC staging converts back or enters an admitted NHWC-safe graph region with axis guards.

Failure cases:

- Timm transform includes custom augmentation or a transform not represented in DinoML preprocessing.
- Tensor inputs depend on timm-version-specific `MaybeToTensor` versus `ToTensor` behavior.

### Rewrite: admitted Conv2d patch/stem lowering

Use only inside a known admitted timm body.

Preconditions:

- Non-overlap patch/stem conv has `kernel_size == stride`, `padding == 0`, `dilation == 1`, `groups == 1`.
- Input spatial dimensions are divisible by kernel size.
- Consumer layout is controlled.

Replacement:

```text
WindowFlatten -> MatMul(weight.reshape(out, in*kh*kw).T) -> BiasAdd -> Reshape
```

Layout constraints:

- Source semantic graph is NCHW. NHWC optimization is legal only for the local controlled stem/patch region or when downstream admitted graph remains channel-last.

## 10. Kernel fusion candidates

Highest priority:

- Architecture admission/provider fallback: without a fixed timm body, every other fusion is unsafe.
- Preprocessing normalization pipeline: resize/crop/normalize dominates small-batch image serving and must match timm exactly.
- NCHW/NHWC boundary planning for admitted CNNs: avoid wrapper-level axis changes; fuse inside admitted conv blocks where layout is controlled.

Medium priority:

- Conv + BatchNorm + activation for ResNet/EfficientNet-style admitted bodies.
- Depthwise Conv + pointwise Conv + activation for EfficientNet/ConvNeXt-style admitted bodies.
- Patch Conv2d to GEMM for ViT/Swin-style admitted bodies.
- Pooling/head fusion for classification: global pool/flatten/linear/logits.

Lower priority:

- Hidden-state extraction fast paths for `forward_intermediates`.
- Broad timm operator registry support. This is high effort and should trail high-value allowlisted architectures.

## 11. Runtime staging plan

Stage 1: implement config and processor parsing.

- Detect timm config via `pretrained_cfg`.
- Normalize `num_classes` to `num_labels`.
- Extract `architecture`, `model_args`, `num_features`, `pretrained_cfg`, label metadata.

Stage 2: add strict architecture admission.

- Start with reject-by-default.
- Admit one checkpoint per already-audited family, e.g. `resnet50`, `convnext_base`, `efficientnet_b0`, `vit_base_patch16_224`, `swin_base_patch4_window7_224`.

Stage 3: implement wrapper ABI parity.

- `TimmWrapperForImageClassification`: logits only.
- `TimmWrapperModel`: `forward_features` plus optional `forward_head` pooling.
- Stub or reject `output_hidden_states` until each admitted body supports an equivalent intermediate-output ABI.

Stage 4: implement architecture-owned graph lowering.

- Reuse native DinoML family implementations when the timm body matches an already-audited architecture.
- Otherwise route to fallback instead of silently lowering unknown ops.

Stage 5: optimize layout/fusion.

- Add NHWC/channel-last fusions inside admitted conv-heavy regions.
- Keep wrapper input/output ABI explicit and tested.

Stage 6: expand allowlist.

- Add architectures only with source/body inspection, config sweep, and parity tests.

## 12. Parity and validation plan

- Config parity: load sampled `config.json` dictionaries and verify effective `model_type`, `architecture`, `num_labels`, `label_names`, `pretrained_cfg`, `do_pooling`, and `model_args`.
- Processor parity: compare preprocessed `pixel_values` against Transformers/timm processor for PIL and tensor inputs. Tolerances should be exact or near-exact for deterministic transforms after interpolation differences are controlled.
- Wrapper classification parity: for each admitted checkpoint, compare logits shape `[B,num_labels]` and numeric logits.
- Feature parity: compare `TimmWrapperModel.last_hidden_state` and `pooler_output` for `do_pooling=True/False`.
- `features_only` parity: compare feature map count, layout, channel widths, strides, and values for admitted feature-extractor configs.
- Hidden-state parity: only for admitted timm models with `forward_intermediates`; test both boolean and list-of-indices forms.
- Rejection tests: unknown architecture, unsupported `model_args`, `output_attentions=True`, unsupported `output_hidden_states=True`, non-`pt` processor output.

Recommended tolerances:

- fp32: `rtol=1e-4`, `atol=1e-5` for model outputs; stricter if using identical PyTorch kernels.
- fp16/bf16: `rtol=5e-2`, `atol=5e-2` for full model outputs, with per-block probes for diagnosing drift.

## 13. Performance probes

- Preprocessing throughput for each representative `pretrained_cfg`: PIL path, tensor path, batch-size sweep.
- Classification throughput per admitted architecture.
- Feature extraction throughput with `do_pooling=False` and `do_pooling=True`.
- `features_only` feature-map throughput and memory by output feature count.
- Batch-size sweep: 1, 8, 32, 128 for common 224x224 configs.
- Resolution sweep for configs with `fixed_input_size=false`, including advertised `test_input_size` when present.
- Layout comparison inside admitted CNN bodies: NCHW baseline versus guarded NHWC/channel-last fusions.
- Head-only overhead: `forward_features` versus `forward_features + forward_head`.
- Fallback rate: percent of encountered `timm_wrapper` checkpoints rejected or routed to external timm fallback due to unsupported architecture/model_args.

## 14. Skip/defer list

- Arbitrary timm architecture support.
- Training and gradient checkpointing.
- `output_attentions`; wrapper source rejects it.
- `output_hidden_states` for timm models without `forward_intermediates`.
- Exotic `model_args` that alter topology, output stride, input channels, or dynamic behavior.
- Non-PyTorch processor outputs; source requires `return_tensors="pt"`.
- Quantization-specific timm loaders and custom external weights unless separately admitted.
- Multi-GPU/tensor parallel; wrapper marks timm as not supporting model parallelism and sets `_no_split_modules`.

## 15. Final implementation checklist

- [ ] Parse timm `config.json` and inject/recognize `model_type="timm_wrapper"` when `pretrained_cfg` is present.
- [ ] Normalize `num_classes`/`num_labels` and label metadata.
- [ ] Parse `architecture`, `model_args`, `num_features`, `global_pool`, and `pretrained_cfg`.
- [ ] Implement strict allowlist for `(architecture, model_args)`.
- [ ] Normalize checkpoint keys with optional `timm_model.` prefix.
- [ ] Implement `TimmWrapperImageProcessor` parity from `pretrained_cfg`.
- [ ] Implement classification wrapper ABI: `pixel_values -> logits`.
- [ ] Implement base wrapper ABI: `pixel_values -> last_hidden_state, optional pooler_output`.
- [ ] Reject `output_attentions=True`.
- [ ] Gate `output_hidden_states` behind admitted `forward_intermediates` parity.
- [ ] Add one checkpoint parity test per admitted timm architecture.
- [ ] Add preprocessing parity tests for PIL and tensor inputs.
- [ ] Add unknown-architecture and unsupported-`model_args` rejection tests.
- [ ] Benchmark preprocessing, feature extraction, classification head, and layout variants.
