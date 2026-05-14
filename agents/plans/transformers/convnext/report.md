# ConvNeXT Transformers Family Audit

## 1. Source basis

Transformers commit/version: local checkout `X:/H/transformers` at `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`.

Model id sweep:

- `facebook/convnext-tiny-224`
- `facebook/convnext-small-224`
- `facebook/convnext-base-224`
- `facebook/convnext-large-224`
- `facebook/convnext-large-384`

Config source: downloaded from `https://huggingface.co/<model-id>/resolve/main/config.json` and `preprocessor_config.json`, cached under `agents/plans/transformers/convnext/_sources/`.

Source files inspected:

- `X:/H/transformers/src/transformers/models/convnext/modeling_convnext.py`
- `X:/H/transformers/src/transformers/models/convnext/configuration_convnext.py`
- `X:/H/transformers/src/transformers/models/convnext/image_processing_convnext.py`
- `X:/H/transformers/src/transformers/models/convnext/image_processing_pil_convnext.py`
- `X:/H/transformers/src/transformers/models/convnextv2/modeling_convnextv2.py`, only to verify that Global Response Normalization belongs to the separate `convnextv2` family, not this `convnext` target.

Primary runtime target: image classification with `ConvNextForImageClassification`. `ConvNextModel` is required as the base encoder. `ConvNextBackbone` is optional but useful for future detection/segmentation integrations. Training loss paths, stochastic DropPath behavior, and gradients are deferred.

Any missing files or assumptions: no remote code was required. The current source has no attention, text tokenizer, cache, or sequence generation path. The official HF configs omit `num_labels` as a scalar but include 1000-entry `id2label`/`label2id`; effective `num_labels` is therefore 1000 by `PreTrainedConfig` convention.

## 2. High-level architecture

ConvNeXT is a pure vision encoder/classifier:

```text
CPU image preprocessing -> pixel_values [B, 3, H, W]
  -> patch/stem Conv2d stride 4 + channels-first LayerNorm
  -> 4 ConvNeXT stages with depthwise 7x7 conv blocks and stride-2 downsample between stages
  -> global average pool over H,W
  -> final LayerNorm over channels
  -> Linear classifier -> logits [B, num_labels]
```

The modeling code uses source NCHW tensors at stage boundaries. Inside each residual block, it creates a local NHWC island:

```text
NCHW -> depthwise Conv2d -> permute to NHWC -> LayerNorm(C) -> Linear(C,4C)
     -> GELU -> Linear(4C,C) -> layer scale -> permute to NCHW -> residual add
```

This is not a transformer despite living in the Transformers library. There is no attention, no positional encoding, no token packing, no KV cache, and no generation controller.

Stage decomposition:

- CPU/data pipeline: decode image, resize/crop or warp, rescale, normalize, produce NCHW `pixel_values`.
- Stem: non-overlapping patch Conv2d with kernel/stride `patch_size=4`.
- Encoder stages: independently testable NCHW feature maps with known channel/spatial scales.
- Pool/head: mean over spatial axes, LayerNorm, classifier.
- Backbone output: optional feature maps from `stem`, `stage1`..`stage4`, each normalized with channels-first LayerNorm.

## 3. Important config dimensions

Source defaults from `ConvNextConfig`:

| Field | Default / meaning |
| --- | --- |
| `model_type` | `convnext` |
| `num_channels` | 3 |
| `patch_size` | 4 |
| `image_size` | 224 |
| `num_stages` | 4 |
| `hidden_sizes` | `(96, 192, 384, 768)` |
| `depths` | `(3, 3, 9, 3)` |
| `hidden_act` | `gelu` |
| `layer_norm_eps` | `1e-12` for final pooler LayerNorm; stem/block/downsample use hard-coded `1e-6` |
| `layer_scale_init_value` | `1e-6`; if `<= 0`, layer scale parameter is absent |
| `drop_path_rate` | 0.0 in inspected configs; inference path is identity regardless |
| `out_features` / `out_indices` | backbone-only output selection via `BackboneConfigMixin` |

Representative checkpoint sweep:

| Model id | Image size | Processor size | Depths | Hidden sizes | Blocks | Last map for square input | Classifier |
| --- | ---: | ---: | --- | --- | ---: | --- | --- |
| `facebook/convnext-tiny-224` | 224 | 224 | 3,3,9,3 | 96,192,384,768 | 18 | `[B,768,7,7]` | Linear 768 -> 1000 |
| `facebook/convnext-small-224` | 224 | 224 | 3,3,27,3 | 96,192,384,768 | 36 | `[B,768,7,7]` | Linear 768 -> 1000 |
| `facebook/convnext-base-224` | 224 | 224 | 3,3,27,3 | 128,256,512,1024 | 36 | `[B,1024,7,7]` | Linear 1024 -> 1000 |
| `facebook/convnext-large-224` | 224 | 224 | 3,3,27,3 | 192,384,768,1536 | 36 | `[B,1536,7,7]` | Linear 1536 -> 1000 |
| `facebook/convnext-large-384` | 384 | 384 | 3,3,27,3 | 192,384,768,1536 | 36 | `[B,1536,12,12]` | Linear 1536 -> 1000 |

For square `S x S`, the stem emits `S/4`, then stages 2, 3, and 4 each downsample by 2. Effective final stride is 32 when all divisions are exact. The source Conv2d formulas use no padding for stem/downsample and padding 3 for depthwise 7x7 blocks, so odd or non-divisible dimensions should use normal Conv2d floor behavior unless Dinoml adds explicit divisibility guards.

## 3a. Family variation traps

- Plain `convnext` has no Global Response Normalization. GRN appears in `convnextv2` as `ConvNextV2GRN`, with `vector_norm(..., dim=(1,2))` over NHWC spatial axes after the first pointwise expansion. Treat GRN as a separate `convnextv2` follow-up unless a checkpoint has `model_type="convnextv2"`.
- The source block explicitly permutes NCHW -> NHWC -> NCHW around LayerNorm and pointwise Linear. A layout pass may keep NHWC internally, but initial translation should preserve the source axis semantics.
- Channels-first LayerNorm is implemented as `permute(0,2,3,1) -> LayerNorm(C) -> permute(0,3,1,2)`. A native channels-first LayerNorm is allowed only if it normalizes per pixel across channel axis, not across spatial axes.
- The final pooled LayerNorm uses `config.layer_norm_eps` (`1e-12` in configs), while other LayerNorm instances use `1e-6`.
- `patch_size` is annotated as int or pair. Source passes it directly to `nn.Conv2d(kernel_size=patch_size, stride=patch_size)`, so pair-valued patch sizes imply pair-valued stride and shape equations.
- 224 processors use `crop_pct=0.875`: resize shortest edge to `int(224 / 0.875)=256`, then center crop to 224. 384 configs have `crop_pct=null` in downloaded configs; current processor default logic warps to 384 square for `shortest_edge >= 384`.
- Backbone outputs add extra channels-first LayerNorms on selected feature maps. This is not part of classification logits but matters for detection-style use.
- Training DropPath is present but inference should lower it to identity. Do not compile random DropPath into an inference artifact.
- `hidden_act` comes from config. Inspected checkpoints use GELU, but source dispatches through `ACT2FN`, so non-GELU configs would change the block MLP activation.

Axis-sensitive layout rewrite notes:

| Source op | Source axis/layout | NHWC/channel-last rewrite requirement |
| --- | --- | --- |
| pixel input | `[B,C,H,W]` | Processor/runtime contract must change or insert boundary transpose |
| stem/downsample Conv2d | NCHW PyTorch Conv2d | Use NHWC conv provider or transform weights `[O,I,kH,kW]` for chosen kernel |
| channels-first LayerNorm | normalizes `C` via NCHW<->NHWC permutes | Rewrite normalized axis from `dim=1` semantic channel to last axis |
| block Linear | source input NHWC, Linear over last C | Already channel-last; can be fused as 1x1 conv/GEMM over `B*H*W` rows |
| global average pool | `mean([-2,-1])` over H,W | In NHWC, mean over axes `[1,2]`; do not reduce channel axis |
| backbone feature maps | returned NCHW | Either preserve NCHW ABI or document NHWC feature-map ABI separately |

## 4. Operator coverage checklist

Tensor/layout ops:

- NCHW input validation for `pixel_values`, channel count equals `config.num_channels`.
- `permute(0,2,3,1)` and `permute(0,3,1,2)` or equivalent layout-region representation.
- Mean reduction over two spatial axes for global average pooling.
- Residual add with same NCHW shape.
- Optional hidden-state capture for backbone/classification diagnostics.

Neural network primitives:

- Conv2d stem: `[B,3,H,W] -> [B,C0,floor((H-4)/4)+1,floor((W-4)/4)+1]`, kernel 4, stride 4, padding 0, groups 1, bias true by PyTorch default.
- Stage downsample Conv2d for stages 2-4: `C_in -> C_out`, kernel 2, stride 2, padding 0, groups 1, bias true.
- Depthwise Conv2d in every block: `C -> C`, kernel 7, stride 1, padding 3, groups `C`, bias true.
- LayerNorm over channel dimension:
  - stem/downsample/backbone: channels-first wrapper, effective per-location channel LN, eps `1e-6`.
  - block: native channels-last LN, eps `1e-6`.
  - final pooled vector: rank-2 LN over last C, eps `config.layer_norm_eps`.
- Pointwise MLP per spatial position:
  - `Linear(C -> 4C)` with bias.
  - GELU or configured activation.
  - `Linear(4C -> C)` with bias.
- Layer scale: multiply NHWC `[B,H,W,C]` by learned `[C]` when enabled.
- Classifier: Linear `C_last -> 1000` for inspected checkpoints.

Attention primitives: none.

Position/rotary/relative-bias ops: none.

Preprocessing-coupled ops:

- Bicubic resize.
- Optional center crop for sizes below 384.
- Rescale by default image pipeline factor, then normalize by ImageNet mean/std.
- Output tensor is `pixel_values` in channel-first form `[B,3,S,S]` for the standard processor path.

Backbone optional ops:

- Feature-map selection by `out_features`/`out_indices`.
- Per-output channels-first LayerNorm.
- Return tuple of feature maps, preserving source NCHW ABI.

## 5. Layer/block breakdown

Stem:

```text
pixel_values: [B, 3, H, W]
x = Conv2d(3 -> C0, kernel=4, stride=4, padding=0, bias=True)(pixel_values)
x = channels_first_layernorm(x, normalized_shape=C0, eps=1e-6)
```

Stage 1, no spatial downsample in source when `in_channels == out_channels` and `stride=1`:

```text
repeat depths[0]:
  residual = x                         # [B,C0,H/4,W/4]
  y = depthwise_conv7x7(x)              # groups=C0, padding=3
  y = permute_nchw_to_nhwc(y)
  y = LayerNorm(C0, eps=1e-6)(y)
  y = Linear(C0 -> 4*C0)(y)
  y = GELU(y)
  y = Linear(4*C0 -> C0)(y)
  y = layer_scale[C0] * y
  y = permute_nhwc_to_nchw(y)
  x = residual + y                      # DropPath is identity in inference
```

Stages 2-4:

```text
x = channels_first_layernorm(x, C_in, eps=1e-6)
x = Conv2d(C_in -> C_out, kernel=2, stride=2, padding=0, bias=True)(x)
repeat depths[i]:
  same ConvNeXT residual block at C_out
```

Classifier model:

```text
last_hidden_state = encoder(x)          # [B,C_last,H/32,W/32] for divisible inputs
pooled = mean(last_hidden_state, dims=[-2,-1])  # [B,C_last]
pooled = LayerNorm(C_last, eps=1e-12)(pooled)
logits = Linear(C_last -> num_labels)(pooled)
```

Concrete block dimensions:

| Variant | Stage channels | Pointwise MLPs per block |
| --- | --- | --- |
| Tiny/Small | 96,192,384,768 | 96->384->96, 192->768->192, 384->1536->384, 768->3072->768 |
| Base | 128,256,512,1024 | 128->512->128, 256->1024->256, 512->2048->512, 1024->4096->1024 |
| Large | 192,384,768,1536 | 192->768->192, 384->1536->384, 768->3072->768, 1536->6144->1536 |

## 6. Attention requirements

No attention is implemented or required for ConvNeXT. There is no MHA/MQA/GQA, mask, RoPE, ALiBi, packed sequence metadata, sliding window, FlashAttention path, or KV cache. Any Dinoml transformer-specific attention lowering should be guarded off for `model_type="convnext"`.

## 7. Position encoding and custom math

Plain ConvNeXT has no explicit position encoding. Spatial locality is carried by convolutions and downsampling.

Custom math to reproduce:

```python
def convnext_channels_first_layernorm(x, weight, bias, eps=1e-6):
    # Source-equivalent: x is [B, C, H, W], normalize C for each B,H,W.
    y = x.permute(0, 2, 3, 1)
    y = layer_norm(y, normalized_shape=[x.shape[1]], weight=weight, bias=bias, eps=eps)
    return y.permute(0, 3, 1, 2)
```

```python
def convnext_block(x, dwconv, ln, fc1, fc2, gamma):
    residual = x
    y = dwconv(x)
    y = y.permute(0, 2, 3, 1)
    y = ln(y)
    y = fc1(y)
    y = gelu(y)
    y = fc2(y)
    if gamma is not None:
        y = gamma * y
    y = y.permute(0, 3, 1, 2)
    return residual + y
```

ConvNeXT V2 note, out of scope for this report but important for future routing:

```python
def convnextv2_grn(x, weight, bias):
    # x is NHWC in the V2 block.
    gx = vector_norm(x, ord=2, dim=(1, 2), keepdim=True)
    nx = gx / (mean(gx, dim=-1, keepdim=True) + 1e-6)
    return weight * (x * nx) + bias + x
```

## 8. Preprocessing and input packing

The processor produces `pixel_values`; there are no token IDs or packed metadata.

Downloaded `preprocessor_config.json` facts:

| Model group | `size` | `crop_pct` | Resize behavior | Normalize |
| --- | ---: | ---: | --- | --- |
| 224 checkpoints | 224 | 0.875 | resize shortest edge to 256, center crop 224 | ImageNet mean/std |
| 384 checkpoint | 384 | null | current source path treats `shortest_edge >= 384` as square resize/warp to 384 | ImageNet mean/std |

The current `ConvNextImageProcessor` class defaults are slightly newer than the old downloaded feature extractor configs: class default `size={"shortest_edge":384}`, `crop_pct=224/256`, `default_to_square=False`, bicubic resampling, rescale, normalize. For parity, instantiate from the checkpoint preprocessor config instead of relying on class defaults.

Runtime graph boundary recommendation:

- First Dinoml integration should accept already-processed `pixel_values` `[B,3,S,S]` in float format.
- CPU/data pipeline should own resize/crop/rescale/normalize initially.
- A later GPU preprocessing path can fuse rescale+normalize and possibly NHWC conversion, but must match bicubic/crop semantics for end-to-end parity.

## 9. Graph rewrite / lowering opportunities

### Rewrite: stem non-overlap Conv2d -> Linear or GEMM

Source pattern:

```text
Conv2d(C_in=3, C_out=C0, kernel=patch_size, stride=patch_size, padding=0, groups=1)
```

Replacement:

```text
WindowFlatten([B,C,H,W], kh,kw,stride=kh,kw) -> GEMM(flat_patch, W_flat.T) -> BiasAdd -> Reshape [B,C0,H',W']
```

Preconditions:

- `kernel_size == stride == patch_size`.
- `padding == 0`, `dilation == 1`, `groups == 1`.
- Input channel equals config `num_channels`.
- Use Conv2d floor output shape for non-divisible H/W, or require divisibility for the optimized path and fall back otherwise.
- Flatten order must match PyTorch NCHW convolution weight order `[O,I,kH,kW]`.

Weight transform:

```python
w_flat = conv.weight.reshape(out_channels, in_channels * kh * kw)
bias = conv.bias
```

Failure cases: asymmetric pair patch sizes need pair-aware shape math; dynamic H/W need guarded shape cases; NHWC input requires a different flatten order or weight transform.

Parity test sketch: compare stem Conv2d+LayerNorm to WindowFlatten+GEMM+LayerNorm for random `[B,3,224,224]` and `[B,3,384,384]`.

### Rewrite: stage downsample Conv2d -> local GEMM

Source pattern:

```text
channels_first_layernorm(C_in) -> Conv2d(C_in -> C_out, kernel=2, stride=2)
```

Replacement is the same window-flatten GEMM pattern with `kh=kw=2`, `C_in` from previous stage, and `C_out` from next stage.

Preconditions:

- `kernel_size == stride == 2`, `padding == 0`, `groups == 1`, `dilation == 1`.
- Preserve the preceding channels-first LayerNorm exactly.
- Guard output shape against odd spatial dimensions or implement PyTorch floor behavior.

### Rewrite: NHWC pointwise Linear pair -> batched GEMM / 1x1 Conv

Source pattern:

```text
NHWC LayerNorm -> Linear(C -> 4C) -> GELU -> Linear(4C -> C)
```

Replacement:

```text
reshape [B,H,W,C] to [B*H*W,C]
GEMM + bias -> GELU -> GEMM + bias
reshape back to [B,H,W,C]
```

Preconditions:

- Input is logically NHWC and contiguous or represented by a layout-aware accessor.
- Linear weights preserve PyTorch shape `[out_features, in_features]`.
- Activation is a supported `ACT2FN` value; inspected configs use GELU.

Fusion opportunity: `Linear1 + GELU + Linear2` can be lowered through CUTLASS GEMM families only if the intermediate materialization policy is explicit. A custom 1x1 MLP kernel may beat two GEMMs for small C/spatial maps; benchmark before adding surface.

### Rewrite: channels-first LayerNorm without physical permutes

Source pattern:

```text
permute NCHW->NHWC -> LayerNorm(C) -> permute NHWC->NCHW
```

Replacement: native per-location channel LayerNorm on NCHW storage, or keep tensor in NHWC across a fully controlled region.

Preconditions:

- Normalized shape is exactly channel count C.
- No consumer observes the intermediate NHWC tensor.
- Epsilon matches the source site (`1e-6` except final pooled LN).
- For an NHWC region, all downstream Conv2d/depthwise Conv2d providers must accept NHWC or the pass must insert a correct boundary transform.

Failure cases: reducing over H/W by mistake, using final LN epsilon for block LN, or returning NHWC from `ConvNextBackbone` when callers expect NCHW.

### Rewrite: block layout-region elimination

Source pattern:

```text
NCHW depthwise conv -> permute -> LN/Linear/GELU/Linear/layer-scale -> permute -> residual add
```

Replacement candidates:

- Keep block input/output NCHW and fuse only the two permutes into channel-axis-aware kernels.
- Or translate the entire block/stage to NHWC if depthwise conv, pointwise MLP, residual add, and downsample boundaries are all layout-aware.

Required axis rewrites:

- LayerNorm channel axis `C`: NCHW axis 1 becomes NHWC axis -1.
- Global mean `[-2,-1]`: NCHW H/W becomes NHWC `[1,2]`.
- Conv weights need provider-specific OIHW/NHWC handling.

No-layout-translation guards:

- Public `pixel_values` ABI.
- Public `last_hidden_state` and `BackboneOutput.feature_maps` unless the API explicitly advertises NHWC.
- Any shape/debug hidden-state capture expected to match Transformers.

## 10. Kernel fusion candidates

Highest priority:

- Depthwise Conv2d 7x7 NCHW/NHWC provider path. It appears in every block and dominates unique convolution coverage beyond GEMM.
- Channel LayerNorm for 4D tensors. ConvNeXT uses it in stem, downsample, block, and backbone output paths.
- Pointwise MLP GEMMs over `[B*H*W, C]`, including GELU and bias. These are repeated per block and map naturally to existing GEMM infrastructure.
- Layout-region canonicalization around NCHW<->NHWC permutes. Avoiding physical transposes is likely crucial.

Medium priority:

- Stem/downsample Conv2d-to-GEMM rewrite for non-overlapping kernels.
- Fused `LayerNorm + Linear(C->4C)` for NHWC block input, if profiling shows bandwidth pressure.
- Fused `Linear + GELU` or `Linear + GELU + Linear` pointwise MLP for common C values.
- Global average pool + final LayerNorm, especially for batch throughput.

Lower priority:

- Backbone feature-map normalization and output packing.
- GPU image preprocessing. Useful end-to-end, but not needed for first graph parity if `pixel_values` are supplied.
- ConvNeXT V2 GRN. Important for a separate `convnextv2` report, not required here.

## 11. Runtime staging plan

Stage 1: config/weight loader and processor contract.

- Parse `ConvNextConfig`.
- Load Conv2d, LayerNorm, Linear, layer-scale, and classifier weights.
- Accept preprocessed `pixel_values` only.

Stage 2: one-block parity.

- Implement/compose depthwise Conv2d, NHWC LayerNorm, pointwise Linear, GELU, layer-scale multiply, permutes, residual add.
- Validate each stage channel size.

Stage 3: encoder parity, NCHW-faithful.

- Lower stem, all stages, and final pooler using explicit permutes.
- Defer layout optimization until source-faithful parity is stable.

Stage 4: classifier parity.

- Add final Linear head and ImageNet 1000-label logits.
- Validate tiny/base/large shape sweeps.

Stage 5: backbone optional path.

- Implement `out_features`/`out_indices` and feature-map normalization.

Stage 6: optimized layout/fusion passes.

- Add guarded NCHW/NHWC region rewrites.
- Add Conv2d-to-GEMM rewrites for stem/downsample.
- Add depthwise and channel-LN fused kernels if profiling justifies them.

Stage 7: processor integration.

- Optionally add CPU or GPU resize/crop/rescale/normalize pipeline for end-to-end image parity.

## 12. Parity and validation plan

Random tensor tests:

- Channels-first LayerNorm against Transformers for `[B,C,H,W]` across C in `96,128,192,384,768,1536`.
- Depthwise Conv2d 7x7 groups=C with padding 3.
- Conv2d stem/downsample shape and values for even and odd spatial dimensions.
- NHWC pointwise Linear/GELU/Linear block core.
- Layer-scale broadcast `[C]` over `[B,H,W,C]`.
- Spatial mean over H/W.

Model tests:

- Single stem parity for tiny/base/large.
- Single ConvNeXT block parity per channel width.
- Stage-by-stage parity after each of four stages.
- Full `ConvNextModel` last hidden state and pooler parity.
- Full `ConvNextForImageClassification` logits parity for `tiny-224`, `base-224`, `large-384`.
- Backbone feature-map parity for selected `out_features`.

Suggested tolerances:

- fp32 source-faithful: `rtol=1e-4`, `atol=1e-5` for block/stage; logits may need `rtol=2e-4` after many layers.
- fp16/bf16 optimized: compare against PyTorch reduced-precision run, not fp32, with tolerance chosen per accumulated Conv/GEMM precision. Keep preprocessing in fp32 for first parity.

End-to-end image parity:

- Use HF processor from each checkpoint config.
- Compare top-1/top-5 logits ordering and raw logits against Transformers for a fixed public image.
- Separate processor mismatch from model mismatch by saving and reusing `pixel_values`.

## 13. Performance probes

- Processor throughput: PIL/torchvision preprocessing images/sec for 224 and 384.
- Stem/downsample Conv2d time versus Conv2d-to-GEMM rewrite.
- Depthwise 7x7 Conv2d throughput by channel/spatial size:
  - 224 tiny: `56x56 C=96`, `28x28 C=192`, `14x14 C=384`, `7x7 C=768`.
  - 384 large: `96x96 C=192`, `48x48 C=384`, `24x24 C=768`, `12x12 C=1536`.
- Pointwise MLP GEMM throughput for `[B*H*W,C] x [C,4C]` and `[B*H*W,4C] x [4C,C]`.
- Physical permute cost per block versus layout-elided kernels.
- Encoder-only latency and throughput by batch size.
- Classification head cost; expected to be small, but measure for large batch.
- NCHW baseline versus guarded NHWC/channel-last optimized region.

No benchmark observations are included here; these are source-derived probe recommendations.

## 14. Skip/defer list

- Training loss and labels.
- DropPath stochastic training behavior.
- Gradient checkpointing and backpropagation.
- Quantization and compressed weights.
- Multi-GPU/tensor parallelism.
- Dynamic public image preprocessing on GPU.
- ConvNeXT V2 GRN and `model_type="convnextv2"` checkpoints.
- Detection/segmentation heads that consume `ConvNextBackbone`, except for optional normalized feature-map extraction.
- Any attention, KV-cache, or generation machinery.

## 15. Final implementation checklist

- [ ] Parse `ConvNextConfig` fields: `num_channels`, `patch_size`, `hidden_sizes`, `depths`, `hidden_act`, `layer_norm_eps`, `layer_scale_init_value`, `out_features`.
- [ ] Load stem, stage, block, pooler, and classifier weights with PyTorch-compatible names and shapes.
- [ ] Accept preprocessed NCHW `pixel_values` `[B,3,H,W]`.
- [ ] Implement Conv2d stem with kernel/stride `patch_size`.
- [ ] Implement stage downsample Conv2d kernel 2 stride 2.
- [ ] Implement depthwise Conv2d 7x7 padding 3 groups=C.
- [ ] Implement channels-first LayerNorm as channel-axis LN with eps `1e-6`.
- [ ] Implement NHWC LayerNorm and pointwise Linear block path.
- [ ] Implement GELU via `ACT2FN`-compatible semantics.
- [ ] Implement layer-scale `[C]` broadcast multiply and residual add.
- [ ] Implement global mean over source spatial axes and final LayerNorm eps `config.layer_norm_eps`.
- [ ] Implement classifier Linear `C_last -> 1000` for inspected checkpoints.
- [ ] Add one-block, one-stage, full-encoder, and logits parity tests.
- [ ] Add 224 and 384 processor-contract parity fixtures using saved `pixel_values`.
- [ ] Add guarded Conv2d-to-GEMM rewrite for stem/downsample.
- [ ] Add guarded layout-region rewrite to remove NCHW/NHWC physical permutes.
- [ ] Benchmark depthwise conv, pointwise MLP GEMMs, layout transforms, encoder-only, and end-to-end classification.
