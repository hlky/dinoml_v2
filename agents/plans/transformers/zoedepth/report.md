# ZoeDepth DinoML Operator Assessment

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: Intel/zoedepth-nyu, Intel/zoedepth-kitti, Intel/zoedepth-nyu-kitti
Config source: official Hugging Face config.json and preprocessor_config.json snapshots
Source files inspected:
  X:/H/transformers/src/transformers/models/zoedepth/configuration_zoedepth.py
  X:/H/transformers/src/transformers/models/zoedepth/modeling_zoedepth.py
  X:/H/transformers/src/transformers/models/zoedepth/image_processing_zoedepth.py
  X:/H/transformers/src/transformers/models/zoedepth/image_processing_pil_zoedepth.py
Any missing files or assumptions:
  The native source composes a BEiT backbone through load_backbone(config). This report owns the ZoeDepth neck/head and records
  the BEiT feature contract consumed by ZoeDepth, but detailed BEiT operator coverage should reuse a separate BEiT audit.
  hf-internal-testing/tiny-random-ZoeDepthForDepthEstimation returned 401 and was not used as representative.
```

Snapshots are stored under `agents/plans/transformers/zoedepth/_sources/`.

## 2. High-level architecture

ZoeDepth is an image-to-metric-depth estimator:

```text
CPU/image preprocessing -> BEiT backbone feature maps -> DPT-style reassemble/neck/fusion -> relative depth head
  -> metric bin/attractor head -> predicted depth -> postprocess resize/remove padding
```

Runtime stages:

- CPU/data pipeline: rescale, reflect pad, aspect-preserving resize to a multiple of 32, normalize.
- Backbone: BEiT image encoder returns four selected hidden states from `stage6`, `stage12`, `stage18`, and `stage24`.
- Neck: reassemble token sequences into NCHW feature maps, project channels, resize, and fuse top-down.
- Heads: relative depth conv head plus metric depth head. Single-domain checkpoints use one bin head; `Intel/zoedepth-nyu-kitti` adds a tiny transformer classifier to choose between NYU and KITTI bin heads.
- Postprocessing: optional flipped-output averaging, bicubic resize to source size plus synthetic padding, crop padding, optional resize to target size.

## 3. Important config dimensions

Representative checkpoint sweep:

| Field | `Intel/zoedepth-nyu` | `Intel/zoedepth-kitti` | `Intel/zoedepth-nyu-kitti` | Source/default note |
|---|---:|---:|---:|---|
| `model_type` | `zoedepth` | `zoedepth` | `zoedepth` | config |
| `backbone_config.model_type` | `beit` | `beit` | `beit` | config/default |
| backbone hidden size | 1024 | 1024 | 1024 | config |
| backbone layers | 24 | 24 | 24 | config |
| backbone heads | 16 | 16 | 16 | config |
| backbone patch size | 16 | 16 | 16 | omitted in sampled configs; effective BEiT default |
| backbone features | stages 6/12/18/24 | stages 6/12/18/24 | stages 6/12/18/24 | token sequences, `reshape_hidden_states=false` |
| preprocessing size | 384 x 512 | 384 x 512 | 384 x 512 | preprocessor config |
| patch grid at 384x512 | 24 x 32 | 24 x 32 | 24 x 32 | inference from input size / patch size |
| neck hidden sizes | 256,512,1024,1024 | same | same | config |
| reassemble factors | 4,2,1,0.5 | same | same | source default; configs omit |
| fusion hidden size | 256 | 256 | 256 | config |
| relative features | 32 | 32 | 32 | source default; configs omit |
| bottleneck features | 256 | 256 | 256 | config |
| bin embedding dim | 128 | 128 | 128 | source default; configs omit |
| bin centers type | `softplus` | `normed` | `softplus` | config |
| bin configurations | NYU 64 bins, 0.001..10 | config says NYU 64 bins, 0.001..10 | NYU 64 bins 0.001..10; KITTI 64 bins 0.001..80 | config |
| attractors per stage | 16,8,4,1 | same | same | config |
| patch transformer layers | n/a | n/a | 4 | only multi-head routing |
| patch transformer hidden/heads | n/a | n/a | 128 / 4 | only multi-head routing |
| dtype | not fixed | not fixed | not fixed | no runtime dtype requirement in config |
| cache support | none | none | none | image encoder only, no generation cache |

## 3a. Family variation traps

- The sampled public checkpoints all compose BEiT-large-like token features, not a standalone ZoeDepth encoder. Treat BEiT as a nested backbone contract.
- `reshape_hidden_states=false` is essential: ZoeDepth reassembles token sequences itself and expects a CLS token plus patch tokens.
- `swinv2` backbones skip the reassemble stage in source and expect image-like feature maps. This is a separate feature contract, even though sampled Intel configs use BEiT.
- Single bin configuration uses `ZoeDepthMetricDepthEstimationHead`; multiple configurations use `ZoeDepthMultipleMetricDepthEstimationHeads` with classifier-driven runtime branch selection.
- `bin_centers_type` changes math: `softplus` uses unbounded centers and memory-efficient attractor loop; `normed` normalizes widths, cumsums edges, sorts/clips centers.
- Multi-head routing calls `.item()` on `argmax(domain_vote)`, which is a graph break/dynamic host branch in eager PyTorch. DinoML should either specialize per selected domain or model this as an explicit control-flow boundary.
- The preprocessing `do_pad` and postprocess `do_remove_padding` are coupled through source image sizes; end-to-end parity requires passing original source sizes.
- Layout-sensitive axes are common: channel concat and reductions use `dim=1`, token reshape uses `(B, H, W, C) -> NCHW`, and bin sorting/reduction happens over channel/bin axis.

## 4. Operator coverage checklist

### Tensor/layout ops

- NCHW tensor input/output, reshape, flatten, permute, contiguous.
- Token-to-grid reassembly: `(B, 1 + H_p*W_p, 1024)` -> drop/read CLS -> `(B, 1024, H_p, W_p)`.
- Concatenate along channel/bin axis: `dim=1` for condition features, bin/probability tensors, and relative-depth conditioning.
- Split/slice first two channels from `Conv2d(... -> 4)` outputs for probability and temperature.
- `pad` on channel/bin axis for normed bin widths; reflect pad in image preprocessing.
- `sort(dim=1)`, `clip`, `cumsum(dim=1)`, `sum(dim=1)`, `mean/sum(dim=1)` attractor reductions.
- `argmax`, host-visible scalar branch for multi-domain bin-head selection.

### Neural network primitives

- Nested BEiT backbone primitives: patch Conv2d/embedding, encoder MHA with relative position bias, LayerNorm, GELU MLP.
- Reassemble projections: `Conv2d(1024 -> 256/512/1024/1024, kernel=1)` plus resize by factors 4, 2, 1, 0.5.
- Fusion projections: `Conv2d(neck_channels -> 256, kernel=3, padding=1, bias=False)`.
- Residual fusion units: ReLU -> `Conv2d(256 -> 256, 3x3)` -> optional BatchNorm -> ReLU -> `Conv2d(256 -> 256, 3x3)` -> optional BatchNorm -> residual add.
- Relative head: `Conv2d(256 -> 128, 3x3)`, bilinear upsample x2, `Conv2d(128 -> 32, 3x3)`, ReLU, `Conv2d(32 -> 1, 1x1)`, ReLU.
- Metric head: `Conv2d(256 -> 256, 1x1)`, seed bin regressor `Conv2d(256 -> 64, 1x1)` then `Conv2d(64 -> n_bins, 1x1)`, projectors `Conv2d(C -> 128, 1x1)` via hidden 128.
- Conditional log-binomial MLP: single head uses `Conv2d(33 + 128 -> 80, 1x1) -> GELU -> Conv2d(80 -> 4, 1x1) -> Softplus`; multi-head uses `Conv2d(32 + 128 -> 40, 1x1) -> GELU -> Conv2d(40 -> 4, 1x1) -> Softplus`.

### Attention primitives

- Required through BEiT backbone: noncausal image self-attention with BEiT relative position bias.
- Only multi-domain ZoeDepth head: noncausal MHA, hidden 128, 4 heads, head dim 32, no cache, sequence length `1 + H_b*W_b` after 1x1 bottleneck embedding.

### Position/custom math ops

- BEiT relative position bias from nested backbone.
- Multi-head patch transformer 1D sinusoidal position encoding generated dynamically from sequence length and hidden size.
- Attractor/bin math: log, clamp, softplus/ReLU, pad, cumsum, interpolate, sort, clip, `dx / (1 + alpha * dx**gamma)`.

### Generation/cache ops

- None. No autoregressive decode, KV cache, beam search, or token sampling.

### Preprocessing-coupled ops

- Rescale, reflect pad, bilinear resize with `align_corners=True`, ImageNet normalization.
- Postprocess bicubic resize with `antialias=False`, padding crop, optional horizontal flip averaging.

## 5. Layer/block breakdown

BEiT backbone, composed from `backbone_config`:

```text
pixel_values: [B, 3, H, W]
backbone -> feature_maps:
  4 token tensors, each [B, 1 + (H/16)*(W/16), 1024] for BEiT configs with reshape_hidden_states=false
```

Reassemble + neck for BEiT token features:

```text
for each selected stage:
  cls = x[:, 0]
  patches = x[:, 1:].reshape(B, H/16, W/16, 1024).permute(0,3,1,2)
  if readout_type == "project":
    patches = Linear(2048 -> 1024) + GELU over per-token concat([patch, cls])
  patches = Conv2d(1024 -> neck_C, 1x1, bias=True)
  patches = resize: ConvTranspose2d for factors 4/2, Identity for 1, Conv2d stride 2 for 0.5
  patches = Conv2d(neck_C -> 256, 3x3, padding=1, bias=False)
top-down fusion:
  process deepest first
  optional residual feature is interpolated to current size with bilinear align_corners=False
  residual branch = ReLU -> Conv2d(256 -> 256, 3x3) -> ReLU -> Conv2d(256 -> 256, 3x3)
  hidden = hidden + residual_branch
  hidden = second residual conv unit
  hidden = bilinear upsample x2, align_corners=True
  hidden = Conv2d(256 -> 256, 1x1, bias=True)
```

Relative depth head:

```text
select hidden_states[head_in_index]  # default -1, last fused map
optional Conv2d(256 -> 256, 3x3) + ReLU if add_projection
x = Conv2d(256 -> 128, 3x3)
x = bilinear upsample x2, align_corners=True
x = Conv2d(128 -> 32, 3x3)
features = ReLU(x)
relative_depth = ReLU(Conv2d(32 -> 1, 1x1)).squeeze(1)
```

Single metric head:

```text
bottleneck = Conv2d(256 -> 256, 1x1)(features[-1])
seed = Conv2d(256 -> 64, 1x1) -> ReLU -> Conv2d(64 -> 64 bins, 1x1) -> ReLU or Softplus
prev_bin = normalized seed centers for normed checkpoints, otherwise seed centers
prev_embed = Projector(256 -> 128)
for four fused feature blocks:
  bin_embed = Projector(256 -> 128)
  prev_bin, bin_centers = Attractor(bin_embed, prev_bin, prev_embed)
  prev_embed = bin_embed
last = concat(outconv_activation[32], resized relative_depth[1])  # [B,33,H,W]
prob = ConditionalLogBinomial(last, resized bin_embed)  # [B, n_bins, H, W]
metric_depth = sum(prob * resized_bin_centers, dim=1, keepdim=True).squeeze(1)
```

Multi metric head:

```text
x = Conv2d(256 -> 256, 1x1)(bottleneck)
emb = Conv2d(256 -> 128, 1x1)(x).flatten(2)
emb = pad one CLS-like zero token, add 1D sinusoidal position encoding
emb = 4 x TransformerEncoderLayer(hidden=128, heads=4, intermediate=128)
domain_logits = MLP(128 -> 128 -> 2)(emb[:,0])
domain_vote = softmax(domain_logits.sum(dim=0, keepdim=True), dim=-1)
selected head = argmax(domain_vote).item()
run selected seed/projector/attractor/conditional-log-binomial path
```

Most Conv2d layers use bias by PyTorch default unless the source passes `bias=False`.

## 6. Attention requirements

ZoeDepth itself is not a generation model and has no KV cache. Attention appears in two places:

- BEiT nested backbone: noncausal image self-attention over patch tokens plus CLS, with relative position bias. Reuse the BEiT audit/operator coverage.
- Multi-domain metric routing only: standard MHA over bottleneck patch embeddings.
  - Self-attention, noncausal.
  - MHA, not MQA/GQA.
  - Heads: 4, hidden 128, head dim 32.
  - Q/K/V are separate `Linear(128 -> 128)` with bias; output projection `Linear(128 -> 128)` with bias.
  - Query length equals key/value length `1 + H_b*W_b`.
  - Attention mask is optional but the ZoeDepth caller does not construct one.
  - Eager implementation uses explicit matmul, divide by sqrt(head_dim), mask add, softmax, dropout, matmul. Dropout is inactive in eval but still present in source.
  - SDPA/FlashAttention are not used by the ZoeDepth head; source comment explicitly avoids sdpa/flash support in the base class.

## 7. Position encoding and custom math

Multi-head routing patch transformer uses generated 1D sinusoidal positions:

```python
def zoe_posenc(batch, seq, dim, dtype):
    pos = arange(seq, dtype=dtype)[:, None]
    idx = arange(0, dim, 2, dtype=dtype)[None, :]
    div = exp(idx * (-log(10000.0) / dim))
    pe = concat([sin(pos * div), cos(pos * div)], dim=1)
    return pe[None].repeat(batch, 1, 1)
```

Conditional log-binomial distribution:

```python
def log_binomial_distribution(p, temp, k):
    idx = arange(k).view(1, k, 1, 1)
    n = (k - 1)
    p = clamp(p, 1e-4, 1)
    q = clamp(1 - p, 1e-4, 1)
    y = log_binom(n, idx) + idx * log(p) + (n - idx) * log(q)
    return softmax(y / temp, dim=1)
```

Attractor update:

```python
def inv_attractor(dx, alpha=300, gamma=2):
    return dx / (1 + alpha * dx.pow(gamma))

def update_bins(attractors, bins, kind="mean"):
    delta = inv_attractor(attractors[:, :, None] - bins[:, None])
    delta = delta.mean(dim=1) if kind == "mean" else delta.sum(dim=1)
    return bins + delta
```

Source-specific details:

- `ZoeDepthAttractorLayer` reads `config.attractor_alpha` and stores `self.gemma = config.attractor_gamma`, but calls `inv_attractor(...)` without passing either value, so the scripted defaults `alpha=300, gamma=2` control the math in the vectorized normed path.
- `ZoeDepthAttractorLayerUnnormed` sets `self.gamma = config.attractor_alpha` and also calls `inv_attractor(...)` without passing alpha/gamma. Treat config alpha/gamma as currently ignored by the native source.
- Normed centers are sorted on `dim=1` and clipped to `[min_depth, max_depth]`; softplus centers are not sorted/clipped in the unnormed path.
- Precompute candidates: `log_binom(k-1, idx)`, `k_idx`, `k_minus_1`, and fixed sinusoidal encodings for fixed spatial shapes. Dynamic input sizes change sequence length and postprocess crop sizes.

## 8. Preprocessing and input packing

Processor contract:

```text
input images -> float tensor, channel-first
rescale by image processor default factor
reflect pad: pad_h=int(sqrt(H/2)*3), pad_w=int(sqrt(W/2)*3)
resize: aspect-preserving toward 384x512, constrain dimensions to multiple of 32
normalize with ImageNet mean/std
output pixel_values: [B, 3, H_resized, W_resized]
```

Official preprocessor configs set `do_pad=true`, `do_resize=true`, `size={height:384,width:512}`, `keep_aspect_ratio=true`, `ensure_multiple_of=32`, `resample=2` (bilinear).

GPU/runtime graph receives only `pixel_values`. Original source sizes are postprocessing metadata, not model inputs, but they are required if `do_remove_padding=True`.

Postprocess contract:

- Raw `predicted_depth` shape is `[B, H_out, W_out]`.
- Optional `outputs_flipped` must match shape; final depth averages `predicted_depth` with horizontally flipped flipped-output depth.
- If source size is supplied, resize raw depth to `(source_h + 2*pad_h, source_w + 2*pad_w)` using bicubic, `antialias=False`, then crop `pad_h/pad_w`.
- If target size is supplied, resize again to target `(height, width)` using bicubic, `antialias=False`.
- Output is a list of per-image dictionaries with `predicted_depth: [H, W]`.

## 9. Graph rewrite / lowering opportunities

### Rewrite: BEiT patch Conv2d -> Linear

Preconditions:

- Apply inside the nested BEiT backbone only when patch embedding has `kernel_size == stride == patch_size`, `padding == 0`, `dilation == 1`, `groups == 1`.
- Input H/W must be divisible by patch size after preprocessing.
- Preserve BEiT positional/relative-bias behavior separately.

Replacement:

```text
WindowFlatten(NCHW or guarded NHWC) -> MatMul(weight_flat.T) -> BiasAdd -> token reshape
```

Weight transform:

```python
w = conv.weight.reshape(out_channels, in_channels * patch_h * patch_w)
```

Failure cases: dynamic non-divisible sizes, non-BEiT backbones, or any checkpoint with different patch embedding semantics.

### Rewrite: ZoeDepth 1x1 Conv2d -> pointwise GEMM

Preconditions:

- `kernel_size == 1`, `stride == 1`, `padding == 0`, `groups == 1`.
- Source tensor is image-like `[B,C,H,W]`; axis rewrite must be explicit if optimized NHWC is used.

Replacement:

```text
NCHW -> optional NHWC layout region -> MatMul(Cin -> Cout) + bias -> restore consumer layout if needed
```

Applies to reassemble projections, bottleneck convs, projectors, seed heads, and conditional-log-binomial MLP. Failure case: consumers requiring source NCHW and no enclosing layout/fusion region.

### Rewrite: top-down fusion interpolation + residual conv region

Preconditions:

- Inputs are all NCHW feature maps from ZoeDepth neck/fusion.
- Bilinear `align_corners` values must match source: residual matching uses `False`, top-down x2 uses `True`.
- Channel axis remains 1 unless a full guarded NHWC region rewrites concat/reduction axes and Conv weights.

Replacement:

```text
Resize -> residual Conv/ReLU/Conv -> Add -> residual Conv/ReLU/Conv -> Resize x2 -> 1x1 projection
```

Layout constraints: good candidate for a `no_translation()` boundary at public neck input/output, with optional local NHWC inside conv-heavy fusion blocks only if all consumers are controlled.

### Rewrite: conditional log-binomial constants

Preconditions:

- `n_bins` fixed by selected bin configuration.
- dtype/device known for constant materialization.

Replacement:

```text
precompute log_binom(k-1, idx), idx, k_minus_1 -> elementwise log/clamp/softmax
```

Failure cases: dynamic `n_bins` or mixed selected heads in one batch without bucketing/specialization.

### Rewrite: multi-head branch specialization

Preconditions:

- Deployment admits one depth domain per compiled graph or batches are bucketed by selected domain.
- Domain classifier output is either precomputed outside the fused metric head or a conservative fallback handles ambiguous/mixed batches.

Replacement:

```text
domain-specific metric head graph for "nyu" or "kitti"
```

Failure cases: source parity for a batch where summed domain logits select a different single path than per-sample routing would; source uses batch-global `domain_logits.sum(dim=0)`.

## 10. Kernel fusion candidates

Highest priority:

- BEiT encoder kernels from the nested backbone: MHA with relative position bias, LayerNorm, GELU MLP, and patch embedding dominate compute.
- Conv-heavy DPT/ZoeDepth neck fusion in NHWC-local kernels: repeated 3x3/1x1 convs and bilinear resizes operate at image-map resolution.
- Conditional log-binomial + weighted bin sum: avoids materializing large `[B,64,H,W]` probability and bin tensors where possible.

Medium priority:

- Attractor update kernel: broadcast `[B,A,1,H,W] - [B,1,N,H,W]`, inverse update, reduce over attractors, sort/clip for normed path.
- Projector chains: fuse 1x1 Conv + ReLU + 1x1 Conv where channel counts are fixed.
- Postprocess resize/crop pipeline for batch throughput.

Lower priority:

- Multi-domain patch transformer routing: only used by `zoedepth-nyu-kitti`, hidden 128 and 4 layers; correctness matters more than peak performance.
- Optional BatchNorm in fusion residual blocks; sampled Intel configs use source default without fusion BatchNorm.

## 11. Runtime staging plan

1. Parse ZoeDepth config plus nested `backbone_config`; reject or defer non-BEiT/SwinV2 backbones until separately validated.
2. Load weights and verify BEiT feature-map extraction contract from the existing BEiT path.
3. Implement token reassembly + neck fusion parity for fixed 384x512 preprocessed input.
4. Implement relative depth head parity.
5. Implement single metric head for `Intel/zoedepth-nyu` and `Intel/zoedepth-kitti`, including softplus and normed bin center paths.
6. Add multi-domain routing for `Intel/zoedepth-nyu-kitti`; initially allow a graph break/domain-specialized path.
7. Implement processor/postprocessor parity around reflect padding, align-corners resize, bicubic resize/crop.
8. Add NHWC-local conv/layout fusions and log-binomial/attractor fusion after source-faithful parity.

Can be stubbed initially: training labels/loss, output attentions/hidden states beyond backbone pass-through, flipped-output test-time augmentation, SwinV2 feature-map path.

## 12. Parity and validation plan

- Custom op tests: `get_resize_output_image_size`, reflect padding sizes, log-binomial constants, normed seed bin cumsum, softplus seed path, attractor update, conditional weighted sum.
- Reassemble parity: synthetic `[B, 1+H*W, 1024]` feature maps through readout/project and resize factors 4/2/1/0.5.
- Neck parity: four feature maps through fusion with odd/even spatial sizes to catch interpolation `align_corners` and shape matching.
- Head parity: relative head, single metric head softplus, single metric head normed, multi-head routing with deterministic logits.
- End-to-end fixed image parity: preprocessed 384x512 input, compare raw `predicted_depth`.
- Postprocess parity: with source sizes, target sizes, `do_remove_padding` true/false, and flipped outputs.
- Tolerances: fp32 `rtol=1e-4, atol=1e-4`; fp16/bf16 use `rtol=5e-2, atol=5e-2` for full depth maps because bin sorting/log/softmax/resize amplify small differences.

## 13. Performance probes

- Processor throughput: rescale/pad/resize/normalize for mixed image sizes.
- BEiT backbone throughput and memory at 384x512 and other multiples of 32.
- Neck-only throughput: reassemble + fusion maps.
- Relative head and metric head separately, including materialized `[B,64,H,W]` tensors.
- Attractor update kernel timings for `n_attractors` 16/8/4/1 and `n_bins=64`.
- Single-head versus multi-domain routing overhead.
- Batch-size sweep and resolution sweep after padding/resizing.
- NCHW baseline versus guarded NHWC conv/fusion regions.
- Postprocess resize/crop throughput for large original images.

## 14. Skip/defer list

- Training and loss; source raises `NotImplementedError` for labels.
- Autoregressive generation, KV cache, beam search, sampling.
- Non-BEiT and SwinV2 ZoeDepth backbones until representative checkpoints are selected.
- Multi-GPU tensor parallelism.
- Quantization/packed weights; no source-coupled packed format is present here.
- Flipped-output test-time augmentation can be added after single-pass parity.

## 15. Final implementation checklist

- [ ] Parse `ZoeDepthConfig` and nested `backbone_config`.
- [ ] Load BEiT backbone weights and ZoeDepth neck/head weights.
- [ ] Route BEiT feature maps with `reshape_hidden_states=false`.
- [ ] Implement readout token handling: `ignore`, `add`, `project`.
- [ ] Implement reassemble Conv/ConvTranspose/stride-Conv resize path.
- [ ] Implement DPT-style feature fusion with exact `align_corners` settings.
- [ ] Implement relative depth head.
- [ ] Implement seed bin regressor for `softplus` and `normed`.
- [ ] Implement attractor update and bin center sort/clip behavior.
- [ ] Implement conditional log-binomial softmax and weighted depth sum.
- [ ] Implement multi-domain patch transformer routing and graph-break/specialization policy.
- [ ] Implement ZoeDepth image preprocessing and depth postprocessing.
- [ ] Add guarded NHWC/local conv fusion candidates.
- [ ] Add parity tests for neck, heads, processor, postprocessor, and end-to-end depth.
- [ ] Benchmark backbone, neck, metric head, postprocess, and layout variants.
