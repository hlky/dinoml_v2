# PVT DinoML Operator Audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: Zetatech/pvt-tiny-224, Zetatech/pvt-small-224, Zetatech/pvt-medium-224, Zetatech/pvt-large-224, Xrenya/pvt-tiny-224
Config source: HF config.json snapshots plus configuration_pvt.py defaults
Source files inspected:
- transformers/src/transformers/models/pvt/configuration_pvt.py
- transformers/src/transformers/models/pvt/modeling_pvt.py
- transformers/src/transformers/models/pvt/image_processing_pvt.py
- transformers/src/transformers/models/pvt/image_processing_pil_pvt.py
- transformers/src/transformers/models/pvt/convert_pvt_to_pytorch.py
Any missing files or assumptions: no remote code required; no gated files observed. Source snapshots and HF JSON snapshots are under agents/plans/transformers/pvt/_sources/.
```

Primary DinoML target: image classification with `PvtForImageClassification`; base encoder feature parity is independently useful. DinoML assumptions: inference-only first, CUDA target, NHWC preferred only as guarded layout/fusion optimization, and initial translation should preserve PyTorch/NCHW axes.

## 2. High-level architecture

PVT is a hierarchical vision Transformer encoder. Each stage applies a Conv2d patch embedding from NCHW image/feature maps into `[batch, tokens, channels]`, runs several encoder blocks with efficient sequence-reduction self-attention, then reshapes non-final stages back to NCHW for the next stage. The final stage prepends a CLS token and the classifier reads `sequence_output[:, 0, :]`.

```text
CPU image preprocessing -> NCHW pixel_values -> 4-stage PVT encoder -> final CLS token -> linear classifier logits
```

Stage decomposition:

- CPU/data pipeline: resize to 224x224, rescale, normalize, emit `pixel_values` as NCHW.
- GPU encoder: Conv2d patch embeddings, sequence/NCHW reshapes, LayerNorm, noncausal self-attention with K/V spatial reduction, MLP.
- Head: CLS gather and Linear(`hidden_sizes[-1] -> num_labels`).

## 3. Important config dimensions

| Field | Default/source value | Notes |
|---|---:|---|
| image_size | 224 | HF configs use 224. Source has a dynamic interpolation path but initial DinoML should reject or carefully validate non-config sizes. |
| num_channels | 3 | Input `pixel_values` are NCHW. |
| num_encoder_blocks | 4 | Four hierarchical stages. |
| hidden_sizes | `[64, 128, 320, 512]` | Same across sampled tiny/small/medium/large configs. |
| depths | `[2, 2, 2, 2]` default | Varies by checkpoint; see sweep. |
| patch_sizes / strides | `[4, 2, 2, 2]` / `[4, 2, 2, 2]` | Patch embedding Conv2d uses `kernel_size=stride`, `stride=patch_size`; equivalent for sampled configs. |
| sequence_reduction_ratios | `[8, 4, 2, 1]` | Stages 0-2 reduce K/V tokens with strided Conv2d; stage 3 does full K/V length. |
| num_attention_heads | `[1, 2, 5, 8]` | Head dims are `[64, 64, 64, 64]`. |
| mlp_ratios | `[8, 8, 4, 4]` | MLP intermediate sizes per stage are `[512, 1024, 1280, 2048]`. |
| hidden_act | `gelu` | ACT2FN lookup. |
| layer_norm_eps | `1e-6` | Used in patch embeddings, blocks, SR path, final norm. |
| qkv_bias | `true` | Source creates separate biased Q, K, V Linear layers. |
| dropout | `0.0` in sampled configs | DropPath is also `0.0` in sampled HF configs, but source implements training stochastic depth. |
| dtype | `float32` in HF configs | From `config.json`. |
| cache support | none | Encoder-only image model; no KV cache. |

Representative checkpoint sweep:

| HF config | depths | hidden_sizes | heads | head_dim | SR ratios | MLP ratios | dtype |
|---|---:|---:|---:|---:|---:|---:|---|
| `Zetatech/pvt-tiny-224` | `[2,2,2,2]` | `[64,128,320,512]` | `[1,2,5,8]` | 64 | `[8,4,2,1]` | `[8,8,4,4]` | float32 |
| `Xrenya/pvt-tiny-224` | `[2,2,2,2]` | `[64,128,320,512]` | `[1,2,5,8]` | 64 | `[8,4,2,1]` | `[8,8,4,4]` | float32 |
| `Zetatech/pvt-small-224` | `[3,4,6,3]` | `[64,128,320,512]` | `[1,2,5,8]` | 64 | `[8,4,2,1]` | `[8,8,4,4]` | float32 |
| `Zetatech/pvt-medium-224` | `[3,4,18,3]` | `[64,128,320,512]` | `[1,2,5,8]` | 64 | `[8,4,2,1]` | `[8,8,4,4]` | float32 |
| `Zetatech/pvt-large-224` | `[3,8,27,3]` | `[64,128,320,512]` | `[1,2,5,8]` | 64 | `[8,4,2,1]` | `[8,8,4,4]` | float32 |

## 3a. Family variation traps

- Depth is the main sampled variant: medium/large greatly increase stage-3 block count, where token count is 196 at 224 input and SR ratio is 2.
- `hidden_size == num_heads * head_dim` for sampled configs, but heads are per-stage; do not assume a single global head count.
- Final stage has a CLS token; earlier stages do not. Classifier parity depends on preserving final token order.
- Source checkpoint conversion splits original packed `kv` weights into separate K then V matrices. Native HF weights are separate K/V; importers for original PVT checkpoints must preserve K-before-V split order.
- HF configs include `reshape_last_stage: true`, but inspected `modeling_pvt.py` does not read it. Treat it as ignored for this source basis.
- Patch embeddings and SR attention cross the sequence/NCHW boundary repeatedly. A global NCHW->NHWC translation would need axis rewrites for `permute(0,2,1)`, `reshape(B,C,H,W)`, `flatten(2)`, `transpose(1,2)`, concat `dim=1`, and final `sequence_output[:,0,:]`; initial lowering should guard these regions.
- Position interpolation in source reshapes position embeddings using current `height,width`. For first integration, prefer fixed 224x224 inputs from the preprocessor and add explicit dynamic-size parity tests before admitting arbitrary resolutions.

## 4. Operator coverage checklist

Tensor/layout ops:

- NCHW input validation: `pixel_values.shape[1] == num_channels`.
- Conv output shape tracking: stage spatial sizes at 224 are `56x56`, `28x28`, `14x14`, `7x7`.
- `flatten(2)`, `transpose(1,2)`, `permute(0,2,1)`, `reshape(B,C,H,W)`, `reshape(B,H,W,C)`, `contiguous`.
- CLS token expand and `cat(..., dim=1)` in final patch embedding.
- CLS gather `sequence_output[:, 0, :]`.

Neural network primitives:

- Patch Conv2d with bias:
  - stage 0: Conv2d(`3 -> 64`, kernel/stride 4)
  - stage 1: Conv2d(`64 -> 128`, kernel/stride 2)
  - stage 2: Conv2d(`128 -> 320`, kernel/stride 2)
  - stage 3: Conv2d(`320 -> 512`, kernel/stride 2)
- LayerNorm over last sequence/channel dim with eps `1e-6`.
- Linear Q/K/V with bias per stage: `C -> C`.
- Attention output Linear with bias: `C -> C`.
- MLP with bias:
  - stage 0: Linear(`64 -> 512`) -> GELU -> Linear(`512 -> 64`)
  - stage 1: Linear(`128 -> 1024`) -> GELU -> Linear(`1024 -> 128`)
  - stage 2: Linear(`320 -> 1280`) -> GELU -> Linear(`1280 -> 320`)
  - stage 3: Linear(`512 -> 2048`) -> GELU -> Linear(`2048 -> 512`)
- Classifier Linear(`512 -> 1000`) for sampled ImageNet heads.

Attention primitives:

- Noncausal encoder self-attention with separate Q, K, V.
- MHA only; no MQA/GQA.
- Dense batched matmul + scale by `1/sqrt(head_dim)` + softmax over K length + value matmul.
- Sequence-reduction Conv2d for K/V only when `sequence_reduction_ratio > 1`:
  - stage 0: Conv2d(`64 -> 64`, kernel/stride 8), K/V length `7*7=49`.
  - stage 1: Conv2d(`128 -> 128`, kernel/stride 4), K/V length `7*7=49`.
  - stage 2: Conv2d(`320 -> 320`, kernel/stride 2), K/V length `7*7=49`.
  - stage 3: no reduction; K/V length includes CLS + 49 patch tokens at 224 input.

Position/custom ops:

- Learned absolute position embeddings per stage.
- Bilinear interpolation of position embeddings in source path; fixed 224x224 can use stored tables.

Preprocessing-coupled ops:

- Resize to 224x224 bicubic, rescale, normalize with ImageNet mean/std, emit NCHW.

Distributed/tensor-parallel ops:

- None required by source.

## 5. Layer/block breakdown

For 224x224 input, stage shapes are:

```text
stage 0 input:  [B, 3, 224, 224]
patch conv:     [B, 64, 56, 56] -> tokens [B, 3136, 64]
blocks:         [B, 3136, 64], heads=1, sr=8, K/V tokens=49
to NCHW:        [B, 64, 56, 56]

stage 1 input:  [B, 64, 56, 56]
patch conv:     [B, 128, 28, 28] -> tokens [B, 784, 128]
blocks:         [B, 784, 128], heads=2, sr=4, K/V tokens=49
to NCHW:        [B, 128, 28, 28]

stage 2 input:  [B, 128, 28, 28]
patch conv:     [B, 320, 14, 14] -> tokens [B, 196, 320]
blocks:         [B, 196, 320], heads=5, sr=2, K/V tokens=49
to NCHW:        [B, 320, 14, 14]

stage 3 input:  [B, 320, 14, 14]
patch conv:     [B, 512, 7, 7] -> patch tokens [B, 49, 512]
cls+pos:        [B, 50, 512]
blocks:         [B, 50, 512], heads=8, sr=1, K/V tokens=50
final norm:     [B, 50, 512]
classifier:     logits = Linear(sequence[:, 0, :])
```

PVT block, repeated according to stage `depths[i]`:

```text
x_norm = LayerNorm(x)
q = Linear(C -> C, bias=qkv_bias)(x_norm)
kv_input = x_norm
if sr_ratio > 1:
  kv_input = reshape sequence [B,N,C] -> NCHW [B,C,H,W]
  kv_input = Conv2d(C -> C, kernel=stride=sr_ratio, bias=True)(kv_input)
  kv_input = reshape NCHW -> sequence [B,N_sr,C]
  kv_input = LayerNorm(kv_input)
k = Linear(C -> C, bias=qkv_bias)(kv_input)
v = Linear(C -> C, bias=qkv_bias)(kv_input)
attn = softmax((q @ k^T) / sqrt(64), dim=-1)
x = x + Linear(C -> C, bias=True)(attn @ v)
mlp = Linear(C -> C*mlp_ratio, bias=True) -> GELU -> Linear(C*mlp_ratio -> C, bias=True)
x = x + mlp(LayerNorm(x))
```

Dropout and DropPath are identity in inference and sampled configs.

## 6. Attention requirements

- Variant: noncausal encoder self-attention only.
- Heads: stage heads `[1,2,5,8]`, KV heads equal query heads, head dim 64.
- Masking: no attention mask input in source. Attention probabilities are dense over reduced K/V tokens.
- Packed/varlen: not required.
- Sliding/local: not required; SR attention is strided spatial downsampling of K/V, not local-window attention.
- Positional interactions: learned absolute positions are added before LayerNorm/Q/K/V; no RoPE/ALiBi/relative bias.
- KV cache: none. Encoder outputs could be cached by application-level image reuse, but no autoregressive cache exists.
- Backend compatibility: a dense attention backend can handle each stage if it supports rectangular query/key lengths (`N_q != N_kv`) for SR stages. Eager-style matmul/softmax is simple but stage 0 has Q length 3136, so optimized rectangular attention matters for throughput.

Attention score order:

```text
q,k,v linear -> reshape to [B,H,N,D] -> scores = q @ k^T -> divide by sqrt(D) -> softmax(dim=-1) -> dropout -> probs @ v
```

## 7. Position encoding and custom math

Position embeddings are learned per patch-embedding stage. Only the final stage has a CLS position row.

Source interpolation pattern:

```python
def pvt_pos_embed(position_embeddings, height, width, has_cls):
    if has_cls:
        cls_pos = position_embeddings[:, :1]
        patch_pos = position_embeddings[:, 1:]
    else:
        cls_pos = None
        patch_pos = position_embeddings
    patch_pos = patch_pos.reshape(1, height, width, -1).permute(0, 3, 1, 2)
    patch_pos = interpolate(patch_pos, size=(height, width), mode="bilinear")
    patch_pos = patch_pos.reshape(1, -1, height * width).permute(0, 2, 1)
    return cat([cls_pos, patch_pos], dim=1) if has_cls else patch_pos
```

For fixed 224x224, the stored position table shape already matches each stage's token grid. Dynamic shapes depend on runtime `height,width`; because the inspected source reshapes the stored table with current dimensions, DinoML should admit non-224 sizes only after a parity test against the exact source.

## 8. Preprocessing and input packing

`PvtImageProcessor` and `PvtImageProcessorPil` defaults:

- Resize to `{"height":224,"width":224}` / legacy configs use `"size": 224`.
- Bicubic resampling.
- Rescale enabled by backend default.
- Normalize with ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`.
- Output tensor expected by model: `pixel_values` `[B,3,224,224]` NCHW.

No tokenizer, masks, packed sequence metadata, or modality stitching. Postprocessing for classification is outside the model graph: softmax/top-k/label mapping if an application needs probabilities/labels.

## 9. Graph rewrite / lowering opportunities

### Rewrite: patch Conv2d -> WindowFlatten + GEMM

Source pattern:

```text
Conv2d(Cin -> Cout, kernel_size=stride, stride=patch_size, padding=0)
-> flatten(2) -> transpose(1,2) -> LayerNorm(Cout)
```

Replacement:

```text
WindowFlatten(NCHW, kh=kw=stride) -> MatMul(weight_flat.T) -> BiasAdd -> LayerNorm
```

Preconditions:

- `kernel_size == stride == patch_size`.
- `padding == 0`, `dilation == 1`, `groups == 1`.
- Input H/W divisible by stride.
- Flatten order must match PyTorch Conv2d NCHW weight layout `[Cout,Cin,kh,kw]`.

Weight transform:

```python
w = conv.weight.reshape(out_channels, in_channels * kh * kw)
y = windows @ w.T + conv.bias
```

Layout constraints: this can be a direct NHWC translation only inside a fully controlled patch-embedding region. If consumers expect NCHW for later reshapes, materialize or track layout explicitly.

Failure cases: dynamic input sizes not divisible by stride; non-square patch configs; padding/dilation/groups changes.

Parity test sketch: compare patch embedding output after LayerNorm for each stage against source for random NCHW tensors.

### Rewrite: sequence-reduction Conv2d for K/V -> reduced-token projection

Source pattern:

```text
[B,N,C] -> permute/reshape [B,C,H,W]
-> Conv2d(C -> C, kernel=stride=sr)
-> reshape/permute [B,N_sr,C]
-> LayerNorm
-> K/V Linear
```

Replacement:

```text
SequenceToGrid -> WindowFlatten(sr x sr) -> MatMul(sr_weight_flat.T) -> BiasAdd -> LayerNorm -> K/V Linear
```

Preconditions:

- `sr_ratio > 1`, `kernel_size == stride == sr_ratio`, `padding == 0`, `groups == 1`, `dilation == 1`.
- `N == H * W` and the stage passes exact `height,width`.
- H/W divisible by `sr_ratio`; for sampled 224 path all reduced maps are 7x7.
- CLS token must not be present in SR stages. In sampled source, final stage has CLS but `sr_ratio == 1`.

Layout constraints: this region is axis-sensitive. A layout pass may keep NHWC internally, but must rewrite channel axis from `dim=1`/NCHW to last-channel and preserve sequence token order when returning `[B,N_sr,C]`.

Failure cases: final-stage CLS with `sr_ratio > 1`, non-divisible H/W, any future grouped/depthwise SR conv, or consumers requesting exact NCHW intermediate outputs.

Parity test sketch: random `[B,H*W,C]` per stage, compare reduced K/V input and final attention output.

### Rewrite: rectangular MHA dispatch

Source pattern:

```text
Q length Nq, K/V length Nkv after SR, dense noncausal attention
```

Replacement: call a generic encoder attention kernel supporting `Nq != Nkv`, no mask, head_dim 64.

Preconditions: no attention mask, no dropout in eval, MHA with equal Q/K/V head counts.

Failure cases: `output_attentions=True` requires materializing attention probabilities `[B,H,Nq,Nkv]`.

### Layout guard: NCHW/sequence crossings

Protect these source regions with a conceptual `no_layout_translation()` unless a pass owns every consumer:

```text
flatten(2).transpose(1,2)
hidden_states.permute(0,2,1).reshape(B,C,H,W)
hidden_states.reshape(B,H,W,C).permute(0,3,1,2)
torch.cat((cls, patches), dim=1)
sequence_output[:, 0, :]
```

Axis rewrite examples for an NHWC fusion pass: Conv/LayerNorm channel axis becomes `-1`, Conv2d weights need OIHW-to-HWIO or provider-native transforms, flatten spatial axes must preserve row-major token order, and concat/gather over token dimension must not be rewritten as channel concat.

## 10. Kernel fusion candidates

Highest priority:

- Patch Conv2d + flatten/transpose + LayerNorm: repeated at every stage and cheap to validate; can remove layout churn.
- SR Conv2d + reshape + LayerNorm feeding K/V: PVT-specific hot path and the main nonstandard operator surface.
- Rectangular attention for stage 0/1/2: Q length is large while K/V is 49 at 224, so backend overhead and materialized attention tensors can dominate.

Medium priority:

- Q/K/V separate Linear batching: source has separate Q and separate K/V after SR; K and V can be fused after SR into one GEMM if weights are concatenated `[K;V]`.
- LayerNorm + Linear fusion for block norms feeding projections/MLP.
- GELU MLP fusion for `[Linear -> GELU -> Linear]` with stage-specific intermediate sizes.

Lower priority:

- Final CLS gather + classifier Linear.
- Dropout/DropPath elimination in inference.
- Position add + dropout elimination for sampled configs with zero dropout.

## 11. Runtime staging plan

1. Parse `PvtConfig`, enforce supported image size/channels, depths, SR ratios, and qkv bias.
2. Load native HF weights and verify parameter names, including separate K/V; optionally add original PVT import split later.
3. Implement patch embedding parity for all four stages with fixed 224x224 inputs.
4. Implement one encoder block with SR attention for stages 0-2 and full attention for stage 3.
5. Run full base encoder parity and final classifier parity.
6. Add optimized patch/SR Conv2d-to-GEMM rewrites behind strict guards.
7. Add rectangular attention backend and K/V fused projection optimization.
8. Admit dynamic image sizes only after source parity tests for position interpolation.

Can stub initially: `output_attentions=True`, training losses, DropPath training behavior, original PVT checkpoint conversion.

## 12. Parity and validation plan

- Config parser tests for tiny/small/medium/large depths and stage shape equations.
- Random tensor patch embedding parity per stage: Conv2d, flatten/transposed token order, LayerNorm, position add.
- Random tensor SR path parity per stage for `sr_ratio` 8/4/2.
- Single block parity for one block from each stage with fp32 tolerance `atol=1e-5, rtol=1e-4`.
- After-N-layer parity for tiny, then deeper stage-3 parity for small/medium/large.
- End-to-end classifier logits for one preprocessed 224x224 image.
- `output_attentions=True` parity if exposed; attention tensor shapes should be `[B,H,Nq,Nkv]`.
- fp16/bf16 suggested tolerance after fp32: `atol=2e-3, rtol=2e-2`, with attention softmax drift checked per stage.
- Dynamic resolution parity tests before enabling non-224 inputs.

## 13. Performance probes

- CPU preprocessing throughput for bicubic resize/normalize.
- Patch embedding throughput by stage, Conv2d backend vs GEMM rewrite.
- SR Conv2d throughput by stage and batch size.
- Rectangular attention throughput by stage: `[Nq,Nkv]` of `[3136,49]`, `[784,49]`, `[196,49]`, `[50,50]`.
- Full encoder throughput sweep for tiny/small/medium/large depths.
- Batch-size sweep at 224x224.
- Optional image-resolution sweep only after dynamic-size parity is admitted.
- Memory probe for materialized `output_attentions=True` vs hidden-state-only fast path.
- Layout-pass probe comparing faithful NCHW crossings against guarded NHWC fused patch/SR regions.

## 14. Skip/defer list

- Training losses and stochastic DropPath behavior.
- `output_attentions=True` materialization in the first optimized backend.
- Dynamic image sizes until position interpolation parity is confirmed.
- Original PVT checkpoint conversion path unless importing non-HF weights.
- Multi-GPU/tensor parallel lowering.
- Quantization; no source-coupled packed/quantized weights in inspected native code.

## 15. Final implementation checklist

- [ ] Parse `PvtConfig` stage arrays and validate equal lengths.
- [ ] Load HF native weights with separate Q/K/V parameters.
- [ ] Implement NCHW image input and PVT image preprocessor contract.
- [ ] Implement patch Conv2d embeddings with learned position add and final-stage CLS token.
- [ ] Implement sequence/NCHW reshape transitions with token-order tests.
- [ ] Implement SR Conv2d + LayerNorm K/V reduction.
- [ ] Implement noncausal rectangular MHA with no mask/cache.
- [ ] Implement stage MLPs with GELU and LayerNorm eps `1e-6`.
- [ ] Implement final CLS gather and classifier Linear.
- [ ] Add guarded patch Conv2d -> GEMM rewrite.
- [ ] Add guarded SR Conv2d -> GEMM rewrite.
- [ ] Add layout-pass guards for NCHW/sequence crossings.
- [ ] Add single-block, full-encoder, and classifier logits parity tests.
- [ ] Benchmark patch embedding, SR attention, rectangular attention, and full encoder throughput.
