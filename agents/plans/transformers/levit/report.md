# DinoML Transformers Audit: LeViT

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: facebook/levit-128S, facebook/levit-128, facebook/levit-192, facebook/levit-256, facebook/levit-384
Config source: official Hugging Face config.json and preprocessor_config.json snapshots
Source files inspected:
- X:/H/transformers/src/transformers/models/levit/modeling_levit.py
- X:/H/transformers/src/transformers/models/levit/configuration_levit.py
- X:/H/transformers/src/transformers/models/levit/image_processing_levit.py
- X:/H/transformers/src/transformers/models/levit/image_processing_pil_levit.py
- X:/H/transformers/src/transformers/models/levit/convert_levit_timm_to_pytorch.py
Any missing files or assumptions: no gated files; no remote code required. Scope is inference for image classification, including the official `LevitForImageClassificationWithTeacher` wrapper.
```

Source URLs:

- [modeling_levit.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/levit/modeling_levit.py)
- [configuration_levit.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/levit/configuration_levit.py)
- [image_processing_levit.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/levit/image_processing_levit.py)
- [image_processing_pil_levit.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/levit/image_processing_pil_levit.py)
- [convert_levit_timm_to_pytorch.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/levit/convert_levit_timm_to_pytorch.py)
- [facebook/levit-128S](https://huggingface.co/facebook/levit-128S)
- [facebook/levit-128](https://huggingface.co/facebook/levit-128)
- [facebook/levit-192](https://huggingface.co/facebook/levit-192)
- [facebook/levit-256](https://huggingface.co/facebook/levit-256)
- [facebook/levit-384](https://huggingface.co/facebook/levit-384)

Local snapshots are under `agents/plans/transformers/levit/_sources/`.

## 2. High-level architecture

LeViT is an image classification model with a four-layer convolutional stem followed by three transformer stages over image tokens. It is not an autoregressive model: no tokenizer, prefill/decode, sampling, causal mask, or KV cache is required.

```text
image preprocessing -> NCHW pixel_values -> Conv/BN/Hardswish patch stem
  -> flatten image map to row-major [B,L,C] sequence
  -> full 2D-relative-bias attention + MLP blocks
  -> attention downsample
  -> full attention + MLP blocks
  -> attention downsample
  -> full attention + MLP blocks
  -> mean over tokens -> BatchNorm1d + Linear classifier(s) -> logits
```

CPU/data pipeline owns resize, crop, rescale, and normalize. GPU/runtime owns the NCHW stem, sequence transformer, attention downsampling, token pooling, and classifier heads. Official checkpoints use `LevitForImageClassificationWithTeacher`, which runs two heads on the same mean-pooled feature and averages their logits.

## 3. Important config dimensions

All official configs inspected use `image_size=224`, `patch_size=16`, `num_channels=3`, `kernel_size=3`, `stride=2`, `padding=1`, `attention_ratio=[2,2,2]`, `mlp_ratio=[2,2,2]`, `num_labels=1000`, and architecture `LevitForImageClassificationWithTeacher`.

| Checkpoint | Hidden sizes | Heads | Depths | Key dim | Stage resolutions | Final tokens | Drop path |
|---|---:|---:|---:|---:|---:|---:|---:|
| `facebook/levit-128S` | 128,256,384 | 4,6,8 | 2,3,4 | 16,16,16 | 14 -> 7 -> 4 | 16 | 0 |
| `facebook/levit-128` | 128,256,384 | 4,8,12 | 4,4,4 | 16,16,16 | 14 -> 7 -> 4 | 16 | 0 |
| `facebook/levit-192` | 192,288,384 | 3,5,6 | 4,4,4 | 32,32,32 | 14 -> 7 -> 4 | 16 | 0 |
| `facebook/levit-256` | 256,384,512 | 4,6,8 | 4,4,4 | 32,32,32 | 14 -> 7 -> 4 | 16 | 0 |
| `facebook/levit-384` | 384,512,768 | 6,9,12 | 4,4,4 | 32,32,32 | 14 -> 7 -> 4 | 16 | 0.1 |

Attention dimensions are source-derived, not inferred from hidden size alone:

```text
full attention qkv width = heads * key_dim * (2 + attention_ratio)
full attention pre-projection width = heads * key_dim * attention_ratio
q split width = key_dim per head
k split width = key_dim per head
v split width = attention_ratio * key_dim per head
```

Examples where widths differ from stage hidden size:

| Checkpoint/stage | Hidden C | Heads | Key dim | Q/K total | V total / attention output |
|---|---:|---:|---:|---:|---:|
| 128S stage 1 | 256 | 6 | 16 | 96 | 192 |
| 128S stage 2 | 384 | 8 | 16 | 128 | 256 |
| 192 stage 1 | 288 | 5 | 32 | 160 | 320 |
| 384 stage 1 | 512 | 9 | 32 | 288 | 576 |

Downsample attention comes from `down_ops`: stage 0 -> 1 and stage 1 -> 2 use stride 2, `attention_ratio=4`, and a following MLP ratio 2. The downsample head counts are `hidden_sizes[0] // key_dim[0]` and `hidden_sizes[1] // key_dim[0]`.

## 4. Operator coverage checklist

Tensor/layout ops:

- NCHW input validation against `num_channels`.
- Four stride-2 Conv2d layers producing `[B,C0,14,14]` from `[B,3,224,224]`.
- `flatten(2).transpose(1,2)` to `[B,196,C0]`.
- Sequence-to-grid view for downsample queries: `[B,L,C] -> [B,R,R,C]`, slice `[:, ::2, ::2]`, reshape to `[B,Rout^2,C]`.
- View/reshape, permute, transpose, split, residual add, mean over `dim=1`.

Neural primitives:

- Conv2d bias-free + BatchNorm2d.
- Linear bias-free + BatchNorm1d over flattened `[B*L,C]`.
- Linear with bias for classifier heads after BatchNorm1d.
- Hardswish activation.
- Residual add. DropPath is training-only; inference can omit it.

Attention primitives:

- Dense noncausal self-attention over square 2D grids.
- Fused QKV projection with split order `query, key, value`.
- Attention downsample with full-resolution K/V and strided-query Q.
- Additive per-head relative attention bias, softmax over key length.

Position/relative-bias ops:

- Learned `attention_biases[heads,num_offsets]`.
- Integer `attention_bias_idxs[Q,K]` buffers for gather to `[heads,Q,K]`.
- Eval cache of gathered bias per device.

Generation/cache ops:

- None.

Preprocessing-coupled ops:

- Bicubic resize shortest edge to `int(256 / 224 * target)`, center crop, rescale, ImageNet normalize.

Distributed/tensor-parallel ops:

- None required for first integration.

## 5. Layer/block breakdown

Patch stem:

```text
x: [B,3,224,224] NCHW
x = Conv2d(3 -> C0/8, k=3, s=2, p=1, bias=False) -> BN2d -> Hardswish
x = Conv2d(C0/8 -> C0/4, k=3, s=2, p=1, bias=False) -> BN2d -> Hardswish
x = Conv2d(C0/4 -> C0/2, k=3, s=2, p=1, bias=False) -> BN2d -> Hardswish
x = Conv2d(C0/2 -> C0, k=3, s=2, p=1, bias=False) -> BN2d
x = flatten spatial row-major -> transpose -> [B,196,C0]
```

Full attention residual block:

```text
x: [B,L,C]
qkv = LinearNoBias(C -> heads * key_dim * (2 + attention_ratio)) -> BN1d
q,k,v = qkv.view(B,L,heads,-1).split([key_dim, key_dim, attention_ratio * key_dim], dim=3)
q,k,v = permute to [B,heads,L,D]
scores = q @ k.T * key_dim**-0.5 + relative_bias[heads,L,L]
attn = softmax(scores, dim=-1)
y = attn @ v
y = transpose/reshape to [B,L,heads * attention_ratio * key_dim]
y = Hardswish(y)
y = LinearNoBias(heads * attention_ratio * key_dim -> C) -> BN1d
x = x + y
```

MLP residual block:

```text
y = LinearNoBias(C -> C * mlp_ratio) -> BN1d -> Hardswish
y = LinearNoBias(C * mlp_ratio -> C) -> BN1d
x = x + y
```

Attention downsample:

```text
x: [B,Rin^2,Cin]
k,v = LinearNoBias(Cin -> heads * key_dim * (1 + attention_ratio)) -> BN1d
k,v = split([key_dim, attention_ratio * key_dim], dim=3), permute to [B,heads,Rin^2,D]
q_base = x.view(B,Rin,Rin,Cin)[:, ::2, ::2].reshape(B,Rout^2,Cin)
q = LinearNoBias(Cin -> heads * key_dim) -> BN1d
q = view/permute to [B,heads,Rout^2,key_dim]
scores = q @ k.T * key_dim**-0.5 + relative_bias[heads,Rout^2,Rin^2]
y = softmax(scores, dim=-1) @ v
y = reshape [B,Rout^2,heads * attention_ratio * key_dim] -> Hardswish
y = LinearNoBias(heads * attention_ratio * key_dim -> Cout) -> BN1d
```

Classifier:

```text
pooled = sequence.mean(dim=1)          # [B,Cfinal]
logits = BatchNorm1d(Cfinal) -> Linear(Cfinal -> num_labels, bias=True)
WithTeacher: logits = (classifier(pooled) + classifier_distill(pooled)) / 2
```

Source comments mention class/distillation tokens, but the inspected implementation does not add CLS or distillation tokens; both heads consume the mean-pooled sequence.

## 6. Attention requirements

Full attention is dense, noncausal MHA over image-grid tokens. Q and K use `key_dim`; V uses `attention_ratio * key_dim`. There is no padding mask, causal mask, sliding window, local attention, GQA/MQA, or KV cache.

Downsample attention is a separate primitive: Q length is `Rout^2`, K/V length is `Rin^2`, and Q tokens come from row-major strided slicing of the input grid. Bias is `[heads,Rout^2,Rin^2]`.

SDPA/FlashAttention compatibility requires additive per-head bias support. For official 224x224 configs, sequence lengths are small (`196`, `49`, `16`), so correctness-first dense attention is acceptable; repeated Linear+BN and reshapes are likely more visible than attention FLOPs at small batch sizes.

## 7. Position encoding and custom math

LeViT uses learned relative attention-bias tables indexed by absolute 2D offsets. No absolute position embeddings, RoPE, ALiBi, convolutional positional embedding, or token type embeddings are used.

```python
def levit_full_bias_indices(resolution):
    points = [(i, j) for i in range(resolution) for j in range(resolution)]
    offsets, indices = {}, []
    for p1 in points:
        for p2 in points:
            offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
            offsets.setdefault(offset, len(offsets))
            indices.append(offsets[offset])
    return indices.reshape(resolution * resolution, resolution * resolution)
```

```python
def levit_downsample_bias_indices(resolution_in, resolution_out, stride):
    src = [(i, j) for i in range(resolution_in) for j in range(resolution_in)]
    dst = [(i, j) for i in range(resolution_out) for j in range(resolution_out)]
    offsets, indices = {}, []
    for q in dst:
        for k in src:
            offset = (abs(q[0] * stride - k[0]), abs(q[1] * stride - k[1]))
            offsets.setdefault(offset, len(offsets))
            indices.append(offsets[offset])
    return indices.reshape(resolution_out * resolution_out, resolution_in * resolution_in)
```

```python
def levit_attention_bias(attention_biases, attention_bias_idxs):
    return attention_biases[:, attention_bias_idxs]  # [heads, query_len, key_len]
```

Indices depend only on resolution and stride and can be built at load/compile time. Gathered bias depends on learned weights and can be cached per device in inference.

## 8. Preprocessing and input packing

The processor produces channel-first `pixel_values` `[B,3,224,224]`. Defaults are bicubic resize, shortest edge scaled to 256 for target 224, center crop to 224x224, rescale, and ImageNet normalization with mean `[0.485,0.456,0.406]` and std `[0.229,0.224,0.225]`.

No masks, token IDs, placeholder tokens, grid metadata, or packed descriptors are produced.

Layout caution: the runtime must preserve the token order from NCHW `flatten(2).transpose(1,2)`. Any NHWC optimization must still emit the same row-major `[B,L,C]` sequence used by later `view(B,R,R,C)[:, ::2, ::2]` downsampling.

## 9. Graph rewrite / lowering opportunities

### Rewrite: Conv2d + BatchNorm2d folding

Source pattern:

```text
Conv2d(bias=False, k=3, s=2, p=1) -> BatchNorm2d -> optional Hardswish
```

Replacement:

```text
Conv2d(weight_fused, bias_fused) -> optional Hardswish
```

Preconditions:

- Inference mode with frozen BN stats.
- `groups == 1`, `dilation == 1`, kernel/stride/padding match config.

Weight transform:

```python
scale = bn.weight / sqrt(bn.running_var + bn.eps)
w_fused = conv.weight * scale[:, None, None, None]
b_fused = bn.bias - bn.running_mean * scale
```

Failure cases: training mode or missing BN running stats. Parity test: compare each stem layer before/after folding on random NCHW images.

### Rewrite: LinearNoBias + BatchNorm1d folding

Source pattern:

```text
Linear(Cin -> Cout, bias=False) on [B,L,Cin]
BatchNorm1d(Cout) on flatten(0,1), reshape back
```

Replacement:

```text
MatMul([B,L,Cin], W_fused.T) + bias_fused
```

Preconditions:

- Inference mode with frozen BN.
- Feature dimension is last; BN is over flattened batch-token rows.

Weight transform:

```python
scale = bn.weight / sqrt(bn.running_var + bn.eps)
w_fused = linear.weight * scale[:, None]
b_fused = bn.bias - bn.running_mean * scale
```

Failure cases: training mode or layout pass changes the feature axis. Test QKV, projection, MLP, and downsample projections independently.

### Rewrite: QKV split canonicalization

Source pattern:

```text
LinearBN -> view(B,L,heads,-1) -> split([key_dim,key_dim,attention_ratio*key_dim], dim=3)
```

Replacement:

```text
FusedLinearBN -> SplitQKV(order=q,k,v, source_layout=per_head_interleaved)
```

Preconditions:

- Preserve source packed layout or explicitly repack. The view occurs before split, so source weights are per-head interleaved, not one global Q block followed by K then V.

Failure cases: using a global Q/K/V split without repacking. Parity test: compare Q/K/V tensors before attention.

### Rewrite: AttentionSubsample query gather

Source pattern:

```text
x.view(B,Rin,Rin,C)[:, ::2, ::2].reshape(B,Rout^2,C)
```

Replacement:

```text
GridView(row-major) -> StridedGather2D(stride=2) -> FlattenGrid(row-major)
```

Preconditions:

- `L == Rin * Rin`.
- Token order is the canonical stem flatten order.
- `Rout == (Rin - 1) // 2 + 1`.

Failure cases: non-square token sequence or altered token order. Test with coordinate-coded tokens.

### Rewrite: guarded NHWC Conv stem region

Source pattern:

```text
NCHW pixel_values -> four ConvBN/Hardswish layers -> flatten(2).transpose(1,2)
```

Replacement:

```text
NCHW->NHWC boundary -> NHWC fused ConvBN/Hardswish chain -> row-major flatten -> [B,L,C]
```

Preconditions:

- Region is local and all consumers are controlled.
- Axis-sensitive attrs are rewritten: channel axis `1 -> -1`, BN channel axis, spatial flatten axes.
- Output token order exactly matches source.

Failure cases: exposing intermediate image maps in NHWC or changing token order before downsample attention. Parity test: compare full stem sequence output before the first transformer block.

## 10. Kernel fusion candidates

Highest priority:

- LinearNoBias + BatchNorm1d folding for all attention/MLP projections.
- Conv2d + BatchNorm2d + Hardswish fusion in the stem.
- Relative-bias gather precompute/cache.

Medium priority:

- Dense attention with additive per-head relative bias.
- Hardswish fused into GEMM epilogues after LinearBN folding.
- Fold classifier BatchNorm1d into Linear; compute two teacher heads together when using `WithTeacher`.

Lower priority:

- Custom AttentionSubsample kernel combining strided query gather, Q projection, and attention.
- Larger-resolution relative-bias memory optimizations beyond official 224x224 configs.

## 11. Runtime staging plan

Stage 1: parse config and load weights; admit official-style square `image_size`, `patch_size=16`, `kernel_size=3`, `stride=2`, `padding=1`.

Stage 2: implement patch stem parity through `[B,196,C0]`.

Stage 3: implement full attention block with relative bias and LinearBN folding.

Stage 4: implement attention downsample and validate final sequence `[B,16,Cfinal]`.

Stage 5: implement mean pooling plus single and `WithTeacher` classification heads.

Stage 6: add preprocessing parity or explicitly accept precomputed `pixel_values`.

Stage 7: enable NHWC stem/folding/fused-attention optimizations.

Initially stubbable: labels/loss, training DropPath, output hidden states, non-224 arbitrary images.

## 12. Parity and validation plan

- Random tensor tests for relative-bias index construction at resolutions 14, 7, 4.
- Coordinate-token tests for downsample query gather.
- ConvBN and LinearBN folding tests.
- Single-block parity for full attention and MLP.
- Downsample block parity for 14 -> 7 and 7 -> 4.
- Stage parity for stem `[B,196,C0]`, stage 0 `[B,49,C1]`, stage 1 `[B,16,C2]`, and stage 2 `[B,16,C2]`.
- End-to-end logits parity for `LevitForImageClassificationWithTeacher`; assert averaged logits match `(cls_logits + distillation_logits) / 2`.
- Suggested tolerances: fp32 `rtol=1e-4, atol=1e-5`; fp16/bf16 start at `rtol=5e-2, atol=5e-2` after BN folding, then tighten with backend calibration.

## 13. Performance probes

- CPU preprocessing images/sec.
- Stem-only throughput: NCHW baseline vs guarded NHWC ConvBN/Hardswish.
- LinearBN folding impact per block.
- Full attention vs downsample attention time.
- Batch sweep: 1, 8, 32, 128.
- Image-size sweep if non-224 is admitted; track relative-bias materialization.
- End-to-end images/sec for 128S, 128, 192, 256, 384.
- Teacher two-head classifier cost vs fused/batched head.

## 14. Skip/defer list

- Training, DropPath stochastic behavior, and label loss.
- Fine-tuning with teacher; source marks teacher wrapper as inference-only.
- Quantization and packed weights.
- Multi-GPU tensor parallel.
- Arbitrary non-square image/token grids.
- Public backbone-style feature-map outputs; current public model returns sequence hidden states.

## 15. Final implementation checklist

- [ ] Parse `LevitConfig` and representative checkpoint dimensions.
- [ ] Load Conv2d, BatchNorm2d, Linear, BatchNorm1d, attention-bias, and classifier weights.
- [ ] Implement NCHW patch stem.
- [ ] Implement `[B,C,H,W] -> [B,L,C]` row-major tokenization.
- [ ] Implement LinearNoBias + BatchNorm1d projection primitive or folding.
- [ ] Implement Hardswish.
- [ ] Implement LeViT full attention with per-head relative bias.
- [ ] Implement LeViT attention-downsample query gather and bias.
- [ ] Implement MLP residual blocks.
- [ ] Implement global mean pooling over sequence dimension.
- [ ] Implement classifier and classifier-distill average.
- [ ] Add ConvBN and LinearBN folding rewrites.
- [ ] Add guarded NHWC Conv stem rewrite with token-order parity test.
- [ ] Add relative-bias index/gather parity tests.
- [ ] Add one-block, one-stage, and end-to-end logits parity tests.
- [ ] Benchmark stem, attention, downsample attention, classifier heads, and end-to-end throughput.
