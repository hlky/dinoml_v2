# Hiera Transformers audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: facebook/hiera-* official HF checkpoints sampled
Config source: official HF config.json and preprocessor_config.json snapshots saved in this folder
Source files inspected:
  transformers/src/transformers/models/hiera/configuration_hiera.py
  transformers/src/transformers/models/hiera/modeling_hiera.py
  transformers/src/transformers/models/hiera/convert_hiera_to_hf.py
Any missing files or assumptions: no Hiera-specific image_processing file exists; processors are BitImageProcessor configs. No sampled official repos were gated or 401.
```

Saved snapshots:

- `facebook__hiera-tiny-224-hf.config.json`
- `facebook__hiera-small-224-hf.config.json`
- `facebook__hiera-base-224-hf.config.json`
- `facebook__hiera-base-plus-224-hf.config.json`
- `facebook__hiera-large-224-hf.config.json`
- `facebook__hiera-huge-224-hf.config.json`
- `facebook__hiera-tiny-224-mae-hf.config.json`
- `facebook__hiera-base-plus-224-in1k-hf.config.json`
- `facebook__hiera-tiny-224-hf.preprocessor_config.json`
- `facebook__hiera-base-plus-224-hf.preprocessor_config.json`
- `facebook__hiera-tiny-224-mae-hf.preprocessor_config.json`

Primary HF URLs sampled:

- [facebook/hiera-tiny-224-hf](https://huggingface.co/facebook/hiera-tiny-224-hf)
- [facebook/hiera-base-plus-224-hf](https://huggingface.co/facebook/hiera-base-plus-224-hf)
- [facebook/hiera-tiny-224-mae-hf](https://huggingface.co/facebook/hiera-tiny-224-mae-hf)
- [facebook/hiera-base-plus-224-in1k-hf](https://huggingface.co/facebook/hiera-base-plus-224-in1k-hf)

## 2. High-level architecture

Hiera is a hierarchical image encoder, not an autoregressive model. The first useful DinoML runtime target should be `HieraModel`/`HieraForImageClassification`/`HieraBackbone` inference on fixed 224x224 images.

Dataflow:

```text
image CPU preprocessing -> NCHW pixel_values -> Conv2d patch embedding
-> absolute position add -> token unroll/window reordering
-> staged mask-unit/global self-attention with query pooling
-> sequence pooling or feature-map reroll -> classifier/backbone outputs
```

Implemented heads:

- `HieraModel`: required for base feature extraction. Returns sequence plus optional average-pooled output.
- `HieraForImageClassification`: required for ImageNet-style classification. Adds linear classifier after sequence average pooling plus LayerNorm.
- `HieraBackbone`: required if DinoML targets DETR/MaskFormer-style feature maps. Returns selected NCHW feature maps after stage-specific LayerNorm.
- `HieraForPreTraining`: optional/deferred for first inference target. It adds MAE random masking, masked conv, multiscale fusion, a decoder block, pixel reconstruction, boolean indexing, and training loss.

Independently stageable pieces:

- CPU/data pipeline: resize, center crop, rescale, normalize, NCHW packing.
- Encoder body: patch embedding, unroll, staged attention/MLP.
- Pool/classifier head: tiny and independent.
- Backbone feature extraction: reroll hidden states to NHWC, LayerNorm over channels, permute to NCHW.
- MAE decoder: separate optional path with mask/scatter-like reconstruction.

## 3. Important config dimensions

Source defaults from `HieraConfig`:

| Field | Default |
|---|---:|
| `image_size` | `[224, 224]` |
| `num_channels` | `3` |
| `patch_size` / `patch_stride` / `patch_padding` | `[7,7]` / `[4,4]` / `[3,3]` |
| initial token grid | `56 x 56 = 3136` |
| `embed_dim` | `96` |
| `depths` | `[2, 3, 16, 3]` |
| `num_heads` | `[1, 2, 4, 8]` |
| `embed_dim_multiplier` | `2.0` |
| final `hidden_size` | `embed_dim * 2^3 = 768` |
| `num_query_pool` / `query_stride` | `3` / `[2,2]` |
| `masked_unit_size` | `[8,8]` |
| `masked_unit_attention` | `[true,true,false,false]` |
| `hidden_act` | `gelu` |
| `layer_norm_eps` | `1e-6` |

Representative checkpoint sweep:

| Checkpoint | Architecture | `embed_dim` | Final width | Depths | Heads | Query pool | Notes |
|---|---|---:|---:|---|---|---:|---|
| `facebook/hiera-tiny-224-hf` | `HieraModel` | 96 | 768 | 1/2/7/2 | 1/2/4/8 | 3 | feature extraction |
| `facebook/hiera-small-224-hf` | `HieraModel` | 96 | 768 | 1/2/11/2 | 1/2/4/8 | 3 | deeper stage3 |
| `facebook/hiera-base-224-hf` | `HieraModel` | 96 | 768 | 2/3/16/3 | 1/2/4/8 | 3 | config default body |
| `facebook/hiera-base-plus-224-hf` | `HieraModel` | 112 | 896 | 2/3/16/3 | 2/4/8/16 | 3 | wider, more heads |
| `facebook/hiera-large-224-hf` | `HieraModel` | 144 | 1152 | 2/6/36/4 | 2/4/8/16 | 3 | much deeper stage3 |
| `facebook/hiera-huge-224-hf` | `HieraModel` | 256 | 2048 | 2/6/36/4 | 4/8/16/32 | 3 | very wide |
| `facebook/hiera-tiny-224-mae-hf` | `HieraForPreTraining` | 96 | 768 | 1/2/7/2 | 1/2/4/8 | 2 | decoder width 512, depth 8, heads 16 |
| `facebook/hiera-base-plus-224-in1k-hf` | `HieraForImageClassification` | 112 | 896 | 2/3/16/3 | 2/4/8/16 | 3 | classifier head, ImageNet labels |

Preprocessor snapshots are uniform in sampled repos: `BitImageProcessor`, resize shortest edge 256, center crop 224x224, resample id 3, rescale, normalize with ImageNet mean/std, output `pixel_values` in NCHW.

## 3a. Family variation traps

- Stage widths are `embed_dim * embed_dim_multiplier^stage`; final hidden size is not a separate checkpoint field in sampled configs.
- Attention `head_dim = hidden_size_output / num_heads[stage]`; sampled configs divide evenly, but admission should guard divisibility.
- The first layer of stages 2 to 4 changes width and query-pools. It projects the residual branch from `hidden_size` to `hidden_size_output`, then max-pools along the unrolled query-stride axis.
- Attention alternates between mask-unit/window attention and global attention by `masked_unit_attention`. Stage 1 and 2 use windowed mask-unit attention by default, stage 3 and 4 use global attention, and the first layer after a previous masked-attention stage is also marked as masked to align with pooled lower resolution.
- `num_query_pool` changes MAE output geometry. MAE checkpoints use `num_query_pool=2`, while base/classifier checkpoints use 3.
- The source performs many `view`/`reshape`/`permute` operations whose axes encode NCHW-to-token and token-to-NHWC layout. Do not apply a broad NHWC rewrite across these regions without rewriting `dim`/axis constants and validating flatten order.
- `HieraBackbone.forward` in the inspected source calls embeddings then `encoder` directly, while `HieraModel.forward` applies `unroll` before `encoder`. Treat this as exact source behavior for this commit and test it explicitly before sharing lowering code between model/backbone entrypoints.
- `interpolate_pos_encoding=True` uses bicubic interpolation over a square learned position table. Static 224x224 inference can reject or stub it; higher-resolution inference must implement it.
- MAE pretraining uses random `argsort`, `gather`, boolean indexing, and in-place boolean assignment. That path is not necessary for classification/feature extraction.

## 4. Operator coverage checklist

Tensor/layout ops:

- NCHW image input validation and CPU preprocessing handoff.
- `Conv2d(3 -> embed_dim, kernel=7x7, stride=4x4, padding=3x3)`.
- Flatten spatial tokens: `[B,C,H',W'] -> [B,H'W',C]`.
- Position add with optional bicubic interpolation.
- Static `view`, `reshape`, `flatten`, `transpose`, `permute`, `contiguous`.
- Token unroll and reroll patterns with exact stride schedule.
- Max reduction over query-stride axis for query pooling and residual projection pooling.
- Average pool over sequence length in `HieraPooler` via transpose + `AdaptiveAvgPool1d(1)`.
- Backbone channel-last LayerNorm then NHWC -> NCHW permute for feature maps.

Neural primitives:

- LayerNorm over last dimension, eps `1e-6`.
- Linear QKV, packed as one `nn.Linear(hidden_size, 3 * hidden_size_output)` with split order `q, k, v`.
- Linear attention output projection.
- MLP: `Linear(C -> int(C * mlp_ratio)) -> GELU -> Linear(int(C * mlp_ratio) -> C)`.
- Residual add and optional DropPath. For inference with `drop_path_rate=0.0` in sampled configs, DropPath is identity.
- Classifier: `Linear(final_width -> num_labels)` for classification repos.

Attention primitives:

- Noncausal self-attention only.
- Local/mask-unit attention for selected stages: attention is batched over `num_windows`.
- Global attention when `use_mask_unit_attn=False`: `num_windows=1`.
- Query pooling when `query_stride > 1`: max over query groups before score matmul.
- Score scale by `head_dim ** -0.5`, softmax over key axis, then value matmul.
- No causal mask, cross-attention, KV cache, RoPE, ALiBi, GQA/MQA, sliding decode, or generation controller.

Preprocessing-coupled ops:

- `BitImageProcessor`: resize shortest edge 256, center crop 224, rescale, normalize ImageNet mean/std, emit NCHW float pixels.

Optional MAE-only ops:

- Per-sample random noise, `argsort`, inverse `argsort`, `gather`.
- Boolean mask upsample with `nn.functional.interpolate`.
- Boolean indexing and in-place scatter-like assignment.
- Multiscale Conv2d fusion heads over mask-unit feature maps.
- Pixel label extraction via NHWC permute + `unfold` + flatten.
- Mean/variance normalization and MSE loss.

## 5. Layer/block breakdown

Patch embedding:

```text
pixel_values: [B,3,224,224]
conv2d k=7 s=4 p=3 -> [B,E,56,56]
flatten+transpose -> [B,3136,E]
add position_embeddings [1,3136,E]
unroll by three [2,2] schedules -> [B,3136,E] with mask-unit/window order
```

Encoder stage `s`, repeated `depths[s]`:

```text
input width C_in, output width C_out = embed_dim * multiplier^s
x_norm = LayerNorm(C_in)(x)
if C_in != C_out:
  residual = Linear(C_in -> C_out)(x_norm)
  residual = max_pool_over_query_stride(residual)  # first layer of pooled stages
else:
  residual = x
qkv = Linear(C_in -> 3*C_out)(x_norm)
q,k,v = reshape/permute/unbind(qkv)
if query_stride > 1:
  q = max_pool_over_query_stride(q)
attn = softmax((q * scale) @ k.T)
y = Linear(C_out -> C_out)(attn @ v)
x = residual + y
z = LayerNorm(C_out)(x)
z = Linear(C_out -> 4*C_out)(z)
z = GELU(z)
z = Linear(4*C_out -> C_out)(z)
x = x + z
```

Default base dimensions by stage:

| Stage | Token grid after pool | Width | Heads | Head dim | Depth | Attention |
|---|---:|---:|---:|---:|---:|---|
| 1 | 56x56 | 96 | 1 | 96 | 2 | mask-unit |
| 2 | 28x28 | 192 | 2 | 96 | 3 | mask-unit |
| 3 | 14x14 | 384 | 4 | 96 | 16 | global |
| 4 | 7x7 | 768 | 8 | 96 | 3 | global |

Classifier head:

```text
last_hidden_state [B,49,C]
transpose -> [B,C,49]
AdaptiveAvgPool1d(1) -> [B,C,1]
flatten -> [B,C]
LayerNorm(C)
Linear(C -> num_labels)
```

Backbone head:

```text
selected rerolled hidden state [B,H,W,C]
view [B,H*W,C]
LayerNorm(C)
view [B,H,W,C]
permute contiguous -> [B,C,H,W]
```

## 6. Attention requirements

Hiera uses encoder-style noncausal self-attention. There is no generation, no decode cache, and no KV cache.

Attention shape basis:

```text
qkv linear output: [B, L, 3*C_out]
reshape: [B, tokens_per_window_or_query, num_windows, 3, heads, head_dim]
permute: [3, B, heads, num_windows, tokens_per_window_or_query, head_dim]
query/key/value: [B, heads, num_windows, Q_or_K, head_dim]
scores: [B, heads, num_windows, Q, K]
```

For mask-unit attention:

- `window_size = masked_unit_area * query_stride_area ** -stage_index`.
- With 224x224 defaults, `masked_unit_size=[8,8]`, so stage window sizes are 64, 16, 4, 1 before the `use_mask_unit_attn` schedule.
- `num_windows = seq_len // (query_stride * window_size)`, where `query_stride` is flattened area, usually 4 on query-pool layers and 1 otherwise.

For global attention:

- `num_windows=1`.
- Attention covers the full current stage token sequence.

SDPA/FlashAttention compatibility:

- The math is standard scaled dot-product attention after the source reshape/window grouping.
- A fused backend can treat `(B * heads * num_windows)` as batch rows if the pre-attention window layout is exact.
- Query pooling changes Q length while K/V still come from the unpooled sequence. Fused attention must support rectangular Q/K for those layers.
- `output_attentions=True` returns attention weights and may require a dense materialized weights path; optimized inference can initially reject it.

## 7. Position encoding and custom math

Position embeddings are learned absolute embeddings with optional interpolation:

```python
def interpolate_hiera_pos(pos, num_positions, dim, new_height, new_width):
    side = int(num_positions ** 0.5)
    pos = pos.reshape(1, side, side, dim).permute(0, 3, 1, 2)
    pos = bicubic_interpolate(pos, size=(new_height, new_width), align_corners=False)
    return pos.permute(0, 2, 3, 1).reshape(1, new_height * new_width, dim)
```

Token unroll is source-specific and layout-sensitive:

```python
def unroll_2d(x, current_h, current_w, stride_h, stride_w):
    # x starts as [B,H,W,C]
    x = x.view(B, current_h // stride_h, stride_h, current_w // stride_w, stride_w, C)
    x = x.permute(0, 2, 4, 1, 3, 5)
    x = x.flatten(0, 2)
    return x
```

The source applies the stride schedule repeatedly, then reshapes back to `[B, H*W, C]` with window-contiguous order. This ordering is part of attention semantics.

## 8. Preprocessing and input packing

The model-coupled preprocessor is `BitImageProcessor` from checkpoint config:

```text
resize shortest_edge=256
center_crop height=224 width=224
resample=3
rescale=True
normalize mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]
output pixel_values=[B,3,224,224]
```

CPU/data-pipeline work:

- Image decode.
- Resize/crop/rescale/normalize.
- Batch packing to contiguous NCHW float tensor.

GPU/runtime work:

- Patch Conv2d and all subsequent neural graph ops.

MAE path input packing:

- `noise` optional tensor `[B,num_mask_units]`, mainly for reproducibility.
- Random masking generates `bool_masked_pos` where `True` means keep in the encoder path, then later the pretraining loss inverts it to select masked patches.
- First integration can reject MAE or require caller-supplied deterministic `noise`.

## 9. Graph rewrite / lowering opportunities

### Rewrite: fixed patch Conv2d as provider Conv2d

Source pattern:

```text
Conv2d(3 -> E, kernel=7, stride=4, padding=3) -> flatten(2) -> transpose(2,1)
```

Replacement:

```text
Conv2d NCHW -> token flatten
```

Preconditions:

- Rank-4 NCHW input.
- `num_channels == 3`.
- Static or guarded input size.
- Preserve PyTorch Conv2d padding semantics.

Failure cases:

- `interpolate_pos_encoding=True` with non-224 sizes needs dynamic position interpolation and grid guards.
- NHWC rewrite must transform Conv2d weight layout and flatten order.

Parity sketch:

- Compare patch embeddings before position add for random NCHW tensors.

### Rewrite: Hiera unroll + mask-unit attention batching

Source pattern:

```text
view/permute/flatten token unroll -> qkv -> reshape by num_windows -> attention
```

Replacement:

```text
WindowTokenReorder -> BatchedAttention(batch=B*heads*num_windows)
```

Preconditions:

- Static stride schedule `[2,2]` repeated `len(depths)-1`.
- Token grid divisible by query strides.
- `window_size` and `num_windows` computed from config/source equation.
- Preserve row-major order exactly.

Failure cases:

- Dynamic image sizes without full position and window schedule validation.
- Backbone entrypoint if source unroll behavior differs from `HieraModel`.

Parity sketch:

- Compare q/k/v window tensors and attention output for one layer.

### Rewrite: query-pool max as reshape reduction

Source pattern:

```text
query.view(B,H,num_windows,query_stride,-1,D).max(dim=3)
residual.view(B,query_stride,-1,C).max(dim=1)
```

Replacement:

```text
Reshape -> reduce_max(axis=query_stride_axis)
```

Preconditions:

- `query_stride > 1`.
- Input is already in Hiera unrolled token order.
- Reduction axis is the source query-stride axis, not spatial channel axis.

Failure cases:

- Applying this to NHWC feature maps before unroll.

### Rewrite: pooler AdaptiveAvgPool1d(1)

Source pattern:

```text
x.transpose(1,2) -> AdaptiveAvgPool1d(1) -> flatten -> LayerNorm
```

Replacement:

```text
reduce_mean over sequence axis -> LayerNorm
```

Preconditions:

- Output size is exactly 1.
- No padding or count override.

### Layout rewrite candidate: backbone NHWC LayerNorm region

Source pattern:

```text
[B,H,W,C] -> view [B,H*W,C] -> LayerNorm(C) -> view [B,H,W,C] -> permute [B,C,H,W]
```

Candidate optimized layout:

```text
Keep NHWC internally until consumer requires NCHW, or fuse LayerNorm with producer.
```

Required axis rewrites:

- LayerNorm stays over last/channel dimension in NHWC.
- Feature map output ABI currently requires NCHW.

Failure cases:

- Downstream consumers that assume NCHW strides.
- Shared lowering with classifier sequence path.

## 10. Kernel fusion candidates

Highest priority:

- LayerNorm over channel dimension for token tensors. It appears before every attention and MLP plus pooler/backbone norms.
- Packed QKV projection + reshape/split. The source stores QKV as one Linear with `q,k,v` split order.
- Windowed/rectangular SDPA for mask-unit attention and query-pool layers.
- MLP `Linear -> GELU -> Linear`, with GEMM epilogue GELU if available.
- Query-pool reduce_max fused with residual projection or Q path reshape.

Medium priority:

- Patch Conv2d + flatten/transpose output packing.
- Position add fused into first token reorder or first LayerNorm input load.
- Pooler sequence reduce_mean + LayerNorm.
- Backbone reroll + LayerNorm + NHWC-to-NCHW output packing.

Lower priority:

- MAE random masking and boolean compaction.
- MAE multiscale fusion Conv2d heads.
- Position interpolation, since fixed 224x224 inference can avoid it.
- `output_attentions=True` dense attention materialization.

## 11. Runtime staging plan

Stage 1: parse Hiera configs and load weights for `HieraModel`; support tiny/base-plus snapshots first.

Stage 2: implement patch embedding, position add, unroll, one encoder block parity with static 224x224 NCHW input.

Stage 3: implement full encoder for `HieraModel` with `output_attentions=False`, `output_hidden_states=False`, fixed 224x224.

Stage 4: add pooler and classifier head for `HieraForImageClassification`.

Stage 5: add `HieraBackbone` feature maps with reroll, stage LayerNorm, and NCHW output ABI. Validate exact source entrypoint behavior for this commit before reusing `HieraModel` unroll.

Stage 6: optimize attention/window lowering, LayerNorm, MLP/GEMM epilogues, and query pooling.

Stage 7: optional MAE pretraining parity, including masking, multiscale fusion, decoder, and reconstruction loss. This can be stubbed or rejected for inference-only integration.

Initially stub or reject:

- `interpolate_pos_encoding=True`.
- `output_attentions=True`.
- `output_hidden_states=True` except when needed for backbone implementation.
- MAE `HieraForPreTraining`.
- Training losses and labels.

## 12. Parity and validation plan

Recommended tests:

- Preprocessor parity against saved `preprocessor_config.json`: one PIL image to NCHW tensor, compare with Transformers.
- Patch embedding parity for random NCHW input.
- Position interpolation parity for 224x224 no-op and one larger guarded shape if supported.
- `unroll` and `reroll` round-trip/order tests with arange token tensors.
- Single `HieraMaskUnitAttention` parity for mask-unit and global attention layers.
- Query-pooling parity for rectangular attention layer where `query_stride=4`.
- Single `HieraLayer` parity with and without width change.
- Full `HieraModel` parity on tiny checkpoint, fp32 tolerance around `1e-4` to `1e-5`.
- Classification logits parity for an `*-in1k-hf` checkpoint.
- Backbone feature-map parity for selected `out_features`, including NCHW shape and values.
- Optional MAE parity with fixed `noise` tensor; compare `bool_masked_pos`, logits, and loss.

For reduced precision, start with fp32 graph parity and then compare fp16/bf16 kernel variants with tolerances chosen per primitive: LayerNorm/attention likely need relaxed relative tolerance around `1e-2` for fp16.

## 13. Performance probes

- CPU preprocessing images/sec.
- Patch Conv2d + token pack throughput.
- Per-stage encoder throughput, separated by mask-unit and global attention.
- Query-pooling layers versus non-pooling layers.
- Batch-size sweep for 224x224: 1, 2, 4, 8, 16.
- Model-size sweep: tiny, base, base-plus, large, huge.
- Windowed attention backend comparison: source-shaped batched GEMM/softmax/GEMM versus fused SDPA over `B*heads*num_windows`.
- LayerNorm and MLP GEMM time share per stage.
- Backbone feature extraction cost when `output_hidden_states=True`/rerolling is required.
- Optional position interpolation cost for higher-resolution images.
- Optional MAE masking/decoder overhead with deterministic `noise`.

## 14. Skip/defer list

Safe to defer for first integration:

- Training losses and label handling.
- `HieraForPreTraining` MAE path.
- Random masking, `argsort`, boolean scatter/indexing.
- `output_attentions=True`.
- Higher-resolution `interpolate_pos_encoding=True`.
- Gradient checkpointing and DropPath training behavior.
- Dynamic image sizes.
- Quantization and packed weights.
- Multi-GPU/tensor parallel execution.

Do not defer if targeting backbone users:

- `output_hidden_states=True` internally for feature maps.
- Reroll/NHWC LayerNorm/NCHW output packing.
- `out_features`/`out_indices` admission and channel metadata.

## 15. Final implementation checklist

- [ ] Parse `HieraConfig`, including `depths`, `num_heads`, `query_stride`, `masked_unit_attention`, and backbone output fields.
- [ ] Load official HF weights and preserve packed QKV split order `q,k,v`.
- [ ] Implement NCHW patch Conv2d and token flatten/transpose.
- [ ] Implement learned position add and fixed-size admission guard.
- [ ] Implement Hiera `unroll`/`reroll` token-order transforms with parity tests.
- [ ] Implement LayerNorm over last dimension with eps `1e-6`.
- [ ] Implement packed QKV self-attention with mask-unit windows and global mode.
- [ ] Implement query-pooling reduce_max for Q and residual projection paths.
- [ ] Implement MLP GELU block and residual adds.
- [ ] Implement sequence reduce_mean pooler plus classifier head.
- [ ] Implement backbone feature-map path with stage LayerNorm and NCHW outputs.
- [ ] Add tiny/base-plus full-model fp32 parity tests.
- [ ] Add classification logits parity for an `*-in1k-hf` checkpoint.
- [ ] Benchmark per-stage attention, MLP, and LayerNorm bottlenecks.
- [ ] Defer or explicitly reject MAE pretraining, dynamic position interpolation, and attention-weight outputs until separately scoped.
