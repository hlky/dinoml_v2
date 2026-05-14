# SigLIP2 DinoML Operator Audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: google/siglip2-base-patch16-naflex, google/siglip2-so400m-patch16-naflex
Config source: local Transformers source defaults plus official HF config/preprocessor/tokenizer JSON snapshots
Source files inspected:
- X:/H/transformers/src/transformers/models/siglip2/configuration_siglip2.py
- X:/H/transformers/src/transformers/models/siglip2/modeling_siglip2.py
- X:/H/transformers/src/transformers/models/siglip2/modular_siglip2.py
- X:/H/transformers/src/transformers/models/siglip2/image_processing_siglip2.py
- X:/H/transformers/src/transformers/models/siglip2/image_processing_pil_siglip2.py
- X:/H/transformers/src/transformers/models/siglip2/processing_siglip2.py
- X:/H/transformers/src/transformers/models/siglip2/tokenization_siglip2.py
Any missing files or assumptions: no gated native SigLIP2 configs observed. `modeling_siglip2.py`, `configuration_siglip2.py`, and `tokenization_siglip2.py` are generated from `modular_siglip2.py`; future source edits should target the modular file. Several official repos named `siglip2-*` still have `model_type: siglip` and route to the older SigLIP implementation/image processor, so they are listed as out-of-scope comparison configs rather than native SigLIP2 operator requirements.
```

Snapshots were written under `agents/plans/transformers/siglip2/_sources/`.

## 2. High-level architecture

SigLIP2 is a dual encoder for image-text retrieval / zero-shot image classification. It is not an autoregressive generator and has no KV cache. The native SigLIP2 path differs from classic ViT/SigLIP because the image processor converts an image into a padded sequence of flattened patches before the model; the model's vision embedding is `Linear(C * patch_size * patch_size -> vision_hidden_size)`, not `Conv2d`.

```text
CPU image/text preprocessing
  -> text tokens + attention_mask
  -> flattened image patches + pixel_attention_mask + spatial_shapes
  -> independent text encoder and vision encoder
  -> branch pooling/projection
  -> L2 normalize text/image embeddings
  -> logits_per_text = text_embeds @ image_embeds.T * exp(logit_scale) + logit_bias
  -> logits_per_image = logits_per_text.T
```

Independently stageable units:
- Text encoder feature path: token/position embeddings, bidirectional encoder, final LayerNorm, last-token projection.
- Vision encoder feature path: patch Linear, dynamic positional interpolation/padding, bidirectional encoder, final LayerNorm, masked multi-head attention pooling.
- Similarity head: L2 normalize both branches, matrix multiply, scalar scale and scalar bias.

## 3. Important config dimensions

Native SigLIP2 checkpoint sweep:

| Checkpoint | Native model type | Text hidden/layers/heads | Vision hidden/layers/heads | Head dim | Text MLP | Vision MLP | Projection | Vocab | Text max pos | Patch input | Max patches / pos table | dtype source |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `google/siglip2-base-patch16-naflex` | `siglip2` | 768 / 12 / 12 effective defaults | 768 / 12 / 12 effective defaults | 64 | 3072 | 3072 | 768 default | 256000 | 64 default | `3*16*16=768` | 256 default | config `float32` |
| `google/siglip2-so400m-patch16-naflex` | `siglip2` | 1152 / 27 / 16 | 1152 / 27 / 16 | 72 | 4304 | 4304 | 1152 | 256000 | 64 default | `3*16*16=768` | 256 default | config `float32` |

Source defaults when omitted by config:

| Field | Text default | Vision default | Notes |
|---|---:|---:|---|
| `hidden_act` | `gelu_pytorch_tanh` | `gelu_pytorch_tanh` | Ungated FFN. |
| `layer_norm_eps` | `1e-6` | `1e-6` | Standard `LayerNorm`. |
| `attention_dropout` | `0.0` | `0.0` | Dropout disabled for inference. |
| `projection_size` | `hidden_size` | n/a | Text projection must match vision pool width for similarity. |
| `patch_size` | n/a | 16 | Current native configs rely on source default. |
| `num_patches` | n/a | 256 | Position table is square-root reshaped to `16 x 16`. |

Out-of-scope named `siglip2-*` comparison configs:

| Checkpoint | Observed config routing | Why not native SigLIP2 here |
|---|---|---|
| `google/siglip2-base-patch16-224` | `model_type: siglip`, `SiglipImageProcessor` | Fixed-image older SigLIP implementation, not `siglip2` packed patch ABI. |
| `google/siglip2-base-patch32-256` | `model_type: siglip`, `SiglipImageProcessor` | Same; useful for a separate SigLIP report, not this one. |
| `google/siglip2-large-patch16-512` | `model_type: siglip`, `SiglipImageProcessor` | Same; includes larger dimensions but different source family. |
| `google/siglip2-so400m-patch14-384` | `model_type: siglip`, `SiglipImageProcessor` | Patch14 fixed-size SigLIP-routed variant. |
| `google/siglip2-giant-opt-patch16-384` | `model_type: siglip`, `SiglipImageProcessor` | SigLIP-routed and has text/vision width asymmetry handled by the older source. |

## 3a. Family variation traps

- Native SigLIP2 currently means `model_type: siglip2`; do not route fixed-size `model_type: siglip` repos through this packed-patch graph.
- The vision branch input contract is already patchified: `pixel_values[B, max_num_patches, C*p*p]`, `pixel_attention_mask[B, max_num_patches]`, and `spatial_shapes[B, 2]`.
- Position embeddings are stored as a fixed square table of length `num_patches`; for each image, source bilinear-interpolates to `spatial_shapes[i]`, flattens, and pads remaining positions with the first resized embedding.
- Vision pooling is a learned single-probe cross-attention over patch tokens, not CLS-token pooling.
- Text pooling uses `last_hidden_state[:, -1, :]` after fixed-length padding, and the source comment says this may be padding. Processor defaults train/use `padding="max_length"`, `truncation=True`, `max_length=64`.
- Source `Siglip2Tokenizer` declares `padding_side = "left"`, but sampled HF `tokenizer_config.json` for native NaFlex repos says `tokenizer_class: GemmaTokenizer` and `padding_side: right`. For end-to-end parity, prefer the checkpoint tokenizer config over source-class defaults.
- Both encoders use bidirectional self-attention; there is no causal mask, no decode loop, and no KV cache.
- Attention is MHA only; no GQA/MQA, RoPE, ALiBi, sliding window, or relative bias.
- Layout trap: NCHW exists only in the CPU/PIL/Torchvision image processor before patch flattening. The model graph operates on `[B, Npatch, flat_patch]` and `[B, S, H]`. NHWC conversion should be guarded around patchification only, not a global model translation.

## 4. Operator coverage checklist

### Tensor/layout ops

- CPU/data-pipeline image resize to dynamic `(height, width)` divisible by `patch_size`.
- Image normalize/rescale with mean/std `[0.5, 0.5, 0.5]`.
- Patchify CHW image: reshape `[C, H, W] -> [C, Hp, p, Wp, p]`, permute to `[Hp, Wp, p, p, C]`, flatten to `[Hp*Wp, C*p*p]`.
- Pad patch sequence to `max_num_patches`, produce `pixel_attention_mask`.
- Runtime reshape/transposes for QKV: `[B, T, H] -> [B, heads, T, head_dim]`.
- L2 norm along feature dim, matrix transpose, scalar broadcast add/mul.

### Neural network primitives

Base native path:
- Text token embedding `Embedding(256000, 768)`, position embedding `Embedding(64, 768)`.
- Vision patch projection `Linear(768 -> 768, bias=True)`.
- Encoder layers: `LayerNorm(768, eps=1e-6)`, `Linear(768 -> 768)` Q/K/V/O, `Linear(768 -> 3072)`, GELU-tanh, `Linear(3072 -> 768)`.
- Text head `Linear(768 -> 768)`.
- Vision pooling `MultiheadAttention(embed=768, heads=12)` with packed `in_proj_weight[2304, 768]`, `out_proj[768, 768]`, followed by `LayerNorm` and MLP `768 -> 3072 -> 768`.

SO400M native path:
- Text token embedding `Embedding(256000, 1152)`, position embedding `Embedding(64, 1152)`.
- Vision patch projection `Linear(768 -> 1152, bias=True)`.
- Encoder layers: `LayerNorm(1152, eps=1e-6)`, Q/K/V/O `Linear(1152 -> 1152)`, MLP `1152 -> 4304 -> 1152`.
- Text head `Linear(1152 -> 1152)`.
- Vision pooling `MultiheadAttention(embed=1152, heads=16)` with packed `in_proj_weight[3456, 1152]`, then MLP `1152 -> 4304 -> 1152`.

### Attention primitives

- Bidirectional dense encoder self-attention for text and vision.
- Single-query cross-attention pooling: learned probe `q[B,1,H]`, keys/values `vision_hidden[B,N,H]`, mask over source patches.
- SDPA-compatible encoder attention via `ALL_ATTENTION_FUNCTIONS`; source marks FlashAttention unsupported.
- Eager fallback: matmul scores, add additive mask, `softmax(..., dtype=float32)`, dropout, value matmul.

### Position/custom math ops

- Text absolute position embedding add.
- Vision dynamic bilinear interpolation of position table with `align_corners=False`, `antialias=True`, CPU upcast to fp32 for interpolation, pad tail with first resized position row.

### Generation/cache ops

- No autoregressive cache, no encoder-decoder cache, no generation controller.
- Optional retrieval cache: text embeddings and image embeddings can be persisted independently before the similarity matrix.

### Preprocessing-coupled ops

- Binary search scale selection for max patch budget.
- Patch padding mask and `spatial_shapes` must be preserved into the model graph.
- Tokenizer max-length padding/truncation to length 64 for training-parity text features.

## 5. Layer/block breakdown

Text encoder, repeated `Lt` times:

```text
input_ids[B,S] -> token_embedding[B,S,Ht] + position_embedding[1,S,Ht]
mask = bidirectional_mask(attention_mask[B,S])
for layer:
  r = x
  x = LayerNorm(x)
  q,k,v = Linear(Ht -> Ht, bias=True)(x), split to heads
  x = dense noncausal self_attention(q,k,v, mask)
  x = r + Linear(Ht -> Ht, bias=True)(x)
  r = x
  x = LayerNorm(x)
  x = Linear(Ht -> It, bias=True) -> gelu_pytorch_tanh -> Linear(It -> Ht, bias=True)
  x = r + x
x = final LayerNorm(x)
pool = Linear(Ht -> projection_size, bias=True)(x[:, -1, :])
```

Vision encoder, repeated `Lv` times:

```text
pixel_values[B,N,C*p*p]
patch = Linear(C*p*p -> Hv, bias=True)(pixel_values)
pos = resize_and_pad_position_table(position_embedding[num_patches,Hv], spatial_shapes[B,2], N)
x = patch + pos
mask = bidirectional_mask(pixel_attention_mask[B,N])
for layer:
  same pre-norm noncausal MHA + ungated MLP as text, using Hv/Iv
x = post LayerNorm(x)
pool = learned_probe_cross_attention(query[B,1,Hv], key/value=x, source_mask=pixel_attention_mask)
pool = residual + MLP(LayerNorm(pool))
pool = pool[:, 0, :]
```

Similarity head:

```text
image = image_pool / ||image_pool||_2
text = text_pool / ||text_pool||_2
logits_per_text[Bt,Bi] = text @ image.T
logits_per_text = logits_per_text * exp(logit_scale[1]) + logit_bias[1]
logits_per_image[Bi,Bt] = logits_per_text.T
```

## 6. Attention requirements

Encoder attention:
- Noncausal bidirectional self-attention for both text and image.
- MHA only, with `num_kv_heads == num_attention_heads`.
- Base: 12 heads, head dim 64. SO400M: 16 heads, head dim 72.
- Query/key/value widths are all `hidden_size`.
- Mask comes from `create_bidirectional_mask`; runtime should accept additive masks broadcastable to `[B, heads, Q, K]`.
- Eager math order is score matmul, scale by `head_dim**-0.5`, add mask, softmax upcast to fp32, cast back, value matmul.
- Source declares SDPA support and FlashAttention unsupported.

Vision pooling attention:
- Query-driven cross-attention, not generation decode.
- Query source: learned `probe[1,1,Hv]`, repeated to `[B,1,Hv]`.
- Key/value source: final vision sequence `[B,N,Hv]`.
- Q/K/V widths are all `Hv`; output length is 1.
- Source uses `torch.nn.MultiheadAttention(batch_first=True)` and reshapes mask to `[B*num_heads, 1, N]`. Boolean masks are converted to additive 0 / dtype-min values because PyTorch MHA does not accept the boolean mask form used by SDPA.
- No KV cache; this pooling result is independently cacheable only as an image embedding.

## 7. Position encoding and custom math

Vision position interpolation is the only model-specific position math:

```python
def siglip2_resize_pos(pos_weight, spatial_shapes, max_length):
    # pos_weight: [num_patches, dim], num_patches must be square
    side = int(num_patches ** 0.5)
    grid = pos_weight.reshape(side, side, dim).permute(2, 0, 1)[None]
    out = empty([batch, max_length, dim], dtype=pos_weight.dtype)
    for i, (h, w) in enumerate(spatial_shapes):
        resized = interpolate(grid, size=(h, w), mode="bilinear",
                              align_corners=False, antialias=True)
        flat = resized.reshape(dim, h * w).T.to(pos_weight.dtype)
        out[i, : h * w] = flat
        out[i, h * w :] = flat[0]
    return out
```

Precompute candidates:
- The source position table can be reshaped to `[side, side, H]` at load time.
- Per-`(patch_h, patch_w)` resized tables can be cached if batches reuse shapes.

Dynamic inputs:
- `spatial_shapes[B,2]` drives interpolation and the number of valid patch positions.
- Tail padding values depend on each sample's first resized position row.

## 8. Preprocessing and input packing

Image CPU/data-pipeline contract:
- Accept image in channel-first internal processor tensor `[C,H,W]`.
- If resize enabled, find a scale by binary search so `ceil(H*scale/p)*ceil(W*scale/p) <= max_num_patches`, with both dimensions rounded up to patch multiples and at least one patch.
- Resize bilinear, rescale by `1/255`, normalize `(x - 0.5) / 0.5`.
- Patchify to `[num_patches, C*p*p]`, pad to `[max_num_patches, C*p*p]`.
- Emit `pixel_attention_mask[max_num_patches]` with 1 for real patches and 0 for pad patches.
- Emit `spatial_shapes=(patch_h, patch_w)` before padding.

Text CPU/data-pipeline contract:
- Processor defaults: `padding="max_length"`, `truncation=True`, `max_length=64`.
- Source tokenizer is Gemma/BPE-like with lowercase normalization, special IDs in source defaults `pad=0`, `eos=1`, `bos=2`, `unk=3`, `mask=4`.
- Sampled HF tokenizer config for native NaFlex repos advertises `GemmaTokenizer`, `add_eos_token=true`, `add_bos_token=false`, `padding_side=right`; preserve checkpoint tokenizer behavior for end-to-end parity.

Dual-encoder cache contract:
- Text embeddings `[Bt,D]` and image embeddings `[Bi,D]` can be cached independently after L2 normalization.
- Similarity output orientation is explicit: `logits_per_text[Bt,Bi]`, `logits_per_image[Bi,Bt]`.

## 9. Graph rewrite / lowering opportunities

### Rewrite: processor patchify -> local NHWC patch flatten

Source pattern:
```text
CHW image -> reshape(C, Hp, p, Wp, p) -> permute(Hp, Wp, p, p, C) -> flatten
```

Replacement:
```text
NHWC image -> WindowFlatten(k=p, stride=p, order=(row, col, channel)) -> padded patch sequence
```

Preconditions:
- Processor image dimensions are divisible by `patch_size`.
- Patch windows are non-overlapping.
- Flatten order exactly matches source final order `[patch_h, patch_w, row_in_patch, col_in_patch, channel]`.
- All downstream consumers accept `[B,N,C*p*p]`; no raw NCHW image tensor enters the model.

Layout constraints:
- This is a local processor/lowering rewrite, not a global model layout conversion.
- A conceptual `no_layout_translation()` guard should protect `pixel_values[B,N,flat]`, `pixel_attention_mask[B,N]`, and `spatial_shapes[B,2]`.

Failure cases:
- Different processor backend changes flatten order.
- `do_resize=False` with non-divisible image dimensions.
- Future checkpoints with non-square or list-valued patch sizes need separate guards.

Parity sketch:
- Compare patch rows and masks for random images with several aspect ratios, including images needing pad tokens.

### Rewrite: Linear patch projection fused with patch flatten

Source pattern:
```text
patches[B,N,C*p*p] -> Linear(C*p*p -> H)
```

Replacement:
```text
WindowFlatten -> MatMul(W.T) -> BiasAdd
```

Preconditions:
- Same patch flatten order as above.
- `patch_embedding.weight` has shape `[H, C*p*p]`, bias present.
- No per-patch preprocessing remains between flatten and Linear.

Weight transform:
```python
w = patch_embedding.weight  # [out_hidden, C*p*p], no Conv2d permutation unless source order changes
b = patch_embedding.bias
```

### Rewrite: QKV linears -> fused projection

Source pattern:
```text
q = Linear(H -> H); k = Linear(H -> H); v = Linear(H -> H)
```

Replacement:
```text
Linear(H -> 3H) -> split [Q, K, V] row blocks -> MHA
```

Preconditions:
- Same input tensor feeds all three projections.
- Biases are enabled and concatenated `[q_bias, k_bias, v_bias]`.
- Split order is all-Q, all-K, all-V row blocks. Do not reuse PyTorch `MultiheadAttention` packed weight order for encoder layers without transforming names.

### Rewrite: single-probe pooling MHA -> specialized attention pooling

Source pattern:
```text
probe[B,1,H] attends to hidden[B,N,H] with source mask
```

Replacement:
```text
Q(probe) + KV(hidden) -> masked attention with Q=1 -> out_proj -> residual/MLP
```

Preconditions:
- Query length remains 1.
- Mask applies only to source patch tokens.
- Packed `in_proj_weight[3H,H]` is split in PyTorch MHA order `[Q;K;V]`.

Failure cases:
- Future configs disable `vision_use_head`, in which case no pooling head exists.

## 10. Kernel fusion candidates

Highest priority:
- Dense bidirectional encoder attention with fp32 softmax parity: both branches are transformer stacks and dominate runtime.
- QKV fused projection for text and vision encoder layers: repeated across 12/27 layers.
- Patchify + patch Linear fusion: removes temporary `[B,N,C*p*p]` materialization when the image pipeline runs on GPU or a fused preprocessor path.
- L2 normalize + similarity GEMM + scalar scale/bias: retrieval workloads often compare many texts and images; this is the production scoring hot path.

Medium priority:
- Dynamic position interpolation cache keyed by `(patch_h, patch_w, hidden_size)`.
- LayerNorm + Linear epilogues in encoder blocks.
- GELU-tanh MLP fusion.
- Single-query attention pooling specialized kernel.

Lower priority:
- Loss path for `return_loss=True`; training/eval-only.
- Image classification average-pooling head; optional target separate from contrastive retrieval.
- Tokenizer implementation; mostly CPU/data pipeline.

## 11. Runtime staging plan

1. Parse native `Siglip2Config`, reject or reroute `model_type: siglip` checkpoints to the SigLIP audit.
2. Load text and vision weights, including scalar `logit_scale` and `logit_bias`; preserve `MultiheadAttention` packed pooling weights.
3. Implement processor-compatible inputs: accept precomputed `pixel_values`, `pixel_attention_mask`, `spatial_shapes`, `input_ids`, and `attention_mask`.
4. Validate text encoder only: embeddings, masks, encoder, last-token projection.
5. Validate vision encoder only: patch Linear, resized position embeddings, mask, encoder, pooling head.
6. Validate contrastive logits orientation and scalar scale/bias.
7. Add optimized QKV/attention/MLP fusions.
8. Add GPU/NHWC patchify and patch-projection fusion as an optional guarded preprocessing/runtime path.

Stubs acceptable initially:
- CPU image preprocessing can remain outside DinoML if `pixel_values` and metadata are supplied.
- `return_loss`, image classification head, and hidden-state/attention recording can be skipped for retrieval parity.

## 12. Parity and validation plan

- Unit test `get_image_size_for_max_num_patches` against source logic for square, tall, wide, tiny, and already-divisible images.
- Unit test patch flatten order and padding mask for small synthetic CHW images.
- Unit test position resize/pad for several `spatial_shapes`, tolerances: fp32 `1e-5` absolute, fp16/bf16 `2e-2` around interpolation.
- One-layer text encoder parity for base and SO400M dimensions with random tokens/masks.
- One-layer vision encoder parity with random patch sequences and varied `spatial_shapes`.
- Pooling head parity with masks that include padded patches.
- Full `get_text_features`, `get_image_features`, and `forward` logits parity for native NaFlex checkpoints.
- Check logits orientation: `logits_per_image == logits_per_text.T`.
- Recommended tolerances: fp32 `rtol=1e-4, atol=1e-5`; fp16/bf16 `rtol=5e-2, atol=5e-2` for full graph with interpolation and attention.

## 13. Performance probes

- CPU image preprocessing throughput by image aspect ratio and `max_num_patches`.
- Patchify/patch-projection temporary memory and bandwidth with and without fusion.
- Text encoder throughput for batch-size and sequence-length 64 sweeps.
- Vision encoder throughput for `N=256` and mixed valid patch counts.
- Position interpolation cache hit/miss benchmark by `(patch_h, patch_w)`.
- Retrieval scoring sweep: `Bt x Bi` similarity matrix sizes and embedding-cache reuse.
- Attention backend comparison: eager vs SDPA for encoder attention; separate pooling MHA.
- End-to-end zero-shot image classification throughput with many labels cached as text embeddings.

## 14. Skip/defer list

- Training loss and `return_loss=True`.
- Gradient checkpointing.
- Image classification head unless classification is the first product target.
- Fixed-size `model_type: siglip` repos named `siglip2-*`; route to the SigLIP report.
- Multi-GPU tensor parallelism.
- Quantization-specific loading.
- FlashAttention enablement until source parity is checked; source explicitly does not claim support.
- Full tokenizer implementation inside DinoML; accept tokenized inputs first.

## 15. Final implementation checklist

- [ ] Parse `Siglip2Config` and effective defaults.
- [ ] Reject/reroute `model_type: siglip` named checkpoints.
- [ ] Load text encoder, vision encoder, pooling MHA, text projection, `logit_scale`, and `logit_bias`.
- [ ] Implement packed patch input ABI: `pixel_values`, `pixel_attention_mask`, `spatial_shapes`.
- [ ] Implement dynamic position resize/pad.
- [ ] Implement bidirectional dense MHA with fp32 softmax parity.
- [ ] Implement single-probe masked attention pooling.
- [ ] Implement L2 normalize and contrastive logits orientation.
- [ ] Add guarded patchify + Linear fusion.
- [ ] Add QKV fused projection rewrite.
- [ ] Add text, vision, pooling, and full-logits parity tests.
- [ ] Benchmark preprocessing, encoders, pooling, and retrieval GEMM separately.
