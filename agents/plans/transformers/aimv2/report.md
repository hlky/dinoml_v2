# AIMv2 Transformers Audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: apple/aimv2-large-patch14-224-lit for dual-encoder target; vision-only variants listed below
Config source: local configuration_aimv2.py plus HF raw config/preprocessor JSON fetched 2026-05-13
Source files inspected:
- X:/H/transformers/src/transformers/models/aimv2/configuration_aimv2.py
- X:/H/transformers/src/transformers/models/aimv2/modeling_aimv2.py
- X:/H/transformers/src/transformers/models/aimv2/modular_aimv2.py
- X:/H/transformers/src/transformers/models/aimv2/convert_aimv2_original_pytorch_to_hf.py
Any missing files or assumptions:
- No family-owned tokenizer/image processor implementation exists in-tree; AIMv2 uses CLIP image processor/tokenizer metadata on HF repos.
- modeling_aimv2.py and configuration_aimv2.py are generated from modular_aimv2.py. Runtime behavior was checked in generated modeling_aimv2.py; future Transformers source edits should be made in modular_aimv2.py.
- Some HF repos still expose remote-code auto_map names. This report is scoped to the in-library source at the pinned commit, not remote-code copies.
```

Representative HF configs/processors inspected: `apple/aimv2-large-patch14-224`, `apple/aimv2-huge-patch14-224`, `apple/aimv2-1B-patch14-224`, `apple/aimv2-3B-patch14-224`, `apple/aimv2-large-patch14-336`, `apple/aimv2-large-patch14-448`, `apple/aimv2-large-patch14-native`, and `apple/aimv2-large-patch14-224-lit`. See `config_snapshot.md`.

## 2. High-level architecture

Primary target: contrastive image-text inference for `Aimv2Model` (`apple/aimv2-large-patch14-224-lit`), with independently stageable `Aimv2VisionModel` feature extraction as the first useful subtarget.

```text
image preprocessing -> vision patch embedding -> vision encoder -> final RMSNorm -> attention-pool head -> visual projection -> L2 normalize
text tokenization -> token + position embedding -> causal-masked text encoder -> final RMSNorm -> EOS pooling -> text projection -> L2 normalize
normalized text/image features -> clamped exp(logit_scale) -> text @ image.T -> logits_per_text/logits_per_image
```

There is no autoregressive generation decode path and no KV cache. The text branch is causal self-attention because the source builds a causal additive mask when `attention_mask` is supplied, but it returns pooled sequence features for contrastive scoring.

Stage decomposition:

- CPU/data pipeline: image decode/resize/crop/RGB/rescale/normalize to NCHW `pixel_values`; CLIP BPE tokenization, padding, special tokens.
- Vision encoder: cacheable per image; outputs patch sequence `[B, H_p * W_p, D_v]` and optional pooled embedding.
- Text encoder: cacheable per text prompt; outputs sequence `[B, L, D_t]` and EOS-pooled embedding.
- Similarity head: projections to `projection_dim`, L2 norms, scalar logit scale, matrix multiply. This is independently testable and tiny relative to encoders.

## 3. Important config dimensions

Source defaults:

| Field | Vision default | Text default | `lit` composite |
|---|---:|---:|---:|
| `hidden_size` | 1024 | 768 | vision 1024, text 768 |
| `num_hidden_layers` | 24 | 12 | vision 24, text 12 |
| `num_attention_heads` | 8 | 6 | vision 8, text 6 |
| `head_dim` | `hidden_size / heads` = 128 | 128 | both 128 |
| `intermediate_size` | 2816 | 2048 | same as branch |
| `hidden_act` | `silu` | `silu` | SwiGLU-style gated MLP |
| `qkv_bias` / `mlp_bias` | false / false | false / false | false / false |
| `rms_norm_eps` | `1e-5` | `1e-5` | same |
| `image_size` / `patch_size` | 224 / 14 | n/a | vision 224 / 14 |
| `max_position_embeddings` | n/a | 77 | text 77 |
| `vocab_size` | n/a | 49408 | text 49408 |
| `projection_dim` | n/a | n/a | 512 default, 768 in `lit` config |
| cache support | none | none | none |

Representative checkpoint sweep:

| Model id | Target | Hidden | Layers | Heads | MLP | Image/patch grid | Head | Operator-significant variation |
|---|---|---:|---:|---:|---:|---|---|---|
| `apple/aimv2-large-patch14-224` | vision features | 1024 | 24 | 8 | 2816 | 224 -> 16x16 | no | base vision-only fixed learned pos table |
| `apple/aimv2-huge-patch14-224` | vision features | 1536 | 24 | 12 | 4096 | 224 -> 16x16 | no | wider GEMMs/attention, same token count |
| `apple/aimv2-1B-patch14-224` | vision features | 2048 | 24 | 16 | 5632 | 224 -> 16x16 | no | wider GEMMs/attention, same token count |
| `apple/aimv2-3B-patch14-224` | vision features | 3072 | 24 | 24 | 8192 | 224 -> 16x16 | no | very wide GEMMs/attention, same token count |
| `apple/aimv2-large-patch14-336` | vision features | 1024 | 24 | 8 | 2816 | 336 -> 24x24 | no | longer sequence and larger learned pos table |
| `apple/aimv2-large-patch14-448` | vision features | 1024 | 24 | 8 | 2816 | 448 -> 32x32 | no | longest fixed-resolution sequence in sampled configs |
| `apple/aimv2-large-patch14-native` | native vision features | 1024 | 24 | 8 | 2816 | dynamic by input | no | generated 2D sinusoidal positions; no resize/crop |
| `apple/aimv2-large-patch14-224-lit` | contrastive image-text | vision 1024, text 768 | 24 + 12 | 8 + 6 | 2816 + 2048 | 224 -> 16x16, text 77 | yes | dual encoder, attention pooling, projections, similarity logits |

## 3a. Family variation traps

- `use_head` changes vision outputs. Vision-only sampled repos set `use_head=false`, so `pooler_output=None`; `lit` sets `use_head=true` and needs the attention pooling head.
- `is_native=true` removes learned vision position embeddings and generates a 2D sinusoid at runtime from input height/width.
- Fixed-resolution checkpoints do not implement position interpolation in source. `get_image_features(interpolate_pos_encoding=...)` accepts the kwarg but `Aimv2VisionEmbeddings.forward` does not read it; mismatched image size versus learned position table should be rejected or separately handled.
- Vision input source layout is NCHW. NHWC is an optimization only around the processor-to-patch-embedding region.
- The patch embedding source uses `Conv2d(C, D, kernel_size=patch_size, stride=patch_size)` then `flatten(2).transpose(1, 2)`, so row-major patch order is part of the ABI.
- `qkv_bias` and `mlp_bias` are config-driven even though sampled checkpoints set them false.
- The converter source splits original packed `qkv` weights into separate `q_proj`, `k_proj`, `v_proj` in Q, K, V order. HF in-library weights are separate Linear modules.
- The text model uses EOS pooling by first EOS occurrence: `(input_ids == eos_token_id).int().argmax(dim=-1)`. Missing EOS pools token index 0, so admission should require tokenizer-generated EOS for parity.
- `projection_dim` defaults to 512 in source but is 768 in `lit`.
- HF configs contain fields such as `projection_dropout`, `use_bias`, `num_queries`, `is_causal`, `init_temperature`, or `max_context_length` that the inspected in-library source mostly does not read. Do not treat them as required source behavior.
- HF model repos are public and ungated in sampled official IDs. Probes for mistyped `apple/aimv2-large-patch14-384` and `apple/aimv2-large-patch14-distilled` returned 401/unauthorized; collection-listed distilled IDs use `...-224-distilled` and `...-336-distilled`.

## 4. Operator coverage checklist

Tensor/layout ops:

- NCHW image tensor input `[B, 3, H, W]`.
- Conv2d patch embedding, non-overlap: `3 -> D_v`, kernel/stride `14`, no explicit padding/dilation/groups in source.
- `flatten(2)`, `transpose(1, 2)`, reshape/view, permute, contiguous.
- Add position embeddings with broadcast `[1, S, D]`.
- EOS gather: equality, cast to int, `argmax(dim=-1)`, batch gather.
- L2 normalization via pow, sum over `dim=-1`, sqrt, divide.
- Feature similarity: projection GEMMs, transpose, matrix multiply, scalar multiply.

Neural primitives:

- RMSNorm over last dim with fp32 variance path and learned scale.
- Linear projections with optional bias.
- Gated MLP: `down_proj(silu(gate_proj(x)) * up_proj(x))`.
- Residual adds after attention and MLP.
- CLIP token and absolute position embeddings for text.

Attention primitives:

- Dense noncausal MHA for vision, no KV cache.
- Dense causal self-attention for text when `attention_mask` is supplied; source passes an additive 4D causal mask from `create_causal_mask`.
- Attention pooling head: learned `[1,1,D]` query token, K/V from vision sequence, SDPA with query length 1, no explicit mask.
- Attention backends: source declares SDPA, FlashAttention, and FlexAttention support through Transformers attention interface; eager fallback is matmul -> additive mask -> fp32 softmax -> dropout -> matmul.

Position/math:

- Learned absolute position table for fixed vision configs: `Embedding(num_patches, D_v)`.
- Runtime-generated 2D sinusoidal table for native vision configs, requiring `D % 4 == 0`.
- Learned text position table, length 77 by default.
- Logit scale: `exp(clamp(logit_scale, 0, log(max_logit_scale)))`.

Preprocessing-coupled ops:

- CLIP image preprocessing to channels-first float tensor.
- CLIP BPE tokenizer for `lit`, max length 77, BOS id 49406, EOS/PAD/UNK id 49407.

Not applicable:

- No RoPE, ALiBi, relative bias, MoE, quantized packed weight format, recurrent state, KV cache, varlen packed metadata, multimodal scatter/stitch, or generation logits head in inspected source.

## 5. Layer/block breakdown

Vision embeddings:

```text
pixel_values: [B, 3, H, W] in NCHW
patch = Conv2d(3 -> Dv, kernel=P, stride=P)(pixel_values)       # [B, Dv, H/P, W/P]
x = patch.flatten(2).transpose(1, 2)                            # [B, S, Dv], S=(H/P)*(W/P)
x = RMSNorm(x)
pos = learned_position[0:S] or generated_native_2d_sinusoid      # [1, S, Dv]
x = x + pos
```

Encoder block, repeated `num_hidden_layers`:

```text
y = RMSNorm(x)
q = Linear(D -> D, bias=qkv_bias)(y).view(B, S, heads, head_dim).transpose(1, 2)
k = Linear(D -> D, bias=qkv_bias)(y).view(B, S, heads, head_dim).transpose(1, 2)
v = Linear(D -> D, bias=qkv_bias)(y).view(B, S, heads, head_dim).transpose(1, 2)
a = Attention(q, k, v, additive_mask, scale=head_dim**-0.5)
x = x + Linear(D -> D, bias=qkv_bias)(a)
y = RMSNorm(x)
m = Linear(intermediate -> D, bias=mlp_bias)(silu(Linear(D -> intermediate)(y)) * Linear(D -> intermediate)(y))
x = x + m
```

Vision model tail:

```text
last_hidden_state = RMSNorm(encoder_output)
if use_head:
  cls = learned_cls.expand(B, 1, Dv)
  q = cls.reshape(B, 1, H, Dh).permute(0, 2, 1, 3)
  k/v = Linear(Dv -> Dv)(last_hidden_state).reshape(B, S, H, Dh).permute(0, 2, 1, 3)
  pooled = SDPA(q, k, v).transpose(1, 2).reshape(B, 1, Dv).mean(dim=1)
  pooled = Linear(Dv -> Dv, bias=true)(pooled)
else:
  pooled = None
```

Text model:

```text
x = token_embedding(input_ids) + position_embedding(arange(seq_len))   # [B, L, Dt]
mask = create_causal_mask(...) if attention_mask is not None else None
x = shared Aimv2Encoder blocks with Dt/text heads/text MLP
last_hidden_state = RMSNorm(x)
pooler_output = last_hidden_state[batch_index, first_index(input_ids == eos_token_id)]
```

Composite contrastive head:

```text
image = visual_projection(vision_pooler)      # [B_i, projection_dim]
text = text_projection(text_pooler)           # [B_t, projection_dim]
image = image / sqrt(sum(image**2, dim=-1, keepdim=True))
text = text / sqrt(sum(text**2, dim=-1, keepdim=True))
scale = exp(clamp(logit_scale, 0, log(max_logit_scale)))
logits_per_text = (scale * text) @ image.T    # [B_t, B_i]
logits_per_image = logits_per_text.T          # [B_i, B_t]
```

## 6. Attention requirements

Vision encoder attention:

- Noncausal self-attention over patch tokens.
- MHA only; no GQA/MQA because Q/K/V widths all equal `hidden_size`.
- Head dim is `hidden_size // num_attention_heads`; sampled configs keep it 128.
- No attention mask in ordinary vision forward.
- Source may dispatch to SDPA/Flash/Flex through Transformers interface; eager fallback uses fp32 softmax and returns `[B, heads, S, Dh] -> [B, S, D]`.

Text encoder attention:

- Causal self-attention used for contrastive text encoding, not autoregressive generation.
- No past key/value cache and `past_key_values=None` in mask creation.
- Attention mask, when supplied, is converted to an additive causal mask. If no `attention_mask` is supplied, source passes `None`, and backend causal behavior depends on `self.is_causal=False`; first integration should supply and validate tokenizer attention masks for padded batches.

Attention pooling:

- Query-driven cross-attention from a learned single query to the vision patch sequence.
- Query shape `[B, heads, 1, Dh]`, key/value `[B, heads, S, Dh]`.
- No mask, no cache, no generation semantics.

Packed/varlen, local/sliding-window, relative bias, RoPE, ALiBi, and dense attention reconstruction are not required.

## 7. Position encoding and custom math

Native vision position embedding:

```python
def aimv2_native_pos(height_patches, width_patches, dim, temperature=10000.0):
    assert dim % 4 == 0
    pos_dim = dim // 4
    omega = arange(pos_dim, float64) / pos_dim
    omega = 1.0 / (temperature ** omega)
    grid_h, grid_w = meshgrid(arange(height_patches), arange(width_patches), indexing="ij")
    emb_h = flatten(grid_h).outer(omega)
    emb_w = flatten(grid_w).outer(omega)
    canonical = concat([sin(emb_h), cos(emb_h), sin(emb_w), cos(emb_w)], dim=1)
    half = dim // 2
    return concat([canonical[..., half:], canonical[..., :half]], dim=-1)
```

The final half rotation is source-specific: AIMv2 native checkpoints expect `[sin_w | cos_w | sin_h | cos_h]` layout even though the helper builds h-first. This can be precomputed per admitted `(H/P, W/P, D)` bucket or generated at runtime. Fixed-resolution models use learned position weights and should not use this path.

Text positions are learned absolute positions indexed from 0 to `seq_len - 1`; source rejects `seq_len > max_position_embeddings`.

RMSNorm custom math:

```python
variance = mean(x.float() ** 2, dim=-1, keepdim=True)
y = weight * (x.float() * rsqrt(variance + eps)).to(input_dtype)
```

## 8. Preprocessing and input packing

Vision preprocessing from sampled `preprocessor_config.json`:

- `CLIPImageProcessorFast`.
- Convert RGB, rescale by `1/255`, normalize with CLIP mean `[0.48145466, 0.4578275, 0.40821073]` and std `[0.26862954, 0.26130258, 0.27577711]`.
- Output `data_format="channels_first"`, so runtime tensor is `[B, 3, H, W]`.
- Fixed 224/336/448 repos resize shortest edge then center crop to square crop size.
- Native repo disables resize and center crop. Dinoml should require input H/W divisible by patch size and either precompute a position bucket or enable native position generation.

Text preprocessing for `lit`:

- `CLIPTokenizer` through `CLIPProcessor`.
- `model_max_length=77`; BOS `<start_of_text>` id 49406; EOS/PAD/UNK `<end_of_text>` id 49407.
- `attention_mask` enters causal mask construction.
- Source ignores a `position_ids` kwarg in `Aimv2TextModel.forward`; embeddings create positions internally. Do not expose a required external `position_ids` ABI for the in-library source.

No image/text embedding stitch, placeholder tokens, modality type ids, grid metadata, packed sequence descriptors, or scatter APIs are used.

## 9. Graph rewrite / lowering opportunities

### Rewrite: non-overlap Conv2d patch embed -> GEMM

Source pattern:

```text
Conv2d(C -> D, kernel=P, stride=P, padding=0, dilation=1, groups=1)
flatten(2).transpose(1, 2)
```

Replacement:

```text
WindowFlatten_NCHW_row_major([B,C,H,W], P, P) -> Linear(C*P*P -> D, bias) -> [B, (H/P)*(W/P), D]
```

Preconditions:

- `kernel_size == stride == patch_size`.
- `padding == 0`, `dilation == 1`, `groups == 1`.
- `H % P == 0` and `W % P == 0`.
- Flatten order must match PyTorch Conv2d plus `flatten(2).transpose(1,2)`: token order is row-major over output height/width; each patch uses NCHW convolution weight layout.

Weight transform:

```python
linear_weight = conv.weight.reshape(out_channels, in_channels * patch_h * patch_w)
linear_bias = conv.bias
```

Failure cases: native odd-sized inputs, non-square/list patch config not fully normalized, dynamic sizes without divisibility guards, or any future padding/dilation/groups.

Parity test: compare Conv2d path versus window-flatten GEMM for 224, 336, native rectangular divisible input, fp32/fp16.

### Rewrite: separate Q/K/V Linear -> packed QKV GEMM

Source pattern: three independent `Linear(D -> D, bias=qkv_bias)` then view/transpose.

Replacement: one `Linear(D -> 3D)` with row-concatenated weights `[q_proj; k_proj; v_proj]`, split output in Q,K,V order.

Preconditions:

- Same input tensor to all three projections.
- Same dtype/device and identical bias setting.
- Output split order Q, K, V. This matches converter split logic from original packed `qkv` weights.

Failure cases: any branch-specific quantization, LoRA/adapters, or config variants with unequal Q/K/V widths.

### Rewrite: RMSNorm + projection fusion

Source pattern: `RMSNorm(x)` immediately consumed by Q/K/V projections or MLP projections.

Replacement: fused RMSNorm kernel feeding packed GEMM or vectorized normalization followed by GEMM.

Preconditions: last-dim contiguous logical layout and fp32 variance semantics preserved.

Failure cases: exported hidden states requiring the normalized intermediate, non-contiguous views, or dtype tolerance too loose around fp32 accumulation.

### Rewrite: Attention pooling as single-query attention

Source pattern: learned cls query cross-attends to full vision tokens and `mean(dim=1)` over a length-1 axis.

Replacement: specialized single-query SDPA or GEMV-style attention pooling, removing redundant mean.

Preconditions: query length exactly 1, no mask, `hidden_size % num_heads == 0`, output length remains 1.

Failure cases: future configs with `num_queries > 1` actually implemented in source. Sampled configs have `num_queries` metadata, but source always creates one cls token.

### Layout rewrite: NCHW patch region to NHWC/channel-last

Candidate region: processor output through patch embedding/window flatten only.

Required axis rewrites:

- Source `Conv2d` expects channel axis 1. NHWC lowering must transform patch extraction from `[B,C,H,W]` to `[B,H,W,C]` and transform Conv weight flatten order to match NCHW source values.
- `flatten(2).transpose(1,2)` disappears if NHWC window flatten directly emits `[B,S,C*P*P]`.

Guard boundary: after patch embedding, source is sequence layout `[B,S,D]`; do not translate transformer `dim=-1` RMSNorm/softmax axes or EOS pooling axes as image layout axes.

Failure cases: callers provide already-normalized NCHW tensors and expect exact `get_input_embeddings()` Conv2d behavior; non-divisible native inputs; learned fixed-position table length mismatch.

## 10. Kernel fusion candidates

Highest priority:

- RMSNorm last-dim kernel with fp32 accumulation. It appears before attention, before MLP, after encoder, and after patch embedding.
- Packed QKV projection + reshape/transpose for MHA. This dominates encoder blocks and maps cleanly because all heads have head_dim 128.
- Dense attention for vision/text sequences, with fp32 softmax parity. Vision S is 256/576/1024 for common fixed configs; text L is <=77.
- SwiGLU MLP fusion: `silu(gate) * up` before down projection.

Medium priority:

- Conv patch embedding -> GEMM/window kernel, especially for 336/448 and native resolution.
- Attention pooling single-query kernel for `lit`.
- Projection + L2 normalize for text/image features.
- Similarity head GEMM with scalar scale, especially batched image-vs-text retrieval.

Lower priority:

- Runtime native 2D position generation; useful for dynamic/native inputs but precompute per bucket first.
- Full CLIP preprocessing on GPU. CPU/data pipeline parity is acceptable initially.
- Flash/Flex attention backend parity. Sequences are moderate; use dense SDPA first, then optimize.

## 11. Runtime staging plan

Stage 1: parse configs and load weights for `Aimv2VisionModel` fixed 224 vision-only. Reject `is_native=true` and `use_head=true` initially if needed.

Stage 2: implement one vision block parity with patch embedding, learned positions, RMSNorm, MHA, SwiGLU MLP.

Stage 3: full fixed-resolution vision encoder parity for 224, then 336/448 by learned position length.

Stage 4: add `use_head=true` attention pooling and visual projection for `lit` image features.

Stage 5: add text branch with CLIP token/position embeddings, causal additive mask, EOS pooling, text projection.

Stage 6: add contrastive similarity output orientation and logit scale clamp/exp.

Stage 7: add guarded rewrites/fusions: patch Conv->GEMM, packed QKV, fused RMSNorm, fused SwiGLU, optimized attention.

Stage 8: add native-resolution vision path with generated/precomputed 2D sin/cos positions and strict divisibility guards.

Stubs acceptable early: tokenizer/image processor as external CPU pipeline, no training losses, no `output_attentions`, no gradient checkpointing, no remote-code variants.

## 12. Parity and validation plan

- Unit parity for RMSNorm fp32 accumulation across fp32/fp16/bf16, tolerances around `1e-5` fp32 and `1e-2` fp16/bf16 depending on backend.
- Unit parity for native 2D position generation including final half rotation and dynamic rectangular grids.
- Patch embedding Conv2d versus GEMM rewrite parity for 224, 336, 448, and native divisible H/W.
- Single encoder layer parity with random tensors and real config dims for large and text branches.
- Full `Aimv2VisionModel` parity against Transformers for `apple/aimv2-large-patch14-224`, comparing `last_hidden_state`.
- `use_head` parity for `apple/aimv2-large-patch14-224-lit`, comparing `vision_model.pooler_output` and projected image features.
- Text branch parity with tokenizer-generated padded inputs, comparing `last_hidden_state`, EOS pooled output, and projected text features.
- End-to-end `lit` parity: compare `image_embeds`, `text_embeds`, `logits_per_text`, and `logits_per_image`; assert output orientation `[text_batch, image_batch]` and transpose.
- Native-resolution parity: compare `apple/aimv2-large-patch14-native` for at least one non-resized input divisible by 14.

## 13. Performance probes

- Image preprocessing throughput versus model throughput for 224/336/448.
- Patch embedding Conv2d versus GEMM/window kernel at batch sizes 1, 8, 32.
- Vision encoder throughput by sequence length 256/576/1024 and hidden widths 1024/1536/2048/3072.
- Attention backend comparison: eager-style dense, SDPA, FlashAttention-compatible path.
- MLP GEMM and SwiGLU fusion timing by hidden/intermediate width.
- `lit` branch split: image-only, text-only, projection/norm/similarity-only.
- Retrieval matrix scaling: `B_text x B_image` similarity at large candidate pools.
- Native position generation cost versus bucket precompute/cache.
- Memory probes for largest sampled configs, especially 3B at 448 if admitted later.

## 14. Skip/defer list

- Training, contrastive loss, dropout behavior beyond inference zeros, and gradient checkpointing.
- Remote-code copies and Flax/MLX weights.
- `output_attentions` materialization and hidden-state capture unless debugging requires them.
- Native-resolution path until fixed learned-position models pass.
- GPU image preprocessing.
- Distilled checkpoints until their exact public configs are fetched under correct IDs.
- Any historical config fields not read by in-library source, such as `projection_dropout`, `use_bias`, `num_queries`, and `max_context_length`.
- Autoregressive generation and KV cache; they are not source behavior for AIMv2.

## 15. Final implementation checklist

- [ ] Parse `Aimv2VisionConfig`, `Aimv2TextConfig`, and `Aimv2Config`.
- [ ] Load separate HF Q/K/V Linear weights and preserve Q,K,V split order for packed rewrites.
- [ ] Implement NCHW patch embedding and row-major patch tokenization.
- [ ] Implement learned vision/text absolute position embeddings.
- [ ] Implement AIMv2 RMSNorm with fp32 variance.
- [ ] Implement dense MHA with additive mask and fp32 softmax parity.
- [ ] Implement SwiGLU MLP with config-driven biases.
- [ ] Implement fixed-resolution vision encoder parity.
- [ ] Implement optional attention pooling head.
- [ ] Implement text encoder, causal mask admission, and EOS pooling.
- [ ] Implement projection, L2 normalization, logit scale clamp/exp, and similarity output orientation.
- [ ] Add guarded Conv2d patch -> GEMM rewrite.
- [ ] Add guarded Q/K/V -> packed QKV rewrite.
- [ ] Add native 2D sinusoidal position path with divisibility and `hidden_size % 4` guards.
- [ ] Add parity tests for single block, full vision, text branch, and end-to-end `lit`.
- [ ] Add performance probes for patch embedding, attention, MLP, and contrastive similarity.
