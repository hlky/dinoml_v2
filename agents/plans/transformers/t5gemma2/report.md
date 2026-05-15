# DinoML Transformers Audit: t5gemma2

## 1. Source basis

```text
Transformers commit/version:
  transformers @ b75feb2af64c3e29cbbc1bd859958c5432cc7ed4

Model id:
  google/t5gemma-2-270m-270m, google/t5gemma-2-1b-1b, google/t5gemma-2-4b-4b

Config source:
  Official config.json files are license-gated/manual-gated on Hugging Face and
  returned 401 without accepted Gemma access during this audit. Exact operator
  structure is from local Transformers source defaults. Public Hub/API metadata
  confirms the three official repos, `t5gemma2` tags, `image-text-to-text`
  pipeline, and manual gating.

Source files inspected:
  transformers/src/transformers/models/t5gemma2/configuration_t5gemma2.py
  transformers/src/transformers/models/t5gemma2/modeling_t5gemma2.py
  transformers/src/transformers/models/t5gemma2/modular_t5gemma2.py
  transformers/src/transformers/models/siglip/configuration_siglip.py
  transformers/src/transformers/models/siglip/modeling_siglip.py
  transformers/src/transformers/models/gemma3/processing_gemma3.py
  transformers/src/transformers/models/gemma3/image_processing_gemma3.py
  transformers/tests/models/t5gemma2/test_modeling_t5gemma2.py

Any missing files or assumptions:
  `modeling_t5gemma2.py` and `configuration_t5gemma2.py` are generated from
  `modular_t5gemma2.py`; generated files are the runnable source, modular is the
  future edit authority. Exact 1B/4B config dimensions need gated config access.
```

Primary DinoML target for this report: `T5Gemma2ForConditionalGeneration` for
image/text-to-text generation on CUDA. Sequence and token classification heads
are optional follow-ups. Training losses, dropout, gradient checkpointing, and
tensor parallel plans are out of scope for first inference parity.

## 2. High-level architecture

T5Gemma2 is an encoder-decoder multimodal generation family. The encoder is a
Gemma3-like text Transformer with an optional SigLIP vision tower and a
projection/stitch path for image tokens. The decoder is a causal text decoder,
but each decoder layer uses a single merged attention module that concatenates
self-attention keys/values with cross-attention keys/values before one attention
backend call.

```text
CPU image/text preprocessing
  -> SigLIP vision encoder for pixel_values
  -> AvgPool2d + RMSNorm + Linear projector to text hidden size
  -> image-token embedding stitch into encoder text embeddings
  -> bidirectional encoder with mixed sliding/full attention
  -> causal decoder with merged self+cross attention
  -> tied LM head, optional final logit softcap
  -> generation controller / sampling
```

Stage decomposition:

- CPU/data pipeline: Gemma3Processor expands each `<start_of_image>` into
  `boi + 256 image tokens + eoi`, optionally emits pan-and-scan crops, tokenizes
  text, and image-processes to NCHW `pixel_values`.
- Independently cacheable stage: SigLIP image tower plus T5Gemma2 multimodal
  projector produces `[num_images, mm_tokens_per_image, hidden_size]`.
- Encoder stage: text embeddings with stitched image features, bidirectional
  local/full attention, final RMSNorm.
- Prefill stage: decoder prompt plus encoder hidden states. Cross-attention
  K/V can be cached independently from autoregressive self K/V.
- Decode stage: one-token or small-chunk causal decoder with self-attention KV
  growth and full cross-attention cache reuse.

## 3. Important config dimensions

Source-default dimensions from `T5Gemma2TextConfig` / `T5Gemma2DecoderConfig`:

| Field | Source default | Notes |
|---|---:|---|
| `vocab_size` | 262208 | Shared encoder/decoder vocab required by config validation. |
| `hidden_size` | 2304 | Encoder text hidden size must equal decoder hidden size. |
| `intermediate_size` | 9216 | Gated MLP expansion. |
| `num_hidden_layers` | 26 | Encoder and decoder defaults match. |
| `num_attention_heads` | 8 | Q/O width is `8 * head_dim`. |
| `num_key_value_heads` | 4 | GQA, `num_key_value_groups = 2`. |
| `head_dim` | 256 | Q width is 2048, not equal to hidden size. |
| `max_position_embeddings` | 131072 | 128K input context claim matches model card. |
| `sliding_window` | 4096 | Used on `sliding_attention` layers. |
| `layer_types` | generated pattern | Default: every 6th layer is `full_attention`, others `sliding_attention`. |
| local/global RoPE theta | 10000 / 1000000 | Separate RoPE parameters per layer type. |
| `hidden_activation` | `gelu_pytorch_tanh` | Gated GELU MLP. |
| `attention_bias` | `False` | Text Q/K/V/O linears are bias-free by default. |
| `query_pre_attn_scalar` | 256 | Attention scale is `256**-0.5`, not `head_dim**-0.5` unless equal. |
| `attn_logit_softcapping` | `None` | Source eager path supports it. |
| `final_logit_softcapping` | `None` | Optional tanh softcap after LM head. |
| `mm_tokens_per_image` | 256 | Projected image tokens per image. |
| encoder image token ids | boi 255999, eoi 256000, image 262144 | Top-level `image_token_index` default is 256001 but copied into encoder in `T5Gemma2Config.__post_init__`; gated configs must be checked. |
| SigLIP vision default | 224 image, 16 patch, hidden 768, 12 layers | T5Gemma2 model card says images normalized to 896x896 and 256 tokens; exact official `vision_config` is gated. |

Representative checkpoint sweep:

| Checkpoint | Access | Public metadata | Config facts available |
|---|---|---|---|
| `google/t5gemma-2-270m-270m` | `config.json` 401/manual-gated | image-text-to-text, t5gemma2, Gemma license, model card says 270M encoder + 270M decoder and 0.8B total | Source defaults and integration tests target this id with bf16. |
| `google/t5gemma-2-1b-1b` | `config.json` 401/manual-gated | image-text-to-text, t5gemma2, 2B public collection label | Exact layer/head widths unavailable without access. |
| `google/t5gemma-2-4b-4b` | `config.json` 401/manual-gated | image-text-to-text, t5gemma2, 9B public collection label | Exact layer/head widths unavailable without access. |

## 3a. Family variation traps

- `hidden_size != num_attention_heads * head_dim` in source defaults:
  `2304 != 8 * 256`. Q/O projection width is 2048, while residual hidden width
  remains 2304.
- GQA is required: `num_key_value_heads < num_attention_heads`; K/V cache stores
  KV heads before repeat expansion.
- Layer type alternation matters. Local/sliding layers use local RoPE theta and
  sliding-window masks; full layers use global theta and full masks.
- Decoder attention is not ordinary separate self-attention plus cross-attention.
  It projects decoder and encoder states with the same K/V weights, caches them
  separately, then concatenates self K/V with cross K/V along sequence.
- Cross-attention cache must be forced to full attention; it must not inherit
  decoder sliding-window limits.
- Attention scale comes from `query_pre_attn_scalar**-0.5`; do not infer from
  `head_dim` unless the config makes them equal.
- Q/K post-projection RMSNorm is per-head-dim and happens before RoPE.
- Text embeddings are scaled by `sqrt(hidden_size)`, but the EOI token has a
  separate learned `eoi_embedding` and is not scaled through the normal table.
- Encoder/decoder embeddings and EOI embedding are tied; LM head weight is tied
  to encoder text embeddings. Preserve weight aliases.
- Image feature stitch uses broad `masked_scatter`, but the processor guarantees
  strict image-token expansion counts. DinoML should lower this only with
  placeholder count/order guards, not admit general boolean scatter.
- Processor emits NCHW `pixel_values`; SigLIP patch Conv2d consumes NCHW. NHWC is
  an optimization opportunity only for the controlled vision region.
- SigLIP position interpolation uses NHWC/NCHW permutes around bicubic
  interpolation if `interpolate_pos_encoding=True`; protect that path with
  no-layout-translation guards until explicitly owned.

## 4. Operator coverage checklist

Tensor/layout ops:

- Embedding lookup with scale and special EOI overwrite.
- Boolean equality masks for image token and EOI token detection.
- `masked_scatter` equivalent for replacing flattened image-token embedding
  slots with projected image features.
- `view`/`reshape`, `transpose`, `contiguous`, `flatten`, `cat`, `expand`,
  slicing, `argmax`, `clamp`, `where`/`masked_fill` for helper paths.
- Last-token-only logits slicing through `logits_to_keep`.

Neural network primitives:

- Bias-free Linear for text Q/K/V/O and MLP projections:
  - Q: `hidden_size -> num_attention_heads * head_dim`
  - K/V: `hidden_size -> num_key_value_heads * head_dim`
  - O: `num_attention_heads * head_dim -> hidden_size`
  - MLP gate/up/down: `hidden_size -> intermediate_size`, `hidden_size ->
    intermediate_size`, `intermediate_size -> hidden_size`
- RMSNorm with `(1 + weight)` scale, fp32 variance, output cast back to input
  dtype.
- Gated MLP: `gelu_pytorch_tanh(gate_proj(x)) * up_proj(x) -> down_proj`.
- Residual adds around post-attention and post-MLP RMSNorm.
- Optional dropout is training-only; treat as identity for inference.
- LM head: Linear `decoder.hidden_size -> vocab_size`, bias false.
- Optional tanh softcap after attention scores and after logits.

Vision/preprocessing-coupled ops:

- Gemma3 image processor: RGB conversion, resize, rescale, normalize, optional
  pan-and-scan crop generation; CPU/data pipeline first.
- SigLIP patch Conv2d: `Conv2d(3 -> vision_hidden, kernel=patch, stride=patch,
  padding=valid)` over NCHW.
- SigLIP learned 2D position embedding add; optional bicubic interpolation.
- SigLIP encoder: LayerNorm, dense noncausal MHA, GELU MLP, residuals.
- T5Gemma2 projector:
  - sequence `[B, patches^2, V]`
  - transpose to `[B, V, patches^2]`
  - reshape to NCHW-like `[B, V, patches_per_side, patches_per_side]`
  - AvgPool2d with `kernel=stride=patches_per_side / sqrt(mm_tokens_per_image)`
  - flatten/transpose to `[B, mm_tokens_per_image, V]`
  - RMSNorm(V)
  - MatMul with `mm_input_projection_weight` stored `[V, hidden_size]`.

Attention primitives:

- Dense noncausal self-attention in encoder text and SigLIP.
- Causal self-attention in decoder with mandatory explicit mask.
- Sliding-window causal and bidirectional masks for selected layers.
- Merged decoder self+cross attention with K/V concat along sequence.
- GQA repeat of K/V heads for eager attention or equivalent grouped attention.
- SDPA-compatible backend is advertised; FlashAttention is disabled for
  T5Gemma2 because masks and sliding-window creation are nonstandard.

Position/custom math:

- Dual RoPE tables keyed by layer type (`sliding_attention`, `full_attention`).
- `rotate_half`, cos/sin broadcast to `[B, heads, T, head_dim]`.
- Dynamic RoPE update path exists for non-default RoPE types; source defaults are
  `default`.

Generation/cache ops:

- `EncoderDecoderCache` with self-attention cache and cross-attention cache.
- Self cache shapes per layer: `[B, num_key_value_heads, decoded_T, head_dim]`
  for key and value, before GQA expansion.
- Cross cache shapes per layer: `[B, num_key_value_heads, encoder_T, head_dim]`
  for key and value, full attention only.
- `is_updated[layer_idx]` controls one-time cross K/V projection/cache fill.
- StaticCache handling for cross cache max length equals encoder sequence
  length.

Optional classification heads:

- Sequence classification uses decoder hidden states, Linear to labels, then
  selects rightmost non-pad decoder token via `argmax` over non-pad indices.
- Token classification applies Linear per decoder token.

## 5. Layer/block breakdown

Encoder text block, repeated `encoder.text_config.num_hidden_layers`:

```text
residual = x
x = RMSNorm(hidden_size)(x)
q = Linear(hidden_size -> n_heads * head_dim, bias=attention_bias)(x)
k = Linear(hidden_size -> n_kv_heads * head_dim, bias=attention_bias)(x)
v = Linear(hidden_size -> n_kv_heads * head_dim, bias=attention_bias)(x)
q = view(B,T,n_heads,head_dim).transpose(1,2)
k/v = view(B,T,n_kv_heads,head_dim).transpose(1,2)
q = RMSNorm(head_dim)(q)
k = RMSNorm(head_dim)(k)
q,k = RoPE_layer_type(q,k, position_ids)
attn = Attention(q,k,v, bidirectional full or sliding mask, GQA repeat)
x = Linear(n_heads * head_dim -> hidden_size, bias=attention_bias)(attn)
x = RMSNorm(hidden_size)(x)
x = residual + x

residual = x
x = RMSNorm(hidden_size)(x)
x = gelu_tanh(gate_proj(x)) * up_proj(x)
x = Linear(intermediate_size -> hidden_size, bias=False)(x)
x = RMSNorm(hidden_size)(x)
x = residual + x
```

Decoder block, repeated `decoder.num_hidden_layers`:

```text
residual = x
x = RMSNorm(hidden_size)(x)
q,k_self,v_self = shared text projections from decoder x
q,k_self = per-head RMSNorm -> RoPE
self K/V update cache if present
k_cross,v_cross = same K/V projections from encoder_hidden_states
k_cross = per-head RMSNorm, no RoPE
cross K/V update or reuse cache
k = cat([k_self, k_cross], dim=sequence)
v = cat([v_self, v_cross], dim=sequence)
mask = cat([causal-or-sliding self mask, bidirectional encoder mask], dim=-1)
x = Attention(q,k,v, merged mask, GQA repeat)
x = O projection -> RMSNorm -> residual add
MLP path same as encoder block
```

Vision tower, owned by SigLIP audit but consumed here:

```text
pixel_values [B,3,H,W] -> Conv2d patch embedding -> [B,V,Hp,Wp]
flatten/transpose -> [B, Hp*Wp, V]
+ learned position embedding
N SigLIP encoder blocks: LayerNorm -> MHA -> residual -> LayerNorm -> MLP -> residual
post LayerNorm -> last_hidden_state
```

## 6. Attention requirements

Encoder text attention:

- Noncausal self-attention.
- GQA with `num_attention_heads / num_key_value_heads` repeat.
- Full or sliding window per `layer_types`.
- Sliding-window bidirectional mask uses left window `(sliding_window + 1)//2`
  and right window `sliding_window//2 + 1`.
- Q/K per-head RMSNorm before RoPE.
- RoPE applied to Q/K only.

Decoder merged attention:

- Causal self-attention plus encoder cross-attention in one attention call.
- Self part may be full or sliding-window causal by layer type.
- Cross part is full bidirectional over encoder sequence and concatenated after
  self keys/values.
- Cache stores self and cross K/V separately; concatenation is a runtime view or
  materialized concat before attention.
- Cross K/V are projected with the same `k_proj`/`v_proj` modules as self K/V.
  Cross K gets RMSNorm but no RoPE.
- Eager math order:

```text
scores = matmul(q, repeat_kv(k).transpose(-2,-1)) * query_pre_attn_scalar**-0.5
scores = tanh(scores / softcap) * softcap     # only if configured
scores = scores + additive_mask
probs = softmax(scores, dim=-1, dtype=float32).to(q.dtype)
out = matmul(probs, repeat_kv(v))
```

SigLIP vision attention:

- Dense noncausal MHA, equal Q/K/V widths, LayerNorm pre-attention, standard
  softmax attention. This should compose a separate SigLIP/vision audit if
  broader SigLIP reuse is planned.

Backend implications:

- SDPA can be used if it supports the exact additive masks and GQA/merged K/V
  shape. T5Gemma2 explicitly disables FlashAttention in Transformers because
  the merged mask/sliding-window construction is incompatible with the library
  path as wired there.
- First DinoML attention should be a faithful dense/sliding implementation. Fuse
  merged attention only after cache and mask parity are proven.

## 7. Position encoding and custom math

RoPE is per layer type, with default local/global theta values. Source-default
RoPE computes inverse frequency from `head_dim`, not residual hidden size.

```python
def t5gemma2_default_inv_freq(head_dim, theta):
    i = torch.arange(0, head_dim, 2, dtype=torch.float32)
    return 1.0 / (theta ** (i / head_dim))

def t5gemma2_rope(position_ids, inv_freq):
    # position_ids: [B, T], inv_freq: [head_dim // 2]
    freqs = (inv_freq[None, :, None] @ position_ids[:, None, :].float()).transpose(1, 2)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()

def apply_t5gemma2_rope(q, k, cos, sin):
    # q: [B, Hq, T, D], k: [B, Hkv, T, D], cos/sin: [B, T, D]
    cos = cos[:, None, :, :]
    sin = sin[:, None, :, :]
    def rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)
```

Precompute opportunities:

- Static `inv_freq` per layer type and dtype/device.
- Cos/sin for fixed maximum positions, if memory is acceptable.
- Encoder position embeddings for fixed text length and layer type.
- Decoder decode-step cos/sin from scalar positions.

Dynamic inputs:

- `position_ids` may be supplied; otherwise source uses arange from sequence
  length or cache length. Do not assume contiguous positions unless guarded.
- Advanced `rope_type != default` is routed through `ROPE_INIT_FUNCTIONS` and
  `dynamic_rope_update`; reject or route to a separate audit initially.

## 8. Preprocessing and input packing

Processor-owned CPU/data-pipeline work:

- `Gemma3Processor` wraps a Gemma3 image processor and tokenizer.
- It expands each `<start_of_image>`/BOI occurrence into:
  `\n\n + boi + image_token * image_seq_length + eoi + \n\n`.
- Default `image_seq_length` is 256.
- Optional pan-and-scan rewrites prompts by inserting additional BOI tokens for
  crops; default processor kwargs disable pan-and-scan.
- Image processor emits `pixel_values` and `num_crops`; processor removes
  `num_crops` from model inputs.
- Image processor default source values are 224 square resize, ImageNet mean/std,
  RGB conversion, rescale, normalize. The T5Gemma2 model card says official
  images are normalized to 896x896 and encoded to 256 tokens each, so official
  preprocessor config must be loaded after gated access.

Runtime graph inputs:

- `input_ids`: encoder token ids with image placeholders already expanded.
- `attention_mask`: encoder text/image token mask.
- `pixel_values`: NCHW images, rank 4. With crops, batch dimension is flattened
  over original images plus crop images.
- Optional `decoder_input_ids`, `decoder_attention_mask`, position ids, and
  `past_key_values`.

Embedding stitch contract:

- Source computes normal input embeddings first.
- If `pixel_values` is present:
  - `vision_tower(pixel_values).last_hidden_state`
  - projector returns `[num_images_or_crops, mm_tokens_per_image, hidden_size]`
  - count check compares flattened image-token slots with image feature elements
  - `inputs_embeds.masked_scatter(image_mask, image_features)`
- DinoML can replace general `masked_scatter` with a guarded row-copy:
  source order is row-major flatten order over the boolean mask. Processor
  should guarantee exactly 256 contiguous image placeholder positions per image
  between BOI and EOI. Reject if image token count does not equal
  `num_images * mm_tokens_per_image` or if placeholders are not in the expected
  contiguous regions for the chosen fast path.

## 9. Graph rewrite / lowering opportunities

### Rewrite: image placeholder masked_scatter -> guarded row copy

Source pattern:

```text
image_mask = (input_ids == image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_features)
```

Replacement:

```text
Validate placeholder runs -> copy image_features.reshape(total_image_tokens, H)
into matching rows of inputs_embeds
```

Preconditions:

- `input_ids` is available.
- Placeholder count equals `num_images * mm_tokens_per_image`.
- Each image placeholder region is contiguous and in processor order.
- `image_features` is row-major `[image, token, hidden]`.

Failure cases:

- Caller supplies arbitrary `inputs_embeds` without `input_ids`.
- Pan-and-scan changes image count/order and the runtime does not receive crop
  metadata.
- Placeholder positions are arbitrary; fall back or reject instead of admitting
  general boolean scatter.

Parity test sketch:

- One and two image prompts, batched prompts with padding, count mismatch error,
  and synthetic non-contiguous placeholder rejection.

### Rewrite: SigLIP patch Conv2d -> Linear/GEMM for fixed non-overlap patches

Preconditions:

- `kernel_size == stride == patch_size`.
- `padding == valid`, `dilation == 1`, `groups == 1`.
- NCHW input height/width divisible by patch size.
- Weight flatten order preserves PyTorch Conv2d NCHW layout:
  `w_flat = conv.weight.reshape(out_channels, in_channels * ph * pw)`.

Replacement:

```text
NCHW PatchExtract row-major over output h,w
-> GEMM([B*Hp*Wp, C*ph*pw], w_flat.T)
-> BiasAdd
-> [B, Hp*Wp, V]
```

Layout notes:

- Source is NCHW. An NHWC fast path is safe only if patch extraction, weight
  transform, and downstream token flatten order are rewritten together.
- Optional position interpolation path has NCHW/NHWC permutes around bicubic
  interpolation; keep a no-layout-translation guard there initially.

### Rewrite: multimodal projector AvgPool2d -> token-grid pooling

Source pattern:

```text
[B, P^2, V] -> transpose/reshape [B,V,P,P]
AvgPool2d(kernel=s, stride=s)
-> flatten/transpose [B, (P/s)^2, V]
```

Replacement:

```text
Reshape [B,P,P,V] -> block mean over s x s -> [B,T,T,V]
```

Preconditions:

- `P = image_size // patch_size`.
- `T = sqrt(mm_tokens_per_image)` is integer.
- `P % T == 0`, `kernel_size = stride = P // T`.
- No padding, no ceil mode, divisor is kernel area.

Layout notes:

- This is a strong NHWC candidate after SigLIP token output, because the source
  temporarily uses NCHW only for AvgPool2d. Preserve token order when rewriting.

### Rewrite: merged decoder attention -> two attention calls plus combine

This is a staging rewrite, not the final optimized path.

Replacement:

```text
self_out = Attention(q, self_k, self_v, self_mask)
cross_out = Attention(q, cross_k, cross_v, cross_mask)
merged_out = attention-over-concatenated-KV reference, not equal to self_out + cross_out
```

Because softmax normalizes over the concatenated self+cross sequence, separate
self and cross attention calls are not algebraically equivalent without custom
log-sum-exp recombination. Use this only as a diagnostic with exact
concatenated-score reconstruction, or implement merged K/V attention directly.

## 10. Kernel fusion candidates

Highest priority:

- RMSNorm with `(1 + weight)` scale for hidden and per-head Q/K norms. This is
  everywhere and parity-sensitive because multiplication order differs from
  Llama-style RMSNorm.
- Bias-free GEMM families for Q/K/V/O, MLP, projector, and LM head, including
  hidden size != Q width.
- Gated GELU MLP fusion: `gelu_tanh(gate) * up -> down`.
- RoPE + Q/K per-head RMSNorm + attention prefill/decode preparation.
- Merged GQA attention with concatenated self+cross K/V and additive masks.
- KV cache layout/update for `EncoderDecoderCache`, including full cross cache.

Medium priority:

- Sliding-window mask/attention kernels for local encoder and decoder layers.
- Last-token-only logits via `logits_to_keep` to avoid full vocab GEMM during
  decode.
- Image placeholder row-copy stitch with processor guards.
- SigLIP patch Conv2d-to-GEMM and projector pooling rewrite.
- Optional tanh softcap for attention scores/logits.

Lower priority:

- SigLIP multihead pooling head, because T5Gemma2 consumes `last_hidden_state`
  for the projector and does not use `pooler_output` from the SigLIP head.
- Sequence/token classification heads.
- Pan-and-scan crop prompt rewriting in runtime; keep in CPU processor first.
- Tensor parallel plan metadata.

## 11. Runtime staging plan

1. Parse gated or source-default config and load weights, preserving tied
   encoder/decoder embeddings, EOI embedding, and LM head aliases.
2. Text-only encoder block parity with static small shapes from the local test
   config: RMSNorm, Q/K norm, RoPE, bidirectional full/sliding masks, gated MLP.
3. Text-only decoder block parity without cache, using merged self+cross K/V on
   synthetic encoder hidden states.
4. Full text seq2seq prefill parity: encoder -> decoder -> LM head, no images.
5. Decode parity with `EncoderDecoderCache`: self KV append, cross KV one-time
   projection/cache fill, full cross cache for sliding layers.
6. Add image path: SigLIP tower composed from separate audit, projector, guarded
   image embedding stitch.
7. Optimize: fused RMSNorm/MLP, merged attention kernels, sliding-window
   kernels, last-token logits, Conv2d-to-GEMM and projector pooling rewrites.
8. Optional heads and processor variants: classification heads, pan-and-scan,
   position interpolation, non-default RoPE.

Initial stubs allowed:

- CPU/data pipeline can call Transformers processor for parity harnesses.
- SigLIP vision tower can be delegated or separately audited while text-only
  seq2seq comes up.
- Use dense additive masks before specialized sliding-window attention.

## 12. Parity and validation plan

Concrete tests:

- RMSNorm numeric parity for fp32/fp16/bf16, including zero-initialized weight
  and nonzero learned weight.
- RoPE parity for local theta 10000 and global theta 1000000, custom
  `position_ids`, and decode offsets.
- One encoder block parity for both `full_attention` and `sliding_attention`.
- One decoder block parity with synthetic encoder states and both layer types;
  verify concatenated mask and attention output match source.
- Cross-cache parity: first decode step populates cross K/V, next step reuses it
  without recomputing.
- Text-only prefill logits parity for small test config.
- Decode token parity for greedy generation on a gated 270M checkpoint after
  access is available.
- Image stitch parity with one image, two images, batched padding, and count
  mismatch error.
- Projector parity on random SigLIP hidden states, including pool size derived
  from official config.
- End-to-end image-text parity against `google/t5gemma-2-270m-270m` once gated
  access is available; Transformers integration expected output for the bee
  prompt is ` a bumble bee in a flower bed.` on CUDA/bf16.

Tolerance guidance:

- fp32 custom ops: `atol=1e-5`, `rtol=1e-5`.
- fp16/bf16 block parity: start with `atol=3e-2`, `rtol=3e-2`, then tighten by
  op after fused kernels stabilize.
- Attention parity should compare logits/scores before and after softmax when
  debugging mask or softcap differences.

## 13. Performance probes

- Processor throughput: image resize/normalize/token expansion with and without
  pan-and-scan.
- SigLIP encoder throughput vs text encoder throughput at official image size.
- Projector-only latency and memory traffic for `[B, patches^2, V] -> [B,256,H]`.
- Text-only encoder prefill throughput over 1K, 4K, 32K, 128K tokens.
- Decoder prefill throughput with varying encoder length and decoder prompt
  length.
- Decode tokens/sec with cross cache enabled; sweep encoder length because
  merged attention attends over `self_T + encoder_T`.
- KV/cache memory: self cache grows with decoded length; cross cache stores
  full encoder length per layer.
- Sliding-window vs full layer timing; every 6th full layer is a distinct
  bottleneck profile.
- LM head last-token-only vs full-sequence logits.
- Projector/vision NHWC guarded rewrite speedup versus faithful NCHW path.
- Dense weights vs GGUF/load-time dequant experiments only after text path is
  correct; no source-coupled quantized format is implemented in Transformers
  T5Gemma2 itself.

## 14. Skip/defer list

- Training, loss functions, dropout behavior, and gradient checkpointing.
- Tensor parallel and pipeline parallel plan execution.
- Sequence classification and token classification heads for first generation
  target.
- Pan-and-scan runtime support beyond CPU processor parity.
- Non-default/dynamic RoPE variants until a checkpoint requiring them is found.
- FlashAttention-specific lowering; source disables it for this family.
- SigLIP pooling head unless another target consumes it.
- Arbitrary `inputs_embeds` image placeholder matching fast path; support
  `input_ids`-guarded row copy first.
- Beam search and advanced generation controllers beyond greedy parity.
- Multi-GPU/offloaded cache behavior.

## 15. Final implementation checklist

- [ ] Obtain gated official `config.json`, processor config, tokenizer config,
      and safetensors index for all three official checkpoints.
- [ ] Parse nested `T5Gemma2Config`, including encoder text, vision, decoder,
      layer types, RoPE parameters, special token ids, and tied weights.
- [ ] Preserve tied encoder/decoder token embeddings, EOI embedding alias, and
      LM head weight alias.
- [ ] Implement T5Gemma2 RMSNorm with `(1 + weight)` scale.
- [ ] Implement text Q/K/V/O projections where Q width may differ from hidden
      size.
- [ ] Implement per-head Q/K RMSNorm and dual local/global RoPE.
- [ ] Implement full and sliding bidirectional encoder masks.
- [ ] Implement full and sliding causal decoder masks.
- [ ] Implement merged decoder self+cross GQA attention over concatenated K/V.
- [ ] Implement `EncoderDecoderCache` self/cross cache ABI and cross-cache
      one-time update semantics.
- [ ] Implement gated GELU MLP and residual ordering.
- [ ] Implement LM head with `logits_to_keep` and optional final logit softcap.
- [ ] Add text-only one-block, prefill, and decode parity tests.
- [ ] Compose or separately audit SigLIP vision tower for T5Gemma2 image path.
- [ ] Implement multimodal projector pooling/RMSNorm/matmul.
- [ ] Lower image embedding `masked_scatter` to guarded row copy.
- [ ] Add image projector/stitch and end-to-end image-text parity tests.
- [ ] Benchmark encoder, decoder prefill, decode, vision, projector, and LM head
      as separate probes.
