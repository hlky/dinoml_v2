# Idefics Transformers Audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4, local checkout X:/H/transformers
Model id: HuggingFaceM4/idefics-9b, HuggingFaceM4/idefics-9b-instruct, HuggingFaceM4/idefics-80b, HuggingFaceM4/idefics-80b-instruct, HuggingFaceM4/tiny-random-idefics
Config source: official Hugging Face config/preprocessor/tokenizer JSON files from main branches, fetched 2026-05-13
Source files inspected:
- X:/H/transformers/src/transformers/models/idefics/configuration_idefics.py
- X:/H/transformers/src/transformers/models/idefics/modeling_idefics.py
- X:/H/transformers/src/transformers/models/idefics/vision.py
- X:/H/transformers/src/transformers/models/idefics/perceiver.py
- X:/H/transformers/src/transformers/models/idefics/processing_idefics.py
- X:/H/transformers/src/transformers/models/idefics/image_processing_idefics.py
- X:/H/transformers/src/transformers/models/idefics/image_processing_pil_idefics.py
- X:/H/transformers/tests/models/idefics/test_modeling_idefics.py
- X:/H/transformers/tests/models/idefics/test_processing_idefics.py
Any missing files or assumptions: tokenizer implementation is standard LlamaTokenizer; tokenizer coupling was inspected through tokenizer_config.json and special_tokens_map.json, not a family-local tokenizer file. No remote-code files are required for the in-library idefics family.
```

Primary runtime target: multimodal autoregressive image-text-to-text generation through `IdeficsForVisionText2Text`. `IdeficsModel` is useful for feature extraction and one-block parity but does not own the final LM logits head. Training loss can be deferred.

## 2. High-level architecture

Idefics is a Flamingo-style multimodal causal decoder: CLIP-like vision transformer -> optional Perceiver Resampler -> LLaMA-like text decoder with gated cross-attention inserted every `cross_layer_interval` decoder layers -> decoupled LM head.

```text
processor text/images
  -> input_ids, attention_mask, pixel_values[B, I, C, H, W], image_attention_mask[B, T, I]
  -> vision encoder over flattened images[B*I, C, H, W]
  -> optional perceiver resampler per image
  -> image sequence flatten[B, I*image_seq, vision_dim]
  -> gated image cross-attention before selected decoder blocks
  -> causal self-attention decode with text KV cache
  -> final RMSNorm -> decoupled LM head -> logits/sampling
```

Stage decomposition:

- CPU/data pipeline: URL/image fetch, RGB conversion, resize/rescale/normalize, prompt construction with image special tokens, image attention mask construction.
- Cacheable vision stage: `pixel_values` can be replaced by `image_encoder_embeddings[B, I, image_seq, vision_dim]`.
- Cacheable perceiver stage: when `use_resampler=True`, `perceiver_embeddings[B, I, n_latents, vision_dim]` can be supplied instead of recomputing image processing.
- Prefill: full text sequence plus image cross-attention and text self-attention cache creation.
- Decode: one or few text tokens, cached image hidden states carried in generation kwargs, text self-attention KV cache updated. Cross-attention KV cache is explicitly not implemented.

## 3. Important config dimensions

Source defaults from `IdeficsConfig` unless overridden by checkpoint config:

| Field | Source default | Production configs inspected |
|---|---:|---:|
| `vocab_size` | 32000 | 32000 |
| `additional_vocab_size` | 0 | 2 base, 3 instruct |
| `hidden_size` | 4096 | 4096 or 8192 |
| `num_hidden_layers` | 32 | 32 or 80 |
| `num_attention_heads` | 32 | 32 or 64 |
| `head_dim` | `hidden_size / heads` | 128 |
| `intermediate_size` | 11008 | 11008 or 22016 |
| `hidden_act` | `silu` | `silu` |
| `rms_norm_eps` | 1e-6 | 1e-6 for 9B, 1e-5 for 80B |
| `cross_layer_interval` | 1 | 4 |
| cross-attn layers | `layers // interval` | 8 for 9B, 20 for 80B |
| `qk_layer_norms` | false | true |
| `use_resampler` | false | true |
| vision `embed_dim` | 768 | 1280 |
| vision layers/heads | 32 / 16 | 32 / 16 |
| vision patch/image | 14 / 224 | 14 / 224 |
| vision tokens/image | 257 | 257 before perceiver |
| perceiver latents | 64 | 64 |
| perceiver depth/heads/head_dim | 6 / 16 / 96 | 6 / 16 / 96 |
| dtype | not fixed by source | bf16 production, fp16 tiny |
| cache support | true | text self-attention only |

Representative checkpoint sweep:

| Checkpoint | Text H/L/A | MLP | Vision | Perceiver | Cross interval | Added tokens | dtype |
|---|---:|---:|---:|---:|---:|---:|---|
| `HuggingFaceM4/tiny-random-idefics` | 16 / 2 / 4 | 11008 | 32d, 5L, patch 2, image 30 | 16 latents, 2L, 2H x 8 | 1 | 2 | float16 |
| `HuggingFaceM4/idefics-9b` | 4096 / 32 / 32 | 11008 | 1280d, 32L, 16H, patch 14, image 224 | 64 latents, 6L, 16H x 96 | 4 | 2 | bfloat16 |
| `HuggingFaceM4/idefics-9b-instruct` | 4096 / 32 / 32 | 11008 | same as 9B | same | 4 | 3 | bfloat16 |
| `HuggingFaceM4/idefics-80b` | 8192 / 80 / 64 | 22016 | same as 9B | same | 4 | 2 | bfloat16 |
| `HuggingFaceM4/idefics-80b-instruct` | 8192 / 80 / 64 | 22016 | same as 9B | same | 4 | 3 | bfloat16 |

## 3a. Family variation traps

- Production configs set `cross_layer_interval=4`, but source default is 1. Do not assume every decoder layer has cross-attention unless reading config.
- Production configs use `use_resampler=True`; source supports no-resampler mode, where cross-attention attends all 257 vision tokens per image.
- `additional_vocab_size` changes with tokenizer: base checkpoints add `<fake_token_around_image>` and `<image>`; instruct adds `<end_of_utterance>` too.
- `qk_layer_norms=True` in production cross-attention, adding per-head RMSNorm after RoPE/cache update for self-attn and after projection for cross-attn. Tiny random disables it.
- There is no GQA/MQA in this implementation. `num_key_value_heads` is absent; K/V heads equal query heads.
- Text projections are bias-free; vision projections and vision MLP linears have bias.
- LM head and token embeddings are decoupled for added vocabulary and must preserve weight aliasing when tied.
- Cross-attention cache is not implemented; passing `past_key_values` to `IdeficsGatedCrossAttentionLayer` raises.
- `max_sequence_length` appears in production configs, but the inspected source does not use it for RoPE construction; `IdeficsEmbedding` defaults to 2048 and extends dynamically.
- Pixel tensors are semantically NCHW. Layout translation must be guarded around Conv2d patch embedding, position interpolation, flatten/transposes, and processor output.
- `IdeficsPreTrainedModel` advertises SDPA support but `_supports_flash_attn=False`; source tests also mark a hard SDPA requirement for generation-oriented common tests.

## 4. Operator coverage checklist

Tensor/layout ops:

- `view`, `reshape`, `flatten`, `transpose`, `permute`, `contiguous`, `cat`, `repeat`, `expand`, `unsqueeze`, `squeeze`, `index_select`, slicing, `where`, `masked_fill`.
- Boolean/integer mask construction for image gates and position ids: `cumsum`, comparisons, any-reduction, one-hot in processor/data path.
- Decoupled embedding indexed overwrite: select ids `>= vocab_size`, subtract base vocab, embed added table, scatter/advanced-index write into base embedding result.
- Decoupled LM head concat: base linear logits plus optional added-token linear logits concatenated on vocab axis.

Neural network primitives:

- Text RMSNorm over last dim with fp32 variance and dtype restoration.
- Vision LayerNorm.
- Linear/GEMM: text q/k/v/o, MLP gate/up/down, cross-attn q plus vision-dim k/v, perceiver projections, vision q/k/v/out and MLP.
- Bias-free text/perceiver linears; biased vision linears.
- Text MLP: SwiGLU, `down_proj(silu(gate_proj(x)) * up_proj(x))`.
- Perceiver MLP: LayerNorm -> bias-free Linear -> ReLU -> bias-free Linear.
- Vision MLP: Linear -> GELU -> Linear.
- Conv2d patch embedding: NCHW, `kernel_size=stride=patch_size`, no bias.
- Bicubic positional interpolation for non-default image sizes when `interpolate_pos_encoding=True`.

Attention primitives:

- Causal text MHA with RoPE, SDPA/eager backend, cache update.
- Gated image cross-attention MHA, non-cached, text queries to image K/V, optional q/k RMSNorm, dense additive image mask.
- Perceiver cross/self attention: latent queries attend `cat(context, latents)`, stable eager softmax, no dropout.
- Vision noncausal MHA over `[CLS]+patches`, SDPA/eager backend.

Position/cache/preprocessing-coupled ops:

- LLaMA-style RoPE with duplicated frequency layout.
- Causal mask from Transformers `create_causal_mask`.
- Image attention mask expansion from `[B, T, I]` to `[B, 1, T, I*image_seq]`, invert to additive mask, gate derived from zero entries.
- Generation kwargs update: slice image mask to current decode length and carry cached `image_hidden_states`.

## 5. Layer/block breakdown

Vision encoder:

```text
pixel_values[B*I, 3, H, W]
  -> Conv2d(3 -> 1280, kernel=stride=14, bias=False)
  -> flatten patches to [B*I, P, 1280], prepend class embedding
  -> add learned position embedding or bicubic-interpolated patch positions
  -> LayerNorm
  -> 32 x:
       residual + MHA(LayerNorm(x))        # 16 heads, head_dim 80, biased projections
       residual + MLP(LayerNorm(x))        # 1280 -> 5120 -> 1280, GELU, biased
  -> last_hidden_state[B*I, 257, 1280]
```

Perceiver resampler, when enabled:

```text
context[B*I, 257, 1280], learned latents[64, 1280]
  -> repeat latents per batch
  -> 6 x:
       latents += Attention(LN(context), LN(latents))
         q: latents -> 16*96
         k/v: cat(context, latents) -> 16*96
         softmax((q * scale) @ k.T - rowmax)
         output -> 1280
       latents += LN -> Linear(1280 -> 5120) -> ReLU -> Linear(5120 -> 1280)
  -> LayerNorm -> [B*I, 64, 1280]
```

Text decoder layer, repeated `num_hidden_layers` times:

```text
if layer_idx % cross_layer_interval == 0:
  y = RMSNorm(x)
  y = image_cross_attention(y, image_hidden_states, image_attention_mask)
  y = masked_fill(y, tokens_with_no_image, 0)
  x = x + tanh(alpha_cross_attn) * y
  y = RMSNorm(x)
  y = MLP_swiglu(y)
  x = x + tanh(alpha_dense) * y

y = RMSNorm(x)
y = causal_self_attention_with_rope(y, causal_mask, text_kv_cache)
x = x + dropout(y)
y = RMSNorm(x)
y = down(silu(gate(x)) * up(x))
x = x + dropout(y)
```

Final:

```text
x = RMSNorm(x)
logits = DecoupledLinear(hidden -> vocab_size [+ additional_vocab_size])
```

## 6. Attention requirements

Text self-attention:

- Causal MHA, not GQA/MQA.
- 9B: 32 heads x 128; 80B: 64 heads x 128.
- Q/K/V/O are bias-free.
- RoPE is applied to Q/K before cache update. Cached K is stored after RoPE.
- `past_key_values.update(key_states, value_states, layer_idx)` stores per-layer self-attention KV tensors shaped `[B, heads, cached_text_len, head_dim]`.
- SDPA/eager attention receives additive causal mask and scale `head_dim**-0.5`. Eager path softmaxes in fp32 then casts back.

Gated image cross-attention:

- Text queries attend image sequence K/V.
- K/V input dimension is `vision_config.embed_dim` when present, not text hidden size. Production uses `1280 -> heads*128`.
- Attention mask is additive, expanded to `[B, 1, T, I*image_seq]`.
- Tokens whose image mask attends no image are zeroed with `masked_fill` before residual gating.
- No cross-attention KV cache: decode recomputes K/V from cached image hidden states each step.

Vision attention:

- Noncausal MHA over patch sequence, biased linears, 16 heads x 80 in production.
- Uses `ALL_ATTENTION_FUNCTIONS` with config `_attn_implementation`; eager fallback is matmul/softmax/dropout/matmul.

Perceiver attention:

- Latent queries attend concatenated context and latents.
- Explicit eager implementation with row-max stabilization and no backend dispatch.
- Optional LayerNorm on q/k head vectors from `qk_layer_norms_perceiver`.

## 7. Position encoding and custom math

Text RoPE:

```python
def idefics_rope(q, k, position_ids, inv_freq):
    t = arange(seq_len)
    freqs = outer(t, inv_freq)
    emb = cat([freqs, freqs], dim=-1)
    cos = cos(emb)[position_ids].unsqueeze(1)
    sin = sin(emb)[position_ids].unsqueeze(1)
    def rotate_half(x):
        x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return cat([-x2, x1], dim=-1)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin
```

Vision positions are learned absolute embeddings for `[CLS]+patches`. If `interpolate_pos_encoding=True`, the patch grid position embedding is reshaped to square grid, permuted to NCHW, bicubic-interpolated in fp32 for bf16, permuted back, and concatenated with the class position.

Gated cross-attention residual:

```python
cross = cross_attn(RMSNorm(x), image_hidden, image_mask)
cross = cross.masked_fill((cross_attention_gate == 0)[:, :, None], 0.0)
x = residual + tanh(alpha_cross_attn) * cross
x = x + tanh(alpha_dense) * mlp(RMSNorm(x))
```

RoPE cos/sin can be precomputed up to a max sequence length and extended on demand. Image hidden states and perceiver outputs can be precomputed per prompt/image batch.

## 8. Preprocessing and input packing

Image processor:

- Converts to RGB, resizes to square `image_size` with bicubic resampling, rescales, normalizes by mean `[0.48145466, 0.4578275, 0.40821073]` and std `[0.26862954, 0.26130258, 0.27577711]`.
- Returns NCHW tensors. Processor batches them as `pixel_values[B, max_num_images, 3, image_size, image_size]`.
- If no image exists, processor still emits one zero image slot and an all-zero image mask.

Prompt and special tokens:

- Processor prefixes tokenizer BOS.
- Each image is inserted as `<fake_token_around_image><image><fake_token_around_image>`, except consecutive images use `<image><fake_token_around_image>` for the later image.
- Base tokenizers: `<fake_token_around_image>` id 32000, `<image>` id 32001.
- Instruct tokenizers add `<end_of_utterance>` id 32002, and processor inserts it between consecutive text utterances when the tokenizer advertises the token.

Runtime tensors:

- `input_ids[B, T]`, `attention_mask[B, T]`.
- `pixel_values[B, I, C, H, W]`.
- `image_attention_mask[B, T, I]` bool one-hot-like mask indicating which image each token may attend.
- Model expands image mask to `[B, T, I*image_seq]`, inverts to additive attention mask, and derives `cross_attention_gate[B, T]`.

Generation behavior:

- First generation call can receive `pixel_values`; output includes `image_hidden_states`.
- Subsequent decode calls pass `image_hidden_states` back as `image_encoder_embeddings` or `perceiver_embeddings`, depending on `use_resampler`.
- With cache enabled, `image_attention_mask` is reduced to its last token row for the next decode step.
- Beam expansion index-selects `input_ids`, `attention_mask`, `image_attention_mask`, and exactly one image source tensor.

## 9. Graph rewrite / lowering opportunities

### Rewrite: non-overlap Conv2d patch embed -> GEMM

Source pattern: `Conv2d(C -> D, kernel_size=P, stride=P, padding=0, dilation=1, groups=1, bias=False)` followed by `flatten(2).transpose(1, 2)`.

Replacement:

```text
NCHW image -> non-overlapping patch flatten [B*I, Patches, C*P*P]
  -> MatMul(weight_flat.T) -> [B*I, Patches, D]
```

Preconditions:

- `H` and `W` divisible by `P`.
- NCHW source layout or an explicit NHWC patch extractor with matching flatten order.
- No padding, dilation, groups, or bias.
- Position embedding consumer expects row-major patch order matching PyTorch Conv2d flatten.

Failure cases: interpolated positional embeddings for dynamic/non-square sizes still need the exact patch grid; layout pass must not silently change patch order.

Parity test: compare Conv2d path and GEMM path for random fp32/fp16 inputs and real checkpoint patch weights.

### Rewrite: decoupled embedding/head specialization

Source pattern: split base vocab and added vocab lookup/projection.

Replacement:

```text
if additional_vocab_size == 0:
  normal embedding / linear
else:
  base embedding plus indexed added-token overwrite
  base logits plus added-token logits concat
```

Preconditions: added token ids are contiguous starting at `vocab_size`; LM head aliases embedding weights according to `_tied_weights_keys`.

Failure cases: tokenizer configs with non-contiguous added ids or untied custom weights should stay on generic indexed path.

### Rewrite: image mask expansion precompute

Source pattern: `image_attention_mask[B,T,I] -> repeat image_seq -> invert_attention_mask -> cross_attention_gate`.

Replacement: compute expanded additive mask and gate once per prefill, then slice `[B, 1, 1, I*image_seq]` during decode.

Preconditions: image sequence length and `num_images` fixed for the request; generation cache uses same prompt images.

Failure cases: batched requests with different `I`, no-image all-zero mode, or changing image masks across decode.

### Rewrite: Perceiver attention as cross-attention primitive

Source pattern: q from latents, k/v from `cat(context, latents)`, stable softmax.

Replacement: `CrossAttention(q=latents, kv=concat(context, latents))` plus residual and MLP.

Preconditions: no dropout in inference; q/k optional LayerNorm preserved; row-max stabilization matches fused attention numerics.

Failure cases: backend attention that cannot handle different Q and KV sequence lengths or explicit concat of context+latents.

### Layout candidate: local NHWC vision patch region

The processor and source model are NCHW. A guarded local layout rewrite may keep images NHWC through preprocessing and lower patch extraction to a channel-last patch GEMM, but must rewrite Conv2d axis assumptions, interpolation permutes, flatten order, and downstream vision token consumers. First translation should preserve NCHW semantics.

## 10. Kernel fusion candidates

Highest priority:

- Text RMSNorm and vision LayerNorm. They are on every block boundary and gate all larger kernels.
- Text QKV projection + RoPE + cache update for self-attention. This controls prefill/decode throughput.
- SDPA/Flash-style causal MHA for MHA with head_dim 128 and no GQA.
- Cross-attention projection + attention over image latents. Decode recomputes cross K/V unless DinoML preprojects image K/V.
- SwiGLU MLP fusion: two GEMMs plus SiLU/mul and down GEMM, dominant decoder FLOPs.
- Last-token-only logits through `logits_to_keep`, especially 80B vocab projection.

Medium priority:

- Conv patch embed -> GEMM and vision attention/MLP fusions for image-heavy batches.
- Perceiver attention and MLP kernels; useful because production always uses resampler.
- Image mask expansion/gate generation as a small compiled helper to avoid Python-side tensor churn.
- Decoupled added-token embedding/head specialization.

Lower priority:

- Bicubic position interpolation in runtime; normal production path uses fixed 224 and can defer interpolation.
- Training loss, dropout, gradient checkpointing.
- Full language-only no-image mode optimization; tests note text-only common inference is not the standard path.

## 11. Runtime staging plan

1. Parse config/tokenizer/preprocessor and reject unsupported remote-code or non-contiguous added-token layouts.
2. Load weights with decoupled embedding/LM-head aliasing preserved.
3. Build vision encoder parity for fixed 224 NCHW images.
4. Build Perceiver parity with cached `perceiver_embeddings`.
5. Build one gated cross-attention block and one decoder self-attention block parity.
6. Build full prefill parity using supplied `image_encoder_embeddings` or `perceiver_embeddings` first, then add pixel path.
7. Add text KV cache decode; keep cross-attention uncached initially to match source behavior.
8. Add generation-controller plumbing: beam expansion, image mask slicing, `logits_to_keep`, and cached image hidden states.
9. Optimize attention/GEMM/layout rewrites behind exact shape and layout guards.

Initial stubs: CPU URL fetch and image preprocessing can remain outside DinoML runtime; training loss and position interpolation can be deferred for fixed-size inference.

## 12. Parity and validation plan

- Unit parity for `IdeficsRMSNorm`, RoPE, decoupled embedding, decoupled LM head, image mask expansion/gate.
- Vision patch embedding parity for random NCHW images and real weights.
- Vision encoder after 1 layer, then full 32 layers, fp32 first then bf16/fp16 tolerance.
- Perceiver one block and full resampler parity with `[B*I, 257, 1280]`.
- Gated cross-attention parity with masks containing all-zero and active image tokens.
- Text decoder single-layer parity with and without KV cache.
- Prefill logits parity for tiny-random and 9B shape-compatible slices.
- Decode parity: first token prefill, one-step decode using `past_key_values`, verify image hidden state reuse and last-row image mask.
- End-to-end processor + generate smoke for tiny-random and one production checkpoint.

Suggested tolerances: fp32 `rtol=1e-4, atol=1e-5`; bf16/fp16 block-level `rtol=5e-2, atol=5e-2` initially, tightened per fused kernel after accumulation policy is fixed.

## 13. Performance probes

- Processor throughput: images/sec for fetch/resize/normalize and prompt packing.
- Vision encoder throughput by batch images `B*I` and resolution.
- Perceiver throughput for context length 257 and latents 64.
- Prefill tokens/sec with image count sweep `I=1,2,4,8`.
- Decode tokens/sec with text KV cache and uncached cross-attention K/V recompute.
- Decode memory: text KV cache bytes per layer plus image hidden/perceiver cache bytes.
- Cross-attention benchmark comparing raw vision tokens 257/image versus perceiver 64/image.
- Last-token logits projection time for 32k+added vocab.
- Attention backend comparison: eager/SDPA/DinoML fused for text, cross, vision, and perceiver separately.
- Batch/sequence sweep: 9B and 80B dimensions, `T=1,128,512,2048`.

## 14. Skip/defer list

- Training from scratch, gradient checkpointing, dropout behavior, and loss.
- FlashAttention parity in HF source, because `_supports_flash_attn=False`; DinoML fused attention can be an internal optimization after SDPA parity.
- Cross-attention KV cache as a semantic feature; source explicitly does not implement it.
- Dynamic image-size interpolation and bicubic positional resize for first fixed-224 integration.
- Multi-GPU/tensor parallel and quantization.
- Beam search beyond correct `expand_inputs_for_generation` tensor replication.
- Text-only optimized path; source tests mark text-only inference as nonstandard for Idefics.

## 15. Final implementation checklist

- [ ] Parse `IdeficsConfig`, `IdeficsVisionConfig`, and `IdeficsPerceiverConfig`.
- [ ] Parse preprocessor/tokenizer special-token contract.
- [ ] Load decoupled text embeddings and preserve added-token ids.
- [ ] Load decoupled LM head and tied-weight aliases.
- [ ] Implement fixed-size NCHW vision patch embedding.
- [ ] Implement vision LayerNorm/MHA/MLP encoder parity.
- [ ] Implement optional Perceiver Resampler.
- [ ] Implement image attention mask expansion and cross-attention gate.
- [ ] Implement text RMSNorm, RoPE, MHA KV cache, and SwiGLU MLP.
- [ ] Implement gated cross-attention layers at `layer_idx % cross_layer_interval == 0`.
- [ ] Implement generation input preparation for cached image hidden states.
- [ ] Add prefill logits parity tests.
- [ ] Add one-step decode parity with text KV cache.
- [ ] Add guarded Conv2d patch -> GEMM rewrite.
- [ ] Add attention/GEMM fusion benchmarks.
