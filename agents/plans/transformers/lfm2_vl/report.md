# Transformers audit: `lfm2_vl`

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: LiquidAI/LFM2-VL-450M, LiquidAI/LFM2-VL-1.6B, LiquidAI/LFM2-VL-3B
Config source: official Hugging Face raw config/preprocessor JSON snapshots in _sources/
Source files inspected: lfm2_vl configuration/modeling/modular/processing/image_processing; lfm2 configuration/modeling/modular; siglip2 configuration/modeling/modular; cache_utils.py
Any missing files or assumptions: no model execution, imports, weight loading, or DinoML tests were run. Processor/tokenizer special token strings are source-derived; tokenizer vocab files were not audited beyond processor coupling.
```

`modeling_lfm2_vl.py`, `modeling_lfm2.py`, and `modeling_siglip2.py` are generated from `modular_lfm2_vl.py`, `modular_lfm2.py`, and `modular_siglip2.py`. Future upstream source edits should check the modular files; runtime parity should match the generated modeling files.

Config URLs snapshotted:

- [LiquidAI/LFM2-VL-450M config](https://huggingface.co/LiquidAI/LFM2-VL-450M/raw/main/config.json)
- [LiquidAI/LFM2-VL-1.6B config](https://huggingface.co/LiquidAI/LFM2-VL-1.6B/raw/main/config.json)
- [LiquidAI/LFM2-VL-3B config](https://huggingface.co/LiquidAI/LFM2-VL-3B/raw/main/config.json)
- Matching `preprocessor_config.json` files for all three checkpoints.

## 2. High-level architecture

LFM2-VL is a multimodal generation model:

```text
CPU/image pipeline -> patchified SigLIP2 inputs -> SigLIP2 vision encoder
  -> unpad/restore patch grid -> LFM2-VL projector -> image token embedding stitch
  -> LFM2 hybrid text decoder prefill -> decode with attention KV + conv states
  -> tied LM head logits
```

Stage decomposition:

- CPU/data pipeline: image fetch/resize/split/thumbnail, rescale/normalize, patchify to flattened patch vectors, pad to `max_num_patches`, emit `pixel_attention_mask`, `spatial_shapes`, rows/cols/sizes, and expanded text placeholders.
- Vision encoder/projector: independently cacheable for a fixed image batch. Inputs are patch sequences, not raw NCHW tensors.
- Prefix construction: token embeddings plus image features copied into every `image_token_id=396` placeholder position.
- Prefill: LFM2 text decoder over mixed image/text token embeddings.
- Decode: pixel inputs are forwarded only on the first generation iteration; subsequent tokens use cached text-decoder state.

## 3. Important config dimensions

| Field | 450M | 1.6B | 3B | Source |
| --- | ---: | ---: | ---: | --- |
| dtype | bf16 | bf16 | bf16 | `config.json` |
| image token id | 396 | 396 | 396 | `config.json` |
| text hidden | 1024 | 2048 | 2048 | `text_config` |
| text layers | 16 | 16 | 30 | `text_config` |
| full-attention / conv layers | 6 / 10 | 6 / 10 | 8 / 22 | `text_config.layer_types` |
| text heads / KV heads | 16 / 8 | 32 / 8 | 32 / 8 | `text_config` |
| inferred head dim | 64 | 64 | 64 | source formula |
| text intermediate | 6656 | 12288 | 10752 | `text_config` |
| conv cache length | 3 | 3 | 3 | `text_config` |
| max positions | 128000 | 128000 | 128000 | `text_config` |
| RoPE theta | 1000000 | 1000000 | 1000000 | config/source default |
| vision hidden | 768 | 1152 | 1152 | `vision_config` |
| vision layers / heads | 12 / 12 | 26 / 16 | 27 / 16 | `vision_config` |
| vision patch size | 16 | 16 | 16 | `vision_config` |
| vision num patches table | 256 | 256 | 256 | `vision_config` |
| projector hidden | 2560 | 2560 | 2560 | top config |
| downsample factor | 2 | 2 | 2 | top config |

Representative processor values: `Lfm2VlImageProcessorFast`, `do_image_splitting=true`, `min_tiles=2`, `max_tiles=10`, `tile_size=512`, `encoder_patch_size=16`, `min_image_tokens=64`, `max_image_tokens=256`, `do_pad=true`, rescale `1/255`, mean/std `[0.5,0.5,0.5]`.

## 3a. Family variation traps

- LFM2 is hybrid, not a normal all-attention decoder. `layer_types` is the cache manifest: full-attention layers own KV cache, conv layers own fixed-size `conv_states`.
- 450M config says `use_thumbnail=false`, but its preprocessor config says `use_thumbnail=true`. Treat processor config as the source of preprocessing truth unless caller overrides are explicitly admitted.
- Vision `pixel_values` are patchified rank-3 `[tiles, max_num_patches, 3 * patch_size^2]`, despite some generated docstrings saying image-shaped NCHW.
- SigLIP2 vision positional embeddings are resized per `spatial_shapes` with bilinear interpolation and then padded to the patch-sequence length.
- LFM2 `head_dim` may be omitted; source computes `hidden_size // num_attention_heads`. Do not infer projection widths from hidden size alone in future variants.
- Configs carry legacy or ignored fields (`block_dim`, `block_*`, `conv_dim`, `num_heads`, `theta`, `image_token_index`, `use_pos_enc`). Native source consumes the normalized config fields listed above.
- Initial graph translation should preserve source axes. NHWC is a local projector layout only after vision unpadding, not the whole image pipeline.

## 4. Operator coverage checklist

Tensor/layout ops:

- Embedding lookup for text tokens.
- Rank-3 patch input handling, padding masks, unpad by `sum(pixel_attention_mask, dim=1)`.
- Reshape/view/permute/flatten for patch grids and projector pixel-unshuffle.
- Boolean/equality mask from `input_ids == image_token_id`; bounded indexed copy preferred over general `masked_scatter`.
- `cat` over per-image feature lists; dynamic sequence length from image grid.

Vision preprocessing-coupled ops:

- Resize/split/thumbnail in CPU pipeline.
- Rescale/normalize and patchify from NCHW image tiles to flattened patch rows.
- Pad patch rows to `max_num_patches`; emit flattened `pixel_attention_mask` and `[H_patches,W_patches]`.

Vision encoder ops:

- Linear patch projection: `3 * 16 * 16 = 768 -> vision_hidden`.
- Bilinear positional table resize from 16x16 source table to each patch grid, `align_corners=false`, `antialias=true`.
- Noncausal MHA, LayerNorm, MLP GELU tanh, residual adds.

Projector ops:

- Pixel-unshuffle on `[1,H,W,C]` by factor 2 to `[1,H/2,W/2,4C]`.
- Optional LayerNorm over `4 * vision_hidden`.
- Linear `4C -> 2560`, GELU, Linear `2560 -> text_hidden`, with bias.

LFM2 text ops:

- RMSNorm with fp32 variance.
- Attention layers: biasless Q/K/V/O, q/k per-head RMSNorm, RoPE, GQA repeat or native grouped attention, causal mask, KV cache.
- Conv layers: biasless in-proj `hidden -> 3*hidden`, split B/C/x, multiply `B*x`, depthwise causal Conv1d kernel length 3, multiply by `C`, out-proj.
- SwiGLU MLP: `w2(silu(w1(x)) * w3(x))`, biasless.
- Final RMSNorm and tied LM head.

Generation/cache ops:

- `DynamicCache(config)` builds layer cache classes from `layer_types`.
- Attention cache stores K/V per full-attention layer.
- Conv cache stores `[batch, hidden, conv_L_cache]` per conv layer with static-address in-place updates.
- Beam reorder must index-select both KV and conv states.

## 5. Layer/block breakdown

SigLIP2 vision block, repeated by `vision_config.num_hidden_layers`:

```text
patches = Linear(768 -> vision_hidden)(patchified_pixels)
pos = bilinear_resize(position_table[16,16,C], spatial_shapes)
x = patches + pos
for layer:
  x = x + SelfAttention(LayerNorm(x), bidirectional_mask)
  x = x + MLP(LayerNorm(x))
x = LayerNorm(x)
```

LFM2-VL image projection:

```text
feature = last_hidden_state[i, :valid_patches]
feature = reshape(feature, [1, H_patches, W_patches, C])
feature = pixel_unshuffle_factor_2(feature)
feature = LayerNorm(feature) optional
feature = Linear(4C -> 2560) -> GELU -> Linear(2560 -> text_hidden)
feature = flatten([H/2, W/2], row-major)
```

LFM2 attention layer:

```text
y = RMSNorm(x)
q = RMSNorm_per_head(Linear(hidden -> heads * 64)(y))
k = RMSNorm_per_head(Linear(hidden -> kv_heads * 64)(y))
v = Linear(hidden -> kv_heads * 64)(y)
q,k = RoPE(q,k)
k,v = cache.update(k,v,layer_idx) if cache enabled
attn = causal GQA(q,k,v,mask)
x = x + Linear(heads * 64 -> hidden)(attn)
x = x + w2(silu(w1(RMSNorm(x))) * w3(RMSNorm(x)))
```

LFM2 conv layer:

```text
y = RMSNorm(x)
y = y * attention_mask[...,None] for prefill padding masks
B,C,z = chunk(transpose(Linear(hidden -> 3*hidden)(y)), 3, channel_axis)
u = B * z
state = cache.update_conv_state(pad_or_roll(u), layer_idx) if cache enabled
conv = depthwise causal_conv1d(u, kernel=3) or one-token update
x = x + Linear(hidden -> hidden)(transpose(C * conv))
x = x + w2(silu(w1(RMSNorm(x))) * w3(RMSNorm(x)))
```

## 6. Attention requirements

Vision attention:

- Noncausal self-attention over patch rows.
- MHA with `vision_heads`, `head_dim=vision_hidden/vision_heads`.
- Bidirectional mask from flattened patch mask.
- SDPA-compatible; source marks SigLIP2 as not FlashAttention.

Text attention:

- Causal self-attention only in `full_attention` layers.
- GQA: 450M has 16 Q heads / 8 KV heads; 1.6B and 3B have 32 Q heads / 8 KV heads. Head dim is 64.
- Q and K are RMS-normalized per head before RoPE.
- Cached K/V are stored after RoPE.
- Eager fallback repeats KV heads before matmul; optimized lowering should avoid material repeat.
- Masks: full-attention layers receive a causal mask from `create_causal_mask`; conv layers receive the original 2D attention mask in prefill and no mask for one-token decode.

Hybrid cache contract:

- Cache layer type must match `text_config.layer_types`.
- Attention layers grow with sequence length.
- Conv layers remain fixed-size `[B, hidden, 3]`.
- `get_seq_length()` on mixed cache uses the first attention layer, so position ids advance by attention-cache length.

## 7. Position encoding and custom math

Text RoPE:

```python
inv_freq = 1.0 / (rope_theta ** (arange(0, head_dim, 2) / head_dim))
freqs = inv_freq[None, :, None] @ position_ids[:, None, :]
emb = cat([freqs, freqs], dim=-1).transpose(1, 2)
q = q * cos(emb) + rotate_half(q) * sin(emb)
k = k * cos(emb) + rotate_half(k) * sin(emb)
```

Vision position math:

```text
position_embedding.weight -> [16,16,C]
permute to [1,C,16,16]
F.interpolate(size=(H_patches,W_patches), mode="bilinear",
              align_corners=False, antialias=True)
flatten to [H_patches * W_patches, C]
pad tail positions with first resized embedding
```

Text RoPE can be precomputed per position bucket and dtype. Vision resized position embeddings depend on per-image patch-grid shapes and are cacheable by `(H_patches,W_patches,dtype,device)`.

## 8. Preprocessing and input packing

Processor output tensors for the model:

- `input_ids`, `attention_mask` from tokenizer.
- `pixel_values`: `[total_tiles_and_thumbnails, max_num_patches, 768]` for patch size 16 and RGB.
- `pixel_attention_mask`: `[total_tiles_and_thumbnails, max_num_patches]`, int mask from patch padding.
- `spatial_shapes`: `[total_tiles_and_thumbnails, 2]`, patch-grid height and width.

Placeholder protocol:

- Base image token string defaults to `<image>`; image token id is `396`.
- Optional delimiters: `<|image_start|>` and `<|image_end|>`.
- Multi-tile prompts optionally include row/column tags `<|img_row_r_col_c|>` and a thumbnail tag `<|img_thumbnail|>`.
- Tile order is row-major: rows outer, columns inner, then optional thumbnail.
- Tokens per 512 tile with patch 16/downsample 2: `(512/16/2)^2 = 256`.
- Single image/thumbnail tokens: `ceil((H/16)/2) * ceil((W/16)/2)`.

The source uses `masked_scatter`, but the processor guarantees a bounded row-major placeholder pattern when it owns prompt expansion. DinoML should lower this to validated indexed row copy:

- Count `input_ids == 396` must equal total projected image rows.
- Feature order is concatenated image/tile order from the processor.
- Reject arbitrary placeholder layouts unless a general scatter path is admitted.

## 9. Graph rewrite / lowering opportunities

### Rewrite: patchified SigLIP2 patch projection

Source pattern:

```text
CPU patchify -> Linear(768 -> C)
```

Replacement: keep CPU patchify for first integration; later fuse patchify + projection only if DinoML owns image preprocessing.

Preconditions: RGB, patch size 16, row-major patch flatten order `(ph,pw,c)` as source `convert_image_to_patches`, same resize/normalize pipeline.

Failure cases: caller supplies precomputed `pixel_values`, non-RGB, different patch size, custom processor kwargs.

### Rewrite: projector pixel-unshuffle + Linear

Source pattern:

```text
[1,H,W,C] -> factor-2 pixel_unshuffle -> [1,H/2,W/2,4C] -> Linear
```

Replacement: specialized gather/reshape feeding GEMM, or fused local reorder + GEMM.

Preconditions: `H` and `W` divisible by 2. Processor smart-resize uses `encoder_patch_size * downsample_factor`, so official path satisfies this.

Layout constraints: this is a local NHWC-like region. Do not rewrite upstream SigLIP2 sequence layout or downstream text sequence axes.

### Rewrite: bounded image embedding stitch

Source pattern:

```text
inputs_embeds.masked_scatter(input_ids == image_token_id, image_features)
```

Replacement: indexed row copy into embedding matrix.

Preconditions: input came from LFM2-VL processor, placeholder count equals feature rows, row-major order preserved.

Failure cases: caller-supplied `inputs_embeds` without `input_ids`, arbitrary masks, or mismatched processor config.

### Rewrite: LFM2 conv layer fast path

Source pattern:

```text
Linear -> chunk -> elementwise multiply -> depthwise causal conv1d -> gate multiply -> Linear
```

Replacement: one-token decode kernel using fixed `[B,H,3]` state; prefill kernel for short depthwise causal conv.

Preconditions: `conv_L_cache=3`, `groups=hidden`, `conv_bias=false` for official configs, static hidden size.

## 10. Kernel fusion candidates

Highest priority:

- LFM2 RMSNorm and q/k per-head RMSNorm.
- GQA attention with RoPE and KV cache for full-attention layers.
- Conv-state decode kernel for LFM2 conv layers.
- Image placeholder indexed row copy.
- Projector pixel-unshuffle + LayerNorm + GEMM chain.

Medium priority:

- SigLIP2 vision LayerNorm + attention + MLP kernels for patch sequences up to 1024.
- Vision positional resize cache by grid shape.
- SwiGLU MLP fusion for text decoder.
- Last-token-only LM head using `logits_to_keep`.

Lower priority:

- CPU image resize/split acceleration inside DinoML runtime.
- Vision encoder FlashAttention substitute; source only marks SDPA for SigLIP2.
- General boolean scatter.

## 11. Runtime staging plan

1. Parse configs and reject unsupported mismatches: non-SigLIP2 vision, non-LFM2 text, unknown layer types, unsupported processor overrides.
2. Load text-only LFM2 and validate hybrid attention/conv block parity with synthetic embeddings.
3. Add processor-owned image embedding stitch with precomputed `image_features`.
4. Add SigLIP2 vision encoder on patchified inputs and projector parity.
5. Run full multimodal prefill logits parity for one image and one prompt.
6. Add decode cache parity: attention KV plus conv states, including no pixel forwarding after first iteration.
7. Optimize GQA attention, conv decode, and projector layout rewrites.

Stubs acceptable in early stages: CPU image preprocessing delegated outside DinoML; vision encoder outputs supplied as inputs; generation controller outside graph.

## 12. Parity and validation plan

- Unit parity for processor token counts: single tile, multi-tile row/col order, thumbnail enabled/disabled.
- Random tensor parity for projector pixel-unshuffle, including non-square grids divisible by 2.
- SigLIP2 vision embedding parity for positional resize at several `spatial_shapes`.
- LFM2 one-layer attention parity with and without cache.
- LFM2 one-layer conv parity for prefill and one-token decode state update.
- End-to-end prefill logits parity for 450M first, then 1.6B/3B shape sweeps.
- Decode parity over at least 4 generated steps with mixed attention/conv layer cache.
- Suggested tolerances: fp32 custom ops `1e-5`; bf16/fp16 model paths compare with relaxed absolute/relative tolerances around `1e-2` after full blocks.

## 13. Performance probes

- CPU preprocessing throughput by image resolution and tile count.
- Vision encoder throughput for 1, 2, 4, 10 tiles plus thumbnail.
- Projector throughput by patch grid and hidden size.
- Prefill throughput by text length plus image token count.
- Decode tokens/sec separated by full-attention layer count and conv layer count.
- KV cache memory plus conv-state memory by batch and sequence length.
- GQA attention backend comparison: eager repeat-KV, SDPA/Flash-style grouped, and cached decode.
- Conv decode backend comparison: generic depthwise Conv1d fallback versus fixed-state update kernel.
- Placeholder copy cost versus sequence length and image token count.

## 14. Skip/defer list

- Training and loss parity except basic labels smoke tests.
- Gradient checkpointing and `torch.compile` fullgraph parity.
- General remote-code variants or non-SigLIP2/non-LFM2 backbones.
- General boolean `masked_scatter`; use bounded indexed copy first.
- Beam search cache reorder until single-sequence decode cache is stable.
- Multi-GPU tensor parallel plans.
- Quantized/GGUF loading for VL weights; treat separately from dense source parity.

## 15. Final implementation checklist

- [ ] Parse `Lfm2VlConfig`, nested `Lfm2Config`, and `Siglip2VisionConfig`.
- [ ] Parse processor config and reject mismatches with model config unless explicitly overridden.
- [ ] Implement patchified SigLIP2 vision input ABI.
- [ ] Implement SigLIP2 resized positional embedding cache.
- [ ] Implement SigLIP2 vision encoder blocks.
- [ ] Implement projector pixel-unshuffle, optional LayerNorm, GELU MLP.
- [ ] Implement bounded image-placeholder indexed row copy.
- [ ] Implement LFM2 RMSNorm, q/k RMSNorm, RoPE, GQA attention.
- [ ] Implement hybrid cache manifest from `layer_types`.
- [ ] Implement LFM2 conv prefill and decode state update.
- [ ] Implement tied LM head and `logits_to_keep`.
- [ ] Add single-block, projector, vision, prefill, and decode parity tests.
- [ ] Benchmark preprocessing, vision, prefill, decode, cache memory, and conv-state update.
