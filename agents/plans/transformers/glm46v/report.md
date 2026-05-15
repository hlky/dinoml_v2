# GLM46V DinoML Audit

## 1. Source basis

Transformers commit/version: local checkout `transformers` at
`b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`.

Model id: source docstrings mention `zai-org/GLM-4.1V-9B-Thinking`; fetched
representative configs include `zai-org/GLM-4.6V`, `zai-org/GLM-4.6V-Flash`,
`zai-org/GLM-4.6V-FP8`, and `zai-org/GLM-4.1V-9B-Thinking`.

Config source: local `configuration_glm46v.py` plus HF raw `config.json` and
`preprocessor_config.json` snapshots in `_sources/`.

Source files inspected:

- `src/transformers/models/glm46v/{configuration,modeling,modular,processing,image_processing,image_processing_pil,video_processing}_glm46v.py`
- Delegated bodies: `src/transformers/models/glm4v/configuration_glm4v.py` and
  `src/transformers/models/glm4v/modeling_glm4v.py`

Any missing files or assumptions: `glm46v` modeling files are generated from
`modular_glm46v.py`; the generated files expose the expanded ABI. The official
GLM-4.6V configs inspected do not use `model_type="glm46v"`: main/FP8 use
`glm4v_moe`, Flash uses `glm4v`. This audit therefore covers the checked-in
`glm46v` wrapper/processor and its delegated `glm4v_text`/`glm4v_vision`
operator body. MoE text-body parity for main GLM-4.6V is gated on a separate
`glm4v_moe` audit.

## 2. High-level architecture

Runtime target: multimodal causal generation with image/video prefix features
stitched into a causal text decoder, then autoregressive decode.

Dataflow:

```text
CPU image/video decode and processor packing
-> packed flattened patch tensors + grid_thw + mm_token_type_ids
-> vision encoder/projector
-> placeholder row copy into token embeddings
-> causal text prefill with M-RoPE
-> cached decode without pixel inputs
-> lm_head logits/sampling
```

Stage decomposition:

- CPU/data pipeline: image resize/rescale/normalize, video frame sampling,
  dynamic resize, patch flattening, prompt placeholder expansion, tokenization,
  `mm_token_type_ids`.
- Vision runtime: independently cacheable image/video encoder outputs. Inputs
  are already flattened patch rows, not raw NHWC/NCHW images.
- Prefix construction: text embeddings plus deterministic replacement of
  `<|image|>` placeholder rows by image/video features.
- Prefill: causal decoder over mixed text and modality tokens with M-RoPE
  position ids and cache construction.
- Decode: text-only one/few-token steps; source intentionally drops
  `pixel_values` and `pixel_values_videos` after first cached iteration.

## 3. Important config dimensions

Source defaults from `glm46v` plus delegated `glm4v` config:

| Field | Default / checked source behavior |
| --- | --- |
| top-level `model_type` | `glm46v` |
| text subconfig default | `glm4v_text` |
| vision subconfig default | `glm4v_vision` |
| text hidden size | 4096 |
| text layers | 40 |
| text attention heads / KV heads | 32 / 2 |
| inferred text head dim | 128 |
| text MLP | packed SwiGLU, `Linear(4096 -> 2*13696)` then `Linear(13696 -> 4096)` |
| text RoPE | M-RoPE, `partial_rotary_factor=0.5` when config supplies it |
| source default max positions | `glm4v_text` default 32768; GLM-4.6V configs use 131072 |
| vision hidden / heads / layers | 1536 / 12 / 24 |
| vision patch | temporal 2, spatial 14, merge 2 |
| vision out hidden | 4096 |
| image token defaults in `glm46v` config | image token 151343, video token 151344, starts 151339/151361 |
| processor image/video ABI | `pixel_values`, `pixel_values_videos`, `image_grid_thw`, `video_grid_thw`, `mm_token_type_ids` |
| cache support | standard `Cache`/`DynamicCache`; keys/values updated after RoPE |

Representative checkpoint sweep:

| Checkpoint | Config model type | Text body | Layers | Heads/KV/head dim | MLP/MoE | Vision | Tokens |
| --- | --- | --- | ---: | --- | --- | --- | --- |
| `zai-org/GLM-4.6V-Flash` | `glm4v` | dense `glm4v_text` | 40 | 32/2/128 inferred | 13696 dense SwiGLU | 24 layers, 1536 hidden, patch 14, merge 2 | image/video ids 151363/151364 |
| `zai-org/GLM-4.6V` | `glm4v_moe` | MoE text, separate audit | 46 | 96/8/128 explicit | 128 routed experts, top-8, shared expert | 24 layers, 1536 hidden, out 4096 | image/video ids 151363/151364 |
| `zai-org/GLM-4.6V-FP8` | `glm4v_moe` | MoE text, quantized | 46 | 96/8/128 explicit | same MoE; compressed-tensors FP8 config | same vision; many ignored dense modules | image/video ids 151363/151364 |
| `zai-org/GLM-4.1V-9B-Thinking` | `glm4v` | dense predecessor | 40 | 32/2/128 inferred | 13696 dense SwiGLU | 24 layers, 1536 hidden, out 4096 | image/video ids 151343/151344 |

## 3a. Family variation traps

- `glm46v` source defaults and GLM-4.6V HF configs disagree on top-level
  `model_type` and placeholder token ids. DinoML should not silently load
  `glm4v_moe` checkpoints through the dense `glm46v` path.
- The processor uses the same repeated `<|image|>` placeholder token for
  standalone images and video frames; `mm_token_type_ids` and video start/end
  spans disambiguate M-RoPE, while `get_placeholder_mask` uses `image_token_id`
  for both image and video replacement when `input_ids` are available.
- GLM-4.6V main/FP8 are MoE models with expert routing; this report does not
  claim dense decoder parity for them.
- Text MLP weights are packed as `gate_up_proj` with chunk order `(gate, up)`;
  fused rewrites must preserve this order.
- Text attention is GQA/MQA-like: 32 query heads and 2 KV heads for dense
  defaults, 96 query and 8 KV heads for GLM-4.6V MoE configs.
- Text position ids may be `[4, batch, seq]` for packed multimodal generation:
  first plane is text position ids for mask construction, remaining planes are
  temporal/height/width M-RoPE ids.
- Image/video processor emits flattened patch rows; the model does not own raw
  image decode, video decode, or frame extraction unless DinoML chooses to move
  preprocessing into the runtime.
- Vision branch uses source NCHW/NCTHW assumptions inside preprocessing and
  patch packing. Channel-last/NHWC is only a guarded optimization for the local
  processor-to-patch region.
- Vision learned position interpolation uses `grid_sample` bicubic,
  `align_corners=False`, `padding_mode="border"`; this is not ordinary RoPE.
- Video frame sampling depends on metadata fps/duration. Pre-sampled frames
  without metadata need a processor fallback policy and prompt timestamps.
- FP8 config uses `compressed-tensors` metadata; treat as a loading/provider
  contract, not ordinary dtype conversion.

## 4. Operator coverage checklist

Tensor/layout ops:

- Embedding lookup for token ids.
- Row-wise placeholder copy into `inputs_embeds`; source uses
  `masked_scatter`, but processor guarantees repeated placeholder rows.
- `reshape`, `view`, `permute`, `transpose`, `contiguous`, `cat`, `split`,
  `repeat`, `repeat_interleave`, `cumsum`, `arange`, `where`/mask comparisons,
  `masked_fill`.
- Packed generation expansion by per-sample visual lengths.

Neural primitives:

- RMSNorm over hidden dim, fp32 variance and cast back.
- LayerNorm in vision merger after linear projection.
- Dense Linear/GEMM with optional bias for text Q/K/V and vision qkv.
- Biasless output projections and MLP down projections.
- SiLU-gated MLP multiply for text and vision.
- GELU in vision merger before gated MLP.
- Vision non-overlap `Conv3d(3 -> 1536, kernel=stride=(2,14,14))` over
  flattened patch windows.
- Vision `Conv2d(1536 -> 4096, kernel=stride=2)` after transformer blocks.

Attention primitives:

- Text causal GQA attention with Q heads != KV heads.
- Vision packed noncausal self-attention with varlen `cu_seqlens`.
- Eager fallback: matmul, mask add, fp32 softmax, dropout disabled for
  inference, matmul V.
- Optimized path: SDPA/FlashAttention-compatible dispatch.

Position/rotary/custom math:

- Text M-RoPE over temporal/height/width planes, with partial rotary factor and
  mrope sections.
- Text rotary rotates interleaved even/odd pairs, not half-split pairs.
- Vision RoPE over 2D grid coordinates plus learned 2D positional
  interpolation via `grid_sample`.
- Multimodal `rope_deltas` cached for decode.

Preprocessing-coupled ops:

- Smart resize with dimensions divisible by `patch_size * merge_size`.
- Image/video rescale and CLIP mean/std normalization.
- Video dynamic frame sampling and even frame count padding.
- Patch flatten order exactly as processor `view -> permute -> reshape`.

Scatter/indexed update:

- Replace arbitrary placeholder rows in token sequence with image/video feature
  rows. DinoML should lower to counted indexed row copy or contiguous segment
  copies under processor guards, not admit general boolean scatter at first.

Packed/varlen metadata:

- `image_grid_thw`, `video_grid_thw`, repeated per-frame video grids, and
  vision `cu_seqlens`.
- `mm_token_type_ids`: 0 text, 1 standalone image placeholder, 2 image token
  inside video span.

Quantized/packed weight metadata:

- Source basis has no quantized-kernel path in `glm46v`; FP8 checkpoint config
  advertises `compressed-tensors` with Linear weight/activation FP8 metadata and
  large ignore lists. First DinoML admission should either reject FP8 or route
  to a separate compressed-tensors loader/provider.

## 5. Layer/block breakdown

Vision preprocessing for each image/video:

```text
raw image/video frames -> resize to multiples of 28
-> rescale/normalize in C-first layout
-> pad temporal length to multiple of 2
-> view [B, grid_t, 2, C, grid_h/2, 2, 14, grid_w/2, 2, 14]
-> permute [B, grid_t, gh/2, gw/2, mh, mw, C, tp, ph, pw]
-> flatten [sum(grid_t*grid_h*grid_w), C*2*14*14]
```

Vision encoder, repeated 24 blocks:

```text
patch rows -> Conv3d patch projection -> RMSNorm
-> add interpolated learned 2D position embedding
for each block:
  x = x + VisionAttention(RMSNorm(x), packed cu_seqlens, 2D RoPE)
  x = x + VisionSwiGLU(RMSNorm(x))
-> RMSNorm
-> view spatial merge 2x2 -> Conv2d downsample 1536 -> 4096
-> Linear(4096 -> 4096) -> LayerNorm -> GELU
-> SwiGLU Linear(4096 -> 13696) / Linear(4096 -> 13696)
-> Linear(13696 -> 4096)
```

Dense text decoder, repeated 40 source-default layers:

```text
residual = x
x = RMSNorm(x)
q = Linear(4096 -> 32*128, bias=True)
k = Linear(4096 -> 2*128, bias=True)
v = Linear(4096 -> 2*128, bias=True)
q,k = M-RoPE(q,k)
k,v = cache.update(k,v) if cache is active
x = Attention(q,k,v, causal_mask)
x = Linear(32*128 -> 4096, bias=False)
x = RMSNorm(x)
x = residual + x
residual = x
x = RMSNorm(x)
gate_up = Linear(4096 -> 2*13696, bias=False)
gate, up = chunk(gate_up, 2)
x = Linear(silu(gate) * up -> 4096, bias=False)
x = RMSNorm(x)
x = residual + x
```

LM head:

```text
hidden[:, slice_indices, :] -> Linear(4096 -> vocab_size, bias=False)
```

## 6. Attention requirements

Text attention:

- Causal self-attention, no cross-attention.
- GQA: query heads and KV heads differ. Dense default is 32 Q heads, 2 KV
  heads, head dim 128; GLM-4.6V MoE configs use 96 Q heads, 8 KV heads, head
  dim 128.
- Q/K/V projections are separate biased Linear layers; output projection is
  biasless.
- Cached keys are stored after M-RoPE application because `past_key_values`
  update happens after `apply_rotary_pos_emb`.
- Cache tensor shape before backend repeat is `[batch, num_kv_heads, seq,
  head_dim]`; attention backend may repeat KV to query heads for eager mode.
- Mask is produced by `create_causal_mask`; packed multimodal paths may pass
  text position ids extracted from `[4,batch,seq]` position ids.
- FlashAttention/SDPA compatible via Transformers attention interface.

Vision attention:

- Noncausal self-attention over packed image/video frame sequences.
- MHA with 12 heads, head dim 128 for 1536 hidden.
- One qkv linear stores split order `q, k, v` after reshape to
  `[seq, 3, heads, head_dim]`.
- Uses `cu_seqlens` where each frame contributes `grid_h * grid_w` patch tokens
  before spatial merging.
- If FlashAttention is unavailable, source splits packed tensors per sequence
  length and runs attention per chunk.

No recurrent/state-space cache exists. Independently cacheable non-KV data:
vision outputs and processor-derived `grid_thw`/placeholder metadata.

## 7. Position encoding and custom math

Text M-RoPE summary:

```python
def glm_text_mrope(inv_freq, position_ids_3d, section=(8, 12, 12)):
    # position_ids_3d: [3, batch, seq]
    freqs = matmul(inv_freq[None, None, :, None], position_ids_3d[:, :, None, :])
    freqs = freqs.transpose(2, 3)  # [3, batch, seq, rotary_dim/2]
    chunks = split(freqs, section, dim=-1)
    mixed = cat([chunk[i % 3] for i, chunk in enumerate(chunks)], dim=-1)
    emb = cat([mixed, mixed], dim=-1)
    return cos(emb), sin(emb)
```

Text rotary application:

```python
def glm_apply_rope(q, k, cos, sin):
    cos = cos[..., : cos.shape[-1] // 2].repeat_interleave(2, dim=-1)
    sin = sin[..., : sin.shape[-1] // 2].repeat_interleave(2, dim=-1)
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    return concat(rotate_even_odd(q_rot, cos, sin), q_pass), concat(rotate_even_odd(k_rot, cos, sin), k_pass)
```

Vision position math:

- `rot_pos_emb(grid_thw)` constructs 2D H/W coordinates in spatial-merge order,
  samples sinusoidal RoPE frequencies for max grid size, indexes by the 2D
  coordinate pairs, then flattens H/W rotary frequencies.
- Learned position table is reshaped to a square 2D grid and sampled at patch
  center coordinates normalized to `[-1, 1]` with bicubic `grid_sample`.

`get_rope_index` custom ABI:

- Groups contiguous runs of `mm_token_type_ids`.
- Text groups get 1D increasing positions repeated over T/H/W planes.
- Image/video groups consume `grid_thw`; videos are first split into per-frame
  grids `[1,h,w]`.
- After an image/video group, `current_pos` advances by `max(h,w)/spatial_merge`
  rather than by token count.
- Returns `mrope_position_deltas = max(position_ids)+1 - valid_token_count`,
  cached for incremental decode.

## 8. Preprocessing and input packing

Image ABI:

- Processor returns `pixel_values` shaped
  `[sum_images grid_t*grid_h*grid_w, 3*2*14*14]`; image `grid_t` is normally 1
  after duplicating one image frame to temporal patch size 2.
- `image_grid_thw` is `[num_images, 3]` in patch-grid units before spatial
  merge.
- Placeholder count per image is `prod(grid_thw) / merge_size^2`.

Video ABI:

- Video decode and metadata are data-pipeline owned.
- `sample_frames` requires `VideoMetadata.fps`; duration <= 30s targets 3 fps,
  <= 300s targets 1 fps, otherwise 0.5 fps, multiplies by temporal patch size,
  caps at 640 extracted frames, de-duplicates, and pads to an even count.
- Processor returns `pixel_values_videos` with the same flattened patch width
  as images and `video_grid_thw=[num_videos,3]`.
- Prompt expansion converts one `<|video|>` into repeated per-frame
  `<|begin_of_image|><|image|><|end_of_image|>{timestamp:.1f} seconds` blocks.
  Timestamps are taken from metadata timestamps every two frames.

Embedding stitch:

- `get_image_features` and `get_video_features` return tuples split by
  modality object. Video feature extraction flattens each video into per-frame
  vision grids, then re-splits by original video token count.
- Source uses `masked_scatter` against masks expanded to hidden size. Feature
  count validation compares total selected embedding elements to total feature
  elements.
- Safer DinoML lowering: compute placeholder row indices from `input_ids` plus
  `mm_token_type_ids`, validate expected counts from `grid_thw`, then copy
  contiguous feature rows into those row indices.

## 9. Graph rewrite / lowering opportunities

### Rewrite: processor patch pack + Conv3d -> Linear

Source pattern:

```text
processor flatten patch rows [N, C*tp*ph*pw]
-> vision PatchEmbed view [-1,C,tp,ph,pw]
-> Conv3d(kernel=stride=[tp,ph,pw])
```

Replacement pattern:

```text
Linear(C*tp*ph*pw -> vision_hidden) with conv weight flattened in source order
```

Preconditions:

- Inputs are exactly processor-emitted flattened patches.
- `Conv3d` kernel equals stride, padding 0, dilation 1, groups 1.
- `tp=2`, `ph=pw=14`, `C=3` unless config says otherwise.
- Flatten order matches processor `view/permute/reshape` and patch embed
  `view(-1,C,tp,ph,pw)`.

Weight transform:

```python
w_linear = conv3d.weight.reshape(out_channels, C * tp * ph * pw)
b_linear = conv3d.bias
```

Failure cases: raw video/image tensors, changed patch order, nonstandard
groups/padding/dilation, or channel-last translation without a matching weight
permutation.

Parity sketch: random pre-flattened patches, compare Conv3d patch embed and
Linear replacement for fp32/bf16.

### Rewrite: masked_scatter placeholder stitch -> indexed row copy

Source pattern: `inputs_embeds.masked_scatter(mask.expand_as(inputs_embeds),
vision_features)`.

Replacement: gather row indices where `input_ids == image_token_id`, split by
`mm_token_type_ids` into image/video rows, validate counts from `grid_thw`, and
copy feature rows into `inputs_embeds[row_indices, :]`.

Preconditions:

- Processor-generated prompt text and `mm_token_type_ids` are present.
- Count of selected rows times hidden size equals feature elements.
- Feature rows are in the same order as processor placeholder expansion.

Failure cases: caller-provided arbitrary `inputs_embeds`, missing
`mm_token_type_ids`, custom prompts with placeholders out of processor order,
or token id mismatch between config and tokenizer.

### Rewrite: packed gate_up SwiGLU

Source pattern: `gate, up = Linear(x).chunk(2, -1); down(silu(gate) * up)`.

Replacement: fused packed-GEMM plus SwiGLU epilogue or two logical GEMMs sharing
one packed weight.

Preconditions: split order gate first, up second; no bias in text MLP; hidden
and intermediate sizes match config.

Failure cases: MoE layers, quantized compressed-tensors modules, or alternate
activation.

### Rewrite: last-token-only logits

Source pattern: `logits_to_keep` slices hidden states before `lm_head`.

Replacement: during decode, materialize only requested final-token hidden rows
for `lm_head`.

Preconditions: no loss computation; generation-only logits.

Failure cases: prefill parity requiring all logits, training labels, tensor
`logits_to_keep` requiring arbitrary rows.

### Layout pass guards

Candidate channel-last optimization is limited to processor resize/normalize
and patch packing. Protect the text decoder, token/placeholder row ABI,
`grid_sample` coordinate math, packed `cu_seqlens`, and M-RoPE axis semantics
with no-layout-translation guards unless a full axis rewrite is implemented.

## 10. Kernel fusion candidates

Highest priority:

- RMSNorm, because every vision and text block uses multiple RMSNorms.
- GQA FlashAttention with M-RoPE and KV cache for text prefill/decode.
- Placeholder indexed row copy, because it avoids general boolean scatter.
- Packed SwiGLU MLP for dense text and vision merger.
- Vision packed noncausal attention with `cu_seqlens`.

Medium priority:

- PatchEmbed Linear rewrite from processor-flattened patches.
- Vision qkv projection plus 2D RoPE.
- Vision learned position interpolation cache for repeated grid sizes.
- Last-token-only logits.
- Generation expansion for visual tensors in beam search.

Lower priority:

- Move smart resize/normalization to GPU. It is useful only if DinoML owns the
  input media pipeline.
- FP8 compressed-tensors provider. Required for FP8 checkpoints, but separate
  from dense `glm46v` source parity.
- MoE routing/expert GEMM for main GLM-4.6V; belongs to `glm4v_moe` audit.

## 11. Runtime staging plan

Stage 1: config admission.

- Accept dense `glm46v` / `glm4v_text` / `glm4v_vision` combinations.
- Reject or route `glm4v_moe` and compressed-tensors FP8 configs.
- Validate tokenizer/config placeholder IDs.

Stage 2: vision-only parity.

- Load vision weights, run processor-emitted flattened patch tensors through
  PatchEmbed, positional interpolation, packed vision blocks, downsample, and
  merger.
- Stub text decoder and compare vision `pooler_output`.

Stage 3: text-only dense decoder parity.

- Implement RMSNorm, GQA attention, M-RoPE, packed SwiGLU, causal mask, cache,
  and `lm_head`.

Stage 4: multimodal prefill.

- Add processor metadata ABI, placeholder indexed copy, 3D position-id
  computation, and prefill logits.

Stage 5: decode.

- Implement `rope_deltas`, skip pixel inputs after first step, cache reorder or
  beam expansion, and last-token logits.

Stage 6: optimized kernels.

- Enable FlashAttention paths, fused RMSNorm/SwiGLU, patch linear rewrite, and
  vision varlen attention.

Stage 7: optional GLM-4.6V production variants.

- Separate `glm4v_moe` route for main GLM-4.6V and FP8 compressed-tensors
  provider route.

## 12. Parity and validation plan

- Processor math tests: smart resize divisibility/aspect guard, image patch
  pack order, video frame sampling with synthetic metadata.
- Custom op tests: text M-RoPE cos/sin and apply-rope against Transformers for
  text-only and mixed image/video positions.
- Vision tests: PatchEmbed Linear rewrite parity, position interpolation
  parity for multiple grids, one vision block, all vision blocks, merger output.
- Text tests: one decoder layer, N-layer prefix, text-only prefill logits,
  single-token decode with cache.
- Stitch tests: image-only, video-only, mixed image+video placeholder row copy
  with count mismatch rejection.
- End-to-end tests: dense GLM-4.6V-Flash-style config first; compare prefill
  logits and greedy decode tokens.
- Tolerances: fp32 custom math near `1e-5`; bf16/fp16 block/logit parity
  should use relaxed absolute/relative tolerances and compare token decisions
  for generation.

No DinoML tests or imports were run for this audit.

## 13. Performance probes

- Processor throughput: image resize/pack and video frame sample/pack
  separately.
- Vision encoder throughput by total patch rows and by number of packed
  sequences.
- Vision position interpolation cache hit/miss cost.
- Prefill throughput by text length plus visual token count.
- Decode tokens/sec with cache for batch size and beam-size sweeps.
- KV cache memory by batch, context length, and dense vs MoE head geometry.
- Placeholder copy bandwidth and count validation overhead.
- Attention backend comparison: eager/SDPA/FlashAttention for text GQA and
  vision varlen MHA.
- PatchEmbed Conv3d vs Linear rewrite.
- Last-token logits vs all-token logits.
- FP8/compressed-tensors load/dequant/provider benchmark only after separate
  admission.

## 14. Skip/defer list

- Training, loss, gradient checkpointing.
- GLM-4.6V MoE text body and expert routing for this `glm46v` report.
- FP8 compressed-tensors execution for first dense path.
- Remote-code-only behavior; in-library source exists.
- Beam search beyond source-compatible tensor expansion/reorder basics.
- GPU-owned media decode and resize.
- General boolean scatter; prefer guarded indexed row copy.
- Arbitrary channel-last layout translation across the full model.
- Speculative decoding and continuous batching.

## 15. Final implementation checklist

- [ ] Add config admission for `glm46v` dense source path.
- [ ] Reject or separately route `glm4v_moe` GLM-4.6V configs.
- [ ] Validate tokenizer/config placeholder IDs and `mm_token_type_ids`.
- [ ] Implement/import processor metadata ABI: `grid_thw`, visual token counts,
  and video timestamps.
- [ ] Implement vision patch pack contract and PatchEmbed Linear rewrite.
- [ ] Implement vision learned position interpolation with `grid_sample` parity.
- [ ] Implement vision packed noncausal attention with `cu_seqlens`.
- [ ] Implement RMSNorm and LayerNorm coverage needed by text/vision.
- [ ] Implement dense text GQA attention with post-RoPE KV cache.
- [ ] Implement GLM M-RoPE position ids, `rope_deltas`, and interleaved
  rotate-half math.
- [ ] Implement packed `gate_up_proj` SwiGLU with gate-first split.
- [ ] Implement placeholder indexed row copy with count/order guards.
- [ ] Implement decode input preparation that drops visual tensors after first
  cached step.
- [ ] Add last-token-only `lm_head` lowering.
- [ ] Add vision-only, text-only, multimodal prefill, and decode parity tests.
- [ ] Benchmark processor, vision, prefill, decode, cache memory, and rewrite
  variants separately.
