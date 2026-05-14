# Transformers audit: `glm_ocr`

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4 from X:/H/transformers
Model id: zai-org/GLM-OCR
Config source: local snapshots in _sources plus https://huggingface.co/zai-org/GLM-OCR/blob/main/config.json
Source files inspected:
  X:/H/transformers/src/transformers/models/glm_ocr/configuration_glm_ocr.py
  X:/H/transformers/src/transformers/models/glm_ocr/modeling_glm_ocr.py
  X:/H/transformers/src/transformers/models/glm_ocr/modular_glm_ocr.py
  X:/H/transformers/src/transformers/models/glm46v/processing_glm46v.py
  X:/H/transformers/src/transformers/models/glm46v/image_processing_glm46v.py
  X:/H/transformers/src/transformers/models/glm46v/video_processing_glm46v.py
  X:/H/transformers/src/transformers/models/glm4v/processing_glm4v.py
  X:/H/transformers/src/transformers/models/glm4v/image_processing_glm4v.py
  X:/H/transformers/src/transformers/models/glm4v/video_processing_glm4v.py
Any missing files or assumptions:
  glm_ocr has no local processor file. The official preprocessor config routes to Glm46VProcessor / Glm46VImageProcessor.
  This report targets inference-only OCR/image-to-text generation on CUDA. Video support is documented but deferred.
```

`configuration_glm_ocr.py` and `modeling_glm_ocr.py` are generated from `modular_glm_ocr.py`. Future Transformers edits should inspect the modular source first, but DinoML should translate the generated runtime file.

## 2. High-level architecture

GLM-OCR is a compact multimodal generation model: packed image patches go through a GLM-V-style vision transformer/projector, the resulting visual embeddings are stitched into text token embeddings at image placeholder tokens, then a causal GLM text decoder generates OCR/markup text.

```text
CPU/image preprocessing -> flattened patch tokens + image_grid_thw
tokenizer/chat template -> input_ids + attention_mask + mm_token_type_ids
vision encoder/projector -> image embeddings
masked scatter into token embeddings -> multimodal causal decoder prefill -> KV-cache decode -> logits/sampling -> decoded text
```

Stage decomposition:

| Stage | Runtime contract | Cacheability |
| --- | --- | --- |
| Image preprocessing | Resize to `patch_size * merge_size` multiples, normalize with CLIP mean/std, flatten `[C,tp,ph,pw]` patches into `pixel_values`, emit `image_grid_thw`. | CPU/data pipeline first; GPU preprocessing optional later. |
| Vision encoder/projector | Consumes `pixel_values` rank 2 `[sum(grid_t*grid_h*grid_w), 1176]` for production defaults and `image_grid_thw`; emits one embedding per merged patch group. | Independently cacheable per image. |
| Prefix construction | Token embeddings plus `masked_scatter` into positions where `input_ids == image_token_id`; requires exact placeholder count. | Can cache stitched prompt embeddings for repeated decode. |
| Prefill | Causal decoder with 4-plane position ids in packed paths: text plane plus 3 M-RoPE planes. | Produces per-layer KV cache. |
| Decode | `prepare_inputs_for_generation` drops image tensors after first iteration when cache is used; uses cached `rope_deltas`. | Standard autoregressive KV cache plus model-level `rope_deltas`. |

There is no bounding-box OCR input ABI. OCR is the generation task, not a caller-supplied layout-box preprocessing path.

## 3. Important config dimensions

Official `zai-org/GLM-OCR` config-derived dimensions:

| Field | Value | Source / notes |
| --- | ---: | --- |
| text hidden size | 1536 | `text_config.hidden_size` |
| text layers | 16 | `text_config.num_hidden_layers` |
| attention heads | 16 | `text_config.num_attention_heads` |
| KV heads | 8 | GQA, `num_key_value_groups = 2` |
| head dim | 128 | Explicit `head_dim`; note `1536 != 16 * 128` |
| Q projection | `1536 -> 2048` | `num_heads * head_dim` |
| K/V projection | `1536 -> 1024` each | `num_key_value_heads * head_dim` |
| O projection | `2048 -> 1536` | Attention output width differs from hidden size. |
| MLP | `1536 -> 9216 -> chunk(4608,4608) -> 1536` | Fused `gate_up_proj`, SiLU-gated. |
| vocab size | 59392 | LM head untied by config. |
| max positions | 131072 | Text config. |
| RoPE | default, theta 10000, partial 1.0 | M-RoPE section `[16,24,24]`. |
| dtype | bfloat16 | Config field. |
| vision hidden/depth | 1024 / 24 | Vision config. |
| vision heads/head dim | 16 / 64 | `hidden_size // num_heads`. |
| vision patch input width | 1176 | `3 * temporal_patch_size(2) * 14 * 14`. |
| vision patch embed | Conv3d `3 -> 1024`, kernel/stride `[2,14,14]` | Source modeling. |
| vision downsample | Conv2d `1024 -> 1536`, kernel/stride `2` | After NHWC-like view then NCHW permute. |
| image tokens per image | `grid_t * grid_h * grid_w / 4` | `spatial_merge_size=2`. |

Representative checkpoint sweep:

| Snapshot | Text dims | Vision dims | Operator-significant variation |
| --- | --- | --- | --- |
| `zai-org/GLM-OCR` | `H=1536`, `L=16`, `heads=16`, `kv=8`, `head_dim=128`, `I=4608` | `H=1024`, `L=24`, `out=1536`, `patch=14`, `merge=2`, `tp=2` | Official production path; GQA with attention width 2048. |
| `mlx-community/GLM-OCR-bf16` | Same as official | Same as official | Mirror/export config; no new ops. |
| `onnx-community/GLM-OCR-ONNX` | Same as official | Same as official | ONNX mirror; useful for ABI comparison, not source authority. |
| `unsloth/GLM-OCR` | Same as official | Same as official | Mirror config; no new ops. |
| `tiny-random/glm-ocr` | `H=8`, `L=2`, `heads=8`, `kv=4`, `head_dim=32`, `I=64` | `H=32`, `L=2`, `out=8`, same patch/merge | Debug shape trap: `hidden_size` much smaller than attention output width. |
| `yujiepan/glm-ocr-tiny-random` | Same as tiny-random | Same as tiny-random | Debug mirror. |

## 3a. Family variation traps

- `hidden_size != num_attention_heads * head_dim` for both official and tiny configs. Projection shapes must use explicit `head_dim`.
- Text attention is GQA: K/V heads are fewer than Q heads and are repeated in eager attention.
- Text Q/K/V/O projections are bias-free; vision QKV/proj and vision MLP use `attention_bias=True` in official configs.
- Text MLP packs gate and up in one weight: `gate_up_proj` output split order is `[gate, up]`.
- Text block applies four RMSNorm modules: pre-attn, post-attn-output before residual, pre-MLP, post-MLP-output before residual.
- M-RoPE uses 3 spatial/temporal planes plus a separate text-position plane for packed mask construction. Naive 1D RoPE is insufficient for multimodal prefill.
- Official preprocessor snapshot names `Glm46VProcessor`, not a `GlmOcrProcessor`.
- `image_token_id` and `video_token_id` config fields differ, but source placeholder masking treats both image and video masks as `image_token_id` when `input_ids` are present.
- Image preprocessing emits flattened patch vectors, not NCHW images, to the model. Layout optimizations must preserve the processor flatten order.
- Vision model immediately reshapes flattened patches back to `[N, C, tp, ph, pw]` for Conv3d. This is a guarded processor-to-patch-embed rewrite opportunity.
- Vision packed attention uses `cu_seqlens`; non-Flash fallback splits sequences per image/frame chunk.
- `image_grid_thw` dimensions must be divisible by `spatial_merge_size` for `rot_pos_emb` reshapes and post-transformer downsample views.
- Generation expansion has custom handling for visual tensors whose leading dimension is total visual token count, not batch size.

## 4. Operator coverage checklist

Tensor/layout ops:

- `view`, `reshape`, `permute`, `transpose`, `contiguous`, `split`, `chunk`, `cat`, `stack`, `flatten`, `unsqueeze`, `expand`, `repeat`, `repeat_interleave`, `cumsum`, `pad`, `arange`, `outer`, `where`/masking equivalents.
- Boolean/equality masks over token ids, `masked_fill`, `masked_scatter`, tensor indexing/gather for RoPE tables.
- Dynamic shape arithmetic from `grid_thw`, `attention_mask`, and cache length.

Neural primitives:

- Embedding lookup `vocab_size x hidden_size`.
- Linear/GEMM:
  - text Q `1536 -> 2048`, K/V `1536 -> 1024`, O `2048 -> 1536`.
  - text packed MLP `1536 -> 9216`, chunk into two `4608`, down `4608 -> 1536`.
  - LM head `1536 -> 59392`, untied by config despite `_tied_weights_keys`.
  - vision QKV `1024 -> 3072`, proj `1024 -> 1024`, MLP `1024 -> 4096 -> 1024`.
  - patch merger `1536 -> 1536`, LayerNorm, GELU, gated MLP `1536 -> 4608 -> 1536`.
- RMSNorm with fp32 variance, LayerNorm in patch merger, SiLU, GELU, elementwise multiply/add.
- Conv3d patch embed and Conv2d spatial downsample.

Attention primitives:

- Causal self-attention for text with GQA, RoPE, cache update, causal mask.
- Noncausal packed vision self-attention with MHA, q/k RMSNorm, vision 2D RoPE, optional varlen FlashAttention ABI.
- SDPA/FlashAttention backend dispatch compatibility if DinoML chooses backend lowering.

Position/rotary/custom math:

- Text M-RoPE with section-based interleaving across temporal/height/width planes.
- Vision 2D RoPE from per-grid h/w positions after merge-order permutation.
- `rope_deltas` carried from prefill to decode.

Generation/cache ops:

- DynamicCache-like per-layer K/V append for 16 text layers.
- Cache length reads for position ids.
- Custom generation preparation that removes image/video tensors after first cached step.
- Beam/expand repeat handling for visual packed tensors.

Preprocessing-coupled and multimodal stitch ops:

- Smart resize to multiples of `patch_size * merge_size`.
- Image/video rescale and CLIP normalize.
- Patch flatten order `(B, grid_t, gh/merge, gw/merge, mh, mw, C, tp, ph, pw) -> [B, grid_t*gh*gw, C*tp*ph*pw]`.
- Placeholder expansion in text: one `<|image|>` becomes `grid_t*grid_h*grid_w/merge^2` image tokens.
- `mm_token_type_ids`: 0 text, 1 standalone image, 2 video-frame image tokens inside video spans.

NHWC/layout guards:

- Initial model input is flattened patches. Treat raw NCHW/NHWC layout as a processor concern unless DinoML owns preprocessing.
- Patch embed expects flattened order compatible with `view(-1, C, tp, ph, pw)`; any channel-last rewrite must rewrite flatten order and Conv3d weights together.
- Vision downsample region reshapes token sequence to `[-1, merge, merge, hidden]`, permutes to NCHW, applies Conv2d. A channel-last optimization is local only if Conv2d and following view/merger are fused or fully rewritten.
- Text decoder is rank-3 `[B,S,H]`; no NHWC translation should cross into text/embedding/LM head.

## 5. Layer/block breakdown

Vision preprocessing and patch embed:

```text
raw image/video -> resize/rescale/normalize -> flattened patches [sum_patches, C*tp*14*14]
patches.view(-1, C, tp, 14, 14)
Conv3d(C=3 -> 1024, kernel=stride=(2,14,14))
flatten -> tokens [sum_patches, 1024]
```

Vision block, repeated 24 times:

```text
x = x + VisionAttention(RMSNorm(x), cu_seqlens, vision_2d_rope)
x = x + VisionMLP(RMSNorm(x))

VisionAttention:
  qkv = Linear(1024 -> 3072, bias=True)
  q,k,v = reshape [S,3,16,64] -> split
  q = RMSNorm(q); k = RMSNorm(k)
  q,k = vision_rope(q,k)
  noncausal attention over each packed image/frame segment
  out = Linear(1024 -> 1024, bias=True)

VisionMLP:
  down(SiLU(gate_proj(x)) * up_proj(x))
  1024 -> 4096 -> 1024, bias follows attention_bias
```

Vision post/projector:

```text
x = RMSNorm(x)
x = x.view(-1, 2, 2, 1024).permute(0,3,1,2)
x = Conv2d(1024 -> 1536, kernel=stride=2)(x).view(-1,1536)
pooler = Linear(1536 -> 1536) -> LayerNorm -> GELU -> gated MLP(1536 -> 4608 -> 1536)
```

Text decoder block, repeated 16 times:

```text
res = x
h = RMSNorm(x)
h = causal GQA attention(h, M-RoPE, mask, KV cache)
h = RMSNorm(h)
x = res + h

res = x
h = RMSNorm(x)
gate_up = Linear(1536 -> 9216, bias=False)
gate, up = chunk(gate_up, 2, dim=-1)
h = Linear(4608 -> 1536, bias=False)(up * SiLU(gate))
h = RMSNorm(h)
x = res + h
```

Final head:

```text
x = RMSNorm(x)
logits = Linear(1536 -> 59392, bias=False)(x[:, slice(-logits_to_keep, None), :])
```

## 6. Attention requirements

Text attention:

- Causal self-attention only; no encoder-decoder cross-attention.
- GQA: Q heads 16, KV heads 8, head dim 128, Q/K/V widths 2048/1024/1024.
- Cached K/V are stored after RoPE because `past_key_values.update` occurs after `apply_rotary_pos_emb`.
- Eager fallback repeats K/V from `[B,8,S,128]` to `[B,16,S,128]`, computes fp32 softmax, and casts to query dtype.
- Masking uses `create_causal_mask`; when packed 4-plane position ids are supplied, text plane is passed separately for mask construction.
- FlashAttention/SDPA are advertised by source. DinoML can use fused attention if it preserves scaling, mask addition before softmax, and GQA semantics.
- Per-layer cache tensors before repeat are `[B,8,T,128]` for keys and values. Query is `[B,16,Q,128]`.

Vision attention:

- Noncausal self-attention over packed image/frame segments.
- MHA, not GQA: 16 Q/K/V heads, head dim 64.
- Q and K have per-head RMSNorm before RoPE.
- FlashAttention path consumes `cu_seq_lens_q`, `cu_seq_lens_k`, and `max_length_q/k`; fallback splits Q/K/V by segment lengths and concatenates outputs.
- No KV cache for vision; image features are independently cacheable outside the decoder.

## 7. Position encoding and custom math

Text M-RoPE:

```python
def glm_ocr_text_rope(position_ids_3, inv_freq, mrope_section):
    # position_ids_3: [3, batch, seq]
    freqs = matmul(inv_freq[None, None, :, None], position_ids_3[:, :, None, :]).transpose(2, 3)
    chunks = split(freqs, mrope_section, dim=-1)
    freqs = cat([chunk[i % 3] for i, chunk in enumerate(chunks)], dim=-1)
    emb = cat([freqs, freqs], dim=-1)
    return cos(emb), sin(emb)
```

Text application uses interleaved even/odd rotation and only rotates the prefix width implied by `cos`:

```python
def apply_text_rope(q, k, cos, sin):
    cos = repeat_interleave(cos[..., : cos.shape[-1] // 2], 2, dim=-1).unsqueeze(1)
    sin = repeat_interleave(sin[..., : sin.shape[-1] // 2], 2, dim=-1).unsqueeze(1)
    q_rot, q_pass = q[..., :cos.shape[-1]], q[..., cos.shape[-1]:]
    k_rot, k_pass = k[..., :cos.shape[-1]], k[..., cos.shape[-1]:]
    return cat([q_rot * cos + rotate_even_odd(q_rot) * sin, q_pass], -1), \
           cat([k_rot * cos + rotate_even_odd(k_rot) * sin, k_pass], -1)
```

Vision RoPE:

- `rot_pos_emb(grid_thw)` builds h/w position ids in merge-major order, indexes a 1D rotary table up to `max(grid_h, grid_w)`, flattens h/w rotary pairs, then duplicates them before cos/sin.
- Vision rotation uses half-split rotation, not the text even/odd rotation.

Position ids and dynamic dependencies:

- `mm_token_type_ids`, `image_grid_thw`, `video_grid_thw`, `attention_mask`, and current cache length determine multimodal position ids.
- `rope_deltas` is model state cached after multimodal prefill and used during incremental decode.

## 8. Preprocessing and input packing

Image preprocessing is processor-owned for first integration:

- Input images are converted to RGB, resized with bicubic interpolation, rescaled by `1/255`, and normalized with CLIP mean/std.
- Resize target is rounded to multiples of `patch_size * merge_size = 28`, bounded by `shortest_edge=12544` and `longest_edge=9633792` in the official preprocessor config.
- Aspect ratio greater than 200 is rejected by `smart_resize`.
- Output `pixel_values` shape is `[sum_images, grid_t * grid_h * grid_w, 3 * temporal_patch_size * 14 * 14]`; for images `grid_t=1` after padding a singleton frame to temporal patch size 2, so patch width is 1176.
- `image_grid_thw` is `[num_images, 3]` with `[grid_t, grid_h, grid_w]`.

Text and placeholder packing:

- Processor expands each `<|image|>` placeholder to `prod(image_grid_thw[i]) / merge_size^2` repeated image tokens before tokenization.
- It returns `mm_token_type_ids` by default. The model requires it whenever multimodal grids are passed and `position_ids` are not precomputed.
- Model validates placeholder feature count by comparing masked embedding slots with image/video feature element count, then uses `masked_scatter`.

Video path:

- Video preprocessing emits `pixel_values_videos` and `video_grid_thw`; GLM-OCR modeling can process it.
- The official OCR target can defer video. If admitted, DinoML must preserve frame sampling, timestamp prompt insertion, per-frame splitting of `video_grid_thw`, and visual tensor expansion under beam search.

Output ABI:

- Primary neural output is logits `[B, kept_seq, vocab_size]`.
- End-to-end OCR parity requires tokenizer decode and any application-level formatting outside the neural graph. The Transformers processor only provides `batch_decode`; no NMS, box conversion, or layout-coordinate postprocess is present.

## 9. Graph rewrite / lowering opportunities

### Rewrite: processor-flattened patch embed -> Linear

Source pattern:

```text
pixel_values.view(-1, C, tp, ph, pw) -> Conv3d(C -> hidden, kernel=stride=(tp,ph,pw)) -> view(-1, hidden)
```

Replacement:

```text
Linear(C*tp*ph*pw -> hidden) over flattened patch rows
```

Preconditions:

- Input `pixel_values` was produced by the GLM46V/GLM4V processor flatten order.
- `Conv3d.kernel_size == stride == (temporal_patch_size, patch_size, patch_size)`.
- No padding, dilation 1, groups 1.
- Flatten row order matches `view(-1, C, tp, ph, pw)`.

Weight transform:

```python
w = conv3d.weight.reshape(out_channels, C * tp * ph * pw)
y = x @ w.T + bias
```

Failure cases: raw image tensors, alternate processor flatten order, nondefault patch sizes without matching shape guards.

Parity test: compare patch embed output for randomized flattened patch rows against source Conv3d for official and tiny configs.

### Rewrite: packed `gate_up_proj` -> fused SwiGLU/SiLU MLP

Source pattern:

```text
gate_up = Linear(H -> 2I)
gate, up = chunk(gate_up, 2, -1)
out = Linear(I -> H)(up * SiLU(gate))
```

Replacement: fused `GEMM + split + SiLU + multiply + GEMM`.

Preconditions: split order `[gate, up]`, bias-free text weights, activation `silu`.

Failure cases: future configs with different `hidden_act` or separate gate/up weights.

### Rewrite: last-token-only logits

Source pattern: `slice_indices = slice(-logits_to_keep, None)` before LM head.

Replacement: during decode, project only the final token hidden state.

Preconditions: `logits_to_keep=1` or equivalent generation controller request; no loss computation.

Failure cases: prompt-logit scoring, full-sequence loss, arbitrary tensor indices.

### Rewrite: local channel-last vision downsample

Source pattern:

```text
x.view(-1, 2, 2, hidden).permute(0,3,1,2) -> Conv2d(kernel=stride=2) -> view(-1,out_hidden)
```

Replacement: NHWC-aware 2x2 patch merge GEMM or fused Conv2d+view.

Preconditions: `spatial_merge_size=2`, contiguous source view, Conv2d kernel=stride=2, no padding/dilation/groups.

Layout constraints: guard this as a local region. Do not propagate NHWC into packed attention or text decoder unless all axis-sensitive ops are rewritten.

### Rewrite: multimodal stitch as indexed copy

Source pattern: boolean `masked_scatter` over expanded embedding tensor.

Replacement: explicit checked indexed copy from visual embedding rows into placeholder token positions.

Preconditions: placeholder count exactly equals visual embedding row count; stable row-major token order.

Failure cases: `inputs_embeds` path where special masks are inferred by embedding equality, malformed prompts, video spans using image token ids.

## 10. Kernel fusion candidates

Highest priority:

- Text RMSNorm, including post-attention/post-MLP norms, because every block has four RMSNorm applications.
- GQA causal attention with RoPE and KV cache for prefill/decode. The official attention width mismatch makes generic hidden-size assumptions dangerous.
- Packed SiLU MLP fusion for `gate_up_proj`.
- Vision patch embed as Linear over flattened patch rows, because it removes awkward Conv3d from the runtime graph when preprocessing is trusted.
- Vision varlen noncausal attention using `cu_seqlens`, for multi-image/batched OCR throughput.

Medium priority:

- Vision qkv projection + q/k RMSNorm + vision RoPE setup.
- Vision downsample Conv2d -> local patch-merge GEMM.
- Patch merger Linear + LayerNorm + GELU + gated MLP.
- Placeholder indexed copy to replace dense boolean `masked_scatter`.
- Last-token-only LM head for decode.

Lower priority:

- Full GPU image preprocessing and resize.
- Video frame sampling and timestamp prompt construction.
- Beam expansion for visual tensors.
- Alternative attention backend parity for eager/SDPA/FlashAttention fallbacks.

## 11. Runtime staging plan

Stage 1: config and weights

- Parse `GlmOcrConfig`, nested text/vision configs, tokenizer special ids, generation config, and preprocessor config.
- Reject configs where `model_type != glm_ocr` or processor classes are not in an allowlist.

Stage 2: text-only decoder parity

- Implement text embeddings, RMSNorm, explicit-width Q/K/V/O GQA, M-RoPE with 1D text positions, SiLU MLP, final norm, LM head.
- Validate tiny-random text-only forward first.

Stage 3: image preprocessing ABI and patch embed

- Accept processor-produced `pixel_values` and `image_grid_thw`.
- Lower patch embed through guarded Linear rewrite or direct Conv3d fallback.

Stage 4: vision encoder/projector

- Implement packed noncausal vision attention, q/k RMSNorm, vision RoPE, downsample, and patch merger.
- Validate image feature outputs independently.

Stage 5: multimodal prefill

- Implement placeholder count validation, indexed visual embedding copy, `mm_token_type_ids`, M-RoPE position construction, and `rope_deltas`.

Stage 6: cached decode

- Add KV cache ABI `[layers][K,V] = [B,kv_heads,T,head_dim]`, cache update after RoPE, and decode position id continuation using `rope_deltas`.
- Ensure image tensors are consumed only on first cached iteration.

Stage 7: optimized kernels

- Add FlashAttention/GQA kernels, varlen vision attention, fused MLP/norm, last-token logits, and layout-local vision rewrites.

Stub initially:

- Training/loss, hidden-state/attention outputs, video, beam expansion, GPU processor, and remote-code/mirror-specific quantized loaders.

## 12. Parity and validation plan

- Config parser tests for official, tiny-random, and mirror configs; assert explicit projection widths.
- Unit tests for text RoPE even/odd rotation and M-RoPE `mrope_section` slicing.
- Unit tests for `get_vision_position_ids`, `get_rope_index`, and `rope_deltas` with mixed text/image token groups.
- Patch embed rewrite parity against Conv3d for random flattened patches.
- Vision RoPE and `rot_pos_emb` parity for multiple `grid_thw`, including non-square grids.
- Single text block parity in fp32, then bf16 tolerance.
- Single vision block parity with one image and two packed images.
- Full vision encoder/projector parity on tiny-random config.
- Multimodal stitch parity: placeholder mismatch must fail; valid prompt must match source embeddings after scatter.
- Prefill logits parity for tiny-random image+text prompts.
- Decode parity for one or more generated tokens with cache enabled; verify image inputs are not reread after prefill.
- Suggested tolerances: fp32 `rtol=1e-4, atol=1e-4`; bf16/fp16 block-level `rtol=3e-2, atol=3e-2`, with tighter logits checks where accumulation is fp32.

## 13. Performance probes

- CPU preprocessing throughput by image size and aspect ratio.
- Processor output token count sweep from `image_grid_thw`.
- Patch embed direct Conv3d vs Linear rewrite.
- Vision encoder throughput by packed image count and max segment length.
- Vision varlen attention backend comparison: eager split loop vs fused varlen attention.
- Text prefill throughput by prompt length and image-token count.
- Decode tokens/sec with KV cache for batch sizes 1, 4, 16.
- KV cache memory: `layers * 2 * B * kv_heads * T * head_dim * dtype_size`.
- LM head cost: full sequence vs last-token-only logits.
- End-to-end OCR requests/sec split into preprocessing, vision, prefill, decode, decode/postprocess.
- Placeholder indexed-copy overhead for large image token counts.

## 14. Skip/defer list

- Training, labels/loss, gradient checkpointing.
- Video OCR path, including sampling, timestamp prompt construction, and visual tensor beam expansion.
- GPU image/video preprocessing and resizing.
- Beam search visual expansion and speculative decoding.
- Returning attentions/hidden states as first target.
- Quantized or packed weight formats from mirrors; treat them as separate loading/provider audits.
- Remote-code-only variants and non-`glm_ocr` GLM-V families.
- Full NHWC propagation. Only local guarded vision rewrites should be attempted first.

## 15. Final implementation checklist

- [ ] Parse nested `GlmOcrConfig` and reject unsupported processor/model_type combinations.
- [ ] Load text, vision, projector, and LM-head weights with explicit Q/K/V/O widths.
- [ ] Implement text RMSNorm and four-norm decoder block ordering.
- [ ] Implement explicit-width GQA attention with KV cache stored after RoPE.
- [ ] Implement GLM-OCR text M-RoPE and `rope_deltas` continuation.
- [ ] Implement processor ABI ingestion for `pixel_values`, `image_grid_thw`, `mm_token_type_ids`, and placeholder counts.
- [ ] Add guarded flattened patch embed -> Linear rewrite.
- [ ] Implement vision 2D RoPE and packed noncausal attention with `cu_seqlens`.
- [ ] Implement vision downsample and patch merger.
- [ ] Replace `masked_scatter` with checked indexed copy.
- [ ] Add multimodal prefill parity on tiny-random config.
- [ ] Add cached decode parity and verify image tensors are first-iteration only.
- [ ] Benchmark preprocessing, vision, prefill, decode, LM head, and cache memory separately.

## Gated gaps for DinoML admission

- `head_dim` must be an explicit config field or derived exactly as source does; lowering must not assume `hidden_size / heads` controls projection output width.
- Multimodal admission requires `mm_token_type_ids` or precomputed compatible 3D/4D position ids when image/video grids are present.
- Placeholder token count must equal visual feature rows before any graph execution.
- `image_grid_thw[:,1:]` must be divisible by `spatial_merge_size`; resized processor outputs naturally satisfy this, but external callers need guards.
- Initial integration should reject raw images as model inputs unless DinoML owns the GLM46V preprocessing path.
- NHWC/channel-last rewrites are allowed only inside guarded local vision regions with axis/weight rewrites; text and packed sequence semantics remain layout-neutral `[B,S,H]`.
- Vision varlen attention needs either a real `cu_seqlens` backend or an admitted split-loop fallback with bounded performance expectations.
- Decode admission needs a cache ABI plus persisted `rope_deltas`; a stateless per-token call will not match multimodal generation positions.
