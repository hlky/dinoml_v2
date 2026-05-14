# DinoML Transformers Audit: video_llama_3

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: lkhl/VideoLLaMA3-2B-Image-HF, lkhl/VideoLLaMA3-7B-Image-HF
Config source: HF raw config/preprocessor JSON snapshots saved beside this report
Source files inspected:
- X:/H/transformers/src/transformers/models/video_llama_3/configuration_video_llama_3.py
- X:/H/transformers/src/transformers/models/video_llama_3/modeling_video_llama_3.py
- X:/H/transformers/src/transformers/models/video_llama_3/modular_video_llama_3.py
- X:/H/transformers/src/transformers/models/video_llama_3/processing_video_llama_3.py
- X:/H/transformers/src/transformers/models/video_llama_3/image_processing_video_llama_3.py
- X:/H/transformers/src/transformers/models/video_llama_3/video_processing_video_llama_3.py
- X:/H/transformers/src/transformers/models/qwen2/modeling_qwen2.py
- X:/H/transformers/src/transformers/models/qwen2/configuration_qwen2.py
Any missing files or assumptions:
- Generated video_llama_3 files state they are generated from modular_video_llama_3.py; future source edits should inspect the modular file first.
- The native in-library report target is VideoLlama3ForConditionalGeneration with Qwen2 text_config.
- DAMO-NLP-SG/VideoLLaMA3-2B and DAMO-NLP-SG/VideoLLaMA3-7B configs are remote-code videollama3_qwen2, not native video_llama_3. They are evidence for migration traps, not required native-source parity.
```

Evidence snapshots written:

- `lkhl__VideoLLaMA3-2B-Image-HF__config.json`
- `lkhl__VideoLLaMA3-7B-Image-HF__config.json`
- `lkhl__VideoLLaMA3-2B-Image-HF__preprocessor_config.json`
- `lkhl__VideoLLaMA3-2B-Image-HF__video_preprocessor_config.json`
- `lkhl__VideoLLaMA3-7B-Image-HF__preprocessor_config.json`
- `lkhl__VideoLLaMA3-7B-Image-HF__video_preprocessor_config.json`
- `DAMO-NLP-SG__VideoLLaMA3-2B__config.json`
- `DAMO-NLP-SG__VideoLLaMA3-7B__config.json`

## 2. High-level architecture

VideoLLaMA3 is a multimodal generation wrapper:

```text
CPU/video decode and frame sampling -> image/video patch flattening
-> VideoLlama3 vision transformer -> pixel_unshuffle downsample -> MLP projector
-> placeholder embedding stitch into Qwen2 token embeddings
-> Qwen2 causal decoder prefill/decode -> lm_head logits -> generation controller
```

Stage decomposition:

- CPU/data pipeline: image/video loading, RGB conversion, resize, rescale, normalize, patch flattening, video timestamp prompt expansion, video token compression mask construction.
- Vision encoder/projector: independently cacheable per image/video. It consumes flattened patch rows plus `grid_thw` and `merge_sizes`, and emits projected rows in text hidden size.
- Prefix construction: text embeddings are created by Qwen2 token embedding and image/video features replace placeholder token rows via `masked_scatter`.
- Prefill: Qwen2 causal decoder consumes the full mixed text/media sequence.
- Decode: generation skips forwarding pixel tensors after the first iteration when `use_cache=True`; only text token IDs, masks, positions, and KV cache continue.

First useful DinoML runtime target: native `VideoLlama3ForConditionalGeneration` image/video-to-text prefill and decode for the lkhl HF configs, with Qwen2 delegated to the existing or separately audited Qwen2 decoder plan.

## 3. Important config dimensions

Native checkpoint sweep, from saved `config.json` files:

| Field | 2B Image HF | 7B Image HF | Operator impact |
|---|---:|---:|---|
| dtype | bfloat16 | bfloat16 | bf16 weights/activations expected |
| text model_type | qwen2 | qwen2 | compose Qwen2 decoder |
| text hidden_size | 1536 | 3584 | decoder/projection/logit GEMM widths |
| text layers | 28 | 28 | same depth, larger 7B width |
| text attention heads | 12 | 28 | head_dim 128 for both |
| text KV heads | 2 | 4 | GQA, repeat groups 6 and 7 |
| text intermediate_size | 8960 | 18944 | SwiGLU MLP GEMMs |
| vocab_size | 151936 | 152064 | lm_head and embedding width differ |
| max_position_embeddings | 131072 | 32768 | 2B native HF extends context |
| rope_theta | 1000000.0 | 1000000.0 | Qwen2 RoPE base |
| sliding_window | null | null | no sliding attention in native configs |
| vision hidden_size | 1152 | 1152 | shared vision encoder width |
| vision layers | 27 | 27 | shared vision encoder depth |
| vision heads | 16 | 16 | vision head_dim 72 |
| vision intermediate_size | 4304 | 4304 | GELU MLP width |
| vision patch_size | 14 | 14 | processor emits 3*14*14 rows |
| image_token_id | 151655 | 151655 | placeholder stitch |
| video_token_id | 151656 | 151656 | placeholder stitch |

Processor dimensions, from saved preprocessor files:

| Field | Image processor | Video processor |
|---|---:|---:|
| layout emitted | flattened patch rows | flattened patch rows |
| patch_size | 14 | 14 |
| temporal_patch_size | 1 | 1 |
| merge_size | 1 | 2 |
| min_pixels | 3136 | 12544 |
| max_pixels | 3211264 | 12845056 |
| max_frames | n/a | 180 |
| frame sampling default | n/a | `do_sample_frames=false` |
| token compression | n/a | true |
| normalization | mean/std 0.5 | mean/std 0.5 |

Remote-code trap configs, from DAMO-NLP-SG snapshots:

| Field | DAMO 2B | DAMO 7B | Native admission note |
|---|---:|---:|---|
| model_type | videollama3_qwen2 | videollama3_qwen2 | not native `video_llama_3` |
| auto_map | remote config/modeling | remote config/modeling | separate audit or reject |
| image token field | image_token_index=151665 | image_token_index=151665 | differs from native IDs |
| use_token_compression | true | true | config flag not read by native `VideoLlama3Config` |
| vision config key | vision_encoder_config | vision_encoder_config | differs from native `vision_config` |

## 3a. Family variation traps

- Native source delegates text to `AutoModel.from_config(config.text_config)`. This report owns the wrapper, vision encoder, projector, stitching, and video/image processors; Qwen2 decoder ops should compose a Qwen2 audit.
- `hidden_size != vision_hidden_size`: projector maps 1152 to text hidden size 1536 or 3584.
- Text attention is GQA: 2B has 12 query heads and 2 KV heads; 7B has 28 query heads and 4 KV heads.
- Vision head_dim is 1152 / 16 = 72, so vision RoPE uses `head_dim // 2 = 36` in `VideoLlama3VisionRotaryEmbedding`.
- Video merge size is 2 by default, image merge size is 1. Placeholder counts are `grid_t * grid_h * grid_w / merge_size^2`.
- Video token compression makes the actual number of video placeholders data-dependent through pixel-difference thresholding.
- Processor emits flattened patch rows, not NCHW images, to the model. The model then views each row as `[C, patch, patch]` before a Conv2d.
- `masked_scatter` is broad PyTorch API, but processor-created placeholders are ordered row slots. DinoML can lower to validated indexed row copy if counts and order match.
- Native `get_video_features` reuses `get_image_features`; video temporal information enters through `grid_thw`, vision cu_seqlens, vision RoPE repetition, and processor prompt timestamps, not a 3D convolution.
- Generated files should not be edited directly; modular source is authoritative for upstream patches.
- Remote-code DAMO configs use different class names and token fields. Do not silently load them as native `video_llama_3`.

## 4. Operator coverage checklist

Tensor/layout ops:

- `view`, `reshape`, `permute`, `transpose`, `contiguous`, `unsqueeze`, `expand`, `flatten`, `cat`, `split`, `repeat`, `repeat_interleave`, `cumsum`, `pad`.
- Boolean masks, `masked_scatter`, boolean indexing for `video_embeds[video_compression_mask]`.
- Dynamic row counts from `grid_thw.prod(dim=1)` and `merge_sizes`.
- Beam expansion helpers for flat visual tensors: split per sample, repeat, concatenate.

Neural network primitives:

- Embedding lookup for text tokens.
- Vision patch Conv2d: effectively `Linear(3*14*14 -> 1152)` with kernel=stride=14 on each flattened patch row.
- Vision LayerNorm(1152, eps=1e-6).
- Vision linear projections with bias: q/k/v/o `1152 -> 1152`.
- Vision MLP: `Linear(1152 -> 4304)`, GELU tanh approximation, `Linear(4304 -> 1152)`.
- Vision post LayerNorm and bilinear interpolate inside `pixel_unshuffle`.
- Projector: `Linear(1152 -> text_hidden)`, exact `nn.GELU()`, `Linear(text_hidden -> text_hidden)`.
- Qwen2 decoder: RMSNorm, biased q/k/v projections, biasless o projection, SwiGLU MLP, final RMSNorm, lm_head biasless linear.

Attention primitives:

- Vision: noncausal self-attention over each image/video chunk, MHA 16 heads, head_dim 72. FlashAttention path uses varlen `cu_seqlens`; non-FA path loops over chunks and runs dense attention.
- Text: causal GQA Qwen2 attention with RoPE, KV cache, optional attention backend dispatch. Native configs use full attention only.

Position/rotary ops:

- Vision 2D RoPE from h/w grid positions after merge-size block ordering.
- Text Qwen2 RoPE from `position_ids`, theta 1e6.

Preprocessing-coupled ops:

- Smart resize to dimensions divisible by `patch_size * merge_size`.
- Channels-first RGB, rescale, normalize.
- Image patch flatten order:
  `[B,C,H,W] -> [B, grid_h/merge, grid_w/merge, merge, merge, C, patch, patch] -> [B, grid_h*grid_w, C*Tpatch*patch*patch]`.
- Video patch flatten order:
  `[B,T,C,H,W] -> [B, grid_t, grid_h/merge, grid_w/merge, merge, merge, C, temporal_patch, patch, patch] -> [B, grid_t*grid_h*grid_w, C*Tpatch*patch*patch]`.
- Video compression mask: per-frame token keep mask from mean absolute row difference times 255, threshold 0.1, at least one token per frame.
- Prompt expansion with `Time {t:.1f}s:` before per-frame video placeholder runs.

Generation/cache ops:

- Qwen2 DynamicCache per text layer; cached keys/values are after text RoPE.
- First decode iteration carries media tensors; later cached iterations omit `pixel_values` and `pixel_values_videos`.
- Visual outputs can be cached independently of the text KV cache for repeated prompts.

## 5. Layer/block breakdown

Vision embeddings:

```text
pixel_values: [sum(grid_t*grid_h*grid_w), 3*14*14]
x = view(-1, 3, 14, 14)
x = Conv2d(3 -> 1152, kernel=14, stride=14, valid)
x = view(-1, 1152)
```

Vision encoder block, repeated 27 times in representative HF configs:

```text
residual = x
x = LayerNorm(x)
q,k,v = Linear(1152 -> 1152, bias=True)
q,k = 2D vision RoPE(q,k)
x = noncausal MHA over each cu_seqlens chunk
x = Linear(1152 -> 1152, bias=True)
x = residual + x
residual = x
x = LayerNorm(x)
x = Linear(1152 -> 4304, bias=True)
x = GELU tanh approximation
x = Linear(4304 -> 1152, bias=True)
x = residual + x
```

Vision output:

```text
x = LayerNorm(1152)
x = pixel_unshuffle(x, grid_thw, merge_sizes)
x = Projector: Linear(1152 -> H_text) -> GELU -> Linear(H_text -> H_text)
split per media item by grid.prod / merge_size^2
```

Qwen2 text block, repeated 28 times for both native checkpoints:

```text
residual = x
x = RMSNorm(x)
q = Linear(H -> n_heads*128, bias=True)
k = Linear(H -> n_kv_heads*128, bias=True)
v = Linear(H -> n_kv_heads*128, bias=True)
q,k = RoPE(q,k)
k,v = cache.update(k,v) if decoding
x = causal GQA attention(q,k,v, mask)
x = Linear(n_heads*128 -> H, bias=False)
x = residual + x
residual = x
x = RMSNorm(x)
x = Linear(H -> I, bias=False)
g = silu(gate)
u = Linear(H -> I, bias=False)
x = Linear(I -> H, bias=False)(g * u)
x = residual + x
```

Wrapper:

```text
inputs_embeds = token_embedding(input_ids)
image/video embeds replace placeholder rows
hidden = Qwen2(inputs_embeds, attention_mask, position_ids, cache)
logits = Linear(H_text -> vocab_size, bias=False)
```

## 6. Attention requirements

Vision attention:

- Noncausal self-attention, no mask.
- MHA, not GQA: `num_key_value_groups = 1`.
- Shapes before backend: q/k/v `[1, 16, total_tokens, 72]`.
- Chunk boundaries are described by `cu_seqlens = pad(cumsum(repeat_interleave(h*w, t)))`.
- FlashAttention path should use varlen q/k cu_seqlens and max chunk length.
- Eager/SDPA path splits q/k/v by chunk and concatenates results along sequence.

Text attention:

- Causal self-attention from Qwen2.
- GQA: 2B `Q=12, KV=2, head_dim=128`; 7B `Q=28, KV=4, head_dim=128`.
- KV cache stores un-repeated KV heads, shape conceptually `[B, n_kv_heads, S_cache, 128]` per layer.
- Repeat expansion is only for eager attention; optimized attention should consume GQA directly where possible.
- Native configs set `sliding_window=null` and `use_sliding_window=false`; sliding/local attention can be deferred for native lkhl targets.
- Masking is standard causal mask plus input attention mask. Padding-free/FA helpers are inherited from Transformers attention utilities and Qwen2.

## 7. Position encoding and custom math

Vision RoPE builds h/w position IDs in merge-size block order:

```python
for (t, h, w), merge in zip(grid_thw, merge_sizes):
    h_ids = arange(h).reshape(h//merge, merge, w//merge, merge)
    h_ids = h_ids.permute(0, 2, 1, 3).flatten()
    w_ids = arange(w).reshape(h//merge, merge, w//merge, merge)
    w_ids = w_ids.permute(0, 2, 1, 3).flatten()
    pos_ids.append(stack([h_ids, w_ids], -1).repeat(t, 1))
freqs = outer(arange(max(h,w)), inv_freq)
rot = freqs[pos_ids].flatten(1)
cos, sin = cos(cat([rot, rot], -1)), sin(cat([rot, rot], -1))
```

Vision apply:

```python
q = q.float(); k = k.float()
q = q * cos.unsqueeze(-2) + rotate_half(q) * sin.unsqueeze(-2)
k = k * cos.unsqueeze(-2) + rotate_half(k) * sin.unsqueeze(-2)
q,k = q.to(orig_dtype), k.to(orig_dtype)
```

Text RoPE is Qwen2 default RoPE with theta 1e6. Cos/sin depend on runtime `position_ids` and current cache length; inv_freq can be precomputed per config.

## 8. Preprocessing and input packing

Image ABI:

- Input ownership: processor/data pipeline owns decode, RGB conversion, resize, rescale, normalize, and patch flattening.
- Model receives `pixel_values` as `[sum_patches, 588]` for patch_size 14 and temporal_patch_size 1.
- `image_grid_thw` has one row per image: `[1, grid_h, grid_w]`.
- `image_merge_sizes` has one scalar per image, default 1.
- Placeholder count per image is `grid_h * grid_w`.

Video ABI:

- Input ownership: processor/data pipeline owns decode/frame sampling. Default `do_sample_frames=false`; callers may pass frames directly. If FPS metadata is missing, processor warns and uses fps=1 for timestamps.
- Processor expects/normalizes channel-first video tensors and emits `pixel_values_videos` as `[sum(grid_t*grid_h*grid_w), 588]`.
- `video_grid_thw` has one row per video: `[grid_t, grid_h, grid_w]`.
- `video_merge_sizes` default 2; placeholder count before compression is `grid_t * grid_h * grid_w / 4`.
- `video_compression_mask` has length equal to pre-compression video tokens and is applied after projector.
- Text prompt replacement expands each video token into comma-separated per-frame segments with timestamp text plus repeated video placeholders for kept frame tokens.

Embedding stitch:

- Placeholder IDs: image 151655, video 151656 in native configs.
- Source uses `masked_scatter` after validating `inputs_embeds[mask].numel() == features.numel()`.
- Processor-generated text creates ordered placeholder runs, so DinoML should implement a stricter indexed row-copy with guards: placeholder count equals feature rows, feature width equals hidden size, row-major feature order matches processor order, and unsupported arbitrary boolean scatter falls back/rejects.

## 9. Graph rewrite / lowering opportunities

### Rewrite: flattened patch Conv2d -> Linear

Source pattern:

```text
pixel_values [N, C*P*P] -> view(N,C,P,P) -> Conv2d(C->D,kernel=P,stride=P) -> view(N,D)
```

Replacement:

```text
MatMul(pixel_values, conv.weight.reshape(D, C*P*P).T) + bias
```

Preconditions:

- `kernel_size == stride == patch_size`.
- `padding == valid`, dilation 1, groups 1.
- Input rows are exactly flattened `[C, patch_h, patch_w]` in processor order.
- Patch size and channel count match config.

Failure cases: raw NCHW image tensors, custom patch_size without matching processor flattening, grouped/dilated conv, or altered flatten order.

Parity sketch: compare vision embedding output on random flattened patches and on processor-produced image/video patches.

### Rewrite: placeholder masked_scatter -> indexed row copy

Source pattern:

```text
mask = input_ids == image_token_id/video_token_id
inputs_embeds = inputs_embeds.masked_scatter(mask[...,None].expand_as(inputs_embeds), features)
```

Replacement:

```text
positions = nonzero(input_ids == token_id) in row-major order
copy features[i, :] into inputs_embeds[positions[i], :]
```

Preconditions:

- Feature row count equals placeholder count after video compression.
- Feature width equals text hidden size.
- Processor guarantees row-major placeholder order.

Failure cases: caller supplies `inputs_embeds` without `input_ids`, arbitrary equal-to-placeholder embeddings, or mixed hand-written prompts whose placeholders do not match grids.

### Rewrite: vision chunk loop -> varlen attention

Source pattern:

```text
split q/k/v by cu_seqlens -> dense attention per chunk -> cat
```

Replacement: varlen FlashAttention or grouped batched attention with `cu_seqlens`.

Preconditions: no attention mask, noncausal, same head_dim 72, exact chunk boundaries.

### Layout note: processor-to-vision region

The processor currently emits channel-first flattened patch rows. Treat NTHWC/NHWC as an optimization only if DinoML owns the entire preprocess-to-embedding region and rewrites flatten order plus Conv/Linear weight layout together. The semantic graph should initially preserve source axes.

## 10. Kernel fusion candidates

Highest priority:

- Qwen2 RMSNorm and residual-add patterns: used twice per text layer plus final norm.
- Qwen2 GQA FlashAttention with KV cache: dominates prefill/decode.
- Qwen2 SwiGLU MLP: `silu(gate) * up` between two large GEMMs.
- Projector MLP: small but on every visual token; fuse GELU between two GEMMs where profitable.
- Placeholder indexed copy: removes general scatter admission.

Medium priority:

- Vision flattened-patch Linear rewrite: avoids Conv2d machinery for already flattened rows.
- Vision LayerNorm + MHA + output projection attention stack, especially varlen chunking.
- Vision GELU MLP.
- Vision pixel_unshuffle/interpolate region: axis-heavy and worth isolating after parity.
- Last-token-only logits for decode.

Lower priority:

- Beam visual tensor expansion helpers.
- Processor video compression on GPU. First target can keep it in CPU/data pipeline.
- Layout translation to channel-last/NTHWC for preprocessing kernels.

## 11. Runtime staging plan

Stage 1: Native config admission.

- Parse native `video_llama_3` configs and reject remote-code `videollama3_qwen2` unless separately audited.
- Load shared/tied weights correctly: `lm_head.weight` aliases `model.language_model.embed_tokens.weight` when tie behavior requires it.

Stage 2: Vision embedding/projector parity.

- Implement flattened patch Linear rewrite, vision LayerNorm, noncausal chunk attention, MLP, pixel_unshuffle, projector.
- Validate image-only and video-only visual features against Transformers.

Stage 3: Mixed prefix construction.

- Implement placeholder count validation and indexed row copy for image/video features.
- Support video compression mask as input metadata.

Stage 4: Text prefill with Qwen2.

- Compose Qwen2 decoder for full mixed sequence and compare logits.

Stage 5: Decode with cache.

- Reuse visual prefix only in first iteration, then Qwen2 KV cache decode.

Stage 6: Optimized attention/fusions.

- Add varlen vision attention, GQA FlashAttention, fused RMSNorm/SwiGLU, last-token logits.

Stage 7: Processor integration.

- Decide CPU-only processor boundary first; later evaluate GPU preprocessing/video compression.

## 12. Parity and validation plan

- Config parse tests: native lkhl 2B/7B accepted; DAMO remote-code configs rejected or routed to remote-code audit.
- Vision embedding test: random flattened rows `[N,588]` through Conv2d path versus Linear rewrite.
- Vision RoPE test: fixed `grid_thw` and `merge_sizes` compare cos/sin and q/k rotation.
- Vision block parity: one layer and full 27-layer random tiny config, fp32 tolerance `1e-5`, bf16 tolerance around `2e-2` depending on backend.
- Pixel_unshuffle parity: include `merge_size=1` image and `merge_size=2` video shapes.
- Projector parity: `1152 -> 1536` and `1152 -> 3584`.
- Stitch parity: image-only, video-only with compression mask, mixed image+video, batch with one text-only row.
- Qwen2 prefill parity: logits for short mixed sequence; use fp32/bf16 comparisons separately.
- Decode parity: ensure media tensors are skipped after first cached iteration and token logits match Transformers.
- End-to-end smoke: use processor snapshots for image and video prompts, compare first generated greedy tokens.

DinoML tests were intentionally not run for this audit task.

## 13. Performance probes

- Processor throughput: image resize/patching and video decode/frame sampling separately.
- Video token compression cost versus token reduction ratio across frame counts.
- Vision encoder throughput by total visual tokens and number of chunks.
- Vision attention backend comparison: chunk-loop SDPA/eager versus varlen FlashAttention.
- Projector throughput for image merge size 1 versus video merge size 2.
- Prefill throughput by text length plus visual token count.
- Decode tokens/sec and KV cache memory for 2B and 7B.
- Last-token-only logits versus full-sequence logits during decode.
- Batch expansion/beam search memory for flat visual tensors.
- End-to-end requests/hour split into processor, vision/projector, prefill, decode.

## 14. Skip/defer list

- Training, losses, gradient checkpointing.
- Remote-code DAMO `videollama3_qwen2` checkpoints.
- Sliding-window Qwen2 attention, since native lkhl configs disable it.
- Arbitrary `inputs_embeds` placeholder detection without `input_ids`.
- General boolean `masked_scatter`; use guarded indexed row copy first.
- GPU video decode/frame sampling and GPU token compression.
- Multi-GPU tensor parallel plans.
- Speculative decoding and sampling processors beyond standard generation controller.
- Channel-last/NTHWC layout translation until processor-to-embedding region is controlled.

## 15. Final implementation checklist

- [ ] Parse native `VideoLlama3Config` and nested `VideoLlama3VisionConfig`/Qwen2 `text_config`.
- [ ] Reject or route remote-code `videollama3_qwen2` configs.
- [ ] Load vision, projector, Qwen2, embedding, and lm_head weights with tied-weight aliases.
- [ ] Implement flattened patch Conv2d-to-Linear rewrite.
- [ ] Implement vision 2D RoPE and varlen/chunked noncausal MHA.
- [ ] Implement vision LayerNorm, GELU MLP, post LayerNorm, and pixel_unshuffle.
- [ ] Implement projector MLP for `1152 -> text_hidden`.
- [ ] Implement processor ABI guards for `pixel_values`, `grid_thw`, `merge_sizes`, and video compression mask.
- [ ] Implement placeholder indexed row copy for image/video embedding stitch.
- [ ] Compose Qwen2 causal decoder with GQA KV cache.
- [ ] Add prefill parity for image-only, video-only, mixed, and text-only batch rows.
- [ ] Add decode parity proving visual tensors are first-iteration only.
- [ ] Add performance probes for processor, vision, prefill, decode, and KV memory.
