# PaddleOCR-VL Transformers Family Audit

## 1. Source basis

```text
Transformers commit/version:
  X:/H/transformers @ b75feb2af64c3e29cbbc1bd859958c5432cc7ed4

Model ids:
  PaddlePaddle/PaddleOCR-VL
  PaddlePaddle/PaddleOCR-VL-1.5

Config source:
  agents/plans/transformers/paddleocr_vl/_sources/hf_main_config.json
  agents/plans/transformers/paddleocr_vl/_sources/hf_main_preprocessor_config.json
  agents/plans/transformers/paddleocr_vl/_sources/hf_main_tokenizer_config.json
  agents/plans/transformers/paddleocr_vl/_sources/hf_15_main_config.json
  agents/plans/transformers/paddleocr_vl/_sources/hf_15_main_preprocessor_config.json

Source files inspected:
  X:/H/transformers/src/transformers/models/paddleocr_vl/configuration_paddleocr_vl.py
  X:/H/transformers/src/transformers/models/paddleocr_vl/modeling_paddleocr_vl.py
  X:/H/transformers/src/transformers/models/paddleocr_vl/modular_paddleocr_vl.py
  X:/H/transformers/src/transformers/models/paddleocr_vl/processing_paddleocr_vl.py
  X:/H/transformers/src/transformers/models/paddleocr_vl/image_processing_paddleocr_vl.py
  X:/H/transformers/src/transformers/models/paddleocr_vl/image_processing_pil_paddleocr_vl.py
  X:/H/transformers/tests/models/paddleocr_vl/test_modeling_paddleocr_vl.py
  X:/H/transformers/tests/models/paddleocr_vl/test_image_processing_paddleocr_vl.py

Any missing files or assumptions:
  modeling/config/processor/image_processor files are generated from modular_paddleocr_vl.py; future source edits should target the modular file.
  HF model cards and configs advertise PaddleOCR page-level parsing, JSON/Markdown export, and task prompts, but the native Transformers family implements image-text-to-text generation only. Page-level document parsing and structured JSON/Markdown postprocess are owned by the external PaddleOCR pipeline.
```

Primary DinoML runtime target: native Transformers `PaddleOCRVLForConditionalGeneration` for element-level OCR/table/chart/formula/seal/text-spotting image-to-text generation on CUDA. Page-level document parsing, crop orchestration, multi-page table merge, and JSON/Markdown writers should be treated as external pipeline work unless separately audited.

Primary sources:

- [PaddlePaddle/PaddleOCR-VL config](https://huggingface.co/PaddlePaddle/PaddleOCR-VL/blob/main/config.json)
- [PaddlePaddle/PaddleOCR-VL model card](https://huggingface.co/PaddlePaddle/PaddleOCR-VL)
- [PaddlePaddle/PaddleOCR-VL-1.5 config](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.5/blob/main/config.json)
- [PaddlePaddle/PaddleOCR-VL-1.5 model card](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.5)

## 2. High-level architecture

PaddleOCR-VL is a composite image-to-text decoder:

```text
CPU image resize/RGB/rescale/normalize/patch packing
-> NaViT-like variable-resolution vision encoder
-> 2x2 spatial patch merge projector
-> replace image placeholder token embeddings
-> ERNIE4.5-style causal decoder prefill/decode
-> LM logits
-> tokenizer decode / external OCR structured postprocess
```

Stage decomposition:

- CPU/data pipeline: image load, RGB conversion, bicubic resize to multiples of `patch_size * merge_size`, rescale/normalize, patch flattening, chat-template prompt construction, placeholder expansion, `mm_token_type_ids`.
- Cacheable vision/projector stage: `pixel_values` plus `image_grid_thw` produce one projected embedding row per merged 2x2 patch. These image embeddings are deterministic for an image and can be cached across prompt variants.
- Prefix construction: token embeddings are built from `input_ids`, image placeholders are replaced by projected image embeddings with a count guard, and 3D mRoPE position ids are computed from `mm_token_type_ids` and `image_grid_thw`.
- Prefill: causal decoder attends over text plus image tokens.
- Decode: image inputs are dropped after the first iteration when cache is used; generated tokens use cached `rope_deltas`.
- Postprocess: native Transformers returns decoded text. Structured JSON/Markdown output is outside this family.

Independently useful validation slices: image processor ABI, vision encoder on packed patches, projector merge, placeholder replacement, one decoder block with mRoPE, prefill logits, and cache decode.

## 3. Important config dimensions

Representative checkpoint/config sweep:

| Source | Text layers | Text hidden | Heads / KV heads / head dim | MLP | Vision layers | Vision hidden | Vision heads | Patch / merge | Processor pixels | dtype | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| Source defaults | 18 | 1024 | 16 / 2 / 128 | 3072 SwiGLU | 27 | 1152 | 16 | 14 / 2 | 147456 to 2359296 | config default | `tie_word_embeddings=True`, `use_cache=True` by class defaults |
| `PaddleOCR-VL` main | 18 | 1024 | 16 / 2 / 128 | 3072 SwiGLU | 27 | 1152 | 16 | 14 / 2 | 147384 to 2822400 | bf16 | `use_cache=false`, `tie_word_embeddings=false`, processor mean/std 0.5 |
| `PaddleOCR-VL-1.5` main | 18 | 1024 | 16 / 2 / 128 | 3072 SwiGLU | 27 | 1152 | 16 | 14 / 2 | 112896 to 1003520 | bf16 | Same model graph; tighter processor max pixels |
| Test tiny config | 2 | 32 | 4 / 2 / 128 | 32 | 2 | 144 | 4 | 14 / 2 | 28x28 fixture | fp32 random | Shape/unit test only; not representative performance |
| Historical `7a811607`/`f1e186d3` | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | Only minimal PaddleOCR pipeline `Global.model_name`; out of scope for native Transformers graph |

Operator-significant dimensions:

| Field | Value / behavior |
|---|---|
| `vocab_size` | 103424 |
| `max_position_embeddings` | 131072 |
| `rope_theta` | 500000 |
| `rope_scaling` | `{"mrope_section": [16, 24, 24], "rope_type": "default", "type": "default"}` |
| `image_token_id` | 100295, token `<\|IMAGE_PLACEHOLDER\|>` |
| `vision_start_token_id` / `vision_end_token_id` | 101305 / 101306 |
| `video_token_id` | Source class default says 100296; HF configs use 101307. Native PaddleOCR-VL raises for video paths, so DinoML should reject video for this family. |
| Text attention | Causal GQA: 16 Q heads, 2 KV heads, 8x repeat, `head_dim=128`; Q width 2048 while hidden is 1024. |
| Text MLP | gated `silu(gate_proj(x)) * up_proj(x)` then `down_proj`; no bias in HF configs. |
| Vision attention | Noncausal MHA over variable image chunks, 16 heads, head dim 72. |
| Vision patch input | Processor emits flattened patches `[sum(T*H*W), 3, 14, 14]`; model unsqueezes to `[1, sum_patches, 3, 14, 14]`. |
| Projector | LayerNorm(1152), merge 2x2 -> 4608, Linear 4608->4608 + GELU + Linear 4608->1024. |

## 3a. Family variation traps

- `hidden_size != num_attention_heads * head_dim` for the text decoder: `1024 != 16 * 128`. Q/O projection width is 2048, while K/V are 256.
- GQA is mandatory for HF configs: `num_key_value_heads=2`, `num_attention_heads=16`.
- HF checkpoint configs set `use_cache=false`, while integration examples call `generate(..., use_cache=True)` in the flash-attn path. DinoML should make cache an explicit runtime policy, not infer from config alone.
- Source class defaults and HF configs differ for `tie_word_embeddings`, `use_cache`, processor min/max pixels, image mean/std, and `video_token_id`.
- Native source has generation helper hooks for video-shaped keys inherited from Qwen, but PaddleOCR-VL explicitly does not support video. Reject `pixel_values_videos`, `video_grid_thw`, and `mm_token_type_ids==2`.
- The processor expands each image placeholder into `prod(image_grid_thw) // merge_size^2` image tokens. The model later enforces exact placeholder-feature count with `masked_scatter`.
- `pixel_values` has no batch dimension by sample. It is concatenated across images as `[total_patches, C, patch, patch]`, so batch slicing, beam expansion, and runtime batching need image-grid-aware split lengths.
- Vision encoder supports variable image lengths through `cu_seqlens`. FlashAttention uses varlen metadata; eager/SDPA path loops over image chunks and concatenates results.
- Layout translation is risky across the image processor and patch embedding. The processor already emits patch tensors in NCHW mini-images, and the first `Conv2d` is `kernel=stride=14` over each 14x14 patch. Treat the processor-to-vision boundary as a no-layout-translation guard unless the whole patch packing plus patch embedding region is owned.
- Tokenizer includes location tokens `<|LOC_0|>` through `<|LOC_1000|>`, location delimiters, crop row/column separators, image separators, and table cell tokens. These are text tokens, not numeric tensors in the model graph.

## 4. Operator coverage checklist

Tensor/layout ops:

- Dynamic split by `image_grid_thw.prod(dim=1).tolist()`.
- `reshape`, `view`, `transpose`, `permute`, `flatten`, `squeeze`, `unsqueeze`, `cat`, `concat`, `repeat`, `repeat_interleave`, `expand`.
- `arange`, `%`, `//`, `cumsum`, `pad`, `stack`, `where`/mask fill behavior.
- Variable-length batch packing: `cu_seqlens` int32 for FA2; per-image split fallback for non-FA2.

Neural primitives:

- Embedding lookup for 103424 vocab and position tables.
- Vision `Conv2d(3 -> 1152, kernel=14, stride=14, padding=valid)` applied to each packed 14x14 patch. Since each input is exactly one patch, this is equivalent to Linear(3*14*14 -> 1152) with bias under strict guards.
- LayerNorm for vision/projector, RMSNorm for text.
- Linear projections:
  - Text Q: `1024 -> 2048`, K/V: `1024 -> 256`, O: `2048 -> 1024`.
  - Text MLP: gate/up `1024 -> 3072`, down `3072 -> 1024`.
  - Vision Q/K/V/O: `1152 -> 1152`.
  - Vision MLP: `1152 -> 4304 -> 1152`.
  - Projector: `4608 -> 4608 -> 1024`.
  - LM head: `1024 -> 103424`.
- Activations: SiLU for text MLP, GELU tanh approximation for vision MLP and projector.
- `softmax(dtype=torch.float32).to(query.dtype)` in eager attention.

Attention primitives:

- Text causal GQA self-attention with mRoPE, cache update, optional SDPA/FA2/Flex backend.
- Vision noncausal self-attention over packed image chunks, with 2D rotary embeddings and varlen metadata.
- KV repeat expansion from 2 KV heads to 16 Q heads in eager path.

Position/rotary/custom math:

- Vision 2D RoPE over height/width ids.
- Text multimodal 3D RoPE using temporal/height/width `position_ids` and `mrope_section=[16,24,24]`.
- `rope_deltas` state carried from prefill into decode.

Preprocessing-coupled ops:

- Bicubic resize to multiple of `patch_size * merge_size` with min/max pixel guards.
- RGB conversion, rescale by `1/255`, normalize with checkpoint mean/std.
- Patch packing order:
  `[B,T,C,H,W] -> view(B,grid_t,tps,C,grid_h,14,grid_w,14) -> permute(0,1,4,6,3,2,5,7) -> [B, grid_t*grid_h*grid_w, C, 14, 14]`.

Scatter/indexed update ops:

- Source uses `inputs_embeds.masked_scatter(image_mask, image_embeds)`.
- Bounded lowering should reject general boolean scatter and lower only the processor-guaranteed pattern: ordered image placeholder positions in row-major text order, exact count `sum(t*h*w/merge^2)`, and `image_embeds` flattened in the same image order as `image_grid_thw`.

Structured output/tokenizer ops:

- End-to-end OCR/layout parity needs tokenizer decode plus optional downstream parsing of `<|LOC_*|>`, `<|LOC_BEGIN|>`, `<|LOC_END|>`, `<|LOC_SEP|>`, `<|CROP_COL_SEP|>`, `<|CROP_ROW_SEP|>`, `<ecel>`, `<fcel>`, `<xcel>`, `<lcel>`, `<ucel>`, and `<nl>`.
- Native Transformers source does not implement NMS, box scaling, polygon reconstruction, JSON export, or Markdown export.

## 5. Layer/block breakdown

Vision preprocessing and embeddings:

```text
raw image
-> RGB, resize, rescale, normalize
-> packed pixel_values [total_patches, 3, 14, 14]
-> model unsqueeze [1, total_patches, 3, 14, 14]
-> Conv2d patch_embedding over each 14x14 patch
-> [total_patches, 1152]
-> add interpolated 2D absolute position embedding per image/grid
```

Vision encoder, repeated 27 times:

```text
x = LayerNorm(x)
q,k,v = Linear(1152 -> 1152)
q,k = 2D RoPE(q,k)
x = residual + noncausal attention(q,k,v, cu_seqlens)
x = x + Linear(gelu_tanh(Linear(LayerNorm(x), 1152 -> 4304)), 4304 -> 1152)
```

Projector per image:

```text
x = split vision features by t*h*w
x = LayerNorm(x)
x = reshape(t, h/2, 2, w/2, 2, 1152)
x = transpose merge axes
x = reshape(t*h/2*w/2, 4608)
x = Linear(4608 -> 4608, bias=True)
x = GELU
x = Linear(4608 -> 1024, bias=True)
```

Multimodal prefix:

```text
inputs_embeds = token_embedding(input_ids)
image_embeds = projector(vision(pixel_values, image_grid_thw))
guard count(image_token_id) * hidden_size == image_embeds.numel()
inputs_embeds = masked_scatter(inputs_embeds, image_embeds)
position_ids = get_rope_index(input_ids, mm_token_type_ids, image_grid_thw, attention_mask)
```

Text decoder, repeated 18 times:

```text
res = x
x = RMSNorm(x)
q = Linear(1024 -> 2048, bias=False)
k = Linear(1024 -> 256, bias=False)
v = Linear(1024 -> 256, bias=False)
q,k = multimodal RoPE(q,k)
k,v = cache.update(k,v) when enabled
x = attention(q,k,v, causal_mask, GQA repeat 8)
x = res + Linear(2048 -> 1024, bias=False)
res = x
x = RMSNorm(x)
x = res + down_proj(silu(gate_proj(x)) * up_proj(x))
```

LM head:

```text
x = final RMSNorm(x)
logits = Linear(1024 -> vocab_size, bias=False), optionally sliced by logits_to_keep
```

## 6. Attention requirements

Text decoder attention:

- Causal self-attention.
- GQA: 16 query heads, 2 KV heads, 8 repeats.
- `head_dim=128`, Q width 2048, KV width 256, output projection input width 2048.
- Mask comes from `create_causal_mask`; text `position_ids` are passed for FA2 when 4-row generation position ids are used.
- RoPE is applied before cache update; cached keys are post-RoPE.
- Eager math order: matmul, multiply by `head_dim**-0.5`, add mask, fp32 softmax, cast to query dtype, dropout, matmul V.
- Decode cache is `DynamicCache`; native class can drop image inputs after first iteration when `use_cache=True`.
- Sliding window is in config plumbing but HF configs set `sliding_window=null`; no sliding-window layer types are present in inspected configs.

Vision attention:

- Noncausal self-attention, 16 heads, head dim 72.
- Runs on one concatenated sequence of all image patches but respects image boundaries.
- FA2 path passes `cu_seq_lens_q`, `cu_seq_lens_k`, `max_length_q`, `max_length_k`, `is_causal=False`.
- Non-FA2 path splits Q/K/V along sequence by `cu_seqlens` lengths, attends per image, and concatenates.
- No KV cache for vision.

Packed/varlen support:

- Required for efficient vision encoder parity. A fallback can loop per image, but production should expose varlen attention provider metadata and layout guards.

## 7. Position encoding and custom math

Vision absolute plus rotary positions:

- Absolute learned table has source default `image_size=384`, `patch_size=14`, so `num_positions=(384//14)^2`. Interpolation reshapes this table to square, bilinear interpolates to runtime `(grid_h, grid_w)`, then repeats over `t`.
- Vision RoPE computes `inv_freq` for `head_dim//2`, builds height/width ids per flattened image grid, indexes cos/sin by `[h_id,w_id]`, flattens to head dim, then repeats.

Text mRoPE:

```python
def multimodal_rope(q, k, cos, sin, mrope_section):
    sections = [s * 2 for s in mrope_section]
    cos = cat([chunk[i % 3] for i, chunk in enumerate(cos.split(sections, dim=-1))], dim=-1)
    sin = cat([chunk[i % 3] for i, chunk in enumerate(sin.split(sections, dim=-1))], dim=-1)
    q = q * cos[:, None, :, :] + rotate_half(q) * sin[:, None, :, :]
    k = k * cos[:, None, :, :] + rotate_half(k) * sin[:, None, :, :]
    return q, k
```

Position id construction:

- Text runs get identical temporal/height/width ids.
- Image runs get 3D ids from `image_grid_thw / spatial_merge_size`.
- After each image run, `current_pos += max(grid_h, grid_w) // spatial_merge_size`, not the number of image tokens.
- `rope_deltas = max(position_ids) + 1 - nonpad_sequence_length`; decode uses this delta to shift ordinary generated-token positions.

Precompute candidates:

- Text inv_freq can be precomputed per dtype/device.
- Vision `inv_freq` can be precomputed; per-image indexed cos/sin depends on `image_grid_thw`.
- Interpolated absolute vision position embeddings depend on runtime grid size and are worth caching by `(grid_h, grid_w, dtype)`.

## 8. Preprocessing and input packing

Image processor ABI:

| Tensor/key | Shape | Owner | Notes |
|---|---|---|---|
| `pixel_values` | `[sum_i(t_i*h_i*w_i), 3, 14, 14]` | image processor | dtype depends on `return_tensors`; model casts to vision dtype |
| `image_grid_thw` | `[num_images, 3]` | image processor | entries are patch grid `(t,h,w)` before 2x2 merge |
| `input_ids` | `[batch, seq]` | tokenizer/chat template | placeholders expanded before tokenization |
| `attention_mask` | `[batch, seq]` | tokenizer | left padding supported in integration test |
| `mm_token_type_ids` | `[batch, seq]` | processor | text=0, image=1; required for multimodal mRoPE |

Processor placeholder rule:

```text
for each image token in prompt:
  replace one image token string with image_token repeated prod(image_grid_thw[index]) // merge_size // merge_size
```

OCR/document specifics:

- The processor does not run OCR. It consumes images plus task prompt text such as `OCR:`, `Table Recognition:`, `Formula Recognition:`, `Chart Recognition:`, `spotting`, or `seal` prompts from model-card examples.
- No caller-supplied bounding boxes enter the native graph.
- Layout/location outputs are generated as text tokens, including `<|LOC_*|>` and delimiters. Coordinate normalization and structured parsing are not implemented in the Transformers family.
- Page-level document parsing and `save_to_json` / `save_to_markdown` are exposed by PaddleOCR `PaddleOCRVL` pipeline examples, not by the native model class.

Guards DinoML should enforce at the processor/model boundary:

- image height/width after resize must be divisible by `patch_size * merge_size`.
- aspect ratio must be <= 200, matching `smart_resize`.
- `grid_h` and `grid_w` must be divisible by `spatial_merge_size=2` for projector reshape.
- number of image placeholder tokens must equal `sum_i(t_i*h_i*w_i/(merge_size^2))`.
- image token runs and `mm_token_type_ids==1` must agree.
- `pixel_values.shape[0] == sum_i(t_i*h_i*w_i)`.

## 9. Graph rewrite / lowering opportunities

### Rewrite: packed Conv2d patch embedding -> Linear

Source pattern:

```text
pixel_values [N,3,14,14] -> Conv2d(3,1152,kernel=14,stride=14,padding=valid) -> flatten/squeeze
```

Replacement:

```text
reshape [N, 3*14*14] -> GEMM(weight_flat.T) -> bias
```

Preconditions:

- input patch height and width exactly equal `patch_size`.
- `kernel_size == stride == patch_size`, `padding=valid`, `dilation=1`, `groups=1`.
- source patch packing order is preserved as channel-first `[C, patch_h, patch_w]`.

Weight transform:

```python
w = conv.weight.reshape(out_channels, in_channels * patch_h * patch_w)
```

Failure cases: raw full images, different patch size, channel-last patch packing not rewritten with weight permutation.

Parity test sketch: compare Conv2d and Linear outputs for random `[N,3,14,14]` in fp32/bf16 tolerances.

### Rewrite: masked_scatter image stitch -> ordered indexed row copy

Source pattern:

```text
inputs_embeds.masked_scatter(input_ids == image_token_id, image_embeds)
```

Replacement:

```text
dst_rows = nonzero(input_ids == image_token_id) in row-major order
copy image_embeds rows into inputs_embeds[dst_rows]
```

Preconditions:

- mask is exactly `input_ids == image_token_id`.
- `mm_token_type_ids` marks the same rows as image.
- `num_dst_rows == image_embeds.shape[0]`.
- no video/audio placeholders admitted.

Failure cases: `inputs_embeds`-only mode with embedding-value comparison, arbitrary boolean masks, mixed image/video placeholders.

### Rewrite: projector 2x2 merge -> layout-aware reshape plus GEMM

Source pattern:

```text
[t,h,w,d] -> [t,h/2,2,w/2,2,d] -> transpose merge axes -> [t*h/2*w/2, 4*d] -> Linear/GELU/Linear
```

Preconditions:

- `h % 2 == 0` and `w % 2 == 0`.
- flattened vision feature order is processor order `(t,h,w)`.
- `spatial_merge_size == 2`.

Failure cases: odd grids, changed merge size, layout-translated vision sequence without corresponding axis rewrite.

### Rewrite: mRoPE generation as explicit shape artifact

Source pattern: Python loops over `input_type_group`, `image_grid_thw`, and attention mask.

Replacement: generate an explicit runtime shape/position kernel that writes `[3,B,S]` int64 position ids plus `[B,1]` rope deltas.

Preconditions: static modality token type values, no video, valid grid/token counts.

Failure cases: `inputs_embeds`-only prefill without input ids, stale `rope_deltas`, left-padding bugs.

### Layout guard: processor-to-vision patch region

Do not apply a generic NHWC conversion across the patch processor and patch embedding unless the whole patch packing transform and patch-embedding weight layout are rewritten together. Axis-sensitive ops include `_preprocess` `shape[-2:]`, patch reshape/permute, `Conv2d`, position interpolation `size=(h,w)`, projector reshape, and all `grid_h/grid_w` arithmetic.

## 10. Kernel fusion candidates

Highest priority:

- Text RMSNorm, causal GQA attention with mRoPE and KV cache, and SwiGLU MLP. These dominate decoder prefill/decode.
- Vision varlen noncausal attention with 2D RoPE and `cu_seqlens`. The eager split loop is likely too slow for multi-image batches.
- Placeholder stitch as ordered row copy. This avoids admitting general boolean scatter.
- Projector 2x2 merge + Linear/GELU/Linear, because it is shape-regular after grid guards and sits on the image prefix critical path.

Medium priority:

- Packed Conv2d patch embedding to GEMM/linear.
- Interpolated absolute position embedding cache by grid size.
- Last-token-only logits via `logits_to_keep=1` for decode.
- Processor patch packing acceleration if DinoML owns preprocessing on GPU later.

Lower priority:

- Beam expansion for packed visual tensors. Source tests skip beam search, and first integration can reject beams.
- Full structured output parser for JSON/Markdown. Useful product work, but not a neural graph blocker.
- Video/audio placeholder paths, because native PaddleOCR-VL rejects video despite inherited helper signatures.

## 11. Runtime staging plan

Stage 1: config and processor ABI

- Parse nested text/vision configs and HF flat text fields.
- Load tokenizer special token metadata needed for image placeholders and location tokens.
- Implement or wrap image processor to produce `pixel_values`, `image_grid_thw`, `input_ids`, `attention_mask`, and `mm_token_type_ids`.

Stage 2: vision/projector parity

- Lower packed patch embedding, absolute position interpolation, vision encoder, and projector.
- Start with eager per-image attention fallback; add varlen provider after correctness.

Stage 3: multimodal prefill

- Implement ordered image embedding stitch and mRoPE position id generation.
- Validate prefill logits for one image and one prompt.

Stage 4: decoder cache

- Implement GQA KV cache with post-RoPE key storage.
- Drop image inputs after first iteration and use `rope_deltas` for decode positions.

Stage 5: optimized kernels

- Add fused RMSNorm, SwiGLU, GQA FlashAttention, varlen vision attention, last-token logits, and projector fusion.

Stage 6: OCR task wrappers

- Add prompt/task presets for OCR/table/chart/formula/seal/spotting.
- Route page-level document parsing and JSON/Markdown to an external PaddleOCR-compatible pipeline or a separate audited implementation.

## 12. Parity and validation plan

- Processor ABI tests:
  - resize dimensions are multiples of 28 and respect min/max pixels.
  - `pixel_values` shape/order for one image and mixed-size batch.
  - placeholder expansion count equals `prod(grid)//4`.
- Custom math tests:
  - `get_vision_position_ids` for known small grids.
  - mRoPE `apply_multimodal_rotary_pos_emb` versus Transformers on random Q/K.
  - `rope_deltas` with left padding and one image run.
- Vision tests:
  - patch Conv2d-to-Linear rewrite parity.
  - one vision layer, then full 27-layer vision encoder on tiny/random config.
  - projector merge parity for even grids.
- Multimodal tests:
  - placeholder mismatch raises.
  - one-image prefill logits parity.
  - batch of same-size and different-size images.
- Decode tests:
  - prefill plus 1-token decode parity with `use_cache=True`.
  - generation with image inputs omitted after first step.
- End-to-end tests:
  - HF integration example expects decoded `"生甘草"` for the demo image with `max_new_tokens=30`.
  - OCR/table/formula/chart prompt smoke tests compare decoded strings against Transformers.

Suggested tolerances: fp32 `1e-4` for unit math, bf16/fp16 `1e-2` to `3e-2` for full model logits depending on attention backend and accumulation. Text decode parity should compare generated token ids under greedy deterministic generation.

## 13. Performance probes

- CPU preprocessing throughput by image size, min/max pixel settings, and batch composition.
- Vision encoder throughput versus total patches and number of image chunks.
- Varlen FA2 vision attention versus per-image split fallback.
- Projector throughput and memory bandwidth by grid size.
- Prefill throughput as a function of text length plus image-token count.
- Decode tokens/sec with and without last-token-only logits.
- KV cache memory: 18 layers * 2 tensors * 2 KV heads * sequence * 128 * dtype.
- Placeholder stitch cost for large image prefixes.
- End-to-end element-level OCR/table/formula/chart prompt latency, separated into processor, vision/projector, prefill, decode, and tokenizer decode.

## 14. Skip/defer list

- Training, labels/loss parity, gradient checkpointing.
- Beam search and assisted decoding; source tests explicitly skip them.
- Video and audio paths; reject for this family.
- Page-level document parsing, crop scheduling, multi-page merge, JSON/Markdown export.
- General boolean scatter; lower only the image-placeholder row-copy contract.
- Generic layout translation across image preprocessing and patch embedding.
- Full tokenizer-controlled structured parser for `<|LOC_*|>` unless the product target requires structured layout output.
- Tensor parallel / multi-GPU plans.

## 15. Final implementation checklist

- [ ] Parse `PaddleOCRVLConfig`, including flat HF text fields and nested `vision_config`.
- [ ] Load tokenizer image/location/table special tokens.
- [ ] Implement image processor ABI or require prepacked `pixel_values`/`image_grid_thw`.
- [ ] Add guards for resize factor, aspect ratio, grid divisibility, patch count, and placeholder count.
- [ ] Lower packed patch embedding Conv2d or rewrite to Linear under strict guards.
- [ ] Implement/interpose vision absolute position interpolation.
- [ ] Implement vision 2D RoPE and noncausal varlen attention.
- [ ] Implement projector 2x2 merge plus MLP.
- [ ] Implement ordered image embedding stitch.
- [ ] Implement mRoPE position ids and `rope_deltas`.
- [ ] Implement text GQA causal attention with post-RoPE KV cache.
- [ ] Implement RMSNorm, SwiGLU, and GELU-tanh coverage.
- [ ] Add prefill and decode parity against Transformers.
- [ ] Add greedy generation parity for demo OCR prompt.
- [ ] Reject video/audio, beam search, and page-level PaddleOCR pipeline features for first integration.
- [ ] Benchmark preprocessing, vision/projector, prefill, decode, and structured decode/postprocess separately.

## Gated gaps for DinoML

- `masked_scatter` must be gated to ordered image-placeholder row copy; do not add a general boolean scatter op for this family.
- Vision attention needs a varlen or per-image chunk contract with `cu_seqlens`; treating all packed patches as one dense image sequence would be incorrect.
- Text decoder dimensions require `hidden_size != q_width`; GEMM lowering must use explicit projection widths rather than deriving from hidden size.
- mRoPE is required for multimodal prompts and decode; plain 1D RoPE is only valid for pure text fallback.
- Processor/runtime must preserve `pixel_values` as concatenated patch rows plus `image_grid_thw`; ordinary batch slicing is invalid.
- Structured OCR/layout outputs are tokenizer/pipeline semantics, not model graph tensors. DinoML should either return decoded text first or separately admit a PaddleOCR-compatible postprocessor.
