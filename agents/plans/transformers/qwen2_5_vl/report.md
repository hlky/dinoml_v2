# Qwen2.5-VL Transformers audit

Primary target: `Qwen2_5_VLForConditionalGeneration` multimodal image/video + text causal generation on CUDA. This is a docs-only source/config audit; no DinoML runtime code was edited and no DinoML tests were run.

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model family: qwen2_5_vl
Primary task: multimodal causal LM prefill/decode/generation
Local source root: X:/H/transformers
```

Source files inspected:

- `src/transformers/models/qwen2_5_vl/configuration_qwen2_5_vl.py`
- `src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py`
- `src/transformers/models/qwen2_5_vl/modular_qwen2_5_vl.py`
- `src/transformers/models/qwen2_5_vl/processing_qwen2_5_vl.py`
- Shared processor sources used by Qwen2.5-VL configs: `src/transformers/models/qwen2_vl/image_processing_qwen2_vl.py`, `src/transformers/models/qwen2_vl/video_processing_qwen2_vl.py`
- Supporting source: `src/transformers/cache_utils.py`, `src/transformers/masking_utils.py`, `src/transformers/modeling_rope_utils.py`

Authoritative source note: `modeling_qwen2_5_vl.py` and `configuration_qwen2_5_vl.py` are generated from `modular_qwen2_5_vl.py`; future Transformers source edits should be checked in the modular file, while the generated modeling/config files are the concrete implementation audited here.

Representative official configs fetched from Hugging Face on 2026-05-13 and snapshotted under `_sources/`:

- `Qwen/Qwen2.5-VL-3B-Instruct`
- `Qwen/Qwen2.5-VL-7B-Instruct`
- `Qwen/Qwen2.5-VL-32B-Instruct`
- `Qwen/Qwen2.5-VL-72B-Instruct`

Any missing files or assumptions:

- `processor_config.json` returned 404 for all four inspected repos. Processor behavior is from `preprocessor_config.json`, `tokenizer_config.json`, `generation_config.json`, and local `processing_qwen2_5_vl.py`.
- No remote code is required for the audited in-library class.
- Official repos use `Qwen2VLImageProcessor` in `preprocessor_config.json`; Qwen2.5-VL has its own `Processor` and model code but reuses the Qwen2-VL image/video processors.
- The 32B `vision_config` uses a shorter/historical schema with fields such as `in_chans` and `spatial_patch_size` and omits `depth`, `num_heads`, `patch_size`, `spatial_merge_size`, `temporal_patch_size`, `window_size`, and `fullatt_block_indexes`. Current source fills these from `Qwen2_5_VLVisionConfig` defaults.
- `rope_scaling` may use either `{"type": "mrope"}` or already-standardized `{"rope_type": "default", "type": "default"}`. Current config conversion maps `mrope` to default RoPE while preserving `mrope_section`.

## 2. High-level architecture

Qwen2.5-VL is a multimodal vision encoder/projector plus Qwen2-style decoder-only LLM:

```text
image/video/text preprocessing
  -> packed visual patch rows + grid metadata + text placeholder expansion
  -> Qwen2.5 vision transformer with windowed/full attention mix
  -> spatial patch merger/projector to text hidden size
  -> masked scatter into text token embeddings
  -> 3D M-RoPE position-id construction
  -> causal text decoder prefill
  -> cached autoregressive decode
  -> lm_head logits / sampling
```

Stage decomposition:

- CPU/data-pipeline work: RGB conversion, resize, rescale, normalize, image/video patch packing, optional video frame sampling, placeholder-token expansion, tokenizer chat template, `mm_token_type_ids`, and `second_per_grid_ts`.
- Independently cacheable vision stage: `pixel_values` / `pixel_values_videos` and `*_grid_thw` feed the vision transformer. The output is a packed tuple/list of visual embeddings split per image or video item after spatial merge.
- Prefix construction: token embeddings are looked up, then image/video embeddings are inserted at `<|image_pad|>` and `<|video_pad|>` placeholders with `masked_scatter`.
- Position construction: M-RoPE position ids are computed from contiguous modality groups in `mm_token_type_ids`, image/video grids, attention mask, and video `second_per_grid_ts`.
- Prefill: the full multimodal prompt runs through the causal text decoder and creates per-layer GQA KV cache.
- Decode: subsequent cached generation drops visual tensors, uses cached `rope_deltas`, and projects only requested hidden positions through `lm_head`.

Heads implemented:

- Required: `Qwen2_5_VLForConditionalGeneration`, `Qwen2_5_VLModel`, `Qwen2_5_VLTextModel`, `Qwen2_5_VisionTransformerPretrainedModel`.
- Optional/deferred for first target: loss computation, hidden-state/attention returns, beam expansion of visual tensors, tensor/pipeline parallel plans.

## 3. Important config dimensions

Source defaults that matter when checkpoint configs omit fields:

| Field | Source default / behavior |
| --- | --- |
| vision `depth` | 32 |
| vision `hidden_size` | 3584 default, but official configs set/derive 1280 |
| vision `hidden_act` | `silu` |
| vision `intermediate_size` | 3420 default |
| vision `num_heads` | 16 |
| vision `in_channels` | 3 |
| vision patch / temporal / merge | `patch_size=14`, `temporal_patch_size=2`, `spatial_merge_size=2` |
| vision `tokens_per_second` | 4 default, official configs set 2 |
| vision `window_size` | 112 |
| vision full attention blocks | `[7, 15, 23, 31]` |
| text `vocab_size` | 152064 default |
| text `hidden_size` / layers | 8192 / 80 default |
| text `num_key_value_heads` | if `None`, becomes `num_attention_heads` |
| text `sliding_window` | set to `None` unless `use_sliding_window=True` |
| text layer types | full attention unless sliding window remains active |
| text RoPE | `rope_theta=1e6`; `mrope` config becomes default RoPE plus `mrope_section` |

Representative checkpoint sweep from `config.json`:

| Model id | dtype | text hidden | layers | Q heads | KV heads | head dim | MLP | vocab | max pos | tied | vision depth | vision hidden | vision MLP | vision heads | vision out | tokens/sec |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Qwen2.5-VL-3B-Instruct` | bf16 | 2048 | 36 | 16 | 2 | 128 | 11008 | 151936 | 128000 | true | 32 | 1280 | 3420 | 16 | 2048 | 2 |
| `Qwen2.5-VL-7B-Instruct` | bf16 | 3584 | 28 | 28 | 4 | 128 | 18944 | 152064 | 128000 | false | 32 | 1280 | 3420 | 16 | 3584 | 2 |
| `Qwen2.5-VL-32B-Instruct` | bf16 | 5120 | 64 | 40 | 8 | 128 | 27648 | 152064 | 128000 | false | default 32 | 1280 | 3456 | default 16 | 5120 | 2 |
| `Qwen2.5-VL-72B-Instruct` | bf16 | 8192 | 80 | 64 | 8 | 128 | 29568 | 152064 | 128000 | false | 32 | 1280 | 3456 | 16 | 8192 | 2 |

Shared config facts from official configs:

| Field | Value / behavior |
| --- | --- |
| text activation | `silu`; gated MLP `down(silu(gate) * up)` |
| text Q/K/V bias | true |
| text O/MLP/LM bias | false |
| text RMSNorm eps | `1e-6` in inspected configs |
| RoPE theta | `1000000.0` |
| M-RoPE section | `[16, 24, 24]`, doubled internally before cos/sin splitting |
| sliding window | raw `sliding_window=32768`, but `use_sliding_window=false`, so text sliding attention is inactive |
| vision patching | `patch_size=14`, `temporal_patch_size=2`, `spatial_merge_size=2` |
| vision attention | windowed attention in most layers, full attention in blocks 7/15/23/31 |
| image pixels | min 3136, max 12845056 |
| image/video normalization | OpenAI CLIP mean/std |
| tokenizer max length | 131072 in tokenizer config |
| special ids | `<|vision_start|>` 151652, `<|vision_end|>` 151653, `<|image_pad|>` 151655, `<|video_pad|>` 151656 |
| generation config | `do_sample=true`, `temperature=1e-6`, `repetition_penalty=1.05`, EOS `[151645, 151643]`, pad `151643` |

## 3a. Family variation traps

- Qwen2.5-VL is not just Qwen2-VL with larger text context. Compared with Qwen2-VL, the vision tower uses RMSNorm, a window-index reorder/reverse path, windowed attention for most vision blocks, and full attention only at configured block indexes.
- Text attention is GQA in all inspected production configs. Cache shape is KV heads, not query heads.
- Q/K/V projections are biased; O projection, text MLP projections, and `lm_head` are biasless.
- Vision QKV/proj/MLP and patch merger linears are biased; patch Conv3d is biasless.
- Vision config field names vary. DinoML should instantiate/evaluate the effective Transformers config rather than assuming raw JSON contains final field names.
- Processor output is flattened patch rows, not raw NCHW images. The model docstrings still mention image-shaped tensors in some places, but source expects packed rows.
- Images have `image_grid_thw` rows `[1, grid_h, grid_w]` and duplicate patch data across temporal patch size in the processor.
- Videos have `video_grid_thw` rows `[ceil_or_sampled_T / 2, grid_h, grid_w]`; preprocessing pads frame count by repeating the last frame until divisible by 2.
- Placeholder count must equal `prod(grid_thw) // spatial_merge_size**2` for each image/video.
- M-RoPE position generation requires `mm_token_type_ids` for multimodal prompts. Without it, source cannot compute correct 3D positions and falls back to language-model position inference only in limited cases.
- Video temporal M-RoPE spacing uses `time_interval = tokens_per_second * int(second_per_grid_t)`. The processor computes `second_per_grid_ts = temporal_patch_size / sampled_fps`, so the source truncates that value to int. This is source behavior and deserves parity tests.
- Text-model `position_ids` may be `[4, batch, seq]` in generation prep: row 0 is text-only positions for mask creation, rows 1:4 are temporal/height/width M-RoPE positions.
- Vision attention metadata has two `cu_seqlens` families: full frame-level `cu_seqlens` and window-level `cu_window_seqlens`.
- `window_index` reorders merged spatial groups before vision blocks; after `PatchMerger`, `argsort(window_index)` restores original merged-token order.
- Layout-sensitive regions need no-layout-translation guards: image/video patch packing, vision rotary position flattening, window indexing, patch merger `.view(-1, hidden * 4)`, placeholder scatter order, and M-RoPE group construction.

## 4. Operator coverage checklist

Tensor/layout ops:

- Embedding lookup `[vocab, hidden]`.
- View/reshape/flatten, transpose/permute, contiguous materialization.
- Split/cat by grid-derived lengths, `torch.split`, `torch.cat`.
- `repeat`, `repeat_interleave`, `cumsum`, `pad`, `unique_consecutive`.
- `arange`, `stack`, `argsort`, `roll`, boolean masks, comparisons.
- Dynamic indexing/gather/scatter for `window_index`, `reverse_indices`, and placeholder masks.
- `masked_scatter` into `[batch, seq, hidden]` embeddings.
- `logits_to_keep` integer or tensor slicing.
- dtype casts: processor float to vision dtype; RoPE/vision RoPE fp32 math then cast back.

Neural network primitives:

- Text RMSNorm over last dim with fp32 variance.
- Vision RMSNorm over last dim in blocks and patch merger.
- Linear with bias:
  - 7B text Q `3584 -> 3584`, K/V `3584 -> 512`.
  - Vision QKV `1280 -> 3840`, vision proj `1280 -> 1280`.
  - Vision MLP 7B/3B `1280 -> 3420 -> 1280`; 32B/72B `1280 -> 3456 -> 1280`.
  - PatchMerger 7B `5120 -> 5120 -> 3584`; 72B `5120 -> 5120 -> 8192`.
- Linear without bias:
  - Text O `num_heads * head_dim -> hidden`.
  - Text gate/up `hidden -> intermediate`, down `intermediate -> hidden`.
  - LM head `hidden -> vocab`.
- Activations: SiLU, GELU.
- Conv3d patch embed: `in=3`, `out=1280`, `kernel=stride=(2,14,14)`, no bias.

Attention primitives:

- Text causal GQA self-attention, full attention for inspected configs.
- Text optional sliding attention path exists in source but is inactive for inspected official configs.
- Vision noncausal self-attention over packed patch tokens.
- Vision window attention with varlen `cu_window_seqlens` for most layers.
- Vision full attention with frame-level `cu_seqlens` for layers 7/15/23/31.
- Backend dispatch through `ALL_ATTENTION_FUNCTIONS`, including eager/SDPA/FlashAttention-compatible paths.

Position/rotary/custom math:

- Text M-RoPE cos/sin generation over `[3, batch, seq]` position ids.
- `apply_multimodal_rotary_pos_emb` splitting final head dimension by doubled `mrope_section`.
- Vision 2D RoPE gathered from height/width positions in spatial-merge order.
- `rope_deltas` computation and reuse during cached decode.

Generation/cache ops:

- DynamicCache-compatible per-layer K/V update after M-RoPE.
- Cache-aware generation input slicing via `GenerationMixin`.
- Drop visual tensors after first cached iteration.
- Beam expansion helpers for visual tensors by per-sample image/video counts.

Preprocessing-coupled ops:

- Smart resize to multiples of `patch_size * merge_size = 28`.
- Image/video rescale and CLIP normalize.
- Image/video patch flatten order.
- Optional video sampling and frame padding.
- Placeholder expansion and `mm_token_type_ids` construction.
- `second_per_grid_ts` construction from sampled fps.

Packed/varlen metadata ops:

- `image_grid_thw`, `video_grid_thw`: integer `[num_items, 3]`.
- Full vision `cu_seqlens`: cumulative `grid_h * grid_w` repeated `grid_t`.
- Window `cu_window_seqlens`: cumulative per-window merged lengths multiplied by `spatial_merge_unit`.
- Split sizes after merger: `prod(grid_thw) // 4`.

Distributed/tensor-parallel ops:

- Source declares TP/PP plans for text projections and embeddings. This is optional/deferred for first single-GPU DinoML integration.

## 5. Layer/block breakdown

Image preprocessing to model input:

```text
image [C,H,W]
  -> resize H,W to multiples of 28 within min/max pixels
  -> rescale/normalize
  -> reshape [B,C,grid_h/2,2,14,grid_w/2,2,14]
  -> permute [B,grid_h/2,grid_w/2,2,2,C,14,14]
  -> duplicate along temporal_patch_size=2
  -> pixel_values [sum_images, grid_h*grid_w, 3*2*14*14]
  -> image_grid_thw rows [1, grid_h, grid_w]
```

Video preprocessing to model input:

```text
video [B,T,C,H,W] in processor convention
  -> optional uniform frame sampling
  -> resize frames to multiples of 28
  -> rescale/normalize
  -> repeat last frame so T % 2 == 0
  -> reshape/permute temporal + spatial patch groups
  -> pixel_values_videos [sum_videos, grid_t*grid_h*grid_w, 1176]
  -> video_grid_thw rows [T/2, grid_h, grid_w]
  -> second_per_grid_ts = temporal_patch_size / sampled_fps
```

Vision patch embed:

```text
x: [total_patch_rows, 1176]
x.view(-1, 3, 2, 14, 14)
Conv3d(3 -> 1280, kernel=stride=(2,14,14), bias=False)
x.view(-1, 1280)
```

Vision position/window preparation:

```text
rotary_pos_emb = rot_pos_emb(grid_thw)                  # [seq, head_dim]
window_index, cu_window_seqlens = get_window_index(...)
x = x.reshape(seq/4, 4, hidden)[window_index].reshape(seq, hidden)
rotary_pos_emb = rotary_pos_emb.reshape(seq/4, 4, -1)[window_index].reshape(seq, -1)
position_embeddings = (cat(rotary, rotary).cos(), cat(rotary, rotary).sin())
cu_seqlens = pad(cumsum(repeat_interleave(grid_h * grid_w, grid_t)), left=0)
```

Vision block, repeated 32 times:

```text
if layer in [7,15,23,31]:
    cu = full frame-level cu_seqlens
else:
    cu = window-level cu_window_seqlens

res = x
x = RMSNorm(1280)(x)
qkv = Linear(1280 -> 3840, bias=True)(x)
q,k,v = reshape to [seq, 16, 80]
q,k = vision_2d_rope(q,k, position_embeddings)
x = res + Linear(1280 -> 1280, bias=True)(noncausal varlen attention(q,k,v,cu))
res = x
x = res + Linear(intermediate -> 1280, bias=True)(silu(gate(x_norm)) * up(x_norm))
```

Patch merger/projector:

```text
x = RMSNorm(1280)(x)
x.view(-1, 1280 * 2 * 2)       # [merged_tokens, 5120]
x = Linear(5120 -> 5120, bias=True)
x = GELU(x)
x = Linear(5120 -> text_hidden, bias=True)
x = x[argsort(window_index), :]
```

Multimodal stitch:

```text
inputs_embeds = Embedding(input_ids)
image_embeds = cat(split(vision.pooler_output, prod(image_grid_thw)/4))
image_mask = (input_ids == image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
repeat for video_token_id
position_ids = compute_3d_position_ids(...)
```

Text decoder block:

```text
res = x
x_norm = RMSNorm(hidden)(x)
q = Linear(hidden -> num_heads*head_dim, bias=True)(x_norm)
k = Linear(hidden -> kv_heads*head_dim, bias=True)(x_norm)
v = Linear(hidden -> kv_heads*head_dim, bias=True)(x_norm)
q,k = M-RoPE(q,k, cos, sin, mrope_section)
k,v = cache.update(k,v,layer_idx) if cache is present
attn = causal GQA attention(q,k,v,mask)
x = res + Linear(num_heads*head_dim -> hidden, bias=False)(attn)
res = x
x_norm = RMSNorm(hidden)(x)
x = res + down_proj(silu(gate_proj(x_norm)) * up_proj(x_norm))
```

Final path:

```text
x = final RMSNorm(hidden)(x)
logits = lm_head(x[:, slice_indices, :])
```

## 6. Attention requirements

Text attention:

- Type: causal self-attention.
- Variant: GQA in all inspected configs.
- 7B example: Q heads 28, KV heads 4, head dim 128.
- 72B example: Q heads 64, KV heads 8, head dim 128.
- Masking: `create_causal_mask`; `create_sliding_window_causal_mask` only if effective config has sliding layers.
- Sliding attention: source supports it through `layer_types`, but inspected official configs set `use_sliding_window=false`, so `sliding_window` becomes `None`.
- Cache layout before logical GQA repeat: K/V `[batch, num_key_value_heads, cache_seq, head_dim]`.
- Cached keys are stored after M-RoPE; values are unrotated.
- Eager math order: repeat KV, matmul, multiply by `head_dim**-0.5`, add mask, softmax in fp32, cast to query dtype, dropout, matmul with V.
- Optimized path should avoid physical KV repeat and accept GQA directly.

Vision attention:

- Type: noncausal self-attention.
- Heads: 16 heads, head dim 80 for `hidden_size=1280`.
- QKV packed projection: `Linear(1280 -> 3840, bias=True)` split in Q/K/V order.
- Source reshapes Q/K/V to `[1, heads, total_seq, head_dim]`.
- Full-attention blocks use `cu_seqlens` made from per-frame lengths `grid_h * grid_w`.
- Other blocks use `cu_window_seqlens`, after `window_index` reorders merged spatial groups into local windows.
- FlashAttention-requested path passes `cu_seq_lens_q/k`, `max_length_q/k`, and `is_causal=False`.
- Non-Flash path splits Q/K/V by lengths and loops over chunks, then concatenates outputs.

FlashAttention/SDPA compatibility:

- Text and vision both dispatch through `ALL_ATTENTION_FUNCTIONS`.
- For text, packed-mask construction depends on text-only position ids when generation passes `[4, batch, seq]` positions.
- For vision, `cu_seqlens` dtype is int32 unless tracing, where it follows `grid_thw.dtype`.

## 7. Position encoding and custom math

Text M-RoPE cos/sin generation:

```python
inv = 1.0 / (rope_theta ** (arange(0, head_dim, 2) / head_dim))
freqs = (inv[None, None, :, None].expand(3, batch, -1, 1)
         @ position_ids[:, :, None, :].float()).transpose(2, 3)
emb = cat([freqs, freqs], dim=-1)
cos, sin = emb.cos() * attention_scaling, emb.sin() * attention_scaling
```

M-RoPE application:

```python
def apply_qwen25_vl_mrope(q, k, cos, sin, section):
    section = section * 2
    cos = cat([part[i % 3] for i, part in enumerate(cos.split(section, -1))], -1).unsqueeze(1)
    sin = cat([part[i % 3] for i, part in enumerate(sin.split(section, -1))], -1).unsqueeze(1)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin
```

3D vision/text position IDs:

```python
text run: positions = arange(text_len) + current_pos, repeated for T/H/W
image run: time_interval = 1
video run: time_interval = tokens_per_second * int(next(second_per_grid_ts))
vision temporal = arange(grid_t) * time_interval + current_pos
vision height = arange(grid_h/merge) + current_pos
vision width = arange(grid_w/merge) + current_pos
current_pos += max(grid_h, grid_w) // merge
rope_delta = max(position_ids) + 1 - unpadded_seq_len
```

Vision 2D RoPE:

```python
hpos = arange(h).reshape(h//2,2,w//2,2).permute(0,2,1,3).flatten()
wpos = arange(w).reshape(h,w).reshape(h//2,2,w//2,2).permute(0,2,1,3).flatten()
pos_ids = stack([hpos, wpos], -1).repeat(t, 1)
rotary = rotary_pos_emb(max(h, w))[pos_ids].flatten(1)
emb = cat([rotary, rotary], -1)
cos, sin = emb.cos(), emb.sin()
```

Precomputable:

- Text inverse frequencies are fixed for default RoPE; cos/sin tables can be precomputed for admitted positions, but multimodal position ids and `rope_deltas` are prompt-dependent.
- Vision rotary frequency tables can be cached per max grid size; gathered position IDs depend on exact grid and merge order.
- `window_index` and `cu_window_seqlens` are grid-dependent and can be cached with visual embeddings for repeated prompts.

## 8. Preprocessing and input packing

Processor runtime tensors:

| Tensor | Shape | Dtype | Producer | Meaning |
| --- | --- | --- | --- | --- |
| `input_ids` | `[batch, text_seq]` | int64 | tokenizer | Text plus expanded image/video placeholders |
| `attention_mask` | `[batch, text_seq]` | int/bool | tokenizer | Padding mask |
| `mm_token_type_ids` | `[batch, text_seq]` | int | tokenizer helper | 0 text, 1 image, 2 video |
| `pixel_values` | `[total_image_patch_rows, 1176]` after concatenation | float | image processor | Packed flattened image patch rows |
| `image_grid_thw` | `[num_images, 3]` | long | image processor | `[1, grid_h, grid_w]` |
| `pixel_values_videos` | `[total_video_patch_rows, 1176]` | float | video processor | Packed flattened video patch rows |
| `video_grid_thw` | `[num_videos, 3]` | long | video processor | `[grid_t, grid_h, grid_w]` |
| `second_per_grid_ts` | `[num_videos]` list/tensor | float-ish | processor | temporal patch seconds per video grid step |

Placeholder expansion:

- The chat template emits one `<|image_pad|>` or `<|video_pad|>` within `<|vision_start|>...<|vision_end|>`.
- The processor replaces each pad-token string with `prod(grid_thw) // merge_size**2` copies before tokenization.
- The tokenizer helper emits `mm_token_type_ids` by recognizing these special multimodal token spans.
- The model validates placeholder element count against visual feature element count before `masked_scatter`.

CPU/data-pipeline versus GPU/runtime:

- First DinoML integration can externalize image/video resize, normalization, patch packing, tokenizer/chat template, and placeholder expansion to Transformers.
- Runtime graph still needs packed visual tensors, grid metadata, modality token types, placeholder scatter, M-RoPE position IDs or supplied precomputed positions, and `second_per_grid_ts` for video parity.
- Vision encoder/projector outputs can be cached independently from decoder KV cache when visual input and grid metadata are unchanged.

Generation-controller behavior:

- `prepare_inputs_for_generation` drops `pixel_values` and `pixel_values_videos` after the first cached iteration.
- `_prepare_position_ids_for_generation` returns `[4, batch, seq]` positions for prefill when multimodal grids are present.
- Beam expansion is source-specific for visual tensors because their leading dimension is total visual tokens/items, not batch.
- Generation config uses EOS `[151645, 151643]`, pad `151643`, and almost-greedy sampling via `temperature=1e-6`.

## 9. Graph rewrite / lowering opportunities

### Rewrite: packed Conv3d patch embed -> Linear

Source pattern:

```text
x.view(-1, C, T_patch, H_patch, W_patch)
Conv3d(C -> E, kernel=stride=(T_patch,H_patch,W_patch), bias=False)
view(-1, E)
```

Replacement:

```text
MatMul(packed_patch_rows, weight_flat.T)
```

Preconditions:

- Input is exactly processor-packed rows with last dim `3 * 2 * 14 * 14 = 1176`.
- Conv3d kernel and stride equal patch volume.
- Padding=0, dilation=1, groups=1, bias=false.
- Processor flatten order is preserved.

Weight transform:

```python
w = conv.weight.reshape(embed_dim, 3 * temporal_patch_size * patch_size * patch_size)
```

Failure cases: raw image/video tensors, changed patch sizes, nonzero bias, different flatten order.

Parity test sketch: compare source Conv3d path to Linear rewrite on random packed rows and checkpoint patch weights for fp32/bf16.

### Rewrite: vision window attention metadata lowering

Source pattern:

```text
reshape(seq/4,4,H) -> gather by window_index -> attention with cu_window_seqlens -> merger -> gather by argsort(window_index)
```

Replacement:

```text
precompute window_index/cu_window_seqlens per grid, run varlen noncausal attention per window, reverse after merger
```

Preconditions:

- `spatial_merge_size=2`, `window_size=112`, `patch_size=14`.
- `vit_merger_window_size = window_size // spatial_merge_size // patch_size`.
- Token order matches source processor and `rot_pos_emb`.

Failure cases: layout pass changes token order; grids not divisible by merge; different window config.

Parity test sketch: synthetic grids with non-square sizes and padding windows; compare `window_index`, `cu_window_seqlens`, and post-merger order.

### Rewrite: Qwen gated MLP -> fused SwiGLU

Source pattern:

```text
down_proj(silu(gate_proj(x)) * up_proj(x))
```

Replacement:

```text
DualLinear or packed GEMM -> split gate/up -> fused SiLU*multiply -> down GEMM
```

Preconditions:

- Text MLP projections are no-bias and share input.
- Vision MLP projections are biased; bias handling must be included for the vision variant.
- Activation is exactly `silu`.

Parity test sketch: one text MLP and one vision MLP with random inputs and checkpoint weights.

### Rewrite: text GQA attention with M-RoPE/cache

Source pattern:

```text
Q/K/V linears -> M-RoPE(Q,K) -> cache.update(K,V) -> causal GQA attention -> O linear
```

Replacement:

```text
native GQA prefill/decode attention accepting KV heads, M-RoPE-rotated K cache, and causal/sliding masks
```

Preconditions:

- Inference dropout is zero.
- Cache stores post-RoPE K.
- Sliding attention is either inactive or explicitly supported.
- M-RoPE section and position ids match source.

Failure cases: unsupported packed-mask path, wrong `[4,batch,seq]` handling, physical KV repeat in hot path.

### Rewrite: placeholder masked_scatter -> indexed copy

Source pattern:

```text
mask = (input_ids == image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
inputs_embeds = inputs_embeds.masked_scatter(mask, image_embeds)
```

Replacement:

```text
find placeholder token offsets -> copy visual rows into embedding rows
```

Preconditions:

- Placeholder count equals visual feature row count per modality.
- Source flattened order of visual embeddings is preserved.
- Embeddings are dense `[batch, seq, hidden]`.

Parity test sketch: mismatched count failure plus mixed image/video prompt with multiple items per sample.

### Layout pass guard regions

Safe regions:

- Text decoder linears/norms/activations on `[batch, seq, hidden]`.
- Vision block linears/norms after patch embedding on `[seq, 1280]`.

No-layout-translation guards:

- Processor patch-packing reshapes and permutes.
- Vision `rot_pos_emb`, `get_window_index`, and `argsort(window_index)` restore.
- PatchMerger `.view(-1, hidden * 4)`.
- Placeholder expansion/scatter.
- M-RoPE grouped modality position construction.

Axis-sensitive attrs:

- RMSNorm reductions are last-dim.
- Attention softmax is key dimension `-1`.
- Image processor permute is `(0,2,5,3,6,1,4,7)`.
- Video processor permute is `(0,1,4,7,5,8,3,2,6,9)`.

## 10. Kernel fusion candidates

Highest priority:

- Text RMSNorm: two per decoder layer plus final norm; decode hot path.
- GQA FlashAttention/paged attention with post-M-RoPE K cache and KV heads < Q heads.
- M-RoPE generation/application fused with Q/K preparation where possible.
- Text SwiGLU MLP fusion and packed gate/up projection.
- Packed Conv3d patch embed lowered to GEMM.
- Placeholder indexed-copy/scatter kernel for multimodal stitch.

Medium priority:

- Vision varlen attention with both window-level and full-frame `cu_seqlens`.
- Vision RMSNorm + QKV projection fusion.
- Vision window reorder/attention/reverse scheduling.
- PatchMerger RMSNorm/view/GEMM/GELU/GEMM.
- Last-token-only logits via `logits_to_keep`.
- Grid/window/rotary metadata caching for repeated visual inputs.

Lower priority:

- Beam-search visual tensor expansion.
- Full GPU resize/rescale/normalize/patch packing.
- Active text sliding-window attention, because inspected configs disable it.
- Tensor/pipeline parallel execution.

## 11. Runtime staging plan

Stage 1: config and weights.

- Parse flat Qwen2.5-VL configs into text/vision subconfigs and apply effective defaults.
- Handle historical vision fields and omitted source defaults.
- Load embeddings, vision tower, patch merger, text decoder, and LM head, including tied 3B embeddings.

Stage 2: vision encoder/projector parity.

- Accept external processor outputs: `pixel_values`, `image_grid_thw`.
- Implement patch embed, vision 2D RoPE, window indexing, window/full varlen attention, RMSNorm MLP blocks, PatchMerger, and reverse index restore.

Stage 3: multimodal prefix construction.

- Accept `input_ids`, `attention_mask`, `mm_token_type_ids`, visual features/grids.
- Implement placeholder count checks and indexed scatter.
- Accept precomputed `position_ids` initially, then implement `get_rope_index`.

Stage 4: text prefill parity.

- Run full multimodal prompt through text decoder.
- Implement default M-RoPE, causal GQA, and `logits_to_keep`.

Stage 5: cached decode.

- Implement per-layer KV cache storing post-RoPE K.
- Reproduce `rope_deltas` handling and visual-input dropping after prefill.

Stage 6: video parity.

- Add `pixel_values_videos`, `video_grid_thw`, `second_per_grid_ts`, video placeholder scatter, and temporal M-RoPE spacing.

Stage 7: optimized kernels.

- Add native GQA attention, varlen vision attention, fused MLP/norm, patch embed GEMM, and production batching.

Initial stubs:

- Tokenizer/chat template and image/video processors can stay external.
- Position IDs can be explicit runtime inputs before DinoML reproduces the Python grouping logic.
- Beam search and TP/PP can be deferred.

## 12. Parity and validation plan

Random tensor tests:

- RMSNorm for text/vision hidden sizes 1280, 2048, 3584, 5120, 8192.
- Text M-RoPE with text-only, image, and video position ids.
- Vision 2D RoPE gather/order for several `grid_thw` values.
- `get_window_index` and `cu_window_seqlens`, including padded windows.
- Packed Conv3d-to-Linear patch embed rewrite.
- Placeholder indexed scatter with mismatch failure cases.
- Video temporal position spacing with fractional `second_per_grid_ts` to confirm source truncation behavior.

Single-layer parity:

- One vision block using window attention and one full-attention block.
- PatchMerger with token counts divisible by 4.
- One text decoder block without cache.
- One text decoder block prefill + one-token decode with cache.

After-N-layer parity:

- Vision tower after 1, 8, and 32 blocks on a small synthetic grid.
- Text decoder after 1, 4, and full layers for short prompts.

End-to-end parity:

- One-image prompt with external Transformers processor.
- Multi-image prompt to test split sizes and placeholder ordering.
- One-video prompt after video path lands.
- Prefill logits and greedy first token.
- Several-token cached decode verifying `rope_deltas` and cache length.

Tolerance guidance:

- fp32 isolated ops: `rtol=1e-5`, `atol=1e-6`.
- bf16/fp16 graph parity: start `rtol=3e-2`, `atol=3e-2` for logits; tighten per op.
- Attention parity should compare Q/K after RoPE, attention outputs, and logits to localize errors.

## 13. Performance probes

- Processor throughput: image resize/patch packing images/sec and video frames/sec.
- Vision encoder throughput: patch rows/sec and merged tokens/sec by grid size.
- Windowed versus full vision attention time by layer and grid.
- PatchMerger throughput by text hidden size.
- Multimodal scatter cost by visual token count and hidden size.
- Prefill-only throughput by text length and visual token count.
- Decode-only tokens/sec by batch size and cache length.
- KV cache memory: `layers * 2 * batch * kv_heads * seq * head_dim * dtype_bytes`.
- LM head cost: full logits versus `logits_to_keep=1`.
- End-to-end request split: processor, vision, stitch/position, prefill, decode, logits.

No benchmark observations are included; these are proposed probes from source-derived bottlenecks.

## 14. Skip/defer list

- Training and loss parity beyond smoke tests.
- Gradient checkpointing.
- Hidden-state/attention output materialization.
- Tensor parallel and pipeline parallel runtime.
- Beam search visual-tensor expansion for first greedy integration.
- Full tokenizer/chat-template implementation inside DinoML.
- CPU/GPU image/video resize kernels; keep preprocessing external initially.
- Quantized checkpoint ingestion.
- Active text sliding-window attention until a target config enables it.
- Speculative decoding.
- Remote-code or historical config behavior not consumed by pinned in-library source.

## 15. Final implementation checklist

- [ ] Parse Qwen2.5-VL config and apply effective text/vision defaults.
- [ ] Normalize `rope_scaling` / `rope_parameters` and preserve `mrope_section`.
- [ ] Load tied and untied embedding/LM-head variants.
- [ ] Support processor-produced `pixel_values`, `pixel_values_videos`, `image_grid_thw`, `video_grid_thw`, `mm_token_type_ids`, and `second_per_grid_ts`.
- [ ] Implement packed patch-row Conv3d-to-Linear parity.
- [ ] Implement vision 2D RoPE and grid/window metadata.
- [ ] Implement vision windowed/full varlen noncausal attention.
- [ ] Implement Qwen2.5 vision RMSNorm MLP blocks.
- [ ] Implement PatchMerger and reverse window-index restore.
- [ ] Implement placeholder count validation and indexed scatter.
- [ ] Implement M-RoPE position-id generation or accept explicit precomputed `position_ids`.
- [ ] Implement text RMSNorm, biased Q/K/V, biasless O, GQA attention, and post-RoPE KV cache.
- [ ] Implement Qwen2 SwiGLU text MLP.
- [ ] Implement final RMSNorm and `logits_to_keep` LM head.
- [ ] Add single-op tests for M-RoPE, vision RoPE, window index, patch embed rewrite, PatchMerger, and scatter.
- [ ] Add one-block and full-vision parity tests.
- [ ] Add text prefill and one-token decode parity tests.
- [ ] Add one-image end-to-end greedy decode parity with external Transformers processor.
- [ ] Add video parity after image path is stable.
- [ ] Benchmark processor, vision, stitch, prefill, decode, LM head, and cache memory separately.
