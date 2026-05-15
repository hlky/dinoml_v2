# Qwen2-VL Transformers audit

## 1. Source basis

Transformers commit/version:

- Local source checkout: `transformers`, commit `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`.
- Model implementation inspected at that commit.
- Representative checkpoint configs fetched from official Hugging Face repos on 2026-05-13.

Model id:

- Primary common target: `Qwen/Qwen2-VL-7B-Instruct`.
- Sweep configs: `Qwen/Qwen2-VL-2B-Instruct`, `Qwen/Qwen2-VL-7B-Instruct`, `Qwen/Qwen2-VL-72B-Instruct`.

Config source:

- `https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct/resolve/main/config.json`
- `https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct/resolve/main/config.json`
- `https://huggingface.co/Qwen/Qwen2-VL-72B-Instruct/resolve/main/config.json`
- `preprocessor_config.json`, `tokenizer_config.json`, and `generation_config.json` from the same repos.

Source files inspected:

- `transformers/src/transformers/models/qwen2_vl/configuration_qwen2_vl.py`
- `transformers/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py`
- `transformers/src/transformers/models/qwen2_vl/processing_qwen2_vl.py`
- `transformers/src/transformers/models/qwen2_vl/image_processing_qwen2_vl.py`
- `transformers/src/transformers/models/qwen2_vl/video_processing_qwen2_vl.py`
- `transformers/src/transformers/models/qwen2_vl/image_processing_pil_qwen2_vl.py` was present but not deeply inspected because the torch/torchvision image processor is the processor path used by current `Qwen2VLProcessor`.

Any missing files or assumptions:

- Official repos do not contain `processor_config.json`; fetch returned 404 for all three inspected checkpoints. Processor behavior comes from `preprocessor_config.json`, `tokenizer_config.json`, and local `processing_qwen2_vl.py`.
- `config.json` files use the older flat Qwen2-VL schema (`hidden_size`, `num_hidden_layers`, etc. at top level) plus `vision_config`; current source wraps those fields into `Qwen2VLTextConfig` during `Qwen2VLConfig.__post_init__`.
- Hub configs use `rope_scaling: {"type": "mrope", "mrope_section": [...]}`. Current source converts this into `rope_parameters`, maps `rope_type == "mrope"` to `"default"`, and keeps `mrope_section`.
- The source comments say some modeling code is copied from Llama/Qwen2; this file is still authoritative for Qwen2-VL behavior.
- Primary runtime target for this report: multimodal image/video + text causal generation through `Qwen2VLForConditionalGeneration`. Training loss, gradients, and beam-search-only expansion behavior are secondary.

## 2. High-level architecture

Qwen2-VL is a multimodal projector plus decoder-only LLM:

```text
image/video/text preprocessing
  -> packed visual patch rows + grid metadata + tokenized text placeholders
  -> vision transformer over packed patch sequences
  -> patch merger/projector to text hidden size
  -> masked scatter into text token embeddings
  -> Qwen2-style causal decoder prefill
  -> cached autoregressive decode
  -> lm_head logits / sampling
```

Stage decomposition:

- CPU/data-pipeline work: image/video loading, RGB conversion, resize, rescale, normalize, temporal frame sampling/padding for videos, patch flattening, text chat template, placeholder-token expansion, `mm_token_type_ids`.
- Independently cacheable vision stage: `pixel_values` or `pixel_values_videos` plus `*_grid_thw` feed the vision transformer. Its output is a packed list of visual token embeddings after spatial merge. This can be validated independently from text decode.
- Prefix construction: text embeddings are looked up, then image/video features are inserted with `masked_scatter` at `<|image_pad|>` and `<|video_pad|>` placeholders. M-RoPE position ids are derived from `input_ids`, `mm_token_type_ids`, and image/video grids.
- Prefill: full multimodal token sequence runs through the causal decoder, with GQA KV cache creation.
- Decode: subsequent iterations omit `pixel_values` and `pixel_values_videos`; stored `rope_deltas` adjust text position ids for continued generation.

Heads implemented:

- Required for target: `Qwen2VLForConditionalGeneration` with `lm_head`.
- Required internal modules: `Qwen2VLModel`, `Qwen2VLTextModel`, `Qwen2VisionTransformerPretrainedModel`.
- Optional/deferred: hidden-state/attention outputs, loss computation for labels, gradient checkpointing, tensor parallel plans, beam expansion of visual tensors.

## 3. Important config dimensions

Representative checkpoint sweep:

| Config fact source | Model | dtype | text hidden | layers | Q heads | KV heads | head dim | MLP intermediate | vocab | max pos | tie embeddings | vision depth | vision embed | vision heads | projector out |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `config.json` | Qwen2-VL-2B-Instruct | bf16 | 1536 | 28 | 12 | 2 | 128 | 8960 | 151936 | 32768 | true | 32 | 1280 | 16 | 1536 |
| `config.json` | Qwen2-VL-7B-Instruct | bf16 | 3584 | 28 | 28 | 4 | 128 | 18944 | 152064 | 32768 | false | 32 | 1280 | 16 | 3584 |
| `config.json` | Qwen2-VL-72B-Instruct | bf16 | 8192 | 80 | 64 | 8 | 128 | 29568 | 152064 | 32768 | false | 32 | 1280 | 16 | 8192 |

Shared source/config fields:

| Field | Value / behavior | Source |
|---|---|---|
| text activation | `silu`; MLP is gated `silu(gate) * up` | `config.json`, source |
| text attention biases | Q/K/V `bias=True`, O `bias=False` | source |
| text RMSNorm eps | `1e-6` in inspected configs; source class default is `1e-5` | `config.json`, source default |
| GQA grouping | `num_attention_heads / num_key_value_heads` = 6 for 2B, 7 for 7B, 8 for 72B | `config.json` |
| RoPE theta | `1000000.0` | `config.json` |
| M-RoPE section | `[16, 24, 24]`, doubled internally to `[32, 48, 48]` before splitting cos/sin | `config.json`, source |
| sliding window | `use_sliding_window=false`; `sliding_window=32768` is present but disabled | `config.json`, source |
| vision patch | temporal patch 2, spatial patch 14, merge 2 | `config.json`, preprocessor |
| vision input channels | source field is `in_channels`; hub `vision_config` uses `in_chans: 3` | source + config compatibility caveat |
| vision attention | noncausal full self-attention over per-image/video patch chunks; FA2 varlen path if requested | source |
| vision hidden act | source default `quick_gelu`; configs omit `hidden_act`, so source default applies unless mapping supplies one | source default inferred |
| image min/max pixels | min 3136, max 12845056 | `preprocessor_config.json` |
| video processor source defaults | min pixels `128*28*28`, max pixels `28*28*768`, min frames 4, max frames 768 | source |
| special token ids | `<|vision_start|>` 151652, `<|vision_end|>` 151653, `<|image_pad|>` 151655, `<|video_pad|>` 151656 | `config.json`, `tokenizer_config.json` |
| tokenizer padding | left padding, pad token `<|endoftext|>` | `tokenizer_config.json` |

Effective default omissions:

- Hub configs omit `text_config` object; current source constructs it from top-level fields.
- Hub configs omit `rope_parameters`; current source accepts `rope_scaling` and standardizes it.
- Hub `vision_config` omits `hidden_act` and `initializer_range`; source defaults are `quick_gelu` and `0.02`.
- Hub `vision_config` uses `in_chans`, while current source config field is `in_channels`; a loader compatibility layer may map this. DinoML should not assume source defaults for loaded checkpoint weights without checking the instantiated config object.

## 3a. Family variation traps

- This is GQA in all inspected checkpoint configs: KV heads are much fewer than query heads.
- `hidden_size == num_heads * head_dim` for inspected checkpoints, but source computes `head_dim = hidden_size // num_attention_heads` and checks divisibility.
- Text Q/K/V projections have bias; O projection and MLP projections are biasless. Vision QKV/proj/MLP/projector linears use biases except patch Conv3d.
- The vision tower is shared across 2B/7B/72B in depth/embed/head count; only projector output/context hidden size changes with text hidden size.
- Processor output is not NCHW image tensors for the model. It is flattened patch rows shaped `[sum_patches, 3 * temporal_patch_size * patch_size * patch_size]`.
- Images get temporal dimension `T=1` in `image_grid_thw`, but their flattened rows duplicate spatial patches across `temporal_patch_size=2` before Conv3d.
- Video frame count is padded by repeating the last frame until divisible by temporal patch size.
- Placeholder count must equal merged visual token count: `grid_t * grid_h * grid_w / merge_size^2`.
- M-RoPE requires `mm_token_type_ids` when multimodal grids are passed and `input_ids` are available; current source raises if omitted.
- Text-model position ids may be `[4, batch, seq]` when generation prepares both text-only positions and 3D M-RoPE positions for packed mask construction.
- Vision attention uses `cu_seqlens` with repeated per-frame spatial lengths. Non-FA implementations split by lengths and loop over chunks.
- Layout-sensitive region: processor patch packing has exact `reshape -> permute -> reshape` order. Treat it as CPU/data-pipeline semantics initially, not an arbitrary layout optimization.
- Layout-sensitive vision patch embed: source reshapes flattened patch rows to `[-1, C, T_patch, H_patch, W_patch]` and applies Conv3d with kernel=stride equal to the patch volume. A Conv3d-to-Linear rewrite is safe only under strict packed-row preconditions.
- Layout-sensitive merger: `PatchMerger` does `LayerNorm(context_dim)` then `.view(-1, context_dim * merge_size^2)`, relying on prior token order where each group of four spatially merged tokens is contiguous.

## 4. Operator coverage checklist

Tensor/layout ops:

- reshape/view, flatten, unsqueeze, expand, permute, transpose, contiguous materialization.
- split, torch.split by dynamic grid-derived lengths, concatenate along sequence axis.
- repeat, repeat_interleave, cumsum, pad for `cu_seqlens`.
- arange, meshgrid-like expand/repeat patterns for 2D/3D position ids.
- masked_fill, boolean comparisons, logical masks, roll for generation expansion.
- dtype casts: processor float to model dtype; RoPE forced fp32 then cast back.

Neural network primitives:

- Embedding lookup `[vocab_size, hidden_size]`.
- Linear with and without bias:
  - Text 7B Q: `3584 -> 3584` bias, K/V: `3584 -> 512` bias, O: `3584 -> 3584` no bias.
  - Text 7B MLP gate/up: `3584 -> 18944` no bias, down: `18944 -> 3584` no bias.
  - Vision QKV: `1280 -> 3840` bias, proj: `1280 -> 1280` bias.
  - Vision MLP: `1280 -> 5120 -> 1280` bias.
  - PatchMerger 7B: after LN/view `5120 -> 5120 -> 3584` with GELU.
- RMSNorm over last dimension with fp32 variance.
- LayerNorm over last dimension for vision blocks and patch merger.
- Activations: SiLU, GELU, quick_gelu.
- Conv3d patch embed with `in=3`, `out=1280`, `kernel=stride=(2,14,14)`, no bias; can be rewritten for packed rows.
- LM head: `hidden_size -> vocab_size`, biasless; 2B may tie with embedding.

Attention primitives:

- Text causal self-attention with GQA, RoPE/M-RoPE before cache update, KV cache.
- Vision noncausal self-attention, full attention per image/frame chunk, optional FlashAttention varlen using `cu_seqlens`.
- Mask construction for full causal and optional sliding window masks via Transformers common masking utilities.
- Softmax in fp32 for eager path, dropout disabled in inference.

Position/rotary/custom math:

- Standard RoPE inv-freq with theta `1e6` for text.
- M-RoPE split/recombine over temporal/height/width sections.
- Vision 2D rotary embeddings over per-patch height/width positions.

Generation/cache ops:

- DynamicCache or equivalent per-layer key/value state.
- Cache update after RoPE.
- Cache-aware input slicing in `GenerationMixin`.
- Omit visual tensors after first cached iteration.
- `logits_to_keep` last-token or indexed slicing before LM head.

Preprocessing-coupled ops:

- Smart resize to multiples of `patch_size * merge_size = 28`, preserving aspect ratio and min/max pixel bounds.
- Image rescale/normalize using CLIP mean/std.
- Video frame sampling or all-frame path, frame padding to temporal multiple.
- Packed patch row construction for images and videos.

Scatter/indexed update ops for multimodal stitch:

- Compare `input_ids` against image/video token ids.
- Expand boolean placeholder masks to embedding shape.
- Check flattened token element count equals feature element count.
- `masked_scatter` image/video embeddings into text embeddings.

Packed/varlen metadata ops:

- `image_grid_thw` and `video_grid_thw`: `LongTensor[num_items, 3]`.
- Vision `cu_seqlens`: `int32` normally, length `sum(grid_t over items)+1`, values are cumulative repeated `grid_h * grid_w`.
- Split sizes after merger: `grid_t * grid_h * grid_w / spatial_merge_size^2`.
- M-RoPE deltas: `[batch, 1]`.

Distributed/tensor-parallel ops:

- Source config includes TP/PP plans for common linear projections. This is optional/deferred for first DinoML integration.

## 5. Layer/block breakdown

Image/video preprocessing to vision inputs:

```text
image RGB/NCHW tensor
  -> resize to H,W divisible by 28 and within pixel bounds
  -> rescale/normalize
  -> reshape by grid_h, grid_w, merge groups, patch_size
  -> permute to group spatial-merge order
  -> duplicate image patches across temporal_patch_size
  -> pixel_values: [sum_images, grid_h*grid_w, 3*2*14*14]
  -> image_grid_thw: [num_images, 3] rows [1, grid_h, grid_w]
```

Video preprocessing:

```text
video tensor [B, T, C, H, W] in processor convention
  -> optional frame sample
  -> resize each frame to H,W divisible by 28
  -> rescale/normalize
  -> pad T by repeating last frame until divisible by 2
  -> reshape/permute temporal and spatial patches
  -> pixel_values_videos: [sum_videos, grid_t*grid_h*grid_w, 3*2*14*14]
  -> video_grid_thw: [num_videos, 3] rows [T/2, grid_h, grid_w]
```

Vision patch embed:

```text
x: [total_patch_rows, 1176]
x.view(-1, 3, 2, 14, 14)
Conv3d(3 -> 1280, kernel=stride=(2,14,14), bias=False)
x.view(-1, 1280)
```

Vision block, repeated 32 times:

```text
res = x
x1 = LayerNorm(1280)(x)
qkv = Linear(1280 -> 3840, bias=True)(x1)
q,k,v = reshape to [seq, 16, 80]
q,k = vision_2d_rope(q,k)
attn = noncausal attention over each cu_seqlens chunk
x = res + Linear(1280 -> 1280, bias=True)(attn)
res = x
x = res + Linear(5120 -> 1280, bias=True)(quick_gelu(Linear(1280 -> 5120, bias=True)(LayerNorm(x))))
```

Patch merger/projector:

```text
x: [sum_patches, 1280]
x = LayerNorm(1280)(x)
x.view(-1, 1280 * 2 * 2)  # [merged_tokens, 5120]
x = Linear(5120 -> 5120, bias=True)
x = GELU(x)
x = Linear(5120 -> text_hidden_size, bias=True)
```

Multimodal stitch:

```text
inputs_embeds = Embedding(input_ids)
image_embeds = cat(split vision pooler output by grid prod / 4)
image_mask = (input_ids == image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
repeat for video_token_id / video_embeds
position_ids = compute_3d_position_ids(...)
```

Text decoder block, repeated `num_hidden_layers`:

```text
res = x
x_norm = RMSNorm(hidden_size, eps=1e-6)(x)
q = Linear(hidden -> num_heads*head_dim, bias=True)(x_norm)
k = Linear(hidden -> kv_heads*head_dim, bias=True)(x_norm)
v = Linear(hidden -> kv_heads*head_dim, bias=True)(x_norm)
q,k = M-RoPE(q,k, position_embeddings)
k,v = cache.update(k,v, layer_idx) if cache is present
attn = causal GQA attention(q,k,v, mask)
x = res + Linear(num_heads*head_dim -> hidden, bias=False)(attn)
res = x
x_norm = RMSNorm(hidden, eps=1e-6)(x)
mlp = Linear(intermediate -> hidden, bias=False)(silu(gate_proj(x_norm)) * up_proj(x_norm))
x = res + mlp
```

Final:

```text
x = RMSNorm(hidden)(x)
logits = Linear(hidden -> vocab_size, bias=False)(x[:, slice_indices, :])
```

## 6. Attention requirements

Text attention:

- Type: causal self-attention.
- Head shape: q `[batch, q_len, num_heads, head_dim] -> [batch, heads, q_len, head_dim]`; k/v use `num_key_value_heads`.
- GQA expansion: eager path repeats KV from `[batch, kv_heads, kv_len, head_dim]` to `[batch, heads, kv_len, head_dim]`; fused path should avoid physical repeat.
- 7B example: q heads 28, KV heads 4, head dim 128, hidden 3584.
- Masking: common Transformers `create_causal_mask`; sliding-window mask only if config layer type is `sliding_attention`.
- Cache: keys and values are stored after M-RoPE has been applied to keys. Values are unrotated.
- Cache shapes before repeat: per layer K/V `[batch, num_key_value_heads, cache_seq, head_dim]`. After logical GQA repeat for eager attention: `[batch, num_attention_heads, cache_seq, head_dim]`.
- Attention math in eager fallback: matmul, multiply by `head_dim**-0.5`, add mask, softmax in fp32, cast to query dtype, dropout, matmul with V, transpose back.
- Backend dispatch: source uses `ALL_ATTENTION_FUNCTIONS.get_interface(config._attn_implementation, eager_attention_forward)` and advertises FlashAttention/SDPA support.
- Sliding window trap: source warns if `use_sliding_window` is true without FlashAttention; inspected configs disable it.

Vision attention:

- Type: noncausal self-attention.
- q/k/v shape before backend: source makes `[1, num_heads, total_seq, head_dim]`.
- Sequence partitioning: `cu_seqlens` partitions packed patches by each temporal frame of each image/video. Length per segment is `grid_h * grid_w`, repeated `grid_t` times.
- FlashAttention path: passes `cu_seq_lens_q`, `cu_seq_lens_k`, `max_length_q`, `max_length_k`, `is_causal=False`.
- Non-Flash path: splits Q/K/V by `lengths = cu_seqlens[1:] - cu_seqlens[:-1]`, runs attention per chunk, and concatenates outputs.
- Vision RoPE is applied to Q/K before backend call; no cache.

Fused attention parity notes:

- Preserve scaling placement before softmax for eager parity.
- Preserve fp32 softmax accumulation in composed fallback.
- For text, pass text-only position ids to FA2-style packed mask creation when position ids have the `[4, batch, seq]` form.
- For vision, `cu_seqlens` dtype is normally int32 but may match `grid_thw.dtype` during tracing.

## 7. Position encoding and custom math

Text RoPE:

```python
inv_freq = 1.0 / (rope_theta ** (arange(0, head_dim, 2) / head_dim))
freqs = (inv_freq[None, None, :, None].expand(3, batch, -1, 1)
         @ position_ids[:, :, None, :].float()).transpose(2, 3)
emb = cat(freqs, freqs, dim=-1)
cos, sin = emb.cos(), emb.sin()
```

M-RoPE application:

```python
def apply_mrope(q, k, cos, sin, section):
    section = section * 2
    cos = cat([part[i % 3] for i, part in enumerate(cos.split(section, -1))], -1).unsqueeze(1)
    sin = cat([part[i % 3] for i, part in enumerate(sin.split(section, -1))], -1).unsqueeze(1)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin
```

Important details:

- `mrope_section=[16,24,24]` from configs is doubled before splitting because cos/sin have duplicated half dimensions.
- Text-only tokens use identical temporal/height/width position ids, so M-RoPE reduces to normal 1D RoPE.
- Vision tokens use 3D positions derived from multimodal token groups and grids.
- During incremental decode, `rope_deltas` are cached on the model and added to new text positions.

Vision 2D RoPE:

```python
hpos = arange(h).reshape(h, 1).expand(h, w)
wpos = arange(w).reshape(1, w).expand(h, w)
# reshape/permute by spatial_merge_size, flatten in packed merge order
pos_ids = stack([hpos_flat, wpos_flat], dim=-1).repeat(t, 1)
rotary = rotary_pos_emb(max(h, w))[pos_ids].flatten(1)
emb = cat([rotary, rotary], dim=-1)
cos, sin = emb.cos(), emb.sin()
```

Precomputable:

- Text inv-frequencies and static sin/cos tables can be precomputed up to a max position, but multimodal position ids and `rope_deltas` are input-dependent.
- Vision rotary frequency table up to `max(grid_h, grid_w)` is reusable per grid size; gathered position ids are grid/order-dependent.

## 8. Preprocessing and input packing

Processor runtime tensors:

| Tensor | Shape | Dtype | Produced by | Meaning |
|---|---|---|---|---|
| `input_ids` | `[batch, text_seq]` | int64 | tokenizer | Text plus expanded image/video placeholders |
| `attention_mask` | `[batch, text_seq]` | int/bool | tokenizer | Padding mask |
| `mm_token_type_ids` | `[batch, text_seq]` | int | tokenizer helper | 0=text, 1=image, 2=video |
| `pixel_values` | `[sum_images, grid_h*grid_w, 1176]`, concatenated as `[total_image_patches, 1176]` after batching | float | image processor | Packed flattened image patch rows |
| `image_grid_thw` | `[num_images, 3]` | long | image processor | `[1, grid_h, grid_w]` per image |
| `pixel_values_videos` | `[sum_videos, grid_t*grid_h*grid_w, 1176]`, concatenated as `[total_video_patches, 1176]` | float | video processor | Packed flattened video patch rows |
| `video_grid_thw` | `[num_videos, 3]` | long | video processor | `[T/2, grid_h, grid_w]` per video |

Note: the model docstrings still describe `pixel_values` as image tensors, but source processor/model behavior expects flattened patch rows. DinoML should follow source code.

Placeholder expansion:

- The chat template emits one `<|image_pad|>` or `<|video_pad|>` inside `<|vision_start|>...<|vision_end|>`.
- Processor replaces each image/video pad token string with `num_tokens = prod(grid_thw) // merge_size^2` repeated pad tokens, then tokenizes.
- The model validates that the number of placeholder embedding elements equals image/video feature elements before `masked_scatter`.

Image packing:

- Resize factor is `patch_size * merge_size = 28`.
- `smart_resize` rounds to multiples of 28, enforces min/max pixels, and rejects extreme aspect ratio > 200.
- Image grid token count before merge is `grid_h * grid_w`; after merger/placeholder expansion it is `grid_h * grid_w / 4`.
- The source duplicates each image patch across the temporal patch dimension by `unsqueeze(6).expand(..., temporal_patch_size, ...)`, so patch embed Conv3d sees a 2-frame patch volume.

Video packing:

- If frame sampling is enabled or requested, `num_frames` or `fps` is rounded/floored to a multiple of `temporal_patch_size`.
- If actual `T` is not divisible by 2 at preprocess time, the last frame is repeated.
- Grid token count before merge is `grid_t * grid_h * grid_w`, with `grid_t = padded_T / 2`; after merger it is divided by 4.

M-RoPE input packing:

- `get_rope_index` groups contiguous token type runs in each sample.
- Text groups advance `current_pos` by text length.
- Image/video groups consume the next row from the corresponding grid iterator, generate 3D positions after spatial merge, then advance `current_pos` by `max(grid_h, grid_w) / spatial_merge_size`.
- `mrope_position_deltas = max(position_ids) + 1 - unpadded_sequence_length`.

CPU/data-pipeline versus GPU/runtime:

- First integration can keep resize/rescale/normalize/patch packing/token placeholder expansion on CPU.
- GPU/runtime must support the resulting packed tensors, grid metadata, feature scatter, M-RoPE position-id generation or accept precomputed position ids.
- Vision encoder/projector output can be cached per image/video for repeated prompts if token order and grid metadata are stable.

Generation-controller behavior:

- In cached decode after the first iteration, `prepare_inputs_for_generation` drops `pixel_values` and `pixel_values_videos`.
- `_prepare_position_ids_for_generation` returns `[4, batch, seq]` when it prepends text-only positions to 3D vision positions; later decode uses cached `rope_deltas`.
- `generation_config.json` uses eos/pad ids from tokenizer config; no model-specific suppress-token or timestamp processors were found.

## 9. Graph rewrite / lowering opportunities

### Rewrite: packed Conv3d patch embed -> Linear

Source pattern:

```text
hidden_states.view(-1, C, T_patch, H_patch, W_patch)
Conv3d(C -> E, kernel=(T_patch,H_patch,W_patch), stride=same, bias=False)
view(-1, E)
```

Replacement:

```text
MatMul(packed_patch_rows, weight_flat.T)
```

Preconditions:

- Input is exactly processor-packed rows with last dim `C*T_patch*H_patch*W_patch`.
- Conv3d kernel and stride equal `[temporal_patch_size, patch_size, patch_size]`.
- Padding = 0, dilation = 1, groups = 1, bias = false.
- Source reshape order is `[C, T, H, W]`; processor flatten order must be preserved.

Shape equations:

- `K = 3 * 2 * 14 * 14 = 1176`.
- `E = vision_embed_dim = 1280`.
- Output `[total_patch_rows, 1280]`.

Weight transform:

```python
w = conv.weight.reshape(embed_dim, in_channels * temporal_patch_size * patch_size * patch_size)
y = x @ w.T
```

Layout constraints:

- Do not reinterpret source as NHWC/NTHWC unless the processor flatten order and weight permutation are rewritten together.

Failure cases:

- Raw image/video tensors, different temporal patch size, nonzero conv bias, padding/dilation/groups changes.

Parity test sketch:

- Random packed rows and real checkpoint patch weight; compare Conv3d source against Linear rewrite in fp32 and bf16.

### Rewrite: Qwen2 gated MLP -> fused SwiGLU GEMM chain

Source pattern:

```text
down_proj(silu(gate_proj(x)) * up_proj(x))
```

Replacement:

```text
DualLinear(x, [gate, up]) -> SiLU(gate) * up -> Linear(down)
```

Preconditions:

- `gate_proj` and `up_proj` share input shape, dtype, and no bias.
- Activation is `silu`.
- Intermediate sizes match.

Shape equations:

- 7B: `[B*S,3584] -> [B*S,18944]` twice, elementwise multiply, then `[B*S,3584]`.

Weight transform:

- Concatenate gate/up weights along output dimension for a dual-GEMM or single larger GEMM if backend supports split outputs.

Failure cases:

- Non-SiLU activation or projection biases.

Parity test sketch:

- Single decoder MLP with random hidden states and checkpoint weights.

### Rewrite: vision QKV packed projection

Source pattern:

```text
Linear(1280 -> 3840)(x).reshape(seq, 3, heads, head_dim).permute(1,0,2,3)
```

Replacement:

```text
single GEMM with packed QKV output, then strided views/slices into Q,K,V
```

Preconditions:

- Bias present and packed in Q,K,V order.
- Output is immediately reshaped as source order.

Failure cases:

- Backend cannot represent strided QKV views or needs materialized contiguous tensors for attention.

Parity test sketch:

- Compare Q/K/V slices before RoPE for random seq lengths.

### Rewrite: text GQA attention with cache

Source pattern:

```text
Q/K/V linears -> M-RoPE(Q,K) -> cache update -> GQA attention -> O linear
```

Replacement:

```text
fused prefill/decode attention kernel accepting q_heads, kv_heads, head_dim,
M-RoPE cos/sin or pre-rotated Q/K, and KV cache
```

Preconditions:

- Causal mask, no dropout in inference.
- Cache stores rotated K.
- RoPE scaling/type matches default M-RoPE.
- Sliding window disabled or supported by backend.

Failure cases:

- Packed FA2 position-id mask path not reproduced.
- `use_sliding_window=true` without a matching local attention kernel.

Parity test sketch:

- Prefill logits and one-token decode with identical cache against Transformers.

### Rewrite: PatchMerger grouped projection

Source pattern:

```text
LayerNorm(1280)(x).view(-1, 5120) -> Linear -> GELU -> Linear(hidden)
```

Replacement:

```text
group-aware view + GEMM/GELU/GEMM
```

Preconditions:

- Vision token order matches processor/rotary order.
- `spatial_merge_size == 2`.
- Number of tokens is divisible by 4 for each item.

Failure cases:

- Layout pass changes vision token order or fuses away the required spatial grouping.

Parity test sketch:

- Compare per-image split outputs using several grid sizes.

### Layout pass: safe regions and guards

Candidate safe regions:

- Text decoder hidden states are `[batch, seq, hidden]`; last-dim linear/norm/activation regions are layout-stable.
- Vision linears/norms after patch embedding operate on `[seq, 1280]`; no channel-last conversion needed.

No-layout-translation guard regions:

- Processor patch packing `reshape/permute/reshape`.
- Vision `rot_pos_emb` position-id flatten order.
- PatchMerger `.view(-1, context_dim * merge_size^2)`.
- `masked_scatter` placeholder order.

Axis-sensitive attrs:

- Norm/reductions are last-dim (`dim=-1`) in RMSNorm/LayerNorm.
- Vision processor permutes exact axes `(0,2,5,3,6,1,4,7)` for images and `(0,1,4,7,5,8,3,2,6,9)` for videos.
- Attention reshapes and transposes distinguish sequence/head axes; do not silently convert to NHWC semantics.

## 10. Kernel fusion candidates

Highest priority:

- RMSNorm last-dim kernel for text hidden sizes 1536/3584/8192.
- GQA FlashAttention/paged attention with KV cache, supporting KV heads < Q heads and rotated-K cache.
- M-RoPE application fused with Q/K projection or attention input preparation.
- SwiGLU MLP fusion: dual projection, SiLU multiply, down projection.
- Packed Conv3d patch embed lowered to GEMM for the processor-packed patch-row contract.
- `masked_scatter` or indexed copy for multimodal embedding stitch.

Medium priority:

- Vision varlen noncausal attention using `cu_seqlens`.
- Vision LayerNorm + QKV projection fusion.
- PatchMerger LN/view/GEMM/GELU/GEMM path.
- Last-token-only logits through `logits_to_keep`.
- Precomputed/gathered RoPE cos/sin tables for common grid sizes and decode steps.

Lower priority:

- Beam-search visual tensor expansion.
- Full CPU/GPU preprocessing kernels for resize/rescale/normalize/patch packing.
- Tensor-parallel collective support from source TP plans.
- Sliding-window attention, because inspected configs disable it.

## 11. Runtime staging plan

Stage 1: config and weights.

- Parse flat Qwen2-VL configs into text and vision subconfigs.
- Load embeddings, vision tower, patch merger, text decoder, and LM head.
- Treat processor and tokenizer as external for the first graph.

Stage 2: vision encoder/projector parity.

- Accept precomputed `pixel_values` and `image_grid_thw`.
- Implement packed patch Linear rewrite or faithful Conv3d path.
- Implement vision 2D RoPE, `cu_seqlens`, 32 vision blocks, and PatchMerger.

Stage 3: multimodal prefix construction.

- Accept `input_ids`, `attention_mask`, `mm_token_type_ids`, `pixel_values`, `image_grid_thw`.
- Implement placeholder mask checks and indexed scatter.
- Implement or accept precomputed 3D `position_ids`.

Stage 4: text prefill parity.

- Run full multimodal prompt through decoder with causal mask and M-RoPE.
- Produce prefill logits for `logits_to_keep=1` and full logits debug mode.

Stage 5: cached decode.

- Implement per-layer GQA KV cache with rotated K.
- Reproduce `rope_deltas` decode position handling.
- Drop vision inputs after first iteration.

Stage 6: video path.

- Add `pixel_values_videos`, `video_grid_thw`, video placeholder scatter, and video M-RoPE positions.
- Keep frame sampling/resize/packing outside DinoML initially.

Stage 7: optimized kernels and scheduling.

- Add FlashAttention/paged attention, fused MLP/norm, projector fusions, and batch scheduling.

Stub initially:

- Processor CPU path can be delegated to Transformers.
- Position ids may be supplied as explicit inputs before DinoML reproduces `get_rope_index`.
- Beam expansion and tensor parallelism can be deferred.

## 12. Parity and validation plan

Random tensor tests:

- RMSNorm fp32/bf16, LayerNorm, quick_gelu, SiLU gated MLP.
- M-RoPE function for text-only and mixed 3D positions.
- Vision 2D RoPE gather/order for several `grid_thw` values.
- Packed Conv3d-to-Linear patch embed rewrite.
- Placeholder `masked_scatter` with exact count checks and mismatch failures.

Single-layer parity:

- One vision block with random packed patch rows and fixed `grid_thw`.
- PatchMerger over token counts divisible by 4.
- One text decoder block with GQA and no cache.
- One text decoder block with prefilled cache and one decode token.

After-N-layer parity:

- Vision tower after 1, 4, 32 blocks on a small synthetic grid.
- Text decoder after 1, 4, full layers on short prompts.

Encoder/projector parity:

- Compare `get_image_features(pixel_values, image_grid_thw).pooler_output` split sizes and values.
- Repeat for video after video path is admitted.

Prefill logits parity:

- Use official processor to build a one-image prompt for 2B or 7B.
- Feed packed tensors to DinoML and Transformers.
- Compare `logits[:, -1, :]` and selected top-k logits.

Decode token parity:

- Prefill a multimodal prompt, decode 1 token with cache, then decode several tokens greedily.
- Verify cache lengths and token ids.

End-to-end parity:

- Image-text prompt with processor externalized.
- Video-text prompt once video path lands.

Recommended tolerances:

- fp32 unit ops: `rtol=1e-5`, `atol=1e-6`.
- bf16/fp16 graph parity: `rtol=3e-2`, `atol=3e-2` for logits initially, tighten per op.
- Attention parity should compare pre-softmax or attention outputs as well as logits to localize RoPE/cache mistakes.

## 13. Performance probes

- Processor-only throughput: image resize/rescale/patch packing images/sec and video frames/sec.
- Vision encoder only: total patches/sec and merged tokens/sec versus grid size.
- PatchMerger throughput: merged tokens/sec by hidden size.
- Prefill-only: multimodal sequence length sweep, visual token count sweep, batch-size sweep.
- Decode-only: tokens/sec for cached GQA at batch sizes 1, 4, 16; cache length sweep.
- End-to-end requests/hour: split processor, vision, prefill, decode, logits.
- KV cache memory: layers * 2 * batch * kv_heads * seq * head_dim * dtype bytes.
- Attention backend comparison: eager/SDPA/FlashAttention-style for text and vision varlen separately.
- Placeholder scatter cost: visual token count and hidden size sweep.
- LM head cost: full logits versus `logits_to_keep=1`, vocab 151936/152064.

No benchmark results are included here; all entries are proposed probes from source-derived bottlenecks.

## 14. Skip/defer list

- Training and loss parity beyond smoke tests.
- Gradient checkpointing.
- Tensor parallel and pipeline parallel execution.
- Beam-search visual tensor expansion for first greedy integration.
- Sliding-window attention unless a checkpoint with `use_sliding_window=true` is targeted.
- Full tokenizer/chat-template implementation inside DinoML.
- CPU/GPU image/video resize kernels; keep preprocessing external at first.
- Quantized checkpoint ingestion.
- Speculative decoding.
- Attention/hidden-state debug outputs.
- PIL-specific image processor path unless a deployment requires it.

## 15. Final implementation checklist

- [ ] Parse flat Qwen2-VL config into text and vision config objects.
- [ ] Load 2B/7B representative weights, including tied 2B embedding/lm_head handling.
- [ ] Support processor-produced `pixel_values`, `pixel_values_videos`, `image_grid_thw`, `video_grid_thw`, and `mm_token_type_ids`.
- [ ] Implement packed patch-row Conv3d-to-Linear parity.
- [ ] Implement vision 2D RoPE position gathering.
- [ ] Implement vision varlen noncausal attention using `cu_seqlens`.
- [ ] Implement PatchMerger order-sensitive view/projector.
- [ ] Implement placeholder mask validation and indexed scatter into embeddings.
- [ ] Implement M-RoPE position id generation or accept explicit precomputed `position_ids`.
- [ ] Implement text RMSNorm, biased Q/K/V, biasless O, GQA attention, and rotated-K KV cache.
- [ ] Implement Qwen2 SwiGLU MLP.
- [ ] Implement final RMSNorm and `logits_to_keep` LM head.
- [ ] Add single-op tests for RoPE, M-RoPE, patch embed rewrite, PatchMerger, and scatter.
- [ ] Add one-block and full-vision parity tests.
- [ ] Add text prefill and one-token decode parity tests.
- [ ] Add one-image end-to-end greedy decode parity with external Transformers processor.
- [ ] Add video parity after image path is stable.
- [ ] Benchmark processor, vision, prefill, decode, LM head, and cache memory separately.
