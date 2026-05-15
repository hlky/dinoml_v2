# Transformers family audit: `lighton_ocr`

## 1. Source basis

```text
Transformers commit/version:
  Local checkout transformers at b75feb2af64c3e29cbbc1bd859958c5432cc7ed4.

Model id:
  Primary current-source target: lightonai/LightOnOCR-1B-1025.
  Representative configs inspected:
    - lightonai/LightOnOCR-1B-1025
    - lightonai/LightOnOCR-0.9B-16k-1025
    - lightonai/LightOnOCR-0.9B-32k-1025
    - lightonai/LightOnOCR-2-1B-bbox-soup

Config source:
  HF raw config/processor/preprocessor/generation/tokenizer snapshots saved under
  agents/plans/transformers/lighton_ocr/_sources/.

Source files inspected:
  transformers/src/transformers/models/lighton_ocr/configuration_lighton_ocr.py
  transformers/src/transformers/models/lighton_ocr/modeling_lighton_ocr.py
  transformers/src/transformers/models/lighton_ocr/modular_lighton_ocr.py
  transformers/src/transformers/models/lighton_ocr/processing_lighton_ocr.py
  transformers/src/transformers/models/pixtral/modeling_pixtral.py
  transformers/src/transformers/models/pixtral/configuration_pixtral.py
  transformers/src/transformers/models/pixtral/image_processing_pixtral.py
  transformers/src/transformers/models/qwen3/modeling_qwen3.py
  transformers/src/transformers/models/qwen3/configuration_qwen3.py

Any missing files or assumptions:
  modeling_lighton_ocr.py, configuration_lighton_ocr.py, and processing_lighton_ocr.py
  are generated from modular_lighton_ocr.py; modular_lighton_ocr.py is the future
  source-edit basis. The audit focuses on inference for image/document-to-text
  conditional generation. Training losses, gradient checkpointing, and generic
  classification/QA heads from the delegated Qwen3 family are out of scope.
```

Primary source scope warning: public checkpoint configs currently use
`model_type: "mistral3"` with `architectures: ["LightOnOCRForConditionalGeneration"]`,
while the inspected in-library LightOn source defines `LightOnOcrConfig.model_type =
"lighton_ocr"`. DinoML should gate admission on an explicit route: either native
`lighton_ocr` source for current Transformers, or the older Mistral3-compatible
remote/checkpoint route as a separately audited compatibility path.

## 2. High-level architecture

LightOn OCR is a vision encoder + projector + causal text decoder generation
model. The neural body composes:

```text
CPU image preprocessing + prompt image-token expansion
  -> Pixtral vision encoder over padded NCHW pixel_values
  -> LightOn OCR RMSNorm + 2x2 patch merger + GELU MLP projector
  -> masked image-feature stitch into Qwen token embeddings
  -> Qwen3 causal LM prefill/decode with KV cache
  -> lm_head logits -> generation controller -> decoded OCR/markup text
```

Stage decomposition:

- CPU/data pipeline: RGB conversion, resize to longest edge with patch-aligned
  output, rescale/normalize, pad to batch max height/width, tokenizer prompt
  handling, and replacement of each image placeholder with one token per merged
  2x2 patch.
- Cacheable vision/projector stage: `pixel_values` plus `image_sizes` produce
  projected image features. These can be cached for repeated text prompts over
  the same page image.
- Prefix construction: image features are copied into positions where
  `input_ids == image_token_id`. The source uses `masked_scatter`, but the
  processor gives a stricter ordered placeholder-token pattern.
- Text prefill/decode: Qwen3 causal decoder owns autoregressive KV cache.
  LightOn `prepare_inputs_for_generation` forwards `pixel_values` only on the
  first generation iteration unless cache is disabled.
- Structured output: the current source has no box decoder, NMS, layout-head, or
  postprocessing function. Bounding boxes or markup are generated as text tokens.

## 3. Important config dimensions

Source defaults from `LightOnOcrConfig`:

| Field | Effective value |
| --- | --- |
| `spatial_merge_size` | 2 |
| `image_token_id` | 151655 |
| `tie_word_embeddings` | true |
| vision `model_type` | `pixtral` |
| vision hidden/layers/heads/head_dim | 1024 / 24 / 16 / 64 |
| vision MLP intermediate | 4096 |
| vision image/patch size | 1540 / 14 |
| vision activation | `silu` |
| vision RoPE theta | 10000 |
| text `model_type` | `qwen3` |
| text hidden/layers/heads/KV heads/head_dim | 1024 / 28 / 16 / 8 / 128 |
| text MLP intermediate | 3072 |
| text vocab | 151936 default |
| text max positions | 40960 source default, but public checkpoints vary |
| text RoPE theta | 1000000 |
| text activation | `silu` |
| text cache | `use_cache=True` in text config |
| text sliding window | `None` in LightOn defaults and inspected configs |

Representative checkpoint sweep:

| Model | HF repo sha | Config route | Text vocab | Max pos | dtype source | Processor class | Notable variation |
| --- | --- | --- | ---: | ---: | --- | --- | --- |
| `lightonai/LightOnOCR-1B-1025` | `a8a4a065974c82d6c707ca13132bc19f857625fc` | `model_type=mistral3`, architecture `LightOnOCRForConditionalGeneration` | 151936 | 8192 | `dtype=bfloat16` in config | `LightOnOCRProcessor` | Main public OCR checkpoint; image token id 151655. |
| `lightonai/LightOnOCR-0.9B-16k-1025` | `e56908e08b233d4f800d073ad61d05ecae019de9` | `model_type=mistral3` | 16384 | 8192 | `torch_dtype=bfloat16` in snapshot summary | `PixtralProcessor` in processor file, `LightOnOCRProcessor` in preprocessor | Small-vocab compatibility/config drift; special token fields are sparse. |
| `lightonai/LightOnOCR-0.9B-32k-1025` | `2329700684730dbacbb8d760c2701e6a7dc73066` | `model_type=mistral3` | 32768 | 8192 | `torch_dtype=bfloat16` in snapshot summary | `PixtralProcessor` in processor file, fast Pixtral image processor | Small-vocab variant; same text layer geometry. |
| `lightonai/LightOnOCR-2-1B-bbox-soup` | `dfdbd3e3627d80e28ddadece14098131aa485700` | `model_type=mistral3` | 151936 | 16384 | `dtype=bfloat16` in config | `PixtralProcessor` with embedded image processor object | Bbox-tuned generation checkpoint; no source-level structured-output head. |

All inspected configs keep the same operator-significant hidden sizes, layer
counts, vision patch size, and GQA geometry. Variations are mostly tokenizer
vocab, sequence length, generation config, and routing/processor metadata.

## 3a. Family variation traps

- Current-source `model_type="lighton_ocr"` differs from public configs using
  `model_type="mistral3"`. Gate this before weight loading.
- Public configs include historical fields such as `vision_feature_layer`,
  `multimodal_projector_bias`, `projector_hidden_act`, and `use_qk_norm`.
  Current `modeling_lighton_ocr.py` hard-codes last vision hidden state,
  bias-free projector linears, GELU projector activation, and Qwen3 q/k RMSNorm;
  do not expose those as free graph variants without source support.
- Qwen3 has `hidden_size != num_attention_heads * head_dim` in this family:
  1024 hidden, 16 heads, head_dim 128 gives attention width 2048. Q/K/O
  projection widths must be read from config fields, not inferred from hidden
  size.
- Text attention is GQA: 16 query heads, 8 KV heads, 2 query groups per KV head.
- Vision attention is noncausal full self-attention with a block mask separating
  images in the concatenated image sequence unless FlashAttention is requested.
- Processor output is NCHW. NHWC/channel-last is an optimization opportunity
  only inside guarded patch-conv and normalization regions.
- Patch merger depends on `image_sizes`, `patch_size`, and exact row-major
  patch order. Layout passes must not silently rewrite its `view/permute/unfold`
  axes.
- The image stitch uses `masked_scatter`; DinoML should lower it through a
  stricter ordered indexed row-copy only when placeholder-token count/order
  guards pass.
- `logits_to_keep` can be an integer slice or tensor indices. First integration
  can require last-token logits with `logits_to_keep=1`.
- Generation config is checkpoint-owned: 1B and bbox checkpoints use sampling
  defaults with EOS list `[151645, 151643]`, while 0.9B snapshots use older
  `bos_token_id=2`, `eos_token_id=1`.
- `image_break_token` and `image_end_token` are processor/tokenizer ABI fields
  but the current neural graph only consumes `image_token_id` for embedding
  replacement.

## 4. Operator coverage checklist

Tensor/layout ops:

- Embedding lookup for `input_ids -> [B, S, 1024]`.
- `masked_scatter` equivalent for image-feature stitch; preferred bounded lower:
  ordered `input_ids == image_token_id` row-copy into token embedding rows.
- `cat`, `split`, `view`, `reshape`, `transpose`, `permute`, `unsqueeze`,
  `squeeze`, `expand`, contiguous materialization.
- Dynamic shape arithmetic from `image_sizes`: patch grid sizes, token counts,
  split sizes, merged grid sizes.
- `torch.meshgrid`, `arange`, `stack`, `chunk`, indexing/gather for Pixtral
  position ids.
- NCHW pad and crop/slice for padded batch images.

Neural network primitives:

- Conv2d patch embedding: `Conv2d(3 -> 1024, kernel=14, stride=14, bias=False)`.
- RMSNorm over hidden width 1024 and q/k head width 128; fp32 variance and
  rsqrt, output cast back to input dtype.
- GEMM/Linear, all bias-free in inspected source:
  - vision Q/K/V/O: `1024 -> 1024`
  - vision MLP gate/up/down: `1024 -> 4096`, `1024 -> 4096`, `4096 -> 1024`
  - patch merger: `4096 -> 1024`
  - projector: `1024 -> 1024`, `1024 -> 1024`
  - text Q: `1024 -> 2048`
  - text K/V: `1024 -> 1024` each
  - text O: `2048 -> 1024`
  - text MLP gate/up/down: `1024 -> 3072`, `1024 -> 3072`, `3072 -> 1024`
  - lm_head: `1024 -> vocab_size`
- Activations: `silu` for Pixtral/Qwen gated MLPs, `GELU` for LightOn projector.
- Residual adds.

Attention primitives:

- Pixtral dense noncausal MHA with 16 heads, head_dim 64, 2D RoPE, optional
  block-diagonal image mask.
- Qwen3 causal GQA with 16 Q heads, 8 KV heads, head_dim 128, q/k RMSNorm before
  RoPE, KV cache update, optional sliding-window code path gated off by configs.
- Backend dispatch through Transformers attention interface: eager, SDPA,
  FlashAttention, and FlexAttention are advertised by the source.

Position/rotary/custom math:

- Pixtral 2D RoPE table indexed by patch grid ids from `position_ids_in_meshgrid`.
- Qwen3 1D RoPE using `rope_theta=1000000`; dynamic rope variants exist in
  generic Qwen3 source but inspected LightOn configs use default rope.

Generation/cache ops:

- Dynamic KV cache for Qwen3 text layers. Per layer:
  - key: `[B, 8, past_seq + T, 128]`
  - value: `[B, 8, past_seq + T, 128]`
  Stored after q/k RMSNorm and RoPE for keys.
- Cache-dependent position id generation from `past_key_values.get_seq_length()`.
- First-iteration image forwarding; decode iterations normally omit
  `pixel_values`.
- `logits_to_keep` slicing before LM head.

Preprocessing-coupled ops:

- Resize preserving aspect ratio, floor when shrinking, then align height/width
  upward to patch size.
- CLIP-style RGB image normalization with mean `[0.48145466, 0.4578275,
  0.40821073]` and std `[0.26862954, 0.26130258, 0.27577711]`.
- Batch padding to max resized height/width.
- Prompt image-token expansion to `(resized_h // 28) * (resized_w // 28)`
  placeholder tokens, because effective patch is `14 * spatial_merge_size`.

Structured output postprocess:

- No source-level postprocessor was found for boxes, HTML, Markdown, NMS, mask
  decoding, or coordinate conversion. Bbox/checkpoint-specific output is text
  generation controlled by tokenizer/template and downstream application code.

Quantized/packed weights:

- No source-coupled packed/quantized weight format is implemented in this
  Transformers family. GGUF/community conversions should be treated as DinoML
  loading/provider policy, not as native Transformers model behavior.

## 5. Layer/block breakdown

Processor and vision input:

```text
images -> RGB -> resize longest_edge<=1540 and patch-align to 14
       -> rescale/normalize -> pad batch -> pixel_values [B,3,Hmax,Wmax]
       -> image_sizes list[(Hi, Wi)] after resize
text prompt containing image_token -> tokenizer input_ids with repeated image tokens
```

Pixtral vision model:

```text
patch_embeds = Conv2d(pixel_values, kernel=14, stride=14, bias=False)
for each image i:
  patch_i = patch_embeds[i, :, :Hi/14, :Wi/14]
patch_seq = cat([patch_i.flatten(1).T], dim=0).unsqueeze(0)  # [1, total_patches, 1024]
patch_seq = RMSNorm(patch_seq)
position_ids = meshgrid ids per image
cos, sin = Pixtral2DRoPE(position_ids)
attention_mask = block diagonal per image unless flash attention
repeat 24:
  x = x + MHA(RMSNorm(x), 2D RoPE, block/full mask)
  x = x + Linear(silu(Linear(RMSNorm(x))) * Linear(RMSNorm(x)))
```

LightOn projector:

```text
image_features = RMSNorm(last_hidden_state.squeeze(0))
for each image:
  grid = image_tokens.view(h, w, d).permute(2,0,1).unsqueeze(0)
  merged = unfold(grid, kernel=2, stride=2).view(d*4, -1).T
merged = cat(merged_images, dim=0)
merged = Linear(4096 -> 1024, bias=False)
projected = Linear(1024 -> 1024, bias=False)
projected = GELU(projected)
projected = Linear(1024 -> 1024, bias=False)
split by (Hi // 28) * (Wi // 28)
```

Embedding stitch and Qwen3 decoder:

```text
inputs_embeds = token_embedding(input_ids)
image_features = cat(projected_image_features).to(inputs_embeds dtype/device)
mask = input_ids == image_token_id
guard mask_count * hidden == image_features.numel()
inputs_embeds = row-copy image_features into masked embedding rows

position_ids = arange(T) + cache_seen
causal_mask = full causal mask, or sliding mask if future configs enable it
cos, sin = Qwen3RoPE(position_ids)
repeat 28:
  x = x + O(Attention(RMSNorm(x), q/k RMSNorm, RoPE, GQA, cache))
  x = x + down(silu(gate(RMSNorm(x))) * up(RMSNorm(x)))
x = RMSNorm(x)
logits = lm_head(x[:, slice(-logits_to_keep, None), :])
```

## 6. Attention requirements

Pixtral vision attention:

- Noncausal self-attention.
- MHA: 16 query/key/value heads, head_dim 64.
- Q/K/V projections are bias-free `1024 -> 1024`.
- Masking: block-diagonal additive mask when multiple images are concatenated
  and FlashAttention is not requested. Each image attends only within its own
  patch sequence.
- Position: 2D patch-grid RoPE applied to Q/K before attention.
- KV cache: none; this is an encoder-style branch.
- Flash/SDPA compatibility: source advertises these backends; flash path relies
  on position ids and skips explicit block mask in the inspected Pixtral source.
  DinoML should only use that optimization when the batch/image packing semantics
  are reproduced.

Qwen3 text attention:

- Causal self-attention.
- GQA: Q heads 16, KV heads 8, repeat factor 2 for eager attention.
- `head_dim=128`, so Q/O attention width is 2048 while hidden size is 1024.
- Q/K/V projections are bias-free in inspected configs.
- q/k RMSNorm is applied on per-head dim before RoPE.
- RoPE is applied before cache update; cached keys are post-RoPE.
- Masking: generated causal mask from Transformers masking utilities; future
  sliding-window mask is possible via Qwen3 config but disabled here.
- Cache ABI: per layer stores K/V as `[B, 8, cache_len, 128]`; attention backend
  may repeat KV to 16 heads internally.
- First integration can support prefill with cache and single-token decode with
  last-token logits; multi-token chunked decode follows the same cache ABI.

## 7. Position encoding and custom math

Qwen3 RoPE:

```python
inv_freq = 1.0 / (rope_theta ** (arange(0, head_dim, 2) / head_dim))
freqs = inv_freq[None, :, None] @ position_ids[:, None, :]
emb = cat([freqs.transpose(1, 2), freqs.transpose(1, 2)], dim=-1)
cos, sin = cos(emb), sin(emb)
q = q * cos[:, None, :, :] + rotate_half(q) * sin[:, None, :, :]
k = k * cos[:, None, :, :] + rotate_half(k) * sin[:, None, :, :]
```

Pixtral 2D RoPE:

```python
max_side = image_size // patch_size
freqs = 1 / (theta ** (arange(0, head_dim, 2) / head_dim))
freqs_h = outer(arange(max_side), freqs[::2])
freqs_w = outer(arange(max_side), freqs[1::2])
table = cat([freqs_h[:, None, :].repeat(1, max_side, 1),
             freqs_w[None, :, :].repeat(max_side, 1, 1)], dim=-1)
table = cat([table.reshape(-1, head_dim // 2)] * 2, dim=-1)
cos, sin = cos(table[position_ids]), sin(table[position_ids])
```

Patch position ids:

```python
for patch_grid in patch_embeds_list:
    h, w = patch_grid.shape[-2:]
    ids = row_index * max_width + col_index
```

Precomputable:

- Qwen3 inv_freq and fixed-length cos/sin tables up to admitted max context.
- Pixtral 2D table for `image_size // patch_size` squared positions.

Dynamic inputs:

- Pixtral position id list depends on each resized image size.
- Qwen3 decode position ids depend on cache length.

## 8. Preprocessing and input packing

Image processor ABI:

- Input ownership: image decode is outside the neural graph; processor accepts
  PIL/array/tensor-like images through Transformers image utilities.
- Resize: longest edge bounded by 1540 in inspected LightOn configs; aspect
  ratio preserved; shrinking uses floor; output height/width are rounded up to
  multiples of patch size 14.
- Layout: emitted `pixel_values` are channels-first `[B, 3, Hmax, Wmax]`.
- Padding: each image is padded on bottom/right to batch max resized height/width.
- `image_sizes` are the unpadded resized sizes, consumed by both Pixtral cropping
  and LightOn patch merger.

Prompt/token ABI:

- Tokenizer class in snapshots: `Qwen2Tokenizer`.
- Main image placeholder token: `<|image_pad|>`; neural graph consumes
  `image_token_id`, 151655 for 1B/bbox configs.
- Processor expands each image placeholder in text into one repeated image token
  per merged 2x2 patch: `(resized_h // 28) * (resized_w // 28)`.
- Order: image sizes are consumed in encounter order while scanning samples;
  projected features are concatenated in the same image order.
- Guard required: count of image placeholder embedding slots must equal
  projected image feature rows. The source raises through `torch_compilable_check`
  before `masked_scatter` if counts differ.
- `mm_token_type_ids` can be returned by processor but are not consumed by the
  LightOn model forward in inspected source.

Document/OCR ABI:

- There is no separate OCR engine invoked by the processor. OCR is end-to-end
  image-to-text generation.
- There are no caller-supplied word boxes, normalized bounding boxes, subword box
  expansion rules, or bbox embedding indices in the current source.
- Bbox-tuned checkpoints should be treated as text-output variants unless a
  downstream application defines parsing/postprocessing outside Transformers.

## 9. Graph rewrite / lowering opportunities

### Rewrite: Pixtral non-overlap Conv2d patch embedding -> Linear/GEMM

Source pattern:

```text
Conv2d(C=3, out=1024, kernel=14, stride=14, padding=0, bias=False)
```

Replacement:

```text
PatchExtract[NCHW, 14x14 non-overlap] -> GEMM(weight.reshape(1024, 3*14*14).T)
```

Preconditions:

- `kernel_size == stride == patch_size`.
- `padding == dilation == 0/1` respectively.
- `groups == 1`, `bias is None`.
- Input is NCHW or a guarded NHWC layout rewrite transforms both extraction and
  weight flattening.
- Runtime image sizes are patch-aligned as guaranteed by processor.

Failure cases:

- Any future overlapping patch config, bias, non-RGB input, or arbitrary stride
  must fall back to Conv2d.

Parity test sketch:

- Random padded NCHW image batch with distinct `image_sizes`; compare patch
  tokens before cropping/flattening against PyTorch Conv2d.

### Rewrite: LightOn patch merger unfold -> grouped row gather + GEMM

Source pattern:

```text
view(h,w,d) -> permute(d,h,w) -> unfold(kernel=2,stride=2)
-> view(d*4, -1).T -> Linear(4096 -> 1024)
```

Replacement:

```text
For each 2x2 patch block in row-major order:
  concatenate [top-left, top-right, bottom-left, bottom-right] channel vectors
  GEMM merged_rows x W_merger.T
```

Preconditions:

- `spatial_merge_size == 2`.
- `h` and `w` are even after processor alignment to effective patch 28.
- Source row-major order and `permute(2,0,1)` flatten order are reproduced.
- No cross-image merging; split by `image_sizes`.

Failure cases:

- Odd patch grids, non-2 merge, or a layout pass that changes patch sequence
  order without rewriting gather indices.

Parity test sketch:

- Create synthetic `[total_patches, d]` with identifiable row/col/channel values;
  compare merged rows and final linear output.

### Rewrite: image `masked_scatter` -> ordered image row copy

Source pattern:

```text
mask = (input_ids == image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
inputs_embeds.masked_scatter(mask, image_features)
```

Replacement:

```text
positions = nonzero(input_ids == image_token_id) in row-major batch/sequence order
copy image_features[row] -> inputs_embeds[positions[row]]
```

Preconditions:

- `input_ids` is available; `inputs_embeds`-only matching by embedding equality
  can be rejected initially.
- Placeholder count equals image feature rows.
- Image tokens are not interleaved in a way that violates processor ordering.
  Arbitrary positions are okay if row-major `nonzero` order is used.
- Dtype/device cast to token embedding dtype is preserved.

Failure cases:

- Missing `input_ids`, duplicate embedding-equality path, or mismatched count.

Parity test sketch:

- Prompts with image tokens at prefix, middle, and suffix positions; compare
  row-copy output to `masked_scatter`.

### Rewrite: GQA attention without materialized `repeat_kv`

Source pattern:

```text
key/value [B, 8, K, 128] -> repeat_kv(..., n_rep=2) -> attention with Q [B,16,Q,128]
```

Replacement:

```text
Grouped-query attention kernel maps query_head // 2 to KV head.
```

Preconditions:

- `num_attention_heads % num_key_value_heads == 0`.
- No attention backend requires materialized repeated KV in observable outputs.
- Cache stores unrepeated KV.

Failure cases:

- Debug attentions requiring exact repeated weight shape can be deferred.

Parity test sketch:

- Compare eager repeated-KV attention to grouped kernel for prefill and decode.

### Rewrite: last-token-only logits

Source pattern:

```text
logits = lm_head(hidden_states[:, slice(-logits_to_keep, None), :])
```

Replacement:

```text
When generation requests `logits_to_keep=1`, run lm_head only on final token.
```

Preconditions:

- No loss computation.
- Sampling only needs final logits.

Failure cases:

- Full-sequence logits, tensor `logits_to_keep`, or training labels.

## 10. Kernel fusion candidates

Highest priority:

- Qwen3 RMSNorm and q/k head RMSNorm: repeated 28 layers, fp32 variance with
  reduced-precision output.
- Qwen3 GQA FlashAttention with RoPE and KV cache: dominant text prefill/decode
  cost; must support attention width 2048 with hidden width 1024.
- SwiGLU MLP fusion: `silu(gate) * up -> down` for both Pixtral and Qwen3.
- Image row-copy stitch replacing generic boolean scatter under guards.
- Last-token-only LM head for generation.

Medium priority:

- Pixtral patch Conv2d lowered to GEMM or optimized non-overlap patch kernel.
- Pixtral noncausal block attention over concatenated image patches.
- Patch merger gather/unfold + merger GEMM fusion.
- Projector GELU MLP fusion.
- RoPE table precompute and Q/K application fusion.

Lower priority:

- CPU processor acceleration; useful for throughput but outside first GPU graph.
- Full logits and training loss.
- Sliding-window Qwen3 attention, since inspected LightOn configs disable it.
- Generic `inputs_embeds`-only image placeholder matching path.

## 11. Runtime staging plan

Stage 1: config and routing admission.

- Accept explicit current-source `lighton_ocr` configs.
- Add a compatibility gate for public `model_type=mistral3` LightOnOCR configs,
  or reject with a clear message until the Mistral3 route is separately audited.
- Parse nested Pixtral and Qwen3 configs; validate hidden/head geometry.

Stage 2: processor ABI and vision/projector parity.

- Reproduce `pixel_values`/`image_sizes` contract using saved preprocessor configs.
- Run Pixtral encoder and LightOn projector for one image and multiple image
  sizes.
- Stub text decode by returning image feature checksums if necessary.

Stage 3: bounded image-token stitch + Qwen3 prefill.

- Implement ordered row-copy stitch for `input_ids` path.
- Compile Qwen3 full-attention prefill without sliding-window support.
- Validate logits for short prompts and one image.

Stage 4: decode with KV cache.

- Add per-layer unrepeated KV cache `[B, 8, T, 128]`.
- Ensure LightOn generation only forwards `pixel_values` on first iteration.
- Support last-token logits.

Stage 5: optimized kernels.

- Enable GQA FlashAttention/SDPA backend, RMSNorm, SwiGLU, patch Conv2d rewrite,
  patch merger rewrite, and last-token LM head.

Stage 6: checkpoint variants.

- Admit 0.9B vocab variants and bbox-tuned checkpoint once tokenizer/generation
  ABI and output parsing expectations are documented.

## 12. Parity and validation plan

- Processor parity:
  - Fixed images with odd aspect ratios; compare resized `image_sizes`,
    padded `pixel_values` shape, and image-token expansion count.
  - Check prompt with multiple image placeholders consumes image sizes in order.
- Pixtral patch embedding:
  - Random NCHW inputs with padding; compare Conv2d and patch sequence after
    `image_sizes` cropping.
- Pixtral attention block:
  - One-layer parity with eager attention and block mask for two images in one
    batch.
- Pixtral full encoder:
  - Compare `last_hidden_state` for one page and two pages.
- Patch merger/projector:
  - Synthetic patch grids to prove 2x2 row-major merge order, then random
    feature parity through projector.
- Stitch:
  - Compare guarded row-copy against `masked_scatter` for prefix/middle/suffix
    image tokens.
- Qwen3 block:
  - Single-layer prefill parity with q/k RMSNorm, RoPE, GQA and causal mask.
  - Single-token decode parity with a populated cache.
- End-to-end:
  - One real document image from HF examples, greedy or fixed sampling disabled
    where possible, compare logits/token sequence against Transformers.

Suggested tolerances:

- fp32 component tests: `rtol=1e-4`, `atol=1e-5`.
- bf16/fp16 fused attention and MLP tests: start with `rtol=2e-2`, `atol=2e-2`,
  tighten per kernel after matching accumulation policy.

## 13. Performance probes

- CPU preprocessing throughput by image resolution and batch size.
- Vision encoder throughput by total patch count and number of images per batch.
- Patch merger/projector throughput by merged token count.
- Prefill throughput by text length plus image token count.
- Decode tokens/sec with and without cached image prefix.
- KV cache memory: 28 layers x 2 tensors x `[B, 8, T, 128]` x dtype bytes.
- Pixtral block attention scaling by page resolution; compare eager/SDPA/flash
  strategies and block-mask overhead.
- Last-token LM head versus full logits for vocab 151936, 32768, and 16384.
- Image stitch implementation: boolean scatter versus indexed row copy.
- GGUF/dequant provider experiments only if using non-Transformers converted
  checkpoints; keep separate from this native-source audit.

## 14. Skip/defer list

- Training, labels/loss, gradient checkpointing.
- Qwen3 sequence/token classification and QA heads.
- Sliding-window attention for LightOn OCR until a checkpoint enables it.
- `inputs_embeds`-only placeholder matching by embedding equality.
- Generic boolean scatter; use guarded row-copy for first integration.
- Full structured OCR postprocessing, NMS, mask decoding, or bbox coordinate
  conversion; not present in inspected source.
- Multi-GPU tensor parallel plans.
- Remote-code or older Mistral3 compatibility beyond a clear route/reject gate.
- Quantized/GGUF community checkpoints as native model behavior.

## 15. Final implementation checklist

- [ ] Add LightOn OCR config parser with nested Pixtral/Qwen3 validation.
- [ ] Gate public `model_type=mistral3` LightOnOCR configs separately from native `lighton_ocr`.
- [ ] Load token embedding, lm_head tied-weight alias, vision, projector, and Qwen3 weights.
- [ ] Implement Pixtral image processor ABI or define CPU-pipeline boundary.
- [ ] Implement NCHW patch Conv2d and patch-sequence cropping by `image_sizes`.
- [ ] Implement Pixtral 2D RoPE table and position-id generation.
- [ ] Implement Pixtral block-diagonal noncausal attention.
- [ ] Implement LightOn RMSNorm + 2x2 patch merger + projector.
- [ ] Implement guarded image-token row-copy stitch.
- [ ] Implement Qwen3 RMSNorm, q/k head RMSNorm, RoPE, GQA causal attention, and KV cache.
- [ ] Implement Qwen3 SwiGLU MLP and final norm.
- [ ] Implement generation prefill/decode with first-iteration image forwarding.
- [ ] Implement last-token-only LM head path.
- [ ] Add parity tests for processor, Pixtral block, projector, stitch, Qwen3 block, cache decode, and end-to-end logits.
- [ ] Benchmark processor, vision/projector, prefill, decode, LM head, and cache memory.
- [ ] Revisit bbox-tuned output parsing only after text-generation parity lands.

## Gated gaps for DinoML

- Config routing drift: current native source and public checkpoint metadata do
  not name the same `model_type`.
- Operator gap: Pixtral block attention and patch merger need bounded layout and
  dynamic-image-size guards.
- Operator gap: Qwen3 attention width differs from hidden width, so GEMM shapes
  and cache ABI must use explicit `head_dim`.
- Runtime gap: generic `masked_scatter` should not be admitted; lower only the
  guarded image-row-copy pattern.
- Runtime gap: KV cache and generation controller must preserve first-iteration
  image handling.
- Layout gap: NCHW image graph is semantic source; NHWC/channel-last rewrites
  require explicit guards around patch conv, patch sequence order, and patch
  merger axes.
- Product gap: structured OCR/bbox outputs are text-only in source; no native
  bbox postprocess exists to validate yet.
