# Transformers audit: kosmos2_5

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: microsoft/kosmos-2.5
Config source: Hugging Face config.json plus Transformers config defaults
Primary runtime target: Kosmos2_5ForConditionalGeneration, multimodal OCR/markdown generation
Dinoml assumptions: inference-only first; CUDA target; keep source PyTorch axes for initial parity; treat image preprocessing as CPU/data-pipeline unless explicitly fused later.
```

Source files inspected from `X:/H/transformers/src/transformers/models/kosmos2_5/`:

- `configuration_kosmos2_5.py`
- `modeling_kosmos2_5.py`
- `processing_kosmos2_5.py`
- `image_processing_kosmos2_5.py`
- `image_processing_pil_kosmos2_5.py`
- `convert_kosmos2_5.py`

Representative config snapshots saved in this folder:

- `microsoft__kosmos-2.5__config.json`, `preprocessor_config.json`, `generation_config.json`
- `microsoft__kosmos-2.5-chat__config.json`, `preprocessor_config.json`, `generation_config.json`
- `merve__kosmos-2.5-ft__config.json`, `generation_config.json`
- `textum-org__kosmos2.5__config.json`, `preprocessor_config.json`, `generation_config.json`
- `Fireblossom__kosmos-2.5-4bit-vision__config.json`
- `Fireblossom__kosmos-2.5-4bit-text__config.json`, `generation_config.json`

Hub links inspected:

- [microsoft/kosmos-2.5](https://huggingface.co/microsoft/kosmos-2.5)
- [microsoft/kosmos-2.5-chat](https://huggingface.co/microsoft/kosmos-2.5-chat)
- [merve/kosmos-2.5-ft](https://huggingface.co/merve/kosmos-2.5-ft)
- [textum-org/kosmos2.5](https://huggingface.co/textum-org/kosmos2.5)
- [Fireblossom/kosmos-2.5-4bit-vision](https://huggingface.co/Fireblossom/kosmos-2.5-4bit-vision)
- [Fireblossom/kosmos-2.5-4bit-text](https://huggingface.co/Fireblossom/kosmos-2.5-4bit-text)
- [kirp/kosmos2_5](https://huggingface.co/kirp/kosmos2_5) returned HTTP 401 for raw config files; access would be needed to verify whether it is a true variant or a private mirror.

Any missing files or assumptions:

- No remote-code files are required for the native source basis.
- Only Microsoft/full mirrors expose the full multimodal architecture; Fireblossom repos are split vision/text submodule checkpoints and are not a separate full multimodal runtime target.
- Tokenizer coupling was checked only for runtime-significant special tokens from `tokenizer_config.json`: BOS `<s>` id 0, PAD id 1, EOS id 2, `<image>` id 100283, `</image>` id 100284, `<ocr>` id 100282, `<md>` id 100288. The full tokenizer file was not snapshotted because it is large.

## 2. High-level architecture

KOSMOS-2.5 is a multimodal literate model:

```text
image -> CPU image patch extraction -> vision encoder -> L2 normalize -> latent-query projection
text prompt with 2048 image placeholders -> token embeddings -> image embedding stitch -> causal text decoder -> LM logits -> generated OCR/markdown/bbox tokens
```

Stage decomposition:

- CPU/data pipeline: RGB conversion, per-image normalization, resize to a patch grid bounded by `max_patches`, non-overlapping 16x16 patch extraction, row/column id prefixing, padding to 4096 patches, tokenizer prompt construction.
- Vision encoder: Pix2Struct/T5-style patch sequence encoder over `flattened_patches`.
- Projector: dense projection from vision width to text width, learned latent queries, noncausal cross-attention from latent queries to `[vision_tokens, latent_queries]`.
- Prefix construction: processor emits `bos + <image> + <s> * 2048 + </image> + text`; model replaces only mask value `1` positions with projected image embeddings.
- Prefill: causal text decoder consumes full prompt plus image embeddings and builds self-attention KV cache.
- Decode: subsequent cached steps should not re-run vision or embedding stitch; generated tokens use the text decoder cache.

Independently stageable pieces:

- Vision encoder can be validated from `flattened_patches -> last_hidden_state`.
- Image-to-text projection can be validated from normalized vision features -> `[B, 2048, 1536]`.
- Text-only causal LM can be validated with `image_embeds=None`.
- Full multimodal prefill validates processor masks, image embedding insertion, and logits.
- Cached decode validates `prepare_inputs_for_generation` position-id offset behavior and no repeated vision execution.

## 3. Important config dimensions

Primary Microsoft config values are from `microsoft/kosmos-2.5/config.json`; defaults are from `configuration_kosmos2_5.py`.

| Field | Value | Source |
|---|---:|---|
| `latent_query_num` | 2048 | config/default |
| text `vocab_size` | 108481 | config/default |
| text `embed_dim` / hidden size | 1536 | config/default |
| text layers | 24 | config/default |
| text attention heads | 16 | config/default |
| text head dim | 96 | inferred from source: `embed_dim // attention_heads` |
| text FFN dim | 6144 | config/default |
| text activation | `gelu` | config/default |
| text max positions | 4096 | config/default |
| text cache | `use_cache=True` | config/default |
| text embedding scale | `sqrt(embed_dim)` | config/default `scale_embedding=True` |
| vision hidden size | 1536 | config/default |
| vision patch input width | 768 | config/default; `2 + 16*16*3 = 770` arrives, first two are row/col ids |
| vision layers | 18 | config/default |
| vision attention heads | 24 | config/default |
| vision head dim | 64 | config/default |
| vision MLP intermediate | 3968 | config/default |
| vision activation | `gelu_new` | config/default |
| vision max patches | 4096 | config/default |
| patch size | 16x16 | image processor default |
| generation | greedy, BOS 0, EOS 2, PAD 1 | generation config |

Representative checkpoint sweep:

| Checkpoint | Scope | Architecture | Dtype metadata | Operator-significant notes |
|---|---|---|---|---|
| `microsoft/kosmos-2.5` | official full model | `Kosmos2_5ForConditionalGeneration` | top-level `float32`; examples use bf16/fp16 | canonical full multimodal config |
| `microsoft/kosmos-2.5-chat` | official full chat variant | same | similar metadata | same dimensions/operator graph; prompt style may differ outside model graph |
| `textum-org/kosmos2.5` | mirror | same | same as official | no operator variation found in config |
| `merve/kosmos-2.5-ft` | fine-tune/mirror | same | same dimensions | no preprocessor config present; top-level `tie_word_embeddings` omitted but text config default/source ties LM head |
| `Fireblossom/kosmos-2.5-4bit-vision` | split vision submodule | `Kosmos2_5VisionModel` | `float16` | config uses historical aliases `d_ff`, `d_kv`, `seq_len`; native `strict` config ignores/rejects unknown aliases unless mapped before load |
| `Fireblossom/kosmos-2.5-4bit-text` | split text submodule | `Kosmos2_5TextForCausalLM` | `float16` | text-only graph; no multimodal projector or image stitch |

## 3a. Family variation traps

- The full architecture is not encoder-decoder cross-attention generation. The text decoder has only causal self-attention; image conditioning is injected by replacing placeholder token embeddings before the decoder.
- Vision attention head dim is explicitly configured as 64, so `hidden_size == num_heads * head_dim` happens to hold as `1536 == 24 * 64`. Text head dim is not a config field; source computes `1536 // 16 = 96`.
- Vision patch input has row/column ids in the first two feature slots. The learned patch projection consumes only the remaining 768 pixel features.
- The image processor pads `flattened_patches` to 4096, but the projector emits exactly 2048 latent image embeddings. The processor prompt must reserve exactly 2048 replacement positions unless `num_image_tokens` and `latent_query_num` are changed together.
- `image_embeds_position_mask` uses values `[0, -1, 1..., -1, 0...]`. Only `1` triggers embedding replacement; any nonzero value selects segment embedding id 1.
- The text source applies query scaling before calling the attention backend and passes backend scaling `1.0`. Vision attention passes unscaled Q and backend scaling `head_dim^-0.5`.
- Text FFN is not a standard two-linear GELU MLP only: it applies `fc1 -> activation -> dropout -> LayerNorm(ffn_dim) -> fc2 -> dropout`.
- Vision MLP is gated: `act(wi_0(x)) * wi_1(x) -> wo`, all bias-free.
- `Kosmos2_5LayerNorm` in vision is RMSNorm/T5-style with no mean subtraction and no bias; text uses ordinary `nn.LayerNorm`.
- The model has no RoPE, ALiBi, or learned absolute text position table. Text positions are sinusoidal with a padding offset.
- Layout pass trap: processor and model source operate on NCHW images for resize/unfold, then flatten to sequence. Initial graph parity should preserve these axes; NHWC/channel-last is a guarded fusion opportunity only inside controlled preprocessing/patch extraction.
- The Fireblossom 4-bit repos are split checkpoints with quantization implied by repo name/metadata, not source-coupled packed weight logic in the native modeling file. DinoML should handle them as loading-policy variants or reject until a quantized weight contract is explicit.

## 4. Operator coverage checklist

Tensor/layout ops:

- `reshape`, `view`, `permute`, `transpose`, `contiguous`
- `slice`/last-dim feature split for row/column ids and patch payload
- `cat` along sequence dimension for projector key/value source
- `expand` learned latent queries over batch
- boolean/equality masks, `ne`, `sum(dim=-1)`, `cumsum(dim=1)`, `index_select`
- indexed row update for image embedding stitch: `inputs_embeds[mask == 1] = image_embeds.reshape(-1, D)`
- `normalize(..., dim=-1)` L2 normalization on vision features

Neural primitives:

- Embedding lookup: token embeddings `[108481, 1536]`, segment embedding `[2, 1536]`, vision row/column embeddings `[4096, 1536]`.
- Dense/GEMM:
  - vision patch projection `Linear(768 -> 1536)` with bias
  - vision attention Q/K/V `Linear(1536 -> 1536)` bias-free, out `1536 -> 1536` bias-free
  - vision gated MLP `wi_0/wi_1: 1536 -> 3968` bias-free, `wo: 3968 -> 1536` bias-free
  - image projector dense `Linear(1536 -> 1536)` with bias
  - projection cross-attention Q/K/V/O `1536 -> 1536` with bias
  - text self-attention Q/K/V/O `1536 -> 1536` with bias
  - text FFN `Linear(1536 -> 6144)` with bias, `Linear(6144 -> 1536)` with bias
  - LM head `Linear(1536 -> 108481)` bias-free, tied to token embeddings
- Norms: RMSNorm/T5-style for vision; LayerNorm for text and text FFN intermediate.
- Activations: `gelu_new` in vision, `gelu` in text.
- Dropout is source-present but inference should constant-fold/remove.

Attention primitives:

- Vision noncausal self-attention MHA, 24 heads, head dim 64, padding mask.
- Projector noncausal cross-attention-like MHA, 16 heads, head dim 96, query length 2048, key/value length `vision_seq + 2048`.
- Text causal self-attention MHA, 16 heads, head dim 96, DynamicCache support.
- Backend dispatch can use Transformers attention interface / SDPA when available; eager path defines exact softmax upcast behavior.

Position and custom math:

- Vision row/column learned embeddings from processor-generated ids offset by 1, zero for padded rows/cols.
- Text sinusoidal positional embeddings with `padding_idx=1` and offset 2.
- Position ids from input ids use non-pad cumulative count plus padding index.

Preprocessing-coupled ops:

- Per-image mean/std normalization across all non-batch dims.
- Bilinear resize with `align_corners=False`, `antialias=True`.
- Non-overlapping `unfold` with kernel and stride 16.
- Row/column id feature construction and patch padding to `max_patches`.

Generation/cache ops:

- Dynamic self-attention KV cache per text layer.
- `logits_to_keep` slicing before LM head.
- `prepare_inputs_for_generation` suppresses image inputs after cache is populated and adjusts position ids by `1 + pad_token_id`.

Postprocessing:

- Neural graph emits token logits only. OCR/markdown/bbox rendering is tokenizer/generation-controller postprocessing. Bounding boxes are represented as generated special tokens such as `<bbox>`, `<x_...>`, `<y_...>`, not a detector head with NMS.

## 5. Layer/block breakdown

Image preprocessing:

```text
image [B, C, H, W]
normalize per image over C,H,W
resize to rows*16, cols*16 with rows*cols <= 4096
unfold non-overlap 16x16
patches [B, rows*cols, 768]
prepend row_id, col_id -> [B, rows*cols, 770]
pad sequence -> flattened_patches [B, 4096, 770]
attention_mask [B, 4096]
```

Vision embedding:

```text
row_ids = flattened_patches[:, :, 0].long()
col_ids = flattened_patches[:, :, 1].long()
patch = flattened_patches[:, :, 2:]                # [B, 4096, 768]
x = Linear(768 -> 1536)(patch) + RowEmbed(row_ids) + ColEmbed(col_ids)
```

Vision block, repeated 18 times:

```text
residual = x
x = RMSNorm(x)
a = MHA_noncausal(x, mask)                         # Q/K/V/O bias=False
x = residual + a
residual = x
m = RMSNorm(x)
m = gelu_new(Linear(1536 -> 3968, bias=False)(m)) *
    Linear(1536 -> 3968, bias=False)(m)
m = Linear(3968 -> 1536, bias=False)(m)
x = residual + m
```

Vision output:

```text
vision = RMSNorm(x)
vision = l2_normalize(vision, dim=-1)
```

Image-to-text projection:

```text
features = Linear(1536 -> 1536)(vision)
latent = learned [2048, 1536].expand(B, 2048, 1536)
kv = cat([features, latent], dim=1)                 # [B, 4096+2048, 1536] usually
image_embeds = noncausal_attention(query=latent, key/value=kv)
```

Text embedding and image stitch:

```text
tok = Embedding(input_ids)                          # [B, S, 1536]
tok[image_embeds_position_mask == 1] = image_embeds.reshape(-1, 1536)
tok = tok * sqrt(1536)
pos = sinusoidal_position(input_ids or inputs_embeds)
pos += segment_embedding(image_embeds_position_mask.ne(0).long())
x = tok + pos
```

Text decoder block, repeated 24 times:

```text
residual = x
x = LayerNorm(x)
a = causal_self_attention(x, causal_mask, KV cache) # Q/K/V/O all bias=True
x = residual + dropout(a)
residual = x
x = LayerNorm(x)
m = gelu(Linear(1536 -> 6144)(x))
m = LayerNorm(6144)(m)
m = Linear(6144 -> 1536)(m)
x = residual + dropout(m)
```

LM head:

```text
x = final LayerNorm(x)
logits = tied_lm_head(x[:, slice(-logits_to_keep, None), :])
```

## 6. Attention requirements

Vision encoder attention:

- Noncausal self-attention.
- MHA, 24 query heads, 24 KV heads, head dim 64.
- Query/key/value width 1536.
- Sequence length up to 4096 padded patches.
- Mask source is patch attention mask; expanded eager mask shape is `[B, 1, T, S]`.
- No KV cache, no RoPE, no ALiBi, no local/sliding window.

Projection attention:

- Query-driven noncausal attention, not autoregressive decode.
- Query source: learned latent query table `[2048, 1536]` expanded over batch.
- Key/value source: `cat([dense(vision_features), latent_query], dim=1)`.
- MHA, 16 heads, head dim 96.
- Query length 2048; key/value length `num_valid_or_padded_patches + 2048`, source uses full padded vision sequence unless optimized with masks.
- No mask in source call.
- Output `[B, 2048, 1536]` is independently cacheable for a given image.

Text decoder attention:

- Causal self-attention only.
- MHA, 16 heads, 16 KV heads, head dim 96; no GQA/MQA.
- Query/key/value width 1536.
- Source scales query before attention backend and passes `scaling=1.0`.
- Mask is created by Transformers `create_causal_mask` from config, attention mask, inputs, and cache.
- DynamicCache updates projected keys/values per layer. Cached keys/values are stored after linear projection and before any nonexistent position encoding.
- During cached decode, `prepare_inputs_for_generation` clears image embeddings and image mask once `past_key_values.get_seq_length() > 0`.

FlashAttention/SDPA compatibility:

- Source uses `ALL_ATTENTION_FUNCTIONS` for selected attention implementation.
- Eager parity requires softmax computed in fp32 and cast back to query dtype.
- DinoML first path can use dense attention for parity; optimized path should provide three attention modes: noncausal vision, noncausal latent projection, causal cached decoder.

## 7. Position encoding and custom math

Text sinusoidal embeddings:

```python
def kosmos25_sinusoidal(num_embeddings, dim, padding_idx=1):
    half = dim // 2
    inv = exp(arange(half) * -(log(10000) / (half - 1)))
    phase = arange(num_embeddings)[:, None] * inv[None, :]
    emb = cat([sin(phase), cos(phase)], dim=1)
    if dim % 2:
        emb = cat([emb, zeros(num_embeddings, 1)], dim=1)
    emb[padding_idx] = 0
    return emb
```

Position ids from token ids:

```python
mask = (input_ids != pad_token_id).int()
position_ids = (cumsum(mask, dim=1) + past_key_values_length) * mask
position_ids = position_ids.long() + pad_token_id
```

Vision RMSNorm:

```python
variance = mean(float32(x) * float32(x), dim=-1, keepdim=True)
y = x * rsqrt(variance + eps)
y = weight * cast_if_needed(y)
```

Precomputable:

- Sinusoidal table up to the static max plus offset, extendable for longer positions.
- Vision row/column embedding weights are learned, but row/column ids come from preprocessing.
- Learned latent queries are constants and can be cached expanded only by batch shape.

Dynamic:

- Position ids depend on padding and decode step.
- Image row/column ids depend on resized patch grid.
- Projection output depends on image features and should be cached across decode for the same image/prompt prefill.

## 8. Preprocessing and input packing

Processor contract:

- Images are required by `Kosmos2_5Processor`; text-only generation should route to `Kosmos2_5TextForCausalLM` or bypass the processor.
- Default image output: `flattened_patches`, `attention_mask`, `width`, `height`; processor removes `rows` and `cols` before model input.
- `flattened_patches` shape is `[B, max_patches, 2 + 16*16*3]`, normally `[B, 4096, 770]`.
- `attention_mask` is reused name-wise: image processor emits patch attention mask; processor later overwrites/updates with tokenizer attention mask in the combined encoding. For the full model call, the public `attention_mask` passed to text model is the text/token mask, while vision model recomputes patch mask if not passed.
- Text prompt prefix is:

```text
<s><image><s> repeated 2048 times</image>
```

- `image_embeds_position_mask`:

```text
[0, -1] + [1] * 2048 + [-1] + [0] * remaining_text_positions
```

- Replacement guard: number of `mask == 1` positions must equal `B * latent_query_num`; source relies on PyTorch assignment shape failure if not.
- Segment embedding guard: all nonzero mask values become segment id 1, so `<image>`, image placeholder positions, and `</image>` get image segment embedding.

CPU/data-pipeline work for first integration:

- PIL/torch image decode and RGB conversion.
- Per-image normalization.
- Resize and patch extraction.
- Tokenization and special-token construction.
- OCR/markdown/bbox token decoding.

GPU/runtime candidates after parity:

- Patch extraction and resize can be fused only if DinoML controls the processor-to-vision boundary.
- Image embedding stitch can lower to a bounded contiguous row copy because processor-generated mask positions are a known contiguous block after BOS and `<image>`.

## 9. Graph rewrite / lowering opportunities

### Rewrite: contiguous image placeholder scatter -> indexed row copy

Source pattern:

```text
inputs_embeds = clone(token_embeddings)
inputs_embeds[image_embeds_position_mask == 1] = image_embeds.reshape(-1, D)
```

Replacement:

```text
copy image_embeds[B, 2048, D] into token embeddings[:, start:start+2048, :]
```

Preconditions:

- Processor-created prompt is used.
- Mask equals `[0, -1] + [1] * latent_query_num + [-1] + zeros`.
- `latent_query_num == num_image_tokens`.
- No arbitrary user-provided `image_embeds_position_mask`.

Shape equations:

- `inputs_embeds: [B, S, 1536]`
- `image_embeds: [B, 2048, 1536]`
- destination slice starts at token index 2.

Failure cases:

- Custom prompt with multiple image regions.
- Non-contiguous `mask == 1`.
- Mismatch between placeholder count and projected latent count.

Parity test sketch:

- Compare PyTorch masked assignment with indexed slice copy for batch 1/2, variable text suffix length, and malformed masks that must reject.

### Rewrite: patch extraction -> non-overlap im2col/linear input

Source pattern:

```text
interpolate NCHW -> unfold(kernel=16, stride=16) -> permute -> reshape -> Linear(768 -> 1536)
```

Replacement:

```text
controlled preprocessing: Patchify16x16NCHW -> GEMM(weight.T) + row/col embeddings
```

Preconditions:

- `patch_size == stride == 16`.
- `padding == 0`, `dilation == 1`.
- Image resized dimensions are multiples of 16.
- Source layout is NCHW and flatten order matches `unfold -> reshape -> permute`.

Weight transform:

```python
w = patch_projection.weight  # [1536, 768]
```

Layout constraints:

- Initial semantic lowering should preserve NCHW.
- NHWC path requires a patch feature permutation or transformed weights.

Failure cases:

- Different patch size.
- Processor bypassed with caller-supplied `flattened_patches`.

Parity test sketch:

- Random images over several aspect ratios, compare `flattened_patches[:, :, 2:]` and post-projection outputs.

### Rewrite: vision gated MLP fusion

Source pattern:

```text
gelu_new(wi_0(x)) * wi_1(x) -> wo
```

Replacement:

```text
two GEMMs -> fused activation/multiply -> GEMM
```

Preconditions:

- `dense_act_fn == gelu_new`.
- Bias-free `wi_0`, `wi_1`, `wo`.
- Same input tensor and same intermediate size 3968.

Failure cases:

- Config changes activation.
- Quantized or split weight format without dense materialization.

### Rewrite: last-token-only logits

Source pattern:

```text
logits = lm_head(hidden_states[:, slice_indices, :])
```

Replacement:

```text
if logits_to_keep == 1 during decode: GEMM only for final token
```

Preconditions:

- Generation decode/prefill caller requests `logits_to_keep=1`.
- Loss is not computed.

Failure cases:

- Full logits requested for training/evaluation.
- Tensor-valued `logits_to_keep` with non-contiguous indices.

## 10. Kernel fusion candidates

Highest priority:

- Text causal attention with KV cache: dominant decode path; preserve query pre-scaling and fp32 softmax parity.
- Vision encoder attention over up to 4096 patches: dominant prefill/image cost; noncausal dense attention can use a separate optimized path.
- Projection latent attention: fixed 2048 latent queries over large image+latent KV; independently cacheable and expensive.
- GEMM + bias for QKV/O/FFN and LM head: existing CUTLASS GEMM path is directly relevant.
- Image embedding stitch as bounded row copy: avoids admitting general boolean scatter.

Medium priority:

- Vision RMSNorm and text LayerNorm kernels.
- Vision gated MLP activation multiply fusion.
- Text FFN `GELU -> LayerNorm(ffn_dim)` fusion around intermediate normalization.
- Patch projection after patchify as GEMM.
- Last-token-only logits for decode.

Lower priority:

- CPU/GPU fused image preprocessing. Useful only after model parity because tokenizer/image pipeline likely remains CPU-owned initially.
- Attention mask expansion fusion. Can be folded into attention kernels once mask contracts are stable.
- Quantized split checkpoint loading. Needs a separate storage/provider contract; not required for the official dense source basis.

## 11. Runtime staging plan

Stage 1: config and dense weight loading.

- Parse nested `Kosmos2_5Config`, `Kosmos2_5TextConfig`, `Kosmos2_5VisionConfig`.
- Preserve tied token embedding / LM head alias.
- Reject unsupported split/quantized checkpoint configs unless mapped explicitly.

Stage 2: text-only causal LM parity.

- Implement embeddings, sinusoidal positions, segment embedding default path, 24 decoder blocks, cache, and LM head.
- Validate `Kosmos2_5TextForCausalLM` without image inputs.

Stage 3: vision encoder parity.

- Treat `flattened_patches` as runtime input.
- Implement patch projection, row/column embeddings, RMSNorm, noncausal vision attention, gated MLP.

Stage 4: projector parity.

- Implement L2 normalize, dense projection, learned latent queries, noncausal latent attention.
- Cache image embeddings as a reusable prefix artifact for a fixed image.

Stage 5: full multimodal prefill parity.

- Add processor-compatible image embedding stitch.
- Validate prompt prefix, mask semantics, and first-token logits.

Stage 6: cached decode parity.

- Ensure vision/projector are skipped after prefill.
- Validate position-id offset and DynamicCache-compatible self-attention state.

Stage 7: optimized kernels/fusions.

- Enable fused attention variants, row-copy stitch, gated MLP fusion, last-token logits, and optional processor fusion.

Can be stubbed initially:

- Training loss and labels.
- Dropout.
- Output attentions/hidden states beyond debug parity.
- OCR/markdown formatting helpers beyond tokenizer decode.

## 12. Parity and validation plan

Concrete tests:

- Config round-trip: official full config, chat config, split text/vision configs, and malformed alias fields from 4-bit configs.
- Custom math: RMSNorm fp32/fp16/bf16 tolerance; sinusoidal table and position id generation with padding and decode offset.
- Image processor contract: random image aspect ratios, verify resized dimensions, row/col ids, patch flatten order, padding mask, and `[B, 4096, 770]`.
- Vision single-layer parity: one layer with random `flattened_patches` and mask.
- Vision full-encoder parity: 18 layers at reduced batch/sequence first, then max patch count.
- Projector parity: normalized features -> image embeds and attention output.
- Embedding stitch parity: source masked assignment versus bounded slice copy under processor mask; rejection for malformed masks.
- Text single-layer and full decoder parity without image embeds.
- Prefill logits parity for official prompt `<ocr>` and `<md>` with fixed image.
- Decode parity: one-step and multi-step cached decode, comparing generated token ids under greedy generation.
- End-to-end smoke: receipt image example output prefix/token sequence, not exact long-text string as the first gate.

Recommended tolerances:

- fp32: absolute/relative around `1e-4` for block outputs; logits may need `2e-4`.
- fp16/bf16: compare with Transformers in same dtype, `1e-2` class tolerance for deep block/logit paths; tighten per-kernel tests separately.

## 13. Performance probes

- CPU preprocessing throughput: images/sec by resolution/aspect ratio.
- Patch count sweep: rows*cols near 512, 1024, 2048, 4096.
- Vision encoder throughput and memory at batch 1/2/4 and max patches.
- Projection attention throughput: query length 2048, KV length 4096+2048.
- Text prefill throughput by prompt length including 2051 image prefix tokens plus generated text prompt.
- Decode tokens/sec with cached image prefix, batch sweep.
- KV cache memory: 24 layers, `[B, heads=16, seq, head_dim=96]` for K and V.
- Last-token LM head cost for vocab 108481.
- Attention backend comparison: eager dense, SDPA/Flash-compatible, DinoML fused.
- Image embedding cache benefit: full image+prefill versus reuse projected image embeds for repeated prompts.
- Quantized loading probe only after a DinoML weight-storage policy exists for split/4-bit repos.

## 14. Skip/defer list

- Training loss and gradient checkpointing.
- Beam search, sampling processors, and chat prompt policies beyond greedy generation parity.
- General boolean scatter for arbitrary image masks; use guarded row-copy first.
- Full GPU image processor; keep CPU/data pipeline initially.
- Split/4-bit checkpoint support until explicit quantized storage metadata is audited.
- Multi-image or multiple image-region prompts; source processor only constructs one contiguous image block.
- Tensor parallel/distributed execution.
- Output attentions and hidden-state recording for production path.
- NMS/detector postprocessing; KOSMOS-2.5 emits bbox tokens, not detector tensors.

## 15. Final implementation checklist

- [ ] Parse nested `Kosmos2_5Config` and representative checkpoint variants.
- [ ] Load dense weights and preserve token embedding / LM head tying.
- [ ] Implement text sinusoidal positions and padding-aware position ids.
- [ ] Implement text decoder block with causal MHA, LayerNorm, FFN intermediate LayerNorm, and KV cache.
- [ ] Implement `logits_to_keep` LM head slicing.
- [ ] Implement vision `flattened_patches` ABI with row/column embeddings.
- [ ] Implement vision RMSNorm, noncausal attention, and gated MLP.
- [ ] Implement image feature L2 normalization and latent-query projection attention.
- [ ] Implement processor-compatible image embedding stitch as guarded row copy.
- [ ] Add CPU/data-pipeline adapter or importer for `flattened_patches`, `input_ids`, `attention_mask`, and `image_embeds_position_mask`.
- [ ] Add text-only single-layer/full-model parity tests.
- [ ] Add vision encoder and projector parity tests.
- [ ] Add full multimodal prefill logits parity.
- [ ] Add cached decode token parity.
- [ ] Benchmark preprocessing, vision, projector, prefill, decode, LM head, and KV memory separately.
