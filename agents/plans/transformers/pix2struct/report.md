# Pix2Struct Transformers audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: google/pix2struct-base as the source-default reference; sweep also covers google/pix2struct-large, google/pix2struct-textcaps-base, google/pix2struct-docvqa-base, google/pix2struct-screen2words-base, and google/deplot.
Config source: Hugging Face raw config.json, preprocessor_config.json, tokenizer_config.json, and model repo metadata fetched 2026-05-13.
Source files inspected:
- X:/H/transformers/src/transformers/models/pix2struct/configuration_pix2struct.py
- X:/H/transformers/src/transformers/models/pix2struct/modeling_pix2struct.py
- X:/H/transformers/src/transformers/models/pix2struct/processing_pix2struct.py
- X:/H/transformers/src/transformers/models/pix2struct/image_processing_pix2struct.py
- X:/H/transformers/src/transformers/models/pix2struct/image_processing_pil_pix2struct.py
Any missing files or assumptions: no remote code is required for the inspected checkpoints. No generation_config.json is present in the sampled repos; generation defaults come from model/config/generation mixin behavior. Tokenizer is T5Tokenizer with SentencePiece files in the repos.
```

Primary raw source URLs:

- https://github.com/huggingface/transformers/tree/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/pix2struct
- https://huggingface.co/google/pix2struct-base/raw/main/config.json
- https://huggingface.co/google/pix2struct-base/raw/main/preprocessor_config.json
- https://huggingface.co/google/pix2struct-large/raw/main/config.json
- https://huggingface.co/google/pix2struct-docvqa-base/raw/main/preprocessor_config.json

## 2. High-level architecture

Pix2Struct is an image/screenshot-to-text encoder-decoder. The vision side consumes pre-flattened image patches, not raw image tensors. The decoder is T5-like: autoregressive text self-attention with a relative-position bias in layer 0, per-layer cross-attention over vision encoder tokens, gated GELU feed-forward blocks, and an untied LM head.

```text
CPU/data preprocessing -> flattened_patches + patch attention_mask
  -> vision patch projection + row/column embeddings
  -> noncausal vision encoder
  -> autoregressive text decoder prefill/decode with cross-attention
  -> LM logits -> generation controller / tokenizer decode
```

Stage decomposition:

- CPU/data-pipeline: RGB conversion, optional VQA header rendering, per-image normalization, aspect-preserving resize to a patch grid, non-overlap 16x16 patch extraction, row/column ID insertion, padding to `max_patches`, and T5 tokenization.
- Independently cacheable encoder: `flattened_patches` and `attention_mask` produce `encoder_last_hidden_state [B, S_img, H]`; this can be cached across decoder tokens and across repeated decoding for the same image/question render.
- Decoder prefill: consumes `decoder_input_ids` or shifted labels plus encoder states. Conditional prompts for non-VQA image-to-text use `decoder_input_ids`; VQA tasks instead render the question/header into the image before patch extraction.
- Decode: `EncoderDecoderCache` can hold autoregressive decoder self-attention K/V and cross-attention K/V. Source config defaults `use_cache=false`, but generation may request caching.

The primary runtime target for DinoML should be `Pix2StructForConditionalGeneration` inference for image-to-text and VQA/document QA. `Pix2StructVisionModel` and `Pix2StructTextModel` are useful independently stageable parity targets, but the end-to-end task head is seq2seq LM.

## 3. Important config dimensions

| Field | Base/default | Large | Source / notes |
|---|---:|---:|---|
| `text_config.hidden_size` | 768 | 1536 | `config.json` |
| `text_config.num_layers` | 12 | 18 | `config.json` |
| `text_config.num_heads` | 12 | 24 | `config.json` |
| `text_config.d_kv` | 64 | 64 | head dim; `hidden_size == heads * d_kv` in sampled configs |
| `text_config.d_ff` | 2048 | 3968 | gated FFN intermediate |
| `text_config.vocab_size` | 50244 | 50244 | T5 SentencePiece + extra IDs |
| `text_config.dense_act_fn` | `gelu_new` | `gelu_new` | source uses `ACT2FN` |
| `text_config.use_cache` | false | false | default/config; generation can override |
| `text_config.relative_attention_num_buckets` | 32 | 32 | decoder self-attn layer 0 only |
| `text_config.relative_attention_max_distance` | 128 | 128 | decoder self-attn |
| `vision_config.hidden_size` | 768 | 1536 | encoder width |
| `vision_config.num_hidden_layers` | 12 | 18 | encoder blocks |
| `vision_config.num_attention_heads` | 12 | 24 | MHA |
| `vision_config.d_kv` | 64 | 64 | head dim |
| `vision_config.d_ff` | 2048 | 3968 | gated FFN intermediate |
| `vision_config.patch_embed_hidden_size` | 768 | 768 | flattened RGB 16x16 patch payload, excluding 2 ID columns |
| `vision_config.seq_len` | 4096 | 4096 | row/column embedding table length, not necessarily processor max patches |
| Processor `patch_size` | 16x16 | 16x16 | `preprocessor_config.json` |
| Processor `max_patches` | usually 2048 | 4096 for `pix2struct-large` | runtime input length |
| `torch_dtype` | float32 | float32 | config metadata |

Representative checkpoint sweep:

| Checkpoint | Repo task metadata | Top-level `is_vqa` | Processor `is_vqa` | Processor `max_patches` | Text/vision depth | Width | Dropout notes |
|---|---|---:|---:|---:|---:|---:|---|
| `google/pix2struct-base` | image-to-text | false | omitted/null | 2048 | 12/12 | 768 | text/vision dropout 0.2 |
| `google/pix2struct-large` | image-to-text | false | true | 4096 | 18/18 | 1536 | text dropout 0.1, vision dropout 0 |
| `google/pix2struct-textcaps-base` | image-to-text | omitted in sampled config | omitted/null | 2048 | 12/12 | 768 | older vision config omits `patch_embed_hidden_size`; source default gives 768 |
| `google/pix2struct-docvqa-base` | visual-question-answering | false | true | 2048 | 12/12 | 768 | VQA header render path matters |
| `google/pix2struct-screen2words-base` | visual-question-answering | false | true | 2048 | 12/12 | 768 | VQA-style processor |
| `google/deplot` | visual-question-answering / chart-to-table | false | true | 2048 | 12/12 | 768 | chart/question rendered into image |

## 3a. Family variation traps

- The processor, not the model config, determines `max_patches`; base model config has `vision_config.seq_len=4096`, while common processors emit `[B, 2048, 770]`.
- `flattened_patches` length is padded processor output length. The final dimension is `2 + patch_embed_hidden_size`; the first two values are float row/column IDs, cast to `long` in the model.
- Some older configs omit `vision_config.patch_embed_hidden_size`; current strict config default is 768. DinoML should treat missing field as source-default 768 only for native Pix2Struct configs.
- VQA/doc QA does not stitch text tokens into the decoder prefix by default. The processor renders `text` as an image header when `image_processor.is_vqa=true`.
- The model top-level `is_vqa` in sampled configs is unreliable for preprocessing. The active branch in `Pix2StructProcessor.__call__` is `self.image_processor.is_vqa`.
- No GQA/MQA: sampled configs use MHA with `hidden_size == num_heads * d_kv`.
- All Q/K/V/O and MLP linears are bias-free in the modeling source. Config fields like `qkv_bias`/`mlp_bias` from historical vision configs are not read by current native source.
- Token embeddings and LM head are not tied in the conditional generation configs (`tie_word_embeddings=false`); do not alias them unless a config/source path explicitly ties.
- Vision processor uses NCHW tensors internally, then emits flattened patch sequences. DinoML runtime can start from `flattened_patches`; raw-image NHWC optimization must be guarded as a processor acceleration, not a semantic model-layout rewrite.
- Axis-sensitive ops in preprocessing: `mean/std dim=(1,2,3)`, `unfold` over NCHW H/W, `permute(0,4,2,3,1)`, `cat(..., dim=-1)`, pad along patch sequence length, and `attention_mask = patches.sum(dim=-1) != 0`.
- Decoder cache support exists despite `use_cache=false` defaults. Generation code can pass `use_cache=true`; runtime should not assume cache is absent.

## 4. Operator coverage checklist

Tensor/layout ops:

- Dense reshape/view, contiguous, transpose, permute, flatten, unsqueeze/squeeze, pad, concatenation on last dimension, row-major sequence packing.
- Integer casts from float row/column IDs to embedding indices.
- Embedding lookup for token, row, column, and relative-position-bias tables.
- Attention mask creation, inversion, broadcast to `[B, 1 or heads, Q, K]`, `masked_fill`, clamp/max with dtype min.
- `torch.arange`, compare, abs, min, where, log, division, and cast for relative-position bucket computation.

Neural network primitives:

- Bias-free Linear:
  - Vision patch projection: base `Linear(768 -> 768)`, large `Linear(768 -> 1536)`.
  - Vision/text Q/K/V/O: base `Linear(768 -> 768)`, large `Linear(1536 -> 1536)`.
  - Gated FFN: base `wi_0/wi_1 Linear(768 -> 2048)`, `wo Linear(2048 -> 768)`; large `1536 -> 3968 -> 1536`.
  - LM head: base `Linear(768 -> 50244)`, large `Linear(1536 -> 50244)`.
- T5/RMS-style layer norm: variance over last dim, no mean subtraction, scale only, fp32 accumulation.
- `gelu_new` gated activation: `act(wi_0(x)) * wi_1(x)`.
- Residual adds and dropout nodes, with dropout disabled for inference.
- Optional cross-entropy loss is training/eval-only and can be deferred for inference.

Attention primitives:

- Vision noncausal MHA over patch sequence with processor mask.
- Decoder causal self-attention with T5 relative-position bias on layer 0; later layers reuse the position bias.
- Decoder cross-attention over vision encoder states, noncausal encoder mask, no relative bias.
- Softmax in fp32 then cast back to score dtype.
- KV cache update and read for self-attention and cross-attention under `EncoderDecoderCache`.

Preprocessing-coupled ops:

- RGB conversion, optional PIL/text rendering for VQA headers.
- Per-image normalization `(image - mean) / max(std, 1/sqrt(C*H*W))`.
- Dynamic resize to `rows * patch_h` and `cols * patch_w`.
- Non-overlap patch extraction with `kernel_size=stride=patch_size`.
- Row/column ID generation offset by 1; zero rows remain padding.

Generation/cache ops:

- Shift-right labels for teacher forcing: start token 0, replace `-100` with pad 0.
- Decoder-start token 0, EOS 1, pad 0.
- T5Tokenizer decode; no repo-level generation_config in sampled checkpoints.

## 5. Layer/block breakdown

Vision embeddings:

```text
flattened_patches: [B, S_img, 2 + P], P = 3 * patch_h * patch_w = 768
row_ids = flattened_patches[:, :, 0].long()
col_ids = flattened_patches[:, :, 1].long()
patch_payload = flattened_patches[:, :, 2:]
x = Linear(P -> H)(patch_payload) + row_embedding(row_ids) + col_embedding(col_ids)
```

Vision block, repeated `vision_config.num_hidden_layers`:

```text
residual = x
x_norm = RMSNorm(x)
q,k,v = Linear(H -> heads*d_kv, bias=False)(x_norm)
q,k,v -> [B, heads, S_img, d_kv]
scores = q @ k^T
scores += patch mask converted to dtype minimum
attn = softmax(scores, dim=-1, fp32).to(scores.dtype)
x = residual + Linear(heads*d_kv -> H, bias=False)(attn @ v)
x = x + Linear(d_ff -> H, bias=False)(dropout(gelu_new(Linear(H -> d_ff)(RMSNorm(x))) * Linear(H -> d_ff)(RMSNorm(x))))
```

After the vision encoder, a final RMS-style layer norm produces `encoder_last_hidden_state [B, S_img, H]`.

Decoder block, repeated `text_config.num_layers`:

```text
self_residual = y
y_norm = RMSNorm(y)
q,k,v = bias-free self-attn projections
self_position_bias = compute causal T5 relative bias in layer 0, then reuse
self_attn = causal_attention(q,k,v,self_position_bias, decoder mask, self KV cache)
y = self_residual + self_attn_out

cross_residual = y
y_norm = RMSNorm(y)
q = Linear(y_norm)
k,v = Linear(encoder_hidden_states), cached after first decode step if cache enabled
cross_attn = attention(q,k,v, inverted patch attention mask)
y = cross_residual + cross_attn_out

y = y + gated_gelu_ffn(RMSNorm(y))
```

Decoder head:

```text
y = final RMSNorm(y)
logits = Linear(H -> vocab_size, bias=False)(y)
```

## 6. Attention requirements

Vision encoder attention:

- Noncausal self-attention, full `S_img x S_img`.
- MHA, no GQA/MQA, `heads=12/24`, `head_dim=64`.
- Uses additive mask derived from `attention_mask`; padding positions become dtype min before softmax.
- No RoPE/ALiBi/relative bias in vision attention.
- Flash/SDPA-compatible in principle after expressing mask as additive bias, but the source math clamps scores to dtype min after adding mask.

Decoder self-attention:

- Causal autoregressive self-attention.
- MHA, `heads=12/24`, `head_dim=64`.
- Layer 0 owns a T5-style learned relative-position bias with `bidirectional=False`, 32 buckets, max distance 128. Later layers reuse the same computed bias tensor through the forward loop, matching T5 behavior.
- Cache tensors are shaped logically `[B, heads, cached_text_len, d_kv]` for keys and values. Cached self-attention keys are stored after projection/reshape and before relative-position bias application; there is no RoPE.
- With `EncoderDecoderCache`, self K/V update through `self_attention_cache`.

Decoder cross-attention:

- Noncausal attention from decoder queries to encoder patch tokens.
- Q from text hidden states; K/V from `encoder_hidden_states [B, S_img, H]`.
- Cross K/V cache uses `cross_attention_cache`; after the first generated id, `past_key_values.is_updated[layer_idx]` makes later decode steps reuse cross K/V.
- Cross-attention has no learned relative bias; only inverted encoder attention mask.

Backend dispatch:

- Current source implements eager matmul + softmax. There is no family-specific FlashAttention dispatch in `modeling_pix2struct.py`.
- A fused attention path must preserve fp32 softmax, additive mask ordering, causal mask slicing to current key length, and decoder relative-bias reuse.

## 7. Position encoding and custom math

Vision position is row/column embedding, not absolute sequence embedding. Row and column IDs start at 1; 0 is reserved for padded patch rows and maps to embedding row 0.

Decoder self-attention relative bias is T5-style logarithmic bucketing:

```python
def pix2struct_relative_position_bucket(relative_position, num_buckets=32, max_distance=128):
    # Decoder self-attention uses bidirectional=False.
    relative_position = -minimum(relative_position, 0)
    max_exact = num_buckets // 2
    is_small = relative_position < max_exact
    large = max_exact + (
        log(relative_position.float() / max_exact)
        / log(max_distance / max_exact)
        * (num_buckets - max_exact)
    ).long()
    large = minimum(large, num_buckets - 1)
    return where(is_small, relative_position, large)
```

For cached decode, `context_position = arange(query_length) + past_seen_tokens`, while `memory_position = arange(key_length)`. Bias depends on current cache length and cannot be a single static table unless keyed by `(query_length, key_length, past_seen_tokens)`.

Precomputable pieces:

- Row/column ID grids for a fixed patch grid and processor `max_patches`.
- Relative-position bucket indices for fixed prefill/decode shapes.
- Cross-attention K/V for a fixed encoder output during generation.

Dynamic pieces:

- Processor resize grid depends on original image aspect and `max_patches`.
- Decoder causal mask and relative bias depend on current generated length and cache length.

## 8. Preprocessing and input packing

Processor tensor contract:

- Model inputs from image processor: `flattened_patches [B, max_patches, 2 + 3*patch_h*patch_w]`, float32 by default, and `attention_mask [B, max_patches]`, float.
- With 16x16 RGB patches, feature width is `770`: row ID, column ID, and 768 patch pixels.
- Pixel payload flatten order after `unfold` is effectively `[patch_h, patch_w, channels]` per patch because source reshapes to `[B,C,ph,pw,num_patches]`, permutes to `[B,num_patches,ph,pw,C]`, then flattens.
- `attention_mask = (patches.sum(dim=-1) != 0).float()`. Padding rows are all zeros, including row/column IDs.

Patch grid sizing:

```text
scale = sqrt(max_patches * patch_h / image_h * patch_w / image_w)
rows = clamp(int(scale * image_h / patch_h), min=1, max=max_patches)
cols = clamp(int(scale * image_w / patch_w), min=1, max=max_patches)
resized_h = rows * patch_h
resized_w = cols * patch_w
```

The formula is intended to keep `rows * cols <= max_patches`; DinoML should still guard this before padding because runtime configs can be malformed.

VQA/doc QA packing:

- If `image_processor.is_vqa=true`, processor requires `header_text`, normally copied from `text`.
- The header is rendered as a white-background text image, resized to match image width, concatenated above the image, then the combined image is normalized/resized/patchified.
- Decoder input text is not separately tokenized in this branch. The question/context becomes pixels.

Non-VQA conditional generation:

- Images are patchified.
- Optional `text` is tokenized as decoder prefix and returned as `decoder_input_ids` and `decoder_attention_mask`, with `add_special_tokens` defaulting to false when images are present.

GPU/runtime boundary recommendation:

- First DinoML integration should accept processor-produced `flattened_patches` and masks as runtime inputs.
- Raw-image preprocessing can remain CPU/data pipeline initially. A later GPU processor path must be guarded by exact NCHW/NHWC layout conversion, patch flatten order, resize rounding, antialias behavior, and header-rendering exclusion.

## 9. Graph rewrite / lowering opportunities

### Rewrite: processor non-overlap patch extract -> layout-aware patch GEMM input

Source pattern:

```text
NCHW image -> bilinear resize -> unfold(kernel=patch, stride=patch)
-> reshape/permute -> [B, rows*cols, patch_h*patch_w*C]
```

Replacement:

```text
PatchExtractNHWC-or-NCHW -> flattened patch payload -> patch Linear
```

Preconditions:

- `kernel_size == stride == (patch_h, patch_w)`.
- `padding == 0`, `dilation == 1`.
- RGB channel count is 3.
- Resized height/width are divisible by patch height/width.
- Flatten order must match `[ph, pw, C]`.
- Header rendering is either already applied by CPU pipeline or excluded.

Failure cases:

- Any non-16x16 processor config not reflected in `patch_embed_hidden_size`.
- Different resize antialias semantics causing end-to-end drift.
- NHWC pass that changes flatten order or row/column ID generation.

Parity test sketch:

- Compare processor `flattened_patches` for synthetic images with known channel ramps, several aspect ratios, and both `max_patches=2048` and `4096`.

### Rewrite: patch projection as GEMM with row/column embedding fusion

Source pattern:

```text
payload = flattened_patches[:, :, 2:]
emb = Linear(payload) + row_embed(row_id) + col_embed(col_id)
```

Replacement:

```text
GEMM(payload, W_patch^T) + bias(optional absent) + gathered_row + gathered_col
```

Preconditions:

- `flattened_patches[...,0:2]` IDs are integer-valued floats in `[0, seq_len)`.
- Payload width equals `patch_embed_hidden_size`.
- Padding rows are all zero so row/col ID 0 is used consistently.

Failure cases:

- Processor or caller supplies raw `pixel_values` instead of flattened patches.
- Malformed row/column IDs exceed embedding table.

### Rewrite: T5 RMSNorm -> DinoML RMSNorm primitive

Source pattern:

```text
x * rsqrt(mean(x.float() ** 2, dim=-1, keepdim=True) + eps) * weight
```

Replacement: fused RMSNorm with fp32 accumulation and scale-only affine.

Preconditions: last-dim contiguous or supported strided accessor; no bias; epsilon from config (`1e-6`).

### Rewrite: gated GELU FFN -> fused GEGLU/GatedGELU MLP

Source pattern:

```text
gelu_new(x @ wi_0.T) * (x @ wi_1.T) -> wo
```

Replacement: two GEMMs plus fused activation multiply, then output GEMM; later fuse first two projections as a packed dual-output GEMM.

Preconditions:

- Both input projections are bias-free and share input shape.
- Activation is `gelu_new`.
- Weight order is not packed in source; packed lowering must preserve logical names `wi_0`, `wi_1`.

### Rewrite: decoder cross-attention K/V precompute

Source pattern:

```text
for each decode step: if cross cache not updated, project encoder_hidden_states to K/V
```

Replacement: encoder-stage or first-decode-stage K/V projection cache per layer.

Preconditions:

- Encoder hidden states and encoder attention mask are unchanged for the generation request.
- Layer weights are resident and cache memory is allocated for `[B, heads, S_img, d_kv]`.

Failure cases: encoder states differ across beams or image batch expansion without explicit cache reorder support.

### Layout guard: no blind NHWC translation inside flattened sequence model

The model body after the processor is `[B, sequence, hidden]`. Treat it as token sequence layout. NHWC/channel-last optimization applies only before or inside patch extraction. Required axis rewrites for a GPU processor path include NCHW `mean(dim=(1,2,3))`, resize H/W axes, unfold spatial axes, and final `cat(dim=-1)`.

## 10. Kernel fusion candidates

Highest priority:

- RMSNorm scale-only kernel with fp32 accumulation. It appears before every attention/FFN and at encoder/decoder outputs.
- Bias-free GEMM families for Q/K/V/O, FFN, patch projection, and LM head. Base and large widths align well with existing GEMM work.
- Gated GELU activation multiply between `wi_0` and `wi_1`, ideally fused after dual projection.
- Decoder causal attention with relative bias and KV cache. This is required for generation throughput.
- Cross-attention K/V projection caching for image encoder outputs.

Medium priority:

- Vision full-sequence attention over up to 2048 or 4096 patch tokens. This can dominate prefill/encoder cost; SDPA/Flash-style kernels with additive masks are worth probing.
- Patch projection + row/column embedding add fusion.
- Last-token-only LM head for decode; prefill still needs full logits only if the generation controller asks for them.
- Processor patch extraction GPU kernel for high-throughput screenshot batches, excluding VQA header rendering.

Lower priority:

- Training loss (`CrossEntropyLoss`) and label shift on device.
- Output attentions/hidden states.
- PIL-equivalent VQA header rendering on GPU; keep CPU pipeline unless it becomes a bottleneck.

## 11. Runtime staging plan

Stage 1: config and weight loading.

- Parse nested `Pix2StructConfig`, source-default omitted fields, T5 tokenizer metadata, and processor config.
- Load untied embeddings/LM head and all bias-free linears.

Stage 2: processor-bound vision encoder parity.

- Accept `flattened_patches` and `attention_mask` from HF processor.
- Implement patch projection, row/column embeddings, vision RMSNorm, attention, FFN, and final norm.

Stage 3: decoder prefill without cache.

- Run `decoder_input_ids` plus encoder states through full decoder and LM head.
- Validate image captioning and doc-QA first-token/logit parity.

Stage 4: generation decode with cache.

- Add self-attention KV cache and cross-attention cache semantics compatible with `EncoderDecoderCache`.
- Add cache reorder only when beam search is in scope; greedy decode can skip beam reorder initially.

Stage 5: graph rewrites/fusions.

- Lower GEMMs to CUTLASS where possible, add RMSNorm and GEGLU fusion, then attention backend replacement.

Stage 6: processor acceleration.

- Optional GPU patch extraction for non-VQA images; leave VQA header rendering in CPU preprocessing.

Initially stub/defer: training loss, dropout, output attentions, hidden-state returns, beam search, and raw-image preprocessing inside DinoML runtime.

## 12. Parity and validation plan

- Processor contract tests: compare HF `flattened_patches` and `attention_mask` for synthetic RGB images with exact known values, multiple aspect ratios, `max_patches=2048/4096`, and VQA header enabled/disabled. Use strict or near-strict fp32 tolerance after resize differences are controlled.
- Patch embedding test: feed crafted flattened patches with row/column IDs including padding row 0; compare patch projection + row/column embedding sum.
- RMSNorm test: random fp32/fp16/bf16 tensors, verify fp32 accumulation. Suggested tolerances: fp32 `1e-5`, fp16/bf16 `5e-3` to `1e-2` depending backend.
- Relative-bucket test: compare bucket IDs and bias tensors for prefill lengths and cached decode `(query_length=1, past_seen_tokens>0)`.
- Single vision layer parity, then N-layer encoder parity with processor output.
- Single decoder block parity for self-attn only, cross-attn only, and full block.
- Prefill logits parity for `Pix2StructForConditionalGeneration` on a fixed image and optional decoder prefix.
- Decode parity: greedy generation for 1, 2, and N tokens with cache enabled; compare logits and generated token IDs.
- VQA/doc QA end-to-end: use `google/pix2struct-docvqa-base` processor path with header text rendered into the image.

## 13. Performance probes

- CPU processor throughput: images/sec for resize/normalize/patchify; separate VQA header rendering cost.
- Encoder-only throughput across `max_patches` 512, 1024, 2048, 4096.
- Vision attention backend comparison: eager matmul/softmax vs fused attention for `S_img` sweep.
- Decoder prefill throughput as a function of target prefix length.
- Decode tokens/sec with and without self/cross KV cache.
- Cross-attention cache memory and projection time per layer.
- LM head decode cost with full-sequence logits vs last-token-only logits.
- Batch-size sweep for image-to-text and doc-QA.
- Layout/processor probe: CPU flattened patches as input vs GPU patch extraction for non-VQA screenshot batches.

## 14. Skip/defer list

- Training loss and gradient checkpointing.
- Dropout behavior in training mode.
- Beam search/cache reorder for first greedy integration.
- Output attentions and hidden-state collection.
- GPU/PIL-equivalent header rendering.
- Remote-code variants; sampled repos use native source.
- Quantization-specific dtype casts around int8 weights.
- Multi-GPU/tensor parallelism.
- Raw-image preprocessing in the compiled runtime, until model parity from `flattened_patches` is stable.

## 15. Final implementation checklist

- [ ] Parse nested Pix2Struct config and processor config, including omitted `patch_embed_hidden_size=768`.
- [ ] Load untied token embedding and LM head weights.
- [ ] Implement processor-produced `flattened_patches [B,S,770]` input contract.
- [ ] Implement row/column embedding extraction from first two patch fields.
- [ ] Implement scale-only RMSNorm with fp32 accumulation.
- [ ] Implement bias-free Linear/GEMM coverage for base and large dimensions.
- [ ] Implement GatedGELU FFN pattern and fusion candidate.
- [ ] Implement vision noncausal MHA with additive padding mask.
- [ ] Implement decoder causal self-attention with T5 relative-position bias.
- [ ] Implement decoder cross-attention over encoder patch states.
- [ ] Implement `EncoderDecoderCache`-equivalent self and cross K/V semantics.
- [ ] Add cross-attention K/V precompute/cache rewrite.
- [ ] Add last-token-only LM head decode optimization.
- [ ] Add guarded processor patch-extract rewrite with flatten-order tests.
- [ ] Add VQA header-rendering pipeline compatibility tests using HF processor.
- [ ] Add single-block, encoder, prefill-logit, and cached-decode parity tests.
- [ ] Benchmark processor, encoder, prefill, decode, and cache memory separately.
