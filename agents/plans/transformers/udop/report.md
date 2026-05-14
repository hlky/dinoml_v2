# UDOP Transformers Audit

## 1. Source basis

Transformers commit/version: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4` in `X:/H/transformers`.

Model id: primary official checkpoint [`microsoft/udop-large`](https://huggingface.co/microsoft/udop-large); official variants [`microsoft/udop-large-512`](https://huggingface.co/microsoft/udop-large-512) and [`microsoft/udop-large-512-300k`](https://huggingface.co/microsoft/udop-large-512-300k).

Config source: HF `config.json`, `preprocessor_config.json`, and `generation_config.json` for the above official repos, plus public mirrors/test repos [`nielsr/udop-test`](https://huggingface.co/nielsr/udop-test) and [`nielsr/udop-large`](https://huggingface.co/nielsr/udop-large).

Source files inspected:

- [`configuration_udop.py`](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/udop/configuration_udop.py)
- [`modeling_udop.py`](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/udop/modeling_udop.py)
- [`processing_udop.py`](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/udop/processing_udop.py)
- [`tokenization_udop.py`](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/udop/tokenization_udop.py)
- [`image_processing_layoutlmv3.py`](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/layoutlmv3/image_processing_layoutlmv3.py), because UDOP has no `image_processing_udop.py` and its processor composes `LayoutLMv3ImageProcessor`.

Local evidence snapshots: [`_sources/source_snippets.md`](./_sources/source_snippets.md) and [`_sources/representative_configs.md`](./_sources/representative_configs.md).

Any missing files or assumptions: no gated/401/403 gaps were observed for representative configs. No runtime import/tests were run. The report target is inference for `UdopForConditionalGeneration` image/text-to-text generation; `UdopModel` base outputs and `UdopEncoderModel` are useful subtargets, while training loss and token-classification fine-tunes are deferred.

## 2. High-level architecture

UDOP is a document multimodal encoder-decoder: LayoutLMv3-style image preprocessing and optional OCR feed a T5-like text/layout/image encoder, then a T5-like autoregressive decoder generates text.

Dataflow:

```text
document image + optional prompt/OCR words/boxes
-> LayoutLMv3 image processor + UdopTokenizer
-> text embeddings + Conv2d patch embeddings + OCR-box/patch merge + 2D bbox embeddings
-> encoder self-attention with 1D + horizontal + vertical relative bias
-> decoder causal self-attention + encoder-decoder cross-attention
-> tied LM head -> logits/generation controller
```

Stage decomposition:

- CPU/data pipeline: image resize/rescale/normalize; optional Tesseract OCR; word-to-subword box expansion; overflow image duplication.
- Encoder prefix stage: patch convolution, dynamic OCR patch gather, remaining-patch concatenation, bbox embeddings, encoder attention. Encoder outputs and encoder attention masks are cacheable across decoder steps.
- Decode stage: standard encoder-decoder autoregressive generation with decoder self-attention KV cache and cross-attention KV cache.
- Independent validation targets: processor tensor ABI, patch embedding/merge, one encoder block, full encoder, one decoder block with cross-attention, logits/generate.

## 3. Important config dimensions

| Field | Official large / large-512 values | Source/config provenance |
|---|---:|---|
| `vocab_size` | 33201 | `config.json` |
| `d_model` / hidden size | 1024 | `config.json` |
| `num_layers` | 24 encoder | `config.json` |
| `num_decoder_layers` | 24 decoder | `config.json`; defaults to `num_layers` if omitted |
| `num_heads` | 16 | `config.json` |
| `d_kv` / head dim | 64 | `config.json`; Q/K/V width = `16*64=1024` |
| `d_ff` | 4096 | `config.json` |
| `feed_forward_proj` | `relu` | official configs |
| Gated MLP support | optional `gated-*` | config/source default parser, not used by official configs |
| `relative_attention_num_buckets` | 32 | `config.json` |
| `relative_attention_max_distance` | 128 | `config.json` |
| encoder relative biases | `1d`, `horizontal`, `vertical` | `relative_bias_args` |
| `max_2d_position_embeddings` | 1024 | `config.json` |
| `image_size` | 224 or 512 | checkpoint config |
| `patch_size` | 16 | `config.json` |
| patch tokens | 196 for 224, 1024 for 512 | inferred from `image_size//patch_size` |
| `num_channels` | 3 | `config.json` |
| cache | `use_cache=true` for decoder | `config.json` |
| generation start/end/pad | start 0, EOS 1, pad 0 | `generation_config.json` |
| dtype | `torch_dtype=float32` | `config.json` metadata |

Representative checkpoint sweep:

| Checkpoint | Official? | Image size | Patch tokens | Topology notes |
|---|---|---:|---:|---|
| `microsoft/udop-large` | yes | 224 | 196 | common production checkpoint; 24/24 layers, relu FFN |
| `microsoft/udop-large-512` | yes | 512 | 1024 | same neural dimensions, much longer encoder sequence from patches |
| `microsoft/udop-large-512-300k` | yes | 512 | 1024 | same operator surface; training-step variant |
| `nielsr/udop-test` | public test/mirror | 224 | 196 | same topology; historical preprocessor type string |
| `nielsr/udop-large` | public mirror | 224 | 196 | same topology; older tokenizer/preprocessor metadata |

## 3a. Family variation traps

- Source consumes `pixel_values` in NCHW `[B,C,H,W]`; NHWC is only a guarded local optimization around patch embedding and image preprocessing.
- `image_size=512` increases patch tokens from 196 to 1024 before OCR patch filtering, which can dominate encoder attention and relative-bias memory.
- The encoder input length after merge is `text_len + padded_remaining_patch_len`, not a simple `text_len + all_patches`: patch tokens assigned to OCR text boxes are removed from the standalone patch list and their embeddings are added into text tokens.
- `bbox` from tokenizer/preprocessor is 0-1000 integer scale, but model code treats boxes as normalized 0-1 floats for patch gathering and 2D position clipping. Current conversion script casts `encoding.bbox.float()`; DinoML should explicitly normalize or verify caller conventions.
- Encoder relative bias is aggregated from 1D sequence distance plus horizontal/vertical bbox-center distances. A plain T5 relative-bias implementation is insufficient.
- Decoder uses causal T5-style self-attention relative bias and cross-attention mask bias, but no horizontal/vertical bias in the decoder.
- Official configs use `feed_forward_proj="relu"`; source supports `gated-gelu` style gated FFN. Reject or separately audit gated checkpoints if they appear.
- Tokenizer special boxes matter: separator box `[1000,1000,1000,1000]`, pad box `[0,0,0,0]`; those values affect target patch suppression and relative layout bias.
- Historical configs can say `UdopImageProcessor`; pinned native processor composes `LayoutLMv3ImageProcessor`.
- Tied weights/aliases: shared token embedding, decoder/encoder embedding aliases, LM head tied to shared embedding, patch embedding alias keys, and selected relative-bias alias keys in conditional generation.

## 4. Operator coverage checklist

Tensor/layout ops:

- NCHW image input, resize/rescale/normalize in data pipeline, Conv2d patch embedding, flatten, transpose, contiguous, reshape/view, concat, pad/truncate, dynamic gather, boolean mask indexing, scatter-style patch exclusion, clipping/floor/cast/long.
- Dynamic sequence construction from ragged remaining patch lists padded to fixed max patch count.

Neural primitives:

- Embedding lookup for tokens and x/y bbox positions.
- Bias-free Linear/GEMM: Q/K/V/O `1024 -> 1024`; FFN `1024 -> 4096 -> 1024`; LM head `1024 -> 33201`.
- Optional gated FFN path: two `1024 -> 4096` projections, activation, elementwise multiply, output projection.
- RMSNorm/T5LayerNorm with fp32 accumulation and learned scale only.
- ReLU official activation; GELU possible for gated configs.
- Residual add, dropout disabled for inference, fp16 clamp branches can be omitted for inference unless matching low-precision eager behavior is required.

Attention primitives:

- Dense MHA for encoder self-attention, decoder causal self-attention, and decoder cross-attention.
- Mask addition with dtype min values; softmax upcast to fp32 then cast back.
- Encoder-decoder cache with self-attention and cross-attention subcaches.

Position/relative-bias ops:

- T5 relative position bucket function.
- 1D sequence relative bias.
- Horizontal/vertical bbox-center relative bias with scaling factor 100 and max distance 100 defaults.
- 2D cell embeddings from clipped/scaled bbox coordinates.

Preprocessing-coupled ops:

- Tesseract OCR boundary or caller-supplied words/boxes.
- Word-to-subword bbox expansion and overflow-to-image duplication.
- Prompt/question plus OCR words are tokenized as pair sequences.

Generation/cache ops:

- `_shift_right` for labels/training; generation starts at pad token id 0.
- DynamicCache/EncoderDecoderCache update/reuse, cache reorder through generic generation support.

## 5. Layer/block breakdown

Encoder embedding path:

```text
input_ids [B,T] -> shared token embedding [B,T,1024]
pixel_values [B,3,H,W] -> Conv2d(k=s=16, out=1024) -> flatten/transpose [B,P,1024]
bbox [B,T,4] -> patch-center gather -> add selected patch embed into text token embed
remaining patch embeds -> pad to P -> concat after text tokens
bbox + visual_bbox -> concat/pad [B,T+P_rem,4]
cell_2d_embedding(bbox) -> add to combined embeddings
dropout -> 24 encoder blocks -> final RMSNorm
```

Encoder block, repeated 24 times:

```text
x_norm = RMSNorm(x)
q,k,v = Linear(x_norm) with no bias, shapes [B,H,S,64]
scores = q @ k.T + aggregated_relative_bias + padding_mask
attn = softmax(scores.float()).to(dtype)
x = x + Linear(attn @ v)
x = x + FFN(RMSNorm(x))
```

Decoder block, repeated 24 times:

```text
x_norm = RMSNorm(x)
q,k,v = decoder self-attention Linear(x_norm), causal relative bias, KV cache update
x = residual + self_attention
x_norm = RMSNorm(x)
q = Linear(x_norm), k/v = Linear(encoder_hidden_states) or cached cross-attention states
x = residual + cross_attention(masked by encoder attention mask)
x = x + FFN(RMSNorm(x))
```

LM head:

```text
sequence_output *= d_model**-0.5  # because tie_word_embeddings is true
logits = tied_linear(sequence_output, shared_embedding.T)
```

## 6. Attention requirements

Encoder self-attention is bidirectional dense MHA over the combined text/layout/patch sequence. It has 16 heads, head dim 64, Q/K/V width 1024, and requires aggregated relative bias from sequence positions plus bbox-derived horizontal/vertical distances. No KV cache is used in the encoder.

Decoder self-attention is causal dense MHA with the same heads/head dim. It uses T5-style relative position bias computed in the first layer and shared through later layers, plus the causal mask from `create_causal_mask`. Cached keys/values are stored after projection and before score matmul; relative bias uses `past_seen_tokens` when computing decode positions.

Decoder cross-attention is dense MHA from decoder queries to encoder hidden states. It uses projected encoder K/V and the inverted encoder attention mask; no learned cross relative bias. In `EncoderDecoderCache`, cross K/V are computed once and reused after `is_updated[layer_idx]` is set.

FlashAttention/SDPA compatibility: source does manual matmul/softmax/dropout/matmul and returns optional attention weights. A fused attention backend is valid only if it accepts additive bias tensors, fp32 softmax semantics, causal decode offsets, and rectangular cross-attention masks. Encoder 2D relative bias is the biggest nonstandard requirement.

## 7. Position encoding and custom math

UDOP uses no RoPE/ALiBi. Required custom math is T5 bucketed relative bias plus bbox-derived horizontal/vertical variants.

```python
def udop_relative_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
    if bidirectional:
        half = num_buckets // 2
        bucket = (relative_position > 0).long() * half
        relative_position = abs(relative_position)
        num_buckets = half
    else:
        bucket = 0
        relative_position = -minimum(relative_position, 0)
    max_exact = num_buckets // 2
    large = max_exact + log(relative_position / max_exact) / log(max_distance / max_exact) * (num_buckets - max_exact)
    large = minimum(large.long(), num_buckets - 1)
    return bucket + where(relative_position < max_exact, relative_position, large)
```

```python
def udop_horizontal_vertical_positions(bbox):
    x = mean(bbox[..., [0, 2]], dim=-1)  # left/right center
    y = mean(bbox[..., [1, 3]], dim=-1)  # top/bottom center
    return x, y  # each becomes memory_position - context_position, scaled by 100
```

Precomputable: visual patch boxes for fixed `image_size/patch_size`; 1D decoder relative-bias buckets for bounded decode lengths. Dynamic: horizontal/vertical bias depends on per-example OCR boxes and on which patch tokens remain after patch exclusion.

## 8. Preprocessing and input packing

Processor ABI:

- `pixel_values`: `[B,3,H,W]`, float tensor after resize to checkpoint size, rescale `1/255`, ImageNet mean/std normalization.
- `input_ids`: tokenized prompt/question and/or OCR words.
- `attention_mask`: text-token mask before model appends remaining patch masks.
- `bbox`: token-level boxes. OCR boxes are normalized to 0-1000 by LayoutLMv3 preprocessing, then expanded to subwords by `UdopTokenizer`. For model parity, convert/guard to the normalized 0-1 convention expected by bbox math before GPU graph execution.
- `visual_bbox`: optional `[B,P,4]`; if omitted, model creates regular patch boxes.

OCR ownership: first integration should allow caller-supplied words/boxes and run OCR in CPU/data pipeline only. Tesseract is not a DinoML GPU operator. If `apply_ocr=True`, the HF processor rejects caller-provided boxes/labels.

Overflow/chunking: tokenizer can return `overflow_to_sample_mapping`; processor duplicates the corresponding image for each overflowed token chunk. This is part of data packing, not the neural graph.

Prompt layout: for DocVQA-style examples, `text` is the question/prefix and `text_pair` is OCR words. Token boxes for question-side tokens in a pair use the pad token box; OCR-side subwords use OCR boxes.

## 9. Graph rewrite / lowering opportunities

### Rewrite: patch Conv2d -> Linear

Source pattern: `Conv2d(3,1024,kernel_size=patch_size,stride=patch_size)` followed by `flatten(2).transpose(1,2)`.

Replacement: local `WindowFlatten(NCHW, kh=kw=patch_size, stride=patch_size)` -> `GEMM(weight_flat.T)` -> bias add -> `[B,P,1024]`.

Preconditions: `kernel_size == stride == patch_size`, `padding=0`, `dilation=1`, `groups=1`, fixed source image dimensions equal config image size, and source NCHW flatten order preserved. NHWC optimization requires a weight/layout transform and a guard that all consumers remain inside the patch embed region.

Failure cases: non-square config tuple not handled by naive scalar `image_size // patch_size` paths in `combine_image_text_embeddings`; reject until tuple path is audited.

Parity test: compare Conv2d path and lowered WindowFlatten+GEMM for 224 and 512 configs with fixed random images and real weights.

### Rewrite: text-patch merge as explicit packing op

Source pattern: bbox center -> patch index gather -> add patch embedding to text token -> boolean removal of used patches -> pad remaining patch list -> concat.

Replacement: a bounded `udop_pack_document_tokens` helper that emits combined embeddings, combined bbox, and combined attention mask.

Preconditions: square image grid, normalized bbox in 0-1 range, fixed max patch count, no duplicate-patch semantic shortcuts; duplicate OCR tokens mapping to the same patch must remove the patch once but gather it for each token.

Failure cases: ragged dynamic outputs without fixed padding, bbox scale mismatch, empty input special branch.

Parity test: randomized boxes including duplicate centers, pad/sep boxes, all-zero/all-one boxes, compare combined tensors and masks.

### Rewrite: relative bias precompute/cache

Source pattern: recompute 1D/horizontal/vertical bucket embeddings before the encoder stack and share `position_bias`.

Replacement: compile/run-time bias materialization once per encoder input, then reuse across encoder layers. Cache 1D buckets for sequence length; compute bbox components per batch.

Preconditions: inference mode, no augmentation, fixed `relative_bias_args`, and no `prefix_bucket` expansion mutation.

Failure cases: training augmentation, `prefix_bucket`, unsupported custom bias types.

### Guarded NHWC opportunity

Only the image preprocessing/patch embedding window is a good NHWC candidate. The model immediately flattens to sequence tokens, and all attention/MLP tensors are `[B,S,D]`. Do not translate bbox axes or token sequence axes. Axis-sensitive operations to preserve: Conv2d input channels, `flatten(2)`, patch grid index equation, `bbox[...,0/1/2/3]`, softmax `dim=-1`, RMSNorm/mean `dim=-1`, concatenation along sequence dim 1.

## 10. Kernel fusion candidates

Highest priority:

- Patch Conv2d/WindowFlatten+GEMM for 224 and 512 image sizes, especially 512's 1024 patches.
- UDOP packing helper for gather/add/filter/pad/concat, because generic PyTorch-style boolean indexing is awkward and performance-sensitive.
- RMSNorm + QKV projections for T5-like blocks.
- Dense attention with additive relative bias, preserving fp32 softmax.
- FFN `Linear + ReLU + Linear` and optional gated GELU variant.

Medium priority:

- Aggregated relative-bias kernel from bbox centers and sequence positions.
- Cross-attention K/V projection cache for decode.
- Tied LM head GEMM with last-token-only decode optimization.

Lower priority:

- Tokenizer/OCR acceleration, because it belongs in CPU/data pipeline.
- Training-only loss and label packing.
- Rare `prefix_bucket`/augmentation relative-bias branches.

## 11. Runtime staging plan

Stage 1: parse config/processors; load weights with alias preservation; admit official `relu` large configs only.

Stage 2: implement processor ABI boundary with caller-supplied OCR words/boxes; normalize/guard bbox scale; run patch embedding and packing parity.

Stage 3: encoder-only parity for 224, then 512, including aggregated bbox relative bias.

Stage 4: decoder block and full seq2seq prefill parity with encoder outputs.

Stage 5: generation decode with `EncoderDecoderCache`, self-attention KV growth, and cross-attention K/V reuse.

Stage 6: enable fused attention/GEMM/RMSNorm and patch Conv2d lowering; add layout guards for NHWC patch window only.

Stage 7: throughput tuning for long 512 encoder sequences and decode batching.

Can be stubbed initially: Tesseract OCR, training loss, output attentions/hidden states, token-label paths, historical `UdopImageProcessor` metadata.

## 12. Parity and validation plan

- Config/load tests: official 224 and 512 configs, tied token/LM weights, patch-embedding alias keys, relative-bias aliases.
- Processor ABI tests: supplied words/boxes vs OCR-disabled path; overflow image duplication; special token boxes.
- Custom op tests: `get_visual_bbox`, patch-index mapping, packing helper, 2D cell embeddings, T5 bucket function, horizontal/vertical relative bias.
- Single-layer parity: encoder block with random normalized boxes and masks; decoder self-attention with and without cache; cross-attention reuse.
- Full encoder parity: 224 and 512 image sizes, small text lengths, duplicate patch centers.
- Prefill logits parity: `UdopForConditionalGeneration` with fixed `decoder_input_ids`.
- Decode parity: greedy first N tokens with cached and uncached decode.
- Tolerances: fp32 `rtol=1e-4, atol=1e-4`; fp16/bf16 start with `rtol=5e-2, atol=5e-2` for full logits, tighter per-op where accumulation is controlled.

## 13. Performance probes

- Processor throughput: image resize/normalize plus optional OCR separately from GPU graph.
- Patch embedding throughput for 224 vs 512.
- Packing helper latency vs text length, duplicate patch rate, and patch count.
- Encoder-only throughput over total sequence length after packing.
- Relative-bias materialization time and memory for 224/512.
- Decoder prefill throughput by target length.
- Decode tokens/sec with cross-attention cache on/off.
- Attention backend comparison for additive-bias encoder and causal decoder.
- LM head last-token-only vs full-sequence logits.
- Batch-size sweep and image-size sweep.

## 14. Skip/defer list

- Training loss, gradient checkpointing, dropout behavior.
- Tesseract OCR inside DinoML runtime; keep in CPU/data pipeline.
- Token classification fine-tunes and `word_labels`.
- Historical/remote `UdopImageProcessor` implementation unless a target repo requires it.
- `gated-gelu` checkpoints until one is identified and admitted.
- Relative-bias `augmentation`, `prefix_bucket`, and `expand` mutation paths.
- Output attentions/hidden states unless debugging requires them.
- Beam search-specific generation controller details beyond cache reorder.

## 15. Final implementation checklist

- [ ] Parse `UdopConfig` and reject unsupported custom relative-bias types.
- [ ] Load official weights with shared embedding/LM-head and patch aliases preserved.
- [ ] Implement processor ABI boundary for `pixel_values`, `input_ids`, `attention_mask`, `bbox`, optional `visual_bbox`.
- [ ] Add bbox scale validation/normalization policy.
- [ ] Implement patch Conv2d or guarded WindowFlatten+GEMM rewrite.
- [ ] Implement UDOP image/text packing helper.
- [ ] Implement 2D cell embeddings.
- [ ] Implement aggregated encoder relative bias.
- [ ] Implement T5-style decoder relative bias with cache offsets.
- [ ] Implement encoder dense MHA with additive bbox bias.
- [ ] Implement decoder causal self-attention and cross-attention cache.
- [ ] Implement T5 RMSNorm, relu FFN, optional gated FFN rejection/admission.
- [ ] Implement tied LM head scaling and logits.
- [ ] Add 224 and 512 config parity tests.
- [ ] Benchmark patch packing, encoder attention, relative bias, prefill, and decode.
