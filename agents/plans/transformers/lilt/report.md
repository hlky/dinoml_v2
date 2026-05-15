# LiLT Transformers Audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4 from local checkout transformers
Model id: SCUT-DLVCLab/lilt-roberta-en-base, SCUT-DLVCLab/lilt-infoxlm-base; representative mirrors/fine-tunes listed below
Config source: Hugging Face Hub config/tokenizer files fetched 2026-05-13 into _sources/
Source files inspected:
  transformers/src/transformers/models/lilt/configuration_lilt.py
  transformers/src/transformers/models/lilt/modeling_lilt.py
  transformers/src/transformers/models/layoutlmv3/tokenization_layoutlmv3.py
  transformers/src/transformers/models/layoutxlm/tokenization_layoutxlm.py
Any missing files or assumptions:
  No LiLT processor/image/OCR source exists in the LiLT directory. Official LiLT repos did not expose preprocessor_config.json. OCR and word boxes are caller/data-pipeline responsibilities.
```

Representative config snapshots are under `_sources/`, with a compact table in `_sources/config_sweep.md`. The modeling file is the authoritative source for the current in-library implementation; no modular LiLT source file was present.

## 2. High-level architecture

LiLT is an encoder-only document layout model. It combines a RoBERTa/XLM-R-like token encoder stream with a shrunk layout stream derived from token-aligned bounding boxes. The two streams compute separate Q/K/V projections but couple attention scores by adding normalized text and layout score matrices before each stream softmax. The public heads are base encoder, sequence classification, token classification, and extractive QA.

```text
words + OCR/layout boxes -> layout-aware tokenizer -> input_ids, bbox, masks
input_ids/token_type_ids/position_ids -> text embeddings
bbox + position_ids -> layout embeddings
paired text/layout encoder blocks -> text last_hidden_state
task head -> sequence logits, token logits, or QA start/end logits
```

First useful DinoML target: `LiltForTokenClassification` for OCR/KIE-style document token labeling. The base encoder and token classification head can be validated independently from OCR and word segmentation because the neural ABI consumes token IDs, attention masks, and token-level boxes.

## 3. Important config dimensions

| Field | Source default | Official RoBERTa base | Official InfoXLM base | Runtime significance |
| --- | ---: | ---: | ---: | --- |
| `hidden_size` | 768 | 768 | 768 | text stream width |
| `num_hidden_layers` | 12 | 12 | 12 | repeated paired blocks |
| `num_attention_heads` | 12 | 12 | 12 | MHA heads |
| text `head_dim` | 64 | 64 | 64 | `hidden_size / heads` |
| `channel_shrink_ratio` | 4 | 4 | 4 | layout width divider |
| layout hidden width | 192 | 192 | 192 | `hidden_size / shrink` |
| layout `head_dim` | 16 | 16 | 16 | text head dim / shrink |
| `intermediate_size` | 3072 | 3072 | 3072 | text FFN width |
| layout intermediate width | 768 | 768 | 768 | `intermediate_size / shrink` |
| `vocab_size` | 30522 | 50265 | 250002 | tokenizer-family dependent |
| `max_position_embeddings` | 512 | 514 | 514 | absolute token and layout position table size |
| `max_2d_position_embeddings` | 1024 | 1024 | 1024 | bbox coordinate/height/width embedding range |
| `type_vocab_size` | 2 | 1 | 1 | official checkpoints normally have one segment type |
| `layer_norm_eps` | 1e-12 | 1e-5 | 1e-5 | checkpoint-sensitive LayerNorm parity |
| `hidden_act` | `gelu` | `gelu` | `gelu` | BERT GELU FFNs |
| `torch_dtype` | not set | float32 | float32 | config metadata only |
| `use_cache` | inherited/unused | true | true | ignored by inspected LiLT source |

Checkpoint sweep:

| Checkpoint | Role | Architecture | Hidden/layers/heads | Vocab | Tokenizer/layout ABI variation |
| --- | --- | --- | --- | ---: | --- |
| `hf-internal-testing/tiny-random-LiltForSequenceClassification` | tiny/debug | sequence classification | 24/2/6 | 1024 | nonstandard FFN 37, type vocab 16; useful for guard tests only |
| `SCUT-DLVCLab/lilt-roberta-en-base` | common English base | base encoder | 768/12/12 | 50265 | `LayoutLMv3Tokenizer`, special boxes all zero |
| `SCUT-DLVCLab/lilt-infoxlm-base` | common multilingual base | base encoder | 768/12/12 | 250002 | `LayoutXLMTokenizer`, SEP box `[1000,1000,1000,1000]` |
| `dharmik3005/lilt-en-funsd` | fine-tuned English KIE | token classification | 768/12/12 | 50265 | inherits LayoutLMv3-style tokenizer ABI |
| `koshkidadanet/lilt-xlm-roberta-base-finetuned-piad` | fine-tuned multilingual KIE | token classification | 768/12/12 | 250002 | config only; tokenizer files absent in fetched repo snapshot |

## 3a. Family variation traps

- Official configs differ from `LiltConfig` defaults in `pad_token_id`, `max_position_embeddings`, `type_vocab_size`, and `layer_norm_eps`; do not instantiate source defaults for checkpoint parity.
- `hidden_size`, `intermediate_size`, and per-head text dimension must be divisible by `channel_shrink_ratio` for layout projections and reshapes. The source only explicitly checks `hidden_size % num_attention_heads == 0`; DinoML should add stricter layout guards.
- `hidden_size // 6` is used for each of six bbox embeddings, then concatenated and fed to `Linear(config.hidden_size -> layout_hidden)`. This silently assumes `hidden_size` is divisible by 6; for non-divisible configs the concat width will not match the linear input.
- Config fields `use_cache`, `output_past`, and `position_embedding_type` are historical/ignored by this source basis. LiLT is not an autoregressive cache model.
- Tokenizer classes change bbox special-token policy: LayoutLMv3 uses SEP box `[0,0,0,0]`, LayoutXLM uses `[1000,1000,1000,1000]`.
- There is no image tensor branch in LiLT. Any OCR/image handling belongs outside the neural graph or in a composed processor.
- Layout stream outputs are not returned by `LiltEncoder`; only text hidden states are public. Lowering must preserve the private layout state across layers but not expose it as a normal output.

## 4. Operator coverage checklist

Tensor/layout ops:

- integer `input_ids != pad_token_id`, cast to int, `cumsum(dim=1)`, multiply mask, cast to int64, add padding offset for default position IDs.
- `arange`, `unsqueeze`, `expand` for `inputs_embeds` position IDs.
- bbox slicing `bbox[:,:,0..3]`, integer subtraction for height/width indices, `cat(dim=-1)`.
- reshape/view `[B,S,H] -> [B,S,heads,head_dim]`, `permute(0,2,1,3)`, transpose last two dims, contiguous + view back to `[B,S,H]`.
- first-token gather `hidden[:,0]`, logits split/squeeze for QA, optional argmax/decode outside compiled graph.

Neural primitives:

- Embedding lookups: word `[vocab, H]`, token type `[type_vocab, H]`, text position `[max_pos, H]`.
- Layout embeddings: x/y/h/w tables `[max_2d_pos, H//6]`; six lookups concatenate to `H`, then `Linear(H -> H/4)`.
- Layout position embedding `[max_pos, H/4]`.
- LayerNorm eps from checkpoint, residual add, dropout as identity for inference.
- Per layer text linears: Q/K/V/O `Linear(768 -> 768)`, FFN `Linear(768 -> 3072) -> GELU -> Linear(3072 -> 768)` for base checkpoints.
- Per layer layout linears: Q/K/V/O `Linear(192 -> 192)`, FFN `Linear(192 -> 768) -> GELU -> Linear(768 -> 192)`.
- Heads: token classifier `Linear(768 -> num_labels)`, QA `Linear(768 -> 2)` then split, sequence classifier first-token `Linear(768 -> 768) -> tanh -> Linear(768 -> num_labels)`.

Attention primitives:

- Noncausal dense self-attention over token sequence, no KV cache.
- Text attention score GEMM `[B,heads,S,64] x [B,heads,64,S] -> [B,heads,S,S]`.
- Layout attention score GEMM `[B,heads,S,16] x [B,heads,16,S] -> [B,heads,S,S]`.
- Score coupling: text score/sqrt(64) + layout score/sqrt(16) is used for both streams before mask/softmax.
- Additive extended attention mask broadcastable to `[B,heads,S,S]`; source adds it separately to both score tensors.
- Softmax along key dimension, context GEMMs for text and layout values.

Position/custom math:

- Absolute 1D token positions plus 2D bbox coordinate embeddings. No RoPE, ALiBi, relative bias, local attention, or packed varlen metadata.

Preprocessing-coupled ops:

- Tokenizer must expand word-level boxes to subword/token-level boxes using tokenizer `word_ids`.
- Special token and padding boxes are tokenizer-family-specific.
- Overflow chunks need `overflow_to_sample_mapping`; image/page data may be duplicated by caller if end-to-end document examples are batched.

## 5. Layer/block breakdown

Base checkpoint shapes use `B=batch`, `S<=512 tokenizer model_max_length`, `H=768`, `L=192`, `heads=12`, text head `64`, layout head `16`.

Embeddings:

```text
position_ids = cumsum(input_ids != pad_id) * mask + pad_id
text = Emb(input_ids) + Emb(token_type_ids) + Emb(position_ids)
text = LayerNorm(text)
layout6 = cat(EmbX(x0), EmbY(y0), EmbX(x1), EmbY(y1), EmbH(y1-y0), EmbW(x1-x0))
layout = Linear(768 -> 192)(layout6) + EmbLayoutPos(position_ids)
layout = LayerNorm(layout)
```

Encoder block, repeated 12 times:

```text
text_q,k,v = Linear(768 -> 768)(text) -> [B,12,S,64]
layout_q,k,v = Linear(192 -> 192)(layout) -> [B,12,S,16]
text_scores = matmul(text_q, text_k.T) / sqrt(64)
layout_scores = matmul(layout_q, layout_k.T) / sqrt(16)
joint_scores = text_scores + layout_scores + additive_mask
text_ctx = softmax(joint_scores) @ text_v -> Linear(768 -> 768) -> residual LayerNorm
layout_ctx = softmax(joint_scores) @ layout_v -> Linear(192 -> 192) -> residual LayerNorm
text = Linear(768 -> 3072) -> GELU -> Linear(3072 -> 768) -> residual LayerNorm
layout = Linear(192 -> 768) -> GELU -> Linear(768 -> 192) -> residual LayerNorm
```

All linears in the inspected source have bias because they use `nn.Linear(...)` with default `bias=True`.

## 6. Attention requirements

LiLT requires encoder self-attention only:

- noncausal bidirectional self-attention;
- MHA, not GQA/MQA;
- text Q/K/V width 768, 12 heads, head dim 64 for base checkpoints;
- layout Q/K/V width 192, 12 heads, head dim 16 for base checkpoints;
- query and key lengths are both `S`; rectangular cross-attention is not present;
- source mask path is standard Transformers additive extended mask from `attention_mask`;
- no sliding-window/local/sparse attention;
- no packed/varlen/cu-seqlens support in source;
- no KV cache, decode, beam reorder, or causal generation ABI;
- FlashAttention/SDPA is not dispatched by source, but the coupled-score pattern can be lowered to a custom fused attention family or to two score GEMMs plus shared softmax.

The source computes two softmax calls from algebraically identical tensors after score coupling. In inference, DinoML can share one masked softmax result for both text and layout context if the same additive mask and dtype/order are preserved.

## 7. Position encoding and custom math

LiLT uses learned absolute token positions and learned bbox-derived embeddings.

```python
def lilt_position_ids(input_ids, pad):
    mask = (input_ids != pad).int()
    return (cumsum(mask, dim=1) * mask).long() + pad

def lilt_layout_embedding(bbox, pos):
    x0, y0, x1, y1 = bbox[..., 0], bbox[..., 1], bbox[..., 2], bbox[..., 3]
    pieces = [emb_x(x0), emb_y(y0), emb_x(x1), emb_y(y1), emb_h(y1 - y0), emb_w(x1 - x0)]
    return layer_norm(linear(cat(pieces, -1)) + emb_box_pos(pos))
```

Precompute candidates: all learned embedding tables and position table rows. Dynamic inputs: `position_ids` if not supplied, bbox coordinate/height/width indices, and attention masks. DinoML should guard `0 <= x0 <= x1 < max_2d_position_embeddings` and `0 <= y0 <= y1 < max_2d_position_embeddings`; the source catches some out-of-range coordinate lookups but lets negative width/height fail through embedding indexing behavior.

## 8. Preprocessing and input packing

Neural input ABI:

```text
input_ids: int64 [B,S]
bbox: int64 [B,S,4], normalized x0,y0,x1,y1 token boxes
attention_mask: numeric/bool [B,S], 1 for real tokens, 0 for padding
token_type_ids: int64 [B,S], optional; official checkpoints normally use only 0
position_ids: int64 [B,S], optional
```

OCR is not invoked by LiLT. The caller supplies words and word-level boxes, typically normalized to a 0-1000 page coordinate scale. The tokenizer expands each word box to all subword tokens. For sequence pairs, LayoutLMv3/LayoutXLM assign pad boxes to the first sequence's word tokens and real boxes to the second sequence's word tokens. Padding appends or prepends `pad_token_box` depending on tokenizer padding side.

Tokenizer coupling:

- `lilt-roberta-en-base`: `LayoutLMv3Tokenizer`, BPE, `add_prefix_space=true`, special boxes `[0,0,0,0]`.
- `lilt-infoxlm-base`: `LayoutXLMTokenizer`, sentencepiece/unigram, SEP box `[1000,1000,1000,1000]`.
- `only_label_first_subword=true` affects training labels, not the inference neural graph.
- Overflow handling emits `overflow_to_sample_mapping`; the model consumes only chunked tensors.

No `pixel_values` are consumed by LiLT. If a product flow starts from PDFs/images, OCR, page resize/normalization, word ordering, and chunking are CPU/data-pipeline work for first integration.

## 9. Graph rewrite / lowering opportunities

### Rewrite: shared coupled softmax

Source pattern:

```text
text_scores = Qtext @ Ktext.T / sqrt(text_head_dim)
layout_scores = Qlayout @ Klayout.T / sqrt(layout_head_dim)
text_probs = softmax(text_scores + layout_scores + mask)
layout_probs = softmax(layout_scores + text_scores + mask)
```

Replacement:

```text
joint_probs = softmax(text_scores + layout_scores + mask)
text_ctx = joint_probs @ Vtext
layout_ctx = joint_probs @ Vlayout
```

Preconditions: dropout disabled, same additive mask, same score dtype/upcast policy, no `output_attentions` requirement that distinguishes the two tensors. Failure cases: training/dropout, attention output parity requiring object identity/order, or backend softmax precision mismatch. Test: compare one layer and full encoder logits against Transformers fp32 with random valid boxes and masks.

### Rewrite: layout six-embedding concat into gathered projection

Source pattern: six independent coordinate embedding lookups concatenate to width `H`, followed by `Linear(H -> H/r)`.

Replacement: treat the post-concat linear as six smaller projections and sum:

```text
sum_i Linear_i(Emb_i(index_i)) + bias
```

Preconditions: split `box_linear_embeddings.weight` into six contiguous width `H//6` blocks; `hidden_size % 6 == 0`; preserve shared x/y tables for left/right and upper/lower. Failure cases: non-divisible hidden size or checkpoint with unexpected table widths. Test: layout embedding parity before LayerNorm.

### Rewrite: attention projections to batched GEMM families

Source pattern: separate Q/K/V text linears and separate Q/K/V layout linears.

Replacement: either keep six GEMMs initially, or pack per-stream QKV weights into fused projection outputs with split order `[q,k,v]`.

Preconditions: all projections present with bias, identical input tensor per stream, fixed output widths. Weight transform: concatenate output-feature rows of PyTorch linear weights and biases in q,k,v order. Failure cases: partial weight override, quantized per-module metadata, or head pruning. Test: projection tensor parity before reshape.

### Rewrite: head-specific first-token classification

Source pattern: `features[:,0,:] -> dropout -> Linear -> tanh -> dropout -> out_proj`.

Replacement: first-token gather plus two GEMMs. For batch-one low latency, specialize gather and classifier head separately from full sequence token logits.

Preconditions: sequence classification head only. Failure cases: token classification or QA heads.

### Layout/no-layout guards

The model tensors are rank-3 token sequences, so NHWC/NCHW image layout translation is not relevant inside LiLT. Protect tokenizer/OCR output ordering, bbox axis order, `cat(dim=-1)`, softmax `dim=-1`, first-token gather, and QA split `dim=-1` with no-layout-translation guards.

## 10. Kernel fusion candidates

Highest priority:

- LayerNorm + residual add for text width 768 and layout width 192.
- Dense bidirectional attention with coupled text/layout scores and shared softmax.
- GEMM coverage for small layout stream projections and FFNs; layout FFN `192 -> 768 -> 192` is narrow but repeated.
- Embedding-add-LayerNorm for text and layout embeddings, including bbox guard/index path.

Medium priority:

- Fused QKV projection packing per stream.
- Layout six-embedding projection rewrite to reduce concat temporary bandwidth.
- Token-classification head fusion for `dropout(identity) -> Linear`.
- QA head split/squeeze as metadata/view lowering after `Linear(768 -> 2)`.

Lower priority:

- Sequence classification first-token-only specialization.
- Output attentions materialization; useful for debugging but expensive and optional for first runtime.
- Chunked feed-forward path; source config does not set nonzero chunking for representative checkpoints.

## 11. Runtime staging plan

Stage 1: parse LiLT config and reject unsupported combinations with clear guards: no cache/generation, `hidden_size % heads == 0`, `hidden_size % channel_shrink_ratio == 0`, `head_dim % channel_shrink_ratio == 0`, `hidden_size % 6 == 0`, bbox rank `[B,S,4]`, valid bbox monotonic ranges.

Stage 2: load base encoder weights and run embedding + one encoder block parity with caller-supplied `input_ids`, `bbox`, and `attention_mask`.

Stage 3: full `LiltModel` parity for `last_hidden_state`; keep pooler optional.

Stage 4: token classification head parity as first product target. Stub OCR, tokenizer, and label postprocessing outside DinoML.

Stage 5: add sequence classification and QA heads. QA postprocessing can remain controller-side.

Stage 6: optimize coupled attention/shared softmax, QKV packing, and layout embedding projection rewrite.

Stage 7: add processor integration adapters for LayoutLMv3/LayoutXLM tokenizer metadata and overflow chunk bookkeeping, still outside the compiled neural graph.

## 12. Parity and validation plan

- Config/shape guard tests for official configs, tiny debug config, and deliberately invalid bbox/shape combinations.
- Layout embedding random parity with valid boxes, special-token boxes, and max coordinate/height/width edge cases.
- Default `position_ids` parity for padded and unpadded `input_ids`; separate `inputs_embeds` sequential-position path if admitted.
- One-block parity for text and private layout stream outputs in fp32.
- Full encoder `last_hidden_state` parity for `B=1/2`, `S=16/128/512`, with padding masks.
- Token classification logits parity on a fine-tuned checkpoint config with synthetic tensors; recommended tolerance fp32 `atol=1e-4`, fp16/bf16 after enablement `atol=5e-3` or model-specific calibration.
- QA head parity for `Linear(768 -> 2)` plus split/squeeze.
- End-to-end tokenizer-to-model smoke using pre-tokenized words/boxes from FUNSD-style examples; OCR itself is not a DinoML parity target initially.

## 13. Performance probes

- Tokenizer/OCR/data-pipeline throughput separately from neural encoder throughput.
- Encoder-only latency/throughput sweep over `B in {1,4,8}` and `S in {128,256,512}`.
- Attention backend comparison: unfused two-softmax source-equivalent path vs shared-softmax coupled attention.
- Layout stream overhead probe: text-only BERT-like block vs paired text/layout LiLT block.
- Embedding path bandwidth: six bbox lookups + concat + linear vs split-projection rewrite.
- Head-only probes for token classification, sequence classification, and QA.
- Memory footprint sweep for materialized `[B,heads,S,S]` scores/probs; LiLT has no KV cache but full dense attention still scales quadratically in `S`.

## 14. Skip/defer list

- Training losses, dropout behavior, and gradient checkpointing.
- Autoregressive cache/generation despite historical `use_cache` config fields.
- OCR, image decoding, PDF parsing, and word ordering inside the compiled graph.
- `output_attentions` and `output_hidden_states` materialization for optimized production path.
- Non-token-classification heads can follow after base/token parity.
- Quantized or packed weights; no source-coupled quantization format is present.
- Multi-GPU/tensor parallel.
- General boolean scatter, image-layout NHWC rewrites, RoPE/relative-position kernels, and local/sparse attention are not required for LiLT.

## 15. Final implementation checklist

- [ ] Parse `LiltConfig` and checkpoint tokenizer metadata.
- [ ] Load text, layout, encoder, and selected task-head weights with source names preserved.
- [ ] Add LiLT admission guards for bbox rank/range/monotonicity and layout divisibility.
- [ ] Implement default position ID generation or require explicit `position_ids` for first slice.
- [ ] Implement text embedding add + LayerNorm.
- [ ] Implement bbox layout embedding path and special-token box ABI.
- [ ] Implement paired text/layout encoder block.
- [ ] Add coupled-attention shared-softmax rewrite with strict parity guards.
- [ ] Add token classification head as first task target.
- [ ] Add one-block, full-encoder, and token-logit parity tests.
- [ ] Benchmark encoder and coupled-attention variants.
