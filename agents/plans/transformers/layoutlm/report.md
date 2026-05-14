# LayoutLM DinoML operator assessment

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: microsoft/layoutlm-base-uncased, microsoft/layoutlm-large-uncased, microsoft/layoutlm-base-cased, impira/layoutlm-document-qa
Config source: Hugging Face config/tokenizer JSON snapshots under _sources/hf_configs plus source defaults in configuration_layoutlm.py
Source files inspected:
  X:/H/transformers/src/transformers/models/layoutlm/modeling_layoutlm.py
  X:/H/transformers/src/transformers/models/layoutlm/configuration_layoutlm.py
  X:/H/transformers/src/transformers/models/layoutlm/__init__.py
  X:/H/transformers/src/transformers/models/auto/tokenization_auto.py
  X:/H/transformers/src/transformers/models/auto/modeling_auto.py
Any missing files or assumptions:
  LayoutLM has no family-local tokenization, processing, or image-processing file in this checkout. LayoutLMTokenizer aliases BERT tokenizer classes through __init__.py/auto mappings. Bbox alignment is caller/data-pipeline supplied.
```

Primary source URLs at the pinned commit:

- `modeling_layoutlm.py`: https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/layoutlm/modeling_layoutlm.py
- `configuration_layoutlm.py`: https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/layoutlm/configuration_layoutlm.py
- `__init__.py`: https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/layoutlm/__init__.py
- `tokenization_auto.py`: https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/auto/tokenization_auto.py

Representative checkpoint snapshots:

- `microsoft/layoutlm-base-uncased`: config, tokenizer config, special tokens, vocab accessible.
- `microsoft/layoutlm-large-uncased`: config, tokenizer config, special tokens, vocab accessible.
- `microsoft/layoutlm-base-cased`: config/tokenizer/special tokens accessible; `vocab.txt` returned 404 because this checkpoint declares `tokenizer_class: RobertaTokenizer` and uses RoBERTa-style files rather than BERT `vocab.txt`.
- `impira/layoutlm-document-qa`: config/tokenizer/special tokens accessible; `vocab.txt` returned 404 for the same RoBERTa-tokenizer reason.
- [microsoft/layoutlm-base-uncased-finetuned-funsd](https://huggingface.co/microsoft/layoutlm-base-uncased-finetuned-funsd) returned 401 for raw config/tokenizer/special-token/vocab files. Access would resolve task-head label metadata and whether the fine-tune changes `num_labels`; it should not change the encoder operator surface.

## 2. High-level architecture

LayoutLM is a text-and-layout encoder. The model consumes token IDs plus one normalized 2D bounding box per token and runs a BERT-style bidirectional encoder. There is no image branch in this family; OCR/image parsing and bbox normalization happen before the model graph.

```text
OCR/word boxes + tokenizer alignment on CPU -> token IDs + bbox[B,S,4] -> text/layout embeddings -> bidirectional encoder -> pooled/token/logit heads
```

Primary DinoML runtime target: encoder inference for document understanding, with token classification and question answering heads prioritized. Masked LM and sequence classification heads are implemented and straightforward optional heads. Autoregressive prefill/decode and KV-cache scheduling are not applicable.

## 3. Important config dimensions

Source defaults from `LayoutLMConfig`:

| Field | Default |
|---|---:|
| `vocab_size` | 30522 |
| `hidden_size` | 768 |
| `num_hidden_layers` | 12 |
| `num_attention_heads` | 12 |
| `head_dim` | 64 |
| `intermediate_size` | 3072 |
| `max_position_embeddings` | 512 |
| `max_2d_position_embeddings` | 1024 |
| `type_vocab_size` | 2 |
| `hidden_act` | `gelu` |
| `layer_norm_eps` | `1e-12` |
| `tie_word_embeddings` | `True` |
| `use_cache` | `True` in config only; not used by current modeling source |

Representative checkpoint sweep:

| Checkpoint | Task/head | Hidden | Layers | Heads | Head dim | FFN | Vocab | 1D pos | 2D pos | Type vocab | LN eps | Tokenizer |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `microsoft/layoutlm-base-uncased` | base/MLM | 768 | 12 | 12 | 64 | 3072 | 30522 | 512 | 1024 | 2 | `1e-12` | BERT uncased |
| `microsoft/layoutlm-large-uncased` | base/MLM | 1024 | 24 | 16 | 64 | 4096 | 30522 | 512 | 1024 | 2 | `1e-12` | BERT uncased |
| `microsoft/layoutlm-base-cased` | base | 768 | 12 | 12 | 64 | 3072 | 50265 | 514 | 1024 | 1 | `1e-5` | RoBERTa |
| `impira/layoutlm-document-qa` | QA | 768 | 12 | 12 | 64 | 3072 | 50265 | 514 | 1024 | 1 | `1e-5` | RoBERTa, `add_prefix_space: true` |

No RoPE/theta, MoE, grouped-query attention, sliding windows, vision/audio dimensions, or patch parameters are present for native LayoutLM.

## 3a. Family variation traps

- Native LayoutLM is not LayoutLMv2/v3: no visual backbone, no image processor, no spatial-aware attention bias, and no patch/image tokens.
- The model adds nine embedding sources before LayerNorm: word, 1D position, left x, upper y, right x, lower y, bbox height, bbox width, plus token type.
- `bbox` must be integral and within embedding-table bounds after deriving `h = y1 - y0` and `w = x1 - x0`. The source catches out-of-range x/y lookups but does not separately validate negative width/height before embedding lookup.
- `use_cache` and historical `output_past` appear in configs, but current `LayoutLMModel.forward` has no `past_key_values`, no cache object, and no decode path. Treat them as ignored for this source basis.
- `LayoutLMForQuestionAnswering.__init__` accepts `has_visual_segment_embedding`, but the argument is not used by the native source. Do not infer a visual segment/image branch from that signature.
- BERT-style checkpoints use `type_vocab_size=2`; RoBERTa-style LayoutLM checkpoints use `type_vocab_size=1`, `max_position_embeddings=514`, `pad_token_id=1`, and `layer_norm_eps=1e-5`.
- `microsoft/layoutlm-base-cased` declares `tokenizer_class: RobertaTokenizer` despite `model_type: layoutlm`; tokenizer/model coupling must come from the config/tokenizer files, not from family name alone.
- MLM decoder weights are tied to `layoutlm.embeddings.word_embeddings.weight`; the output bias is a separate logical parameter aliased through `_tied_weights_keys`.
- Layout is `[batch, sequence, hidden]`; NHWC is irrelevant except for downstream OCR/image pipelines outside this model. Put the encoder under `no_layout_translation()` style guards.

## 4. Operator coverage checklist

### Tensor/layout ops

- Integer gather/embedding lookup for `input_ids[B,S]`, `position_ids[1,S]` or `[B,S]`, `token_type_ids[B,S]`, and `bbox[B,S,4]`.
- Slice bbox channels `x0,y0,x1,y1`.
- Integer subtract for bbox width/height indices: `w = x1 - x0`, `h = y1 - y0`.
- Elementwise add chain over nine `[B,S,H]` tensors.
- Unsqueeze attention mask `[B,S] -> [B,1,1,S]`, cast to runtime dtype, transform to additive mask `(1 - mask) * finfo_min`.
- Reshape/view/transpose/contiguous for attention: `[B,S,H] -> [B,heads,S,head_dim]`, output back to `[B,S,H]`.
- First-token indexing for pooler: `hidden[:,0]`.
- Optional chunked FFN along sequence dim for `chunk_size_feed_forward`; first integration can require `0`/disabled.

### Neural network primitives

Base/uncased shapes:

- Embeddings: `word(30522 x 768)`, `position(512 x 768)`, `x/y/h/w 2D tables(1024 x 768 each)`, `token_type(2 x 768)`.
- LayoutLM-large: same operators with `H=1024`, `layers=24`, `heads=16`, `FFN=4096`.
- LayerNorm over last dim with eps from config (`1e-12` or `1e-5`).
- Linear with bias for Q/K/V: `Linear(H -> H)`.
- Attention output projection: `Linear(H -> H)`.
- FFN: `Linear(H -> intermediate) -> GELU -> Linear(intermediate -> H)`.
- Pooler: `Linear(H -> H) -> tanh`.
- Token/sequence classifiers: `Linear(H -> num_labels)`.
- QA head: `Linear(H -> 2)` then split/squeeze.
- MLM head: `Linear(H -> H) -> GELU -> LayerNorm -> tied vocab projection Linear(H -> vocab_size) + output bias`.

### Attention primitives

- Dense bidirectional self-attention, MHA only.
- Additive padding mask broadcast over heads and query positions.
- Softmax in fp32 in eager path, downcast to query dtype, dropout only in training.
- Source can dispatch through `ALL_ATTENTION_FUNCTIONS` using `config._attn_implementation`; eager parity must be preserved first.

### Position/layout ops

- Learned 1D absolute position embedding.
- Learned 2D x/y/h/w embedding tables.
- No RoPE, ALiBi, relative bias, or convolutional positions.

### Generation/cache ops

- No required generation, no KV cache, no beam reorder, no causal mask.
- Config `use_cache`/`output_past` should be ignored or rejected for LayoutLM execution plans.

### Preprocessing-coupled ops

- Tokenization is BERT or RoBERTa depending on checkpoint tokenizer config.
- Caller must align word boxes to subword tokens and insert bbox entries for special tokens.
- Runtime graph consumes already packed `input_ids`, `attention_mask`, `token_type_ids`, optional `position_ids`, and `bbox`.

## 5. Layer/block breakdown

Embedding stage for `[B,S]` tokens and `[B,S,4]` boxes:

```text
word = Embedding(input_ids)                         # [B,S,H]
pos = Embedding(position_ids or arange(S))          # [1,S,H] broadcast or [B,S,H]
x0 = x_position_embeddings(bbox[:,:,0])             # [B,S,H]
y0 = y_position_embeddings(bbox[:,:,1])
x1 = x_position_embeddings(bbox[:,:,2])
y1 = y_position_embeddings(bbox[:,:,3])
h = h_position_embeddings(bbox[:,:,3] - bbox[:,:,1])
w = w_position_embeddings(bbox[:,:,2] - bbox[:,:,0])
typ = token_type_embeddings(token_type_ids)
x = LayerNorm(word + pos + x0 + y0 + x1 + y1 + h + w + typ)
x = Dropout(x)                                      # inference identity
```

Encoder block, repeated `N` times:

```text
residual = x
q = Linear(H -> H, bias=True)(x).view(B,S,heads,head_dim).transpose(1,2)
k = Linear(H -> H, bias=True)(x).view(B,S,heads,head_dim).transpose(1,2)
v = Linear(H -> H, bias=True)(x).view(B,S,heads,head_dim).transpose(1,2)
a = softmax((q @ k^T) * head_dim^-0.5 + additive_padding_mask, dim=-1)
x = (a @ v).transpose(1,2).reshape(B,S,H)
x = LayerNorm(Dropout(Linear(H -> H, bias=True)(x)) + residual)
residual = x
x = Linear(H -> intermediate, bias=True)(x)
x = GELU(x)
x = LayerNorm(Dropout(Linear(intermediate -> H, bias=True)(x)) + residual)
```

Heads:

- Pooler: take token 0, `Linear(H -> H)`, tanh.
- Sequence classification: pooler output, dropout identity in inference, `Linear(H -> num_labels)`.
- Token classification: sequence output, dropout identity, `Linear(H -> num_labels)`.
- QA: sequence output, `Linear(H -> 2)`, split last dim to start/end `[B,S]`.
- MLM: sequence output, transform `Linear(H -> H) + GELU + LayerNorm`, tied vocab projection.

## 6. Attention requirements

- Type: noncausal encoder self-attention.
- Heads: MHA, no GQA/MQA. Base has 12 heads x 64, large has 16 heads x 64.
- Query/key/value width: all `H`; head dim is `H / num_attention_heads`.
- Query length equals key/value length `S`; no rectangular cross-attention.
- Masking: padding-only additive mask `[B,1,1,S]` using dtype minimum for masked positions. It applies to keys and broadcasts over all queries/heads.
- Packed/varlen: not implemented in source.
- Sliding/local/block sparse: not implemented.
- Position interactions: position/layout information is embedded before attention; attention scores have no relative layout bias.
- KV cache: absent. `use_cache` is config-only in this source.
- FlashAttention/SDPA compatibility: mathematically compatible with dense bidirectional attention if backend supports noncausal additive key padding masks and fp32-softmax parity requirements. Eager path is O(`B * heads * S^2 * head_dim`) and likely too slow for long documents, but `S` is normally capped at 512/514.

## 7. Position encoding and custom math

The only custom position math is the 2D bbox embedding composition. This can be implemented as integer index prep plus embedding gathers.

```python
def layoutlm_bbox_embeddings(bbox, x_table, y_table, h_table, w_table):
    x0 = x_table[bbox[:, :, 0]]
    y0 = y_table[bbox[:, :, 1]]
    x1 = x_table[bbox[:, :, 2]]
    y1 = y_table[bbox[:, :, 3]]
    h = h_table[bbox[:, :, 3] - bbox[:, :, 1]]
    w = w_table[bbox[:, :, 2] - bbox[:, :, 0]]
    return x0 + y0 + x1 + y1 + h + w
```

Precomputable:

- Static `position_ids = arange(max_position_embeddings)` and position table weights.
- Special-token bbox policy if the tokenizer pipeline standardizes it.

Dynamic per batch:

- Tokenized subword layout and bbox indices.
- Width/height derived from per-token bbox.
- Attention mask conversion.

## 8. Preprocessing and input packing

CPU/data-pipeline responsibilities:

- OCR or external parser produces words and normalized boxes in `[0, max_2d_position_embeddings - 1]`, typically `[0,1000]` for public examples.
- Tokenizer splits words into subwords. Each subword inherits its word bbox.
- Special tokens need explicit boxes. Source examples use `[0,0,0,0]` for CLS/non-document tokens and `[1000,1000,1000,1000]` for SEP in some paths.
- For QA, tokenizer packs question and document words; boxes for question tokens are usually zero, document tokens use word boxes, and SEP uses all-1000.
- BERT checkpoints use `[CLS] text [SEP]`; RoBERTa checkpoints use `<s> ... </s>` and often `add_prefix_space`.

GPU/runtime graph inputs:

- `input_ids[B,S]`
- `bbox[B,S,4]` as integer indices
- `attention_mask[B,S]`
- `token_type_ids[B,S]`; defaults to zeros if absent and may have only one valid value for RoBERTa-style checkpoints.
- optional `position_ids[B,S]` or default arange slice

No image tensors, pixel masks, grids, `cu_seqlens`, modality token IDs, or placeholder scatter are used by native LayoutLM.

## 9. Graph rewrite / lowering opportunities

### Rewrite: LayoutLM embedding fan-in fusion

Source pattern:

```text
Embedding(word) + Embedding(pos) + Embedding(token_type)
+ Embedding(x0) + Embedding(y0) + Embedding(x1) + Embedding(y1)
+ Embedding(y1-y0) + Embedding(x1-x0)
-> LayerNorm -> Dropout
```

Replacement pattern:

```text
FusedGatherAdd9 -> LayerNorm
```

Preconditions:

- Inference mode so dropout is identity.
- All embedding outputs have identical `[B,S,H]`.
- Bbox indices and derived width/height are in range `[0, max_2d_position_embeddings - 1]`.
- `position_ids` are either explicit or equal to the default arange slice.

Shape equations:

- `word_ids, token_type_ids, position_ids: [B,S]`
- `bbox: [B,S,4]`
- output `[B,S,H]`

Weight transform:

- None; keep separate tables to preserve weight names and checkpoint loading.

Layout constraints:

- Sequence-major semantic layout `[B,S,H]` must remain visible. Use `no_layout_translation()` around this region; no NHWC rewrite applies.

Failure cases:

- Negative `x1-x0` or `y1-y0`.
- Position length exceeds `max_position_embeddings`.
- RoBERTa-style checkpoint with nonzero token type IDs when `type_vocab_size=1`.

Parity test sketch:

- Random in-range IDs/boxes against source embedding module in fp32 with dropout disabled; include special boxes and RoBERTa-style `type_vocab_size=1`.

### Rewrite: separate Q/K/V projections -> packed QKV GEMM

Source pattern:

```text
q = Linear(H,H)(x); k = Linear(H,H)(x); v = Linear(H,H)(x)
```

Replacement:

```text
qkv = MatMul(x, Wqkv.T) + bqkv -> split [Q,K,V] along last dim
```

Preconditions:

- Same input tensor `x` feeds all three projections.
- All projections have bias and output width `H`.
- Split order is all-Q, all-K, all-V because source has separate Linear modules.

Shape equations:

- `x: [B,S,H]`
- `Wqkv: [3H,H]`, `bqkv: [3H]`
- split to three `[B,S,H]`, then reshape to `[B,heads,S,head_dim]`.

Weight transform:

```python
Wqkv = concat([Wq, Wk, Wv], axis=0)
bqkv = concat([bq, bk, bv], axis=0)
```

Layout constraints:

- Preserve `[B,S,H]`; optional internal attention kernels may choose packed head layouts.

Failure cases:

- Quantized checkpoints with incompatible packed storage, pruned heads, or missing biases.

Parity test sketch:

- Compare separate projections and packed projection for base and large shapes.

### Rewrite: attention mask canonicalization

Source pattern:

```text
mask = attention_mask.unsqueeze(1).unsqueeze(2)
mask = (1.0 - mask.to(dtype)) * finfo(dtype).min
```

Replacement:

```text
KeyPaddingMask[B,S] -> backend additive/boolean mask
```

Preconditions:

- Mask is binary 1 keep / 0 pad.
- Backend applies mask to key positions before softmax with equivalent numerical behavior.

Failure cases:

- Non-binary masks or user-supplied additive masks masquerading as `attention_mask`.

Parity test sketch:

- Include all-valid, right-padded, and interspersed masked tokens; compare attention output/logits.

### Rewrite: MLM tied vocab projection

Source pattern:

```text
hidden -> Linear(H,H) -> GELU -> LayerNorm -> Linear(H,V) with decoder.weight tied to word_embeddings.weight
```

Replacement:

```text
HiddenTransform -> MatMul(hidden, word_embedding_weight.T) -> BiasAdd
```

Preconditions:

- `_tied_weights_keys` active and decoder weight aliases input embedding weight.
- No untied output embedding override was loaded.

Failure cases:

- Custom checkpoint with untied decoder or resized embeddings not reflected consistently.

Parity test sketch:

- Verify pointer/weight identity at load manifest level, then compare logits.

## 10. Kernel fusion candidates

Highest priority:

- Embedding gather-add fan-in plus LayerNorm: LayoutLM spends a distinctive amount of front-end work gathering eight position/layout tables and summing them before every encoder run.
- Packed QKV projection plus attention: same dense encoder pattern as BERT; high reuse across document encoder families.
- Bias-dropout-residual-LayerNorm collapsed for inference as `Linear -> residual add -> LayerNorm`: dropout is identity but source ordering must be preserved.

Medium priority:

- GELU FFN fusion: `Linear -> GELU -> Linear` dominates encoder FLOPs after attention.
- Attention mask conversion and dense noncausal SDPA backend: reduce repeated mask materialization and improve attention throughput.
- QA head split/squeeze fusion: small but easy for document QA latency.

Lower priority:

- Pooler `first-token -> Linear -> tanh` fusion for sequence classification.
- Last-layer-only output materialization when hidden states/attentions are not requested.
- MLM transform plus tied vocab projection fusion for masked LM deployments.

## 11. Runtime staging plan

1. Parse LayoutLM config and reject/ignore config-only cache fields for native source parity.
2. Load base encoder weights, preserving MLM tied embedding aliases when the MLM head is present.
3. Implement embedding stage with bbox-derived h/w indices and default `position_ids`/`token_type_ids`.
4. Implement one encoder block parity for dense bidirectional attention and FFN.
5. Run full encoder parity for base and RoBERTa-tokenizer configs.
6. Add token classification and QA heads as first document-understanding targets.
7. Add sequence classification and MLM heads.
8. Enable packed QKV, fused embedding fan-in, fused residual LayerNorm, and optimized dense attention.

Initial stubs allowed:

- Training losses.
- Dropout as identity under inference.
- `chunk_size_feed_forward` rejected unless zero/disabled.
- Hidden-state/attention output collection disabled until needed.

## 12. Parity and validation plan

- Random bbox embedding tests: in-range boxes, zero boxes, all-1000 boxes, RoBERTa-style `type_vocab_size=1`, explicit and default position IDs.
- Negative/admission tests: out-of-range bbox, negative width/height, `S > max_position_embeddings`, invalid token type for `type_vocab_size=1`.
- Single attention block parity in fp32 for base and large dimensions with all-valid and padded masks.
- Full encoder parity after 1, 6, 12, and 24 layers, depending on checkpoint.
- Head parity: token classification logits `[B,S,num_labels]`, QA start/end logits `[B,S]`, sequence logits `[B,num_labels]`, MLM logits `[B,S,V]`.
- End-to-end document QA packing parity against `impira/layoutlm-document-qa` using precomputed tokenizer outputs and boxes.
- Recommended tolerances: fp32 `rtol=1e-4, atol=1e-5`; fp16/bf16 `rtol=5e-2, atol=5e-2` for full encoder logits, tighter for isolated linear/embedding ops.

## 13. Performance probes

- CPU tokenizer/OCR-box alignment throughput separate from GPU graph.
- Embedding fan-in time as a function of `B,S,H`.
- Encoder throughput for base vs large at `S=128,256,512/514`.
- Attention backend comparison: eager math parity, SDPA/Flash-style dense noncausal backend, packed QKV.
- FFN throughput and GELU fusion effect.
- Head-only cost for token classification vs QA vs MLM vocab projection.
- Memory probe for attention score tensors at large batch and sequence cap.
- Batch-size sweep for document batches with variable padding ratios.

## 14. Skip/defer list

- Training losses and gradient checkpointing.
- Autoregressive generation, beam search, speculative decode, and KV cache.
- LayoutLMv2/v3 image branches and visual embeddings; those are separate family audits.
- OCR/image preprocessing and bbox normalization implementation, beyond accepting model-ready tensors.
- Quantization and multi-GPU tensor parallelism.
- Head pruning and nonzero `chunk_size_feed_forward` until a checkpoint requires it.

## 15. Final implementation checklist

- [ ] Parse LayoutLM config and tokenizer-coupling metadata.
- [ ] Load encoder weights and preserve MLM tied embedding aliases.
- [ ] Implement word/1D/2D/token-type embedding gather-add with bbox h/w derivation.
- [ ] Implement dense bidirectional MHA with additive key padding mask.
- [ ] Implement GELU FFN and residual LayerNorm ordering.
- [ ] Implement pooler, token classification, QA, sequence classification, and MLM heads.
- [ ] Add admission guards for bbox range, width/height nonnegative, sequence length, token type range, and ignored cache fields.
- [ ] Add rewrite for embedding fan-in fusion.
- [ ] Add rewrite for packed QKV projection.
- [ ] Add rewrite for key-padding-mask canonicalization.
- [ ] Add parity tests for embedding stage, one block, full encoder, and heads.
- [ ] Benchmark embedding fan-in, encoder attention/FFN, and head-specific throughput.
