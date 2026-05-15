# DPR Transformers Family Audit

## 1. Source basis

```text
Transformers commit/version:
  Local checkout transformers at b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id:
  facebook/dpr-question_encoder-single-nq-base
  facebook/dpr-ctx_encoder-single-nq-base
  facebook/dpr-reader-single-nq-base
  facebook/dpr-question_encoder-multiset-base
  facebook/dpr-ctx_encoder-multiset-base
Config source:
  Hugging Face resolve/main config.json snapshots under _sources/
Source files inspected:
  transformers/src/transformers/models/dpr/configuration_dpr.py
  transformers/src/transformers/models/dpr/modeling_dpr.py
  transformers/src/transformers/models/dpr/tokenization_dpr.py
  transformers/src/transformers/models/dpr/tokenization_dpr_fast.py
  transformers/src/transformers/models/dpr/convert_dpr_original_checkpoint_to_pytorch.py
  transformers/src/transformers/models/bert/modeling_bert.py
  transformers/src/transformers/models/bert/configuration_bert.py
  transformers/src/transformers/masking_utils.py
Any missing files or assumptions:
  No remote-code files are required for the sampled official checkpoints.
  special_tokens_map.json returned 404 for sampled repos; tokenizer behavior is
  BERT WordPiece plus repo tokenizer_config.json and BERT defaults.
```

Source URLs for reproducibility:

- [DPR modeling source](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/dpr/modeling_dpr.py)
- [DPR configuration source](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/dpr/configuration_dpr.py)
- [DPR tokenizer source](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/dpr/tokenization_dpr.py)
- [BERT modeling source used by DPR](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/bert/modeling_bert.py)

Primary DinoML runtime target for this report: retrieval encoders,
`DPRQuestionEncoder` and `DPRContextEncoder`, returning pooled dense embeddings.
`DPRReader` is a secondary optional target because it has different input
packing and span/relevance heads.

## 2. High-level architecture

DPR is not an autoregressive text-generation model. It is a BERT-family
encoder used in two independent retrieval branches:

```text
question text -> WordPiece packing -> BERT encoder -> first-token pooling -> optional projection -> question embedding
context title/text -> WordPiece packing -> BERT encoder -> first-token pooling -> optional projection -> context embedding
question embeddings x context embeddings^T -> retrieval scores outside model class
```

The in-library neural body is delegated to `BertModel(config,
add_pooling_layer=False)`. DPR does not use BERT's tanh pooler. Instead it takes
`sequence_output[:, 0, :]` directly and optionally applies `encode_proj:
Linear(hidden_size -> projection_dim)` when `projection_dim > 0`.

Stage decomposition:

- CPU/data pipeline: WordPiece tokenization, `[CLS]`/`[SEP]` packing, optional
  question/title/text packing for reader, right padding, attention mask and
  token type IDs.
- GPU/runtime branch: embedding lookup, absolute position and token type
  embeddings, bidirectional BERT encoder, first-token gather, optional dense
  projection.
- Retrieval post-step: dot-product similarity matrix between independently
  cacheable question and context embeddings. The sampled model classes do not
  normalize embeddings or own ANN/index lookup.
- Reader optional branch: same encoder body plus token-wise start/end logits
  and first-token passage relevance logits, followed by tokenizer-owned span
  decoding/postprocessing.

## 3. Important config dimensions

Effective source defaults from `DPRConfig`:

| Field | Default | Runtime effect |
|---|---:|---|
| `vocab_size` | 30522 | Word embedding rows |
| `hidden_size` | 768 | Encoder width and default embedding width |
| `num_hidden_layers` | 12 | BERT encoder blocks |
| `num_attention_heads` | 12 | MHA heads |
| `head_dim` | 64 | Inferred as `hidden_size / num_attention_heads` |
| `intermediate_size` | 3072 | FFN up projection width |
| `hidden_act` | `gelu` | FFN activation |
| `max_position_embeddings` | 512 | Absolute position embedding table |
| `type_vocab_size` | 2 | Token type embedding rows |
| `layer_norm_eps` | `1e-12` | BERT LayerNorm epsilon |
| `projection_dim` | 0 | Optional DPR pooled projection; `0` disables it |
| `is_decoder` | `False` | Retrieval target is bidirectional encoder, no cache |
| `add_cross_attention` | `False` | No cross-attention for sampled DPR targets |
| `pad_token_id` | 0 | Default attention mask from input IDs |

Representative checkpoint sweep:

| Checkpoint | Architecture | h/layers/heads/ffn | `max_position_embeddings` | `projection_dim` | Tokenizer note |
|---|---|---:|---:|---:|---|
| `facebook/dpr-question_encoder-single-nq-base` | `DPRQuestionEncoder` | 768/12/12/3072 | 512 | 0 | `do_lower_case=true` |
| `facebook/dpr-ctx_encoder-single-nq-base` | `DPRContextEncoder` | 768/12/12/3072 | 512 | 0 | `do_lower_case=true` |
| `facebook/dpr-reader-single-nq-base` | `DPRReader` | 768/12/12/3072 | 512 | 0 | `do_lower_case=true` |
| `facebook/dpr-question_encoder-multiset-base` | `DPRQuestionEncoder` | 768/12/12/3072 | 512 | 0 | `do_lower_case=true` |
| `facebook/dpr-ctx_encoder-multiset-base` | `DPRContextEncoder` | 768/12/12/3072 | 512 | 0 | `do_lower_case=true` |

All sampled configs share operator-significant dimensions. The source supports
`projection_dim > 0`, but the sampled official configs do not exercise it.

## 3a. Family variation traps

- DPR weights are branch-specific. Question and context encoders have the same
  architecture but are separate modules/checkpoints and should not be treated as
  tied weights.
- `projection_dim=0` means pooled output width is `hidden_size`; if positive,
  `encode_proj.weight` changes retrieval embedding width and reader head input
  width.
- DPR uses direct CLS hidden state pooling, not BERT's `BertPooler` dense+tanh.
- Retrieval similarity is outside `DPRQuestionEncoder`/`DPRContextEncoder`.
  First integration should expose embeddings and optionally a separate
  `Q @ C.T` score op with explicit orientation.
- `DPRReader` does not accept `token_type_ids` in its public forward, even
  though its internal BERT embeddings can consume token types. Its tokenizer
  `model_input_names` are `input_ids` and `attention_mask`.
- Context/question encoders default missing `attention_mask` to
  `input_ids != pad_token_id`; reader defaults missing `attention_mask` to all
  ones.
- Fast DPR tokenizer classes inherit `BertTokenizer` in this checkout rather
  than `BertTokenizerFast`; do not infer tokenizers-library-only ABI from class
  name without checking the active tokenizer instance.
- `gradient_checkpointing` appears in checkpoint configs but is training-only
  for this inference audit.
- BERT source has decoder/cache/cross-attention branches, but DPR defaults set
  `is_decoder=False` and `add_cross_attention=False`; route such configs to a
  BERT decoder audit or reject for DPR retrieval parity.
- No NHWC/NCHW layout translation is relevant: runtime tensors are token
  sequences `[B, S, H]`. Layout guards should protect sequence axis `dim=1` and
  feature axis `dim=-1`.

## 4. Operator coverage checklist

Required for retrieval encoders:

- Tensor/layout ops: embedding lookup, arange/slice position IDs, expand/gather
  for default token type IDs, broadcast add, reshape/view, transpose, contiguous,
  first-token gather `sequence_output[:, 0, :]`.
- Neural primitives: `Embedding(vocab_size -> hidden_size)`,
  `Embedding(max_position_embeddings -> hidden_size)`,
  `Embedding(type_vocab_size -> hidden_size)`, LayerNorm over `H` with epsilon
  `1e-12`, Linear with bias, GELU, residual add.
- Attention primitives: bidirectional self-attention, MHA with Q/K/V
  `Linear(768 -> 768)`, head shape `[B, 12, S, 64]`, score scaling
  `head_dim**-0.5`, additive padding mask, softmax over key length, value matmul.
- Position encoding: learned absolute position embeddings only; no RoPE, ALiBi,
  relative bias, local attention, or sliding window.
- Retrieval head: CLS row gather, optional `Linear(768 -> projection_dim)`.
- Similarity/post-step: optional matrix multiply `scores = question_embeddings
  @ context_embeddings.T`; no source-owned normalization or logit scale.
- Preprocessing-coupled ops: BERT WordPiece IDs, `[CLS]`, `[SEP]`, `[PAD]`,
  `token_type_ids`, attention mask.

Optional for `DPRReader`:

- Token-wise `Linear(embedding_width -> 2)`, `split(dim=-1)`, `squeeze(-1)`,
  `contiguous`, `view(n_passages, sequence_length)`.
- Relevance `Linear(embedding_width -> 1)` on CLS, then `view(n_passages)`.
- CPU/tokenizer postprocessing: sort relevance logits, locate second `[SEP]`,
  ignore padding tail, enumerate start/end spans up to `max_answer_length`, sort
  by start+end score, suppress nested overlaps, decode WordPiece spans.

Not required for primary target:

- KV cache, causal mask, cross-attention, generation logits, beam search.
- Image/audio/video preprocessing or layout ops.
- Quantized or packed DPR-specific weight metadata.

## 5. Layer/block breakdown

Embedding stage:

```text
input_ids: [B, S] int64
token_type_ids: [B, S] int64, default zeros for question/context encoders
position_ids: [1, S] int64 from learned absolute table

x = word_embedding(input_ids)                # [B, S, 768]
x = x + token_type_embedding(token_type_ids) # [B, S, 768]
x = x + position_embedding(position_ids)     # [B, S, 768]
x = LayerNorm(x, eps=1e-12)
```

BERT encoder block, repeated 12 times for sampled configs:

```text
q = Linear(768 -> 768, bias=True)(x).view(B, S, 12, 64).transpose(1, 2)
k = Linear(768 -> 768, bias=True)(x).view(B, S, 12, 64).transpose(1, 2)
v = Linear(768 -> 768, bias=True)(x).view(B, S, 12, 64).transpose(1, 2)
a = softmax((q @ k.transpose(-2, -1)) * 0.125 + padding_mask)
ctx = (a @ v).transpose(1, 2).reshape(B, S, 768)
x = LayerNorm(Linear(768 -> 768, bias=True)(ctx) + x, eps=1e-12)
ff = GELU(Linear(768 -> 3072, bias=True)(x))
x = LayerNorm(Linear(3072 -> 768, bias=True)(ff) + x, eps=1e-12)
```

DPR retrieval head:

```text
pooled = x[:, 0, :]                          # [B, 768]
if projection_dim > 0:
    pooled = Linear(768 -> projection_dim)(pooled)
```

DPR reader head:

```text
span_logits = Linear(embedding_width -> 2)(x) # [P, S, 2]
start_logits, end_logits = split(span_logits, 1, dim=-1)
start_logits = squeeze(start_logits, -1).contiguous().view(P, S)
end_logits = squeeze(end_logits, -1).contiguous().view(P, S)
relevance_logits = Linear(embedding_width -> 1)(x[:, 0, :]).view(P)
```

## 6. Attention requirements

DPR retrieval uses encoder-style noncausal self-attention:

| Field | Requirement |
|---|---|
| Causal? | No for DPR defaults |
| Self/cross | Self-attention only |
| MHA/MQA/GQA | MHA, no grouped KV |
| Heads | 12 sampled, source requires `hidden_size % num_attention_heads == 0` |
| Head dim | 64 sampled, inferred |
| Q/K/V width | all 768 sampled |
| Query/key lengths | rectangular only through padding mask; self-attn `S x S` |
| Mask | 2D `[B, S]` padding mask converted to bidirectional backend mask |
| Packed/varlen | Not implemented by DPR source |
| Local/sliding | None |
| RoPE/ALiBi | None |
| KV cache | Not applicable for DPR defaults |
| SDPA/Flash | `DPRPreTrainedModel._supports_sdpa=True`; BERT dispatches through Transformers attention interface using `_attn_implementation` |

For DinoML, the first target should admit dense bidirectional attention with a
padding mask. BERT decoder/cache branches should be gated off for DPR configs.

## 7. Position encoding and custom math

DPR inherits BERT learned absolute position embeddings:

```python
def dpr_position_ids(max_position_embeddings, seq_length, past_key_values_length=0):
    table = arange(max_position_embeddings).expand(1, -1)
    return table[:, past_key_values_length : seq_length + past_key_values_length]
```

For DPR retrieval defaults `past_key_values_length=0`. Position IDs can be
precomputed per sequence bucket. Dynamic `S` only slices the prefix of the fixed
table. There is no rotary or relative-bias math to reproduce.

Attention math order for eager parity:

```python
scores = matmul(q, transpose(k, -2, -1)) * (head_dim ** -0.5)
scores = scores + attention_mask  # when mask exists
probs = softmax(scores, dim=-1)
out = matmul(probs, v)
```

## 8. Preprocessing and input packing

Question/context encoders use BERT tokenizer packing:

- Single sequence: `[CLS] tokens [SEP]`.
- Sequence pair, used for title+text contexts when caller chooses pair packing:
  `[CLS] first [SEP] second [SEP]`.
- Token type IDs are `0` for the first segment including first `[SEP]`, `1` for
  the second segment when present. DPR model wrappers default missing token
  types to zeros.
- Right padding is advised because positions are absolute.
- For question/context wrappers, missing `attention_mask` is generated as
  `input_ids != pad_token_id`.

Reader tokenizer packing differs:

```text
[CLS] question token ids [SEP] title ids [SEP] text ids
```

The reader tokenizer first tokenizes question+title with BERT pair special
tokens, tokenizes text without special tokens, concatenates, optionally truncates
the concatenated IDs to `max_length`, computes attention mask from non-pad IDs,
then pads. Span decoding finds the second `[SEP]` and only considers tokens
after it up to the first pad.

CPU/data pipeline should own WordPiece tokenization, special-token construction,
padding, truncation, and reader span text decoding. GPU/runtime should consume
already packed `input_ids`, `attention_mask`, and optionally `token_type_ids`
for retrieval encoders.

Branch feature contracts:

- Question encoder output: `question_embeddings [Bq, E]`.
- Context encoder output: `context_embeddings [Bc, E]`.
- `E = hidden_size` when `projection_dim == 0`, else `projection_dim`.
- Embeddings are independently cacheable before similarity.
- Source does not L2-normalize; ANN systems may add external normalization, but
  that is not Transformers DPR model behavior.

Similarity orientation:

```text
scores[q, c] = dot(question_embeddings[q, :], context_embeddings[c, :])
scores shape = [Bq, Bc]
```

If DinoML provides an end-to-end retrieval helper, use `Q @ C.T` with this
orientation and reject mismatched embedding widths.

## 9. Graph rewrite / lowering opportunities

### Rewrite: DPR encoder to audited BERT encoder primitive

Source pattern:

```text
DPREncoder -> BertModel(add_pooling_layer=False) -> sequence_output[:, 0, :] -> optional encode_proj
```

Replacement:

```text
BERTEncoderOnly(config, no_pooler=True) -> FirstTokenPool(axis=1,index=0) -> optional Linear
```

Preconditions:

- `model_type == "dpr"`.
- `is_decoder == False`.
- `add_cross_attention == False`.
- `hidden_size % num_attention_heads == 0`.
- BERT-family fields are supported by the separately audited BERT encoder path.

Failure cases:

- Config enables decoder/cache/cross-attention behavior.
- Unsupported `hidden_act` callable or non-GELU activation.
- `projection_dim > 0` without loaded `encode_proj` weights.

Parity test sketch: compare hidden-state and pooled outputs against
Transformers for random token IDs and masks over several sequence lengths.

### Rewrite: separate Q/K/V linears to packed QKV GEMM

Source pattern:

```text
query(x), key(x), value(x) with three Linear(768 -> 768)
```

Replacement:

```text
Linear(768 -> 2304) -> split [Q, K, V] in that order
```

Weight transform:

```python
w_packed = cat([q.weight, k.weight, v.weight], dim=0)
b_packed = cat([q.bias, k.bias, v.bias], dim=0)
```

Preconditions:

- Same input tensor for Q/K/V.
- All projections have bias and equal output width.
- Split order remains Q, K, V row blocks.
- Consumer expects `[B, S, heads, head_dim]` after split.

Failure cases: cross-attention, asymmetric widths, missing bias, or future
source variants with packed per-head storage.

### Rewrite: CLS first-token pool to narrow gather

Source pattern:

```text
sequence_output[:, 0, :]
```

Replacement:

```text
FirstRowCopy(sequence_axis=1)
```

Preconditions:

- CLS token is at position 0 from tokenizer contract.
- Output only needs pooled embedding, not full sequence.

Optimization note: full encoder still computes all sequence positions. Do not
skip token positions unless an attention-aware pruning design exists.

### Rewrite: retrieval similarity as GEMM RCR

Source pattern:

```text
scores = question_embeddings @ context_embeddings.T
```

Replacement:

```text
gemm_rcr(Q [Bq,E], C [Bc,E]) -> scores [Bq,Bc]
```

Preconditions:

- Both branches use same embedding width `E`.
- No normalization or temperature/logit scaling is requested.
- Context matrix is row-major `[Bc, E]`; use RCR interpretation for transpose.

Failure cases: external normalized embeddings, ANN index lookup instead of dense
score matrix, or mixed projection dimensions.

### Rewrite: reader span head split

Source pattern:

```text
Linear(E -> 2) -> split(1, dim=-1) -> squeeze(-1)
```

Replacement:

```text
Linear(E -> 2) -> view/copy two columns into start/end [P,S]
```

Preconditions:

- Last dimension exactly 2.
- Dense row-major logits.

Failure cases: training loss paths or custom heads not in current source.

## 10. Kernel fusion candidates

Highest priority:

- BERT encoder block parity: LayerNorm, QKV GEMM, bidirectional attention with
  padding mask, output projection, residual LayerNorm, GELU FFN.
- QKV packed projection fusion because every layer has three equal linears from
  the same hidden state.
- Attention backend with `[B, H, S, D]` and 2D padding masks for common
  sequence lengths up to 512.
- Retrieval similarity GEMM `Q @ C.T`, important for batched reranking or
  offline dense matrix scoring.

Medium priority:

- Embedding sum + LayerNorm fusion for token/position/type embeddings.
- Bias+GELU FFN fusion for `Linear(768 -> 3072)`.
- First-token pool + optional projection fusion for `projection_dim > 0`.
- Reader token-wise span head and relevance head fusion if reader becomes a
  first-class target.

Lower priority:

- Dropout removal/canonicalization for inference artifacts.
- Reader CPU postprocessing acceleration; it is usually not the GPU bottleneck.
- Tokenizer integration inside DinoML runtime; keep in data pipeline initially.

## 11. Runtime staging plan

Stage 1: Parse DPR configs and instantiate encoder-only metadata. Gate on
`is_decoder=False`, `add_cross_attention=False`, supported activation, and
`hidden_size % num_attention_heads == 0`.

Stage 2: Load one question encoder and one context encoder independently.
Validate embedding lookup, BERT block, CLS pooling, and optional projection
against Transformers.

Stage 3: Add dense retrieval score helper `Q @ C.T` outside the encoder module,
with explicit orientation tests and embedding-width guards.

Stage 4: Optimize BERT encoder lowering using existing GEMM, LayerNorm, GELU,
and attention kernels. Add QKV packing as a guarded graph rewrite.

Stage 5: Add `DPRReader` only after retrieval encoders are stable. Its neural
ops are simple, but its end-to-end parity depends on reader-specific packing and
span decoding.

Stage 6: Add performance-focused batching: cached context embeddings,
batched question encoding, and batched similarity GEMM. ANN/vector-index
integration should remain outside this DPR model audit.

Initial stubs allowed: tokenizer and span decoding can stay in Python/data
pipeline; similarity may be a separate runtime op instead of embedded in model.

## 12. Parity and validation plan

- Config sweep validation: load sampled configs and assert effective dimensions,
  architecture class, `projection_dim`, and tokenizer lower-case setting.
- Embedding stage parity: random `input_ids`, `token_type_ids`, and masks for
  `S in {1, 8, 128, 512}`; compare embedding output before encoder if hooks are
  available.
- Single-layer BERT parity: one block with copied weights, fp32 tolerance around
  `1e-4` absolute/relative.
- Full encoder parity: compare `last_hidden_state` and DPR `pooler_output` for
  question/context checkpoints.
- Mask parity: all-ones mask, right-padded mask, and omitted-mask behavior for
  question/context wrappers.
- Projection parity: synthetic config with `projection_dim > 0` because sampled
  official configs do not cover it.
- Similarity orientation test: small known Q/C matrices verifying `scores[q,c]`.
- Reader neural parity: compare start/end/relevance logits for packed
  `[P,S]` inputs.
- Reader postprocessing parity: fixed logits with known best spans, second
  `[SEP]` offset, max answer length, and overlap suppression.
- Reduced precision: after fp32 is stable, validate fp16/bf16 encoder kernels
  with tolerances chosen per BERT attention backend; keep tokenizer/postprocess
  exact.

## 13. Performance probes

- Tokenization throughput for question-only, context title/text, and reader
  question/title/text packing.
- Encoder throughput by batch size and sequence length: `B in {1, 8, 32, 128}`,
  `S in {32, 128, 256, 512}`.
- Attention backend comparison for bidirectional padded attention at DPR lengths.
- Context offline embedding throughput and storage bandwidth for cached
  `[num_contexts, E]` embeddings.
- Dense retrieval GEMM sweep: `Bq x Bc x E`, including `E=768` and synthetic
  projection widths.
- End-to-end retrieval latency split: tokenizer, question encoder, context
  encoder or cache fetch, similarity.
- Reader throughput split: encoder body, span/relevance heads, CPU span decode.

No benchmark measurements are included here; these are source-derived probe
recommendations.

## 14. Skip/defer list

- Training, losses, gradient checkpointing, dropout behavior in training mode.
- BERT decoder mode, causal masks, cross-attention, and KV cache.
- ANN index/search integration; expose embeddings and dense score orientation
  first.
- Quantization or packed weight formats beyond DinoML's generic constant
  machinery.
- Tokenizer implementation inside GPU/runtime.
- Reader span decoding inside compiled graph.
- Remote-code or non-official DPR variants until configs are allowlisted.

## 15. Final implementation checklist

- [ ] Parse `DPRConfig` and reject decoder/cross-attention DPR configs for first target.
- [ ] Compose or import audited BERT encoder-only lowering.
- [ ] Load question/context encoder weights as independent branches.
- [ ] Implement embedding lookup plus position/token-type embedding sum.
- [ ] Implement bidirectional MHA with padding mask and BERT LayerNorm/GELU blocks.
- [ ] Implement CLS first-token pooling.
- [ ] Implement optional `encode_proj` for `projection_dim > 0`.
- [ ] Add retrieval similarity helper `Q @ C.T` with orientation guard.
- [ ] Add tokenizer/data-pipeline ABI docs for `[CLS]`/`[SEP]` packing and masks.
- [ ] Add encoder parity tests for sampled official checkpoints.
- [ ] Add synthetic projection-dim parity test.
- [ ] Add similarity orientation parity test.
- [ ] Gate or defer `DPRReader` until retrieval encoders are stable.
- [ ] Add reader neural-head parity tests if `DPRReader` is admitted.
- [ ] Benchmark encoder throughput and dense similarity GEMM separately.

## Gated gaps for DinoML

- BERT encoder coverage is the real prerequisite; DPR-specific retrieval logic
  is small once BERT encoder-only parity exists.
- Attention admission must explicitly reject DPR configs that activate decoder,
  cross-attention, cache, or unsupported `_attn_implementation` behavior.
- `projection_dim > 0` is source-supported but not covered by sampled official
  configs; require synthetic or alternate-checkpoint validation before marking
  complete.
- Similarity is not owned by the Transformers model classes; any end-to-end DPR
  helper must define `Q @ C.T` orientation and normalization policy explicitly.
- Reader parity needs tokenizer-owned span postprocessing and should be staged
  separately from retrieval embedding parity.
