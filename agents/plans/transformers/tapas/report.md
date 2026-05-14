# TAPAS Full Audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model family: tapas
Primary source checkout: X:/H/transformers
Report path: H:/dinoml_v2/agents/plans/transformers/tapas/report.md
Config snapshot: H:/dinoml_v2/agents/plans/transformers/tapas/hf_config_snapshot.json
```

Model ids inspected:

| Model id | Config source | Notes |
|---|---|---|
| `google/tapas-small` | [config.json](https://huggingface.co/google/tapas-small/resolve/main/config.json) | base encoder, small dimensions |
| `google/tapas-base` | [config.json](https://huggingface.co/google/tapas-base/resolve/main/config.json) | base encoder |
| `google/tapas-large` | [config.json](https://huggingface.co/google/tapas-large/resolve/main/config.json) | base encoder, large dimensions |
| `google/tapas-base-finetuned-sqa` | [config.json](https://huggingface.co/google/tapas-base-finetuned-sqa/resolve/main/config.json) | QA, no aggregation head |
| `google/tapas-base-finetuned-wtq` | [config.json](https://huggingface.co/google/tapas-base-finetuned-wtq/resolve/main/config.json) | QA, weak aggregation |
| `google/tapas-base-finetuned-wikisql-supervised` | [config.json](https://huggingface.co/google/tapas-base-finetuned-wikisql-supervised/resolve/main/config.json) | QA, supervised aggregation |
| `google/tapas-base-finetuned-tabfact` | [config.json](https://huggingface.co/google/tapas-base-finetuned-tabfact/resolve/main/config.json) | sequence classification |
| `google/tapas-small-finetuned-sqa@no_reset` | [config.json](https://huggingface.co/google/tapas-small-finetuned-sqa/resolve/no_reset/config.json) | absolute position embedding revision |

Source files inspected:

- `src/transformers/models/tapas/configuration_tapas.py`
- `src/transformers/models/tapas/modeling_tapas.py`
- `src/transformers/models/tapas/tokenization_tapas.py`
- `src/transformers/models/tapas/convert_tapas_original_tf_checkpoint_to_pytorch.py`
- `docs/source/en/model_doc/tapas.md`
- `tests/models/tapas/test_modeling_tapas.py`
- `tests/models/tapas/test_tokenization_tapas.py`

Any missing files or assumptions:

- No representative config returned 401/403. Checkpoint weight tensors were not downloaded; weight shapes below are source-derived from module definitions.
- The source supports decoder/cross-attention cache plumbing inherited from BERT-like modules, but all representative TAPAS configs inspected are encoder-style table models.
- Historical Hub configs include both `type_vocab_size` and `type_vocab_sizes`; the current source reads `type_vocab_sizes`.

## 2. High-level architecture

TAPAS is a BERT-style text/table encoder with table-aware embeddings and task heads. The primary production family is an encoder-only model for table question answering, masked LM, and table entailment/classification.

```text
Pandas/text table preprocessing -> flattened WordPiece sequence + 7-channel token_type_ids
  -> word + position + 7 token-type embeddings
  -> BERT-like noncausal encoder
  -> task head:
       masked LM head
       table QA cell-selection head + optional aggregation head
       sequence classification head
  -> tokenizer-side table-coordinate postprocess
```

Stage decomposition:

- CPU/data pipeline: Pandas table validation, string conversion, WordPiece tokenization, row/column flattening, answer label mapping, numeric parsing, numeric rank/relation features, truncation by dropping rows or trimming per-cell token budget.
- GPU/runtime graph: embedding lookup/add/layernorm/dropout, dense MHA encoder blocks, segmented reductions for relative position IDs and QA heads, classifier heads.
- Independently cacheable pieces: tokenized table structure and numeric annotations can be cached per table; encoder outputs can be cached per exact table/query input but TAPAS has no autoregressive decode loop.
- Postprocess: sigmoid token probabilities -> mean probability per cell -> cell coordinate thresholding; optional aggregation argmax to labels such as `NONE`, `SUM`, `AVERAGE`, `COUNT`.

## 3. Important config dimensions

| Field | Source default | Operator significance |
|---|---:|---|
| `vocab_size` | 30522 | word embedding and MLM decoder |
| `hidden_size` | 768 | encoder width and head projection width |
| `num_hidden_layers` | 12 | encoder block count |
| `num_attention_heads` | 12 | MHA head count |
| `head_dim` | `hidden_size / num_attention_heads` | 64 for inspected small/base/large configs |
| `intermediate_size` | 3072 | FFN up-projection width |
| `hidden_act` | `gelu` | FFN activation |
| `max_position_embeddings` | 1024 | absolute/relative position embedding table length |
| `type_vocab_sizes` | `[3,256,256,2,256,256,10]` | seven independent token-type embedding tables |
| `max_num_rows` / `max_num_columns` | 64 / 32 | segmented QA cell grid shape, max cells 2048 |
| `reset_position_index_per_cell` | true | relative position IDs per table cell |
| `select_one_column` | true | hierarchical column+cell selection path |
| `num_aggregation_labels` | 0 | optional aggregation classifier output width |
| `aggregation_labels` | optional | usually `{0:NONE,1:SUM,2:AVERAGE,3:COUNT}` for aggregation checkpoints |
| `temperature` | 1.0 default, checkpoint-specific | divides cell logits |
| `aggregation_temperature` | 1.0 | divides aggregation logits for expected-result loss |
| `average_logits_per_cell` | false | optional per-cell logit averaging before Bernoulli |
| `use_answer_as_supervision` | optional | enables weak aggregation regression path |
| cache support | not used by representative configs | decoder cache path exists only if configured as decoder |

Representative checkpoint sweep:

| Model | Arch | Layers | Hidden | Heads | FFN | Max pos | Dropout | Aggregation | Position reset |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| `google/tapas-small` | `TapasModel` | 4 | 512 | 8 | 2048 | 512 | 0.07 | none | true |
| `google/tapas-base` | `TapasModel` | 12 | 768 | 12 | 3072 | 1024 | 0.07 | none | true |
| `google/tapas-large` | `TapasModel` | 24 | 1024 | 16 | 4096 | 1024 | 0.07 | none | true |
| `google/tapas-base-finetuned-sqa` | QA | 12 | 768 | 12 | 3072 | 1024 | 0.1 | none | true |
| `google/tapas-base-finetuned-wtq` | QA | 12 | 768 | 12 | 3072 | 1024 | 0.1 | 4 labels, weak answer supervision | true |
| `google/tapas-base-finetuned-wikisql-supervised` | QA | 12 | 768 | 12 | 3072 | 1024 | 0.1 | 4 labels, supervised | true |
| `google/tapas-base-finetuned-tabfact` | sequence classification | 12 | 768 | 12 | 3072 | 1024 | 0.07 | none | true |
| `google/tapas-small-finetuned-sqa@no_reset` | QA | 4 | 512 | 8 | 2048 | 512 | 0.1 | none | false |

## 3a. Family variation traps

- Token type ABI is rank-3: `token_type_ids[B, S, 7]`, not BERT's `token_type_ids[B, S]`.
- The seven channels have fixed order: `segment_ids`, `column_ids`, `row_ids`, `prev_labels`, `column_ranks`, `inv_column_ranks`, `numeric_relations`.
- `type_vocab_sizes` has one embedding table per channel; do not collapse into one segment embedding.
- Relative position mode depends on `column_ids` and `row_ids`; absolute-position `no_reset` revisions must bypass segmented position recomputation.
- Header tokens have `row_id == 0`; table body tokens have `segment_id == 1`, `column_id >= 1`, `row_id >= 1`.
- Conversational SQA mutates `prev_labels` from previous predictions. This is a processor/session concern, not an encoder weight change.
- WTQ/WikiSQL aggregation heads add `aggregation_classifier: Linear(hidden_size -> num_aggregation_labels)`.
- QA cell-selection heads are vector parameters, not `nn.Linear` modules: `output_weights[H]`, `column_output_weights[H]`, and scalar biases.
- `select_one_column=True` changes inference logits by masking cells outside the model-selected column.
- Hub configs may contain historical `softmax_temperature` and `type_vocab_size` fields that this source does not read for forward math.
- `attention_probs_dropout_prob` and `hidden_dropout_prob` differ between base pretraining configs and finetuned configs.
- `allow_empty_column_selection` differs for WTQ and WikiSQL-supervised checkpoints.
- The tokenizer expects table cells to be strings; `.astype(str)` is required for Pandas numeric columns.

## 4. Operator coverage checklist

Tensor/layout ops:

- `reshape`, `view`, `transpose`, `permute`, `contiguous`.
- `arange`, `expand`, `unsqueeze`, `cat`, `repeat`.
- `where`, `logical_and`, `logical_not`, equality/comparison ops.
- `min` clamp for row/column IDs and position IDs.
- `gather` along segment dimension.
- `scatter_reduce` with `sum`, `mean`, `amax`, `amin` for segmented reductions.
- `argmax`, `max`, `sum`, `mean`, division with epsilon.

Neural network primitives:

- Embedding lookup: word `[30522, H]`, position `[max_position_embeddings, H]`, token type tables `[3,H]`, `[256,H]`, `[256,H]`, `[2,H]`, `[256,H]`, `[256,H]`, `[10,H]`.
- LayerNorm epsilon `1e-12`.
- Dropout for training; inference can elide.
- Linear with bias for Q/K/V/O and FFN.
- GELU FFN activation.
- Pooler: take hidden state at token 0 -> `Linear(H -> H)` -> tanh.
- MLM head: `Linear(H -> H)` + activation + LayerNorm + tied/untied decoder to vocab.

Attention primitives:

- Dense noncausal self-attention for primary configs.
- Matmul QK, scale by `1/sqrt(head_dim)`, add extended attention mask, softmax, dropout, matmul PV.
- Optional decoder/cross-attention/cache path is source-present but not required for representative TAPAS checkpoints.

Preprocessing-coupled ops:

- Pandas table flattening with header row first.
- WordPiece tokenization with `[CLS] question [SEP] table_tokens`.
- Row-drop truncation and optional per-cell token budget; no overflowing-token return.
- Numeric parsing for dates/numbers/number words/ordinals.
- Numeric rank and inverse-rank per column.
- Numeric relation bitset for EQ/LT/GT question-to-cell relations.
- Answer coordinate mapping from SQA `(row, column)` to internal `(column, row)`.

Aggregation/postprocess ops:

- Token-level cell logits: `einsum("bsh,h->bs") + scalar_bias`, divided by `temperature`.
- Column logits: per-token projection, reduce mean per cell, reduce sum per column, divide by cell count, mask missing/empty columns with `-10000`.
- Single-column constrained logits: reduce token logits to cells, choose selected column by `argmax(column_logits)`, mask other cells with `-10000`, gather back to tokens.
- Aggregation logits: pooled output -> `Linear(H -> num_aggregation_labels)`.
- Inference postprocess: sigmoid token logits with lower clamp, multiply attention mask, average per cell, threshold `> 0.5`, return sorted `(row, column)` coordinates, optional aggregation `argmax`.

Quantized/packed weight metadata ops:

- None in the inspected TAPAS source/configs.

## 5. Layer/block breakdown

Embeddings:

```text
input_ids[B,S], token_type_ids[B,S,7]
word = Embedding(vocab_size,H)(input_ids)
position_ids = absolute arange or relative-per-cell positions
position = Embedding(max_position_embeddings,H)(position_ids)
type_i = Embedding(type_vocab_sizes[i],H)(token_type_ids[:,:,i]) for i in 0..6
x = LayerNorm(word + position + sum(type_i))
x = Dropout(x)
```

Relative position path when `reset_position_index_per_cell=True`:

```text
col_index = token_type_ids[:,:,1]
row_index = token_type_ids[:,:,2]
cell_index = row_index * type_vocab_sizes[1] + col_index
first_position_per_cell = segmented_min(arange(S), cell_index)
position_ids = min(max_position_embeddings - 1, arange(S) - gather(first_position_per_cell, cell_index))
```

Encoder block, repeated `N` times:

```text
q = Linear(H -> H, bias=True)(x).view(B,S,heads,64).transpose(1,2)
k = Linear(H -> H, bias=True)(x).view(B,S,heads,64).transpose(1,2)
v = Linear(H -> H, bias=True)(x).view(B,S,heads,64).transpose(1,2)
scores = (q @ k.transpose(-1,-2)) / sqrt(64)
scores += extended_attention_mask
p = softmax(scores, dim=-1)
ctx = (dropout(p) @ v).transpose(1,2).reshape(B,S,H)
x = LayerNorm(Linear(H -> H)(ctx) + residual)
ff = Linear(H -> intermediate_size)(x)
ff = GELU(ff)
x = LayerNorm(Linear(intermediate_size -> H)(ff) + residual)
```

QA head:

```text
sequence_output[B,S,H], pooled_output[B,H]
token_logits[B,S] = dot(sequence_output, output_weights[H]) + output_bias
token_logits /= temperature
column_logits[B,max_num_columns] = segmented cell/column projection path
if select_one_column:
    token_logits = single_column_masked_logits(token_logits, column_logits, row/column cell_index)
if num_aggregation_labels > 0:
    logits_aggregation[B,A] = Linear(H -> A)(pooled_output)
```

Sequence classification:

```text
pooled = tanh(Linear(H -> H)(last_hidden_state[:,0]))
logits = Linear(H -> num_labels)(Dropout(pooled))
```

## 6. Attention requirements

Primary TAPAS checkpoints require dense encoder self-attention:

| Requirement | TAPAS behavior |
|---|---|
| Causal | No |
| Self/cross | Self-attention only for inspected configs |
| MHA/MQA/GQA | MHA |
| Heads/head dim | small 8x64, base 12x64, large 16x64 |
| Q/K/V width | all `hidden_size` |
| Masking | extended attention mask from `attention_mask`; additive mask before softmax |
| Packed/varlen | no source varlen attention path |
| Sliding/local/block | no |
| ALiBi/RoPE | no |
| KV cache | not required for primary configs |
| Flash/SDPA compatibility | mathematically compatible with ordinary noncausal dense attention in inference, as long as additive mask and dropout-disabled behavior match |

The source can instantiate decoder/cross-attention if config flags are changed, but representative TAPAS configs do not use this path. First integration should reject `is_decoder=True` or `add_cross_attention=True` unless explicitly staged.

## 7. Position encoding and custom math

TAPAS has learned position embeddings. The custom piece is relative position ID construction per table cell:

```python
def tapas_relative_position_ids(token_type_ids, max_position_embeddings, type_vocab_sizes):
    # token_type_ids: [B, S, 7]
    col = token_type_ids[:, :, 1]
    row = token_type_ids[:, :, 2]
    cell = row * type_vocab_sizes[1] + col
    pos = arange(S).expand(B, S)
    first = segmented_min(pos, cell, num_segments=type_vocab_sizes[1] * type_vocab_sizes[2])
    first_for_token = gather(first, cell)
    return minimum(max_position_embeddings - 1, pos - first_for_token)
```

Precomputable:

- Absolute position IDs for `reset_position_index_per_cell=False`.
- Static `arange(S)`.

Dynamic/input-dependent:

- Relative cell position IDs depend on `token_type_ids[:,:,1:3]`, truncation, padding side, and table flattening.

## 8. Preprocessing and input packing

CPU/data-pipeline contract:

- Input table is a Pandas `DataFrame`; source docs require text-only cell values.
- Tokenization is WordPiece with TAPAS-specific empty cell token `[EMPTY]`.
- Sequence format is `[CLS] query_tokens [SEP] flattened_table_tokens`; no trailing `[SEP]` is appended after the table in this implementation.
- Header cells are flattened before data rows. Header row has `row_id=0`; body rows start at 1; columns start at 1.
- `max_row_id`/`max_column_id` default from `model_max_length` unless explicitly set; model config max rows/columns for QA head are separate defaults 64/32.
- Truncation strategy is `drop_rows_to_fit`; tokenizer does not return overflowing chunks.
- Padding appends/prepends zeros for `input_ids`, `attention_mask`, `token_type_ids`, labels, and numeric arrays; numeric padding uses NaN for `numeric_values` and 1.0 for scales.

Token type ABI:

| Channel | Name | Values |
|---:|---|---|
| 0 | `segment_ids` | 0 for question/special/pad, 1 for table |
| 1 | `column_ids` | 0 outside table, table columns start at 1 |
| 2 | `row_ids` | 0 outside body cells and for headers, body rows start at 1 |
| 3 | `prev_labels` | SQA previous answer labels, 0/1 |
| 4 | `column_ranks` | numeric rank within column, 0 if not applicable |
| 5 | `inv_column_ranks` | inverse numeric rank within column, 0 if not applicable |
| 6 | `numeric_relations` | bitset over EQ/LT/GT relations to numeric question spans |

Training-only or weak-supervision inputs:

- `labels[B,S]`: token answer labels from answer coordinates/text.
- `numeric_values[B,S]`: float numeric value per token, NaN for nonnumeric.
- `numeric_values_scale[B,S]`: token-count scale for multi-token numeric cells.
- `float_answer[B]`: scalar answer for weak aggregation supervision.
- `aggregation_labels[B]`: required for strong aggregation supervision.

Postprocess requirements:

- Convert logits to NumPy on CPU in source tokenizer.
- Clamp token logits below `-88.7` before sigmoid to avoid float32 exponential overflow.
- Mean token probabilities per `(column,row)` cell.
- Select cells with mean probability strictly greater than threshold.
- Return coordinates sorted as `(row, column)`.
- Aggregation indices are `argmax(logits_aggregation, axis=-1)`; mapping to labels comes from config or caller.

## 9. Graph rewrite / lowering opportunities

### Rewrite: TAPAS token-type embedding sum

Source pattern:

```text
word_embedding + position_embedding + sum(Embedding_i(token_type_ids[:,:,i]) for i in 0..6)
```

Replacement:

```text
Gather8/16/32 token embeddings -> fused elementwise sum -> LayerNorm
```

Preconditions:

- `token_type_ids` rank is `[B,S,7]`.
- Each channel is in range for its own `type_vocab_sizes[i]`.
- Do not pack the seven embedding tables unless the compiler preserves per-channel offsets and range checks.

Shape equations:

- every embedding output is `[B,S,H]`; sum output `[B,S,H]`.

Failure cases:

- Historical config missing `type_vocab_sizes`.
- Any custom config changes channel count away from 7.

Parity test sketch:

- Compare embedding output for random `token_type_ids` plus real tokenizer outputs, both reset and no-reset position modes.

### Rewrite: relative position segmented-min

Source pattern:

```text
ProductIndexMap(column,row) -> reduce_min(position_ids) -> gather -> subtract
```

Replacement:

```text
SegmentFirstPosition(row_ids,column_ids) -> token-relative-position IDs
```

Preconditions:

- `reset_position_index_per_cell=True`.
- Segment key equals `row * type_vocab_sizes[1] + column`, matching source.
- Headers/question/pad use row/column 0 and therefore share the special segment.

Failure cases:

- Left padding or caller-supplied custom `position_ids` can change expected IDs.
- Custom `type_vocab_sizes[1]`/`[2]` must participate in key construction.

Parity test sketch:

- Use tokenizer integration examples and assert `position_ids` against source for right/left padding and `no_reset`.

### Rewrite: QA vector projection to GEMV/linear

Source pattern:

```text
einsum("bsh,h->bs", sequence_output, weight) + scalar_bias
```

Replacement:

```text
Linear(H -> 1) -> squeeze last dim
```

Preconditions:

- Weight is logically `[H]`, bias scalar.
- Preserve `/ temperature` for cell logits and no temperature divide for column logits.

Failure cases:

- Treating these parameters as ordinary named `nn.Linear` weights may break checkpoint loading aliases.

Parity test sketch:

- Compare token logits and column projection logits against source with random sequence output.

### Rewrite: segmented reductions for QA head

Source pattern:

```text
reduce_mean token logits per cell
reduce_sum/mask/divide per column
gather per-cell logits back to tokens
```

Replacement:

```text
custom segmented_cell_column_head(row_ids,column_ids,sequence_output,mask)
```

Preconditions:

- `max_num_rows` and `max_num_columns` are known from config.
- Segment flattening keeps batch elements distinct.
- Empty segment behavior matches PyTorch `scatter_reduce(..., include_self=False)` plus explicit masks.

Failure cases:

- Dynamic rows/columns beyond config clamps.
- Different `allow_empty_column_selection` changes column 0 masking.

Parity test sketch:

- Random cell grids with empty columns, padding, headers, and selected-column constraints.

## 10. Kernel fusion candidates

Highest priority:

- Embedding multi-sum + LayerNorm. TAPAS pays eight embedding gathers per token before the encoder.
- Dense encoder attention/FFN fusions. Base and large checkpoints are ordinary BERT-like compute.
- Segmented reductions/gather for relative positions and QA heads. These are the nonstandard runtime blockers.

Medium priority:

- QA cell/column head fused segmented kernel: vector projections, per-cell mean, per-column mean, masking, and gather.
- Postprocess cell probability averaging on CPU or GPU for batch inference.
- Pooler + aggregation/classification linear heads.

Lower priority:

- MLM head optimization, unless masked-LM TAPAS is a target.
- Training-only weak supervision regression loss kernels.
- Decoder/cross-attention/cache support, because representative TAPAS configs do not use it.

## 11. Runtime staging plan

Stage 1: config and tokenizer ABI admission.

- Parse `TapasConfig`, require seven `type_vocab_sizes`, reject decoder/cross-attention initially.
- Accept precomputed `input_ids`, `attention_mask`, `token_type_ids`; treat Pandas/tokenizer as CPU-side preprocessor.

Stage 2: embedding parity.

- Implement word/position/seven-token-type embedding sum.
- Implement both absolute and relative-per-cell position IDs.

Stage 3: encoder parity.

- Lower BERT-like encoder blocks for small/base shapes.
- Validate `TapasModel` last hidden state and pooler output.

Stage 4: QA inference head.

- Implement token and column projections, cell/column segmented reductions, selected-column masking, optional aggregation classifier.
- Return raw token logits and aggregation logits.

Stage 5: tokenizer-compatible postprocess.

- Implement or call CPU postprocess: sigmoid, mean per cell, threshold, sorted `(row,column)`, aggregation argmax.
- Keep conversational `prev_labels` update in the host/session layer.

Stage 6: classification and MLM heads.

- Add TabFact sequence classification.
- Add MLM only if fill-mask parity is required.

Stage 7: optimization.

- Fuse embedding sum, encoder kernels, and segmented QA head after parity is stable.

Can be stubbed initially:

- Training losses, weak aggregation regression loss, Gumbel sampling, gradient checkpointing, decoder/cache path, MLM head.

## 12. Parity and validation plan

- Tokenizer snapshot parity: reproduce `tests/models/tapas/test_tokenization_tapas.py` expected `input_ids`, `segment_ids`, `column_ids`, and `row_ids` for `google/tapas-base-finetuned-wtq`.
- Embedding parity: compare source `TapasEmbeddings` for reset and `no_reset` configs with fixed tokenized inputs.
- Segmented op parity: unit-test `ProductIndexMap`, `gather`, `reduce_sum`, `reduce_mean`, `reduce_min`, `reduce_max` on random grids including empty segments.
- Single-block parity: run one encoder layer for small config fp32, tolerance `rtol=5e-4`, `atol=5e-4`.
- Encoder parity: compare `TapasModel` hidden-state and pooler slices from integration tests.
- QA SQA parity: `google/tapas-base-finetuned-sqa`, expect logits shape `[B,S]`, no aggregation logits.
- QA WTQ parity: `google/tapas-base-finetuned-wtq`, expect logits `[2,28]` and aggregation logits `[2,4]` for integration example.
- QA WikiSQL supervised parity: validate aggregation logits `[B,4]`.
- TabFact parity: `google/tapas-base-finetuned-tabfact`, logits `[B,2]`.
- Postprocess parity: compare `convert_logits_to_predictions` answer coordinates and aggregation indices.
- Recommended fp32 tolerances: encoder/head slices `rtol=5e-4` to `1e-3`, aggregation logits `1e-3` for fp32. fp16/bf16 should start with looser end-to-end logits tolerances and exact coordinate parity after threshold stress cases.

## 13. Performance probes

- CPU preprocessing throughput: tables/sec by rows, columns, average cell token length.
- Numeric annotation throughput: parse/rank/relation cost split from WordPiece cost.
- Encoder-only throughput: small/base/large, batch and sequence-length sweeps.
- Relative position construction cost: segmented-min/gather versus precomputed IDs.
- QA head cost: segmented reductions over `B x S` into `B x 2048` cell grid and `B x 32` columns.
- End-to-end table QA latency: preprocessing + encoder + QA head + postprocess.
- Batch shape sweep: `B={1,2,8,16}`, `S={64,128,256,512,1024}` where configs allow.
- Table shape sweep: rows/columns near `64 x 32`, sparse/empty columns, long cells requiring trimming.
- Postprocess CPU cost: sigmoid/mean/threshold over batch and max cell count.
- Attention backend comparison: eager matmul/softmax versus fused dense attention for noncausal encoder.

## 14. Skip/defer list

- Training losses and gradient support.
- Weak-supervision expected-result regression loss and Gumbel sampling.
- Gradient checkpointing.
- Decoder/cross-attention/cache path.
- Masked LM head unless fill-mask is a product target.
- Custom production OCR/GoldMine/Aqua numeric parsing replacements; source tokenizer notes academic numeric utilities.
- Multi-GPU/tensor parallel.
- Quantization/packed weights.
- Automatic Pandas ingestion inside compiled runtime; keep it as host preprocessing first.

## 15. Final implementation checklist

- [ ] Parse `TapasConfig` and normalize historical `type_vocab_size`/`type_vocab_sizes` handling.
- [ ] Load TAPAS weights, including seven token-type embedding tables and vector QA head parameters.
- [ ] Implement TAPAS input ABI: `input_ids[B,S]`, `attention_mask[B,S]`, `token_type_ids[B,S,7]`.
- [ ] Implement absolute and relative-per-cell position ID paths.
- [ ] Implement segmented `gather` and reductions with TAPAS empty-segment behavior.
- [ ] Implement embedding sum + LayerNorm parity.
- [ ] Implement BERT-like encoder block parity for small/base/large.
- [ ] Implement pooler.
- [ ] Implement QA token cell-selection projection.
- [ ] Implement QA column projection and selected-column masking.
- [ ] Implement optional aggregation classifier.
- [ ] Implement CPU-compatible postprocess for answer coordinates and aggregation ids.
- [ ] Add tokenizer fixture parity for table flattening/token type channels.
- [ ] Add one-layer, encoder, QA, and TabFact parity tests.
- [ ] Add performance probes for preprocessing, encoder, segmented QA head, and postprocess.
