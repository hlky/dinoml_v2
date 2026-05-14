# DinoML Transformers Audit: `luke`

## 1. Source basis

Transformers commit/version:
`b75feb2af64c3e29cbbc1bd859958c5432cc7ed4` from local checkout `X:/H/transformers`. Sampled checkpoint configs report older `transformers_version` values (`4.6.0.dev0` or `4.15.0`), so source behavior below is based on the current local in-library implementation, not historical remote code.

Model id:
Primary source/checkpoint family `studio-ousia/luke-*`. Representative configs were fetched from Hugging Face and saved under `_sources/`.

Config source:

- `https://huggingface.co/studio-ousia/luke-base/raw/main/config.json`
- `https://huggingface.co/studio-ousia/luke-large/raw/main/config.json`
- `https://huggingface.co/studio-ousia/luke-large-finetuned-open-entity/raw/main/config.json`
- `https://huggingface.co/studio-ousia/luke-large-finetuned-tacred/raw/main/config.json`
- `https://huggingface.co/studio-ousia/luke-large-finetuned-conll-2003/raw/main/config.json`

Source files inspected:

- `X:/H/transformers/src/transformers/models/luke/configuration_luke.py`
- `X:/H/transformers/src/transformers/models/luke/modeling_luke.py`
- `X:/H/transformers/src/transformers/models/luke/tokenization_luke.py`
- `X:/H/transformers/src/transformers/models/luke/convert_luke_original_pytorch_checkpoint_to_pytorch.py`

Any missing files or assumptions:
No remote-code files were required. The report target is inference for the base LUKE encoder plus entity classification, entity-pair classification, and entity-span classification heads. Masked LM/entity prediction is optional for first integration; training losses are deferred.

## 2. High-level architecture

LUKE is an encoder-only, RoBERTa-like text transformer with a second entity-token stream. Word tokens and entity records each get embeddings, then every encoder layer jointly attends over the concatenated sequence `[word tokens, entity tokens]`. When `use_entity_aware_attention=True`, the query projection is relation-specific for word-to-word, word-to-entity, entity-to-word, and entity-to-entity score blocks.

Dataflow:

```text
CPU tokenizer/entity span packing
  -> word embeddings + entity embeddings
  -> repeated entity-aware encoder blocks
  -> word hidden states + entity hidden states
  -> task head
```

Stage decomposition:

- CPU/data pipeline: byte-level BPE tokenization, character span to token span conversion, entity ID lookup, insertion of `<ent>`/`<ent2>` markers for some tasks, padding/truncation, and fixed-size `entity_position_ids`.
- GPU/runtime encoder: embedding lookups, entity mention-position averaging, dense noncausal self-attention over `W + E`, FFN, LayerNorm, and residuals.
- Independently optimizable heads: pooled `[CLS]` sequence classifier, token classifier, QA span logits, entity classifier, entity-pair classifier, entity-span classifier, and MLM/entity prediction heads.

## 3. Important config dimensions

| Field | Source default | Sampled checkpoint values | DinoML note |
| --- | ---: | --- | --- |
| `vocab_size` | 50267 | 50267 | RoBERTa BPE word vocab. |
| `entity_vocab_size` | 500000 | 500000 | Large entity embedding/output table. |
| `hidden_size` | 768 | 768 base, 1024 large | Attention/FFN width. |
| `entity_emb_size` | 256 | 256 | Projected to `hidden_size` by biasless dense when different. |
| `num_hidden_layers` | 12 | 12 base, 24 large | Encoder depth. |
| `num_attention_heads` | 12 | 12 base, 16 large | Head dim is `hidden_size / heads` = 64 in sampled configs. |
| `intermediate_size` | 3072 | 3072 base, 4096 large | BERT/RoBERTa FFN expansion. |
| `hidden_act` | `gelu` | `gelu` | ACT2FN path. |
| `max_position_embeddings` | 512 | 514 | Checkpoint fact overrides source default. |
| `type_vocab_size` | 2 | 1 | Checkpoint fact; token type IDs default to zeros. |
| `layer_norm_eps` | `1e-12` | `1e-5` | Checkpoint fact matters for parity. |
| `use_entity_aware_attention` | true | true | Main ABI/attention trap. |
| `classifier_dropout` | null | usually absent/null | Falls back to hidden dropout. |

Representative checkpoint sweep:

| Checkpoint | Primary class | Hidden | Layers | Heads | Labels | Task ABI |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `studio-ousia/luke-base` | `LukeForMaskedLM` | 768 | 12 | 12 | n/a | MLM + entity prediction. |
| `studio-ousia/luke-large` | `LukeForMaskedLM` | 1024 | 24 | 16 | n/a | MLM + entity prediction. |
| `studio-ousia/luke-large-finetuned-open-entity` | `LukeForEntityClassification` | 1024 | 24 | 16 | 9 | Single entity at `entity_last_hidden_state[:,0]`. |
| `studio-ousia/luke-large-finetuned-tacred` | `LukeForEntityPairClassification` | 1024 | 24 | 16 | 42 | Two entities concatenated; classifier has no bias in source. |
| `studio-ousia/luke-large-finetuned-conll-2003` | `LukeForEntitySpanClassification` | 1024 | 24 | 16 | 5 | Candidate spans with start/end word gathers plus entity state. |

## 3a. Family variation traps

- `max_position_embeddings`: source default 512, common checkpoints 514. Do not compile fixed 512 from the config class default when loading checkpoint configs.
- `type_vocab_size`: sampled checkpoints use 1, while source default is 2. Any tokenizer-provided nonzero token/entity type IDs would be invalid for these weights.
- `use_entity_aware_attention=False` is implemented and collapses to ordinary MHA over concatenated word/entity states. First DinoML admission can require `true` for checkpoint parity and add the false path later.
- Historical config fields `output_past`, `use_cache`, and `position_embedding_type` are not runtime requirements in current source; LUKE has no KV cache and no causal generation.
- `classifier_bias` appears in TACRED config but current source hardcodes `nn.Linear(hidden_size * 2, num_labels, False)`. Treat it as ignored for this source basis unless auditing old code.
- Entity-packing behavior is tokenizer-task-dependent. The same model body accepts variable `entity_length`, but task tokenizers force 1, 2, or up to `max_entity_length`.
- Entity position IDs have shape `[B, E, max_mention_length]`, use `-1` sentinel padding, clamp before embedding lookup, then masked average over mention positions. This is not ordinary token position embedding.
- Layout translation should be guarded off for the encoder ABI. All semantic axes are sequence-major `[B, S, H]` and `[B, E, H]`; attention uses `[B, heads, query, key]`. No NHWC opportunity exists.

## 4. Operator coverage checklist

Tensor/layout ops:

- Integer embedding lookup for word IDs, word positions, word token types, entity IDs, entity mention positions, and entity token types.
- `arange`, `ne`, `cumsum`, multiply, and add for default word position IDs from `input_ids` when not provided.
- `clamp(min=0)` for entity position IDs before position embedding.
- Broadcast/unsqueeze/expand for masks and gather indices.
- Concatenate along sequence axis for `[word, entity]`, split/slice back into word/entity portions.
- `view`, `permute`, `contiguous`, reshape for attention heads and multiple-choice batch flattening.
- `gather` on word sequence for entity span start/end states.

Neural network primitives:

- Dense Linear with bias for attention Q/K/V/output, FFN, pooler, most heads.
- Biasless Linear for `entity_embedding_dense` and entity-pair classifier.
- LayerNorm with checkpoint `eps=1e-5`.
- GELU, tanh, residual add, dropout-as-noop in inference.
- Large LM/entity output projections: word vocab `H -> 50267`, entity vocab `entity_emb_size -> 500000`.

Attention primitives:

- Noncausal dense self-attention over total length `T = W + E`.
- Entity-aware attention requires four query projections and four score GEMMs, then score block concatenation:
  `ww [B,A,W,W]`, `we [B,A,W,E]`, `ew [B,A,E,W]`, `ee [B,A,E,E]`.
- Mask addition uses extended additive mask `[B,1,1,T]` or `[B,1,Q,T]` with masked value `torch.finfo(dtype).min`.
- Softmax over key axis, value matmul, output projection and residual LayerNorm.

Position/custom math:

- Word positions are padding-aware cumsum positions offset by `pad_token_id`.
- Entity mention positions are averaged over up to `max_mention_length` word positions, ignoring `-1`.

Preprocessing-coupled ops:

- Byte-level BPE and entity vocab lookup are CPU/tokenizer work.
- Character span to token span mapping, truncation filtering, marker insertion, and task-specific entity count checks are CPU/tokenizer work.

Optional/deferred categories:

- Training losses: cross entropy, BCE, MSE.
- No RoPE, ALiBi, local attention, recurrent state, KV cache, quantized/packed weight format, image/audio/video, or distributed tensor parallelism in source.

Aliasing/tied weights:

- `LukeForMaskedLM` ties `entity_predictions.decoder.weight` to `luke.entity_embeddings.entity_embeddings.weight`.
- `lm_head.decoder` follows standard `PreTrainedModel` word embedding tying behavior for `tie_word_embeddings=True`; preserve logical aliasing for MLM parity.

## 5. Layer/block breakdown

Embeddings:

```text
word = WordEmbedding(input_ids) + PositionEmbedding(position_ids) + TokenTypeEmbedding(token_type_ids)
word = LayerNorm(word)

entity = EntityEmbedding(entity_ids)                  # [B,E,entity_emb_size]
entity = Linear(entity_emb_size -> H, bias=False)      # when entity_emb_size != H
entity_pos = PositionEmbedding(clamp(entity_position_ids, min=0))  # [B,E,M,H]
entity_pos = masked_mean(entity_pos, mask=(entity_position_ids != -1), axis=M)
entity = entity + entity_pos + EntityTokenTypeEmbedding(entity_token_type_ids)
entity = LayerNorm(entity)
```

Encoder block, repeated `L` times:

```text
concat = cat(word, entity, dim=1) if entities exist else word
K,V = Linear(H -> H)(concat)
if entity-aware and entities exist:
  Qww = Linear(H -> H)(word)
  Qwe = Linear(H -> H)(word)
  Qew = Linear(H -> H)(entity)
  Qee = Linear(H -> H)(entity)
  scores = block_cat(Qww@Kw.T, Qwe@Ke.T, Qew@Kw.T, Qee@Ke.T) / sqrt(head_dim)
else:
  Q = Linear(H -> H)(concat)
  scores = Q@K.T / sqrt(head_dim)
scores += additive_mask
probs = softmax(scores, dim=-1)
context = probs @ V
attn_out = LayerNorm(Linear(H -> H)(context) + concat)
ffn = Linear(intermediate -> H)(GELU(Linear(H -> intermediate)(attn_out)))
out = LayerNorm(ffn + attn_out)
word, entity = split(out, [W,E], dim=1)
```

Task heads:

- Entity classification: `entity[:,0,:] -> dropout -> Linear(H -> num_labels)`.
- Entity pair: `cat(entity[:,0,:], entity[:,1,:]) -> dropout -> Linear(2H -> num_labels, bias=False)`.
- Entity span: gather `word[start]` and `word[end]`, concatenate with entity state, then `Linear(3H -> num_labels)`.
- Sequence classification/multiple choice: pool `word[:,0] -> Linear(H -> H) -> tanh -> classifier`.
- Token classification/QA: per-word `Linear(H -> labels)` or `Linear(H -> 2)`.
- Masked LM: RoBERTa LM transform and word vocab projection; entity prediction transform to `entity_emb_size` then entity vocab projection.

## 6. Attention requirements

LUKE attention is encoder-only noncausal self-attention. There is no autoregressive cache, no cross-attention, no grouped-query attention, and no sliding/local sparse mode.

Shape contract:

- Word hidden: `[B, W, H]`.
- Entity hidden: absent or `[B, E, H]`.
- Total keys/values: `[B, W + E, H]`.
- Heads: `A = num_attention_heads`; sampled `head_dim=64`.
- Scores/probs: `[B, A, W + E, W + E]`.

Entity-aware mode:

- Shared K/V projections consume the concatenated word/entity sequence.
- Queries are not shared by relation block: word queries use separate `query` and `w2e_query`; entity queries use `e2w_query` and `e2e_query`.
- Fusing into one dense attention backend requires either precomputing a block-assembled Q tensor with different Q rows per target key region, or a custom entity-aware score kernel. A vanilla FlashAttention call over one Q tensor is not equivalent when `use_entity_aware_attention=True`.

Masking:

- Word and entity masks concatenate along key axis.
- The extended mask broadcasts over heads and query positions.
- Values are cast to model dtype, then `(1 - mask) * finfo(dtype).min`.

Packed/varlen support:
Not implemented in source. DinoML can initially require dense padded `[B,W]` and `[B,E]` inputs.

## 7. Position encoding and custom math

No RoPE/ALiBi/relative bias is used. LUKE uses learned absolute positions for both word tokens and entity mention spans.

Word position IDs:

```python
def luke_word_position_ids(input_ids, pad_token_id):
    mask = (input_ids != pad_token_id).int()
    inc = torch.cumsum(mask, dim=1).type_as(mask) * mask
    return inc.long() + pad_token_id
```

Entity mention position average:

```python
def luke_entity_position_embedding(position_ids, position_table):
    pos = position_table(position_ids.clamp(min=0))      # [B,E,M,H]
    mask = (position_ids != -1).to(pos.dtype).unsqueeze(-1)
    summed = (pos * mask).sum(dim=-2)
    denom = mask.sum(dim=-2).clamp(min=1e-7)
    return summed / denom
```

Precomputable:
Position embedding weights are constants. Entity `position_ids` depend on tokenizer span packing and padding; the average is runtime graph work unless DinoML precomputes entity position embeddings in the CPU pipeline for a bounded integration.

## 8. Preprocessing and input packing

Tokenizer/runtime ABI:

- Word inputs: `input_ids [B,W]`, optional `attention_mask [B,W]`, optional `token_type_ids [B,W]`, optional `position_ids [B,W]`.
- Entity inputs: `entity_ids [B,E]`, `entity_attention_mask [B,E]`, optional `entity_token_type_ids [B,E]`, `entity_position_ids [B,E,M]` where `M=max_mention_length` (default 30 in tokenizer configs).
- Entity span classification adds `entity_start_positions [B,E]` and `entity_end_positions [B,E]` for word-state gathers.

Tokenizer details that affect parity:

- LUKE tokenizer is byte-level BPE with RoBERTa-style tokens.
- `model_max_length` is 512 in sampled tokenizer configs while model position table is commonly 514.
- `task="entity_classification"` requires exactly one entity span, inserts `<ent>` markers around the mention, and emits one `[MASK]` entity.
- `task="entity_pair_classification"` requires exactly two spans, inserts `<ent>` and `<ent2>` markers around head/tail mentions in source-order-safe fashion, and emits `[MASK]`, `[MASK2]` entity IDs.
- `task="entity_span_classification"` accepts a list of candidate spans, emits one mask entity per span, and records start/end word-token positions.
- Generic entity encoding can use actual Wikipedia entity titles via `entity_vocab`; absent titles become `[UNK]`, absent entity names with spans become `[MASK]`.
- Padding pads entity IDs with `[PAD]`, entity masks with 0, and entity positions with all `-1`.

Suggested DinoML boundary:
Keep tokenizer/entity packing on CPU for first integration. The GPU graph should accept already-packed dense tensors and guard shapes/ranges: word length <= model max, entity length <= configured max, mention length fixed, position IDs either `-1` or in range, and entity token type IDs compatible with `type_vocab_size`.

## 9. Graph rewrite / lowering opportunities

### Rewrite: entity mention position average fusion

Source pattern:
`Embedding(clamp(position_ids,0)) -> mask multiply -> sum(axis=-2) -> divide(clamp(sum(mask),1e-7))`.

Replacement:
One custom `masked_position_embedding_mean(position_ids, table)` helper or a generated gather-reduce kernel.

Preconditions:

- `position_ids` rank is `[B,E,M]`.
- Sentinel is exactly `-1`.
- Non-sentinel IDs are in `[0, max_position_embeddings - 1]`.
- Reduction axis is mention axis `M`, not entity axis.

Failure cases:
Different sentinel, weighted spans, or dynamic ragged span lengths without padded `M`.

Parity test sketch:
Random `position_ids` with all-padding, one-token, and multi-token spans; compare fp32 and fp16 against PyTorch, especially denominator clamp.

### Rewrite: entity-aware attention block lowering

Source pattern:
Four Q projections plus shared K/V, four score GEMMs, two concat operations, softmax, and value GEMM.

Replacement:
Custom entity-aware attention provider or a guarded graph expansion using existing BMM/GEMM/softmax primitives.

Preconditions:

- Dense padded word/entity tensors.
- `use_entity_aware_attention=True`.
- No attention dropout in inference.
- Same head dim for all projections.

Shape equations:
`W=input word length`, `E=entity length`, `T=W+E`, scores `[B,A,T,T]` assembled from `[W,W]`, `[W,E]`, `[E,W]`, `[E,E]` blocks.

Failure cases:
No entity tensor, `use_entity_aware_attention=False`, requests for attention probability outputs if optimized backend does not materialize dense probs.

Parity test sketch:
Single layer with small `W,E`, compare final word/entity states and optionally full attention matrix.

### Rewrite: task-specific entity gather heads

Source pattern:
Entity span head expands start/end indices to `[B,E,H]`, gathers word hidden states along sequence axis, concatenates `start`, `end`, and entity hidden states.

Replacement:
`batch_gather(word_hidden, indices)` plus concat and GEMM, or a fused gather-concat-linear head.

Preconditions:

- Start/end indices are valid word token positions after tokenizer special-token offsets.
- Output entity length equals classifier entity axis.

Failure cases:
Out-of-range padded start/end positions if caller does not mask downstream logits; incompatible with arbitrary ragged output records.

Parity test sketch:
Compare candidate-span logits for padded and unpadded entity batches.

### Rewrite: inference dropout removal

Source pattern:
Dropout after embeddings, attention probs, dense outputs, and heads.

Replacement:
Identity in eval mode.

Preconditions:
Model is compiled for inference/eval only.

Failure cases:
Training or stochastic MC-dropout requests.

## 10. Kernel fusion candidates

Highest priority:

- Entity-aware attention graph expansion/provider. This is the defining LUKE cost and cannot be treated as vanilla BERT attention when entities are present.
- LayerNorm + residual patterns around attention and FFN. LUKE has BERT-like post-norm blocks where residual+LayerNorm fusion is broadly useful.
- Entity position masked embedding mean. Small but custom and repeated for every call with entities.

Medium priority:

- FFN Linear + GELU + Linear with residual LayerNorm. Standard encoder throughput path.
- Entity span gather-concat-classifier for NER-style checkpoints with many candidate spans.
- Large entity prediction head `256 -> 500000` only if MLM/entity prediction is a target.

Lower priority:

- Pooler + classifier fusion for sequence/multiple-choice tasks.
- Word position ID generation on GPU. Prefer CPU-supplied `position_ids` or simple generated helper first.
- Attention-probability materialization. Defer unless output attentions are required.

## 11. Runtime staging plan

Stage 1: Config and packed-input ABI.
Parse LUKE configs, load word/entity/token-type/position embeddings, and accept CPU-packed tensors. Reject cache/generation fields as unsupported because source does not implement them.

Stage 2: Embedding parity.
Implement word embeddings and entity embeddings including `-1` mention mask averaging. Validate base and large dimensions.

Stage 3: One encoder block parity.
Lower entity-aware attention using explicit BMM/softmax graph first, plus FFN and LayerNorm. Run single-layer parity with small synthetic `W,E`.

Stage 4: Full encoder parity.
Run full `LukeModel` for `luke-base` with entities present and entities absent. Entities absent should use the ordinary attention path.

Stage 5: First useful heads.
Prioritize entity classification, entity-pair classification, and entity-span classification. Sequence/token/QA heads are straightforward optional follow-ups.

Stage 6: Optimized attention/fusion.
Add a custom entity-aware attention kernel/provider or guarded rewrite. Add residual LayerNorm and FFN fusions.

Stage 7: Optional MLM/entity prediction.
Add tied word/entity output heads, large vocab/entity-vocab GEMMs, and output alias checks.

## 12. Parity and validation plan

- Tokenizer snapshot tests: verify CPU packing for one entity, two entities, pair text, truncation, left/right padding, and span classification start/end indices.
- Custom op tests: `create_position_ids_from_input_ids`; entity position masked average with all `-1`, one valid position, max mention length, and mixed padding.
- Attention tests: compare relation-block score assembly before softmax for small `B=1`, `W=3`, `E=2`, `A=2`.
- Single block parity: fixed random weights, no dropout, fp32 tolerance around `1e-5` to `1e-4`.
- Full encoder parity: `studio-ousia/luke-base`, entities absent and present, compare word and entity hidden states.
- Head parity: open-entity, TACRED, and CoNLL configs using packed tokenizer outputs; compare logits.
- Reduced precision: fp16/bf16 tolerances should be looser around softmax and LayerNorm; start with fp32 acceptance before provider fusion.
- Guard tests: reject `type_vocab_size=1` with nonzero type IDs, invalid entity positions, missing `entity_position_ids` when `entity_ids` are present, and unsupported output-attentions materialization if not implemented.

## 13. Performance probes

- CPU tokenizer/entity packing throughput by task and entity count.
- Encoder throughput sweep over `B`, `W`, and `E`; include `E=0`, `E=1`, `E=2`, `E=32`, and many candidate spans.
- Entity-aware attention microbench: explicit four-GEMM expansion versus custom provider.
- FFN and LayerNorm bandwidth probes for base and large.
- Entity span classification head sweep over candidate span count.
- MLM/entity prediction head GEMM probes if that target is admitted; entity vocab projection is very large.
- Memory probe for attention probs `[B,A,W+E,W+E]`, especially if output attentions or unfused softmax stores full matrices.
- Batch-size sweep for entity-pair TACRED shape, entity-classification Open Entity shape, and CoNLL span-candidate shape.

## 14. Skip/defer list

- Training losses and gradient checkpointing.
- Autoregressive generation, KV cache, beam search, and `use_cache`/`output_past` config fields.
- Output attentions/hidden-state tuples for optimized runtime unless needed for debugging.
- Masked LM and masked entity prediction for the first entity-classification integration.
- Dynamic tokenizer execution inside DinoML GPU runtime.
- General ragged entity sequences; use dense padded `[B,E,M]` first.
- Quantized/packed weights and multi-GPU tensor parallelism.

## 15. Final implementation checklist

- [ ] Parse `LukeConfig` and sampled checkpoint overrides, especially `max_position_embeddings`, `type_vocab_size`, and `layer_norm_eps`.
- [ ] Define packed LUKE input ABI: word tensors, entity tensors, entity mention positions, and optional span start/end indices.
- [ ] Load word, entity, position, token-type, encoder, and head weights with MLM/entity-head alias preservation when needed.
- [ ] Implement word position ID helper or require caller-supplied `position_ids`.
- [ ] Implement entity embedding projection and masked mention-position average.
- [ ] Lower ordinary no-entity attention path.
- [ ] Lower entity-aware attention score block assembly.
- [ ] Implement encoder FFN, residual, LayerNorm, and pooler.
- [ ] Implement entity classification head.
- [ ] Implement entity-pair classification head with no-bias classifier.
- [ ] Implement entity-span gather-concat-classifier head.
- [ ] Add tokenizer packing parity fixtures from saved configs.
- [ ] Add one-layer and full-encoder fp32 parity tests.
- [ ] Add guarded rejection tests for ignored historical cache fields and invalid packed entity tensors.
- [ ] Benchmark entity-aware attention expansion versus fused/provider path.

