# Splinter Transformers Audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: tau/splinter-base, tau/splinter-base-qass, tau/splinter-large, tau/splinter-large-qass
Config source: official Hugging Face config.json files fetched 2026-05-13
Source files inspected:
  transformers/src/transformers/models/splinter/modeling_splinter.py
  transformers/src/transformers/models/splinter/configuration_splinter.py
  transformers/src/transformers/models/splinter/tokenization_splinter.py
  transformers/docs/source/en/model_doc/splinter.md
  transformers/tests/models/splinter/test_modeling_splinter.py
Any missing files or assumptions: no gated configs in the representative sweep; no remote-code path required.
```

Primary source URLs for future review:

- [modeling_splinter.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/splinter/modeling_splinter.py)
- [configuration_splinter.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/splinter/configuration_splinter.py)
- [tokenization_splinter.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/splinter/tokenization_splinter.py)

The modeling file is the authoritative implementation. The Splinter encoder layers are copied from BERT/Align-style modules, but this family owns a custom Question-Aware Span Selection head.

## 2. High-level architecture

Splinter is a text-only encoder transformer for extractive span selection. There is no autoregressive decoder and no KV cache. The first DinoML target should be `SplinterForQuestionAnswering` inference, because all official configs name that architecture and the QASS head is the task-defining runtime surface.

Dataflow:

```text
WordPiece preprocessing -> token/segment/position embeddings -> encoder stack -> QASS span head -> start/end logits
```

Stage decomposition:

```text
CPU/data pipeline: WordPiece tokenization, [QUESTION] insertion for paired inputs, padding/attention mask
GPU/runtime: embeddings, noncausal encoder self-attention, FFN, question-position gather, QASS matmuls, mask addition
Postprocessing: argmax or top-k span selection over start/end logits, task-specific answer text reconstruction outside model graph
```

`SplinterModel` alone is a feature extractor. `SplinterForPreTraining` reuses the same encoder and QASS head but supports multiple question tokens per sample and a dynamic question-position discovery path; it is optional for first inference parity unless pretraining-style RSS inference is desired.

## 3. Important config dimensions

| Field | Source default | tau base | tau large | Runtime effect |
|---|---:|---:|---:|---|
| `vocab_size` | 30522 | 28996 | 28996 | word embedding rows |
| `hidden_size` | 768 | 768 | 1024 | encoder/QASS width |
| `num_hidden_layers` | 12 | 12 | 24 | repeated encoder blocks |
| `num_attention_heads` | 12 | 12 | 16 | MHA heads |
| `head_dim` | derived | 64 | 64 | `hidden_size / num_attention_heads` |
| `intermediate_size` | 3072 | 3072 | 4096 | FFN expansion |
| `max_position_embeddings` | 512 | 512 | 512 | absolute position table and max tokenizer length |
| `type_vocab_size` | 2 | 2 | 2 | token type embedding rows |
| `hidden_act` | `gelu` | `gelu` | `gelu` | FFN and QASS FC activation |
| `layer_norm_eps` | 1e-12 | 1e-12 | 1e-12 | all LayerNorms |
| `question_token_id` | 104 | omitted -> 104 | omitted -> 104 | question-position discovery |
| cache support | none in source path | none | none | encoder-only, no decode cache |

Representative checkpoint sweep:

| Model id | Access | Hidden | Layers | Heads | FFN | Vocab | Variant |
|---|---:|---:|---:|---:|---:|---:|---|
| `tau/splinter-base` | public | 768 | 12 | 12 | 3072 | 28996 | base encoder, QASS head may be randomly initialized if missing weights |
| `tau/splinter-base-qass` | public | 768 | 12 | 12 | 3072 | 28996 | base plus QASS checkpoint weights |
| `tau/splinter-large` | public | 1024 | 24 | 16 | 4096 | 28996 | large encoder, QASS head may be randomly initialized if missing weights |
| `tau/splinter-large-qass` | public | 1024 | 24 | 16 | 4096 | 28996 | large plus QASS checkpoint weights |

The `*-qass` configs contain `initialize_new_qass: true`, but the inspected current source does not read that field. Treat it as historical metadata for this source basis.

## 3a. Family variation traps

- Official configs omit `question_token_id`; current `SplinterConfig` supplies `104`. Do not infer it from tokenizer special-token order without checking the config/default.
- The source requires `hidden_size % num_attention_heads == 0` unless an unused `embedding_size` attribute exists.
- There is no GQA/MQA, RoPE, ALiBi, relative position bias, sliding window, causal mask, or cross-attention in the inspected source.
- `SplinterTokenizer` defaults to lowercase in code, but official tokenizer configs set `do_lower_case: false`. Use repo tokenizer config for parity.
- Paired-input tokenization inserts `[QUESTION] .` differently depending on padding side: right-padding path places it between sequence A and B; left-padding path places it before the final separator after B.
- QASS logits orientation is `[B, Q, S]` for explicit/multiple question positions and `[B, S]` only after the QA wrapper squeezes a single discovered question position.
- `SplinterForQuestionAnswering` without `question_positions` uses `argmax(eq(input_ids, question_token_id))`; if no question token exists, PyTorch returns position `0`. Admission should guard this for production parity.
- `SplinterForPreTraining._prepare_question_positions` uses `where`, `bincount`, dynamic max question count, `arange`, `cat`, and indexed assignment. This is a harder dynamic-shape path than the QA wrapper and should be deferred or bounded with explicit `question_positions`.
- Attention mask addition uses large negative floating values from inherited Transformers mask conversion and an additional QASS logit mask add. Mask dtype/value parity matters for fp16.
- There are no NCHW/NHWC tensors. Layout guidance is sequence-major `[B, S, H]`; any transpose-elimination pass must preserve attention head axes.

## 4. Operator coverage checklist

Tensor/layout ops:

- `Embedding(input_ids)` -> `[B, S, H]`, `Embedding(token_type_ids)`, `Embedding(position_ids)`.
- `arange`, slice, expand for default position IDs `[1, max_pos] -> [1, S]`.
- zeros/ones defaults for token types and attention mask.
- reshape/view `[B, S, H] -> [B, S, heads, Dh]`, transpose to `[B, heads, S, Dh]`, transpose back, contiguous, reshape.
- `permute(0, 2, 1)` for QASS sequence reps `[B, S, H] -> [B, H, S]`.
- `unsqueeze`, `repeat`, `gather(dim=1)` for question reps.
- `squeeze(1)` for single-question QA output.
- `eq`, `int`, `argmax(dim=-1)` for QA question-position discovery.
- Optional pretraining path: `where`, `bincount`, `full`, `arange`, `cat`, dynamic indexed assignment.

Neural primitives:

- LayerNorm over `H` with `eps=1e-12`.
- Dropout is present in source but disabled for inference.
- Linear projections with bias for Q, K, V, attention output, FFN up/down, and QASS FC transforms.
- Bias-free QASS classifiers: `Linear(H -> H, bias=False)` for start and end query transforms.
- GELU activation for FFN and QASS fully connected layers.
- Residual add before post-attention and post-FFN LayerNorm.

Attention primitives:

- Dense noncausal self-attention only.
- Scores: `matmul(q, k.transpose(-2, -1)) * Dh**-0.5`.
- Mask add before softmax.
- Softmax on `dim=-1`, explicitly computed in fp32 and cast back to query dtype in eager attention.
- Attention/value matmul.

Position encoding:

- Learned absolute position embeddings only; no runtime sinusoidal/RoPE math.

Preprocessing-coupled ops:

- WordPiece tokenizer with BERT normalizer/pre-tokenizer/decoder.
- Special `[QUESTION]` token and period insertion for paired inputs.
- Attention masks and token type IDs enter the GPU graph.

Postprocessing:

- End-to-end QA parity needs span selection from start/end logits and tokenizer offset/text reconstruction outside this model module. The modeling source itself returns logits only.

## 5. Layer/block breakdown

Embeddings:

```text
input_ids: [B, S] int64
token_type_ids: [B, S] int64, default zeros
position_ids: [1 or B, S] int64, default arange slice
x = word_embedding(input_ids) + token_type_embedding(token_type_ids) + position_embedding(position_ids)
x = LayerNorm(x, eps=1e-12)
```

Encoder block, repeated `L` times:

```text
residual = x                                      # [B, S, H]
q = Linear(H -> H, bias=True)(x).view(B,S,A,Dh).transpose(1,2)
k = Linear(H -> H, bias=True)(x).view(B,S,A,Dh).transpose(1,2)
v = Linear(H -> H, bias=True)(x).view(B,S,A,Dh).transpose(1,2)
attn = softmax((q @ k.transpose(-2,-1)) * Dh**-0.5 + mask, dim=-1)
y = (attn @ v).transpose(1,2).reshape(B,S,H)
y = Linear(H -> H, bias=True)(y)
x = LayerNorm(y + residual)
residual = x
y = GELU(Linear(H -> I, bias=True)(x))
y = Linear(I -> H, bias=True)(y)
x = LayerNorm(y + residual)
```

QASS head:

```text
question_positions: [B, Q]
gathered = gather(sequence_output, dim=1, positions repeated to [B,Q,H])
query_start = LayerNorm(GELU(Linear(H -> H)(gathered)))
query_end   = LayerNorm(GELU(Linear(H -> H)(gathered)))
start_reps  = LayerNorm(GELU(Linear(H -> H)(sequence_output)))
end_reps    = LayerNorm(GELU(Linear(H -> H)(sequence_output)))
start_logits = Linear(H -> H, bias=False)(query_start) @ start_reps.transpose(1,2)
end_logits   = Linear(H -> H, bias=False)(query_end) @ end_reps.transpose(1,2)
```

Base shapes are `H=768, A=12, Dh=64, I=3072`; large shapes are `H=1024, A=16, Dh=64, I=4096`.

## 6. Attention requirements

Splinter requires encoder self-attention only:

- Causal: no.
- Cross-attention: no.
- Head pattern: standard MHA, `num_key_value_heads == num_attention_heads`.
- Query/key/value width: all `H`, split to `A x Dh`.
- Query length and key length: both `S`.
- Masking style: attention mask is converted to an additive broadcast mask, then added to scores before softmax.
- Packed/varlen support: not implemented in source.
- Sliding/local attention: not implemented.
- Position interaction: learned absolute embeddings are added before attention; attention itself has no position bias.
- KV cache: not applicable.
- FlashAttention/SDPA compatibility: source dispatches through `ALL_ATTENTION_FUNCTIONS` with eager fallback. A fused SDPA/Flash-style implementation is valid when it preserves noncausal additive-mask semantics, fp32 softmax behavior where required, no dropout in eval, and output layout `[B,S,H]`.

## 7. Position encoding and custom math

Position encoding is a learned table:

```python
position_ids = arange(max_position_embeddings).expand((1, -1))[:, :seq_length]
embeddings = word_embeddings + token_type_embeddings + position_embeddings(position_ids)
```

This can be precomputed only as an index vector; the embedding lookup remains weight-dependent. No RoPE, ALiBi, sinusoidal, or relative-bias math is required.

QASS question-position discovery is source-specific:

```python
# QA wrapper fallback
question_positions = argmax((input_ids == question_token_id).int(), dim=-1).unsqueeze(-1)
```

```python
# Pretraining fallback, dynamic Q per batch
rows, flat_positions = where(input_ids == question_token_id)
num_questions = bincount(rows)
positions = full((B, max(num_questions)), pad_token_id)
positions[rows, concat([arange(n) for n in num_questions])] = flat_positions
```

For DinoML first integration, prefer explicit `question_positions` or a guarded single-question-token discovery path.

## 8. Preprocessing and input packing

Tokenizer contract:

- `SplinterTokenizer` is WordPiece-backed with BERT normalization and pre-tokenization.
- Official tokenizer config sets `do_lower_case=false` and `model_max_length=512`.
- Special tokens are `[UNK]`, `[SEP]`, `[PAD]`, `[CLS]`, `[MASK]`, and `[QUESTION]`.
- Model input names from tokenizer source are `input_ids` and `attention_mask`; token type IDs are still accepted by the model and created by the backend post-processor.

Pair template:

```text
right padding: [CLS] A [QUESTION] . [SEP] B [SEP]
left padding:  [CLS] A [SEP] B [QUESTION] . [SEP]
```

GPU graph inputs:

- `input_ids [B,S]`, or alternatively `inputs_embeds [B,S,H]`.
- `attention_mask [B,S]`; if absent, source creates all ones.
- `token_type_ids [B,S]`; if absent, source creates zeros.
- `position_ids [1,S]` or `[B,S]`; if absent, source slices the registered absolute position buffer.
- Optional `question_positions [B,Q]`; strongly recommended for bounded DinoML lowering.

There is no OCR, image/audio/video preprocessing, packed sequence metadata, or modality embedding stitch.

## 9. Graph rewrite / lowering opportunities

### Rewrite: split Q/K/V projections -> packed QKV GEMM

Source pattern:

```text
q = Linear(H,H)(x); k = Linear(H,H)(x); v = Linear(H,H)(x)
```

Replacement:

```text
qkv = Linear(H, 3H)(x) -> split last dim as [q, k, v]
```

Preconditions:

- Same input tensor `x`.
- All three projections have bias.
- Packed weight rows are concatenated in source order `[q, k, v]`.
- No consumer observes individual module outputs before reshape.

Shape equations:

- Input `[B,S,H]`.
- Packed output `[B,S,3H]`.
- Split to three `[B,S,H]` tensors, then view to `[B,A,S,Dh]`.

Failure cases:

- Weight tying or hooks on individual projections.
- Quantized/packed checkpoint formats that store projections separately and cannot be packed at load time.

Parity test sketch:

- Compare one layer attention output before/after rewrite for random fp32/fp16 inputs and masks.

### Rewrite: QASS classifier matmul -> batched GEMM

Source pattern:

```text
start_logits = LinearNoBias(H,H)(query_start) @ start_reps.transpose(1,2)
```

Replacement:

```text
query_projected [B,Q,H] x reps_T [B,H,S] -> logits [B,Q,S]
```

Preconditions:

- `question_positions` has bounded static or bucketed `Q`.
- Sequence reps are contiguous or materialized with known strides.
- Mask add is applied after logits.

Failure cases:

- Dynamic `Q` from pretraining `where/bincount` path without bounded output allocation.

Parity test sketch:

- Test `Q=1`, `Q=2`, and padded question positions with mask additions.

### Rewrite: explicit question-token search -> validated question_positions input

Source pattern:

```text
argmax(eq(input_ids, question_token_id)) -> [B,1]
```

Replacement:

```text
Host/tokenizer supplies question_positions [B,1]
Runtime validates 0 <= position < S and input_ids[position] == question_token_id when input_ids are present.
```

Preconditions:

- First integration controls tokenizer output.
- Exactly one question token per sample for QA wrapper.

Failure cases:

- Multiple question tokens where first occurrence semantics matter.
- Inputs supplied as `inputs_embeds` without token IDs.

### Rewrite: attention transpose cleanup

Source pattern:

```text
[B,S,H] -> view [B,S,A,Dh] -> transpose [B,A,S,Dh] -> attention -> transpose/reshape [B,S,H]
```

Replacement:

Use attention kernel ABI that consumes projected packed layout directly or fuses view/transpose into projection epilogue.

Layout constraints:

- Preserve sequence-major public tensor ABI `[B,S,H]`.
- Head axis must remain before query/key sequence inside attention.
- This is not an NHWC/NCHW rewrite; protect text sequence axes from generic channel-last passes.

## 10. Kernel fusion candidates

Highest priority:

- Encoder LayerNorm, especially `residual + LayerNorm` after attention and FFN.
- Packed QKV GEMM plus split/view for prefill-style encoder throughput.
- Dense noncausal attention using SDPA/Flash-compatible backend for `[B,A,S,S]`.
- FFN `Linear -> GELU -> Linear` with GEMM epilogue activation where practical.
- QASS batched GEMM path for `[B,Q,H] x [B,H,S]`, because it is task-critical and not covered by generic classification heads.

Medium priority:

- Embedding sum plus LayerNorm fusion.
- QASS `Linear -> GELU -> LayerNorm` repeated four times.
- Mask-add fusion into attention and QASS logits.
- Static/bucketed `Q=1` specialization for `SplinterForQuestionAnswering`.

Lower priority:

- Dynamic pretraining question-position discovery with `where/bincount/scatter`.
- Training losses and gradient checkpointing.
- Tokenizer CPU throughput optimizations.

## 11. Runtime staging plan

Stage 1: parse Splinter configs and tokenizer metadata; load base and large checkpoints; admit `SplinterModel` encoder-only with explicit `input_ids`, `attention_mask`, and `token_type_ids`.

Stage 2: single encoder block parity for embeddings, attention, FFN, residual LayerNorm, and masks.

Stage 3: full encoder parity for base and large sizes at fixed `S <= 512`.

Stage 4: implement QASS head with explicit `question_positions [B,Q]`; validate `Q=1` QA output and `Q>1` pretraining-style output without dynamic discovery.

Stage 5: add guarded QA fallback for discovering a single `[QUESTION]` token from `input_ids`.

Stage 6: optimize packed QKV, fused attention, residual LayerNorm, FFN, and QASS batched GEMMs.

Stage 7: optionally support `SplinterForPreTraining._prepare_question_positions` only after DinoML has a bounded variable-`Q` output policy.

Can be stubbed initially: dropout, training losses, hidden-state/attention tuple capture, gradient checkpointing, dynamic pretraining question discovery, and tokenizer-in-runtime execution.

## 12. Parity and validation plan

- Config/load tests for all four official tau configs; verify omitted `question_token_id` resolves to `104`.
- Embedding parity with explicit and default `token_type_ids`/`position_ids`.
- Attention parity for fp32 with masks containing padding and all-ones masks.
- One-block and full-encoder parity for base-sized and large-sized synthetic configs.
- QASS parity with explicit `question_positions` for `Q=1`, `Q=2`, and padded positions.
- QA wrapper parity for fallback argmax discovery with exactly one `[QUESTION]` token per sample.
- Masked-logit parity: verify padded positions receive `torch.finfo(dtype).min` additions in both start and end logits.
- Official slow-test examples: `tau/splinter-base-qass` should predict start/end positions `(10,12)` for the documented QA example and `(7,7)`, `(10,12)` for the two-question pretraining example.

Suggested tolerances:

- fp32: absolute/relative `1e-4` for logits.
- fp16/bf16: compare against Transformers in the same dtype where possible; start with `5e-2` absolute for logits around mask boundaries and tighten after fused attention policy is selected.

## 13. Performance probes

- Encoder throughput sweep: base vs large, `B in {1,4,16}`, `S in {64,128,256,512}`.
- Attention backend comparison: eager matmul/softmax vs SDPA/Flash-compatible noncausal path.
- QASS head sweep: `Q in {1,2,8,32}` to isolate batched logits cost.
- Mask density sweep for attention and QASS logit masks.
- Packed QKV rewrite benchmark versus three independent GEMMs.
- FFN GEMM/GELU fusion probe.
- End-to-end extractive QA latency split into tokenizer, encoder, QASS head, and CPU span postprocessing.
- Memory probe for attention score tensor `[B,A,S,S]`, especially large checkpoint at `S=512`.

## 14. Skip/defer list

- Training losses and `CrossEntropyLoss`.
- Gradient checkpointing.
- Hidden-state and attention capture outputs, unless needed for debugging.
- `SplinterForPreTraining._prepare_question_positions` dynamic `where/bincount/scatter` path.
- Multi-GPU data parallel behavior.
- Runtime tokenizer execution.
- Beam search, KV cache, causal decode, RoPE, ALiBi, GQA/MQA, MoE, vision/audio/video paths: not part of this family.

## 15. Final implementation checklist

- [ ] Parse `SplinterConfig`, including defaulted `question_token_id=104`.
- [ ] Load official tau base/large weights and preserve QASS head presence/missing-weight behavior.
- [ ] Implement embedding sum with token, position, and token-type embeddings.
- [ ] Implement encoder MHA with additive mask and fp32 softmax semantics.
- [ ] Implement FFN `Linear -> GELU -> Linear` and residual LayerNorm blocks.
- [ ] Implement QASS gather by `question_positions`.
- [ ] Implement QASS start/end batched GEMMs and logit mask add.
- [ ] Add guarded single-question-token discovery from `input_ids`.
- [ ] Add parity tests for encoder-only, QA head `Q=1`, and QASS `Q>1`.
- [ ] Add official checkpoint smoke tests for `tau/splinter-base-qass`.
- [ ] Add packed QKV rewrite with source-order `[q,k,v]` weight transform.
- [ ] Add performance probes for attention, FFN, and QASS batched GEMM.
