# BigBird Transformers Family Audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: google/bigbird-roberta-base primary; google/bigbird-roberta-large, google/bigbird-base-trivia-itc, l-yohai/bigbird-roberta-base-mnli representative configs
Config source: Hugging Face config.json snapshots saved under agents/plans/transformers/big_bird/_sources/
Source files inspected:
- X:/H/transformers/src/transformers/models/big_bird/configuration_big_bird.py
- X:/H/transformers/src/transformers/models/big_bird/modeling_big_bird.py
- X:/H/transformers/src/transformers/models/big_bird/tokenization_big_bird.py
- X:/H/transformers/src/transformers/modeling_utils.py for full-attention mask helper behavior
Any missing files or assumptions:
- google/bigbird-base-natural-questions config fetch returned an authentication error, so it is not used in the sweep.
- This report scopes DinoML first integration to encoder-style BigBird inference: base encoder, masked LM/pretraining heads, span QA, and sequence classification. `BigBirdForCausalLM` exists but should be a separate/full-attention decoder follow-up.
```

Primary source URLs:

- `https://github.com/huggingface/transformers/tree/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/big_bird`
- `https://huggingface.co/google/bigbird-roberta-base/blob/main/config.json`
- `https://huggingface.co/google/bigbird-roberta-large/blob/main/config.json`
- `https://huggingface.co/google/bigbird-base-trivia-itc/blob/main/config.json`
- `https://huggingface.co/l-yohai/bigbird-roberta-base-mnli/blob/main/config.json`

## 2. High-level architecture

BigBird in this directory is a BERT/RoBERTa-like text encoder with absolute position embeddings and a configurable attention module:

```text
tokenization + special-token packing -> token/type/position embeddings
-> N encoder blocks with block-sparse or full self-attention
-> task head: MLM/pretraining, QA span logits, CLS classification, token classification, multiple choice
```

The production-significant path is long-sequence encoder inference with `attention_type="block_sparse"`. Full attention is still required as a fallback for short sequences, decoder/cross-attention use, and easier parity bring-up.

Stageable pieces:

- CPU/data pipeline: SentencePiece-like Unigram tokenizer, `[CLS] A [SEP]` or `[CLS] A [SEP] B [SEP]`, attention mask, optional token type IDs.
- GPU/runtime: embeddings, encoder blocks, task head.
- Independently testable: embeddings, one encoder block in `original_full`, one encoder block in `block_sparse`, final heads.
- Cacheable: encoder outputs for downstream retrieval/classification pipelines. This is not a KV-cache for the primary encoder target.

## 3. Important config dimensions

Source defaults from `BigBirdConfig`:

| field | default | runtime effect |
|---|---:|---|
| `vocab_size` | 50358 | token embedding and LM head width |
| `hidden_size` | 768 | encoder width |
| `num_hidden_layers` | 12 | repeated blocks |
| `num_attention_heads` | 12 | MHA heads |
| `head_dim` | 64 | inferred as `hidden_size / num_attention_heads` |
| `intermediate_size` | 3072 | FFN expansion |
| `hidden_act` | `gelu_new` | FFN and MLM transform activation |
| `max_position_embeddings` | 4096 | absolute position table and `max_seqlen` for sparse planning |
| `type_vocab_size` | 2 | token type embedding table; QA configs may override |
| `attention_type` | `block_sparse` | `block_sparse` or `original_full` |
| `block_size` | 64 | sparse block granularity and padding multiple |
| `num_random_blocks` | 3 | sparse random blocks per query block |
| `use_bias` | true | Q/K/V projection bias |
| `rescale_embeddings` | false | optional multiply by `sqrt(hidden_size)` |
| `use_cache` | true | only meaningful for full-attention decoder path |

Representative checkpoint sweep:

| model id | architecture | layers | hidden | heads | head dim | FFN | max pos | attention | block/random | type vocab | notable head |
|---|---|---:|---:|---:|---:|---:|---:|---|---|---:|---|
| `google/bigbird-roberta-base` | `BigBirdForPreTraining` | 12 | 768 | 12 | 64 | 3072 | 4096 | block sparse | 64 / 3 | 2 | MLM + NSP |
| `google/bigbird-roberta-large` | `BigBirdForMaskedLM` | 24 | 1024 | 16 | 64 | 4096 | 4096 | block sparse | 64 / 3 | 2 | MLM |
| `google/bigbird-base-trivia-itc` | `BigBirdForQuestionAnswering` | 12 | 768 | 12 | 64 | 3072 | 4096 | block sparse | 64 / 3 | 16 | span QA |
| `l-yohai/bigbird-roberta-base-mnli` | `BigBirdForSequenceClassification` | 12 | 768 | 12 | 64 | 3072 | 4096 | block sparse | 64 / 3 | 2 | 3-label CLS |

Checkpoint facts above come from `config.json` snapshots. `head_dim` is inferred from source shape logic. `torch_dtype` is absent for the Google configs inspected and `float32` only in the MNLI config.

## 3a. Family variation traps

- `attention_type` is mutable at runtime. If `seq_length <= (5 + 2 * num_random_blocks) * block_size`, the model logs a warning and calls `set_attention_type("original_full")`. With default `block_size=64`, `num_random_blocks=3`, the threshold is `704`; lengths `<=704` switch to dense full attention.
- Block-sparse inputs are padded internally to a multiple of `block_size`; output sequence states are unpadded afterward.
- `block_sparse` is encoder-only in practice. Cross-attention raises unless attention is `original_full`; causal generation should use `original_full`.
- The source sparse implementation has fixed two global blocks: first and last block. Window size is fixed to three blocks. ETC/global-token variants are explicitly not supported.
- During evaluation, random attention plans are all zeros in the inspected source (`if not self.training: return rand_attn`). DinoML must match this for HF parity unless intentionally routing to a different trained/eval mode.
- The sparse path always reconstructs a dense `attention_probs` tensor `[B, H, S, S]` with Python loops after computing the context. This is not needed for hidden states and should be optional in DinoML fast paths.
- QA fine-tunes may use `type_vocab_size=16`, not the base default `2`.
- `BigBirdForQuestionAnswering` derives `question_lengths` by the first `sep_token_id` if not provided, then masks question tokens out of start/end competition.
- Tokenizer config has BigBird-specific `[CLS]`/`[SEP]` post-processing and a SentencePiece/Unigram-style metaspace path; tokenizer work is CPU pipeline unless compiling end-to-end preprocessing.
- No RoPE, ALiBi, GQA/MQA, MoE, sliding-window config knob, or relative-bias table exists in this BigBird source.

## 4. Operator coverage checklist

Tensor/layout ops:

- `view`/reshape, transpose/permute, contiguous, squeeze/unsqueeze, slice, pad last sequence dimension.
- `cat` over block/key/value fragments.
- `index_select` / gather by block indices for random attention.
- `argmax`, `eq`, `arange`, `where` for QA question masking.
- Dense output slicing for `padding_len` removal and `logits_to_keep`.

Neural network primitives:

- Embedding lookup: word `[vocab, hidden]`, position `[max_position_embeddings, hidden]`, token type `[type_vocab_size, hidden]`.
- Elementwise adds for embeddings and residuals.
- LayerNorm with `eps=1e-12`.
- Linear with bias for Q/K/V when `use_bias=True`.
- Attention output projection `Linear(hidden -> hidden)`.
- FFN `Linear(hidden -> intermediate) -> gelu_new -> Linear(intermediate -> hidden)`.
- Task heads:
  - MLM transform `Linear(hidden -> hidden) -> gelu_new -> LayerNorm -> tied Linear(hidden -> vocab) + bias`.
  - NSP `Linear(hidden -> 2)`.
  - sequence classification on token 0: dropout, `Linear(hidden -> hidden)`, activation, dropout, `Linear(hidden -> num_labels)`.
  - QA head: dropout, FFN-style intermediate/output block, `Linear(hidden -> 2)`, split start/end.

Attention primitives:

- Full MHA for `original_full`: Q/K/V projections, `[B,H,S,D] @ [B,H,D,T]`, scale by `1/sqrt(D)`, additive mask, softmax, dropout, value matmul.
- Block-sparse MHA for encoder: first/last global blocks, three-block local window, random block gathers, per-region softmaxes, and sparse value matmuls.

Generation/cache ops:

- Optional/deferred for primary target. `BigBirdForCausalLM` uses `GenerationMixin`, `DynamicCache`, and `logits_to_keep`, but correct decoder/cross-attention integration requires `original_full`.

Preprocessing-coupled ops:

- Tokenizer emits `input_ids` and `attention_mask`; token type IDs may be generated by post-processing or by model defaults.
- Pair input layout is `[CLS] A [SEP] B [SEP]` with type IDs 0 for A and 1 for B.

Aliasing/tied weights:

- MLM/causal/pretraining decoder weight is tied to `bert.embeddings.word_embeddings.weight`.
- Decoder bias is tied through `cls.predictions.bias`.

## 5. Layer/block breakdown

Embeddings:

```text
input_ids [B,S] or inputs_embeds [B,S,H]
token_emb = Embedding(vocab_size, H)
type_emb = Embedding(type_vocab_size, H), default all-zero type ids
pos_emb = Embedding(max_position_embeddings, H), sliced by past length for decoder full-attn path
x = token_emb + type_emb + pos_emb
x = dropout(x)
x = LayerNorm(x)
```

Encoder block, repeated `num_hidden_layers`:

```text
q = Linear(H -> H, bias=use_bias)(x).view(B,S,Hh,D).transpose(1,2)
k = Linear(H -> H, bias=use_bias)(x).view(B,S,Hh,D).transpose(1,2)
v = Linear(H -> H, bias=use_bias)(x).view(B,S,Hh,D).transpose(1,2)
attn = full_or_block_sparse_attention(q, k, v, masks)
x_attn = Linear(H -> H)(attn)
x = LayerNorm(dropout(x_attn) + residual)
ff = Linear(H -> I)(x)
ff = gelu_new(ff)
ff = Linear(I -> H)(ff)
x = LayerNorm(dropout(ff) + residual)
```

For base: `H=768`, `Hh=12`, `D=64`, `I=3072`. For large: `H=1024`, `Hh=16`, `D=64`, `I=4096`.

## 6. Attention requirements

Full attention:

- Noncausal encoder self-attention for normal encoder use.
- Causal self-attention when `config.is_decoder=True`, via `get_extended_attention_mask`.
- Cross-attention only when `add_cross_attention=True`; this forces `original_full` in `BigBirdModel.__init__`.
- Cache shape per layer is effectively `[B, num_heads, cache_seq, head_dim]` for key and value, before any expansion. There is no GQA/MQA.
- Full attention mask is additive with `torch.finfo(dtype).min`, after 2D/3D mask broadcasting and optional causal masking.

Block-sparse attention:

- Noncausal self-attention only.
- Input sequence is padded to a multiple of `block_size`; the source raises if sparse attention sees non-divisible lengths.
- Sparse masks:
  - `blocked_encoder_mask`: `[B, S/block, block]`
  - `band_mask`: `[B, 1, S/block - 4, block, 3*block]`
  - `from_mask`: `[B, 1, S, 1]`
  - `to_mask`: `[B, 1, 1, S]`
  - `rand_mask`: `[B, H, S/block - 2, block, num_random_blocks*block]`
- Score masking uses `-10000.0`, not dtype minimum.
- First and last query blocks attend all key tokens.
- Middle query blocks attend first global block, previous/current/next sliding blocks, random blocks, and last global block.
- Second and second-last query blocks have special edge patterns combining nearby sliding blocks, both global sides, and random blocks.
- The source computes several small BMM/einsum regions rather than one generic sparse matrix multiply.

Packed/varlen support:

- None in source. Sequence padding is explicit and mask-driven.

FlashAttention/SDPA compatibility:

- `original_full` can map to standard dense MHA/SDPA.
- `block_sparse` needs a custom sparse attention kernel or a rewrite to full attention for bring-up and short sequences. Standard FlashAttention does not directly represent the random/global/sliding block pattern.

## 7. Position encoding and custom math

Position encoding is learned absolute embedding only:

```python
position_ids = arange(max_position_embeddings)[None, past_len:past_len + seq_len]
x = word_embeddings(input_ids) + token_type_embeddings(token_type_ids)
x = x + position_embeddings(position_ids)
```

No RoPE, ALiBi, relative position bias, or convolutional positional encoding is present.

Sparse random plan sketch:

```python
if seq_len in [1024, 3072, 4096]:
    rand_attn = old_plan(max_position_embeddings, last_idx=1024)
else:
    plan_len, plan_rand = get_rand_attn_plan(seq_len, block_size, num_random_blocks)
    rand_attn = per_head_plan(seq_len, block_size, plan_len, plan_rand)
if not training:
    rand_attn = zeros_like(rand_attn)
```

For inference parity against the inspected source, the eval zero-random behavior should be treated as source-derived behavior, even though the config still names `num_random_blocks`.

## 8. Preprocessing and input packing

Tokenizer/runtime contract:

- Text tokenizer is a tokenizers-backed Unigram model using `spiece.model`/`tokenizer.json`.
- Normalizer strips neither side and replaces repeated spaces with the metaspace underline.
- Metaspace pre-tokenizer/decoder uses replacement `▁`; `add_prefix_space=True` by default.
- Post-processor:
  - single sequence: `[CLS] A [SEP]`
  - pair: `[CLS] A [SEP] B [SEP]`
- `model_input_names = ["input_ids", "attention_mask"]`; token type IDs are still accepted by the model and default to zeros if omitted.
- Model defaults create an all-ones `attention_mask` if missing, but callers should provide masks for padded batches.
- QA path: if `question_lengths` is absent, first `[SEP]` determines question length; question tokens are masked from start/end logits except token 0.

CPU/data-pipeline work:

- Tokenization, truncation/padding policy, and special-token insertion.

GPU/runtime work:

- Embeddings, block padding, attention masks, QA logits mask, task head.

## 9. Graph rewrite / lowering opportunities

### Rewrite: Separate Q/K/V linears -> packed QKV projection

Source pattern:

```text
q = Linear(H,H)(x); k = Linear(H,H)(x); v = Linear(H,H)(x)
```

Replacement:

```text
qkv = Linear(H, 3H)(x); split last dim into q,k,v
```

Preconditions:

- Same input tensor.
- Same dtype/device/layout.
- All three projections have matching `use_bias`.

Weight transform:

```python
w_qkv = cat([w_q, w_k, w_v], dim=0)
b_qkv = cat([b_q, b_k, b_v], dim=0) if bias else None
```

Failure cases: separate quantization policies, weight aliasing, or need to preserve per-projection debug outputs.

Parity test: compare q/k/v tensors before attention for one layer in fp32 and fp16.

### Rewrite: Full attention path -> dense SDPA/FlashAttention

Preconditions:

- `attention_type == "original_full"`.
- Mask is representable as padding/causal additive mask.
- Dropout disabled for inference.

Replacement:

```text
QKV projection -> SDPA/FlashAttention -> output projection
```

Failure cases: requested dense `attentions` output, cross-attention cache edge cases, unsupported dtype.

### Rewrite: Block-sparse attention -> BigBird sparse kernel

Source pattern:

```text
block Q/K/V -> gather random blocks -> local/global/random score regions
-> per-region softmax -> sparse value matmuls -> concat blocks
```

Replacement:

```text
BigBirdSparseAttention(Q, K, V, attention_mask, block_size, random_plan, return_attn=False)
```

Preconditions:

- Encoder self-attention only.
- `S > (5 + 2*num_random_blocks)*block_size` and padded `S % block_size == 0`.
- First and last blocks are global; local window is exactly one block left/current/right.
- Eval random plan matches HF source, including zero plan behavior.

Failure cases:

- Decoder or cross-attention.
- `output_attentions=True` unless dense attention reconstruction fallback exists.
- Sequence length above `max_position_embeddings` without position-table extension.

Parity test: compare block-sparse context tensors with HF for `S=768`, `1024`, `3072`, `4096` and a non-special padded length.

### Rewrite: Omit dense sparse-attention probability reconstruction

Source pattern:

```text
attention_probs = zeros(B,H,S,S); fill sparse-visible positions with weights
```

Replacement:

```text
skip unless attentions are requested
```

Preconditions:

- Caller does not request attentions.
- Only hidden states/logits are validated.

Failure cases: APIs that expose `outputs.attentions`.

### Rewrite: FFN fusion

Source pattern:

```text
Linear(H,I) -> gelu_new -> Linear(I,H) -> dropout -> residual add -> LayerNorm
```

Replacement:

```text
CUTLASS GEMM + activation epilogue where possible; fuse residual+LayerNorm separately
```

Preconditions: static hidden/intermediate sizes, inference dropout disabled.

## 10. Kernel fusion candidates

Highest priority:

- BigBird block-sparse attention kernel. This is the family-defining operator and avoids falling back to quadratic dense attention at long context.
- Dense full-attention/SDPA fallback. Needed for short sequences, decoder experiments, and block-sparse bring-up parity.
- LayerNorm and residual+LayerNorm. Every block has post-attention and post-FFN LayerNorm.
- FFN GEMMs with `gelu_new`. These dominate non-attention compute.

Medium priority:

- Packed QKV projection and split.
- Sparse block gather/index-select optimization for random blocks.
- Optional dense attention-probability reconstruction only for `output_attentions`.
- QA head fusion: FFN-style transform plus final `Linear(H,2)`, plus question mask subtract.
- Last-token/selected-token LM logits via `logits_to_keep` for causal LM follow-up.

Lower priority:

- Tokenizer on GPU. Keep CPU pipeline first.
- NSP, multiple-choice, and classification head micro-fusions.
- Training dropout/loss paths.

## 11. Runtime staging plan

Stage 1: Parse BigBird config and load weights for `google/bigbird-roberta-base`; support embeddings, LayerNorm, Linear, `gelu_new`, tied MLM weights.

Stage 2: Run one encoder block in `original_full` with random tensors and then checkpoint weights. Validate masks and short-sequence fallback.

Stage 3: Implement full encoder in `original_full` and run MLM/pretraining logits parity for short sequences.

Stage 4: Add block padding and sparse mask construction, then run the HF block-sparse Python-equivalent path as a reference lowering if needed.

Stage 5: Replace sparse reference with a DinoML BigBird sparse attention kernel; skip dense `attention_probs` unless requested.

Stage 6: Add task heads: masked LM/pretraining, QA, sequence classification. QA needs `sep_token_id` question-length path and question logits mask.

Stage 7: Add performance rewrites: packed QKV, FFN fusions, dense fallback dispatch, sparse attention profiling.

Stage 8: Optional CausalLM/full-attention decoder with `DynamicCache`-style KV cache and `logits_to_keep`.

Can stub initially: training losses, dropout randomness, dense `attentions` outputs, multiple choice, token classification, CausalLM.

## 12. Parity and validation plan

- Config/load tests:
  - Source-default config and the four checkpoint configs above.
  - Verify `head_dim * num_heads == hidden_size`.
  - Verify tied MLM decoder weight aliases token embeddings.
- Embedding parity:
  - `input_ids` path with and without explicit `token_type_ids`.
  - `inputs_embeds` path and `rescale_embeddings=True` synthetic config.
- Full attention parity:
  - One block and N blocks at `S=128` and `S=704`, where source should use/switch to `original_full`.
  - Encoder mask with padding and 3D custom mask.
- Block-sparse parity:
  - `S=768`, `1024`, `3072`, `4096`, and non-multiple input length padded to block size.
  - Compare hidden states with `output_attentions` disabled and enabled separately.
  - Eval mode must match source random-plan behavior.
- Head parity:
  - MLM logits `[B,S,V]`.
  - QA start/end logits, including inferred `question_lengths` from first `[SEP]`.
  - Sequence classification logits from token 0.
- Tolerances:
  - fp32: `rtol=1e-4`, `atol=1e-5` for full attention; sparse may need `rtol=2e-4` because region order differs if fused.
  - fp16/bf16: start with `rtol=2e-2`, `atol=2e-2`, then tighten after kernel math order is fixed.

## 13. Performance probes

- Sequence-length sweep: `S=512`, `704`, `768`, `1024`, `2048`, `3072`, `4096`; separate fallback/full versus sparse.
- Batch sweep: `B=1`, `2`, `4`, `8` at `S=4096`.
- Sparse attention breakdown:
  - QKV projection time.
  - random block gather time.
  - local/global/random score+softmax time.
  - value aggregation time.
  - dense `attention_probs` reconstruction cost when enabled.
- Full attention baseline versus sparse attention at the fallback boundary.
- FFN GEMM throughput for base and large.
- End-to-end per-head probes: MLM logits, QA logits, sequence classification.
- Memory probes:
  - Sparse intermediates and masks.
  - Optional dense `attention_probs [B,H,S,S]`.
  - Block padding overhead for non-multiple sequence lengths.

## 14. Skip/defer list

- Training losses and gradient checkpointing.
- Dropout randomness.
- Causal LM generation and KV cache for first encoder target.
- Cross-attention / seq2seq decoder mode.
- Dense `attentions` output in optimized sparse path, except an explicit debug/parity mode.
- ETC variant and extra global tokens; source says unsupported.
- Quantization and multi-GPU/tensor parallel.
- GPU tokenizer.
- Non-primary heads: multiple choice and token classification can wait behind MLM/QA/classification.

## 15. Final implementation checklist

- [ ] Parse `BigBirdConfig`, including `attention_type`, `block_size`, `num_random_blocks`, `use_bias`, `rescale_embeddings`, and task-head fields.
- [ ] Load embeddings, encoder blocks, and tied MLM/pretraining weights without breaking aliases.
- [ ] Implement `gelu_new` activation parity.
- [ ] Implement full MHA path with padding/causal/cross masks for fallback.
- [ ] Implement block padding and sparse mask construction.
- [ ] Implement BigBird block-sparse attention context path with global/sliding/random blocks.
- [ ] Add optional dense attention-probability reconstruction only for requested attentions.
- [ ] Add runtime dispatch: short sequence or decoder/cross-attn -> `original_full`; long encoder -> block sparse.
- [ ] Add MLM/pretraining head parity.
- [ ] Add QA head and `question_lengths`/`sep_token_id` masking parity.
- [ ] Add sequence classification head parity.
- [ ] Add packed QKV rewrite with weight transform tests.
- [ ] Add sparse attention performance probes across sequence and batch sweeps.
- [ ] Benchmark dense fallback versus sparse kernel at `S=704/768/1024/4096`.
