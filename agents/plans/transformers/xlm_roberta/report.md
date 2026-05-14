# XLM-RoBERTa Transformers Audit

## 1. Source basis

```text
Transformers commit/version:
  Local checkout X:/H/transformers at b75feb2af64c3e29cbbc1bd859958c5432cc7ed4.

Model id:
  Primary: FacebookAI/xlm-roberta-base.
  Representative: FacebookAI/xlm-roberta-large,
  cardiffnlp/twitter-xlm-roberta-base, joeddav/xlm-roberta-large-xnli,
  Davlan/xlm-roberta-base-ner-hrl.

Config source:
  https://huggingface.co/FacebookAI/xlm-roberta-base/raw/main/config.json
  https://huggingface.co/FacebookAI/xlm-roberta-large/raw/main/config.json
  https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base/raw/main/config.json
  https://huggingface.co/joeddav/xlm-roberta-large-xnli/raw/main/config.json
  https://huggingface.co/Davlan/xlm-roberta-base-ner-hrl/raw/main/config.json
  Snapshots are under agents/plans/transformers/xlm_roberta/_sources/.

Source files inspected:
  X:/H/transformers/src/transformers/models/xlm_roberta/configuration_xlm_roberta.py
  X:/H/transformers/src/transformers/models/xlm_roberta/modeling_xlm_roberta.py
  X:/H/transformers/src/transformers/models/xlm_roberta/modular_xlm_roberta.py
  X:/H/transformers/src/transformers/models/xlm_roberta/tokenization_xlm_roberta.py
  Comparison: RoBERTa and CamemBERT modeling/config/tokenizer files at same commit.

Any missing files or assumptions:
  modeling_xlm_roberta.py is generated from modular_xlm_roberta.py. Runtime
  behavior was audited in the generated file, but upstream source edits belong
  in the modular file. [FacebookAI/xlm-roberta-xl](https://huggingface.co/FacebookAI/xlm-roberta-xl)
  and [FacebookAI/xlm-roberta-xxl](https://huggingface.co/FacebookAI/xlm-roberta-xxl)
  raw files returned 401 Unauthorized.
  Primary DinoML target is encoder inference plus XLMRobertaForMaskedLM.
  XLMRobertaForCausalLM exists but is a non-primary decoder/CLM branch.
  No DinoML tests were run, per task scope.
```

## 2. High-level architecture

XLM-RoBERTa is a text-only multilingual RoBERTa-style bidirectional encoder. The useful first target is masked-LM or encoder-feature inference with SentencePiece/Unigram tokenization, learned token embeddings, learned absolute positions with padding-aware offsets, a normally all-zero token-type embedding, post-residual LayerNorm encoder blocks, and a tied large-vocab LM head.

```text
SentencePiece/Unigram tokenizer -> input_ids/attention_mask
-> word + token_type + learned absolute position embeddings
-> embedding LayerNorm
-> N bidirectional encoder layers
-> MLM transform + tied vocab projection
-> logits [B, S, V]
```

CPU/data-pipeline work owns tokenizer normalization, Unigram segmentation, special token insertion, padding, and attention masks. GPU/runtime work starts at embedding gather and can optionally accept precomputed position ids.

## 3. Important config dimensions

Source defaults from `XLMRobertaConfig` are RoBERTa-like but not production-like for XLM-R: `vocab_size=30522`, `max_position_embeddings=512`, `type_vocab_size=2`, and `layer_norm_eps=1e-12`. Public XLM-R configs inspected override these to the multilingual shape below.

| Field | Base checkpoint | Large checkpoint | Runtime meaning |
|---|---:|---:|---|
| `vocab_size` | 250002 | 250002 | Multilingual SentencePiece vocabulary and LM projection width |
| `hidden_size` | 768 | 1024 | Encoder width `H` |
| `num_hidden_layers` | 12 | 24 | Encoder block count |
| `num_attention_heads` | 12 | 16 | MHA heads |
| `head_dim` | 64 | 64 | Inferred as `H / heads` |
| `intermediate_size` | 3072 | 4096 | FFN expansion |
| `hidden_act` | `gelu` | `gelu` | FFN and MLM transform activation |
| `max_position_embeddings` | 514 | 514 | 512 real tokens plus pad offset rows |
| `type_vocab_size` | 1 | 1 | Token type ids must normally be all zero |
| `layer_norm_eps` | `1e-5` | `1e-5` | Production LayerNorm epsilon |
| `pad/bos/eos` ids | `1/0/2` | `1/0/2` | Padding and special-token ids |
| `use_cache` | true in config | true in config | Forced off for non-decoder encoder path |
| `tie_word_embeddings` | true by default | true by default | LM decoder weight aliases word embeddings |

Representative checkpoint sweep:

| Model | Architecture | Layers | H | Heads | D | FFN | Vocab | Max pos | Type vocab | LN eps | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `FacebookAI/xlm-roberta-base` | `XLMRobertaForMaskedLM` | 12 | 768 | 12 | 64 | 3072 | 250002 | 514 | 1 | `1e-5` | Primary fill-mask checkpoint |
| `FacebookAI/xlm-roberta-large` | `XLMRobertaForMaskedLM` | 24 | 1024 | 16 | 64 | 4096 | 250002 | 514 | 1 | `1e-5` | Larger encoder |
| `cardiffnlp/twitter-xlm-roberta-base` | `XLMRobertaForMaskedLM` | 12 | 768 | 12 | 64 | 3072 | 250002 | 514 | 1 | `1e-5` | Domain-adapted Twitter MLM |
| `joeddav/xlm-roberta-large-xnli` | `XLMRobertaForSequenceClassification` | 24 | 1024 | 16 | 64 | 4096 | 250002 | 514 | 1 | `1e-5` | XNLI classifier, 3 labels |
| `Davlan/xlm-roberta-base-ner-hrl` | `XLMRobertaForTokenClassification` | 12 | 768 | 12 | 64 | 3072 | 250002 | 514 | 1 | `1e-5` | NER head, 9 labels |

Large-vocab cost is a first-order runtime issue. Base word embeddings and tied LM decoder contain about `250002 * 768 = 192M` weights; large uses about `256M`. Full MLM logits cost scales as `[B,S,250002]`, so fill-mask serving should eventually support masked-position-only logits.

## 3a. Family variation traps

- The generated modeling body is essentially RoBERTa with XLM-R names and config class. The family-specific runtime differences are mainly tokenizer, vocabulary, config defaults, and checkpoint values.
- Production configs use `type_vocab_size=1`, while the source class default says `2`. Pair sentence information is represented by `<s> A </s></s> B </s>` and attention masks, not segment embeddings. Nonzero token type ids are out of range for common checkpoints.
- Position ids are RoBERTa/fairseq-style. With `pad_token_id=1`, pad tokens keep position `1`; the first non-pad token is position `2`; a 512-token sequence reaches `513`, requiring `max_position_embeddings=514`.
- `XLMRobertaTokenizer.model_input_names` is only `["input_ids", "attention_mask"]`. The model accepts `token_type_ids`, but tokenizer output normally omits them.
- The tokenizer is multilingual SentencePiece/Unigram through `tokenizers.models.Unigram`, not RoBERTa byte-level BPE. It uses `WhitespaceSplit` followed by Metaspace and may load a SentencePiece precompiled charsmap.
- Compared with CamemBERT, XLM-R has a much larger vocabulary, no CamemBERT unused special-token defaults, and a different tokenizer normalizer/pre-tokenizer sequence. Both production families usually use `type_vocab_size=1`, `max_position_embeddings=514`, and `layer_norm_eps=1e-5`.
- Compared with RoBERTa, XLM-R has the same encoder math but different tokenizer and common vocab (`250002` vs RoBERTa `50265`). RoBERTa uses byte-level BPE; XLM-R uses Unigram/SentencePiece-style metaspace.
- `use_cache=true` appears in configs, but `XLMRobertaModel.forward` forces `use_cache=False` whenever `config.is_decoder` is false. Do not require KV cache for the primary encoder/MLM target.
- `XLMRobertaForCausalLM` can be constructed with `is_decoder=true`, `DynamicCache`, `EncoderDecoderCache`, causal masks, and `logits_to_keep`; this is optional and separate from the ordinary XLM-R encoder.
- [FacebookAI/xlm-roberta-xl](https://huggingface.co/FacebookAI/xlm-roberta-xl) and [FacebookAI/xlm-roberta-xxl](https://huggingface.co/FacebookAI/xlm-roberta-xxl) were not inspected because raw files returned 401. Access could reveal larger operator-significant dimensions, but not a different source implementation unless their configs use remote code or nonstandard fields.

## 4. Operator coverage checklist

Tensor/layout ops:

- Embedding gathers: `input_ids [B,S] -> [B,S,H]`, `position_ids [B,S] -> [B,S,H]`, optional `token_type_ids [B,S] -> [B,S,H]`.
- Position id generation: `ne(input_ids, pad_id)`, int cast, `cumsum(dim=1)`, multiply by mask, scalar add.
- Elementwise embedding sum, residual add, dropout-as-noop in inference.
- Reshape/view/transpose/contiguous for attention: `[B,S,H] -> [B,A,S,D]` and back.
- Bidirectional attention mask creation through Transformers `create_bidirectional_mask`.
- Optional head indexing: first-token select for sequence classification, flatten/unflatten for multiple choice, split/squeeze for QA.

Neural network primitives:

- LayerNorm over last dimension with checkpoint `eps=1e-5`.
- Bias Linear/GEMM for Q/K/V/O, FFN up/down, LM transform, LM decoder, pooler/classifier/task heads.
- GELU for FFN and LM transform; tanh for pooler and sequence classification head.
- Tied LM output projection: `lm_head.decoder.weight` aliases `roberta.embeddings.word_embeddings.weight`; `lm_head.bias` is a separate `[V]` parameter tied to decoder bias key.

Attention primitives:

- Bidirectional self-attention, MHA only; no GQA/MQA.
- Eager order: `q @ k^T * D^-0.5`, add mask, softmax over key dimension, dropout in training, `attn @ v`.
- Source dispatches through `ALL_ATTENTION_FUNCTIONS` and advertises FlashAttention, SDPA, and Flex attention support, but the fallback is ordinary dense attention.

Position/rotary/relative-bias ops:

- Learned absolute position embedding only.
- No RoPE, M-RoPE, ALiBi, relative bias, sliding window, block sparse attention, or convolutional positional embedding in the primary source path.

Generation/cache ops:

- None required for encoder/MLM.
- Optional decoder branch uses full MHA K/V tensors `[B,A,T,D]` in `DynamicCache`; cross-attention uses `EncoderDecoderCache`.

Preprocessing-coupled ops:

- SentencePiece/Unigram tokenizer with `sentencepiece.bpe.model` or `tokenizer.json`.
- Special token layout: single `<s> A </s>`; pair `<s> A </s> </s> B </s>`.
- Metaspace replacement `U+2581`, default `add_prefix_space=True`.

## 5. Layer/block breakdown

Embedding block:

```text
input_ids: [B,S]
mask = input_ids != pad_token_id
position_ids = cumsum(mask, dim=1) * mask + pad_token_id
token_type_ids = zeros([B,S]) unless explicitly supplied

x = word_embedding[input_ids]             # [B,S,H], V=250002 common
x += token_type_embedding[token_type_ids] # common table [1,H]
x += position_embedding[position_ids]     # common table [514,H]
x = LayerNorm(x, eps=config.layer_norm_eps)
```

Encoder block, repeated `N` times:

```text
q = Linear(H -> H, bias=True)(x).view(B,S,A,D).transpose(1,2)
k = Linear(H -> H, bias=True)(x).view(B,S,A,D).transpose(1,2)
v = Linear(H -> H, bias=True)(x).view(B,S,A,D).transpose(1,2)

ctx = Attention(q, k, v, bidirectional_padding_mask)
ctx = ctx.transpose(1,2).reshape(B,S,H)
x = LayerNorm(Linear(H -> H, bias=True)(ctx) + x)

ff = GELU(Linear(H -> I, bias=True)(x))
x = LayerNorm(Linear(I -> H, bias=True)(ff) + x)
```

Masked-LM head:

```text
h = Linear(H -> H, bias=True)(x)
h = GELU(h)
h = LayerNorm(h, eps=config.layer_norm_eps)
logits = Linear(H -> V, bias=True, tied_weight=word_embeddings)(h)
```

Concrete production shapes:

- Base: Q/K/V/O `768 -> 768`, FFN `768 -> 3072 -> 768`, MLM decoder `768 -> 250002`.
- Large: Q/K/V/O `1024 -> 1024`, FFN `1024 -> 4096 -> 1024`, MLM decoder `1024 -> 250002`.

## 6. Attention requirements

Primary target:

- Noncausal bidirectional self-attention.
- Self-attention only; cross-attention only exists for decoder-mode variants.
- MHA with `num_key_value_heads == num_attention_heads`; no repeat-KV path.
- Head dim is inferred as `hidden_size / num_attention_heads`; source rejects non-divisible hidden size unless an `embedding_size` escape hatch is present.
- Masking is additive after score scaling. Fully unpadded batches can skip materializing a dense mask only if parity with `create_bidirectional_mask` is preserved.
- No packed/varlen source ABI, no local/sparse attention, no RoPE/ALiBi/relative bias.
- FlashAttention/SDPA compatibility is straightforward dense bidirectional attention for inference, but full `[B,A,S,S]` attention output is still needed if `output_attentions=True` is admitted.

Optional decoder/causal branch:

- `XLMRobertaModel.forward` forces `use_cache=False` when `is_decoder=false`.
- If `is_decoder=true` and `use_cache`, source creates `DynamicCache`; with encoder states it creates `EncoderDecoderCache`.
- Self-attention caches projected K/V after linear projection and before attention, shape `[B,A,T,D]`.
- Cross-attention caches encoder-projected K/V once and tracks `is_updated[layer_idx]`.
- This branch should not block first encoder/MLM integration.

## 7. Position encoding and custom math

XLM-R uses learned absolute positions with RoBERTa-style padding offsets:

```python
def xlm_roberta_position_ids(input_ids, padding_idx=1, past_len=0):
    mask = (input_ids != padding_idx).int()
    incremental = (cumsum(mask, dim=1) + past_len) * mask
    return incremental.long() + padding_idx
```

For common checkpoints, pads have position `1`; first real token has position `2`; a full 512-token non-pad sequence reaches row `513`. If `inputs_embeds` are supplied, source cannot infer padding and creates sequential ids from `padding_idx + 1` through `padding_idx + S`.

Precomputable:

- Position ids can be CPU/data-pipeline output for a fixed padded batch.
- All-zero token type ids can be omitted or represented as a guarded constant for `type_vocab_size=1`.

Dynamic:

- Left/right padding changes position ids; they must be derived from `input_ids != pad_id`, not plain `arange(S)`.
- Decoder-mode `past_key_values_length` offsets positions, but that is outside the primary target.

## 8. Preprocessing and input packing

CPU/data-pipeline:

- `XLMRobertaTokenizer` uses `Unigram` from `tokenizers` and loads `sentencepiece.bpe.model` or `tokenizer.json`.
- Default special tokens are `<s>`, `</s>`, `<unk>`, `<pad>`, and `<mask>`.
- Source tokenizer can install a precompiled SentencePiece charsmap if supplied by tokenizer JSON.
- Pretokenizer is `WhitespaceSplit` followed by Metaspace with replacement `U+2581` and `prepend_scheme="always"` by default.
- Pair inputs use a double separator: `<s> A </s> </s> B </s>`.

GPU/runtime:

- Required tensors: `input_ids [B,S]`, `attention_mask [B,S]`.
- Optional tensors: `position_ids [B,S]`, `inputs_embeds [B,S,H]`, `token_type_ids [B,S]`.
- For production checkpoints, validate `token_type_ids` are omitted or all zeros because `type_vocab_size=1`.
- Enforce tokenizer `model_max_length=512` or at least `max(position_ids) < max_position_embeddings`.

## 9. Graph rewrite / lowering opportunities

### Rewrite: all-zero token type embedding fold

Source pattern:

```text
token_type_ids omitted -> gather registered zero buffer -> token_type_embedding[0]
```

Replacement:

```text
broadcast token_type_embedding[0] over [B,S,H], or fold into embedding-sum kernel
```

Preconditions:

- `type_vocab_size == 1`, or caller guarantees every token type id is zero.
- Position ids are valid because source gathers the zero buffer using `position_ids`.

Failure cases:

- Source-default or debug configs with `type_vocab_size > 1`.
- Caller supplies pair segment ids as `1` for production XLM-R; this is out of range and should fail rather than silently behave like BERT.

Parity test sketch:

- Compare omitted `token_type_ids` and explicit zeros for base and large.

### Rewrite: position ids outside graph

Source pattern:

```text
cumsum(input_ids != pad_id, dim=1) * mask + pad_id
```

Replacement:

```text
CPU/tokenizer-side position_ids -> position embedding gather
```

Preconditions:

- Graph boundary includes `position_ids`.
- Data pipeline uses checkpoint `pad_token_id`.

Failure cases:

- `inputs_embeds` path uses sequential ids without pad detection.
- Decoder mode adds `past_key_values_length`.

Parity test sketch:

- Right-padded and left-padded batches: pads stay `1`; first non-pad token is `2`.

### Rewrite: QKV linears -> packed self-attention projection

Source pattern:

```text
q = Linear(H,H)(x); k = Linear(H,H)(x); v = Linear(H,H)(x)
```

Replacement:

```text
Linear(H,3H) -> split [q,k,v]
```

Weight transform:

```python
w_qkv = concat([w_q, w_k, w_v], axis=0)
b_qkv = concat([b_q, b_k, b_v], axis=0)
```

Preconditions:

- Self-attention only; same hidden input feeds Q/K/V.
- Bias is present for all three projections.

Failure cases:

- Cross-attention computes Q from decoder states and K/V from encoder states.

Parity test sketch:

- Compare q/k/v tensors before reshape for random block weights.

### Rewrite: full MLM logits -> masked-position logits

Source pattern:

```text
MLMHead(sequence_output) -> [B,S,250002]
```

Replacement:

```text
Gather masked hidden states -> MLMHead -> [num_masked,250002]
```

Preconditions:

- Serving API only needs fill-mask positions and accepts changed output shape.

Failure cases:

- Hugging Face `XLMRobertaForMaskedLM` parity requires full `[B,S,V]` logits.

Parity test sketch:

- Compare gathered HF full logits at mask positions against optimized head output.

## 10. Kernel fusion candidates

Highest priority:

- Large-vocab LM projection and masked-position-only logits. The `V=250002` decoder dominates fill-mask output cost.
- Bias GEMM coverage for Q/K/V/O, FFN, LM transform, and task heads.
- Embedding sum + LayerNorm, preserving padding-aware position ids and all-zero token-type behavior.
- Packed QKV projection plus attention layout handling.
- Bidirectional MHA with padding-mask fast path.

Medium priority:

- Residual add + LayerNorm fusion after attention and FFN.
- GEMM + GELU for FFN up projection and LM transform.
- Position-id generation kernel if position ids are not supplied by CPU pipeline.
- Attention mask creation/skip path for all-ones masks.

Lower priority:

- Pooler, sequence classification, token classification, QA, and multiple-choice convenience heads.
- Optional decoder cache/CausalLM branch.
- Output-attention materialization on optimized attention backends.

## 11. Runtime staging plan

Stage 1: config/tokenizer/weights

- Parse checkpoint config, not source defaults.
- Load tokenizer metadata needed for special tokens, pad id, max length, and Unigram assets.
- Load word, position, token-type, encoder, and LM-head weights; preserve tied decoder/embedding alias.

Stage 2: embedding parity

- Accept `input_ids`, `attention_mask`, and precomputed or generated `position_ids`.
- Fold all-zero token types only behind `type_vocab_size=1` guards.

Stage 3: one encoder block

- Implement post-norm MHA and FFN for `[B,S,H]`.
- Validate base and large geometries.

Stage 4: full encoder

- Run all layers with eager-equivalent bidirectional padding mask.

Stage 5: masked LM

- Implement full `[B,S,250002]` logits for HF parity.
- Add masked-position-only API/rewrite as the practical serving path.

Stage 6: optional task heads

- Token classification and sequence classification first, because representative public checkpoints use them.

Stage 7: optional decoder

- Admit `XLMRobertaForCausalLM`, causal masks, and caches only as a separate target.

## 12. Parity and validation plan

- Config parsing tests: source defaults vs production checkpoint values, especially vocab, max positions, token types, and eps.
- Tokenizer coupling tests: special-token layout, pair double separator, Metaspace leading-space behavior, and mask token.
- Position-id tests for left/right padded batches and `inputs_embeds` path.
- Embedding parity against `FacebookAI/xlm-roberta-base`.
- One-layer parity with no padding and mixed padding.
- Full encoder parity for base and large at short sequence lengths, then `S=512`.
- MLM head parity for full `[B,S,V]` logits.
- Masked-position-only rewrite parity against gathered full logits.
- Token-classification parity for `Davlan/xlm-roberta-base-ner-hrl` if task heads are admitted.
- Sequence-classification parity for `joeddav/xlm-roberta-large-xnli` if classifier heads are admitted.

Suggested tolerances:

- fp32: `atol=1e-4`, `rtol=1e-4` end-to-end; tighter for isolated GEMMs and LayerNorm.
- fp16/bf16: start around `atol=2e-2`, `rtol=2e-2` end-to-end, then tighten per kernel.

## 13. Performance probes

- Tokenizer throughput for multilingual Unigram plus special-token insertion.
- Position-id generation in graph vs CPU precompute.
- Encoder throughput sweep for base and large: `B in {1,8,32}`, `S in {16,64,128,512}`.
- Attention backend comparison: eager additive mask, SDPA/Flash-compatible mask, unpadded fast path.
- FFN GEMM/GELU time for base vs large.
- LayerNorm/residual bandwidth probes.
- Full MLM logits `[B,S,250002]` vs masked-position-only logits.
- Embedding table and LM decoder memory residency: fp32/fp16/bf16, dense and future quantized loading policies.
- Task-head overhead for token/sequence classification over encoder-only runtime.

## 14. Skip/defer list

Safe to defer for first encoder/MLM integration:

- Training losses, dropout randomness, gradient checkpointing, and chunked feed-forward.
- `XLMRobertaForCausalLM`, decoder cache, cross-attention, beam search, and generation helpers.
- Sequence classification, token classification, QA, and multiple choice unless a target checkpoint requires the head.
- Optimized-backend exact `output_attentions=True` parity.
- XL/XXL gated config coverage until access is available.
- Quantization and tensor parallelism.

## 15. Final implementation checklist

- [ ] Parse `XLMRobertaConfig` from checkpoint and reject source-default mismatch hazards.
- [ ] Load tokenizer metadata/assets for Unigram, special tokens, `pad_token_id`, and max length.
- [ ] Load/tie word embedding and LM decoder weights for `V=250002`.
- [ ] Implement RoBERTa-style padding-aware position ids or accept precomputed `position_ids`.
- [ ] Implement word, position, and token-type embedding lookup plus embedding LayerNorm.
- [ ] Add all-zero token-type fast path for `type_vocab_size=1`.
- [ ] Implement post-norm encoder block with bias Q/K/V/O linears, bidirectional MHA, residuals, LayerNorm, and GELU FFN.
- [ ] Implement full MLM head logits for `V=250002`.
- [ ] Add masked-position-only LM-head rewrite/API.
- [ ] Add packed QKV rewrite with self-attention-only guard.
- [ ] Add fused residual + LayerNorm candidates.
- [ ] Add position-id and tokenizer-coupling parity tests.
- [ ] Add one-block, full-encoder, and MLM parity tests against HF.
- [ ] Add optional token/sequence classification head parity.
- [ ] Benchmark tokenizer, encoder, attention, FFN, LayerNorm, and LM head separately.
