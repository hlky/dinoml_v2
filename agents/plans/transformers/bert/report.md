# Transformers BERT Operator and Integration Report

## 1. Source basis

```text
Transformers commit/version:
  Local checkout X:/H/transformers at b75feb2af64c3e29cbbc1bd859958c5432cc7ed4.
  Remote upstream: https://github.com/huggingface/transformers.git

Model id:
  Primary worked example: google-bert/bert-base-uncased.
  Additional sizing references: google-bert/bert-large-uncased,
  google-bert/bert-base-cased, google-bert/bert-large-cased,
  google-bert/bert-base-multilingual-cased, emilyalsentzer/Bio_ClinicalBERT.

Config source:
  https://huggingface.co/google-bert/bert-base-uncased/raw/main/config.json
  Additional configs fetched from Hugging Face model repos listed above.
  HF plugin metadata confirmed common BERT repos as transformers `bert`,
  mostly AutoModelForMaskedLM/fill-mask; Bio_ClinicalBERT advertises AutoModel.

Source files inspected:
  X:/H/transformers/src/transformers/models/bert/modeling_bert.py
  X:/H/transformers/src/transformers/models/bert/configuration_bert.py
  X:/H/transformers/src/transformers/models/bert/tokenization_bert.py
  X:/H/transformers/src/transformers/models/bert/tokenization_bert_legacy.py was
  present but not needed for operator coverage.

Any missing files or assumptions:
  No remote-code files are required for standard BERT. This report prioritizes
  encoder inference and masked-LM/fill-mask because that is the common base
  checkpoint task. Classification, token classification, QA, NSP, and decoder
  mode are documented as staged variants.
```

## 2. High-level architecture

BERT is a text-only encoder transformer. The standard path is bidirectional self-attention with learned absolute position embeddings, token type embeddings, post-attention/post-MLP LayerNorm, and optional task heads. Unlike Llama/Mistral, standard BERT is not a generation-first architecture and does not require KV cache for encoder inference.

```text
WordPiece tokenization + [CLS]/[SEP]/token_type_ids
  -> word + token_type + absolute position embeddings
  -> LayerNorm/dropout
  -> bidirectional encoder stack
  -> optional pooler
  -> masked-LM / classifier / QA / token head
```

## 3. Important config dimensions

Worked example: `google-bert/bert-base-uncased`.

| Field | BERT base uncased value | Source |
|---|---:|---|
| architecture | BertForMaskedLM | HF repo metadata |
| vocab_size / V | 30522 | config.json |
| hidden_size / H | 768 | config.json |
| num_hidden_layers | 12 | config.json |
| num_attention_heads / A | 12 | config.json |
| head_dim / D | 64 | inferred from `H/A` |
| intermediate_size / I | 3072 | config.json |
| hidden_act | gelu | config.json |
| max_position_embeddings | 512 | config.json |
| type_vocab_size | 2 | config.json |
| layer_norm_eps | 1e-12 | config.json |
| position_embedding_type | absolute | config or source default |
| is_decoder | false inferred | BertConfig default when omitted |
| add_cross_attention | false inferred | BertConfig default when omitted |
| tie_word_embeddings | true inferred | BertConfig default when omitted |
| cache support | disabled for encoder | source sets `use_cache=False` when not decoder |

Representative checkpoint sweep:

| Checkpoint | H | I | layers | A | D | V | max pos | type vocab | task/model metadata |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| google-bert/bert-base-uncased | 768 | 3072 | 12 | 12 | 64 | 30522 | 512 | 2 | fill-mask, AutoModelForMaskedLM |
| google-bert/bert-large-uncased | 1024 | 4096 | 24 | 16 | 64 | 30522 | 512 | 2 | fill-mask |
| google-bert/bert-base-cased | 768 | 3072 | 12 | 12 | 64 | 28996 | 512 | 2 | fill-mask |
| google-bert/bert-large-cased | 1024 | 4096 | 24 | 16 | 64 | 28996 | 512 | 2 | fill-mask |
| google-bert/bert-base-multilingual-cased | 768 | 3072 | 12 | 12 | 64 | 119547 | 512 | 2 | fill-mask, multilingual |
| emilyalsentzer/Bio_ClinicalBERT | 768 | 3072 | 12 | 12 | 64 | 28996 | 512 | 2 | AutoModel metadata |

## 3a. Family variation traps

- BERT has many heads/tasks over the same encoder. A Dinoml target must choose whether it needs only `BertModel`, masked LM, sequence classification, token classification, QA, NSP, or decoder mode.
- Standard BERT uses post-LayerNorm after residuals, not Llama-style pre-RMSNorm.
- Attention projections and dense layers include bias by default.
- Token type embeddings are part of the model graph, not only tokenizer metadata.
- Multilingual BERT keeps the same hidden geometry but has a much larger vocab/LM head.
- `BertConfig` supports decoder/cross-attention/cache paths, but common BERT checkpoints are encoder-only and source disables cache for that path.

## 4. Operator coverage checklist

### Tensor/layout ops

- Word embedding gather: `[B,S] -> [B,S,H]`.
- Token type embedding gather: `[B,S] -> [B,S,H]`, defaulting to zeros when not supplied.
- Position embedding gather: position IDs `[1,S]` or `[B,S] -> [*,S,H]`.
- Elementwise embedding sum: word + token type + position.
- Attention reshape/transpose: Q/K/V `[B,S,H] -> [B,A,S,D]`.
- Dense mask broadcasting to attention score shape `[B,A,S,S]`.
- Pooler first-token select: `hidden[:,0]`.
- MLM head transform and tied decoder projection.

### Neural network primitives

- Embedding tables: word `[V,H]`, position `[max_pos,H]`, token type `[type_vocab,H]`.
- LayerNorm with mean subtraction, variance, weight, and bias.
- Linear with bias:
  - BERT base Q/K/V/O: `Linear(768 -> 768, bias=True)`.
  - FFN: `Linear(768 -> 3072, bias=True)`, GELU, `Linear(3072 -> 768, bias=True)`.
  - Pooler: `Linear(768 -> 768)`, tanh.
  - MLM transform: `Linear(768 -> 768)`, GELU, LayerNorm.
  - MLM decoder: `Linear(768 -> 30522, bias=True)` with tied decoder weight plus output bias.
- Residual adds before LayerNorm in attention output and FFN output.
- GELU activation; tanh for pooler.
- Dropout is inference no-op.

### Attention primitives

- Standard encoder bidirectional MHA.
- Source can use SDPA through `ALL_ATTENTION_FUNCTIONS`, with eager fallback matmul + additive mask + softmax + matmul.
- No GQA/MQA in standard BERT.
- Optional decoder/cross-attention path exists, but not required for primary encoder/masked-LM integration.

### Position/rotary/relative-bias ops

- Learned absolute position embeddings only for common BERT configs.
- `position_ids` buffer is serialized/exported as `[1,max_position_embeddings]`.
- Configs may include `position_embedding_type="absolute"`; source snippets inspected do not require RoPE.

### Generation/cache ops

- Primary encoder path does not use cache.
- If `config.is_decoder=True`, source can allocate `DynamicCache` or `EncoderDecoderCache`, but this is a non-primary BERT variant.

### Preprocessing-coupled ops

- WordPiece tokenizer with BERT normalization, optional lowercasing, Chinese char handling, and `##` continuation pieces.
- Post-processor builds `[CLS] A [SEP]` and `[CLS] A [SEP] B [SEP]`.
- Token type IDs distinguish segment A/B and must align with tokenizer post-processing.

### Distributed/tensor-parallel ops

- No BERT-specific TP plan in inspected config. Large/vocab-heavy MLM heads may still benefit from sharded LM projection.

## 5. Layer/block breakdown

Embedding path:

```text
word = Embedding(V,H)(input_ids)
tok = Embedding(type_vocab,H)(token_type_ids or zeros)
pos = Embedding(max_pos,H)(position_ids)
x = LayerNorm(word + tok + pos)
```

Encoder block, repeated `N` times:

```text
q = Linear(H -> H, bias=True)(x) -> [B,A,S,D]
k = Linear(H -> H, bias=True)(x) -> [B,A,S,D]
v = Linear(H -> H, bias=True)(x) -> [B,A,S,D]
attn = Attention(q,k,v, bidirectional_mask)
y = Linear(H -> H, bias=True)(attn)
x = LayerNorm(y + x)

z = Linear(H -> I, bias=True)(x)
z = GELU(z)
z = Linear(I -> H, bias=True)(z)
x = LayerNorm(z + x)
```

Common heads:

```text
pooler = tanh(Linear(H -> H)(x[:,0]))
mlm = LayerNorm(GELU(Linear(H -> H)(x)))
logits = Linear(H -> V, tied_weight=word_embeddings, bias=True)(mlm)
classification_logits = Linear(H -> num_labels)(pooler)
qa_logits = Linear(H -> 2)(x)
```

## 6. Attention requirements

- Standard path: noncausal, bidirectional self-attention.
- MHA only: `A=12`/`16`, `D=64`, `KvH=A`.
- Masking style: additive attention mask broadcastable over heads and query positions.
- No sliding-window/local attention, RoPE, ALiBi, or relative bias required for common BERT checkpoints.
- SDPA compatibility: straightforward noncausal MHA with additive mask. Eager fallback is matmul-softmax-matmul and is likely acceptable for small batches but should be optimized for high-throughput encoder serving.
- Cache layout is not required for the primary encoder path. Decoder-mode BERT would cache `[B,A,past,D]` after K/V projection, but that should be a separate target.

## 7. Position encoding and custom math

BERT common checkpoints use learned absolute position embeddings.

```python
def bert_embeddings(input_ids, token_type_ids, position_ids):
    word = gather(word_embedding, input_ids)
    tok = gather(token_type_embedding, token_type_ids)
    pos = gather(position_embedding, position_ids)
    return layer_norm(word + tok + pos)
```

Precompute:

- Default position IDs `[0..S-1]`.
- Default token type IDs all zeros for single-sentence inputs.

Dynamic:

- Pair inputs require tokenizer-produced token type IDs.
- Decoder-mode past length offsets position IDs, but primary encoder path does not.

## 8. Preprocessing and input packing

CPU/data-pipeline work:

- WordPiece tokenization, normalization/lowercasing, special token insertion.
- Segment/token type ID construction.
- Padding/truncation to max 512.

GPU/runtime work:

- Accept `input_ids[B,S]`, `attention_mask[B,S]`, optional `token_type_ids[B,S]`, optional `position_ids`.
- Build/additive attention mask.
- For fill-mask, compute logits for all tokens or gather masked positions before/after MLM head as an optimization.

## 9. Graph rewrite / lowering opportunities

### Rewrite: embedding triple-sum -> fused embedding add

Preconditions:

- Inputs are word IDs, token type IDs, and position IDs.
- All embedding tables produce same `[B,S,H]` dtype/layout.

Replacement:

```text
GatherWord + GatherTokenType + GatherPosition -> FusedEmbeddingSum -> LayerNorm
```

Failure cases:

- Missing token_type_ids must use zero segment IDs exactly.
- Position IDs can be caller-provided.

Parity test sketch:

- Compare single and pair tokenized inputs with explicit and default token_type_ids.

### Rewrite: Linear with bias -> GEMM_RCR_Bias

Preconditions:

- Source is `nn.Linear(..., bias=True)`.
- Input is dense row-major after flattening leading dims.

Replacement:

```text
FlattenLeadingDims -> GEMM_RCR_Bias -> Reshape
```

Failure cases:

- Tied MLM decoder weight requires preserving alias with word embeddings.

Parity test sketch:

- Compare Q/K/V/O, FFN, pooler, MLM transform, and decoder projections.

### Rewrite: QKV projections -> packed QKV

Preconditions:

- Same input feeds Q/K/V.
- All projections have equal output width `H` and bias.

Replacement:

```text
PackedLinear(H -> 3H, bias=True) -> Split(q,k,v)
```

Weight transform:

```python
w_qkv = concat([w_q, w_k, w_v], axis=0)
b_qkv = concat([b_q, b_k, b_v], axis=0)
```

Failure cases:

- Cross-attention/decoder variants need separate handling.

Parity test sketch:

- Compare split q/k/v against independent projections.

### Rewrite: masked-LM positions only

Preconditions:

- Caller only needs logits for known mask positions.
- No loss/full-token scoring requirement.

Replacement:

```text
GatherHidden(mask_positions) -> MLMTransform -> DecoderGEMM
```

Failure cases:

- Standard `BertForMaskedLM.forward` returns `[B,S,V]`; changing output shape is only safe behind an explicit optimized API.

Parity test sketch:

- Compare gathered logits to full logits indexed at mask positions.

## 10. Kernel fusion candidates

Highest priority:

- LayerNorm with residual add: BERT uses post-LN after attention and FFN plus embedding LN.
- Bias GEMM coverage: nearly all BERT Linear layers have bias.
- Packed QKV projection with bias.
- GELU activation fusion in FFN and MLM transform.
- Masked-position-only MLM head for fill-mask serving.

Medium priority:

- Embedding sum + LayerNorm fusion.
- SDPA/noncausal attention backend.
- Pooler first-token select + dense + tanh.
- Large-vocab MLM projection for multilingual BERT.

Lower priority:

- Decoder/cache/cross-attention mode.
- NSP and miscellaneous task heads unless targeted.

## 11. Runtime staging plan

Stage 1: Parse BertConfig and load embeddings/encoder weights.

Stage 2: Implement embedding path parity, including token_type defaults.

Stage 3: Implement one encoder block parity with bias GEMMs and post-LayerNorm.

Stage 4: Full encoder parity for BERT base.

Stage 5: Add masked-LM head parity with tied word embedding decoder.

Stage 6: Add selected downstream heads as needed: sequence classification, token classification, QA.

Stage 7: Add packed QKV, fused LayerNorm/residual, and masked-position-only logits.

Stage 8: Scale to BERT large and multilingual vocab.

## 12. Parity and validation plan

- Embedding path parity with default and explicit token type IDs.
- LayerNorm parity with eps `1e-12`.
- Single attention parity for bidirectional mask.
- Single encoder layer parity.
- Full encoder last-hidden-state and pooler parity.
- Masked-LM logits parity for base and large.
- Masked-position-only logits parity against full logits.
- Sequence classification/QA head parity when those heads are admitted.
- Suggested tolerances: fp32 `rtol=1e-5, atol=1e-6`; fp16/bf16 `rtol=2e-2, atol=2e-2` if reduced precision is used.

## 13. Performance probes

- Tokenization throughput.
- Encoder-only throughput over `B` and `S<=512`.
- Attention backend comparison for bidirectional MHA.
- FFN GEMM/activation time.
- LayerNorm/residual bandwidth.
- MLM full logits vs masked-position-only logits.
- Vocab-size sweep: 30K vs 119K.
- BERT base vs large latency/throughput.

## 14. Skip/defer list

- Training and pretraining losses.
- Dropout.
- Decoder mode, cross-attention, and cache.
- NSP unless explicitly needed.
- Beam/generation workflows.
- Multi-GPU sharding for first parity.
- Token dropping conversion scripts.

## 15. Final implementation checklist

- [ ] Parse BertConfig and reconcile default omitted fields.
- [ ] Load word, position, and token type embeddings.
- [ ] Implement embedding sum and LayerNorm.
- [ ] Implement bias GEMM lowering.
- [ ] Implement bidirectional MHA.
- [ ] Implement GELU FFN with post-residual LayerNorm.
- [ ] Implement pooler.
- [ ] Implement masked-LM transform and tied decoder projection.
- [ ] Add embedding parity tests.
- [ ] Add one-layer and full-encoder parity tests.
- [ ] Add masked-LM parity tests.
- [ ] Add masked-position-only rewrite and parity test.
- [ ] Benchmark encoder, attention, FFN, LayerNorm, and MLM head.
