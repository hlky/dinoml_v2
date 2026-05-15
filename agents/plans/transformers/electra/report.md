# Transformers ELECTRA Operator and Integration Report

## 1. Source basis

```text
Transformers commit/version:
  Local checkout transformers at b75feb2af64c3e29cbbc1bd859958c5432cc7ed4.
  Remote upstream: https://github.com/huggingface/transformers.git

Model id:
  Primary target: google/electra-base-discriminator for replaced-token detection
  and downstream encoder inference.
  Generator/head target: google/electra-base-generator and google/electra-small-generator.
  Additional sizing references: google/electra-small-discriminator,
  google/electra-large-discriminator.

Config source:
  https://huggingface.co/google/electra-small-discriminator/raw/main/config.json
  https://huggingface.co/google/electra-base-discriminator/raw/main/config.json
  https://huggingface.co/google/electra-large-discriminator/raw/main/config.json
  https://huggingface.co/google/electra-small-generator/raw/main/config.json
  https://huggingface.co/google/electra-base-generator/raw/main/config.json
  Tokenizer metadata checked from tokenizer_config.json for the same Google repos.

Source files inspected:
  transformers/src/transformers/models/electra/modeling_electra.py
  transformers/src/transformers/models/electra/configuration_electra.py
  transformers/src/transformers/models/auto/tokenization_auto.py

Any missing files or assumptions:
  The ELECTRA model directory has no family-local tokenizer implementation.
  AutoTokenizer maps `electra` to BertTokenizer/BertTokenizerFast, and the
  Google tokenizer metadata only records lowercase WordPiece behavior plus
  max length. No remote-code files are required for the standard checkpoints.
  This report is docs-only; no DinoML tests were run.
```

## 2. High-level architecture

ELECTRA is a BERT-like text encoder family with learned word, token type, and absolute position embeddings, bidirectional self-attention for standard discriminator/generator checkpoints, post-residual LayerNorm, and task-specific heads. The commonly deployed discriminator predicts whether each input token was original or replaced. The generator uses the same encoder body but an MLM-style projection head.

```text
WordPiece tokenization + [CLS]/[SEP]/token_type_ids
  -> word + token_type + absolute position embeddings in embedding_size E
  -> embedding LayerNorm/dropout
  -> optional Linear(E -> H) embedding projection
  -> bidirectional encoder stack in hidden_size H
  -> discriminator replaced-token head or generator MLM head
```

Primary DinoML runtime scope should be encoder inference plus the discriminator replaced-token head. Masked-LM generator support is useful because official generator checkpoints are common and expose the `embedding_size != hidden_size` projection pattern. Sequence classification, token classification, QA, and multiple choice are optional downstream heads. `ElectraForCausalLM` and decoder/cross-attention paths exist in source but are not the primary ELECTRA inference contract.

## 3. Important config dimensions

`ElectraConfig` source defaults resemble `google/electra-small-discriminator`.

| Field | Source default | Operator relevance |
|---|---:|---|
| vocab_size / V | 30522 | word embeddings and generator LM head |
| embedding_size / E | 128 | embedding table width before optional projection |
| hidden_size / H | 256 | encoder width |
| num_hidden_layers | 12 | encoder block repeat count |
| num_attention_heads / A | 4 | MHA head count |
| head_dim / D | 64 | inferred as `H / A` |
| intermediate_size / I | 1024 | FFN expansion |
| hidden_act | gelu | FFN/discriminator activation; generator head uses GELU |
| max_position_embeddings | 512 | learned absolute position table |
| type_vocab_size | 2 | segment/token type table |
| layer_norm_eps | 1e-12 | all LayerNorms |
| pad_token_id | 0 | word embedding padding index |
| is_decoder | false | standard path uses bidirectional mask |
| use_cache | true default | ignored for encoder because source forces `use_cache=False` |
| tie_word_embeddings | true | generator LM head may tie to word embedding table |

Representative checkpoint sweep:

| Checkpoint | Architecture | E | H | I | Layers | Heads | D | V | Max pos | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| google/electra-small-discriminator | ElectraForPreTraining | 128 | 256 | 1024 | 12 | 4 | 64 | 30522 | 512 | source-default shape; requires E->H projection |
| google/electra-base-discriminator | ElectraForPreTraining | 768 | 768 | 3072 | 12 | 12 | 64 | 30522 | 512 | BERT-base geometry; no embedding projection |
| google/electra-large-discriminator | ElectraForPreTraining | 1024 | 1024 | 4096 | 24 | 16 | 64 | 30522 | 512 | BERT-large geometry |
| google/electra-small-generator | ElectraForMaskedLM | 128 | 256 | 1024 | 12 | 4 | 64 | 30522 | 512 | generator head projects H->E before vocab |
| google/electra-base-generator | ElectraForMaskedLM | 768 | 256 | 1024 | 12 | 4 | 64 | 30522 | 512 | unusual: wide embeddings and narrow encoder, E->H then H->E |

Fields commonly omitted from these configs but supplied by `ElectraConfig`: `attention_probs_dropout_prob=0.1`, `hidden_dropout_prob=0.1`, `initializer_range=0.02`, `classifier_dropout=None`, `summary_type="first"`, `summary_use_proj=True`, `summary_activation="gelu"`, `summary_last_dropout=0.1`, `is_decoder=False`, `add_cross_attention=False`, `use_cache=True`, `tie_word_embeddings=True`.

## 3a. Family variation traps

- ELECTRA is not just BERT with a different head: `embedding_size` can differ from `hidden_size`. DinoML must preserve the optional `embeddings_project: Linear(E -> H)` after embedding LayerNorm.
- Generator checkpoints use `ElectraGeneratorPredictions`: `Linear(H -> E) -> GELU -> LayerNorm(E) -> Linear(E -> V)`. For `google/electra-base-generator`, this means `768 -> 256` at the embedding projection, then `256 -> 768` before the tied vocab projection.
- Discriminator checkpoints use `ElectraDiscriminatorPredictions`: `Linear(H -> H) -> activation -> Linear(H -> 1) -> squeeze(-1)`, producing `[B,S]` logits rather than `[B,S,V]`.
- Standard discriminator/generator configs are encoder-only and bidirectional. Source supports decoder causal masks, cache, and cross-attention, but those are non-primary variants.
- Attention is standard MHA; there is no GQA/MQA, RoPE, ALiBi, sliding window, or relative-position bias in the inspected standard configs.
- Sequence classification uses an ELECTRA-specific CLS head with GELU, not BERT's pooler tanh path.
- Multiple choice uses `ElectraSequenceSummary`, whose pooling type is config-driven (`first`, `last`, `mean`, or `cls_index`).
- Token type IDs are semantically active. Missing `token_type_ids` are filled from an all-zero buffer indexed by `position_ids`.
- Layout translation is not a useful family-level optimization: text tensors are `[B,S,H]`, attention reshapes to `[B,A,S,D]`, and all axis-sensitive ops are sequence/feature/head transforms that should stay in source order.

## 4. Operator coverage checklist

### Tensor/layout ops

- Input validation: exactly one of `input_ids[B,S]` or `inputs_embeds[B,S,E]`.
- Embedding gathers:
  - word embedding `[V,E]`, padding index 0 for Google checkpoints.
  - token type embedding `[type_vocab_size,E]`.
  - position embedding `[max_position_embeddings,E]`.
- Default position ID slice `[past_len:past_len+S]`, normally `[0:S]`.
- Default token type IDs all zeros, gathered by position IDs then expanded to `[B,S]`.
- Elementwise embedding sum and LayerNorm over `E`.
- Optional embedding projection `Linear(E -> H, bias=True)`.
- Attention projection reshape/transpose: `[B,S,H] -> [B,A,S,D]`.
- Attention output transpose/contiguous/reshape: `[B,A,S,D] -> [B,S,H]`.
- Additive attention mask construction and broadcasting over heads/query positions.
- Squeeze/split/gather ops for discriminator logits, QA logits, CLS pooling, multiple-choice flatten/unflatten.

### Neural network primitives

- LayerNorm with affine weight/bias and eps `1e-12`.
- Bias Linear/GEMM for all projections.
- GELU activation in FFN, generator head, classifier head, and common discriminator configs.
- Residual add followed by LayerNorm in attention output and FFN output.
- Dropout is inference no-op but must not perturb graph shape.
- Generator LM projection can be tied to `electra.embeddings.word_embeddings.weight`; weight aliasing or explicit tied-weight metadata matters.

Base discriminator shapes:

- Q/K/V/O: `Linear(768 -> 768, bias=True)`.
- FFN: `Linear(768 -> 3072) -> GELU -> Linear(3072 -> 768)`.
- Discriminator head: `Linear(768 -> 768) -> GELU -> Linear(768 -> 1) -> squeeze`.

Base generator shapes:

- Embedding projection: `Linear(768 -> 256)`.
- Q/K/V/O: `Linear(256 -> 256, bias=True)`.
- FFN: `Linear(256 -> 1024) -> GELU -> Linear(1024 -> 256)`.
- Generator head: `Linear(256 -> 768) -> GELU -> LayerNorm(768) -> Linear(768 -> 30522)`.

### Attention primitives

- Standard noncausal bidirectional self-attention for primary target.
- Optional causal self-attention and cross-attention only for decoder-mode `ElectraForCausalLM`.
- Backend dispatch through `ALL_ATTENTION_FUNCTIONS`; source declares FlashAttention, SDPA, and FlexAttention support.
- Eager fallback: QK matmul, scale by `D ** -0.5`, add mask, softmax over last dim, dropout, AV matmul.

### Position/relative-bias ops

- Learned absolute position embeddings only.
- No RoPE, ALiBi, or relative-position bias for standard ELECTRA.

### Preprocessing-coupled ops

- Bert WordPiece tokenization via AutoTokenizer mapping.
- Lowercasing for Google uncased checkpoints (`do_lower_case=true`).
- Special tokens `[CLS]`, `[SEP]`, `[PAD]`, `[MASK]`, `[UNK]` by BERT tokenizer convention.
- Segment/token type IDs for sentence pairs.

### Generation/cache ops

- Not required for primary discriminator/generator encoder inference.
- Decoder-mode cache uses DynamicCache or EncoderDecoderCache with per-layer keys/values shaped `[B,A,T,D]`.

## 5. Layer/block breakdown

Embedding path:

```text
word = Embedding(V,E)(input_ids)
tok = Embedding(type_vocab,E)(token_type_ids or zeros)
pos = Embedding(max_pos,E)(position_ids)
x_e = LayerNorm_E(word + tok + pos)
x = Linear(E -> H)(x_e) if E != H else x_e
```

Encoder block, repeated `N` times:

```text
q = Linear(H -> H, bias=True)(x) -> view [B,S,A,D] -> transpose [B,A,S,D]
k = Linear(H -> H, bias=True)(x) -> view [B,S,A,D] -> transpose [B,A,S,D]
v = Linear(H -> H, bias=True)(x) -> view [B,S,A,D] -> transpose [B,A,S,D]
attn = Attention(q, k, v, additive_bidirectional_mask)
y = Linear(H -> H, bias=True)(attn)
x = LayerNorm(y + x)

z = Linear(H -> I, bias=True)(x)
z = GELU(z)
z = Linear(I -> H, bias=True)(z)
x = LayerNorm(z + x)
```

Discriminator head:

```text
h = Linear(H -> H, bias=True)(x)
h = hidden_act(h)
logits = Linear(H -> 1, bias=True)(h).squeeze(-1)  # [B,S]
```

Generator MLM head:

```text
h = Linear(H -> E, bias=True)(x)
h = GELU(h)
h = LayerNorm_E(h)
logits = Linear(E -> V, bias=True, tied_weight=word_embeddings)(h)  # [B,S,V]
```

Downstream heads:

```text
sequence classification: x[:,0] -> dropout -> Linear(H,H) -> GELU -> dropout -> Linear(H,num_labels)
token classification: dropout(x) -> Linear(H,num_labels)
question answering: Linear(H,2)(x) -> split start/end -> squeeze
multiple choice: flatten choices -> encoder -> SequenceSummary -> Linear(H,1) -> reshape [B,num_choices]
```

## 6. Attention requirements

Primary ELECTRA attention is encoder self-attention:

- Noncausal, bidirectional MHA.
- `A=4/12/16`, `D=64` in representative Google checkpoints.
- K/V heads equal Q heads; no grouped-query or multi-query expansion.
- Masking is additive and created by `create_bidirectional_mask` from `attention_mask`.
- Dropout is passed as zero during inference.
- No packed/varlen metadata is required by source semantics.
- FlashAttention/SDPA compatibility is good for the primary path: dense Q/K/V, standard scale, additive padding mask, no relative bias.

Decoder-mode path is optional:

- `config.is_decoder=True` switches to `create_causal_mask`.
- `use_cache` can instantiate `DynamicCache`; source stores projected K/V after the linear projections and before attention backend dispatch.
- Cross-attention is admitted only when `add_cross_attention=True` and `encoder_hidden_states` are passed.
- This path should be a separate DinoML target because common ELECTRA checkpoints are not autoregressive LMs.

Eager attention math:

```python
scores = matmul(q, k.transpose(-2, -1)) * (head_dim ** -0.5)
scores = scores + attention_mask
probs = softmax(scores, dim=-1)
out = matmul(probs, v).transpose(1, 2).contiguous()
```

## 7. Position encoding and custom math

Position math is simple learned absolute embedding lookup. The only dynamic offset is `past_key_values_length`, which is zero for standard encoder inference.

```python
def electra_embeddings(input_ids, token_type_ids, position_ids):
    word = gather(word_embeddings, input_ids)       # [B,S,E]
    tok = gather(token_type_embeddings, token_type_ids)
    pos = gather(position_embeddings, position_ids)
    x = layer_norm(word + tok + pos, eps=1e-12)
    return embeddings_project(x) if E != H else x
```

Precomputable:

- Default position IDs `[0, 1, ..., max_position_embeddings-1]`.
- Default all-zero token type buffer.

Dynamic:

- Caller-supplied `position_ids`.
- Pair-input token type IDs from tokenizer.
- Decoder-mode past offset, if that non-primary path is admitted.

## 8. Preprocessing and input packing

CPU/data-pipeline work:

- Bert WordPiece tokenization with lowercasing for Google uncased checkpoints.
- Special token insertion: single sequence `[CLS] A [SEP]`; pair sequence `[CLS] A [SEP] B [SEP]`.
- Token type IDs: 0 for first segment/specials around it, 1 for second segment.
- Padding/truncation to `model_max_length=512` for checked Google tokenizer metadata.

GPU/runtime inputs:

- `input_ids[B,S]` int64.
- `attention_mask[B,S]`, typically 1 for valid tokens and 0 for padding.
- Optional `token_type_ids[B,S]`; default all zeros.
- Optional `position_ids[1,S]` or `[B,S]`.
- Optional `inputs_embeds[B,S,E]` instead of `input_ids`.

No multimodal placeholder, scatter stitching, image/audio preprocessing, or packed sequence descriptors are involved.

## 9. Graph rewrite / lowering opportunities

### Rewrite: embedding triple-sum plus optional projection

Source pattern:

```text
GatherWord + GatherTokenType + GatherPosition -> Add/Add -> LayerNorm(E) -> optional Linear(E,H)
```

Replacement:

```text
FusedEmbeddingSumLayerNorm(E) -> optional GEMM_RCR_Bias(E,H)
```

Preconditions:

- All embeddings produce identical `[B,S,E]` shape.
- Missing token type IDs use the source all-zero behavior.
- Caller-provided `position_ids` are honored.

Failure cases:

- `inputs_embeds` bypasses word embedding gather.
- Decoder-mode nonzero `past_key_values_length` shifts default position IDs.

Parity test sketch:

- Compare default and explicit `token_type_ids`/`position_ids` for small, base-discriminator, and base-generator configs.

### Rewrite: bias Linear over `[B,S,*]` to GEMM

Source pattern:

```text
nn.Linear(in,out)(dense [B,S,in])
```

Replacement:

```text
Flatten B*S -> GEMM_RCR_Bias(in,out) -> reshape [B,S,out]
```

Preconditions:

- Dense contiguous row-major logical input.
- Bias is present.
- Weight orientation follows PyTorch Linear: weight stored `[out_features,in_features]`.

Failure cases:

- Tied generator LM head must preserve alias or explicit weight sharing with word embeddings.

Parity test sketch:

- Per-layer compare Q/K/V/O, FFN, embedding projection, generator projection, discriminator dense, and classifier heads.

### Rewrite: independent Q/K/V projections to packed QKV

Source pattern:

```text
q = Linear(H,H)(x); k = Linear(H,H)(x); v = Linear(H,H)(x)
```

Replacement:

```text
PackedLinear(H,3H) -> split [q,k,v] -> attention
```

Weight transform:

```python
w_qkv = concat([w_q, w_k, w_v], axis=0)
b_qkv = concat([b_q, b_k, b_v], axis=0)
```

Preconditions:

- Self-attention only.
- Same source tensor feeds Q, K, V.
- All projections have matching `H`.

Failure cases:

- Cross-attention uses query from decoder hidden states and K/V from encoder hidden states.

Parity test sketch:

- Compare packed split tensors before attention for discriminator and generator configs.

### Rewrite: discriminator token head fusion

Source pattern:

```text
Linear(H,H) -> GELU -> Linear(H,1) -> squeeze(-1)
```

Replacement:

```text
GEMM_Bias_GELU -> GEMM_Bias -> Squeeze
```

Preconditions:

- `hidden_act` is a supported activation, GELU for official Google configs.
- Output consumer accepts `[B,S]` logits.

Failure cases:

- Training loss path uses attention-mask-indexed active logits; not needed for inference.

Parity test sketch:

- Compare `ElectraForPreTraining` logits before sigmoid/rounding.

### Rewrite: masked generator logits only

Source pattern:

```text
Full [B,S,V] generator logits
```

Replacement:

```text
Gather hidden at mask positions -> GeneratorPredictions -> vocab GEMM
```

Preconditions:

- Caller only needs logits at known masked positions.
- Optimized API may return gathered logits rather than the standard full `[B,S,V]`.

Failure cases:

- Standard `ElectraForMaskedLM.forward` returns full sequence logits and must remain available for parity.

Parity test sketch:

- Compare gathered optimized logits with full logits indexed at mask positions.

## 10. Kernel fusion candidates

Highest priority:

- Bias GEMM for Linear layers, including narrow generator `H=256` and large discriminator `H=1024`.
- LayerNorm plus residual add for attention and FFN outputs.
- Packed QKV projection with bias.
- Bidirectional SDPA/FlashAttention path for encoder MHA.
- Embedding sum + LayerNorm + optional projection, because generator/discriminator small/base variants exercise `E != H`.

Medium priority:

- GELU fusion in FFN and generator/discriminator heads.
- Discriminator head fusion for `[B,S]` replaced-token logits.
- Generator masked-position-only vocab projection for fill-mask workloads.
- Large-vocab GEMM tuning for generator LM head.

Lower priority:

- Multiple-choice `SequenceSummary` variants beyond `first`.
- Decoder-mode causal/cache/cross-attention support.
- Training losses and active-loss masking.

## 11. Runtime staging plan

Stage 1: Parse `ElectraConfig`, load embeddings, and support `embedding_size != hidden_size`.

Stage 2: Implement embedding path parity with default token type and position IDs.

Stage 3: Implement one encoder block with bidirectional MHA, bias GEMMs, GELU FFN, residual post-LayerNorm.

Stage 4: Full `ElectraModel` encoder parity for `google/electra-small-discriminator` and `google/electra-base-discriminator`.

Stage 5: Add `ElectraForPreTraining` discriminator head and replaced-token logits.

Stage 6: Add `ElectraForMaskedLM` generator head, tied LM projection, and base-generator `E=768,H=256` parity.

Stage 7: Add optional downstream heads: sequence classification, token classification, QA, multiple choice.

Stage 8: Add optimized rewrites: packed QKV, fused residual LayerNorm, embedding fusion, masked generator logits.

Stage 9: Treat decoder/cross-attention/cache as a separate non-primary follow-up.

## 12. Parity and validation plan

- Config parsing tests for all five representative configs, including omitted defaults.
- Embedding parity with explicit and default `token_type_ids` and `position_ids`.
- `embedding_size != hidden_size` projection parity for small discriminator and base generator.
- Single self-attention parity with additive bidirectional padding mask.
- Single encoder block parity after attention output LayerNorm and FFN output LayerNorm.
- Full encoder last-hidden-state parity for small/base/large discriminator shapes.
- Discriminator head logits parity `[B,S]` against `ElectraForPreTraining`.
- Generator head logits parity `[B,S,V]`, including tied vocab projection.
- Downstream head parity as admitted: sequence classification CLS head, QA split/squeeze, token classification.
- Suggested tolerances: fp32 `rtol=1e-5, atol=1e-6`; fp16/bf16 `rtol=2e-2, atol=2e-2` after reduced-precision backend admission.

## 13. Performance probes

- Tokenization throughput and padding strategy impact for `S<=512`.
- Encoder throughput sweep over batch size and sequence length for small/base/large discriminator.
- Separate attention, FFN, LayerNorm/residual, and embedding-projection timings.
- Compare eager attention, SDPA, and FlashAttention for bidirectional masks.
- Generator MLM full logits vs masked-position-only logits.
- Discriminator head throughput for full-token replaced-token scoring.
- Shape contrast: base discriminator `H=768,A=12` vs base generator `H=256,A=4,E=768`.
- Memory bandwidth probe for embedding tables and large generator vocab projection.

## 14. Skip/defer list

- Training losses: BCE/CrossEntropy/MSE and active-loss masking.
- Dropout and gradient checkpointing.
- `ElectraForCausalLM`, causal masks, cross-attention, and KV cache.
- Beam search or generation-controller behavior.
- Multi-GPU/tensor parallel sharding.
- Nonstandard tokenizer variants outside Bert WordPiece-compatible ELECTRA repos.
- SequenceSummary `attn` mode, which source marks unimplemented.

## 15. Final implementation checklist

- [ ] Parse `ElectraConfig` and effective defaults.
- [ ] Load word, token type, and position embeddings with width `E`.
- [ ] Implement embedding sum, LayerNorm(E), and optional `Linear(E -> H)`.
- [ ] Implement bidirectional additive attention mask creation.
- [ ] Implement bias GEMM lowering for all Linear layers.
- [ ] Implement MHA reshape/transpose contracts `[B,S,H] <-> [B,A,S,D]`.
- [ ] Implement GELU FFN and post-residual LayerNorm.
- [ ] Implement discriminator replaced-token head.
- [ ] Implement generator predictions head and tied LM projection.
- [ ] Add parity for small/base/large discriminator configs.
- [ ] Add parity for small/base generator configs, especially `E != H`.
- [ ] Add optional downstream classifier/QA/token heads.
- [ ] Add packed-QKV rewrite with weight transform tests.
- [ ] Add embedding-fusion and residual-LayerNorm fusion tests.
- [ ] Benchmark encoder, attention backend, FFN, head, and full-logit vs masked-logit paths.
