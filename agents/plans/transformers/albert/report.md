# Transformers ALBERT Operator and Integration Report

## 1. Source basis

```text
Transformers commit/version:
  Local checkout X:/H/transformers at b75feb2af64c3e29cbbc1bd859958c5432cc7ed4.
  Remote upstream: https://github.com/huggingface/transformers.git

Model id:
  Primary worked example: albert/albert-base-v2.
  Additional sizing references: albert/albert-base-v1,
  albert/albert-large-v2, albert/albert-xlarge-v2,
  albert/albert-xxlarge-v2.

Config source:
  https://huggingface.co/albert/albert-base-v1/raw/main/config.json
  https://huggingface.co/albert/albert-base-v2/raw/main/config.json
  https://huggingface.co/albert/albert-large-v2/raw/main/config.json
  https://huggingface.co/albert/albert-xlarge-v2/raw/main/config.json
  https://huggingface.co/albert/albert-xxlarge-v2/raw/main/config.json
  Tokenizer metadata checked from tokenizer_config.json, tokenizer.json,
  and Hugging Face repo file metadata for albert-base-v2 and albert-xxlarge-v2.

Source files inspected:
  X:/H/transformers/src/transformers/models/albert/modeling_albert.py
  X:/H/transformers/src/transformers/models/albert/configuration_albert.py
  X:/H/transformers/src/transformers/models/albert/tokenization_albert.py
  X:/H/transformers/src/transformers/masking_utils.py for the shared
  bidirectional attention-mask helper used by AlbertModel.

Any missing files or assumptions:
  No remote-code files are required for standard ALBERT. This report targets
  encoder inference and masked-LM/fill-mask first. Base encoder, pooling, and
  masked-LM are required for that target. Sequence classification, token
  classification, QA, and multiple-choice heads are optional staged heads.
  Pretraining sentence-order prediction and training losses are documented but
  deferred unless a checkpoint or product explicitly needs them.
```

## 2. High-level architecture

ALBERT is a text-only BERT-like bidirectional encoder with two defining changes:
factorized embeddings and cross-layer parameter sharing. Token, position, and
segment embeddings live in `embedding_size` space, then a learned projection maps
them into `hidden_size`. The encoder loop runs `num_hidden_layers` logical
layers, but selects one of `num_hidden_groups` physical `AlbertLayerGroup`
modules; layers assigned to the same group reuse the exact same attention and
FFN parameters.

```text
SentencePiece/Unigram tokenization + [CLS]/[SEP]
  -> word + token_type + learned absolute position embeddings in E
  -> LayerNorm(E) + dropout
  -> Linear(E -> H) embedding_hidden_mapping_in
  -> repeated shared-parameter bidirectional encoder layers
  -> optional pooler
  -> masked-LM / classifier / QA / token head
```

Primary runtime path:

```text
input_ids, attention_mask, optional token_type_ids/position_ids
  -> AlbertModel encoder
  -> AlbertMLMHead
  -> logits [B, S, vocab_size]
```

Independently stageable units are tokenizer/input packing, embedding path,
embedding-to-hidden projection, one physical `AlbertLayer`, the logical
layer-sharing loop, pooler, and masked-LM head.

## 3. Important config dimensions

Worked example: `albert/albert-base-v2`.

| Field | Value | Source |
|---|---:|---|
| model_type | albert | config/source |
| primary task metadata | fill-mask | HF repo metadata |
| vocab_size / V | 30000 | config.json |
| embedding_size / E | 128 | config.json |
| hidden_size / H | 768 | config.json |
| num_hidden_layers / L | 12 | config.json |
| num_hidden_groups / G | 1 | config.json |
| inner_group_num | 1 | config.json |
| physical layer modules | 1 | inferred: `G * inner_group_num` |
| num_attention_heads / A | 12 | config.json |
| head_dim / D | 64 | inferred: `H / A` |
| intermediate_size / I | 3072 | config.json |
| hidden_act | gelu_new | config.json |
| max_position_embeddings | 512 | config.json |
| type_vocab_size | 2 | config.json |
| hidden_dropout_prob | 0.0 | config default / inspected configs |
| attention_probs_dropout_prob | 0.0 | config default / inspected configs |
| layer_norm_eps | 1e-12 | config.json |
| classifier_dropout_prob | 0.1 | config.json |
| tie_word_embeddings | true | source default/config class |
| cache support | none for primary encoder | source behavior |

Representative checkpoint sweep:

| Checkpoint | E | H | I | L | G | inner | physical groups/layers | A | D | V | max pos | activation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `albert/albert-base-v1` | 128 | 768 | 3072 | 12 | 1 | 1 | 1 | 12 | 64 | 30000 | 512 | gelu |
| `albert/albert-base-v2` | 128 | 768 | 3072 | 12 | 1 | 1 | 1 | 12 | 64 | 30000 | 512 | gelu_new |
| `albert/albert-large-v2` | 128 | 1024 | 4096 | 24 | 1 | 1 | 1 | 16 | 64 | 30000 | 512 | gelu_new |
| `albert/albert-xlarge-v2` | 128 | 2048 | 8192 | 24 | 1 | 1 | 1 | 16 | 128 | 30000 | 512 | gelu_new |
| `albert/albert-xxlarge-v2` | 128 | 4096 | 16384 | 12 | 1 | 1 | 1 | 64 | 64 | 30000 | 512 | gelu_new |

Config default trap: the current `AlbertConfig` class defaults to the
xxlarge-v2 geometry (`H=4096`, `A=64`, `I=16384`, `L=12`), not base. Real
checkpoint JSON must drive sizing.

## 3a. Family variation traps

- `embedding_size != hidden_size` for official checkpoints. The word,
  position, token-type embeddings and MLM transform/decoder operate in
  `E=128`, while encoder attention/FFN operate in `H`.
- Cross-layer parameter sharing is architectural. For official configs
  `num_hidden_groups=1`, so every logical encoder layer calls the same physical
  group and same physical `AlbertLayer` weights. Lowering must not clone
  constants or mutate shared-weight state per logical layer.
- `group_idx = int(i / (num_hidden_layers / num_hidden_groups))`; nonstandard
  configs with `G > 1` partition logical layers across physical groups.
- `inner_group_num > 1` creates multiple physical `AlbertLayer` modules inside
  each group and calls all of them each time that group is selected. Effective
  layer applications are `num_hidden_layers * inner_group_num`.
- v1/v2 official checkpoint variants differ in activation: base-v1 uses
  `gelu`, v2 checkpoints use `gelu_new`.
- `albert-xlarge-v2` has `D=128`; do not assume BERT-like `head_dim=64`.
- Position embeddings are learned absolute embeddings in `E` space, not RoPE or
  relative bias.
- Token type embeddings are model inputs. The tokenizer metadata lists
  `model_input_names=["input_ids", "attention_mask"]`, but source accepts
  `token_type_ids` and defaults them to zeros from a registered buffer.
- `position_ids` default to `[0..S-1]`; if caller supplies position IDs, the
  token-type default path gathers zeros using those IDs before expanding to
  `[B,S]`.
- ALBERT attention is noncausal encoder MHA. No KV cache, cross-attention,
  sliding window, ALiBi, or RoPE is needed for the primary target.
- `AlbertMLMHead` has a standalone `bias` parameter and a decoder linear. HF
  tie metadata ties `predictions.decoder.weight` to
  `albert.embeddings.word_embeddings.weight` and decoder bias to
  `predictions.bias`; DinoML should represent this alias explicitly.
- Layout translation should be guarded off for the core sequence graph. Source
  tensors are `[B,S,H]` or `[B,S,E]`, and axis-sensitive operations include
  attention softmax over `dim=-1`, pooler `sequence_output[:,0]`,
  multiple-choice flatten/unflatten, QA split/squeeze on the last dimension,
  and MLM logits `[B,S,V]`.

## 4. Operator coverage checklist

### Tensor/layout ops

- Integer embedding gather:
  - word table `[V,E]`, input IDs `[B,S] -> [B,S,E]`
  - position table `[512,E]`, position IDs `[1,S]` or `[B,S] -> broadcast/add
  - token type table `[2,E]`, token type IDs `[B,S] -> [B,S,E]`
- Default `arange`/slice for position IDs and zeros/gather/expand for default
  token type IDs.
- Elementwise add for three embedding sources.
- Reshape/view/transpose/contiguous for Q/K/V:
  `[B,S,H] -> [B,S,A,D] -> [B,A,S,D]`.
- Additive attention-mask creation/broadcast from `attention_mask[B,S]` to a
  backend-compatible bidirectional score mask.
- First-token select `sequence_output[:,0]`.
- Multiple-choice flatten `[B,C,S] -> [B*C,S]` and logits reshape
  `[B*C,1] -> [B,C]` for optional head.
- QA split last dim `Linear(H -> 2)` into start/end, squeeze `[-1]`.

### Neural network primitives

- LayerNorm with affine parameters and eps `1e-12` in both E and H spaces.
- Linear with bias:
  - embedding hidden mapping: `Linear(E -> H)`.
  - Q/K/V/O: `Linear(H -> H)`.
  - FFN: `Linear(H -> I)`, activation, `Linear(I -> H)`.
  - pooler: `Linear(H -> H)`, tanh.
  - MLM transform: `Linear(H -> E)`, activation, `LayerNorm(E)`.
  - MLM decoder: `Linear(E -> V)` with tied word embedding weight plus output
    bias.
  - optional classifiers: `Linear(H -> num_labels)`, QA `Linear(H -> 2)`,
    multiple choice `Linear(H -> 1)`.
- Residual add + LayerNorm after attention output and after FFN output.
- Activations: `gelu_new` for v2, `gelu` for base-v1, tanh for pooler.
- Dropout is an inference no-op; official configs set hidden and attention
  dropout to `0.0`, classifier dropout remains head-only and disabled in eval.

### Attention primitives

- Bidirectional encoder self-attention, MHA only.
- Eager math: `matmul(q, k^T) * D^-0.5`, add mask, softmax over keys, dropout,
  `matmul(probs, v)`.
- Source advertises FlashAttention, SDPA, and FlexAttention support through the
  shared attention interface. For DinoML, the first semantic target can be
  normal additive-mask MHA; optimized attention can be staged behind parity.

### Position/relative-bias ops

- Learned absolute position embedding gather only.
- No rotary, ALiBi, relative-bias, convolutional position encoding, or cache
  position math for the primary target.

### Generation/cache ops

- Not applicable. ALBERT is an encoder/fill-mask model for this report.

### Preprocessing-coupled ops

- Tokenizer is Unigram/SentencePiece-style with `spiece.model` or
  `tokenizer.json`.
- Normalization lowercases by default, strips accents by default, collapses
  repeated spaces, and uses metaspace prefix handling.
- Post-processing creates `[CLS] A [SEP]` or `[CLS] A [SEP] B [SEP]` with type
  IDs 0 for A and 1 for B in tokenizer output metadata. Runtime still supports
  missing token type IDs by using all zeros.

## 5. Layer/block breakdown

Embedding path:

```text
word = Embedding(V,E, padding_idx=0)(input_ids)
tok = Embedding(type_vocab,E)(token_type_ids or default zeros)
pos = Embedding(max_pos,E)(position_ids or [0..S-1])
x_e = LayerNorm_E(word + tok + pos, eps=1e-12)
x_h = Linear(E -> H, bias=True)(x_e)
```

Shared encoder layer application, logically repeated `L` times with group
selection:

```text
group_idx = int(layer_i / (L / G))
for physical_layer in albert_layer_groups[group_idx].albert_layers:
    q = Linear(H -> H, bias=True)(x).view(B,S,A,D).transpose(1,2)
    k = Linear(H -> H, bias=True)(x).view(B,S,A,D).transpose(1,2)
    v = Linear(H -> H, bias=True)(x).view(B,S,A,D).transpose(1,2)
    attn = Attention(q, k, v, bidirectional_additive_mask)
    attn = Linear(H -> H, bias=True)(attn)
    x_attn = LayerNorm_H(x + attn, eps=1e-12)

    ff = Linear(H -> I, bias=True)(x_attn)
    ff = GELU or GELU_NEW(ff)
    ff = Linear(I -> H, bias=True)(ff)
    x = LayerNorm_H(x_attn + ff, eps=1e-12)
```

Pooler and heads:

```text
pooler = tanh(Linear(H -> H)(sequence_output[:, 0]))

mlm_hidden = Linear(H -> E)(sequence_output)
mlm_hidden = activation(mlm_hidden)
mlm_hidden = LayerNorm_E(mlm_hidden)
logits = Linear(E -> V, weight=tied_word_embeddings)(mlm_hidden) + bias

sequence_logits = Linear(H -> num_labels)(dropout(pooler))
token_logits = Linear(H -> num_labels)(sequence_output)
qa_start_end = split(Linear(H -> 2)(sequence_output), dim=-1)
multiple_choice = reshape(Linear(H -> 1)(pooler), [B, C])
```

## 6. Attention requirements

- Type: noncausal bidirectional self-attention.
- Heads: MHA, no MQA/GQA. `num_key_value_heads == num_attention_heads`.
- Shapes:
  - hidden input `[B,S,H]`
  - Q/K/V `[B,A,S,D]`
  - scores `[B,A,S,S]`
  - output after attention interface `[B,S,A,D]`, reshaped to `[B,S,H]`
- Scale: `D ** -0.5`, passed explicitly as `self.scaling`.
- Masking: source calls `create_bidirectional_mask(config, inputs_embeds,
  attention_mask)`. The eager fallback adds the returned mask to attention
  scores before softmax. With no padding and an SDPA-like backend, the shared
  helper may allow the mask to be skipped; semantic lowering should still model
  an optional bidirectional additive mask.
- Dropout: disabled in inference; official encoder dropout probs are `0.0`.
- No packed/varlen metadata, sliding-window attention, local attention, RoPE,
  ALiBi, relative bias, or KV cache is required.
- FlashAttention/SDPA compatibility: straightforward full bidirectional MHA
  when the backend supports encoder masks. First integration can use a generic
  matmul-softmax-matmul path, then switch to SDPA/Flash-style kernels after
  mask parity is verified.

## 7. Position encoding and custom math

ALBERT uses learned absolute position embeddings in embedding space:

```python
def albert_embeddings(input_ids, token_type_ids=None, position_ids=None):
    if position_ids is None:
        position_ids = arange(max_position_embeddings)[None, :seq_len]
    if token_type_ids is None:
        token_type_ids = zeros_like(input_ids)
    x = word_embedding[input_ids]
    x = x + token_type_embedding[token_type_ids]
    x = x + position_embedding[position_ids]
    return layer_norm(x, eps=1e-12)
```

The position ID buffer `[1,512]` and default all-zero token type buffer can be
precomputed. Caller-supplied `position_ids` remain dynamic inputs because the
source allows them and uses them when generating default token type IDs.

`gelu_new` must match Transformers `ACT2FN["gelu_new"]`. DinoML should treat it
as a distinct activation from exact GELU unless the existing activation helper
has verified parity.

## 8. Preprocessing and input packing

CPU/data-pipeline work:

- Unigram/SentencePiece tokenization from `spiece.model` or `tokenizer.json`.
- Default text normalization: quote replacement, NFKD/strip accents, lowercase,
  repeated-space collapse, whitespace split, metaspace with prefix space.
- Special token packing:

```text
single: [CLS]:0 A:0 [SEP]:0
pair:   [CLS]:0 A:0 [SEP]:0 B:1 [SEP]:1
```

- Padding/truncation to `model_max_length=512` for official tokenizer configs
  inspected.

GPU/runtime graph inputs:

- `input_ids`: integer `[B,S]`.
- `attention_mask`: optional integer/bool/float `[B,S]`; if omitted the model
  behaves as all valid tokens through the shared mask helper.
- `token_type_ids`: optional integer `[B,S]`; if omitted source uses zeros.
- `position_ids`: optional integer `[1,S]` or `[B,S]`; if omitted source uses
  `[0..S-1]`.
- `inputs_embeds`: optional `[B,S,E]` alternative to `input_ids`; the primary
  integration can defer this alternate entry point.

For fill-mask serving, the public HF head returns full logits `[B,S,V]`. A
deployment-specific optimized API may gather only masked positions, but that is
a graph rewrite with a changed output contract and should not replace the
standard forward path silently.

## 9. Graph rewrite / lowering opportunities

### Rewrite: embedding triple-sum + E-space LayerNorm

Source pattern:

```text
Embedding(input_ids) + Embedding(token_type_ids) + Embedding(position_ids)
  -> LayerNorm(E)
```

Replacement:

```text
FusedEmbedding3Add(input_ids, token_type_ids, position_ids) -> LayerNorm(E)
```

Preconditions:

- All three tables have output width `E`.
- Token type IDs and position IDs are either explicit or generated exactly as
  source defaults.
- Position IDs are in `[0, max_position_embeddings)`.

Shape equations:

- IDs `[B,S]`, position IDs `[1,S]` or `[B,S]`.
- Output `[B,S,E]`.

Failure cases:

- Caller uses `inputs_embeds` instead of `input_ids`.
- Caller supplies unusual `position_ids`; default-token-type gather semantics
  must still match.

Parity test sketch:

- Compare embeddings for single sentence, pair sentence with explicit
  `token_type_ids`, omitted token type IDs, and custom position IDs.

### Rewrite: Linear with bias -> GEMM_RCR_Bias

Source pattern: every `nn.Linear` in ALBERT has bias in inspected source.

Replacement:

```text
Flatten leading dims -> GEMM_RCR_Bias -> restore leading dims
```

Preconditions:

- Dense row-major activation storage.
- Weight stored in PyTorch `nn.Linear` orientation `[out_features, in_features]`
  and transformed to DinoML GEMM RHS orientation consistently.

Shape equations:

- `[B,S,K] x [O,K]^T -> [B,S,O]`.

Failure cases:

- Tied MLM decoder must preserve alias to the word embedding table instead of
  materializing a divergent copy.

Parity test sketch:

- Compare embedding mapping, Q/K/V/O, FFN, pooler, MLM transform, and decoder
  projections independently.

### Rewrite: physical shared layer loop

Source pattern:

```text
for i in range(num_hidden_layers):
    group_idx = int(i / (num_hidden_layers / num_hidden_groups))
    x = albert_layer_groups[group_idx](x)
```

Replacement:

```text
Unrolled logical layer calls that reference shared constant handles
```

Preconditions:

- Constants retain identity/provenance across all logical calls to the same
  physical group/layer.
- Runtime scheduling treats each logical call as a separate op instance but uses
  the same weight storage.

Shape equations:

- Each logical call preserves `[B,S,H]`.

Lowering implications:

- Generated manifests should reference the same constant name/storage for all
  reused Q/K/V/O/FFN/LayerNorm weights.
- Provider profiling may need per-logical-call workload entries but should not
  duplicate constant residency or GGUF encoded storage.
- Future weight offload must not unload a shared weight after an early logical
  use if later logical uses still need it in the same forward pass.

Failure cases:

- Weight import code that keys constants by logical layer index and silently
  clones shared parameters.
- In-place temporary planning that assumes physical layer weights are consumed
  only once.

Parity test sketch:

- Build a tiny config with `L=4`, `G=1`, `inner_group_num=1`; verify all four
  logical applications use the same source tensors and match HF output.

### Rewrite: packed QKV projection

Source pattern: separate `query`, `key`, and `value` biased linears over the
same hidden states.

Replacement:

```text
PackedLinear(H -> 3H, bias=True) -> split q,k,v
```

Weight transform:

```python
w_qkv = concat([w_q, w_k, w_v], axis=0)
b_qkv = concat([b_q, b_k, b_v], axis=0)
```

Preconditions:

- Self-attention path, same input tensor for Q/K/V.
- Equal output width `H` for Q/K/V.
- Shared-layer aliasing is preserved for the packed replacement if the physical
  layer is reused.

Failure cases:

- Future nonstandard cross-attention path would need separate Q and KV inputs;
  not applicable to standard ALBERT.

Parity test sketch:

- Compare packed split outputs with independent projections before attention.

### Rewrite: masked-position-only MLM logits

Source pattern:

```text
AlbertMLMHead(sequence_output) -> logits [B,S,V]
```

Replacement:

```text
Gather(sequence_output, mask_positions) -> MLMHead -> logits [num_masks,V]
```

Preconditions:

- Caller requests fill-mask logits only for known masked token positions.
- API explicitly changes output shape or returns gathered logits as an
  additional optimized output.

Failure cases:

- Standard HF forward requires full `[B,S,V]` logits.
- Training/loss paths need full labels or an equivalent indexed loss.

Parity test sketch:

- Compare gathered optimized logits to full logits indexed at the same mask
  positions.

## 10. Kernel fusion candidates

Highest priority:

- Bias GEMM coverage for all linears, including very large FFN GEMMs in
  xlarge/xxlarge and E-to-H/H-to-E factorized projection GEMMs.
- Shared-weight-aware constant lowering. This is correctness-critical for
  ALBERT and also important for memory footprint.
- LayerNorm + residual add in H space and embedding LayerNorm in E space.
- Bidirectional MHA with additive mask, eventually through SDPA/Flash-style
  encoder attention.

Medium priority:

- Packed QKV projection with bias, preserving physical shared-layer aliases.
- GELU/GELU_NEW fusion in FFN and MLM transform.
- Embedding triple-add + LayerNorm fusion.
- Masked-position-only MLM head for fill-mask serving.

Lower priority:

- Pooler + classifier fusion for small downstream heads.
- SOP head and pretraining loss.
- Multiple-choice flatten/unflatten convenience lowering.
- Output attentions/hidden states materialization.

## 11. Runtime staging plan

Stage 1: Parse `AlbertConfig`, load tokenizer-independent input contract, and
load factorized embedding/encoder weights while preserving shared parameter
identity.

Stage 2: Implement embedding path parity in E space, including token type and
position defaults.

Stage 3: Implement `embedding_hidden_mapping_in` and one physical
`AlbertLayer` in fp32 using generic bias GEMMs, LayerNorm, activation, and
bidirectional attention.

Stage 4: Implement logical encoder loop with group-based shared weights.
Validate official `G=1` checkpoints and a synthetic `G>1`/`inner_group_num>1`
config.

Stage 5: Add `AlbertForMaskedLM` head with tied decoder weight and output bias.

Stage 6: Add pooler and selected downstream heads as needed.

Stage 7: Optimize packed QKV, fused residual LayerNorm, fused embeddings, and
encoder attention backend.

Stubbable initially: dropout, losses, SOP, output hidden states/attentions,
`inputs_embeds` entry point, task-specific heads beyond masked LM.

## 12. Parity and validation plan

- Config parsing tests for base-v1, base-v2, large-v2, xlarge-v2, xxlarge-v2,
  including `E`, `H`, `G`, `inner_group_num`, activation, and head dim.
- Embedding path parity for explicit/default token type IDs and position IDs.
- `gelu` versus `gelu_new` activation parity against Transformers.
- Single attention parity with additive bidirectional masks and padding.
- Single physical `AlbertLayer` parity.
- Shared-loop parity for `L > 1` with `G=1`: verify repeated use of identical
  physical weights matches HF.
- Synthetic `G=2` and `inner_group_num=2` parity to guard the group-index math.
- Full encoder parity for `albert-base-v2` at short and max sequence lengths.
- Masked-LM head parity including tied decoder weight and bias.
- Fill-mask end-to-end parity on a tokenized sentence after tokenizer output is
  fixed.
- Optional head parity: pooler/sequence classification, token classification,
  QA split/squeeze, multiple choice reshape.
- Suggested tolerances: fp32 `rtol=1e-5, atol=1e-6`; fp16/bf16
  `rtol=2e-2, atol=2e-2`, with attention/softmax validated carefully.

## 13. Performance probes

- Tokenization throughput and padding overhead separate from GPU runtime.
- Encoder throughput over `S=16, 32, 128, 512` and batch sizes
  `B=1, 4, 8, 16`.
- Shared-weight memory footprint versus a naive cloned-layer import.
- Constant residency/offload probe for shared weights: peak loaded bytes and
  reload count across logical layer calls.
- Attention backend comparison: eager matmul/softmax, SDPA, Flash-style
  bidirectional attention.
- FFN GEMM time for base, large, xlarge, and xxlarge geometries.
- LayerNorm/residual bandwidth in H space and embedding LayerNorm in E space.
- Full logits `[B,S,V]` versus masked-position-only MLM logits.
- Activation variant cost and parity for `gelu` and `gelu_new`.

## 14. Skip/defer list

- Training, dropout behavior, and all loss computation.
- Sentence order prediction/pretraining head unless explicitly needed.
- Sequence classification, token classification, QA, and multiple-choice heads
  for the first masked-LM target.
- Output attentions and hidden-state capture as optimized runtime outputs.
- `inputs_embeds` alternate entry point.
- Quantization and multi-GPU/tensor parallel.
- Generation, beam search, and KV cache; not applicable to this encoder family.
- Exotic nonstandard configs after the guarded `G`/`inner_group_num` loop is
  validated.

## 15. Final implementation checklist

- [ ] Parse `AlbertConfig` and checkpoint JSON defaults.
- [ ] Load word, position, and token type embeddings in `embedding_size` space.
- [ ] Implement default `position_ids` and `token_type_ids`.
- [ ] Implement E-space embedding sum and LayerNorm eps `1e-12`.
- [ ] Implement `embedding_hidden_mapping_in` `Linear(E -> H)`.
- [ ] Implement bias GEMM lowering for ALBERT linears.
- [ ] Implement bidirectional MHA with additive mask.
- [ ] Implement H-space residual LayerNorm after attention and FFN.
- [ ] Implement `gelu` and `gelu_new` activation parity.
- [ ] Preserve shared constants across `num_hidden_layers` logical calls.
- [ ] Implement `num_hidden_groups` and `inner_group_num` loop semantics.
- [ ] Implement pooler first-token select, dense, and tanh.
- [ ] Implement MLM transform `Linear(H -> E)`, activation, LayerNorm.
- [ ] Implement tied MLM decoder using word embedding weight and output bias.
- [ ] Add embedding, one-layer, shared-loop, full-encoder, and MLM parity tests.
- [ ] Add optional downstream head parity tests when admitted.
- [ ] Benchmark attention, FFN GEMMs, LayerNorm, full logits, and masked-only logits.
