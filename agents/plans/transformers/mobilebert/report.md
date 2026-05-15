# Transformers MobileBERT Operator and Integration Report

## 1. Source basis

```text
Transformers commit/version:
  Local checkout transformers at b75feb2af64c3e29cbbc1bd859958c5432cc7ed4.

Model id:
  Primary worked example: google/mobilebert-uncased.
  Additional representative configs: RedHatAI/mobilebert-uncased-finetuned-squadv1,
  mrm8488/mobilebert-finetuned-ner, vumichien/emo-mobilebert,
  optimum-intel-internal-testing/tiny-random-MobileBertModel.

Config source:
  https://huggingface.co/google/mobilebert-uncased/raw/main/config.json
  Additional raw config URLs are summarized in _sources/config_snapshots.md.

Source files inspected:
  transformers/src/transformers/models/mobilebert/configuration_mobilebert.py
  transformers/src/transformers/models/mobilebert/modeling_mobilebert.py
  transformers/src/transformers/models/mobilebert/tokenization_mobilebert.py
  transformers/src/transformers/masking_utils.py
  transformers/tests/models/mobilebert/test_modeling_mobilebert.py

Any missing files or assumptions:
  No remote-code files are required for the inspected standard MobileBERT
  family. This report targets inference. Training losses are described only as
  head behavior to avoid admitting training-only surface.
```

## 2. High-level architecture

MobileBERT is a text-only bidirectional encoder. Its external ABI looks like a
BERT encoder, but the standard checkpoint keeps a 512-wide hidden state around a
128-wide bottleneck core. Each encoder layer applies bottleneck projections,
noncausal self-attention, several small FFN sublayers, then an inverted
projection back to the 512-wide residual stream.

```text
BERT WordPiece preprocessing
  -> compact word embeddings + trigram concatenation + projection
  -> token type + learned absolute position embeddings
  -> 24-layer MobileBERT encoder
  -> optional pooler
  -> MLM / NSP / sequence classification / QA / token classification heads
```

The encoder can be validated independently from task heads. Tokenization and
special-token packing are CPU/data-pipeline work; embedding, mask construction,
encoder, pooler, and heads are GPU/runtime graph work.

## 3. Important config dimensions

Primary checkpoint: `google/mobilebert-uncased`.

| Field | Value | Source |
|---|---:|---|
| architecture | MobileBertForPreTraining | config.json |
| vocab_size / V | 30522 | config.json |
| hidden_size / H | 512 | config.json |
| true_hidden_size / T | 128 | config.json/source derived |
| embedding_size / E | 128 | config.json |
| num_hidden_layers | 24 | config.json |
| num_attention_heads / A | 4 | config.json |
| head_dim / D | 32 | inferred from `T/A` |
| intermediate_size / I | 512 | config.json |
| hidden_act | relu | config.json |
| max_position_embeddings | 512 | config.json |
| type_vocab_size | 2 | config.json |
| trigram_input | true | config.json |
| use_bottleneck | true | config.json |
| intra_bottleneck_size | 128 | config.json |
| use_bottleneck_attention | false | config.json |
| key_query_shared_bottleneck | true | config.json |
| num_feedforward_networks | 4 | config.json |
| normalization_type | no_norm | config.json |
| layer_norm_eps | 1e-12 | config.json |
| classifier_activation | false | config.json |
| cache support | none for primary encoder | source |

Representative checkpoint sweep:

| Checkpoint | task architecture | H | T | E | I | layers | heads | FFNs/block | V | notable variation |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| google/mobilebert-uncased | PreTraining | 512 | 128 | 128 | 512 | 24 | 4 | 4 | 30522 | standard pretraining, pooler activation disabled |
| RedHatAI/mobilebert-uncased-finetuned-squadv1 | QA | 512 | 128 | 128 | 512 | 24 | 4 | 4 | 30522 | QA head, attention dropout 0 |
| mrm8488/mobilebert-finetuned-ner | TokenClassification | 512 | 128 | 128 | 512 | 24 | 4 | 4 | 30522 | 8 labels |
| vumichien/emo-mobilebert | SequenceClassification | 512 | 128 | 128 | 512 | 24 | 4 | 4 | 2016 | smaller vocab, pooler activation enabled |
| optimum-intel-internal-testing/tiny-random-MobileBertModel | Model | 64 | 128 | 32 | 37 | 5 | 4 | 4 | 1124 | tiny stress config has `T > H` |

## 3a. Family variation traps

- Do not infer attention width from `hidden_size`. Attention uses
  `true_hidden_size`, which is `intra_bottleneck_size` when bottlenecks are on.
- `hidden_size` remains the external residual/output width for embeddings,
  pooler, task heads, and value projection when `use_bottleneck_attention=false`.
- Standard layers have four FFN applications per block: three `FFNLayer`
  applications after attention plus the final intermediate/output path.
- `normalization_type` can be `"no_norm"` or `"layer_norm"`. `NoNorm` is a
  learned affine `x * weight + bias`, not an identity.
- `classifier_activation` changes pooler ABI: disabled returns `hidden[:,0]`;
  enabled applies `Linear(H -> H)` and tanh.
- `trigram_input=true` changes embedding width from `E` to `3E` before the
  embedding projection. This is not an ordinary Conv1d module in source; it is
  pad/shift/cat then `Linear(3E -> H)`.
- `use_bottleneck_attention` and `key_query_shared_bottleneck` change the
  source tensors and physical bottleneck modules for Q/K/V. They need explicit
  graph guards.
- The MLM head is MobileBERT-specific and split-shaped; it is not a plain tied
  BERT decoder from `H` to `V`.
- Tiny/random configs may be shape-valid for tests while unlike production:
  `hidden_size=64`, `true_hidden_size=128`, and `intermediate_size=37`.

## 4. Operator coverage checklist

### Tensor/layout ops

- Integer embedding gathers for word `[V,E]`, position `[max_pos,H]`, token type
  `[type_vocab,H]`.
- Default position IDs slice `[1,S]` from `[0..max_pos-1]`.
- Default token type IDs as zeros `[B,S]`.
- Trigram embedding path: shift left/right with zero padding, concatenate on
  last axis to `[B,S,3E]`.
- Reshape/transpose for attention: `[B,S,T] -> [B,A,S,D]` for Q/K and
  `[B,S,H or T] -> [B,A,S,D]` for V.
- Additive bidirectional attention mask broadcast to scores.
- First-token select `hidden[:,0]`.
- Multiple-choice flatten/unflatten: `[B,C,S] -> [B*C,S] -> [B,C]`.
- QA split/squeeze/contiguous: logits `[B,S,2] -> start/end [B,S]`.

### Neural network primitives

- Linear with bias for embeddings, bottlenecks, Q/K/V/O, FFN, output
  bottleneck, pooler, and heads.
- Standard checkpoint shapes:
  - Embedding projection: `Linear(384 -> 512)`.
  - Per-layer input bottleneck: `Linear(512 -> 128)`.
  - Shared Q/K bottleneck when enabled: `Linear(512 -> 128)`.
  - Q/K: `Linear(128 -> 128)`.
  - V: `Linear(512 -> 128)` because `use_bottleneck_attention=false`.
  - attention output: `Linear(128 -> 128)`.
  - each FFN: `Linear(128 -> 512)`, ReLU, `Linear(512 -> 128)`.
  - output bottleneck after final FFN: `Linear(128 -> 512)`.
  - pooler when active: `Linear(512 -> 512)`, tanh.
- `NoNorm`: affine scale+bias over the last axis.
- `LayerNorm` for configs using `normalization_type="layer_norm"` and always
  inside MLM transform.
- Residual adds at 128-wide bottleneck and 512-wide hidden paths.
- Dropout is inference no-op.

### Attention primitives

- Noncausal bidirectional self-attention only for primary family.
- MHA, not MQA/GQA: `A=4`, `D=32`, K/V head count equals Q head count.
- Eager path is `matmul(q,k^T) * D^-0.5`, additive mask, softmax, dropout,
  `matmul(probs,v)`.
- Source also advertises SDPA and FlashAttention dispatch through
  `ALL_ATTENTION_FUNCTIONS`; DinoML can first target eager-equivalent dense
  attention and later substitute a noncausal SDPA provider.

### Position/rotary/relative-bias ops

- Learned absolute position embeddings only. No RoPE, ALiBi, relative bias, or
  sliding-window position math is required.

### Generation/cache ops

- No autoregressive KV cache for the primary encoder. `past_key_values` paths in
  the generic mask utility are not used by `MobileBertModel.forward`.

### Preprocessing-coupled ops

- Tokenizer aliases BERT tokenization. CPU preprocessing owns lowercasing,
  WordPiece, `[CLS]`/`[SEP]`, padding, and segment IDs.
- Runtime graph consumes `input_ids[B,S]`, optional `token_type_ids[B,S]`,
  optional `position_ids[B,S or 1,S]`, and optional `attention_mask[B,S]` or
  already prepared `[B,1,Q,K]` mask.

### Quantized/packed weight metadata ops

- No source-specific quantized or packed weight format in inspected code.
  External quantized checkpoints would be a loader/provider contract, not a
  MobileBERT modeling requirement.

## 5. Layer/block breakdown

Embedding path for standard config:

```text
word = Embedding(V, E=128)(input_ids)
tri = concat(pad(word[:,1:]), word, pad(word[:,:-1]))  # [B,S,384]
x = Linear(384 -> H=512)(tri)
x = x + PositionEmbedding(max_pos, H)(position_ids)
x = x + TokenTypeEmbedding(type_vocab, H)(token_type_ids)
x = NoNorm_or_LayerNorm(H)(x)
```

Encoder block, repeated `N=24` times for the standard checkpoint:

```text
b_in = NoNorm_or_LayerNorm(Linear(H -> T)(x))
qk_in = NoNorm_or_LayerNorm(Linear(H -> T)(x))  # if key_query_shared_bottleneck
q = Linear(T -> T)(qk_in) -> [B,A,S,D]
k = Linear(T -> T)(qk_in) -> [B,A,S,D]
v = Linear(H -> T)(x) -> [B,A,S,D]              # standard use_bottleneck_attention=false
a = Attention(q, k, v, bidirectional_mask)
a = NoNorm_or_LayerNorm(Linear(T -> T)(a) + b_in)

repeat num_feedforward_networks - 1 times:
  a = NoNorm_or_LayerNorm(Linear(I -> T)(act(Linear(T -> I)(a))) + a)

z = act(Linear(T -> I)(a))
z = NoNorm_or_LayerNorm(Linear(I -> T)(z) + a)
x = NoNorm_or_LayerNorm(Linear(T -> H)(z) + x)
```

Heads:

```text
pool = hidden[:,0]                         # if classifier_activation=false
pool = tanh(Linear(H -> H)(hidden[:,0]))    # if classifier_activation=true
sequence logits = Linear(H -> num_labels)(dropout(pool))
token logits = Linear(H -> num_labels)(dropout(sequence))
qa logits = Linear(H -> 2)(sequence); split last dim
nsp logits = Linear(H -> 2)(pool)
```

MLM head:

```text
h = LayerNorm(act(Linear(H -> H)(sequence)))
packed_decoder = concat([word_embedding.T, dense_vocab_tail], axis=0)  # [H,V]
logits = h @ packed_decoder + decoder_bias
```

The tied-weight contract is `cls.predictions.decoder.weight` aliases
`mobilebert.embeddings.word_embeddings.weight`, while the tail dense weight
covers `hidden_size - embedding_size` additional rows.

## 6. Attention requirements

- Attention is encoder-style, noncausal, self-attention.
- Standard shape: Q/K/V `[B,4,S,32]`, scores `[B,4,S,S]`, output `[B,S,128]`.
- Query/key input width is `T=128`. Value input width is `H=512` unless
  `use_bottleneck_attention=true`, in which case it is also `T=128`.
- Masking: `create_bidirectional_mask` accepts padding mask `[B,S]` or prepared
  4D mask `[B,1,Q,K]`; eager attention adds it to scores before softmax.
- Packed/varlen support is not model-specific. It can be introduced only behind
  attention-provider admission with mask equivalence tests.
- No sliding window, local attention, RoPE, ALiBi, relative bias, cross-attn, or
  KV cache is required for the primary target.
- SDPA/Flash compatibility: source declares support, but the first DinoML
  parity target should preserve eager math order. A later noncausal attention
  provider can consume the same Q/K/V ABI.

## 7. Position encoding and custom math

Position encoding is learned absolute embedding lookup.

The custom MobileBERT embedding trigram path is the important math to reproduce:

```python
def mobilebert_trigram(word):  # word: [B, S, E]
    left = pad(word[:, 1:], right_seq=1, value=0.0)
    center = word
    right = pad(word[:, :-1], left_seq=1, value=0.0)
    return concat([left, center, right], dim=-1)
```

The source pad ordering means token `i` receives embeddings from `i+1`, `i`,
and `i-1`, with zeros at sequence boundaries. Position IDs can be precomputed
for fixed maximum length; token type IDs and attention masks are input-dependent.

`NoNorm` is another source-specific primitive:

```python
def no_norm(x, weight, bias):
    return x * weight + bias
```

## 8. Preprocessing and input packing

CPU/data pipeline:

- BERT WordPiece tokenizer, including normalizer/lowercasing behavior inherited
  from BERT tokenizer files.
- Build `[CLS] A [SEP]` or `[CLS] A [SEP] B [SEP]`.
- Build segment/token type IDs aligned to sequence pairs.
- Pad/truncate to `S <= max_position_embeddings` unless a checkpoint-specific
  serving policy imposes a smaller max length.

GPU/runtime graph:

- Validate exactly one of `input_ids` or `inputs_embeds`.
- If `input_ids`: gather word embeddings and run trigram/projection path.
- If `inputs_embeds`: source expects last dim `embedding_size`; the same trigram
  and embedding projection behavior still applies.
- Create or accept bidirectional additive attention mask.
- Run encoder and selected head.

## 9. Graph rewrite / lowering opportunities

### Rewrite: trigram embedding gather -> fused shifted gather

Source pattern:

```text
Embedding(input_ids) -> pad/shift left/right -> concat -> Linear(3E -> H)
```

Replacement:

```text
FusedTrigramEmbedding(input_ids, word_table) -> GEMM_RCR_Bias(3E -> H)
```

Preconditions:

- `trigram_input=true`.
- Word embedding layout is `[V,E]`.
- Boundary padding is exact zeros.
- Concatenation order is `[next_token, current_token, previous_token]`.

Failure cases:

- Caller supplies `inputs_embeds`; fused ID gather is not applicable.
- `trigram_input=false`; use ordinary embedding/projection guard.

Parity test sketch:

- Compare first/middle/last tokens for short sequences against source pad/cat
  behavior.

### Rewrite: NoNorm affine -> elementwise affine

Source pattern:

```text
NoNorm(x) = x * weight + bias
```

Replacement:

```text
FusedElementwiseAffine(last_axis)
```

Preconditions:

- `normalization_type="no_norm"`.
- Broadcast only along the final feature axis.

Failure cases:

- `normalization_type="layer_norm"` must use true LayerNorm.

Parity test sketch:

- Compare all NoNorm sites with nontrivial learned weight/bias.

### Rewrite: bottleneck linear family -> GEMM_RCR_Bias

Source pattern:

```text
Linear(..., bias=True)` over `[B,S,K]`
```

Replacement:

```text
Flatten B*S -> GEMM_RCR_Bias -> reshape [B,S,N]
```

Preconditions:

- Dense row-major logical layout.
- Last axis is contiguous or layout metadata admits equivalent GEMM.

Shape equations:

- Standard bottleneck: `M=B*S`, `K=512`, `N=128`.
- FFN up: `M=B*S`, `K=128`, `N=512`.
- FFN down: `M=B*S`, `K=512`, `N=128`.

Failure cases:

- Layout pass must not silently translate attention axes after reshape/transpose.

Parity test sketch:

- One-layer parity with all bottleneck flags and tiny/random dimensions.

### Rewrite: Q/K shared bottleneck CSE

Source pattern:

```text
qk_in = BottleneckLayer(hidden)
q = Linear(T -> T)(qk_in)
k = Linear(T -> T)(qk_in)
```

Replacement:

```text
compute qk_in once -> two GEMMs or packed QK GEMM
```

Preconditions:

- `use_bottleneck=true`.
- `use_bottleneck_attention=false`.
- `key_query_shared_bottleneck=true`.

Failure cases:

- If `key_query_shared_bottleneck=false`, Q/K source tensors are full hidden
  states in inspected source, not the shared bottleneck tensor.

Parity test sketch:

- Compare Q and K inputs for configs toggling both bottleneck flags.

### Rewrite: packed Q/K/V projection

Source pattern:

```text
q = Linear(T -> T)(q_input)
k = Linear(T -> T)(k_input)
v = Linear(H or T -> T)(v_input)
```

Replacement:

```text
Packed QKV only when q_input, k_input, and v_input are the same tensor and K dims match
```

Preconditions:

- Usually only safe for `use_bottleneck_attention=true`, where bottleneck returns
  the same tensor four times.
- Equal input width for Q/K/V.

Failure cases:

- Standard checkpoint has V projected from `hidden_size=512`, while Q/K are
  projected from `T=128`; do not pack into one GEMM.

Parity test sketch:

- Config sweep for standard and `use_bottleneck_attention=true`.

### Rewrite: MLM decoder packing

Source pattern:

```text
h @ concat([decoder.weight.T, dense.weight], dim=0) + decoder.bias
```

Replacement:

```text
Materialize logical LM matrix [H,V] with tied first E rows and tail dense rows
```

Preconditions:

- Preserve alias between `decoder.weight` and word embeddings.
- Tail dense has shape `[H-E, V]` in logical matmul orientation.

Failure cases:

- Resizing token embeddings also resizes the tail dense head in source; loader
  must keep vocab-sized structures coherent.

Parity test sketch:

- Compare logits and alias metadata before and after vocab resize where admitted.

## 10. Kernel fusion candidates

Highest priority:

- GEMM_RCR_Bias for bottleneck-heavy linear layers; MobileBERT has many small
  `128 <-> 512` projections.
- NoNorm affine plus residual add where `normalization_type=no_norm`.
- Noncausal attention provider for `[B,4,S,32]`; attention is small per head but
  repeated 24 times.
- Trigram embedding shift/concat/projection to avoid temporary `[B,S,384]`.

Medium priority:

- ReLU/GELU activation fusion in FFN and MLM transform.
- Shared bottleneck CSE and packed Q/K where flags permit.
- Pooler select+dense+tanh for classification checkpoints with activation on.
- QA split/squeeze fused head epilogue.

Lower priority:

- Full-vocab MLM projection optimizations and masked-position-only logits.
- FlashAttention-specific path after eager/SDPA parity.
- Training losses and dropout.

## 11. Runtime staging plan

Stage 1: Parse `MobileBertConfig`, derive `true_hidden_size`, and reject
unsupported bottleneck flag combinations explicitly.

Stage 2: Load embeddings and run embedding parity, including trigram input,
default token type IDs, position IDs, and `NoNorm`.

Stage 3: Implement one MobileBERT layer with standard flags:
`use_bottleneck=true`, `use_bottleneck_attention=false`,
`key_query_shared_bottleneck=true`, `num_feedforward_networks=4`.

Stage 4: Full encoder parity for `google/mobilebert-uncased`.

Stage 5: Add pooler and task heads: sequence classification, token
classification, QA, NSP, then pretraining/MLM.

Stage 6: Add alternate flag support: no bottleneck, bottleneck attention,
unshared Q/K bottleneck, `normalization_type=layer_norm`, and non-ReLU
activations.

Stage 7: Add optimized rewrites/fusions: trigram fused gather, NoNorm residual
affine, attention provider, and guarded Q/K/V packing.

## 12. Parity and validation plan

- Config parser tests for derived `true_hidden_size` and omitted defaults.
- Trigram embedding parity for short sequences and boundary padding.
- NoNorm and LayerNorm parity with eps `1e-12`.
- Single bottleneck layer parity under standard flags.
- Flag sweep parity for `use_bottleneck_attention`,
  `key_query_shared_bottleneck`, `use_bottleneck=false`, and
  `normalization_type=layer_norm`.
- Full encoder parity on `google/mobilebert-uncased`; note Transformers test
  comments that outputs can span very large magnitudes, so relative checks are
  more useful than pure absolute checks for fp32.
- Head parity for QA, sequence classification, token classification, NSP, and
  MLM.
- Suggested tolerances: fp32 relative `1e-4` with bounded absolute fallback;
  fp16/bf16 should start at `rtol=2e-2, atol=2e-2` after attention/normalization
  backend choices are fixed.

## 13. Performance probes

- Tokenization and sequence packing throughput.
- Embedding path: ordinary pad/cat temporary vs fused trigram gather.
- Encoder-only latency over `B` and `S <= 512`.
- Per-layer breakdown: bottleneck projections, attention, repeated FFNs, output
  bottleneck.
- Attention backend comparison: eager-equivalent, SDPA, FlashAttention if dtype
  and mask constraints fit.
- Small GEMM provider efficiency for `M=B*S`, `K/N` in `{128,512}`.
- NoNorm/residual bandwidth versus true LayerNorm variants.
- Head probes: QA/token heads vs full MLM vocab head; masked-position-only MLM
  when serving fill-mask.

## 14. Skip/defer list

- Training, gradient checkpointing, dropout behavior, and loss computation.
- Vocab resizing at runtime.
- Quantized checkpoint loading unless a specific checkpoint is targeted.
- FlashAttention as a requirement; SDPA/eager parity comes first.
- Packed/varlen attention metadata.
- Multi-GPU tensor parallel.
- Nonstandard remote-code variants; none were needed for inspected configs.

## 15. Final implementation checklist

- [ ] Parse `MobileBertConfig` and derive `true_hidden_size`.
- [ ] Load tied word embeddings, position embeddings, token type embeddings, and
  MobileBERT MLM tail dense weights.
- [ ] Implement trigram embedding shift/concat/projection.
- [ ] Implement `NoNorm` affine and `LayerNorm` fallback.
- [ ] Implement standard bottleneck encoder layer.
- [ ] Implement bidirectional MHA with `true_hidden_size` head geometry.
- [ ] Add guards for bottleneck flags and `num_feedforward_networks`.
- [ ] Implement pooler with `classifier_activation` guard.
- [ ] Implement QA, sequence classification, token classification, NSP, and MLM
  heads.
- [ ] Add parity tests for embedding, one block, full encoder, and heads.
- [ ] Add guarded rewrites for trigram gather, bottleneck GEMMs, Q/K sharing, and
  MLM decoder packing.
- [ ] Benchmark embedding, encoder, attention, FFN, NoNorm/LayerNorm, and heads.
