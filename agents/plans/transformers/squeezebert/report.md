# SqueezeBERT DinoML Audit

## 1. Source basis

Transformers commit/version:
`b75feb2af64c3e29cbbc1bd859958c5432cc7ed4` from local checkout `transformers`.

Model id:
Primary source/config target is `squeezebert/squeezebert-uncased`; representative task configs include `squeezebert/squeezebert-mnli`, `squeezebert/squeezebert-mnli-headless`, `mrm8488/squeezebert-finetuned-squadv2`, and tiny random SqueezeBERT configs.

Config source:
Local `SqueezeBertConfig` plus HF `config.json` snapshots under `agents/plans/transformers/squeezebert/_sources/`.

Source files inspected:

- `transformers/src/transformers/models/squeezebert/configuration_squeezebert.py`
- `transformers/src/transformers/models/squeezebert/modeling_squeezebert.py`
- `transformers/src/transformers/models/squeezebert/tokenization_squeezebert.py`
- `transformers/tests/models/squeezebert/test_modeling_squeezebert.py`

Any missing files or assumptions:
No modular source file exists for this family in the inspected checkout. Tokenization is an alias to BERT tokenization. This report targets inference/eval; dropout and losses are training/eval-head concerns, not first-runtime graph requirements. No DinoML tests/imports were run.

## 2. High-level architecture

SqueezeBERT is a text-only bidirectional encoder. The public ABI resembles BERT, but the encoder body is channels-first NCW with grouped `Conv1d(kernel_size=1)` projections instead of ordinary `nn.Linear` modules.

```text
BERT tokenizer -> input_ids/token_type_ids/attention_mask
-> word + position + token-type embeddings
-> embedding LayerNorm
-> [B,S,C] to [B,C,S]
-> N encoder modules with grouped pointwise Conv1d Q/K/V, dense self-attention, grouped pointwise FFN
-> [B,C,S] to [B,S,C]
-> pooler and/or task head
```

Stage decomposition:

- CPU/data pipeline: BERT-style WordPiece tokenization, special-token packing, padding, segment IDs.
- GPU/runtime encoder: embeddings, LayerNorm, NCW pointwise grouped conv projections, noncausal self-attention, residuals, channel LayerNorm.
- Independently stageable heads: pooler + sequence classification, MLM transform + tied decoder, token classification, QA span logits, multiple-choice flatten/reshape.

First useful DinoML target:
`SqueezeBertModel` plus `SqueezeBertForSequenceClassification` for MNLI-style inference. MLM and QA are important follow-ups because they exercise tied embedding/output projection and split/squeeze head behavior.

## 3. Important config dimensions

Source defaults:

| Field | Default | Runtime meaning |
| --- | ---: | --- |
| `vocab_size` | 30522 | Source default table size; official checkpoints use 30528. |
| `embedding_size` | 768 | Embedding output width. Encoder asserts this equals `hidden_size`. |
| `hidden_size` | 768 | Encoder channel count `C`. |
| `num_hidden_layers` | 12 | Repeated SqueezeBERT modules. |
| `num_attention_heads` | 12 | MHA heads; `head_dim = hidden_size / heads = 64` for official configs. |
| `intermediate_size` | 3072 | FFN expansion channel count. |
| `hidden_act` | `gelu` | FFN and MLM transform activation. |
| `max_position_embeddings` | 512 | Learned absolute position table. |
| `type_vocab_size` | 2 | Segment/token-type table. |
| `layer_norm_eps` | `1e-12` | Embedding/head LayerNorm and NCW channel LayerNorm. |
| `q_groups/k_groups/v_groups` | 4/4/4 | Grouped Q/K/V pointwise conv groups. |
| `post_attention_groups` | 1 | Attention output pointwise conv groups. |
| `intermediate_groups` | 4 | FFN expansion pointwise conv groups. |
| `output_groups` | 4 | FFN contraction pointwise conv groups. |
| `tie_word_embeddings` | true | MLM decoder weight tied to word embedding by model utility. |

Representative checkpoint sweep:

| Model id | Architecture/task from config or repo | Layers | Hidden | Heads | Intermediate | Vocab | Groups `(q,k,v,post,inter,out)` | Notable fields |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `squeezebert/squeezebert-uncased` | base encoder / fill-mask capable | 12 | 768 | 12 | 3072 | 30528 | `4,4,4,1,4,4` | Omits `layer_norm_eps`, `pad_token_id`, `tie_word_embeddings`; source defaults apply. |
| `squeezebert/squeezebert-mnli` | sequence classification, `num_labels=3` | 12 | 768 | 12 | 3072 | 30528 | `4,4,4,1,4,4` | Integration test checks logits shape `[1,3]`. |
| `squeezebert/squeezebert-mnli-headless` | encoder/headless MNLI weights | 12 | 768 | 12 | 3072 | 30528 | `4,4,4,1,4,4` | No classifier-specific fields in config. |
| `mrm8488/squeezebert-finetuned-squadv2` | `SqueezeBertForQuestionAnswering` | 12 | 768 | 12 | 3072 | 30528 | `4,4,4,1,4,4` | Explicit `layer_norm_eps=1e-12`, `pad_token_id=0`. |
| `hf-tiny-model-private/tiny-random-SqueezeBertModel` | tiny feature extraction | 5 | 32 | 4 | 64 | 1124 | `2,2,2,2,4,1` | Varies group pattern; includes historical ignored `attention_dropout`. |
| `hf-tiny-model-private/tiny-random-SqueezeBertForTokenClassification` | tiny token classification | 5 | 32 | 4 | 64 | 1124 | `2,2,2,2,4,1` | Same operator-shape variation as tiny base. |

## 3a. Family variation traps

- `embedding_size != hidden_size` is not supported by current source: `SqueezeBertEncoder` asserts equality and says an adapter Conv1d would be needed.
- Group counts are config-driven per projection family. DinoML must validate `in_channels % groups == 0` and `out_channels % groups == 0`.
- Tiny configs prove `post_attention_groups` can be `2` and `output_groups` can be `1`; do not hardcode the official `1/4` feed-forward group pattern.
- Official checkpoint configs omit source defaults such as `layer_norm_eps`, `pad_token_id`, and `tie_word_embeddings`; these are effective defaults, not absent behavior.
- `attention_dropout` appears in test/tiny configs but the inspected source reads `attention_probs_dropout_prob`, not `attention_dropout`.
- Source layout is axis-sensitive: embeddings and heads use `[B,S,C]`, while the encoder body uses `[B,C,S]`. A layout pass needs explicit guards at the encoder entry/exit and around LayerNorm axes.
- No causal mask, KV cache, RoPE, ALiBi, cross-attention, MQA/GQA, or local attention exists in the inspected source.
- MLM weight tying must preserve one logical word embedding table and decoder matrix identity.

## 4. Operator coverage checklist

Tensor/layout ops:

- Embedding lookup for `input_ids`, `position_ids`, and `token_type_ids`.
- Broadcast/default creation for token types and attention masks.
- Additive embedding sum and residual adds.
- `permute(0,2,1)` at encoder entry/exit and inside NCW LayerNorm.
- `view`, `reshape`, `contiguous`, `split`, `squeeze`, first-token slice `hidden_states[:,0]`.
- Multiple-choice flatten `[B, choices, S] -> [B*choices, S]` and logits reshape `[B*choices,1] -> [B, choices]`.

Neural network primitives:

- `LayerNorm(C, eps=1e-12)` on `[B,S,C]`.
- NCW channel LayerNorm implemented as `[B,C,S] -> [B,S,C] -> LayerNorm(C) -> [B,C,S]`.
- Pointwise `Conv1d(kernel_size=1)` with groups:
  - Q/K/V official: `Conv1d(768 -> 768, groups=4)`.
  - Post-attention official: `Conv1d(768 -> 768, groups=1)`.
  - Intermediate official: `Conv1d(768 -> 3072, groups=4)`.
  - Output official: `Conv1d(3072 -> 768, groups=4)`.
  - Tiny variation: `32 -> 32` with groups 2, `32 -> 64` with groups 4, `64 -> 32` with groups 1.
- Dense `Linear(768 -> 768)` pooler and MLM transform.
- Dense task heads: `Linear(768 -> num_labels)`, `Linear(768 -> 1)`, `Linear(768 -> 2)` for QA default.
- Tied MLM decoder: `Linear(hidden_size -> vocab_size, bias=true)` with decoder weight tied to word embedding.
- GELU activation, tanh pooler activation.

Attention primitives:

- Encoder self-attention only, noncausal MHA.
- Batched matmul `q @ k` where q is `[B,H,S,D]`, k is `[B,H,D,S]`.
- Scale by `1 / sqrt(head_dim)`.
- Additive extended attention mask broadcast to `[B,1,1,S]`.
- Softmax over source sequence dimension `dim=-1`.
- Batched matmul `attention_probs @ v`, with v `[B,H,S,D]`.

Position encoding:

- Learned absolute position embedding table `[max_position_embeddings, embedding_size]`.
- Default `position_ids = arange(max_position_embeddings)[None, :seq_len]`.

Preprocessing-coupled ops:

- BERT tokenizer-compatible `input_ids`, segment IDs, attention mask, special token layout.
- No model-owned preprocessing beyond ID tensors.

Quantized/packed/distributed ops:

- None in native source. ONNX/transformers.js mirrors are out of scope for the native PyTorch source basis.

Current DinoML coverage implication:

- GEMM/BMM and common elementwise ops have bounded support.
- Public `Conv1d`/grouped pointwise conv, mature LayerNorm, BERT embeddings helper, and production fused attention are gated gaps for faithful SqueezeBERT import.
- Grouped pointwise conv can be lowered to guarded grouped GEMM slices, but that still needs explicit operator/provider admission.

## 5. Layer/block breakdown

Embeddings:

```text
word = Embedding(vocab_size, C)(input_ids)
pos = Embedding(max_position_embeddings, C)(position_ids)
seg = Embedding(type_vocab_size, C)(token_type_ids)
x = LayerNorm(word + pos + seg)
x = Dropout(x)       # disabled in eval
```

Encoder module, repeated `num_hidden_layers` times:

```text
# input x: [B,C,S]
q0 = Conv1d(C -> C, kernel=1, groups=q_groups)(x)
k0 = Conv1d(C -> C, kernel=1, groups=k_groups)(x)
v0 = Conv1d(C -> C, kernel=1, groups=v_groups)(x)

q = view(q0, [B,H,D,S]).permute(0,1,3,2)  # [B,H,S,D]
k = view(k0, [B,H,D,S])                   # [B,H,D,S]
v = view(v0, [B,H,D,S]).permute(0,1,3,2)  # [B,H,S,D]

score = (q @ k) / sqrt(D) + mask           # [B,H,S,S]
prob = softmax(score, dim=-1)
ctx = prob @ v                             # [B,H,S,D]
attn = ctx.permute(0,1,3,2).contiguous().view(B,C,S)

y = Conv1d(C -> C, groups=post_attention_groups)(attn)
y = LayerNorm_NCW(y + x)

z = gelu(Conv1d(C -> intermediate_size, groups=intermediate_groups)(y))
out = Conv1d(intermediate_size -> C, groups=output_groups)(z)
x = LayerNorm_NCW(out + y)
```

Pooler:

```text
pooled = tanh(Linear(C -> C)(last_hidden_state[:, 0]))
```

Heads:

- Masked LM: `Linear(C -> C) -> GELU -> LayerNorm(C) -> tied Linear(C -> vocab_size) + bias`.
- Sequence classification: pooled output, dropout, `Linear(C -> num_labels)`.
- Multiple choice: flatten choices into batch, pooled output, `Linear(C -> 1)`, reshape.
- Token classification: per-token `Linear(C -> num_labels)`.
- QA: per-token `Linear(C -> num_labels)`, split start/end, squeeze last dim.

## 6. Attention requirements

SqueezeBERT attention is encoder-only noncausal self-attention.

| Requirement | Source behavior |
| --- | --- |
| Causal? | No. Bidirectional encoder attention. |
| Attention type | Self-attention only. No cross-attention. |
| Head pattern | MHA. No MQA/GQA. |
| Head count / dim | Official 12 heads, head dim 64. Tiny 4 heads, head dim 8. |
| Q/K/V width | All equal to hidden channel count `C`. |
| Query/key lengths | Square self-attention `S x S`. |
| Masking | Extended additive mask from `get_extended_attention_mask`, added before softmax. |
| Packed/varlen | None in source. Padded dense batches only. |
| Sliding/local/block attention | None. |
| KV cache | None. This is not an autoregressive decoder. |
| FlashAttention/SDPA | Source does not dispatch to SDPA; it uses explicit matmul/softmax/matmul wrappers. A fused backend is an optimization with strict additive-mask parity. |

Output attentions:
The source returns pre-softmax `attention_score` when `output_attentions=True`, not the post-softmax probabilities. First DinoML integration can reject or defer attention outputs if hidden-state/logit parity is the target.

## 7. Position encoding and custom math

Position encoding is simple learned absolute embedding lookup.

```python
def squeezebert_default_position_ids(max_position_embeddings, seq_len):
    return torch.arange(max_position_embeddings).expand((1, -1))[:, :seq_len]
```

Custom math worth reproducing exactly is the NCW channel LayerNorm:

```python
def squeezebert_layer_norm_ncw(x, weight, bias, eps):
    # x: [B, C, S]; normalize across C for each token position.
    y = x.permute(0, 2, 1)
    y = layer_norm(y, normalized_shape=C, weight=weight, bias=bias, eps=eps)
    return y.permute(0, 2, 1)
```

Everything else is standard BERT-style learned embeddings, GELU, tanh, matmul scaling, and additive mask softmax.

## 8. Preprocessing and input packing

Tokenizer:
`SqueezeBertTokenizer` is an alias for `BertTokenizer` in the inspected source. Runtime inputs are ordinary BERT-style tensors:

- `input_ids`: `[B,S]`, integer token IDs.
- `attention_mask`: `[B,S]`, optional; source defaults to ones.
- `token_type_ids`: `[B,S]`, optional; source defaults to zeros.
- `position_ids`: `[1,S]` or `[B,S]`, optional; source defaults to first `S` rows from a registered arange buffer.
- `inputs_embeds`: optional alternative `[B,S,C]`; mutually exclusive with `input_ids`.

CPU/data pipeline:
Own WordPiece tokenization, truncation, padding, and special-token packing outside the GPU graph for first integration.

GPU/runtime:
Own embedding lookup, default token types and positions where the caller omits them, attention mask conversion to additive broadcast form, and the encoder/head graph.

No multimodal placeholder scatter, image/audio/video packing, OCR/layout boxes, or packed sequence descriptors are present.

## 9. Graph rewrite / lowering opportunities

### Rewrite: pointwise Conv1d groups=1 to GEMM

Source pattern:
`nn.Conv1d(cin, cout, kernel_size=1, groups=1)` on `[B,Cin,S]`.

Replacement:

```text
Transpose/view to tokens X: [B*S, Cin]
Y = X @ W.T + bias
Reshape/transpose to [B,Cout,S]
```

Preconditions:

- `kernel_size == 1`, `stride == 1`, `padding == 0`, `dilation == 1`.
- `groups == 1`.
- Input is dense contiguous or has a validated layout view.
- Conv weight layout is `[cout, cin, 1]`; use `W2 = weight[:, :, 0]`.

Shape equations:
`X[B,S,Cin] -> Y[B,S,Cout]`, then NCW if the surrounding region stays channels-first.

Failure cases:
Any grouped conv, non-unit kernel/stride/padding/dilation, or unexpected NCW strides should reject or fall back to a Conv1d provider.

Parity test sketch:
Compare the post-attention official `Conv1d(768 -> 768, groups=1)` and tiny output `Conv1d(64 -> 32, groups=1)` against PyTorch for random `[B,C,S]`.

### Rewrite: grouped pointwise Conv1d to grouped GEMM family

Source pattern:
`nn.Conv1d(cin, cout, kernel_size=1, groups=G)` on `[B,Cin,S]`.

Replacement:

```text
For each group g:
  Xg = X[:, :, g*Cin/G:(g+1)*Cin/G]          # [B,S,Cin/G]
  Yg = Xg @ Wg.T + bg                        # [B,S,Cout/G]
Concat Yg along channel dimension
```

Weight transform:

```python
w = conv.weight[:, :, 0]                     # [Cout, Cin/G]
w_g = w.view(G, Cout // G, Cin // G)
b_g = conv.bias.view(G, Cout // G)
```

Exact preconditions:

- `cin % G == 0`, `cout % G == 0`.
- Same Conv1d unit-kernel constraints as above.
- Channel grouping is contiguous in PyTorch Conv1d order.
- Either lower to a first-class grouped pointwise provider or emit an artifact-visible sequence of split/GEMM/concat nodes; do not hide block-diagonal expansion in a dense weight unless memory blowup is explicitly accepted.

Failure cases:
Dynamic `cin/cout/groups`, non-contiguous group slicing, or missing concat/split support.

Parity test sketch:
Use official Q/K/V `768 -> 768, G=4`, official intermediate `768 -> 3072, G=4`, official output `3072 -> 768, G=4`, and tiny `32 -> 32, G=2`.

### Rewrite: NCW LayerNorm to native channel LayerNorm

Source pattern:
`permute(0,2,1) -> LayerNorm(C) -> permute(0,2,1)`.

Replacement:
Single `layernorm_axis(x, axis=1)` or a specialized NCW channel-LayerNorm kernel.

Preconditions:

- Input rank is exactly 3 `[B,C,S]`.
- Normalized shape equals `C`.
- No consumer observes the intermediate NWC tensor.
- Epsilon and affine parameters match source.

Failure cases:
Generalizing this to arbitrary rank/axis without tests, or applying a layout pass that normalizes over `S` instead of `C`.

Parity test sketch:
Random NCW tensors with official `C=768` and tiny `C=32`, compare against source subclass.

### Rewrite: encoder NCW island with boundary elision

Source pattern:

```text
embedding [B,S,C] -> permute [B,C,S] -> all encoder modules -> permute [B,S,C]
```

Replacement:
Keep an internal NCW layout island or rewrite the whole encoder to BSC grouped pointwise GEMMs and fused attention.

Preconditions:

- The island begins after embedding LayerNorm and ends before pooler/head.
- Every internal op has an axis-aware lowering: grouped pointwise conv, attention reshape, NCW LayerNorm.
- Hidden-state/attention-output optional returns either materialize source layout or are rejected.

Failure cases:
`output_hidden_states=True` requires hidden states in `[B,S,C]` between layers; a purely internal NCW island must insert materialization for that debug ABI or reject it.

### Rewrite: tied MLM decoder projection

Source pattern:
`cls.predictions.decoder.weight` tied to `transformer.embeddings.word_embeddings.weight`, plus separate decoder/bias alias handling.

Replacement:
Use one logical constant table for embedding rows and LM projection.

Preconditions:

- `tie_word_embeddings=true` and source weight tying metadata is honored.
- Projection uses `embedding_weight.T`.
- Decoder bias remains separate and resize-compatible.

Failure cases:
Untied or resized embeddings without preserving decoder bias and alias metadata.

## 10. Kernel fusion candidates

Highest priority:

- Grouped pointwise Conv1d provider or grouped-GEMM rewrite. This is the family-defining ABI and blocks encoder parity.
- NCW channel LayerNorm. Every encoder module uses it twice.
- Encoder attention fusion for dense noncausal self-attention with additive mask. The eager chain is matmul/scale/add/softmax/matmul and dominates long sequence work.

Medium priority:

- Conv1d + residual + LayerNorm fusion for `ConvDropoutLayerNorm` in eval mode.
- Conv1d + GELU fusion for the intermediate FFN expansion.
- Boundary-elided NCW encoder island to avoid repeated permute materialization.
- Pooler/classifier GEMM fusion for sequence-classification throughput.

Lower priority:

- MLM transform + tied projection optimization, because first target can be sequence classification.
- Multiple-choice flatten/reshape specialization.
- Attention-output materialization support for debug APIs.

## 11. Runtime staging plan

Stage 1: Parse SqueezeBERT config and reject unsupported variants clearly:
`embedding_size != hidden_size`, invalid group divisibility, unsupported output attentions/hidden states if not materialized.

Stage 2: Load weights for embeddings, grouped Conv1d modules, LayerNorms, pooler, and sequence-classification head. Preserve MLM tied embedding metadata even if MLM is deferred.

Stage 3: Implement one encoder module parity using explicit operators: embedding output input, grouped pointwise Conv1d, dense attention, NCW LayerNorm.

Stage 4: Full `SqueezeBertModel` hidden-state parity with base and tiny configs.

Stage 5: Add `SqueezeBertForSequenceClassification` MNLI-style inference.

Stage 6: Add QA and token-classification heads, then MLM tied-head parity.

Stage 7: Add grouped pointwise and attention fusions, then layout-island optimization.

Can be stubbed initially:
Training losses, dropout behavior in train mode, output attentions, output hidden states, runtime embedding resize, multiple-choice if sequence classification is the first target.

## 12. Parity and validation plan

- Config parser tests for official and tiny group patterns.
- Unit parity for grouped pointwise Conv1d:
  - `768 -> 768, G=4`
  - `768 -> 3072, G=4`
  - `3072 -> 768, G=4`
  - `32 -> 64, G=4`
  - `64 -> 32, G=1`
- NCW LayerNorm parity against the source subclass.
- Attention parity for one layer with fixed additive masks, including padded tokens.
- Single-block parity with random weights and tiny source config.
- Full encoder parity for tiny random config.
- Sequence-classification parity against `squeezebert/squeezebert-mnli`; Transformers integration test expected output shape is `[1,3]`.
- QA head parity against an SQuAD-style config: logits split/squeeze produce two `[B,S]` tensors.
- MLM parity: tied word embedding and decoder projection share one logical table.

Suggested tolerances:
fp32 initial parity `rtol=1e-4, atol=1e-4`; fp16/bf16 after fused kernels can start at `rtol=1e-2, atol=1e-2`, with tighter checks around LayerNorm and softmax once kernels are stable.

## 13. Performance probes

- Encoder-only throughput over `(B,S)` sweep: `B={1,8,32}`, `S={32,128,512}`.
- Grouped pointwise Conv1d rewrite comparison: native Conv1d provider vs split-GEMM vs block-diagonal dense GEMM.
- Attention backend comparison: eager BMM/softmax/BMM vs fused noncausal attention.
- Layout probe: materialized `[B,S,C] <-> [B,C,S]` permutes vs NCW island vs all-BSC grouped GEMM lowering.
- LayerNorm kernel probe for NCW channel normalization.
- Head-specific probes: MNLI classifier latency, token-classifier throughput, MLM projection bandwidth.
- Weight-load probe: tied MLM projection without cloning embedding table.

## 14. Skip/defer list

Safe to defer for first sequence-classification integration:

- Training losses and train-mode dropout.
- Output attentions and output hidden states, if the runtime target is logits only.
- MLM tied head, unless fill-mask is selected as the first product target.
- QA/token/multiple-choice heads, after the base encoder and sequence head are validated.
- Runtime resize of token embeddings.
- ONNX, transformers.js, and external quantized mirrors.
- Any `embedding_size != hidden_size` variant; current native source asserts against it.
- General Conv1d beyond `kernel_size=1`; SqueezeBERT only needs pointwise Conv1d.

## 15. Final implementation checklist

- [ ] Parse `SqueezeBertConfig` and apply source defaults for omitted fields.
- [ ] Add admission guards for `embedding_size == hidden_size`, head divisibility, and grouped Conv1d divisibility.
- [ ] Load BERT-style embeddings and default position/token-type IDs.
- [ ] Implement or lower pointwise `Conv1d(kernel=1, groups=G)` on NCW tensors.
- [ ] Implement NCW channel LayerNorm or guarded rewrite to axis LayerNorm.
- [ ] Implement dense encoder self-attention with additive mask and no KV cache.
- [ ] Implement encoder NCW layout island with optional hidden-state materialization policy.
- [ ] Load pooler and sequence-classification head.
- [ ] Preserve MLM tied embedding/decoder alias metadata.
- [ ] Add one-block parity tests for official and tiny group patterns.
- [ ] Add full tiny encoder parity.
- [ ] Add MNLI sequence-classification parity.
- [ ] Add QA/token/MLM head parity as follow-up tasks.
- [ ] Benchmark grouped pointwise provider choices and attention fusion.

Gated gaps for DinoML before production SqueezeBERT:

- First-class or rewritable grouped pointwise Conv1d with artifact-visible provider/lowering.
- Axis-aware LayerNorm, specifically NCW channel LayerNorm.
- BERT embedding/default-ID/mask lowering.
- Dense noncausal encoder attention path with additive mask semantics.
- Layout guards around `[B,S,C]` public ABI and `[B,C,S]` encoder internals.
- Weight alias handling for MLM tied decoder.
