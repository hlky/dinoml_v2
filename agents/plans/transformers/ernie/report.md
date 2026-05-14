# ERNIE Transformers Family Audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: ernie family; representative open configs listed below
Config source: local configuration_ernie.py plus raw Hugging Face config.json files
Source files inspected:
  X:/H/transformers/src/transformers/models/ernie/configuration_ernie.py
  X:/H/transformers/src/transformers/models/ernie/modular_ernie.py
  X:/H/transformers/src/transformers/models/ernie/modeling_ernie.py
  X:/H/transformers/src/transformers/models/bert/modeling_bert.py for inheritance comparison
Any missing files or assumptions:
  No ERNIE tokenizer or processor source lives in this model directory. Tokenization is BERT-style and is treated as CPU/data-pipeline work.
  modeling_ernie.py is generated from modular_ernie.py; future source edits should target modular_ernie.py.
```

Source URLs at the inspected commit:

- [configuration_ernie.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/ernie/configuration_ernie.py)
- [modular_ernie.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/ernie/modular_ernie.py)
- [modeling_ernie.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/ernie/modeling_ernie.py)
- [bert/modeling_bert.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/bert/modeling_bert.py)

Representative configs were fetched from raw Hub URLs and summarized in `config_sweep.json` in this folder. `PaddlePaddle/ernie-3.0-base-zh` returned 401 from the raw config endpoint; the gated/unavailable link is [PaddlePaddle/ernie-3.0-base-zh](https://huggingface.co/PaddlePaddle/ernie-3.0-base-zh). Access would resolve whether that official repo differs from the open `nghuyong/ernie-3.0-base-zh` mirror.

Report runtime target: encoder-only ERNIE base plus masked-LM head. Causal LM, cross-attention decoder mode, and task heads are documented but not first-target requirements.

## 2. High-level architecture

ERNIE in this Transformers directory is a text-only BERT-family encoder with absolute position embeddings, token type embeddings, optional ERNIE task type embeddings, post-attention LayerNorm, and an ungated feed-forward MLP. The main source-owned delta from BERT is `task_type_ids` support when `config.use_task_id=True`.

```text
tokenizer/data pipeline -> input_ids/attention_mask/token_type_ids/task_type_ids
  -> word + position + token type (+ optional task type) embeddings
  -> LayerNorm + dropout
  -> N encoder layers: self-attention -> add+LayerNorm -> MLP -> add+LayerNorm
  -> last hidden state
  -> optional pooler or masked-LM/classification/QA heads
```

Stage decomposition:

- CPU/data pipeline: tokenization, special tokens, segment IDs, optional task IDs, padding mask construction.
- GPU/runtime first target: embedding lookup/add, encoder stack, masked-LM transform and tied decoder logits.
- Independently stageable heads: pooler/sequence classification, token classification, QA split head, NSP, multiple choice.
- Optional decoder stage: `ErnieForCausalLM` can use causal masks and dynamic KV cache if `config.is_decoder=True`; representative open ERNIE configs are encoder or masked-LM oriented.

## 3. Important config dimensions

Source defaults from `ErnieConfig`:

| Field | Default | Runtime meaning |
|---|---:|---|
| `vocab_size` | 30522 | word embedding rows and LM decoder rows |
| `hidden_size` | 768 | model width |
| `num_hidden_layers` | 12 | encoder layer count |
| `num_attention_heads` | 12 | MHA head count |
| `head_dim` | `hidden_size / num_attention_heads` | source computes integer division after divisibility guard |
| `intermediate_size` | 3072 | MLP expansion width |
| `hidden_act` | `gelu` | MLP and LM transform activation through `ACT2FN` |
| `max_position_embeddings` | 512 | absolute position embedding rows |
| `type_vocab_size` | 2 | token/segment type embedding rows |
| `task_type_vocab_size` | 3 | optional ERNIE task embedding rows |
| `use_task_id` | false | gates task type embedding module and add |
| `layer_norm_eps` | 1e-12 | LayerNorm epsilon |
| `use_cache` | true | only effective when `is_decoder=True` |
| `is_decoder` | false | switches bidirectional vs causal self-attention |
| `add_cross_attention` | false | decoder-only optional cross-attention |

Representative checkpoint sweep:

| Model id | Arch | Layers | Hidden | Heads x dim | MLP | Vocab | Max pos | Type vocab | Task IDs | Act | LN eps |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---:|
| `hf-internal-testing/tiny-random-ErnieModel` | `ErnieModel` | 5 | 32 | 4 x 8 | 37 | 1124 | 512 | 16 | no | gelu | 1e-12 |
| `nghuyong/ernie-1.0-base-zh` | `ErnieForMaskedLM` | 12 | 768 | 12 x 64 | 3072 | 18000 | 513 | 2 | no | relu | 1e-5 |
| `nghuyong/ernie-2.0-base-en` | `ErnieModel` | 12 | 768 | 12 x 64 | 3072 | 30522 | 512 | 4 | no | gelu | 1e-5 |
| `nghuyong/ernie-2.0-large-en` | `ErnieModel` | 24 | 1024 | 16 x 64 | 4096 | 30522 | 512 | 4 | no | gelu | 1e-5 |
| `nghuyong/ernie-3.0-base-zh` | `ErnieForMaskedLM` | 12 | 768 | 12 x 64 | 3072 | 40000 | 2048 | 4 | yes, 3 rows | gelu | 1e-5 |

`torch_dtype`, parameter count, and license are not source-derived for this report except where a fetched `config.json` explicitly included `torch_dtype` (`float32` for the tiny random config).

## 3a. Family variation traps

- `hidden_size` must be divisible by `num_attention_heads` unless a nonstandard `embedding_size` attribute exists; current inspected config class does not define `embedding_size`.
- ERNIE 1.0 uses `hidden_act="relu"` and `max_position_embeddings=513`; do not bake in GELU or 512.
- ERNIE 2.0/3.0 examples use `type_vocab_size=4`; BERT assumptions of only two segment IDs are too narrow.
- `use_task_id=True` adds a fourth embedding table and an extra elementwise add. `task_type_ids=None` defaults to zeros.
- `position_embedding_type` appears in the tiny historical config but is not read by the inspected ERNIE source; current runtime should treat it as ignored for this source basis.
- `use_cache=True` in configs is ignored for encoder mode because `ErnieModel.forward` forces `use_cache=False` unless `config.is_decoder=True`.
- `ErnieForCausalLM` exists, but a valid generation target requires `is_decoder=True`; otherwise source logs a warning and the base model remains bidirectional.
- LM decoder weight is tied to `ernie.embeddings.word_embeddings.weight`; lowering must preserve logical aliasing.
- No RoPE, ALiBi, GQA/MQA, sliding window, MoE, quantized packed weights, vision/audio branches, or NHWC/NCHW layout-sensitive operators are present in native ERNIE.

## 4. Operator coverage checklist

Tensor/layout ops:

- Shape extraction for `[B, S]`, `[B, S, H]`, multiple-choice flatten `[B, C, S] -> [B*C, S]`.
- `view`, `reshape`, `transpose(1, 2)`, `transpose(2, 3)`, `contiguous`, `split`, `squeeze`, `slice`, `gather` for buffered token type defaults.
- Broadcast and elementwise add for embedding sums, residuals, and additive attention masks.
- Optional `to(device)` move for `inputs_embeds` parity is a framework/device concern, not a graph math op.

Neural network primitives:

- Embedding lookup: word `[vocab_size, H]`, position `[max_position_embeddings, H]`, token type `[type_vocab_size, H]`, optional task type `[task_type_vocab_size, H]`.
- LayerNorm over last dim `H` with epsilon from config.
- Linear with bias: Q/K/V `H -> H`, attention output `H -> H`, MLP `H -> I -> H`, pooler `H -> H`, LM transform `H -> H`, LM decoder `H -> vocab_size`.
- Activations: GELU and ReLU required by representative configs; Tanh for pooler.
- Dropout is present in source but can be disabled for inference.

Attention primitives:

- Dense MHA self-attention with query/key/value `[B, heads, S, head_dim]`.
- Eager math: `softmax((Q @ K^T) * head_dim^-0.5 + mask) @ V`.
- Backend dispatch through Transformers attention implementations (`eager`, SDPA, FlashAttention, flex attention); DinoML can initially lower source semantics to dense MHA.

Position/custom math:

- Absolute position embedding lookup only. No rotary or relative bias.

Generation/cache ops for optional decoder mode:

- Dynamic self-attention KV cache per layer with stored key/value before attention matmul, shape `[B, heads, T, head_dim]`.
- Optional encoder-decoder cache when cross-attention is enabled.
- `logits_to_keep` slicing before LM head for CausalLM.

Preprocessing-coupled ops:

- Token IDs, attention mask, token type IDs, optional task type IDs, and position IDs. These are rank-2 integer inputs.

Aliasing contract:

- `cls.predictions.decoder.weight` is tied to `ernie.embeddings.word_embeddings.weight`.
- `cls.predictions.decoder.bias` is tied/logically paired with `cls.predictions.bias` in tied weight keys.

## 5. Layer/block breakdown

Embedding block:

```text
input_ids: [B, S]
word = Embedding(vocab_size, H)(input_ids) -> [B, S, H]
position_ids default = arange(max_position_embeddings)[past_len:past_len+S] -> [1, S]
pos = Embedding(max_position_embeddings, H)(position_ids) -> [1 or B, S, H]
token_type_ids default = gathered zero buffer by position_ids -> [B, S]
segment = Embedding(type_vocab_size, H)(token_type_ids) -> [B, S, H]
x = word + segment + pos
if use_task_id: x += Embedding(task_type_vocab_size, H)(task_type_ids or zeros)
x = LayerNorm(x, eps)
```

Encoder block, repeated `num_hidden_layers`:

```text
q = Linear(H -> H, bias=True)(x).view(B, S, heads, D).transpose(1, 2)
k = Linear(H -> H, bias=True)(x).view(B, S, heads, D).transpose(1, 2)
v = Linear(H -> H, bias=True)(x).view(B, S, heads, D).transpose(1, 2)
attn = Attention(q, k, v, additive_mask, scale=D**-0.5)
attn = attn.transpose(1, 2).reshape(B, S, H)
x = LayerNorm(Linear(H -> H, bias=True)(attn) + residual)
mlp = Linear(H -> I, bias=True)(x)
mlp = gelu_or_relu(mlp)
x = LayerNorm(Linear(I -> H, bias=True)(mlp) + residual)
```

Masked LM head:

```text
y = Linear(H -> H, bias=True)(last_hidden)
y = activation(y)
y = LayerNorm(y)
logits = Linear(H -> vocab_size, bias=True, weight tied to word embeddings)(y)
```

Pooler/head variants:

- Pooler takes `last_hidden_state[:, 0]`, then `Linear(H -> H)` and Tanh.
- Sequence classification: dropout, `Linear(H -> num_labels)`.
- Token classification: dropout, `Linear(H -> num_labels)` on each token.
- QA: `Linear(H -> num_labels)`, then `split(1, dim=-1)`, squeeze start/end logits.
- Multiple choice: flatten choices into batch, pool, `Linear(H -> 1)`, reshape `[B, C]`.

## 6. Attention requirements

Primary encoder target:

- Noncausal bidirectional self-attention.
- MHA only; `num_key_value_heads` is absent and KV heads equal query heads.
- Query/key/value widths are all `H`; per-head width is `H / num_attention_heads`.
- Attention mask is created by `create_bidirectional_mask` and consumed as an additive mask broadcastable to attention scores `[B, heads, Q, K]`.
- No packed/varlen, local/sliding, ALiBi, relative bias, RoPE, or cross-attention for first target.
- FlashAttention/SDPA/flex compatibility is advertised by the source class, but parity target should preserve eager math order first.

Optional decoder mode:

- Set `config.is_decoder=True` to use causal self-attention masks.
- Cache stores per-layer K/V after projection and reshape, before score matmul, shape `[B, heads, cached_length, head_dim]`.
- Cross-attention exists only when `config.add_cross_attention=True` and `encoder_hidden_states` is passed. Cross K/V can be cached in `EncoderDecoderCache`.
- This should be a separate admission mode because representative ERNIE configs do not require it.

## 7. Position encoding and custom math

ERNIE uses learned absolute position embeddings. No custom RoPE/ALiBi math is required.

Minimal source-equivalent position path:

```python
def ernie_position_ids(position_ids, max_position_embeddings, seq_length, past_key_values_length=0):
    if position_ids is not None:
        return position_ids
    full = arange(max_position_embeddings).reshape(1, max_position_embeddings)
    return full[:, past_key_values_length : past_key_values_length + seq_length]
```

The position table can be loaded as a constant. Position ID slicing depends on dynamic sequence length and optional decoder cache length.

## 8. Preprocessing and input packing

Runtime tensor ABI:

- `input_ids`: int64-like `[B, S]`.
- `attention_mask`: optional `[B, S]`; source mask helpers convert it to backend-compatible additive/bool mask form.
- `token_type_ids`: optional `[B, S]`; if absent, defaults to zeros gathered from a registered `[1, max_position_embeddings]` buffer using `position_ids`.
- `task_type_ids`: optional `[B, S]`; only consumed when `use_task_id=True`, defaulting to zeros.
- `position_ids`: optional `[1, S]` or `[B, S]`; default is contiguous absolute positions.
- `inputs_embeds`: optional `[B, S, H]`; exactly one of `input_ids` or `inputs_embeds` must be supplied.

Tokenizer ownership remains outside the GPU graph. Special-token layout and vocab files are checkpoint/tokenizer artifacts, not modeled in `src/transformers/models/ernie`. For first integration, DinoML should accept already-tokenized tensors and validate ID ranges against vocab/type/task/position embedding row counts.

No image/audio/video preprocessing, placeholder scatter, packed sequence metadata, OCR/layout boxes, or postprocessing is present.

## 9. Graph rewrite / lowering opportunities

### Rewrite: separate Q/K/V linears -> packed QKV GEMM

Source pattern:

```text
q = Linear(H -> H)(x)
k = Linear(H -> H)(x)
v = Linear(H -> H)(x)
```

Replacement:

```text
qkv = Linear(H -> 3H)(x)
split qkv last dim into [q, k, v]
```

Preconditions:

- Same input tensor `x`.
- All three projections have bias.
- Output widths equal `H`.
- Split order must be source order `[q, k, v]`.

Weight transform:

```python
packed_weight = concat([Wq, Wk, Wv], axis=0)
packed_bias = concat([bq, bk, bv], axis=0)
```

Failure cases: any missing projection, nonstandard width, quantized storage without a defined packing rule, or graph consumers that need individual projection materialization.

Parity sketch: compare packed and unpacked q/k/v tensors before reshape for random `[B, S, H]`.

### Rewrite: inference dropout removal

Source pattern: dropout after embeddings, attention probabilities, attention output, MLP output, and classifier inputs.

Replacement: identity.

Preconditions: `model.eval()` or inference-only artifact. Training and stochastic parity are out of scope.

### Rewrite: Embedding add chain fusion

Source pattern: word + token type + position + optional task type, then LayerNorm.

Replacement: fused gather-add-LayerNorm kernel or embedding-add followed by LayerNorm.

Preconditions: all embedding indices validated in range; task branch selected by `use_task_id`.

Failure cases: `inputs_embeds` path bypasses word embedding but still adds token/position/task embeddings.

### Rewrite: last-token-only CausalLM logits

Source pattern: `hidden_states[:, slice_indices, :]` before LM head.

Replacement: slice hidden states before LM transform/decoder to reduce logits work.

Preconditions: CausalLM target, `logits_to_keep` is int or explicit tensor indices, loss is not being computed. Not needed for masked-LM first target.

### Rewrite: tied LM decoder as embedding GEMM

Source pattern: decoder weight aliases word embedding.

Replacement: use word embedding matrix as LM projection RHS.

Preconditions: tied weights preserved during load; bias added once. Do not duplicate or transpose inconsistently.

## 10. Kernel fusion candidates

Highest priority:

- LayerNorm + residual around attention and MLP outputs, because every layer has two post-norm residual blocks.
- QKV packed projection plus reshape/transposes, because it removes three small GEMM launches per layer.
- Dense MHA prefill for bidirectional encoder, because encoder throughput is dominated by attention and GEMMs.
- GELU/ReLU MLP activation fused with first or second MLP GEMM epilogue where available.

Medium priority:

- Embedding gather/add + LayerNorm for ERNIE 3.0 with optional task type add.
- LM head transform fusion: `Linear -> activation -> LayerNorm -> tied decoder`.
- Pooler `slice[:,0] -> Linear -> Tanh` for classification-heavy deployments.

Lower priority:

- Decoder KV cache and cross-attention cache kernels, because native ERNIE checkpoints are primarily encoder/masked-LM.
- Multiple-choice flatten/reshape specialization.
- Attention backend dispatch parity beyond dense eager/SDPA unless a benchmark proves it matters.

## 11. Runtime staging plan

Stage 1: parse `ErnieConfig`, reject unsupported config combinations, and load encoder weights with tied LM alias metadata.

Stage 2: implement embedding block including optional task type embeddings and default position/token/task IDs.

Stage 3: run one encoder block parity with dense MHA, LayerNorm, residuals, and GELU/ReLU MLP.

Stage 4: full `ErnieModel` encoder parity for open base configs.

Stage 5: add masked-LM head with tied decoder weight and vocab-size variations.

Stage 6: add classification/QA/pooler heads as thin optional heads.

Stage 7: add optional CausalLM decoder admission with causal mask and DynamicCache only after encoder/masked-LM parity is stable.

Stubbable initially: losses, dropout randomness, gradient checkpointing, output attentions/hidden-states capture, cross-attention, and generation controller conveniences.

## 12. Parity and validation plan

- Config parse tests for all config sweep entries, including ERNIE 1.0 ReLU, ERNIE 3.0 `use_task_id=True`, and tiny `position_embedding_type` ignored behavior.
- Embedding parity with explicit and default `token_type_ids`, `position_ids`, and `task_type_ids`.
- Single-layer parity for random `[B, S, H]` using fp32, comparing hidden states after attention output LayerNorm and after MLP LayerNorm.
- Full encoder parity for tiny random checkpoint at short sequences such as `S=7` and padded masks.
- Masked-LM logits parity on `hf-internal-testing/tiny-random-ErnieForMaskedLM` or equivalent tiny weights.
- Representative base config smoke tests for shape only if full weights are too large.
- Optional decoder tests: one-step prefill and decode cache parity only for synthetic `is_decoder=True`.

Suggested tolerances: fp32 absolute/relative `1e-5` for unfused paths; fp16/bf16 start around `1e-2` relative and tighten per kernel after backend selection. Compare logits and intermediate hidden states separately so mask or LayerNorm issues are localized.

## 13. Performance probes

- Encoder throughput by sequence length: `S = 16, 128, 512, 2048` where config permits.
- Batch sweep for base and large configs: `B = 1, 4, 16, 64`.
- Attention backend comparison: eager dense lowering vs SDPA/Flash-style fused attention for bidirectional masks.
- GEMM profile split: QKV projections, attention output projection, MLP up/down, LM decoder.
- Embedding + LayerNorm microbenchmark with and without task type embeddings.
- Mask construction overhead for dense masks and padded batches.
- LM head cost by vocab size: 18k vs 30,522 vs 40k.
- Memory usage by activation checkpoint boundary for encoder-only inference, especially long ERNIE 3.0 sequence length 2048.

No benchmark observations were collected in this audit; these are source-derived probe recommendations.

## 14. Skip/defer list

- Training losses and gradient checkpointing.
- Dropout stochastic parity.
- Beam search and generation controllers.
- Causal LM decoder mode and KV cache for first encoder/masked-LM target.
- Cross-attention and encoder-decoder cache.
- Output attentions/hidden-states capture unless needed for debug.
- Quantization and packed-weight loading; native source has no ERNIE-specific packed format.
- Multi-GPU/tensor parallel.
- Remote-code ERNIE 4.5 / ERNIE VL / ERNIE Image families; their `model_type` and architecture are separate audits, not this `ernie` family.

## 15. Final implementation checklist

- [ ] Parse `ErnieConfig` fields and source defaults.
- [ ] Reject or route unsupported `is_decoder=True` and `add_cross_attention=True` for first target.
- [ ] Load word, position, token type, optional task type embeddings.
- [ ] Preserve tied LM decoder and word embedding alias.
- [ ] Implement embedding add + LayerNorm path with default IDs.
- [ ] Implement dense bidirectional MHA with additive masks.
- [ ] Implement `Linear(H -> H)` Q/K/V/output projections with bias.
- [ ] Implement MLP `Linear(H -> I) -> gelu/relu -> Linear(I -> H)`.
- [ ] Implement post-attention and post-MLP residual LayerNorm.
- [ ] Implement pooler and masked-LM head.
- [ ] Add QKV packed projection rewrite with `[q, k, v]` split-order guard.
- [ ] Add inference dropout elimination rewrite.
- [ ] Add embedding-add-LayerNorm fusion candidate behind parity tests.
- [ ] Add config sweep parser tests for tiny, ERNIE 1.0, 2.0 base/large, and 3.0.
- [ ] Add single-block, full-encoder, and masked-LM logits parity tests.
- [ ] Benchmark encoder attention/GEMM/LM-head bottlenecks across batch and sequence length.
