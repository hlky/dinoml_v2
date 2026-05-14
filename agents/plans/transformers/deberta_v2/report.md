# DeBERTa-v2 Transformers audit

## 1. Source basis

Transformers commit/version: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`, local checkout `X:/H/transformers`.

Model family: `deberta_v2`, config `model_type="deberta-v2"`.

Primary runtime target: encoder/base and masked-LM inference on CUDA. Classification, token classification, QA, and multiple-choice heads are optional/deferred for the first target.

Source files inspected:

- `X:/H/transformers/src/transformers/models/deberta_v2/modeling_deberta_v2.py`
- `X:/H/transformers/src/transformers/models/deberta_v2/configuration_deberta_v2.py`
- `X:/H/transformers/src/transformers/models/deberta_v2/tokenization_deberta_v2.py`
- `X:/H/transformers/src/transformers/models/deberta_v2/__init__.py`

Representative config sources fetched from Hugging Face raw config JSON:

- `microsoft/deberta-v2-xlarge`
- `microsoft/deberta-v2-xxlarge`
- `microsoft/deberta-v3-base`
- `microsoft/deberta-v3-large`
- `microsoft/mdeberta-v3-base`
- Debug-only contrast: `hf-internal-testing/tiny-random-DebertaV2Model`, `hf-internal-testing/tiny-random-DebertaV2ForMaskedLM`

Any missing files or assumptions: no remote-code files are required for these model ids. `special_tokens_map.json` was not present for the checked Microsoft repos; tokenizer defaults come from `tokenization_deberta_v2.py` and fetched `tokenizer_config.json` (`do_lower_case=false`, `vocab_type=spm`). Dtype is not specified in the Microsoft config JSON files inspected; runtime dtype should come from loaded weights or deployment policy.

## 2. High-level architecture

DeBERTa-v2 is a text-only bidirectional encoder. The core model is:

```text
SentencePiece/tokenizer preprocessing -> token embeddings -> optional absolute/token-type embeddings
  -> embedding projection/LayerNorm/dropout/mask
  -> repeated DeBERTa encoder layers with disentangled relative self-attention
  -> task head
```

For the primary target:

```text
input_ids, attention_mask, optional token_type_ids/position_ids
  -> DebertaV2Model encoder
  -> masked-LM transform/head
  -> logits [B, S, vocab_size]
```

Staging split:

- CPU/data pipeline: SentencePiece Unigram tokenization, special token insertion, padding, attention mask construction.
- GPU/runtime graph: embeddings, mask application, relative-position id construction or constant/buffer input, encoder blocks, MLM head.
- Independently testable units: tokenizer output contract, embedding path, relative-position bucket function, one attention layer, full encoder, MLM head.

## 3. Important config dimensions

Production configs inspected share these operator-relevant defaults unless listed otherwise:

| Field | Effective behavior |
|---|---|
| `relative_attention` | `true` for inspected Microsoft production configs |
| `position_buckets` | `256` for inspected Microsoft production configs |
| `max_relative_positions` | `-1` in config, resolved to `max_position_embeddings` (`512`) in source |
| `pos_att_type` | config string `"p2c|c2p"` is converted to `["p2c", "c2p"]` |
| `share_att_key` | `true`, so relative position projections reuse query/key projection weights |
| `norm_rel_ebd` | `"layer_norm"`; relative embedding table is LayerNormed before use |
| `position_biased_input` | `false`; no absolute position embedding is added to token embeddings |
| `type_vocab_size` | `0`; no token-type embedding table for production configs |
| `layer_norm_eps` | `1e-7` |
| `hidden_act` | `gelu` |
| `cache support` | none for generation; bidirectional encoder, no KV cache |

Representative checkpoint sweep:

| Model id | Layers | Hidden | Heads | Head dim | Intermediate | Vocab | Max pos | Relative/buckets | Conv |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| `microsoft/deberta-v3-base` | 12 | 768 | 12 | 64 inferred | 3072 | 128100 | 512 | p2c+c2p, 256 buckets | omitted -> disabled |
| `microsoft/deberta-v3-large` | 24 | 1024 | 16 | 64 inferred | 4096 | 128100 | 512 | p2c+c2p, 256 buckets | omitted -> disabled |
| `microsoft/deberta-v2-xlarge` | 24 | 1536 | 24 | 64 config | 6144 | 128100 | 512 | p2c+c2p, 256 buckets | `kernel=3`, GELU |
| `microsoft/deberta-v2-xxlarge` | 48 | 1536 | 24 | 64 config | 6144 | 128100 | 512 | p2c+c2p, 256 buckets | `kernel=3`, GELU |
| `microsoft/mdeberta-v3-base` | 12 | 768 | 12 | 64 inferred | 3072 | 251000 | 512 | p2c+c2p, 256 buckets | omitted -> disabled |
| `hf-internal-testing/tiny-random-DebertaV2ForMaskedLM` | 5 | 32 | 4 | 8 inferred | 37 | 128001 | 512 | `relative_attention=false` | disabled |

Config default traps from `configuration_deberta_v2.py`: source defaults have `relative_attention=false`, `position_biased_input=true`, `type_vocab_size=0`, and `legacy=true`. Production DeBERTa-v2/v3 configs override the relative and position-bias behavior. Do not rely on class defaults when loading real checkpoints.

## 3a. Family variation traps

- `attention_head_size` can override `hidden_size // num_attention_heads`; source only checks `hidden_size % num_attention_heads == 0`, then uses `all_head_size = num_heads * attention_head_size`.
- Production DeBERTa-v2 xlarge/xxlarge include an optional Conv1d layer after layer 0 attention. Production DeBERTa-v3 configs inspected omit `conv_kernel_size`, so source disables it.
- `embedding_size` may differ from `hidden_size`, creating a bias-free `embed_proj` and changing legacy MLM transform/output dimensions.
- `legacy=true` changes the masked-LM head. Legacy uses a standalone decoder linear over `embedding_size`; non-legacy uses explicit matmul against input word embeddings.
- `share_att_key=false` adds separate `pos_key_proj` and/or `pos_query_proj` linears. Production configs inspected set `share_att_key=true`.
- Relative attention can be absent (`relative_attention=false`) in debug or nonstandard configs.
- `position_biased_input=false` means position embeddings are not added in the embedding path, but relative position embeddings are still used inside attention.
- `type_vocab_size=0` means tokenizer may still emit `token_type_ids`, but the model ignores them. If `type_vocab_size>0`, token-type embedding addition is required.
- There is no causal mask or KV cache. Attention masks are bidirectional pair masks.
- Source at this commit does not use an `XSoftmax` class. Masked softmax is `masked_fill(~attention_mask, finfo(dtype).min)` then `softmax(dim=-1)`.
- Layout translation should be guarded off for sequence-major semantics. Source tensors are `[B, S, H]`; Conv1d temporarily permutes to `[B, H, S]`, then returns `[B, S, H]`.

## 4. Operator coverage checklist

Tensor/layout ops:

- Embedding gather for word, optional absolute position, optional token type.
- `arange`, slice, expand, unsqueeze/squeeze, view/reshape, permute, contiguous.
- Mask expansion: `[B, S] -> [B, 1, S] -> [B, 1, S, S]`.
- Elementwise add, multiply, subtract, comparisons, bool conversion, where, clamp.
- `gather` for relative c2p/p2c bias.
- Optional split/squeeze for QA head.

Neural network primitives:

- LayerNorm with eps `1e-7`, affine.
- Linear with bias for Q/K/V, attention output, FFN, pooler/head projections.
- Linear without bias for `embed_proj` when `embedding_size != hidden_size`.
- GELU and optional configured activations through `ACT2FN`; production configs use GELU, conv uses GELU for v2 xlarge/xxlarge.
- Dropout is present in source but disabled by inference/eval.
- Optional Conv1d `[B, H, S] -> [B, H, S]` with `kernel=3`, padding 1, groups default 1 for v2 xlarge/xxlarge.

Attention primitives:

- Bidirectional MHA, no GQA/MQA.
- Q/K/V shapes after projection: `[B, S, H] -> [B * heads, S, head_dim]`.
- Scores: batched matmul `[B*heads, Q, Dh] x [B*heads, Dh, K] -> [B*heads, Q, K]`.
- Scale divides key operand by `sqrt(Dh * scale_factor)` before bmm; `scale_factor = 1 + has_c2p + has_p2c`.
- Add disentangled relative bias before mask/softmax.
- Masked softmax with `finfo(dtype).min` fill for masked positions.
- Context bmm and reshape back to `[B, S, H]`.

Position/relative-bias ops:

- Build relative position ids from `q_ids[:, None] - k_ids[None, :]`.
- Optional logarithmic bucketing with `bucket_size=position_buckets`, `max_position=max_relative_positions`.
- Relative embedding table shape `[2 * pos_ebd_size, H]`, where `pos_ebd_size = position_buckets` if buckets > 0 else resolved max relative positions.
- Optional LayerNorm over relative embedding table before per-layer reuse.
- c2p and p2c bias paths with `bmm`, clamp, gather, transpose for p2c.

Preprocessing-coupled ops:

- SentencePiece/Unigram tokenizer with metaspace, special `[CLS]` and `[SEP]`.
- Runtime receives `input_ids`, `attention_mask`, and optionally `token_type_ids`.
- Padding mask must enter both embeddings and attention.

Heads:

- Required for primary target: `DebertaV2ForMaskedLM`.
- Optional: base `DebertaV2Model`.
- Deferred: sequence classification, token classification, question answering, multiple choice.

## 5. Layer/block breakdown

Embedding path:

```text
input_ids [B, S] -> word_embeddings [B, S, E]
position_ids [1, S] -> position_embeddings [1, S, E] if position_biased_input
token_type_ids [B, S] -> token_type_embeddings [B, S, E] if type_vocab_size > 0
emb = sum enabled embeddings
if E != H: emb = Linear(E -> H, bias=False)
emb = LayerNorm(H, eps=1e-7)(emb)
emb = emb * attention_mask[..., None]
```

Encoder block, repeated `num_hidden_layers`:

```text
q = Linear(H -> heads * Dh, bias=True)(query_states or x)
k = Linear(H -> heads * Dh, bias=True)(x)
v = Linear(H -> heads * Dh, bias=True)(x)
scores = bmm(q, k^T / sqrt(Dh * scale_factor))
scores += disentangled_relative_bias(q, k, rel_pos, rel_embeddings)
scores = masked_fill(scores, ~pair_mask, finfo.min)
probs = softmax(scores, dim=-1)
context = bmm(probs, v) -> reshape [B, S, H]
x_attn = LayerNorm(Linear(H -> H)(context) + residual_query)
ff = Linear(H -> intermediate)(x_attn)
ff = GELU(ff)
x = LayerNorm(Linear(intermediate -> H)(ff) + x_attn)
```

Optional first-layer conv:

```text
after layer 0 attention/FFN output:
conv = Conv1d(H -> H, kernel=conv_kernel_size, padding=(k-1)//2, groups=conv_groups)
conv_out = conv(original_embedding_hidden_states.permute[B,H,S]).permute[B,S,H]
conv_out = masked_fill(pad positions, 0)
conv_out = GELU(dropout(conv_out))
x = LayerNorm(layer0_output + conv_out)
x = x * attention_mask[..., None]
```

Masked-LM heads:

Legacy head (`config.legacy=true`, default):

```text
h [B,S,H] -> Linear(H -> E) -> GELU -> LayerNorm(E)
logits = Linear(E -> vocab_size, bias=True)
```

Non-legacy head:

```text
h [B,S,H] -> Linear(H -> H) -> GELU -> LayerNorm(H)
logits = matmul(h, word_embeddings.weight.T) + bias[vocab]
```

## 6. Attention requirements

Required attention is bidirectional self-attention, MHA, no causal masking, no cross-attention for the primary model path. `query_states` exists for the `z_steps` path and for encoder internals, but normal inference uses self-attention where Q/K/V come from the same hidden states.

Shape contract for production base/large:

- Input hidden states: `[B, S, H]`.
- Per-head hidden: `Dh=64` for inspected production configs.
- Q/K/V after `transpose_for_scores`: `[B * heads, S, Dh]`.
- Raw scores before head reshape: `[B * heads, S, S]`.
- Final attention scores/probs: `[B, heads, S, S]`.
- Context output: `[B, S, H]`.

Masking:

- A 2D mask `[B, S]` becomes pair mask `[B, 1, S, S]` through outer product of valid tokens.
- A 3D mask `[B, S, S]` becomes `[B, 1, S, S]`.
- The mask is converted to bool and applied with `masked_fill(~mask, torch.finfo(dtype).min)`.
- Parity note: all-masked rows would softmax over all `finfo.min`, not produce a custom zero row as some `XSoftmax` implementations do. Validate padding-only edge cases explicitly.

Backend compatibility:

- Vanilla SDPA/FlashAttention can cover only the content-content score path if relative bias is supplied as an additive score bias. The c2p/p2c bias computation itself is a separate custom pre-attention region.
- Dropout must be disabled for inference. Source applies dropout after softmax and on relative embeddings/output in training.
- Query scaling is implemented as `bmm(query, key.transpose / scale)`, equivalent to scores divided by scale; fused kernels must use `sqrt(Dh * scale_factor)`.

## 7. Position encoding and custom math

No RoPE, ALiBi, or cache position encoding. Position logic is DeBERTa disentangled relative attention.

Short source-equivalent snippets:

```python
def relative_position_ids(q_len, k_len):
    q = arange(q_len)
    k = arange(k_len)
    return (q[:, None] - k[None, :])[None, :, :]
```

```python
def make_log_bucket_position(relative_pos, bucket_size, max_position):
    sign = sign(relative_pos)
    mid = bucket_size // 2
    abs_pos = where((relative_pos < mid) & (relative_pos > -mid),
                    tensor(mid - 1).type_as(relative_pos),
                    abs(relative_pos))
    log_pos = ceil(log(abs_pos / mid) / log((max_position - 1) / mid) * (mid - 1)) + mid
    return where(abs_pos <= mid, relative_pos.type_as(log_pos), log_pos * sign)
```

```python
def disentangled_bias(q, k, rel_pos, rel_embeddings):
    # q,k: [B*heads, S, Dh], rel_embeddings: [2*span, H]
    # production configs use shared query/key projections for position embeddings.
    c2p_scores = bmm(q, pos_key.T)
    c2p_index = clamp(rel_pos + span, 0, 2 * span - 1)
    c2p = gather(c2p_scores, dim=-1, index=expanded_c2p_index) / scale

    p2c_scores = bmm(k, pos_query.T)
    p2c_index = clamp(-rel_pos + span, 0, 2 * span - 1)
    p2c = gather(p2c_scores, dim=-1, index=expanded_p2c_index).transpose(-1, -2) / scale
    return c2p + p2c
```

Precompute candidates:

- Relative position id matrix can be precomputed per `S` and reused across layers for self-attention.
- Bucketed ids can be precomputed for each supported sequence bucket.
- Relative embedding LayerNorm output changes only with weights and can be computed once per forward, as source does.
- Projected position embeddings can be cached per forward if weights are shared; for production `share_att_key=true`, they use Q/K projection weights and depend only on `rel_embeddings`.

## 8. Preprocessing and input packing

Tokenizer coupling:

- `DebertaV2Tokenizer` uses Hugging Face `tokenizers` Unigram model, files `spm.model` or `tokenizer.json`.
- Normalization: optional lowercase, whitespace/newline/tab collapse to single space, NFC, strip right.
- Pretokenization: optional punctuation isolation, then Metaspace replacement `"\u2581"` with prepend scheme `always` when `add_prefix_space=true`.
- Default special tokens: `[CLS]`, `[SEP]`, `[UNK]`, `[PAD]`, `[MASK]`.
- Postprocessor single sequence: `[CLS]:0 $A:0 [SEP]:0`.
- Pair sequence: `[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1`.
- `model_input_names = ["input_ids", "attention_mask", "token_type_ids"]`.

Runtime tensors:

- `input_ids`: int tensor `[B, S]`.
- `attention_mask`: optional; if omitted, source creates ones `[B, S]`.
- `token_type_ids`: optional; if omitted, source creates zeros `[B, S]`.
- `position_ids`: optional; if omitted, source uses buffer `[1, 0:S]`.
- Production configs have `type_vocab_size=0`, so `token_type_ids` are accepted but ignored by the model graph.

CPU/data-pipeline work: tokenization, padding/truncation, special token insertion. GPU/runtime work starts at embedding gather.

## 9. Graph rewrite / lowering opportunities

### Rewrite: precompute relative position ids

Source pattern: per-forward `arange`, subtract, optional log bucket.

Replacement: bucketed constant tensor `[1, S, S]` or runtime-selected sequence-bucket constant.

Preconditions:

- Self-attention with `Q_len == K_len == S`.
- Fixed or bucketed maximum `S`.
- Same `position_buckets` and `max_relative_positions`.

Failure cases: non-self `query_states` path, dynamic arbitrary sequence length without a supported bucket, custom external `relative_pos`.

Parity test sketch: compare source `build_relative_position` to compiled constant for several lengths including 1, 2, 128, 512.

### Rewrite: relative bias subgraph fusion

Source pattern: project position embeddings, `bmm`, clamp/gather, divide, sum c2p/p2c.

Replacement: custom relative-bias kernel that emits additive score bias `[B*heads, S, S]` or `[B, heads, S, S]`.

Preconditions:

- Known `pos_att_type` subset.
- Known `share_att_key` behavior.
- `rel_embeddings` already LayerNormed when `norm_rel_ebd` contains `layer_norm`.
- Index dtype int64/int32 behavior matches PyTorch gather.

Failure cases: `share_att_key=false` needing separate projection weights, `p2c` with `query_len != key_len`, external `relative_pos` rank 2/3/4 variants.

Parity test sketch: random q/k and rel embeddings, compare c2p-only, p2c-only, and p2c+c2p outputs.

### Rewrite: attention with additive relative bias

Source pattern: content scores + relative bias -> masked softmax -> context bmm.

Replacement: fused attention accepting additive score bias and pair mask.

Preconditions:

- Inference dropout disabled.
- Noncausal full attention.
- Bias is materialized or computed inside the fused kernel before softmax.
- Scale is `sqrt(Dh * scale_factor)`, not plain `sqrt(Dh)`.

Failure cases: training dropout, output attentions requested, all-masked row semantics if backend differs from PyTorch.

Parity test sketch: compare layer attention output/probs over fp32 and fp16 tolerances with padded masks.

### Rewrite: optional Conv1d as sequence convolution

Source pattern: `[B,S,H] -> permute [B,H,S] -> Conv1d(H,H,k,pad) -> permute [B,S,H]`.

Replacement: direct sequence-axis Conv1d kernel over `[B,S,H]` or lowered im2col/GEMM for small fixed `k=3`.

Preconditions:

- `groups`, padding, and kernel size match config.
- Axis semantics preserved: convolution is over sequence dimension, channels are hidden.
- Masked positions zeroed before activation.

Failure cases: grouped conv not implemented, non-odd kernel, custom `conv_act`, layout pass that treats hidden as spatial/channel incorrectly.

Parity test sketch: v2 xlarge config one-block parity with and without padding.

### Rewrite: MLM head weight tying

Source pattern non-legacy: transform then matmul with `word_embeddings.weight.T` plus bias.

Replacement: GEMM using embedding weight as RHS constant; if legacy head, use standalone decoder linear.

Preconditions:

- Correct `legacy` branch from config.
- Correct `embedding_size` for legacy decoder.
- Weight tying represented explicitly when non-legacy.

Failure cases: checkpoint uses legacy but runtime assumes tied non-legacy path, resized embeddings, `embedding_size != hidden_size`.

Parity test sketch: compare logits for both branches using synthetic configs.

## 10. Kernel fusion candidates

Highest priority:

- LayerNorm + residual patterns after attention and FFN. Every layer has two post-residual LayerNorms with eps `1e-7`.
- Dense GEMMs for Q/K/V, attention output, FFN up/down, and MLM head. These dominate encoder compute.
- Relative bias kernel for c2p+p2c. This is the custom family-specific blocker for optimized attention parity.
- Masked softmax attention with additive bias. A fused path should preserve `finfo.min` masking and DeBERTa scale.

Medium priority:

- Q/K/V projection packing. Source has separate biased linears; a packed QKV lowering can reduce launch overhead if weights are transformed.
- GELU FFN fusion: Linear -> GELU -> Linear is standard, but the first integration can leave it as GEMMs plus activation.
- Optional Conv1d sequence kernel for v2 xlarge/xxlarge.
- Relative embedding LayerNorm and projection cache per forward.

Lower priority:

- Pooler/classification heads.
- Multiple-choice flatten/unflatten convenience path.
- Training losses and label filtering.
- Output attentions materialization.

## 11. Runtime staging plan

Stage 1: parse config and load weights for `DebertaV2Model`; support production fields `relative_attention`, `position_buckets`, `share_att_key`, `norm_rel_ebd`, `position_biased_input`, `conv_kernel_size`, `embedding_size`, `legacy`.

Stage 2: implement embedding path and one encoder layer in fp32, including pair mask expansion and relative-position id/bucket parity.

Stage 3: implement full encoder for DeBERTa-v3-base style configs without Conv1d.

Stage 4: add optional first-layer Conv1d for DeBERTa-v2 xlarge/xxlarge.

Stage 5: add masked-LM head, initially legacy branch because `DebertaV2Config` defaults to `legacy=true`; also support non-legacy tied matmul for checkpoints that set it.

Stage 6: optimize relative bias and attention with fused kernels/additive-bias SDPA path.

Stage 7: add optional task heads: sequence classification, token classification, QA, multiple choice.

Stubbable initially: losses, training dropout, gradient checkpointing, output hidden states/attentions, `z_steps > 0`, non-production `share_att_key=false`.

## 12. Parity and validation plan

- Unit test `make_log_bucket_position` over negative, zero, positive positions and production `bucket_size=256`, `max_position=512`.
- Unit test `build_relative_position` for `S=1,2,17,512` against Transformers.
- Unit test mask expansion from `[B,S]` and `[B,S,S]`, including padding.
- Single embedding parity for production `position_biased_input=false` and debug `position_biased_input=true/type_vocab_size>0`.
- Single attention parity with relative attention p2c+c2p, fp32 tolerance around `1e-5`.
- Single layer parity including residual LayerNorm and FFN.
- Conv-enabled one-block parity for `microsoft/deberta-v2-xlarge` config.
- Full encoder parity for `microsoft/deberta-v3-base` at short sequence lengths before scaling.
- MLM logits parity for legacy and non-legacy synthetic configs.
- End-to-end masked token logits parity on a small text batch after tokenizer output is fixed.
- fp16/bf16 validation should use looser tolerances, especially around softmax and GELU; start with fp32 as source-of-truth.

## 13. Performance probes

- Tokenizer throughput and padding/batching overhead, separate from GPU runtime.
- Encoder-only latency/throughput by sequence length: 32, 128, 384, 512.
- Batch-size sweep: 1, 4, 8, 16 for base and large.
- Relative-bias subgraph time versus content attention time.
- Attention backend comparison: eager bmm/softmax, fused additive-bias attention, custom relative-bias fused path.
- Conv-enabled v2 xlarge/xxlarge first-layer overhead.
- MLM head logits cost for vocab 128100 and multilingual vocab 251000.
- Memory bandwidth and activation footprint for full `[B, heads, S, S]` score/prob tensors.

## 14. Skip/defer list

- Training, dropout behavior, losses.
- Gradient checkpointing.
- `output_attentions=True` and full hidden-state output as optimized runtime outputs.
- `z_steps > 1` iterative last-layer refinement path.
- Classification, token classification, QA, and multiple-choice heads for first masked-LM target.
- Non-production `share_att_key=false` and exotic `pos_att_type` combinations after source-compatible fallback exists.
- Quantization and multi-GPU/tensor parallel.
- Beam search/generation and KV cache; not applicable to this encoder target.

## 15. Final implementation checklist

- [ ] Parse `DebertaV2Config` including non-schema fields from checkpoint JSON.
- [ ] Load word embeddings, optional position/type embeddings, optional `embed_proj`.
- [ ] Implement embedding mask multiply and LayerNorm eps `1e-7`.
- [ ] Implement pair attention mask expansion and masked softmax parity.
- [ ] Implement relative position id and log bucket math.
- [ ] Implement relative embedding table and optional relative LayerNorm.
- [ ] Implement c2p/p2c disentangled relative bias for `share_att_key=true`.
- [ ] Add fallback for `share_att_key=false`.
- [ ] Implement MHA content attention with DeBERTa scale factor.
- [ ] Implement encoder block residual LayerNorm and GELU FFN.
- [ ] Implement optional first-layer Conv1d path.
- [ ] Implement legacy MLM head.
- [ ] Implement non-legacy tied MLM head.
- [ ] Add tokenizer/input contract notes to model loader docs.
- [ ] Add single-op parity tests for bucket positions and relative bias.
- [ ] Add one-layer and full-encoder parity tests.
- [ ] Add masked-LM logits parity test.
- [ ] Benchmark relative-bias, attention, FFN, and MLM head separately.
