# DeBERTa Transformers audit

## 1. Source basis

Transformers commit/version: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`, local checkout `transformers`.

Model id: family `deberta`, config `model_type="deberta"`. Primary production references: `microsoft/deberta-base`, `microsoft/deberta-large`, `microsoft/deberta-xlarge`, and `microsoft/deberta-base-mnli`.

Config source: fetched Hugging Face `config.json` files into `_sources/` from:

- [`hf-internal-testing/tiny-random-DebertaModel`](https://huggingface.co/hf-internal-testing/tiny-random-DebertaModel)
- [`microsoft/deberta-base`](https://huggingface.co/microsoft/deberta-base)
- [`microsoft/deberta-large`](https://huggingface.co/microsoft/deberta-large)
- [`microsoft/deberta-xlarge`](https://huggingface.co/microsoft/deberta-xlarge)
- [`microsoft/deberta-base-mnli`](https://huggingface.co/microsoft/deberta-base-mnli)

Source files inspected and snapshotted:

- `transformers/src/transformers/models/deberta/modeling_deberta.py`
- `transformers/src/transformers/models/deberta/configuration_deberta.py`
- `transformers/src/transformers/models/deberta/tokenization_deberta.py`
- Comparison only: `transformers/src/transformers/models/deberta_v2/modeling_deberta_v2.py`

Commit-pinned source URLs:

- [`modeling_deberta.py`](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/deberta/modeling_deberta.py)
- [`configuration_deberta.py`](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/deberta/configuration_deberta.py)
- [`tokenization_deberta.py`](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/deberta/tokenization_deberta.py)
- Comparison only: [`modeling_deberta_v2.py`](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/deberta_v2/modeling_deberta_v2.py)

Any missing files or assumptions: no 401/403/gated repos were encountered. `microsoft/deberta-base/special_tokens_map.json` returned 404, so special-token defaults are taken from `tokenization_deberta.py`; `tokenizer_config.json` was available and records `do_lower_case=false`, `vocab_type="gpt2"`. Dtype is omitted by the Microsoft config JSON files; runtime dtype should come from weights or deployment policy.

Primary runtime target: encoder/base and masked-LM inference on CUDA. Sequence classification, token classification, and QA heads are optional first-wave heads. Training losses are out of scope.

## 2. High-level architecture

DeBERTa v1 is a text-only bidirectional encoder with disentangled relative self-attention. It is not an autoregressive generator and has no KV cache or decode loop.

```text
ByteLevel BPE preprocessing -> input_ids/attention_mask/token_type_ids
  -> word embeddings + optional absolute/token-type embeddings
  -> embedding projection/LayerNorm/dropout/mask
  -> repeated DeBERTa encoder layers with disentangled relative attention
  -> task head
```

For masked LM:

```text
input_ids, attention_mask, optional token_type_ids/position_ids
  -> DebertaModel encoder
  -> legacy or non-legacy MLM prediction head
  -> logits [B, S, vocab_size]
```

Stage split:

- CPU/data pipeline: byte-level BPE tokenization, special `[CLS]`/`[SEP]` insertion, padding, attention-mask construction.
- GPU/runtime graph: embeddings, mask multiplication, relative-position id construction or supplied `relative_pos`, encoder blocks, head projection.
- Independently testable units: embedding path, relative-position builder, disentangled bias function, one encoder layer, full encoder, each task head.

## 3. Important config dimensions

Production configs inspected override several source defaults:

| Field | Production DeBERTa v1 behavior |
|---|---|
| `relative_attention` | `true` for Microsoft base/large/xlarge/MNLI configs |
| `pos_att_type` | `"c2p|p2c"` in JSON, normalized by config to `["c2p", "p2c"]` |
| `max_relative_positions` | `-1` in JSON, resolved in source to `max_position_embeddings` (`512`) |
| `position_biased_input` | `false`; no absolute position embedding contribution in production configs |
| `type_vocab_size` | `0`; no token-type embedding table in production configs |
| `layer_norm_eps` | `1e-7` |
| `hidden_act` | `gelu` |
| `cache support` | none; encoder-only, no generation cache |

Representative checkpoint sweep:

| Model id | Layers | Hidden | Heads | Head dim | Intermediate | Vocab | Max pos | Rel attn | Head/task |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| `hf-internal-testing/tiny-random-DebertaModel` | 5 | 32 | 4 | 8 inferred | 37 | 1024 | 512 | disabled, `pos_att_type=["none"]` | base |
| `microsoft/deberta-base` | 12 | 768 | 12 | 64 inferred | 3072 | 50265 | 512 | c2p+p2c, unbucketed | base/MLM |
| `microsoft/deberta-large` | 24 | 1024 | 16 | 64 inferred | 4096 | 50265 | 512 | c2p+p2c, unbucketed | base/MLM |
| `microsoft/deberta-xlarge` | 48 | 1024 | 16 | 64 inferred | 4096 | 50265 | 512 | c2p+p2c, unbucketed | base/MLM |
| `microsoft/deberta-base-mnli` | 12 | 768 | 12 | 64 inferred | 3072 | 50265 | 512 | c2p+p2c, unbucketed | sequence classification, 3 labels |

Defaults from `configuration_deberta.py` that differ from production configs: `relative_attention=false`, `position_biased_input=true`, `type_vocab_size=0`, `legacy=true`, `pooler_dropout=0.0`, and `pos_att_type=None`. DinoML should trust loaded config values, not the class defaults alone.

## 3a. Family variation traps

- DeBERTa v1 differs from DeBERTa-v2 in projection layout: v1 uses one packed `in_proj: Linear(H -> 3H, bias=False)` plus separate learned `q_bias` and `v_bias`; v2 uses separate biased `query_proj`, `key_proj`, and `value_proj`.
- Packed QKV split is per-head interleaved. Source reshapes `in_proj(hidden)` to `[B, S, heads, 3*Dh]`, permutes to `[B, heads, S, 3*Dh]`, then chunks the last dim into Q/K/V. For source weight rows, that means each head has contiguous `[q_head, k_head, v_head]` rows, not all-Q/all-K/all-V row blocks.
- Relative attention is unbucketed in v1. DeBERTa-v2 adds `position_buckets` and optional log bucketing; DeBERTa v1 source does not.
- `position_biased_input=false` removes absolute position embeddings from production configs, but relative embeddings are still used inside attention.
- `type_vocab_size=0` means tokenizer may emit `token_type_ids`, but production models ignore them. Debug configs can set `type_vocab_size>0`.
- `embedding_size` can differ from `hidden_size`, enabling a bias-free `embed_proj` and changing legacy MLM head dimensions.
- `legacy=true` changes the MLM head. Legacy uses a decoder linear over `embedding_size`; non-legacy explicitly matmuls against `word_embeddings.weight.T`.
- `talking_head` is read by source but absent from standard config defaults; if present and true, it adds head-mixing linears over the head axis before and after softmax.
- `z_steps` is a mutable model attribute initialized to `0`, not a config field. If set above `1`, source re-applies the last encoder layer with `query_states`; first integration should reject or ignore nonzero `z_steps`.
- Source uses custom `DebertaLayerNorm` in the encoder/embedding path: fp32 mean/variance, epsilon inside sqrt, cast back before affine. Some heads use standard `nn.LayerNorm`.
- No causal masking, no cross-attention, no generation cache. Treat DeBERTa v1 as an encoder family, not a prefill/decode family.
- Source tensors are `[B, S, H]`; no NHWC/channel-last rewrite applies. Axis-sensitive ops include softmax `dim=-1`, LayerNorm `dim=-1`, QA split/squeeze on last dim, and classification pooling at token index `0`.

## 4. Operator coverage checklist

Tensor/layout ops:

- Embedding gather for word, optional absolute position, optional token type.
- `arange`, slice, expand, unsqueeze/squeeze, view/reshape, permute, contiguous.
- Pair mask expansion: `[B, S] -> [B, 1, 1, S]`, then pairwise multiplication to `[B, 1, S, S]`.
- Elementwise add, subtract, multiply, division, pow, sqrt, comparisons, bool conversion, masked fill, clamp.
- `gather` for c2p and p2c relative-bias lookup.
- `split`/`chunk` for QKV and QA logits.

Neural network primitives:

- Custom DeBERTa LayerNorm over last dim with fp32 reduction and affine.
- Standard `nn.LayerNorm` for MLM transforms.
- Linear with bias for attention output, FFN, pooler, classifier, QA, and most heads.
- Packed bias-free QKV linear `Linear(H -> 3H, bias=False)` plus separate q/v bias adds.
- Bias-free `embed_proj` when `embedding_size != hidden_size`.
- GELU and ACT2FN-configured activations.
- Dropout is in source but disabled for inference/eval.

Attention primitives:

- Bidirectional MHA only; no GQA/MQA.
- Q/K/V logical shapes `[B, heads, S, Dh]`.
- Scores `matmul(q / sqrt(Dh * scale_factor), k^T)`.
- Add disentangled relative bias before mask/softmax.
- Optional talking-head projection over head axis if admitted.
- Masked softmax with `torch.finfo(dtype).min` fill.
- Context matmul and reshape back to `[B, S, H]`.

Position/relative-bias ops:

- Relative ids from `q_ids[:, None] - k_ids[None, :]`, shape `[1, Q, K]`.
- Relative embedding table `[2 * max_relative_positions, H]`, shared by all layers.
- c2p and p2c projection/matmul/clamp/gather paths.

Generation/cache ops: none. There is no `past_key_values`, cache update, cache reorder, or logits sampling path.

Preprocessing-coupled ops:

- ByteLevel BPE tokenizer, `[CLS]` prefix, `[SEP]` suffix, pair template `[CLS] A [SEP] [SEP] B [SEP]`.
- Runtime graph consumes `input_ids`, `attention_mask`, optional `token_type_ids`, optional `position_ids`, or `inputs_embeds`.

Heads:

- Required for primary target: base encoder and `DebertaForMaskedLM`.
- Optional: sequence classification, token classification, QA.
- Deferred: training losses and any user-mutated `z_steps` behavior.

## 5. Layer/block breakdown

Embedding path:

```text
input_ids [B,S] -> word_embeddings [B,S,E]
position_ids [1,S] -> position_embeddings [1,S,E] if position_biased_input
token_type_ids [B,S] -> token_type_embeddings [B,S,E] if type_vocab_size > 0
emb = word + enabled position/type embeddings
if E != H: emb = Linear(E -> H, bias=False)
emb = DebertaLayerNorm(H, eps=1e-7)(emb)
emb = emb * attention_mask[..., None]
```

Encoder block, repeated `num_hidden_layers`:

```text
packed = Linear(H -> 3H, bias=False)(x)
q, k, v = reshape/permute/chunk(packed)  # [B, heads, S, Dh]
q = q + q_bias
v = v + v_bias
scores = matmul(q / sqrt(Dh * (1 + len(pos_att_type))), k^T)
scores += c2p/p2c disentangled relative bias if enabled
scores = optional talking_head_logits(scores)
scores = masked_fill(scores, ~pair_mask, finfo.min)
probs = softmax(scores, dim=-1)
probs = optional talking_head_weights(probs)
context = matmul(probs, v) -> [B,S,H]
x_attn = DebertaLayerNorm(Linear(H -> H)(context) + residual)
ff = Linear(H -> intermediate)(x_attn)
ff = GELU(ff)
x = DebertaLayerNorm(Linear(intermediate -> H)(ff) + x_attn)
```

Masked-LM heads:

```text
legacy=true:
  h [B,S,H] -> Linear(H -> E) -> GELU -> LayerNorm(E)
  logits = Linear(E -> vocab_size, bias=True)

legacy=false:
  h [B,S,H] -> Linear(H -> H) -> GELU -> LayerNorm(H)
  logits = matmul(h, word_embeddings.weight.T) + bias[vocab]
```

Other heads:

- Sequence classification: take `hidden_states[:, 0]`, dropout, `Linear(H -> H)`, activation, dropout, `Linear(H -> num_labels)`.
- Token classification: dropout, `Linear(H -> num_labels)` for every token.
- QA: `Linear(H -> num_labels)`, then split last dim into start/end logits and squeeze.

## 6. Attention requirements

Required attention is bidirectional self-attention. There is no causal mask, cross-attention, sliding window, block sparse attention, packed varlen metadata, or KV cache.

Production base/large/xlarge shapes:

- Hidden input `[B, S, H]`.
- Heads: 12 or 16.
- Head dim: `H / heads = 64` for inspected Microsoft production configs.
- Q/K/V after split: `[B, heads, S, 64]`.
- Scores/probs: `[B, heads, Q, K]`, normally `Q=K=S`.
- Context: `[B, S, H]`.

Masking style:

- If attention mask is rank 2, source builds `[B, 1, S, S]` pair mask by multiplying query-valid and key-valid masks.
- If rank 3, source unsqueezes to `[B, 1, Q, K]`.
- Mask fill uses `torch.finfo(query_layer.dtype).min`, then softmax over the key dimension.

FlashAttention/SDPA compatibility: the plain content attention is compatible with dense bidirectional attention, but c2p/p2c relative bias must be computed and added before the backend call. A fused attention path needs an additive bias tensor or custom kernel support for the gather-based relative terms. Talking-head mode, if admitted, blocks simple SDPA replacement because it projects over heads before/after softmax.

## 7. Position encoding and custom math

DeBERTa v1 production checkpoints use relative attention without absolute position-biased input. Relative positions are not log-bucketed.

Short source-equivalent snippets:

```python
def build_relative_position(q_len, k_len, device):
    q_ids = torch.arange(q_len, dtype=torch.long, device=device)
    k_ids = torch.arange(k_len, dtype=torch.long, device=device)
    return (q_ids[:, None] - k_ids[None, :]).unsqueeze(0)  # [1,Q,K]
```

```python
def deberta_layer_norm(x, weight, bias, eps):
    y = x.float()
    mean = y.mean(-1, keepdim=True)
    var = (y - mean).pow(2).mean(-1, keepdim=True)
    y = (y - mean) / torch.sqrt(var + eps)
    return weight * y.to(x.dtype) + bias
```

Relative bias outline:

```python
scale_factor = 1 + len(pos_att_type)
att_span = min(max(Q, K), max_relative_positions)
rel = rel_embeddings[max_rel - att_span : max_rel + att_span]  # [2*att_span,H]

if "c2p" in pos_att_type:
    pos_k = pos_proj(rel) -> [heads, 2*att_span, Dh]
    c2p = matmul(q, pos_k.T)
    c2p = gather(c2p, dim=-1, index=clamp(relative_pos + att_span))

if "p2c" in pos_att_type:
    pos_q = pos_q_proj(rel) / sqrt(Dh * scale_factor)
    p2c = matmul(k, pos_q.T)
    p2c = gather(p2c, dim=-1, index=clamp(-relative_pos + att_span)).transpose(-1, -2)
```

Precomputable: relative id matrix for each `(Q, K, max_relative_positions)` shape and rel-embedding slices for fixed sequence lengths. Dynamic: gather indices depend on runtime sequence lengths; relative embedding weights are model parameters.

## 8. Preprocessing and input packing

Tokenizer/runtime contract:

- `DebertaTokenizer` is a fast tokenizers-backed ByteLevel BPE tokenizer.
- Defaults: `[CLS]` id 1, `[SEP]` id 2, `[PAD]` id 3, `[MASK]` id 4 in the constructor fallback; loaded vocab supplies actual mapping.
- Single sequence template: `[CLS] A [SEP]`.
- Pair template: `[CLS] A [SEP] [SEP] B [SEP]`.
- Model input names: `input_ids`, `attention_mask`, `token_type_ids`.
- Production `type_vocab_size=0`, so `token_type_ids` are accepted but ignored by the embedding path.

GPU graph inputs:

- `input_ids [B,S]` or `inputs_embeds [B,S,E]`, exactly one required.
- `attention_mask [B,S]`, default all ones if omitted.
- `token_type_ids [B,S]`, default zeros if omitted.
- `position_ids [1,S]` default from registered arange buffer.

No multimodal packing, placeholder scatter, `cu_seqlens`, image/audio preprocessing, or generation controller is involved.

## 9. Graph rewrite / lowering opportunities

### Rewrite: packed DeBERTa v1 QKV to three linears

Source pattern:

```text
packed = Linear(H -> 3H, bias=False)(x)
packed.view(B,S,heads,3*Dh).permute(0,2,1,3).chunk(3, dim=-1)
q += q_bias; v += v_bias
```

Replacement:

```text
Q = Linear(H -> H, bias=q_bias)(x)
K = Linear(H -> H, bias=0 or absent)(x)
V = Linear(H -> H, bias=v_bias)(x)
```

Preconditions:

- `all_head_size == hidden_size`.
- Weight rows are transformed from per-head `[q,k,v]` groups to all-Q/all-K/all-V blocks.
- `q_bias` and `v_bias` become projection biases; K bias is zero/absent.

Weight transform:

```python
w = in_proj.weight.reshape(num_heads, 3, head_dim, hidden_size)
q_w = w[:, 0].reshape(hidden_size, hidden_size)
k_w = w[:, 1].reshape(hidden_size, hidden_size)
v_w = w[:, 2].reshape(hidden_size, hidden_size)
```

Failure cases: nonstandard `all_head_size`, `query_states` z-step path unless separately validated, or checkpoints with incompatible packed layout. Parity test: compare Q/K/V tensors before attention for base and large configs.

### Rewrite: relative-position id precompute

Source pattern: build `arange` ids and subtract every forward.

Replacement: cache an int64 `[1,Q,K]` relative-position table per profiled sequence shape.

Preconditions: fixed or bucketed `Q,K` and no externally supplied `relative_pos`. Failure case: dynamic sequence length not in cache; fall back to runtime arange/subtract.

### Rewrite: additive bias FlashAttention

Source pattern: dense scores + relative bias + mask + softmax + value matmul.

Replacement: compute relative bias `[B, heads, Q, K]`, combine with mask as additive bias, call dense attention backend that accepts additive bias.

Preconditions: no talking-head mode, dense bidirectional attention, dropout disabled, additive bias dtype/masking exactly matches source. Failure cases: backend cannot represent `finfo.min` mask semantics or cannot accept per-head additive bias.

### Rewrite: last-token-free classification pooling

Source pattern: `hidden_states[:, 0]` then pooler dense/activation/classifier.

Replacement: slice/gather first token only before head GEMMs.

Preconditions: sequence classification head only; encoder still needs full sequence. Failure case: tasks requiring token-level logits.

## 10. Kernel fusion candidates

Highest priority:

- Custom DeBERTa LayerNorm + residual add in embedding, attention output, and FFN output.
- Packed QKV projection transform into GEMM-friendly linears or a custom packed projection loader.
- Dense attention with precomputed c2p/p2c relative bias and mask.
- FFN `Linear -> GELU -> Linear` with residual LayerNorm around the second projection.

Medium priority:

- Relative c2p/p2c bias gather kernels, especially for common `S=512`.
- Masked softmax with additive relative bias.
- MLM head matmul against tied word embeddings for `legacy=false`.
- Classification pooler first-token gather plus small GEMMs.

Lower priority:

- Talking-head projections over the head axis; rare/unadvertised in representative configs.
- `z_steps` last-layer reapplication; mutable runtime knob, not normal config.
- Dropout removal/eval-mode canonicalization.

## 11. Runtime staging plan

Stage 1: parse config, load embeddings/encoder weights, reject unsupported `talking_head=true` and `z_steps>1`.

Stage 2: implement base encoder without relative attention using tiny random config parity.

Stage 3: add DeBERTa v1 packed QKV handling, q/v biases, custom LayerNorm, and production relative c2p+p2c bias.

Stage 4: full encoder parity for `microsoft/deberta-base` on fixed sequence lengths, including mask behavior.

Stage 5: add masked-LM heads, with legacy head first because `legacy=true` is the source default and common load path.

Stage 6: add optional sequence classification, token classification, and QA heads.

Stage 7: optimize relative-bias precompute, fused attention, LayerNorm/residual, and GEMM epilogues.

Initially stub/defer training losses, output attentions, hidden-state tuple materialization, talking-head mode, and nonzero `z_steps`.

## 12. Parity and validation plan

- Unit parity for `build_relative_position` over `(Q,K) = (4,4), (3,5), (512,512)`.
- Unit parity for `DebertaLayerNorm` fp32/fp16/bf16 with eps `1e-7`.
- Packed QKV rewrite parity: compare Q/K/V after q/v bias for random weights.
- Disentangled bias parity for c2p only, p2c only, and c2p+p2c.
- Single-layer encoder parity with and without padding masks.
- After-N-layer encoder parity for tiny random config, then `microsoft/deberta-base`.
- Masked-LM logits parity for legacy and non-legacy heads where configs/weights are available.
- Head parity for MNLI sequence classification: logits shape `[B,3]`, first-token pooler behavior.
- Recommended tolerances: fp32 `rtol=1e-4, atol=1e-4`; fp16/bf16 start at `rtol=5e-2, atol=5e-2` for full encoder, tighten after fused kernels are validated.

No DinoML tests were run for this audit per user instruction.

## 13. Performance probes

- Encoder throughput sweep by batch and sequence length: `S=64,128,256,512`.
- Relative-bias construction time separated from attention matmul/softmax.
- Dense attention backend comparison: unfused matmul/softmax/matmul vs additive-bias attention.
- QKV loader/projection comparison: native packed layout handling vs transformed separate Q/K/V weights.
- LayerNorm/residual kernel timing across `[B,S,H]` for base/large/xlarge.
- MLM head throughput and memory bandwidth for vocab `50265`.
- Classification head latency for first-token pooling workloads.
- Mask-density sweep for padded batches to quantify pair-mask and masked-softmax cost.

## 14. Skip/defer list

- Training losses and label filtering paths.
- Dropout behavior outside eval mode.
- `output_attentions` and full `hidden_states` materialization unless requested.
- `talking_head=true` until a real checkpoint requires it.
- Mutable `z_steps > 1` path.
- Generation, beam search, KV cache, cache reorder: not applicable.
- Tokenizer execution on GPU; keep ByteLevel BPE in CPU/data pipeline.
- Remote-code behavior: none required by inspected sources/configs.

## 15. Final implementation checklist

- [ ] Parse `DebertaConfig`, including production overrides and string `pos_att_type`.
- [ ] Load word, optional position, optional token-type embeddings.
- [ ] Implement DeBERTa custom LayerNorm.
- [ ] Implement packed v1 `in_proj` QKV split with q/v bias.
- [ ] Implement c2p/p2c disentangled relative attention bias.
- [ ] Implement bidirectional masked MHA with additive relative bias.
- [ ] Implement encoder FFN and residual LayerNorm blocks.
- [ ] Add legacy MLM head and tied/non-legacy MLM variant.
- [ ] Add optional sequence classification, token classification, and QA heads.
- [ ] Add config admission rejects for `talking_head=true` and nonzero `z_steps` until supported.
- [ ] Add parity tests for relative ids, LayerNorm, packed QKV, one layer, full encoder, and heads.
- [ ] Benchmark relative-bias, attention, LayerNorm, FFN, and MLM head paths separately.
