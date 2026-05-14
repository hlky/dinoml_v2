# SwitchTransformers audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: google/switch-base-8 as the primary common checkpoint; sweep also covers google/switch-base-{16,32,64,128,256}, google/switch-large-128, google/switch-xxl-128, google/switch-c-2048.
Config source: official Hugging Face config.json files fetched from https://huggingface.co/{model_id}/resolve/main/config.json on 2026-05-13; snapshots stored beside this report.
Source files inspected: X:/H/transformers/src/transformers/models/switch_transformers/modeling_switch_transformers.py, configuration_switch_transformers.py, modular_switch_transformers.py; T5 inherited helpers in src/transformers/models/t5/modeling_t5.py; tests/models/switch_transformers/test_modeling_switch_transformers.py.
Any missing files or assumptions: no gated/401 official configs were encountered for the Google Switch checkpoints listed above. Tokenizer coupling is T5-style SentencePiece from google-t5/t5-small in tests, but tokenizer internals are not model-graph-owned.
```

Primary source URLs for review:

- [modeling_switch_transformers.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/switch_transformers/modeling_switch_transformers.py)
- [configuration_switch_transformers.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/switch_transformers/configuration_switch_transformers.py)
- [modular_switch_transformers.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/switch_transformers/modular_switch_transformers.py)
- [T5 modeling helpers](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/t5/modeling_t5.py)

`modeling_switch_transformers.py` is generated from `modular_switch_transformers.py`; future source edits should target the modular file. This report treats generated modeling code as the exact runtime basis.

## 2. High-level architecture

SwitchTransformers is a text encoder-decoder with T5-style relative-position attention and sparse MoE feed-forward layers. The primary DinoML target should be `SwitchTransformersForConditionalGeneration`: encoder prefill, decoder prefill/decode, encoder-decoder cross-attention, and logits for seq2seq generation.

```text
tokenized text + masks
  -> shared token embedding
  -> encoder stack: self-attention + dense or sparse MoE FFN
  -> decoder stack: causal self-attention + cross-attention + dense or sparse MoE FFN
  -> optional tied-embedding scale by d_model^-0.5
  -> lm_head logits
  -> generation controller / sampling
```

Stage decomposition:

- CPU/data pipeline: T5 tokenizer, task prefixes from config metadata, padding masks, decoder start token handling.
- Encoder: independently cacheable `encoder_last_hidden_state` with noncausal dense self-attention and sparse FFNs.
- Decoder prefill: causal self-attention, cross-attention over encoder states, MoE FFNs, logits over full decoder prefix.
- Decode: one or more decoder tokens with autoregressive self-attention KV cache plus encoder-decoder cross-attention cache.
- Optional outputs: router logits/losses and hidden/attention records are useful for parity but not needed for first inference.

## 3. Important config dimensions

Source defaults from `SwitchTransformersConfig`:

| Field | Default | Runtime meaning |
| --- | ---: | --- |
| `vocab_size` | 32128 | Shared input embedding and tied LM head width. |
| `d_model` | 768 | Hidden size. |
| `d_kv` | 64 | Per-head Q/K/V dimension. |
| `num_heads` | 12 | MHA heads; `inner_dim = num_heads * d_kv`. |
| `d_ff` | 2048 | Expert/dense FFN intermediate width by source default; common configs override to 3072+. |
| `num_layers` | 12 | Encoder layers. |
| `num_decoder_layers` | defaults to `num_layers` | Decoder layers; can be asymmetric. |
| `num_experts` | 8 | Experts per sparse FFN layer. |
| `expert_capacity` | 64 | Per-expert token capacity; overflow tokens are dropped. |
| `num_sparse_encoder_layers` / `num_sparse_decoder_layers` | 3 / 3 | Used to compute sparse step in `__post_init__`. |
| `router_dtype` | `float32` | Router projection/softmax dtype. |
| `router_jitter_noise` | 0.01 | Training-only multiplicative jitter; inference should disable by eval mode. |
| `relative_attention_num_buckets` | 32 | T5 relative bias buckets. |
| `relative_attention_max_distance` | 128 | T5 relative bias logarithmic cap. |
| `dense_act_fn` | `relu` | Single-branch FFN activation; current source does not use T5 gated FFN class. |
| `use_cache` | true | Decoder cache support. |
| `tie_word_embeddings` | true | Shared input embedding / LM head alias and output scaling. |

Representative official config sweep:

| Model | `d_model` | `d_ff` | `d_kv` | Heads | Enc/dec layers | Experts | Capacity | Sparse step enc/dec | Sparse layers enc/dec | Act | dtype metadata |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | --- | --- |
| `google/switch-base-8` | 768 | 3072 | 64 | 12 | 12 / 12 | 8 | 64 | 2 / 2 | 6 / 6 | relu | bfloat16 |
| `google/switch-base-16` | 768 | 3072 | 64 | 12 | 12 / 12 | 16 | 64 | 2 / 2 | 6 / 6 | relu | bfloat16 |
| `google/switch-base-32` | 768 | 3072 | 64 | 12 | 12 / 12 | 32 | 64 | 2 / 2 | 6 / 6 | relu | float32 |
| `google/switch-base-128` | 768 | 3072 | 64 | 12 | 12 / 12 | 128 | 64 | 2 / 2 | 6 / 6 | relu | float32 |
| `google/switch-base-256` | 768 | 3072 | 64 | 12 | 12 / 12 | 256 | 64 | 2 / 2 | 6 / 6 | relu | float32 |
| `google/switch-large-128` | 1024 | 4096 | 64 | 16 | 24 / 24 | 128 | 64 | 2 / 2 | 12 / 12 | relu | bfloat16 |
| `google/switch-xxl-128` | 4096 | 10240 | 64 | 64 | 48 / 12 | 128 | 64 | 2 / 2 | 24 / 6 | gelu | bfloat16 |
| `google/switch-c-2048` | 2080 | 6144 | 64 | 30 | 15 / 12 | 2048 | 64 | 1 / 0 | 15 / 0 | relu | omitted |

## 3a. Family variation traps

- `d_model != num_heads * d_kv` is possible. `switch-c-2048` has `d_model=2080`, `num_heads=30`, `d_kv=64`, so attention `inner_dim=1920`; Q/K/V/O projections are not square hidden-to-hidden.
- Encoder and decoder layer counts can differ: `switch-xxl-128` is 48 encoder / 12 decoder; `switch-c-2048` is 15 encoder / 12 decoder.
- Sparse placement is derived from `encoder_sparse_step` / `decoder_sparse_step`, not just `num_sparse_*` names. Source marks layer `i` sparse when `i % sparse_step == 1 or sparse_step == 1`. `switch-c-2048` has all encoder layers sparse (`step=1`) and no decoder sparse layers (`step=0` path creates none).
- Source uses top-1 routing only. Config fields such as `num_selected_experts`, `router_type`, and `batch_prioritized_routing` appear in historical configs but are not read by current modeling code.
- `router_ignore_padding_tokens` is stored but not used in router forward in the inspected source. Padding tokens can still route and consume capacity.
- Router output naming is misleading in source: the local `router_logits` from the classifier is overwritten by `torch.max(router_probs, ...)` before return. The sparse MLP uses that returned third value as routing weight, so first-inference parity should preserve selected max probability weighting rather than relying on the docstring's "raw logits" wording.
- `feed_forward_proj` and `is_gated_act` appear in configs but current Switch FFN instantiates `SwitchTransformersDenseActDense`, not T5 gated FFN. Treat gated Switch variants as unsupported unless source changes.
- Router projection may force weights/module to `router_dtype` during forward. DinoML should model router math as fp32 by default even when hidden states are bf16.
- No RoPE, ALiBi, local/sliding attention, GQA/MQA, packed varlen attention, or native quantized-weight path is implemented in this source.
- Tied embeddings are logical aliases: `shared.weight`, encoder embedding, decoder embedding, and usually `lm_head.weight`.

## 4. Operator coverage checklist

Tensor/layout ops:

- Token embedding lookup: `[B,S] int64 -> [B,S,d_model]`.
- Reshape/view: `[B,S,H] -> [B*S,H]` for sparse FFN routing; expert slices reshape back.
- Transpose/permute/contiguous for attention heads: `[B,S,inner_dim] -> [B,num_heads,S,d_kv]`.
- Mask expansion/inversion: encoder mask `[B,S] -> [B,1,1,S]`; decoder causal mask `[B,1,T,K]`; cross mask from encoder attention mask.
- `arange`, broadcast, compare, `where`, min/max, abs, log for relative-position buckets.
- Dynamic gather/scatter/indexing for MoE: one-hot, cumulative sum, capacity compare, nonzero, where, per-expert gather, `index_add_`.

Neural network primitives:

- Bias-free Linear: embeddings/LM head, Q/K/V/O, dense FFN `wi: d_model -> d_ff`, `wo: d_ff -> d_model`, router `d_model -> num_experts`.
- RMSNorm/T5LayerNorm: mean of squared hidden states over last dim in fp32, rsqrt, scale only, no bias and no mean subtraction.
- Activations: `relu` in most checkpoints, `gelu` in `switch-xxl-128`; `dense_act_fn` is source-selected.
- Residual add and dropout; inference dropout should be disabled.
- Optional fp16 clamp after attention/FFN is training-stability behavior; eval path should not hit it unless infinities exist.

Attention primitives:

- Dense MHA self-attention and encoder-decoder cross-attention.
- Matmul QK^T, add relative/mask bias, softmax in fp32, dropout, matmul AV, output projection.
- T5 relative-position bias only on the first self-attention layer of each stack; following layers receive no reusable bias in this implementation because `position_bias` is not propagated out of `SwitchTransformersBlock`.

MoE routing ops:

- Router Linear over flattened tokens `[B*S,d_model] -> [B*S,num_experts]`.
- Softmax over experts in `router_dtype`, then top-1 via `max`.
- The returned "routing weights" consumed by experts are selected max probabilities with shape `[B*S,1]`; raw classifier logits are not consumed by the inference FFN path.
- One-hot selected expert, cumulative token priority per expert over flattened token order, capacity mask, dropped overflow tokens.
- Per-expert FFN on selected token rows and `index_add_` back into `[B*S,d_model]`.

Generation/cache ops:

- Encoder-decoder cache wrapper containing self-attention dynamic cache and cross-attention cache.
- Self-attention cache appends per layer with shape `[B,num_heads,past,d_kv]`.
- Cross-attention cache stores projected encoder K/V once per decoder layer and reuses on later decode steps.
- Reorder cache for beams is inherited through Transformers cache/generation utilities, not custom in this file.

## 5. Layer/block breakdown

Encoder block, repeated `num_layers`:

```text
x0: [B,S,d_model]
y = RMSNorm(x0)
q = Linear_q(y) -> [B,S,num_heads*d_kv] -> [B,H,S,D]
k = Linear_k(y) -> [B,H,S,D]
v = Linear_v(y) -> [B,H,S,D]
scores = q @ k^T -> [B,H,S,S]
scores += encoder mask + optional first-layer relative position bias [1,H,S,S]
a = softmax(scores.float(), dim=-1).to(scores.dtype)
attn = a @ v -> [B,H,S,D] -> [B,S,H*D]
x1 = x0 + Linear_o(attn)
z = RMSNorm(x1)
ff = DenseFFN(z) or SparseMoEFFN(z)
out = x1 + ff
```

Decoder block, repeated `num_decoder_layers`:

```text
x0: [B,T,d_model]
self = causal T5 self-attention(x0, decoder self KV cache)
x1 = x0 + self_output
cross = T5 cross-attention(RMSNorm(x1), encoder_hidden_states, cross KV cache)
x2 = x1 + cross_output
z = RMSNorm(x2)
ff = DenseFFN(z) or SparseMoEFFN(z)
out = x2 + ff
```

Dense FFN:

```text
hidden -> Linear(d_model,d_ff,bias=False) -> activation -> Linear(d_ff,d_model,bias=False)
```

Sparse MoE FFN:

```text
flat = hidden.view(B*S, d_model)
classifier_logits = Linear(flat, num_experts, bias=router_bias)
router_probs = softmax(classifier_logits, dim=-1, dtype=router_dtype)
score, expert = max(router_probs, dim=-1)
dispatch = one_hot(expert) masked by cumulative capacity <= expert_capacity
for each hit expert:
  rows = where(dispatch for expert)
  expert_out = DenseFFN(flat[rows]) * score[rows]
  final.index_add_(rows, expert_out)
reshape final to [B,S,d_model]
```

## 6. Attention requirements

Required variants:

- Encoder self-attention: noncausal dense MHA, no KV cache, mask additive with `torch.finfo(dtype).min`.
- Decoder self-attention: causal dense MHA with autoregressive KV cache.
- Decoder cross-attention: dense MHA over encoder hidden states; K/V can be cached after first generated token.

Shapes:

- Q/K/V projection width: `inner_dim = num_heads * d_kv`.
- Query/key/value head shape: `[B,num_heads,Q,d_kv]`, `[B,num_heads,K,d_kv]`.
- Attention score shape: `[B,num_heads,Q,K]`.
- Output projection input width: `inner_dim`, output width `d_model`.

Masking and math order:

- QK scores are not scaled by `1/sqrt(d_kv)` in source; parity needs T5/Switch unscaled score behavior.
- Position/mask bias is added before softmax.
- Softmax explicitly computes in fp32 via `scores.float()`, then casts back to score dtype.
- Relative bias is bidirectional in encoder and causal/unidirectional in decoder.
- Cross-attention has no relative bias in this source.

Cache:

- Self-attention cache stores encoded K/V after projection/head reshape and after relative-position-independent projection. Relative bias uses `past_seen_tokens` for decoder positions.
- Cross-attention cache stores K/V projected from `encoder_hidden_states`; `EncoderDecoderCache.is_updated[layer_idx]` gates reuse.
- No GQA/MQA repeat expansion is needed.
- FlashAttention/SDPA compatibility requires custom handling for unscaled scores and T5 relative bias. A fused backend can accept additive bias, but must disable default scale or use scale `1.0`.

## 7. Position encoding and custom math

SwitchTransformers uses T5 relative-position bias, not absolute position embeddings or RoPE. The embedding table has shape `[relative_attention_num_buckets, num_heads]` and is present only in the first self-attention block of encoder and decoder stacks.

Concise bucket logic:

```python
def switch_relative_bucket(memory_pos_minus_query_pos, bidirectional, num_buckets, max_distance):
    bucket = 0
    rp = memory_pos_minus_query_pos
    if bidirectional:
        num_buckets //= 2
        bucket += (rp > 0).long() * num_buckets
        rp = abs(rp)
    else:
        rp = -minimum(rp, 0)
    max_exact = num_buckets // 2
    large = max_exact + (log(rp.float() / max_exact) / log(max_distance / max_exact) * (num_buckets - max_exact)).long()
    large = minimum(large, num_buckets - 1)
    return bucket + where(rp < max_exact, rp, large)
```

Precompute opportunities:

- Encoder full-sequence relative bucket matrix can be precomputed for fixed `S`.
- Decoder prefill bucket matrix can be precomputed per `(T,K)` and `past_seen_tokens=0`.
- Decode step bias depends on `past_seen_tokens`; for one-token decode it is a row over `K=past+1`.

## 8. Preprocessing and input packing

The model consumes normal T5-style text tensors:

- `input_ids`: `[B,S_enc]`, integer token ids.
- `attention_mask`: `[B,S_enc]`, 1 for valid tokens and 0 for padding.
- `decoder_input_ids`: `[B,S_dec]`; during training, labels are shifted right with `decoder_start_token_id`, usually `pad_token_id=0`.
- `decoder_attention_mask`: optional `[B,S_dec]`; if omitted source creates ones adjusted for cache length.

Generation-controller behavior outside the compiled graph:

- Task prefixes in `task_specific_params`, such as summarization/translation prefixes in `switch-base-8`, are tokenizer/controller work.
- `decoder_start_token_id=0`, `pad_token_id=0`, `eos_token_id=1` in representative configs.
- Beam search, no-repeat ngram, length penalties, and task max lengths are generation-controller policy, not model ops.

There are no image/audio/video tensors, placeholder-token embedding scatters, cu_seqlens inputs, or packed varlen metadata.

## 9. Graph rewrite / lowering opportunities

### Rewrite: tied LM head alias and output scale

Source pattern:

```text
if tie_word_embeddings: sequence_output *= d_model ** -0.5
lm_logits = Linear(sequence_output, shared.weight, bias=False)
```

Replacement:

```text
ScaledGEMM(sequence_output, shared_embedding_weight.T, alpha=d_model^-0.5)
```

Preconditions:

- `tie_word_embeddings=True`.
- `lm_head.weight` aliases `shared.weight`.
- No logits processors folded into graph.

Failure cases: untied LM head, quantized/packed embedding storage without a matching GEMM materialization policy.

Parity test sketch: compare logits for random decoder hidden states and real tied weights for base and asymmetric configs.

### Rewrite: router top-1 dispatch to segmented expert GEMM

Source pattern:

```text
one_hot(argmax(softmax(router_linear(flat)))) -> cumsum capacity mask -> per-expert gather -> expert FFN -> index_add
```

Replacement:

```text
RouterTop1Capacity(flat) -> per-expert compact token buffers -> grouped GEMM wi/wo -> scatter-add
```

Preconditions:

- `num_selected_experts == 1` or ignored by source.
- `router_ignore_padding_tokens` either false or explicitly rejected until implemented.
- Stable flattened token order `[batch, sequence]`.
- Capacity exactly `expert_capacity` per expert; dropped tokens must produce zero FFN contribution.
- Inference/eval mode, so no jitter and dropout disabled.

Shape equations:

- Input tokens `N = B*S`.
- Router logits `[N,E]`.
- Per-expert compact capacity max `[E,C,d_model]`, where `C=expert_capacity`.
- Expert weights `wi[e]: [d_ff,d_model]`, `wo[e]: [d_model,d_ff]` in PyTorch linear orientation.

Failure cases: training jitter, padding-ignore routing, top-k routing, custom router remote code, dynamic capacity not captured in manifest.

Parity test sketch: fixed router weights with capacity overflow; assert dropped rows are zero and expert rows match PyTorch `index_add_`.

### Rewrite: T5 attention into fused bias attention

Source pattern:

```text
scores = q @ k.transpose(-1,-2)
scores += relative_position_bias + mask
attn = softmax(scores.float()).to(scores.dtype)
out = attn @ v
```

Replacement:

```text
FusedAttention(q,k,v, additive_bias, scale=1.0, fp32_softmax=True)
```

Preconditions:

- Dense attention, no local/sparse pattern.
- Backend supports arbitrary additive bias and `scale=1.0`.
- Relative bias table lookup either precomputed or fused as a small bias kernel.

Failure cases: backend assumes `1/sqrt(D)` scaling, cannot express `finfo(dtype).min` mask, or cannot handle rectangular cross-attention.

### Rewrite: first-layer relative bias precompute

Preconditions:

- Fixed max/bucketed sequence lengths.
- Same `relative_attention_num_buckets` and `relative_attention_max_distance`.
- Decoder decode tracks `past_seen_tokens`.

Replacement: cache bucket ids or bias tensors per length case; emit embedding gather only when length changes.

## 10. Kernel fusion candidates

Highest priority:

- RMSNorm: every attention and FFN sublayer uses T5 RMSNorm; fp32 accumulation with scale-only output is required.
- Dense T5 attention with additive relative/mask bias: core encoder, prefill, and decode bottleneck.
- RouterTop1Capacity + expert dispatch: the distinctive Switch cost and correctness risk; needs deterministic drop semantics.
- Expert FFN grouped GEMM: many independent `d_model -> d_ff -> d_model` FFNs, especially 128 to 2048 experts.

Medium priority:

- Router linear + softmax + argmax + cumsum capacity in one kernel or tight pipeline.
- LM head scaled tied GEMM, especially for large vocab and decode last-token path.
- Decode cache update and one-token relative bias row generation.
- Cross-attention K/V projection cache materialization after encoder.

Lower priority:

- Router z-loss and load-balancing loss; training/diagnostic only for first inference.
- Dropout and jitter paths; disabled in eval.
- Hidden-state/attention output recording; useful for debug but not core runtime.

## 11. Runtime staging plan

Stage 1: config and weights

- Parse Switch config including asymmetric layer counts and sparse steps.
- Load shared/tied embeddings, attention weights, dense FFN weights, per-expert weights, router weights, and relative bias tables.
- Reject ignored historical router/gated flags unless matching current source behavior is confirmed.

Stage 2: one dense block parity

- Implement RMSNorm, T5 unscaled attention with relative bias, dense FFN, residuals.
- Validate encoder and decoder dense layers with dropout disabled.

Stage 3: sparse FFN parity

- Implement source-equivalent top-1 router, capacity mask, token dropping, per-expert execution, and scatter-add.
- Start with simple loop or CPU reference; then add CUDA compact/grouped path.

Stage 4: encoder full-stack parity

- Run encoder-only for `SwitchTransformersEncoderModel`; cache encoder output as reusable input to decoder.

Stage 5: seq2seq prefill logits

- Compile full `ForConditionalGeneration` prefill with cross-attention and LM head.

Stage 6: decode with caches

- Add decoder self KV cache append, cross K/V cache reuse, one-token decode, and last-token logits.

Stage 7: optimized kernels

- Fused attention, RouterTop1Capacity, grouped expert GEMM, scaled tied LM head, and batch/sequence bucket profiling.

Initially stub or defer router losses, output attentions/hidden states, generation beam policies, dropout, training jitter, and gradient checkpointing.

## 12. Parity and validation plan

- Unit tests for RMSNorm against Transformers/T5 with fp32 accumulation and bf16/fp16 output.
- Unit tests for relative-position buckets with bidirectional encoder and causal decoder cases, including long distances beyond max distance.
- Attention parity for self/cross attention with unscaled scores, additive masks, relative bias, and fp32 softmax.
- Router parity with fixed small weights: verify argmax, one-hot shape, cumsum capacity, dropped tokens, and routing weights.
- Sparse MLP parity with `expert_capacity=0` must produce all-zero FFN contribution, matching HF test coverage.
- Single encoder block parity for dense and sparse layers.
- Single decoder block parity with and without encoder cross-attention.
- Full encoder parity for `google/switch-base-8` bf16 and one asymmetric config.
- Prefill logits parity for `SwitchTransformersForConditionalGeneration`.
- Decode parity: compare next-token logits over several decode steps with cache enabled and disabled.
- Recommended tolerances: fp32 `rtol=1e-4, atol=1e-5`; bf16/fp16 attention/MoE `rtol=1e-2, atol=1e-2` initially, tighten per kernel.

## 13. Performance probes

- Encoder-only throughput over `(B,S_enc)` and expert count.
- Decoder prefill throughput over `(B,S_dec,S_enc)`.
- Decode tokens/sec with cache for batch size and beam/batch-expanded cases.
- Router distribution probe: token drop rate and expert utilization versus batch/sequence length.
- Expert grouped GEMM probe: compare per-expert loop, compacted batched GEMM, grouped GEMM, and dense fallback.
- Attention backend probe: unfused matmul/softmax/matmul versus fused bias attention with scale 1.0.
- KV cache memory and bandwidth by layer count, `num_heads`, `d_kv`, and sequence length.
- LM head last-token GEMM throughput, tied scaled path versus generic projection.
- Config sweep: base-8, base-128, large-128, xxl-128, switch-c-2048 because each stresses different axes.

## 14. Skip/defer list

- Training losses: cross-entropy labels, router z-loss, and auxiliary load-balancing loss.
- Router jitter noise and dropout.
- Gradient checkpointing.
- `output_attentions`, `output_hidden_states`, and `output_router_logits` as public ABI outputs.
- Beam-search controller details beyond cache reorder support.
- Quantization and packed weights; no source-coupled quantized format is implemented here.
- Historical config behavior not read by current source: batch-prioritized routing, padding-ignore routing, top-k experts, gated FFN flags.
- Multi-GPU tensor parallelism and expert parallel placement.
- Tokenizer special-case parity beyond producing standard T5 input tensors.

## 15. Final implementation checklist

- [ ] Parse `SwitchTransformersConfig`, including sparse-step-derived layer layout.
- [ ] Load and alias shared encoder/decoder embeddings and tied LM head.
- [ ] Implement T5 RMSNorm.
- [ ] Implement T5 relative-position bucket and bias lookup.
- [ ] Implement unscaled dense MHA with fp32 softmax and additive masks.
- [ ] Implement decoder self-attention KV cache append.
- [ ] Implement encoder-decoder cross-attention K/V cache reuse.
- [ ] Implement dense FFN `Linear -> activation -> Linear`.
- [ ] Implement source-equivalent top-1 router with capacity drop semantics.
- [ ] Implement per-expert gather/FFN/scatter-add reference path.
- [ ] Add CUDA RouterTop1Capacity and grouped expert GEMM provider plan.
- [ ] Add scaled tied LM head rewrite.
- [ ] Add encoder block, decoder block, encoder stack, prefill logits, and decode parity tests.
- [ ] Benchmark encoder, prefill, decode, router, expert GEMM, attention, and LM head separately.
