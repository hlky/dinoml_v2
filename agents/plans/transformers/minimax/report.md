# MiniMax Transformers Family Audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: MiniMaxAI/MiniMax-Text-01-hf
Config source: Hugging Face raw config.json, downloaded 2026-05-13
Source files inspected:
- X:/H/transformers/src/transformers/models/minimax/configuration_minimax.py
- X:/H/transformers/src/transformers/models/minimax/modeling_minimax.py
- X:/H/transformers/src/transformers/models/minimax/modular_minimax.py
- X:/H/transformers/tests/models/minimax/test_modeling_minimax.py
- X:/H/transformers/docs/source/en/model_doc/minimax.md
Any missing files or assumptions:
- modeling_minimax.py and configuration_minimax.py are generated from modular_minimax.py. Future source edits should target modular_minimax.py, but this report treats generated modeling_minimax.py as the runtime source basis.
- MiniMaxAI/MiniMax-Text-01-hf is not gated. The full checkpoint is very large and sharded; this audit uses config/tokenizer metadata and source inspection, not weights.
- MiniMaxAI/MiniMax-M1-40k and MiniMaxAI/MiniMax-M1-80k configs were downloaded as related snapshots, but they declare model_type minimax_m1 with remote custom code and are out of scope for this in-library minimax report.
```

Local snapshots written beside this report:

- `MiniMaxAI_MiniMax-Text-01-hf_config.json`
- `MiniMaxAI_MiniMax-Text-01-hf_tokenizer_config.json`
- `hf-internal-testing_MiniMax-tiny_config.json`
- `MiniMaxAI_MiniMax-M1-40k_config.json`
- `MiniMaxAI_MiniMax-M1-80k_config.json`

HF repo metadata consulted with the Hugging Face plugin and raw Hub API:

- [MiniMaxAI/MiniMax-Text-01-hf](https://hf.co/MiniMaxAI/MiniMax-Text-01-hf): public, architecture `minimax`, task `text-generation`, license metadata `other`.
- [hf-internal-testing/MiniMax-tiny](https://hf.co/hf-internal-testing/MiniMax-tiny): public test checkpoint, architecture `minimax`.
- [MiniMaxAI/MiniMax-M1-40k](https://hf.co/MiniMaxAI/MiniMax-M1-40k) and [MiniMaxAI/MiniMax-M1-80k](https://hf.co/MiniMaxAI/MiniMax-M1-80k): public but `minimax_m1`, `custom_code`, separate audit target.

## 2. High-level architecture

Primary DinoML target: inference-only causal language modeling for `MiniMaxForCausalLM`.

MiniMax is a text-only decoder with hybrid attention and sparse MoE:

```text
tokenizer/chat template -> input_ids/attention_mask -> token embedding
-> repeated decoder layers:
   RMSNorm -> full causal GQA attention or lightning attention -> scaled residual
   RMSNorm -> top-2 sparse MoE SwiGLU experts -> scaled residual
-> final RMSNorm -> LM head -> logits/sampling
```

Stage decomposition:

- CPU/data pipeline: GPT2Tokenizer-compatible tokenization, chat template expansion, tool/function prompt formatting.
- Prefill: full prompt through all layers. Full-attention layers build KV cache; lightning layers build recurrent `[B, heads, head_dim, head_dim]` linear cache.
- Decode: full-attention layers append one-token KV; lightning layers update fixed-size linear state token by token.
- Optional heads: sequence classification, token classification, and QA are implemented through generic wrappers but are deferred for the first DinoML target.

## 3. Important config dimensions

| Field | MiniMax-Text-01-hf config | HF tiny config | Source default |
| --- | ---: | ---: | ---: |
| `model_type` | `minimax` | `minimax` | `minimax` |
| `vocab_size` | 200064 | 3200 | 32000 |
| `hidden_size` | 6144 | 1024 | 4096 |
| `num_hidden_layers` | 80 | 2 | 32 |
| `num_attention_heads` | 64 | 32 | 32 |
| `num_key_value_heads` | 8 | 8 | 8 |
| `head_dim` | 128 | omitted/null -> 32 | omitted/null -> 128 |
| Q width | 8192 | 1024 | 4096 |
| KV width each | 1024 | 256 | 1024 |
| `intermediate_size` | 9216 | 3584 | 14336 |
| `num_local_experts` | 32 | 8 | 8 |
| `num_experts_per_tok` | 2 | 2 | 2 |
| `hidden_act` | `silu` | `silu` | `silu` |
| `max_position_embeddings` | 10240000 | 32768 | 131072 |
| `rope_theta` / effective RoPE theta | 10000000 | 1000000.0 | 1000000.0 |
| `rotary_dim` | 64 in config, not directly read by current source | absent | absent |
| `block_size` | omitted -> source default 256 | 256 | 256 |
| `sliding_window` | null | null | null |
| `use_cache` | true | true | true |

Representative checkpoint sweep:

| Repo | Status for this report | Operator-significant notes |
| --- | --- | --- |
| `hf-internal-testing/MiniMax-tiny` | In scope test/debug | 2 layers, explicit `[full_attention, linear_attention]`, small vocab, fp32 safetensors metadata. Useful for parity and cache tests. |
| `MiniMaxAI/MiniMax-Text-01-hf` | In scope production | 80 layers, 64 Q heads, 8 KV heads, 32 experts, top-2 routing, 10.24M max positions, layer pattern has one full-attention layer after runs of lightning attention. |
| `MiniMaxAI/MiniMax-M1-40k` | Out of scope | `model_type=minimax_m1`, `auto_map` remote code, historical config names like `attn_type_list` and `layernorm_*`. Do not load with this report's `minimax` source. |
| `MiniMaxAI/MiniMax-M1-80k` | Out of scope | Same `minimax_m1` remote-code family as 40k. Separate audit needed. |

## 3a. Family variation traps

- `hidden_size != num_attention_heads * head_dim` for MiniMax-Text-01-hf: 6144 vs 64 * 128 = 8192. Attention output projection maps 8192 -> 6144.
- Full attention is GQA: 64 query heads, 8 KV heads, repeat factor 8.
- The source config uses `rope_parameters`; older checkpoint configs expose `rope_theta` and `rotary_dim`. Current source reads `config.rope_parameters["rope_theta"]`; DinoML config loading should normalize legacy RoPE fields before graph construction.
- `rotary_dim=64` appears in the production config, but current in-library `MiniMaxRotaryEmbedding.compute_default_rope_parameters` uses `head_dim` for RoPE dimension. Treat `rotary_dim` as ignored by this source basis unless a remote-code audit says otherwise.
- Layer type is not a simple alternation in production. MiniMax-Text-01-hf has 70 lightning layers and 10 full-attention layers, with full attention at indices 7, 15, ..., 79.
- Source defaults would alternate full/linear by layer parity if `layer_types` is omitted. Production config overrides this.
- Lightning attention ignores RoPE and causal mask semantics from full attention. It receives the raw `attention_mask` and only masks values when no linear cache exists.
- `MiniMaxCache` is custom and does not support `crop`; speculative/prompt-lookup generation paths that require cache crop are skipped in Transformers tests.
- All projection modules in inspected source are bias-free.
- MoE expert weights are packed 3D tensors, not per-expert child Linear modules: `gate_up_proj[E, 2I, H]`, `down_proj[E, H, I]`.
- `MiniMaxForCausalLM` declares tied weight keys for `lm_head.weight` and `model.embed_tokens.weight`, but config has `tie_word_embeddings=false`. Preserve config-controlled aliasing rather than forcing a tie.
- `postnorm`, `shared_intermediate_size`, and `shared_moe_mode` exist in the production config but are not read by current in-library `modeling_minimax.py`; route remote/historical behavior to a separate audit.

## 4. Operator coverage checklist

Tensor/layout ops:

- Embedding lookup `[B,S] -> [B,S,H]`.
- `arange`, unsqueeze, position id construction with past length offset.
- Reshape/view/transpose/contiguous for `[B,S,H] <-> [B,heads,S,D]`.
- Split/chunk packed QKV in lightning attention.
- Concatenate along sequence axis for lightning block outputs.
- Dynamic slice for `logits_to_keep` over sequence dimension.

Neural network primitives:

- Bias-free Linear/GEMM:
  - Token LM head: `Linear(H -> vocab_size)`.
  - Full attention: `q_proj H -> A*D`, `k_proj H -> K*D`, `v_proj H -> K*D`, `o_proj A*D -> H`.
  - Lightning attention: packed `qkv_proj H -> 3*A*D`, `output_gate H -> A*D`, `out_proj A*D -> H`.
  - Router: `F.linear([B*S,H], weight[E,H]) -> [B*S,E]`.
  - Experts: per selected expert `Linear(H -> 2I)` then `Linear(I -> H)` from packed 3D weights.
- RMSNorm with fp32 variance accumulation.
- SiLU for MoE gate branch and lightning QKV projection, sigmoid output gate in lightning attention.
- Elementwise multiply/add with source-specific residual scale factors.

Attention primitives:

- Full causal GQA attention with RoPE, optional sliding-window mask, KV cache.
- Backend-dispatched attention via Transformers `ALL_ATTENTION_FUNCTIONS` with eager fallback, SDPA, FlashAttention, and Flex attention support flags.
- Lightning attention custom block recurrence:
  - Prefill block intra attention `(Q @ K.T) * diagonal_decay @ V`.
  - Inter-block state `KV = decay * KV + (K * key_decay).T @ V`.
  - Decode state update `KV = exp(-slope_rate) * KV + K.T @ V`.

Position/rotary ops:

- Standard RoPE cos/sin computed in fp32 from `position_ids`.
- `rotate_half` and apply to Q/K for full-attention layers only.

Generation/cache ops:

- Custom `MiniMaxCache`.
- Full-attention KV cache per full layer: keys/values `[B, num_key_value_heads, T, head_dim]`.
- Lightning cache per linear layer: `[B, num_attention_heads, head_dim, head_dim]`.
- Cache reorder for beam/batch selection uses `batch_repeat_interleave` and `batch_select_indices`; `crop` unsupported.

Sparse/MoE ops:

- Router softmax fp32 over experts.
- Top-k expert selection.
- One-hot expert mask, expert hit detection, `where`, per-expert token gather.
- Per-token expert weighted output and `index_add_` accumulation.

Preprocessing-coupled ops:

- GPT2Tokenizer vocabulary/merges.
- Chat template constructs system/user/assistant/function/tool blocks. This is ABI/controller work, not a GPU graph op.

Distributed/tensor-parallel hints:

- Source config includes TP plan names: column-wise Q/K/V, row-wise O/down, packed column-wise expert gate_up, MoE expert TP. Treat as future distributed lowering metadata.

## 5. Layer/block breakdown

Decoder block, repeated `num_hidden_layers` times:

```text
x = RMSNorm(x)
residual = x
if layer_type == "full_attention":
    q = Linear(H -> A*D, bias=False)(x).view(B,S,A,D).transpose(1,2)
    k = Linear(H -> K*D, bias=False)(x).view(B,S,K,D).transpose(1,2)
    v = Linear(H -> K*D, bias=False)(x).view(B,S,K,D).transpose(1,2)
    q,k = RoPE(q,k, cos, sin)
    k,v = KVCache.update(k,v, layer_idx)
    y = causal_or_sliding_GQA(q,k,v, repeat_kv= A/K, scale=D^-0.5)
    y = Linear(A*D -> H, bias=False)(y)
else:
    qkv = silu(Linear(H -> 3*A*D, bias=False)(x))
    q,k,v = split(qkv.view(B,S,A,3*D), D, dim=-1).transpose_to_BHSD()
    y = lightning_attention(q,k,v, attention_mask, linear_cache)
    y = RMSNorm(A*D)(y)
    y = sigmoid(Linear(H -> A*D, bias=False)(x)) * y
    y = Linear(A*D -> H, bias=False)(y)
x = residual * attn_alpha_factor + y * attn_beta_factor
x = RMSNorm(x)
residual = x
router_logits = Linear(H -> E, bias=False)(x.reshape(B*S,H))
router_probs = softmax(router_logits.float(), dim=-1)
top_values, top_indices = topk(router_probs, k=num_experts_per_tok)
top_values = top_values / sum(top_values, dim=-1)
expert_y = sum_selected_experts(weight * down(silu(gate) * up))
x = residual * mlp_alpha_factor + expert_y.reshape(B,S,H) * mlp_beta_factor
```

For MiniMax-Text-01-hf, `H=6144`, `A=64`, `K=8`, `D=128`, `E=32`, `I=9216`, and `A*D=8192`.

## 6. Attention requirements

Full-attention layers:

- Causal self-attention.
- GQA with `num_attention_heads / num_key_value_heads` repeat groups.
- Q shape `[B, A, Q, D]`; K/V cache shape `[B, K, T, D]`; repeated K/V logical shape `[B, A, T, D]`.
- Attention scores are `(Q @ K.T) * D^-0.5 + mask`.
- Eager fallback softmax is fp32 then cast back to query dtype.
- RoPE is applied before cache update, so cached keys are position-encoded.
- `sliding_window` is passed to attention backend when configured; production config uses null.

Lightning-attention layers:

- Custom recurrent linear attention, not KV-cache attention.
- No RoPE use despite receiving `position_embeddings`.
- Prefill state is initialized to zeros `[B,A,D,D]`.
- Prefill processes sequence in `ceil(S / block_size)` blocks and uses causal-like `diagonal_decay` within each block.
- Decode path uses the existing linear cache and updates one token at a time, although source loops over `seq_len`.
- Padding mask is applied to V only when there is no existing linear cache.

Cache ABI:

- `MiniMaxCache` owns both DynamicCache KV layers and a parallel `linear_cache` list.
- Full-attention cache length drives `past_seen_tokens`; if only linear layers have cache, source still relies on full-attention layers being present for sequence length.
- `batch_repeat_interleave` and `batch_select_indices` must handle both cache families.
- `crop` is intentionally unsupported.

## 7. Position encoding and custom math

RoPE is standard full-head RoPE in the current in-library source:

```python
def minimax_rope(config, position_ids, dtype):
    dim = config.head_dim or config.hidden_size // config.num_attention_heads
    theta = config.rope_parameters["rope_theta"]
    inv_freq = 1.0 / (theta ** (arange(0, dim, 2).float() / dim))
    freqs = (inv_freq[None, :, None] @ position_ids[:, None, :].float()).transpose(1, 2)
    emb = cat([freqs, freqs], dim=-1)
    return cos(emb).to(dtype), sin(emb).to(dtype)

def apply_minimax_rope(q, k, cos, sin):
    cos = cos[:, None, :, :]
    sin = sin[:, None, :, :]
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin
```

Lightning attention decay factors are source-specific and should be reproduced exactly:

```python
base = 1 / (2 ** (8 / num_attention_heads))
slope_rate[h] = base ** (h + 1) * (1 - layer_idx / (num_hidden_layers - 1 + 1e-5) + 1e-5)
query_decay[h,t] = exp(-slope_rate[h] * (t + 1))
key_decay[h,t] = exp(-slope_rate[h] * (block_size - (t + 1)))
diagonal_decay[h,i,j] = exp(-slope_rate[h] * (i - j)) if i >= j else 0
```

Precompute candidates:

- `slope_rate`, `query_decay`, `key_decay`, and `diagonal_decay` are registered buffers and can be constants per layer/config.
- RoPE `inv_freq` is constant; cos/sin depends on runtime `position_ids` and dtype.

## 8. Preprocessing and input packing

MiniMax is text-only. No image/audio/video tensors are required.

Tokenizer contract from `MiniMaxAI_MiniMax-Text-01-hf_tokenizer_config.json`:

- `tokenizer_class`: `GPT2Tokenizer`.
- `bos_token`: `<beginning_of_sentence>`.
- `eos_token`: `<end_of_sentence>`.
- `unk_token`: `<end_of_document>`.
- `model_max_length`: 40960000.
- Chat template emits role-tagged text blocks for system/user/assistant/function/tool settings.

GPU graph inputs:

- `input_ids [B,S]` or mutually exclusive `inputs_embeds [B,S,H]`.
- Optional `attention_mask [B,S]`, used by causal mask construction for full attention and raw value masking for lightning attention prefill.
- Optional `position_ids [B,S]`; if absent, source constructs `[0..S-1] + past_seen_tokens`.

Generation controller work outside the graph:

- Chat template application.
- Sampling/beam search.
- Handling unsupported cache crop paths.
- `logits_to_keep` can reduce final LM-head GEMM to last token(s) during decode.

## 9. Graph rewrite / lowering opportunities

### Rewrite: packed lightning QKV projection

Source pattern:

```text
silu(Linear(H -> 3*A*D)(x)) -> view(B,S,A,3D) -> split(q,k,v)
```

Replacement:

```text
one GEMM -> fused SiLU -> logical split views
```

Preconditions:

- Weight shape `[3*A*D, H]`.
- Bias is absent.
- Split order is `[query, key, value]` along the last projected dimension after view to `[B,S,A,3D]`.

Failure cases:

- If a remote-code variant moves SiLU after split or uses partial rotary dim, reject this rewrite.

Parity test sketch:

- Compare projected q/k/v tensors before attention for random fp32/fp16 inputs and production dimensions scaled down.

### Rewrite: full-attention QKV grouped projection

Source pattern:

```text
q_proj(x), k_proj(x), v_proj(x) as three bias-free linears
```

Replacement:

```text
packed GEMM H -> (A*D + 2*K*D), then split [Q, K, V]
```

Preconditions:

- Same input `x`, no bias, no intervening operations.
- Preserve separate weight names or provide deterministic packing metadata.
- Split order must be all-Q, all-K, all-V, not per-head interleave.

Failure cases:

- Tensor-parallel loading that expects separate modules unless packing metadata is artifact-visible.

### Rewrite: lightning prefill block kernel

Source pattern:

```text
for block:
  intra = (Qb @ Kb.T) * diagonal_decay @ Vb
  inter = (Qb * query_decay) @ KV_state
  KV_state = KV_state * block_decay + (Kb * key_decay).T @ Vb
```

Replacement:

```text
custom block-linear-attention provider
```

Preconditions:

- Static or bucketed `block_size`.
- Head dim and block size within kernel limits.
- Padding mask has already zeroed V for prefill.

Failure cases:

- Decode with existing cache follows a different recurrence.
- Output attention matrices are not meaningful for lightning layers.

### Rewrite: packed MoE expert GEMM

Source pattern:

```text
topk router -> per-expert token gather -> gate_up linear -> chunk -> silu(gate)*up -> down linear -> weighted index_add
```

Replacement:

```text
top-k routing provider + grouped GEMM experts + scatter-add
```

Preconditions:

- `num_experts_per_tok=2`.
- Expert weight layout is `gate_up_proj[E, 2I, H]` and `down_proj[E, H, I]`.
- Activation is SiLU.

Failure cases:

- Training jitter noise, router aux loss, or expert parallel sharding not supported in first inference path.

## 10. Kernel fusion candidates

Highest priority:

- RMSNorm with fp32 accumulation. It appears before every attention/MLP and after lightning attention.
- Full GQA attention with RoPE and KV cache. This covers the full-attention layers and decode correctness.
- Lightning attention prefill/decode provider. Eager Python loops over blocks/tokens will dominate long context.
- Sparse MoE top-2 grouped expert GEMM and scatter-add. Production model has 32 experts per layer and MoE every layer.

Medium priority:

- QKV projection packing for full attention and lightning attention.
- SwiGLU expert fusion: `gate_up -> chunk -> silu(gate) * up`.
- Last-token-only LM head via `logits_to_keep`.
- Router softmax + top-k fusion for `[B*S, E]` with small E.

Lower priority:

- Router aux loss. Training/eval diagnostic only for first inference target.
- Sequence/token classification and QA heads.
- Tensor-parallel plan lowering.
- Quantization through BitsAndBytes. Treat as loader/backend work, not core graph parity.

## 11. Runtime staging plan

Stage 1: Parse config and load tiny checkpoint metadata. Normalize legacy `rope_theta` into `rope_parameters`; reject `minimax_m1`/remote-code configs.

Stage 2: Implement one full-attention block parity with RMSNorm, RoPE, GQA, KV cache, and MoE stubbed or dense-disabled for tiny synthetic configs.

Stage 3: Implement MiniMaxSparseMoeBlock for inference with top-2 routing and packed expert weights.

Stage 4: Implement lightning attention prefill without cache, using exact block recurrence and decay buffers.

Stage 5: Implement custom `MiniMaxCache` with full KV and linear state, then decode parity.

Stage 6: Run end-to-end tiny checkpoint logits and greedy generation parity against `hf-internal-testing/MiniMax-tiny`.

Stage 7: Add production-shape compile/load admission for MiniMax-Text-01-hf, initially with dense weights and clear memory limits.

Stage 8: Add optimized providers: FlashAttention-compatible full layers, lightning provider, grouped MoE GEMM, and optional distributed/runtime weight residency.

## 12. Parity and validation plan

- Config parsing tests:
  - Production config normalizes RoPE theta and preserves ignored historical fields as ignored metadata.
  - M1 configs are rejected or routed to a separate `minimax_m1` target.
- RMSNorm random tensor tests in fp32/fp16/bf16, tolerance fp32 `1e-5`, fp16/bf16 `1e-2`.
- RoPE tests against Transformers for nonzero `position_ids` and cache offset.
- Full attention single-layer tests for prefill and one-token decode, including GQA repeat and cached position encoding.
- Lightning attention tests:
  - Decay buffer values by layer index.
  - Block prefill for `S < block_size`, `S == block_size`, and tail block.
  - Decode recurrence with existing `[B,A,D,D]` state.
  - Attention-mask value zeroing in prefill.
- MoE tests:
  - Router top-k normalization.
  - Packed expert layout split.
  - Duplicate-token accumulation via `index_add`.
- End-to-end tests:
  - `hf-internal-testing/MiniMax-tiny` logits slice parity.
  - Tiny greedy generation parity for 5 new tokens.
  - Production config graph construction without weight materialization.

## 13. Performance probes

- Tokenizer/chat-template throughput for long prompts.
- Prefill full-attention layer throughput by sequence length and batch.
- Lightning attention prefill throughput by `block_size`, sequence length, and head dim.
- Decode tokens/sec split by full-attention KV layers vs lightning recurrent layers.
- MoE router/top-k latency and grouped expert GEMM occupancy.
- Expert load balance histogram for representative prompts.
- LM-head full-vocab vs `logits_to_keep=1` decode cost.
- KV cache memory for 10 full-attention layers at long context.
- Lightning cache memory for 70 linear layers: `layers * B * A * D * D * dtype_size`.
- Dense weight loading and possible GGUF/dequant/offload experiments as separate provider probes.

## 14. Skip/defer list

- Training, losses, gradient checkpointing, and router aux loss in the first inference runtime.
- Sequence classification, token classification, and QA heads.
- Cache `crop`, prompt lookup decoding, assisted/speculative decoding.
- `minimax_m1`, `minimax_m2`, MiniMax-VL, and any remote-code variants.
- BitsAndBytes quantization. First DinoML path should load dense or separately audited GGUF/encoded weights.
- Tensor-parallel and pipeline-parallel execution despite source metadata.
- Sliding-window attention unless a target config sets `sliding_window`.
- Output attentions for lightning layers.

## 15. Final implementation checklist

- [ ] Parse `MiniMaxConfig` and normalize legacy RoPE fields into `rope_parameters`.
- [ ] Reject or route `model_type=minimax_m1` and `model_type=minimax_m2` configs.
- [ ] Load token embeddings, LM head, full-attention weights, lightning weights, router weights, and packed expert weights.
- [ ] Implement RMSNorm with fp32 variance.
- [ ] Implement MiniMax RoPE and full GQA attention with KV cache.
- [ ] Implement custom `MiniMaxCache` with linear cache state.
- [ ] Implement lightning attention prefill and decode recurrence.
- [ ] Implement top-2 router and packed sparse MoE inference.
- [ ] Add packed QKV and MoE grouped-GEMM rewrites with artifact-visible weight transforms.
- [ ] Add tiny checkpoint logits parity.
- [ ] Add one-token decode parity with both cache families.
- [ ] Benchmark full attention, lightning attention, MoE, LM head, and cache memory independently.
