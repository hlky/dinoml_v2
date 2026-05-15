# OLMoE Transformers Audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: allenai/OLMoE-1B-7B-0924, plus official instruct/SFT/DPO configs
Config source: local configuration_olmoe.py plus HF config.json snapshots
Source files inspected:
  transformers/src/transformers/models/olmoe/configuration_olmoe.py
  transformers/src/transformers/models/olmoe/modeling_olmoe.py
  transformers/src/transformers/models/olmoe/modular_olmoe.py
  transformers/src/transformers/models/olmoe/convert_olmoe_weights_to_hf.py
Any missing files or assumptions: no gated source gaps found; no tokenizer internals audited because the runtime target is causal-LM tensor execution, not tokenizer parity.
```

`modeling_olmoe.py` is generated from `modular_olmoe.py`; generated source was used for exact behavior, while `modular_olmoe.py` is the upstream edit source. Config snapshots and line anchors are in `_sources/`.

Primary DinoML target: inference-only CUDA causal LM, with prefill and autoregressive decode. Training loss and router auxiliary loss are not first-target runtime requirements.

## 2. High-level architecture

OLMoE is a text-only decoder MoE language model:

```text
token ids / input embeddings -> embedding lookup -> N decoder blocks -> final RMSNorm -> LM head -> logits/sampling
decoder block: RMSNorm -> causal self-attention with RoPE/cache -> residual -> RMSNorm -> sparse MoE MLP -> residual
```

Stage decomposition:

```text
CPU tokenizer/generation controller -> GPU prefill -> KV-cache decode -> last-token logits -> sampling
```

Independently stageable pieces are dense attention block parity, RoPE/cache parity, router top-k parity, expert grouped-GEMM parity, and last-token-only LM head.

## 3. Important config dimensions

Source defaults come from `OlmoeConfig`; official checkpoint values come from HF `config.json`.

| Field | Source default | Official configs inspected |
|---|---:|---:|
| `hidden_size` | 2048 | 2048 |
| `num_hidden_layers` | 16 | 16 |
| `num_attention_heads` | 16 | 16 |
| `num_key_value_heads` | defaults to Q heads | 16 |
| inferred `head_dim` | 128 | 128 |
| Q projection | 2048 -> 2048 | 2048 -> 2048 |
| K/V projection | 2048 -> 2048 | 2048 -> 2048 |
| O projection | 2048 -> 2048 | 2048 -> 2048 |
| `intermediate_size` per expert | 2048 | 1024 |
| experts | 64 | 64 |
| experts per token | 8 | 8 |
| expert gate/up packed weight | `[64, 2 * intermediate, 2048]` | `[64, 2048, 2048]` |
| expert down weight | `[64, 2048, intermediate]` | `[64, 2048, 1024]` |
| `vocab_size` | 50304 | 50304 |
| context | 4096 | 4096 |
| RoPE theta | config-normalized | 10000.0 in checkpoint JSON |
| activation | SiLU | SiLU |
| attention bias | false | false |
| dtype | not fixed by class | bf16 |
| cache | true | true |

Representative checkpoint sweep:

| Model id | Topology-significant differences |
|---|---|
| `allenai/OLMoE-1B-7B-0924` | base model; bf16; 16 layers; 64 experts; top-8; `intermediate_size=1024` |
| `allenai/OLMoE-1B-7B-0924-Instruct` | same graph dimensions; instruction weights/chat behavior is tokenizer/controller level |
| `allenai/OLMoE-1B-7B-0125-SFT` | same graph dimensions; adds explicit `rms_norm_eps=1e-05` |
| `allenai/OLMoE-1B-7B-0125-DPO` | same graph dimensions; same runtime graph |

## 3a. Family variation traps

- Do not infer expert width from source default. Official checkpoints use `intermediate_size=1024`, not the config-class default `2048`.
- Do not infer `hidden_size == num_heads * head_dim` as a permanent rule. Source uses `getattr(config, "head_dim", hidden_size // num_attention_heads)` for projections and RoPE.
- `num_key_value_heads` can differ from Q heads by config, even though inspected configs use MHA (`16 == 16`). DinoML should admit MHA first and gate GQA separately.
- Attention projections are separate HF weights, but conversion code shows original fused QKV row order `[Q, K, V]`.
- Attention has q/k RMSNorm after q/k projection and before reshape/RoPE. Tensor parallel comments say Q/K/V output must be gathered because of these norms.
- `clip_qkv` is implemented but disabled in inspected configs. If non-null, clamp happens after Q/K/V projections and q/k norms, before head reshape and RoPE.
- `sliding_window` is passed to the attention backend if present, but official configs do not set it and model-level mask creation comments say no sliding.
- MoE source eager path uses `one_hot`, `where`, per-expert gather, two expert GEMMs, top-k weight scaling, and `index_add_`. Treat this as a sparse dispatch pattern, not a general-purpose scatter admission.
- `norm_topk_prob` is implemented; official configs set it false. When true, selected top-k probabilities are renormalized by their top-k sum.
- LM head is declared tie-capable via `_tied_weights_keys`, but official configs set `tie_word_embeddings=false`; do not alias weights unless config requires it.

## 4. Operator coverage checklist

Tensor/layout ops:
- token embedding lookup `[B, S] -> [B, S, 2048]`
- reshape/view `[B, S, H] -> [B*S, H]` for router and experts
- head reshape/transpose: Q/K/V `[B, S, heads, 128] -> [B, heads, S, 128]`
- concat/chunk for packed expert gate/up output split
- gather/select token rows by expert assignment
- scatter-add/index-add into flattened `[B*S, 2048]`
- optional last-token/logits slice via `logits_to_keep`

Neural primitives:
- RMSNorm with fp32 variance accumulation
- Linear/GEMM: embeddings, Q/K/V/O, router, expert gate-up, expert down, LM head
- SiLU and elementwise multiply for expert SwiGLU-like MLP
- residual add

Attention primitives:
- causal self-attention
- MHA first; GQA/MQA possible by config
- softmax in fp32 for eager fallback
- additive causal/padding mask
- RoPE applied to Q/K before KV cache update
- DynamicCache per-layer K/V update

MoE primitives:
- router GEMM `Linear(2048 -> 64)`
- softmax over experts in fp32
- `topk(k=8, dim=-1)`
- optional top-k renormalization
- expert dispatch by selected expert id
- packed gate-up expert GEMM: `Linear(2048 -> 2048)` for official configs, split into two 1024 halves
- down expert GEMM: `Linear(1024 -> 2048)`
- route-weight multiply and scatter-add to original token order

Position/custom math:
- RoPE with rotate-half split on the last dimension
- optional dynamic RoPE types via shared Transformers rope utilities, but inspected configs use default theta 10000

Generation/cache:
- per-layer KV cache shape before repeat: K/V `[B, num_key_value_heads, T, head_dim]`
- for official configs: `[B, 16, T, 128]`
- repeated K/V for attention only when `num_key_value_heads < num_attention_heads`

## 5. Layer/block breakdown

Decoder block, repeated 16 times for official checkpoints:

```text
x0: [B, S, 2048]
a = RMSNorm(x0)
q = Linear(2048 -> num_attention_heads * head_dim, bias=attention_bias)(a)
k = Linear(2048 -> num_key_value_heads * head_dim, bias=attention_bias)(a)
v = Linear(2048 -> num_key_value_heads * head_dim, bias=attention_bias)(a)
q = RMSNorm(q) over width 2048
k = RMSNorm(k) over width num_key_value_heads * head_dim
optional q/k/v clamp if clip_qkv != None
q,k,v = view/transpose to [B, heads_or_kv_heads, S, 128]
q,k = RoPE(q,k, cos/sin)
k,v = cache.update(k,v, layer_idx) if cache is active
attn = causal_attention(q,k,v, mask, scale=1/sqrt(128))
x1 = x0 + Linear(2048 -> 2048, bias=attention_bias)(attn)
m = RMSNorm(x1)
router_logits = Linear(2048 -> 64)(flatten(m))
top8_scores, top8_experts = topk(softmax(router_logits, fp32))
for each hit expert:
  gate, up = Linear(2048 -> 2048, packed expert weight).chunk(2)
  y = Linear(1024 -> 2048)(SiLU(gate) * up)
  scatter_add token contribution y * route_score
out = x1 + unflatten(moe_out)
```

The CausalLM head applies final RMSNorm, then `Linear(2048 -> 50304, bias=False)` on either all positions or a selected suffix/index set.

## 6. Attention requirements

Required first path:
- causal self-attention only
- no cross-attention
- official configs are MHA: Q heads 16, KV heads 16, head dim 128
- source can express GQA if `num_key_value_heads < num_attention_heads`; K/V cache is stored before repeat expansion
- Q/K/V width is explicit: Q width `num_attention_heads * head_dim`, K/V width `num_key_value_heads * head_dim`
- mask is built by `create_causal_mask` from attention mask, cache, and position ids
- source eager fallback computes `query @ key.T * scaling`, adds mask, softmaxes in fp32, casts to query dtype, then `attn @ value`
- source advertises FlashAttention and SDPA support through Transformers attention interfaces
- dropout is zero in inference

Cache ordering:

```text
project -> q/k RMSNorm -> optional clamp -> reshape/transpose -> RoPE(q,k) -> cache.update(k,v) -> attention
```

This means cached keys are already RoPE-applied. Values are unclamped only if `clip_qkv` is null; with `clip_qkv`, values are clamped before caching.

## 7. Position encoding and custom math

Default RoPE:

```python
dim = head_dim
inv_freq = 1.0 / (rope_theta ** (arange(0, dim, 2) / dim))
freqs = inv_freq[None, :, None] @ position_ids[:, None, :]
emb = cat(freqs, freqs, dim=-1)
cos = cos(emb) * attention_scaling
sin = sin(emb) * attention_scaling
```

Apply:

```python
def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return cat((-x2, x1), dim=-1)

def apply_olmoe_rope(q, k, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)
```

Cos/sin can be precomputed for static/bucketed positions on default RoPE. Position IDs depend on `past_seen_tokens` during decode.

## 8. Preprocessing and input packing

No model-coupled vision/audio preprocessing. Runtime graph inputs are either:
- `input_ids: [B, S]` integer tokens, or
- `inputs_embeds: [B, S, 2048]`

If `position_ids` is absent, source computes `[0..S-1] + past_seen_tokens` and unsqueezes to `[1, S]`. `attention_mask` is consumed by causal mask creation. Tokenizer/chat-template and sampling policies are generation-controller concerns outside the core graph.

## 9. Graph rewrite / lowering opportunities

### Rewrite: separate Q/K/V projections -> packed QKV GEMM

Source pattern:

```text
q = linear(x, q_proj); k = linear(x, k_proj); v = linear(x, v_proj)
```

Replacement:

```text
qkv = Linear(2048 -> q_width + k_width + v_width)
split rows as [Q, K, V]
```

Preconditions:
- all three projections share the same input tensor and dtype
- bias settings are identical and present/absent per config
- q/k RMSNorm remains after split, not fused before split
- packed weight row order must be `[Q, K, V]`
- output widths use explicit `head_dim` and KV-head count

Failure cases:
- incompatible quantization/storage layout
- per-projection tensor parallel partitioning that cannot gather before q/k norm
- future configs with different q/k/v bias policies

Parity test sketch: compare projected Q/K/V tensors before reshape for fp32/bf16 and for any `head_dim` override.

### Rewrite: MoE eager dispatch -> grouped expert GEMM plus scatter-add

Source pattern:

```text
router = softmax(linear(flat_x, W_router), fp32)
scores, experts = topk(router, k=8)
for expert in hit_experts:
  rows = where(experts == expert)
  gate, up = linear(flat_x[rows], W_gate_up[expert]).chunk(2)
  y = linear(silu(gate) * up, W_down[expert])
  out.index_add_(token_idx, y * scores)
```

Replacement:

```text
TopKRouter -> grouped token packing by expert -> batched/grouped gate_up GEMM -> SiLU*up -> grouped down GEMM -> weighted scatter-add
```

Preconditions:
- `num_experts=64`, `top_k=8` or bounded compile-time values
- deterministic top-k tie behavior matches PyTorch enough for parity envelope
- route scores are from fp32 softmax and cast back to router dtype
- if `norm_topk_prob`, divide by top-k sum before cast
- scatter-add accumulates multiple expert contributions per token
- token packing records `(token_idx, top_k_pos)` so scores align with expert outputs

Failure cases:
- admitting arbitrary boolean scatter/gather beyond this structured token routing
- non-deterministic top-k ties in exact parity tests
- unsupported expert sharding or tensor-parallel expert ownership

Parity test sketch: fixed random router logits with forced ties avoided; compare selected experts, scores, per-expert packed rows, and final scatter-add output.

### Rewrite: last-token-only logits

Source pattern: `lm_head(hidden_states[:, slice_indices, :])`.

Replacement: for decode, pass only final hidden row to LM head.

Preconditions:
- caller requests `logits_to_keep=1` or decode path only needs next-token logits
- no loss computation over all positions
- generation controller does not require full logits history

Failure cases: prompt scoring, training loss, or user-requested full logits.

### Rewrite: RoPE/cached attention fusion

Source pattern: q/k reshape, RoPE, cache update, attention backend.

Replacement: fused prefill/decode attention kernel that accepts q, k, v projections and writes/reads cache.

Preconditions:
- cached K is RoPE-applied
- cache layout exactly `[layer][K/V][B, KVH, T, D]` or manifest equivalent
- additive mask semantics match Transformers mask values
- MHA first; GQA requires repeat-free grouped attention support

Failure cases: sliding-window configs, non-default RoPE scaling not implemented, attention output requests requiring dense attention weights.

## 10. Kernel fusion candidates

Highest priority:
- RMSNorm, including q/k projection RMSNorm. It is on every block and q/k norms block naive tensor-parallel/fused projection assumptions.
- Causal attention with RoPE and KV cache. This is the prefill/decode core.
- MoE router plus expert grouped GEMMs. Top-8-of-64 routing dominates non-attention compute and cannot be lowered as a dense FFN.
- Structured scatter-add for expert outputs. This needs a bounded primitive or fused MoE provider, not a general scatter surface.

Medium priority:
- Packed QKV GEMM with `[Q,K,V]` split.
- Gate-up packed expert GEMM plus SiLU multiply.
- Last-token-only LM head for decode.
- Router softmax/top-k specialized to 64 experts and top-8.

Lower priority:
- Router auxiliary loss and router logits output recording.
- Training labels/loss.
- Sliding-window attention admission; source hook exists but official configs do not need it.

## 11. Runtime staging plan

Stage 1: parse configs and load weights. Admit official bf16 configs with `hidden_size=2048`, `layers=16`, `heads=16`, `kv_heads=16`, `experts=64`, `top_k=8`, `intermediate=1024`.

Stage 2: dense decoder block parity with MoE stubbed by eager reference or one expert path. Validate RMSNorm, Q/K RMSNorm, RoPE, attention, residuals.

Stage 3: full prefill parity with eager MoE dispatch using bounded structured gathers and scatter-add.

Stage 4: decode parity with DynamicCache-equivalent ABI; cached K must be post-RoPE.

Stage 5: optimized MoE provider: token counting/packing, grouped gate-up GEMM, activation, grouped down GEMM, weighted scatter-add.

Stage 6: optimized attention backend and packed QKV rewrite.

Stage 7: production scheduling: batch-size/sequence buckets, cache allocation policy, optional expert load balancing/permutation optimizations.

## 12. Parity and validation plan

- RMSNorm random tests: fp32 accumulation, bf16/fp16 cast back, eps `1e-5`.
- RoPE tests: default theta 10000, dynamic positions with nonzero `past_seen_tokens`, compare q/k before cache update.
- Attention tests: one-layer prefill with and without padding mask; decode one token against full recompute.
- Router tests: softmax fp32, top-8 indices/scores, optional `norm_topk_prob`.
- Expert tests: forced routing to selected experts, compare gate/up split, SiLU multiply, down projection, weighted accumulation.
- Full block tests: single layer and all 16 layers on small sequence with fixed weights.
- CausalLM tests: `logits_to_keep=1` decode logits and full prefill logits.
- Recommended tolerances: fp32 `1e-5`/`1e-4`; bf16 `2e-2` absolute for full model, tighter per-op where accumulation is fp32.

## 13. Performance probes

- Prefill tokens/sec sweep over sequence lengths 1, 128, 512, 2048, 4096.
- Decode tokens/sec sweep over batch sizes and cache lengths.
- Attention backend comparison: eager reference, SDPA/flash equivalent, DinoML fused attention.
- MoE routing breakdown: router GEMM, top-k, token packing, expert gate-up GEMM, expert down GEMM, scatter-add.
- Expert load distribution probe: tokens per expert for realistic prompts and synthetic uniform/skewed routing.
- Last-token LM head cost with full vocab 50304.
- KV cache memory: layers * 2 * B * KVH * T * 128 * dtype bytes.
- MoE workspace memory for route indices, per-expert counts, packed token buffers, and scatter output.
- Packed QKV versus separate Q/K/V projection timing.

## 14. Skip/defer list

- Training loss and router auxiliary loss.
- Gradient checkpointing.
- Returning attentions/router logits as a production fast path.
- Tensor parallel execution beyond preserving weight/layout metadata.
- Sliding-window attention unless a checkpoint requiring it is admitted.
- Non-default/dynamic RoPE variants beyond rejecting or routing to reference.
- Quantized checkpoints and GGUF loading; treat separately as a weight/provider contract.
- Beam search and advanced generation controller policies.

## 15. Final implementation checklist

- [ ] Parse `OlmoeConfig` and reject unsupported topology/config flags.
- [ ] Load embedding, per-layer attention, q/k norm, expert, router, final norm, and LM head weights.
- [ ] Preserve expert packed weights: `gate_up_proj[E, 2I, H]`, `down_proj[E, H, I]`.
- [ ] Implement RMSNorm with fp32 accumulation.
- [ ] Implement default RoPE and post-RoPE KV cache update.
- [ ] Implement causal self-attention MHA first; gate GQA separately.
- [ ] Implement router softmax/top-k with optional top-k renorm.
- [ ] Implement bounded MoE token dispatch and weighted scatter-add.
- [ ] Add grouped expert GEMM provider path.
- [ ] Add safe packed QKV rewrite with `[Q,K,V]` split order.
- [ ] Add last-token-only LM head rewrite.
- [ ] Add single-op, single-layer, prefill, and decode parity tests.
- [ ] Add prefill/decode/MoE performance probes.
