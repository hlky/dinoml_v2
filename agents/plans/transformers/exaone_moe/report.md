# Transformers family audit: `exaone_moe`

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: LGAI-EXAONE/K-EXAONE-236B-A23B
Config source: HF config.json, main branch plus one historical revision and one quantized mirror
Source files inspected: configuration_exaone_moe.py, modeling_exaone_moe.py, modular_exaone_moe.py, __init__.py
Any missing files or assumptions: no model weights loaded; no imports/tests; only one official model shape found
```

Primary local source is under
`X:/H/transformers/src/transformers/models/exaone_moe`. The generated
`modeling_exaone_moe.py` and `configuration_exaone_moe.py` are the runtime basis;
they are generated from `modular_exaone_moe.py`, which is authoritative for
future upstream source edits. Config/source notes are in
`agents/plans/transformers/exaone_moe/_sources/source_notes.md`.

## 2. High-level architecture

EXAONE MoE is a text-only decoder-only causal LM with hybrid sliding/full
self-attention and sparse MoE feed-forward layers:

```text
token ids -> token embedding -> repeated decoder blocks -> final RMSNorm -> lm_head -> logits/sampling
```

Each decoder block is pre-norm attention plus pre-norm MLP/MoE. The official
checkpoint uses one dense SwiGLU layer followed by sparse MoE layers. First
DinoML target should be `ExaoneMoeForCausalLM` prefill and decode logits.
Generation sampling, chat templates, and tool-call formatting are controller or
tokenizer work, not neural graph ops.

## 3. Important config dimensions

| Field | Source default | `K-EXAONE-236B-A23B` main |
|---|---:|---:|
| `vocab_size` | 102400 | 153600 |
| `hidden_size` | 4096 | 6144 |
| `num_hidden_layers` | 32 | 48 |
| `num_attention_heads` | 32 | 64 |
| `num_key_value_heads` | 32 | 8 |
| `head_dim` | inferred | 128 |
| attention q width | 4096 | 8192 |
| attention kv width | 4096 | 1024 |
| `intermediate_size` dense MLP | 16384 | 18432 |
| `moe_intermediate_size` | 1024 | 2048 |
| `num_experts` | 64 | 128 |
| `num_experts_per_tok` | 8 | 8 |
| `num_shared_experts` | 1 | 1 |
| `max_position_embeddings` | 2048 | 262144 |
| RoPE | `rope_parameters=None` default must be filled | `default`, theta 1000000 |
| `sliding_window` | 4096 | 128 |
| `sliding_window_pattern` | 4 | `LLLG` |
| dtype metadata | not fixed by source | `bfloat16` |

Representative config sweep:

| Config | Status | Operator-significant notes |
|---|---|---|
| `LGAI-EXAONE/K-EXAONE-236B-A23B` main | official | 48L, 6144 hidden, GQA 64/8, top-8 of 128 experts, hybrid LLLG attention |
| same repo historical `2159cb4...` | official historical | same graph shape; uses old `first_last_k_dense_replace` field not read by current config class |
| `mlx-community/K-EXAONE-236B-A23B-6bit` | mirror | same graph; adds 6-bit affine quantization metadata not read by Transformers source |
| `inferencerlabs/K-EXAONE-236B-A23B-MLX-6.5bit` | mirror/search result | same family; MLX quantized loading should be separately gated |

## 3a. Family variation traps

- `hidden_size != num_attention_heads * head_dim` for the official config:
  hidden is 6144, q/o attention width is 8192.
- GQA is required: 64 query heads, 8 KV heads, repeat factor 8.
- Full-attention layers in the hybrid pattern use NoPE; sliding layers use RoPE.
- Cache stores keys after the layer-specific RoPE/NoPE decision.
- Source defaults may synthesize `layer_types` from integer pattern 4, but the
  official config ships explicit `LLLG` layer types.
- Current source reads `first_k_dense_replace` or `mlp_layer_types`; historical
  `first_last_k_dense_replace`, `is_moe_layer`, `scoring_func`, and
  `topk_method` are not used by the inspected modeling source.
- `num_nextn_predict_layers` appears in config and `mtp.*` weights are ignored;
  no MTP head is implemented in this source basis.
- MLX quantization metadata is not native source behavior and must not be
  treated as a graph requirement.

## 4. Operator coverage checklist

Tensor/layout ops:
- token embedding gather `[B,S] -> [B,S,H]`
- reshape/view and transpose for `[B,S,heads,head_dim] <-> [B,heads,S,D]`
- contiguous materialization after attention transpose
- gather/scatter/indexing for MoE dispatch: top-k gather, boolean/group masks,
  token row gather, scatter-add/index-add

Neural primitives:
- bias-free Linear/GEMM, including rectangular attention q/o widths
- RMSNorm fp32 reduction over last dim
- SiLU and SwiGLU: `down(silu(gate(x)) * up(x))`
- residual adds and final LM head

Attention primitives:
- causal self-attention
- GQA KV repeat or native grouped attention
- sliding-window causal attention for selected layers
- fp32 softmax in eager fallback
- KV cache update and per-layer cache lookup

Position/cache ops:
- default RoPE cos/sin generation in fp32
- apply RoPE to Q/K for sliding layers only
- NoPE full attention layers
- position id generation from cache length

MoE ops:
- router fp32 linear `[tokens,H] x [E,H] -> [tokens,E]`
- sigmoid scoring
- group top-2 scoring inside each group, group top-k, scatter group mask
- masked top-k experts, score gather, optional route-weight normalization
- packed expert GEMMs and weighted scatter-add
- shared dense expert added to routed output

Quantized/packed weight metadata:
- no native Transformers quantized path in source; MLX mirror metadata should
  route to a loading/provider policy such as dense materialize or GGUF/MLX
  provider, with source-parity fallback to dense bf16/fp32 weights.

## 5. Layer/block breakdown

Decoder block, repeated `num_hidden_layers`:

```text
residual = x
x = RMSNorm(x)
q = Linear(H -> num_attention_heads * head_dim, no bias)
k = Linear(H -> num_key_value_heads * head_dim, no bias)
v = Linear(H -> num_key_value_heads * head_dim, no bias)
q,k,v = reshape/transpose to [B, heads, S, D]
q = RMSNorm(q over D); k = RMSNorm(k over D)
if sliding_window is None or layer_type == sliding_attention: q,k = RoPE(q,k)
k,v = cache.update(k,v, layer_idx) if cache is enabled
x_attn = causal/sliding attention(q,k,v, mask, scale=D^-0.5)
x = residual + Linear(num_attention_heads * head_dim -> H, no bias)(x_attn)
residual = x
x = RMSNorm(x)
x = dense SwiGLU MLP or sparse MoE block
x = residual + x
```

Official checkpoint shapes: `H=6144`, `D=128`, q width `8192`, kv width `1024`,
dense MLP width `18432`, expert intermediate `2048`.

Sparse MoE block:

```text
router_logits = fp32 Linear(H -> num_experts)
topk_indices, topk_weights = route(router_logits)
routed = experts(flatten(x), topk_indices, topk_weights)
shared = dense_swiglu(x, intermediate=moe_intermediate_size * num_shared_experts)
return routed.view_as(x) + shared
```

## 6. Attention requirements

Attention is causal decoder self-attention. The official config requires GQA,
with 64 query heads, 8 KV heads, head dim 128, and KV repeat factor 8 if using a
dense MHA fallback. Prefer native GQA FlashAttention/SDPA-style kernels that do
not materialize repeated KV.

There are two masks per forward when both layer types exist: full causal and
sliding-window causal. The official `LLLG` pattern means layers 0,1,2 are local,
layer 3 is full, repeated. Sliding attention passes `sliding_window=128` to the
attention backend. Full layers pass no sliding window and also skip RoPE.

KV cache shape per layer before any repeat:

```text
key:   [B, num_key_value_heads, cache_seq, head_dim]
value: [B, num_key_value_heads, cache_seq, head_dim]
```

Cached keys are post-RoPE for sliding layers and NoPE for full layers. Decode
must maintain the layer-type-specific position/caching contract. Eager fallback
orders math as repeat KV, `q @ k^T * scale`, add mask, softmax in fp32, cast
back to query dtype, dropout, `weights @ v`.

## 7. Position encoding and custom math

Default RoPE parameters:

```python
inv_freq = 1.0 / (rope_theta ** (arange(0, head_dim, 2).float() / head_dim))
freqs = (inv_freq[None, :, None] @ position_ids[:, None, :].float()).transpose(1, 2)
emb = cat([freqs, freqs], dim=-1)
cos, sin = cos(emb), sin(emb)
```

Apply function:

```python
def exaone_moe_rope(q, k, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    def rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return cat([-x2, x1], dim=-1)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin
```

Cos/sin depend on runtime `position_ids`, cache length, dtype, and device.
For static windows they can be precomputed by position bucket, but long-context
decode needs a cache-aware position path.

## 8. Preprocessing and input packing

Model graph inputs are `input_ids` or `inputs_embeds`, optional
`attention_mask`, optional `position_ids`, optional `past_key_values`, and
`use_cache`. Tokenizer config says right padding and model inputs
`input_ids`, `attention_mask`. Chat template and special tokens are tokenizer
or generation-controller ABI. The modeling source is text-only; vision-looking
tokens in tokenizer metadata do not imply multimodal tensors or masked scatter.

Default `position_ids` are generated as `arange(S) + past_seen_tokens` and
unsqueezed to `[1,S]`. DinoML should accept caller-supplied position ids for
parity and use generated ids only under the same defaulting rule.

## 9. Graph rewrite / lowering opportunities

### Rewrite: fused QKV projection

Source pattern: three bias-free linears from the same normalized hidden states.

Replacement: one GEMM `H -> q_width + k_width + v_width`, then split as
`[q, k, v]`.

Preconditions: identical input tensor, no projection bias, source weight order
preserved, split widths exactly `[num_attention_heads*D, num_key_value_heads*D,
num_key_value_heads*D]`.

Failure cases: `hidden_size != q_width` must not be assumed away; no packed
checkpoint source was observed, so weight packing is a DinoML artifact transform.

Parity test sketch: compare q/k/v tensors before q/k norms for random inputs.

### Rewrite: native GQA attention

Source pattern: repeat KV heads then dense attention.

Replacement: grouped attention kernel consuming Q heads and KV heads directly.

Preconditions: repeat factor `num_attention_heads // num_key_value_heads` is an
integer; attention backend preserves mask, scale, fp32 softmax tolerance, and
layer sliding-window setting.

Failure cases: attention output requests may require dense attention weights;
first integration can defer returned attentions.

### Rewrite: MoE packed expert batch GEMM

Source pattern: per-hit expert loop with token gather, packed gate/up linear,
activation multiply, down linear, route-weight multiply, `index_add_`.

Replacement: dispatch tokens by expert, run grouped GEMM for gate/up and down,
then weighted scatter-add.

Preconditions: stable token-to-expert pairs from top-k, expert weights in source
layout `[E, 2I, H]` and `[E, H, I]`, route weights already normalized/scaled.

Failure cases: top-k tie ordering can affect expert choice; preserve or bound
tie behavior in tests. Do not replace with dense all-expert GEMM except as a
debug fallback.

### Rewrite: shared expert as regular SwiGLU GEMM

Source pattern: shared expert is a dense `ExaoneMoeMLP` with intermediate
`moe_intermediate_size * num_shared_experts`.

Replacement: standard fused SwiGLU epilogue plus down GEMM.

Preconditions: no bias and `hidden_act == "silu"`.

## 10. Kernel fusion candidates

Highest priority:
- RMSNorm, including head-dim Q/K RMSNorm.
- GQA FlashAttention with sliding-window and KV cache.
- MoE routing plus dispatch: sigmoid/top-k/group mask and expert token packing.
- Grouped expert GEMMs plus weighted scatter-add.
- SwiGLU dense/shared expert fusion.

Medium priority:
- QKV fused projection plus Q/K norm plus RoPE preparation.
- RoPE apply fused with attention input layout.
- Last-token-only `lm_head` using `logits_to_keep`.
- Route-weight normalization/scaling fused into dispatch metadata.

Lower priority:
- Returned attention weights.
- Training-only dropout/loss.
- Hub kernel replacement paths, after source-parity kernels exist.

## 11. Runtime staging plan

Stage 1: parse config and load dense bf16/fp32 weights, rejecting unsupported
historical fields only when they change behavior.

Stage 2: one dense layer parity with embedding, RMSNorm, attention without cache,
dense SwiGLU, final norm/head.

Stage 3: sparse MoE block parity with eager dispatch fallback and small token
counts.

Stage 4: full prefill parity with hybrid sliding/full masks and layer-specific
RoPE/NoPE.

Stage 5: decode parity with per-layer KV cache and last-token logits.

Stage 6: optimized kernels: RMSNorm, GQA attention, grouped expert GEMM,
scatter-add, and fused SwiGLU.

Stage 7: optional quantized loading/provider work for MLX/GGUF mirrors, with
dense materialization fallback and explicit provenance.

## 12. Parity and validation plan

- RMSNorm fp32 accumulation tests over `[B,S,H]` and `[B,heads,S,D]`.
- RoPE tests with explicit `position_ids`, including cached offset positions.
- Attention tests for sliding and full layers separately, proving full layers
  skip RoPE.
- Router tests for sigmoid, correction bias choice-only behavior, group top-k,
  expert top-k, normalization, and scaling.
- Expert tests with synthetic token/expert assignments covering duplicate token
  destinations and scatter-add accumulation.
- One-block dense and one-block sparse parity against source tensors.
- Full prefill logits for short prompts and longer sliding-window prompts.
- Decode token parity for cache length growth and mixed layer types.

Recommended tolerances: fp32 `1e-4` absolute/relative for block internals;
bf16/fp16 `2e-2` for logits initially, tightened after fused kernels mature.

## 13. Performance probes

- Prefill throughput by sequence length: 128, 512, 2048, 8192+.
- Decode tokens/sec by batch size and cache length.
- Sliding-window vs full-attention layer time split.
- KV cache memory by batch/cache length: raw KV heads, not repeated heads.
- Router/top-k latency and token distribution entropy.
- Expert grouped GEMM occupancy by active expert count and tokens per expert.
- Scatter-add bandwidth and contention for top-8 routing.
- Dense bf16 weights vs quantized load/dequant provider if quantized mirrors are
  admitted later.
- Last-token-only logits vs full-sequence logits.

## 14. Skip/defer list

- Training, loss, dropout, and gradient checkpointing.
- Returned attentions and hidden-state/router-logit capture unless debugging.
- MTP/speculative heads: config/weights may mention them, but source ignores
  `mtp.*` and implements no head.
- MLX 6-bit/6.5-bit quantized execution as native graph behavior.
- Tensor parallel and pipeline parallel plans.
- Chat template/tool-call sampling logic beyond token ids and attention mask.

## 15. Final implementation checklist

- [ ] Parse `ExaoneMoeConfig`, including explicit `layer_types` and `mlp_layer_types`.
- [ ] Reject or document ignored historical fields: `is_moe_layer`, `scoring_func`, `topk_method`, `num_nextn_predict_layers`.
- [ ] Load dense weights with source layouts and untied embedding/lm head.
- [ ] Implement RMSNorm with fp32 variance.
- [ ] Implement attention projections where q width may differ from hidden size.
- [ ] Implement Q/K head RMSNorm.
- [ ] Implement layer-specific RoPE/NoPE and cache update ordering.
- [ ] Implement full and sliding causal masks.
- [ ] Implement GQA prefill/decode with KV cache.
- [ ] Implement dense SwiGLU MLP.
- [ ] Implement MoE router, group top-k, expert top-k, normalization, scaling.
- [ ] Implement packed expert GEMM dispatch and weighted scatter-add.
- [ ] Add one-block dense and sparse parity tests.
- [ ] Add full prefill logits and decode-cache parity tests.
- [ ] Add performance probes for attention, routing, expert GEMM, scatter-add, and logits.
