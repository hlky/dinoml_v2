# Transformers Audit: `ernie4_5_moe`

## 1. Source Basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: baidu/ERNIE-4.5-21B-A3B-PT, baidu/ERNIE-4.5-300B-A47B-PT, related Base/Thinking variants
Config source: official Hugging Face config.json from Baidu repos, fetched 2026-05-13
Source files inspected:
  X:/H/transformers/src/transformers/models/ernie4_5_moe/configuration_ernie4_5_moe.py
  X:/H/transformers/src/transformers/models/ernie4_5_moe/modular_ernie4_5_moe.py
  X:/H/transformers/src/transformers/models/ernie4_5_moe/modeling_ernie4_5_moe.py
  X:/H/transformers/src/transformers/integrations/moe.py
Any missing files or assumptions:
  Tokenizer and weights were not downloaded.
  modeling_ernie4_5_moe.py is generated from modular_ernie4_5_moe.py; modular is authoritative for future source edits.
  HF configs were accessible; no gated config gap was observed.
  Cache/mask helper internals were identified by source call sites but not fully expanded.
```

Primary DinoML target for this report: text-only causal LM inference, with prefill, decode, logits, GQA KV cache, and MoE routing/expert execution. Training loss, auxiliary router loss, gradient checkpointing, and MTP are deferred.

## 2. High-Level Architecture

ERNIE 4.5 MoE is a decoder-only causal language model with RMSNorm pre-norm blocks, GQA self-attention, full-dimension ERNIE/GLM-style RoPE, and mostly sparse MoE FFN blocks. Some early layers may use dense SwiGLU MLPs before MoE starts.

```text
input_ids/inputs_embeds
  -> token embedding
  -> shared RoPE cos/sin for current positions
  -> N decoder blocks:
       RMSNorm -> Q/K/V -> RoPE(Q,K) -> causal GQA attention with KV cache -> O projection -> residual
       RMSNorm -> dense SwiGLU or routed MoE SwiGLU experts plus optional shared experts -> residual
  -> final RMSNorm
  -> optional last-token/logits_to_keep slice
  -> lm_head
  -> logits / generation controller
```

Stage decomposition:

```text
CPU/tokenizer pipeline -> token ids + attention mask
GPU prefill -> decoder stack over prompt -> logits + initialized KV cache
GPU decode -> one/few new tokens with cache update -> logits
controller -> sampling / stopping / chat template behavior
```

Independently stageable pieces are RMSNorm, RoPE, GQA attention/cache, dense MLP, router top-k, routed experts, shared experts, and final logits slicing. There is no vision/audio/preprocessor branch in this family.

## 3. Important Config Dimensions

Source defaults from `Ernie4_5_MoeConfig`:

| Field | Default / meaning |
|---|---|
| `vocab_size` | 103424 |
| `hidden_size` | 2560 |
| `num_hidden_layers` | 28 |
| `num_attention_heads` | 20 |
| `num_key_value_heads` | 4 |
| `head_dim` | inferred as `hidden_size // num_attention_heads`; 128 in official configs |
| `intermediate_size` | 12288 dense MLP intermediate |
| `hidden_act` | `silu` |
| `max_position_embeddings` | 131072 |
| `rope_theta` / `default_theta` | 500000.0 |
| `rms_norm_eps` | 1e-5 |
| `use_bias` | false by default; applies attention, MLP, lm_head projections |
| `use_cache` | true |
| `tie_word_embeddings` | true by default, but false for 300B configs |
| `moe_num_experts` | 64 |
| `moe_k` | 6 by default |
| `moe_num_shared_experts` | 2 by default |
| `moe_intermediate_size` | 1536 |
| `moe_layer_start_index/end/interval` | default MoE starts at layer 1, ends at final layer, interval 1 |
| `moe_norm_min` | 1e-12 top-k weight normalization clamp |
| `output_router_logits` | false |
| `router_aux_loss_coef` | 0.001, training/loss path |

Representative official config sweep:

| Model id | hidden | layers | q/kv heads | dense FFN | expert FFN | experts/top-k | shared experts | MoE layers | tied emb | dtype |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| `baidu/ERNIE-4.5-21B-A3B-PT` | 2560 | 28 | 20 / 4 | 12288 | 1536 | 64 / 6 | 2 | 1..27 | true | bf16 |
| `baidu/ERNIE-4.5-21B-A3B-Base-PT` | 2560 | 28 | 20 / 4 | 12288 | 1536 | 64 / 6 | 2 | 1..27 | true | bf16 |
| `baidu/ERNIE-4.5-21B-A3B-Thinking` | 2560 | 28 | 20 / 4 | 12288 | 1536 | 64 / 6 | 2 | default end -> 27 | true | bf16 |
| `baidu/ERNIE-4.5-300B-A47B-PT` | 8192 | 54 | 64 / 8 | 28672 | 3584 | 64 / 8 | 0 | 3..53 | false | bf16 |
| `baidu/ERNIE-4.5-300B-A47B-Base-PT` | 8192 | 54 | 64 / 8 | 28672 | 3584 | 64 / 8 | 0 | 3..53 | false | bf16 |

## 3a. Family Variation Traps

- `num_key_value_heads < num_attention_heads` in official configs; this is GQA, not full MHA. KV cache stores 4 or 8 KV heads, then attention repeats to 20 or 64 query heads.
- `head_dim` is not an explicit config field in this MoE config class; source computes `hidden_size // num_attention_heads` unless a checkpoint adds `head_dim`.
- Q/K/V are separate linear modules, not a single packed QKV weight. Any fused QKV rewrite must preserve split order `q_proj`, `k_proj`, `v_proj`.
- `use_bias` is global. Official configs set false, but DinoML should guard if a future config enables projection biases.
- RoPE is not Llama rotate-half pairing. Source uses full-dim GLM style: truncate cos/sin to half, repeat-interleave pairs, and rotate even/odd pairs.
- Official 21B and 300B differ materially: top-k 6 vs 8, shared experts 2 vs 0, MoE start layer 1 vs 3, tied lm head true vs false, and hidden size 2560 vs 8192.
- `moe_layer_end_index=-1` in source defaults is normalized to `num_hidden_layers - 1`.
- `Thinking` config includes `moe_capacity`, `moe_gate`, and `moe_use_aux_free`; the inspected native source does not read these fields.
- PT configs include `num_nextn_predict_layers=1`, but the source declares MTP unsupported and ignores unexpected `mtp` weight keys. Do not treat MTP as required for this audit.
- `_attn_implementation` can select eager/SDPA/Flash/Flex through Transformers shared attention dispatch. DinoML should define its own attention admission rather than inheriting that runtime switch blindly.
- Optimized experts can be selected with `experts_implementation`; eager, batched-mm, grouped-mm, and SonicMoE paths may differ in sorting, accumulation order, sentinel handling, and availability.

## 4. Operator Coverage Checklist

Tensor/layout ops:

- `Embedding(input_ids) -> [B, T, H]`, padding index 0.
- `view`, `reshape`, `transpose`, `contiguous`, `slice`, `chunk(2)`, `repeat_interleave`, `expand`.
- Last-token or index tensor slicing for `logits_to_keep` before `lm_head`.
- Mask creation/combination for causal and padding masks.

Neural primitives:

- RMSNorm over last dimension with fp32 variance and scale.
- Linear GEMMs:
  - 21B attention: Q `2560 -> 2560`, K/V `2560 -> 512`, O `2560 -> 2560`.
  - 300B attention: Q `8192 -> 8192`, K/V `8192 -> 1024`, O `8192 -> 8192`.
  - 21B dense SwiGLU: gate/up `2560 -> 12288`, down `12288 -> 2560`.
  - 300B dense SwiGLU: gate/up `8192 -> 28672`, down `28672 -> 8192`.
  - 21B expert gate_up packed weight per expert `[3072, 2560]`, down `[2560, 1536]`.
  - 300B expert gate_up packed weight per expert `[7168, 8192]`, down `[8192, 3584]`.
  - 21B shared experts are a dense MLP with intermediate `1536 * 2 = 3072`; 300B official configs have no shared experts.
  - `lm_head`: `H -> 103424`, tied to embedding only when `tie_word_embeddings=true`.
- SiLU/SwiGLU: `silu(gate) * up`.
- Residual adds.

Attention primitives:

- Causal self-attention only.
- GQA repeat of KV heads.
- QK matmul, scale by `head_dim**-0.5`, additive mask, fp32 softmax, dropout disabled for inference, probability/V matmul.
- KV cache update per layer.
- Optional backend-compatible FlashAttention/SDPA equivalent for prefill/decode if it preserves RoPE/cache/mask semantics.

MoE/routing primitives:

- Router linear in fp32: `[tokens, H] x [experts, H]`.
- Expert-score correction bias add via `moe_statics`.
- Softmax over experts in fp32.
- Top-k expert selection along expert dimension.
- Gather top-k probabilities from original softmax weights.
- Normalize selected routing weights by `clamp(sum, min=moe_norm_min)`.
- Active-expert compaction, token gather, per-expert or grouped GEMMs, route scaling, scatter-add or deterministic reshape/sum accumulation.

Position/rotary:

- Precompute or generate `inv_freq`, `cos`, `sin` in fp32.
- Dynamic RoPE update only if non-default `rope_parameters["rope_type"]` appears; official configs use default theta/null scaling.

Generation/cache:

- Dynamic cache allocation when `use_cache=True`.
- Per-layer K/V cache shape before repeat: `[B, kv_heads, cache_len, head_dim]`.
- Cache stores post-RoPE keys and values before KV repeat.
- Reorder/cache management for generation should follow Transformers cache ABI, but first DinoML target can use append-only batch-stable decode.

Quantized/packed weight metadata:

- No native source-level quantized weight format is implemented in this family. Official source weights are normal PyTorch parameters.
- Community GGUF/4-bit conversions should be treated as external loading/provider contracts, not native Transformers behavior.

## 5. Layer/Block Breakdown

Decoder block, repeated `num_hidden_layers`:

```text
residual = x
x_norm = RMSNorm(x)
q = Linear_q(x_norm).view(B, T, q_heads, head_dim).transpose(1, 2)
k = Linear_k(x_norm).view(B, T, kv_heads, head_dim).transpose(1, 2)
v = Linear_v(x_norm).view(B, T, kv_heads, head_dim).transpose(1, 2)
q, k = ErnieRoPE(q, k, cos, sin)
k, v = cache.update(k, v, layer_idx) if cache is present
attn = causal_gqa_attention(q, k, v, mask, scale=head_dim^-0.5)
x = residual + Linear_o(attn.transpose(1, 2).reshape(B, T, q_heads * head_dim))

residual = x
x_norm = RMSNorm(x)
if layer is dense:
    x_ffn = down(silu(gate(x_norm)) * up(x_norm))
else:
    tokens = x_norm.reshape(B*T, H)
    router_logits = tokens @ router_weight.T
    probs = softmax(router_logits, dim=experts)
    selected = topk(probs + expert_correction_bias, k=moe_k)
    weights = gather(probs, selected) / clamp(sum(gathered), min=moe_norm_min)
    x_ffn = sum_over_selected_experts(weight * expert_down(silu(expert_gate(tokens)) * expert_up(tokens)))
    x_ffn += shared_swiglu(tokens) if configured
    x_ffn = x_ffn.reshape(B, T, H)
x = residual + x_ffn
```

MoE layer admission:

```text
((layer_idx + 1) % moe_layer_interval == 0)
and layer_idx >= moe_layer_start_index
and layer_idx <= moe_layer_end_index
```

For official configs, 21B has dense layer 0 then MoE layers 1-27; 300B has dense layers 0-2 then MoE layers 3-53.

## 6. Attention Requirements

- Type: causal decoder self-attention.
- Heads: GQA.
- Official 21B: `q_heads=20`, `kv_heads=4`, `groups=5`, `head_dim=128`.
- Official 300B: `q_heads=64`, `kv_heads=8`, `groups=8`, `head_dim=128`.
- Q width equals hidden size in official configs; K/V width is `kv_heads * head_dim`.
- Masking: Transformers `create_causal_mask` combines causal and optional padding behavior; attention receives additive mask and adds it before softmax.
- Softmax: eager path explicitly computes softmax in fp32 then casts to query dtype.
- Dropout: source sets `attention_dropout=0.0`; inference uses zero dropout.
- Cache: if cache exists, `past_key_values.update` happens after RoPE and before attention backend dispatch.
- Packed/varlen: no model-specific packed sequence ABI is present. Backend-specific Flash/Flex may use internal metadata, but source-level model inputs are ordinary `input_ids`, optional `attention_mask`, optional `position_ids`, and optional cache.
- Sliding window/local/block sparse: not present in this model source.
- Cross-attention: not present.

FlashAttention/SDPA compatibility: source declares support for FlashAttention, SDPA, and FlexAttention through Transformers backend interfaces. DinoML optimized attention should preserve the exact order:

```text
linear q/k/v -> reshape/transpose -> RoPE q/k -> cache update -> causal mask + scale -> softmax -> V matmul -> transpose/reshape -> o_proj
```

## 7. Position Encoding and Custom Math

Default inverse frequency:

```python
dim = config.head_dim if present else config.hidden_size // config.num_attention_heads
inv_freq = 1.0 / (rope_theta ** (arange(0, dim, 2, fp32) / dim))
```

Forward RoPE table:

```python
freqs = (inv_freq[None, :, None] @ position_ids[:, None, :].float()).transpose(1, 2)
emb = cat((freqs, freqs), dim=-1)
cos = cos(emb) * attention_scaling
sin = sin(emb) * attention_scaling
```

ERNIE/GLM-style apply:

```python
def rotate_half(x):
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return stack((-x2, x1), dim=-1).flatten(-2)

def apply_ernie_rope(q, k, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    cos = cos[..., : cos.shape[-1] // 2].repeat_interleave(2, dim=-1)
    sin = sin[..., : sin.shape[-1] // 2].repeat_interleave(2, dim=-1)
    q = q.float() * cos + rotate_half(q).float() * sin
    k = k.float() * cos + rotate_half(k).float() * sin
    return q.to(original_dtype), k.to(original_dtype)
```

Precomputation: for default RoPE, `inv_freq` is static and cos/sin can be cached by position range and device. `position_ids` depend on cache length during decode. If future configs use non-default `rope_parameters["rope_type"]`, dynamic update and scaling must be audited separately.

## 8. Preprocessing and Input Packing

This family consumes standard text token inputs:

- `input_ids`: `[B, T]`, exactly one of `input_ids` or `inputs_embeds`.
- `inputs_embeds`: `[B, T, H]`, bypasses embedding lookup.
- `attention_mask`: optional `[B, T_total]` style mask consumed by shared causal-mask helper.
- `position_ids`: optional `[B, T]`; if absent, generated as `arange(T) + past_seen_tokens`, unsqueezed to one batch row.

No model-coupled tokenizer logic, multimodal placeholder scatter, image/audio/video preprocessing, packed `cu_seqlens`, or modality token stitching is present in the inspected model files. Generation-controller details such as sampling and chat templating sit outside the neural graph.

## 9. Graph Rewrite / Lowering Opportunities

### Rewrite: Separate Q/K/V Linears -> Fused QKV Projection

Source pattern:

```text
q = linear(x, q_weight)
k = linear(x, k_weight)
v = linear(x, v_weight)
```

Replacement:

```text
qkv = linear(x, concat_rows(q_weight, k_weight, v_weight))
split qkv as [q_heads*D, kv_heads*D, kv_heads*D]
```

Preconditions:

- Same input tensor and dtype.
- Same bias policy; concatenate biases only if all three biases are present.
- Preserve output split order Q, K, V.
- Do not assume equal Q/K/V widths because GQA makes K/V smaller.
- Weight layout must match DinoML GEMM convention. PyTorch linear weight is `[out_features, in_features]`.

Failure cases: mixed quantization/provider policies per projection, future configs with different bias handling, or graph consumers that need individual projection outputs.

Parity test sketch: compare q/k/v tensors before and after fusion for random hidden states in fp32 and bf16, then compare one full attention block.

### Rewrite: RoPE + GQA Attention -> Fused Attention Region

Source pattern:

```text
RoPE(q,k) -> cache.update(k,v) -> repeat_kv -> QK^T scale + mask -> fp32 softmax -> PV
```

Replacement: backend attention kernel that accepts post-RoPE Q/K/V or fuses RoPE inside the pre-attention path.

Preconditions:

- Causal self-attention only.
- Additive mask semantics match source.
- Cache stores post-RoPE K with unexpanded KV heads.
- fp32 softmax or validated tolerance for backend softmax.
- No attention output requested for first optimized path.

Failure cases: non-default RoPE scaling not audited, attention output tensors requested, backend mask convention mismatch, or cache reorder not implemented.

### Rewrite: Eager MoE Expert Loop -> Grouped Expert GEMM + Deterministic Sum

Source pattern:

```text
one_hot(top_k_index) -> per expert token gather -> gate_up GEMM -> swiglu -> down GEMM
-> multiply route weight -> index_add_ into token rows
```

Replacement:

```text
flatten token-expert pairs S = tokens * top_k
sort/group by expert
grouped GEMM gate_up over active expert buckets
SwiGLU
grouped GEMM down
route-weight multiply
unsort
reshape [tokens, top_k, H] and sum over top_k
```

Preconditions:

- `top_k_index` contains valid expert ids only; if DinoML supports EP sentinels later, add explicit sentinel masks.
- Expert weights use native source layout:
  - `gate_up_proj[E, 2I, H]`, concatenated as `[gate; up]`.
  - `down_proj[E, H, I]`.
- Accumulation-order differences are accepted within tolerance or the path is selected as the canonical deterministic lowering.
- Shared expert dense MLP is added separately when configured.

Failure cases: requested exact eager `index_add_` bit parity, unavailable grouped GEMM provider, or dynamic top-k unsupported.

Parity test sketch: fixed router outputs with repeated expert assignments; compare eager loop and grouped lowering for fp32 and bf16, including duplicate token routes and empty experts.

### Rewrite: Shared Experts -> Dense SwiGLU

Source pattern:

```text
shared_experts = MLP(hidden, intermediate = moe_intermediate_size * moe_num_shared_experts)
final = routed + shared_experts(tokens)
```

Replacement: ordinary dense SwiGLU GEMM epilogue chain, independent of router.

Preconditions:

- `moe_num_shared_experts > 0`.
- Same `hidden_act`.
- No route-dependent scaling.

Failure cases: 300B official configs have no shared experts; do not instantiate empty shared path.

### Rewrite: Last-Token Logits

Source pattern:

```text
slice_indices = slice(-logits_to_keep, None)
logits = lm_head(hidden_states[:, slice_indices, :])
```

Replacement: if `logits_to_keep == 1` during decode, run lm_head only for final token row.

Preconditions:

- Caller does not request full-sequence logits.
- Loss is not being computed.
- Sampling controller accepts `[B, 1, vocab]` logits.

Failure cases: prefill full-logit parity tests, teacher-forced loss, or tensor index `logits_to_keep` that selects non-suffix positions.

## 10. Kernel Fusion Candidates

Highest priority:

- RMSNorm: every block has two RMSNorms plus final norm; fp32 variance with reduced-precision storage is required.
- GQA attention with RoPE and KV cache: primary prefill/decode bottleneck; cache stores unexpanded KV heads.
- MoE routing + grouped expert GEMM: official models route 6 or 8 experts per token over 64 experts; eager scatter/add loop is not production viable.
- SwiGLU GEMM epilogues for dense and expert MLPs: gate/up activation multiply is ubiquitous.
- Last-token-only logits: large vocab `103424` makes full-sequence logits expensive.

Medium priority:

- Fused QKV projection for separate source weights.
- Router linear + softmax + top-k + gather normalization pipeline.
- Shared expert dense SwiGLU for 21B.
- Expert weight dequantization/provider path if DinoML uses GGUF or other converted checkpoints.

Lower priority:

- Auxiliary router loss and router logits capture.
- Attention weights output.
- MTP keys or speculative `num_nextn_predict_layers`; source does not implement this path.
- Tensor-parallel plan metadata; useful later, not required for single-device first parity.

## 11. Runtime Staging Plan

1. Parse config and reject unsupported native-source mismatches: non-default RoPE types, unsupported `experts_implementation`, MTP-only assumptions, or unsupported bias/quantization combinations.
2. Load weights for embeddings, attention, dense MLP, router, expert tensors, optional shared experts, final norm, and lm head. Preserve tied embedding/lm-head alias when `tie_word_embeddings=true`.
3. Build one dense early layer parity path: embedding, RMSNorm, attention without cache, dense SwiGLU, residuals.
4. Add RoPE and GQA KV cache for prefill/decode with append-only dynamic cache.
5. Add router top-k parity and eager reference MoE lowering for one MoE block.
6. Replace eager MoE with grouped/batched expert GEMM provider path and explicit route accumulation semantics.
7. Add full model prefill logits parity for 21B-shaped small synthetic configs and then official config shape smoke without full weights if needed.
8. Add decode-token parity with cache.
9. Enable optimized attention, last-token logits, and MoE fusions.
10. Add converted-weight paths such as GGUF only as a separate provider contract, not as native Transformers source behavior.

Stub initially: generation sampling, auxiliary loss, attention weight outputs, router logits outputs, tensor parallel execution, MTP/speculative layers, and non-default RoPE variants.

## 12. Parity and Validation Plan

- RMSNorm random tensor tests: fp32 reference, bf16/fp16 storage with fp32 variance, `eps=1e-5`.
- RoPE tests: compare ERNIE full-dim repeat-interleave apply for random Q/K and nonzero `position_ids`; include decode offset.
- Attention unit tests: Q/K/V shapes for 21B and 300B configs, GQA repeat factor, additive mask, fp32 softmax tolerance.
- Cache tests: prefill then decode one token; verify cache length and that K is stored after RoPE before repeat.
- Router tests: fp32 router linear/softmax, score-correction bias, top-k selection, gathered-prob normalization with clamp.
- Expert tests: fixed top-k assignments, empty experts, duplicate token routes, shared expert on/off, compare eager and grouped lowering within dtype tolerance.
- Single-layer parity: dense layer 0 for 21B/300B configs.
- MoE layer parity: layer 1 for 21B and layer 3 for 300B.
- Full prefill logits parity: random tiny config with same source structure.
- Decode parity: same prompt, one-step continuation with cache.
- Recommended tolerances: fp32 `rtol=1e-4, atol=1e-5`; bf16/fp16 block-level `rtol=2e-2, atol=2e-2`, tighten for isolated ops where possible.

No DinoML tests or imports were run for this audit.

## 13. Performance Probes

- Prefill throughput by batch and sequence length: 1k, 4k, 16k, 64k, 131k where memory permits.
- Decode tokens/sec by batch size with populated KV cache.
- GQA attention backend comparison: eager matmul, SDPA-like, FlashAttention-like, DinoML fused.
- KV cache memory usage: 21B and 300B shapes, varying batch/context.
- Router throughput: `[B*T, H] -> [B*T, 64]` plus softmax/top-k.
- MoE dispatch overhead: sort/histogram/offsets versus eager active-expert loop.
- Grouped expert GEMM occupancy: expert load imbalance sweeps, top-k 6 vs 8, empty expert rates.
- Shared expert cost for 21B.
- Last-token-only logits versus full-sequence logits.
- Converted/quantized checkpoint probe: dense bf16 load versus GGUF dequant-before-GEMM provider path, if DinoML chooses to support community conversions.

## 14. Skip/Defer List

- Training loss and `labels`.
- Auxiliary router loss and `output_router_logits` as a production output.
- Gradient checkpointing.
- Attention weights output.
- Beam-search cache reorder beyond a simple first decode target.
- MTP/speculative layers advertised by `num_nextn_predict_layers`; native source ignores MTP.
- SonicMoE and external optimized expert implementations until DinoML owns a provider contract.
- Tensor parallel and pipeline parallel plans.
- Non-default/dynamic RoPE variants unless an official config requiring them is targeted.
- Quantized/community GGUF checkpoints for first native-source parity.

## 15. Final Implementation Checklist

- [ ] Parse `Ernie4_5_MoeConfig` and normalize `moe_layer_end_index`.
- [ ] Reject or separately route ignored/historical fields such as MTP-only assumptions and `moe_gate` variants.
- [ ] Load embeddings, norms, attention weights, dense MLP weights, router weights, expert tensors, shared experts, and lm head.
- [ ] Preserve tied `lm_head.weight` / `embed_tokens.weight` alias when configured.
- [ ] Implement RMSNorm fp32 variance.
- [ ] Implement ERNIE full-dim RoPE.
- [ ] Implement GQA causal attention with unexpanded KV cache.
- [ ] Implement dense SwiGLU MLP.
- [ ] Implement router fp32 softmax, score-correction bias, top-k, gather, and normalization clamp.
- [ ] Implement eager-reference MoE expert path for parity.
- [ ] Implement grouped expert GEMM lowering with deterministic route accumulation.
- [ ] Add optional shared expert dense SwiGLU path.
- [ ] Add last-token/logits-to-keep lm-head lowering.
- [ ] Add one-layer dense parity tests.
- [ ] Add one-layer MoE parity tests.
- [ ] Add prefill logits parity.
- [ ] Add decode/cache parity.
- [ ] Benchmark attention, router, expert GEMM, logits, and cache memory separately.

