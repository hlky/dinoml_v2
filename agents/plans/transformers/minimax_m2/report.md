# MiniMax M2 Transformers Family Audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: MiniMaxAI/MiniMax-M2.7 as the primary current representative; MiniMaxAI/MiniMax-M2, M2.1, and M2.5 also inspected.
Config source: Hugging Face raw config.json snapshots fetched 2026-05-13 and stored beside this report.
Source files inspected:
- transformers/src/transformers/models/minimax_m2/modular_minimax_m2.py
- transformers/src/transformers/models/minimax_m2/configuration_minimax_m2.py
- transformers/src/transformers/models/minimax_m2/modeling_minimax_m2.py
- transformers/src/transformers/models/flex_olmo/modeling_flex_olmo.py
- transformers/src/transformers/models/glm4_moe/modeling_glm4_moe.py
- transformers/src/transformers/modeling_gguf_pytorch_utils.py
- transformers/src/transformers/integrations/ggml.py
- transformers/src/transformers/conversion_mapping.py
Any missing files or assumptions: no MiniMax M2 processor exists; tokenizer/generation configs were sampled from the Hub only for ABI notes. No weights were downloaded.
```

The generated `modeling_minimax_m2.py` and `configuration_minimax_m2.py` are the runtime source actually imported by this Transformers checkout. Their headers state they are generated from `modular_minimax_m2.py`; future upstream source edits should be made to the modular file.

Config snapshots saved in this folder:

- `MiniMaxAI__MiniMax-M2_config.json`
- `MiniMaxAI__MiniMax-M2.1_config.json`
- `MiniMaxAI__MiniMax-M2.5_config.json`
- `MiniMaxAI__MiniMax-M2.7_config.json`
- `QuantTrio__MiniMax-M2-AWQ_config.json`
- `saricles__MiniMax-M2.7-NVFP4-GB10-AC_config.json`
- `saricles__MiniMax-M2.7-REAP-172B-A10B-NVFP4-GB10_config.json`

Primary URLs:

- [MiniMaxAI/MiniMax-M2.7](https://huggingface.co/MiniMaxAI/MiniMax-M2.7)
- [MiniMaxAI/MiniMax-M2.5](https://huggingface.co/MiniMaxAI/MiniMax-M2.5)
- [MiniMaxAI/MiniMax-M2.1](https://huggingface.co/MiniMaxAI/MiniMax-M2.1)
- [MiniMaxAI/MiniMax-M2](https://huggingface.co/MiniMaxAI/MiniMax-M2)
- [QuantTrio/MiniMax-M2-AWQ](https://huggingface.co/QuantTrio/MiniMax-M2-AWQ), quantized mirror/out-of-scope for native `minimax_m2` because its config says `model_type: mixtral`
- [saricles/MiniMax-M2.7-NVFP4-GB10-AC](https://huggingface.co/saricles/MiniMax-M2.7-NVFP4-GB10-AC), open quantized derivative
- [saricles/MiniMax-M2.7-REAP-172B-A10B-NVFP4-GB10](https://huggingface.co/saricles/MiniMax-M2.7-REAP-172B-A10B-NVFP4-GB10), open pruned/quantized derivative

Official MiniMaxAI repos were accessible for `config.json`, tokenizer config, and generation config. No gated 401/403 was observed for those metadata files. HF repo metadata reports `license: other`, `custom_code`, `fp8`, text generation, and approximately 228.7B parameters for the official M2/M2.1/M2.5/M2.7 repos; these are Hub metadata facts, not source-derived graph facts.

## 2. High-level architecture

MiniMax M2 is a text-only causal language model with a decoder-only transformer body and sparse MoE MLP blocks.

```text
token ids / input embeddings
  -> token embedding
  -> 62 decoder blocks:
       RMSNorm -> GQA self-attention with Q/K RMSNorm + RoPE + KV cache -> residual
       RMSNorm -> sigmoid top-k MoE SwiGLU experts -> residual
  -> final RMSNorm
  -> LM head
  -> logits / generation controller
```

Primary DinoML runtime target: `MiniMaxM2ForCausalLM` inference, covering prefill and autoregressive decode. Training-only loss, router auxiliary loss, jitter noise, gradient checkpointing, and output-router-logits capture can be deferred.

Independent stages:

- CPU/data pipeline: GPT2-style tokenizer, chat template if supplied by caller, generation controller fields.
- GPU prefill: dense embedding, all decoder layers over full prompt, causal mask, position ids, cache write.
- GPU decode: one or more new tokens, per-layer KV cache append/read, last-token logits.
- Optional loading/provider stage: FP8, AWQ, GGUF, and NVFP4 derivatives require separate weight materialization/provider admission.

## 3. Important config dimensions

Source defaults from `MiniMaxM2Config`:

| Field | Default |
| --- | --- |
| `vocab_size` | 200064 |
| `hidden_size` | 3072 |
| `intermediate_size` | 1536 |
| `num_hidden_layers` | 62 |
| `num_attention_heads` | 48 |
| `num_key_value_heads` | 8 |
| `head_dim` | 128 |
| Query width | 48 * 128 = 6144 |
| Key/value width | 8 * 128 = 1024 each |
| Attention output projection input | 6144 |
| `max_position_embeddings` | 196608 |
| `default_theta` / common `rope_theta` | 5000000 |
| `hidden_act` | `silu` |
| `rms_norm_eps` | 1e-6 |
| `num_local_experts` | 256 |
| `num_experts_per_tok` | 8 |
| `attention_dropout` | 0.0 |
| `use_cache` | true |
| `tie_word_embeddings` | false |

Representative config sweep:

| Repo/config | Source | Layers | Hidden | Q heads / KV heads / head dim | Experts / top-k | Context | Dtype/quantization notes | Operator-significant variation |
| --- | --- | ---: | ---: | --- | --- | ---: | --- | --- |
| `MiniMaxAI/MiniMax-M2` | official config | 62 | 3072 | 48 / 8 / 128 | 256 / 8 | 196608 | FP8 quant config, block 128x128, excludes gate, routing bias, LM head | has historical fields such as `use_mtp`, `attn_type_list`, `rotary_dim`; native pinned source ignores them except standard config fields |
| `MiniMaxAI/MiniMax-M2.1` | official config | 62 | 3072 | 48 / 8 / 128 | 256 / 8 | 196608 | same FP8 config | same native operator graph |
| `MiniMaxAI/MiniMax-M2.5` | official config | 62 | 3072 | 48 / 8 / 128 | 256 / 8 | 196608 | same FP8 config | same native operator graph |
| `MiniMaxAI/MiniMax-M2.7` | official config | 62 | 3072 | 48 / 8 / 128 | 256 / 8 | 204800 | `dtype: bfloat16` plus same FP8 config | longer context than earlier official configs |
| `saricles/MiniMax-M2.7-NVFP4-GB10-AC` | open derivative config | 62 | 3072 | 48 / 8 / 128 | 256 / 8 | 196608 | `dtype: bfloat16`; repo tags mention NVFP4/compressed tensors | `partial_rotary_factor: 0.5`, BOS/EOS 1/2; no `quantization_config` in config snapshot |
| `saricles/MiniMax-M2.7-REAP-172B-A10B-NVFP4-GB10` | open derivative config | 62 | 3072 | 48 / 8 / 128 | 192 / 8 | 196608 | `dtype: bfloat16`; repo name/tags imply pruned experts and NVFP4 | fewer experts, so router/expert tensor counts differ |
| `QuantTrio/MiniMax-M2-AWQ` | quantized mirror config | 62 | 3072 | 48 / 8 / 128 | 256 / 8 | 196608 | AWQ 4-bit group 128, zero point, `torch_dtype: float16` | config says `model_type: mixtral`; route to separate compatibility audit or reject for native `minimax_m2` path |

Tokenizer/generation ABI sampled from official M2.7:

- `tokenizer_class: GPT2Tokenizer`
- tokenizer config `model_max_length: 40960000`
- tokenizer text markers: BOS string `]~!b[` and EOS string `[e~[`
- generation config: `bos_token_id: 200019`, `eos_token_id: 200020`, `do_sample: true`, `temperature: 1.0`, `top_p: 0.95`, `top_k: 40`

## 3a. Family variation traps

- `hidden_size != num_attention_heads * head_dim`: hidden size is 3072 but Q/O attention width is 6144. Do not infer projection widths from hidden size.
- GQA is required: 48 query heads and 8 KV heads, with `num_key_value_groups = 6`.
- Q and K are RMS-normalized after projection and before reshape/RoPE.
- RoPE is partial for many Hub configs: configs expose `rotary_dim: 64`; derivative configs expose `partial_rotary_factor: 0.5`. The pinned source computes rotary dim from `config.rope_parameters`, so config normalization must preserve equivalent `partial_rotary_factor`.
- The native pinned source does not implement sliding-window attention despite some configs carrying `sliding_window: null`; source comment says no sliding window.
- Official configs carry historical/remote-code fields such as `use_mtp`, `num_mtp_modules`, `attn_type_list`, `use_qk_norm`, `qk_norm_type`, `scoring_func`, `rotary_dim`, and layernorm beta fields. The inspected in-library source does not read most of these flags; do not require MTP or alternate attention-type operators for this native-source scope.
- MoE routing uses sigmoid scores plus `e_score_correction_bias`, then unsorted top-k and renormalization of gathered sigmoid weights. This is not standard softmax MoE routing in the forward path.
- Experts store `gate_up_proj` as one packed tensor `[num_experts, 2 * intermediate_size, hidden_size]`, split as gate then up.
- Official configs advertise FP8 quantization; the native source only defines neural modules. DinoML should treat FP8/AWQ/NVFP4/GGUF as loading/provider contracts, not ordinary dtype annotations.
- `tie_word_embeddings` defaults and sampled official configs are false. `MiniMaxM2ForCausalLM` still declares `_tied_weights_keys`; DinoML should honor actual config/weight aliasing rather than assuming tying.
- `QuantTrio/MiniMax-M2-AWQ` has `architectures: MiniMaxM2ForCausalLM` but `model_type: mixtral`; this is a route/reject trap for automatic config dispatch.
- No NCHW/NHWC issue exists for the core model. Layout-sensitive ops are sequence/head reshapes and transposes, not image layouts.

## 4. Operator coverage checklist

Tensor/layout ops:

- Token embedding lookup `[B, S] -> [B, S, 3072]`.
- Reshape/view flatten `[B, S, H] -> [B*S, H]` for routing and experts.
- View Q/K/V into `[B, S, heads, 128]`, transpose to `[B, heads, S, 128]`.
- Transpose attention output `[B, heads, S, 128] -> [B, S, heads, 128]`, contiguous/reshape to `[B, S, 6144]`.
- Split/chunk packed expert projection into gate/up halves along last dim.
- Gather selected token rows by expert, indexed scatter-add/index-add back to flattened token order.
- Top-k indices, gather weights, one-hot/expert masks for fallback route; production should avoid full one-hot when possible.
- Causal/padding mask construction and broadcast to attention backend.

Neural network primitives:

- Bias-free Linear `q_proj: 3072 -> 6144`.
- Bias-free Linear `k_proj: 3072 -> 1024`.
- Bias-free Linear `v_proj: 3072 -> 1024`.
- Bias-free Linear `o_proj: 6144 -> 3072`.
- RMSNorm over 3072, 6144, and 1024 widths, fp32 variance.
- MoE router Linear `3072 -> num_local_experts` with no bias.
- Per-expert packed gate/up Linear `3072 -> 2 * 1536`, no bias.
- SiLU gate activation, elementwise multiply with up projection.
- Per-expert down Linear `1536 -> 3072`, no bias.
- Weighted expert output multiply and accumulation.
- Final LM Linear `3072 -> 200064`, no bias.
- Residual adds.

Attention primitives:

- Causal self-attention only.
- GQA/MQA-compatible grouped-query attention, 48 query heads, 8 KV heads, head dim 128.
- KV repeat only for eager fallback; optimized attention should consume unexpanded KV heads.
- SDPA/FlashAttention/FlexAttention backends are source-supported through `ALL_ATTENTION_FUNCTIONS`; exact backend selected by `_attn_implementation`.
- Softmax is fp32 in eager attention then cast to query dtype.

Position/rotary ops:

- RoPE inverse frequency from theta 5,000,000 and partial rotary dim.
- Cos/sin computed in fp32 and cast to hidden dtype.
- Apply RoPE to the leading rotary slice only, concatenate pass-through tail.

Generation/cache ops:

- DynamicCache creation when `use_cache` and no cache is passed.
- Cache update per layer after RoPE. Cached keys are post-RoPE and post-Q/K norm.
- Position ids are `arange(current_seq_len) + past_seen_tokens`, shape `[1, S]` unless caller supplies custom ids.
- `logits_to_keep` can be int or tensor; first integration can support int `0` and `1` only with guards.

Quantized/packed weight metadata ops:

- FP8 `quantization_config`: dynamic activation scheme, E4M3FN, 128x128 weight blocks, excludes `gate`, `e_score_correction_bias`, and `lm_head`.
- AWQ mirror: 4-bit GEMM, group size 128, zero point true, but not native `minimax_m2` model_type.
- GGUF mapping exists: Minimax M2 GGUF maps gate/up expert tensors into packed `gate_up_proj`, maps `exp_probs_b.bias` to `e_score_correction_bias`, and converts integer expert gating metadata to `scoring_func`.

Distributed/tensor-parallel ops:

- Config includes TP plans: Q/K/V colwise gather output, O rowwise split input, expert `gate_up_proj` packed colwise, expert `down_proj` rowwise, LM head colwise gather. Multi-GPU is deferrable for single-GPU DinoML but the weight layout is relevant.

## 5. Layer/block breakdown

Decoder block, repeated `num_hidden_layers = 62` times:

```text
residual = x                                      # [B, S, 3072]
x = RMSNorm_3072(x)

q = Linear_no_bias(x, 3072 -> 6144)
k = Linear_no_bias(x, 3072 -> 1024)
v = Linear_no_bias(x, 3072 -> 1024)
q = RMSNorm_6144(q)
k = RMSNorm_1024(k)
q = view(q, [B, S, 48, 128]).transpose(1, 2)
k = view(k, [B, S, 8, 128]).transpose(1, 2)
v = view(v, [B, S, 8, 128]).transpose(1, 2)
q, k = RoPE(q, k, cos, sin)
k, v = cache.update(k, v, layer_idx)             # if cache is enabled
attn = CausalGQA(q, k, v, mask, scale=1/sqrt(128))
attn = reshape(attn, [B, S, 6144])
attn = Linear_no_bias(attn, 6144 -> 3072)
x = residual + attn

residual = x
x = RMSNorm_3072(x)
router_logits = Linear_no_bias(flatten(x), 3072 -> E)
routing_weights = sigmoid(router_logits.float())
scores_for_choice = routing_weights + e_score_correction_bias
top_idx = topk(scores_for_choice, k=8, sorted=False)
top_w = gather(routing_weights, top_idx)
top_w = top_w / sum(top_w, axis=-1)
for selected expert e:
    gate, up = Linear_no_bias(tokens_for_e, W_gate_up[e]).chunk(2)
    expert = Linear_no_bias(silu(gate) * up, W_down[e])
    scatter_add(token, expert * top_w)
x = residual + moe_out
```

Final head:

```text
x = RMSNorm_3072(x)
logits = Linear_no_bias(x[:, slice_indices, :], 3072 -> vocab_size)
```

Projection biases: all attention projections, expert projections, router, and LM head are bias-free in the inspected source. Routing bias is a separate per-expert buffer/weight-like tensor `e_score_correction_bias`.

## 6. Attention requirements

Required attention is causal decoder self-attention with GQA.

| Property | Value |
| --- | --- |
| Causal/noncausal | causal |
| Self/cross | self-attention only |
| Heads | 48 query heads |
| KV heads | 8 |
| Head dim | 128 |
| Q width | 6144 |
| K/V width | 1024 each |
| Scale | `head_dim ** -0.5` |
| Masking | `create_causal_mask` combines causal and provided attention mask |
| Sliding window | not implemented in native source |
| RoPE placement | after Q/K norm and reshape, before cache update |
| Cache storage | post-RoPE K, raw V after projection/reshape |
| Eager fallback | repeats KV heads to 48 heads before matmul |
| Optimized backend | `ALL_ATTENTION_FUNCTIONS` dispatch supports SDPA/Flash/Flex if configured |

Cache ABI:

- Per layer key shape after update: `[B, 8, total_kv_len, 128]`.
- Per layer value shape after update: `[B, 8, total_kv_len, 128]`.
- A GQA attention kernel should consume 8 KV heads directly and map query head `h` to KV head `h // 6`.
- Cached keys are already position-encoded; do not reapply RoPE to cached keys.
- Position ids must account for `past_seen_tokens`.

Eager math order:

```text
K_rep = repeat_kv(K, 6)
V_rep = repeat_kv(V, 6)
scores = matmul(Q, K_rep.transpose(-2, -1)) * (1 / sqrt(128))
scores += attention_mask
prob = softmax(scores, dim=-1, dtype=float32).to(Q.dtype)
out = matmul(prob, V_rep)
```

## 7. Position encoding and custom math

RoPE uses standard rotate-half on the leading rotary dimension. For default source settings:

```python
def minimax_m2_inv_freq(head_dim=128, partial_rotary_factor=0.5, theta=5_000_000.0):
    rotary_dim = int(head_dim * partial_rotary_factor)  # 64 for sampled configs
    return 1.0 / (theta ** (arange(0, rotary_dim, 2).float() / rotary_dim))

def minimax_m2_rope(q, k, cos, sin):
    cos = cos.unsqueeze(1)  # [B, 1, S, rotary_dim]
    sin = sin.unsqueeze(1)
    rotary_dim = cos.shape[-1]
    q_rot, q_tail = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_tail = k[..., :rotary_dim], k[..., rotary_dim:]
    q2 = q_rot * cos + rotate_half(q_rot) * sin
    k2 = k_rot * cos + rotate_half(k_rot) * sin
    return cat([q2, q_tail], -1), cat([k2, k_tail], -1)
```

Cos/sin depend on runtime `position_ids` and can be precomputed by maximum context bucket for default RoPE. The source also decorates rotary embedding with `dynamic_rope_update`, so non-default `rope_type` values from normalized `rope_parameters` should be admitted only after a separate RoPE audit.

Custom MoE routing math:

```python
router_logits = linear(hidden.reshape(-1, 3072).to(router_weight.dtype), router_weight)
routing_weights = sigmoid(router_logits.float())
scores_for_choice = routing_weights + e_score_correction_bias
top_idx = topk(scores_for_choice, k=8, sorted=False).indices
top_w = gather(routing_weights, dim=1, index=top_idx)
top_w = top_w / top_w.sum(dim=-1, keepdim=True)
```

Training-only `load_balancing_loss_func` uses softmax over router logits, not the sigmoid routing path. It can be ignored for inference unless router logits are requested.

## 8. Preprocessing and input packing

The neural graph consumes either:

- `input_ids: LongTensor[B, S]`, or
- `inputs_embeds: FloatTensor[B, S, 3072]`.

Exactly one must be provided. Attention mask, if provided, is a text mask consumed by `create_causal_mask`; no image/audio/video processor is involved.

Tokenizer ABI from sampled official config:

- GPT2Tokenizer family.
- Very large tokenizer `model_max_length` is present in tokenizer config and should not be conflated with model `max_position_embeddings`.
- Generation config supplies BOS/EOS ids even when the model config has null BOS/EOS for older M2.
- Chat template is empty in the sampled official tokenizer config; prompt formatting is caller/controller work.

GPU graph inputs for first integration:

- `input_ids`, optional `attention_mask`, optional `position_ids`, optional cache handles.
- Prefer generating default `position_ids` and causal masks inside DinoML graph/runtime for parity with source.
- No scatter placeholder, multimodal packing, cu-seqlens, or layout metadata is required.

## 9. Graph rewrite / lowering opportunities

### Rewrite: attention projections as explicit wide GEMMs

Source pattern:

```text
q = Linear(3072 -> 6144)
k = Linear(3072 -> 1024)
v = Linear(3072 -> 1024)
```

Replacement:

```text
Either three GEMMs, or a packed QKV provider with split widths [6144, 1024, 1024].
```

Preconditions:

- All projection biases absent.
- Weight packing preserves source split order Q, K, V.
- Q/K RMSNorm must happen after projection and before reshape/RoPE.
- Packed provider must allow unequal output widths.

Failure cases:

- Any checkpoint with projection bias or changed q/k/v widths.
- Quantized providers where Q/K/V have incompatible block metadata.

Parity test sketch: compare Q/K/V tensors after Q/K norm and before RoPE for random `[B, S, 3072]`.

### Rewrite: GQA attention without KV repeat

Source pattern:

```text
repeat_kv(K,V, 6) -> dense causal attention
```

Replacement:

```text
GQA causal attention kernel using KV head index q_head // 6.
```

Preconditions:

- `num_attention_heads % num_key_value_heads == 0`.
- K/V cache stored as `[B, 8, L, 128]`.
- Attention backend preserves fp32 softmax accumulation and mask addition order within tolerance.

Failure cases:

- Requested attention weights output requiring dense `[B, 48, Q, K]` reconstruction.
- Unsupported custom masks.

### Rewrite: expert loop to grouped MoE GEMMs

Source pattern:

```text
topk routing -> per-expert token gather -> gate/up GEMM -> SiLU multiply -> down GEMM -> weighted scatter_add
```

Replacement:

```text
TopKRouter -> token permutation by expert -> grouped GEMM gate_up -> fused SwiGLU -> grouped GEMM down -> inverse permutation / weighted reduce.
```

Preconditions:

- `hidden_act == "silu"`.
- `gate_up_proj` layout `[E, 2*I, H]` with gate first, up second.
- Top-k weights are normalized sigmoid gathers, not softmax router values.
- Stable enough top-k tie behavior is either matched to torch or tie cases are excluded in tests.

Failure cases:

- Quantized expert weights without supported dequant/grouped provider.
- `num_local_experts` differs from loaded weight metadata.
- Requests for router logits/aux loss if optimized path drops logits.

### Rewrite: last-token-only logits

Source pattern:

```text
slice_indices = slice(-logits_to_keep, None)
logits = lm_head(hidden_states[:, slice_indices, :])
```

Replacement:

```text
For decode and normal generation, compute only final token hidden row through LM head.
```

Preconditions:

- `logits_to_keep == 1` or generation controller only needs last-token logits.
- No training loss over full sequence.

Failure cases:

- Caller requests full logits or tensor-valued `logits_to_keep`.

### Rewrite: FP8/GGUF/AWQ materialization to dense fallback

Source pattern:

```text
quantization_config or GGUF tensor conversion metadata
```

Replacement:

```text
Load quantized storage as explicit encoded constants; either dequantize to dense before GEMM or route to quantized provider.
```

Preconditions:

- Provider manifest records original storage, block size/group size, excluded modules, and dense logical dtype.
- Dense fallback must preserve packed expert gate/up layout.

Failure cases:

- Quantized mirror config uses `model_type: mixtral` or non-native remote code.
- NVFP4/compressed tensor format lacks DinoML provider support.

## 10. Kernel fusion candidates

Highest priority:

- RMSNorm widths 3072, 6144, 1024. This appears twice per block plus Q/K post-projection norms and final norm.
- GQA FlashAttention with RoPE-ready Q/K and KV cache. Avoiding KV repeat is mandatory for memory/perf.
- MoE top-k routing plus grouped expert GEMMs. Expert work dominates active FLOPs; fallback Python-style loops are not viable.
- SwiGLU expert fusion: `gate_up GEMM -> split -> silu(gate) * up` should be fused before down projection.
- Last-token-only LM head for decode; vocab is 200064.

Medium priority:

- Q/K/V projection packing with unequal widths.
- RoPE apply fused with attention input preparation.
- Top-k/router normalization kernel: sigmoid, add correction bias, unsorted top-k, gather, renormalize.
- Weighted scatter/index-add after expert down projection.
- Quantized FP8 dequant + GEMM for official configs.

Lower priority:

- Dense eager attention output weights reconstruction for debugging.
- Router auxiliary loss path.
- Tensor-parallel sharding plans.
- Dynamic/non-default RoPE variants beyond default/partial rotary factor.

## 11. Runtime staging plan

Stage 1: config and weight schema admission.

- Parse native `minimax_m2` configs.
- Normalize `rope_theta`/`partial_rotary_factor`/`rotary_dim` into DinoML RoPE metadata.
- Reject or route configs where `model_type != minimax_m2`, such as the sampled AWQ mirror.
- Load dense bf16/fp16 weights first; record quantized configs as encoded constants but allow dense fallback.

Stage 2: one-block dense parity.

- Implement embeddings, RMSNorm, Q/K/V/O GEMMs with nonstandard widths, partial RoPE, causal GQA eager/reference, and dense MoE fallback for tiny synthetic configs.
- Validate a reduced local config rather than the 228B-scale checkpoint.

Stage 3: prefill parity.

- Run full decoder stack on synthetic or small randomly initialized config.
- Add causal mask and position id parity.
- Compute full or last-token logits.

Stage 4: decode with KV cache.

- Define per-layer cache manifest `[K,V]` shapes.
- Validate cache append, position offsets, and one-token decode logits.

Stage 5: optimized providers.

- Add GQA FlashAttention provider.
- Add grouped MoE provider and router/top-k kernels.
- Add last-token LM head path.

Stage 6: quantized loading.

- First dense-dequant materialization for official FP8 metadata or supported GGUF.
- Later direct FP8/NVFP4/AWQ expert/GEMM providers if manifests can express block metadata.

Stage 7: production scheduling.

- Continuous batching with paged or block KV cache.
- Optional tensor parallel and expert parallel support.

Initially stub/defer: aux loss, training labels, router logits capture, full attention outputs, MTP fields, sampling controller beyond logits.

## 12. Parity and validation plan

Recommended tests:

- Config normalization tests for official M2, M2.1, M2.5, M2.7, and derivative REAP configs.
- Reject/routing test for `QuantTrio/MiniMax-M2-AWQ` because `model_type` is `mixtral` despite MiniMax architecture string.
- RMSNorm random tensor tests for widths 1024, 3072, 6144, fp32/bf16/fp16. Tolerances: fp32 `1e-5`, bf16/fp16 `1e-2` relative depending on accumulation.
- RoPE tests for rotary dim 64 and full dim 128, comparing post-RoPE Q/K.
- Attention single-layer tests: prefill `[B=1/2, S=1, 16, 257]`, GQA cache and no-cache, custom attention masks.
- KV cache tests: prefill then decode equals single full forward for logits at the appended token.
- Router tests: sigmoid routing plus correction bias, top-k unsorted set equality, normalized weights sum to 1.
- Expert layout tests: packed `gate_up_proj` gate/up split order and down projection.
- MoE block parity on tiny configs with `E=4`, `top_k=2`, including repeated tokens per expert and empty experts.
- LM head slicing tests for `logits_to_keep=0`, `1`, and small positive values.
- End-to-end random-weight parity for a tiny MiniMax M2 config through `MiniMaxM2ForCausalLM`.
- Quantization admission tests: official FP8 config produces encoded/dense fallback plan; unsupported NVFP4/AWQ routes to fallback/reject with clear reason.

## 13. Performance probes

- Prefill throughput by sequence length: 1K, 8K, 32K, 128K, 196K/204K if memory permits.
- Decode tokens/sec by batch size and cache length.
- GQA backend comparison: eager repeat, SDPA-style, FlashAttention GQA direct.
- KV cache memory per batch/context and cache append bandwidth.
- Router/top-k latency vs token count and expert count.
- MoE grouped GEMM utilization by active tokens per expert; sweep `num_local_experts` 192 vs 256 and top-k 8.
- Expert permutation/scatter overhead separately from expert GEMMs.
- LM head last-token GEMM throughput for vocab 200064.
- Dense bf16 vs FP8 dequantized weight throughput.
- Encoded-constant load/dequant time for FP8/GGUF/AWQ/NVFP4 candidates where supported.
- End-to-end prefill/decode split so MoE, attention, and logits costs do not get averaged together.

## 14. Skip/defer list

Safe to defer for first native inference integration:

- Training loss and labels.
- Router auxiliary load-balancing loss.
- Router logits/hidden states/attention output capture APIs.
- Gradient checkpointing.
- MTP fields (`use_mtp`, `num_mtp_modules`, `mtp_transformer_layers`) because pinned native source does not implement MTP modules.
- Tensor parallel and pipeline parallel execution.
- Beam search and sampling processors beyond exposing logits.
- Full-sequence logits in decode, except as a guarded debug mode.
- Non-default dynamic RoPE types.
- Quantized direct kernels for FP8/AWQ/NVFP4; start with dense materialization or clear rejection.
- GGUF end-to-end loading beyond metadata-aware weight conversion unless it fits DinoML's existing GGUF provider path.

Do not defer for useful causal LM parity:

- GQA causal attention.
- Partial RoPE.
- Q/K RMSNorm.
- Sigmoid top-k MoE routing with correction bias.
- KV cache.
- Packed expert gate/up layout.

## 15. Final implementation checklist

- [ ] Parse native `minimax_m2` config and normalize RoPE metadata.
- [ ] Reject or route `model_type != minimax_m2` mirrors.
- [ ] Load token embeddings, untied LM head, attention weights, RMSNorm weights, router weights, routing bias, and packed expert tensors.
- [ ] Implement RMSNorm with fp32 variance.
- [ ] Implement Q/K/V/O projection widths `3072 -> 6144/1024/1024` and `6144 -> 3072`.
- [ ] Implement partial RoPE with theta 5,000,000 and rotary dim 64 for sampled configs.
- [ ] Implement causal GQA attention without materializing repeated KV heads.
- [ ] Define KV cache ABI `[B, 8, L, 128]` per layer for K and V.
- [ ] Implement MiniMax M2 router: sigmoid, add `e_score_correction_bias`, top-k 8, gather, renormalize.
- [ ] Implement packed expert gate/up split and SiLU multiply.
- [ ] Implement grouped expert down projection and weighted scatter-add.
- [ ] Implement final RMSNorm and last-token LM head.
- [ ] Add tiny-config one-block parity tests.
- [ ] Add prefill vs cached decode parity tests.
- [ ] Add config sweep/admission tests for official and derivative snapshots.
- [ ] Add quantized-loading admission plan for FP8/GGUF/AWQ/NVFP4 with dense fallback or explicit rejection.
- [ ] Benchmark attention, MoE routing/expert GEMMs, LM head, and KV cache memory separately.
