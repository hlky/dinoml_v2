# OLMo Hybrid Transformers Audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: olmo_hybrid
Config source: local OlmoHybridConfig plus public HF configs for allenai/Olmo-Hybrid-7B and chat variants
Source files inspected: configuration_olmo_hybrid.py, modeling_olmo_hybrid.py, modular_olmo_hybrid.py, convert_olmo_hybrid_weights_to_hf.py, tests/models/olmo_hybrid/test_modeling_olmo_hybrid.py, generation/utils.py, conversion_mapping.py
Any missing files or assumptions: no imports/tests; optional FLA kernels not audited, torch fallback is treated as parity reference
```

Primary source notes are in `_sources/source_notes.md`.

## 2. High-level architecture

OLMo Hybrid is a text-only causal decoder that alternates GatedDeltaNet linear-attention/state layers with ordinary full self-attention layers. The default and public 7B configs use 32 decoder blocks with `linear_attention` at most layers and `full_attention` every fourth layer.

```text
input ids -> token embeddings -> hybrid decoder blocks -> final RMSNorm -> LM head -> logits/sampling
```

Stage decomposition:

```text
CPU/tokenizer: text tokenization, chat template for instruct variants
GPU prefill: embeddings, linear layers with convolution/state updates, periodic full causal attention
GPU decode: one-token or chunked continuation using hybrid cache
Generation controller: sampling/beam handling, cache reorder, logits_to_keep slicing
```

The independently testable units are full-attention layers, GatedDeltaNet layers, the hybrid cache ABI, and final logits. DinoML should not lower this as a standard Llama/OLMo3 clone because most layers are recurrent linear-attention layers with convolutional state.

## 3. Important config dimensions

| Field | Source default | Public 7B configs |
|---|---:|---:|
| `vocab_size` | 100352 | 100352 |
| `hidden_size` | 3840 | 3840 |
| `intermediate_size` | 11008 | 11008 |
| `num_hidden_layers` | 32 | 32 |
| `num_attention_heads` | 30 | 30 |
| `num_key_value_heads` | defaults to attention heads | 30 |
| full attention `head_dim` | `hidden_size // num_attention_heads` unless explicit | 128 |
| `linear_num_key_heads` | defaults to attention heads | 30 |
| `linear_num_value_heads` | defaults to attention heads | 30 |
| `linear_key_head_dim` | `int(0.75 * hidden_size / linear_num_key_heads)` | 96 |
| `linear_value_head_dim` | `2 * linear_key_head_dim` | 192 |
| linear key width | derived | 2880 |
| linear value width | derived | 5760 |
| `linear_conv_kernel_dim` | 4 | 4 |
| `linear_allow_neg_eigval` | true | true |
| activation | `silu` | `silu` |
| `rms_norm_eps` | `1e-6` | `1e-6`; Gated output norm uses `1e-5` |
| `attention_bias` | false | false |
| `max_position_embeddings` | 65536 | base 65536, instruct/think 32768 |
| `rope_parameters` | optional | base/SFT/Think null; DPO has default object with `rope_theta: null` |
| `use_cache` | true | true |
| tied embeddings | false | false |

Representative checkpoint sweep:

| Checkpoint | Context | RoPE config encoding | Dtype/config notes | Operator-significant variation |
|---|---:|---|---|---|
| `allenai/Olmo-Hybrid-7B` | 65536 | `null` | `transformers_version=4.52.0` | NoPE path; largest public context |
| `allenai/Olmo-Hybrid-Instruct-SFT-7B` | 32768 | `null` | chat tokenizer/template files | Same graph, shorter context |
| `allenai/Olmo-Hybrid-Instruct-DPO-7B` | 32768 | `{rope_type: default, rope_theta: null}` | config has `dtype=bfloat16`, newer dev version | Loader must handle object-but-disabled RoPE safely |
| `allenai/Olmo-Hybrid-Think-SFT-7B` | 32768 | `null` | chat/reasoning variant | Same graph, shorter context |

## 3a. Family variation traps

- `hidden_size != num_attention_heads * linear_key_head_dim`; linear q/k projection width is 2880, not 3840.
- Linear value width is 5760, so the linear-attention output projection is `Linear(5760 -> 3840)`.
- Full attention uses 30 heads with `head_dim=128`, while linear attention uses 30 heads with key dim 96 and value dim 192.
- Layer type is a per-layer ABI. Default pattern is three linear layers then one full-attention layer; configs can supply a custom `layer_types` list but strict validation requires at least one of each.
- NoPE is real. `OlmoHybridModel` sets `rotary_emb=None` when `rope_parameters` is absent or has null `rope_theta`; full-attention layers then skip RoPE.
- DPO-style `rope_parameters={"rope_theta": null, "rope_type": "default"}` must not accidentally instantiate RoPE.
- Attention outputs exist only for `full_attention` layers.
- GatedDeltaNet uses split q/k/v convolutions and recurrent state, so generic `DynamicCache`, static cache, and quantized cache assumptions are unsafe.
- `linear_allow_neg_eigval=True` scales beta by 2.0, changing recurrence range from `[0,1]` to `[0,2]`.
- OLMo-core conversion names linear layer norms as `attention_layer_norm` and `feedforward_layer_norm`; conversion mapping aliases them to HF names.

## 4. Operator coverage checklist

Tensor/layout ops:

- Embedding lookup `[B, T] -> [B, T, 3840]`.
- Reshape/view/transpose/contiguous for attention heads and linear heads.
- `repeat_interleave` for linear q/k when value heads exceed key heads; full attention uses repeat-kv when KV heads are fewer than Q heads.
- Cache concatenation along sequence axis for full-attention KV.
- Beam `index_select` and `repeat_interleave` over batch for all cache tensors.

Neural primitives:

- Biasless dense linear: `3840 -> 2880`, `3840 -> 5760`, `5760 -> 3840`, `3840 -> 30`, `3840 -> 11008`, `11008 -> 3840`, `3840 -> 100352`.
- Optional bias full-attention projections if a config sets `attention_bias=true`.
- RMSNorm with fp32 variance and output cast to input dtype.
- Gated RMSNorm: RMSNorm over `head_v_dim=192`, multiply by `silu(gate)` in fp32 path.
- SwiGLU MLP: `down(silu(gate(x)) * up(x))`.

Attention primitives:

- Causal self-attention, MHA/GQA capable, shape `[B, Hq, Tq, Dh] x [B, Hkv, Tk, Dh]`.
- Attention mask addition before fp32 softmax.
- SDPA/FlashAttention compatible through Transformers attention interface for full-attention layers.

Recurrent/state-space and convolution ops:

- Depthwise causal `Conv1d` with groups equal channels, kernel `linear_conv_kernel_dim`, separate q/k/v convs.
- Gated DeltaNet chunk recurrence for prefill/chunked decode.
- Gated DeltaNet recurrent one-token path for cached single-token decode.
- `sigmoid`, `softplus`, `exp`, `rsqrt`, `cumsum`, triangular masks, chunk padding to multiples of 64.
- L2 normalization of q/k inside the GatedDeltaNet kernels.

Position/cache/generation:

- Optional RoPE only for full-attention layers.
- Hybrid cache with per-layer `key_cache`, `value_cache`, `conv_states_q`, `conv_states_k`, `conv_states_v`, and `recurrent_states`.
- `logits_to_keep` slicing before LM head.
- Left-padding-sensitive linear attention mask application.

## 5. Layer/block breakdown

Full-attention decoder block, repeated where `layer_types[i] == "full_attention"`:

```text
residual = x
q = RMSNorm(Linear(3840 -> 3840, bias=attention_bias)(x))
k = RMSNorm(Linear(3840 -> 3840, bias=attention_bias)(x))
v = Linear(3840 -> 3840, bias=attention_bias)(x)
q,k = optional RoPE(q,k)
k,v = hybrid_cache.update(k,v, layer=i) if cache is present
a = causal_attention(q,k,v, mask)
x = residual + RMSNorm(Linear(3840 -> 3840, bias=attention_bias)(a))
residual = x
x = residual + RMSNorm(Linear(11008 -> 3840)(silu(Linear(3840 -> 11008)(x)) * Linear(3840 -> 11008)(x)))
```

Linear-attention decoder block, repeated where `layer_types[i] == "linear_attention"`:

```text
residual = x
x_norm = RMSNorm(x)
q = Linear(3840 -> 2880)(x_norm)
k = Linear(3840 -> 2880)(x_norm)
v = Linear(3840 -> 5760)(x_norm)
q,k,v = separate depthwise causal Conv1d + silu(q/k/v)
q,k -> [B,T,30,96], v -> [B,T,30,192]
beta = sigmoid(Linear(3840 -> 30)(x_norm)) * (2 if allow_neg_eigval else 1)
g = -exp(A_log) * softplus(Linear(3840 -> 30)(x_norm) + dt_bias)
y, recurrent_state = gated_delta_rule(q,k,v,g,beta,state)
gate = Linear(3840 -> 5760)(x_norm)
y = GatedRMSNorm(y.reshape(-1,192), gate.reshape(-1,192)).reshape(B,T,5760)
x = residual + Linear(5760 -> 3840)(y)
residual = x
x = residual + Linear(11008 -> 3840)(silu(Linear(3840 -> 11008)(RMSNorm(x))) * Linear(3840 -> 11008)(RMSNorm(x)))
```

## 6. Attention requirements

Full attention:

- Causal self-attention only; no cross-attention.
- Public 7B: MHA with `num_attention_heads=30`, `num_key_value_heads=30`, `head_dim=128`.
- Config admits GQA if `num_key_value_heads < num_attention_heads`; repeat-kv expands cache reads for eager attention.
- Cache stores full-attention keys/values as `[B, num_key_value_heads, S, head_dim]`.
- Cached keys are stored after optional RoPE.
- Mask is from `create_causal_mask`; full-attention mask is separate from linear mask.
- Attention math order: q/k projection -> q/k RMSNorm -> reshape -> optional RoPE -> cache update -> backend attention -> output projection.

Linear attention:

- Not standard softmax attention. It is GatedDeltaNet with q/k L2 norm, beta/g gates, depthwise causal conv state, and recurrent matrix state.
- Prefill/chunk path pads sequence length to chunk size 64 and returns final recurrent state when cache is enabled.
- Cached single-token decode uses `fused_recurrent_gated_delta_rule` when FLA is available, otherwise torch fallback.
- Linear cache tensors:
  - `conv_states_q`: `[B, linear_key_width, conv_kernel_dim - 1]`, public 7B `[B, 2880, 3]`.
  - `conv_states_k`: `[B, 2880, 3]`.
  - `conv_states_v`: `[B, 5760, 3]`.
  - `recurrent_states`: `[B, linear_num_value_heads, linear_key_head_dim, linear_value_head_dim]`, public 7B `[B, 30, 96, 192]`.

Generation/cache ABI:

- `past_key_values` is an `OlmoHybridDynamicCache`, not a plain tuple or generic DynamicCache.
- `has_previous_state()` keys off the last linear layer's q-conv state. This means all earlier linear layers must update state consistently before decode.
- Beam reorder must reorder KV caches, q/k/v conv states, and recurrent states together.
- Transformers generation explicitly treats OLMoHybrid as unsupported by the default dynamic-cache preparation path because its linear cache has split q/k/v conv states.

## 7. Position encoding and custom math

RoPE is optional and only used by full-attention layers. If `rotary_emb` exists, it returns fp32 `cos`/`sin`; otherwise full-attention layers run NoPE.

```python
def olmo_hybrid_rope(q, k, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin
```

Gated DeltaNet recurrence, simplified single-token form:

```python
def recurrent_delta(q_t, k_t, v_t, g_t, beta_t, state):
    q_t = l2norm(q_t) * (q_t.shape[-1] ** -0.5)
    k_t = l2norm(k_t)
    state = state * exp(g_t)[..., None, None]
    kv_mem = (state * k_t[..., None]).sum(dim=-2)
    delta = (v_t - kv_mem) * beta_t[..., None]
    state = state + k_t[..., None] * delta[..., None, :]
    y_t = (state * q_t[..., None]).sum(dim=-2)
    return y_t, state
```

Precomputable: RoPE inverse frequencies when enabled, conv weights, projection weights, `A_log`, `dt_bias`. Dynamic inputs: position ids, masks, conv state, recurrent state, and chunk padding.

## 8. Preprocessing and input packing

Runtime graph input is text-only:

- `input_ids` or `inputs_embeds`, exactly one.
- Optional `attention_mask`; linear attention expects left-padding semantics when padding is present.
- If `position_ids` is absent, source builds a contiguous range offset by full-attention cache length.
- Chat variants rely on tokenizer chat templates outside the neural graph.

No image/audio/video packing, multimodal scatter, tokenizer codebook, or discrete codec path is present.

## 9. Graph rewrite / lowering opportunities

### Rewrite: NoPE full-attention specialization

Source pattern: `rotary_emb is None`, full-attention layer passes `position_embeddings=None`.

Replacement: omit RoPE graph entirely for full-attention layers.

Preconditions:

- `config.rope_parameters is None`, or `config.rope_parameters["rope_theta"] is None`.
- Preserve `position_ids` for mask/cache length logic even if not consumed by RoPE.

Failure cases:

- Any config with a real `rope_theta` or non-default RoPE scaling.

Parity test sketch: compare one full-attention block with and without the rewrite for a config whose RoPE is disabled.

### Rewrite: depthwise causal Conv1d to rolling window kernel

Source pattern: `Conv1d(groups=channels, kernel=K, padding=K-1)`, output cropped to input length, activation `silu`.

Replacement: specialized depthwise causal convolution over `[B,T,C]` plus explicit state `[B,C,K-1]`.

Preconditions:

- `groups == in_channels == out_channels`.
- `bias == false` for public OLMo Hybrid q/k/v convs.
- `padding == kernel_size - 1`.
- Preserve source transpose order `[B,T,C] <-> [B,C,T]`.
- Decode path must prepend cached state and drop oldest token exactly as source.

Failure cases:

- Any future non-depthwise, biased, non-silu, or noncausal convolution.

Parity test sketch: prefill, cached one-token decode, and cached multi-token continuation must match source output and state.

### Rewrite: linear q/k/v projection scheduling

Source pattern: separate `q_proj`, `k_proj`, `v_proj` with separate convs.

Replacement: optional packed GEMM only if outputs are split back before three independent conv kernels.

Preconditions:

- Preserve weight split order `[q, k, v]`.
- Packed output widths are `[2880, 2880, 5760]` for public 7B.
- No fusion across convs unless state update ABI remains separate for q/k/v.

Failure cases:

- Do not import Qwen3Next's fused qkvz layout; OLMo Hybrid intentionally uses separate projections.

Parity test sketch: packed projection output split equals three standalone linears bitwise or within dtype tolerance before conv.

### Rewrite: last-token-only logits

Source pattern: `lm_head(hidden_states[:, slice_indices, :])` using `logits_to_keep`.

Replacement: only materialize requested logits rows for generation.

Preconditions:

- No loss computation requiring full shifted logits.
- `logits_to_keep` is positive int or explicit tensor indices.

Failure cases:

- Training/loss or callers requesting all logits.

## 10. Kernel fusion candidates

Highest priority:

- Hybrid cache/state ABI kernels: q/k/v rolling depthwise conv plus recurrent state update. This dominates decode correctness and cannot be faked with a KV-only cache.
- GatedDeltaNet provider: chunked prefill and recurrent one-token decode, including q/k L2 norm and beta/g gates.
- RMSNorm and GatedRMSNorm+silu gate for 3840 and 192-wide vectors.
- Full-attention FlashAttention/SDPA only for the eight full-attention layers in public 32-layer configs.

Medium priority:

- MLP SwiGLU fused GEMM/activation/multiply/down projection.
- Packed q/k/v projection for linear layers with safe split.
- Last-token-only LM head.
- Full-attention q/k RMSNorm + RoPE + attention prefill fusion when RoPE is enabled.

Lower priority:

- Beam cache reorder kernels.
- Generic RoPE variants, because public inspected configs mostly use NoPE.
- Tensor parallel sharding plans.

## 11. Runtime staging plan

Stage 1: parse config and layer schedule, load weights, construct one full-attention block and one linear-attention block.

Stage 2: implement RMSNorm, GatedRMSNorm, NoPE handling, and shape guards for explicit linear projection dims.

Stage 3: implement linear layer depthwise conv fallback and exact cache state ABI.

Stage 4: implement torch-reference GatedDeltaNet chunk and recurrent kernels for parity; use FLA-like optimized kernels only behind provider manifests.

Stage 5: implement full prefill parity for a short sequence and mixed layer stack.

Stage 6: implement decode with `OlmoHybridDynamicCache` equivalent, including multi-token cached continuation.

Stage 7: add optimized attention/GatedDeltaNet providers, last-token logits, and scheduling/fusion.

Initially stub: beam search, tensor parallel, quantized cache, custom FLA provider acceleration, and training.

## 12. Parity and validation plan

- Config parsing tests for base, SFT, DPO, and Think configs, especially disabled RoPE encodings.
- Random tensor tests for RMSNorm, GatedRMSNorm, short convolution state update, l2norm, beta/g calculation.
- Single linear-attention layer parity for prefill with `use_cache=false` and `use_cache=true`.
- Multi-token cached continuation parity: first token in a chunk must match single-token cached decode.
- Full-attention layer parity with NoPE and with synthetic real RoPE config.
- Hybrid stack after N layers, checking hidden states and cache tensors.
- Prefill logits parity against `hf-internal-testing/olmo-hybrid` or a locally converted small checkpoint.
- Greedy decode token parity for a short prompt.
- Suggested tolerances: fp32 `1e-5`/`1e-5`; bf16/fp16 block-level `1e-2` to `5e-2` depending on GatedDeltaNet provider, with tighter reference tolerances before provider substitution.

No validation was run for this audit.

## 13. Performance probes

- Prefill throughput by sequence length, separating linear layers from full-attention layers.
- Decode tokens/sec with cache enabled, batch-size sweep.
- GatedDeltaNet chunk-size sensitivity; source fallback uses chunk size 64.
- Cache memory by component: full KV, q/k/v conv states, recurrent states.
- Optional FLA provider vs DinoML torch-reference equivalent.
- Full-attention backend comparison on only full-attention layers.
- Last-token LM-head cost for vocab 100352.
- Left-padded vs unpadded batch behavior for linear attention mask path.

## 14. Skip/defer list

- Training and gradient checkpointing.
- Generic cache implementations, static cache, quantized cache, and offloaded cache.
- Tensor parallel and pipeline parallel.
- Beam search until cache reorder is validated for all state tensors.
- FLA-native provider acceleration before reference parity.
- Real RoPE scaling variants unless a target config uses non-null `rope_theta`.
- Quantized or packed weights; inspected source is dense safetensors.

## 15. Final implementation checklist

- [ ] Parse `OlmoHybridConfig`, including `layer_types` validation and disabled-RoPE encodings.
- [ ] Load dense weights with OLMo-core conversion aliases.
- [ ] Implement explicit linear-attention projection dims: key width, value width, q/k/v/a/b/g/o.
- [ ] Implement RMSNorm and GatedRMSNorm.
- [ ] Implement separate q/k/v depthwise causal Conv1d with rolling state.
- [ ] Implement GatedDeltaNet chunk prefill reference.
- [ ] Implement GatedDeltaNet recurrent one-token decode reference.
- [ ] Implement hybrid cache with KV, q/k/v conv states, and recurrent states.
- [ ] Implement full causal attention layers with optional NoPE/RoPE.
- [ ] Implement SwiGLU MLP and final LM head with `logits_to_keep`.
- [ ] Add safe rewrites for NoPE, rolling conv, packed linear projections, and last-token logits.
- [ ] Add config sweep tests for public 7B variants.
- [ ] Add single-layer, hybrid-stack, prefill-logit, and decode-token parity tests.
- [ ] Benchmark prefill, decode, cache memory, and provider candidates separately.
