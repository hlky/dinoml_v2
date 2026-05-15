# Qwen3-Next Transformers Family Audit

Primary target: `Qwen3NextForCausalLM` inference and generation on CUDA. This is a source/config audit only; no DinoML runtime code was edited, no DinoML tests were run, and no commit was made.

## 1. Source basis

```text
Transformers commit/version: local checkout transformers at b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model family: qwen3_next
Primary task: causal LM prefill/decode/generation
Local source root: transformers
```

Source files inspected:

- `transformers/src/transformers/models/qwen3_next/configuration_qwen3_next.py`
- `transformers/src/transformers/models/qwen3_next/modeling_qwen3_next.py`
- `transformers/src/transformers/models/qwen3_next/modular_qwen3_next.py`
- Cross-checks: `src/transformers/cache_utils.py`, `src/transformers/models/qwen3_moe/modeling_qwen3_moe.py`, `src/transformers/models/qwen2_moe/modeling_qwen2_moe.py`, `src/transformers/models/gemma3/modeling_gemma3.py`, `src/transformers/models/bamba/modeling_bamba.py`, `src/transformers/models/mixtral/modeling_mixtral.py`

Source URLs at the inspected commit:

- `https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/qwen3_next/configuration_qwen3_next.py`
- `https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/qwen3_next/modeling_qwen3_next.py`
- `https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/qwen3_next/modular_qwen3_next.py`

Representative configs inspected from Hugging Face raw files:

- `Qwen/Qwen3-Next-80B-A3B-Instruct`
- `Qwen/Qwen3-Next-80B-A3B-Thinking`
- `Qwen/Qwen3-Next-80B-A3B-Instruct-FP8`
- `Qwen/Qwen3-Coder-Next`
- `Qwen/Qwen3-Coder-Next-Base`
- `Qwen/Qwen3-Coder-Next-FP8`
- `tiny-random/qwen3-next-moe`, used only as an open tiny/debug config mirror

Small snapshots are under `agents/plans/transformers/qwen3_next/_sources/`: fetched `config.json` files and selected `generation_config.json` files.

Authoritative source note: `modeling_qwen3_next.py` is generated from `modular_qwen3_next.py`; future Transformers source edits should be checked in the modular file. This report uses both: modular source for intended inheritance and generated source for expanded classes such as the native `Qwen3NextForCausalLM`.

Missing files or assumptions:

- No remote code is required for the audited in-library class.
- No multimodal processor is consumed by `Qwen3NextForCausalLM`; tokenization routes through Qwen2 tokenizer auto mapping.
- FP8 configs add quantization metadata, but the native model graph remains `Qwen3NextForCausalLM`. Treat FP8/NVFP4/GGUF/AWQ repos as separate weight-admission work.
- `Qwen3-Next` and `Qwen3-Coder-Next` official configs sampled omit explicit `layer_types`; source defaults make 3 linear-attention layers followed by 1 full-attention layer repeatedly.

## 2. High-level architecture

Qwen3-Next is a text-only hybrid decoder with linear-attention/state layers, sparse MoE MLPs, and periodic full GQA attention:

```text
tokenization/input_ids -> embedding -> 48 hybrid decoder blocks -> final RMSNorm -> lm_head -> logits/sampling
                         | each block: RMSNorm -> token mixer -> residual -> RMSNorm -> MoE/shared MLP -> residual
                         | token mixer: mostly gated-delta linear attention, every 4th layer full causal GQA
```

Stage decomposition:

- CPU/data pipeline: Qwen tokenizer/chat template, special-token handling, attention mask, generation controller fields from `generation_config.json`.
- GPU/runtime prefill: embedding, RoPE cos/sin construction for full-attention layers, 36 gated-delta linear-attention layers, 12 full GQA layers, MoE routing/expert execution in every official sampled layer, final RMSNorm, logits.
- GPU/runtime decode: position IDs from cache sequence length, per-layer state update; full-attention layers append KV cache, linear-attention layers update fixed-size convolution and recurrent state tensors.
- Independently stageable pieces: full-attention block parity can share Qwen3/Qwen3-MoE coverage with one extra query gate; linear-attention block requires new gated-delta scan/cache work; MoE block can reuse Qwen2-MoE shared-expert behavior.

Implemented heads:

- Required for target: `Qwen3NextForCausalLM`.
- Optional/deferred: base `Qwen3NextModel` hidden-state output.
- Deferred for first causal-LM target: sequence classification, token classification, and question answering generic heads.

## 3. Important config dimensions

Source defaults from `Qwen3NextConfig`:

| Field | Source default / behavior |
| --- | --- |
| `vocab_size` | 151936 |
| `hidden_size` | 2048 |
| `intermediate_size` | 5632 for dense fallback MLP layers |
| `num_hidden_layers` | 48 |
| `num_attention_heads` / `num_key_value_heads` | 16 / 2 |
| `head_dim` | 256; do not assume `hidden_size / heads` |
| `hidden_act` | `silu` |
| `max_position_embeddings` | 32768 default; sampled official configs use 262144 |
| `rms_norm_eps` | 1e-6 |
| `use_cache` | true |
| `tie_word_embeddings` | false |
| `attention_bias` / `attention_dropout` | false / 0.0 |
| `rope_parameters` | post-init default includes `partial_rotary_factor=0.25`; sampled configs also provide raw `rope_theta` |
| `linear_conv_kernel_dim` | 4 |
| `linear_key_head_dim` / `linear_value_head_dim` | 128 / 128 |
| `linear_num_key_heads` / `linear_num_value_heads` | 16 / 32 |
| `decoder_sparse_step` / `mlp_only_layers` | 1 / `[]` after post-init, so official sampled layers are MoE |
| `moe_intermediate_size` / `shared_expert_intermediate_size` | 512 / 512 |
| `num_experts` / `num_experts_per_tok` | 512 / 10 |
| `norm_topk_prob` | true |
| `layer_types` | if omitted, `"linear_attention"` except every `full_attention_interval=4` layer is `"full_attention"` |

Representative checkpoint sweep. Dimensions are from fetched `config.json`; `layer_types=default` means the source post-init pattern described above.

| Model id | H | Layers | Heads/KV | Head dim | Max pos | RoPE theta | Linear K/V heads | Experts/top-k | Expert I/shared I | Layer types | Dtype/quant |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `Qwen/Qwen3-Next-80B-A3B-Instruct` | 2048 | 48 | 16/2 | 256 | 262144 | 10000000 | 16/32 | 512/10 | 512/512 | default | bf16 |
| `Qwen/Qwen3-Next-80B-A3B-Thinking` | 2048 | 48 | 16/2 | 256 | 262144 | 10000000 | 16/32 | 512/10 | 512/512 | default | bf16 |
| `Qwen/Qwen3-Next-80B-A3B-Instruct-FP8` | 2048 | 48 | 16/2 | 256 | 262144 | 10000000 | 16/32 | 512/10 | 512/512 | default | fp8 config, bf16 logical dtype |
| `Qwen/Qwen3-Coder-Next` | 2048 | 48 | 16/2 | 256 | 262144 | 5000000 | 16/32 | 512/10 | 512/512 | default | bf16 |
| `Qwen/Qwen3-Coder-Next-Base` | 2048 | 48 | 16/2 | 256 | 262144 | 5000000 | 16/32 | 512/10 | 512/512 | default | bf16 |
| `Qwen/Qwen3-Coder-Next-FP8` | 2048 | 48 | 16/2 | 256 | 262144 | 5000000 | 16/32 | 512/10 | 512/512 | default | fp8 config, bf16 logical dtype |
| `tiny-random/qwen3-next-moe` | 8 | 4 | 16/8 | 32 | 262144 | 10000000 | 8/16 | 32/10 | 32/32 | 3 linear, 1 full | debug only |

Generation config sweep:

| Model id | Generation fields observed |
| --- | --- |
| `Qwen3-Next-80B-A3B-Instruct` | `do_sample=true`, `temperature=0.7`, `top_p=0.8`, `top_k=20`, `eos_token_id=[151645,151643]`, `pad_token_id=151643` |
| `Qwen3-Next-80B-A3B-Thinking` | `do_sample=true`, `temperature=0.6`, `top_p=0.95`, `top_k=20`, same EOS/pad IDs |
| `Qwen3-Coder-Next` and `Qwen3-Coder-Next-Base` | `do_sample=true`, `temperature=1`, `top_p=0.95`, `top_k=40`, same EOS/pad IDs |

## 3a. Family variation traps

- This is not Qwen3 dense and not plain Qwen3-MoE. It is a hybrid decoder: most layers are gated-delta linear attention with convolution/recurrent state, while every fourth default layer is full causal GQA.
- The full-attention query projection is Qwen3-Next-specific: `q_proj` outputs `num_attention_heads * head_dim * 2`, split into query and a sigmoid gate. The gate multiplies attention output before `o_proj`.
- Full attention uses Q/K per-head RMSNorm before RoPE, like Qwen3/Qwen3-MoE in placement, but Qwen3-Next's norm class is Gemma3-style `(1 + weight)` rather than Qwen3-MoE's direct weight multiply. Norm weights are length `head_dim=256`.
- RoPE is partial by default: `partial_rotary_factor=0.25`, so with `head_dim=256` only 64 rotary dimensions are produced for full-attention Q/K. This differs from ordinary Qwen3/Qwen3-MoE reports where sampled dense configs use full head-dim RoPE.
- Linear attention has no KV cache. It owns two fixed-size states per linear layer: convolution state and recurrent gated-delta state.
- Linear attention has separate key/value head spaces: K heads/dim `16 x 128`; V heads/dim `32 x 128`. Q/K are repeated from key heads to value heads before the gated-delta rule when `num_v_heads / num_k_heads > 1`.
- Source fallback linear attention is algorithmically correct but extremely slow for production: chunked prefill uses Python loops over chunks and intra-chunk rows, while single-token decode loops over sequence length. DinoML should treat FLA/causal-conv1d semantics as the production target, not the fallback implementation strategy.
- All sampled official configs route every MLP through Qwen2-MoE-style sparse block with a shared expert and shared expert gate. This differs from Qwen3-MoE, which does not have the Qwen2-MoE shared expert path.
- Expert count/top-k is large: 512 experts with top-10 selected per token. Do not reuse Mixtral top-2-over-8 assumptions.
- Expert weights are packed as `gate_up_proj[E, 2*moe_intermediate_size, hidden_size]` and `down_proj[E, hidden_size, moe_intermediate_size]`.
- Official FP8 configs are graph-identical but add `quantization_config`. First native-graph integration can reject FP8 weight admission while still accepting bf16 configs.
- `hidden_size != num_attention_heads * head_dim` for official configs: `2048 != 16 * 256`. Attention output width is 4096 before `o_proj`, and query projection output is 8192 including the gate.
- Layout translation is low value. The semantic layout is `[batch, sequence, hidden]`, with many axis-sensitive ops: RMSNorm `dim=-1`, Q/K/V split order, grouped Conv1d channels, chunk padding on sequence, top-k over experts, RoPE last-dim pairing, and cache state layouts.

## 4. Operator coverage checklist

Tensor/layout ops:

- Integer embedding lookup: `input_ids [B,S] -> [B,S,H]`.
- `view`, `reshape`, `transpose`, `contiguous`, `split`, `chunk`, `cat`, `pad`, `roll`, slicing, `expand`, `repeat_interleave`.
- Residual adds and dtype casts.
- `logits_to_keep` int or tensor slicing before LM head.

Neural network primitives:

- Gemma3-style RMSNorm: fp32 variance, multiply by `(1 + weight.float())`, cast back. Qwen3-Next initializes these RMSNorm weights to zero for this centered form, including hidden-state norms and Q/K head norms.
- Gated RMSNorm for linear attention output: normalize value head dim, multiply learned weight, then multiply `silu(z.float())`.
- Bias-free linears in sampled configs:
  - Full attention Q gate: `Linear(2048 -> 8192)`, view to `[B,S,16,512]`, chunk to query/gate `[B,S,16,256]`.
  - Full attention K/V: `Linear(2048 -> 512)` each, `2 x 256`.
  - Full attention O: `Linear(4096 -> 2048)`.
  - Linear attention `in_proj_qkvz`: `Linear(2048 -> 12288)` from `key_dim*2 + value_dim*2`.
  - Linear attention `in_proj_ba`: `Linear(2048 -> 64)` for beta/a gates.
  - Linear attention `out_proj`: `Linear(4096 -> 2048)`.
  - Shared dense expert: gate/up `2048 -> 512`, down `512 -> 2048`, shared gate `2048 -> 1`.
  - Routed expert: packed gate/up `2048 -> 1024` per selected expert, split to 512/512; down `512 -> 2048`.
  - LM head: `Linear(2048 -> 151936)`.
- Depthwise grouped Conv1d over `conv_dim=8192`, `kernel_size=4`, `groups=8192`, no bias, causal padding/truncation.
- SiLU, sigmoid, softplus, exp, log-parameter `A_log`, elementwise multiply/add/sub.

Full-attention primitives:

- Causal self-attention with GQA, Q heads 16, KV heads 2, `head_dim=256`, scaling `head_dim**-0.5`.
- Eager fallback repeats KV from 2 to 16 heads, QK matmul, mask add, fp32 softmax, cast to query dtype, dropout only in training, AV matmul.
- Backend dispatch through `ALL_ATTENTION_FUNCTIONS` for eager/SDPA/FlashAttention-compatible paths.

Linear-attention/state primitives:

- Padding-mask multiply before projection for linear-attention layers only.
- Gated-delta rule with Q/K L2 normalization inside the kernel, beta sigmoid, decay `g = -exp(A_log.float()) * softplus(a.float() + dt_bias)`.
- Chunked prefill default chunk size 64, pads sequence to chunk multiple, uses recurrent state across chunks.
- Recurrent single-token decode path that updates fixed recurrent state.
- Cache state copy/update with static address preservation.

MoE/router ops:

- Router linear `2048 -> 512`, fp32 softmax, top-k 10, top-k probability renormalization, cast to router dtype.
- Expert grouping via one-hot/permute/nonzero/where in eager source; optimized lowering should avoid physical one-hot when possible.
- Per-expert token gather, packed gate/up GEMM, SiLU multiply, down GEMM, route-weight multiply, scatter/index-add back to flattened token order.
- Shared expert output plus `sigmoid(shared_expert_gate(x))`.

Position/rotary ops:

- Default RoPE inverse frequency from `rope_theta` and `dim = int(head_dim * partial_rotary_factor)`.
- Cos/sin computed in fp32, cast to hidden dtype, applied to Q/K after Q/K RMSNorm and before KV cache update.

Generation/cache ops:

- `DynamicCache(config)` creation when `use_cache=true`.
- Full-attention cache: KV tensors per full-attention layer, stored after RoPE, before KV repeat.
- Linear-attention cache: per-linear-layer conv/recurrent states, no sequence growth in those state tensors.
- Cache reorder/reset/offload semantics from shared `Cache` and `LinearAttentionLayer`.

Distributed/tensor-parallel metadata:

- Config declares TP plans for attention, Q/K norms, MoE packed experts, shared expert, and LM head. First DinoML integration can be single GPU, but packed expert weight identity and row/col orientation should be preserved.

## 5. Layer/block breakdown

Default layer pattern for 48 layers:

```text
layers where (idx + 1) % 4 != 0: linear_attention  # 36 layers
layers where (idx + 1) % 4 == 0: full_attention    # 12 layers
```

Full-attention decoder block:

```text
x0: [B,S,2048]
residual = x0
x = Gemma3RMSNorm_H(x0)
q_gate = Linear(2048 -> 8192)(x).view(B,S,16,512)
q, gate = chunk(q_gate, 2, dim=-1)        # each [B,S,16,256]
k = Linear(2048 -> 512)(x).view(B,S,2,256)
v = Linear(2048 -> 512)(x).view(B,S,2,256)
q = RMSNorm_D(q).transpose(1,2)           # [B,16,S,256]
k = RMSNorm_D(k).transpose(1,2)           # [B,2,S,256]
v = v.transpose(1,2)
q,k = RoPE_partial(q,k,cos,sin)
k,v = cache.update(k,v,layer_idx) if cache enabled
a = causal_gqa_attention(q,k,v,mask)
a = a.reshape(B,S,4096) * sigmoid(gate.reshape(B,S,4096))
x = residual + Linear(4096 -> 2048)(a)
x = x + moe_or_dense_mlp(Gemma3RMSNorm_H(x))
```

Linear-attention decoder block:

```text
x0: [B,S,2048]
residual = x0
x = Gemma3RMSNorm_H(x0)
x = x * attention_mask[..., None] only when a left-padding mask is active
qkvz = Linear(2048 -> 12288)(x)
ba = Linear(2048 -> 64)(x)
q,k,v,z,b,a = reorder/split(qkvz, ba)
mixed_qkv = cat(q,k,v).transpose(1,2)      # [B,8192,S]
mixed_qkv = causal_depthwise_conv1d_or_update(mixed_qkv, conv_state)
q,k,v = split(mixed_qkv.transpose(1,2), [2048,2048,4096])
q,k -> [B,S,16,128], repeat to [B,S,32,128]
v -> [B,S,32,128]
beta = sigmoid(b)
g = -exp(A_log.float()) * softplus(a.float() + dt_bias)
y, recurrent_state = gated_delta_rule(q,k,v,g,beta,state)
y = gated_rmsnorm_per_value_head(y, z)
x = residual + Linear(4096 -> 2048)(y.reshape(B,S,4096))
x = x + moe_or_dense_mlp(Gemma3RMSNorm_H(x))
```

Official sampled configs make `moe_or_dense_mlp` a Qwen2-MoE sparse block in every layer:

```text
flat = x.reshape(B*S,2048)
shared = sigmoid(Linear(2048 -> 1)(flat)) * down(silu(gate(flat)) * up(flat))
router = softmax(Linear(2048 -> 512)(flat), dtype=float32)
top10_values, top10_indices = topk(router)
top10_values = top10_values / sum(top10_values)
routed = sum_selected_experts(flat, top10_indices, top10_values)
return (routed + shared).reshape(B,S,2048)
```

## 6. Attention requirements

Full attention:

- Causal self-attention only; no cross-attention in the audited primary target.
- GQA: 16 query heads, 2 KV heads, 8 query groups per KV head in sampled official configs.
- KV cache stores `[B, 2, cached_S, 256]` keys and values per full-attention layer. The eager attention path expands to `[B,16,cached_S,256]`; optimized kernels should avoid materializing this repeat.
- Keys are cached after Q/K RMSNorm and RoPE.
- Masking uses `create_causal_mask`; attention mask is additive for eager attention.
- Source advertises FlashAttention and SDPA support. Fused attention parity must preserve the order: Q/K RMSNorm, RoPE, cache update, scale by `head_dim**-0.5`, mask add, fp32 softmax semantics.

Linear attention:

- Not a KV-cache mechanism. The cache ABI is per linear layer:
  - `conv_states`: `[B, conv_dim=8192, linear_conv_kernel_dim=4]`, dtype/device from the first update.
  - `recurrent_states`: `[B, linear_num_value_heads=32, linear_key_head_dim=128, linear_value_head_dim=128]`.
- `conv_states` are updated before causal convolution output. Multi-token cached continuation prepends conv context; single-token decode calls `causal_conv1d_update` and mutates `conv_state`.
- `recurrent_states` are the final gated-delta state and are copied into the cache each call. They remain fixed-size with sequence length.
- `Cache.has_previous_state(layer_idx)` controls decode branch choice. Calling linear state methods on full-attention layers is an error.
- `Cache.reorder_cache(beam_idx)` reorders both conv and recurrent states along batch dimension, while full-attention layers reorder KV tensors.
- `Cache.get_seq_length()` uses the first full-attention layer because linear layers do not track sequence length. Position IDs therefore follow full-attention KV length.

Packed/varlen support:

- `FlashAttentionKwargs` can flow through full-attention backend dispatch. The Qwen3-Next linear-attention source also imports FLA kernels but does not expose a separate public packed metadata contract in the model forward beyond standard kwargs and masks.

## 7. Position encoding and custom math

Default RoPE for full-attention layers:

```python
def qwen3_next_default_inv_freq(config):
    dim = int(config.head_dim * config.rope_parameters.get("partial_rotary_factor", 1.0))
    base = config.rope_parameters["rope_theta"]
    return 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
```

RoPE application is the standard Qwen/Bamba half-rotation on the final rotary dimension, with `cos`/`sin` unsqueezed on the head axis. It is applied only to full-attention Q/K after Q/K RMSNorm. Linear-attention layers do not use RoPE.

Gated-delta linear attention math, reduced to the runtime-significant pieces:

```python
beta = sigmoid(b)
g = -exp(A_log.float()) * softplus(a.float() + dt_bias)
q = l2norm(q, dim=-1, eps=1e-6)
k = l2norm(k, dim=-1, eps=1e-6)
q = q * (q.shape[-1] ** -0.5)
state = state * exp(g_t) + k_t[..., None] * ((v_t - (state * k_t[..., None]).sum(-2)) * beta_t)[..., None, :]
out_t = (state * q_t[..., None]).sum(-2)
```

The source chunked fallback uses a more parallel chunk formulation for prefill and a recurrent formulation for single-token decode. DinoML can precompute RoPE frequencies by max position/base/dim, but gated-delta state depends on runtime sequence, masks, and previous state.

## 8. Preprocessing and input packing

Text-only runtime contract:

- Inputs are `input_ids [B,S]`, optional `attention_mask [B,S]`, optional `position_ids [B,S]`, optional `inputs_embeds [B,S,H]`, optional `past_key_values`.
- Exactly one of `input_ids` or `inputs_embeds` is required.
- If `position_ids` are omitted, source computes `arange(S) + past_seen_tokens`, where `past_seen_tokens` comes from full-attention cache length.
- For linear-attention layers, `attention_mask` is passed as a padding-state mask only until cache state exists or when all tokens are unmasked. The source note says left padding is used for linear attention masking.
- No image/audio/video tensors, grid metadata, cu-seqlens requirement, or embedding scatter is part of this family.
- Generation controller uses standard `GenerationMixin`; sampling fields in generation configs are outside the compiled module graph.

## 9. Graph rewrite / lowering opportunities

### Rewrite: Full-Attention Q Projection Split

Source pattern:

```text
Linear(H -> 2 * QH * D) -> view(B,S,QH,2D) -> chunk(query, gate)
```

Replacement:

```text
single GEMM -> two logical views, or two GEMMs sharing packed source weight
```

Preconditions:

- `attention_bias` support matches config.
- Output split is exactly equal halves along final head dimension.
- Gate is consumed only by `sigmoid(gate.reshape(B,S,QH*D)) * attn_output`.

Failure cases: nonstandard remote-code projection packing or biasful configs if bias split is not represented.

Parity test sketch: compare full-attention layer output before/after split rewrite with random hidden states and fixed cos/sin/cache disabled.

### Rewrite: Depthwise Causal Conv1d Kernel

Source pattern:

```text
[B,S,8192] -> transpose -> depthwise Conv1d(groups=8192,k=4,pad=3) -> SiLU -> truncate
```

Replacement:

```text
custom causal depthwise temporal kernel, with decode update path over cached 4-token state
```

Preconditions:

- `groups == channels == conv_dim`.
- `kernel_size == linear_conv_kernel_dim`.
- `bias is None` in current source.
- Activation is `silu`.

Failure cases: different activation, non-depthwise groups, biasful conv, or chunked decode with cached context not prepended/trimmed exactly.

### Rewrite: Gated Delta Rule to Scan/State Kernel

Source pattern:

```text
l2norm(q,k) -> beta/g transforms -> chunk_gated_delta_rule or recurrent_gated_delta_rule
```

Replacement:

```text
prefill scan kernel producing [B,S,VH,VD] output and final [B,VH,KD,VD] state;
decode step kernel mutating/returning fixed recurrent state
```

Preconditions:

- `linear_key_head_dim == linear_value_head_dim == 128` for first optimized kernel.
- `num_v_heads % num_k_heads == 0`.
- Q/K repeat ratio is explicit.
- `use_qk_l2norm_in_kernel=True`.

Failure cases: fallback chunk size changes, masking semantics not reproduced, state dtype/address mutation contract not honored.

### Rewrite: MoE Top-K Expert Grouped GEMM

Source pattern:

```text
router softmax -> topk -> per-expert token gather -> packed gate/up GEMM -> down GEMM -> index_add
```

Replacement:

```text
top-k router + token bucketing + grouped GEMM for selected experts + scatter-add
```

Preconditions:

- Packed weights match `gate_up_proj[E,2I,H]`, `down_proj[E,H,I]`.
- `norm_topk_prob` is applied when true.
- Shared expert branch is preserved and added once per token.

Failure cases: token order instability, non-deterministic top-k tie handling, training aux-loss-only paths mixed into inference.

### Rewrite: Last-Token Logits

Source pattern:

```text
slice_indices = slice(-logits_to_keep, None)
lm_head(hidden_states[:, slice_indices, :])
```

Replacement:

```text
for decode/default logits_to_keep=1, apply LM GEMM only to last token
```

Preconditions: no loss computation and caller requests last-token logits.

Failure cases: labels provided, tensor-valued `logits_to_keep`, or full-sequence logits requested.

## 10. Kernel fusion candidates

Highest priority:

- Linear-attention state kernels: causal depthwise conv update plus gated-delta prefill/decode. This is the primary architectural blocker and cannot be replaced by KV attention kernels.
- MoE router plus grouped expert GEMMs for 512 experts/top-10, including shared expert. Sparse execution dominates dense MLP cost and naive Python-style expert loops are not viable.
- Full-attention GQA FlashAttention with KV cache and Q/K RMSNorm/RoPE order preserved. Only 12 layers are full attention, but long context makes them important.

Medium priority:

- Gemma3-style RMSNorm and gated RMSNormGated kernels.
- Fused full-attention Q projection split + per-head Q RMSNorm + RoPE prep.
- Fused linear-attention projections/split/reorder for `qkvz` and `ba`.
- Last-token-only LM head for decode.

Lower priority:

- Router aux-loss and router-logit capture, useful for diagnostics/training but not required for first inference.
- Tensor-parallel/expert-parallel lowering from HF plan metadata.
- FP8 weight-runtime path; graph parity can land first with bf16 weights.

## 11. Runtime staging plan

Stage 1: parse config and instantiate an artifact description for bf16 Qwen3-Next official configs. Reject unsupported quantization first.

Stage 2: load weights and run isolated modules against Transformers: Gemma3 RMSNorm, Q/K RMSNorm, Qwen3-Next full-attention block with cache disabled, and Qwen2-MoE shared sparse block.

Stage 3: implement full-attention layers only as a bounded path: GQA + partial RoPE + query gate + KV cache for the 12 default full-attention layers.

Stage 4: implement linear-attention prefill without cache reuse using a clear state ABI and compare one linear layer. The fallback math is acceptable for correctness tests, but production should target scan kernels.

Stage 5: implement linear-attention decode state ABI: conv state `[B,8192,4]` and recurrent state `[B,32,128,128]`, plus reorder/reset semantics.

Stage 6: integrate MoE grouped expert execution and shared expert branch for all 48 layers.

Stage 7: end-to-end prefill logits parity, then one-token and multi-token decode parity with mixed full-attention KV and linear-attention recurrent states.

Stage 8: add optimized kernels/fusions, FP8 admission, and batching/scheduling once graph parity is stable.

## 12. Parity and validation plan

- Config parser tests: official Instruct, Thinking, Coder, Coder Base, and FP8 configs; verify default layer type expansion to 36 linear and 12 full layers.
- Unit tests:
  - Gemma3-style RMSNorm vs Transformers, including zero-initialized weight behavior.
  - Partial RoPE with `partial_rotary_factor=0.25`, `head_dim=256`, theta 5e6 and 1e7.
  - Q projection split/gate shape and numerical parity.
  - Depthwise causal conv prefill and single-token update with random conv state.
  - Gated-delta recurrent update for small dims, fp32 reference tolerance.
  - Router top-k renormalization and shared expert branch.
- Single-layer parity:
  - One default linear-attention layer with and without padding mask.
  - One full-attention layer with KV cache disabled and enabled.
  - One MoE block with deterministic router inputs.
- End-to-end tests:
  - tiny/debug config prefill logits.
  - Official-shape synthetic-weight smoke for shape/cache ABI.
  - One-token decode parity after prefill, verifying full-attention cache length and linear state shapes.
- Recommended tolerances: fp32 reference `rtol=1e-4, atol=1e-5`; bf16/fp16 graph parity `rtol=2e-2, atol=2e-2` initially, tightened per kernel once accumulation order is fixed.

## 13. Performance probes

- Prefill-only throughput by sequence length: 1K, 8K, 32K, 128K, 262K where feasible.
- Decode tokens/sec with batch sweep and cache memory accounting split into full-attention KV and linear recurrent states.
- Gated-delta scan backend comparison: fallback PyTorch, Triton/custom, FLA if available.
- Depthwise conv update bandwidth and latency for `[B,8192,S]`.
- MoE router + grouped GEMM timing by batch/tokens, top-10 occupancy, expert imbalance, and shared expert overhead.
- Full-attention backend comparison: eager, SDPA, FlashAttention-compatible GQA.
- LM head last-token vs full-sequence logits.
- Memory probe: 12-layer KV cache vs 36-layer fixed recurrent/conv state at long context.

## 14. Skip/defer list

- Training, labels/loss, router aux loss, gradient checkpointing.
- Classification/token-classification/question-answering heads.
- FP8/NVFP4/AWQ/GGUF quantized weight execution.
- Tensor parallel, expert parallel, and multi-GPU sharding.
- Beam search cache reorder beyond preserving the ABI in tests.
- Non-default `layer_types`, non-default RoPE variants, and custom remote-code configs until explicitly admitted.
- Packed/varlen FlashAttention metadata beyond standard dense attention masks for first parity.

## 15. Final implementation checklist

- [ ] Parse `Qwen3NextConfig`, including default `layer_types` expansion and partial RoPE normalization.
- [ ] Load bf16 weights with packed expert tensors and shared expert branch.
- [ ] Implement Gemma3-style RMSNorm and Q/K per-head RMSNorm.
- [ ] Implement partial RoPE for full-attention layers.
- [ ] Implement Qwen3-Next full attention: gated Q projection, GQA, KV cache, output gate.
- [ ] Define mixed cache manifest: full-attention KV layers plus linear-attention conv/recurrent state layers.
- [ ] Implement linear-attention depthwise causal conv prefill/update.
- [ ] Implement gated-delta rule prefill and decode state update.
- [ ] Implement MoE router top-k 10 over 512 experts with renormalization.
- [ ] Implement routed expert grouped GEMM and shared expert addition.
- [ ] Add last-token LM-head lowering for decode.
- [ ] Add single-layer parity tests for full attention, linear attention, and MoE.
- [ ] Add end-to-end tiny/debug prefill and decode parity.
- [ ] Add official-shape synthetic cache/state ABI smoke tests.
- [ ] Benchmark prefill, decode, gated-delta state kernels, MoE grouped GEMM, and cache memory.
