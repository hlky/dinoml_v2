# Jamba Transformers Family Audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: ai21labs/Jamba-v0.1 as the primary in-library reference
Config source: HF config.json files listed below, plus JambaConfig defaults
Source files inspected:
  transformers/src/transformers/models/jamba/configuration_jamba.py
  transformers/src/transformers/models/jamba/modeling_jamba.py
  transformers/src/transformers/models/jamba/modular_jamba.py
  transformers/src/transformers/cache_utils.py for hybrid cache classes
  transformers/src/transformers/models/llama/modeling_llama.py for inherited attention conventions
  transformers/src/transformers/models/mistral/modeling_mistral.py for dense SwiGLU MLP
  transformers/src/transformers/models/mixtral/modeling_mixtral.py for expert tensor layout
Any missing files or assumptions:
  Official AI21 Jamba 1.5 Mini/Large configs were gated when fetched unauthenticated.
  Their model cards were used only for repo/task context, not exact operator dimensions.
  This report scopes current in-library Jamba source, not older remote-code behavior.
```

Representative configs inspected:

- `ai21labs/Jamba-v0.1` raw `config.json`, open.
- `ai21labs/Jamba-tiny-random` raw `config.json`, open debug/random checkpoint.
- `ai21labs/Jamba-tiny-dev` raw `config.json`, open debug/development checkpoint.
- `TechxGenus/Mini-Jamba-v2` raw `config.json`, open community small checkpoint.
- `ai21labs/AI21-Jamba-Reasoning-3B` raw `config.json`, open official 3B-style variant.
- `ai21labs/AI21-Jamba2-Mini-FP8` raw `config.json`, open compressed-tensors variant.
- `ai21labs/AI21-Jamba-1.5-Mini` and `ai21labs/AI21-Jamba-1.5-Large` repository pages/model cards, gated config downloads.

`modeling_jamba.py` starts with a generated-file warning: it is generated from `modular_jamba.py`. Future source edits should target `modular_jamba.py`; audits should still inspect generated `modeling_jamba.py` because that is what users import.

## 2. High-level architecture

Jamba is a text-only autoregressive decoder for causal LM inference. The distinguishing feature is a hybrid decoder stack:

```text
token IDs / embeddings -> periodic Mamba or attention decoder layers -> final RMSNorm -> LM head -> logits/sampling
```

Each decoder layer has pre-norm residual structure. The sequence mixer is either:

- `JambaMambaMixer`: input projection, depthwise causal Conv1d, selective SSM scan/update, output projection.
- `JambaAttention`: causal grouped-query self-attention with KV cache.

The FFN after each mixer is config-scheduled independently:

- dense `JambaMLP` SwiGLU when the layer has one expert,
- `JambaSparseMoeBlock` top-k MoE when the layer has more than one expert.

Stage decomposition:

- CPU/data pipeline: tokenizer/chat template, attention mask construction inputs, optional generation controls.
- GPU prefill: embedding lookup, full prompt pass through mixed Mamba/attention layers, populate both attention KV cache and Mamba conv/recurrent caches.
- GPU decode: one-token steps using attention KV append plus Mamba `causal_conv1d_update`/`selective_state_update`.
- Logits: source supports `logits_to_keep` to project only selected trailing positions.

The primary runtime target for DinoML should be `JambaForCausalLM`. `JambaModel` is required as the body. `JambaForSequenceClassification` is implemented through a generic classification head and can be deferred for generation-first integration.

## 3. Important config dimensions

Effective source defaults from `JambaConfig`:

| Field | Default | Operator impact |
|---|---:|---|
| `vocab_size` | 65536 | token embedding and LM head width |
| `hidden_size` | 4096 | model width |
| `num_hidden_layers` | 32 | decoder depth |
| `num_attention_heads` | 32 | query heads |
| `num_key_value_heads` | 8 | GQA KV heads |
| `head_dim` | `hidden_size // num_attention_heads` | 128 by default; not an explicit config default |
| `intermediate_size` | 14336 | dense and expert FFN hidden width |
| `hidden_act` | `silu` | SwiGLU and Mamba gate activation |
| `max_position_embeddings` | 262144 | mask/cache position range, not a RoPE table in Jamba source |
| `attn_layer_period` / `offset` | 8 / 4 | attention at layer indices `i % 8 == 4` |
| `expert_layer_period` / `offset` | 2 / 1 | MoE at layer indices `i % 2 == 1` |
| `num_experts` | 16 | expert count for MoE layers |
| `num_experts_per_tok` | 2 | top-k routing |
| `mamba_expand` | 2 | Mamba inner channels `E = 2 * hidden_size` |
| `mamba_d_state` | 16 | SSM recurrent state width |
| `mamba_d_conv` | 4 | depthwise causal conv kernel/cache width |
| `mamba_dt_rank` | `ceil(hidden_size / 16)` if `"auto"` | dt projection rank |
| `mamba_conv_bias` | true | Conv1d bias present |
| `mamba_proj_bias` | false | Mamba linear projection biases absent by default |
| `rms_norm_eps` | `1e-6` | RMSNorm epsilon |
| `tie_word_embeddings` | false | LM head is separate by default |
| `use_cache` | true | hybrid attention + Mamba cache |
| `use_mamba_kernels` | true | prefers optional `mamba-ssm` and `causal-conv1d` kernels |

Checkpoint sweep:

| Checkpoint | Source | Hidden | Layers | Attn schedule | Mamba layers | Attention layers | MoE schedule | Experts/top-k | KV heads | Tie embeddings | Notes |
|---|---|---:|---:|---|---:|---:|---|---:|---:|---|---|
| `ai21labs/Jamba-tiny-random` | config | 128 | 8 | `i % 4 == 2` | 6 | 2 | `i % 2 == 1` | 4 / 2 | 1 | false | `use_mamba_kernels=false`; random debug |
| `ai21labs/Jamba-tiny-dev` | config | 512 | 16 | `i % 8 == 4` | 14 | 2 | `i % 2 == 1` | 8 / 2 | 2 | false | small valid dev model |
| `TechxGenus/Mini-Jamba-v2` | config | 256 | 16 | `i % 3 == 1` | 11 | 5 | `i % 2 == 1` | 8 / 2 | 8 | false | community config has legacy fields current source ignores |
| `ai21labs/Jamba-v0.1` | config | 4096 | 32 | `i % 8 == 4` | 28 | 4 | `i % 2 == 1` | 16 / 2 | 8 | false | official open primary reference |
| `ai21labs/AI21-Jamba-Reasoning-3B` | config | 2560 | 28 | `i % 14 == 7` | 26 | 2 | no MoE effectively | 1 / 1 | 1 | true | MQA attention; dense FFN only |
| `ai21labs/AI21-Jamba2-Mini-FP8` | config | 4096 | 32 | `i % 8 == 4` | 28 | 4 | `i % 2 == 1` | 16 / 2 | 8 | false | compressed-tensors FP8 metadata outside core module graph |

For default `Jamba-v0.1`, attention layers are indices `4, 12, 20, 28`. MoE layers are all odd indices, so the default attention layers use dense MLPs and all 16 MoE layers are Mamba layers. Other configs can overlap attention and MoE, for example `TechxGenus/Mini-Jamba-v2` has attention indices `1,4,7,10,13`, with MoE on `1,7,13`.

## 3a. Family variation traps

- Layer type is computed, not listed in most configs. DinoML must materialize `layers_block_type` and `layers_num_experts` from period/offset fields.
- Attention and MoE schedules are independent. Do not assume attention layers are dense or Mamba layers are MoE.
- `num_key_value_heads` can be less than, equal to, or much less than `num_attention_heads`; configs include GQA and MQA.
- Current source uses no attention projection biases; Mamba conv bias is separate from Mamba projection bias.
- `hidden_size == num_attention_heads * head_dim` by source default, but `head_dim` can be explicit in inherited attention conventions. Validate divisibility.
- `sliding_window` appears in configs but current Jamba source only creates full attention or Mamba cache types through `layer_types`; no Jamba sliding-window attention path is implemented here.
- Generated `modeling_jamba.py` contains rotary helper code, but `JambaAttention.forward` does not apply RoPE. `position_ids` feed causal mask/cache position logic, not attention rotation.
- Legacy config fields such as `n_ctx`, `mamba_inner_layernorms`, and `calc_logits_for_entire_prompt` occur in some community configs but are not consumed by the inspected in-library source.
- `AI21-Jamba2-Mini-FP8` has `quantization_config`; that affects weight loading/quantized linear implementations outside core Jamba modeling code.
- `tie_word_embeddings=true` appears in `AI21-Jamba-Reasoning-3B`; weight aliasing must be preserved when set.

## 4. Operator coverage checklist

Tensor/layout ops:

- Token embedding lookup `[B, S] -> [B, S, H]`.
- Reshape/view/transposes for attention `[B,S,H] -> [B,heads,S,D]`, Mamba `[B,S,E] <-> [B,E,S]`, and MoE flatten `[B,S,H] -> [B*S,H]`.
- `chunk`/split for Mamba in-proj and SwiGLU gate/up projections.
- Padding for Mamba conv state prefill: left pad along sequence/channel-last-to-conv layout.
- `where`, `one_hot`, `nonzero`, `index_select`/advanced indexing, `index_add_` for eager MoE routing.
- `topk`, `softmax`, dtype casts, reductions over expert masks if router logits are requested.

Neural network primitives:

- Bias-free GEMM/linear: attention Q/K/V/O, dense MLP gate/up/down, Mamba `in_proj`, `x_proj`, `out_proj`.
- Mamba `dt_proj` has bias and a special fast-path behavior: in CUDA-kernel path the matmul is performed with bias temporarily zeroed and the bias is passed separately to selective scan.
- LM head `Linear(H -> vocab_size, bias=False)`, optionally tied to embeddings.
- RMSNorm over last dim with fp32 variance and output cast back to input dtype.
- SiLU activation and multiply for SwiGLU.
- Depthwise grouped Conv1d for Mamba with `groups=E`, `kernel=mamba_d_conv`, left padding, optional bias.

Attention primitives:

- Causal self-attention only.
- GQA/MQA repeat of KV heads for eager path.
- Backend-dispatched attention via `ALL_ATTENTION_FUNCTIONS`, with eager fallback, SDPA, and FlashAttention support flags.
- Dynamic KV cache update for attention layers.

Position/cache/generation ops:

- Causal mask from `create_causal_mask`.
- Dynamic position IDs based on attention cache sequence length.
- Hybrid cache layers: full-attention KV layers and Mamba linear-attention layers.
- Mamba cache update for conv states `[B,E,Kconv]` and recurrent states `[B,E,Dstate]`.
- `logits_to_keep` slicing before LM head.

Preprocessing-coupled ops:

- Tokenizer/chat template are outside the module graph.
- Attention mask `[B,S]` enters both causal mask construction and Mamba padding mask logic.

## 5. Layer/block breakdown

Let `H=hidden_size`, `A=num_attention_heads`, `KV=num_key_value_heads`, `D=H/A`, `E=mamba_expand*H`, `I=intermediate_size`, `R=mamba_dt_rank`, `N=mamba_d_state`, `K=mamba_d_conv`, `T=B*S`.

Common decoder residual frame:

```text
residual = x
x = RMSNorm(H)(x)
x = sequence_mixer(x, mask, cache)
x = residual + x
residual = x
x = RMSNorm(H)(x)
x = feed_forward(x)
x = residual + x
```

Attention mixer:

```text
q = Linear(H -> A*D, bias=False)(x).view(B,S,A,D).transpose(1,2)
k = Linear(H -> KV*D, bias=False)(x).view(B,S,KV,D).transpose(1,2)
v = Linear(H -> KV*D, bias=False)(x).view(B,S,KV,D).transpose(1,2)
k, v = cache.update(k, v, layer_idx) when cache is present
attn = causal_attention(q, k, v, mask, scale=D**-0.5)
y = Linear(A*D -> H, bias=False)(attn.transpose/reshape)
```

Dense FFN:

```text
gate = Linear(H -> I, bias=False)(x)
up = Linear(H -> I, bias=False)(x)
y = Linear(I -> H, bias=False)(silu(gate) * up)
```

MoE FFN:

```text
router_logits = Linear(H -> num_experts, bias=False)(x.view(T,H))
weights = softmax(router_logits, dim=-1, dtype=float32)
top_k_weights, top_k_index = topk(weights, k=num_experts_per_tok)
for each hit expert:
  selected = hidden[token_idx]
  gate, up = Linear(H -> 2*I, expert_weight[expert]).chunk(2)
  expert_y = Linear(I -> H, expert_down[expert])(silu(gate) * up)
  final.index_add_(token_idx, expert_y * selected_top_k_weight)
y = final.view(B,S,H)
```

MoE expert storage layout is packed 3D tensors:

```text
gate_up_proj: [num_experts, 2*intermediate_size, hidden_size]
down_proj:    [num_experts, hidden_size, intermediate_size]
```

Mamba mixer:

```text
projected = Linear(H -> 2*E, bias=mamba_proj_bias)(x).transpose(1,2)
hidden, gate = projected.chunk(2, dim=1)  # [B,E,S]
hidden *= attention_mask[:,None,:] when needed
hidden = depthwise_causal_conv1d(hidden, weight=[E,K], bias?, activation=silu)
hidden *= attention_mask[:,None,:] when needed
params = Linear(E -> R + 2*N, bias=False)(hidden.transpose(1,2))
dt, Bpar, Cpar = split(params, [R,N,N])
dt = RMSNorm(R)(dt); Bpar = RMSNorm(N)(Bpar); Cpar = RMSNorm(N)(Cpar)
dt_projected = Linear(R -> E, bias=True)(dt)
A = -exp(A_log.float())  # [E,N]
scan = selective_scan/update(hidden, dt_projected, A, Bpar, Cpar, D, gate, dt_bias)
y = Linear(E -> H, bias=mamba_proj_bias)(scan.transpose(1,2))
```

## 6. Attention requirements

Jamba attention is causal self-attention only, present only on layers where `i % attn_layer_period == attn_layer_offset`.

Requirements:

- MHA/GQA/MQA depending on `num_key_value_heads`.
- Query shape before attention: `[B, A, Sq, D]`.
- Cached key/value shape before KV repetition: `[B, KV, Skv, D]`.
- Eager path repeats KV to `[B, A, Skv, D]` using `num_key_value_groups = A // KV`.
- Attention math order in eager path: QK matmul, multiply by `D**-0.5`, add mask, fp32 softmax, cast to query dtype, dropout, AV matmul.
- Dropout is zero during inference.
- Source advertises FlashAttention and SDPA support, but routing is through `ALL_ATTENTION_FUNCTIONS`.
- No RoPE is applied by `JambaAttention.forward` in the inspected generated source.
- No sliding-window/local attention is implemented for native Jamba despite `sliding_window: null` fields in configs.

Cache distinction:

- Attention layers use autoregressive KV cache: keys and values grow with sequence length.
- Mamba layers use conv/recurrent state cache, not KV cache.
- `DynamicCache(config=config)` builds layer cache classes from `config.layer_types`: `"full_attention"` for attention layers and `"mamba"` for Mamba layers.
- `past_key_values.get_seq_length()` is used to offset generated `position_ids`; this sequence length is attention-cache driven.

Decode behavior:

- With no cache, prefill processes the full sequence and initializes both attention KV and Mamba states when `use_cache=True`.
- In one-token decode, attention appends one KV position on attention layers.
- In one-token decode on Mamba layers, `cache.has_previous_state(layer_idx)` selects `causal_conv1d_update` plus `selective_state_update` in the fast path, or rolling conv state plus one-step recurrence in slow path.
- Mamba padding masks are disabled when any previous Mamba state exists or when the input attention mask is all ones.

## 7. Position encoding and custom math

The current in-library Jamba source has no explicit learned position embeddings, ALiBi, or applied RoPE in the attention forward path. `position_ids` exist to drive `create_causal_mask` and cache-aware sequence positions.

Mamba custom math is the main nonstandard runtime requirement. Slow-path selective scan parity can be summarized as:

```python
def jamba_mamba_slow_scan(u, gate, dt, b, c, a_log, d, dt_bias):
    # u/gate: [B, E, S], dt after dt_proj+softplus: [B, E, S]
    A = -torch.exp(a_log.float())              # [E, N]
    state = zeros([B, E, N])
    ys = []
    for t in range(S):
        discrete_A = torch.exp(A[None, :, :] * dt[:, :, t, None])
        discrete_B = dt[:, :, t, None] * b[:, t, None, :].float()
        state = discrete_A * state + discrete_B * u[:, :, t, None].float()
        y = torch.matmul(state.to(u.dtype), c[:, t, :, None])[:, :, 0]
        ys.append(y)
    y = torch.stack(ys, dim=-1)
    return (y + u * d[None, :, None]) * silu(gate)
```

Precomputable:

- `A = -exp(A_log.float())` per layer.
- Conv weights flattened to `[E,K]`.

Dynamic/input-dependent:

- `dt`, `B`, `C` from `x_proj(hidden)`.
- Conv and recurrent cache states.
- Attention and Mamba masks.

## 8. Preprocessing and input packing

Runtime graph inputs are standard text generation tensors:

- `input_ids: [B,S]` or `inputs_embeds: [B,S,H]`, exactly one required.
- `attention_mask: [B,S]` optional. Used by causal mask construction and by Mamba masking before/after convolution unless disabled.
- `position_ids: [1,S]` or compatible optional. If absent, generated from current cache length plus token offsets.
- `past_key_values` optional hybrid cache object.

There is no multimodal packing, image/audio preprocessing, placeholder scatter, or packed varlen metadata in this family.

Generation-controller behavior outside core graph:

- `logits_to_keep` is model-forward visible and important for efficient prefill/decode logits.
- Chat template/tool/JSON behavior belongs to tokenizer/model-card usage, not the core Jamba module graph.
- Training-only labels and router auxiliary loss can be deferred for inference.

## 9. Graph rewrite / lowering opportunities

### Rewrite: attention GQA to fused causal attention

Source pattern:

```text
q/k/v Linear -> view/transpose -> cache.update -> repeat_kv -> scaled causal attention -> transpose/reshape -> o_proj
```

Replacement:

```text
QKV linear family -> cache append -> fused GQA causal attention backend -> output linear
```

Preconditions:

- `num_attention_heads % num_key_value_heads == 0`.
- No RoPE transform is required for current Jamba source.
- Mask is causal plus optional padding in source-compatible additive form.
- Dropout disabled for inference.

Shape equations:

- `D = H / A`.
- KV cache stores `[B, KV, S, D]`, not repeated `[B, A, S, D]`.

Failure cases:

- Unexpected `_attn_implementation` with incompatible mask semantics.
- Configs adding position transforms outside current native source.

Parity sketch:

- Compare eager attention, SDPA, and fused backend for random prompt and one-token decode with GQA/MQA.

### Rewrite: Mamba depthwise Conv1d plus selective scan to provider op

Source pattern:

```text
in_proj -> chunk -> optional mask -> depthwise causal conv+silu -> x_proj -> three RMSNorms -> dt_proj -> selective_scan/update -> out_proj
```

Replacement:

```text
MambaBlock(H,E,R,N,K, biases) provider op with explicit conv_state and recurrent_state IO
```

Preconditions:

- `groups == E`, `kernel_size == mamba_d_conv`, `padding == K-1`.
- Activation is `silu`.
- `mamba_inner_layernorms` behavior is present in source as dt/B/C RMSNorm; configs that disable it are not honored by current source.
- Cache state shapes are explicit and per layer.

Weight transform:

```python
conv_w = conv1d.weight.view(E, K)
A = -torch.exp(A_log.float())
```

Failure cases:

- Optional kernels unavailable: source falls back to slow PyTorch recurrence; DinoML should not.
- Quantized linear replacement around `dt_proj` requires preserving the "bias passed separately" fast-path semantics.

Parity sketch:

- Single Mamba layer prefill for `S > K`, short `S < K`, and decode step after prefill.

### Rewrite: MoE eager routing to grouped expert GEMM

Source pattern:

```text
router linear -> fp32 softmax -> topk -> one_hot/where per expert -> expert gate_up GEMM -> silu*up -> down GEMM -> weighted index_add
```

Replacement:

```text
TopKRouter -> token permutation/grouping -> grouped GEMM gate_up -> activation multiply -> grouped GEMM down -> weighted scatter-add
```

Preconditions:

- Inference only, no router auxiliary loss required.
- Top-k weights are not renormalized after `topk`; preserve source behavior.
- No dropped tokens; full-capacity routing.
- Expert tensors use `[expert, out, in]` layout.

Failure cases:

- Very small batches where eager per-expert loop is acceptable but grouping overhead dominates.
- Config `num_experts=1` should canonicalize to dense MLP or bypass MoE path.

Parity sketch:

- Fixed router logits with ties avoided; compare token-level outputs and final scatter accumulation for top-k=1 and top-k=2.

### Rewrite: dense SwiGLU FFN to fused GEMM epilogue

Source pattern:

```text
gate_proj(x), up_proj(x), silu(gate) * up, down_proj
```

Replacement:

```text
dual GEMM or packed gate/up GEMM -> fused SiLU multiply -> down GEMM
```

Preconditions:

- Bias-free projections.
- Activation exactly SiLU.
- Weight packing preserves gate/up order.

Parity sketch:

- Dense MLP layer random tensor tests across fp32/bf16.

## 10. Kernel fusion candidates

Highest priority:

- Mamba provider op with prefill scan and decode update. This is the largest architecture-specific gap and avoids O(S) Python/PyTorch loop fallback.
- GQA/MQA causal attention with KV cache. Attention layers are sparse in default Jamba, but decode still depends on them.
- MoE routing plus grouped expert GEMMs. Default large Jamba has many MoE layers, and eager `where`/per-expert loops are not production shaped.
- RMSNorm. Used twice per layer plus three inner Mamba norms.

Medium priority:

- Fused dense SwiGLU and expert SwiGLU.
- Mamba depthwise causal conv update kernel if not folded into full Mamba provider.
- Last-token-only LM head for prefill/decode via `logits_to_keep`.
- Attention Q/K/V projection packing when weights are available as separate matrices.

Lower priority:

- Router auxiliary loss and router logits capture.
- Classification head.
- Beam cache reorder for hybrid caches.
- Quantized compressed-tensors loading for FP8 checkpoints, after dense parity.

## 11. Runtime staging plan

Stage 1: config and schedule loader.

- Parse period/offset schedules into explicit layer manifests.
- Reject unsupported legacy fields that current source ignores only if they would imply changed behavior.

Stage 2: one dense block parity.

- Run attention+dense and Mamba+dense blocks without cache.
- Use tiny/dev configs first.

Stage 3: Mamba prefill parity.

- Implement depthwise causal conv, dt/B/C projections and norms, selective scan, and recurrent state output.
- Validate masks and short-prompt padding.

Stage 4: attention prefill parity.

- Implement GQA causal attention with source mask/scaling/softmax order.
- Populate KV cache only on attention layers.

Stage 5: hybrid decode.

- Add explicit cache object with two state kinds: attention KV and Mamba conv/recurrent state.
- Validate one-token decode after prefill across mixed layer schedules.

Stage 6: MoE inference.

- Start with eager but graph-visible top-k routing and per-expert GEMMs.
- Then lower to grouped expert GEMM/permutation/scatter.

Stage 7: optimized production path.

- Fuse Mamba provider, attention backend, RMSNorm, SwiGLU, grouped MoE, and last-token logits.
- Add cache memory accounting and batching probes.

Initially stubbable:

- Training loss, router auxiliary loss, output attentions, output hidden states, sequence classification, beam cache reorder, quantized checkpoint loaders.

## 12. Parity and validation plan

- Config schedule tests: compute layer type and expert count arrays for each representative config.
- RMSNorm tests: fp32 and bf16 against Transformers with tolerances around `1e-5` fp32 and `1e-2` bf16.
- Dense SwiGLU tests: random `[B,S,H]` against `JambaMLP`.
- MoE tests: fixed small `num_experts`, deterministic router logits, top-k=1/2, compare final scatter-add.
- Mamba unit tests: one layer, `use_mamba_kernels=false`, no mask/all-ones mask/padded mask, prefill and decode-after-prefill.
- Attention tests: GQA and MQA random tensors, eager fallback parity, cache append shape and output parity.
- Hybrid block tests: one attention layer and one Mamba layer with both dense and MoE FFNs.
- Full tiny model parity: `ai21labs/Jamba-tiny-random` and `ai21labs/Jamba-tiny-dev` prefill logits.
- Decode parity: prompt prefill plus 1, 2, and N token decode with exact same sampled/argmax tokens.
- Logits slicing parity: `logits_to_keep=1`, `0`, and tensor indices.

Recommended tolerances:

- fp32 block-level: `rtol=1e-4`, `atol=1e-4`.
- bf16/fp16 block-level: start `rtol=2e-2`, `atol=2e-2`; tighten per fused op when accumulation order is matched.
- Full-model logits: compare relative/top-k agreement before demanding strict elementwise parity for fused Mamba/attention.

## 13. Performance probes

- Config schedule probe: attention/Mamba/MoE layer counts and active parameter slices.
- Mamba prefill throughput by sequence length: 1K, 8K, 32K, 128K, 256K where memory allows.
- Mamba decode tokens/sec with conv/recurrent state update only.
- Attention prefill throughput on sparse attention layers, with GQA/MQA variants.
- Attention KV cache memory per layer: `2 * B * KV * S * D * dtype_size`.
- Mamba state memory per layer: `B * E * K * dtype_size + B * E * N * dtype_size`.
- MoE routing overhead versus grouped GEMM time at batch/token counts.
- Last-token LM head versus full-prompt LM head.
- End-to-end prefill/decode split for tiny/dev and one production-shaped synthetic config.
- Quantized checkpoint load/dequant cost for FP8 or compressed-tensors variants, labeled separately from dense graph runtime.

## 14. Skip/defer list

- Training labels/loss and router auxiliary loss.
- `output_router_logits`, `output_attentions`, and full hidden-state recording.
- Sequence classification head.
- Beam search cache reorder and speculative decoding.
- Multi-GPU tensor parallel plans from `_tp_plan`/`_pp_plan`.
- Compressed-tensors/FP8 quantized loading and bitsandbytes-specific paths.
- Remote-code-only or historical config behavior not present in current in-library source.
- Chat template, tool-use, JSON-mode, and grounded-generation controller behavior beyond tokenizer/prompt construction.

## 15. Final implementation checklist

- [ ] Parse `JambaConfig` and materialize explicit layer type/expert schedules.
- [ ] Load embedding, LM head, attention, Mamba, dense MLP, and expert weights with alias handling for tied embeddings.
- [ ] Implement RMSNorm with fp32 variance.
- [ ] Implement dense SwiGLU MLP.
- [ ] Implement GQA/MQA causal attention with `[B, KV, S, D]` cache storage.
- [ ] Implement hybrid cache manifest with distinct attention KV and Mamba conv/recurrent state slots.
- [ ] Implement Mamba depthwise causal conv prefill and decode update.
- [ ] Implement Mamba selective scan prefill and selective state decode update.
- [ ] Preserve `dt_proj` bias semantics for fast-path selective scan.
- [ ] Implement MoE router softmax/top-k without post-top-k renormalization.
- [ ] Implement expert grouped GEMM or an initially correct per-expert fallback.
- [ ] Implement `logits_to_keep` LM-head slicing.
- [ ] Add one-block parity tests for attention, Mamba, dense MLP, and MoE.
- [ ] Add tiny/dev full-model prefill parity.
- [ ] Add decode parity with hybrid cache state inspection.
- [ ] Benchmark Mamba prefill/decode, attention prefill/decode, MoE routing, and LM-head slicing separately.
