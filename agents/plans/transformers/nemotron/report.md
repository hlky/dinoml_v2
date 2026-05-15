# Transformers Nemotron / Nemotron-H audit for DinoML

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: nemotron and nemotron_h families
Config source: Hugging Face Hub config.json snapshots saved beside this report
Primary runtime target: causal LM prefill/decode
DinoML assumptions: inference-only first, CUDA GPU target, faithful PyTorch axes first, then guarded GEMM/attention/norm/state fusions.
```

Source files inspected:

- `transformers/src/transformers/models/nemotron/configuration_nemotron.py`
- `transformers/src/transformers/models/nemotron/modeling_nemotron.py`
- `transformers/src/transformers/models/nemotron/convert_nemotron_nemo_to_hf.py`
- `transformers/src/transformers/models/nemotron_h/configuration_nemotron_h.py`
- `transformers/src/transformers/models/nemotron_h/modular_nemotron_h.py`
- `transformers/src/transformers/models/nemotron_h/modeling_nemotron_h.py`
- Related owners for inherited behavior: `models/zamba2/modeling_zamba2.py`, `models/deepseek_v3/modeling_deepseek_v3.py`, and `cache_utils.py`.

Pinned source URLs:

- https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/nemotron/configuration_nemotron.py
- https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/nemotron/modeling_nemotron.py
- https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/nemotron_h/configuration_nemotron_h.py
- https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/nemotron_h/modular_nemotron_h.py
- https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/nemotron_h/modeling_nemotron_h.py

Representative config URLs:

- https://huggingface.co/thhaus/nemotron3-8b/resolve/main/config.json
- https://huggingface.co/nvidia/Nemotron-Mini-4B-Instruct/resolve/main/config.json
- https://huggingface.co/nvidia/Nemotron-4-Mini-Hindi-4B-Instruct/resolve/main/config.json
- https://huggingface.co/nvidia/Nemotron-H-4B-Base-8K/resolve/main/config.json
- https://huggingface.co/nvidia/Nemotron-H-8B-Reasoning-128K/resolve/main/config.json
- https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-12B-v2/resolve/main/config.json
- https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/resolve/main/config.json
- https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16/resolve/main/config.json

`nemotron_h/modeling_nemotron_h.py` is generated from `modular_nemotron_h.py`; the modular file is authoritative for source edits, while the generated file shows the expanded Mamba fallback implementation used at runtime.

Config snapshots saved:

- `config_thhaus_nemotron3_8b.json`
- `config_nvidia_nemotron_mini_4b_instruct.json`
- `config_nvidia_nemotron_4_mini_hindi_4b_instruct.json`
- `config_nvidia_nemotron_h_4b_base_8k.json`
- `config_nvidia_nemotron_h_8b_reasoning_128k.json`
- `config_nvidia_nemotron_nano_12b_v2.json`
- `config_nvidia_nemotron_3_nano_30b_a3b_bf16.json`
- `config_nvidia_nemotron_3_nano_30b_a3b_base_bf16.json`

No gated config fetch failed. Some older official Nemotron-4 340B repositories expose NeMo `model_config.yaml` rather than an in-library HF `config.json`; those should be treated as a separate NeMo conversion/loading audit, not as direct `NemotronConfig` checkpoints.

## 2. High-level architecture

There are two materially different families:

```text
nemotron:
token ids -> embedding -> N dense decoder blocks -> final LayerNorm1P -> LM head -> logits

nemotron_h:
token ids -> embedding -> layer-type schedule of Mamba / attention / MoE / MLP blocks -> RMSNorm -> LM head -> logits
```

`nemotron` is Llama-like causal decoder inference with separate Q/K/V projections, partial RoPE, GQA/MHA, additive causal masks, KV cache, a non-gated `relu2` MLP, and a custom `LayerNorm1P` that applies `weight + 1`.

`nemotron_h` is a hybrid stateful decoder. Each layer has exactly one mixer selected by `layers_block_type` or legacy `hybrid_override_pattern`: Mamba2, dense causal attention, MoE, or plain MLP. The first useful DinoML split is:

```text
CPU/tokenizer -> embedding
prefill:
  attention blocks build KV cache
  mamba blocks build conv/recurrent state
  MoE blocks route tokens to experts
decode:
  attention blocks append/use KV
  mamba blocks update fixed-size state
  MoE blocks route current tokens
last-token LM head -> sampling controller
```

The dense `nemotron` path can be staged independently from `nemotron_h`. Do not treat `nemotron_h` as just a decoder with a KV cache; its Mamba layers require fixed-size recurrent/session state.

## 3. Important config dimensions

Source defaults:

| Family | hidden | layers | heads | KV heads | head_dim | MLP/intermediate | vocab | max positions | activation | cache |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `NemotronConfig` default | 6144 | 32 | 48 | defaults to heads | `hidden/heads` | 24576 | 256000 | 4096 | `relu2` | KV |
| `NemotronHConfig` default | 4096 | derived from `layers_block_type` | 32 | 8 | 128 | 21504 | 131072 | 4096 | `relu2` MLP, `silu` Mamba | KV + Mamba state |

Representative checkpoint sweep:

| Checkpoint | type | hidden | layers / schedule | heads / KV | head_dim | MLP / MoE | max pos | notable operator surface |
|---|---|---:|---|---|---:|---|---:|---|
| `thhaus/nemotron3-8b` | `nemotron` | 4096 | 32 dense | 32 / 32 | 128 | 16384 | 4096 | MHA, partial RoPE 0.5, dense `relu2` MLP |
| `nvidia/Nemotron-Mini-4B-Instruct` | `nemotron` | 3072 | 32 dense | 24 / 8 | config omits; effective 128 | 9216 | 4096 | GQA, partial RoPE 0.5 |
| `nvidia/Nemotron-4-Mini-Hindi-4B-Instruct` | `nemotron` | 3072 | 32 dense | 24 / 8 | config omits; effective 128 | 9216 | 4096 | same graph as Mini 4B, different EOS |
| `nvidia/Nemotron-H-4B-Base-8K` | `nemotron_h` | 3072 | 52: 24 Mamba, 4 attention, 24 MLP | 32 / 8 | config uses `attention_head_dim=128`; current source default would use `head_dim=128` | MLP 12288 | 8192 | no MoE, hybrid Mamba/attention/MLP |
| `nvidia/Nemotron-H-8B-Reasoning-128K` | `nemotron_h` | 4096 | 52: 24 Mamba, 4 attention, 24 MLP | 32 / 8 | source default 128 | MLP 21504 | 131072 | long context hybrid, no MoE |
| `nvidia/NVIDIA-Nemotron-Nano-12B-v2` | `nemotron_h` | 5120 | 62: 28 Mamba, 6 attention, 28 MLP | 40 / 8 | 128 | MLP 20480 | 131072 | larger hybrid, Mamba head_dim 80 |
| `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` | `nemotron_h` | 2688 | 52: 23 Mamba, 23 MoE, 6 attention | 32 / 2 | 128 | MoE 128 experts, top-6, shared expert | 262144 | hybrid Mamba + sparse MoE + GQA |

Several HF configs carry legacy field names. Current `NemotronHConfig` maps `hybrid_override_pattern` to `layers_block_type` and maps M/E/*/- to Mamba/MoE/attention/MLP. Some older configs use `attention_head_dim` rather than `head_dim`; DinoML config parsing should either normalize that alias or reject with a clear message after confirming Transformers behavior for the target version.

## 3a. Family variation traps

- `nemotron` and `nemotron_h` are separate runtime targets despite related names.
- `NemotronConfig.head_dim` may be omitted; source computes `hidden_size // num_attention_heads`.
- `num_key_value_heads` may be smaller than query heads; GQA repeat is required.
- `NemotronConfig` uses partial RoPE by default (`partial_rotary_factor=0.5`), so only a prefix of Q/K head_dim is rotated.
- `NemotronLayerNorm1P` is not standard LayerNorm weight use: source passes `self.weight + 1` to `F.layer_norm`.
- MLP is non-gated: `down_proj(act(up_proj(x)))`, not SwiGLU.
- `relu2` is operator-significant; do not replace with ordinary ReLU.
- `nemotron_h` layer count is the length of the layer schedule, not a free-standing `num_hidden_layers`.
- `nemotron_h` Mamba layers are stateful and require conv state plus recurrent state; they cannot be represented by a KV-cache-only ABI.
- `nemotron_h` MoE experts are non-gated two-matrix MLP experts stored as 3D tensors, with optional latent projection and a shared expert.
- Current `NemotronHAttention` generated source does not apply local RoPE itself; it delegates attention backend selection through `ALL_ATTENTION_FUNCTIONS` and cache update.
- `sliding_window` exists in config but inspected source does not set a per-layer sliding-window attention path beyond generic mask/config handling. Treat non-null sliding window as requiring source-specific confirmation.
- `num_nextn_predict_layers` and MTP config fields exist, but current model ignores unexpected `mtp.*` keys on load and no MTP head is implemented in the inspected `NemotronHForCausalLM`. Defer/reject MTP.
- `auto_map` in configs may point to remote code names. The audited scope is the in-library source at the pinned commit.

## 4. Operator coverage checklist

Tensor/layout ops:

- Embedding lookup for token ids.
- `view`, `reshape`, `transpose`, `contiguous`, `expand`, split, concat, slice.
- `repeat_kv`: `[B, KVH, S, D] -> [B, QH, S, D]` by expand/reshape.
- Last-token or selected-index logits slicing via `logits_to_keep`.

Neural network primitives:

- Dense `Linear(H -> QH*D)`, `Linear(H -> KVH*D)`, `Linear(QH*D -> H)`.
- GEMM-backed MLP: `Linear(H -> I) -> relu2 -> Linear(I -> H)`.
- `NemotronLayerNorm1P`: layer norm over last dim using effective gamma `weight + 1` and bias.
- RMSNorm for `nemotron_h`.
- `Zamba2RMSNormGated` in Mamba layers: gated RMSNorm over Mamba intermediate states.
- Depthwise causal `Conv1d` for Mamba: groups equal channels, kernel `conv_kernel`, padding `kernel-1`, crop to sequence length.

Attention primitives:

- Causal self-attention, MHA/GQA.
- Additive causal/padding masks from `create_causal_mask`.
- Eager matmul-softmax-matmul with fp32 softmax accumulation.
- SDPA/FlashAttention-compatible path for optimized lowering.
- KV cache update before KV repeat in dense `nemotron`; cache stores un-repeated KV heads.

Position/rotary ops:

- RoPE cos/sin generation from `rope_theta`, `head_dim`, and `partial_rotary_factor`.
- Partial RoPE split/cat for dense `nemotron`.
- Dynamic RoPE variants if `rope_parameters["rope_type"] != "default"` appears in configs.

Generation/cache ops:

- Dynamic KV cache per attention layer: K/V shape `[B, num_key_value_heads, S, head_dim]`.
- Mamba cache per Mamba layer: conv state `[B, conv_dim, conv_kernel]` and recurrent state shaped from `[B, num_heads, head_dim, ssm_state_size]`.
- Beam/cache reorder must handle both KV and linear-attention state.
- `num_logits_to_keep` default of 1 in `nemotron_h` generation preparation.

MoE ops for `nemotron_h` 30B A3B:

- Router `F.linear(hidden.float(), router_weight.float())`.
- Add `e_score_correction_bias` for expert choice.
- Group top-k, expert top-k, gather router weights, optional normalization, scaling.
- Token-to-expert dispatch, expert GEMMs from 3D expert weights, weighted accumulation via `index_add_`.
- Shared expert MLP added to routed expert result.

Mamba/state-space ops for `nemotron_h`:

- Packed input projection split into gate, conv/BC channels, and dt.
- `softplus(dt + dt_bias)`, clamp/limit behavior.
- `A = -exp(A_log.float())`, D residual/skip.
- Chunked scan fallback: padding to `chunk_size`, lower-triangular masks, `cumsum`, `exp`, einsum/matmul-like contractions.
- Optional external fast kernels: `causal-conv1d` and `mamba-ssm`.

Quantized/packed weight metadata:

- No source-coupled quantized weight format is required by the inspected modeling files. Separate GGUF/FP8 checkpoints should be treated as loading/provider contracts with dense fallback.

## 5. Layer/block breakdown

Dense `NemotronDecoderLayer`, repeated `num_hidden_layers`:

```text
residual = x
x = LayerNorm1P(x)
q = Linear(H -> num_heads * head_dim, bias=attention_bias)
k = Linear(H -> num_key_value_heads * head_dim, bias=attention_bias)
v = Linear(H -> num_key_value_heads * head_dim, bias=attention_bias)
q,k,v -> [B, heads, S, head_dim]
q,k = partial_rope(q,k, cos, sin)
k,v = cache.update(k,v) when cache is present
k,v = repeat_kv(k,v, num_heads / num_key_value_heads)
attn = softmax((q @ k.T) / sqrt(head_dim) + mask, fp32)
x = residual + Linear(attn @ v -> H, bias=attention_bias)
residual = x
x = LayerNorm1P(x)
x = residual + Linear(I -> H)(relu2(Linear(H -> I)(x)))
```

`NemotronHBlock`, repeated according to `layers_block_type`:

```text
residual = x
x = RMSNorm(x)
if block_type == "attention":
  x = causal GQA attention mixer(x, KV cache, mask)
elif block_type == "mamba":
  x = Mamba2 mixer(x, conv/recurrent cache, mamba mask)
elif block_type == "moe":
  x = routed non-gated experts(x) + shared_expert(x)
elif block_type == "mlp":
  x = Linear(I -> H)(relu2(Linear(H -> I)(x)))
x = residual + x
```

For the 30B A3B config, MoE expert shapes are:

```text
router: [n_routed_experts=128, hidden=2688]
expert up_proj: [128, moe_intermediate_size=1856, input_dim]
expert down_proj: [128, input_dim, 1856]
shared expert: 2688 -> 3712 -> 2688
top_k experts per token: 6
```

## 6. Attention requirements

`nemotron` attention:

- Causal self-attention only for the primary CausalLM target.
- MHA when `num_key_value_heads == num_attention_heads`; GQA otherwise.
- Query width is `num_attention_heads * head_dim`; key/value width is `num_key_value_heads * head_dim`.
- Rotary is applied before cache update.
- Cached keys are stored after RoPE and before `repeat_kv`.
- Eager path uses additive mask before fp32 softmax. SDPA path slices `attention_mask[:, :, :, : key_states.shape[-2]]` and passes `is_causal=True` only when no explicit mask exists and `q_len > 1`.
- FlashAttention2 path rejects `StaticCache` in the inspected source.

`nemotron_h` attention:

- Only layers marked `"attention"` use dense causal attention.
- Query heads and KV heads are config-driven; representative configs use 32/8 or 32/2.
- The generated attention class projects Q/K/V, updates cache, then dispatches through `ALL_ATTENTION_FUNCTIONS`.
- `create_causal_mask` is built once per forward and passed only to attention blocks.
- Mamba and MoE layers do not consume attention masks except Mamba's separate padding mask logic.

Cache manifest for `nemotron_h`:

```text
attention layer:
  key:   [B, num_key_value_heads, cached_seq, head_dim]
  value: [B, num_key_value_heads, cached_seq, head_dim]

mamba layer:
  conv_state:      [B, conv_dim, conv_kernel]
  recurrent_state: [B, mamba_num_heads, mamba_head_dim, ssm_state_size]

moe/mlp layer:
  no persistent decode state
```

## 7. Position encoding and custom math

Dense `nemotron` default RoPE:

```python
def nemotron_rope_inv_freq(head_dim, partial_rotary_factor, rope_theta):
    dim = int(head_dim * partial_rotary_factor)
    return 1.0 / (rope_theta ** (torch.arange(0, dim, 2).float() / dim))

def partial_rope(q, k, cos, sin):
    rot_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rot_dim], q[..., rot_dim:]
    k_rot, k_pass = k[..., :rot_dim], k[..., rot_dim:]
    q_out = q_rot * cos + rotate_half(q_rot) * sin
    k_out = k_rot * cos + rotate_half(k_rot) * sin
    return cat(q_out, q_pass), cat(k_out, k_pass)
```

`NemotronRotaryEmbedding.forward` computes cos/sin in fp32 and casts to the hidden dtype. It supports non-default RoPE through `ROPE_INIT_FUNCTIONS` and `dynamic_rope_update`; DinoML can precompute default fixed tables for static/bucketed contexts, but dynamic RoPE types need separate admission.

Custom norm:

```python
def layer_norm_1p(x, weight, bias, eps):
    return layer_norm(x, gamma=weight + 1, beta=bias, eps=eps)
```

Mamba fallback math is too large to inline fully. Required source-derived primitives are depthwise causal conv, `softplus(dt + dt_bias)`, `A = -exp(A_log.float())`, chunk padding to `chunk_size`, lower-triangular segment sums, exponentials of cumulative A, B/C state contractions, D residual, gated RMSNorm, and output projection.

## 8. Preprocessing and input packing

The inspected model source is text-only. Tokenization and chat templates are outside the neural graph. Runtime graph inputs are:

```text
input_ids: [B, S] int64, or inputs_embeds: [B, S, H]
attention_mask: optional [B, S] style mask consumed by create_causal_mask and Mamba padding masking
position_ids: optional [B, S], otherwise arange plus cache length
past_key_values: optional cache/state object
```

No multimodal placeholder scatter, image/audio processor, packed varlen descriptor, or channel-layout conversion is present in the in-library `nemotron` or `nemotron_h` source. NVIDIA Omni or other remote-code Nemotron-branded multimodal repositories are out of scope for this report.

## 9. Graph rewrite / lowering opportunities

### Rewrite: separate Q/K/V projections -> packed QKV GEMM

Source pattern:

```text
q = Linear(x, q_weight)
k = Linear(x, k_weight)
v = Linear(x, v_weight)
```

Replacement: one GEMM with concatenated output, then split as Q, K, V.

Preconditions:

- Same input tensor and dtype.
- Same bias policy and compatible storage layout.
- Split order must be all-Q, all-K, all-V, because source modules are separate.
- Preserve different Q and KV widths.

Failure cases: quantized/packed external weight format, missing one projection, or custom remote code.

Parity test: compare per-projection outputs and post-reshape Q/K/V for random `B,S,H`.

### Rewrite: partial RoPE fused into attention prep

Source pattern: Q/K reshape, transpose, split rotary prefix, apply rotate-half, concat pass-through suffix.

Replacement: fused Q/K rotary kernel over `rot_dim = int(head_dim * partial_rotary_factor)`.

Preconditions:

- Default or explicitly supported RoPE type.
- Q/K shape `[B, heads, S, head_dim]`.
- Rotation prefix is contiguous in head dimension.

Failure cases: dynamic/non-default RoPE scaling not admitted, layout pass changes head dimension order.

### Rewrite: LayerNorm1P as standard LayerNorm with transformed gamma

Source pattern: `F.layer_norm(input, ..., self.weight + 1, self.bias, eps)`.

Replacement: materialize effective gamma `gamma_eff = gamma + 1` at load time or fuse `+1` in norm kernel.

Preconditions:

- Inference weights immutable.
- Preserve checkpoint parameter identity in metadata if round-tripping.

### Rewrite: non-gated MLP fusion

Source pattern: `down_proj(relu2(up_proj(x)))`.

Replacement: GEMM + fused `relu(x)^2` activation + GEMM.

Preconditions:

- `hidden_act == "relu2"`.
- Bias flags honored.

Failure cases: configs using unknown activation.

### Rewrite: Mamba prefill external provider

Source pattern: Mamba fallback with chunked scan and depthwise causal conv.

Replacement: provider-backed Mamba2 scan using `causal-conv1d` / `mamba-ssm` compatible ABI, with fallback disabled or separately validated.

Preconditions:

- Exact `conv_kernel`, `chunk_size`, `mamba_num_heads`, `mamba_head_dim`, `n_groups`, and `ssm_state_size`.
- Cache state ABI is explicit in artifact metadata.
- Padding mask semantics match source.

Failure cases: missing provider, non-silu activation, unsupported chunk size, external-stream hazards.

### Rewrite: MoE grouped expert GEMM

Source pattern: route top-k tokens, loop over hit experts, expert up/down linear, weighted `index_add_`.

Replacement: sort/group tokens by expert and use grouped GEMM plus scatter-add.

Preconditions:

- `has_gate=False` expert MLP.
- Stable top-k/gather semantics and top-k normalization match source.
- Token dispatch metadata shape is visible to the runtime.

Failure cases: dynamic token counts without allocation plan, unimplemented top-k tie parity, unsupported latent projection.

## 10. Kernel fusion candidates

Highest priority:

- `LayerNorm1P` and RMSNorm kernels; every block uses pre-norm and dense `nemotron` uses the custom `weight+1` contract.
- QKV projection + reshape + partial RoPE for `nemotron`.
- GQA FlashAttention/SDPA with cache update; dense `nemotron` becomes useful once this is solid.
- Mamba2 provider path for `nemotron_h`; eager fallback is graph-heavy and likely too slow.
- Last-token-only LM head using `logits_to_keep`, especially for decode.

Medium priority:

- `relu2` MLP fusion.
- Mamba depthwise causal conv + activation.
- Mamba decode state update kernel.
- MoE router top-k + grouped expert GEMM for 30B A3B.
- Shared expert MLP fusion in MoE blocks.

Lower priority:

- Classification / QA / token-classification heads for dense `nemotron`.
- Dynamic/non-default RoPE variants.
- Training-only dropout/loss paths.
- Remote-code multimodal Nemotron-branded models.

## 11. Runtime staging plan

Stage 1: dense `nemotron` config/weights. Implement `NemotronForCausalLM` with embeddings, LayerNorm1P, partial RoPE, GQA attention, `relu2` MLP, final norm, and LM head.

Stage 2: dense prefill parity. Use eager attention decomposition first, then route to DinoML fused attention once masks/cache are stable.

Stage 3: dense decode parity. Add KV cache update/reorder and last-token logits.

Stage 4: `nemotron_h` no-MoE hybrid skeleton. Support Mamba/attention/MLP schedules for 4B/8B/12B with explicit state manifest.

Stage 5: Mamba provider. Admit exact Mamba2 dimensions and chunk sizes; compare fallback math to provider output.

Stage 6: `nemotron_h` MoE. Add router/top-k/dispatch/grouped GEMM for 30B A3B; start with small token counts and dense fallback for experts.

Stage 7: production fusions. Enable packed QKV, fused RoPE attention, grouped expert GEMM, Mamba decode kernels, and continuous batching/cache management.

Can be stubbed initially: non-CausalLM heads, training loss, dropout, MTP fields, remote-code Omni/multimodal models, NeMo-only 340B conversion.

## 12. Parity and validation plan

- Unit test `LayerNorm1P` against PyTorch `F.layer_norm(x, gamma + 1, beta, eps)`.
- Unit test `relu2` activation and MLP block for fp32/bf16.
- Unit test partial RoPE for `partial_rotary_factor=0.5` and `1.0`.
- Single attention-layer parity for MHA and GQA, with and without cache.
- Dense `nemotron` one-block parity, then 2/4/all-layer hidden-state parity.
- Prefill logits parity for representative 4B/8B configs, `logits_to_keep=0`.
- Decode parity for one-token steps with cache length growth.
- `nemotron_h` Mamba fallback parity for short prefill and one-token decode, checking conv and recurrent state after each layer.
- MoE router parity: top-k indices, normalized top-k weights, expert accumulation output for controlled logits.
- Full `nemotron_h` layer-schedule parity on tiny synthetic configs before large checkpoints.

Recommended tolerances: fp32 `rtol=1e-4, atol=1e-5` for isolated ops; bf16/fp16 `rtol=2e-2, atol=2e-2` for full-block comparisons, tightened per kernel where accumulation is fp32.

## 13. Performance probes

- Dense `nemotron` prefill tokens/sec by sequence length: 128, 512, 4096.
- Dense decode tokens/sec with KV cache lengths: 128, 4096, 128K where supported.
- GQA attention backend comparison: eager GEMM/softmax, SDPA, FlashAttention.
- QKV packed GEMM versus three GEMMs.
- `LayerNorm1P` fused versus decomposed.
- Last-token logits versus full-sequence logits.
- `nemotron_h` per block-type timing: Mamba, attention, MLP, MoE.
- Mamba prefill chunk-size sweep: 128 and 256 from representative configs.
- Mamba decode state-update bandwidth and latency.
- MoE token routing distribution and grouped GEMM occupancy for top-2/top-6.
- Cache memory footprint split by KV cache and Mamba recurrent/conv state.
- Weight-load and dequant/provider comparison for FP8/GGUF variants if they become target checkpoints.

## 14. Skip/defer list

- Training, gradients, dropout behavior, and loss parity.
- Sequence/token/classification/QA heads for first CausalLM integration.
- StaticCache with FlashAttention2 for dense `nemotron`, because source rejects it.
- MTP / next-token prediction auxiliary layers in `nemotron_h`.
- Remote-code multimodal Nemotron Omni models.
- NeMo-only 340B checkpoints and conversion scripts.
- Tensor parallel / pipeline parallel metadata from NeMo configs.
- Non-default/dynamic RoPE until a checkpoint requires it.
- Sliding-window attention unless a representative in-library checkpoint uses non-null `sliding_window`.
- Quantized FP8/GGUF loading beyond dense fallback and explicit provider admission.

## 15. Final implementation checklist

- [ ] Parse `NemotronConfig` and normalize omitted `head_dim` / KV heads.
- [ ] Parse `NemotronHConfig`, legacy `hybrid_override_pattern`, and layer schedule.
- [ ] Load dense `nemotron` weights and preserve tied-weight metadata while respecting `tie_word_embeddings=false`.
- [ ] Implement `LayerNorm1P`.
- [ ] Implement `relu2` MLP.
- [ ] Implement partial RoPE and cache-position handling.
- [ ] Implement GQA causal attention with KV cache.
- [ ] Add packed QKV rewrite with split-order tests.
- [ ] Add dense `nemotron` prefill and decode parity tests.
- [ ] Define `nemotron_h` cache manifest for KV, conv state, and recurrent state.
- [ ] Implement/admit Mamba2 provider or validated fallback.
- [ ] Implement `nemotron_h` MLP and attention blocks.
- [ ] Implement MoE router/top-k/expert dispatch for A3B checkpoints.
- [ ] Add per-block and full-model synthetic parity tests.
- [ ] Benchmark prefill, decode, Mamba state update, MoE routing, and logits slicing.
