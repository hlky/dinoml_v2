# Falcon H1 Transformers Audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: tiiuae/Falcon-H1 family, primary target FalconH1ForCausalLM
Config source: local configuration_falcon_h1.py plus HF config.json sweep
Source files inspected:
  X:/H/transformers/src/transformers/models/falcon_h1/configuration_falcon_h1.py
  X:/H/transformers/src/transformers/models/falcon_h1/modeling_falcon_h1.py
  X:/H/transformers/src/transformers/models/falcon_h1/modular_falcon_h1.py
  X:/H/transformers/src/transformers/models/falcon_h1/convert_mamba_ssm_checkpoint.py
  X:/H/transformers/src/transformers/cache_utils.py
Any missing files or assumptions: no weights fetched; no imports/tests run; config reads were direct HF raw JSON.
```

`modeling_falcon_h1.py` is generated from `modular_falcon_h1.py`. Runtime behavior was read from the generated file; future source edits in Transformers should target the modular source.

## 2. High-level architecture

Falcon H1 is a text-only causal LM decoder whose every layer is hybrid: a Mamba2-style state-space mixer and a causal GQA attention block run from the same pre-normalized hidden state, their outputs are summed, then a gated MLP runs after a second RMSNorm.

```text
input_ids -> token embedding * embedding_multiplier
  -> repeated hybrid decoder blocks:
       RMSNorm -> parallel Mamba2 SSM branch + causal GQA attention branch -> sum -> residual
       RMSNorm -> SwiGLU MLP -> residual
  -> final RMSNorm -> last-token/window lm_head * lm_head_multiplier -> logits/sampling
```

Generation has two persistent state classes in the same `past_key_values` object: attention KV states that grow with sequence length and Mamba conv/recurrent states that remain fixed-size per layer.

## 3. Important config dimensions

Source defaults:

| Field | Default / source behavior |
|---|---:|
| `vocab_size` | 128000 |
| `hidden_size` | 4096 |
| `num_hidden_layers` | 32 |
| `num_attention_heads` | 32 |
| `num_key_value_heads` | 8, or attention heads if `None` |
| `head_dim` | used if present; otherwise `hidden_size // num_attention_heads` |
| `intermediate_size` | 14336 for MLP |
| `max_position_embeddings` | 8192 |
| `rope_parameters` | default RoPE via `rope_theta`; sampled checkpoints use `rope_theta=1e11` |
| `mamba_d_ssm` | 1024; overrides `mamba_expand * hidden_size` |
| `mamba_n_heads`, `mamba_d_head` | 128, `auto` -> `mamba_intermediate // mamba_n_heads` |
| `mamba_n_groups`, `mamba_d_state`, `mamba_d_conv` | 1, 256, 4 |
| `mamba_chunk_size` | 256 default; sampled checkpoints often 128 |
| biases | attention/MLP/projector false by default, conv bias true |
| cache | `use_cache=True`; model marked stateful |

Representative HF config sweep:

| Checkpoint | Layers | Hidden | MLP | Vocab | Attn heads/KV/head_dim | Attn width | Mamba heads/head_dim | Mamba width | Groups/state | Context | Chunk | RMS gate |
|---|---:|---:|---:|---:|---|---:|---|---:|---|---:|---:|---|
| Tiny-R-90M | 24 | 512 | 768 | 32768 | 8/2/64 | 512 | 24/32 | 768 | 1/64 | 262144 | 128 | no |
| 0.5B Base | 36 | 1024 | 2048 | 32784 | 8/2/64 | 512 | 24/64 | 1536 | 1/128 | 16384 | 128 | no |
| 1.5B Base | 24 | 2048 | 4608 | 65536 | 8/2/128 | 1024 | 48/64 | 3072 | 1/256 | 131072 | 128 | yes |
| 1.5B Deep Base | 66 | 1280 | 3072 | 65536 | 6/2/128 | 768 | 24/64 | 1536 | 1/256 | 131072 | 128 | yes |
| 3B Base | 32 | 2560 | 6144 | 65536 | 10/2/128 | 1280 | 32/128 | 4096 | 1/256 | 131072 | 128 | yes |
| 7B Base | 44 | 3072 | 12288 | 130048 | 12/2/128 | 1536 | 24/128 | 3072 | 1/256 | 262144 | 128 | yes |
| 7B Instruct | 44 | 3072 | 12288 | 130049 | 12/2/128 | 1536 | 24/128 | 3072 | 1/256 | 262144 | 256 | yes |
| 34B Base | 72 | 5120 | 21504 | 261120 | 20/4/128 | 2560 | 32/128 | 4096 | 2/256 | 262144 | 128 | yes |

## 3a. Family variation traps

- Do not infer attention projection width from `hidden_size`: 0.5B, 1.5B, deep, 3B, 7B, and 34B configs all have `num_attention_heads * head_dim < hidden_size`.
- Mamba width is independent: `mamba_n_heads * mamba_d_head == mamba_d_ssm` in sampled configs, and may be greater than, equal to, or less than `hidden_size`.
- GQA is always present in sampled configs; KV heads are fewer than query heads.
- `vocab_size` differs between base and instruct variants by one token in sampled 1.5B/7B configs.
- `mamba_rms_norm` changes the branch post-scan math: false uses `scan_output * silu(gate)`, true uses gated RMSNorm.
- `mamba_chunk_size` varies, including 7B Instruct using 256 while 7B Base uses 128.
- Long-context settings use very large `rope_theta` and contexts up to 262144.
- GPTQ repos advertise packed quantization metadata, but native Falcon H1 modeling only defines dense modules; packed weights need provider admission or fallback.
- No MoE is implemented in native `falcon_h1`.
- `configuration_falcon_h1.py` has no annotated `head_dim`/`rope_theta` fields, but source and conversion paths read/use those attributes. DinoML config parsing should preserve unknown compatible fields and validate effective values.

## 4. Operator coverage checklist

Tensor/layout ops:
- embedding lookup, scalar multipliers, reshape/view, transpose/permute, contiguous, split, concat, pad, chunk reshape, repeat/interleave or broadcast expansion, roll/update for conv cache, index select for beam reorder.

Neural primitives:
- dense `Linear`, `Embedding`, RMSNorm, optional gated grouped RMSNorm, SiLU, elementwise multiply/add, `softplus`, `clamp`, `exp`, `cumsum`, depthwise causal `Conv1d`.

Attention primitives:
- causal self-attention, GQA KV repeat, additive causal/padding mask, fp32 softmax, dropout only in training, RoPE on Q/K before cache update.

Position ops:
- default RoPE with `inv_freq = 1 / rope_theta^(arange(0, head_dim, 2) / head_dim)`, dynamic RoPE plumbing if non-default `rope_parameters` appears.

Generation/cache ops:
- `DynamicCache` with attention KV plus fixed Mamba state per layer; last-token-only logits via `num_logits_to_keep`.

Recurrent/state-space ops:
- Mamba2 selective scan, depthwise causal conv update, recurrent state update, chunked scan fallback, optional fused `mamba-ssm` / `causal-conv1d` kernels.

Quantized/packed weight metadata:
- GPTQ sampled config has `bits`, `group_size`, `pack_dtype=int32`, `desc_act`, `sym`; source has no native packed matmul.

## 5. Layer/block breakdown

For each of `N` decoder layers:

```text
residual = x
h = RMSNorm(x)

m = Mamba2Mixer(h, mamba_mask, state)
  h0 = h * ssm_in_multiplier
  p = Linear(hidden_size -> mamba_d_ssm + conv_dim + mamba_n_heads)(h0) * mup_vector
  gate, hidden_B_C, dt = split(p, [mamba_d_ssm, conv_dim, mamba_n_heads])
  hidden_B_C = depthwise causal Conv1d(conv_dim, kernel=mamba_d_conv) + SiLU
  hidden, B, C = split(hidden_B_C, [mamba_d_ssm, groups * d_state, groups * d_state])
  y = Mamba2 selective scan/update over [B, T, mamba_n_heads, mamba_d_head]
  y = gated RMSNorm(y, gate) if mamba_rms_norm else y * silu(gate)
  m = Linear(mamba_d_ssm -> hidden_size)(y) * ssm_out_multiplier

a = Attention(h * attention_in_multiplier)
  q = Linear(hidden_size -> num_attention_heads * head_dim)
  k = Linear(hidden_size -> num_key_value_heads * head_dim) * key_multiplier
  v = Linear(hidden_size -> num_key_value_heads * head_dim)
  q,k = RoPE(q,k)
  k,v = cache.update(k,v)
  a = causal GQA attention(q,k,v) -> Linear(num_attention_heads * head_dim -> hidden_size)
  a = a * attention_out_multiplier

x = residual + m + a
residual = x
x = residual + Linear(MLP -> hidden)(SiLU(gate_proj(x) * gate_multiplier) * up_proj(x)) * down_multiplier
```

Biases are config-controlled. Sampled native checkpoints use no attention, MLP, or Mamba projection bias; depthwise conv bias is true.

## 6. Attention requirements

The attention branch is causal self-attention with GQA. Q shape is `[B, Hq, Tq, Dh]`; K/V shape before repeat and cache is `[B, Hkv, Tk, Dh]`. Cached K is already RoPE-applied. Eager attention repeats K/V to query heads, computes `q @ k^T * Dh^-0.5`, adds the causal mask, softmaxes in fp32, casts back to query dtype, then computes `weights @ v`.

No sliding-window, block-sparse, ALiBi, cross-attention, or MoE attention is present. FlashAttention and SDPA are supported through Transformers `ALL_ATTENTION_FUNCTIONS`, but DinoML parity should preserve the eager math order, especially `key_multiplier` before RoPE/cache and fp32 softmax.

Cache manifest per layer:

| State | Shape | Grows with sequence? | Notes |
|---|---|---|---|
| attention key | `[B, num_key_value_heads, cached_T, head_dim]` | yes | stored after RoPE and `key_multiplier` |
| attention value | `[B, num_key_value_heads, cached_T, head_dim]` | yes | before GQA repeat |
| conv state | `[B, conv_dim, mamba_d_conv]` | fixed | rolling/static address update |
| recurrent state | `[B, mamba_n_heads, mamba_d_head, mamba_d_state]` | fixed | overwritten each token/chunk |

Beam reorder must index-select the batch dimension for all four states.

## 7. Position encoding and custom math

Default RoPE is standard half-rotation over `head_dim`:

```python
inv_freq = 1.0 / (rope_theta ** (arange(0, head_dim, 2) / head_dim))
freqs = inv_freq[None, :, None] @ position_ids[:, None, :]
emb = cat([freqs, freqs], dim=-1)
q = q * cos(emb) + rotate_half(q) * sin(emb)
k = k * cos(emb) + rotate_half(k) * sin(emb)
```

`position_ids` start at `past_key_values.get_seq_length()` during decode. Cos/sin depend on batch position IDs but not layer, so they can be computed once per model forward and shared across layers.

Mamba custom math:
- `A = -exp(A_log.float())`
- `dt = softplus(dt + dt_bias)`, then clamp to `time_step_limit`
- decode recurrence: `state = state * exp(dt * A) + (dt * B) * x`
- output: `y = state @ C + D * x`, then gated RMSNorm or SiLU gate.

## 8. Preprocessing and input packing

There is no model-coupled image/audio/video preprocessing. Runtime inputs are ordinary text `input_ids` or `inputs_embeds`, optional `[B, T]` attention mask, optional `position_ids`, and optional hybrid cache. Tokenizer-specific chat templates and instruct formatting are generation-controller/data-pipeline concerns, not neural graph ops.

The source creates a causal mask via `create_causal_mask`. Mamba receives a simpler mask: padding masks are applied to hidden states only when not decoding from previous state and not all-ones. For cached decode, `_update_mamba_mask` returns `None`.

`prepare_inputs_for_generation` sets `logits_to_keep = config.num_logits_to_keep`; source defaults to last-logit generation.

## 9. Graph rewrite / lowering opportunities

### Rewrite: separate Q/K/V projections -> grouped GEMM

Preconditions:
- Same input tensor and dtype.
- Preserve exact split order `q, k, v` and per-projection bias flags.
- Apply `key_multiplier` only to K after K projection.
- Use explicit widths: `q = Hq * Dh`, `k = v = Hkv * Dh`.

Replacement:
```text
MatMul(x, concat_rows(q_w,k_w,v_w).T) -> split([q_wide,k_wide,v_wide])
```

Weight transform: concatenate dense row-major PyTorch linear weights along output rows. Failure cases: provider cannot materialize fused transient weight, quantized per-module metadata, or future configs with different bias behavior.

### Rewrite: last-token-only logits

Preconditions:
- `labels is None`.
- `logits_to_keep` is integer 1 or a concrete slice/tensor accepted by source semantics.

Replacement:
```text
hidden[:, -K:, :] -> lm_head -> scalar lm_head_multiplier
```

Failure cases: full-sequence loss/perplexity or callers requesting all logits.

### Rewrite: depthwise causal Conv1d update -> rolling state dot

Preconditions:
- `seq_len == 1`, cache has previous state, `groups == conv_dim`, `padding == kernel-1`, activation is SiLU/Swish or source fallback activation is reproduced.
- Conv state has static shape `[B, conv_dim, kernel]`.

Replacement:
```text
state = roll_append(state, new_hidden_B_C)
y = sum(state * weight.squeeze(1), dim=-1) + bias
y = silu(y)
```

Failure cases: prefill/chunk path, non-SiLU activation, missing conv bias handling.

### Rewrite: Mamba chunk scan -> provider primitive

Preconditions:
- Validate `mamba_n_heads * mamba_d_head == mamba_d_ssm`.
- `mamba_n_heads % mamba_n_groups == 0`.
- Chunk padding follows source formula `(chunk_size - T % chunk_size) % chunk_size`.
- `time_step_limit` and gated RMSNorm mode match config.

Replacement: lower the scan region to a single provider `mamba2_scan` op returning output and final recurrent state. Failure cases: dynamic chunk sizes not supported, non-default multipliers not folded, or grouped B/C repetition mismatch.

### Rewrite: multiplier folding into weights

Preconditions:
- Inference-only, constants known, no weight sharing violation.
- Fold `embedding_multiplier`, `key_multiplier`, `attention_in/out`, `ssm_in/out`, `mlp_multipliers`, `lm_head_multiplier`, and `mup_vector` at boundaries that preserve dtype and rounding policy.

Failure cases: quantized weights, calibration/export needing unfused parameters, or mixed precision where source multiplication order matters.

Layout constraints: source is sequence-major `[B, T, C]` with attention transposes to `[B, H, T, D]` and Conv1d as `[B, C, T]`. No NCHW/NHWC image layout work applies. A layout pass can eliminate local transposes around attention and conv only if consumers remain in the fused region.

## 10. Kernel fusion candidates

Highest priority:
- Mamba2 prefill scan and decode update, including causal conv update, because it is half of every layer and fixed-state decode depends on it.
- GQA FlashAttention/SDPA with KV cache for long contexts.
- RMSNorm and gated RMSNorm.
- Q/K/V projection + reshape + RoPE + cache update.

Medium priority:
- SwiGLU MLP fusion with gate/down multipliers.
- Last-token-only LM head.
- Depthwise causal Conv1d prefill fused with activation.
- Multipliers folded into adjacent matmuls where parity permits.

Lower priority:
- Training-only fused `mamba_split_conv1d_scan_combined`.
- Quantized GPTQ kernels until dense native parity is established.
- Tensor-parallel plans.

## 11. Runtime staging plan

1. Parse native configs and reject unsupported/ambiguous fields; preserve explicit `head_dim`, RoPE fields, Mamba dimensions, multipliers, and quantization metadata.
2. Load dense weights for one small checkpoint and verify tensor shapes without executing imports.
3. Implement one hybrid block eager parity: RMSNorm, attention, MLP, Mamba eager scan.
4. Add full prefill parity with chunked Mamba state initialization and attention KV cache creation.
5. Add single-token decode with four-part cache update.
6. Swap attention and Mamba scan to optimized provider kernels behind shape/config guards.
7. Add generation-controller support for last-token logits and cache reorder/reset.
8. Admit GPTQ variants only after provider/loader contract is audited separately.

## 12. Parity and validation plan

- Random tensor tests for RoPE, RMSNorm, gated RMSNorm, GQA repeat, causal conv update, and Mamba decode recurrence.
- Single-layer parity with fixed random weights for both `mamba_rms_norm=false` and true.
- Prefill parity at sequence lengths below, equal to, and above `mamba_chunk_size`; include non-multiple lengths.
- Decode parity after prefill: one-token and several-token continuation, checking attention KV growth and fixed Mamba state mutation.
- Mask parity with all-ones masks, padded masks in prefill, and cached decode where Mamba mask is intentionally disabled.
- Logit parity with `logits_to_keep=1` and full logits.
- Suggested tolerances: fp32 custom ops near `1e-5`; bf16/fp16 end-to-end looser, compare against source dtype behavior and isolate Mamba scan numeric drift.

No validation was run for this audit per user scope.

## 13. Performance probes

- Prefill throughput split into attention time, Mamba scan time, MLP time, and LM head time.
- Decode tokens/sec with cache memory accounting for attention KV versus fixed Mamba states.
- Context-length sweep up to 16K, 128K, and 256K where model config allows.
- Chunk-size sweep for configs using 128 versus 256.
- Batch-size sweep with fixed prompt and fixed decode length.
- Mamba eager fallback versus provider primitive.
- Attention backend comparison: eager, SDPA, FlashAttention, DinoML fused.
- KV cache memory versus recurrent/conv state memory by checkpoint.
- GPTQ load/dequant/provider probe as a separate admission benchmark.

## 14. Skip/defer list

- Training, loss, gradient checkpointing, dropout.
- MoE, vision/audio/video, cross-attention: not present in native source.
- Beam search beyond cache batch reorder in first integration.
- Tensor parallel and pipeline plans.
- GPTQ packed execution until dense path is correct.
- Non-default/dynamic RoPE variants unless a sampled checkpoint requires them.
- Conversion from external `mamba_ssm` checkpoints.

## 15. Final implementation checklist

- [ ] Parse Falcon H1 config including explicit `head_dim`, RoPE compatibility fields, Mamba dimensions, multipliers, and quantization metadata.
- [ ] Add dense weight shape validation for attention, Mamba, MLP, embedding, and LM head.
- [ ] Implement RMSNorm and gated grouped RMSNorm.
- [ ] Implement causal GQA attention with RoPE-before-cache and KV repeat.
- [ ] Implement Falcon H1 hybrid cache ABI: attention K/V plus conv/recurrent state.
- [ ] Implement Mamba2 prefill scan with source chunk padding and grouped B/C repeat.
- [ ] Implement Mamba2 single-token decode recurrence and depthwise conv update.
- [ ] Implement SwiGLU MLP with gate/down multipliers.
- [ ] Implement last-token-only logits and LM head multiplier.
- [ ] Add guarded QKV projection fusion.
- [ ] Add guarded Mamba scan provider lowering.
- [ ] Add parity tests for one block, full prefill, cached decode, masks, and logits slicing.
- [ ] Add performance probes for prefill, decode, cache memory, and scan backend.
- [ ] Gate or reject GPTQ variants until packed-weight provider support is audited.
