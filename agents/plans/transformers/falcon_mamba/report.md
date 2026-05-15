# Falcon Mamba DinoML Audit

## 1. Source basis

Transformers commit/version: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4` from local checkout `transformers`.

Model id: primary reachable configs were `tiiuae/falcon-mamba-7b`, `tiiuae/falcon-mamba-7b-instruct`, and `tiiuae/falcon-mamba-7b-instruct-4bit`.

Config source: `transformers/src/transformers/models/falcon_mamba/configuration_falcon_mamba.py`, plus saved HF config snapshots in `_sources/`.

Source files inspected:

- `transformers/src/transformers/models/falcon_mamba/modeling_falcon_mamba.py`
- `transformers/src/transformers/models/falcon_mamba/configuration_falcon_mamba.py`
- `transformers/src/transformers/models/falcon_mamba/modular_falcon_mamba.py`
- `transformers/src/transformers/cache_utils.py`
- `transformers/src/transformers/generation/utils.py`
- Shared Mamba files only for inheritance/behavior comparison: `models/mamba/modeling_mamba.py`, `models/mamba/configuration_mamba.py`

Any missing files or assumptions: `modeling_falcon_mamba.py` and `configuration_falcon_mamba.py` are generated from `modular_falcon_mamba.py`; generated files are the runtime source basis, modular is the future upstream edit source. `tiiuae/falcon-mamba-7b-base` and `hf-internal-testing/tiny-random-FalconMambaForCausalLM` config fetches returned 401. No DinoML tests/imports were run.

## 2. High-level architecture

Falcon Mamba is a text-only causal language model with a pure Mamba/selective-state-space decoder stack. There is no Transformer attention, no RoPE, and no KV cache. The generation state is fixed-size per layer: a depthwise convolution state plus a recurrent SSM state.

Dataflow:

```text
token ids -> token embedding -> repeated Mamba blocks -> final RMSNorm -> tied/untied LM head -> logits/sampling
```

Runtime stages:

- CPU/data pipeline: tokenizer, chat template, attention mask construction for padded prompts.
- Prefill: embedding plus all Mamba blocks over `[B, S]`, optionally initializes per-layer conv/recurrent cache.
- Decode: one new token per step, updates fixed-size conv state and recurrent state in place, computes last-token logits.
- Generation controller: owns sampling, EOS, beam/cache reorder, and `logits_to_keep=1`.

## 3. Important config dimensions

| Field | Source default | 7B / instruct configs | Runtime significance |
|---|---:|---:|---|
| `vocab_size` | 50280 | 65024 | Embedding and LM head width |
| `hidden_size` | 768 | 4096 | Residual/model width |
| `num_hidden_layers` | 32 | 64 | Number of Mamba state owners |
| `state_size` | 16 | 16 | Recurrent state width per intermediate channel |
| `expand` | 2 | 16 in saved configs | Source computes `intermediate_size = expand * hidden_size`; see trap below |
| `intermediate_size` | computed | 8192 serialized | Mixer/channel width for conv, scan, out projection |
| `time_step_rank` | `ceil(hidden_size/16)` | 256 | Low-rank dt projection width |
| `conv_kernel` | 4 | 4 | Conv cache length |
| `hidden_act` | `silu` | `silu` | Conv and gate activation |
| `use_bias` | false | false | `in_proj`/`out_proj` bias |
| `use_conv_bias` | true | true | Depthwise conv bias |
| `tie_word_embeddings` | true | false | Official 7B snapshots do not tie embeddings/head |
| `torch_dtype` | not fixed | bf16 or fp16 | Weight/runtime dtype; recurrent math upcasts selected terms |
| `use_cache` | true | true | Enables fixed-size SSM/conv cache |

Representative config sweep:

| Model id | dtype | hidden | layers | serialized `intermediate_size` | `expand` | vocab | token ids | quantization |
|---|---|---:|---:|---:|---:|---:|---|---|
| `tiiuae/falcon-mamba-7b` | bf16 | 4096 | 64 | 8192 | 16 | 65024 | BOS 0, EOS/PAD 11 | none |
| `tiiuae/falcon-mamba-7b-instruct` | bf16 | 4096 | 64 | 8192 | 16 | 65024 | BOS 8, EOS 11, PAD 0 | none |
| `tiiuae/falcon-mamba-7b-instruct-4bit` | fp16 compute snapshot | 4096 | 64 | 8192 | 16 | 65024 | BOS 8, EOS 11, PAD 0 | bitsandbytes FP4 |

## 3a. Family variation traps

- **Projection dimension trap:** current source derives `intermediate_size` from `expand * hidden_size`; saved configs serialize `expand=16` and `intermediate_size=8192`, which disagree for `hidden_size=4096`. DinoML must not infer projection widths from `hidden_size` alone. First loader should resolve dimensions from weight shapes or require a config normalization rule.
- No attention heads exist. `num_attention_heads`, `num_key_value_heads`, `head_dim`, RoPE, ALiBi, sliding-window attention, and KV cache are not applicable.
- Cache is not sequence-growing. It is fixed-size state per layer and batch row, so cache ABI must support static-address mutation rather than append-style KV tensors.
- `attention_mask` is multiplicative on hidden states before and after convolution during prefill; generation removes it after the first cached step.
- Official 4-bit config uses bitsandbytes metadata. That is a loading/provider contract, not a graph op.
- `use_falcon_mambapy` and training-only fused `falcon_mamba_inner_fn` are not required for inference parity.
- `use_associative_scan` affects tracing/training/reference path; production CUDA should use a scan/state-update provider or a clearly gated reference fallback.

## 4. Operator coverage checklist

Tensor/layout ops:

- Embedding lookup `[B,S] -> [B,S,H]`.
- Transpose/permute `[B,S,2I] <-> [B,2I,S]`, chunk/split on channel and projection dims.
- Pad left on sequence axis for conv state initialization.
- Slice/crop causal conv output to original sequence length.
- Last-token or indexed logits slice via `logits_to_keep`.
- Batch `index_select` for state reorder during beam search.

Neural primitives:

- RMSNorm over hidden width `H`.
- Linear `in_proj: H -> 2I`, no bias in official configs.
- Depthwise causal Conv1d: channels `I`, kernel `K=4`, groups `I`, optional bias, SiLU activation.
- Linear `x_proj: I -> R + 2N`, no bias, split order `[time_step, B, C]`.
- RMS normalization without learnable weights for `time_step`, `B`, and `C`.
- Linear/dense matmul `dt_proj: R -> I` with bias for slow path; fast path may pass weight matmul and separate bias to scan kernel.
- Elementwise `exp`, `softplus`, multiply/add recurrence, SiLU gate, residual add.
- Linear `out_proj: I -> H`, no bias in official configs.
- Final LM head `H -> vocab_size`, bias false.

Recurrent/state-space cache ops:

- Per-layer `conv_states`: `[B, I, K]`, same dtype/device as incoming hidden conv state, static address once initialized.
- Per-layer `recurrent_states`: `[B, I, N]`, where `N=state_size`, static address once initialized.
- `has_previous_state` boolean per layer chooses prefill versus decode path.
- Reset zeroes initialized states and clears `has_previous_state`.
- Reorder selects batch rows for both states with `beam_idx`.

Generation/cache ops:

- Create `DynamicCache(config)` whose `layer_types == ["mamba"] * num_hidden_layers`, mapped to `LinearAttentionLayer`.
- On first cached prefill, update both conv and recurrent states.
- On decode, mutate conv state with rolling append and mutate recurrent state with selective update.
- No `cache_position` is consumed by Falcon Mamba forward.

Attention primitives: none required.

Position encoding: none required.

Quantized/packed weight metadata ops:

- Defer bitsandbytes FP4 until DinoML has an explicit source-coupled quantized weight admission path. Dense bf16/fp16 loading is the safe first target.

## 5. Layer/block breakdown

Repeated decoder block, `L = num_hidden_layers`:

```text
residual = x
x_norm = RMSNorm(x)
projected = Linear_in(x_norm)                  # [B,S,H] -> [B,S,2I]
u, gate = split(transpose(projected), dim=C)   # each [B,I,S]
u = u * attention_mask                         # prefill/padded prompts only
u = depthwise_causal_conv1d(u) + bias
u = silu(u)
u = u * attention_mask                         # prefill/padded prompts only
params = Linear_x(transpose(u))                # [B,S,I] -> [B,S,R+2N]
dt_raw, B_t, C_t = split(params, [R,N,N])
dt_raw = rms(dt_raw); B_t = rms(B_t); C_t = rms(C_t)
dt = softplus(Linear_dt(dt_raw))               # [B,I,S]
A = -exp(A_log.float())                        # [I,N]
state_t = exp(A * dt_t) * state_{t-1} + dt_t * B_t * u_t
y_t = state_t @ C_t + D * u_t
y_t = y_t * silu(gate_t)
block_out = Linear_out(transpose(y))           # [B,S,I] -> [B,S,H]
x = residual + block_out                       # residual may be fp32
```

For 7B configs, intended source/weight-derived dimensions appear to be `H=4096`, `I=8192`, `R=256`, `N=16`, `K=4`, `L=64`. Because config `expand` conflicts with serialized `intermediate_size`, DinoML should validate these against checkpoint weight shapes.

## 6. Attention requirements

No attention is required for the primary causal LM target. There is no self-attention, cross-attention, MHA/MQA/GQA, attention mask addition, softmax attention, RoPE, ALiBi, or KV cache.

The closest attention-adjacent input is `attention_mask`, but source applies it as a multiplicative mask to mixer hidden states before and after causal convolution during prompt processing. During cached generation, `prepare_inputs_for_generation` sets `attention_mask=None` after the first iteration.

The cache manifest is per layer:

```text
layer i:
  type: mamba / linear-attention cache
  conv_states:      [batch, intermediate_size, conv_kernel]
  recurrent_states: [batch, intermediate_size, state_size]
  has_previous_state: bool
```

Cached keys are not stored; positions are represented only implicitly by the fixed recurrence state.

## 7. Position encoding and custom math

There is no explicit position encoding. Sequence order enters through causal depthwise convolution and the SSM recurrence.

Core recurrence sketch:

```python
def falcon_mamba_step(u_t, gate_t, dt_raw_t, b_t, c_t, state, A_log, D, dt_w, dt_b):
    b_t = rms(b_t)
    c_t = rms(c_t)
    dt_raw_t = rms(dt_raw_t)
    dt_t = softplus(dt_raw_t @ dt_w.T + dt_b)
    A = -exp(float32(A_log))
    state = exp(A * dt_t[:, :, None]) * state + dt_t[:, :, None] * b_t[:, None, :] * u_t[:, :, None]
    y_t = matmul(state, c_t[:, :, None]).squeeze(-1) + D[None, :] * u_t
    return y_t * silu(gate_t), state
```

`A_log` and `D` are learned, input-independent. `B`, `C`, and `dt` are token-dependent. `A_log`, `D`, `dt_proj.bias`, and several scan arguments are explicitly treated as float32 in source/fast paths; parity-sensitive kernels should preserve that upcast behavior.

## 8. Preprocessing and input packing

Model-coupled preprocessing is ordinary causal LM tokenization. The model consumes either:

- `input_ids: [B,S]` integer token ids, or
- `inputs_embeds: [B,S,H]`.

`attention_mask: [B,S]` is optional but important for padded prompts. It is multiplied into mixer channel states, not converted to an additive attention mask. No token type ids, position ids, packed sequence descriptors, placeholder tokens, or modality metadata are consumed by the model graph.

Generation controller behavior required for parity:

- `logits_to_keep=1` is set by generation utilities when supported to avoid full prompt logits.
- For cached generation after the first iteration, Falcon Mamba drops `attention_mask`.
- Beam reorder must reorder fixed Mamba states along batch dimension.

## 9. Graph rewrite / lowering opportunities

### Rewrite: depthwise causal Conv1d prefill to bounded channelwise FIR

Source pattern:

```text
pad-left by K-1 -> depthwise Conv1d(groups=I, kernel=K, padding=K-1) -> crop to S -> SiLU
```

Replacement:

```text
Channelwise causal FIR over [B,I,S] with K taps -> SiLU
```

Preconditions:

- `groups == in_channels == out_channels == I`.
- `stride == 1`, `dilation == 1`.
- Padding/crop exactly match source causal behavior.
- Weight layout converted from Conv1d `[I,1,K]` to per-channel taps `[I,K]`.
- Preserve source activation and optional bias.

Failure cases: non-depthwise conv, different padding/stride/dilation, missing crop, or layout pass that changes sequence/channel axes without rewriting all consumers.

Parity test sketch: compare a single mixer conv path for prompt lengths `1`, `K-1`, `K`, and `K+3`, including left-padded prompts.

### Rewrite: decode conv update to in-place ring/shift state

Source pattern:

```text
conv_state = roll_left(conv_state, num_new_tokens)
conv_state[..., -num_new_tokens:] = new_u
y = sum(conv_state * conv_weight, axis=-1) + bias
```

Replacement:

```text
Static state update + channelwise dot(K)
```

Preconditions:

- Decode token count is bounded; first target should require `num_new_tokens == 1`.
- State tensor address remains stable for CUDA graph compatibility.
- Batch shape matches initialized cache or explicit reorder/reset happened.

Failure cases: chunked decode with `num_new_tokens > 1` before ring-buffer semantics are validated; prefill shorter than `conv_kernel` has a source TODO about logical correctness.

### Rewrite: selective scan prefill to provider op

Source pattern:

```text
discrete_A, deltaB_u -> associative/sequential scan over sequence -> y, last_state
```

Replacement:

```text
falcon_mamba_selective_scan(u, dt, A, B, C, D, gate, dt_bias, return_last_state=True)
```

Preconditions:

- Inputs have canonical `[B,I,S]` for `u/dt/gate`, `[I,N]` for `A`, `[B,N,S]` or `[B,S,N]` for B/C with documented layout.
- `dt_softplus=True`.
- B/C/dt RMS has already been applied or is fused with exact epsilon.
- Kernel returns both output `[B,I,S]` and final state `[B,I,N]`.

Failure cases: training-only fused inner path, quantized `dt_proj` path, or missing state return when cache is enabled.

### Rewrite: decode selective state update to provider op

Source pattern:

```text
state = exp(A * softplus(dt + bias)) * state + softplus(dt + bias) * B * u
y = (state @ C) + D * u
y *= silu(gate)
```

Replacement:

```text
falcon_mamba_state_update_inplace(state, u_t, dt_t, A, B_t, C_t, D, gate_t, dt_bias)
```

Preconditions:

- One token per batch row.
- State address remains stable and mutation ordering is layer-serial.
- dtype/upcast policy matches source.

Failure cases: general sequence chunk update without scan equivalence tests; concurrent decode that aliases the same state rows.

### Rewrite: last-token-only logits

Source pattern:

```text
lm_head(hidden_states[:, -1:, :])
```

Replacement:

```text
slice_last_hidden -> GEMM(H, vocab)
```

Preconditions: `logits_to_keep == 1` or explicit tensor indices are supported. Preserve output float32 cast.

## 10. Kernel fusion candidates

Highest priority:

- Mamba selective scan prefill provider returning output and final recurrent state.
- Mamba selective state update decode provider with in-place `[B,I,N]` mutation.
- Depthwise causal conv prefill and decode update, including static-address `[B,I,K]` state.
- RMSNorm over hidden and small RMS over B/C/dt projection slices.
- Linear projection bundles around `in_proj`, `x_proj`, `dt_proj`, and `out_proj`, with shape guards from weight dimensions.

Medium priority:

- Fuse `x_proj -> split -> RMS(B,C,dt) -> dt_proj` where it reduces memory traffic.
- Fuse scan output `+ D*u`, gate SiLU multiply, and `out_proj` input preparation.
- Last-token-only LM head GEMM for decode.
- State reorder/reset kernels for batched generation.

Lower priority:

- Training-only `falcon_mamba_inner_fn` parity.
- Associative scan reference lowering for torch.compile-style export.
- Bitsandbytes FP4 load/dequant provider for the 4-bit checkpoint.

## 11. Runtime staging plan

Stage 1: config and weight admission. Parse Falcon Mamba configs, validate projection dims against checkpoint weight shapes, and reject the `expand/intermediate_size` ambiguity unless normalized.

Stage 2: dense single-block parity without cache. Implement embedding, RMSNorm, projections, depthwise causal conv, reference scan, residual, and output projection.

Stage 3: full prefill parity. Run all layers, final norm, and LM head. Cache may be disabled at first.

Stage 4: cache ABI. Add `MambaStateCache` with per-layer `conv_states`, `recurrent_states`, `has_previous_state`, reset, reorder, and static-address mutation semantics.

Stage 5: decode parity. Implement one-token conv update and selective state update, then last-token logits.

Stage 6: optimized providers. Replace reference conv/scan/state update with CUDA kernels and profile prefill/decode separately.

Stage 7: optional weight formats and scheduling. Add bitsandbytes/dense fallback policy, then batching/beam-state management.

## 12. Parity and validation plan

- Config loader tests for default config, 7B config, instruct config, and 4-bit config admission/rejection.
- Single-op tests: RMSNorm, B/C/dt RMS, depthwise causal conv crop, decode conv state update, softplus dt with bias, recurrent update.
- Single-block parity against Transformers for fp32, then bf16/fp16 with tolerances around `1e-2` absolute for reduced precision.
- Prefill parity for prompt lengths `1`, `3`, `4`, `5`, and longer sequences with padded attention masks.
- Cache parity: prefill `S` tokens then decode one token must match full recompute on `S+1` tokens.
- Beam reorder parity: reorder batch states then decode and compare with reordered full recompute.
- End-to-end logits parity for `tiiuae/falcon-mamba-7b` dense weights when accessible.
- 4-bit checkpoint: initially validate rejection or dense materialization fallback, not numerical parity through bitsandbytes.

## 13. Performance probes

- Prefill throughput by sequence length: 1, 4, 16, 128, 1024, 4096.
- Decode tokens/sec by batch size, especially state update bandwidth for `[B,I,N]`.
- Conv update bandwidth and latency for `[B,I,K]`.
- Selective scan provider comparison: sequential reference, associative scan, custom CUDA/Triton provider.
- LM head cost with full logits versus `logits_to_keep=1`.
- State memory footprint: `layers * B * I * (K + N) * dtype_size`.
- Weight load memory for dense bf16/fp16 versus 4-bit materialization.
- Batch reorder/reset overhead during beam or continuous batching.

## 14. Skip/defer list

- Training, loss parity, gradient checkpointing, and training-only fused `falcon_mamba_inner_fn`.
- mambapy fallback.
- General chunked decode beyond one token per step.
- Bitsandbytes FP4 execution, except explicit reject/fallback metadata.
- Torch associative scan export path.
- Multi-GPU tensor parallelism.
- Continuous batching until fixed-state ABI and reorder/reset are proven.

## 15. Final implementation checklist

- [ ] Parse Falcon Mamba config and normalize/gate `expand` versus `intermediate_size`.
- [ ] Load dense weights with projection shape validation.
- [ ] Implement/tokenize input ABI for `input_ids`, `inputs_embeds`, and `[B,S]` attention masks.
- [ ] Implement RMSNorm and small unweighted RMS for B/C/dt slices.
- [ ] Implement `in_proj`, `x_proj`, `dt_proj`, `out_proj`, final LM head.
- [ ] Implement depthwise causal Conv1d prefill rewrite.
- [ ] Implement one-token convolutional state update.
- [ ] Implement selective scan prefill with returned recurrent state.
- [ ] Implement selective state update decode with in-place recurrent state mutation.
- [ ] Add `MambaStateCache` manifest/ABI with conv and recurrent states per layer.
- [ ] Add reset, reorder, and admission guards for cache state.
- [ ] Add last-token-only logits lowering.
- [ ] Add single-block, prefill, decode, and reorder parity tests.
- [ ] Add prefill/decode/state-memory benchmarks.
- [ ] Add explicit dense fallback or rejection for bitsandbytes FP4 configs.
