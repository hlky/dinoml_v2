# Mamba Transformers Audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: state-spaces/mamba-130m-hf, 370m-hf, 790m-hf, 1.4b-hf, 2.8b-hf; hf-internal-testing/tiny-random-MambaForCausalLM
Config source: Hugging Face config.json snapshots in agents/plans/transformers/mamba/_sources/
Source files inspected:
- transformers/src/transformers/models/mamba/modeling_mamba.py
- transformers/src/transformers/models/mamba/configuration_mamba.py
- transformers/src/transformers/models/mamba/convert_mamba_ssm_checkpoint_to_pytorch.py
- transformers/src/transformers/cache_utils.py
Any missing files or assumptions: tokenizer is not model-coupled beyond normal input_ids; no remote code required for the in-library Mamba class. Report target is inference-only CUDA causal LM.
```

Source URLs at the inspected commit:

- https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/mamba/modeling_mamba.py
- https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/mamba/configuration_mamba.py
- https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/cache_utils.py

## 2. High-level architecture

Mamba is a text-only causal language model, but it is not a Transformer decoder block. Each layer is a residual pre-norm Mamba state-space block:

```text
input_ids -> token embedding -> repeated Mamba blocks -> final RMSNorm -> tied LM head -> logits/sampling
```

Generation stages:

```text
CPU/tokenizer -> embedding -> prefill selective scan + conv/SSM state creation -> decode one-token conv/state update -> last-token logits
```

There is no self-attention, cross-attention, RoPE, ALiBi, or KV cache. The cache is a per-layer linear-attention-style state container with:

- convolution state: `[batch, intermediate_size, conv_kernel]`
- recurrent SSM state: `[batch, intermediate_size, state_size]`

Prefill and decode can be validated independently. Prefill must reproduce the full selective scan and write final conv/recurrent states. Decode must consume and mutate those states with one-token causal convolution and `selective_state_update`.

## 3. Important config dimensions

Effective source defaults from `MambaConfig`:

| Field | Default / rule | Runtime significance |
|---|---:|---|
| `vocab_size` | 50280 | embedding and LM head rows |
| `hidden_size` | 768 | residual width |
| `expand` | 2 | `intermediate_size = expand * hidden_size` |
| `state_size` | 16 | SSM recurrent state per channel |
| `num_hidden_layers` | 32 | repeated Mamba blocks |
| `conv_kernel` | 4 | depthwise causal conv state width |
| `time_step_rank` | `"auto"` -> `ceil(hidden_size / 16)` | low-rank dt projection |
| `hidden_act` | `silu` | conv output activation and gate activation |
| `use_bias` | false | no bias on `in_proj` or `out_proj` |
| `use_conv_bias` | true | depthwise conv bias |
| `residual_in_fp32` | true | residual addition upcasts residual |
| `layer_norm_epsilon` | `1e-5` | RMSNorm epsilon |
| `use_cache` | true | returns linear-attention state cache |
| `tie_word_embeddings` | true | LM head tied to token embedding |

Representative checkpoint sweep, from downloaded `config.json` snapshots:

| Model id | Hidden | Intermediate | Layers | State | dt rank | Conv | Vocab | Biases | dtype |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| `hf-internal-testing/tiny-random-MambaForCausalLM` | 32 | source recomputes 64 from `expand=2`; JSON says 32 | 2 | 16 | 2 | 4 | 1024 | linear no, conv yes | float32 |
| `state-spaces/mamba-130m-hf` | 768 | 1536 | 24 | 16 | 48 | 4 | 50280 | linear no, conv yes | float32 |
| `state-spaces/mamba-370m-hf` | 1024 | 2048 | 48 | 16 | 64 | 4 | 50280 | linear no, conv yes | float32 |
| `state-spaces/mamba-790m-hf` | 1536 | 3072 | 48 | 16 | 96 | 4 | 50280 | linear no, conv yes | float32 |
| `state-spaces/mamba-1.4b-hf` | 2048 | 4096 | 48 | 16 | 128 | 4 | 50280 | linear no, conv yes | float32 |
| `state-spaces/mamba-2.8b-hf` | 2560 | 5120 | 64 | 16 | 160 | 4 | 50280 | linear no, conv yes | float32 |

## 3a. Family variation traps

- This is not an attention decoder. Do not map it to QKV, RoPE, SDPA, FlashAttention, or KV cache.
- The public state-spaces configs include historical fields such as `fused_add_norm`, `rms_norm`, `d_inner`, `d_model`, `n_layer`, `ssm_cfg`, and `pad_vocab_size_multiple`; the inspected `MambaConfig`/`modeling_mamba.py` only uses the normalized fields such as `hidden_size`, `num_hidden_layers`, `intermediate_size`, and Mamba-specific step/state fields.
- `MambaConfig.__post_init__` recomputes `intermediate_size = expand * hidden_size`. If a config carries a conflicting `intermediate_size`, DinoML should follow the source config rule or reject the checkpoint after verifying actual weight shapes.
- `time_step_rank="auto"` is source-supported, but public configs pin the computed integer.
- Fast CUDA inference requires optional external kernels from `mamba-ssm` and `causal-conv1d`. Missing kernels fall back to a slow PyTorch sequential path unless `use_mambapy=True` selects the `mambapy` training fallback.
- Attention masks are multiplicative `[batch, seq]` masks applied to hidden channels before and after convolution during prefill. During generation, `prepare_inputs_for_generation` clears `attention_mask` after the first cached iteration.
- Cache state uses static-address mutation for graph capture. Replacing cache tensors with newly allocated tensors can break expected cudagraph behavior.
- No layout translation opportunity exists at the model semantic level; tensors are `[batch, seq, hidden]` around linears and `[batch, intermediate, seq]` inside convolution/scan.

## 4. Operator coverage checklist

Tensor/layout ops:

- Token embedding lookup `[batch, seq] -> [batch, seq, hidden]`.
- Transpose between `[B, S, I]` and `[B, I, S]`.
- Chunk/split `in_proj` output into hidden and gate along intermediate channel dim.
- Split `x_proj` output into `time_step`, `B`, `C` with sizes `[rank, state_size, state_size]`.
- Pad left on sequence dimension for conv state initialization.
- Slice/truncate conv output to original `seq_len`.
- `roll` or equivalent state shift for slow decode conv cache.
- Last-token slice / `logits_to_keep` slice before LM head.

Neural network primitives:

- Tied token embedding / LM head alias.
- RMSNorm over hidden dim with fp32 variance.
- Linear `hidden_size -> 2 * intermediate_size` (`in_proj`, no bias in public configs).
- Depthwise causal Conv1d over intermediate channels, `groups=intermediate_size`, kernel 4, padding 3, conv bias true.
- SiLU activation for convolution output and gate.
- Linear `intermediate_size -> time_step_rank + 2 * state_size` (`x_proj`, no bias).
- Linear/time projection `time_step_rank -> intermediate_size` (`dt_proj`, bias true).
- Linear `intermediate_size -> hidden_size` (`out_proj`, no bias in public configs).
- Residual add, optionally with residual operand in fp32.
- LM head `hidden_size -> vocab_size`, bias false, tied to embedding.

Selective scan / state-space primitives:

- `A = -exp(A_log.float())`, `D` skip vector.
- `dt = softplus(dt_proj(time_step))`.
- Per-token recurrence `state_t = exp(A * dt_t) * state_{t-1} + (dt_t * B_t) * u_t`.
- Output `y_t = dot(state_t, C_t) + D * u_t`, gated by `silu(gate_t)`.
- Optimized CUDA prefill path delegates to `selective_scan_fn`.
- Optimized CUDA decode path delegates to `selective_state_update`.

Generation/cache ops:

- `DynamicCache(config)` builds `LinearAttentionLayer` for every `layer_type == "mamba"`.
- Per-layer cache update for conv state and recurrent state.
- Beam/cache reorder uses batch `index_select` on both conv and recurrent states.
- Cache reset zeros both states and clears `has_previous_state`.
- No KV append, no KV repeat for GQA/MQA, no attention mask growth.

## 5. Layer/block breakdown

Mamba block, repeated `num_hidden_layers` times:

```text
residual = x
x_norm = RMSNorm(x).to(norm_weight_dtype)
if residual_in_fp32: residual = residual.float()
mixer_out = MambaMixer(x_norm, cache, attention_mask)
x = residual + mixer_out
```

MambaMixer prefill path:

```text
projected = in_proj(x).transpose(1, 2)              # [B, 2I, S]
u, gate = chunk(projected, 2, dim=1)                # [B, I, S] each
u = u * mask[:, None, :] if attention_mask
u = causal_depthwise_conv1d(u)[:, :, :S]            # [B, I, S]
u = silu(u)
u = u * mask[:, None, :] if attention_mask
params = x_proj(u.transpose(1, 2))                  # [B, S, R + 2N]
time_step, Bp, Cp = split(params, [R, N, N], -1)
dt = softplus(dt_proj(time_step)).transpose(1, 2)   # [B, I, S]
scan = selective_scan(u, dt, A, Bp, Cp, D, gate)    # [B, I, S]
out = out_proj(scan.transpose(1, 2))                # [B, S, H]
```

MambaMixer decode path, after cache exists:

```text
projected = in_proj(x).transpose(1, 2)              # normally [B, 2I, 1]
u, gate = chunk(projected, 2, dim=1)
u = causal_conv1d_update(u[:, :, 0], conv_state, conv_weight, bias, "silu")
params = x_proj(u[:, None, :])                      # [B, 1, R + 2N]
dt = dt_proj.weight @ time_step.transpose(1, 2)     # [B, I, 1] before softplus in kernel
scan = selective_state_update(recurrent_state, u, dt, A, Bp, Cp, D, gate, dt_bias)
out = out_proj(scan[:, None, :])
```

Notation: `H=hidden_size`, `I=intermediate_size`, `R=time_step_rank`, `N=state_size`, `S=sequence length`.

## 6. Attention requirements

No attention is required for the primary Mamba target.

- Causality is implemented by depthwise causal convolution plus a recurrent state-space scan, not by a causal attention mask.
- There are no query/key/value projections, no attention heads, no `num_key_value_heads`, no RoPE, no ALiBi, no sliding-window attention, and no SDPA/FlashAttention path.
- The cache is not `[layer, key/value, batch, heads, seq, head_dim]`. It is two fixed-size state tensors per layer: conv state `[B, I, K]` and recurrent state `[B, I, N]`.
- Cached recurrent states are stored after the selective scan update. Cached conv states store the latest convolution window.
- Optimized backend dispatch is through `causal_conv1d_fn`, `causal_conv1d_update`, `selective_scan_fn`, `selective_state_update`, and training-only `mamba_inner_fn`. The slow fallback is a Python/PyTorch loop over sequence length and is not a production inference path.

## 7. Position encoding and custom math

There is no explicit absolute or rotary position encoding. Sequence order enters through causal convolution and the recurrent SSM.

Core recurrence to reproduce:

```python
def mamba_scan(u, gate, time_step, Bp, Cp, A_log, D, dt_bias):
    A = -torch.exp(A_log.float())                       # [I, N]
    dt = torch.nn.functional.softplus(time_step + dt_bias[:, None])
    dA = torch.exp(A[None, :, None, :] * dt[:, :, :, None])
    dB_u = (dt[:, :, :, None] * Bp[:, None, :, :].float()) * u[:, :, :, None].float()
    state = zeros([batch, intermediate, state_size])
    ys = []
    for t in range(seq):
        state = dA[:, :, t, :] * state + dB_u[:, :, t, :]
        y = (state.to(u.dtype) @ Cp[:, t, :, None]).squeeze(-1)
        ys.append(y)
    y = torch.stack(ys, dim=-1) + u * D[None, :, None]
    return y * torch.nn.functional.silu(gate)
```

`A_log`, `D`, convolution weights, and projection weights are static parameters. `time_step`, `B`, and `C` are input-dependent and must be computed per token.

## 8. Preprocessing and input packing

Runtime inputs are standard language model tensors:

- `input_ids`: `[batch, seq]` integer token IDs, or `inputs_embeds`: `[batch, seq, hidden]`; exactly one is required.
- `attention_mask`: optional `[batch, seq]`, multiplicatively applied to the mixer hidden stream. In cached generation it is used for the first iteration and cleared for later decode steps by `prepare_inputs_for_generation`.
- No token type IDs, position IDs, packed sequence descriptors, cu-seqlens, multimodal placeholders, image/audio preprocessors, or indexed embedding stitch are required.

End-to-end generation parity still depends on the standard Transformers generation controller for sampling/beam search. Core module parity can ignore beam search initially, but cache reorder must be validated before beam support.

## 9. Graph rewrite / lowering opportunities

### Rewrite: depthwise causal Conv1d prefill -> channelwise causal convolution kernel

Source pattern:

```text
Conv1d(in_channels=I, out_channels=I, groups=I, kernel=K, padding=K-1)(u)[:, :, :S] -> SiLU
```

Replacement:

```text
DepthwiseCausalConv1d([B,I,S], weight[I,K], bias[I]) -> SiLU -> [B,I,S]
```

Preconditions:

- `groups == in_channels == out_channels == intermediate_size`.
- `padding == conv_kernel - 1`.
- Output is truncated to original sequence length.
- Activation is exactly config `hidden_act`, public configs use `silu`.
- Preserve `[B, I, S]` channel-first local layout.

Failure cases: non-depthwise conv, non-SiLU activation if not implemented, sequence layout already fused into another op without a compatible local layout.

Parity test sketch: compare full sequence conv outputs and final cached conv state for random `[B,S,H]` after `in_proj`.

### Rewrite: decode conv update -> rolling state update kernel

Source pattern:

```text
conv_state = update_conv_state(u_t)
y_t = sum(conv_state * weight[:, 0, :], dim=-1) + bias
y_t = silu(y_t)
```

Replacement:

```text
StatefulDepthwiseCausalConvUpdate(u_t, conv_state, weight, bias, activation)
```

Preconditions: one-token decode or bounded multi-token update with identical roll/truncate semantics; state tensor shape `[B,I,K]`.

Failure cases: prefill shorter than `conv_kernel` has a known source caveat in cache roll behavior; first integration should test `S >= K`.

### Rewrite: selective scan -> provider-backed SSM scan

Source pattern:

```text
x_proj -> split(dt,B,C) -> dt_proj/softplus -> exp(A*dt) recurrence -> D skip -> SiLU gate
```

Replacement:

```text
SelectiveScanMamba(u, gate, x_proj_weight, dt_proj_weight/bias, A_log, D) -> y, last_state
```

Preconditions:

- `state_size`, `time_step_rank`, `intermediate_size` static per artifact.
- `A` is diagonal per intermediate channel, represented by `A_log`.
- `B` and `C` are input-dependent vectors from `x_proj`, not learned static matrices.
- Return/update last recurrent state when `use_cache=True`.
- Preserve fp32 math for `A`, `D`, `dt_bias`, exponentials, and softplus behavior.

Failure cases: missing optimized scan provider should not silently lower to a sequential Python-style loop for production sequence lengths.

Parity test sketch: compare slow recurrence, associative scan if available, and optimized provider on random tensors for fp32/fp16 with fixed seeds.

### Rewrite: last-token-only logits

Source pattern:

```text
logits = lm_head(hidden_states[:, slice_indices, :]).float()
```

Replacement:

```text
Slice last K hidden tokens before GEMM; for decode K=1.
```

Preconditions: `logits_to_keep` is integer or static tensor index set; no loss computation.

Failure cases: full-sequence logits requested for perplexity or training.

## 10. Kernel fusion candidates

Highest priority:

- RMSNorm + residual dtype handling: every layer uses pre-norm and residual add.
- Depthwise causal conv prefill and decode update: required for Mamba performance and cache correctness.
- Selective scan prefill: dominant sequence-length-dependent kernel; needs provider-level implementation rather than generic elementwise loops.
- Selective state update decode: core token/sec path with fixed-size recurrent cache.
- Fused Mamba mixer prefill: `in_proj`, depthwise conv, `x_proj/dt_proj`, scan, gate, and `out_proj` have strong fusion/staging opportunities, but should be built after standalone parity.

Medium priority:

- `x_proj + split + dt_proj` fusion to reduce intermediate writes.
- Gate SiLU multiply fused into scan output.
- Last-token-only LM head GEMM.
- Cache update kernels with static addresses suitable for CUDA graphs.

Lower priority:

- Training-only `mamba_inner_fn`.
- mambapy fallback.
- Beam reorder; required for beam search but not first greedy decode.
- Attention-style optimizations; not applicable.

## 11. Runtime staging plan

Stage 1: Parse `MambaConfig`, load tied embedding/LM head weights, and instantiate one Mamba block with static shapes.

Stage 2: Implement reference CPU or CUDA-debug Mamba block parity using explicit ops and the slow recurrence for short sequences.

Stage 3: Add provider-backed depthwise causal Conv1d prefill and decode update, with conv state materialization.

Stage 4: Add provider-backed selective scan prefill returning final recurrent state.

Stage 5: Add one-token decode with `selective_state_update`, conv state update, recurrent state mutation, and last-token logits.

Stage 6: Run full-model prefill logits parity for `mamba-130m-hf`, then decode token-by-token parity against Transformers.

Stage 7: Optimize/fuse scan, projection staging, cache residency, and CUDA graph capture. Stub training loss, gradient checkpointing, `mambapy`, and beam search until core greedy inference is stable.

## 12. Parity and validation plan

- Config parity: assert computed `intermediate_size`, `time_step_rank`, layer count, and tied embedding/LM head alias against Transformers.
- RMSNorm random tensor parity in fp32/fp16/bf16; tolerance fp32 `1e-5`, fp16/bf16 around `1e-2` depending on accumulation.
- Depthwise causal conv parity for prefill across `S < K`, `S == K`, and `S > K`; include attention-mask multiplication.
- Decode conv state parity for repeated one-token updates versus full prefill, especially cache contents.
- Selective scan random tensor parity against slow source recurrence for short sequences.
- Selective state update parity: one step from a saved recurrent state must match the final step of full scan.
- Single-block parity with random weights, then first N layers of `mamba-130m-hf`.
- Prefill logits parity on short prompts with `use_cache=False` and `use_cache=True`.
- Decode parity: prefill a prompt, decode several tokens greedily, compare logits and generated IDs.
- Cache reorder/reset parity before beam search.

## 13. Performance probes

- Prefill throughput sweep: batch sizes 1, 4, 16 and sequence lengths 64, 512, 2048.
- Decode tokens/sec sweep: batch sizes 1, 4, 16, 64 with one-token updates.
- Selective scan kernel time vs projection GEMMs and depthwise conv time.
- Cache memory: `layers * batch * intermediate * (conv_kernel + state_size) * dtype_size`.
- Last-token logits GEMM cost for full vocab, with `logits_to_keep=1`.
- Fast kernel path versus slow fallback to quantify the non-production gap.
- CUDA graph capture probe with static-address cache tensors.

## 14. Skip/defer list

- Training, loss, and gradient checkpointing.
- `mamba_inner_fn` training fused path.
- `mambapy` fallback.
- Beam search until cache reorder is validated.
- CPU production performance for long sequences.
- Quantization and GGUF loading.
- Multi-GPU/tensor parallel sharding.
- Any attention/KV-cache implementation work; it is unrelated to Mamba.

## 15. Final implementation checklist

- [ ] Parse `MambaConfig`, including computed `intermediate_size` and `time_step_rank`.
- [ ] Load Mamba weights and preserve tied `lm_head.weight` / embedding alias.
- [ ] Implement RMSNorm with fp32 variance.
- [ ] Implement depthwise causal Conv1d prefill.
- [ ] Implement conv state cache `[B, I, conv_kernel]`.
- [ ] Implement decode conv update with source-compatible roll/truncate behavior.
- [ ] Implement `x_proj`, `dt_proj`, split, `A=-exp(A_log)`, softplus dt math.
- [ ] Implement provider-backed selective scan returning final recurrent state.
- [ ] Implement recurrent state cache `[B, I, state_size]`.
- [ ] Implement one-token `selective_state_update`.
- [ ] Implement residual fp32 option and output projection.
- [ ] Implement final RMSNorm and last-token-only tied LM head.
- [ ] Add one-block, prefill logits, and decode parity tests.
- [ ] Add performance probes for scan, conv update, and decode token/sec.
- [ ] Reject or route unsupported config/remote-code fields that current source does not consume.
