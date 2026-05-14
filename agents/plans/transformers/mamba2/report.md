# Mamba2 Transformers Audit

## 1. Source basis

Transformers commit/version: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`.

Model id: primary native family `mamba2`; representative public checkpoints inspected from `state-spaces/mamba2-130m`, `state-spaces/mamba2-370m`, `state-spaces/mamba2-780m`, `state-spaces/mamba2-1.3b`, `state-spaces/mamba2-2.7b`, plus `state-spaces/mamba2attn-2.7b` as an out-of-scope hybrid variation.

Config source: local Transformers `src/transformers/models/mamba2/configuration_mamba2.py`; HF config snapshots saved under `agents/plans/transformers/mamba2/_sources/`; representative HF raw URLs:

- https://huggingface.co/state-spaces/mamba2-130m/raw/main/config.json
- https://huggingface.co/state-spaces/mamba2-370m/raw/main/config.json
- https://huggingface.co/state-spaces/mamba2-780m/raw/main/config.json
- https://huggingface.co/state-spaces/mamba2-1.3b/raw/main/config.json
- https://huggingface.co/state-spaces/mamba2-2.7b/raw/main/config.json
- https://huggingface.co/state-spaces/mamba2attn-2.7b/raw/main/config.json

Source files inspected:

- `X:/H/transformers/src/transformers/models/mamba2/modeling_mamba2.py`
- `X:/H/transformers/src/transformers/models/mamba2/configuration_mamba2.py`
- `X:/H/transformers/src/transformers/models/mamba2/convert_mamba2_ssm_checkpoint_to_pytorch.py`
- `X:/H/transformers/src/transformers/cache_utils.py`, around `LinearAttentionLayer`.
- GitHub source URLs at the inspected commit:
  - https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/mamba2/modeling_mamba2.py
  - https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/mamba2/configuration_mamba2.py
  - https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/mamba2/convert_mamba2_ssm_checkpoint_to_pytorch.py

Any missing files or assumptions: no processor/tokenizer source is model-coupled for the native runtime graph. The public `state-spaces/mamba2-*` configs use the original `mamba_ssm` schema (`d_model`, `n_layer`, `ssm_cfg`) and omit `model_type`; the conversion script maps them into native `Mamba2Config`. The `mamba2attn-2.7b` config advertises attention layers, but the inspected native `mamba2` source always returns `["mamba"] * num_hidden_layers` and does not implement `attn_layer_idx`, so that checkpoint should be rejected or audited separately.

## 2. High-level architecture

Runtime target: causal language modeling with `Mamba2ForCausalLM`.

Architecture: text-only autoregressive decoder-like SSM stack. There is no Transformer self-attention in the native source. Each layer is an RMSNorm-preconditioned Mamba2 selective state-space block with a depthwise causal convolution front-end, grouped `B/C` state projections, SSD chunk scan for prefill, recurrent state update for decode, gated RMSNorm, and output projection.

Dataflow:

```text
input_ids -> token embedding -> repeated Mamba2 blocks -> final RMSNorm -> LM head -> logits/sampling
```

Stage decomposition:

- CPU/data pipeline: tokenization, left/right padding policy, `attention_mask` construction.
- GPU prefill: embedding, all Mamba2 blocks over `[batch, seq]`, final norm, optional last-token-only logits.
- Decode: one token at a time, using per-layer convolution states and recurrent SSM states. There is no KV cache.
- Independently optimizable regions: embedding/LM head, per-layer RMSNorm, per-layer `in_proj`, causal depthwise conv, SSD chunk scan, decode state update, gated RMSNorm, `out_proj`.

## 3. Important config dimensions

Native `Mamba2Config` defaults from source:

| Field | Default | Runtime significance |
| --- | ---: | --- |
| `hidden_size` | 4096 | Token/state width. |
| `expand` | 2 | `intermediate_size = hidden_size * expand`. |
| `num_heads` | 128 | SSM heads, not attention heads. |
| `head_dim` | 64 | `intermediate_size == num_heads * head_dim` must hold. |
| `state_size` | 128 | Per-head recurrent SSM state length. |
| `num_hidden_layers` | 64 | Number of Mamba blocks. |
| `n_groups` | 8 | Number of groups for input-dependent `B/C` states. |
| `conv_kernel` | 4 | Causal depthwise convolution state length. |
| `chunk_size` | 256 | SSD prefill chunk length. |
| `vocab_size` | 32768 | Embedding and LM-head rows. |
| `hidden_act` | `silu` | Conv branch activation; fast path assumes `silu`/`swish` for causal-conv kernel. |
| `use_bias` | `False` | Bias on `in_proj` and `out_proj`. |
| `use_conv_bias` | `True` | Bias on depthwise conv. |
| `rms_norm` | `True` | Source always constructs RMSNorm modules. |
| `residual_in_fp32` | `True` | Residual path upcast before addition. |
| `time_step_limit` | `(0.0, inf)` | Optional clamp on softplus time steps. |
| `use_cache` | `True` | Uses `DynamicCache` with linear-attention cache layers. |
| `tie_word_embeddings` | `False` | Native default; official SSM configs set `tie_embeddings: true` before conversion. |

Representative checkpoint sweep:

| Checkpoint config | Config schema | Hidden | Layers | Effective SSM heads | Head dim | Intermediate | Groups | Raw/padded vocab | Tie embeddings |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |
| `state-spaces/mamba2-130m` | original SSM | 768 | 24 | 24 | 64 | 1536 | converter default `1` | 50277 / 50288 | true |
| `state-spaces/mamba2-370m` | original SSM | 1024 | 48 | 32 | 64 | 2048 | converter default `1` | 50277 / 50288 | true |
| `state-spaces/mamba2-780m` | original SSM | 1536 | 48 | 48 | 64 | 3072 | converter default `1` | 50277 / 50288 | true |
| `state-spaces/mamba2-1.3b` | original SSM | 2048 | 48 | 64 | 64 | 4096 | converter default `1` | 50277 / 50288 | true |
| `state-spaces/mamba2-2.7b` | original SSM | 2560 | 64 | 80 | 64 | 5120 | converter default `1` | 50277 / 50288 | true |
| `state-spaces/mamba2attn-2.7b` | original hybrid | 2560 | 64 | Mamba plus advertised attention | attention cfg uses 30 heads, head dim 128 | 5120 for Mamba | converter default `1` for Mamba | 50277 / 50288 | true |

The sweep values above are config-derived plus conversion-script-derived. The original SSM configs omit `head_dim`, `expand`, `state_size`, `conv_kernel`, and `chunk_size`; effective values come from native defaults used by `convert_mamba2_ssm_checkpoint_to_pytorch.py`.

## 3a. Family variation traps

- `num_heads` are SSM heads, not attention heads. Do not route this family through attention lowering.
- `hidden_size * expand == num_heads * head_dim` is a hard native validation rule.
- `B` and `C` are grouped by `n_groups`; naive fallback expands them to all heads with `repeat_interleave(num_heads // n_groups)`. Kernels must either consume grouped `B/C` directly or reproduce the exact expansion.
- Public `state-spaces/mamba2-*` configs are not native `Mamba2Config` JSON. They need conversion or a loader shim.
- `mamba2attn-2.7b` advertises attention layers and RoPE in config, but the native source inspected here does not implement them.
- The `in_proj` split includes two zero-length `d_mlp` slices in this source because `projection_size = 2 * intermediate + 2 * n_groups * state_size + num_heads`.
- `attention_mask` is not an attention mask; it masks hidden states by multiplication in selected prefill paths and is dropped after the first generation iteration.
- Fast path depends on optional `mamba-ssm` Triton kernels and `causal-conv1d`. Missing kernels fall back to a naive PyTorch scan that is likely too slow for production.
- Decode cache is `[conv_states, recurrent_states]`, not KV cache. Beam reorder only index-selects batch dimension.
- `conv_kernel=4` is assumed by cache comments; shorter-than-kernel prefill has an acknowledged approximation in `LinearAttentionLayer.update_conv_state`.
- Layout is sequence-major `[batch, seq, channel]` for projections and `[batch, channel, seq]` only inside Conv1d. A layout pass must guard all `transpose(1, 2)`, split, chunk, and cache shape assumptions.

## 4. Operator coverage checklist

Tensor/layout ops:

- Embedding lookup: `[batch, seq] -> [batch, seq, hidden]`.
- Slice/select for `logits_to_keep`: `hidden_states[:, slice_indices, :]`.
- `transpose(1, 2)` around causal Conv1d.
- `split` with sizes `[0, 0, intermediate, conv_dim, num_heads]` for standard configs.
- `reshape/view` between `[B, L, intermediate]` and `[B, L, num_heads, head_dim]`.
- `reshape/view` for grouped states `[B, L, n_groups, state_size]`.
- `pad` on sequence/cache axes, crop `[..., :seq_len]`.
- `permute`, `cat`, lower-triangular masks, `masked_fill`, `cumsum`, `repeat_interleave`, `expand`, `contiguous`.

Neural network primitives:

- Token embedding and LM head. If `tie_word_embeddings=True`, `lm_head.weight` and embedding weight must remain one logical parameter.
- RMSNorm over hidden width, fp32 variance, output cast to input dtype.
- Gated RMSNorm: `RMSNorm(hidden * silu(gate))` over `intermediate_size`.
- Linear `in_proj`: `hidden_size -> 2 * intermediate + 2 * n_groups * state_size + num_heads`.
- Depthwise causal Conv1d: channels `conv_dim = intermediate + 2 * n_groups * state_size`, groups `conv_dim`, kernel `conv_kernel`, padding `conv_kernel - 1`, optional bias, activation.
- Linear `out_proj`: `intermediate_size -> hidden_size`.
- Residual add with optional fp32 residual.

Selective SSM / scan primitives:

- `softplus(dt + dt_bias)` and optional clamp.
- `A = -exp(A_log.float())`.
- Per-token discretization of `x`, `A`, and `B`.
- SSD chunk scan with `chunk_size`, segmented cumulative sums, exponentials, low-rank diagonal/off-diagonal contractions.
- Decode recurrence: `state = state * exp(dt * A) + dt * B * x`, then `y = state @ C + D * x`.
- Optional optimized calls: `mamba_chunk_scan_combined`, `mamba_split_conv1d_scan_combined`, `selective_state_update`.

Generation/cache ops:

- `DynamicCache(config=Mamba2Config)` with one linear-attention cache layer per model layer.
- Per-layer `conv_states`: `[batch, conv_dim, conv_kernel]`.
- Per-layer `recurrent_states`: `[batch, num_heads, head_dim, state_size]`.
- Cache update with roll/truncate for conv states and copy update for recurrent states.
- Cache reset, reorder for beam search, offload/prefetch hooks.

Preprocessing-coupled ops:

- Tokenizer/padding is external. GPU graph sees `input_ids` or `inputs_embeds` and optional 2D `attention_mask`.
- `attention_mask` must be shape `[batch, seq]` and is applied only when both batch and sequence dimensions are greater than one.

## 5. Layer/block breakdown

Model:

```text
input_ids -> Embedding(vocab, hidden)
repeat num_hidden_layers:
  residual = x
  x = RMSNorm(x)
  if residual_in_fp32: residual = residual.float()
  x = Mamba2Mixer(x, cache, attention_mask)
  x = residual + x
x = final RMSNorm(x)
logits = Linear(hidden -> vocab)(selected sequence positions).float()
```

Mamba2Mixer, prefill source shape path:

```text
x: [B, L, HIDDEN]
x = x * attention_mask[:, :, None] when source condition applies
p = in_proj(x): [B, L, 2*I + 2*G*S + NH]
_, _, gate, conv_in, dt = split(p, [0, 0, I, I + 2*G*S, NH])
conv_in_t = conv_in.transpose(1, 2): [B, I + 2*G*S, L]
conv_out = depthwise_causal_conv(conv_in_t, kernel=K)[:, :, :L].transpose(1, 2)
conv_out = activation(conv_out)
hidden, B_state, C_state = split(conv_out, [I, G*S, G*S])
y = SSD_scan(hidden.reshape(B,L,NH,HD), dt, A_log, B_state.reshape(B,L,G,S), C_state.reshape(B,L,G,S), D)
y = GatedRMSNorm(y.reshape(B,L,I), gate)
out = out_proj(y): [B, L, HIDDEN]
```

Decode source shape path:

```text
p = in_proj(x[:, -1:, :])
_, _, gate, conv_in, dt = split(p.squeeze(1), [0, 0, I, I + 2*G*S, NH])
conv_state = roll/update per layer, shape [B, I + 2*G*S, K]
conv_out = activation(sum(conv_state * conv_weight, dim=-1) + conv_bias)
hidden, B_state, C_state = split(conv_out, [I, G*S, G*S])
dt = softplus(dt + dt_bias), expanded to [B, NH, HD]
B_state/C_state expand G groups to NH heads
recurrent = recurrent * exp(dt * A) + dt * B_state * hidden
y = bmm(recurrent.reshape(B*NH, HD, S), C.reshape(B*NH, S, 1))
y = y.reshape(B, NH, HD) + D * hidden
y = GatedRMSNorm(y.reshape(B, 1, I), gate)
out = out_proj(y)
```

Symbols: `I = hidden_size * expand`, `NH = num_heads`, `HD = head_dim`, `G = n_groups`, `S = state_size`, `K = conv_kernel`.

Projection biases: `in_proj` and `out_proj` use `config.use_bias`; defaults and official converted configs use no bias. Conv bias uses `config.use_conv_bias`; default true.

## 6. Attention requirements

No Transformer attention is required for the native `mamba2` family. There is no MHA/MQA/GQA, no attention mask addition, no RoPE, no ALiBi, no sliding-window attention, and no KV cache in the inspected source.

The causal dependency is implemented by:

- Causal depthwise convolution over the projected hidden/B/C stream.
- SSD selective state-space scan in prefill.
- Per-layer recurrent state update in decode.

Cache shapes:

- Convolution cache before/after update: `[batch, conv_dim, conv_kernel]`, where `conv_dim = intermediate + 2 * n_groups * state_size`.
- Recurrent SSM cache before/after update: `[batch, num_heads, head_dim, state_size]`.
- Cached states are after convolution/scan initialization, not position-encoded keys/values.

Optimized backend dispatch:

- Prefill CUDA fast path calls `causal_conv1d_fn` and `mamba_chunk_scan_combined`; training without cache can call `mamba_split_conv1d_scan_combined`.
- Decode CUDA fast path calls `causal_conv1d_update` and `selective_state_update`.
- Fallback path uses PyTorch Conv1d plus explicit chunk/scan math. This is useful as a correctness reference, not as a production lowering target.

## 7. Position encoding and custom math

There is no RoPE or learned absolute position embedding in native `Mamba2Model`. Sequence order enters through causal convolution and recurrent SSM dynamics.

Core decode recurrence:

```python
def mamba2_decode_step(x, dt, B, C, state, A_log, D, dt_bias):
    A = -torch.exp(A_log.float())                  # [heads]
    dt = torch.nn.functional.softplus(dt + dt_bias)
    dA = torch.exp(dt[..., None] * A[..., None, None])
    dB_x = dt[..., None] * B[..., None, :] * x[..., None]
    state = state * dA + dB_x                      # [batch, heads, head_dim, state]
    y = torch.matmul(state, C[..., None]).squeeze(-1)
    y = y + D[..., None] * x
    return y, state
```

Core prefill chunk math, paraphrased from source:

```python
def segment_sum(a):
    # a: [..., chunk]
    expanded = a[..., None].expand(*a.shape, a.shape[-1])
    lower = torch.tril(torch.ones(a.shape[-1], a.shape[-1], dtype=torch.bool), diagonal=-1)
    seg = torch.cumsum(expanded.masked_fill(~lower, 0), dim=-2)
    return seg.masked_fill(~torch.tril(torch.ones_like(lower), diagonal=0), -torch.inf)
```

Precomputable:

- `A = -exp(A_log.float())`.
- Broadcasted `D` and `dt_bias` shapes for fixed model dimensions.
- Lower-triangular masks for each `chunk_size`.

Dynamic:

- `dt`, `B`, `C`, and `hidden` are input-dependent.
- `pad_size = (chunk_size - seq_len % chunk_size) % chunk_size`.
- Recurrent and convolution cache contents.

## 8. Preprocessing and input packing

Inputs:

- `input_ids`: `[batch, seq]` integer token IDs, or `inputs_embeds`: `[batch, seq, hidden]`.
- Exactly one of `input_ids` or `inputs_embeds` is required.
- `attention_mask`: optional `[batch, seq]`; source uses it to zero hidden states, not to construct attention logits.

Generation controller behavior:

- `prepare_inputs_for_generation` delegates to the generic generation helper, then sets `attention_mask=None` on non-first iterations when cache is used.
- `logits_to_keep` may be an integer or tensor of indices; first integration can support integer `1` for last-token-only logits and full logits for parity.

CPU/data-pipeline work:

- Tokenization, special tokens, padding side, and prompt construction are outside this model source.
- The conversion script uses `GPTNeoXTokenizerFast` for `mamba_ssm` checkpoints and `LlamaTokenizerFast` for `codestral` conversion if a tokenizer model path is supplied.

GPU/runtime work:

- Embedding lookup, optional hidden masking, SSM stack, final norm, selected LM-head projection.

## 9. Graph rewrite / lowering opportunities

### Rewrite: depthwise causal Conv1d to specialized causal FIR

Source pattern:

```text
Conv1d(groups=channels, kernel_size=K, padding=K-1)(x.transpose(1,2))[..., :seq].transpose(1,2)
```

Replacement:

```text
CausalDepthwiseConv1dBSC(x, weight.squeeze(1), bias, K) -> activation
```

Preconditions:

- `groups == in_channels == out_channels == conv_dim`.
- `stride == 1`, `dilation == 1`.
- Padding exactly `K - 1`, followed by crop to original sequence length.
- Source layout `[B, L, C]` at graph boundary.

Shape equations:

- Input `[B, L, conv_dim]`; output `[B, L, conv_dim]`.
- Decode state `[B, conv_dim, K]`.

Weight transform:

```python
w = conv1d.weight.squeeze(1)  # [conv_dim, K]
```

Failure cases:

- Non-SiLU activation can still use the rewrite, but cannot call `causal_conv1d_fn` with the same fast activation assumptions.
- Layout-translated regions must preserve causal sequence order.

Parity test sketch: compare prefill and one-step decode against PyTorch Conv1d for random `B,L,C,K`, including `L < K`, `L == K`, and `L > K`.

### Rewrite: grouped B/C expansion elimination

Source pattern:

```text
B/C: [B, L, G, S] -> repeat_interleave(num_heads // G, dim=2) -> [B, L, NH, S]
```

Replacement:

```text
SSD scan consumes grouped B/C with head_to_group = head // (NH/G)
```

Preconditions:

- `num_heads % n_groups == 0`.
- Group replication order matches `repeat_interleave`.
- Kernel implements grouped indexing for both prefill and decode.

Shape equations:

- `B_grouped`, `C_grouped`: `[B, L, G, S]`.
- Logical per-head values: group index `h // (NH / G)`.

Failure cases:

- Configs with invalid divisibility should be rejected.
- Any future nonuniform group mapping needs a separate audit.

Parity test sketch: random grouped `B/C`, compare grouped-kernel output to explicit repeat fallback.

### Rewrite: SSD prefill to provider-backed chunk scan

Source pattern:

```text
pad -> chunk reshape -> cumsum/segment_sum/exp -> diagonal and off-diagonal contractions -> final state
```

Replacement:

```text
Mamba2ChunkScan(x, dt, A, B_grouped, C_grouped, D, dt_bias, chunk_size, dt_limit)
```

Preconditions:

- Source semantic layout `[B, L, NH, HD]` for `x`.
- `B/C` grouped as `[B, L, G, S]`.
- `A_log`, `D`, and `dt_bias` are per-head.
- `chunk_size` is fixed in config for the compiled artifact or guarded as a runtime constant.
- Output final recurrent state shape `[B, NH, HD, S]`.

Failure cases:

- Dynamic sequence length must handle padding and crop exactly.
- `time_step_limit != (0, inf)` requires clamp support.
- Fallback PyTorch implementation materializes large `[chunk, chunk]` tensors; do not lower that literally for production.

Parity test sketch: compare provider scan against `torch_forward` for short sequences, non-multiple chunk lengths, and multiple `n_groups`.

### Rewrite: gated RMSNorm fusion

Source pattern:

```text
hidden = hidden * silu(gate.float())
hidden = hidden.float()
hidden = hidden * rsqrt(mean(hidden ** 2, dim=-1, keepdim=True) + eps)
out = weight * hidden.to(input_dtype)
```

Replacement:

```text
GatedRMSNorm(hidden, gate, weight, eps)
```

Preconditions:

- Reduction axis is last dimension.
- Gate shape matches hidden shape.
- Accumulation in fp32.

Failure cases:

- Do not commute gate after normalization.
- Preserve output dtype cast and weight dtype behavior.

Parity test sketch: fp32/fp16/bf16 random tensors with large/small magnitudes.

### Rewrite: last-token-only logits

Source pattern:

```text
lm_head(hidden_states[:, -1:, :]).float()
```

Replacement:

```text
SliceLastToken -> GEMM(hidden -> vocab)
```

Preconditions:

- `logits_to_keep == 1` or equivalent last-position index.
- Generation path only needs next-token logits.

Failure cases:

- Training/loss and prompt scoring need full or indexed logits.

Parity test sketch: compare full logits slice with last-token-only lowering.

## 10. Kernel fusion candidates

Highest priority:

- Provider-backed Mamba2 chunk scan. This is the core prefill bottleneck and should consume grouped `B/C` without materializing repeat-expanded states.
- Decode selective state update. One-token generation is dominated by per-layer recurrent update plus small matrix/vector contractions.
- Causal depthwise conv update. Decode requires efficient rolling state update over `[B, conv_dim, K]`.
- RMSNorm and gated RMSNorm. Both are present in every layer; gated form includes SiLU and multiplication before variance.
- Linear projection families. `in_proj` and `out_proj` dominate dense math around the scan and should use existing GEMM/CUTLASS paths.

Medium priority:

- Conv + split fusion: produce hidden/B/C directly from the activated convolution stream.
- `in_proj + conv` staging fusion for prefill: reduce layout transposes and temporary writes where kernel boundaries allow.
- Last-token-only LM head for decode.
- Residual-add + RMSNorm preblock fusion, with fp32 residual handling.

Lower priority:

- Full training fast path `mamba_split_conv1d_scan_combined`; inference can ignore initially.
- Beam cache reorder optimization; correctness first with index-select.
- Cache offload/prefetch hooks; useful later for long-running serving, not first parity.

## 11. Runtime staging plan

Stage 1: parse native and converted configs. Add a loader shim for original `state-spaces/mamba2-*` configs or require pre-converted HF config. Reject `mamba2attn-*` for this report.

Stage 2: load weights and run embedding, RMSNorm, `in_proj`, depthwise conv, split, and `out_proj` for one layer against PyTorch.

Stage 3: implement naive/reference Mamba2 SSD prefill as an internal correctness op for small shapes only. Validate chunk padding, grouped `B/C`, `dt_bias`, `D`, `A_log`, and final recurrent state.

Stage 4: add provider-backed prefill chunk scan and replace the reference op under strict shape/config guards.

Stage 5: implement decode caches: conv state `[B, conv_dim, K]`, recurrent state `[B, NH, HD, S]`, one-token selective update, and attention-mask drop after first iteration.

Stage 6: integrate `Mamba2ForCausalLM` with last-token logits and tied embedding handling.

Stage 7: add fusions and scheduling: Conv/update fusion, gated RMSNorm, scan provider profiling, chunk-size/sequence-length sweeps, continuous batching cache layout.

Initial stubs: loss, training-only fused kernel, gradient checkpointing, cache offload, beam search reorder, non-last-token indexed logits.

## 12. Parity and validation plan

- Random tensor parity for `Mamba2RMSNorm` and `MambaRMSNormGated`; tolerances `1e-5` fp32, `2e-3` fp16/bf16.
- Depthwise causal conv parity for prefill and decode cache update, including `L` not divisible by `chunk_size` and `L < conv_kernel`.
- SSD chunk scan parity for small configs against `torch_forward`; use `chunk_size` 4/8 in synthetic configs to keep reference tensors small.
- Grouped `B/C` parity for `n_groups=1`, `n_groups=8`, and `n_groups=num_heads`.
- Single-block parity with random weights, no cache.
- Single-block parity with cache: prefill then decode one token, compare to full-sequence no-cache output at the appended position.
- After-N-layer parity for 2, then full layer count for a small checkpoint/config.
- Causal LM prefill logits parity, including `logits_to_keep=0` and `logits_to_keep=1`.
- Decode token parity over several generated steps with cache reuse and attention mask dropped after first iteration.
- Weight alias parity when `tie_word_embeddings=True`.

Recommended tolerances: fp32 `atol=1e-4, rtol=1e-4` for full blocks because scan uses exponentials; fp16/bf16 `atol=3e-2, rtol=3e-2` initially, then tighten after fused kernels match source dtype/upcast order.

## 13. Performance probes

- Prefill-only tokens/sec by sequence length: `L = 128, 256, 512, 1024, 2048, 4096`.
- Decode-only tokens/sec by batch size with warmed caches.
- Chunk-size sensitivity for provider scan, especially default `256`.
- Conv update bandwidth for `[B, conv_dim, 4]`.
- Recurrent cache memory: `layers * batch * num_heads * head_dim * state_size * dtype_size`, plus conv cache.
- GEMM time for `in_proj`, `out_proj`, and LM head separately.
- Last-token-only LM head versus full logits.
- Fast provider scan versus reference fallback on small shapes to catch launch overhead and numerical drift.
- Cache reorder/reset overhead for beam-style generation, even if beam search is deferred.

## 14. Skip/defer list

- Training and `mamba_split_conv1d_scan_combined` training-only path.
- Loss computation.
- Gradient checkpointing.
- Hybrid `mamba2attn-*` checkpoints.
- Beam search optimization beyond correctness of `reorder_cache`.
- Cache offload/prefetch scheduling.
- Quantization and GGUF ingestion.
- Multi-GPU tensor parallelism.
- Remote-code sequence classification variants not in native source.
- Arbitrary activation functions beyond `silu`/`swish` fast path for first optimized kernel.

## 15. Final implementation checklist

- [ ] Parse native `Mamba2Config`.
- [ ] Add or document conversion for original `state-spaces/mamba2-*` configs.
- [ ] Reject `mamba2attn-*` configs in this native-family path.
- [ ] Load embedding, LM head, RMSNorm, `in_proj`, conv, `A_log`, `D`, `dt_bias`, `out_proj`.
- [ ] Preserve tied embedding/LM-head alias when config requests it.
- [ ] Implement RMSNorm and gated RMSNorm parity.
- [ ] Implement causal depthwise Conv1d prefill and decode update.
- [ ] Implement grouped `B/C` shape handling.
- [ ] Implement small-shape reference SSD chunk scan.
- [ ] Implement provider-backed Mamba2 chunk scan.
- [ ] Implement one-token selective state update.
- [ ] Implement conv and recurrent cache allocation/update/reset/reorder.
- [ ] Add single-layer prefill parity.
- [ ] Add prefill-plus-decode cache parity.
- [ ] Add full causal LM logits parity with `logits_to_keep`.
- [ ] Benchmark prefill, decode, scan, conv update, and LM head separately.
