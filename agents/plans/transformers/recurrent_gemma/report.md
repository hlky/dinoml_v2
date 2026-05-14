# RecurrentGemma Audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: google/recurrentgemma-2b as documented/tested; google/recurrentgemma-{2b,2b-it,9b,9b-it} configs attempted but gated
Config source: local RecurrentGemmaConfig plus converter presets; official HF config.json fetches returned 401 Unauthorized
Source files inspected: configuration_recurrent_gemma.py, modeling_recurrent_gemma.py, convert_recurrent_gemma_to_hf.py, docs/source/en/model_doc/recurrent_gemma.md, tests/models/recurrent_gemma/test_modeling_recurrent_gemma.py
Any missing files or assumptions: official gated Google config.json and checkpoint metadata were not available; no imports or model tests were run
```

See `_sources/source_notes.md` for hashes, line anchors, and gated fetch results.

## 2. High-level architecture

RecurrentGemma is a text-only causal decoder using the Griffin/Hawk style hybrid of linear recurrence and local self-attention. A decoder layer contains one temporal block, selected by `layers_block_type`, followed by a gated MLP. The default block pattern repeats `recurrent, recurrent, attention`, so the inspectable 26-layer default has 18 recurrent layers and 8 local-attention layers.

```text
token ids/embeds -> embedding scale -> hybrid decoder prefill -> recurrent/session state + local KV cache -> decode -> final RMSNorm -> tied LM head -> tanh logits soft cap -> sampling
```

Independent stages for DinoML:

- CPU/tokenizer: Gemma tokenizer, padding side, BOS/EOS/PAD policy.
- Prefill: full sequence through recurrent Conv1d/RG-LRU scans and sliding-window attention.
- Decode: one-token recurrent state update plus attention KV cache update.
- Logits: optional last-token-only `logits_to_keep`, tied LM head, tanh soft cap.

The important runtime distinction is that recurrent state is not ordinary `past_key_values`: it is mutable module state in Transformers and must become explicit DinoML session state.

## 3. Important config dimensions

| Field | Source default | Converter `2B` preset | Runtime significance |
|---|---:|---:|---|
| `vocab_size` | 256000 | 256000 | Embedding and LM-head rows. |
| `hidden_size` | 2560 | 2560 | Residual width. |
| `num_hidden_layers` | 26 | 26 | Layer count; block pattern is repeated/truncated. |
| `block_types` | `["recurrent","recurrent","attention"]` | default | Temporal block schedule. |
| `lru_width` | `None -> hidden_size` | default | Recurrent block channel width and RG-LRU state width. |
| `num_attention_heads` | 10 | 10 | Query heads and RG-LRU gate groups. |
| `num_key_value_heads` | `None -> 10` | 1 | MHA by default, MQA in converter 2B preset. |
| `head_dim` | 256 | 256 | `hidden_size // num_attention_heads`. |
| `intermediate_size` | 7680 | 15360 | Source MLP uses `intermediate_size // 2`; preset gives 7680 branch width. |
| `attention_window_size` | 2048 | default | Sliding-window local causal attention. |
| `conv1d_width` | 4 | default | Depthwise causal convolution kernel and state length 3. |
| `partial_rotary_factor` | 0.5 | default | RoPE applies to first half of q/k head dim. |
| `hidden_activation` | `gelu_pytorch_tanh` | default | Recurrent y branch and MLP gate branch. |
| `rms_norm_eps` | `1e-6` | default | RMSNorm epsilon. |
| `attention_bias` | `False` | default | Q/K/V bias absent; O projection has bias. |
| `attention_dropout` | 0.0 | default | Dropout disabled for inference. |
| `logits_soft_cap` | 30.0 | default | Always applied in source. |
| `tie_word_embeddings` | `True` | default | `lm_head.weight` aliases embeddings logically. |
| `use_cache` | `True` | default | Enables DynamicCache plus recurrent module state. |

Representative checkpoint/config sweep from inspectable sources:

| Variant | Source | Visible dimensions | Gaps |
|---|---|---|---|
| `google/recurrentgemma-2b` | tests/docs and gated HF repo | model id, generation expectations, long-context/window tests | Remote `config.json` gated. |
| Converter `2B` | local converter | `hidden=2560`, `layers=26`, `heads=10`, `kv_heads=1`, `intermediate=15360` | Does not prove current Hub config because gated. |
| Source default / converter `7B` label | config class and converter | `hidden=2560`, `layers=26`, `heads=10`, `kv_heads=10`, `intermediate=7680` | Converter label conflicts with public 9B naming; do not infer 9B shape. |
| `google/recurrentgemma-9b` / `9b-it` | gated HF repos | repository existence from search/model listing | Config and weights unavailable without access approval. |

## 3a. Family variation traps

- `num_key_value_heads` can be MQA/GQA/MHA; the converter `2B` preset uses one KV head even though the class default expands to all heads.
- `intermediate_size` is not the actual MLP branch width; source halves it for `gate_proj` and `up_proj`.
- `lru_width` is configurable but source decode cache allocation assumes `hidden_size` channels for `conv1d_state`; require `lru_width == hidden_size` unless source parity for divergent configs is proven.
- Block schedule is config-driven. Do not assume every third layer is attention unless `block_types` is exactly the default.
- Attention is local/sliding-window, not full-context causal attention.
- Recurrent state is not returned in outputs and is not contained in `DynamicCache`.
- Left and right padding differ in integration tests; padding side is a parity-visible input contract.
- RoPE supports only `rope_type="default"` in this source. LongRoPE/YARN/dynamic scaling should be rejected for this family basis.
- Q/K/V projections have optional `attention_bias`; O projection and recurrent/MLP projections have biases.
- `lm_head.weight` and token embeddings are tied logically. Preserve aliasing.

## 4. Operator coverage checklist

Tensor/layout ops:

- Embedding lookup `[B, T] -> [B, T, 2560]`.
- Multiply embedding output by `sqrt(hidden_size)` stored as a nonpersistent bfloat16 buffer and cast to activation dtype.
- Reshape/view/transpose/contiguous for attention heads and recurrent Conv1d layout.
- Concatenate/slice for partial RoPE and Conv1d state update.
- `logits_to_keep` slice by integer or tensor index.

Neural network primitives:

- RMSNorm with fp32 accumulation and `(1 + weight.float())`.
- Linear layers with bias controls:
  - Attention: `q: 2560 -> H*D`, `k/v: 2560 -> KVH*D`, `o: H*D -> 2560` with O bias.
  - Recurrent block: `linear_x: 2560 -> lru_width`, `linear_y: 2560 -> lru_width`, `linear_out: lru_width -> 2560`.
  - MLP: `gate/up: 2560 -> intermediate_size/2`, `down: intermediate_size/2 -> 2560`.
- Depthwise causal Conv1d over recurrent channels: `groups=lru_width`, kernel width 4, source padding 3, output truncated to input length.
- `gelu_pytorch_tanh`, sigmoid, softplus, exp, sqrt, rsqrt, tanh.

Attention primitives:

- Sliding-window causal self-attention via SDPA with mask from `create_sliding_window_causal_mask`.
- MHA/MQA/GQA KV repeat after cache update.
- Partial RoPE on the first half of q/k head dim.

Generation/cache ops:

- DynamicCache for attention layers only, with sparse layer indices and first-attention-layer methods rebound at runtime.
- Recurrent per-layer session state:
  - `conv1d_state`: `[B, lru_width, conv1d_width - 1]` desired ABI, source allocates `[B, hidden_size, conv1d_width - 1]`.
  - `rg_lru.recurrent_states`: `[B, lru_width]`, fp32.
- Position IDs drive both RoPE and RG-LRU reset (`position_id == 0`).

Position ops:

- Default RoPE inverse-frequency precompute.
- Per-call cos/sin generation in fp32 for current `position_ids`.

Preprocessing-coupled ops:

- Tokenizer, padding side, attention mask, BOS/EOS/PAD IDs.

Quantized/packed weight ops:

- No custom source quantization path in modeling code. BitsAndBytes is used only by a test via standard HF quantization config.

## 5. Layer/block breakdown

Decoder model:

```text
input_ids -> embed_tokens
hidden = hidden * sqrt(hidden_size)
repeat N layers:
  hidden = decoder_layer(hidden, position_ids, sliding_window_mask, cache/state)
hidden = final RMSNorm(hidden)
logits = lm_head(hidden[:, logits_to_keep])
logits = tanh(logits / logits_soft_cap) * logits_soft_cap
```

Decoder layer, repeated `num_hidden_layers`:

```text
raw = x
z = temporal_pre_norm(raw)
z = temporal_block(z)       # recurrent or attention, config scheduled
residual = raw + z
y = channel_pre_norm(residual)
y = MLP(y)
out = residual + y
```

Recurrent temporal block:

```text
y = gelu_pytorch_tanh(Linear(hidden_size -> lru_width, bias)(x))
x_branch = Linear(hidden_size -> lru_width, bias)(x)
x_branch = depthwise causal Conv1d(width=conv1d_width)(x_branch)
x_branch = RG-LRU scan(x_branch, position_ids)
out = Linear(lru_width -> hidden_size, bias)(x_branch * y)
```

Attention temporal block:

```text
q = Linear(hidden_size -> num_attention_heads * head_dim, bias=attention_bias)
k = Linear(hidden_size -> num_key_value_heads * head_dim, bias=attention_bias)
v = Linear(hidden_size -> num_key_value_heads * head_dim, bias=attention_bias)
q,k = partial_default_rope(q,k, first head_dim * partial_rotary_factor dims)
k,v = cache.update(k,v, layer_idx) if cache is present
k,v = repeat_kv(k,v, num_attention_heads // num_key_value_heads)
attn = scaled_dot_product_attention(q,k,v, sliding_window_causal_mask, scale=head_dim**-0.5)
out = Linear(num_attention_heads * head_dim -> hidden_size, bias=True)(attn)
```

MLP:

```text
gate = gelu_pytorch_tanh(Linear(hidden_size -> intermediate_size/2, bias)(x))
up = Linear(hidden_size -> intermediate_size/2, bias)(x)
out = Linear(intermediate_size/2 -> hidden_size, bias)(gate * up)
```

## 6. Attention requirements

Attention is causal local self-attention in attention-scheduled layers only. It is not cross-attention and not encoder-decoder attention.

| Requirement | Source behavior |
|---|---|
| Causality | Causal. |
| Local/window | `attention_window_size`, exposed as `sliding_window`; tests force a window smaller than prompt. |
| Head shape | q `[B, num_attention_heads, Q, head_dim]`; k/v `[B, num_key_value_heads, K, head_dim]` before repeat. |
| MQA/GQA | `repeat_kv` expands after cache update; converter `2B` preset is MQA (`kv_heads=1`). |
| Mask | `create_sliding_window_causal_mask` output is passed directly to SDPA. |
| RoPE | Partial default RoPE before cache update, so cached keys are post-RoPE. |
| KV cache | DynamicCache stores per-attention-layer k/v before repeat. |
| FlashAttention | Source flags `_supports_flash_attn=False`, `_supports_sdpa=False`, yet implementation calls PyTorch SDPA directly; do not assume HF backend dispatch parity. |
| Packed/varlen | No explicit packed sequence metadata in this source. |

Cache/state ABI for DinoML:

- Attention KV cache per attention layer:
  - key/value before repeat: `[B, num_key_value_heads, cached_T, head_dim]`.
  - key/value after repeat is a derived view/materialization for attention only.
  - cache length is taken from the first attention layer because recurrent layers have no KV entries.
- Recurrent state per recurrent layer:
  - Conv state `[B, lru_width, conv1d_width - 1]`, activation dtype.
  - RG-LRU state `[B, lru_width]`, fp32.
  - state resets on batch-size mismatch in source; DinoML should expose explicit allocate/reset/update ownership instead.
- The model output does not return `past_key_values`; generation tests that require returned PKV are skipped upstream.

## 7. Position encoding and custom math

RoPE is default-only and partial. It computes inverse frequencies for `dim = int(head_dim * partial_rotary_factor)`, with the default `partial_rotary_factor=0.5`, so for `head_dim=256` only 128 dimensions are rotated.

```python
def recurrent_gemma_partial_rope(q, k, cos, sin, partial=0.5):
    q_rot, q_pass = split_last_dim(q, [int(q.shape[-1] * partial), -1])
    k_rot, k_pass = split_last_dim(k, [int(k.shape[-1] * partial), -1])
    q_rot = q_rot * cos[:, None, :, :] + rotate_half(q_rot) * sin[:, None, :, :]
    k_rot = k_rot * cos[:, None, :, :] + rotate_half(k_rot) * sin[:, None, :, :]
    return concat_last_dim(q_rot, q_pass), concat_last_dim(k_rot, k_pass)
```

RG-LRU math:

```python
input_gate = sigmoid(group_baddbmm(x, input_gate_weight, input_gate_bias))
rec_gate = sigmoid(group_baddbmm(x, recurrent_gate_weight, recurrent_gate_bias))
log_a = -8.0 * rec_gate * softplus(recurrent_param)
a = exp(log_a)
gamma = sqrt(1.0 - exp(2.0 * log_a))
gamma = where(position_ids[:, :, None] == 0, 1.0, gamma)
u = (x * input_gate) * gamma
state_t = where(reset_t, 0.0, a_t * state_{t-1}) + u_t
```

Source training uses a custom autograd sqrt derivative clamp; inference only needs the forward `sqrt`.

## 8. Preprocessing and input packing

Input is text-only. GPU graph inputs are `input_ids` or `inputs_embeds`, optional `position_ids`, optional `attention_mask`, optional cache/state, and `logits_to_keep`.

Tokenizer and CPU-side behavior:

- Gemma tokenizer is used by docs/tests.
- `pad_token_id=0`, `eos_token_id=1`, `bos_token_id=2`.
- Padding side affects generation outputs in tests and must be treated as input-contract-visible.

Runtime packing:

- If `position_ids` are absent, source computes them from attention cache length, not recurrent state length.
- `attention_mask` feeds sliding-window causal mask construction.
- There is no multimodal stitching, packed varlen metadata, image/audio preprocessing, or codebook path.

## 9. Graph rewrite / lowering opportunities

### Rewrite: recurrent prefill depthwise Conv1d -> causal window gather + channelwise dot

Source pattern:

```text
transpose [B,T,C] -> [B,C,T] -> depthwise Conv1d(C,C,k=conv1d_width, groups=C, padding=k-1) -> truncate [:T]
```

Replacement:

```text
left_pad(k-1) -> rolling_window [B,T,C,k] -> channelwise dot(weight[C,k]) + bias[C]
```

Preconditions:

- `groups == channels == lru_width`.
- `stride == 1`, `dilation == 1`, `padding == conv1d_width - 1`.
- Output is truncated to original `T`.
- Weight orientation matches converted HF Conv1d weight `[C, 1, k]`; converter transposes source weights.

Failure cases:

- Divergent `lru_width`/`hidden_size` without cache-shape audit.
- Non-default Conv1d parameters.

Parity test sketch:

- Compare full prefill Conv1d output for random `[B,T,C]`, including `T < k`, `T == k`, and `T > k`.

### Rewrite: decode Conv1d -> explicit state dot

Source pattern:

```text
conv_state = concat(prev_state[B,C,k-1], x_t[B,C,1])
y_t = sum(conv_state * weight[:,0,:], dim=-1) + bias
new_state = conv_state[:,:,1:]
```

Replacement:

```text
stateful_channelwise_dot(prev_conv_state, x_t, weight, bias) -> y_t, new_conv_state
```

Preconditions:

- Single-token decode (`position_ids.shape[1] == 1`).
- Session owns `prev_conv_state`.
- Batch size/order is unchanged or an explicit reorder/reset operation is provided.

Failure cases:

- Beam search or batch compaction without state reorder ABI.
- Prefill path accidentally using decode update.

### Rewrite: RG-LRU grouped gate projections -> batched small GEMM provider

Source pattern:

```text
reshape [B,T,lru_width] -> [heads, B*T, block_width]
baddbmm with [heads, block_width, block_width] weights
```

Replacement:

```text
grouped_bmm_or_block_diagonal_linear(x, per_head_weight, per_head_bias)
```

Preconditions:

- `lru_width % num_attention_heads == 0`.
- Weight layout `[num_attention_heads, block_width, block_width]`.
- Bias layout `[num_attention_heads, block_width]`.

Failure cases:

- Treating gates as one dense `[lru_width,lru_width]` matrix without block-diagonal zeros changes parameter semantics.

### Rewrite: attention QKV packing

Source pattern:

```text
separate q_proj, k_proj, v_proj with possibly different row counts
```

Replacement:

```text
packed_qkv_linear -> split [Q rows, K rows, V rows]
```

Preconditions:

- Same input tensor and dtype.
- Bias presence is identical or packed bias handles absent q/k/v bias correctly.
- Split order is exactly Q then K then V.
- Packed K/V rows use `num_key_value_heads * head_dim`, not `num_attention_heads * head_dim`.

Failure cases:

- MQA/GQA configs where K/V row count is smaller.
- Quantized/partitioned weights without a matching packing contract.

### Rewrite: last-token-only logits

Source pattern:

```text
hidden_states[:, slice_indices, :] -> lm_head -> soft_cap
```

Replacement:

```text
gather selected hidden positions before vocab GEMM
```

Preconditions:

- `logits_to_keep` is known integer `1` or explicit valid index tensor.
- Loss is not requested for omitted logits.

Failure cases:

- Training/loss path needing all shifted logits.

## 10. Kernel fusion candidates

Highest priority:

- Explicit recurrent state kernels: decode Conv1d state dot plus RG-LRU one-step update. This is the decode hot path and cannot hide state mutation.
- RG-LRU scan for prefill: a sequential scan over `T` with fp32 state; this is the family-defining non-transformer op.
- Sliding-window GQA/MQA attention with post-RoPE cached keys and local causal mask.
- RMSNorm with RecurrentGemma ordering `(norm_fp32 * (1 + weight_fp32)).astype(input_dtype)`.

Medium priority:

- Recurrent block branch fusion: `linear_x`, depthwise Conv1d, RG-LRU, `linear_y` activation, multiply, `linear_out`.
- MLP gated activation multiply.
- QKV projection + partial RoPE for attention layers.
- Last-token-only vocab GEMM plus tanh soft cap.

Lower priority:

- Training-only sqrt gradient clamp.
- BitsAndBytes paths; source modeling has no native quantized layout.
- Beam search state reorder until single-stream decode state ABI is correct.

## 11. Runtime staging plan

Stage 1: Parse config and reject unsupported variants.

- Require default RoPE only.
- Require `lru_width == hidden_size` initially.
- Preserve tied embedding/LM-head alias.

Stage 2: Weight loading and one-block parity.

- Load recurrent, attention, RMSNorm, MLP weights.
- Validate one recurrent block and one attention block on random tensors.

Stage 3: Full prefill parity.

- Implement recurrent Conv1d prefill and RG-LRU scan.
- Implement sliding-window attention mask and local attention.
- Compare hidden states and logits.

Stage 4: Decode state ABI.

- Add explicit per-layer recurrent session state and attention KV cache.
- Validate prefill-plus-one-token equals full forward for short sequences.

Stage 5: Optimized kernels.

- Replace Python-style scans/gathers with provider-backed kernels.
- Add local attention backend only after mask/cache parity.

Stage 6: Generation integration.

- Greedy decode with right padding first.
- Then left padding and batch compaction/state reorder if needed.

## 12. Parity and validation plan

- RMSNorm random tests in fp32/fp16/bf16; tolerance `1e-5` fp32, `2e-3` fp16/bf16.
- Partial RoPE tests for known `position_ids`, including nonzero decode offsets.
- Depthwise Conv1d prefill and decode-state equivalence tests for `T=1,3,4,8`.
- RG-LRU scan tests with resets at `position_ids == 0`, fp32 state retention, and one-step decode equivalence.
- Attention tests for MHA and MQA/GQA shapes, local-window masks, post-RoPE cached keys.
- Single recurrent block parity.
- Single attention block parity with cache update.
- Full 26-layer small-random-config parity, with generated block schedules.
- Prefill logits parity for `google/recurrentgemma-2b` once gated access or a mirrored legal test checkpoint is available.
- Decode token parity: full forward over `prompt + token` versus prefill state plus one-token decode.
- End-to-end greedy text parity after tokenizer/config access is available.

No tests were run for this audit, per request.

## 13. Performance probes

- Prefill RG-LRU scan time versus sequence length.
- Decode one-token recurrent state update latency by batch size.
- Sliding-window attention throughput by `attention_window_size` and KV-head count.
- KV cache memory plus recurrent state memory:
  - recurrent state per recurrent layer: `B * lru_width * (conv1d_width - 1) * activation_bytes + B * lru_width * 4`.
  - attention KV cache per attention layer: `2 * B * num_key_value_heads * cached_window_or_T * head_dim * dtype_bytes`.
- Last-token-only logits GEMM cost versus full-sequence logits.
- Padding-side/batch-shape sweep because source outputs differ under left/right padding.
- Dense versus packed QKV projection benchmark for MQA/GQA.

## 14. Skip/defer list

- Training and custom sqrt backward.
- Gradient checkpointing.
- Assisted/speculative decoding; upstream tests skip it because returned PKV is unsupported.
- Beam search/model parallel generation until recurrent state reorder is explicit.
- Non-default RoPE scaling.
- Divergent `lru_width != hidden_size`.
- Native quantized loading beyond generic HF/BitsAndBytes adapters.
- Multi-GPU/tensor parallel.

## 15. Final implementation checklist

- [ ] Parse `RecurrentGemmaConfig` and gated/config-source provenance.
- [ ] Preserve tied `embed_tokens.weight` / `lm_head.weight` alias.
- [ ] Implement RecurrentGemma RMSNorm ordering.
- [ ] Implement default partial RoPE and reject non-default RoPE types.
- [ ] Implement sliding-window causal mask.
- [ ] Implement MHA/MQA/GQA attention with post-RoPE KV cache.
- [ ] Implement recurrent Conv1d prefill.
- [ ] Implement recurrent Conv1d decode state update.
- [ ] Implement RG-LRU gate projections and fp32 scan/state.
- [ ] Define explicit recurrent state ABI for session allocate/reset/update/reorder.
- [ ] Implement gated MLP branch width `intermediate_size // 2`.
- [ ] Implement logits soft cap and `logits_to_keep`.
- [ ] Add recurrent block parity tests.
- [ ] Add attention block cache parity tests.
- [ ] Add prefill-plus-decode equivalence tests.
- [ ] Add gated-checkpoint end-to-end parity when access is available.
