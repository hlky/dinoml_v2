# Qwen2 Transformers Family Audit

Primary target: `Qwen2ForCausalLM` inference and generation on CUDA. This is a source/config audit only; no DinoML tests were run.

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model family: qwen2
Primary task: causal LM prefill/decode/generation
Local source root: transformers
```

Source files inspected:

- `src/transformers/models/qwen2/configuration_qwen2.py`
- `src/transformers/models/qwen2/modeling_qwen2.py`
- `src/transformers/models/qwen2/modular_qwen2.py`
- `src/transformers/models/qwen2/tokenization_qwen2.py`
- Supporting source: `src/transformers/cache_utils.py`, `src/transformers/masking_utils.py`, `src/transformers/modeling_rope_utils.py`

Authoritative source note: `modeling_qwen2.py` is generated from `modular_qwen2.py`; future Transformers source edits should be checked in `modular_qwen2.py`, while `modeling_qwen2.py` is the concrete generated implementation audited here.

Representative configs fetched from Hugging Face raw `config.json` / `generation_config.json`:

- `Qwen/Qwen2-0.5B`
- `Qwen/Qwen2-1.5B`
- `Qwen/Qwen2-7B`
- `Qwen/Qwen2-72B`
- `Qwen/Qwen2-7B-Instruct`
- `Qwen/Qwen2.5-0.5B`
- `Qwen/Qwen2.5-7B`
- `Qwen/Qwen2.5-72B`
- `Qwen/Qwen2.5-7B-Instruct-1M`
- `Qwen/Qwen2.5-Coder-7B-Instruct`
- `Qwen/Qwen2.5-Math-7B`
- Small/debug open mirrors: `hyper-accel/tiny-random-qwen2`, `llamafactory/tiny-random-qwen2.5`, `trl-internal-testing/tiny-Qwen2ForCausalLM-2.5`. `hf-internal-testing/tiny-random-Qwen2ForCausalLM` returned 401.

Missing files or assumptions:

- No Qwen2 remote code is required for the audited Transformers class.
- Source supports generic RoPE variants through shared `modeling_rope_utils.py`, but the representative official Qwen2/Qwen2.5 configs fetched here use default RoPE with explicit `rope_theta`; none used active `rope_scaling`/YaRN in the consumed qwen2 source path.
- `Qwen/Qwen2.5-7B-Instruct-1M` contains `dual_chunk_attention_config`, but this pinned `Qwen2Config` does not declare or consume that field for layer construction. Treat dual-chunk behavior as out of scope for this qwen2 source path unless another integration layer handles it.

## 2. High-level architecture

Qwen2 is a text-only decoder-only Transformer:

```text
tokenization/input_ids -> embedding -> N decoder blocks -> final RMSNorm -> lm_head -> logits/sampling
```

Generation decomposition:

```text
CPU/tokenizer -> prefill full prompt -> KV cache allocation/update -> decode one or more new tokens -> logits processors/sampling
```

The runtime-critical graph is embedding lookup, repeated RMSNorm + GQA self-attention + SwiGLU MLP, final RMSNorm, and last-token LM projection. Tokenization, chat templates, and sampling/logits processors are controller/data-pipeline work rather than core compiled module work.

Implemented heads:

- Required for target: `Qwen2ForCausalLM`.
- Optional/deferred: base `Qwen2Model` for hidden states.
- Deferred for first causal-LM target: sequence classification, token classification, question answering. These inherit generic heads and do not change decoder block requirements.

## 3. Important config dimensions

Source defaults from `Qwen2Config`:

| Field | Default / behavior |
| --- | --- |
| `vocab_size` | 151936 |
| `hidden_size` | 4096 |
| `intermediate_size` | 22016 |
| `num_hidden_layers` | 32 |
| `num_attention_heads` | 32 |
| `num_key_value_heads` | 32 if omitted; if `None`, set to `num_attention_heads` |
| `head_dim` | Inferred as `hidden_size // num_attention_heads` unless config has `head_dim` |
| `hidden_act` | `silu` |
| `max_position_embeddings` | 32768 |
| `rms_norm_eps` | 1e-6 |
| `use_cache` | true |
| `tie_word_embeddings` | false |
| `attention_dropout` | 0.0 |
| `use_sliding_window` | false |
| `sliding_window` | kept only if `use_sliding_window`; otherwise set to `None` in post-init |
| `layer_types` | if omitted, full attention for layers before `max_window_layers`, sliding attention for layers at/after it only when `sliding_window` remains set |

Representative checkpoint sweep. Dimensions are from fetched `config.json`; `head_dim` and KV grouping are derived from those fields.

| Model id | Layers | H | Heads/KV | Head dim | KV groups | MLP | Vocab | Max pos | RoPE theta | RMS eps | Tied emb | Sliding active |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `hyper-accel/tiny-random-qwen2` | 2 | 512 | 4/4 | 128 | 1 | 1376 | 151936 | 32768 | 1e6 | 1e-6 | false | no |
| `Qwen/Qwen2-0.5B` | 24 | 896 | 14/2 | 64 | 7 | 4864 | 151936 | 131072 | 1e6 | 1e-6 | true | no |
| `Qwen/Qwen2-1.5B` | 28 | 1536 | 12/2 | 128 | 6 | 8960 | 151936 | 131072 | 1e6 | 1e-6 | true | no |
| `Qwen/Qwen2-7B` | 28 | 3584 | 28/4 | 128 | 7 | 18944 | 152064 | 131072 | 1e6 | 1e-6 | false | no |
| `Qwen/Qwen2-72B` | 80 | 8192 | 64/8 | 128 | 8 | 29568 | 152064 | 131072 | 1e6 | 1e-5 | false | no |
| `Qwen/Qwen2-7B-Instruct` | 28 | 3584 | 28/4 | 128 | 7 | 18944 | 152064 | 32768 | 1e6 | 1e-6 | false | no |
| `Qwen/Qwen2.5-0.5B` | 24 | 896 | 14/2 | 64 | 7 | 4864 | 151936 | 32768 | 1e6 | 1e-6 | true | no |
| `Qwen/Qwen2.5-7B` | 28 | 3584 | 28/4 | 128 | 7 | 18944 | 152064 | 131072 | 1e6 | 1e-6 | false | no |
| `Qwen/Qwen2.5-72B` | 80 | 8192 | 64/8 | 128 | 8 | 29568 | 152064 | 131072 | 1e6 | 1e-5 | false | no |
| `Qwen/Qwen2.5-7B-Instruct-1M` | 28 | 3584 | 28/4 | 128 | 7 | 18944 | 152064 | 1010000 | 1e7 | 1e-5 | false | no in this source path |
| `Qwen/Qwen2.5-Math-7B` | 28 | 3584 | 28/4 | 128 | 7 | 18944 | 152064 | 4096 | 10000 | 1e-6 | false | no |

Generation config examples:

- Base Qwen2 0.5B/1.5B/7B/72B: `do_sample=false`, `max_new_tokens=2048`, `bos_token_id=eos_token_id=151643`.
- `Qwen2-7B-Instruct`: `do_sample=true`, `temperature=0.7`, `top_p=0.8`, `top_k=20`, `repetition_penalty=1.05`, `eos_token_id=[151645,151643]`, `pad_token_id=151643`.
- `Qwen2.5-Coder-7B-Instruct`: same sampling family with `repetition_penalty=1.1`.

## 3a. Family variation traps

- GQA is common: `num_key_value_heads < num_attention_heads` for production checkpoints. Cache shape is KV heads, not query heads.
- Projection biases are asymmetric: `q_proj`, `k_proj`, and `v_proj` have bias; `o_proj`, MLP projections, and `lm_head` are bias-free.
- `hidden_size == num_attention_heads * head_dim` for sampled configs, but source permits a separate `head_dim` attr; avoid hard-coding the equality.
- Vocab differs: 151936 for smaller/base variants, 152064 for many 7B+ and instruct/coder/math variants, 151665 for some tiny testing configs.
- `tie_word_embeddings` is true for some 0.5B/1.5B configs and false for many 7B+ configs; loader must handle both shared and separate `lm_head.weight`.
- `rms_norm_eps` changes to `1e-5` for 72B and Qwen2.5 1M configs.
- `rope_theta` varies: 10000 in Qwen2.5-Math-7B, 1e6 in most Qwen2/Qwen2.5 configs, 1e7 in Qwen2.5 1M.
- `sliding_window` can be present even when inactive. The audited config post-init sets `sliding_window=None` unless `use_sliding_window=true`.
- If `use_sliding_window=true`, default `layer_types` makes early layers full attention and layers `i >= max_window_layers` sliding attention.
- The 1M config includes `dual_chunk_attention_config`, but the pinned qwen2 source does not use it. Label any dual-chunk support as a separate follow-up, not baseline qwen2 parity.
- No vision/audio layout translation applies. Text shapes are `[batch, seq, hidden]`; attention internally transposes to `[batch, heads, seq, head_dim]`.
- Axis-sensitive ops: RMSNorm reduces `dim=-1`; softmax is over attention key dimension `dim=-1`; RoPE applies on the final head dimension after `[B,H,S,D]` layout.

## 4. Operator coverage checklist

Tensor/layout ops:

- Embedding lookup: `input_ids [B,S] -> inputs_embeds [B,S,H]`.
- Reshape/view: Q/K/V projection outputs `[B,S,Hq_or_Hkv] -> [B,S,num_heads,head_dim]`.
- Transpose: `[B,S,heads,D] -> [B,heads,S,D]`.
- Contiguous + reshape: attention output `[B,heads,S,D] -> [B,S,H]`.
- Slice/index for `logits_to_keep`: `hidden_states[:, slice_indices, :]`.
- Optional cache concat/update or static cache indexed update.
- Residual adds `[B,S,H]`.

Neural network primitives:

- RMSNorm over final hidden axis, fp32 variance math, weight multiply.
- Linear with bias: Q `H -> num_attention_heads*D`, K/V `H -> num_key_value_heads*D`.
- Linear without bias: O `num_attention_heads*D -> H`, gate/up `H -> I`, down `I -> H`, LM head `H -> vocab`.
- SiLU activation and gated multiply: `down(silu(gate(x)) * up(x))`.

Attention primitives:

- Causal self-attention, full and optionally sliding-window.
- GQA repeat/expand in eager path: KV `[B,Hkv,S,D] -> [B,Hq,S,D]` when not using a native GQA backend.
- Scaled dot-product attention: `q @ k^T * head_dim^-0.5`, mask add, fp32 softmax, dropout only in training, `attn @ v`.
- Backend dispatch through `ALL_ATTENTION_FUNCTIONS` for eager, SDPA, FlashAttention, FlexAttention, and custom backend integrations.

Position/cache/generation ops:

- RoPE cos/sin generation from `position_ids`.
- Apply RoPE to Q and K before cache update.
- DynamicCache and StaticCache support; DynamicCache is default when `use_cache` and no cache is passed.
- Attention mask creation from 2D padding masks, existing 4D masks, cache lengths, and optional packed-sequence position-id detection.
- Logits projection for last token or requested token subset.

Preprocessing-coupled ops:

- BPE tokenizer emits `input_ids` and `attention_mask`. GPU graph needs only these tensors plus optional `position_ids`/cache.
- Generation controller must handle EOS lists, padding token, top-k/top-p/temperature/repetition penalty for instruct configs.

Distributed/tensor-parallel metadata:

- Source config declares TP plan: Q/K/V/gate/up are column-wise; O/down are row-wise; LM head is column-wise gather output. This is optional for first single-GPU integration.

## 5. Layer/block breakdown

Decoder block repeated `num_hidden_layers` times:

```text
x0: [B,S,H]
r = x0
x = RMSNorm(x0)
q = Linear_bias(x): [B,S,Hq=num_attention_heads*D]
k = Linear_bias(x): [B,S,Hkv=num_key_value_heads*D]
v = Linear_bias(x): [B,S,Hkv]
q,k,v -> [B,heads,S,D]
cos,sin = RoPE(position_ids): [B,S,D]
q,k = apply_rope(q,k,cos,sin)
k,v = cache.update(k,v,layer_idx) if cache enabled
attn = causal/GQA attention(q,k,v,mask,scale=D^-0.5,sliding_window?)
attn -> [B,S,H]
x = r + Linear_no_bias(attn)
r = x
x = RMSNorm(x)
mlp = Linear_no_bias(silu(Linear_no_bias(x)) * Linear_no_bias(x))
x = r + mlp
```

Example production shapes:

- Qwen2-7B/Qwen2.5-7B: `H=3584`, `heads=28`, `kv_heads=4`, `D=128`, `I=18944`. Q projection is `3584 -> 3584`; K/V are `3584 -> 512`; O is `3584 -> 3584`; gate/up are `3584 -> 18944`; down is `18944 -> 3584`.
- Qwen2-72B/Qwen2.5-72B: `H=8192`, `heads=64`, `kv_heads=8`, `D=128`, `I=29568`. K/V are `8192 -> 1024`.

Final path:

```text
hidden = RMSNorm(hidden)
logits = Linear_no_bias(hidden[:, slice_indices, :]) -> [B, kept_tokens, vocab_size]
```

## 6. Attention requirements

Attention type:

- Decoder self-attention, causal by default.
- GQA/MQA-ready: `num_key_value_groups = num_attention_heads // num_key_value_heads`.
- No cross-attention in qwen2 causal LM.
- No ALiBi or learned relative bias.
- Dropout is `0.0` during inference; source passes `0.0 if not training else attention_dropout`.

Masking:

- `create_causal_mask` returns backend-specific masks from `attention_mask`, cache metadata, and `position_ids`.
- Existing 4D masks are accepted as already prepared.
- For `position_ids` with no attention mask and no cache, shared masking code can detect packed sequence indices. Inference normally uses monotonic positions and/or cache.
- If sliding layers exist, `create_sliding_window_causal_mask` is used for those layer types.

Cache layout:

- New projected K/V are RoPE-applied before cache update.
- Dynamic full-attention cache stores per layer:
  - key `[B, num_key_value_heads, seen_seq, head_dim]`
  - value `[B, num_key_value_heads, seen_seq, head_dim]`
- Dynamic sliding-window cache stores retained tensors with shape `[B, num_key_value_heads, min(seen_seq, sliding_window-1), head_dim]` and returns full current K/V for the attention call. Its logical `get_seq_length()` is cumulative seen tokens.
- Static full cache allocates `[B, num_key_value_heads, max_cache_len, head_dim]`; static sliding cache allocates `[B, num_key_value_heads, min(max_cache_len, sliding_window), head_dim]`.

Prefill/decode equations:

- Prefill no cache input: Q length `S`, KV length `S`.
- Decode with full cache and one new token: Q length `1`, cache stores/returns KV length `T+1`.
- Per-layer KV memory elements for full cache: `2 * B * num_key_value_heads * T * head_dim`.
- Example Qwen2-7B bf16 per token per layer: `2 * 4 * 128 * 2 bytes = 2048 bytes`; all 28 layers ≈ 56 KiB/token/batch element.
- Example Qwen2-72B bf16 per token per layer: `2 * 8 * 128 * 2 bytes = 4096 bytes`; all 80 layers ≈ 320 KiB/token/batch element.

Math order in eager fallback:

```text
repeat_kv(k/v) -> matmul(q, k^T) -> multiply by head_dim^-0.5 -> add mask -> softmax(dtype=float32) -> cast to query dtype -> dropout -> matmul(weights, v)
```

Optimization note: eager repeat materializes/expands KV to query-head count and is too slow/memory-heavy for production. DinoML should lower to native GQA attention without physical KV repeat.

## 7. Position encoding and custom math

Default RoPE:

```python
inv_freq = 1.0 / (rope_theta ** (arange(0, head_dim, 2) / head_dim))
freqs = (inv_freq[None, :, None] @ position_ids[:, None, :]).transpose(1, 2)
emb = cat(freqs, freqs, dim=-1)
cos = cos(emb) * attention_scaling
sin = sin(emb) * attention_scaling
q = q * cos[:, None, :, :] + rotate_half(q) * sin[:, None, :, :]
k = k * cos[:, None, :, :] + rotate_half(k) * sin[:, None, :, :]
```

`rotate_half(x)` splits the final dimension into two halves and returns `cat(-x2, x1)`. Cos/sin are computed in fp32 under disabled autocast and cast back to the hidden dtype.

RoPE config behavior:

- Older configs carry `rope_theta`; shared config mixin standardizes it into `rope_parameters` with default `rope_type="default"`.
- Shared Transformers utility also supports `linear`, `dynamic`, `yarn`, `longrope`, `llama3`, etc. These are not active in the official sampled qwen2 configs, but a robust parser should preserve `rope_parameters` and reject or stage unsupported rope types clearly.
- `dynamic_rope_update` can recompute inverse frequencies for dynamic/longrope variants based on max `position_ids`; first integration can defer this if it only admits default RoPE.

Precompute opportunities:

- For static max sequence and default RoPE, `inv_freq` is fixed by config; cos/sin can be precomputed per supported position range and sliced.
- Decode can compute only the new position row(s).
- Q/K are cached after RoPE, so cached K never needs position encoding again.

## 8. Preprocessing and input packing

Tokenizer:

- `Qwen2Tokenizer` is BPE over `vocab.json`/`merges.txt`/`tokenizer.json`.
- It applies NFC normalization, a Qwen regex split, byte-level pre-tokenization, and byte-level decoding.
- Model input names are `input_ids` and `attention_mask`.

Runtime tensor contract:

- `input_ids`: integer `[B,S]`, mutually exclusive with `inputs_embeds`.
- `attention_mask`: optional 2D `[B, kv_length]` padding mask or already prepared 4D attention mask.
- `position_ids`: optional `[B,S]`; if omitted, source creates `arange(S) + past_seen_tokens` and unsqueezes to `[1,S]`.
- `past_key_values`: optional cache object; if omitted and `use_cache=True`, `DynamicCache(config)` is created.

Generation controller behavior outside compiled graph:

- Base models use greedy default (`do_sample=false`) and `max_new_tokens=2048`.
- Instruct/coder configs require sampling processors for temperature, top-p, top-k, repetition penalty, EOS list handling, and pad token behavior.
- Chat prompt construction is tokenizer/repo-level behavior, not in `modeling_qwen2.py`; first DinoML graph parity can use raw token IDs.

## 9. Graph rewrite / lowering opportunities

### Rewrite: QKV projections as grouped GEMMs

Source pattern:

```text
q = Linear_bias(x, Wq, bq)
k = Linear_bias(x, Wk, bk)
v = Linear_bias(x, Wv, bv)
```

Replacement pattern:

```text
one or grouped GEMM producing [q | k | v] -> split -> view/transposes
```

Preconditions:

- Same input `x`, dtype, batch/sequence flattening, and contiguous row-major hidden layout.
- Preserve unequal output widths for GQA: `Hq != Hkv`.
- Bias must be concatenated or applied per segment.

Shape equations:

- Flatten `x` to `[B*S, H]`.
- Output widths: `Hq = n_heads*D`, `Hk = Hv = n_kv_heads*D`.

Failure cases:

- Tensor-parallel sharding, quantized per-projection layouts, or independently overridden weights.

Parity test sketch:

- Compare split fused projection outputs to three PyTorch linear calls for GQA and MHA configs.

### Rewrite: native GQA attention

Source pattern:

```text
repeat_kv(k, groups)
repeat_kv(v, groups)
softmax(q @ k.T * scale + mask) @ v
```

Replacement:

```text
GQA FlashAttention/SDPA kernel with q_heads, kv_heads, causal/sliding mask, RoPE-applied K cache
```

Preconditions:

- `num_attention_heads % num_key_value_heads == 0`.
- K/V cache stored as `[B,Hkv,T,D]`.
- Backend supports causal and, when needed, sliding-window masking.
- Preserve fp32 softmax parity tolerance or document backend numeric deltas.

Failure cases:

- Unsupported mask overlays, packed-sequence training masks, attention output requests, or unsupported sliding-window layer mix.

### Rewrite: RMSNorm fused kernel

Source pattern:

```text
x_fp32 = x.to(float32)
variance = mean(x_fp32 * x_fp32, dim=-1, keepdim=True)
y = weight * (x_fp32 * rsqrt(variance + eps)).to(input_dtype)
```

Replacement:

```text
single RMSNorm kernel over last dimension H
```

Preconditions:

- Last dimension contiguous or supported stride.
- Weight length equals H.
- Epsilon from config per checkpoint.

### Rewrite: SwiGLU MLP fusion

Source pattern:

```text
down(silu(gate(x)) * up(x))
```

Replacement:

```text
two GEMMs -> fused SiLU*multiply -> GEMM
```

Preconditions:

- Same input to gate/up; both no-bias in qwen2 source.
- Activation exactly SiLU.

Optimization variant:

- Combine gate/up into one larger GEMM `H -> 2I`, split `[gate, up]`, fused activation multiply.

### Rewrite: last-token-only logits

Source pattern:

```text
logits = lm_head(hidden_states[:, slice_indices, :])
```

Replacement:

```text
for decode, gather/select final hidden row before vocab GEMM
```

Preconditions:

- `logits_to_keep=1` or generation only consumes last token.
- Loss computation disabled.

### Rewrite: inactive sliding-window metadata elimination

Source pattern:

```text
config has sliding_window number but use_sliding_window=false
```

Replacement:

```text
all layers full_attention; no sliding mask/cache path
```

Preconditions:

- Use post-init effective config semantics, not raw JSON field presence.

## 10. Kernel fusion candidates

Highest priority:

- RMSNorm: two per layer plus final norm; small but latency-sensitive in decode.
- GQA FlashAttention with KV cache: required to avoid eager KV repeat and large attention matrices.
- RoPE + cache update path: Q/K RoPE is on the decode hot path; cached K is post-RoPE.
- Gate/up fused MLP plus SiLU multiply: major GEMM and bandwidth region.
- Last-token logits: avoids full-sequence vocab projection during decode.

Medium priority:

- Fused QKV projection or grouped GEMMs with bias.
- Residual add fused with projection epilogues where possible.
- Static cache indexed update for CUDA graph/compile-style decode.
- Sliding-window attention/cache if a qwen2 config with `use_sliding_window=true` is admitted.

Lower priority:

- Classification/QA heads.
- Attention weight output path.
- Training dropout and gradient checkpointing.
- Tensor-parallel execution plans.

## 11. Runtime staging plan

1. Parse `Qwen2Config`, including effective post-init behavior for `num_key_value_heads`, `sliding_window`, `layer_types`, and standardized RoPE parameters.
2. Load weights for embeddings, per-layer Q/K/V/O, RMSNorm, MLP, final norm, and `lm_head`; handle tied embeddings.
3. Implement one-block fp32/bf16 parity without cache using eager reference ops.
4. Implement full prefill logits parity for a small/tiny config and Qwen2-0.5B shape metadata.
5. Add default RoPE and DynamicCache-compatible full-attention decode.
6. Add optimized native GQA attention for prefill/decode.
7. Add last-token logits and generation-controller integration for raw token IDs.
8. Add optional static cache for production decode scheduling.
9. Add sliding-window layer/cache support only after a config with active `use_sliding_window=true` is selected.
10. Add advanced/default-rejected RoPE variants only when a representative config requires them.

Initially stub/defer:

- Sampling processors can run outside DinoML.
- Chat templates/tokenization can remain CPU-side.
- `dual_chunk_attention_config` for Qwen2.5 1M.
- Tensor-parallel sharding.

## 12. Parity and validation plan

Operator tests:

- RMSNorm random tensors across fp32/fp16/bf16 with eps `1e-6` and `1e-5`.
- Default RoPE against Transformers for several `rope_theta` values: 10000, 1e6, 1e7.
- GQA attention with `heads/kv_heads` of 14/2, 28/4, 64/8 and MHA tiny 4/4.
- Dynamic cache update: prefill then one-token decode, checking K/V are post-RoPE and shaped `[B,Hkv,T,D]`.
- SwiGLU MLP parity for `H/I` pairs 896/4864, 3584/18944, 8192/29568.

Model tests:

- Single-layer parity with randomly initialized tiny config.
- After-N-layer parity for 2-layer tiny config.
- Prefill logits parity on fixed token IDs with `logits_to_keep=0`.
- Decode parity token by token with `logits_to_keep=1` and cache reuse.
- Tied and untied LM-head load tests.
- Config effective behavior tests for inactive `sliding_window` despite raw JSON presence.

Tolerance guidance:

- fp32: `rtol=1e-4`, `atol=1e-5` for block/logit parity.
- bf16/fp16: start with `rtol=3e-2`, `atol=3e-2` for full logits; tighter for isolated deterministic ops.
- Optimized attention may need separate tolerances due to softmax accumulation order.

End-to-end:

- Raw prompt token IDs through greedy base generation for base configs.
- Instruct configs with external sampler verifying EOS-list handling and plausible token parity under deterministic settings where possible.

## 13. Performance probes

- Prefill throughput by sequence length: 128, 2048, 32768, 131072 where memory allows.
- Decode tokens/sec by batch size and active cache length.
- KV cache memory usage per model: 0.5B, 7B, 72B, 1M-context config.
- Attention backend comparison: eager reference, SDPA, FlashAttention/native GQA.
- MLP throughput: fused versus unfused gate/up/SwiGLU/down.
- LM-head cost: full-sequence logits versus last-token-only logits.
- Weight-load time and memory footprint for tied versus untied embeddings.
- Sliding-window synthetic benchmark only if active sliding layers are admitted.

No benchmark observations are included; these are proposed probes from source/config requirements.

## 14. Skip/defer list

- Training, labels/loss, dropout behavior, gradient checkpointing.
- Sequence classification, token classification, QA heads.
- Returning attentions/hidden states as a first optimized path.
- Tensor parallel and pipeline parallel runtime.
- Quantized weights beyond whatever DinoML weight loader separately supports.
- YaRN/dynamic/longrope unless an admitted qwen2 checkpoint requires it.
- Dual-chunk attention metadata in Qwen2.5 1M for this pinned qwen2 implementation.
- Beam search/speculative decoding; generation can initially use greedy/sampling controller outside the compiled graph.
- Sliding-window attention unless `use_sliding_window=true` survives config post-init.

## 15. Final implementation checklist

- [ ] Parse Qwen2 config and apply effective defaults/post-init behavior.
- [ ] Standardize and validate RoPE parameters; admit default RoPE first.
- [ ] Load embeddings, decoder weights, RMSNorm weights, and LM head.
- [ ] Support tied and untied `lm_head.weight`.
- [ ] Implement Q/K/V biased linear projections and O no-bias projection.
- [ ] Implement RMSNorm with fp32 variance math.
- [ ] Implement default RoPE and `rotate_half`.
- [ ] Implement GQA causal attention without physical KV repeat.
- [ ] Implement DynamicCache-compatible KV layout `[B,Hkv,T,D]`.
- [ ] Implement prefill causal mask and decode cache-position handling.
- [ ] Implement SwiGLU MLP, preferably with gate/up fusion.
- [ ] Implement final RMSNorm and logits projection.
- [ ] Implement `logits_to_keep` / last-token-only logits lowering.
- [ ] Add tiny/random one-block and two-layer parity tests.
- [ ] Add Qwen2-0.5B/7B shape/config load tests.
- [ ] Add decode cache parity tests.
- [ ] Add generation-controller integration for EOS lists and sampling outside the graph.
- [ ] Benchmark prefill, decode, KV memory, MLP, attention backend, and LM-head slices.
