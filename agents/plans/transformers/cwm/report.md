# CWM Transformers Audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: facebook/cwm
Config source: native CwmConfig defaults plus gated Hub metadata; open mirror snapshots used only where labeled
Source files inspected:
- X:/H/transformers/src/transformers/models/cwm/configuration_cwm.py
- X:/H/transformers/src/transformers/models/cwm/modeling_cwm.py
- X:/H/transformers/src/transformers/models/cwm/modular_cwm.py
- X:/H/transformers/tests/models/cwm/test_modeling_cwm.py
- X:/H/transformers/tests/models/cwm/test_configuration_cwm.py
- X:/H/transformers/docs/source/en/model_doc/cwm.md
Any missing files or assumptions:
- Official facebook/cwm, facebook/cwm-sft, and facebook/cwm-pretrain config.json files are gated and returned 403 for the authenticated CLI.
- dnakov/cwm-mlx is an open mirror snapshot with CWM-sized dimensions but model_type/architecture set to llama; use it as dimensional evidence only.
- mci29/sn29_s2m2_cwmk is a smaller llama checkpoint found by query; it is out of native CWM scope.
```

Source URLs:

- https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/cwm/modular_cwm.py
- https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/cwm/modeling_cwm.py
- https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/cwm/configuration_cwm.py
- https://hf.co/facebook/cwm gated
- https://hf.co/facebook/cwm-sft gated
- https://hf.co/facebook/cwm-pretrain gated

Small snapshots saved under this folder:

- `snapshots/dnakov_cwm-mlx/config.json`
- `snapshots/dnakov_cwm-mlx/generation_config.json`
- `snapshots/dnakov_cwm-mlx/special_tokens_map.json`
- `snapshots/mci29_sn29_s2m2_cwmk/config.json`
- `snapshots/precedentbrute_cwm-baseline-sft/adapter_config.json`
- `snapshots/precedentbrute_cwm-nloc-sft/adapter_config.json`

`modeling_cwm.py` and `configuration_cwm.py` are generated from `modular_cwm.py`; future Transformers source edits should treat `modular_cwm.py` as authoritative.

## 2. High-level architecture

CWM is a dense decoder-only causal language model for text generation. The native implementation is Llama/Qwen2-derived with RMSNorm, GQA self-attention, Llama-3 RoPE parameters, SwiGLU MLPs, untied LM head, and per-layer full versus sliding-window causal attention masks.

```text
tokenizer/chat template -> input_ids/attention_mask -> token embedding
-> repeated decoder blocks with mixed full/sliding causal self-attention
-> final RMSNorm -> LM head -> logits -> generation controller/sampling
```

Stage decomposition:

- CPU/data pipeline: tokenizer, chat template, system prompt, optional thinking-template flags, attention mask construction inputs.
- GPU prefill: embeddings, causal/sliding masks, RoPE tables, all decoder blocks, logits.
- GPU decode: one-token or short-token forward with `DynamicCache`; cache stores RoPE-applied keys and raw values.
- Independently optimizable: RoPE table generation, mask generation, GQA attention, RMSNorm, SwiGLU, last-token-only logits.

Primary DinoML target: `CwmForCausalLM` prefill and decode. `CwmModel` feature extraction is optional. Training loss, gradient checkpointing, and output attentions are deferred.

## 3. Important config dimensions

Native `CwmConfig` defaults:

| Field | Value | Source |
|---|---:|---|
| `vocab_size` | 128256 | source default |
| `hidden_size` | 6144 | source default |
| `intermediate_size` | 21504 | source default |
| `num_hidden_layers` | 64 | source default |
| `num_attention_heads` | 48 | source default |
| `num_key_value_heads` | 8 | source default |
| `head_dim` | 128 | source default |
| Q width | 6144 | derived from heads * head_dim |
| KV width | 1024 | derived from kv_heads * head_dim |
| `max_position_embeddings` | 131072 | source default |
| `sliding_window` | 8192 | source default |
| full/sliding pattern | every 4th layer full; others sliding | source default |
| full layers | 16 of 64 | derived |
| sliding layers | 48 of 64 | derived |
| `hidden_act` | `silu` | source default |
| `rms_norm_eps` | 1e-5 | source default |
| attention bias | false, projections are bias-free | source/modeling |
| MLP bias | false by default | source default |
| `tie_word_embeddings` | false | source default |
| cache | true | source default |
| dtype | bfloat16 in mirror/tests | mirror config and tests |

RoPE defaults:

| Field | Value | Source |
|---|---:|---|
| `rope_type` | `llama3` | source default |
| `rope_theta` | 1000000.0 | source default |
| `factor` | 16.0 | source default |
| `low_freq_factor` | 1.0 | source default |
| `high_freq_factor` | 4.0 | source default |
| `original_max_position_embeddings` | 8192 | source default |

Representative checkpoint/config sweep:

| Repo | Access | Config basis | Operator-significant facts |
|---|---|---|---|
| `facebook/cwm` | gated | Hub metadata only | Native `Architecture: cwm`, 32.581B params, 14 safetensor shards visible, fair-noncommercial-research-license. Config access requires approval. |
| `facebook/cwm-sft` | gated | Hub metadata only | Same visible parameter count and gated file layout as base; likely same graph, but config not accessible. |
| `facebook/cwm-pretrain` | gated | Hub metadata only | Same visible shard layout; Hub metadata omits architecture field in plugin output, so do not assume without access. |
| `dnakov/cwm-mlx` | open mirror | downloaded config | 6144/64/48/8, head_dim 128, 131072 positions, sliding_window 8192, bfloat16, Llama-3 rope scaling; `model_type: llama`, so not a native CWM config. |
| `abhijithmallya/cwm-Q8_0-GGUF` / `cwm-Q4_0-GGUF` | open quantized mirrors | Hub metadata only | Quantized from `facebook/cwm`; no Transformers config in repo, GGUF loading is separate provider work. |
| `precedentbrute/cwm-baseline-sft` / `cwm-nloc-sft` | open PEFT adapters | downloaded adapter config | LoRA rank 64 targeting q/k/v/o and gate/up/down projections. Adapter overlay is out of first native dense target. |
| `mci29/sn29_s2m2_cwmk` | open non-CWM query hit | downloaded config | Llama 4096/48-layer checkpoint; out of native CWM scope despite name match. |

## 3a. Family variation traps

- Native CWM uses `rope_parameters`, not legacy Llama `rope_scaling`; open mirrors may translate fields and set `model_type: llama`.
- `hidden_size == num_attention_heads * head_dim` for defaults, but source uses explicit `head_dim`; do not infer it only from hidden size.
- GQA is mandatory in defaults: 48 query heads, 8 KV heads, 6 query groups per KV head.
- Attention alternates by `layer_types`; a single global dense causal mask is wrong for the default graph.
- Sliding-window size is configurable and tests override it to 4096 for memory; masks and attention backend must honor the effective runtime config.
- All attention projections are bias-free in source; MLP bias is controlled by `mlp_bias` but default false.
- LM head and token embedding are logically tied by `_tied_weights_keys`, but `tie_word_embeddings` default is false and the modules are separate by construction.
- Official configs are gated; open Llama/MLX mirrors should not silently route native CWM to a plain Llama implementation unless a compatibility admission rule explicitly accepts the translated config.
- Tokenizer/chat-template behavior matters for end-to-end parity but is not part of the neural graph.
- No image/audio/video branches, no MoE, no NCHW/NHWC layout work.

## 4. Operator coverage checklist

Tensor/layout ops:

- Embedding lookup `[B,S] -> [B,S,6144]`.
- View/reshape Q/K/V projections to `[B,S,H,D]`, transpose to `[B,H,S,D]`, final attention transpose back to `[B,S,H*D]`, contiguous/reshape.
- Slice logits by `logits_to_keep`: int `0` means all positions because `slice(0, None)`; positive values keep suffix positions; tensor indices are allowed by source.
- Add residuals, multiply, dtype casts for norm/softmax/RoPE.

Neural primitives:

- `Linear(6144 -> 6144, bias=False)` q_proj.
- `Linear(6144 -> 1024, bias=False)` k_proj and v_proj.
- `Linear(6144 -> 6144, bias=False)` o_proj.
- `RMSNorm(6144, eps=1e-5)` before attention, before MLP, and final norm; variance computed in fp32.
- SwiGLU: `down_proj(silu(gate_proj(x)) * up_proj(x))` with `Linear(6144 -> 21504)` gate/up and `Linear(21504 -> 6144)` down.
- LM head `Linear(6144 -> 128256, bias=False)`.

Attention primitives:

- Causal self-attention, GQA repeat from 8 KV heads to 48 query heads for eager backend.
- Full causal mask and sliding-window causal mask, selected per layer.
- SDPA/Flash/Flex dispatch via `ALL_ATTENTION_FUNCTIONS`; eager fallback uses fp32 softmax.

Position/rotary:

- Llama-3 RoPE parameters; cos/sin computed in fp32 and cast to hidden dtype.
- RoPE applied to Q and K before cache update.

Generation/cache:

- `DynamicCache(config)` when `use_cache=True`.
- Per-layer K/V append/update with stored shapes `[B,8,T,128]` before GQA repeat.
- Cache reorder for beam search can be deferred if greedy/sampling-only first.

Preprocessing-coupled ops:

- Tokenizer/chat template and special token IDs are CPU/controller ABI.
- Attention masks are rank/shape-sensitive; no multimodal scatter or packed varlen ABI in native source.

Distributed/tensor parallel:

- Config declares TP hints for q/k/v/gate/up colwise and o/down rowwise, plus LM head colwise gather. This is optional for first single-GPU parity.

## 5. Layer/block breakdown

Decoder block, repeated 64 times:

```text
x: [B,S,6144]
residual = x
x = RMSNorm(x)
q = Linear(x): [B,S,6144] -> view [B,S,48,128] -> [B,48,S,128]
k = Linear(x): [B,S,1024] -> view [B,S,8,128] -> [B,8,S,128]
v = Linear(x): [B,S,1024] -> view [B,S,8,128] -> [B,8,S,128]
q,k = RoPE(q,k, cos/sin)
k,v = cache.update(k,v, layer_idx) when cache is present
attn = causal_attention(q,k,v, mask=full_or_sliding, scale=1/sqrt(128))
x = residual + Linear(attn: [B,S,6144] -> [B,S,6144])
residual = x
x = RMSNorm(x)
x = down_proj(silu(gate_proj(x)) * up_proj(x))
x = residual + x
```

After all blocks:

```text
x = RMSNorm(x)
logits = lm_head(x[:, slice_indices, :])
```

## 6. Attention requirements

CWM requires autoregressive causal self-attention only; no cross-attention.

| Requirement | Value |
|---|---|
| Attention type | causal self-attention |
| Head form | GQA |
| Q heads / KV heads | 48 / 8 |
| Head dim | 128 |
| Q width / KV width | 6144 / 1024 |
| Scale | `head_dim ** -0.5` |
| Mask styles | full causal and sliding-window causal |
| Sliding window | default 8192, configurable |
| RoPE | applied to Q/K before cache update |
| Cache storage | K/V per layer `[B,8,T,128]`; expanded only for eager attention compute |
| Backend compatibility | source advertises FlashAttention, SDPA, FlexAttention, eager fallback |

The layer mask is chosen by `config.layer_types[i]`. Native source defaults make layers with `i % 4 == 0` full-attention and all other layers sliding-attention, yielding 16 full layers and 48 sliding layers for 64 layers. The downloaded `dnakov/cwm-mlx` mirror snapshot lists 17 full layers and a final full layer, so treat that as a mirror/config divergence requiring admission checks rather than native CWM behavior.

Mask preconditions:

- Full layers must see all previous positions subject to causal/pad mask.
- Sliding layers must restrict historical keys outside the window.
- Decode with cache must construct masks using current `position_ids` and `past_key_values`.

## 7. Position encoding and custom math

Native default RoPE uses the Transformers `llama3` initializer from `ROPE_INIT_FUNCTIONS`; default fallback math for plain RoPE is:

```python
inv_freq = 1.0 / (rope_theta ** (arange(0, head_dim, 2) / head_dim))
freqs = inv_freq[None, :, None] @ position_ids[:, None, :]
emb = cat([freqs, freqs], dim=-1)
cos = cos(emb) * attention_scaling
sin = sin(emb) * attention_scaling
q = q * cos[:, None, :, :] + rotate_half(q) * sin[:, None, :, :]
k = k * cos[:, None, :, :] + rotate_half(k) * sin[:, None, :, :]
```

`rotate_half(x)` concatenates `[-x[..., D/2:], x[..., :D/2]]`. Cos/sin are computed under autocast-disabled fp32 and cast back to input dtype. RoPE tables can be precomputed for fixed max length and dtype, but dynamic/advanced RoPE update hooks mean admission should reject unsupported `rope_type` values until implemented.

## 8. Preprocessing and input packing

Runtime tensors:

- `input_ids`: `[B,S]` int token IDs, or `inputs_embeds`: `[B,S,6144]`; exactly one is required.
- `attention_mask`: optional tensor accepted by mask helpers, or a dict with `full_attention` and `sliding_attention` masks already constructed.
- `position_ids`: optional `[1,S]` by default; source creates `arange(S) + past_seen_tokens` and unsqueezes batch dim.
- `past_key_values`: optional `Cache`.

Tokenizer/controller ABI:

- Native defaults: BOS 128000, EOS list `[128001,128008,128009]`, no pad token.
- Open mirror special tokens: BOS `<|begin_of_text|>`, EOS `<|end_of_text|>`.
- Docs and integration tests use a system prompt and `apply_chat_template(..., enable_thinking=True, preserve_previous_think=True)`.

No pixel/audio features, no placeholder token scatter, no packed multimodal metadata, and no NCHW/NHWC layout-sensitive regions.

## 9. Graph rewrite / lowering opportunities

### Rewrite: separate Q/K/V linear to packed QKV/GQA projection

Source pattern:

```text
q_proj(x), k_proj(x), v_proj(x)
```

Replacement:

```text
MatMul(x, packed_weight.T) -> split [6144,1024,1024]
```

Preconditions:

- All three projections are bias-free.
- Same input `x`, same dtype, same batch/sequence axes.
- Packed split order must be Q, K, V and preserve source weight row order.
- If TP plans are active, packing must happen after shard semantics are resolved.

Parity sketch: compare Q/K/V tensors before RoPE for random `[B,S,6144]`.

### Rewrite: GQA eager repeat elimination

Source pattern:

```text
repeat_kv(k, 6), repeat_kv(v, 6), dense attention
```

Replacement:

```text
GQA attention kernel with Hq=48, Hkv=8
```

Preconditions:

- KV grouping is uniform: `num_attention_heads % num_key_value_heads == 0`.
- Attention backend supports full and sliding causal masks.
- Outputs do not require materialized attention weights for first target.

Failure cases: requested dense `attentions`, nonstandard head grouping, unsupported sliding-window backend.

### Rewrite: RoPE table precompute

Source pattern:

```text
inv_freq @ position_ids -> cos/sin per forward
```

Replacement:

```text
lookup precomputed cos/sin by position_ids
```

Preconditions:

- `rope_type=llama3` implemented exactly.
- Max position bounded by configured table.
- Same dtype cast and fp32 generation parity.

### Rewrite: last-token-only logits

Source pattern:

```text
lm_head(hidden_states[:, slice_indices, :])
```

Replacement:

```text
gather/slice hidden suffix before GEMM; decode uses [B,1,6144] -> [B,1,V]
```

Preconditions:

- `logits_to_keep` is known int or bounded tensor indices.
- Loss computation is not requested.

### Layout notes

All neural tensors are sequence-major rank-3/rank-4 logical layouts, not image layouts. Preserve `[B,S,H]` and `[B,H,S,D]` semantics. A layout pass may fuse away transpose/contiguous around attention only within the QKV-attention-output region and must rewrite head/sequence axes consistently.

## 10. Kernel fusion candidates

Highest priority:

- RMSNorm for hidden width 6144 with fp32 variance.
- GQA FlashAttention/SDPA for 48q/8kv heads with full and sliding causal masks.
- QKV packed projection plus RoPE application.
- SwiGLU MLP fusion: gate/up GEMM, SiLU, elementwise multiply, down GEMM.
- Last-token-only LM head for decode.

Medium priority:

- RoPE cos/sin lookup and Q/K rotation fusion.
- Residual add fused with output projection or following norm when numerically acceptable.
- Sliding-window mask generation/attention specialization for window 8192 and 4096.
- Weight-only quantized GGUF loader path for open quantized mirrors, as a separate provider contract.

Lower priority:

- Output attentions materialization.
- Tensor-parallel sharding according to source TP plan.
- Fullgraph compile parity with Transformers decorators/kernel hub hooks.

## 11. Runtime staging plan

1. Parse native `CwmConfig` and reject mirror-only `model_type: llama` unless an explicit compatibility path maps fields safely.
2. Load dense weights for `CwmForCausalLM`; verify embeddings, norms, projections, LM head shapes.
3. Implement one decoder block parity without cache, full attention only.
4. Add default layer-type mask mapping and sliding-window causal attention.
5. Add full-model prefill logits parity for short sequences.
6. Add DynamicCache-compatible decode with K/V shapes `[B,8,T,128]`.
7. Optimize GQA attention, RoPE, RMSNorm, SwiGLU, and last-token logits.
8. Add optional GGUF/quantized mirror support only through existing DinoML quantized constant/provider rules.

Can be stubbed initially: training loss, output attentions, tensor parallel, LoRA adapters, beam cache reorder, chat-template helper APIs.

## 12. Parity and validation plan

- Config validation: defaults, custom `sliding_window`, explicit `layer_types`, invalid layer-type length/value rejection.
- Unit parity: RMSNorm fp32 variance, RoPE rotate-half and Llama-3 scaling, repeat_kv/GQA equivalence.
- Single-block parity: random `[B,S,6144]` with full mask and then sliding mask.
- Prefill parity: short prompt logits against Transformers for first 32 vocab entries; use bf16 tolerance around `atol=1e-2, rtol=1e-2` as upstream tests do.
- Long-sequence parity: sequence longer than configured sliding window; verify sliding layers restrict attention.
- Decode parity: prefill N tokens, decode one token with cache, compare logits to full-prefix recompute.
- Generation parity: greedy 20-token test from upstream once gated model access is available.

Recommended tolerances: fp32 custom op tests `1e-5`; bf16/fp16 block/logit tests `1e-2` initially, tighten per backend.

## 13. Performance probes

- Tokenization/chat-template throughput separately from model runtime.
- Prefill tokens/sec sweep for `S = 512, 2048, 8192, 32768, 131072`.
- Decode tokens/sec sweep for cache lengths crossing the sliding window.
- Full versus sliding layer attention time split.
- GQA backend comparison: eager repeat, SDPA, FlashAttention/Flex equivalent.
- KV cache memory: 64 layers * 2 tensors * `[B,8,T,128]` * dtype bytes.
- LM head suffix GEMM cost for full logits versus last-token-only logits.
- RoPE table generation versus lookup overhead for long context.
- Optional quantized mirror probe: GGUF load/dequant/GEMM overhead, clearly separate from native dense parity.

## 14. Skip/defer list

- Training loss and gradient checkpointing.
- Output attentions and hidden-state capture.
- Beam search/cache reorder beyond basic generation.
- Tensor parallel and pipeline parallel execution.
- LoRA/PEFT adapter application.
- GGUF/MLX mirror loading in the first dense native target.
- Unsupported RoPE types beyond native default `llama3`.
- Any multimodal, image/audio/video, MoE, speculative decoding, or distributed scheduling work.

## 15. Final implementation checklist

- [ ] Parse native CWM config fields and defaults.
- [ ] Add admission guard for gated/native CWM versus Llama-format mirrors.
- [ ] Load embeddings, 64 decoder layers, final norm, and untied LM head.
- [ ] Implement RMSNorm with fp32 variance.
- [ ] Implement bias-free Q/K/V/O projections with explicit head_dim.
- [ ] Implement Llama-3 RoPE for `[B,H,S,128]` Q/K.
- [ ] Implement GQA causal attention for 48 query heads and 8 KV heads.
- [ ] Implement full and sliding-window causal masks selected by `layer_types`.
- [ ] Implement SwiGLU MLP with 21504 intermediate width.
- [ ] Implement `DynamicCache`-compatible K/V append and decode masks.
- [ ] Implement logits suffix slicing / last-token-only logits.
- [ ] Add one-block, prefill, sliding-window, and decode parity tests.
- [ ] Add long-context and cache-memory performance probes.
- [ ] Defer LoRA, GGUF, TP/PP, training, and output-attention support until dense parity is stable.
