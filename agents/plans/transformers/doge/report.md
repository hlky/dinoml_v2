# Doge Transformers Audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: doge / SmallDoge Doge family
Config source: Transformers DogeConfig defaults plus public Hub config snapshots in this folder
Source files inspected:
  X:/H/transformers/src/transformers/models/doge/configuration_doge.py
  X:/H/transformers/src/transformers/models/doge/modeling_doge.py
  X:/H/transformers/src/transformers/models/doge/modular_doge.py
  X:/H/transformers/src/transformers/models/doge/convert_doge_weights_to_hf.py
  X:/H/transformers/tests/models/doge/test_modeling_doge.py
  X:/H/transformers/docs/source/en/model_doc/doge.md
Config snapshots:
  SmallDoge_Doge-20M_config.json
  SmallDoge_Doge-60M_config.json
  SmallDoge_Doge-160M_config.json
  SmallDoge_Doge-320M_config.json
  SmallDoge_Doge-20M-MoE_config.json
  SmallDoge_Doge-120M-MoE_config.json
  SmallDoge_Doge-20M-Instruct_config.json
Any missing files or assumptions: no gated Hub configs encountered; weights were not downloaded. modeling_doge.py is generated from modular_doge.py, but it is the runtime source basis inspected here.
```

HF connector metadata lists public Apache-2.0 checkpoints under `SmallDoge/*`, with base text-generation and instruct/question-answering variants. Representative links: [Doge-20M](https://hf.co/SmallDoge/Doge-20M), [Doge-60M](https://hf.co/SmallDoge/Doge-60M), [Doge-160M](https://hf.co/SmallDoge/Doge-160M), [Doge-320M](https://hf.co/SmallDoge/Doge-320M), [Doge-20M-MoE](https://hf.co/SmallDoge/Doge-20M-MoE), [Doge-120M-MoE](https://hf.co/SmallDoge/Doge-120M-MoE).

## 2. High-level architecture

Doge is a text-only causal decoder with dynamic-mask self-attention and an optional Cross Domain MoE state transformation. The public classes are `DogeModel`, `DogeForCausalLM`, and `DogeForSequenceClassification`.

```text
token ids -> token embedding -> repeated decoder blocks -> final RMSNorm -> LM head / classifier
prefill: embeddings -> dynamic-mask causal attention -> MLP/CDMoE -> logits
decode: new token embedding + DynamicCache KV -> dynamic-mask attention over cached KV -> logits
```

CPU/data-pipeline work is ordinary tokenizer/chat-template handling. Runtime-heavy stages are decoder prefill, decode with KV cache, dynamic mask construction, optional CDMoE routing, and final logits. The embedding table and final LM head are tied when `tie_word_embeddings=true` in checkpoints, even though the config class default is `False`.

## 3. Important config dimensions

Source defaults from `DogeConfig`:

| Field | Default |
|---|---:|
| `vocab_size` | 32768 |
| `hidden_size` | 1024 |
| `intermediate_size` | 2048 |
| `num_hidden_layers` | 32 |
| `num_attention_heads` | 8 |
| `num_key_value_heads` | defaults to `num_attention_heads` if omitted |
| inferred `head_dim` | `hidden_size // num_attention_heads` |
| `max_position_embeddings` | 2048 |
| `rope_parameters` | default RoPE unless converted from checkpoint `rope_scaling`/`rope_theta` |
| `attention_bias` / `mlp_bias` | `False` / `False` |
| `hidden_act` | `silu` |
| `use_cache` | `True` |
| `sliding_window` | `None` |
| `keep_window_size` | 2048 |
| `is_moe` | `False` |
| `num_experts` / `num_experts_per_tok` | 16384 / 64 |

Representative checkpoint sweep, from downloaded `config.json` snapshots:

| Model | Layers | Hidden | Heads / KV | Head dim | FFN | MoE | Experts / top-k | RoPE | Dtype |
|---|---:|---:|---:|---:|---:|---|---:|---|---|
| Doge-20M | 8 | 256 | 2 / 1 | 128 | 512 | no | 16384 / 64 unused | null/default | float32 |
| Doge-60M | 16 | 512 | 4 / 2 | 128 | 1024 | no | 16384 / 64 unused | dynamic factor 4 | bfloat16 |
| Doge-160M | 24 | 768 | 6 / 3 | 128 | 1536 | no | 16384 / 64 unused | dynamic factor 4 | bfloat16 |
| Doge-320M | 16 | 512 | 4 / 2 | 128 | 1024 | no | 16384 / 64 unused | dynamic factor 4 | bfloat16 |
| Doge-20M-MoE | 8 | 256 | 2 / 1 | 128 | 512 | yes | 1024 / 16 | dynamic factor 4 | float32 |
| Doge-120M-MoE | 16 | 512 | 4 / 2 | 128 | 1024 | yes | 4096 / 32 | dynamic factor 4 | float32 |

Note: HF repo metadata reported parameter counts; configs are the operator source. The inspected Doge-60M and Doge-320M config snapshots have identical operator dimensions despite different model names.

## 3a. Family variation traps

- GQA is standard in public checkpoints: `num_key_value_heads < num_attention_heads`, with `num_key_value_groups = heads / kv_heads`.
- `head_dim` is not a declared config field in snapshots; source infers it, currently 128 for all sampled public configs.
- Dynamic mask attention is not ordinary causal attention. It builds a per-layer, value-derived attention mask before calling the attention backend.
- `keep_window_size` gates a `topk` sparse mask only when key length exceeds the window. The default/public value 2048 means short-context runs behave mostly dense after value-derived biasing.
- `sliding_window` is a separate causal-mask option. Public snapshots set it `null`, but source supports `create_sliding_window_causal_mask`.
- Dense MLP and CDMoE are mutually exclusive through `is_moe`.
- CDMoE routing factorizes expert ids through two top-k router halves: `num_keys = floor(sqrt(num_experts))`; the implementation assumes expert ids from `x_key * num_keys + y_key`.
- Checkpoints include historical fields not read by this source basis, including `dynamic_mask_ratio`, `hidden_bias`, `expert_retrieval_size`, `num_cdmoe_*`, and older `auto_map` remote-code metadata.
- Checkpoints use legacy `rope_scaling`/`rope_theta`; current config class exposes `rope_parameters`. Admission should verify how `PreTrainedConfig` normalizes these fields before compiling.
- `tie_word_embeddings` is `true` in sampled public checkpoints while the class default is `False`; preserve embedding/LM-head aliasing.
- Source declares `_supports_flash_attn = False`, `_supports_sdpa = True`, `_supports_flex_attn = True`; optimized attention admission cannot assume FlashAttention.

## 4. Operator coverage checklist

Tensor/layout ops:
- Embedding lookup `[B,S] -> [B,S,H]`.
- Reshape/view/transposes for Q/K/V: `[B,S,H] -> [B,heads,S,D]`.
- `repeat_kv` expansion from `[B,kv_heads,S,D]` to `[B,heads,S,D]`.
- Concatenate/slice for RoPE and logits slicing via `logits_to_keep`.

Neural primitives:
- RMSNorm over `H` and over `head_dim`.
- Linear projections: Q `H -> heads*D`, K/V `H -> kv_heads*D`, dynamic mask `kv_heads*D -> kv_heads`, O `heads*D -> H`.
- Dense SwiGLU MLP: `gate/up H -> I`, `silu(gate) * up`, `down I -> H`.
- Learnable residual scales: elementwise `input_residual * residual + attention` and `post_attention_residual * residual + mlp`.
- Dropout for training only.
- LM head `H -> vocab`, bias false.

Attention primitives:
- Causal self-attention with GQA, RoPE on Q/K, fp32 softmax in eager path, optional SDPA/flex dispatch.
- Dynamic mask: `softplus`, multiply by learned `A`, `exp`, broadcast to `[B,kv_heads,Q,K]`, optional causal/padding mask fill, optional top-k/scatter sparse retention.

MoE primitives, if `is_moe=true`:
- Router linear `H -> 2*num_keys`.
- Two `topk(num_keys)` selections, pairwise score addition, flatten, second `topk(num_experts_per_tok)`.
- `gather`, expert embedding lookups for `down_embed`/`up_embed`, batched matmul token-to-expert weights, `silu`, softmax routing weights, weighted expert accumulation.
- Optional aux load-balancing loss uses `scatter_add_`.

Position/cache/generation:
- Dynamic RoPE update for long context variants.
- `DynamicCache` KV update per layer, cached K/V after RoPE for keys.
- `create_causal_mask` or `create_sliding_window_causal_mask`.

Preprocessing-coupled ops:
- Tokenizer and chat templates only; no multimodal scatter or processor-derived tensors.

Distributed:
- Source includes tensor-parallel plan annotations for Q/K/V, `dt_proj`, O, MLP, CDMoE embeddings, and LM head. This is optional for first DinoML integration.

## 5. Layer/block breakdown

Decoder block, repeated `num_hidden_layers`:

```text
residual = x
x_norm = RMSNorm_H(x)
q = RMSNorm_D(Linear(H -> heads*D)(x_norm).view(B,S,heads,D)).transpose
k = RMSNorm_D(Linear(H -> kv_heads*D)(x_norm).view(B,S,kv_heads,D)).transpose
v = Linear(H -> kv_heads*D)(x_norm).view(B,S,kv_heads,D).transpose
q,k = RoPE(q,k, position_ids)
k,v = DynamicCache.update(k,v,layer_idx) when cache is present
dt = Linear(kv_heads*D -> kv_heads)(v.transpose/reshape)
dt = exp(A * softplus(dt)).transpose to [B,kv_heads,K]
mask = dynamic value-derived mask + causal/padding mask + optional topk window
attn = GQA attention(q,k,v,mask,scale=D**-0.5)
x = input_residual * residual + Linear(heads*D -> H)(attn)

residual = x
y = RMSNorm_H(x)
if dense:
  y = Linear(I -> H)(silu(Linear(H -> I)(y)) * Linear(H -> I)(y))
if CDMoE:
  y = dense shared SwiGLU + routed embedding experts
x = post_attention_residual * residual + y
```

Projection biases follow `attention_bias` and `mlp_bias`; sampled public configs set them false or omit them, which maps to source default false.

## 6. Attention requirements

Doge requires causal self-attention with GQA. Public configs use `heads/kv_heads` of `2/1`, `4/2`, or `6/3`, all with `head_dim=128`. Q width is `num_attention_heads * head_dim`; K/V width is `num_key_value_heads * head_dim`; O projection consumes full attention-head width.

Masking is source-specific:
- Base causal/padding mask comes from Transformers masking utilities.
- Doge then builds a dynamic mask from cached value states through `dt_proj`, `softplus`, learned vector `A`, and `exp`.
- For key length greater than `keep_window_size`, only the top `keep_window_size` dynamic-mask entries per query survive by `topk`/`scatter`; others become dtype min.

Cache:
- Each layer stores K/V as `[B, kv_heads, past_seq, head_dim]`.
- Cached keys are stored after RoPE, because RoPE is applied before `past_key_values.update`.
- Values are raw V projection outputs.
- Attention backend sees repeated K/V only in eager path; flex attention sets `enable_gqa=True`.

Backend compatibility:
- FlashAttention is explicitly not supported by the model class.
- SDPA and flex attention are advertised, but DinoML should first implement an eager-equivalent path because the dynamic mask is not a standard lower-triangular mask.
- The eager path softmaxes in float32 and casts probabilities back to query dtype.

## 7. Position encoding and custom math

Default RoPE inverse frequency:

```python
dim = config.head_dim if present else config.hidden_size // config.num_attention_heads
inv_freq = 1.0 / (rope_theta ** (arange(0, dim, 2).float() / dim))
freqs = inv_freq[None, :, None] @ position_ids[:, None, :].float()
emb = cat([freqs, freqs], dim=-1)
cos, sin = emb.cos() * attention_scaling, emb.sin() * attention_scaling
```

Application:

```python
def doge_rope(q, k, cos, sin):
    cos = cos[:, None, :, :]
    sin = sin[:, None, :, :]
    def rotate_half(x):
        return cat([-x[..., x.shape[-1] // 2:], x[..., :x.shape[-1] // 2]], dim=-1)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin
```

Dynamic RoPE variants are possible through normalized `rope_parameters["rope_type"]`; public configs mostly use dynamic scaling with factor 4 and original max 2048. RoPE cos/sin depend on runtime `position_ids` and current cache length; inverse frequencies and scaling can be precomputed per admitted rope variant, with dynamic update support for long contexts.

Dynamic mask custom math:

```python
dt = dt_proj(value.transpose(1, 2).reshape(B, K, kv_heads * D))
dt = exp(A * softplus(dt)).transpose(-1, -2)  # [B, kv_heads, K]
mask = dt[:, :, None, :].expand(B, kv_heads, Q, K)
if K > keep_window_size:
    idx = topk(mask, keep_window_size, dim=-1).indices
    active = zeros_like(mask).scatter(-1, idx, 1.0)
    mask = mask.masked_fill(active == 0, min_dtype)
```

## 8. Preprocessing and input packing

Inputs are text token ids or caller-provided `inputs_embeds`. If both or neither are supplied, source raises. Default `position_ids` are `arange(seq_len) + past_seen_tokens`, shaped `[1,S]`. `attention_mask` may be absent, boolean, or numeric; source masking utilities convert it before Doge dynamic masking.

No image/audio/video packing, special multimodal placeholders, packed varlen descriptors, or scatter stitching are required for the primary model. Instruct variants rely on tokenizer chat templates and generation settings outside the compiled graph.

Sequence classification uses the generic Transformers classification wrapper over `DogeModel`; first integration can defer it behind causal LM parity.

## 9. Graph rewrite / lowering opportunities

### Rewrite: fused RMSNorm

Source pattern:
```text
x_fp32 = x.to(float32); variance = mean(x_fp32*x_fp32, axis=-1); y = weight * x_fp32 * rsqrt(variance + eps); cast to input dtype
```

Replacement: single RMSNorm kernel.

Preconditions: normalize over last dimension, weight shape equals normalized dimension, no bias, eps from config. Parity test: random fp32/bf16/fp16 tensors for `H` and `head_dim`.

### Rewrite: QKV projection region

Source pattern:
```text
q_proj, k_proj, v_proj separately; q/k head RMSNorm; RoPE q/k
```

Replacement: fused or grouped GEMMs plus head-wise RMSNorm/RoPE fusion.

Preconditions: same input tensor, bias flags known, output split order is separate Q, K, V modules, not packed checkpoint rows. Failure cases: tensor-parallel sharded weights, nonstandard head_dim, attention_bias true until bias path is validated.

### Rewrite: dense SwiGLU MLP

Source pattern:
```text
down_proj(silu(gate_proj(x)) * up_proj(x))
```

Replacement: two GEMMs plus fused SiLU/multiply, then down GEMM.

Preconditions: dense `is_moe=false`, activation exactly `silu`, projection weights not tied. Parity test: one block with random weights.

### Rewrite: CDMoE routed embedding experts

Source pattern:
```text
router topk over two factorized axes -> embedding gather -> token expert matmuls
```

Replacement: specialized routing/gather kernel plus batched small reductions.

Preconditions: `num_experts` perfect-square-friendly enough for `floor(sqrt(num_experts))` product ids, `num_experts_per_tok` bounded, inference only. Failure cases: expert id product can be less than `num_experts` when `num_experts` is not a perfect square; preserve source behavior exactly.

### Rewrite: last-token-only logits

Source pattern: `hidden_states[:, slice_indices, :]` before LM head.

Replacement: for decode, only project requested positions.

Preconditions: `logits_to_keep` is static int or validated tensor indices. Failure cases: training loss over full sequence.

## 10. Kernel fusion candidates

Highest priority:
- Dynamic mask construction: `dt_proj -> softplus/exp -> causal fill -> optional topk/scatter` is Doge-specific and likely dominates long-context prefill overhead.
- GQA attention with dynamic additive mask and KV cache: required for useful decode.
- RMSNorm and head-RMSNorm: appears three times per layer plus final norm.
- Dense SwiGLU MLP GEMM/activation fusion for non-MoE checkpoints.

Medium priority:
- Q/K/V projection plus q/k RMSNorm plus RoPE.
- CDMoE routing top-k and embedding expert accumulation for MoE checkpoints.
- Last-token-only logits and tied embedding/LM-head handling.

Lower priority:
- Aux router loss, training dropout, gradient checkpointing.
- Tensor-parallel plan support.
- Sequence classification head.

## 11. Runtime staging plan

Stage 1: parse Doge configs and normalize legacy `rope_scaling`/`rope_theta` into the effective rope parameters; load dense non-MoE weights with tied LM-head alias support.

Stage 2: implement one dense decoder block parity without cache, using eager dynamic-mask attention and dense SwiGLU.

Stage 3: full dense prefill parity for Doge-20M/60M-style configs; support default and dynamic RoPE.

Stage 4: decode with `DynamicCache` ABI: cached post-RoPE K and raw V, runtime position id offset, dynamic mask over cached values.

Stage 5: optional optimized dynamic-mask attention and last-token logits.

Stage 6: add CDMoE inference path for MoE checkpoints.

Stage 7: sequence classification and distributed/tensor-parallel features if product needs them.

Initially stub or defer training loss, router aux loss, dropout, output attentions, and sequence classification.

## 12. Parity and validation plan

- Unit test RoPE default and dynamic-scaling variants against Transformers for fixed `position_ids`, fp32 and bf16.
- Unit test `prepare_dynamic_mask` for: no attention mask, boolean mask, numeric mask, `K <= keep_window_size`, and `K > keep_window_size` top-k path.
- Single-attention parity with random Q/K/V-producing weights, cache absent and present.
- Dense MLP parity for `H/I` pairs: 256/512, 512/1024, 768/1536.
- CDMoE parity for public MoE settings: `num_experts=1024, top_k=16` and `4096, top_k=32`; include deterministic top-k tie tests if possible.
- One-block and after-N-layer parity for Doge-20M config.
- Prefill logits parity on a short prompt against `SmallDoge/Doge-20M`.
- Decode token parity: prefill N tokens, decode 1 and 4 tokens with cache, compare logits to full recompute.
- Suggested tolerances: fp32 `atol=1e-5, rtol=1e-4`; bf16/fp16 `atol=2e-2, rtol=2e-2` for logits, tighter for isolated ops where accumulation is fp32.

## 13. Performance probes

- Prefill throughput by sequence length around `keep_window_size`: 512, 2048, 4096, 8192.
- Dynamic mask build time separated from attention matmul.
- Top-k/scatter dynamic mask cost as `keep_window_size` varies.
- Decode tokens/sec with KV cache for batch sizes 1, 4, 16.
- KV cache memory: `layers * 2 * B * kv_heads * S * head_dim * dtype_size`.
- Attention backend comparison: eager-equivalent, SDPA-compatible if mask can be represented efficiently, and custom dynamic-mask fused path.
- Dense MLP GEMM throughput versus full block time.
- CDMoE routing/gather/expert accumulation sweep for `num_experts/top_k`.
- LM-head last-token-only versus full-sequence logits.

## 14. Skip/defer list

- Training, dropout parity in training mode, gradient checkpointing.
- Router aux loss except for optional training parity.
- Sequence classification.
- Tensor parallel and pipeline parallel annotations.
- FlashAttention; source marks it unsupported.
- Non-public or remote-code-only historical config fields not read by this source.
- Full instruct generation-controller behavior beyond tokenizer chat template and standard sampling.

## 15. Final implementation checklist

- [ ] Parse Doge config and normalize effective RoPE parameters.
- [ ] Load token embeddings and preserve tied LM-head alias when configured.
- [ ] Implement RMSNorm for hidden and per-head dimensions.
- [ ] Implement default/dynamic RoPE and cache-position handling.
- [ ] Implement Q/K/V/O projections with GQA shape rules.
- [ ] Implement Doge dynamic mask construction, including `dt_proj`, learned `A`, causal/padding fill, and `keep_window_size` top-k.
- [ ] Implement eager-equivalent GQA attention with fp32 softmax.
- [ ] Implement DynamicCache with post-RoPE K and raw V.
- [ ] Implement dense SwiGLU MLP.
- [ ] Implement learnable residual scales.
- [ ] Add full dense decoder and LM-head parity for Doge-20M.
- [ ] Add decode parity against full recompute.
- [ ] Add CDMoE routing and embedding expert path for MoE checkpoints.
- [ ] Benchmark prefill, decode, dynamic mask, and CDMoE routing separately.
