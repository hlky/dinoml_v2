# AFMoE Transformers Audit

## 1. Source basis

Transformers commit/version: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4` from `X:/H/transformers`.

Model id: primary public representative `arcee-ai/Trinity-Mini`; additional official Arcee configs below.

Primary upstream URLs:

- Transformers source directory at pinned commit: `https://github.com/huggingface/transformers/tree/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/afmoe`
- `arcee-ai/Trinity-Mini`: `https://huggingface.co/arcee-ai/Trinity-Mini`
- `arcee-ai/Trinity-Nano-Preview`: `https://huggingface.co/arcee-ai/Trinity-Nano-Preview`
- `arcee-ai/Trinity-Mini-Base-Pre-Anneal`: `https://huggingface.co/arcee-ai/Trinity-Mini-Base-Pre-Anneal`
- `arcee-ai/Trinity-Large-Preview`: `https://huggingface.co/arcee-ai/Trinity-Large-Preview`

Config source:

- Source defaults: `X:/H/transformers/src/transformers/models/afmoe/configuration_afmoe.py`
- Hub snapshots saved in this folder:
  - `config_trinity_nano_preview.json`
  - `config_trinity_mini.json`
  - `config_trinity_mini_base_pre_anneal.json`
  - `config_trinity_large_preview.json`
  - `generation_config_trinity_mini.json`
  - `tokenizer_config_trinity_mini.json`

Source files inspected:

- `X:/H/transformers/src/transformers/models/afmoe/modeling_afmoe.py`
- `X:/H/transformers/src/transformers/models/afmoe/modular_afmoe.py`
- `X:/H/transformers/src/transformers/models/afmoe/configuration_afmoe.py`
- Supporting common source touched for config interpretation:
  - `X:/H/transformers/src/transformers/configuration_utils.py`
  - `X:/H/transformers/src/transformers/modeling_rope_utils.py`

Any missing files or assumptions:

- `modeling_afmoe.py` is generated from `modular_afmoe.py`; future Transformers source edits should target `modular_afmoe.py`, but DinoML parity should follow generated `modeling_afmoe.py`.
- The official Hub repositories expose `auto_map` custom code and older custom config fields. This report scopes required runtime behavior to the in-library source at the pinned commit unless explicitly marked as Hub remote-code or historical config behavior.
- No gated repos were required for the sampled official configs; large safetensor shards were not downloaded.

## 2. High-level architecture

AFMoE is a decoder-only causal language model with sparse MoE feed-forward layers. It is Llama-like in the broad shape, but has AFMoE-specific attention gating, Q/K RMSNorm, hybrid sliding/full causal attention, dual RMSNorm around both attention and MLP/MoE sublayers, and token-choice MoE routing.

Dataflow:

```text
tokenizer/input_ids -> token embedding -> decoder blocks -> final RMSNorm -> tied/untied LM head -> logits/sampling
```

Decoder block stages:

```text
RMSNorm -> Q/K/V projections + gate projection -> Q/K RMSNorm
  -> optional RoPE for sliding layers -> causal attention/full or sliding
  -> sigmoid gate multiply -> output projection -> RMSNorm -> residual add
  -> RMSNorm -> dense MLP or shared+dynamically-routed MoE -> RMSNorm -> residual add
```

Stage decomposition for DinoML:

- CPU/data pipeline: tokenizer, chat template, sampling controls, attention mask construction if not compiled into graph.
- Prefill graph: embeddings, full sequence hybrid attention, MoE routing/expert execution, logits.
- Decode graph: one-token or small-step hidden input, cache update, sliding-window or full causal attention according to layer type.
- Cacheable state: per-layer K/V caches; full-attention layers retain all prior tokens, sliding layers use sliding-window mask semantics but still call the common `Cache.update` path in source.
- Independently optimizable regions: RMSNorm, Q/K/V projection packing, RoPE for sliding layers, attention backend, MoE router + expert GEMMs, final last-token logits.

## 3. Important config dimensions

Source default config dimensions:

| Field | Source default |
|---|---:|
| `vocab_size` | 200192 |
| `hidden_size` | 2048 |
| `intermediate_size` | 6144 |
| `moe_intermediate_size` | 1408 |
| `num_hidden_layers` | 32 |
| `num_dense_layers` | 1 |
| `num_attention_heads` | 16 |
| `num_key_value_heads` | defaults to `num_attention_heads` |
| `head_dim` | 128 |
| `hidden_act` | `silu` |
| `max_position_embeddings` | 16384 |
| `rms_norm_eps` | `1e-5` |
| `num_experts` | 64 |
| `num_experts_per_tok` | 6 |
| `num_shared_experts` | 2 |
| `route_scale` | 1.0 |
| `global_attn_every_n_layers` | 4 |
| `sliding_window` | 1024 |
| `attention_bias` | `False` |
| `mup_enabled` | `False` |
| `use_cache` | `True` |

Representative checkpoint sweep:

| Model config | Hidden | Attn heads / KV heads / head dim | Q width | Layers | Dense layers | Experts / top-k / shared | MLP inter / MoE inter | Context / sliding | Full:sliding layers | Other |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `arcee-ai/Trinity-Nano-Preview` | 1024 | 8 / 2 / 128 | 1024 | 56 | 2 | 128 / 8 / 1 | 3072 / 256 | 131072 / 2048 | 14:42 | bf16, muP |
| `arcee-ai/Trinity-Mini` | 2048 | 32 / 4 / 128 | 4096 | 32 | 2 | 128 / 8 / 1 | 6144 / 1024 | 131072 / 2048 | 8:24 | bf16, muP |
| `arcee-ai/Trinity-Mini-Base-Pre-Anneal` | 2048 | 32 / 4 / 128 | 4096 | 32 | 2 | 128 / 8 / 1 | 6144 / 1024 | 4096 / 2048 | 8:24 | bf16, muP |
| `arcee-ai/Trinity-Large-Preview` | 3072 | 48 / 8 / 128 | 6144 | 60 | 6 | 256 / 4 / 1 | 12288 / 3072 | 262144 / 4096 | 15:45 | bf16, muP |

Notes:

- `Q width = num_attention_heads * head_dim`, and may exceed `hidden_size`.
- K/V width is `num_key_value_heads * head_dim`.
- Hub configs contain `rope_theta: 10000` and `rope_scaling: null`; current `PreTrainedConfig` standardizes this into `rope_parameters`.
- Hub configs omit `attention_bias`; the in-library effective default is `False`.

## 3a. Family variation traps

- Do not infer attention projection width from `hidden_size`. Trinity Mini and Large use `hidden_size != num_attention_heads * head_dim`.
- GQA is common in sampled configs: `num_key_value_heads < num_attention_heads`.
- `layer_types` is operator-significant. Every fourth layer is full attention in sampled configs; other layers use sliding-window causal attention.
- In the pinned in-library source, RoPE is applied only when `config.layer_types[layer_idx] == "sliding_attention"`. Full-attention layers receive no RoPE in `AfmoeAttention.forward`.
- `mup_enabled` multiplies token embeddings by `sqrt(hidden_size)` before the first block.
- MoE starts at `layer_idx >= num_dense_layers`; earlier layers use dense SwiGLU MLP.
- Router uses sigmoid scores, adds `expert_bias` only for top-k selection, gathers original sigmoid scores, normalizes selected scores by their sum, then multiplies by `route_scale`.
- Hub configs include historical or remote-code fields such as `use_grouped_mm`, `route_norm`, `score_func`, `n_group`, `topk_group`, `num_limited_groups`, `num_expert_groups`, and `load_balance_coeff`. The inspected in-library generated source does not read these fields for inference behavior.
- Hub repos use `auto_map` custom code. DinoML should either route exact remote-code checkpoints through this audited native source only after config compatibility checks, or audit remote code separately.
- Weight tying: `AfmoeForCausalLM` declares `lm_head.weight` tied to `model.embed_tokens.weight`, but `tie_word_embeddings` is `False` in source default and sampled configs. Loader behavior should preserve actual checkpoint aliasing if present.
- No NCHW/NHWC image layout issue exists; this is text-only.

## 4. Operator coverage checklist

Tensor/layout ops:

- Token embedding lookup `[B,S] -> [B,S,H]`.
- Shape/view: `view`, `reshape`, `transpose`, `contiguous`, `unsqueeze`, slicing for `logits_to_keep`.
- Concatenate for RoPE half rotation.
- `one_hot`, `permute`, `sum`, `nonzero`, `where`, advanced indexing, `index_add_` for eager MoE fallback.
- Top-k and gather for router selection.

Neural network primitives:

- Bias-free linear projections for embeddings/MLP/MoE/router/LM head.
- Optional attention projection bias if a future config sets `attention_bias=True`.
- RMSNorm over full hidden dimension and over per-head `head_dim`.
- SiLU/SwiGLU: `down_proj(silu(gate_proj(x)) * up_proj(x))`.
- Sigmoid gate multiply on attention output.
- Residual adds around attention and MLP/MoE.

Attention primitives:

- Causal self-attention, both full and sliding-window variants.
- GQA repeat or grouped attention with `num_key_value_groups = num_attention_heads // num_key_value_heads`.
- Attention softmax upcast to fp32 in eager path, then cast back to query dtype.
- KV cache update via Transformers `Cache.update(key, value, layer_idx)`.

Position/rotary ops:

- Default RoPE inverse frequency, float32 cos/sin generation, half rotation, elementwise RoPE.
- Dynamic RoPE types are possible through common `ROPE_INIT_FUNCTIONS`, but sampled configs use default RoPE.

Generation/cache ops:

- Dynamic cache creation when `use_cache=True` and no cache is supplied.
- `position_ids = arange(seq_len) + past_seen_tokens`.
- Last-token or selected-token logits via `logits_to_keep`.

MoE ops:

- Router GEMM `H -> num_experts`.
- Sigmoid, bias add for selection only, top-k, gather, selected-score normalization.
- Shared expert dense SwiGLU with width `moe_intermediate_size * num_shared_experts`.
- Routed expert packed weights:
  - `gate_up_proj`: `[num_experts, 2 * moe_intermediate_size, hidden_size]`
  - `down_proj`: `[num_experts, hidden_size, moe_intermediate_size]`
- Expert dispatch and accumulation by token/expert. Efficient DinoML path should avoid general `one_hot/nonzero/where/index_add_` where possible and use a bounded grouped-GEMM or sorted-token routing plan.

Preprocessing-coupled ops:

- Text tokenizer only. The sampled tokenizer config uses `PreTrainedTokenizerFast`, `add_bos_token=False`, `add_eos_token=False`, BOS `<|begin_of_text|>`, EOS `<|im_end|>`, PAD `<|pad|>`, and `model_max_length=65536` for Trinity Mini.

Distributed/tensor-parallel ops:

- Source declares `_tp_plan` for projections and experts, but single-GPU DinoML can defer tensor-parallel sharding.

## 5. Layer/block breakdown

Decoder block, repeated `num_hidden_layers` times:

```text
residual = x
x = RMSNorm_H(x)
q = Linear(H -> num_attention_heads * head_dim, bias=attention_bias)(x)
k = Linear(H -> num_key_value_heads * head_dim, bias=attention_bias)(x)
v = Linear(H -> num_key_value_heads * head_dim, bias=attention_bias)(x)
gate = Linear(H -> num_attention_heads * head_dim, bias=False)(x)
q = view(q, [B,S,QH,D]); k = view(k, [B,S,KVH,D]); v = view(v, [B,S,KVH,D])
q = RMSNorm_D(q).transpose(1, 2)
k = RMSNorm_D(k).transpose(1, 2)
v = v.transpose(1, 2)
if sliding_attention: q,k = RoPE(q,k, cos, sin)
if cache: k,v = cache.update(k,v, layer_idx)
attn = causal_attention(q,k,v, full_or_sliding_mask)
attn = reshape(attn, [B,S,QH*D]) * sigmoid(gate)
x = residual + RMSNorm_H(Linear(QH*D -> H, bias=attention_bias)(attn))

residual = x
x = RMSNorm_H(x)
if layer_idx < num_dense_layers:
    x = Linear(intermediate_size -> H)(silu(Linear(H -> intermediate_size)(x)) * Linear(H -> intermediate_size)(x))
else:
    shared = Linear(shared_width -> H)(silu(Linear(H -> shared_width)(x)) * Linear(H -> shared_width)(x))
    routed = token_choice_moe(x)
    x = shared + routed
x = residual + RMSNorm_H(x)
```

For Trinity Mini, a MoE layer has `H=2048`, `QH*D=4096`, `KVH*D=512`, dense MLP width `6144`, routed expert width `1024`, and shared expert width `1024`.

For Trinity Large, a MoE layer has `H=3072`, `QH*D=6144`, `KVH*D=1024`, dense MLP width `12288`, routed expert width `3072`, and shared expert width `3072`.

## 6. Attention requirements

Attention type:

- Autoregressive causal self-attention only.
- Hybrid layer pattern: `full_attention` and `sliding_attention`.
- GQA in sampled configs; source also supports MHA when `num_key_value_heads == num_attention_heads`.

Shapes:

- Query before attention: `[B, num_attention_heads, S_q, head_dim]`.
- Key/value before repeat: `[B, num_key_value_heads, S_k, head_dim]`.
- Eager fallback repeats K/V to `[B, num_attention_heads, S_k, head_dim]`.
- Output before projection: `[B, S_q, num_attention_heads * head_dim]`.

Masking:

- `AfmoeModel.forward` builds both `create_causal_mask` and `create_sliding_window_causal_mask`, then selects by `layer_types[i]`.
- Sliding-window layers pass `sliding_window=config.sliding_window` to the attention backend.

RoPE:

- In this source basis, RoPE is applied only on sliding layers. Full-attention layers skip `apply_rotary_pos_emb`.
- Cached keys are stored after any sliding-layer RoPE because RoPE occurs before `past_key_value.update`.

Backend compatibility:

- `_supports_flash_attn = True`, `_supports_flex_attn = True`, and `_supports_attention_backend = True`.
- Eager fallback is standard matmul/softmax/matmul with fp32 softmax. Optimized DinoML attention must preserve Q/K norm, optional RoPE placement, attention gating after attention, and layer-specific full/sliding masks.

## 7. Position encoding and custom math

Default RoPE parameters:

```python
inv_freq = 1.0 / (rope_theta ** (arange(0, head_dim, 2).float() / head_dim))
freqs = (inv_freq[None, :, None] @ position_ids[:, None, :].float()).transpose(1, 2)
emb = cat((freqs, freqs), dim=-1)
cos = cos(emb).to(x.dtype)
sin = sin(emb).to(x.dtype)
```

RoPE application:

```python
def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return cat((-x2, x1), dim=-1)

def apply_afmoe_rope(q, k, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)
```

Custom source math:

- RMSNorm computes variance in fp32: `x * rsqrt(mean(x^2) + eps) * weight`.
- Attention output gate: `attention_output * sigmoid(gate_proj(normed_input))` before `o_proj`.
- muP input scaling: `hidden_states *= sqrt(hidden_size)` immediately after token embedding and before all blocks.
- Router:

```python
scores = sigmoid(router_logits.float())
selected = topk(scores + expert_bias, k=num_experts_per_tok, dim=1).indices
weights = gather(scores, dim=1, index=selected)
weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
weights = weights * route_scale
```

## 8. Preprocessing and input packing

Inputs:

- `input_ids` `[B,S]` or `inputs_embeds` `[B,S,H]`; exactly one must be supplied.
- Optional `attention_mask`, either raw mask or a dict already keyed by `full_attention` and `sliding_attention`.
- Optional `position_ids`; otherwise source derives monotonically increasing ids from cache length.
- Optional `past_key_values`; if missing and `use_cache=True`, source creates `DynamicCache`.

Tokenizer/generation notes from Trinity Mini snapshots:

- Tokenizer: `PreTrainedTokenizerFast`.
- Special tokens: BOS `<|begin_of_text|>`, EOS `<|im_end|>`, PAD `<|pad|>`.
- Tokenizer config sets `add_bos_token=False` and `add_eos_token=False`; chat template/controller owns prompt framing.
- Generation config includes `temperature=0.15`, `top_p=0.75`, `top_k=50`, `min_p=0.06`. These are generation-controller parameters, not neural graph ops.

No multimodal placeholder, scatter stitch, image/video/audio preprocessing, packed varlen `cu_seqlens`, or channel-layout translation is required for the primary target.

## 9. Graph rewrite / lowering opportunities

### Rewrite: pack QKV projections

Source pattern:

```text
q_proj(x), k_proj(x), v_proj(x)
```

Replacement:

```text
single Linear(H -> QH*D + 2*KVH*D) -> split [q, k, v]
```

Preconditions:

- Same input tensor and dtype.
- Bias settings match across q/k/v. Current configs omit bias through effective `attention_bias=False`.
- Preserve weight layout: concatenate output rows in source split order `[q, k, v]`.
- Do not include `gate_proj`; it feeds sigmoid gating and has separate output width `QH*D`.

Parity test sketch: compare q/k/v tensors before Q/K RMSNorm for random hidden states.

### Rewrite: attention gate fusion

Source pattern:

```text
attention(q,k,v) -> reshape -> multiply sigmoid(gate_proj(x)) -> o_proj
```

Replacement:

```text
attention output epilogue computes sigmoid gate multiply before output GEMM
```

Preconditions:

- Gate tensor shape equals attention output width `[B,S,QH*D]`.
- Gate uses the same pre-attention normalized hidden state as Q/K/V.
- Preserve dtype and sigmoid approximation tolerance.

Failure cases: configs with unexpected attention output width or fused attention backend returning non-contiguous layout without a known epilogue contract.

### Rewrite: dense SwiGLU MLP to fused GEMM epilogue

Source pattern:

```text
silu(gate_proj(x)) * up_proj(x) -> down_proj
```

Replacement:

```text
packed gate/up GEMM -> fused SiLU multiply -> down GEMM
```

Preconditions:

- Bias-free gate/up projections.
- `hidden_act == "silu"` for sampled configs.
- Static intermediate width for candidate planning.

### Rewrite: MoE eager dispatch to grouped expert GEMM

Source pattern:

```text
topk router -> one_hot/where token grouping -> per-expert gate_up/down -> index_add
```

Replacement:

```text
router topk -> token/expert assignment table -> grouped gate_up GEMM -> fused SiLU multiply -> grouped down GEMM -> weighted scatter-add
```

Preconditions:

- `num_experts_per_tok` fixed by config.
- Expert weights use source packed layout `[E, 2I, H]` split as `[gate, up]`, and `[E, H, I]` for down.
- Accumulation order tolerance is defined; exact eager `index_add_` order may differ.

Failure cases: remote-code group routing fields becoming active, training load-balance paths, or unsupported sparse dispatch output ordering.

### Rewrite: last-token-only logits

Source pattern:

```text
lm_head(hidden_states[:, slice_indices, :])
```

Replacement:

```text
carry only selected token rows into LM head for decode or `logits_to_keep=1`
```

Preconditions:

- Caller does not require full-sequence logits.
- Slice or tensor index semantics are known at compile/run boundary.

## 10. Kernel fusion candidates

Highest priority:

- RMSNorm for hidden and per-head dimensions. It appears four times per block plus Q/K norm.
- GQA full/sliding causal attention with KV cache. This is the main decode/prefill cost.
- MoE router + grouped expert GEMM. Eager `one_hot/nonzero/where/index_add_` is not a viable production path.
- Packed SwiGLU gate/up projection and fused SiLU multiply for dense and expert MLPs.
- Last-token-only LM head for decode.

Medium priority:

- QKV projection packing with split metadata.
- RoPE generation/application for sliding layers, fused with Q/K norm or attention pre-processing.
- Attention gate sigmoid multiply fused into attention output or `o_proj` input staging.
- Mask construction specialization for layer-type/full/sliding patterns.

Lower priority:

- Tensor-parallel plans from source `_tp_plan`.
- Router logits output capture for diagnostics.
- Non-default RoPE variants from common config machinery; sampled configs do not require them.

## 11. Runtime staging plan

Stage 1: Config and weight loader.

- Parse `AfmoeConfig`, normalize old Hub fields, reject remote-code-only behavior not implemented by in-library source, and preserve projection widths from config.
- Load token embeddings, LM head, attention, dense MLP, router, shared expert, and packed expert tensors.

Stage 2: Dense-only first blocks.

- Run embedding, muP scale, dense decoder layers `layer_idx < num_dense_layers`, final norm/head on a tiny prefix without cache.

Stage 3: One MoE block parity.

- Implement router top-k and a simple deterministic expert dispatch fallback. Optimize later.

Stage 4: Full prefill parity.

- Add hybrid full/sliding causal masks and full model prefill logits.

Stage 5: Decode with cache.

- Implement per-layer K/V cache update, position id offset, sliding-window attention behavior, and `logits_to_keep=1`.

Stage 6: Optimized kernels.

- Swap in fused RMSNorm, packed projections, optimized attention, grouped MoE, and last-token head.

Stage 7: Production scheduling.

- Batch/sequence bucketing, cache memory planning, MoE routing statistics, and optional tensor parallel.

## 12. Parity and validation plan

- Config parsing tests for default config plus Nano, Mini, Mini-base, and Large snapshots.
- Unit parity for `AfmoeRMSNorm` over `[B,S,H]` and `[B,S,heads,D]`.
- Unit parity for default RoPE, including the source-specific guard that only sliding layers apply RoPE.
- Attention single-layer parity for full and sliding layers, with and without cache.
- Router parity: sigmoid, top-k selected indices, normalized selected weights, and `route_scale`.
- Expert parity for a tiny synthetic config with deterministic route assignments.
- One dense decoder block parity for layer 0.
- One MoE decoder block parity for first MoE layer.
- Full-model prefill logits parity on short prompts.
- Decode parity for several tokens with cache and `logits_to_keep=1`.
- Suggested tolerances: fp32 `1e-4` absolute/relative for block tests; bf16/fp16 `1e-2` to `3e-2` for full logits depending on attention backend and MoE accumulation order.

## 13. Performance probes

- Prefill throughput sweep by sequence length: 512, 2048, 4096, 8192, and long-context buckets for Mini/Nano.
- Decode tokens/sec with cache for batch sizes 1, 4, 16.
- Full vs sliding layer attention timing and memory usage.
- KV cache memory by model variant and sequence length.
- MoE router time, selected expert histogram, and grouped GEMM efficiency.
- Expert GEMM batch-size distribution per layer during representative prompts.
- Dense MLP vs MoE layer timing.
- LM head cost with full logits vs last-token-only logits.
- Config width sweep: Nano, Mini, Large to catch `Q width != H` effects.
- Loader/materialization time for bf16 weights and any future quantized/packed variants.

## 14. Skip/defer list

Safe to defer for first integration:

- Training losses, load-balance auxiliary loss, and gradient checkpointing.
- Router diagnostic output unless needed for parity debugging.
- Tensor parallel and pipeline parallel plans.
- Remote-code-only config fields not read by pinned in-library source.
- Non-default RoPE scaling variants until an official AFMoE checkpoint requires them under native source.
- Beam search, speculative decoding, and advanced generation controllers.
- Quantized/AWQ/GGUF loading variants; treat as separate source-coupled weight-format audits.
- Full sequence logits during decode when `logits_to_keep=1` is sufficient.

Do not defer:

- GQA projection widths.
- Hybrid full/sliding attention masks.
- Source-specific RoPE placement.
- Attention output gate.
- Q/K per-head RMSNorm.
- MoE routing and shared+routed expert addition.
- muP input scaling for sampled Trinity configs.

## 15. Final implementation checklist

- [ ] Parse `AfmoeConfig` and normalize `rope_parameters`/legacy Hub `rope_theta`.
- [ ] Add config compatibility guards for ignored remote-code fields.
- [ ] Load attention weights with explicit `q_width`, `kv_width`, and `head_dim`.
- [ ] Load packed expert weights preserving `[E, 2I, H]` split order.
- [ ] Implement RMSNorm hidden/head variants.
- [ ] Implement muP embedding scale.
- [ ] Implement dense SwiGLU MLP.
- [ ] Implement AFMoE router top-k weighting.
- [ ] Implement initial MoE expert dispatch fallback.
- [ ] Add grouped MoE GEMM lowering plan.
- [ ] Implement full and sliding causal attention masks.
- [ ] Implement GQA attention with cache.
- [ ] Implement source-specific RoPE-on-sliding-layers behavior.
- [ ] Implement attention sigmoid gate before `o_proj`.
- [ ] Implement final norm and LM head with `logits_to_keep`.
- [ ] Add single-op parity tests for RMSNorm, RoPE, router, and expert dispatch.
- [ ] Add one-block dense and one-block MoE parity tests.
- [ ] Add prefill and decode logits parity tests.
- [ ] Benchmark prefill, decode, MoE routing, expert GEMM, and KV memory.
