# GLM4 MoE Transformers audit

## 1. Source basis

Transformers commit/version: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`.

Model family: `glm4_moe`.

Primary runtime target: `Glm4MoeForCausalLM` text-only autoregressive generation, including prefill and decode with self-attention KV cache.

Model ids / config sources inspected:

| Model id | Config source | Local snapshot | Notes |
| --- | --- | --- | --- |
| `zai-org/GLM-4.5` | `https://huggingface.co/zai-org/GLM-4.5/raw/main/config.json` | `_sources/zai-org__GLM-4.5.config.json` | production BF16, 355B total / 32B active per HF model card metadata |
| `zai-org/GLM-4.5-FP8` | `https://huggingface.co/zai-org/GLM-4.5-FP8/raw/main/config.json` | `_sources/zai-org__GLM-4.5-FP8.config.json` | same architecture as GLM-4.5, compressed-tensors FP8 weight config |
| `zai-org/GLM-4.5-Base` | `https://huggingface.co/zai-org/GLM-4.5-Base/raw/main/config.json` | `_sources/zai-org__GLM-4.5-Base.config.json` | base model variant, same operator shape as GLM-4.5 |
| `zai-org/GLM-4.5-Air` | `https://huggingface.co/zai-org/GLM-4.5-Air/raw/main/config.json` | `_sources/zai-org__GLM-4.5-Air.config.json` | production BF16 Air, 106B total / 12B active per HF model card metadata |
| `zai-org/GLM-4.5-Air-FP8` | `https://huggingface.co/zai-org/GLM-4.5-Air-FP8/raw/main/config.json` | `_sources/zai-org__GLM-4.5-Air-FP8.config.json` | same architecture as Air, compressed-tensors FP8 weight config |
| `zai-org/GLM-4.5-Air-Base` | `https://huggingface.co/zai-org/GLM-4.5-Air-Base/raw/main/config.json` | `_sources/zai-org__GLM-4.5-Air-Base.config.json` | base Air variant |
| `tiny-random/glm-4-moe` | `https://huggingface.co/tiny-random/glm-4-moe/raw/main/config.json` | `_sources/tiny-random__glm-4-moe.config.json` | debug-sized checkpoint; operator-significant because `hidden_size != heads * head_dim` |

Source files inspected:

- `X:/H/transformers/src/transformers/models/glm4_moe/modular_glm4_moe.py` (`https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/glm4_moe/modular_glm4_moe.py`)
- `X:/H/transformers/src/transformers/models/glm4_moe/configuration_glm4_moe.py` (`https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/glm4_moe/configuration_glm4_moe.py`)
- `X:/H/transformers/src/transformers/models/glm4_moe/modeling_glm4_moe.py` (`https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/glm4_moe/modeling_glm4_moe.py`)
- Comparison files: `models/glm/modeling_glm.py`, `models/glm4/modeling_glm4.py`, `models/glm/configuration_glm.py`, `models/glm4/configuration_glm4.py`
- Shared utilities: `modeling_rope_utils.py`, `masking_utils.py`, `cache_utils.py`, `modeling_utils.py`, `modeling_flash_attention_utils.py`

Authoritative source note: `configuration_glm4_moe.py` and `modeling_glm4_moe.py` are generated from `modular_glm4_moe.py`. Future Transformers edits should be made in the modular file, but DinoML should audit the generated modeling file because that is what users import.

Missing files or assumptions: no tokenizer files were inspected because tokenization does not alter the core GPU graph beyond `input_ids`, `attention_mask`, and optional generation EOS/pad ids. `GLM-4.5V` configs surfaced in search but are out of scope because they are multimodal and not the text-only `glm4_moe` family target.

## 2. High-level architecture

GLM4 MoE is a text-only decoder-only MoE causal LM:

```text
token ids / embeddings -> decoder prefill/decode blocks -> final RMSNorm -> LM head -> logits/sampling
```

Each decoder block is pre-norm:

```text
RMSNorm -> GQA self-attention with partial RoPE -> residual
RMSNorm -> dense SwiGLU MLP for first K layers, then MoE + shared expert -> residual
```

Stage decomposition:

| Stage | Runtime contract | Independently stageable? |
| --- | --- | --- |
| CPU/data pipeline | tokenizer, chat template, EOS/pad handling, optional attention mask | yes; not model graph critical |
| Embedding | `input_ids -> [B, S, hidden]`, or caller-provided `inputs_embeds` | yes |
| Prefill | full causal GQA attention, MoE routing per token, writes KV cache | yes |
| Decode | one or few query tokens, consumes and appends per-layer KV cache | yes |
| Logits | `lm_head(hidden[:, logits_to_keep, :])`, usually last token only | yes |

Implemented heads: `Glm4MoeModel` and `Glm4MoeForCausalLM`. The causal LM head is required. No sequence/token classification heads are implemented for this family in the inspected source, unlike `glm` and `glm4`.

## 3. Important config dimensions

Config-class defaults differ from production configs, so DinoML should parse checkpoint config rather than instantiate defaults and infer behavior.

| Field | Config class default | Production GLM-4.5 | Production GLM-4.5-Air | Runtime significance |
| --- | ---: | ---: | ---: | --- |
| `hidden_size` | 4096 | 5120 | 4096 | residual width |
| `num_hidden_layers` | 46 | 92 | 46 | block count |
| `num_attention_heads` | 96 | 96 | 96 | query head count |
| `num_key_value_heads` | 8 | 8 | 8 | GQA KV head count |
| `head_dim` | absent in class | 128 | 128 | projection width uses explicit value when present |
| `intermediate_size` | 10944 | 12288 | 10944 | dense early MLP width |
| `moe_intermediate_size` | 1408 | 1536 | 1408 | per-routed-expert hidden width |
| `n_routed_experts` | 128 | 160 | 128 | routed expert count |
| `n_shared_experts` | 1 | 1 | 1 | shared SwiGLU width multiplier |
| `num_experts_per_tok` | 8 | 8 | 8 | top-k routed experts per token |
| `first_k_dense_replace` | 1 | 3 | 1 | first K layers use dense MLP, later layers use MoE |
| `use_qk_norm` | false | true | false | optional per-head RMSNorm after Q/K projection |
| `routed_scaling_factor` | 1.0 | 2.5 | 1.0 | scales routed top-k weights |
| `max_position_embeddings` | 131072 | 131072 | 131072 | default maximum context |
| `attention_bias` | false | true | true | Q/K/V bias present in production checkpoints |
| `hidden_act` | `silu` | `silu` | `silu` | SwiGLU activation |
| `torch_dtype` | not class default | `bfloat16` | `bfloat16` | checkpoint dtype metadata |
| `tie_word_embeddings` | false | false | false | LM head not tied in production |

Representative checkpoint sweep:

| Model id | Layers | Hidden | Q heads / KV heads / head dim | Dense-before-MoE | Routed experts / top-k | QK norm | Scaling | Quant config |
| --- | ---: | ---: | --- | ---: | --- | --- | ---: | --- |
| `tiny-random/glm-4-moe` | 2 | 16 | 4 / 2 / 64 | 1 | 16 / 8 | true | 2.5 | none |
| `zai-org/GLM-4.5-Air` | 46 | 4096 | 96 / 8 / 128 | 1 | 128 / 8 | false | 1.0 | none |
| `zai-org/GLM-4.5-Air-FP8` | 46 | 4096 | 96 / 8 / 128 | 1 | 128 / 8 | false | 1.0 | compressed-tensors |
| `zai-org/GLM-4.5` | 92 | 5120 | 96 / 8 / 128 | 3 | 160 / 8 | true | 2.5 | none |
| `zai-org/GLM-4.5-FP8` | 92 | 5120 | 96 / 8 / 128 | 3 | 160 / 8 | true | 2.5 | compressed-tensors |

Effective RoPE defaults: inspected checkpoint configs leave `rope_parameters` null. `Glm4MoeConfig.__post_init__` supplies `partial_rotary_factor=0.5` for backward compatibility, and the shared RoPE config mixin standardizes missing RoPE parameters to `rope_type="default"` and the library default `rope_theta`. Production configs also provide `head_dim=128`, so only the first 64 dimensions of each head are rotary.

## 3a. Family variation traps

- `head_dim` can differ from `hidden_size // num_attention_heads`. The tiny checkpoint has `hidden_size=16`, `num_attention_heads=4`, and `head_dim=64`, so Q output width is `256`, not `16`.
- GLM4 MoE uses separate `q_proj`, `k_proj`, `v_proj`; older GLM conversion code mentions packed `query_key_value`, but that does not apply here.
- Production GLM-4.5 and Air set `attention_bias=True`, despite `Glm4MoeConfig` defaulting to `False`.
- Full GLM-4.5 uses `use_qk_norm=True`; Air uses `False`.
- Full GLM-4.5 keeps the first 3 layers dense, then MoE. Air keeps only the first layer dense.
- `n_routed_experts` varies: 160 for full, 128 for Air, 16 for tiny.
- `routed_scaling_factor` varies: 2.5 for full/tiny, 1.0 for Air.
- FP8 configs advertise compressed-tensors quantization. The inspected native modeling source does not implement quantized math itself; loading should route through Transformers quantization/weight materialization support or a separate DinoML weight path.
- `tie_word_embeddings=False` in production. Do not alias `lm_head.weight` to embeddings unless the checkpoint says so.
- `norm_topk_prob=True` normalizes selected expert sigmoid probabilities; disabling it would change routing math.
- `n_group`/`topk_group` default to 1 in inspected checkpoints, but source implements grouped expert selection. Future configs with `n_group > 1` require group-topk routing parity.
- There is no sliding-window config in the inspected GLM4 MoE configs. The model calls generic `create_causal_mask`, which can detect packed sequences from nonmonotonic `position_ids` when no explicit attention mask and no cache are present.
- No NCHW/NHWC layout translation is relevant for the core text model. Guard layout passes around all sequence/head reshapes and transposes.

## 4. Operator coverage checklist

Tensor/layout ops:

- Embedding lookup: `[B, S] -> [B, S, hidden]`.
- `view` / `reshape` with explicit projection widths.
- `transpose(1, 2)` between `[B, S, H, D]` and `[B, H, S, D]`.
- `contiguous` after attention output transpose.
- `cat` for RoPE rotated/pass-through halves and `freqs` duplication.
- `chunk(2, dim=-1)` for packed expert gate/up projection output.
- `topk`, `gather`, `scatter_`, `masked_fill`, `one_hot`, `where`, `nonzero`, `permute`, `index_add_` for MoE routing/expert combine.
- Optional `slice` / tensor-index `logits_to_keep` before LM head.

Neural network primitives:

- RMSNorm over last dim, fp32 variance and rsqrt, output cast back to input dtype.
- Linear with optional bias for Q/K/V; no bias for O projection, dense MLP, experts, shared experts, and LM head.
- Dense SwiGLU MLP: `down(silu(gate(x)) * up(x))`.
- Shared expert MLP in every MoE layer: `hidden -> n_shared_experts * moe_intermediate -> hidden`.
- Routed expert packed projection: `gate_up_proj[e]` stored as `[2 * moe_intermediate, hidden]`, split `[gate, up]`, then `down_proj[e]` stored as `[hidden, moe_intermediate]`.

Attention primitives:

- Causal self-attention.
- GQA: 96 query heads, 8 KV heads, `num_key_value_groups=12` in production.
- Repeat KV for eager path from `[B, KVH, S, D]` to `[B, QH, S, D]`.
- Attention scale `head_dim ** -0.5`.
- Eager softmax is computed in fp32 and cast to query dtype.
- Backend dispatch through `ALL_ATTENTION_FUNCTIONS`: eager, SDPA, FlexAttention, FlashAttention 2/3/4, and paged FlashAttention are possible source paths.

Position/rotary ops:

- Default RoPE over first `partial_rotary_factor * head_dim` dimensions only.
- Cos/sin generation uses fp32 matmul `inv_freq @ position_ids`, then casts to hidden dtype.
- RoPE is applied after optional Q/K RMSNorm and after Q/K transpose into `[B, heads, S, D]`.

Generation/cache ops:

- `DynamicCache(config=config)` when `use_cache=True` and no cache is passed.
- Per-layer cache update after RoPE: cached K/V are stored post-RoPE for K and unmodified V.
- Cache tensor logical shape before repeat: K/V `[B, num_key_value_heads, cached_seq, head_dim]`.
- Beam cache reorder must index batch dimension.
- `position_ids` default to `arange(S) + past_seen_tokens`.

Distributed/tensor-parallel metadata:

- Source config includes TP plans: Q/K/V colwise, O rowwise, packed expert gate/up colwise, expert down rowwise, MoE expert sharding marker, shared experts split similarly, LM head `colwise_gather_output`. DinoML can initially ignore TP for single-device parity but should preserve weight names and packed expert layout for later sharding.

## 5. Layer/block breakdown

For production GLM-4.5:

```text
Embedding:
  input_ids [B, S] -> hidden [B, S, 5120]

Decoder block 0..2, dense:
  residual = x
  x = RMSNorm(x)                                  # [B, S, 5120]
  q = Linear(5120 -> 96 * 128, bias=True)
  k = Linear(5120 -> 8 * 128, bias=True)
  v = Linear(5120 -> 8 * 128, bias=True)
  q,k = per-head RMSNorm(q,k)                     # only when use_qk_norm=True
  q,k = partial RoPE(q,k, rotary_dim=64)
  attn = causal GQA(q,k,v, cache)
  x = residual + Linear(96 * 128 -> 5120, bias=False)(attn)
  residual = x
  x = RMSNorm(x)
  x = Linear(12288 -> 5120)(silu(Linear(5120 -> 12288)(x)) * Linear(5120 -> 12288)(x))
  x = residual + x

Decoder block 3..91, MoE:
  same attention path
  residual = x
  x_norm = RMSNorm(x)
  logits = router(x_norm)                         # [B*S, 160], fp32
  topk_indices, topk_weights = grouped sigmoid topk(logits)
  routed = sum_e expert_e(x_norm[token]) * weight
  shared = shared_swiglu(x_norm)                  # 5120 -> 1536 -> 5120 for n_shared=1
  x = residual + routed + shared

Final:
  x = RMSNorm(x)
  logits = Linear(5120 -> 151552, bias=False)(x[:, logits_to_keep, :])
```

For GLM-4.5-Air, replace hidden/intermediate/expert dimensions with `4096`, `10944`, `1408`, use 46 layers, only block 0 is dense, and omit Q/K RMSNorm.

## 6. Attention requirements

Attention type: causal autoregressive self-attention.

MHA/MQA/GQA: GQA. Production configs use 96 query heads and 8 key/value heads, so eager attention repeats each KV head 12 times before `QK^T`. Cache storage should remain compact `[B, 8, T, 128]`; repeat is a compute view/implementation detail, not cache layout.

Masking style:

- Source calls `create_causal_mask(config, inputs_embeds, attention_mask, past_key_values, position_ids)`.
- Eager path adds an additive attention mask to attention scores before softmax.
- If `attention_mask` is omitted, `position_ids` is non-monotonic, and no cache exists, shared mask utilities can infer packed sequence boundaries and combine them with causal masking.
- No inspected GLM4 MoE config sets `sliding_window`; use dense causal admission initially.

KV cache:

- Cache update occurs after RoPE. Cached keys are already position-encoded.
- Values are cached after reshape/transpose but without RoPE.
- For prefill with length `S`, each layer writes K/V `[B, 8, S, 128]`.
- For decode with query length `1`, each layer appends K/V `[B, 8, 1, 128]`.
- `DynamicCache.get_seq_length()` drives default decode position ids.

Backend compatibility:

- Native source declares FlashAttention, SDPA, FlexAttention, and attention-backend support.
- Eager fallback is semantically important but too slow for production because it materializes repeated KV and dense score tensors `[B, 96, Q, K]`.
- A DinoML optimized path should target GQA FlashAttention or paged attention with compact KV cache and partial RoPE already applied.

Attention math order:

```text
linear Q/K/V -> optional per-head Q/K RMSNorm -> transpose heads -> partial RoPE on Q/K -> cache update -> attention backend -> transpose/contiguous -> O projection
```

Preserving optional Q/K norm placement before RoPE is required for full GLM-4.5 parity.

## 7. Position encoding and custom math

RoPE generation:

```python
def glm4_moe_default_rope(config, position_ids):
    head_dim = config.head_dim or config.hidden_size // config.num_attention_heads
    rotary_dim = int(head_dim * config.rope_parameters.get("partial_rotary_factor", 1.0))
    inv_freq = 1.0 / (rope_theta ** (arange(0, rotary_dim, 2).float() / rotary_dim))
    freqs = (inv_freq[None, :, None] @ position_ids[:, None, :].float()).transpose(1, 2)
    emb = cat([freqs, freqs], dim=-1)
    return cos(emb) * attention_scaling, sin(emb) * attention_scaling
```

RoPE application:

```python
def apply_glm4_moe_rope(q, k, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_rot = q_rot * cos + rotate_half(q_rot) * sin
    k_rot = k_rot * cos + rotate_half(k_rot) * sin
    return cat([q_rot, q_pass], -1), cat([k_rot, k_pass], -1)
```

What can be precomputed: `inv_freq` and fixed-position cos/sin tables up to the selected max context can be precomputed per dtype/device for default RoPE. Dynamic RoPE types exist in shared Transformers utilities, but inspected configs do not request them. `position_ids` can be runtime-dependent during decode and packed prefill, so lookup/gather by `position_ids` is still needed.

## 8. Preprocessing and input packing

Inputs accepted by the model graph:

- `input_ids: [B, S]` int token ids, mutually exclusive with `inputs_embeds`.
- `inputs_embeds: [B, S, hidden]` optional direct embedding input.
- `attention_mask` optional; passed to shared mask construction.
- `position_ids: [B or 1, S]` optional; default is contiguous positions offset by cache length.
- `past_key_values` optional `Cache` object.

Generation config snapshots for GLM-4.5 and Air contain only `_from_model_config=true`, EOS ids `[151329, 151336, 151338]`, and pad id `151329`. Sampling, chat templates, tool/reasoning parser behavior, and thinking-mode prompt construction are outside the core module graph and can be handled in the controller/tokenizer layer.

Packed sequence note: source mask utilities can derive packed-sequence segment ids from discontinuities in `position_ids` when no explicit attention mask and no cache are used. DinoML can defer this for first parity by admitting only monotonic position ids or explicit dense masks, but should reject packed position ids until supported.

## 9. Graph rewrite / lowering opportunities

### Rewrite: Q/K/V separate linears -> grouped projection schedule

Source pattern:

```text
q = Linear(hidden -> q_heads * head_dim, bias=attention_bias)
k = Linear(hidden -> kv_heads * head_dim, bias=attention_bias)
v = Linear(hidden -> kv_heads * head_dim, bias=attention_bias)
```

Replacement: keep three logical weights, but schedule as one grouped GEMM or fused projection kernel.

Preconditions:

- Same input tensor and dtype.
- All three projections are present.
- Bias presence matches all Q/K/V projections.
- Output split order is exactly Q, K, V as separate modules, not a packed checkpoint row layout.

Shape equations:

- `Q = [B*S, num_attention_heads * head_dim]`
- `K,V = [B*S, num_key_value_heads * head_dim]`

Failure cases: tensor-parallel sharded loading, quantized FP8 materialization, or checkpoint formats that physically pack weights differently.

Parity test sketch: compare projected Q/K/V tensors before reshape against Transformers for random hidden states.

### Rewrite: partial RoPE + GQA FlashAttention

Source pattern:

```text
q,k optional RMSNorm -> transpose -> partial RoPE -> attention backend
```

Replacement: fused pre-attention kernel that normalizes Q/K when enabled, applies partial RoPE, writes compact KV cache, and launches GQA attention.

Preconditions:

- `rope_type == "default"` initially.
- `partial_rotary_factor == 0.5`.
- compact KV cache layout `[B, KVH, T, D]`.
- no packed sequence mask for first implementation.

Failure cases: dynamic/yarn/longrope configs, packed position ids, backend requiring pre-expanded KV.

Parity test sketch: compare Q/K after optional norm+RoPE and prefill/decode attention output for one layer.

### Rewrite: routed MoE eager loop -> token bucketed expert GEMM

Source pattern:

```text
router_logits = linear(fp32(hidden), fp32(router_weight))
topk = grouped_topk(sigmoid(logits) + correction_bias)
for each hit expert:
    expert_out[token_idx] += down(silu(gate) * up) * topk_weight
```

Replacement: route tokens into expert buckets, run batched/grouped GEMMs for packed gate/up and down projections, then weighted scatter-add.

Preconditions:

- `n_group == 1` and `topk_group == 1` first.
- `norm_topk_prob=True`.
- expert weights use native layout `[E, 2I, H]` and `[E, H, I]`.
- top-k selection uses `sorted=False`; parity must not depend on ordering except matching gathered weights to selected expert ids.

Failure cases: grouped expert selection with `n_group > 1`, tensor-parallel expert sharding, ties in `topk`, quantized expert weights without a materialization plan.

Parity test sketch: fixed random logits with tie-avoidance, compare top-k ids/weights and final MoE output against source for several token counts.

### Rewrite: last-token-only logits

Source pattern: `logits = lm_head(hidden_states[:, slice_indices, :])`.

Replacement: for decode or sampling-only prefill, compute only the requested final tokens.

Preconditions:

- `logits_to_keep` is `1` or small static positive integer.
- no loss computation.
- caller does not request full-sequence logits.

Failure cases: training/loss, full logits inspection, tensor-valued `logits_to_keep` with arbitrary indices.

## 10. Kernel fusion candidates

Highest priority:

- RMSNorm: used twice per block plus final norm; Q/K norm adds two more per attention layer for full GLM-4.5.
- GQA FlashAttention with compact KV cache: avoids repeated KV materialization and dense score tensors.
- Partial RoPE + cache update: decode path touches tiny query slices but large cache metadata; fusing avoids extra transposes/copies.
- MoE routing + expert GEMM: dominant non-attention work in all but the first K layers; source eager loop is not production viable.
- SwiGLU packed expert GEMM: expert `gate_up_proj` is already packed in `[2I, H]`.

Medium priority:

- QKV grouped projection scheduling.
- Shared expert MLP fusion with routed expert combine.
- Last-token-only LM head for generation.
- FP8/compressed-tensors weight materialization and dequant before GEMM.

Lower priority:

- Packed sequence causal masks.
- Tensor-parallel plan execution.
- Non-default RoPE variants not present in inspected configs.

## 11. Runtime staging plan

Stage 1: parse config and load BF16 weights for `tiny-random/glm-4-moe`; validate shapes, especially `head_dim` independent of `hidden_size`.

Stage 2: implement one dense decoder block parity without cache: embedding, RMSNorm, Q/K/V bias projections, optional Q/K norm, partial RoPE, causal attention, dense SwiGLU.

Stage 3: implement MoE block parity with naive but deterministic CPU/GPU routing for small token counts; include shared expert addition.

Stage 4: prefill parity for Air-sized config using BF16 weights or synthetic weights; admit dense causal masks only.

Stage 5: decode with compact KV cache stored after RoPE for keys; validate `position_ids` offset by cache length.

Stage 6: replace eager attention with GQA FlashAttention/paged attention and replace MoE eager loop with bucketed expert GEMMs.

Stage 7: add production weight policies: compressed-tensors FP8, tensor-parallel sharding, and optional offload.

Initially stub/defer: training loss, gradient checkpointing, packed sequence mask auto-detection, beam search cache reorder, tensor parallel, FP8 quantization.

## 12. Parity and validation plan

- RoPE unit tests: compare cos/sin and `apply_rotary_pos_emb` for default `partial_rotary_factor=0.5`, odd batch sizes, decode offsets, and explicit `head_dim`.
- RMSNorm tests: fp32 accumulation, BF16/FP16 output cast, `eps=1e-5`.
- Q/K norm placement test: full GLM-4.5 attention path with `use_qk_norm=True`.
- Router tests: sigmoid logits, correction bias add, group top-2 sum, group mask, top-k gather, normalization, scaling.
- Expert tests: packed `gate_up_proj` split order and `index_add_` accumulation for repeated expert hits.
- Single dense-layer parity: layer 0 for Air and full model.
- First MoE-layer parity: layer 1 for Air, layer 3 for full.
- Prefill logits parity: random short prompts with no cache and explicit attention mask.
- Decode token parity: prefill then one-token decode, checking cache length and logits.
- Recommended tolerances: fp32 custom op tests `rtol=1e-5, atol=1e-6`; BF16/FP16 block tests `rtol=3e-2, atol=3e-2` initially, tighten per kernel after accumulation policy is fixed.

## 13. Performance probes

- Prefill throughput sweep: `B in {1,2,4}`, `S in {128,1024,8192,32768}`.
- Decode tokens/sec sweep: batch size and cache length, with compact KV cache memory recorded.
- Attention backend comparison: eager-equivalent, SDPA, FlashAttention, paged attention.
- MoE routing overhead: router+topk time separate from expert GEMM time.
- Expert load balance: histogram selected expert ids per layer and batch.
- Grouped GEMM efficiency: token count per expert buckets, including tiny decode batches.
- Last-token LM head cost vs full-sequence logits.
- FP8 load/dequant benchmark for compressed-tensors checkpoints once weight loading is in scope.
- Memory probes: BF16 weights, FP8 weights, KV cache `[layers, 2, B, 8, T, 128]`, MoE temporary buffers.

## 14. Skip/defer list

- Training and loss computation.
- Gradient checkpointing.
- Beam search and cache reorder until greedy/sampling decode works.
- Tensor parallel and pipeline parallel execution.
- Compressed-tensors FP8 runtime kernels for first BF16 parity.
- Packed sequence mask inference from `position_ids`.
- Non-default RoPE variants absent from inspected configs.
- Multimodal GLM-4.5V models.
- Chat template, tool parser, reasoning parser, and sampling policy beyond EOS/pad handling.

## 15. Final implementation checklist

- [ ] Parse `Glm4MoeConfig`, including explicit `head_dim`, `first_k_dense_replace`, `use_qk_norm`, and MoE fields.
- [ ] Load embeddings, Q/K/V biases, dense MLP weights, packed expert weights, router correction bias, shared experts, final norm, and untied LM head.
- [ ] Implement RMSNorm with fp32 accumulation.
- [ ] Implement default partial RoPE and cache-position handling.
- [ ] Implement GQA attention with compact KV cache stored after RoPE.
- [ ] Implement optional Q/K RMSNorm before RoPE.
- [ ] Implement dense SwiGLU for early layers.
- [ ] Implement GLM4 MoE router: sigmoid, correction bias, group selection, masked top-k, gather, normalize, scale.
- [ ] Implement packed expert gate/up split and weighted scatter-add combine.
- [ ] Add last-token-only LM head lowering.
- [ ] Add one-block dense and one-block MoE parity tests.
- [ ] Add prefill and one-token decode parity tests.
- [ ] Add performance probes for attention, routing, expert GEMM, and LM head.
- [ ] Reject or separately route packed sequence masks, FP8 compressed checkpoints, tensor parallel, and multimodal GLM-4.5V until audited.
