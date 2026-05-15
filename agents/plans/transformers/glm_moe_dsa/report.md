# Transformers Audit: glm_moe_dsa

## 1. Source basis

Transformers commit/version:

- Local checkout `transformers` at commit `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`.
- Source package appears to target the 2026-era Transformers mainline; inspected files are generated from modular source.

Model id:

- Primary source/config target: `zai-org/GLM-5`, `zai-org/GLM-5.1`.
- Quantized/config-variation references: `zai-org/GLM-5-FP8`, `QuantTrio/GLM-5-AWQ`, `spicyneuron/GLM-5.1-MLX-2.9bit`, `tiny-random/glm-moe-dsa`.

Config source:

- Official configs fetched from `https://huggingface.co/zai-org/GLM-5/raw/main/config.json` and `https://huggingface.co/zai-org/GLM-5.1/raw/main/config.json`.
- Additional snapshots are under `_sources/`; see `_sources/source_notes.md` for URLs and SHA256 hashes.

Source files inspected:

- `src/transformers/models/glm_moe_dsa/modeling_glm_moe_dsa.py`
- `src/transformers/models/glm_moe_dsa/modular_glm_moe_dsa.py`
- `src/transformers/models/glm_moe_dsa/configuration_glm_moe_dsa.py`
- Neighbor reference for inherited/generated behavior: `glm4_moe_lite` config/modeling only where needed.

Any missing files or assumptions:

- No tokenizer files were inspected because this is text-only causal LM and tokenizer coupling does not change the neural graph beyond `input_ids`, `attention_mask`, and generation EOS/BOS metadata.
- No model execution, imports, or DinoML tests were run.
- Quantized checkpoint configs advertise FP8, AWQ, or MLX quantization fields. This report treats them as loading/provider contracts, not required first-pass dense runtime behavior.

## 2. High-level architecture

`glm_moe_dsa` is a text-only causal decoder with Multi-head Latent Attention style projections, DeepSeek Sparse Attention index selection, and dense-plus-sparse MoE feed-forward layers.

Dataflow:

```text
input_ids/inputs_embeds + attention_mask
-> token embedding
-> repeated decoder layers:
   RMSNorm -> MLA-style self-attention + DSA top-k sparse mask + cache update -> residual
   RMSNorm -> dense MLP or MoE(shared expert + routed experts) -> residual
-> final RMSNorm
-> lm_head over requested token slice
-> logits/generation controller
```

Primary DinoML runtime target: `GlmMoeDsaForCausalLM` inference for prefill and single-token decode.

Stage decomposition:

- CPU/data pipeline: tokenization, chat template, attention mask construction inputs, generation controller, EOS handling.
- GPU graph prefill: embeddings, full decoder, DSA index computation, sparse-mask attention, MoE routing/expert compute, final logits.
- GPU graph decode: one-token hidden input, DynamicCache K/V update, plus separate DSA indexer key-state update per layer.
- Independently cacheable state: ordinary attention K/V cache and per-layer DSA indexer `_cached_keys`; these are distinct state families.

## 3. Important config dimensions

Representative official GLM-5/GLM-5.1 dimensions:

| Field | Value | Source |
| --- | ---: | --- |
| `model_type` | `glm_moe_dsa` | HF config |
| `architectures` | `GlmMoeDsaForCausalLM` | HF config |
| `dtype` | `bfloat16` | HF config |
| `vocab_size` | 154880 | HF config |
| `hidden_size` | 6144 | HF config/source default |
| `num_hidden_layers` | 78 | HF config/source default |
| `num_attention_heads` | 64 | HF config/source default |
| `num_key_value_heads` | 64 | HF config, but generated source expands K/V per `num_attention_heads` |
| `q_lora_rank` | 2048 | HF config/source default |
| `kv_lora_rank` | 512 | HF config/source default |
| `qk_nope_head_dim` | 192 | HF config/source default |
| `qk_rope_head_dim` | 64 | HF config/source default |
| `qk_head_dim` | 256 | Derived in config as `qk_nope_head_dim + qk_rope_head_dim` |
| `v_head_dim` | 256 | HF config/source default |
| `intermediate_size` | 12288 | Dense MLP |
| `moe_intermediate_size` | 2048 | Routed expert hidden width |
| `n_routed_experts` | 256 | HF config/source default |
| `n_shared_experts` | 1 | HF config/source default |
| `num_experts_per_tok` | 8 | HF config/source default |
| `n_group`, `topk_group` | 1, 1 | HF config/source default |
| `index_n_heads` | 32 | DSA indexer |
| `index_head_dim` | 128 | DSA indexer |
| `index_topk` | 2048 | DSA sparse attention |
| `max_position_embeddings` | 202752 | HF config |
| `rope_parameters` | `rope_type=default`, `rope_theta=1000000` | HF config |
| `attention_bias` | `false` | HF config |
| `attention_dropout` | `0.0` | HF config |
| `hidden_act` | `silu` | HF config |
| `tie_word_embeddings` | `false` for GLM-5/5.1; `true` in tiny random | HF config |
| `use_cache` | `true` | HF config |

Representative checkpoint sweep:

| Checkpoint | Role | Layers | Hidden | Heads | MLA dims | MoE | DSA | Quantization/config trap |
| --- | --- | ---: | ---: | ---: | --- | --- | --- | --- |
| `zai-org/GLM-5` | official dense-ish config | 78 | 6144 | 64 | q LoRA 2048, KV LoRA 512, qk 256, v 256 | 256 routed, top-8, 1 shared | 32 index heads, dim 128, topk 2048 | no `quantization_config` |
| `zai-org/GLM-5.1` | official updated weights | 78 | 6144 | 64 | same as GLM-5 | same | same | Transformers version differs (`5.4.0`) |
| `zai-org/GLM-5-FP8` | official FP8 config | 78 | 6144 | 64 | same | same | same | FP8 dynamic activation, E4M3, block 128x128; many excluded modules |
| `QuantTrio/GLM-5-AWQ` | AWQ mirror | 78 | 6144 | 64 | same | same | same | AWQ 4-bit GEMM, excludes self-attention/shared expert/gate/dense early layers |
| `spicyneuron/GLM-5.1-MLX-2.9bit` | MLX quantized mirror | 78 | 6144 | 64 | same | same | same | huge per-module MLX quantization map, not native Transformers quantization |
| `tiny-random/glm-moe-dsa` | debug checkpoint | 2 | 8 | 4 | q LoRA 32, KV LoRA 512, qk 256, v 256 | 256 routed, top-8, tiny hidden | index heads 4, topk 2048 | intentionally dimension-stressful; `hidden_size < qk_head_dim`, tied embeddings |

## 3a. Family variation traps

- Generated files are not authoritative for upstream edits. `modeling_glm_moe_dsa.py` is generated from `modular_glm_moe_dsa.py`, but DinoML runtime matching should inspect generated `modeling`.
- `num_key_value_heads` is present, but generated attention expands K/V with `num_attention_heads` through `kv_b_proj`; do not infer GQA/MQA cache shape from the config field alone.
- `head_dim` is mapped to `qk_rope_head_dim` by `attribute_map`. `qk_head_dim` is 256, not `hidden_size / num_attention_heads` (96 for GLM-5).
- `qk_head_dim == v_head_dim` for official configs, but the source has a flash-attention padding branch if they differ.
- `q_lora_rank=None` is syntactically allowed in attention, but the DSA indexer requires `q_resid`; DinoML should reject `q_lora_rank=None` until source behavior is verified.
- `mlp_layer_types` decides dense versus sparse per layer. Defaults are first 3 dense, rest sparse.
- `indexer_types` controls whether a layer computes top-k or reuses previous layer top-k. The default `index_topk_freq=1` makes every layer `full`; non-default patterns introduce cross-layer top-k state edges.
- DSA indexer owns a separate `_cached_keys` buffer outside `DynamicCache`. Beam reorder, batch select, session reset, and prefill/decode transitions need explicit handling.
- The source creates a dense `index_mask` of shape `[B, S, T]` and uses `scatter_`; a production path should not allocate full masks for long context unless using a bounded fallback.
- Quantized configs include FP8/AWQ/MLX storage metadata. Dense DinoML runtime should reject or route these unless a provider-specific weight loader is implemented.
- `num_nextn_predict_layers` and `first_k_dense_replace` appear in configs but are not read by the inspected in-library modeling path. Treat them as ignored for this source basis.
- `rope_interleave` and `indexer_rope_interleave` appear in configs, but the inspected source uses split-half NeoX/Llama `rotate_half`; treat these flags as ignored by this source basis.

## 4. Operator coverage checklist

Tensor/layout ops:

- Embedding lookup `[B,S] -> [B,S,6144]`.
- View/reshape/transpose/split/cat/expand/contiguous for attention and MoE packing.
- Slice last tokens for `logits_to_keep`.
- `full`, `scatter_`, `masked_fill`, `gather`, `one_hot`, `where`, `nonzero`, `index_add_`.
- `topk` over expert groups, experts, and DSA token scores.

Neural primitives:

- RMSNorm over hidden 6144, q LoRA 2048, and KV LoRA 512.
- LayerNorm for DSA indexer keys over 128.
- Linear projections:
  - token embedding: `[154880, 6144]`.
  - q path: `Linear(6144 -> 2048)` plus `Linear(2048 -> 64 * 256 = 16384)`.
  - KV path: `Linear(6144 -> 512 + 64 = 576)` plus `Linear(512 -> 64 * (192 + 256) = 28672)`.
  - attention output: `Linear(64 * 256 = 16384 -> 6144)`.
  - DSA indexer: `Linear(2048 -> 32 * 128 = 4096)`, `Linear(6144 -> 128)`, `Linear(6144 -> 32)`.
  - dense MLP: gate/up `Linear(6144 -> 12288)`, down `Linear(12288 -> 6144)`.
  - routed expert packed gate/up: per expert `[2 * 2048, 6144]`, down `[6144, 2048]`.
  - shared expert MLP: gate/up `Linear(6144 -> 2048)`, down `Linear(2048 -> 6144)`.
  - LM head: `Linear(6144 -> 154880)`, bias false.
- SiLU/SwiGLU style `silu(gate) * up`.
- Sigmoid router and ReLU in DSA scoring.

Attention primitives:

- Causal self-attention with MLA-style projection and full expanded K/V cache.
- DSA top-k sparse additive attention mask.
- Dense eager attention fallback: matmul, scale, additive mask, softmax, dropout, matmul.
- SDPA-compatible backend is advertised. Flash attention is disabled in source class, with `kernels-community/flash-mla` listed as compatible but not supported by default.

Position/rotary ops:

- RoPE cos/sin generation in fp32 with `rope_theta=1000000`, `dim=head_dim` where `head_dim` maps to `qk_rope_head_dim=64`.
- Split-half `rotate_half` application for q rope slice, k rope slice, and DSA indexer q/k rope slices.

Generation/cache ops:

- `DynamicCache(config)` for per-layer expanded attention K/V.
- Separate per-layer DSA indexer key cache `[B,T,index_head_dim]`, reset when `seq_len > 1`.
- Position IDs based on `past_key_values.get_seq_length()`.
- Cache reorder/select support is not implemented for DSA `_cached_keys` in the inspected source.

Quantized/packed weight metadata ops:

- FP8 configs require module exclusion handling and special linear weight materialization. Source notes keep `indexer.weights_proj` in fp32-ish module handling.
- AWQ/MLX configs are provider-specific weight formats. First dense DinoML integration should reject them or require an explicit dequant/materialization plan.

## 5. Layer/block breakdown

Decoder block, repeated 78 times for GLM-5/5.1:

```text
residual = x
x = RMSNorm(x)                                      # [B,S,6144]

q_resid = RMSNorm(Linear(6144 -> 2048)(x))
q = Linear(2048 -> 16384)(q_resid)
q = view [B,S,64,256] -> transpose [B,64,S,256]
q_nope, q_pe = split [192,64]
q_pe = RoPE(q_pe)

compressed_kv = Linear(6144 -> 576)(x)
k_compressed, k_pe = split [512,64]
k_compressed = RMSNorm(k_compressed)
kv = Linear(512 -> 28672)(k_compressed)
kv = view [B,S,64,448]
k_nope, v = split [192,256], transpose to [B,64,S,D]
k_pe = RoPE(view [B,1,S,64]), expand to 64 heads
k = cat(k_nope, k_pe)                               # [B,64,S,256]
k,v = DynamicCache.update(k,v,layer)

topk = DSAIndexer(x, q_resid, rope, mask, use_cache)
combined_mask = causal_mask + DSA top-k mask
attn = attention(q, k, v, combined_mask, scale=1/sqrt(256))
x = residual + Linear(16384 -> 6144)(attn)

residual = x
x = RMSNorm(x)
if layer type dense:
  x = Linear(12288 -> 6144)(silu(Linear(6144 -> 12288)(x)) * Linear(6144 -> 12288)(x))
else:
  routed = sigmoid(Linear(6144 -> 256, fp32)(x))
  top groups/expert topk -> per-token top-8 expert indices/weights
  experts = sum_i weight_i * expert_i_swiglu(x)
  shared = shared_swiglu(x)
  x = experts + shared
x = residual + x
```

Bias:

- Official configs set `attention_bias=false`.
- MLP/expert projections are bias-free.
- DSA `k_norm` is PyTorch LayerNorm and has scale/bias behavior; RMSNorm modules have weight only.

## 6. Attention requirements

Required attention variant:

- Causal self-attention only. No cross-attention.
- Logical MHA with 64 query heads and 64 expanded K/V heads for official configs.
- Query/key width per head: `qk_head_dim=256`.
- Value width per head: `v_head_dim=256`.
- Scale: `qk_head_dim ** -0.5`.
- Masking: causal mask plus DSA top-k mask. The DSA mask allows only selected key positions and sets all other positions to `-inf`.
- Packed/varlen support: not explicit in source. Flash MLA receives `indices=topk_indices` only when a compatible backend is selected, but source class declares `_supports_flash_attn=False`.
- Sliding/local attention: none. Sparsity is dynamic top-k over all visible keys.
- RoPE: applied before cache update; cached keys are post-RoPE, expanded full keys.
- KV cache shape after update: keys `[B,64,T,256]`, values `[B,64,T,256]`.
- DSA cache shape: per layer `_cached_keys` `[B,T,128]` for indexer scoring.

Dense fallback attention:

```text
attn_weights = matmul(q, k.transpose(-2,-1)) * scale
attn_weights += combined_mask
attn_weights = softmax(attn_weights, dim=-1)
attn_output = matmul(attn_weights, v)
```

Admission guidance:

- Prefill can be admitted with dense fallback if `[B,S,T]` DSA score and mask memory is bounded by shape guards.
- Decode must carry both DynamicCache K/V and DSA indexer key cache. A KV-only decode path is incorrect.
- Beam search or batch reordering should be rejected until DSA `_cached_keys` reorder is implemented alongside `DynamicCache.reorder_cache`.

## 7. Position encoding and custom math

RoPE generation:

```python
dim = qk_rope_head_dim  # 64 in official configs
inv_freq = 1.0 / (rope_theta ** (arange(0, dim, 2) / dim))
freqs = inv_freq[None, :, None] @ position_ids[:, None, :]
emb = cat([freqs, freqs], dim=-1)
cos = cos(emb).to(x.dtype)
sin = sin(emb).to(x.dtype)
```

Application:

```python
def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return cat([-x2, x1], dim=-1)

def apply_rope(x, cos, sin, unsqueeze_dim):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return x * cos + rotate_half(x) * sin
```

Precompute opportunity:

- `inv_freq` is static per config.
- Cos/sin can be cached by max position and dtype, but dynamic RoPE update decorator means non-default rope types should be admitted only after source-compatible dynamic behavior is implemented.
- Official configs use `rope_type=default`; first integration should reject non-default `rope_parameters`.

## 8. Preprocessing and input packing

Runtime neural inputs:

- `input_ids` `[B,S]` or `inputs_embeds` `[B,S,6144]`, exactly one required.
- `attention_mask` optional; passed into Transformers causal-mask helper before layers.
- `position_ids` optional; default is `arange(S) + past_seen_tokens`.

CPU/data-pipeline work:

- Tokenization, chat template, padding strategy, and generation EOS/BOS handling.
- No image/audio/video preprocessing.

GPU/runtime work:

- Causal mask creation or equivalent masked attention metadata.
- DSA indexer top-k per layer and mask creation.
- For efficient production lowering, avoid materializing full `[B,S,T]` masks when a sparse-attention backend can consume top-k indices directly.

Generation-controller metadata:

- GLM-5/5.1 configs use `eos_token_id=[154820,154827,154829]`, `pad_token_id=154820`, and no tied embeddings.
- Tiny random uses tied embeddings and should not be used to infer official weight aliasing.

## 9. Graph rewrite / lowering opportunities

### Rewrite: Q LoRA projection chain

Source pattern:

```text
RMSNorm(Linear(hidden -> q_lora_rank)) -> Linear(q_lora_rank -> heads*qk_head_dim)
```

Replacement:

```text
GEMM(hidden, Wq_a) -> RMSNorm -> GEMM(q_resid, Wq_b)
```

Preconditions:

- `q_lora_rank` is not `None`.
- Bias behavior matches `attention_bias`; official configs have no bias.
- Preserve `q_resid` because DSA indexer consumes it.

Failure cases:

- Do not fold the two GEMMs across RMSNorm.
- Reject `q_lora_rank=None` until DSA behavior is defined.

Parity sketch:

- Compare q_resid, q_nope/q_pe split, post-RoPE q, and final attention logits for one layer.

### Rewrite: MLA KV expansion

Source pattern:

```text
Linear(hidden -> kv_lora_rank + rope_dim)
split(kv_rank, rope_dim)
RMSNorm(kv_rank)
Linear(kv_rank -> heads*(qk_nope_dim + v_dim))
split(k_nope, value)
cat(k_nope, expanded_rope_k)
```

Replacement:

```text
GEMM -> split -> RMSNorm -> GEMM -> layout pack for K/V cache
```

Preconditions:

- `num_key_value_heads == num_attention_heads` or source-proven expansion semantics are preserved.
- K cache stores post-RoPE full `[qk_nope + rope]` width, not compressed KV.

Failure cases:

- Do not replace with compressed-cache MLA decode unless a new cache class and parity tests are added.

### Rewrite: DSA top-k mask to sparse attention metadata

Source pattern:

```text
index_scores = sum_h(relu(dot(q_h, k)) * weights_h)
topk_indices = topk(index_scores, k=index_topk)
index_mask = full(-inf); index_mask.scatter(-1, topk_indices, 0)
combined_mask = index_mask + causal_mask
dense_attention(q,k,v,combined_mask)
```

Replacement:

```text
DSAIndex(q_resid, hidden, indexer_cache, causal metadata) -> topk_indices
SparseAttention(q,k,v,topk_indices,causal_mask)
```

Preconditions:

- Sparse backend exactly preserves top-k tie behavior or tests tolerate/source documents tie differences.
- Top-k indices are within causal-visible keys.
- `index_topk <= T`; source uses `min(index_topk,total_len)`.

Failure cases:

- Dense fallback may be required for small bounded shapes.
- General boolean scatter admission is too broad; this is a dedicated indexed mask-write pattern.

### Rewrite: MoE routed experts to grouped GEMM

Source pattern:

```text
router = sigmoid(fp32_linear(x))
top groups -> top experts -> gather weights
for expert in active_experts:
  selected = x[token_idx]
  gate, up = linear(selected, packed_gate_up[expert]).chunk(2)
  y = linear(silu(gate) * up, down[expert])
  index_add(token_idx, y * route_weight)
output = routed + shared_expert(x)
```

Replacement:

```text
RouterTopK -> token permutation/by-expert buckets -> grouped GEMM gate_up -> SiLU*up -> grouped GEMM down -> weighted scatter-add -> shared GEMM path add
```

Preconditions:

- Static expert count 256, active experts 8, expert hidden 2048 for GLM-5.
- Preserve fp32 router logits and `e_score_correction_bias`.
- Preserve top-k normalization if `norm_topk_prob=true`.

Failure cases:

- Dynamic expert bucketing and empty experts need explicit shape/state metadata.
- Quantized expert weights require separate provider admission.

### Rewrite: last-token-only logits

Source pattern:

```text
logits = lm_head(hidden_states[:, slice_indices, :])
```

Replacement:

```text
Slice hidden before GEMM -> GEMM only for kept tokens
```

Preconditions:

- `logits_to_keep` is int or static tensor slice equivalent.
- Loss is not requested, or labels align with sliced logits.

## 10. Kernel fusion candidates

Highest priority:

- RMSNorm for hidden, q LoRA, KV LoRA. It is on every block and gates attention parity.
- Dense GEMMs for q/kv/o/mlp/lm_head using CUTLASS-backed `gemm_rrr`/`gemm_rcr` with bias disabled.
- DSA indexer score kernel: q/k RoPE, dot, ReLU, head-weighted sum, mask add, top-k. This dominates long-context sparse selection.
- Sparse attention consuming top-k indices directly. Dense `[B,S,T]` masks are not production-viable at 202k context.
- MoE grouped GEMM and weighted scatter-add for 256 experts, top-8 active.

Medium priority:

- Q/K RoPE fused with projection layout pack.
- SwiGLU fused epilogue for dense/shared MLP.
- Router sigmoid/top-k/group filtering kernel.
- Last-token-only LM head.
- Cache update kernels for expanded K/V plus DSA key cache.

Lower priority:

- Flash MLA hub-kernel compatibility. Source advertises a compatible implementation but disables flash attention support.
- Non-default RoPE variants.
- Quantized FP8/AWQ/MLX loaders.

## 11. Runtime staging plan

Stage 1: config and dense weights

- Parse config, reject unsupported flags, load dense/bfloat16 weights.
- Reject quantized checkpoints unless a provider is selected.

Stage 2: single-layer dense fallback parity

- Implement one decoder block with dense attention and dense DSA mask for small `S`.
- Validate q/K/V projections, RoPE, DSA top-k, MoE routing, and logits.

Stage 3: full prefill bounded parity

- Run all layers for small/medium sequence lengths using dense DSA fallback.
- Add shape guards for `B*S*T` mask and score memory.

Stage 4: decode state parity

- Implement DynamicCache K/V plus DSA indexer key cache.
- Validate prefill reset, one-token append, and position ID increments.
- Reject beam reorder until DSA cache reorder exists.

Stage 5: optimized attention and DSA

- Lower DSA top-k to dedicated sparse metadata.
- Add sparse attention backend that consumes selected key indices.

Stage 6: optimized MoE

- Add router/top-k/grouped expert GEMM/token bucketing.
- Add shared expert fusion and weighted scatter-add.

Stage 7: quantized/provider integration

- Admit FP8/AWQ/GGUF-like storage only through explicit manifests, dequant/materialization policies, and provider tests.

## 12. Parity and validation plan

- RoPE unit tests: compare cos/sin and `apply_rope` for official dims and odd position offsets.
- DSA indexer unit tests: no cache prefill, decode append, `seq_len > 1` cache reset, causal mask application, top-k min with short sequences.
- Attention unit tests: compare q/k/v cache tensors before and after update; compare dense attention with DSA mask.
- MoE router tests: sigmoid logits, correction bias, group top-2 sum, group top-k mask, expert top-k, normalization, scaling.
- Expert tests: packed `gate_up_proj` split order and `index_add_` accumulation with duplicate token indices.
- Single-layer parity: fp32 then bf16 tolerances.
- Full prefill logits parity for small `S` and official-ish shapes where feasible.
- Decode token parity: prefill N tokens, decode one token, compare logits and both cache families.
- Quantized configs: loader rejection tests with explicit error messages unless provider support lands.

Recommended tolerances:

- fp32 custom math: `rtol=1e-4`, `atol=1e-5` for layer-local tests.
- bf16 full block: start with `rtol=3e-2`, `atol=3e-2`; tighten after fused kernels are stable.
- Top-k tests must compare indices exactly for deterministic scores; include no-tie random cases and tie-policy tests if needed.

## 13. Performance probes

- Prefill DSA score/top-k time versus dense attention time by `S` in `{512,2048,8192,32768}`.
- Memory probe for dense `[B,S,T]` index masks and score tensors.
- Decode tokens/sec with K/V cache only versus K/V plus DSA cache update.
- MoE router/top-k overhead by token count.
- Grouped expert GEMM utilization by batch and sequence length.
- Shared expert versus routed expert time split.
- LM head last-token GEMM time versus full-sequence logits.
- Quantized load/dequant time for FP8/AWQ variants once a provider exists.

## 14. Skip/defer list

- Training, loss, gradient checkpointing.
- Beam search and cache reorder until DSA indexer cache reorder is implemented.
- Flash MLA hub kernel as first target; dense/sparse source-compatible path first.
- Non-default RoPE variants and dynamic RoPE scaling.
- Quantized FP8/AWQ/MLX loading.
- Tensor parallel, expert parallel, pipeline parallel plans in config.
- Multi-token prediction / `num_nextn_predict_layers`; not implemented in inspected source path.
- General boolean scatter support. Admit only the bounded DSA indexed-mask pattern.

## 15. Final implementation checklist

- [ ] Parse `GlmMoeDsaConfig` and reject unsupported config flags.
- [ ] Load dense bf16/fp32 weights with correct tied/untied embedding handling.
- [ ] Implement RMSNorm and DSA LayerNorm coverage.
- [ ] Implement split-half RoPE for q, k, and DSA indexer paths.
- [ ] Implement MLA projection path with expanded K/V cache.
- [ ] Implement DSA indexer score/top-k and separate key cache state.
- [ ] Implement bounded dense DSA mask fallback with memory guards.
- [ ] Implement causal attention parity with additive sparse mask.
- [ ] Implement dense MLP SwiGLU.
- [ ] Implement MoE router group top-k, expert top-k, normalization, and scaling.
- [ ] Implement packed expert gate/up split and weighted scatter-add.
- [ ] Implement shared expert path and routed-plus-shared add.
- [ ] Implement final RMSNorm and last-token-only LM head.
- [ ] Add prefill parity tests.
- [ ] Add decode parity tests covering DynamicCache and DSA `_cached_keys`.
- [ ] Add explicit rejection tests for beam reorder and quantized configs.
- [ ] Add performance probes for DSA top-k, sparse attention, MoE grouped GEMM, and logits slicing.
