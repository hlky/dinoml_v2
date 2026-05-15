# DBRX Transformers Family Audit

Primary target: `DbrxForCausalLM` inference and generation on CUDA. This is a source/config audit only; no DinoML runtime code was edited, no DinoML tests were run, and no commit was made.

## 1. Source basis

```text
Transformers commit/version: local checkout transformers at b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model family: dbrx
Primary task: causal LM prefill/decode/generation
Local source root: transformers
```

Source files inspected:

- `transformers/src/transformers/models/dbrx/configuration_dbrx.py`
- `transformers/src/transformers/models/dbrx/modeling_dbrx.py`
- `transformers/src/transformers/models/dbrx/modular_dbrx.py`
- Cross-checks: `src/transformers/cache_utils.py`, `src/transformers/masking_utils.py`, and existing DinoML audits for `mixtral` and `qwen3_moe`.

Source URLs at the inspected commit:

- `https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/dbrx/configuration_dbrx.py`
- `https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/dbrx/modeling_dbrx.py`
- `https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/dbrx/modular_dbrx.py`

Representative config metadata inspected from Hugging Face raw files:

- `https://huggingface.co/Rocketknight1/dbrx-tiny-random/raw/main/config.json`
- `https://huggingface.co/alpindale/dbrx-instruct/raw/main/config.json`
- `https://huggingface.co/mlx-community/dbrx-instruct-4bit/raw/main/config.json`

Gated or unavailable configs:

- `https://huggingface.co/databricks/dbrx-instruct/raw/main/config.json` returned HTTP 401.
- `https://huggingface.co/databricks/dbrx-base/raw/main/config.json` returned HTTP 401.
- `https://huggingface.co/transformers-community/dbrx-instruct/raw/main/config.json` returned HTTP 401.

Authoritative source note: `modeling_dbrx.py` is generated from `modular_dbrx.py`; future Transformers edits should target `modular_dbrx.py`. The generated file is still the concrete runtime source for this audit because it expands inherited Llama RoPE/eager attention helpers.

Important source/config compatibility note: the current in-library `DbrxAttentionConfig` declares `attn_pdrop`, `clip_qkv`, and `kv_n_heads`, while legacy DBRX configs carry `attn_config.rope_theta`. The generated modeling code also reads `config.rope_parameters["rope_type"]` for RoPE and `attn_config.rope_theta` inside attention init. DinoML should normalize legacy `attn_config.rope_theta` into the current `rope_parameters` shape, or reject configs whose RoPE fields cannot be normalized. The pinned generated `DbrxExperts.forward` also appears to reshape token hidden states by `ffn_hidden_size`; for official-style `d_model=6144, ffn_hidden_size=10752`, that is shape-inconsistent with the surrounding expert GEMMs. Treat this as a source-version hazard and validate against an actual loadable Transformers release/checkpoint before claiming parity.

## 2. High-level architecture

DBRX is a text-only decoder-only sparse MoE language model:

```text
token ids -> embedding -> N decoder blocks -> final LayerNorm -> lm_head -> logits/sampling
                         | each block: LayerNorm -> GQA self-attention -> residual
                         |             LayerNorm -> top-4 sparse MoE GLU -> residual
prefill: full prompt causal attention + KV cache fill
decode: new token(s) + cache update + MoE per token + last-token logits
```

Stage decomposition:

- CPU/data pipeline: tokenizer, prompt/chat formatting, padding mask construction, generation sampling controls.
- GPU/runtime prefill: token embedding, shared RoPE cos/sin generation, repeated decoder blocks, causal GQA attention, MoE routing/expert execution, final LayerNorm, logits.
- GPU/runtime decode: position IDs from cache length, Wqkv for new tokens, optional QKV clamp, RoPE, KV cache append, attention over cached KV, MoE, last-token logits.
- Generation controller: standard `GenerationMixin`; no image/audio processor or packed multimodal metadata is involved.

Implemented heads:

- Required for target: `DbrxForCausalLM`.
- Optional/deferred: base `DbrxModel` hidden-state output.
- Training/diagnostic: labels/loss, router aux loss, router logits.

## 3. Important config dimensions

Source defaults from `DbrxConfig`, `DbrxAttentionConfig`, and `DbrxFFNConfig`:

| Field | Default / behavior |
| --- | --- |
| `vocab_size` | 32000 |
| `d_model` / `hidden_size` | 2048 |
| `n_layers` / `num_hidden_layers` | 24 |
| `n_heads` / `num_attention_heads` | 16 |
| `head_dim` | Source attention uses `d_model // n_heads`; not an explicit config field |
| `attn_config.kv_n_heads` | 1 |
| `attn_config.clip_qkv` | `None`; production configs use 8 |
| `attn_config.attn_pdrop` | 0.0 |
| `max_seq_len` / `max_position_embeddings` | 2048 |
| `resid_pdrop`, `emb_pdrop` | 0.0 |
| `ffn_config.hidden_size` | 6144 default sub-config value; should match `d_model` for real checkpoints |
| `ffn_config.ffn_hidden_size` | 3584 default; production configs use 10752 |
| `ffn_config.moe_num_experts` | 4 default; production configs use 16 |
| `ffn_config.moe_top_k` | 1 default; production configs use 4 |
| `ffn_config.moe_normalize_expert_weights` | 1.0 by source default; legacy configs often omit it |
| `ffn_config.ffn_act_fn` | defaults to `{"name": "silu"}` |
| `use_cache` | true |
| `tie_word_embeddings` | false; `validate_architecture` rejects true |
| `output_router_logits` | false |

Representative checkpoint/config sweep. Dimensions are from raw `config.json` files unless noted.

| Model id | Source status | Layers | H | Heads/KV | Head dim | Experts/top-k | Expert I | Max seq | QKV clamp | RoPE theta | Dtype/quant |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `Rocketknight1/dbrx-tiny-random` | open tiny/debug mirror | 2 | 4 | 4/2 | 1 | 16/4 | 8 | 32768 | 8 | 500000 | fp16 metadata |
| `alpindale/dbrx-instruct` | open mirror of DBRX instruct config | 40 | 6144 | 48/8 | 128 | 16/4 | 10752 | 32768 | 8 | 500000 | bf16 |
| `mlx-community/dbrx-instruct-4bit` | MLX quantized mirror; config is noisy/legacy | 40 | 6144 | 48/8 | 128 | 16/4 | 10752 | 32768 | 8 | 500000 | bf16 logical, 4-bit MLX quant metadata |

Operator-significant variation is small across available DBRX configs: the production architecture uses 48 query heads, 8 KV heads, head dim 128, 16 experts, top-4 routing, `clip_qkv=8`, and 32k context. The open tiny config preserves the same expert/top-k pattern but uses `d_model=4`, so it is useful for parser/shape tests but not performance.

## 3a. Family variation traps

- DBRX uses `LayerNorm(bias=False)`, not RMSNorm. This differs from Mixtral and Qwen3-MoE and changes normalization kernels and parity tolerances.
- Attention is a single packed `Wqkv` linear, not three separate `q_proj/k_proj/v_proj` modules. Split order is `[Q, K, V]` with widths `[H, KVH*D, KVH*D]`.
- `clip_qkv` clamps the packed QKV tensor before splitting. Production configs use `clip_qkv=8`; this clamp is required for parity and should not be fused away unless preserved.
- GQA is normal: production config has 48 query heads and 8 KV heads, so `num_key_value_groups=6`. Cache storage uses KV heads only.
- RoPE config is version-sensitive. Legacy configs carry `attn_config.rope_theta=500000`; current generated source expects normalized `config.rope_parameters`. Add config normalization/admission before lowering.
- MoE uses 16 experts/top-4, between Mixtral's 8/top-2 and Qwen3-MoE's 128/top-8. Router/scatter cost is materially larger than Mixtral but far smaller than Qwen3-MoE.
- DBRX expert weights are separate flattened parameters: `w1`, `v1`, and `w2`, each shaped `[E * I, H]` in source construction. Mixtral/Qwen3-MoE use 3D packed `gate_up_proj[E, 2I, H]` plus `down_proj[E, H, I]`.
- DBRX selected expert weights are normalized with an Lp norm when `moe_normalize_expert_weights` is not `None`; official legacy configs omit the field, so current source default may imply p=1 normalization after config migration. Do not assume Mixtral-style sum renormalization unless the source/config path proves it.
- The source has no sliding-window/local attention branch in `DbrxModel`; it always calls `create_causal_mask`.
- `tie_word_embeddings=true` is rejected by current config validation even though the generated LM class declares a tied weight key. Treat embedding and LM head as untied logical parameters for DBRX.
- Layout translation is low value for text DBRX. Protect sequence/head/cache axes, top-k over experts, and expert gather/scatter with a no-layout-translation guard.

## 4. Operator coverage checklist

Tensor/layout ops:

- Integer embedding lookup: `input_ids [B,S] -> [B,S,H]`.
- `view`, `reshape`, `transpose`, `contiguous`, `split`, `unsqueeze`, slicing for logits.
- `clamp` on packed QKV.
- MoE routing ops: `softmax`, `topk`, `norm` over selected top-k weights, `one_hot`, `permute`, `sum`, `greater`, `nonzero`, `where`, token gather, `index_add_`.
- Residual adds, dropout as inference no-op, dtype casts after LayerNorm.

Neural network primitives:

- Bias-free LayerNorm over hidden axis, `H=6144` for production.
- Packed Wqkv production GEMM: `6144 -> 6144 + 2 * 8 * 128 = 8192`, bias false.
- Output projection: `6144 -> 6144`, bias false.
- Router: `6144 -> 16`, bias false.
- Per-expert GLU:
  - gate `w1`: `6144 -> 10752`
  - up `v1`: `6144 -> 10752`
  - down `w2`: `10752 -> 6144`
  - activation `silu(gate) * up`
- LM head: `6144 -> 100352`, bias false.

Attention primitives:

- Causal self-attention with GQA and RoPE.
- Eager fallback repeats K/V from `[B,8,T,128]` to `[B,48,T,128]`; optimized path should avoid physical repeat.
- Attention math order: QK matmul, scale by `head_dim**-0.5`, add mask, fp32 softmax, cast to query dtype, dropout in training only, AV matmul.
- Backend dispatch through `ALL_ATTENTION_FUNCTIONS`: eager, SDPA, FlashAttention, flex attention, or custom integration.

Position/rotary ops:

- Default RoPE inverse frequencies with theta 500000 for production legacy configs.
- `rotate_half` and apply RoPE to Q/K before cache update.
- Dynamic RoPE utilities are present through current generated code; not observed in representative configs.

Generation/cache ops:

- `DynamicCache(config)` creation when `use_cache` and no cache is supplied.
- Cache length used to derive default `position_ids`.
- KV cache update per layer after RoPE; cached K is post-RoPE, cached V is raw value projection.
- `logits_to_keep`: int or tensor index selection before LM head.

Distributed/tensor-parallel ops:

- Current DBRX config/source exposes only `_tp_plan = {"lm_head": "colwise_gather_output"}` in the LM class, unlike Mixtral/Qwen3-MoE config-level plans for attention/expert tensors. First DinoML support can be single-GPU, but expert layout should preserve future grouped/expert parallel metadata.

## 5. Layer/block breakdown

Decoder block, repeated `n_layers` times:

```text
x0: [B,S,H]
residual0 = x0
x = LayerNorm_H_no_bias(x0)
qkv = Linear_no_bias(H -> H + 2*KVH*D)(x)
qkv = clamp(qkv, -clip_qkv, clip_qkv) if clip_qkv is not None
q, k, v = split(qkv, [H, KVH*D, KVH*D], dim=-1)
q = q.view(B,S,QH,D).transpose(1,2)      # [B,QH,S,D]
k = k.view(B,S,KVH,D).transpose(1,2)     # [B,KVH,S,D]
v = v.view(B,S,KVH,D).transpose(1,2)
q, k = RoPE(q, k, cos, sin)
k, v = cache.update(k, v, layer_idx) if cache enabled
attn = causal_GQA_attention(q, k, v, mask, scale=D**-0.5)
x1 = residual0 + Linear_no_bias(H -> H)(attn.reshape(B,S,H))

residual1 = x1
x = LayerNorm_H_no_bias(x1)
flat = x.reshape(B*S, H)
router_logits = Linear_no_bias(H -> E)(flat)
router_probs = softmax(router_logits, dim=1)
top_values, top_indices = topk(router_probs, K=top_k, dim=-1)
if moe_normalize_expert_weights is not None:
    top_values = top_values / norm(top_values, p=moe_normalize_expert_weights, dim=-1, keepdim=True)
for active expert e:
    token_idx, selected_slot = where(one_hot(top_indices)[e])
    gate = tokens @ w1[e].T or equivalent source layout
    up = tokens @ v1[e].T
    y = (silu(gate) * up) @ w2[e]
    scatter_add(token_idx, y * top_values[token_idx, selected_slot])
x2 = residual1 + moe_output.reshape(B,S,H)
```

Model tail:

```text
hidden = final LayerNorm_H_no_bias(hidden)
slice_indices = slice(-logits_to_keep, None) if logits_to_keep is int else logits_to_keep
logits = lm_head(hidden[:, slice_indices, :])
```

## 6. Attention requirements

Variant: decoder causal self-attention with packed Wqkv, optional QKV clamp, GQA, RoPE, and autoregressive KV cache.

Production shapes:

- Hidden input: `[B,S,6144]`.
- Packed Wqkv output: `[B,S,8192]`.
- Query after split/reshape: `[B,48,S,128]`.
- Key/value after split/reshape: `[B,8,S,128]`.
- Full cache per layer stores key/value as `[B,8,T,128]`.
- Eager fallback expands K/V to `[B,48,T,128]`; DinoML optimized attention should consume grouped KV directly.
- Attention output before O projection: `[B,S,6144]`.

Cache semantics:

- If `use_cache` and no cache is passed, source creates `DynamicCache(config=self.config)`.
- Default `position_ids` are `arange(S) + past_seen_tokens`, unsqueezed to `[1,S]`.
- RoPE is applied before `past_key_values.update`, so cached keys are stored post-RoPE.
- There is no DBRX-specific sliding-window cache in the inspected source.

Math order in eager fallback:

```text
Wqkv -> clamp -> split/reshape -> RoPE(q,k) -> cache.update
repeat_kv(k/v) -> matmul(q, k^T) * scale -> add causal/padding mask
softmax(dtype=float32) -> cast to query dtype -> dropout(training only) -> matmul(weights, v)
transpose/reshape -> out_proj
```

Backend compatibility: source advertises FlashAttention, SDPA, flex attention, and generic attention backend support. For DinoML, the production target should be native GQA causal attention with cache shape `[B,KVH,T,D]`, plus a separate pre-attention QKV clamp/fusion hook.

## 7. Position encoding and custom math

Default RoPE, after normalizing legacy `attn_config.rope_theta` to `rope_parameters`:

```python
dim = config.d_model // config.n_heads
base = config.rope_parameters["rope_theta"]  # 500000 in DBRX production configs
inv_freq = 1.0 / (base ** (arange(0, dim, 2).float() / dim))
freqs = (inv_freq[None, :, None] @ position_ids[:, None, :].float()).transpose(1, 2)
emb = cat((freqs, freqs), dim=-1)
cos = emb.cos() * attention_scaling
sin = emb.sin() * attention_scaling
```

Apply RoPE:

```python
def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return cat((-x2, x1), dim=-1)

def apply_dbrx_rope(q, k, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin
```

Precompute opportunities:

- `inv_freq` is static for default RoPE and production theta.
- Cos/sin depend on runtime `position_ids` and sequence length; source computes them once per model forward and shares them across layers.
- Decode can compute only the new position row(s).

## 8. Preprocessing and input packing

Text-only preprocessing:

- GPU graph consumes `input_ids` or `inputs_embeds`, exactly one.
- Optional graph inputs: `attention_mask`, `position_ids`, `past_key_values`, `use_cache`, `logits_to_keep`.
- If `position_ids` is omitted, the model derives it from cache length and current sequence length.
- No image/audio/video tensors, placeholder expansion, `cu_seqlens`, codebooks, or modality stitch ops are present.

Generation-controller behavior:

- Standard tokenizer and `GenerationMixin` handling are outside the compiled graph.
- Available open configs do not provide reliable official `generation_config.json` because official repos were gated in this environment.
- `logits_to_keep=1` or explicit indices are important for efficient decode because vocab is `100352`.

## 9. Graph rewrite / lowering opportunities

### Rewrite: packed Wqkv with clamp

Source pattern:

```text
qkv = Wqkv(LayerNorm(x))
qkv = clamp(qkv, -clip_qkv, clip_qkv)
q, k, v = split(qkv, [H, KVH*D, KVH*D])
```

Replacement pattern:

```text
PackedLinear(H -> H + 2*KVH*D) -> optional clamp epilogue -> split views
```

Preconditions:

- Weight is bias-free and split order is `[Q, K, V]`.
- Clamp bounds are symmetric and scalar, or absent.
- Provider can expose split outputs without copying or a following fused attention can consume the packed layout.

Shape equations:

- Production: `H=6144`, `QH=48`, `KVH=8`, `D=128`, packed width `8192`.

Weight transform:

- Preserve source row order: first `H` rows are Q, next `KVH*D` rows are K, final `KVH*D` rows are V.

Failure cases:

- Legacy remote-code checkpoints with different packed layout.
- Configs whose normalized RoPE/head dimensions do not satisfy `H == QH * D`.

Parity test sketch:

- Compare packed output, clamped output, and split Q/K/V tensors against Transformers for tiny and production shapes.

### Rewrite: RoPE + GQA attention + cache update

Source pattern:

```text
RoPE(q,k) -> cache.update(k,v) -> repeat_kv -> attention
```

Replacement pattern:

```text
FusedGQAAttention(q, k, v, cos, sin, cache, mask_policy, qkv_clip_already_applied)
```

Preconditions:

- Causal self-attention.
- KV cache stores post-RoPE K and raw V.
- Backend supports `QH/KVH=6` grouping for production DBRX.
- No sliding-window branch is required for inspected source.

Failure cases:

- Attention outputs requested from optimized path if backend cannot return weights.
- Non-default/dynamic RoPE types not admitted.

Parity test sketch:

- Prefill plus multi-step decode with omitted `position_ids`; compare cache length, cache tensors, attention output, and logits.

### Rewrite: DBRX flattened experts -> grouped expert GEMM

Source pattern:

```text
one_hot(top_k_index).permute(...)
for active expert:
    gather tokens
    gate = x @ w1[e].T
    up = x @ v1[e].T
    y = (silu(gate) * up) @ w2[e]
    index_add(token_idx, y * route_weight)
```

Replacement pattern:

```text
RouterTopK -> expert assignment histogram/sort -> grouped gate/up GEMMs -> SiLU multiply -> grouped down GEMM -> weighted scatter-add
```

Preconditions:

- Inference mode, so jitter is disabled.
- `E=16`, `K=4` or provider declares support for configured values.
- Selected expert normalization exactly matches source `torch.norm(..., p=moe_normalize_expert_weights)`.
- Expert storage is reshaped from flattened `[E*I,H]` tensors into `[E,I,H]` views without changing row order.

Shape equations:

- `M = B*S`.
- Router logits `[M,16]`; selected experts/weights `[M,4]`.
- Total routed rows before grouping: `4*M`.
- Per expert: `[M_e,H] x [H,I] -> [M_e,I]` for gate/up and `[M_e,I] x [I,H] -> [M_e,H]` for down.

Weight transform:

```python
w1_e = w1.view(E, I, H)[e]
v1_e = v1.view(E, I, H)[e]
w2_e = w2.view(E, I, H)[e]  # source then uses w2_e.T for down projection
```

Failure cases:

- The current pinned generated source's hidden-state reshape by `ffn_hidden_size` must be resolved/validated before using it as a numerical reference.
- Empty experts must not launch invalid GEMMs.
- Top-k tie ordering and scatter-add accumulation order can affect exact parity.

Parity test sketch:

- Compare router logits, selected indices, normalized selected weights, per-expert GLU output, and final scatter for small deterministic inputs with no top-k ties.

### Rewrite: last-token-only LM head

Source pattern:

```text
logits = lm_head(hidden_states[:, slice_indices, :])
```

Replacement pattern:

```text
GatherLastTokenOrIndices -> GEMM(H -> vocab)
```

Preconditions:

- Generation only needs last token or a known subset of prompt positions.
- Labels/loss path is disabled.

Failure cases:

- Prompt log-prob evaluation or training requires full `[B,S,V]` logits.

## 10. Kernel fusion candidates

Highest priority:

- LayerNorm. DBRX uses bias-free LayerNorm three times per layer group path, not RMSNorm; implement fp32 reduction parity.
- Packed Wqkv + clamp + split. This is a DBRX-specific attention entry pattern and avoids three projection launches.
- Native GQA attention with KV cache. Production grouping is 48/8, and eager K/V repeat is too expensive.
- RoPE + cache write. Must preserve post-RoPE K storage.
- MoE routing + grouped expert GEMM. Top-4 over 16 experts is the major DBRX-specific runtime challenge.
- Separate gate/up expert GEMMs plus SiLU multiply. Unlike Mixtral/Qwen3-MoE, gate/up are not source-packed into one 3D `gate_up_proj`, so a provider should either fuse two RHS matrices or prepack them.

Medium priority:

- Specialized router softmax/top-k/norm for `E=16,K=4`.
- Weighted scatter-add for top-4 contributions.
- Last-token-only LM head for `V=100352`.
- Expert residency/offload probes. DBRX has fewer experts than Qwen3-MoE but large expert matrices.

Lower priority:

- Aux load-balancing loss and router-logit diagnostics.
- Training dropout/jitter/gradient checkpointing.
- Tensor/expert parallel execution; preserve metadata but single-GPU parity can land first.
- Quantized MLX mirror metadata; route to a separate quantization admission path.

## 11. Runtime staging plan

1. Config admission and normalization:
   - Parse `DbrxConfig`, normalize legacy `attn_config.rope_theta` into `rope_parameters`, verify `d_model == n_heads * head_dim`, `n_heads % kv_n_heads == 0`, and reject `tie_word_embeddings=true`.
   - Add a source-version guard for the pinned expert reshape hazard until validated against a loadable Transformers release.
2. Weight loading:
   - Load embeddings, packed Wqkv, output projection, LayerNorm weights, router weights, flattened `w1/v1/w2` expert tensors, final norm, and LM head.
3. Tiny one-block parity:
   - Implement LayerNorm, packed Wqkv clamp/split, RoPE, eager GQA attention, router/top-k/norm, expert loop, and scatter-add.
4. Full prefill parity:
   - Run all layers for tiny config and shape/load smoke for production mirror config.
5. Decode with cache:
   - Implement KV cache `[B,8,T,128]` for production, default position IDs from cache length, and last-token logits.
6. Optimized attention:
   - Replace eager repeat-KV with native GQA prefill/decode.
7. Optimized MoE:
   - Add sorted-token grouped expert GEMMs, separate or fused gate/up scheduling, and weighted scatter-add.
8. Production scheduling:
   - Add batching, cache allocation policy, expert weight residency/offload planning, and later tensor/expert parallel support.

Initially stub/defer labels/loss, aux loss, attention weights, router logits from optimized paths, and quantized mirror loading.

## 12. Parity and validation plan

Custom op tests:

- Bias-free LayerNorm over `[B,S,6144]` with fp32 reduction and bf16/fp16 storage.
- Packed Wqkv clamp/split for `clip_qkv=None` and `clip_qkv=8`.
- Default RoPE with theta `500000`.
- GQA attention for `QH=48`, `KVH=8`, `D=128`, including no physical KV repeat in optimized path.
- Router softmax/top-k/norm for `E=16,K=4`.
- Flattened expert weight views from `[E*I,H]` to per-expert matrices.
- Scatter-add with multiple selected experts per token and empty experts.

Model tests:

- Single decoder block parity on `Rocketknight1/dbrx-tiny-random` after source-version compatibility is confirmed.
- Full tiny prefill logits parity with `logits_to_keep=0`.
- Decode parity token by token with omitted `position_ids` and `DynamicCache`.
- `logits_to_keep=1` parity against full logits sliced to the final token.
- Config admission tests for legacy RoPE fields, production dimensions, tie-word rejection, and quantized mirror metadata rejection/routing.

Tolerance guidance:

- fp32 isolated ops: `rtol=1e-4`, `atol=1e-5`.
- bf16/fp16 block/logits: start with `rtol=2e-2`, `atol=2e-2`.
- Optimized attention and grouped MoE scatter may require separate tolerances due to softmax and accumulation order; router selected indices should be exact on non-tie test inputs.

## 13. Performance probes

- Prefill tokens/sec by sequence length: 128, 512, 2048, 8192, 32768.
- Decode tokens/sec by batch size and cache length.
- KV cache memory: production per token per layer is `2 * 8 * 128` elements, about 4096 bytes/token/layer in bf16; all 40 layers are about 160 KiB/token/batch element.
- Attention backend comparison: eager repeat-KV, SDPA/FlashAttention-equivalent, DinoML native GQA.
- Wqkv clamp overhead and packed-projection fusion benefit.
- Router/top-k/norm latency for `M` tokens with `E=16,K=4`.
- Expert token distribution and grouped GEMM utilization for prefill versus decode.
- Separate gate/up GEMM fusion versus prepacked gate-up strategy.
- Last-token LM head versus full prompt logits.
- Expert weight residency/offload probes with dense bf16 baseline and future GGUF/dequant-before-GEMM experiments.

Benchmark observations: none collected. These are source/config-derived probes.

## 14. Skip/defer list

Safe to defer for first causal LM integration:

- Training, labels/loss, dropout, jitter, gradient checkpointing, and aux load-balancing loss.
- Returning attentions and router logits from optimized paths.
- Beam search and speculative decoding; standard logits are enough initially.
- Quantized MLX 4-bit loading/kernels, except explicit admission rejection or routing.
- Tensor/expert/pipeline parallel execution.
- Non-default/dynamic RoPE variants until a checkpoint requiring them is targeted.

Do not defer:

- Legacy RoPE config normalization/admission.
- `clip_qkv` parity.
- LayerNorm rather than RMSNorm.
- GQA cache shape with KV heads only.
- RoPE-before-cache-update order.
- Top-4 routing with selected-weight norm.
- DBRX flattened `w1/v1/w2` expert weight layout.

## 15. Final implementation checklist

- [ ] Parse `DbrxConfig` and normalize legacy `attn_config.rope_theta` to current RoPE metadata.
- [ ] Add config admission for `d_model`, heads/KV heads, top-k/experts, `clip_qkv`, and `tie_word_embeddings=false`.
- [ ] Resolve or guard the pinned source expert reshape hazard before numerical parity claims.
- [ ] Load embeddings, packed Wqkv, output projection, LayerNorm weights, router, flattened experts, final norm, and LM head.
- [ ] Implement bias-free LayerNorm for DBRX.
- [ ] Implement packed Wqkv linear, optional clamp, and split order `[Q,K,V]`.
- [ ] Implement default RoPE with theta 500000 and post-RoPE K cache storage.
- [ ] Implement GQA causal attention with cache `[B,KVH,T,D]`.
- [ ] Implement router linear, softmax, top-4, and selected-weight Lp normalization.
- [ ] Implement DBRX expert GLU from flattened `w1/v1/w2` storage.
- [ ] Implement weighted scatter-add for top-4 expert outputs.
- [ ] Implement final LayerNorm, `logits_to_keep`, and LM head.
- [ ] Add one-block and tiny full-model parity tests after source/config compatibility is confirmed.
- [ ] Add prefill/decode cache parity tests.
- [ ] Add performance probes for packed Wqkv, attention, router, expert grouped GEMM, KV memory, and last-token logits.
