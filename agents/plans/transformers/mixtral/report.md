# Transformers audit: mixtral

## 1. Source basis

Transformers commit/version: local checkout `transformers` at `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`.

Model id: primary target `mistralai/Mixtral-8x7B-v0.1`; representative sweep also includes `mistralai/Mixtral-8x7B-Instruct-v0.1`, `mistralai/Mixtral-8x22B-v0.1`, `TitanML/tiny-mixtral`, and `optimum-intel-internal-testing/tiny-mixtral`.

Config source:
- `https://huggingface.co/mistralai/Mixtral-8x7B-v0.1/raw/main/config.json`
- `https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1/raw/main/config.json`
- `https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1/raw/main/generation_config.json`
- `https://huggingface.co/mistralai/Mixtral-8x22B-v0.1/raw/main/config.json`
- `https://huggingface.co/mistralai/Mixtral-8x22B-v0.1/raw/main/generation_config.json`
- `https://huggingface.co/TitanML/tiny-mixtral/raw/main/config.json`
- `https://huggingface.co/TitanML/tiny-mixtral/raw/main/generation_config.json`
- `https://huggingface.co/optimum-intel-internal-testing/tiny-mixtral/raw/main/config.json`

Source files inspected:
- `transformers/src/transformers/models/mixtral/configuration_mixtral.py`
- `transformers/src/transformers/models/mixtral/modeling_mixtral.py`
- `transformers/src/transformers/models/mixtral/modular_mixtral.py`
- `transformers/src/transformers/models/mixtral/convert_mixtral_weights_to_hf.py`
- `transformers/src/transformers/models/mistral/modeling_mistral.py`, for inherited attention behavior cross-checks
- `transformers/src/transformers/cache_utils.py`
- `transformers/src/transformers/masking_utils.py`

Source URLs at the inspected commit:
- `https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/mixtral/configuration_mixtral.py`
- `https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/mixtral/modeling_mixtral.py`
- `https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/mixtral/modular_mixtral.py`

Any missing files or assumptions: no processor or multimodal preprocessor is involved. `modeling_mixtral.py` is generated from `modular_mixtral.py`; future Transformers source edits should target the modular file, but this report cites the generated file when it exposes expanded inherited Mistral attention/RoPE code. Primary runtime target is inference-only causal LM prefill/decode/generation on CUDA with MoE MLP.

## 2. High-level architecture

Mixtral is a text-only decoder-only sparse MoE LM:

```text
token ids -> embedding -> N decoder blocks -> final RMSNorm -> lm_head -> logits/sampling
                         | each block: GQA self-attention + top-2 sparse MoE MLP
prefill: full prompt causal attention + KV cache fill
decode: one/few new tokens + cache update + last-token logits
```

Stage decomposition:
- CPU/data pipeline: tokenizer, BOS/EOS/chat template, padding/attention mask construction.
- GPU/runtime prefill: embedding, RoPE, causal or sliding-window GQA attention, MoE routing/expert execution, final norm, logits.
- GPU/runtime decode: position id from cache length, Q/K/V for new tokens, RoPE for new positions, KV cache append or sliding-window roll, attention over cached KV, MoE, last-token logits.
- Generation controller: uses standard `GenerationMixin`; checkpoint `generation_config.json` only supplies BOS/EOS and `_from_model_config`.

The MoE MLP is the defining difference from dense Mistral/Llama-style decoders. Each token is routed to `top_k=2` of `num_local_experts=8`, with per-token router probabilities normalized across the selected experts.

## 3. Important config dimensions

Source defaults from `MixtralConfig`:

| Field | Default / behavior |
| --- | --- |
| `vocab_size` | 32000 |
| `hidden_size` | 4096 |
| `intermediate_size` | 14336 |
| `num_hidden_layers` | 32 |
| `num_attention_heads` | 32 |
| `num_key_value_heads` | 8; if `None`, post-init sets it to `num_attention_heads` |
| `head_dim` | optional; attention uses explicit `head_dim` or `hidden_size // num_attention_heads` |
| `hidden_act` | `silu` |
| `max_position_embeddings` | `4096 * 32` by default, but checkpoint configs vary |
| `rope_theta` / `default_theta` | default theta is 1,000,000.0 in current config class |
| `rms_norm_eps` | `1e-5` |
| `use_cache` | `True` |
| `sliding_window` | `None` by default |
| `num_experts_per_tok` | 2 |
| `num_local_experts` | 8 |
| `output_router_logits` | `False` |
| `router_aux_loss_coef` | `0.001` source default; 8x7B configs use `0.02` |
| `router_jitter_noise` | `0.0`; training-only perturbation |
| `attention_dropout` | `0.0` |
| `tie_word_embeddings` | `False` |

Representative checkpoint sweep:

| Model id | Layers | Hidden | Heads / KV heads | Head dim | Intermediate | Experts / top-k | Max positions | Sliding window | Dtype | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `mistralai/Mixtral-8x7B-v0.1` | 32 | 4096 | 32 / 8 | 128 | 14336 | 8 / 2 | 32768 | `null` | bf16 | base production config; aux coef 0.02 |
| `mistralai/Mixtral-8x7B-Instruct-v0.1` | 32 | 4096 | 32 / 8 | 128 | 14336 | 8 / 2 | 32768 | `null` | bf16 | same core graph as base; generation config only BOS/EOS |
| `mistralai/Mixtral-8x22B-v0.1` | 56 | 6144 | 48 / 8 | 128 | 16384 | 8 / 2 | 65536 | `null` | bf16 | wider/deeper; GQA group size 6 |
| `TitanML/tiny-mixtral` | 2 | 1024 | 32 / 8 | 32 | 3584 | 8 / 2 | 131072 | 4096 | fp32 | tiny random/debug; exercises sliding-window path |
| `optimum-intel-internal-testing/tiny-mixtral` | 2 | 1024 | 32 / 8 | 32 | 3584 | 8 / 2 | 131072 | 4096 | fp32 | similar debug config |

Generation config sweep:

| Model id | Generation fields observed |
| --- | --- |
| `Mixtral-8x7B-Instruct-v0.1` | `_from_model_config: true`, `bos_token_id: 1`, `eos_token_id: 2` |
| `Mixtral-8x22B-v0.1` | `_from_model_config: true`, `bos_token_id: 1`, `eos_token_id: 2` |
| `TitanML/tiny-mixtral` | `_from_model_config: true`, `bos_token_id: 1`, `eos_token_id: 2` |

No checkpoint inspected changes expert count or top-k. The operator-significant variation is scale, GQA group count, context length, dtype, and the presence of sliding-window attention in tiny/debug checkpoints.

## 3a. Family variation traps

- `num_key_value_heads < num_attention_heads` is normal. 8x7B has 4 query heads per KV head; 8x22B has 6 query heads per KV head. Cache storage must use KV heads, not expanded query heads.
- `head_dim` can be explicit. Do not assume `hidden_size == num_heads * head_dim` without checking the config. Current inspected configs do match.
- Attention projections are bias-free. `q_proj: hidden -> num_attention_heads * head_dim`; `k_proj` and `v_proj: hidden -> num_key_value_heads * head_dim`; `o_proj: num_attention_heads * head_dim -> hidden`.
- MoE expert weights are packed in current source as `gate_up_proj` shaped `[num_experts, 2 * intermediate, hidden]` and `down_proj` shaped `[num_experts, hidden, intermediate]`. Older conversion script naming (`w1`, `w2`, `w3`, `block_sparse_moe`) may not match current module names.
- `sliding_window` is `null` for official 8x7B/8x22B configs but present in tiny/debug configs. DinoML should implement or explicitly reject this branch; treating all Mixtral as full attention would fail debug configs and any downstream sliding-window variants.
- `rope_theta` is represented in current code through `config.rope_parameters["rope_theta"]`; raw checkpoint configs still contain top-level `rope_theta`.
- RoPE type can be non-default through `rope_parameters["rope_type"]`; current representative configs are default theta-only.
- `output_router_logits` is optional and mostly training/diagnostics. Inference logits do not require aux loss, but parity tests may request router logits.
- Layout translation is low value for text-only Mixtral. The graph is axis-sensitive around token sequence dim, head dim transposes, softmax dim `-1`, top-k dim `-1`, and expert scatter/gather. Protect attention and MoE routing with a no-layout-translation guard until there is a proven local rewrite.

## 4. Operator coverage checklist

Tensor/layout ops:
- Integer token embedding lookup `[B, S] -> [B, S, H]`.
- `reshape`, `view`, `transpose`, `contiguous`, `unsqueeze`, `expand`, `cat`, `chunk`, slicing for last-token logits.
- `where`, `nonzero`, `one_hot`, `permute`, `index_select`/advanced gather, `index_add_` for eager MoE routing.
- Residual adds and dtype casts.

Neural network primitives:
- RMSNorm over last dimension with fp32 variance and output cast back to input dtype.
- Bias-free linear/GEMM:
  - 8x7B: Q `4096 -> 4096`, K/V `4096 -> 1024`, O `4096 -> 4096`.
  - 8x22B: Q `6144 -> 6144`, K/V `6144 -> 1024`, O `6144 -> 6144`.
  - Router 8x7B `4096 -> 8`, router 8x22B `6144 -> 8`.
  - Per expert 8x7B packed gate/up `4096 -> 28672`, down `14336 -> 4096`.
  - Per expert 8x22B packed gate/up `6144 -> 32768`, down `16384 -> 6144`.
  - LM head 8x7B `4096 -> 32000`, 8x22B `6144 -> 32000`.
- SiLU and multiply for SwiGLU-style expert MLP: `silu(gate) * up`.

Attention primitives:
- Causal self-attention with RoPE.
- GQA repeat or backend-native GQA: cache `[B, KVH, T, D]`, logical attention `[B, QH, Q, K]`.
- Softmax with fp32 accumulation in eager path, output cast to query dtype.
- Optional dropout is training-only; inference uses `0.0`.
- Full causal mask and optional sliding-window causal mask.
- FlashAttention, SDPA, and flex attention backend dispatch compatibility through `ALL_ATTENTION_FUNCTIONS`.

Position/rotary ops:
- Default RoPE inverse frequency, cos/sin generation in fp32 autocast-disabled region.
- `rotate_half`, apply RoPE to Q and K before cache update.

Generation/cache ops:
- Dynamic cache allocation when `use_cache` and no cache is passed.
- Cache `get_seq_length()` to derive default `position_ids`.
- KV cache update per layer after RoPE.
- Optional dynamic/static sliding-window cache semantics.
- `logits_to_keep`: compute only last N or indexed logits to avoid full prompt vocab GEMM during generation.

MoE/scatter/indexed ops:
- Router GEMM, fp32 softmax over experts, top-k, selected-probability renormalization.
- Token grouping by selected expert.
- Per-expert packed gate/up GEMM, activation/multiply, down GEMM.
- Multiply by selected router weight.
- Accumulate multiple expert contributions back to token order with `index_add_`.

Distributed/tensor-parallel ops:
- Config declares TP plans: attention Q/K/V colwise, O rowwise, `gate_up_proj` packed colwise, `down_proj` rowwise, experts as `moe_tp_experts`, and LM head colwise with gather output. First DinoML integration can defer multi-GPU, but weight-layout metadata should not paint it into a corner.

## 5. Layer/block breakdown

Decoder block, repeated `num_hidden_layers` times:

```text
residual = x
x = RMSNorm(x)
q = Linear(H -> QH * D, bias=False)(x).view(B, S, QH, D).transpose(1, 2)
k = Linear(H -> KVH * D, bias=False)(x).view(B, S, KVH, D).transpose(1, 2)
v = Linear(H -> KVH * D, bias=False)(x).view(B, S, KVH, D).transpose(1, 2)
q, k = RoPE(q, k, cos, sin)
k, v = cache.update(k, v, layer_idx) if cache exists
attn = causal_or_sliding_GQA_attention(q, k, v, mask, scale=D**-0.5)
x = residual + Linear(QH * D -> H, bias=False)(attn)

residual = x
x = RMSNorm(x)
flat = x.view(B * S, H)
router_logits = Linear(H -> E, bias=False)(flat)
router_probs = softmax(router_logits.float(), dim=-1)
top_values, top_indices = topk(router_probs, K=top_k, dim=-1)
top_values = top_values / sum(top_values, dim=-1, keepdim=True)
for expert e with assigned tokens:
    gate, up = Linear(H -> 2I, bias=False, expert=e)(tokens).chunk(2, dim=-1)
    y_e = Linear(I -> H, bias=False, expert=e)(silu(gate) * up)
    scatter_add token contribution y_e * route_weight
x = residual + moe_output.reshape(B, S, H)
```

Model tail:

```text
x = final RMSNorm(x)
logits = lm_head(x[:, slice_indices, :])
```

For generation, `slice_indices` defaults to all logits when `logits_to_keep=0`; optimized generation should set `logits_to_keep=1` or pass explicit token indices for prompt-logit pruning.

## 6. Attention requirements

Variant: causal self-attention with GQA, RoPE, optional sliding-window masking, and KV cache.

Shapes:
- Input hidden: `[B, S, H]`.
- Query after projection/reshape: `[B, QH, S, D]`.
- Key/value after projection/reshape: `[B, KVH, S, D]`.
- Full cache per layer stores RoPE-applied K and raw V as `[B, KVH, T, D]`.
- Eager fallback repeats KV to `[B, QH, T, D]` using `repeat_kv`; optimized attention should avoid materializing this expansion.
- Attention output before O projection: `[B, S, QH * D]`.

Representative cache sizes per layer:
- 8x7B: K and V each `[B, 8, T, 128]`; query heads are 32. KV elements per token per layer are `2 * 8 * 128 = 2048`.
- 8x22B: K and V each `[B, 8, T, 128]`; query heads are 48. KV elements per token per layer remain 2048, but there are 56 layers.
- Tiny: K and V each `[B, 8, min(T, 4096), 32]` for sliding-window cache.

Math order in eager fallback:
1. Project Q/K/V.
2. Apply RoPE to Q/K.
3. Update cache with RoPE-applied K and V.
4. Repeat KV to query-head count.
5. `attn_weights = matmul(q, k.transpose(-2, -1)) * head_dim**-0.5`.
6. Add mask if present.
7. Softmax over keys with `dtype=torch.float32`, cast to query dtype.
8. Dropout if training.
9. `matmul(attn_weights, v)`, transpose to `[B, S, heads, D]`, contiguous, reshape, O projection.

Masking:
- If `config.sliding_window is None`, `MixtralModel` uses `create_causal_mask`.
- If `config.sliding_window` is set, it uses `create_sliding_window_causal_mask`, passes `local_size=sliding_window` to SDPA-style mask interfaces, and combines sliding-window overlay with causality.
- Dynamic sliding-window cache stores only the last `sliding_window - 1` cached tokens but returns full states for the current update; static sliding-window cache bounds backing storage to `min(max_cache_len, sliding_window)`.

Backend compatibility:
- Source declares support for flash attention, SDPA, flex attention, and attention backend dispatch. Dinoml should map Mixtral to a GQA-capable fused attention provider for prefill and decode. Eager repeat-KV matmul is a correctness fallback only; it is too slow and memory-heavy for production.

## 7. Position encoding and custom math

Default RoPE:

```python
dim = config.head_dim or config.hidden_size // config.num_attention_heads
base = config.rope_parameters["rope_theta"]
inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
freqs = (inv_freq[None, :, None] @ position_ids[:, None, :].float()).transpose(1, 2)
emb = torch.cat((freqs, freqs), dim=-1)
cos = emb.cos() * attention_scaling
sin = emb.sin() * attention_scaling
```

Apply RoPE:

```python
def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_mixtral_rope(q, k, cos, sin):
    cos = cos.unsqueeze(1)  # [B, 1, S, D]
    sin = sin.unsqueeze(1)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)
```

Precompute candidates:
- `inv_freq` is static per config and dtype/device.
- Cos/sin depend on runtime `position_ids`, batch, sequence length, and dynamic RoPE variants if configured.
- Decode can compute cos/sin for one new position per batch item. Prefill should generate `[B, S, D]` once and share across all layers, as the source does.

## 8. Preprocessing and input packing

Text-only preprocessing:
- Tokenizer emits `input_ids` and optional `attention_mask`.
- BOS token id is 1, EOS token id is 2. `pad_token_id` is `None` in source defaults and representative configs.
- Generation configs inspected do not force sampling parameters, forced IDs, suppressed tokens, or custom processors.
- If `position_ids` is omitted, the model creates `arange(S) + past_seen_tokens` and unsqueezes to `[1, S]`. Batched left/right padding must be represented through `attention_mask` and position handling consistent with Transformers generation.

GPU/runtime inputs:
- Required: `input_ids` or `inputs_embeds`, exactly one.
- Optional: `attention_mask`, `position_ids`, `past_key_values`, `use_cache`, `logits_to_keep`.
- No image/audio tensors, placeholder tokens, packed vision grids, or `cu_seqlens`-style metadata are model-coupled for Mixtral.

Generation-controller behavior:
- Standard `GenerationMixin` is sufficient for first parity.
- Chat template/instruction formatting is tokenizer-side and outside the core module graph.
- `logits_to_keep` is important for efficient generation and can be exposed as a runtime slicing/control input or compile-time generation specialization.

## 9. Graph rewrite / lowering opportunities

### Rewrite: QKV projections -> grouped projection region

Source pattern:

```text
q_proj(x), k_proj(x), v_proj(x) with shared input x
```

Replacement pattern:

```text
GroupedGEMM or fused multi-output GEMM -> reshape/transposes
```

Preconditions:
- All three projections are bias-free and consume the same normalized hidden states.
- Weight dtype/layout is compatible with a grouped or packed projection provider.
- Output split sizes are `[QH * D, KVH * D, KVH * D]`, not equal-sized for GQA in general.

Shape equations:
- `Q = H -> QH * D`.
- `K,V = H -> KVH * D`.

Weight transform:
- Concatenate rows `[Wq; Wk; Wv]` only if the provider supports unequal output slices and returning separate views, or use a grouped GEMM batch.

Failure cases:
- Explicit `head_dim` causing unexpected output sizes.
- Runtime requiring separate tensor-parallel sharding plans.

Parity test sketch:
- Random `[B,S,H]`, compare split outputs to three source linear layers before RoPE.

### Rewrite: RoPE + GQA attention + cache update

Source pattern:

```text
linear Q/K/V -> reshape/transpose -> RoPE(Q,K) -> cache.update -> attention backend
```

Replacement pattern:

```text
FusedGQAAttentionPrefillDecode(q, k, v, cos, sin, cache, mask_policy)
```

Preconditions:
- Causal self-attention.
- RoPE is default or a supported rope type.
- KV cache stores RoPE-applied K and raw V, matching source order.
- Sliding-window flag must be represented in mask/cache policy.

Shape equations:
- Cache write shape `[B, KVH, Snew, D]`.
- Logical attention uses `QH / KVH` groups.

Failure cases:
- Non-default RoPE variants not implemented.
- Packed/block masks or external `and_mask_function`/`or_mask_function` overlays.

Parity test sketch:
- Single layer prefill and decode with identical cache growth; compare attention output before O projection.

### Rewrite: eager MoE routing loop -> sorted token expert batches

Source pattern:

```text
one_hot(top_k_index).permute(...)
for expert in active_experts:
    token_idx = where(mask[expert])
    expert_output = expert(tokens)
    final.index_add_(0, token_idx, expert_output * route_weight)
```

Replacement pattern:

```text
TopKRouter -> token/expert assignment sort or histogram -> grouped expert GEMM -> weighted scatter-add
```

Preconditions:
- `num_experts_per_tok=2` and `num_local_experts=8` or provider declares support for configured values.
- Inference mode, no router jitter.
- Router softmax/top-k normalization exactly preserved.

Shape equations:
- Flat tokens `M = B * S`.
- Router logits `[M, E]`; selected experts `[M, K]`.
- Per-expert token count `M_e` varies dynamically; total routed rows `M * K`.
- Gate/up GEMM per expert: `[M_e, H] x [H, 2I] -> [M_e, 2I]`.
- Down GEMM per expert: `[M_e, I] x [I, H] -> [M_e, H]`.

Weight transform:
- Source stores `gate_up_proj[e]` as `[2I, H]`, matching `F.linear` row-major weight. DinoML GEMM RCR can consume RHS transposed/logical column-major or prepack per provider.
- `down_proj[e]` is `[H, I]` for `F.linear(hidden, down_proj[e])`.

Failure cases:
- Training mode with jitter noise.
- Requesting router logits/aux loss can require preserving extra outputs.
- Determinism differences in top-k ties or scatter-add accumulation order.

Parity test sketch:
- Compare router logits, top-k indices, top-k weights, per-expert outputs, and final scattered result for fixed random inputs. Include tie-sensitive tests with separated logits.

### Rewrite: last-token-only LM head

Source pattern:

```text
logits = lm_head(hidden_states[:, slice_indices, :])
```

Replacement pattern:

```text
if decode or logits_to_keep=1: GatherLastToken -> GEMM(H -> vocab)
```

Preconditions:
- Caller does not need full prompt logits.
- `logits_to_keep` is int 1 or a known tensor index set.

Shape equations:
- Full prefill logits `[B,S,V]`; decode logits `[B,1,V]`.

Failure cases:
- Training/loss computation or prompt-logprobs requiring all positions.

Parity test sketch:
- Compare `logits_to_keep=0`, `1`, and tensor-index paths against Transformers.

## 10. Kernel fusion candidates

Highest priority:
- RMSNorm. It appears twice per block plus final norm; fp32 accumulation and dtype restore are required.
- GQA FlashAttention with KV cache. Avoid eager KV expansion; support `[B, KVH, T, D]` cache directly and both prefill/decode.
- RoPE + cache write. Source applies RoPE before cache update; fusing Q/K rotation with cache write reduces memory traffic.
- MoE routing + expert GEMM scheduler. The Python-style one-hot/where/index-add loop is the dominant non-attention obstacle.
- Packed gate/up expert GEMM + SiLU/multiply. Source already packs gate/up into one expert weight; preserve this and fuse activation.

Medium priority:
- QKV grouped projection. Saves launch overhead but output sizes are asymmetric under GQA.
- Weighted scatter-add for top-2 experts. Critical once grouped expert GEMMs exist.
- Last-token LM head. Reduces vocab GEMM cost during decode.
- Router softmax/top-k over 8 experts. Small but called once per layer; can be specialized.
- Sliding-window attention/cache. Needed for tiny/debug and possible long-context variants; official 8x7B/8x22B do not require it.

Lower priority:
- Aux load-balancing loss. Training/diagnostic path; not needed for inference-first.
- Sequence/token classification and QA heads. Implemented via generic heads but deferred for causal LM target.
- Multi-GPU expert/tensor parallel. Config exposes a plan, but single-GPU graph parity can land first.

## 11. Runtime staging plan

Stage 1: config and weights
- Parse `MixtralConfig`, including `rope_theta`/`rope_parameters`, `head_dim`, GQA, MoE counts, and sliding-window policy.
- Load current packed expert weights (`mlp.experts.gate_up_proj`, `mlp.experts.down_proj`) and router weights.
- Stub classification/QA heads.

Stage 2: one-block eager parity
- Implement embedding, RMSNorm, Q/K/V/O linears, RoPE, eager GQA attention, router softmax/top-k, per-expert loop, and final scatter-add.
- Validate one decoder block without cache.

Stage 3: full prefill parity
- Run all layers for a short prompt with full causal attention.
- Enable final norm and LM head.
- Add `logits_to_keep` handling.

Stage 4: decode with cache
- Implement dynamic full KV cache `[B, KVH, T, D]` per layer.
- Match position id generation from cache length.
- Validate token-by-token logits against Transformers.

Stage 5: optimized attention
- Replace eager attention with GQA fused attention for prefill and decode.
- Keep source math order: scale after QK matmul, mask add before fp32 softmax, no dropout in inference.

Stage 6: optimized MoE
- Add top-2 routing provider, expert token grouping, grouped GEMM, and weighted scatter-add.
- Add provider-visible expert GEMM manifests and profile reports. This likely wants grouped GEMM support beyond current dense GEMM surface.

Stage 7: production scheduling
- Add continuous batching, paged/cache allocation policy, optional sliding-window cache, and later expert/tensor parallel support.

## 12. Parity and validation plan

Custom op tests:
- RMSNorm fp32/fp16/bf16 versus PyTorch over `[B,S,H]`.
- RoPE cos/sin and apply function for prefill and decode positions.
- GQA repeat-free attention compared to eager repeat-KV attention.
- Router: softmax float32, top-k indices/values, renormalization.
- MoE scatter-add for top-2 with empty and non-empty experts.

Single-layer parity:
- Random tiny config layer, no cache, compare after attention, after MoE, and block output.
- Repeat with `sliding_window=4096` tiny config at short sequence to exercise mask branch.

Full model parity:
- Tiny random checkpoint prefill logits for short prompts.
- 8x7B shape-only/load smoke if full weights are unavailable locally; exact logits if weights are available.
- Compare `logits_to_keep=1` against full logits sliced to last token.

Decode parity:
- Prefill N tokens, decode 1 token, compare cache length and logits.
- Multi-step decode with cache reuse and position ids omitted.
- Sliding-window tiny decode past the window boundary.

Recommended tolerances:
- fp32 tiny: `rtol=1e-4`, `atol=1e-5`.
- bf16/fp16 full model: start with `rtol=2e-2`, `atol=2e-2` for end-to-end logits; use tighter per-op tolerances where accumulation order is identical.
- MoE optimized scatter may need slightly looser tolerance due to accumulation order; keep router indices exact.

## 13. Performance probes

- Prefill tokens/sec by sequence length: 128, 512, 2048, 8192, 32768 for 8x7B; include 65536 for 8x22B if memory permits.
- Decode tokens/sec by batch size and cache length.
- KV cache memory per batch/layer and total memory; verify GQA cache uses KV heads only.
- Attention backend comparison: eager repeat-KV, SDPA/flash-equivalent, DinoML fused GQA.
- MoE router overhead: router/top-k/scatter time independent of expert GEMM.
- Expert GEMM utilization: distribution of tokens per expert and grouped GEMM efficiency for batch/sequence sweeps.
- Last-token LM head versus full prompt logits.
- Sliding-window cache memory and speed for tiny/debug config.
- Weight-loading/offload probes for expert weights: dense bf16 baseline, GGUF/dequant-before-GEMM exploratory path, and future expert residency scheduling.

Benchmark observations: none collected in this docs-only audit. All probes above are source/config-derived recommendations.

## 14. Skip/defer list

Safe to defer for first causal LM integration:
- Training, gradient checkpointing, router jitter noise, and aux load-balancing loss.
- Sequence classification, token classification, and question answering heads.
- Beam search and speculative decoding; greedy/sampling through standard logits is enough initially.
- Multi-GPU tensor/expert parallel execution, though config TP plans should be preserved as metadata.
- Quantized weight formats beyond existing DinoML GGUF planning experiments.
- Non-default/dynamic RoPE variants until a checkpoint requiring them is targeted.
- Sliding-window optimized kernel for official 8x7B/8x22B parity, but not for tiny/debug config parity.

Do not defer:
- GQA cache shape correctness.
- RoPE-before-cache-update order.
- Top-2 MoE routing, selected-probability renormalization, and weighted scatter-add.
- Last-token logits path for practical generation performance.

## 15. Final implementation checklist

- [ ] Parse `MixtralConfig`, including GQA, optional `head_dim`, MoE fields, RoPE theta, and sliding-window policy.
- [ ] Load token embedding, attention weights, RMSNorm weights, packed `gate_up_proj`, `down_proj`, router weight, final norm, and LM head.
- [ ] Implement Mixtral RMSNorm with fp32 variance.
- [ ] Implement default RoPE generation and apply function.
- [ ] Implement bias-free Q/K/V/O projection lowering for asymmetric GQA outputs.
- [ ] Implement GQA causal attention with KV cache stored as `[B, KVH, T, D]`.
- [ ] Implement optional sliding-window mask/cache branch or explicit unsupported-config rejection.
- [ ] Implement router linear, fp32 softmax, top-k, and top-k renormalization.
- [ ] Implement MoE expert execution with packed gate/up GEMM, SiLU multiply, down GEMM, route weighting, and scatter-add.
- [ ] Add graph rewrite for grouped QKV projection under guarded preconditions.
- [ ] Add graph rewrite/provider plan for sorted-token grouped expert GEMM.
- [ ] Add last-token-only LM head lowering for `logits_to_keep=1`.
- [ ] Add one-block parity tests for attention and MoE.
- [ ] Add tiny checkpoint prefill and decode parity tests.
- [ ] Add cache shape and RoPE-before-cache-update regression tests.
- [ ] Add performance probes for prefill, decode, MoE routing, expert GEMMs, and KV memory.
