# GraniteMoe Transformers Family Audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: granitemoe family; representative primary targets include ibm-research/PowerMoE-3b and ibm-granite Granite 3.0/3.1 MoE checkpoints.
Config source: HF config.json snapshots saved under agents/plans/transformers/granitemoe/_sources/.
Source files inspected:
- transformers/src/transformers/models/granitemoe/configuration_granitemoe.py
- transformers/src/transformers/models/granitemoe/modeling_granitemoe.py
- transformers/src/transformers/models/granitemoe/modular_granitemoe.py
- Contrast only: granite/modeling_granite.py and granitemoehybrid/configuration_granitemoehybrid.py, modeling_granitemoehybrid.py
Any missing files or assumptions: tokenizer files were not needed for the core GPU graph. Inference target is causal LM prefill/decode on CUDA. `modeling_granitemoe.py` is generated; future upstream edits should target `modular_granitemoe.py`, but DinoML parity should follow the generated file at this commit.
```

Saved config snapshots:

- `_sources/ibm-research_PowerMoE-3b_config.json`
- `_sources/ibm-granite_granite-3.0-1b-a400m-base_config.json`
- `_sources/ibm-granite_granite-3.0-3b-a800m-base_config.json`
- `_sources/ibm-granite_granite-3.1-1b-a400m-base_config.json`
- `_sources/ibm-granite_granite-3.1-3b-a800m-base_config.json`
- `_sources/katuni4ka_tiny-random-granite-moe_config.json`

## 2. High-level architecture

GraniteMoe is a text-only decoder-only causal LM with every decoder block containing causal self-attention followed by a sparse MoE SwiGLU-style feed-forward. It is closer to Mixtral/JetMoE routing than dense Granite, but keeps Granite-specific embedding, attention, residual, and logits multipliers.

```text
token ids -> embedding * embedding_multiplier -> N decoder blocks
  -> final RMSNorm -> lm_head(selected hidden states) / logits_scaling
  -> logits -> generation controller
```

Runtime stages:

- CPU/data pipeline: tokenizer, attention mask construction inputs, generation loop policy.
- GPU prefill: embeddings, shared RoPE cos/sin for the prompt positions, causal GQA attention, MoE routing/expert execution, final logits.
- GPU decode: one-token query with per-layer KV cache update; MoE still routes each new token dynamically.
- Independently optimizable regions: attention with KV cache, MoE router plus expert GEMMs, final last-token-only LM head.

Implemented heads:

- Required for target: `GraniteMoeForCausalLM`.
- Useful optional: bare `GraniteMoeModel` for hidden-state parity.
- Deferred: training loss and auxiliary router load-balancing loss.

## 3. Important config dimensions

Source defaults from `GraniteMoeConfig`:

| Field | Source default |
| --- | ---: |
| `vocab_size` | 32000 |
| `hidden_size` | 4096 |
| `intermediate_size` | 11008 |
| `num_hidden_layers` | 32 |
| `num_attention_heads` | 32 |
| `num_key_value_heads` | defaults to `num_attention_heads` if omitted |
| `hidden_act` | `silu` |
| `max_position_embeddings` | 2048 |
| `rms_norm_eps` | 1e-6 |
| `attention_bias` | false |
| `attention_dropout` | 0.0 |
| `embedding_multiplier` | 1.0 |
| `attention_multiplier` | 1.0 |
| `residual_multiplier` | 1.0 |
| `logits_scaling` | 1.0 |
| `num_local_experts` | 8 |
| `num_experts_per_tok` | 2 |
| `tie_word_embeddings` | false |

Representative checkpoint sweep, from saved `config.json` files:

| Checkpoint | H | Layers | Q heads | KV heads | Head dim | Experts | Top-k | Expert hidden | Context | RoPE theta | Multipliers `(emb, attn, resid, logits)` | Dtype | Tied emb |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| `katuni4ka/tiny-random-granite-moe` | 64 | 6 | 4 | 2 | 16 | 2 | 2 | 32 | 4096 | 10000 | `12, 0.015625, 0.22, 6` | float32 | true |
| `ibm-granite/granite-3.0-1b-a400m-base` | 1024 | 24 | 16 | 8 | 64 | 32 | 8 | 512 | 4096 | 10000 | `12, 0.015625, 0.22, 6` | omitted | true |
| `ibm-granite/granite-3.0-3b-a800m-base` | 1536 | 32 | 24 | 8 | 64 | 40 | 8 | 512 | 4096 | 10000 | `12, 0.015625, 0.22, 6` | omitted | true |
| `ibm-research/PowerMoE-3b` | 1536 | 32 | 24 | 8 | 64 | 40 | 8 | 512 | 4096 | 10000 | `12, 0.015625, 0.22, 6` | float32 | true |
| `ibm-granite/granite-3.1-1b-a400m-base` | 1024 | 24 | 16 | 8 | 64 | 32 | 8 | 512 | 131072 | 1500000 | `12, 0.015625, 0.22, 6` | bfloat16 | true |
| `ibm-granite/granite-3.1-3b-a800m-base` | 1536 | 32 | 24 | 8 | 64 | 40 | 8 | 512 | 131072 | 10000000 | `12, 0.015625, 0.22, 6` | bfloat16 | true |

The official production configs use legacy `rope_theta` / `rope_scaling` fields. The current config infrastructure standardizes these into `config.rope_parameters`; the modeling source reads `config.rope_parameters["rope_type"]` and `["rope_theta"]`.

## 3a. Family variation traps

- `attention_multiplier` is the attention score scale. For the production head-dim-64 configs it is `1/64`, not the usual `1/sqrt(64) = 1/8`.
- GQA is required: `num_key_value_heads < num_attention_heads` in representative checkpoints.
- `hidden_size == num_attention_heads * head_dim` in inspected configs, but the source supports explicit `head_dim`; do not infer projection widths from `hidden_size` alone.
- MoE `intermediate_size` is per expert and small in IBM configs (`512`), but `input_linear` projects to `2 * intermediate_size` for gated activation.
- Routing is top-k over raw router logits, then softmax only over the selected top-k logits.
- Router execution is data-dependent: `topk`, scatter, expert-count sum, Python `tolist()`, sort, gather, per-expert split GEMMs, and `index_add`.
- `embedding_multiplier`, `residual_multiplier`, and `logits_scaling` are required checkpoint behavior, not cosmetic metadata.
- `tie_word_embeddings=true` in representative configs, despite the source default being false. Keep `lm_head.weight` and `model.embed_tokens.weight` logically aliased when the checkpoint ties them.
- `activation_function` and `router_jitter_noise` appear in `PowerMoE-3b` but the inspected source uses `hidden_act` and does not read `router_jitter_noise`; treat those as historical/ignored for this source basis.
- `attention_dropout=0.1` appears in Granite 3.1 configs but inference passes `dropout=0.0` unless training.
- No sliding-window attention is used; source comment says GraniteMoe differs from Mixtral by using no sliding mask.
- No NoPE path exists in `granitemoe`; every layer uses RoPE. NoPE/Mamba/hybrid layer manifests belong to `granitemoehybrid`, a separate family.
- Dense `granite` uses dense SwiGLU MLP (`gate_proj`, `up_proj`, `down_proj`) instead of sparse MoE routing/expert weights.

## 4. Operator coverage checklist

Tensor/layout ops:

- Embedding lookup `[B, S] -> [B, S, H]`.
- Reshape/view: `[B, S, H] -> [B*S, H]`, projection output to `[B, S, heads, D]`, transpose to `[B, heads, S, D]`.
- `transpose`, `contiguous`, `cat`, `chunk`, `split`, `expand`, `reshape`, scalar/tensor multiply/divide.
- Indexing/gather by dynamic `batch_index`; sorting indices by selected expert id.
- `scatter` one-hot router mask and `index_add` for expert output accumulation.
- `arange` position ids and cache-length offset.

Neural network primitives:

- RMSNorm over last dim with fp32 variance and output cast back to input dtype.
- Linear projections:
  - `q_proj`: `H -> num_attention_heads * head_dim`.
  - `k_proj`, `v_proj`: `H -> num_key_value_heads * head_dim`.
  - `o_proj`: `num_attention_heads * head_dim -> H`.
  - Router: `H -> num_local_experts`, no bias.
  - Expert input weights: `[num_experts, 2 * intermediate_size, H]`.
  - Expert output weights: `[num_experts, H, intermediate_size]`.
  - LM head: `H -> vocab_size`, no bias.
- SiLU gated expert activation: `silu(first_half) * second_half`.

Attention primitives:

- Causal self-attention only.
- GQA repeat or equivalent grouped attention without materializing expanded KV.
- RoPE on Q and K before cache update.
- Additive causal/padding mask.
- Softmax in fp32, then cast to query dtype in eager path.
- SDPA/Flash/Flex backend dispatch from `ALL_ATTENTION_FUNCTIONS`; eager fallback is matmul-softmax-matmul.

Position/rotary ops:

- Default RoPE from `rope_theta` over full `head_dim`.
- Dynamic RoPE update decorator may change cached inverse frequencies for advanced rope types, but inspected configs use default-style legacy theta fields.

Generation/cache ops:

- DynamicCache per layer.
- Cache update stores post-RoPE key and unrotated value tensors with shape `[B, num_key_value_heads, cached_seq, head_dim]`.
- Position ids for decode start at `past_key_values.get_seq_length()`.
- Optional `logits_to_keep` slice before LM head.

Preprocessing-coupled ops:

- Tokenization and chat templates are outside this report.
- Attention mask is source-compatible `[B, S]` padding mask that feeds `create_causal_mask`.

Distributed/tensor-parallel metadata:

- Source advertises `_tp_plan` for Q/K/V/O projections, embeddings, and LM head. DinoML can defer multi-GPU tensor parallelism for first integration.

## 5. Layer/block breakdown

Decoder block, repeated `num_hidden_layers` times:

```text
residual = x
x_norm = RMSNorm(x)
q = Linear(H -> QH*D)(x_norm).view(B,S,QH,D).transpose(1,2)
k = Linear(H -> KVH*D)(x_norm).view(B,S,KVH,D).transpose(1,2)
v = Linear(H -> KVH*D)(x_norm).view(B,S,KVH,D).transpose(1,2)
q, k = RoPE(q, k, cos, sin)
k_cache, v_cache = cache.update(k, v, layer_idx)
attn = causal_gqa_attention(q, k_cache, v_cache, mask, scale=attention_multiplier)
x = residual + Linear(QH*D -> H)(attn) * residual_multiplier

residual = x
x_norm = RMSNorm(x)
flat = x_norm.reshape(B*S, H)
router_logits = Linear(H -> E, bias=False)(flat).float()
top_logits, top_indices = topk(router_logits, k=top_k, dim=1)
top_gates = softmax(top_logits, dim=1).to(x.dtype)
group tokens by expert
expert_hidden = ExpertLinear[E](H -> 2I)(selected_tokens)
expert_hidden = silu(expert_hidden[..., :I]) * expert_hidden[..., I:]
expert_out = ExpertLinear[E](I -> H)(expert_hidden)
expert_out *= selected_gate
moe = zeros(B*S,H).index_add(0, batch_index, expert_out).view(B,S,H)
x = residual + moe * residual_multiplier
```

Biases:

- Attention projections use `attention_bias` from config; representative configs set false.
- Router has no bias.
- Expert parallel linear layers have no bias.
- LM head has no bias.

## 6. Attention requirements

GraniteMoe requires causal autoregressive self-attention.

| Requirement | Source behavior |
| --- | --- |
| Causal/noncausal | Causal only |
| Cross-attention | None |
| MHA/MQA/GQA | GQA in representative configs |
| Head counts | 1B: Q=16, KV=8, D=64; 3B/PowerMoE: Q=24, KV=8, D=64 |
| KV repeat | `repeat_kv` expands `[B, KVH, S, D]` to `[B, QH, S, D]` in eager path |
| Mask | `create_causal_mask`; no sliding window |
| RoPE | Q/K before cache update |
| Cache | Dynamic per-layer KV cache; no cross-attention cache |
| Backend | FlashAttention, SDPA, Flex, or eager through Transformers attention interface |

Cache ABI for DinoML:

```text
key[layer]:   [batch, num_key_value_heads, cached_seq, head_dim]
value[layer]: [batch, num_key_value_heads, cached_seq, head_dim]
```

The fast attention kernel should consume grouped KV directly. Materializing `repeat_kv` is a correct fallback but wasteful by a factor of `num_attention_heads / num_key_value_heads` (2x for 1B configs, 3x for 3B/PowerMoE).

Source math order:

```text
Q/K/V linear -> reshape/transpose -> RoPE(Q,K) -> cache update
-> attention scores * attention_multiplier -> add mask -> softmax(fp32)
-> cast to query dtype -> dropout(training only) -> AV -> O projection
```

## 7. Position encoding and custom math

Default RoPE inverse frequencies:

```python
def granite_moe_inv_freq(config):
    dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    base = config.rope_parameters["rope_theta"]
    return 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
```

Application:

```python
def apply_granitemoe_rope(q, k, cos, sin):
    cos = cos.unsqueeze(1)  # [B,1,S,D]
    sin = sin.unsqueeze(1)
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)
```

Cos/sin are computed in fp32 under autocast-disabled context and cast to `x.dtype`. For prefill with known positions, cos/sin can be precomputed or cached by `(rope_theta, head_dim, positions, dtype)`. Decode cos/sin depends on current cache length and position ids.

NoPE handling:

- `granitemoe` has no NoPE mode; reject configs that try to use `position_embedding_type=None` under this family.
- `granitemoehybrid` separately supports `position_embedding_type` and Mamba layers. Route those checkpoints to a separate audit.

## 8. Preprocessing and input packing

Core runtime inputs:

- `input_ids`: `[B, S]`, exactly one of `input_ids` or `inputs_embeds`.
- `inputs_embeds`: optional `[B, S, H]`.
- `attention_mask`: optional `[B, S_total]` style padding mask consumed by `create_causal_mask`.
- `position_ids`: optional `[B, S]`; if absent, source uses `arange(S) + past_seen_tokens`, then unsqueezes to `[1, S]`.
- `past_key_values`: optional DynamicCache.

No multimodal placeholder, image/audio/video packing, discrete codebook, or cu-seqlens metadata is part of this family.

Generation-controller behavior:

- `logits_to_keep=0` computes all logits by Python slicing `slice(0, None)`; for decode, pass `1` or an explicit index tensor to avoid full-sequence LM head work.
- End-to-end text parity also depends on tokenizer special-token conventions, but these do not alter the GPU module graph.

## 9. Graph rewrite / lowering opportunities

### Rewrite: last-token-only LM head

Source pattern:

```text
logits = lm_head(hidden_states[:, slice_indices, :]) / logits_scaling
```

Replacement:

```text
Slice hidden states to requested token positions -> GEMM(H, vocab) -> divide by logits_scaling
```

Preconditions:

- `logits_to_keep` is an int `1` during decode, or a known static/gatherable tensor of positions.
- Labels/loss are not requested.

Failure cases:

- Training loss, full-sequence logits, or arbitrary dynamic index tensor without gather support.

Parity test sketch:

- Compare full logits against `logits_to_keep=1` last-token logits sliced from the full result.

### Rewrite: GQA attention without `repeat_kv`

Source pattern:

```text
repeat_kv(K,V) -> attention(Q, repeated K,V)
```

Replacement:

```text
Grouped-query attention kernel using KV head index = q_head // num_key_value_groups
```

Preconditions:

- `num_attention_heads % num_key_value_heads == 0`.
- Same mask and RoPE placement as source.
- Cache stores unexpanded KV.

Failure cases:

- Attention output capture requiring materialized attention weights per Q head can be deferred.

Parity test sketch:

- Random Q/K/V with odd sequence lengths; compare repeated eager path to grouped kernel.

### Rewrite: sparse MoE reference lowering

Source pattern:

```text
router topk -> token regroup -> per-expert linear up/gate -> activation multiply
-> per-expert down -> gate scale -> index_add
```

Replacement:

```text
RouterTopK -> ExpertTokenPack -> grouped GEMM(H -> 2I)
-> SwiGLU -> grouped GEMM(I -> H) -> WeightedScatterAdd
```

Preconditions:

- Top-k is small positive integer.
- Expert weights use source layout `[E, out, in]`.
- Router softmax is over selected top-k logits only.
- Dynamic token counts per expert are represented explicitly.

Failure cases:

- Fullgraph static lowering that assumes fixed expert sizes.
- Any rewrite that softmaxes all experts before top-k changes routing weights.

Parity test sketch:

- Fixed random router/expert weights; compare top indices, gates, expert counts, and final output for small `[B,S]` cases including empty experts.

### Rewrite: expert grouped GEMM fallback

Source pattern:

```text
for expert in range(E): F.linear(input_list[expert], weight[expert])
```

Replacement:

```text
Grouped GEMM over non-empty expert batches, preserving sorted expert order.
```

Preconditions:

- Input packing groups tokens by expert exactly as source sort order.
- Empty expert groups produce no GEMM and preserve offsets.

Failure cases:

- Using dense all-expert GEMM for every token is correct only as an early reference and expensive by `num_local_experts / top_k`.

### Rewrite: Granite multiplier folding

Source pattern:

```text
embed *= embedding_multiplier
residual + branch * residual_multiplier
logits = lm_head(x) / logits_scaling
attention_scores *= attention_multiplier
```

Replacement:

- Fold embedding multiplier into embedding weights at load time if tied LM-head alias is handled separately.
- Fold logits scaling into LM-head weights only when weights are not tied, or model a post-GEMM scalar divide.
- Keep residual multiplier as fused GEMM epilogue scale or elementwise residual epilogue.
- Treat attention multiplier as fused score scale in attention kernel.

Failure cases:

- Tied embeddings make embedding/logit folding easy to get wrong; representative configs tie weights.

## 10. Kernel fusion candidates

Highest priority:

- GQA FlashAttention/SDPA-compatible causal prefill and decode with unexpanded KV cache. This is on every token and avoids `repeat_kv`.
- MoE router/pack/grouped-expert GEMM/scatter pipeline. The source fallback uses Python lists and per-expert loops; production inference needs a fused or provider-backed sparse MoE path.
- RMSNorm. Two per block plus final norm, fp32 variance.
- Last-token LM head. Avoids full prompt logits during decode and most generation paths.

Medium priority:

- QKV projection fusion into one packed GEMM with split order `[Q, K, V]`, if weights are repacked from separate source tensors.
- RoPE fused into attention pre-processing or attention kernel.
- Residual multiplier fused into attention/MLP epilogues.
- SwiGLU expert activation fused between grouped GEMMs.

Lower priority:

- Router aux loss and router logits capture for training/eval diagnostics.
- Tensor-parallel plans.
- Full attention-weight output materialization.
- Dense all-expert fallback optimized beyond correctness reference.

## 11. Runtime staging plan

Stage 1: Config and weight loading.

- Parse legacy `rope_theta` into DinoML RoPE metadata.
- Preserve tied embedding/LM-head aliasing.
- Load expert weights with layout `[E, out, in]`.

Stage 2: Single-block reference parity.

- Implement RMSNorm, Granite multipliers, RoPE, GQA eager attention, and a simple sparse MoE reference.
- Use tiny random and a manually small config first.

Stage 3: Full prefill parity.

- Run all layers for short prompts.
- Keep MoE as explicit token-pack plus per-expert GEMM loops if needed.

Stage 4: Decode with KV cache.

- Implement per-layer cache ABI `[B, KVH, T, D]`.
- Validate position-id offset and post-RoPE key storage.

Stage 5: Optimized attention.

- Replace eager attention with grouped Flash/SDPA provider path.
- Add last-token-only logits.

Stage 6: Optimized MoE.

- Add router top-k, packing, grouped GEMM, and weighted scatter-add kernels.
- Profile expert imbalance and empty-expert cases.

Stage 7: Production scheduling.

- Continuous batching and cache paging.
- Optional quantized/GGUF expert and LM-head storage once dense parity is stable.

## 12. Parity and validation plan

- Config round-trip tests for legacy `rope_theta`, 3.0 vs 3.1 long context, and tied embeddings.
- RMSNorm random tensor parity in fp32/fp16/bf16; tolerance `1e-5` fp32, `2e-2` fp16/bf16 for full blocks.
- RoPE parity for prefill and decode positions, including nonzero `past_seen_tokens`.
- Attention parity against Transformers eager path for GQA heads and masks.
- Router parity: compare logits, selected experts, selected gates, `expert_size`, `batch_index`.
- MoE parity with empty experts and repeated token assignments.
- Single decoder layer parity with cache disabled.
- N-layer prefill hidden-state/logits parity on tiny random config.
- Decode parity: prefill prompt, then one token with cache; compare next-token logits.
- Last-token LM head parity against full logits slice.

## 13. Performance probes

- Prefill tokens/sec by sequence length: 128, 512, 4096, and long-context 131072 admission/profiling shape only if memory allows.
- Decode tokens/sec by batch size and active cache length.
- KV cache memory: `layers * 2 * B * KVH * T * D * dtype_size`.
- Attention backend comparison: eager repeat-KV vs grouped Flash/SDPA.
- Router cost: top-k/sort/pack time versus expert GEMM time.
- Expert imbalance sweep: synthetic routing distributions, including all tokens to one expert and uniform spread.
- Grouped GEMM probe by `(num_experts, top_k, tokens_per_expert, H, I)`.
- LM-head throughput with full sequence versus last-token-only.
- Dense all-expert fallback cost versus sparse top-k path.
- Quantized/expert-weight load and dequant probes only after dense MoE parity.

## 14. Skip/defer list

- Training loss and router auxiliary load-balancing loss.
- Gradient checkpointing.
- Router logits output capture unless needed for diagnostics.
- Multi-GPU tensor parallelism.
- Flash/Flex backend exact replication before eager parity.
- Quantized, GGUF, bitsandbytes, GPTQ, AWQ, or OpenVINO derivative checkpoints.
- `granitemoehybrid` Mamba/NoPE/shared-MLP behavior.
- Chat template/tokenizer parity beyond required token ids and masks.
- Full attention weight output materialization.

## 15. Final implementation checklist

- [ ] Parse `GraniteMoeConfig`, including legacy `rope_theta` / `rope_scaling` standardization.
- [ ] Load tied embeddings and LM head without breaking aliasing.
- [ ] Load expert weights in `[num_experts, out_features, in_features]` layout.
- [ ] Implement RMSNorm with fp32 variance.
- [ ] Implement embedding, attention, residual, and logits multipliers.
- [ ] Implement default RoPE and position-id cache offset handling.
- [ ] Implement GQA causal attention with KV cache `[B, KVH, T, D]`.
- [ ] Add eager reference path that avoids semantic changes from `repeat_kv`.
- [ ] Implement router top-k over logits and softmax over selected logits only.
- [ ] Implement token regrouping, dynamic expert counts, and weighted scatter-add.
- [ ] Implement expert SwiGLU grouped/per-expert GEMMs.
- [ ] Add last-token-only LM-head lowering.
- [ ] Add single-block parity tests.
- [ ] Add full tiny-random prefill parity tests.
- [ ] Add decode-with-cache parity tests.
- [ ] Benchmark attention, MoE routing, grouped expert GEMM, and LM-head slices.
