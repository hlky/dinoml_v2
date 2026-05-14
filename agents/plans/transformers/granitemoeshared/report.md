# GraniteMoeShared Transformers Family Audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: granitemoeshared source family. No public production config with model_type=granitemoeshared was found.
Config source: source defaults, upstream test fixture, and mismatch snapshots saved under _sources/.
Source files inspected:
- X:/H/transformers/src/transformers/models/granitemoeshared/configuration_granitemoeshared.py
- X:/H/transformers/src/transformers/models/granitemoeshared/modeling_granitemoeshared.py
- X:/H/transformers/src/transformers/models/granitemoeshared/modular_granitemoeshared.py
- X:/H/transformers/tests/models/granitemoeshared/test_modeling_granitemoeshared.py
Any missing files or assumptions: tokenizer files are not needed for the core GPU graph. `modeling_granitemoeshared.py` is generated from `modular_granitemoeshared.py`; parity should follow the generated file, while future upstream edits should target the modular file. Public PowerMoE configs are `granitemoe`, not `granitemoeshared`; public speech configs wrap dense `granite`, not this class.
```

Saved config/context snapshots:

- `_sources/ibm__PowerMoE-3b.config.json`
- `_sources/ibm-research__PowerMoE-3b.config.json`
- `_sources/ibm-granite__granite-speech-3.2-8b.config.json`
- `_sources/ibm-granite__granite-speech-3.3-2b.config.json`
- `_sources/ibm-granite__granite-speech-3.3-8b.config.json`
- `_sources/ibm-granite__granite-speech-4.1-2b-plus.config.json`
- `_sources/source_notes.md`

## 2. High-level architecture

GraniteMoeShared is a text-only decoder-only causal LM. It is the GraniteMoe sparse MoE decoder plus an optional shared dense SwiGLU MLP branch inside every decoder layer. The shared branch is enabled only when `shared_intermediate_size > 0`; source defaults leave it disabled, while the upstream tiny test fixture sets it nonzero.

```text
token ids -> embedding * embedding_multiplier -> N decoder blocks
  -> final RMSNorm -> lm_head(selected hidden states) / logits_scaling
  -> logits -> generation controller
```

Per block:

```text
RMSNorm -> causal GQA self-attention with RoPE/KV cache -> residual add
RMSNorm -> sparse top-k MoE SwiGLU experts
        + optional shared dense SwiGLU MLP
        -> residual add
```

Runtime stages:

- CPU/data pipeline: tokenizer, attention mask inputs, generation controller.
- GPU prefill: embeddings, shared RoPE cos/sin, causal attention, sparse MoE routing/expert execution, optional shared MLP, final logits.
- GPU decode: one-token query with per-layer KV cache update; MoE routing remains dynamic per token.
- Independently optimizable regions: attention/cache, sparse MoE router plus grouped expert GEMMs, shared dense MLP, last-token-only LM head.

Implemented heads:

- Required target: `GraniteMoeSharedForCausalLM`.
- Optional: bare `GraniteMoeSharedModel` hidden states.
- Deferred: training loss and router auxiliary load-balancing loss.

## 3. Important config dimensions

Source defaults from `GraniteMoeSharedConfig`:

| Field | Source default |
| --- | ---: |
| `vocab_size` | 32000 |
| `hidden_size` | 4096 |
| `intermediate_size` | 11008 |
| `shared_intermediate_size` | 0 |
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
| `use_cache` | true |

Representative available config/context sweep:

| Source | Scope | H | Layers | Q heads | KV heads | Expert hidden | Shared hidden | Context | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Source default | `granitemoeshared` default | 4096 | 32 | 32 | 32 | 11008 | 0 | 2048 | Shared branch disabled by default; MHA unless KV heads set. |
| Upstream test fixture | synthetic `GraniteMoeSharedConfig` | 32 | 2 | 4 | 4 by post-init | 37 | 174 | 512 | Exercises shared MLP branch with tiny dimensions. |
| `ibm/PowerMoE-3b` | public mismatch context | 1536 | 32 | 24 | 8 | 512 | absent | 4096 | `model_type=granitemoe`, no shared branch field. |
| `ibm-research/PowerMoE-3b` | public mismatch context | 1536 | 32 | 24 | 8 | 512 | absent | 4096 | Same public config structure as `ibm/PowerMoE-3b`. |
| Granite speech 3.2/3.3/4.1 | public mismatch context | varies | varies | varies | varies | dense MLP | absent | varies | Composite speech configs use dense `granite` `text_config`, not this source family. |

The lack of a public `granitemoeshared` production config means DinoML should treat source defaults and explicit user-provided configs as the admission basis until an official checkpoint with `model_type=granitemoeshared` is available.

## 3a. Family variation traps

- `shared_intermediate_size == 0` disables the shared dense MLP entirely. Nonzero values add extra dense SwiGLU work in parallel with sparse MoE.
- The shared MLP output is added to the sparse MoE output before the second residual multiplier. Fusing or quantizing MoE without including the shared branch changes layer math.
- Router softmax is over selected top-k logits only, not over all experts before top-k.
- Router execution is data-dependent: `topk`, one-hot scatter, expert-count sum, Python `tolist()`, sort by expert id, gather, per-expert split GEMMs, gate scaling, and `index_add`.
- Expert weights are stored as `[num_experts, out_features, in_features]`, compatible with grouped/fused MoE providers but not ordinary stacked dense GEMM without layout awareness.
- `attention_multiplier` is the score scale; do not silently replace it with `1/sqrt(head_dim)`.
- `head_dim` may be explicit. Projection widths are `num_attention_heads * head_dim` and `num_key_value_heads * head_dim`, not inferred from `hidden_size` alone.
- `num_key_value_heads` defaults to `num_attention_heads`, but public GraniteMoe context uses GQA with fewer KV heads.
- `embedding_multiplier`, `residual_multiplier`, and `logits_scaling` are required math.
- `_tied_weights_keys` declares `lm_head.weight` tied to `model.embed_tokens.weight`; actual aliasing depends on config/checkpoint tying and must be preserved when present.
- Current source has no sliding-window attention; the causal mask call comments "NO SLIDING".
- `GraniteFlashAttentionKwargs` includes packed/varlen metadata (`cu_seq_lens_*`, `max_length_*`, `seq_idx`) passed through to attention backends. First DinoML integration can reject padding-free packed metadata unless an attention provider consumes it explicitly.
- No source-coupled quantization is implemented in this modeling file. GGUF/bitsandbytes public forks of PowerMoE are outside native source behavior and should be treated as separate loading/provider contracts.

## 4. Operator coverage checklist

Tensor/layout ops:

- Embedding lookup `[B, S] -> [B, S, H]`.
- Reshape/view `[B, S, H] -> [B*S, H]` for MoE routing.
- Projection reshape to `[B, S, heads, D]`, transpose to `[B, heads, S, D]`, final attention transpose/contiguous/reshape.
- `chunk(..., dim=-1)` for sparse expert and shared MLP SwiGLU halves.
- `split(expert_size, dim=0)`, `cat(dim=0)`, `expand`, `reshape`, scalar multiply/divide.
- Dynamic gather by `batch_index`.
- `topk`, `sort`, integer division with truncation, flatten.
- One-hot/scatter for router counts and `index_add(0, batch_index, expert_outputs)` for weighted accumulation.
- `arange` position ids plus cache-length offset.

Neural primitives:

- RMSNorm over last dim with fp32 variance and cast back to input dtype.
- Attention linears:
  - `q_proj`: `H -> num_attention_heads * head_dim`.
  - `k_proj`, `v_proj`: `H -> num_key_value_heads * head_dim`.
  - `o_proj`: `num_attention_heads * head_dim -> H`.
- Router linear: `H -> num_local_experts`, no bias.
- Sparse expert input linear: per expert `H -> 2 * intermediate_size`, no bias.
- Sparse expert output linear: per expert `intermediate_size -> H`, no bias.
- Optional shared MLP input linear: `H -> 2 * shared_intermediate_size`, no bias.
- Optional shared MLP output linear: `shared_intermediate_size -> H`, no bias.
- LM head: `H -> vocab_size`, no bias.
- SiLU-gated activation: `silu(first_half) * second_half`.

Attention primitives:

- Causal self-attention only.
- MHA/GQA depending on `num_key_value_heads`.
- RoPE on Q/K before cache update.
- Additive causal/padding mask.
- Eager fallback softmax computes in fp32 and casts to query dtype.
- Flash/SDPA/Flex backend dispatch via Transformers attention interface.

Position/rotary ops:

- Default RoPE from `config.rope_parameters["rope_theta"]` over full `head_dim`.
- `dynamic_rope_update` wrapper can support advanced RoPE types if config standardization selects non-default rope.

Generation/cache ops:

- DynamicCache per layer.
- Cache update stores post-RoPE key and value tensors shaped `[B, num_key_value_heads, cached_seq, head_dim]`.
- Optional `logits_to_keep` slice before LM head.

Quantized/packed weight metadata:

- None in native source. Weight quantization should be handled as DinoML encoded-constant/provider policy, not as a `granitemoeshared` op requirement.

Distributed/tensor-parallel:

- Source only advertises `_tp_plan = {"lm_head": "colwise_gather_output"}` in the generated shared class. Multi-GPU can be deferred.

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
batch_index = sorted_topk_assignment_to_token_ids(top_indices)
expert_in = flat[batch_index]
expert_hidden = ExpertLinear[E](H -> 2I)(expert_in)
expert_hidden = silu(expert_hidden[..., :I]) * expert_hidden[..., I:]
expert_out = ExpertLinear[E](I -> H)(expert_hidden)
moe = zeros(B*S,H).index_add(0, batch_index, expert_out * gate[:,None]).view(B,S,H)

if shared_intermediate_size > 0:
    shared = Linear(SI -> H)(silu(first_half(Linear(H -> 2SI)(x_norm))) * second_half(...))
    ff = moe + shared
else:
    ff = moe
x = residual + ff * residual_multiplier
```

Biases:

- Attention projections use `attention_bias`; source default false.
- Router, experts, shared MLP, and LM head are biasless.

## 6. Attention requirements

| Requirement | Source behavior |
| --- | --- |
| Causal/noncausal | Causal only |
| Cross-attention | None |
| MHA/MQA/GQA | MHA if `KVH == QH`, GQA if `KVH < QH` |
| Q width | `num_attention_heads * head_dim` |
| K/V width | `num_key_value_heads * head_dim` |
| Mask | `create_causal_mask`; no sliding window |
| RoPE | Q/K before cache update |
| Cache | Dynamic per-layer KV cache |
| Backend | FlashAttention, SDPA, Flex, or eager fallback |
| Packed metadata | Optional kwargs for padding-free attention; not required for basic path |

Cache ABI:

```text
key[layer]:   [batch, num_key_value_heads, cached_seq, head_dim]
value[layer]: [batch, num_key_value_heads, cached_seq, head_dim]
```

Eager attention repeats KV to Q-head count before matmul. DinoML should prefer a grouped-query attention kernel that maps `q_head -> kv_head` without materializing repeated KV. For parity, preserve this math order:

```text
Q/K/V linear -> reshape/transpose -> RoPE(Q,K) -> cache update
-> attention scores * attention_multiplier -> add mask -> softmax(fp32)
-> cast to query dtype -> dropout(training only) -> AV -> O projection
```

## 7. Position encoding and custom math

Default RoPE inverse frequencies:

```python
def granitemoeshared_inv_freq(config):
    dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    base = config.rope_parameters["rope_theta"]
    return 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
```

Application:

```python
def apply_granitemoeshared_rope(q, k, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)
```

Cos/sin are computed in fp32 with autocast disabled, then cast to hidden dtype. Prefill can cache cos/sin by `(rope type, theta, head_dim, positions, dtype)`. Decode positions depend on `past_key_values.get_seq_length()` unless caller supplies explicit `position_ids`.

## 8. Preprocessing and input packing

Core runtime inputs:

- `input_ids`: `[B, S]`.
- `inputs_embeds`: optional `[B, S, H]`; source requires exactly one of `input_ids` and `inputs_embeds`.
- `attention_mask`: optional padding mask consumed by `create_causal_mask`.
- `position_ids`: optional `[B, S]`; if omitted, source creates `[1, S]` from `arange(S) + past_seen_tokens`.
- `past_key_values`: optional DynamicCache.

There is no multimodal placeholder stitching, image/audio/video packing, codebook, or tokenizer-controlled structural metadata in this family. Tokenizer/generation special-token behavior is end-to-end text parity work, not GPU graph work.

## 9. Graph rewrite / lowering opportunities

### Rewrite: shared-dense MLP branch canonicalization

Source pattern:

```text
moe_hidden_states + shared_mlp(normed_hidden_states)
```

Replacement:

```text
SparseMoE(normed) + DenseSwiGLU(H -> 2SI -> H)
```

Preconditions:

- `shared_intermediate_size > 0`.
- Shared MLP uses biasless input/output linears and same `hidden_act` as experts.
- Addition occurs before residual multiplier.

Failure cases:

- Treating `granitemoeshared` as plain `granitemoe` silently drops the shared dense branch.

Parity test sketch:

- Tiny config with nonzero `shared_intermediate_size`; zero sparse expert weights and compare shared branch alone, then zero shared weights and compare sparse branch alone.

### Rewrite: sparse MoE reference lowering

Source pattern:

```text
router topk -> token regroup -> per-expert up/gate GEMM -> activation multiply
-> per-expert down GEMM -> gate scale -> index_add
```

Replacement:

```text
RouterTopK -> ExpertTokenPack -> grouped GEMM(H -> 2I)
-> SwiGLU -> grouped GEMM(I -> H) -> WeightedScatterAdd
```

Preconditions:

- Top-k is positive and small.
- Expert weights remain `[E, out, in]`.
- Router softmax is over selected top-k logits only.
- Dynamic expert counts are explicit runtime values.

Failure cases:

- Softmaxing all experts before top-k.
- Assuming fixed expert sizes.
- Dropping duplicate token destinations across top-k; accumulation must add them back by original token id.

Parity test sketch:

- Small `[B,S,H]` with forced empty experts, all tokens to one expert, and balanced routing; compare `top_indices`, gates, expert counts, `batch_index`, and final `index_add` output.

### Rewrite: GQA attention without `repeat_kv`

Source pattern:

```text
repeat_kv(K,V) -> attention(Q, repeated K,V)
```

Replacement:

```text
Grouped-query attention kernel using kv_head = q_head // (num_attention_heads / num_key_value_heads)
```

Preconditions:

- `num_attention_heads % num_key_value_heads == 0`.
- RoPE/caching/mask ordering matches source.
- Cache stores unexpanded KV.

Failure cases:

- Attention-weight output capture requiring dense `[B,QH,Q,K]` weights can be deferred.

Parity test sketch:

- Compare eager repeated-KV output to grouped kernel for odd sequence lengths and nonzero cache length.

### Rewrite: last-token-only LM head

Source pattern:

```text
logits = lm_head(hidden_states[:, slice_indices, :]) / logits_scaling
```

Replacement:

```text
Slice/gather requested hidden states -> GEMM(H, vocab) -> divide by logits_scaling
```

Preconditions:

- `logits_to_keep=1` for decode or a known bounded index tensor.
- Labels/loss are not requested.

Failure cases:

- Full-sequence logits requested for scoring or training.

### Rewrite: Granite multiplier folding

Source pattern:

```text
embed *= embedding_multiplier
residual + branch * residual_multiplier
scores *= attention_multiplier
logits /= logits_scaling
```

Replacement:

- Fold attention multiplier into attention score scale.
- Fuse residual multiplier into GEMM/attention/MLP epilogues.
- Apply logits scaling as post-LM-head scalar divide, unless weight aliasing proves folding safe.
- Avoid folding embedding and logits scales into tied weights unless aliasing semantics are explicitly modeled.

Failure cases:

- Tied embeddings/LM head can make load-time scaling transforms unsafe.

### Rewrite: quantized expert/LM weights via encoded constants

Source pattern:

```text
native dense nn.Linear weights
```

Replacement:

```text
encoded constant storage -> runtime dense materialization or dequant-before-GEMM provider
```

Preconditions:

- Quantization comes from DinoML import/loading policy or an external checkpoint format, not native Transformers source.
- Expert grouped GEMM and LM-head GEMM admit the quantized RHS materialization policy.
- Materialization/residency is artifact-visible.

Failure cases:

- Treating GGUF/bitsandbytes fork metadata as native `granitemoeshared` source behavior.
- Direct quantized-RHS expert GEMM without a validated grouped MoE provider contract.

## 10. Kernel fusion candidates

Highest priority:

- Sparse MoE router/pack/grouped-expert GEMM/weighted scatter-add. This is the main nonstandard runtime cost and contains data-dependent scatter/gather.
- Shared dense SwiGLU branch plus sparse MoE add. For `shared_intermediate_size > 0`, this is required parity and should share activation/epilogue code with dense MLP paths.
- GQA causal FlashAttention/SDPA with unexpanded KV cache. Avoids materializing `repeat_kv`.
- RMSNorm. Two per layer plus final norm, with fp32 variance.

Medium priority:

- QKV projection fusion from separate source tensors into a packed provider layout, split order `[Q, K, V]`.
- RoPE fused into attention preprocessing or attention kernel.
- Expert SwiGLU activation fused between grouped GEMMs.
- Residual multiplier and shared/sparse add fused into epilogues.
- Last-token-only LM head.

Lower priority:

- Router logits capture and auxiliary load-balancing loss.
- Attention weight materialization.
- Packed/varlen FlashAttention kwargs.
- Tensor parallelism.
- Quantized/GGUF expert and LM-head storage after dense grouped MoE parity.

## 11. Runtime staging plan

Stage 1: Config and weight loading.

- Parse `GraniteMoeSharedConfig`, including `rope_parameters` and legacy rope fields if supplied by external configs.
- Load expert weights in `[E, out, in]`.
- Preserve tied embedding/LM-head aliasing when config/checkpoint ties weights.
- Reject or warn on public `granitemoe` configs routed to this family unless caller explicitly requests shared-class compatibility.

Stage 2: Single-block parity.

- Implement RMSNorm, Granite multipliers, RoPE, eager GQA attention, sparse MoE reference, and shared dense MLP.
- Use a tiny synthetic config with nonzero `shared_intermediate_size`.

Stage 3: Full prefill parity.

- Run all decoder layers for short prompts.
- Keep MoE as explicit token pack plus per-expert GEMM loops if needed.

Stage 4: Decode with KV cache.

- Implement per-layer cache `[B, KVH, T, D]`.
- Validate position offset and post-RoPE key storage.

Stage 5: Optimized attention and logits.

- Add grouped attention provider.
- Add last-token-only LM head.

Stage 6: Optimized MoE.

- Add provider-backed router/top-k/pack/grouped GEMM/scatter-add.
- Profile empty experts, imbalanced experts, and uniform routing.

Stage 7: Optional loading/provider extensions.

- Add encoded-constant quantization policies for expert and LM-head weights only after dense parity is stable.

## 12. Parity and validation plan

- Config tests: source defaults, `num_key_value_heads=None` post-init, `shared_intermediate_size=0` disabled branch, nonzero shared branch.
- RMSNorm random tensor parity in fp32/fp16/bf16; suggested tolerances `1e-5` fp32 and `2e-2` fp16/bf16 for block outputs.
- RoPE parity for prefill and decode positions, including nonzero cache length.
- Attention parity against eager repeated-KV path for MHA and GQA.
- Router parity: logits, selected expert ids, selected gates, expert counts, sorted `batch_index`.
- Sparse MoE parity with empty experts, all tokens routed to one expert, and duplicate token accumulation from top-k.
- Shared MLP parity with sparse branch zeroed and shared branch zeroed.
- Single decoder layer parity with cache disabled.
- N-layer prefill parity on tiny synthetic config.
- Decode parity: prefill prompt then one token with cache; compare next-token logits.
- Last-token LM-head parity against full logits slice.
- Quantization parity, when added: dense baseline vs dequantized encoded expert/LM weights with explicit materialization policy.

No DinoML tests or Transformers imports were run for this report, per user scope.

## 13. Performance probes

- Prefill tokens/sec by sequence length and batch size.
- Decode tokens/sec by batch size and active cache length.
- KV cache memory: `layers * 2 * B * KVH * T * D * dtype_size`.
- Attention backend comparison: eager repeat-KV vs grouped Flash/SDPA.
- Router/top-k/sort/pack time vs expert GEMM time.
- Expert imbalance sweep: all tokens to one expert, uniform routing, long-tail routing.
- Grouped GEMM sweep by `(num_experts, top_k, tokens_per_expert, H, intermediate_size)`.
- Shared MLP cost sweep by `shared_intermediate_size`.
- Scatter-add bandwidth and determinism probe for duplicate token additions.
- LM-head full-sequence vs last-token-only throughput.
- Encoded/quantized weight load and dequant-before-GEMM probe after dense path is correct.

## 14. Skip/defer list

- Training loss and router auxiliary load-balancing loss.
- Gradient checkpointing.
- Router logits output capture unless diagnostics require it.
- Multi-GPU tensor parallelism.
- Packed/varlen FlashAttention kwargs on first path.
- Full attention weight output materialization.
- Public `granitemoe` and `granitemoehybrid` checkpoints unless routed through their separate audits.
- Native quantization, GGUF, bitsandbytes, GPTQ, AWQ, or external remote-code variants.
- Chat template/tokenizer parity beyond token ids, masks, and generation controller inputs.

## 15. Final implementation checklist

- [ ] Parse `GraniteMoeSharedConfig`, including `shared_intermediate_size`.
- [ ] Add admission checks for real `granitemoeshared` configs vs plain `granitemoe` mismatch.
- [ ] Load expert weights in `[num_experts, out_features, in_features]`.
- [ ] Preserve tied embedding/LM-head aliasing when present.
- [ ] Implement RMSNorm with fp32 variance.
- [ ] Implement embedding, attention, residual, and logits multipliers.
- [ ] Implement default RoPE and position-id cache offset handling.
- [ ] Implement causal MHA/GQA attention with KV cache `[B, KVH, T, D]`.
- [ ] Implement router top-k over logits and softmax over selected logits only.
- [ ] Implement token regrouping, dynamic expert counts, and weighted scatter-add.
- [ ] Implement sparse expert SwiGLU grouped/per-expert GEMMs.
- [ ] Implement optional shared dense SwiGLU branch.
- [ ] Add last-token-only LM-head lowering.
- [ ] Add single-block parity tests.
- [ ] Add full tiny synthetic prefill parity tests.
- [ ] Add decode-with-cache parity tests.
- [ ] Benchmark attention, sparse MoE routing, grouped expert GEMMs, shared MLP, scatter-add, and LM-head slices.
