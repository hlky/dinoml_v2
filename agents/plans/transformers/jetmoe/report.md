# JetMoe Transformers Audit

## 1. Source basis

Transformers commit/version: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4` from `X:/H/transformers`.

Model id: primary `jetmoe/jetmoe-8b`; representative configs also checked for `jetmoe/jetmoe-8b-sft`, `jetmoe/jetmoe-8b-chat`, `AndreaUnibo/JetMoE_base_full_trained`, and out-of-scope `thomasgauthier/expanded-jetmoe-untrained`.

Config source: Hub `config.json` snapshots plus native `src/transformers/models/jetmoe/configuration_jetmoe.py`. A compact sweep is saved in `config_sweep.md`.

Source files inspected:

- `src/transformers/models/jetmoe/configuration_jetmoe.py`
- `src/transformers/models/jetmoe/modeling_jetmoe.py`
- `src/transformers/models/jetmoe/modular_jetmoe.py`
- `tests/models/jetmoe/test_modeling_jetmoe.py`
- `docs/source/en/model_doc/jetmoe.md`

Any missing files or assumptions:

- `modeling_jetmoe.py` is generated from `modular_jetmoe.py`; future source-diff work should treat the modular file as authoritative.
- No gated links were encountered. `jetmoe/jetmoe-8b-chat` advertises `auto_map` custom-code metadata, but this report scopes required runtime behavior to the native in-library source at the pinned commit.
- `thomasgauthier/expanded-jetmoe-untrained` has `model_type="expandedjetmoe"` and should be rejected or separately audited.

Primary DinoML runtime target: `JetMoeForCausalLM` text-generation prefill and decode. `JetMoeModel` hidden-state parity is required as the base body. `JetMoeForSequenceClassification` is optional/deferred for this target.

## 2. High-level architecture

JetMoe is a text-only causal decoder with two sparse routed sublayers per block:

```text
tokenizer/input_ids -> token embedding -> N decoder blocks
  -> final RMSNorm -> tied/untied LM head -> logits/sampling
```

Each decoder block has:

```text
RMSNorm -> Mixture-of-Attention query/output experts + dense KV attention -> residual
RMSNorm -> Mixture-of-MLP experts -> residual
```

Stage decomposition:

- CPU/data pipeline: Llama tokenizer, optional chat template, left padding, attention mask construction, sampling/generation controller.
- GPU/runtime prefill: embeddings, rotary positions, causal masked attention, routed attention experts, routed MLP experts, LM head.
- GPU/runtime decode: single-token query routing, dense KV projection, RoPE, per-layer KV cache update, repeated-KV attention, routed output projection, routed MLP.
- Independently stageable validation: router topology, per-expert grouped GEMM, attention without cache, cache update/reorder, final logits slice.

## 3. Important config dimensions

Native source defaults:

| field | default | source/runtime meaning |
|---|---:|---|
| `vocab_size` | 32000 | token embedding and LM head width |
| `hidden_size` | 2048 | residual width |
| `num_hidden_layers` | 12 | source default only; official 8B configs use 24 |
| `num_key_value_heads` | 16 | dense K/V cache heads |
| `num_experts_per_tok` | 2 | router top-k and MoA query-head multiplier |
| `num_attention_heads` | derived 32 | `num_key_value_heads * num_experts_per_tok` |
| `kv_channels` / `head_dim` | 128 | q/k/v head dim |
| `intermediate_size` | 5632 | per-MLP-expert hidden width before gated output |
| `num_local_experts` | 8 | MoA and MoE expert count |
| `max_position_embeddings` | 4096 | RoPE/cache expected context |
| `activation_function` | `silu` | SwiGLU-style expert activation |
| `rms_norm_eps` | `1e-6` | official checkpoints set `1e-5` |
| `attention_dropout` | 0.0 | inference dropout disabled |
| `use_cache` | true | autoregressive KV cache enabled |
| `tie_word_embeddings` | true | LM head aliases token embedding when tied |

Representative checkpoint sweep:

| model id | operator-significant variation |
|---|---|
| `jetmoe/jetmoe-8b` | 24 layers, 8 experts, top-2, 16 KV heads, hidden 2048, bf16 integration tests in source |
| `jetmoe/jetmoe-8b-sft` | same native graph dimensions as base; finetuned weights |
| `jetmoe/jetmoe-8b-chat` | same graph dimensions; adds chat template and `auto_map` remote-code metadata |
| `AndreaUnibo/JetMoE_base_full_trained` | same graph dimensions; includes BitsAndBytes NF4 quantization metadata that native source does not implement |
| `thomasgauthier/expanded-jetmoe-untrained` | out of scope: `expandedjetmoe`, separate expert-count fields for attention and MLP |

## 3a. Family variation traps

- Attention heads are derived from top-k: `num_attention_heads = num_key_value_heads * num_experts_per_tok`.
- JetMoe is not ordinary GQA. Query and output projections are sparse MoA expert projections, while K/V projection is dense.
- K/V heads are repeated with `tensor.repeat(1, top_k, 1, 1)`, not normal `repeat_interleave`; head ordering must match source.
- Cache stores post-RoPE K and V at `[B, num_key_value_heads, T, head_dim]`; repeat to attention heads is transient.
- Router topology is data-dependent: `topk`, `scatter`, `sum(...).tolist()`, `sort`, gather, split by `expert_size`, and `index_add`.
- Official configs contain historical field names. DinoML should parse or normalize them explicitly instead of assuming every field is read by the current source class.
- `rms_norm_eps` differs between source default (`1e-6`) and official checkpoints (`1e-5`).
- `jetmoe-8b-chat` carries remote-code `auto_map`; route to native source only after compatibility checks.
- BitsAndBytes quantization configs are loading/storage policy, not native JetMoe neural ops.
- FlashAttention right-padding is explicitly skipped in tests; left-padding tokenizer metadata is the safer admission path.
- No vision/audio/layout tensors exist. NHWC/channel-last layout translation is not relevant.

## 4. Operator coverage checklist

Tensor/layout ops:

- `Embedding(input_ids) -> [B,S,2048]`
- reshape/view/flatten for `[B,S,H] -> [B*S,H]`, `[B,S,top_k,kv_heads*head_dim]`, and attention transposes
- `chunk(2, dim=-1)` for K/V and MLP gate/value split
- `cat` for RoPE frequency duplication and expert result concatenation
- `gather`/advanced indexing with router-derived token indices
- `topk`, `sort`, integer division, `scatter`, `sum`, `tolist`-equivalent shape/topology materialization
- `zeros`, `index_add`, `split` by dynamic per-expert token counts
- final `logits_to_keep` slicing before LM head

Neural network primitives:

- RMSNorm over last dim, fp32 variance, scale in original dtype
- Dense linear: token embedding, dense K/V projection `2048 -> 4096`, LM head `2048 -> 32000`
- Router linear: `2048 -> 8`, no bias, fp32 logits
- MoA expert input projections: 8 weights `[2048,2048]`, selected top-2 per token
- MoA expert output projections: 8 weights `[2048,2048]`, selected top-2 per token, gate-weighted `index_add`, bias `[2048]`
- MoE expert input projections: 8 weights `[11264,2048]`, split into two `[5632]` halves
- MoE expert output projections: 8 weights `[2048,5632]`, gate-weighted `index_add`, bias `[2048]`
- SiLU/SwiGLU: `silu(up) * gate`
- residual add

Attention primitives:

- causal self-attention, MHA query heads 32, KV heads 16, head dim 128 for official configs
- dense K/V projection and routed query projection
- additive causal/padding mask from `create_causal_mask`
- softmax over key dimension in fp32 for eager path
- SDPA/Flash/Flex backend compatibility with source-specific head repeat

Position/rotary ops:

- default RoPE with `rope_theta=10000`, full `head_dim=128`
- `position_ids = arange(S) + past_seen_tokens` when omitted
- dynamic RoPE machinery exists through shared rope utilities if non-default `rope_parameters` appear, but official sampled configs use default-style fields

Generation/cache ops:

- DynamicCache creation when `use_cache=True`
- per-layer K/V update after RoPE and before top-k head repeat
- cache length used to offset position ids
- cache reorder for beam search comes from Transformers cache utilities; can defer for greedy first pass

Quantized/packed weight metadata ops:

- Native source has no family-owned quantized kernels. BitsAndBytes NF4 metadata in a sampled community checkpoint should be treated as a loading fallback/reject path until DinoML owns that provider.

## 5. Layer/block breakdown

Decoder block, repeated `num_hidden_layers` times:

```text
residual = x
x = RMSNorm(x)

# Mixture of Attention
flat = x.reshape(B*S, H)
router_logits = Linear(flat, H -> E, bias=False).float()
top_values, top_experts = topk(router_logits, K=top_k)
gates = softmax(top_values, dim=-1)
topology = sort(flatten(top_experts))
q_expert_in = flat[batch_index]                         # [B*S*top_k, H]
q_grouped = expert_linear(q_expert_in, [E, KVH*D, H])
q = ungroup_by_index_add(q_grouped).view(B,S,top_k,KVH,D)

kv = Linear(x, H -> 2*KVH*D, bias=False)
k, v = chunk(kv, 2)
q, k = RoPE(q, k)
k, v = cache.update(k, v)
k = repeat(k, repeats=top_k along head axis)
v = repeat(v, repeats=top_k along head axis)
attn = causal_attention(q, k, v, mask)
attn = attn.view(B,S,top_k,KVH*D)
attn = routed_expert_output_projection(attn, same topology)
x = residual + attn

residual = x
x = RMSNorm(x)
x = routed_mlp_experts(x)
x = residual + x
```

Official 8B shape constants: `H=2048`, `E=8`, `top_k=2`, `KVH=16`, `D=128`, `intermediate=5632`.

Projection bias rules:

- Dense K/V projection has no bias.
- Router projections have no bias.
- Expert projection weights are bias-free, but MoA and MoE add a learned final bias `[hidden_size]` after expert merge.
- LM head has no bias.

## 6. Attention requirements

Attention type: causal self-attention with sparse MoA query/output projections.

Head geometry for official configs:

- query heads: 32
- key/value cache heads: 16
- head dim: 128
- query width: `top_k * num_key_value_heads * head_dim = 4096` before per-token top-k grouping; represented as `[B,S,2,2048]` then `[B,32,S,128]`
- K/V projection width: `2 * num_key_value_heads * head_dim = 4096`

Masking style:

- Source builds an additive causal mask through `create_causal_mask`.
- Right-padding FlashAttention inference is explicitly skipped by tests; left-padding tokenizer metadata should be admitted first.

KV cache:

- Stored K/V shape per layer: `[B, num_key_value_heads, T_total, head_dim]`.
- K is cached after RoPE.
- Before attention, K/V are repeated with full-head-block `repeat(1, top_k, 1, 1)` to `[B, num_attention_heads, T_total, head_dim]`.

Backend compatibility:

- Source advertises FlashAttention, SDPA, and Flex support.
- DinoML optimized attention must preserve the unusual K/V repeat ordering and routed query layout. A generic GQA lowering that interleaves repeated KV heads is unsafe.

## 7. Position encoding and custom math

Default RoPE:

```python
inv_freq = 1.0 / (rope_theta ** (arange(0, head_dim, 2) / head_dim))
freqs = inv_freq[None, :, None] @ position_ids[:, None, :]
emb = cat([freqs, freqs], dim=-1).transpose(1, 2)
cos = cos(emb) * attention_scaling
sin = sin(emb) * attention_scaling
```

Application:

```python
def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return cat([-x2, x1], dim=-1)

q = q * cos[:, None, :, :] + rotate_half(q) * sin[:, None, :, :]
k = k * cos[:, None, :, :] + rotate_half(k) * sin[:, None, :, :]
```

Precompute opportunity:

- `inv_freq` is static per config.
- `cos/sin` can be precomputed or cached for bounded position ranges and dtype, but decode must offset by `past_seen_tokens`.

## 8. Preprocessing and input packing

Tokenizer contract from sampled official metadata:

- `LlamaTokenizer`
- `model_max_length=4096`
- `add_bos_token=true`, `add_eos_token=false`
- BOS id 1, EOS id 2, PAD token set to `</s>` in tokenizer metadata
- `padding_side="left"`
- chat repo adds a chat template with `<|user|>`, `<|system|>`, and `<|assistant|>` text markers followed by EOS

GPU/runtime inputs:

- `input_ids [B,S]` or `inputs_embeds [B,S,H]`, exactly one required
- optional `attention_mask [B,S]`
- optional `position_ids [B,S]`; generated from cache length if omitted
- optional `past_key_values`

No image/audio/video processors, placeholder token scatter, packed varlen metadata, or cu-seqlens are family-owned in this source.

## 9. Graph rewrite / lowering opportunities

### Rewrite: grouped expert linear

Source pattern:

```text
expert_inputs = flat[batch_index]
split by expert_size
for expert i: F.linear(input_i, weight[i])
cat(outputs)
```

Replacement pattern: grouped GEMM or segmented GEMM over expert-major token groups.

Preconditions:

- `expert_size` is nonnegative and sums to `B*S*top_k`.
- Sorted topology groups tokens by expert exactly as source `top_k_indices.flatten().sort(0)`.
- Expert weight layout is `[num_experts, out_features, in_features]`.

Failure cases: unsupported dynamic segment sizes, non-top-2 variants without regenerated topology guards, quantized expert weights without provider support.

Parity test sketch: compare grouped-GEMM output and final ungrouped `index_add` against PyTorch for random router logits with ties and empty experts.

### Rewrite: MoE SwiGLU expert block

Source pattern:

```text
Linear(H -> 2I) -> chunk -> silu(first) * second -> Linear(I -> H)
```

Replacement pattern: fused expert GEMM epilogue for SiLU/multiply and optional second expert GEMM.

Preconditions: activation is `silu`, input projection output is exactly `2 * intermediate_size`, no per-expert bias.

Failure cases: checkpoint changes activation, nonstandard hidden sizes, sparse routing provider cannot preserve expert order.

### Rewrite: MoA attention route

Source pattern: routed query projection, dense K/V projection, RoPE, attention, routed output projection with same topology.

Replacement pattern: preserve router topology once and reuse it for Q projection and output projection; fuse topology materialization with expert dispatch.

Preconditions: same `topo_info` is passed to `reduce`; attention output reshapes to `[B,S,top_k,KVH*D]`.

Failure cases: generic attention lowering repeats K/V in wrong order; topology recomputation introduces tie differences.

### Rewrite: last-token-only logits

Source pattern: `lm_head(hidden_states[:, slice_indices, :])` using `logits_to_keep`.

Replacement pattern: decode path computes LM head only for last token or requested indices.

Preconditions: no loss computation; generation only needs final-token logits.

Failure cases: callers request full logits, tensor-valued `logits_to_keep`, or sequence-classification head.

## 10. Kernel fusion candidates

Highest priority:

- Router topology kernel: linear logits, top-k, softmax, sort/group indices, expert sizes. This is the core JetMoe bottleneck and has data-dependent shape effects.
- Grouped/segmented expert GEMM for MoE and MoA weights. Every layer uses four routed expert projection families.
- GQA-like causal attention with JetMoe head-repeat order and KV cache.
- RMSNorm and residual add around attention/MLP.

Medium priority:

- SwiGLU expert activation fuse between expert input and output projections.
- RoPE + attention prefill fusion, with cache update after RoPE.
- Last-token-only LM head for decode.
- Router topology reuse between MoA map and reduce.

Lower priority:

- Auxiliary load-balancing loss and router logits output.
- Sequence classification pooling/head.
- Beam-search cache reorder and generation-controller extras beyond greedy/sampling.
- BitsAndBytes/NF4 loading provider for community checkpoints.

## 11. Runtime staging plan

Stage 1: parse config, normalize historical Hub fields, load dense weights, and run embedding/RMSNorm/one dense K/V projection parity.

Stage 2: implement router topology parity for fixed small shapes, including top-k, softmax gates, sort order, expert sizes, empty experts, and `index_add`.

Stage 3: implement one routed MoE MLP block with dense per-expert fallback GEMMs and compare layer output.

Stage 4: implement MoA attention without cache for prefill: routed Q, dense K/V, RoPE, source-repeat K/V, causal attention, routed output projection.

Stage 5: wire full decoder prefill through all layers and final LM head, with `logits_to_keep` optimization guarded.

Stage 6: add decode with per-layer KV cache, position offset, and greedy token parity.

Stage 7: replace fallback per-expert loops with grouped GEMM/provider-backed kernels and optimized attention.

Initially stub: training loss, aux router loss, `output_attentions`, `output_router_logits`, sequence classification, BitsAndBytes loading, beam search, and remote-code variants.

## 12. Parity and validation plan

- Config normalization tests for official base/SFT/chat configs, including historical field aliases and derived attention heads.
- RMSNorm random tensor tests in fp32/fp16/bf16 with fp32 variance tolerance.
- RoPE tests against source for prefill and decode offset positions.
- Router topology tests with deterministic logits, ties, empty experts, and all experts used.
- Expert linear tests for `[E,out,in]` weight layout and split/concat order.
- Single MoE MLP parity test at small `B,S,E,top_k`.
- Single MoA attention parity test with cache disabled and attention backend forced eager.
- Cache parity test: prefill then one-token decode equals full-prefix run for logits.
- Full one-layer and all-layer logits parity against `jetmoe/jetmoe-8b` for a short prompt.
- Generation smoke parity for greedy completion using the source integration-test prompt.

Suggested tolerances:

- fp32: `rtol=1e-4`, `atol=1e-5` for isolated ops; relax around softmax/router composition if needed.
- fp16/bf16: `rtol=1e-2`, `atol=1e-2` for end-to-end logits, matching source integration-test scale.

## 13. Performance probes

- Tokenizer throughput separately from runtime.
- Router topology latency by `B*S`, `num_experts`, and `top_k`.
- Grouped expert GEMM throughput with skewed and uniform expert loads.
- Prefill attention throughput by sequence length and batch size.
- Decode tokens/sec with KV cache and last-token LM head.
- KV cache memory: `layers * 2 * B * kv_heads * T * head_dim * dtype_size`.
- Router temporary memory and index bandwidth for `B*S*top_k` token copies.
- Dense fallback versus grouped-GEMM provider comparison.
- Attention backend comparison: eager/SDPA/Flash-like JetMoe-compatible lowering.
- Quantized weight load/dequant probes only after a provider contract exists.

## 14. Skip/defer list

- Training, labels loss, and auxiliary load-balancing loss.
- `output_router_logits`, `output_attentions`, and full hidden-state tracing beyond debug parity.
- `JetMoeForSequenceClassification`.
- Beam search, cache reorder, and speculative decoding.
- Remote-code variants and `expandedjetmoe`.
- BitsAndBytes NF4 execution/loading as a first-class provider.
- Right-padded FlashAttention path.
- Multi-GPU tensor parallel and pipeline parallel behavior.

## 15. Final implementation checklist

- [ ] Parse `JetMoeConfig` and normalize historical Hub aliases.
- [ ] Reject `model_type!="jetmoe"` and unaudited remote-code-only variants.
- [ ] Load tied token embedding / LM head weights with alias preservation.
- [ ] Implement RMSNorm and RoPE parity.
- [ ] Implement JetMoe router topology: top-k, gates, sort, expert sizes, gather, `index_add`.
- [ ] Implement expert weight layout `[E,out,in]` for grouped dense fallback.
- [ ] Implement routed MoE MLP with SiLU-gated activation.
- [ ] Implement routed MoA query and output projections with shared topology.
- [ ] Implement dense K/V projection and JetMoe-specific K/V repeat order.
- [ ] Implement causal attention prefill with additive mask.
- [ ] Implement per-layer KV cache storing post-RoPE K/V before repeat.
- [ ] Implement decode position offset and cache update.
- [ ] Implement `logits_to_keep` / last-token LM head optimization.
- [ ] Add config sweep and checkpoint compatibility tests.
- [ ] Add one-block, prefill-logits, and decode-token parity tests.
- [ ] Benchmark router, grouped experts, attention, decode, and KV memory.
