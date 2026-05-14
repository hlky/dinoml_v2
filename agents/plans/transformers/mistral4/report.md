# Transformers audit: mistral4

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model family: mistral4
Primary runtime target: text-only causal LM decoder / prefill+decode logits
Config source: Transformers Mistral4Config plus representative Hub config.json files
```

Source files inspected:

- `X:/H/transformers/src/transformers/models/mistral4/configuration_mistral4.py`
- `X:/H/transformers/src/transformers/models/mistral4/modular_mistral4.py`
- `X:/H/transformers/src/transformers/models/mistral4/modeling_mistral4.py`
- `X:/H/transformers/src/transformers/models/mistral4/convert_mistral4_weight_to_hf.py`
- `X:/H/transformers/tests/models/mistral4/test_modeling_mistral4.py`

`modeling_mistral4.py` is generated from `modular_mistral4.py`; future source edits should target the modular file, but this report uses the generated file as the runtime source of truth.

Representative configs inspected and snapshotted locally:

- [mistralai/Mistral-Small-4-119B-2603](https://hf.co/mistralai/Mistral-Small-4-119B-2603): `mistral3` multimodal wrapper with `text_config.model_type="mistral4"`; snapshot `mistralai_Mistral-Small-4-119B-2603_config.json`.
- [darkc0de/Mistral-Small-4-119B-2603-heretic](https://hf.co/darkc0de/Mistral-Small-4-119B-2603-heretic): open finetune/mirror of the same wrapper; snapshot `darkc0de_Mistral-Small-4-119B-2603-heretic_config.json`.
- [onnx-internal-testing/tiny-random-Mistral4ForCausalLM](https://hf.co/onnx-internal-testing/tiny-random-Mistral4ForCausalLM): tiny text-only test config; snapshot `onnx_tiny-random-Mistral4ForCausalLM_config.json`.
- [katuni4ka/tiny-random-mistral4-text-only](https://hf.co/katuni4ka/tiny-random-mistral4-text-only): remote-code tiny text-only config; snapshot `katuni4ka_tiny-random-mistral4-text-only_config.json`.

Missing files or assumptions:

- No official pure text-only production `Mistral4ForCausalLM` checkpoint was found in the sampled Hub results. The official production checkpoint routes through `Mistral3ForConditionalGeneration`; this report audits only the `mistral4` text decoder subconfig and defers Pixtral vision/projector/image-token stitching to the `mistral3`/Pixtral audits.
- The `mistralai/Mistral-Small-4-119B-2603-NVFP4` config path returned `Entry not found` during raw config fetch; its repo metadata indicates a compressed-tensors/vLLM variant and should be handled as a separate loading/provider policy if needed.

## 2. High-level architecture

Mistral4 is a decoder-only causal language model with MLA-style low-rank attention projections, partial RoPE, optional interleaved RoPE layout, and MoE feed-forward layers with shared experts.

Text-only dataflow:

```text
input_ids -> token embedding -> N decoder blocks -> final RMSNorm -> lm_head -> logits/sampling
```

Production wrapper dataflow for the official checkpoint, out of scope for this text-only report:

```text
text/image processor -> Pixtral vision tower/projector -> image token embedding stitch -> Mistral4 text decoder -> logits
```

Independently stageable pieces:

- Token embedding, causal mask, position id generation.
- One decoder block with low-rank attention and either dense MLP or MoE MLP.
- Autoregressive KV cache update.
- Final norm and optionally last-token-only `lm_head`.
- FP8/compressed weight loading as a provider contract, separate from dense graph parity.

## 3. Important config dimensions

| Field | Source default | Official text_config | Tiny text-only examples | Notes |
|---|---:|---:|---:|---|
| `vocab_size` | 131072 | 131072 | 32000 | Tokenizer/LM head ABI. |
| `hidden_size` | 4096 | 4096 | 64 | Decoder width. |
| `num_hidden_layers` | 36 | 36 | 3 | Repeated blocks. |
| `num_attention_heads` | 32 | 32 | 4 | Source attention materializes per query head. |
| `num_key_value_heads` | 32 | 32 | 4 | Source effectively assumes equal to attention heads; see traps. |
| `qk_nope_head_dim` | 64 | 64 | 16 | Non-rotary Q/K width. |
| `qk_rope_head_dim` | 64 | 64 | 8 | Rotary Q/K width. |
| `qk_head_dim` / `head_dim` | 128 | 128 | 24 | Set post-init as `qk_nope + qk_rope`. |
| `v_head_dim` | 128 | 128 | 16 | Value width; may differ from Q/K width. |
| `q_lora_rank` | 1024 | 1024 | 32 | If `None`, direct Q projection path. |
| `kv_lora_rank` | 256 | 256 | 16 | Low-rank KV compression width. |
| `intermediate_size` | 12288 | 12288 | 32 | Dense MLP width for early dense layers. |
| `moe_intermediate_size` | 2048 | 2048 | 16 | Per expert hidden width. |
| `n_routed_experts` | 128 | 128 | 4 | Routed experts. |
| `num_experts_per_tok` | 4 | 4 | 2 | Router top-k. |
| `n_shared_experts` | 1 | 1 | 1 | Shared dense expert multiplier. |
| `first_k_dense_replace` | 0 | 0 | 0 or 1 | Layers below this use dense MLP instead of MoE. |
| `max_position_embeddings` | 1048576 | 1048576 | 1048576 | Long-context admission pressure. |
| RoPE | YaRN | YaRN factor 128, original 8192 | YaRN factor 128 | Includes Mistral/Llama-4 query scale. |
| `rope_interleave` | true | true | true | Changes RoPE pre-layout. |
| `attention_bias` | false | false | false | If true, source adds bias to Q-a, KV-a, O only. |
| dtype | config default unspecified | bfloat16 plus FP8 quant config | float32 | Official repo metadata reports FP8 weights. |
| cache | true | true | true | DynamicCache supported. |

Representative checkpoint sweep:

| Repo | Class / scope | Dtype / quant | Layers | Heads | Dims | MoE | Variation |
|---|---|---|---:|---:|---|---|---|
| `mistralai/Mistral-Small-4-119B-2603` | `Mistral3ForConditionalGeneration`, text subconfig is Mistral4 | `bfloat16`, FP8 quantization config | 36 | 32 Q / 32 KV | H=4096, qk=128, v=128 | 128 routed, top-4, shared=1 | Multimodal wrapper, Pixtral vision tower, FP8 scale tensors. |
| `darkc0de/Mistral-Small-4-119B-2603-heretic` | Same wrapper | `bfloat16`, no quant config in fetched config | 36 | 32 / 32 | same | same | Finetune/mirror; text dims unchanged. |
| `onnx-internal-testing/tiny-random-Mistral4ForCausalLM` | pure text `Mistral4ForCausalLM` | `float32` | 3 | 4 / 4 | H=64, qk=24, v=16 | 4 routed, top-2, first dense layer | Tests unequal QK/value widths and grouped routing. |
| `katuni4ka/tiny-random-mistral4-text-only` | pure text remote-code checkpoint | `float32` | 3 | 4 / 4 | H=64, qk=24, v=16 | 4 routed, top-2 | Has historical `rope_scaling`/`rope_theta` alongside `rope_parameters`; current in-library source uses `rope_parameters`. |

## 3a. Family variation traps

- Official production checkpoint is not pure `mistral4`: it is a `mistral3` multimodal model whose text decoder is Mistral4. First DinoML target should either load a pure text checkpoint or explicitly extract the text submodule from the wrapper.
- `hidden_size` need not equal `num_attention_heads * qk_head_dim` in tiny configs (`64 != 4 * 24`). Do not infer projection widths from hidden size.
- Source `num_key_value_heads` is used for `num_key_value_groups`, but K/V projections are shaped with `num_attention_heads`. DinoML should initially require `num_key_value_heads == num_attention_heads`; otherwise eager `repeat_kv` can over-expand K/V.
- Q/K width and V width may differ. Flash attention path pads values when `qk_head_dim != v_head_dim`; eager attention naturally returns V-width outputs.
- `q_lora_rank=None` switches Q projection from `q_a_proj -> RMSNorm -> q_b_proj` to a direct `q_proj`.
- `first_k_dense_replace` changes early blocks from MoE to dense SwiGLU MLP.
- `attention_bias=True` does not add bias everywhere: source uses bias on `q_a_proj`, `kv_a_proj_with_mqa`, and `o_proj`, not `q_b_proj`, `kv_b_proj`, or MLP projections.
- `rope_interleave=True` rewrites Q/K rotary halves through a view/transpose/reshape before ordinary rotate-half math. Converted weights may already assume this layout.
- `sliding_window` appears in converted configs but is not a declared `Mistral4Config` field. The model body delegates masking to `create_causal_mask`; admit `sliding_window=None` first and validate non-null masks separately.
- FP8 official weights include `activation_scale` and `weight_scale_inv` tensors and exclude vision/projector/lm_head from conversion. Treat this as a loading/provider contract, not a normal dtype.
- `mlp_bias`, `rope_scaling`, and top-level `rope_theta` appear in some checkpoint configs but are not read by the inspected Mistral4 source path.

## 4. Operator coverage checklist

Tensor/layout ops:

- Embedding gather: `input_ids [B,S] -> hidden [B,S,H]`.
- Shape/view/transpose/reshape/contiguous around `[B,S,H] <-> [B,heads,S,D]`.
- Split/concat on last dim for Q/K rotary and non-rotary parts.
- Causal mask construction with optional external `attention_mask`.
- `F.pad` for flash attention value padding when `v_head_dim < qk_head_dim`.
- `logits_to_keep` slicing over sequence positions before LM head.

Neural primitives:

- RMSNorm over last dim, fp32 variance, cast back to input dtype.
- Linear/GEMM with optional FP8 materialization for official weights.
- Dense SwiGLU MLP: `gate_proj H->I`, `up_proj H->I`, `silu(gate) * up`, `down_proj I->H`.
- MoE shared expert: same SwiGLU with width `moe_intermediate_size * n_shared_experts`.
- Routed expert GEMM: packed `gate_up_proj [E,2*moe_I,H]`, chunk gate/up, `down_proj [E,H,moe_I]`.

Attention primitives:

- Low-rank Q path: `Linear(H->q_lora) -> RMSNorm(q_lora) -> Linear(q_lora -> heads*qk_head_dim)`.
- Low-rank KV path: `Linear(H -> kv_lora_rank + qk_rope_head_dim)`, split compressed K and single-head rotary K.
- KV expansion path: `RMSNorm(kv_lora) -> Linear(kv_lora -> heads*(qk_nope_head_dim + v_head_dim))`, split K-pass and V.
- Partial RoPE on `q_rot [B,heads,S,qk_rope]` and `k_rot [B,1,S,qk_rope]`; expand K rotary part to all heads.
- Query post-scale: `query *= 1 + beta * log(1 + floor(position_ids / original_max_position_embeddings))`.
- Causal self-attention with optional backend dispatch: eager, SDPA, FlashAttention, or flex attention.

Position/rotary/custom math:

- YaRN RoPE via `ROPE_INIT_FUNCTIONS` using `rope_parameters`.
- Optional interleaved RoPE pre-transform.
- Mistral/Llama-4 query scaling.

Generation/cache ops:

- DynamicCache creation when `use_cache=True`.
- Per-layer cache update after RoPE and query scaling preparation: cached K/V shapes are `[B,heads,total_S,qk_head_dim]` and `[B,heads,total_S,v_head_dim]` before any flash value padding.
- Beam/cache reorder is inherited from Transformers cache utilities, not custom in Mistral4.

MoE routing ops:

- Router linear `H -> n_routed_experts`.
- Softmax over experts.
- Group reshape `[tokens, n_group, experts_per_group]`, `topk(2)` per group, sum.
- `topk(topk_group)` groups, `scatter_` group mask, `masked_fill`.
- `topk(num_experts_per_tok)` experts, `gather` weights, optional normalization, weighted `index_add`.

Quantized/packed metadata:

- Official FP8 path has weight tensors plus `activation_scale` and `weight_scale_inv`.
- Expert source weights are converted/fused from original `w1`, `w2`, `w3` into packed `gate_up_proj` and `down_proj`; gate/up split order is `[gate, up]`.

Optional/deferred heads:

- `Mistral4ForSequenceClassification` and `Mistral4ForTokenClassification` are implemented through generic heads. They are optional/deferred for causal LM parity.

## 5. Layer/block breakdown

Decoder block, repeated `num_hidden_layers` times:

```text
residual = x
x = RMSNorm(x)
q = q_b_proj(RMSNorm(q_a_proj(x)))       # or direct q_proj when q_lora_rank is None
q = view [B,heads,S,qk_head_dim]
q_pass, q_rot = split(q, [qk_nope_head_dim, qk_rope_head_dim])
compressed_kv = kv_a_proj_with_mqa(x)
k_low, k_rot = split(compressed_kv, [kv_lora_rank, qk_rope_head_dim])
kv = kv_b_proj(RMSNorm(k_low))
k_pass, v = split(view(kv), [qk_nope_head_dim, v_head_dim])
q_rot, k_rot = RoPE(q_rot, k_rot)
k = concat(k_pass, expand(k_rot))
q = concat(q_pass, q_rot) * llama4_position_scale(position_ids)
attn = causal_attention(q, k, v, cache)
x = residual + o_proj(attn)
residual = x
x = RMSNorm(x)
x = residual + (MoE(x) or dense SwiGLU(x))
```

Official dimensions:

- `q_a_proj`: `4096 -> 1024`, bias false.
- `q_b_proj`: `1024 -> 4096`, bias false.
- `kv_a_proj_with_mqa`: `4096 -> 320`, bias false.
- `kv_b_proj`: `256 -> 6144`, bias false.
- `o_proj`: `4096 -> 4096`, bias false.
- Dense MLP: `4096 -> 12288 -> 4096`.
- MoE expert: per expert `4096 -> 2*2048 -> 4096`; shared expert width `2048`.

## 6. Attention requirements

Required attention is causal self-attention for text generation.

- Pattern: dense causal self-attention; no cross-attention in `mistral4`.
- Heads: effectively MHA for inspected configs (`num_attention_heads == num_key_value_heads`).
- Q/K dim: `qk_head_dim = qk_nope_head_dim + qk_rope_head_dim`.
- V dim: `v_head_dim`, not necessarily equal to Q/K dim.
- Query/key length: rectangular during decode (`Q=1`, `K=past+1`) through cache.
- Masking: `create_causal_mask` combines causal and padding masks; flash right-padding test is explicitly skipped in Transformers.
- Packed/varlen: no explicit cu-seqlens ABI in Mistral4 source; backend attention interfaces may accept backend-specific kwargs.
- Sliding/local attention: no source-level custom local attention path verified; require `sliding_window=None` initially.
- KV cache: store post-RoPE, post-concat K and raw V per layer. K has width `qk_head_dim`; V has width `v_head_dim`.
- Flash/SDPA compatibility: source advertises FlashAttention, SDPA, and flex attention. Flash path pads V to QK width before backend call and slices output back to V width.

Fused attention parity must preserve:

- RoPE before cache update.
- K rotary expansion before concatenation.
- Query scaling after RoPE/concat and before attention score computation.
- Softmax upcast to fp32 in eager fallback.
- Mask addition before softmax.

## 7. Position encoding and custom math

RoPE construction:

```python
inv_freq = 1.0 / (rope_theta ** (arange(0, head_dim, 2) / head_dim))
freqs = inv_freq[None, :, None] @ position_ids[:, None, :]
emb = cat(freqs, freqs, dim=-1)
cos, sin = cos(emb) * attention_scaling, sin(emb) * attention_scaling
```

Interleaved RoPE path:

```python
def interleave_before_rope(x):
    b, h, s, d = x.shape
    return x.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
```

Rotate-half application:

```python
def rope(q, k, cos, sin):
    cos = cos[:, None, :, :]
    sin = sin[:, None, :, :]
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin
```

Mistral/Llama-4 attention scale:

```python
scale = 1 + beta * log(1 + floor(position_ids / original_max_position_embeddings))
query_states = query_states * scale[:, None, :, None]
```

Precompute candidates:

- Base inv-frequencies and static YaRN factors can be cached by config.
- Cos/sin depend on runtime `position_ids`, especially decode offsets.
- Query scale depends on runtime `position_ids`.

## 8. Preprocessing and input packing

Text-only graph inputs:

- `input_ids [B,S]` or `inputs_embeds [B,S,H]`, exactly one required.
- Optional `attention_mask`; model builds `causal_mask` internally.
- Optional `position_ids [B,S]`; otherwise generated from cache length.
- Optional `past_key_values` cache.

Tokenizer/generation metadata from official wrapper:

- BOS=1, EOS=2, PAD=11 in official config/generation config.
- Tokenizer has many extra special tokens, including `[IMG]` id 10. `mistral4` text source itself does not stitch image embeddings; that belongs to the `mistral3` multimodal wrapper.
- `logits_to_keep` can be an integer suffix count or tensor indices and should be modeled as a runtime logits-slicing optimization.

CPU/data-pipeline work:

- Tokenization, chat template/tool tokens, and multimodal image packing are outside this text decoder graph.
- For official multimodal parity, require separate Pixtral processor/projector audit before accepting `[IMG]` placeholders.

## 9. Graph rewrite / lowering opportunities

### Rewrite: last-token-only logits

Source pattern: `lm_head(hidden_states[:, slice_indices, :])` with `logits_to_keep=1`.

Replacement: gather/slice last hidden token before GEMM.

Preconditions:

- Inference-only logits, no loss.
- Caller requests only newest token or fixed suffix.
- Sequence slice is contiguous or expressible as bounded gather.

Shape equations:

- `[B,S,H] -> [B,K,H] -> [B,K,V]`, with `K << S`.

Failure cases:

- Tensor index `logits_to_keep` with arbitrary non-contiguous positions unless gather is admitted.

Parity test sketch:

- Compare full logits slice versus pre-sliced hidden-state LM head for random hidden states.

### Rewrite: packed expert gate/up GEMM

Source pattern: per selected expert `linear(x, gate_up_proj[e]).chunk(2)`.

Replacement: grouped/segmented expert GEMM producing packed `[gate, up]`, then fused SiLU multiply.

Preconditions:

- Expert weight layout `[E, 2*I, H]`.
- Split order exactly `[gate, up]`.
- Router top-k indices are stable for the input.

Failure cases:

- Unsupported FP8 scale metadata or unsorted top-k semantics that change accumulation order beyond tolerance.

Parity test sketch:

- Fixed router indices and weights, compare naive loop/index_add to grouped expert implementation.

### Rewrite: Q low-rank projection fusion

Source pattern: `q_b_proj(RMSNorm(q_a_proj(x)))`.

Replacement: keep as two GEMMs plus RMSNorm initially; later fuse q_a GEMM output with RMSNorm and q_b input staging.

Preconditions:

- `q_lora_rank` known.
- No bias except optional `q_a_proj` bias.

Failure cases:

- `q_lora_rank=None` direct Q path.

### Rewrite: KV-a split and KV-b projection canonicalization

Source pattern: `kv_a_proj_with_mqa(x)` then split `[kv_lora_rank, qk_rope_head_dim]`; only low-rank part feeds RMSNorm/KV-b.

Replacement: one GEMM with two logical outputs or a packed output view.

Preconditions:

- Split sizes exactly from config.
- Preserve K-rotary part before expansion and RoPE.

### Rewrite: interleaved RoPE as weight/layout transform

Source pattern: view/transpose/reshape before rotate-half when `rope_interleave=True`.

Replacement: either preserve source transform or pre-permute Q/K rotary projection weights so runtime RoPE uses standard layout.

Preconditions:

- Conversion/weight provenance proves the expected interleaved layout.
- Applies only to rotary sub-dim.

Failure cases:

- Mixed checkpoints with already converted weights and `rope_interleave=False`.

## 10. Kernel fusion candidates

Highest priority:

- RMSNorm: used before attention, before MoE/MLP, final norm, plus low-rank Q/KV norms.
- Causal attention with partial RoPE and unequal V width: dominant runtime path.
- MoE router + expert dispatch: top-k, scatter/mask, expert GEMM, weighted index-add dominate large model throughput.
- FP8 linear materialization/dequant provider: official weights are FP8 with scale metadata.
- Last-token-only LM head: prevents full `[B,S,V]` GEMM during decode.

Medium priority:

- SwiGLU dense/shared expert fusion.
- Q-a RMSNorm Q-b staging.
- KV-a split plus KV-b projection staging.
- RoPE + query scale fusion into attention pre-processing.

Lower priority:

- Generic sequence/token classification heads.
- Non-null sliding-window mask path until a representative checkpoint requires it.
- Training loss and gradient checkpointing.

## 11. Runtime staging plan

1. Parse `Mistral4Config` and reject unsupported combinations: `num_key_value_heads != num_attention_heads`, non-null sliding window, unsupported quantization, unsupported `q_lora_rank=None` until tested.
2. Load dense or dequantized weights for tiny text-only checkpoints; implement one dense block parity.
3. Implement prefill for dense MLP and MoE blocks with eager router semantics.
4. Implement decode with DynamicCache-equivalent K/V shapes and last-token logits.
5. Add optimized attention backend preserving partial/interleaved RoPE and query scale.
6. Add MoE grouped expert provider and stable routing tests.
7. Add official FP8 loading path: parse `activation_scale`/`weight_scale_inv`, dense fallback, then provider-backed FP8 GEMMs.
8. Compose with `mistral3` multimodal wrapper only after Pixtral/projector/image-token stitch audit is complete.

Initially stub/defer:

- Sequence/token classification heads.
- Multimodal processor/projector.
- Training loss.
- Flash/flex backend-specific kwargs beyond dense causal attention.

## 12. Parity and validation plan

- Unit parity for RMSNorm fp32/fp16/bf16: compare fp32 variance/cast-back behavior.
- RoPE parity for `rope_interleave=True/False`, including odd decode offsets and YaRN configs.
- Query scale parity across positions below, at, and above `original_max_position_embeddings`.
- Single attention layer parity with `v_head_dim == qk_head_dim` and `v_head_dim < qk_head_dim`.
- Cache parity: prefill full sequence versus prefill prefix + decode suffix.
- Router parity: group top-k, mask, expert top-k, normalization, routed scale.
- Expert parity: naive loop/index-add versus DinoML grouped implementation for fixed router outputs.
- One-block and N-block tiny checkpoint parity using `onnx-internal-testing/tiny-random-Mistral4ForCausalLM`.
- Production text submodule parity once weights can be loaded/dequantized: compare prefill logits and greedy decode token sequence against Transformers.

Suggested tolerances:

- fp32 dense: `rtol=1e-4`, `atol=1e-5`.
- bf16/fp16 attention and MLP: `rtol=2e-2`, `atol=2e-2` initially, tighten per kernel.
- FP8 dequant/provider: separate dense-dequant reference tolerance from end-to-end logits tolerance.

## 13. Performance probes

- Prefill throughput sweep: `B`, `S`, and context up to long YaRN ranges.
- Decode tokens/sec sweep: batch size and cache length.
- KV cache memory: per layer K `[B,heads,T,qk_head_dim]` plus V `[B,heads,T,v_head_dim]`.
- Attention backend comparison: eager/reference, SDPA-like, Flash-like with value padding.
- RoPE/query-scale overhead as a standalone pre-attention probe.
- MoE router time: softmax/top-k/group mask/gather separated from expert GEMMs.
- Expert load balance: selected tokens per expert and top-k distribution.
- Grouped expert GEMM throughput vs naive per-expert loop.
- Last-token LM head vs full-sequence LM head.
- FP8 load/dequant/GEMM provider comparison against dense bf16 fallback.

## 14. Skip/defer list

- Training, labels/loss, gradient checkpointing.
- Sequence classification and token classification heads.
- Official multimodal image path, `[IMG]` embedding stitch, Pixtral vision tower, and projector.
- Non-null sliding window until source/config combination is validated.
- `num_key_value_heads < num_attention_heads` until the source shape contract is clarified.
- Beam search/controller policies beyond cache reorder and greedy decode.
- Remote-code-only behavior in tiny community checkpoints.
- NVFP4/compressed-tensors variant loading.
- Tensor parallel/distributed plans.

## 15. Final implementation checklist

- [ ] Parse `Mistral4Config` including `rope_parameters` and post-init `qk_head_dim`.
- [ ] Add admission guards for `num_key_value_heads == num_attention_heads`, supported `v_head_dim`, `sliding_window=None`, and supported quantization.
- [ ] Load dense text-only weights and preserve tied/untied LM head contract.
- [ ] Implement RMSNorm fp32 variance.
- [ ] Implement partial RoPE with `rope_interleave` and YaRN config.
- [ ] Implement Mistral/Llama-4 query scale.
- [ ] Implement low-rank Q and KV attention projections.
- [ ] Implement causal prefill attention and decode KV cache.
- [ ] Implement dense SwiGLU MLP.
- [ ] Implement MoE router and shared experts.
- [ ] Implement routed expert packed `[gate, up]` GEMM and weighted accumulation.
- [ ] Add last-token-only logits lowering.
- [ ] Add tiny checkpoint one-block and full-model parity.
- [ ] Add production text submodule parity with dense/dequantized weights.
- [ ] Add FP8 scale metadata loading and dense fallback.
- [ ] Benchmark prefill, decode, MoE routing, expert GEMM, and LM head separately.
