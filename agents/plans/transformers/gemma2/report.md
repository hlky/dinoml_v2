# Gemma2 Transformers Audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: gemma2 family; primary target Gemma2ForCausalLM text generation
Config source: local Transformers source defaults/conversion script, public tiny-random config, and open Unsloth mirror configs
Source files inspected:
- X:/H/transformers/src/transformers/models/gemma2/configuration_gemma2.py
- X:/H/transformers/src/transformers/models/gemma2/modeling_gemma2.py
- X:/H/transformers/src/transformers/models/gemma2/modular_gemma2.py
- X:/H/transformers/src/transformers/models/gemma2/convert_gemma2_weights_to_hf.py
- X:/H/transformers/src/transformers/masking_utils.py
- X:/H/transformers/src/transformers/cache_utils.py
- X:/H/transformers/src/transformers/modeling_flash_attention_utils.py
- comparison files: gemma/configuration_gemma.py, gemma/modeling_gemma.py, gemma3/configuration_gemma3.py, gemma3/modeling_gemma3.py
Any missing files or assumptions: official google/gemma-2-* config.json fetches returned 401 Unauthorized. Open mirror configs are labeled as mirrors, not primary Google config authority.
```

`modeling_gemma2.py` and `configuration_gemma2.py` are generated from `modular_gemma2.py`; future source edits should target the modular file. Runtime claims below use the generated file where line-level behavior matters. Small snapshots are in `_sources/`.

## 2. High-level architecture

Gemma2 is a text-only decoder-only causal language model.

```text
token ids / input embeddings -> scaled token embedding -> alternating local/full decoder blocks
-> final RMSNorm -> tied LM head -> optional final logit softcap -> logits/sampling
```

Primary runtime target: `Gemma2ForCausalLM` prefill and autoregressive decode. `Gemma2Model` is required as the base. Sequence and token classification heads exist through generic wrappers and can be deferred for the generation target.

Independently stageable pieces:

- tokenizer and prompt construction: CPU/data pipeline.
- prefill: embeddings, RoPE, hybrid sliding/full attention masks, all decoder blocks, logits.
- decode: same block graph with hybrid KV cache updates.
- logits-only optimization: `logits_to_keep` allows last-token-only or indexed logits.

No vision/audio/projector preprocessing is part of this family.

## 3. Important config dimensions

Source defaults in `Gemma2Config` correspond to the 2B-style shape. The conversion script carries explicit 9B and 27B shapes. Official Google configs were gated in this audit.

| Field | 2B/default source | 9B conversion source | 27B conversion source | Operator impact |
| --- | ---: | ---: | ---: | --- |
| `vocab_size` | 256000 | inherited default | inherited default | embedding and LM head rows |
| `hidden_size` | 2304 | 3584 | 4608 | residual width, GEMM MLP/proj width |
| `num_hidden_layers` | 26 | 42 | 46 | decoder block count |
| `num_attention_heads` | 8 | 16 | 32 | Q/O projection shape |
| `num_key_value_heads` | 4 | 8 | 16 | GQA cache/projection shape |
| `head_dim` | 256 | 256 | 128 | RoPE and attention inner dimension |
| `intermediate_size` | 9216 | 14336 | 36864 | gated MLP GEMMs |
| `max_position_embeddings` | 8192 | inherited default | inherited default | default RoPE/cache admission |
| `rope_theta` | 10000 effective default in configs | inherited/default | inherited/default | RoPE frequency base |
| `sliding_window` | 4096 | 4096 | 4096 | local attention and sliding cache |
| `layer_types` | alternating sliding/full | alternating by config default | alternating by config default | hybrid mask/cache |
| `query_pre_attn_scalar` | 256 | 224 | 144 | attention score scale = scalar^-0.5 |
| `attn_logit_softcapping` | 50.0 | 50.0 | 50.0 | tanh on attention logits before mask |
| `final_logit_softcapping` | 30.0 | 30.0 | 30.0 | tanh on final vocabulary logits |
| `attention_bias` | false | default false | default false | Q/K/V/O are bias-free |
| `hidden_activation` | `gelu_pytorch_tanh` | default | default | gated MLP activation |
| `use_cache` | true | default | default | generation KV cache |

Representative checkpoint/config sweep:

| Source | Config provenance | Shape notes | Variation notes |
| --- | --- | --- | --- |
| `hf-internal-testing/tiny-random-Gemma2ForCausalLM` | public config snapshot | 1 layer, hidden 32, 2 Q heads, 2 KV heads, head_dim 16 | useful for parser and graph smoke tests; no real GQA |
| `google/gemma-2-2b` | official repo gated; source default + Unsloth mirror snapshot | 26 layers, hidden 2304, 8 Q heads, 4 KV heads, head_dim 256 | GQA, alternating local/full layers |
| `google/gemma-2-9b` | official repo gated; conversion script + Unsloth mirror snapshot | 42 layers, hidden 3584, 16 Q heads, 8 KV heads, head_dim 256 | conversion script says `query_pre_attn_scalar=224`; mirror snapshot says 256, so DinoML should prefer actual loaded config |
| `google/gemma-2-27b` | official repo gated; conversion script + Unsloth mirror snapshot | 46 layers, hidden 4608, 32 Q heads, 16 KV heads, head_dim 128 | larger MLP ratio, head_dim differs from 2B/9B |

## 3a. Family variation traps

- `hidden_size != num_attention_heads * head_dim` for 9B and 27B. Q projection output is `num_attention_heads * head_dim`; O projection input is the same, not necessarily `hidden_size`.
- GQA is required: `num_key_value_heads < num_attention_heads` in production shapes. KV cache stores KV heads before repeat expansion.
- Layer schedule alternates by default: 1-based odd layers are `sliding_attention`, 1-based even layers are `full_attention`.
- Sliding layers use `sliding_window` for masks and a sliding cache class; full layers retain full prefix cache.
- Attention score scale is `query_pre_attn_scalar ** -0.5`, not always `head_dim ** -0.5`.
- Attention logits softcap happens before adding the mask in eager attention: divide by softcap, `tanh`, multiply back.
- Final vocabulary logits also get optional tanh softcapping.
- RMSNorm parameter is centered as `(1 + weight)`, with fp32 normalization/math then cast back.
- Gemma2 block norm placement differs from Gemma: it has post-attention and post-feedforward RMSNorms in addition to pre-attention/pre-MLP norms.
- Gemma3 text is a related but distinct target: it has Q/K head RMSNorm, separate local/global RoPE parameters, default softcapping disabled in config, a 6-layer full-attention pattern by default, larger max positions, and multimodal Gemma3 variants.
- No NHWC/channel-last rewrite is relevant for Gemma2 text-only runtime. Protect token axis semantics for `view`, `transpose(1, 2)`, softmax `dim=-1`, and logits slicing.

## 4. Operator coverage checklist

Tensor/layout ops:

- token embedding lookup, `input_ids -> [B, S, H]`.
- embedding scale multiply by `sqrt(hidden_size)`.
- `view(..., heads, head_dim)`, `transpose(1, 2)`, `reshape`, `contiguous`.
- residual adds and dtype casts.
- last-token or indexed logits slicing from `hidden_states[:, slice_indices, :]`.

Neural network primitives:

- RMSNorm over last dim with fp32 accumulation and `(1 + weight)` scale.
- bias-free Linear projections:
  - 2B/default: Q `2304 -> 2048`, K/V `2304 -> 1024`, O `2048 -> 2304`, MLP gate/up `2304 -> 9216`, down `9216 -> 2304`.
  - 9B conversion: Q `3584 -> 4096`, K/V `3584 -> 2048`, O `4096 -> 3584`, MLP gate/up `3584 -> 14336`, down `14336 -> 3584`.
  - 27B conversion: Q `4608 -> 4096`, K/V `4608 -> 2048`, O `4096 -> 4608`, MLP gate/up `4608 -> 36864`, down `36864 -> 4608`.
- gated MLP: `down(gelu_pytorch_tanh(gate(x)) * up(x))`.
- tied token embedding / LM head weight alias.

Attention primitives:

- causal self-attention, no cross-attention.
- hybrid sliding/full attention by layer.
- GQA repeat KV from `[B, KVH, K, D]` to `[B, QH, K, D]` for eager math, or native GQA support in optimized kernels.
- softmax over key axis with fp32 softmax dtype in eager mode.
- attention dropout is runtime zero in inference.
- SDPA, FlashAttention, and flex attention are source-supported backends; eager fallback is required for parity tests.

Position/rotary ops:

- default RoPE over `head_dim` with base `rope_theta`.
- cos/sin computed in fp32 from `[B, S]` position ids and cast to hidden dtype.
- rotate-half split is first half / second half, not interleaved even/odd rotation.

Generation/cache ops:

- dynamic and static hybrid caches.
- per-layer cache class follows `config.layer_types`.
- sliding dynamic cache stores only the last `sliding_window - 1` previous tokens but returns full key/value states for the current attention call.
- cache reorder/select for beam/batch operations can be inherited from general KV-cache support later.

Distributed/tensor-parallel hints:

- config includes TP plan: Q/K/V/gate/up colwise, O/down rowwise, LM head colwise gather. Multi-GPU can be deferred.

## 5. Layer/block breakdown

Gemma2 base model:

```text
inputs_embeds = embedding(input_ids) * sqrt(hidden_size)
position_ids = arange(S) + past_seen_tokens if omitted
mask_map = {
  "full_attention": causal mask,
  "sliding_attention": sliding-window causal mask,
}
position_embeddings = rotary_emb(inputs_embeds, position_ids)
for layer i:
  hidden = decoder_layer(hidden, mask_map[layer_types[i]], position_embeddings, cache)
hidden = final RMSNorm(hidden)
```

Decoder block, repeated `num_hidden_layers`:

```text
residual = x
x = input_layernorm(x)
q = Linear(H -> QH * D, bias=False)(x).view(B,S,QH,D).transpose(1,2)
k = Linear(H -> KVH * D, bias=False)(x).view(B,S,KVH,D).transpose(1,2)
v = Linear(H -> KVH * D, bias=False)(x).view(B,S,KVH,D).transpose(1,2)
q,k = RoPE(q,k, cos, sin)
k,v = cache.update(k,v, layer_idx) if cache is present
attn = attention(q,k,v, mask, scale=query_pre_attn_scalar^-0.5, softcap=attn_logit_softcapping, sliding_window=maybe)
x = Linear(QH * D -> H, bias=False)(attn.reshape(B,S,QH*D))
x = post_attention_layernorm(x)
x = residual + x
residual = x
x = pre_feedforward_layernorm(x)
x = down_proj(gelu_pytorch_tanh(gate_proj(x)) * up_proj(x))
x = post_feedforward_layernorm(x)
x = residual + x
```

LM head:

```text
logits = tied_linear(hidden[:, logits_to_keep_or_all, :], embed_tokens.weight)
logits = tanh(logits / final_logit_softcapping) * final_logit_softcapping
```

## 6. Attention requirements

Gemma2 requires autoregressive self-attention only.

- Masking: causal for both full and sliding layers unless `use_bidirectional_attention` is set. First integration should reject or defer bidirectional Gemma2.
- Schedule: default `layer_types[i] = "sliding_attention"` when `(i + 1) % 2 != 0`, otherwise `"full_attention"`.
- Sliding window: source sliding overlay admits keys with `kv_idx > q_idx - sliding_window`, intersected with causal mask. With `sliding_window=4096`, each causal query can attend to itself and the prior 4095 positions in the local layer.
- GQA: Q heads / KV heads = 2 for production 2B, 9B, 27B. Eager repeats KV before matmul; optimized attention should avoid physical repeat when possible.
- Cache shapes before repeat:
  - per full layer: K/V `[B, num_key_value_heads, total_seen, head_dim]`.
  - per sliding layer dynamic storage: K/V `[B, num_key_value_heads, min(total_seen, sliding_window - 1), head_dim]`, while the update returns current full local attention K/V for the call.
  - static sliding cache allocates an effective window-sized backing store.
- Cached keys are stored after RoPE, because RoPE is applied before `past_key_values.update`.
- Eager math order:
  - `q @ k.transpose(-2,-1)`.
  - multiply by `query_pre_attn_scalar ** -0.5`.
  - apply tanh softcap if configured.
  - add mask.
  - softmax with fp32 dtype, cast to query dtype.
  - matmul by V.
- FlashAttention path can pass `window_size=(sliding_window - 1, sliding_window - 1)` when the backend supports windowing and key length exceeds the window; it can pass `softcap` when supported.
- Packed/varlen support is inherited from generic attention utilities, not Gemma2-specific source code.

## 7. Position encoding and custom math

RoPE can be precomputed for static positions or computed per runtime `position_ids`. With cache, `position_ids` starts at `past_seen_tokens`.

```python
def gemma2_rope_cache(position_ids, head_dim, rope_theta, dtype):
    inv = 1.0 / (rope_theta ** (arange(0, head_dim, 2, fp32) / head_dim))
    freqs = matmul(inv[None, :, None], position_ids[:, None, :].float()).transpose(1, 2)
    emb = concat([freqs, freqs], dim=-1)
    return cos(emb).to(dtype), sin(emb).to(dtype)

def gemma2_apply_rope(q, k, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    def rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return concat([-x2, x1], dim=-1)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin
```

Softcapping parity snippets:

```python
attn_scores = tanh((qk * scale) / attn_softcap) * attn_softcap
logits = tanh(logits / final_softcap) * final_softcap
```

RMSNorm parity:

```python
y = x.float() * rsqrt(mean(x.float() ** 2, dim=-1, keepdim=True) + eps)
y = y * (1.0 + weight.float())
return y.to(x.dtype)
```

## 8. Preprocessing and input packing

CPU/data pipeline:

- tokenizer emits `input_ids` and optional 2D `attention_mask`.
- Gemma tokenizer convention uses BOS/EOS/PAD ids from config: pad 0, eos 1, bos 2.
- Generation may prepare masks and position ids before model forward.

GPU/runtime graph:

- accepts exactly one of `input_ids` or `inputs_embeds`.
- `position_ids` are generated as `[0..S-1] + past_seen_tokens` if omitted.
- if `attention_mask` is already a dict, it is assumed to contain `"full_attention"` and `"sliding_attention"` masks; otherwise the model creates both.
- no multimodal placeholder scatter, cu_seqlens input, image/audio metadata, or token type ids are Gemma2-specific requirements.

## 9. Graph rewrite / lowering opportunities

### Rewrite: Q/K/V projections to fused packed GEMM

Source pattern:

```text
q = Linear(H -> QH*D)(x)
k = Linear(H -> KVH*D)(x)
v = Linear(H -> KVH*D)(x)
```

Replacement: one GEMM from `H -> (QH + 2*KVH) * D`, then split rows into Q, K, V.

Preconditions:

- all three projections are bias-free.
- same input tensor, same dtype, same batch/sequence flattening.
- packed weight layout is explicitly generated as `[q_rows, k_rows, v_rows]` to match HF conversion split order.
- no tensor-parallel sharding boundary between projections.

Failure cases: config or checkpoint uses separate quantization/offload policies per projection; Q/K/V are independently sharded; attention bias becomes true.

Parity test sketch: compare separate HF projections versus fused packed GEMM + split for 2B, 9B, and 27B shapes.

### Rewrite: GQA attention without materialized repeat

Source pattern: eager `repeat_kv(k, q_heads // kv_heads)` before attention.

Replacement: attention kernel consumes Q heads and KV heads with group mapping `q_head // groups_per_kv`.

Preconditions:

- `num_attention_heads % num_key_value_heads == 0`.
- no caller requests dense repeated K/V tensors as outputs.
- cache stores unexpanded KV heads.

Shape equations: Q `[B,QH,Sq,D]`, K/V `[B,KVH,Sk,D]`, output `[B,QH,Sq,D]`.

Failure cases: unsupported backend, attention tensor output requested in exact eager repeated-head layout.

Parity test sketch: eager repeated KV versus grouped kernel for full and sliding masks.

### Rewrite: last-token-only logits

Source pattern: `lm_head(hidden_states[:, slice_indices, :])` with `logits_to_keep=1`.

Replacement: slice hidden state before GEMM, compute only `[B,1,V]`.

Preconditions:

- generation path only needs next-token logits.
- no loss computation over full sequence.
- `logits_to_keep` is positive integer 1 or a known suffix length.

Failure cases: full prefill logits requested, labels present, arbitrary tensor index requiring gather.

Parity test sketch: compare full logits suffix to sliced-GEMM logits before and after final softcap.

### Rewrite: RoPE table precompute

Source pattern: compute cos/sin from `position_ids` each forward.

Replacement: precompute cos/sin for static max position or decode positions, gather by `position_ids`.

Preconditions:

- RoPE type is default with fixed `rope_theta`.
- no dynamic RoPE scaling update is active.
- dtype conversion matches source, with fp32 trigonometry and final cast.

Failure cases: non-default `rope_parameters`, dynamic scaling, extremely long context beyond precomputed table.

Parity test sketch: generated table lookup versus source computation over prefill and decode offsets.

## 10. Kernel fusion candidates

Highest priority:

- RMSNorm: appears four times per block plus final norm; must preserve `(1 + weight)` and fp32 accumulation.
- GQA FlashAttention/SDPA with hybrid sliding/full masks, RoPE-applied cache, and attention softcap.
- bias-free GEMM backbone for Q/K/V/O, MLP gate/up/down, and LM head.
- SwiGLU-style gated MLP fusion: `gelu_pytorch_tanh(gate) * up` plus down projection.
- last-token-only logits with final softcap.

Medium priority:

- fused QKV projection plus RoPE layout transform.
- residual add + post-attention/post-MLP RMSNorm fusion.
- RoPE precompute/gather and apply fusion.
- hybrid cache update kernels for sliding layers.

Lower priority:

- classification heads.
- attention output tensors for debugging.
- tensor-parallel plans.
- bidirectional attention mode.

## 11. Runtime staging plan

Stage 1: parse config and load weights for tiny random and 2B-style shape; reject unsupported non-default flags explicitly.

Stage 2: one-block eager parity with RMSNorm, RoPE, GQA eager attention, MLP, and post-norm placement.

Stage 3: full prefill parity with alternating mask map, final norm, tied LM head, and final logit softcap.

Stage 4: dynamic decode cache parity, including both full and sliding cache layers.

Stage 5: optimized attention backend for full and sliding Gemma2 layers with GQA and softcap.

Stage 6: graph rewrites/fusions: fused QKV, RoPE table, MLP fusion, last-token logits.

Stage 7: optional static/hybrid cache export style and production batching.

Initially stub/defer: training loss, classification heads, tensor parallelism, beam-cache reorder beyond generic cache support, and non-default RoPE variants.

## 12. Parity and validation plan

- config parsing tests for source default, conversion 9B/27B snapshots, tiny random config, and mirror snapshots.
- RMSNorm random tensor parity over fp32/fp16/bf16 with `(1 + weight)` and eps.
- RoPE parity for prefill and decode offsets; include `head_dim=256` and `head_dim=128`.
- attention score parity for full and sliding masks with softcap on/off.
- cache update parity: full layers grow to total seen; sliding dynamic layers retain `sliding_window - 1` previous tokens and return full call-local K/V.
- single decoder block parity with fixed random weights.
- after-N-layer parity on tiny random config.
- prefill logits parity with final softcap.
- decode token parity over a few greedy steps.
- last-token-only logits parity against full logits suffix.

Suggested tolerances: fp32 `rtol=1e-4, atol=1e-5`; fp16/bf16 block/logit parity should use looser tolerances such as `rtol=2e-2, atol=2e-2`, with attention backend-specific thresholds recorded.

## 13. Performance probes

- prefill throughput by sequence length: 512, 2048, 4096, 8192.
- decode tokens/sec with hybrid cache at batch 1, 4, 16.
- sliding versus full attention layer timing at equal shapes.
- KV cache memory split: full layers versus sliding layers.
- attention backend comparison: eager, SDPA, FlashAttention, DinoML optimized.
- QKV separate versus fused packed projection.
- MLP gate/up/down timing and gated activation bandwidth.
- LM head full logits versus last-token-only logits.
- 2B/9B/27B shape sweep to expose head_dim and projection-width differences.

## 14. Skip/defer list

- training, labels/loss, gradient checkpointing.
- sequence and token classification heads for first generation integration.
- bidirectional attention mode.
- non-default or dynamic RoPE variants unless a target config requires them.
- tensor parallel and pipeline parallel execution.
- quantized weights as model-family behavior; treat quantization through DinoML weight-loading/GGUF plans separately.
- FlashAttention-specific packed sequence paths until dense prefill/decode parity is stable.
- speculative decoding, beam search, and continuous batching-specific cache managers.

## 15. Final implementation checklist

- [ ] Parse `Gemma2Config`, including `head_dim`, `query_pre_attn_scalar`, softcaps, `sliding_window`, and `layer_types`.
- [ ] Load tied embedding/LM-head weights with alias preservation.
- [ ] Implement Gemma RMSNorm with fp32 accumulation and `(1 + weight)` scale.
- [ ] Implement default RoPE with first-half/second-half rotation.
- [ ] Implement bias-free Linear/GEMM shapes where `QH * head_dim` may differ from `hidden_size`.
- [ ] Implement GQA attention without requiring materialized KV repeat in optimized path.
- [ ] Implement attention tanh softcap before mask addition.
- [ ] Implement alternating full/sliding causal masks.
- [ ] Implement hybrid KV cache with sliding layers retaining `sliding_window - 1` previous tokens.
- [ ] Implement Gemma2 decoder block with post-attention and post-feedforward RMSNorms.
- [ ] Implement gated MLP `gelu_pytorch_tanh(gate) * up`.
- [ ] Implement final RMSNorm, tied LM head, `logits_to_keep`, and final logit softcap.
- [ ] Add tiny-random single-block and full-model parity tests.
- [ ] Add production-shape config parser tests for 2B/9B/27B.
- [ ] Benchmark prefill, decode, cache memory, attention backend, QKV fusion, and last-token logits.
