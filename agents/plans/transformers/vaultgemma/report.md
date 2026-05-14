# VaultGemma Transformers Audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: google/vaultgemma-1b
Config source: official HF repo metadata + gated config URL; open mirrors listed below
Source files inspected:
- X:/H/transformers/src/transformers/models/vaultgemma/configuration_vaultgemma.py
- X:/H/transformers/src/transformers/models/vaultgemma/modeling_vaultgemma.py
- X:/H/transformers/src/transformers/models/vaultgemma/modular_vaultgemma.py
- X:/H/transformers/tests/models/vaultgemma/test_modeling_vaultgemma.py
- X:/H/transformers/docs/source/en/model_doc/vaultgemma.md
- X:/H/transformers/src/transformers/masking_utils.py
- X:/H/transformers/src/transformers/integrations/{sdpa_attention.py,flash_attention.py,flex_attention.py}
Any missing files or assumptions:
- https://huggingface.co/google/vaultgemma-1b/raw/main/config.json returned 401 Unauthorized; license access is required to inspect the official raw config.
- HF API metadata for google/vaultgemma-1b was visible: gated="manual", model sha f9624dafc1760cb2f6039e86e12055d6559d7abb, BF16 safetensors parameter count 1,038,741,120.
- Open mirrors inspected for config shape: onnx-community/vaultgemma-1b-ONNX and OpenKing/vualtgemma-1b-non-gated. These are mirrors/derivatives, not authoritative Google raw config.
```

`modeling_vaultgemma.py` and `configuration_vaultgemma.py` are generated from `modular_vaultgemma.py`; future source edits should target the modular file, but runtime behavior was audited from the generated modeling file.

Primary runtime target for this report: `VaultGemmaForCausalLM` text-only decoder inference, including prefill and autoregressive decode with KV cache. `VaultGemmaModel` is required as the body. Training loss, gradient checkpointing, tensor parallel helpers, and generic generation policy are deferred.

## 2. High-level architecture

VaultGemma is a text-only decoder-only causal LM derived from Gemma 2. The implementation removes Gemma 2's post-attention and post-FFN norms, keeps pre-attention and pre-FFN RMSNorms, uses RoPE, and supports full or sliding-window causal attention per layer through `config.layer_types`.

```text
token ids / embeds -> scaled token embedding -> N decoder blocks -> final RMSNorm
  -> tied/biasless LM head -> optional final logits tanh softcap -> logits/sampling
```

Stage decomposition:

- CPU/data pipeline: tokenizer emits `input_ids`, optional `attention_mask`, optional caller-supplied `position_ids`.
- Runtime prefill: embedding lookup scaled by `sqrt(hidden_size)`, mask construction, RoPE table for the prefill positions, all decoder layers, logits for selected positions.
- Runtime decode: one or more new tokens, position offset from cache length, per-layer KV update, attention over full or local cache depending on layer type.
- Cacheable state: per-layer self-attention K/V tensors after RoPE. No encoder/projector branch and no multimodal embedding stitch.

## 3. Important config dimensions

Source defaults from `VaultGemmaConfig` differ from the released 1B mirror configs. DinoML should not infer projection widths from `hidden_size` alone.

| Field | Source default | 1B mirror configs | Operator impact |
|---|---:|---:|---|
| `vocab_size` | 256000 | 256000 | Embedding and LM head rows |
| `hidden_size` | 2304 | 1152 | Residual stream width |
| `num_hidden_layers` | 26 | 26 | Decoder block repeat count |
| `num_attention_heads` | 8 | 4 | Query/output head count |
| `num_key_value_heads` | 4 | 4 | Source default is GQA; 1B mirror is MHA |
| `head_dim` | 256 | 256 | Explicit head width |
| Q projection width | 2048 | 1024 | `num_attention_heads * head_dim`, not `hidden_size` |
| K/V projection width | 1024 | 1024 | `num_key_value_heads * head_dim` |
| O projection | 2048 -> 2304 | 1024 -> 1152 | Attention output width differs from residual width |
| `intermediate_size` | 9216 | 6912 | MLP gate/up/down GEMMs |
| `hidden_activation` | `gelu_pytorch_tanh` | `gelu_pytorch_tanh` | Gated GELU MLP |
| `max_position_embeddings` | 8192 | 1024 | RoPE/cache admission |
| `rope_theta` / `rope_parameters` | default RoPE expected | `rope_theta: 10000.0` legacy field | Must normalize to RoPE parameters |
| `attention_bias` | false | false | Q/K/V/O projections are biasless |
| `rms_norm_eps` | 1e-6 | 1e-6 | RMSNorm epsilon |
| `query_pre_attn_scalar` | 256 | 256 | Attention scale is `1/sqrt(256)`, not `1/sqrt(head_dim)` by accident |
| `sliding_window` | 4096 | 512 | Only active for layers marked `sliding_attention` |
| `layer_types` | alternating sliding/full starting layer 0 sliding | all `full_attention` in inspected mirrors | Attention/cache structure changes |
| `attn_logit_softcapping` | 50.0 | null | Softcap required only if non-null |
| `final_logit_softcapping` | 30.0 | null | Logit softcap required only if non-null |
| `dtype` | unspecified | bfloat16 | Weight dtype from config/mirror metadata |
| `tie_word_embeddings` | true | omitted in mirrors, source default true | LM head aliases embedding weight |

Representative checkpoint/config sweep:

| Config source | Status | Dimensions | Attention | Softcaps | Notes |
|---|---|---|---|---|---|
| `google/vaultgemma-1b` HF API metadata | Official but gated raw files | `model_type=vaultgemma`, `VaultGemmaForCausalLM`, BF16 params 1,038,741,120 | Raw `config.json` unavailable without license | Unknown from raw config | Requires Gemma license acknowledgement |
| `OpenKing/vualtgemma-1b-non-gated` raw config | Open mirror | H=1152, L=26, heads=4, KV=4, head_dim=256, I=6912, max_pos=1024 | all 26 layers `full_attention`, `sliding_window=512` ignored by layer types | null/null | Same parameter count as official metadata; mirror, not authoritative |
| `onnx-community/vaultgemma-1b-ONNX` raw config | Open ONNX derivative | Same as above | Same all-full layer list | null/null | Adds Transformers.js ONNX external-data metadata and quantized ONNX variants |
| `VaultGemmaConfig()` source defaults | Source-defined synthetic config | H=2304, L=26, heads=8, KV=4, head_dim=256, I=9216, max_pos=8192 | alternating `sliding_attention`, `full_attention` | 50/30 | Useful for admission traps, not the released 1B shape |

## 3a. Family variation traps

- `hidden_size != num_attention_heads * head_dim` in both source defaults and 1B mirrors. Q/O operate on the attention width, not residual width.
- GQA is config-dependent. Source default has 8 Q heads and 4 KV heads; 1B mirror has 4 Q heads and 4 KV heads.
- `layer_types` controls whether `sliding_window` is semantically used. The 1B mirror configs list all layers as full attention even though `sliding_window=512` is present.
- If `layer_types` is omitted, source `__post_init__` creates alternating sliding/full layers starting with sliding layer 0.
- `attn_logit_softcapping` and `final_logit_softcapping` are null in the inspected mirrors but non-null in source defaults. Do not lower source-default synthetic configs through an attention backend that ignores softcap.
- `sdpa_attention_forward` accepts `**kwargs` but does not apply `softcap`; eager, flash, and flex paths do apply or forward softcap. Admission should reject non-null attention softcap with SDPA unless source changes.
- Config mirrors use legacy `rope_theta` instead of `rope_parameters`; the runtime model expects `config.rope_parameters["rope_type"]` and `["rope_theta"]` after config normalization.
- RMSNorm parameter is initialized/stored as an offset and applied as `(1.0 + weight)`, not a plain multiplicative gamma initialized to ones.
- Embedding output is multiplied by `sqrt(hidden_size)` using a persistent false buffer. This scale is part of inference parity.
- `lm_head.weight` is tied to `model.embed_tokens.weight`; keep one logical parameter alias.
- Docs state the pretrained 1B uses 1024 token sequence length and full attention for all layers. Tests contain a copied/stale-looking comment expecting hybrid cache by default; source and inspected mirror configs should win for admission.
- No NCHW/NHWC issue for the model body: all runtime tensors are token sequences `[B, T, C]` and attention tensors. Layout translation passes should mark this family as no image/video layout work.

## 4. Operator coverage checklist

Tensor/layout ops:

- Integer token input `[B, T]`; optional `inputs_embeds [B, T, H]`.
- Embedding lookup `[vocab_size, H] -> [B, T, H]`, then scalar multiply by `sqrt(H)`.
- `arange`, add cache offset, `unsqueeze(0)` for default `position_ids [1, T]`.
- `view(..., heads, head_dim)`, `transpose(1, 2)`, `contiguous`, `reshape`.
- Slice last logits positions via `logits_to_keep`: int suffix slice or tensor indices.
- Tied weight alias: `lm_head.weight == embed_tokens.weight`.

Neural primitives:

- Biasless Linear for 1B mirror:
  - Q: `Linear(1152 -> 1024)`
  - K: `Linear(1152 -> 1024)`
  - V: `Linear(1152 -> 1024)`
  - O: `Linear(1024 -> 1152)`
  - MLP gate: `Linear(1152 -> 6912)`
  - MLP up: `Linear(1152 -> 6912)`
  - MLP down: `Linear(6912 -> 1152)`
  - LM head: `Linear(1152 -> 256000)`, biasless, tied to embedding.
- Source-default shape variants:
  - Q `2304 -> 2048`, K/V `2304 -> 1024`, O `2048 -> 2304`, MLP `2304 -> 9216 -> 2304`.
- RMSNorm over last dim with fp32 accumulation and `(1 + weight)` multiply.
- GELU tanh approximation for gated MLP: `down(gelu_tanh(gate(x)) * up(x))`.
- Residual adds after attention and MLP only; no post-attention/post-FFN norms.

Attention primitives:

- Causal self-attention.
- MHA for 1B mirrors; GQA for source defaults.
- Optional KV repeat path from `[B, KVH, S, D]` to `[B, QH, S, D]`.
- Dense causal mask or sliding-window causal mask depending on `layer_types`.
- Eager math: `softmax((QK^T * scale) + mask)` with optional pre-mask tanh score softcap.
- Dropout is present in source but `attention_dropout=0.0`; inference can compile it away.

Position/rotary/custom math:

- Default RoPE with `inv_freq = 1 / theta ** (arange(0, head_dim, 2) / head_dim)`.
- Cos/sin computed in fp32 then cast to hidden dtype.
- RoPE applied to Q and K before cache update.

Generation/cache ops:

- `DynamicCache(config)` when `use_cache=True` and no cache is passed.
- Per-layer update after RoPE.
- Cache tensor shape before repeat: `[B, num_key_value_heads, S_cache, head_dim]`.
- 1B mirror cache per full layer: K/V `[B, 4, S, 256]`.
- Source default GQA cache per layer: K/V `[B, 4, S, 256]`; Q expands to 8 heads during attention.
- Sliding layers, if admitted, use `DynamicSlidingWindowLayer` and store only roughly the most recent `sliding_window - 1` cached tokens in dynamic tests.

Preprocessing-coupled ops:

- Tokenizer produces `input_ids` and optional `attention_mask`. No processor, image/audio/video preprocessing, packed patch metadata, or scatter stitch.

Quantized/packed weight metadata ops:

- No source-owned quantized weight format in Transformers Python. ONNX mirror has quantized ONNX files, but that is a derivative export artifact, not a native VaultGemma modeling requirement.

Distributed/tensor-parallel ops:

- Source declares TP plans for projections and LM head, but first DinoML target can ignore multi-GPU/tensor parallel and load dense single-rank weights.

## 5. Layer/block breakdown

For 1B mirror config, repeated 26 times:

```text
x: [B, T, 1152]
residual = x
x = RMSNorm(x)                                      # last dim 1152, fp32 accumulator
q = Linear(1152 -> 1024)(x).view(B,T,4,256).T       # [B,4,T,256]
k = Linear(1152 -> 1024)(x).view(B,T,4,256).T       # [B,4,T,256]
v = Linear(1152 -> 1024)(x).view(B,T,4,256).T       # [B,4,T,256]
q,k = RoPE(q,k, cos[position_ids], sin[position_ids])
k,v = cache.update(k,v,layer_idx) if cache enabled
a = causal_attention(q,k,v, mask, scale=1/sqrt(256), optional_softcap)
a = a.transpose(1,2).reshape(B,T,1024)
x = residual + Linear(1024 -> 1152)(a)
residual = x
x = RMSNorm(x)
x = Linear(6912 -> 1152)(gelu_tanh(Linear(1152 -> 6912)(x)) * Linear(1152 -> 6912)(x))
x = residual + x
```

After the last block:

```text
x = RMSNorm(x)
logits = tied_lm_head(x[:, selected_positions, :])  # [B, T_keep, 256000]
if final_logit_softcapping is not None:
    logits = softcap * tanh(logits / softcap)
```

All projections are biasless for inspected configs. If `attention_bias=True` appears in a future checkpoint, Q/K/V/O need bias support; MLP and LM head remain biasless in source.

## 6. Attention requirements

Required for 1B mirror parity:

- Causal self-attention only; no cross-attention.
- MHA: Q heads = 4, KV heads = 4, head_dim = 256.
- Query/key/value width = 1024; output projection maps 1024 back to residual width 1152.
- Full attention for all layers in inspected mirror configs.
- Attention mask: optional padding mask plus causal mask. Eager mask is additive 4D with `-inf` for invalid entries; SDPA/flash/flex use their backend-specific mask forms.
- KV cache stores RoPE-applied K and raw V, before KV repeat.
- FlashAttention compatibility: source declares support and flash wrapper forwards `sliding_window` and `softcap`.
- SDPA compatibility: safe for inspected mirror configs where `attn_logit_softcapping=null`; unsafe for non-null softcap without an explicit rewrite/fallback.
- FlexAttention compatibility: source supports it, but tests skip a large-model flex path due to a Triton resource error. Treat flex as optional optimization.

Config-dependent optional attention:

- Source-default synthetic config has GQA with Q heads = 8, KV heads = 4, `num_key_value_groups=2`.
- Source default alternates sliding and full layers. Sliding mask is causal with local condition `kv_idx > q_idx - sliding_window`; cache storage uses sliding-window layer classes.
- If sliding layers are admitted, DinoML needs a hybrid cache manifest by layer type and should reject missing/inconsistent `layer_types`.

## 7. Position encoding and custom math

RoPE is standard full-head RoPE over `head_dim`, with cosine and sine duplicated over the two half-dim blocks:

```python
def vaultgemma_rope(q, k, position_ids, theta=10000.0, head_dim=256):
    inv = 1.0 / (theta ** (arange(0, head_dim, 2, fp32) / head_dim))
    freqs = position_ids[:, :, None].fp32() * inv[None, None, :]
    emb = cat([freqs, freqs], dim=-1)
    cos, sin = cos(emb).to(q.dtype), sin(emb).to(q.dtype)
    def rotate_half(x):
        return cat([-x[..., head_dim // 2:], x[..., :head_dim // 2]], dim=-1)
    return q * cos[:, None] + rotate_half(q) * sin[:, None], \
           k * cos[:, None] + rotate_half(k) * sin[:, None]
```

Precompute opportunity: for static max context, precompute fp32 or target-dtype cos/sin tables up to admitted max length. Dynamic RoPE variants are wired through generic `ROPE_INIT_FUNCTIONS`, but no inspected 1B mirror uses scaling; reject non-default `rope_type` until separately validated.

Custom scalar math:

- Attention softcap, when non-null: `scores = softcap * tanh(scores / softcap)` before mask addition.
- Final logit softcap, when non-null: `logits = softcap * tanh(logits / softcap)`.
- RMSNorm uses fp32 inner math and `1 + weight` gamma.
- Embedding scale is `sqrt(hidden_size)`.

## 8. Preprocessing and input packing

GPU/runtime inputs:

- `input_ids [B, T]` int64/long, or exactly one alternative `inputs_embeds [B, T, H]`.
- `attention_mask [B, S]` optional, where `S` includes cached tokens plus query tokens for generation.
- `position_ids [B or 1, T]` optional. If omitted, source uses `arange(T) + past_seen_tokens`, then unsqueezes to `[1, T]`.
- `past_key_values` optional `Cache`.

CPU/data-pipeline work:

- SentencePiece tokenizer files are present in official/mirror repos (`tokenizer.model`, tokenizer configs). Tokenization and generation prompt policy are outside the neural graph.
- Special token IDs from inspected mirror configs: `pad=0`, `eos=1`, `bos=2`.

No multimodal placeholders, no masked scatter, no image/audio/video processors, no cu_seqlens-style packed sequence ABI in normal inference. `masking_utils` can detect packed sequence format from `position_ids` for training-like cases; first DinoML inference target can reject packed position patterns.

## 9. Graph rewrite / lowering opportunities

### Rewrite: explicit attention-width projections

Source pattern:

```text
Linear(H -> num_heads * head_dim) -> view(B,T,num_heads,head_dim) -> transpose(1,2)
```

Replacement:

```text
GEMM_RRR/BiaslessLinear -> reshape/metadata view -> attention layout
```

Preconditions:

- `attention_bias == false` for current fast path.
- Projection output size equals configured `num_*_heads * head_dim`, not `hidden_size`.
- Runtime validates `hidden_size`, `num_attention_heads`, `num_key_value_heads`, and `head_dim` from config/weights.

Failure cases:

- Do not assume output width equals input hidden width.
- Future `attention_bias=true` needs bias epilogue.

Parity test sketch: compare Q/K/V/O tensors for one layer against Transformers for 1B-shaped random weights.

### Rewrite: MHA/GQA repeat elimination

Source pattern:

```text
repeat_kv(k, n_rep); repeat_kv(v, n_rep); attention(q, repeated_k, repeated_v)
```

Replacement:

```text
Grouped-query attention kernel consuming KV heads directly
```

Preconditions:

- `num_attention_heads % num_key_value_heads == 0`.
- Kernel supports MHA as `n_rep=1` and GQA as `n_rep>1`.
- Cache stores unrepeated K/V.

Failure cases:

- Fallback to explicit repeat for early parity or unsupported backend.

### Rewrite: fused RoPE + attention prefill/decode

Source pattern:

```text
Q/K projections -> RoPE -> cache update -> attention
```

Replacement:

```text
Attention kernel with RoPE-applied Q/K or fused RoPE prologue
```

Preconditions:

- Default RoPE only.
- Position IDs are monotonic contiguous or supplied and validated.
- Cached keys are stored after RoPE, matching source.

Failure cases:

- Non-default RoPE scaling or packed position IDs require separate parity.

### Rewrite: RMSNorm

Source pattern:

```text
x.float() * rsqrt(mean(x.float() ** 2, dim=-1) + eps) * (1 + weight.float())
```

Replacement:

```text
Fused RMSNormOffsetGamma
```

Preconditions:

- Normalize last dimension.
- Accumulate in fp32.
- Gamma is `1 + stored_weight`, not stored weight directly.

### Rewrite: SwiGLU-like gated GELU MLP

Source pattern:

```text
down(gelu_tanh(gate(x)) * up(x))
```

Replacement:

```text
two GEMMs -> fused gelu_tanh/multiply -> GEMM
```

Preconditions:

- Activation exactly `gelu_pytorch_tanh`.
- Biasless gate/up/down for inspected configs.

### Rewrite: last-token-only logits

Source pattern:

```text
hidden_states[:, slice_indices, :] -> lm_head
```

Replacement:

```text
select positions before GEMM; for decode use only final row
```

Preconditions:

- `logits_to_keep` is int suffix or explicit index tensor supported by runtime.
- Preserve tied LM head weight.

### Layout rewrite notes

No NCHW/NHWC/NCDHW tensors exist in the model body. A layout pass should leave `[B, T, C]`, `[B, heads, T, D]`, and `[B, T, heads, D]` semantics intact. The only axis-sensitive rewrites are sequence/head transposes and reductions over `dim=-1` for RMSNorm/softmax; protect them with a no-image-layout-translation guard.

## 10. Kernel fusion candidates

Highest priority:

- RMSNorm with offset gamma `(1 + weight)`: every block plus final norm.
- Biasless GEMM coverage for attention/MLP/LM head, including non-square attention width.
- RoPE + Q/K layout transform: frequent and cache-sensitive.
- Causal attention for MHA first, then GQA/sliding once source-default variants are admitted.
- Gated GELU MLP elementwise fusion.
- Last-token-only LM head for decode to avoid full `[B,T,V]` logits.

Medium priority:

- Fused QKV projection is possible only if weight packing is created by DinoML; source stores separate `q_proj`, `k_proj`, `v_proj`.
- FlashAttention-style prefill/decode kernels with KV cache update and optional sliding-window bounds.
- Final logit softcap as fused LM-head epilogue for configs with non-null softcap.
- Attention score softcap support in custom attention kernels for source-default synthetic configs.

Lower priority:

- FlexAttention/block-mask parity; source tests show resource-sensitive behavior and 1B mirrors do not need it for first pass.
- Tensor parallel sharding plans.
- Quantized ONNX derivative support; not part of native Transformers model loading.

## 11. Runtime staging plan

Stage 1: Config and dense weights

- Parse `vaultgemma` config with strict shape checks.
- Normalize legacy `rope_theta` into default `rope_parameters`.
- Load tied embedding/LM head as one logical parameter.
- Reject gated official configs unless files are available or caller supplies local weights.

Stage 2: One-block eager parity

- Implement scaled embedding, RMSNorm offset gamma, attention projections with explicit attention width, RoPE, eager dense attention, gated GELU MLP.
- Validate one 1B-shaped layer with random weights.

Stage 3: Full prefill parity

- Compile all 26 full-attention layers for the 1B mirror shape.
- Use dense causal mask and full logits or `logits_to_keep`.
- Stub generation controller; compare prefill logits against Transformers.

Stage 4: Decode with cache

- Add per-layer K/V cache storing RoPE-applied K.
- Decode one token and multi-token loops.
- Validate cache length and output logits.

Stage 5: Optimized attention

- Enable MHA FlashAttention-style path for all-full 1B configs.
- Add GQA direct KV-head support for source defaults.
- Add sliding-window layers only behind explicit `layer_types` admission.

Stage 6: Softcaps and variant admission

- Add attention score softcap and final logit softcap.
- Reject SDPA-like lowering for non-null attention softcap unless fused attention applies it.
- Add tests for source-default synthetic configs.

Stage 7: Production polish

- Add last-token-only logits, batch/sequence sweeps, BF16 provider tuning, and optional GGUF/dense dequant loading once dense parity is stable.

## 12. Parity and validation plan

- Config parsing tests:
  - 1B mirror config: H=1152, attention width 1024, all-full layers.
  - Source-default config: H=2304, attention width 2048, alternating layer types, non-null softcaps.
  - Reject `hidden_size % num_attention_heads != 0` as source validation does.
- Operator tests:
  - RMSNorm offset-gamma against PyTorch fp32 accumulation.
  - RoPE against source for contiguous and offset `position_ids`.
  - Gated GELU MLP against source.
  - Attention score softcap against eager source for non-null softcap.
- Single-layer parity:
  - Random BF16/fp32 weights and inputs for 1B shape.
  - Separate test for `hidden_size != attention_width`.
- Full-model prefill:
  - Local/mirror weights if available; compare selected logits for prompts.
  - Recommended tolerance: fp32 `rtol=1e-4, atol=1e-4`; bf16/fp16 `rtol=3e-2, atol=3e-2` initially, tighten after provider choice.
- Decode parity:
  - One-token decode after prefill.
  - Multi-token greedy decode for small prompts.
  - Cache tensor shape checks per layer.
- Variant parity:
  - GQA synthetic config with KV repeat/direct grouped kernel.
  - Sliding-window synthetic config over sequence longer than window.
- End-to-end:
  - Tokenizer + greedy generation against Transformers for `google/vaultgemma-1b` only when gated weights are locally available.

## 13. Performance probes

- Prefill throughput by sequence length: 128, 512, 1024 for 1B config.
- Decode tokens/sec by batch size with cache resident.
- LM-head cost with full logits versus last-token-only logits.
- GEMM breakdown: Q/K/V/O, gate/up/down, LM head.
- Attention backend comparison: eager dense, DinoML fused MHA, future GQA, future sliding.
- KV cache memory: `layers * 2 * B * heads_kv * seq * head_dim * dtype_size`; for 1B BF16 full cache this is `26 * 2 * B * S * 4 * 256 * 2` bytes.
- Softcap overhead when enabled on source-default synthetic config.
- Weight-load and optional quantized/dequant path separately from compute; native source has no quantized format, but DinoML may later test GGUF/dense dequant as a loading provider.

## 14. Skip/defer list

- Training loss and labels.
- Gradient checkpointing.
- Tensor parallel and pipeline parallel plans.
- FlexAttention production path.
- Sliding-window attention for the released 1B all-full config; keep as a source-default variant follow-up.
- Non-default RoPE scaling/dynamic RoPE.
- Attention/output attentions materialization except eager debug parity.
- Quantized ONNX derivative runtimes.
- Beam search, sampling processors, and generation-controller policy beyond greedy parity.
- Multimodal/image/audio/video preprocessing; not applicable.

## 15. Final implementation checklist

- [ ] Parse `VaultGemmaConfig` and normalize `rope_theta`/`rope_parameters`.
- [ ] Add admission checks for attention width, layer count, layer types, softcaps, and tied weights.
- [ ] Load dense BF16/fp32 weights with embedding/LM-head alias preserved.
- [ ] Implement scaled token embedding.
- [ ] Implement RMSNorm with fp32 accumulation and `(1 + weight)`.
- [ ] Implement biasless Linear/GEMM shapes including `H != heads * head_dim`.
- [ ] Implement default RoPE and cache-position handling.
- [ ] Implement causal MHA prefill for 1B all-full attention.
- [ ] Implement KV cache update and decode.
- [ ] Implement gated GELU MLP fusion.
- [ ] Implement last-token-only logits path.
- [ ] Add optional final logit softcap and attention score softcap.
- [ ] Add GQA direct attention for source-default variants.
- [ ] Add sliding-window attention/cache only behind explicit layer-type guards.
- [ ] Add one-block, full-prefill, and decode parity tests.
- [ ] Add BF16 throughput and cache-memory benchmarks.
