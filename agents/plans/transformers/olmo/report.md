# OLMo Transformers Family Audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: OLMo family, primary checkpoint sweep below
Config source: Hugging Face config.json files, snapshotted in ./_sources/
Source files inspected:
- transformers/src/transformers/models/olmo/configuration_olmo.py
- transformers/src/transformers/models/olmo/modeling_olmo.py
- transformers/src/transformers/models/olmo/modular_olmo.py
- transformers/src/transformers/models/olmo/convert_olmo_weights_to_hf.py
- transformers/src/transformers/models/olmo2/configuration_olmo2.py
- transformers/src/transformers/models/olmo2/modeling_olmo2.py
- transformers/src/transformers/models/olmo2/modular_olmo2.py
- transformers/src/transformers/masking_utils.py
- transformers/src/transformers/cache_utils.py
- transformers/src/transformers/modeling_layers.py
Any missing files or assumptions: tokenizer files were not audited because no model-coupled tokenizer logic is read by the OLMo module graph. The primary runtime target is OlmoForCausalLM text generation.
```

`modeling_olmo.py` is generated from `modular_olmo.py`; future source edits should target the modular file, but DinoML parity should follow the generated file that Transformers imports. OLMo2 source was inspected only to document family-boundary traps. OLMo2 uses a different `model_type` and must not be silently routed through OLMo lowering.

Representative config snapshots:

- `hf-internal-testing/tiny-random-OlmoForCausalLM`: `./_sources/hf-internal-testing__tiny-random-OlmoForCausalLM.config.json`
- `allenai/OLMo-1B-hf`: `./_sources/allenai__OLMo-1B-hf.config.json`
- `allenai/OLMo-7B-hf`: `./_sources/allenai__OLMo-7B-hf.config.json`
- `allenai/OLMo-7B-0424-hf`: `./_sources/allenai__OLMo-7B-0424-hf.config.json`
- `allenai/OLMo-7B-0724-hf`: `./_sources/allenai__OLMo-7B-0724-hf.config.json`
- `allenai/OLMo-7B-0724-Instruct-hf`: `./_sources/allenai__OLMo-7B-0724-Instruct-hf.config.json`
- OLMo2 contrast configs: `./_sources/allenai__OLMo-2-1124-7B.config.json`, `./_sources/allenai__OLMo-2-1124-13B.config.json`, `./_sources/allenai__OLMo-2-1124-7B-Instruct.config.json`

## 2. High-level architecture

OLMo is a text-only, decoder-only causal language model.

```text
token ids / input embeddings -> embedding lookup -> repeated causal decoder blocks -> final parameter-free LayerNorm -> LM head -> logits -> generation cache update/sampling
```

Primary stages:

- CPU/data pipeline: tokenization, padding, attention-mask construction inputs. No OLMo-specific processor exists.
- GPU/runtime prefill: embedding lookup, causal mask handling, RoPE, full-sequence causal self-attention, SwiGLU MLP, logits.
- GPU/runtime decode: one or more new tokens, dynamic or static KV cache read/update, RoPE at absolute positions, causal attention over cached prefix, last-token logits.
- Independently optimizable pieces: RoPE table generation/application, attention backend, SwiGLU MLP, final/token logits projection, KV cache allocation/update.

Implemented heads:

- `OlmoForCausalLM`: required for this audit.
- `OlmoModel`: required as the base decoder.
- `OlmoForSequenceClassification`: optional/deferred. It reuses the decoder, applies a bias-free `score` linear to all hidden states, then selects the rightmost non-pad token if `input_ids` are available.

## 3. Important config dimensions

Source defaults from `OlmoConfig`:

| Field | Default | Runtime significance |
| --- | ---: | --- |
| `vocab_size` | 50304 | embedding rows and LM-head output rows |
| `hidden_size` | 4096 | residual width |
| `intermediate_size` | 11008 | SwiGLU branch width |
| `num_hidden_layers` | 32 | decoder block count |
| `num_attention_heads` | 32 | query/output head count |
| `num_key_value_heads` | defaults to attention heads | MHA by default; GQA/MQA possible if config sets fewer KV heads |
| `head_dim` | inferred as `hidden_size // num_attention_heads` | projection reshape width unless explicit `head_dim` exists |
| `hidden_act` | `silu` | SwiGLU activation |
| `max_position_embeddings` | 2048 | initial RoPE cache/context limit |
| `rope_parameters` | standardized from legacy `rope_theta`/`rope_scaling` | RoPE type and theta consumed by source |
| `attention_bias` | false | q/k/v/o projection bias flag |
| `attention_dropout` | 0.0 | dropout disabled in eval; training only otherwise |
| `clip_qkv` | null | optional in-place clamp on q/k/v projection outputs before reshape/RoPE |
| `tie_word_embeddings` | false | checkpoint-dependent embedding/LM-head alias |
| `use_cache` | true | default generation KV cache behavior |

Representative checkpoint sweep:

| Model/config | type | hidden | layers | q heads | kv heads | head dim | MLP | max pos | theta | `clip_qkv` | tied | dtype |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| `hf-internal-testing/tiny-random-OlmoForCausalLM` | olmo | 32 | 2 | 4 | 4 | 8 | 64 | 4096 | 10000 | 8.0 | false | float32 |
| `allenai/OLMo-1B-hf` | olmo | 2048 | 16 | 16 | 16 | 128 | 8192 | 2048 | 10000 | null | true | float32 |
| `allenai/OLMo-7B-hf` | olmo | 4096 | 32 | 32 | 32 | 128 | 11008 | 2048 | 10000 | null | false | float32 |
| `allenai/OLMo-7B-0424-hf` | olmo | 4096 | 32 | 32 | 32 | 128 | 11008 | 4096 | 10000 | 8.0 | false | float32 |
| `allenai/OLMo-7B-0724-hf` | olmo | 4096 | 32 | 32 | 32 | 128 | 11008 | 4096 | 10000 | 8.0 | false | float32 |
| `allenai/OLMo-7B-0724-Instruct-hf` | olmo | 4096 | 32 | 32 | 32 | 128 | 11008 | 4096 | 10000 | null | false | bfloat16 |
| `allenai/OLMo-2-1124-7B` contrast | olmo2 | 4096 | 32 | 32 | 32 | 128 | 11008 | 4096 | 500000 | absent | false | float32 |
| `allenai/OLMo-2-1124-13B` contrast | olmo2 | 5120 | 40 | 40 | 40 | 128 | 13824 | 4096 | 500000 | absent | false | float32 |

Checkpoint facts are from `config.json`. `pretraining_tp` appears in older OLMo configs but is not read by the inspected OLMo modeling source. `rms_norm_eps` appears in `OLMo-7B-0724-Instruct-hf` config but OLMo source uses fixed `eps=1e-5` parameter-free LayerNorm, not RMSNorm.

## 3a. Family variation traps

- OLMo is not OLMo2. OLMo uses parameter-free `F.layer_norm(..., weight=None, bias=None, eps=1e-5)` before attention, before MLP, and at final norm. OLMo2 uses learned RMSNorm after attention and after MLP, a final learned RMSNorm, and q/k RMSNorm before RoPE.
- OLMo2 configs have `model_type="olmo2"`, vocab size 100352, theta 500000, and different token IDs. They should be separate audit/route targets.
- `clip_qkv` is source-read for OLMo and must be honored when non-null. It clamps q/k/v projection outputs before reshape and RoPE. Some 0424/0724 checkpoints and the tiny checkpoint set `clip_qkv=8.0`.
- `tie_word_embeddings` varies: OLMo-1B ties embedding and LM head; 7B checkpoints generally do not. The source declares `_tied_weights_keys`, but actual aliasing depends on config/loaded weights.
- `num_key_value_heads` defaults to `num_attention_heads`, but source supports GQA/MQA by projecting k/v to `num_key_value_heads * head_dim` and repeating to query heads inside eager attention or delegating to backend attention.
- `attention_bias` is false in inspected configs, but source supports bias on q/k/v/o projections when true.
- Legacy `rope_theta` and `rope_scaling` are standardized into `config.rope_parameters` by Transformers config utilities. Source reads `config.rope_parameters["rope_type"]` and `["rope_theta"]`.
- OLMo source supports non-default RoPE types through `ROPE_INIT_FUNCTIONS` and `dynamic_rope_update`; inspected representative OLMo configs use default RoPE with no scaling.
- `head_dim` may be an explicit config attr in future configs; source uses it if present. DinoML should not assume `hidden_size == num_heads * inferred_head_dim` without checking.
- `logits_to_keep` can restrict LM-head computation to the last tokens or tensor indices; this is a generation memory optimization and affects output shape.
- No NCHW/NHWC layout issue exists for this text-only graph. Layout translation should be disabled for token sequence axes; transposes/views around `[B, S, H] <-> [B, heads, S, D]` are semantic attention reshapes, not image layout candidates.

## 4. Operator coverage checklist

Tensor/layout ops:

- integer input IDs, optional `inputs_embeds`, exact-one input validation
- embedding gather `[B, S] -> [B, S, H]`
- `arange`, add scalar past length, unsqueeze for default `position_ids`
- view/reshape from `[B, S, heads*D]` to `[B, S, heads, D]`
- transpose `[B, S, heads, D] -> [B, heads, S, D]`
- contiguous/reshape back to `[B, S, heads*D]`
- slice/index for `logits_to_keep`
- optional classification gather of rightmost non-pad token

Neural network primitives:

- parameter-free LayerNorm over last dim with fp32 accumulation and fixed eps 1e-5
- bias-free linear projections in common configs:
  - 7B q: `Linear(4096 -> 4096)`, k/v: `Linear(4096 -> 4096)`, o: `Linear(4096 -> 4096)`
  - 7B MLP: gate/up `Linear(4096 -> 11008)`, down `Linear(11008 -> 4096)`
  - 1B MLP: gate/up `Linear(2048 -> 8192)`, down `Linear(8192 -> 2048)`
- optional bias in attention q/k/v/o only if `attention_bias=true`
- SiLU activation and elementwise multiply for SwiGLU
- residual adds
- LM head `Linear(hidden_size -> vocab_size)`, bias false
- optional q/k/v clamp when `clip_qkv` is non-null

Attention primitives:

- causal self-attention
- MHA by inspected configs; GQA/MQA admitted by `num_key_value_heads < num_attention_heads`
- scale by `head_dim ** -0.5`
- mask add before softmax for eager path
- softmax in fp32, cast back to query dtype
- dropout after softmax in training only; eval uses 0.0
- KV cache update per layer after RoPE
- SDPA/Flash/Flex attention dispatch through `ALL_ATTENTION_FUNCTIONS`

Position/rotary ops:

- default RoPE inverse frequency from theta and head dim
- fp32 RoPE frequency matmul, concat `[freqs, freqs]`, cos/sin
- `rotate_half`: split last dim into halves, concat `[-x2, x1]`
- apply RoPE to q/k with cos/sin broadcast at dim 1

Generation/cache ops:

- DynamicCache construction when `use_cache` and no cache is passed
- per-layer append/update for keys/values shaped before repeat: `[B, kv_heads, T_new, D]`
- cache seq length used to derive default `position_ids`
- beam/cache reorder handled by Transformers cache infrastructure, not OLMo-specific source

Distributed/tensor-parallel metadata:

- config exposes TP plan: q/k/v/gate/up colwise, o/down rowwise, LM head colwise gather. This is optimization metadata; single-device DinoML can ignore initially while preserving weight names.

## 5. Layer/block breakdown

OLMo decoder block, repeated `num_hidden_layers` times:

```text
residual = x
x_norm = LayerNormNoWeightBias(x, eps=1e-5, fp32_accum)
q = Linear(hidden -> num_heads * head_dim, bias=attention_bias)(x_norm)
k = Linear(hidden -> num_kv_heads * head_dim, bias=attention_bias)(x_norm)
v = Linear(hidden -> num_kv_heads * head_dim, bias=attention_bias)(x_norm)
if clip_qkv: q,k,v = clamp(q,k,v, -clip_qkv, clip_qkv)
q = view/transposed to [B, num_heads, S, D]
k,v = view/transposed to [B, num_kv_heads, S, D]
q,k = RoPE(q,k, cos[position], sin[position])
k,v = cache.update(k,v, layer_idx) if cache is present
attn = causal_attention(q,k,v, mask, scale=D**-0.5)
x = residual + Linear(num_heads * D -> hidden, bias=attention_bias)(attn)

residual = x
x_norm = LayerNormNoWeightBias(x, eps=1e-5, fp32_accum)
mlp = down_proj(SiLU(gate_proj(x_norm)) * up_proj(x_norm))
x = residual + mlp
```

Model wrapper:

```text
input_ids -> Embedding(vocab_size, hidden_size)
position_ids default to arange(S) + past_seen_tokens
causal_mask = create_causal_mask(...)
cos,sin = rotary_emb(hidden_states, position_ids)
for layer in layers: hidden_states = layer(...)
hidden_states = final LayerNormNoWeightBias(hidden_states)
logits = lm_head(hidden_states[:, slice_indices, :])
```

For 7B OLMo configs, `head_dim=128`, q/k/v/o all have 4096 output/input width because `num_key_value_heads=num_attention_heads=32`. If a future OLMo GQA config sets fewer KV heads, k/v projection width becomes `num_key_value_heads * 128`.

## 6. Attention requirements

OLMo requires causal self-attention only. No cross-attention, encoder cache, sliding-window/local attention, ALiBi, or relative bias is implemented in `modeling_olmo.py`.

Required properties:

- Causal self-attention over token sequence.
- MHA/GQA/MQA contract: q `[B, q_heads, Q, D]`, k/v cache `[B, kv_heads, KV, D]`; eager fallback repeats k/v to q heads by expanding an extra group dimension and reshaping.
- Eager math order: repeat k/v, `q @ k.T`, multiply by scale, add mask, fp32 softmax cast to query dtype, dropout, `weights @ v`, transpose to `[B, Q, heads, D]`.
- Cache stores post-RoPE keys and raw values before KV repetition. This matters for cache ABI and for avoiding repeated RoPE during decode.
- Prefill cache tensors per layer after update: key/value `[B, num_key_value_heads, S_prefill, head_dim]`.
- Decode cache tensors after one token append: key/value `[B, num_key_value_heads, S_prefill + 1, head_dim]`; attention backend may repeat/broadcast to `[B, num_attention_heads, KV, head_dim]` internally.
- Masking is delegated to `create_causal_mask`, which also detects packed sequence position IDs when attention mask is absent and no cache is present. First integration can require ordinary monotonic position IDs and dense causal masks.
- Source advertises FlashAttention, SDPA, and FlexAttention support. DinoML should use eager parity first, then a fused causal attention kernel with GQA and KV-cache guards.

Eager fallback is too slow for production prefill/decode, especially with fp32 softmax and explicit KV repeat. It is a parity reference, not a target kernel path.

## 7. Position encoding and custom math

Default OLMo RoPE:

```python
def olmo_inv_freq(head_dim, rope_theta, device):
    return 1.0 / (rope_theta ** (arange(0, head_dim, 2, dtype=float, device=device) / head_dim))

def olmo_rope_cos_sin(inv_freq, position_ids):
    # position_ids: [B, S]
    freqs = (inv_freq[None, :, None].float() @ position_ids[:, None, :].float()).transpose(1, 2)
    emb = cat([freqs, freqs], dim=-1)
    return cos(emb), sin(emb)  # fp32 in source, scaled by attention_scaling

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return cat([-x2, x1], dim=-1)

def apply_olmo_rope(q, k, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin
```

`cos` and `sin` can be precomputed for static theta/head_dim/position ranges, but dynamic RoPE types can update `inv_freq` based on sequence length through `dynamic_rope_update`. Inspected OLMo configs use default RoPE, so first integration can admit only `rope_type="default"` and `rope_scaling=null`.

OLMo returns cos/sin in fp32 rather than the input dtype, then casts q/k outputs back to their original dtype. Preserve this for fp16/bf16 parity.

## 8. Preprocessing and input packing

No image/audio/multimodal processor is involved.

Runtime text inputs:

- `input_ids`: integer tensor `[B, S]`; exactly one of `input_ids` or `inputs_embeds` must be provided.
- `inputs_embeds`: optional `[B, S, H]`; bypasses embedding lookup.
- `attention_mask`: optional mask input consumed by `create_causal_mask`; common generation uses 2D mask `[B, S_or_KV]`.
- `position_ids`: optional `[B, S]`. If absent, source creates `arange(S) + past_seen_tokens` and unsqueezes to batch 1; broadcasting occurs in mask/RoPE paths.
- Packed sequences: `create_causal_mask` can infer packed sequence indices from non-monotonic/resetting `position_ids` when `attention_mask is None` and no cache is present. This is a source-supported edge case but safe to defer behind an admission guard.

Generation-controller behavior:

- Core module does not force decoder IDs or suppress tokens.
- `logits_to_keep=0` means all logits because `slice(-0, None)` is equivalent to full slice in Python; positive int keeps last N tokens. Tensor indices are also accepted.
- Tokenizer special IDs come from config (`pad_token_id`, `eos_token_id`); tokenizer implementation is outside the GPU graph.

## 9. Graph rewrite / lowering opportunities

### Rewrite: parameter-free LayerNorm

Source pattern:

```text
F.layer_norm(x.float(), normalized_shape=(hidden_size,), weight=None, bias=None, eps=1e-5).to(orig_dtype)
```

Replacement:

```text
MeanVariance(last_dim, fp32) -> rsqrt(var + 1e-5) -> normalize -> cast original dtype
```

Preconditions:

- Last dimension equals `hidden_size`.
- Weight and bias are absent.
- Epsilon is fixed 1e-5 for OLMo source, regardless of any stray `rms_norm_eps` config field.

Failure cases:

- Do not apply to OLMo2 RMSNorm or any checkpoint routed to `model_type="olmo2"`.

Parity test sketch: random fp32/fp16/bf16 `[B,S,H]`, compare against PyTorch `F.layer_norm` with no affine.

### Rewrite: SwiGLU MLP fusion

Source pattern:

```text
down_proj(silu(gate_proj(x)) * up_proj(x))
```

Replacement:

```text
dual GEMM hidden->intermediate for gate/up -> fused SiLU*multiply -> GEMM intermediate->hidden
```

Preconditions:

- `gate_proj`, `up_proj`, `down_proj` are bias-free.
- `hidden_act == "silu"` or a separately implemented activation from `ACT2FN`.
- Same input tensor feeds gate and up projections.

Shape equations:

- gate/up: `[B*S, H] x [H, I] -> [B*S, I]`
- down: `[B*S, I] x [I, H] -> [B*S, H]`

Failure cases:

- Unknown activation, added biases, or nonstandard tensor-parallel sharded weight layout.

### Rewrite: RoPE + attention canonicalization

Source pattern:

```text
q/k/v linear -> optional qkv clamp -> reshape/transpose -> RoPE q/k -> cache update -> attention
```

Replacement:

```text
QKV projections -> optional clamp -> fused RoPE attention prefill/decode
```

Preconditions:

- Default RoPE or implemented RoPE type.
- Cache stores post-RoPE keys.
- Attention is causal self-attention.
- Mask is standard dense causal/padding mask or admitted packed-mask variant.

Failure cases:

- `clip_qkv` omitted from fused path for checkpoints that require it.
- Packed sequence mask without equivalent kernel support.
- Dynamic/non-default RoPE admitted without dynamic update parity.

### Rewrite: last-token-only logits

Source pattern:

```text
lm_head(hidden_states[:, slice_indices, :])
```

Replacement:

```text
gather last N hidden states -> GEMM to vocab
```

Preconditions:

- `logits_to_keep` is positive int or known tensor indices.
- Caller does not require full `[B,S,V]` logits.

Failure cases:

- Loss computation over all labels, tensor indices with dynamic shape unsupported.

Layout notes: preserve `[B,S,H]` semantic order. No global layout rewrite should reinterpret sequence and hidden axes.

## 10. Kernel fusion candidates

Highest priority:

- Parameter-free LayerNorm with fp32 accumulation. It appears twice per block plus final norm and differs from RMSNorm.
- Causal FlashAttention/SDPA-equivalent with GQA and KV-cache ABI. This dominates prefill/decode.
- RoPE application fused with attention input preparation, preserving fp32 cos/sin and post-RoPE cache storage.
- SwiGLU fused activation multiply between two GEMMs.
- Last-token-only logits for decode to avoid full sequence-vocab GEMM.

Medium priority:

- Optional q/k/v clamp fused after projections for `clip_qkv=8.0` checkpoints.
- QKV projection grouping. Source stores separate q/k/v weights, so any packed QKV kernel needs a weight-packing transform and separate k/v widths for GQA.
- Bias-aware attention projections for future configs with `attention_bias=true`.

Lower priority:

- Sequence classification pooling/indexing.
- Training dropout/loss paths.
- Tensor-parallel execution plan from HF TP metadata.

## 11. Runtime staging plan

Stage 1: parse OLMo configs, reject `model_type!="olmo"`, reject unsupported non-default RoPE/GQA variants only if kernels cannot handle them, and load/tie weights correctly.

Stage 2: implement one-block eager parity using embedding, parameter-free LayerNorm, separate attention projections, default RoPE, eager causal attention, and SwiGLU.

Stage 3: full prefill parity for `OlmoForCausalLM` with ordinary 2D attention masks, no packed position IDs, and full logits.

Stage 4: decode parity with DynamicCache-compatible per-layer KV tensors stored post-RoPE as `[B, kv_heads, T, D]`.

Stage 5: optimized attention path for MHA first, then GQA/MQA if admitted by config.

Stage 6: add q/k/v clamp support and last-token-only logits.

Stage 7: optional sequence classification head and tensor-parallel planning.

Initial stubs: training loss, dropout, gradient checkpointing, packed sequence masks, non-default RoPE, sequence classification.

## 12. Parity and validation plan

- Config loader tests: OLMo-1B tied embeddings, OLMo-7B untied embeddings, 0424/0724 `clip_qkv=8.0`, instruct bf16 dtype metadata, OLMo2 rejection.
- Custom op tests:
  - parameter-free LayerNorm vs PyTorch for fp32/fp16/bf16, eps 1e-5
  - RoPE cos/sin and `rotate_half` vs source snippets
  - q/k/v clamp before reshape/RoPE
  - `repeat_kv` for MHA and synthetic GQA
- Single-layer parity: fixed random weights and hidden states through one `OlmoDecoderLayer`.
- N-layer parity: small/tiny config with 2 layers, no cache.
- Prefill logits parity: selected OLMo checkpoint or random tiny model, ordinary attention mask, full logits and `logits_to_keep=1`.
- Decode parity: prefill N tokens, decode one token, compare logits and per-layer cache length/content shape.
- Tolerances: fp32 `rtol=1e-4, atol=1e-5`; fp16/bf16 `rtol=5e-2, atol=5e-2` initially for fused attention, tighten per kernel.

No DinoML tests were run for this docs-only audit.

## 13. Performance probes

- Prefill throughput by sequence length: 128, 512, 2048, 4096.
- Decode tokens/sec by batch size and cache length.
- KV cache memory: `layers * 2 * B * kv_heads * T * head_dim * dtype_size`.
- Attention backend comparison: eager parity, SDPA/Flash-like fused, GQA synthetic if admitted.
- LayerNorm bandwidth probe for parameter-free fp32 accumulation.
- SwiGLU probe split into gate/up GEMMs, activation multiply, down GEMM.
- Logits GEMM probe for full sequence vs last-token-only.
- `clip_qkv` overhead probe on 0424/0724 configs.
- Weight loading probe for tied vs untied embedding/LM-head storage.

## 14. Skip/defer list

- Training loss and gradient checkpointing.
- Attention dropout in training.
- Beam-search cache reorder unless generation integration requires it immediately.
- Packed sequence position-ID masking.
- Non-default/dynamic RoPE scaling types.
- Tensor parallel and pipeline parallel execution.
- Sequence classification head.
- OLMo2 routing and RMSNorm/qk-norm behavior, except explicit rejection from OLMo path.
- Quantization and GGUF ingestion.

## 15. Final implementation checklist

- [ ] Parse `OlmoConfig`, including legacy `rope_theta`/`rope_scaling` standardized into `rope_parameters`.
- [ ] Reject or separately route `model_type="olmo2"`.
- [ ] Load embeddings, decoder blocks, final norm, and LM head with tied-weight alias preservation.
- [ ] Implement parameter-free LayerNorm over last dim with fp32 accumulation and eps 1e-5.
- [ ] Implement default RoPE with fp32 cos/sin and post-RoPE cache storage.
- [ ] Implement q/k/v/o projections with `attention_bias` guard.
- [ ] Implement optional `clip_qkv`.
- [ ] Implement causal MHA plus GQA/MQA `repeat_kv` semantics.
- [ ] Implement KV cache ABI `[B, kv_heads, T, head_dim]` per layer.
- [ ] Implement SwiGLU MLP.
- [ ] Implement `logits_to_keep` slicing and LM-head GEMM.
- [ ] Add config sweep tests for tiny, 1B, 7B, 0424/0724, instruct, and OLMo2 rejection.
- [ ] Add one-block, prefill, and decode parity tests.
- [ ] Add performance probes for prefill, decode, LayerNorm, SwiGLU, logits, and KV memory.
