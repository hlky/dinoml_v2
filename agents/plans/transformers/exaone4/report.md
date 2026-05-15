# DinoML Transformers audit: exaone4

## 1. Source basis

Transformers commit/version: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4` from `transformers`.

Model id: primary target `LGAI-EXAONE/EXAONE-4.0-32B`; representative configs also checked for `LGAI-EXAONE/EXAONE-4.0-1.2B`, `LGAI-EXAONE/EXAONE-4.0-32B-AWQ`, and `LGAI-EXAONE/EXAONE-4.0-32B-FP8`.

Config source: local `configuration_exaone4.py` plus public HF raw `config.json` and `generation_config.json` URLs listed in `_sources/source_notes.md`.

Source files inspected:

- `src/transformers/models/exaone4/configuration_exaone4.py`
- `src/transformers/models/exaone4/modeling_exaone4.py`
- `src/transformers/models/exaone4/modular_exaone4.py`
- supporting shared code: Gemma2 RoPE, Llama CausalLM head/generation inheritance, Olmo2 MLP/block pattern, common cache and masking utilities.

Any missing files or assumptions: no EXAONE4-specific tokenizer/model coupling beyond normal causal-LM token IDs was required for graph lowering. No official tiny debug checkpoint was found; source defaults can serve as a synthetic tiny-ish config, while public 1.2B/32B configs cover operator-significant variation. The generated modeling/config files are the runtime source at this commit; `modular_exaone4.py` is the upstream edit source.

## 2. High-level architecture

EXAONE4 is a decoder-only causal LM with optional classification/QA heads inherited through generic wrappers. The first useful DinoML target should be `Exaone4ForCausalLM` inference: prompt prefill, autoregressive decode with cache, final logits.

Dataflow:

```text
input_ids or inputs_embeds
  -> token embedding
  -> repeated decoder blocks with hybrid full/sliding causal self-attention
  -> final RMSNorm
  -> LM head
  -> logits / generation controller
```

Stage decomposition:

- CPU/data pipeline: GPT2-style tokenizer, chat template, BOS/EOS/PAD handling, sampling/search controller.
- GPU/runtime graph: embeddings, decoder blocks, RoPE/NoPE attention, MLP, final norm, selected logits.
- Cacheable state: per-layer self-attention K/V cache; 32B uses hybrid full and sliding-window layers, while 1.2B is all full attention.
- Independently validatable slices: RMSNorm, Q/K post-projection RMSNorm, RoPE math, one decoder block, prefill logits, single-token decode with cache, hybrid cache mask behavior.

## 3. Important config dimensions

Source default config values:

| Field | Default |
| --- | ---: |
| `vocab_size` | 102400 |
| `hidden_size` | 4096 |
| `intermediate_size` | 16384 |
| `num_hidden_layers` | 32 |
| `num_attention_heads` | 32 |
| `num_key_value_heads` | 32 |
| `head_dim` | absent; effective `hidden_size // num_attention_heads` |
| `hidden_act` | `silu` |
| `max_position_embeddings` | 2048 |
| `rms_norm_eps` | `1e-5` |
| `attention_dropout` | `0.0` |
| `sliding_window` | 4096 |
| `sliding_window_pattern` | 4 |
| `tie_word_embeddings` | false |
| `use_cache` | true |

Representative checkpoint sweep:

| Model/config | Layers | Hidden | Q heads | KV heads | Head dim | Q width | KV width | MLP width | Context | Attention pattern | Tied embed | Quantization |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| source default | 32 | 4096 | 32 | 32 | 128 inferred | 4096 | 4096 | 16384 | 2048 | generated from pattern 4 | no | none |
| EXAONE-4.0-1.2B | 30 | 2048 | 32 | 8 | 64 | 2048 | 512 | 4096 | 65536 | all full attention, no sliding | yes | none |
| EXAONE-4.0-32B | 64 | 5120 | 40 | 8 | 128 | 5120 | 1024 | 27392 | 131072 | `LLLG` repeated, sliding 4096 | no | none |
| EXAONE-4.0-32B-AWQ | 64 | 5120 | 40 | 8 | 128 | 5120 | 1024 | 27392 | 131072 | same as 32B | no | AWQ 4-bit, group 128, zero point, GEMM, `lm_head` dense |
| EXAONE-4.0-32B-FP8 | 64 | 5120 | 40 | 8 | 128 | 5120 | 1024 | 27392 | 131072 | same as 32B | no | FP8 dynamic activation, 128x128 weight blocks |

Config-derived RoPE for public checkpoints: `rope_scaling/rope_parameters` type `llama3`, `factor=16`, `low_freq_factor=1`, `high_freq_factor=4`, `original_max_position_embeddings=8192`, `rope_theta=1000000`.

## 3a. Family variation traps

- `head_dim` is explicit in public configs. Do not assume only `hidden_size / num_attention_heads`, even though current values match.
- GQA is required for public checkpoints: `num_key_value_heads=8` while query heads are 32 or 40. KV projection width is `num_key_value_heads * head_dim`, not `hidden_size`.
- 32B is hybrid attention: three sliding/local layers then one full/global layer. 1.2B is all full/global with `sliding_window=null`.
- RoPE is not applied uniformly. Source comment says global NoPE for hybrid attention; implementation applies RoPE only if `sliding_window is None` or the current layer is sliding. Therefore 32B full layers use NoPE, while 1.2B full layers still use RoPE because `sliding_window is None`.
- Q and K each have post-projection RMSNorm over `head_dim`; this is not standard Llama.
- Decoder block is post-attention/post-MLP norm before residual add, inherited from Olmo2-style generated code, not Llama pre-norm.
- Public 1.2B ties embeddings and LM head; 32B does not. Weight aliasing must remain logical when tied.
- AWQ/FP8 configs change weight loading/provider requirements, not the EXAONE4 Python graph. Unsupported quantized loaders should route to dense fallback or explicit rejection.
- Generation configs for 32B variants set `cache_implementation="hybrid"`, but current Transformers generation code unsets default hybrid before preparing cache, favoring dynamic hybrid cache unless the user explicitly overrides.
- Source default `sliding_window_pattern` is an int; public 32B uses string `LLLG`. DinoML should accept only layer type lists or validated int/string patterns whose expanded `layer_types` length equals `num_hidden_layers`.

## 4. Operator coverage checklist

Tensor/layout ops:

- Integer token gather: `Embedding(vocab_size, hidden_size)`.
- Shape ops: `view`, `transpose(1,2)`, `reshape`, `contiguous`, slicing `hidden_states[:, slice_indices, :]` for `logits_to_keep`.
- Broadcast/expand/reshape for GQA KV repeat in eager fallback.

Neural network primitives:

- Bias-free Linear projections:
  - Q: `Linear(hidden_size -> num_attention_heads * head_dim)`.
  - K/V: `Linear(hidden_size -> num_key_value_heads * head_dim)`.
  - O: `Linear(num_attention_heads * head_dim -> hidden_size)`.
  - MLP gate/up: `Linear(hidden_size -> intermediate_size)`.
  - MLP down: `Linear(intermediate_size -> hidden_size)`.
  - LM head: `Linear(hidden_size -> vocab_size)`, bias false.
- RMSNorm over hidden size and over per-head `head_dim`.
- SiLU, elementwise multiply for SwiGLU, residual adds.

Attention primitives:

- Causal self-attention, full and sliding-window local causal variants.
- GQA/MQA-style KV grouping: `num_attention_heads // num_key_value_heads`.
- Eager fallback computes `softmax((Q @ K^T) * head_dim^-0.5 + mask, dtype=float32).to(query.dtype) @ V`.
- SDPA/Flash/Flex attention are advertised by source flags and dispatch through `ALL_ATTENTION_FUNCTIONS`.

Position/rotary ops:

- Llama3-scaled RoPE with full `head_dim` rotary dimension unless `partial_rotary_factor` appears.
- RoPE applied to Q/K after QK RMSNorm and before cache update on RoPE-enabled layers.

Generation/cache ops:

- `DynamicCache(config=...)` creates one cache layer per configured layer type.
- Full layers store K/V as `[B, kv_heads, seq_len, head_dim]`.
- Sliding layers store K/V as `[B, kv_heads, min(seq_len, sliding_window), head_dim]` in static cache and last `sliding_window - 1` tokens in dynamic sliding cache while returning full current attention states.
- Beam/cache reorder needs batch-index selection/gather over cache batch dimension.

Quantized/packed metadata ops:

- AWQ: 4-bit grouped weight contract with zero point, group size 128, GEMM version, and dense `lm_head`.
- FP8: dynamic activation and block-quantized weights with 128x128 blocks.
- No EXAONE4-specific packed projection layout appears in source; normal PyTorch `nn.Linear` weight layout is `[out_features, in_features]`.

## 5. Layer/block breakdown

Decoder block, repeated `num_hidden_layers` times:

```text
residual = x
q = Linear_Q(x).view(B,S,QH,D).transpose(1,2)
k = Linear_K(x).view(B,S,KVH,D).transpose(1,2)
v = Linear_V(x).view(B,S,KVH,D).transpose(1,2)
q = RMSNorm_head(q)
k = RMSNorm_head(k)
if rope_enabled_for_layer:
  q,k = RoPE(q,k, cos, sin)
k,v = cache.update(k,v, layer_idx) when cache is present
attn = causal_attention(q,k,v, mask, scale=D^-0.5, sliding_window maybe)
x = Linear_O(attn.transpose/reshape)
x = residual + RMSNorm_hidden(x)
residual = x
m = Linear_down(silu(Linear_gate(x)) * Linear_up(x))
x = residual + RMSNorm_hidden(m)
```

For 32B: Q/O widths are 5120, K/V widths are 1024, MLP hidden is 27392. For 1.2B: Q/O widths are 2048, K/V widths are 512, MLP hidden is 4096. All these Linear modules are bias-free in source.

## 6. Attention requirements

Attention is autoregressive causal self-attention only for the primary target.

- Variant: GQA. Query heads are 32 or 40; KV heads are 8 in public checkpoints; head dim is 64 for 1.2B and 128 for 32B.
- Q/K/V widths: Q width `QH*D`, K/V width `KVH*D`, value head dim equals key head dim in source.
- Masking: full causal mask for full layers; sliding causal mask for sliding layers; optional 2D padding mask is folded into the attention mask utilities.
- Sliding/local pattern: 32B `LLLG` repeats over 64 layers, so layer indices 3, 7, ..., 63 are full/global. Sliding layers pass `sliding_window=4096` to attention backend.
- RoPE interaction: sliding layers in 32B use RoPE; full/global layers in 32B use NoPE. All-full 1.2B uses RoPE because `sliding_window is None`.
- Cache storage before repeat: cache stores un-repeated KV heads. Eager attention repeats K/V to query-head count only for matmul fallback.
- Cached K is stored after QK RMSNorm and after RoPE when RoPE is enabled for that layer.
- Packed/varlen: no EXAONE4-specific varlen ABI in modeling source. Common attention backends may optimize masks, but DinoML should first model dense full and local causal masks.
- Flash/SDPA compatibility: source advertises FlashAttention, SDPA, Flex attention. DinoML can lower to provider-backed attention when the backend supports GQA, optional local window, QK norm output, pre-RoPE cache semantics, and NoPE global layers.

## 7. Position encoding and custom math

RoPE cos/sin are produced once per model forward from `position_ids` and broadcast into each attention layer. Computation is forced in float32 and cast back to hidden dtype.

Llama3 frequency scaling from shared rope utility:

```python
inv_freq = 1.0 / (rope_theta ** (arange(0, dim, 2) / dim))
wavelen = 2 * pi / inv_freq
low = original_max_position_embeddings / low_freq_factor
high = original_max_position_embeddings / high_freq_factor
inv_low = where(wavelen > low, inv_freq / factor, inv_freq)
smooth = (original_max_position_embeddings / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
inv_freq = where((wavelen >= high) & (wavelen <= low),
                 (1 - smooth) * inv_low / factor + smooth * inv_low,
                 inv_low)
```

Apply function:

```python
def exaone4_apply_rope(q, k, cos, sin):
    cos = cos[:, None, :, :]
    sin = sin[:, None, :, :]
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)
```

What can be precomputed: inverse frequencies and, for fixed context buckets, cos/sin tables. What depends on runtime inputs: `position_ids`, cache length/past seen tokens, batch shape, and any dynamic RoPE type updates if non-default advanced RoPE is admitted.

## 8. Preprocessing and input packing

Model-coupled preprocessing is ordinary text/token generation:

- Inputs are either `input_ids` `[B,S]` or `inputs_embeds` `[B,S,H]`, exactly one required.
- If `position_ids` is omitted, source creates `arange(S) + past_seen_tokens` and unsqueezes to `[1,S]`.
- `attention_mask` may be a caller mask or a prebuilt dict with `full_attention` and `sliding_attention`; otherwise source constructs both masks as needed.
- Tokenizer config checked from HF reports `GPT2Tokenizer`, `[BOS]` id 1, `[|endofturn|]` EOS id 361, `[PAD]` id 0, `[UNK]` id 3.
- Chat template and sampling are generation-controller/data-pipeline concerns, not neural graph ops for first DinoML parity.

No image/audio/video processors, placeholder scatter, packed modality metadata, or channel layout translation are involved.

## 9. Graph rewrite / lowering opportunities

### Rewrite: separate Q/K/V projections to packed QKV provider call

Source pattern:

```text
q_proj(x), k_proj(x), v_proj(x)
```

Replacement: one packed GEMM producing `[Q | K | V]`, then split into Q width `QH*D`, K width `KVH*D`, V width `KVH*D`.

Preconditions:

- All three projections consume the same contiguous `x`.
- All are bias-free or all bias transforms are represented explicitly; current source is bias-free.
- Packed weight rows are concatenated in exact order Q, K, V from source weight layout `[out, in]`.
- Split sizes must use explicit `head_dim`, `num_attention_heads`, `num_key_value_heads`.

Failure cases: quantized backends with incompatible packed formats, tensor-parallel shards that already split columns, dynamic weight replacement that breaks alias/provenance.

Parity test sketch: compare individual projections and packed projection split for random BF16/FP32 tensors, then one full attention block.

### Rewrite: QK RMSNorm + RoPE + attention preparation

Replacement: fuse per-head RMSNorm and RoPE into the attention input preparation kernel, optionally producing cache-ready K.

Preconditions:

- Norm axis is exactly last `head_dim`.
- RoPE enabled by layer rule: 32B sliding layers only, all 1.2B layers.
- RoPE must occur after QK RMSNorm and before cache update.
- Full/global NoPE layers in 32B must bypass RoPE.

Failure cases: nonstandard `partial_rotary_factor`, advanced dynamic RoPE types not proven, output attentions requiring intermediate tensors.

### Rewrite: sliding-window GQA attention provider

Replacement: provider-backed local causal attention with KV heads and repeat factor implicit.

Preconditions:

- Causal local window equals config `sliding_window`.
- Cache stores un-repeated KV heads.
- Backend supports rectangular prefill/decode KV lengths and optional padding masks.
- Mask offset semantics match cache `get_mask_sizes`: sliding dynamic cache keeps a rolling subset but may return full states during update.

Failure cases: packed sequence masks, block overlays, attention implementations that skip mask materialization differently, output attentions.

### Rewrite: last-token-only logits

Source pattern: `logits_to_keep` slices hidden states before `lm_head`.

Replacement: for decode or sampling-only prefill, run LM head only over requested token positions.

Preconditions:

- Caller does not request full-sequence logits or loss.
- `logits_to_keep` is `1` or a static trailing count/tensor index set known to runtime planner.
- Output ABI records reduced logits shape.

Failure cases: training loss, full logit parity tests, arbitrary dynamic tensor indices not represented.

### Rewrite: tied embedding/LM head alias

For 1.2B, `tie_word_embeddings=true`; `lm_head.weight` and `model.embed_tokens.weight` are one logical parameter.

Preconditions: loader preserves alias identity and does not duplicate/dequantize inconsistently.

Failure cases: quantized path where embedding and LM head require different providers or dense exceptions.

## 10. Kernel fusion candidates

Highest priority:

- RMSNorm hidden and per-head QK RMSNorm. It appears multiple times per layer and source computes in fp32.
- GQA attention with KV cache, both full causal and sliding-window causal.
- QKV projection + QK RMSNorm + RoPE preparation for prefill/decode.
- SwiGLU MLP: gate/up projections, SiLU, multiply, down projection.
- Last-token-only LM head for decode.

Medium priority:

- Packed QKV GEMM and split metadata for provider selection.
- Residual-add-after-RMSNorm pattern in the post-norm block.
- RoPE cos/sin precompute/cache by context bucket and dtype.
- Hybrid cache allocation layout with per-layer full/sliding cache plans.

Lower priority:

- Classification/token/QA heads; useful later but not needed for causal LM.
- Output attentions materialization.
- Quantized AWQ/FP8 native provider paths. First enable dense fallback or explicit rejection with clear metadata.

## 11. Runtime staging plan

Stage 1: parse `Exaone4Config`, expand and validate `layer_types`, load dense BF16 weights, preserve tied embedding aliases.

Stage 2: implement source-default and 1.2B all-full single-block parity with RMSNorm, QK norm, RoPE, GQA, SwiGLU, post-norm residual order.

Stage 3: implement prefill for `Exaone4ForCausalLM`, including full logits and `logits_to_keep` reduced logits.

Stage 4: implement cache decode for all-full layers, then 32B hybrid full/sliding layers with layer-specific RoPE/NoPE.

Stage 5: replace eager attention with provider-backed full GQA and sliding-window GQA attention.

Stage 6: add AWQ/FP8 admission: parse quantization configs, route unsupported providers to dense materialization or reject with explicit source gap.

Stage 7: production generation integration: batching, beam cache reorder, sampling controllers, cache memory probes.

Initial stubs: labels/loss, output attentions, hidden-state capture, classification/QA heads, quantized native kernels.

## 12. Parity and validation plan

- RMSNorm random tensor tests over hidden axis and head_dim axis; compare fp32 and bf16-cast behavior.
- Llama3 RoPE table tests for 1.2B/32B configs and selected positions around 8192, 32768, 65536, 131072.
- Apply-RoPE tests on `[B,H,S,D]`, including NoPE bypass for 32B full/global layers.
- QKV projection shape tests: 32B Q `[B,S,40,128]`, K/V `[B,S,8,128]`; 1.2B Q `[B,S,32,64]`, K/V `[B,S,8,64]`.
- One decoder layer parity with cache disabled, then enabled for one-token decode.
- Hybrid mask tests: verify 32B sliding layers cannot attend outside local window while full layers can attend all prior tokens.
- Prefill logits parity for short prompts with `logits_to_keep=0` and `1`.
- Decode token parity for a fixed greedy step sequence.
- Recommended tolerances: fp32 `rtol=1e-4, atol=1e-5`; bf16/fp16 block-level `rtol=2e-2, atol=2e-2`, logits tighter when using fp32 accumulation where possible.

## 13. Performance probes

- Prefill throughput sweep: batch, sequence length, 1.2B all-full versus 32B hybrid.
- Decode tokens/sec with cache: batch size and generated length sweep.
- KV cache memory by layer type: full layers grow with sequence length; sliding layers cap around window length.
- Attention backend comparison: eager matmul, SDPA/Flash-equivalent full GQA, local sliding-window GQA.
- QKV packed GEMM versus separate GEMMs.
- MLP throughput and fused SwiGLU/down-projection scheduling.
- Last-token-only LM head versus full-sequence LM head for prompt processing.
- Quantized load/dequant/provider probes: dense BF16, AWQ 4-bit materialized to dense, native AWQ GEMM, FP8 block provider.

## 14. Skip/defer list

- Training loss and gradient checkpointing.
- Sequence/token classification and QA heads for first causal-LM target.
- `output_attentions=True` and full attention weight materialization.
- Beam search and advanced sampling controllers beyond cache reorder shape support.
- Packed sequence/block overlay masks until the common masking utilities are audited as a separate feature.
- Native AWQ/FP8 kernels until DinoML has explicit quantized provider manifests and dense fallback admission.
- Multi-GPU tensor/pipeline parallel plans; source records TP/PP plans but single-device inference should land first.
- Offloaded/quantized KV cache variants.

## 15. Final implementation checklist

- [ ] Parse `Exaone4Config`, including `rope_scaling` -> `rope_parameters`.
- [ ] Validate or expand `layer_types` from `sliding_window_pattern`.
- [ ] Load dense weights with tied embedding alias handling.
- [ ] Implement RMSNorm with fp32 variance.
- [ ] Implement bias-free Linear projections with explicit Q/K/V/O/MLP/LM dimensions.
- [ ] Implement per-head Q/K RMSNorm.
- [ ] Implement Llama3 RoPE and layer-specific RoPE/NoPE gating.
- [ ] Implement GQA full causal attention.
- [ ] Implement sliding-window causal attention and hybrid cache manifest.
- [ ] Implement Dynamic/Static cache update and reorder contracts for full and sliding layers.
- [ ] Implement SwiGLU MLP and post-norm residual block order.
- [ ] Implement `logits_to_keep` LM-head slicing.
- [ ] Add dense BF16 prefill and decode parity tests.
- [ ] Add 1.2B all-full and 32B hybrid config shape tests.
- [ ] Add AWQ/FP8 quantization config admission with dense fallback or clear rejection.
- [ ] Benchmark prefill, decode, cache memory, attention backend, and LM-head slicing.
