# Transformers audit: ministral3

## 1. Source basis

Transformers commit/version: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4` from `transformers`.

Model id: source default is `mistralai/Ministral-3-8B-Base-2512`; official current checkpoints are mostly `mistralai/Ministral-3-{3B,8B,14B}-{Base,Instruct,Reasoning}-2512`. Those official repos use top-level `model_type="mistral3"` with nested `text_config.model_type="ministral3"`. This report owns the nested text decoder / `Ministral3ForCausalLM` surface. The multimodal wrapper, vision tower, projector, image packing, and image placeholder stitch belong to a separate `mistral3` audit.

Config source: raw Hugging Face `config.json` snapshots saved beside this report:

- `mistralai__Ministral-3-3B-Base-2512__config.json`
- `mistralai__Ministral-3-3B-Instruct-2512__config.json`
- `mistralai__Ministral-3-3B-Instruct-2512-BF16__config.json`
- `mistralai__Ministral-3-3B-Reasoning-2512__config.json`
- `mistralai__Ministral-3-8B-Base-2512__config.json`
- `mistralai__Ministral-3-8B-Instruct-2512__config.json`
- `mistralai__Ministral-3-8B-Reasoning-2512__config.json`
- `mistralai__Ministral-3-14B-Base-2512__config.json`
- `mistralai__Ministral-3-14B-Instruct-2512__config.json`
- `mistralai__Ministral-3-14B-Instruct-2512-BF16__config.json`
- `tiny-random__ministral-3__config.json`

Source files inspected:

- `src/transformers/models/ministral3/configuration_ministral3.py`
- `src/transformers/models/ministral3/modular_ministral3.py`
- `src/transformers/models/ministral3/modeling_ministral3.py`
- `src/transformers/models/ministral3/convert_ministral3_weights_to_hf.py`
- `src/transformers/modeling_rope_utils.py` for YaRN RoPE parameter math
- Representative generation/tokenizer/processor snapshots from `mistralai/Ministral-3-8B-Instruct-2512`

Any missing files or assumptions: no gated access was needed for the sampled configs. The generated `modeling_ministral3.py` states that `modular_ministral3.py` is authoritative for future source edits.

## 2. High-level architecture

Primary runtime target: text-only autoregressive causal LM prefill/decode for the `ministral3` decoder.

Architecture: dense decoder-only transformer with token embedding, repeated decoder blocks, final RMSNorm, and untied or tied LM head depending on checkpoint. No MoE, no cross-attention, no encoder in this source family.

Dataflow:

```text
input_ids / inputs_embeds
  -> token embedding
  -> causal or sliding-window causal mask
  -> shared RoPE cos/sin for current positions
  -> N decoder blocks
  -> final RMSNorm
  -> optional last-token / selected-token LM head
  -> logits -> generation controller / sampling
```

Stage decomposition:

- CPU/data pipeline: tokenizer, chat template, attention mask construction inputs, optional generation-controller decisions.
- GPU/runtime prefill: embedding, all decoder layers, full causal/GQA attention, logits for requested positions.
- GPU/runtime decode: one or more new tokens, per-layer KV cache update, last-token logits.
- Independently optimizable pieces: RMSNorm, Q/K/V projections, YaRN RoPE plus query scale, GQA attention with KV cache, SwiGLU MLP, LM head.

## 3. Important config dimensions

Effective source defaults from `Ministral3Config`:

| Field | Default |
| --- | ---: |
| vocab_size | 131072 |
| hidden_size | 4096 |
| num_hidden_layers | 34 |
| num_attention_heads | 32 |
| num_key_value_heads | 8 |
| head_dim | 128 |
| q_width | 4096 |
| kv_width | 1024 |
| intermediate_size | 14336 |
| max_position_embeddings | 262144 |
| hidden_act | `silu` |
| rms_norm_eps | 1e-5 |
| use_cache | true |
| sliding_window | null |
| tie_word_embeddings | false |
| RoPE | YaRN, factor 16, original max positions 16384, `llama_4_scaling_beta=0.1` |

Representative checkpoint sweep, using nested `text_config` for official `mistral3` repos:

| Checkpoint | Text layers | Hidden | Q width | KV width | MLP | Vocab | RoPE theta | Tie embed/head | Quant config |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `mistralai/Ministral-3-3B-Base-2512` | 26 | 3072 | 4096 | 1024 | 9216 | 131072 | 1e6 | true | none |
| `mistralai/Ministral-3-3B-Instruct-2512` | 26 | 3072 | 4096 | 1024 | 9216 | 131072 | 1e6 | true | fp8 static |
| `mistralai/Ministral-3-3B-Reasoning-2512` | 26 | 3072 | 4096 | 1024 | 9216 | 131072 | 1e6 | true | none |
| `mistralai/Ministral-3-8B-Base-2512` | 34 | 4096 | 4096 | 1024 | 14336 | 131072 | 1e6 | default false | none |
| `mistralai/Ministral-3-8B-Instruct-2512` | 34 | 4096 | 4096 | 1024 | 14336 | 131072 | 1e6 | default false | fp8 static |
| `mistralai/Ministral-3-8B-Reasoning-2512` | 34 | 4096 | 4096 | 1024 | 14336 | 131072 | 1e6 | default false | none |
| `mistralai/Ministral-3-14B-Base-2512` | 40 | 5120 | 4096 | 1024 | 16384 | 131072 | 1e9 | default false | none |
| `mistralai/Ministral-3-14B-Instruct-2512` | 40 | 5120 | 4096 | 1024 | 16384 | 131072 | 1e9 | default false | fp8 static |
| `tiny-random/ministral-3` | 2 | 8 | 256 | 128 | 64 | 131072 | source default | false | none |

For 8B and 14B official configs, `tie_word_embeddings` is omitted in the nested text config; the source default is false. Config `torch_dtype` was not set in the sampled official JSON files; dtype must come from weights or runtime loading policy.

## 3a. Family variation traps

- `hidden_size` is not always `num_attention_heads * head_dim`. 3B has `hidden=3072` but `q_width=4096`; 14B has `hidden=5120` but `q_width=4096`. Do not infer projection widths from hidden size.
- GQA is required: `num_key_value_heads=8`, `num_attention_heads=32`, repeat factor 4.
- Q/K RoPE weight conversion is not a no-op for original Mistral checkpoints. The converter permutes Q and K projection weights with `permute_for_rope` before saving HF weights.
- All projections in source are bias-free: Q, K, V, O, gate, up, down, LM head.
- Sliding-window attention exists as a config-driven path if `sliding_window` is non-null, although sampled official configs set it to null.
- Official top-level repos are multimodal `mistral3`, not pure `ministral3`. For first text-only admission, accept nested text weights or extracted `Ministral3ForCausalLM`; route full image-text models to the `mistral3` audit.
- Instruct FP8 configs use `quantization_config.quant_method="fp8"`, `activation_scheme="static"`, and leave `lm_head` plus multimodal modules unconverted. DinoML should treat this as a loading/provider contract, not as normal dense dtype.
- Chat/tokenizer configs include image tokens such as `[IMG]`, `[IMG_BREAK]`, `[IMG_END]`, but those are processor/template ABI for the multimodal wrapper. Text-only `Ministral3ForCausalLM` just embeds token ids.
- `llama_4_scaling_beta` is ignored by generic RoPE validation but used directly in attention to multiply query states by a position-dependent scale.

## 4. Operator coverage checklist

Tensor/layout ops:

- `Embedding(vocab_size, hidden_size)` for token ids.
- `view` and `transpose` from `[B, S, width]` to `[B, H, S, D]`.
- `contiguous` after attention output transpose.
- Slice/gather on sequence dimension for `logits_to_keep`.
- Dynamic `arange` for default `position_ids` and scalar cache length offset.

Neural network primitives:

- Bias-free dense linear/GEMM:
  - Q: `Linear(hidden -> num_heads * head_dim)`.
  - K/V: `Linear(hidden -> num_kv_heads * head_dim)`.
  - O: `Linear(num_heads * head_dim -> hidden)`.
  - MLP gate/up: `Linear(hidden -> intermediate)`.
  - MLP down: `Linear(intermediate -> hidden)`.
  - LM head: `Linear(hidden -> vocab)`, bias-free.
- RMSNorm over last dimension with fp32 variance and output cast back to input dtype.
- SiLU and elementwise multiply for SwiGLU.
- Residual adds after attention and MLP.

Attention primitives:

- Causal self-attention, optional sliding-window causal mask.
- GQA repeat or native grouped attention with 32 query heads and 8 KV heads.
- Softmax over key length with fp32 accumulation in eager path.
- Attention dropout is training-only for inference because dropout is 0.0 when not training.

Position/rotary/custom math:

- YaRN RoPE parameter computation.
- RoPE apply to Q/K.
- Extra query scale: `1 + beta * log(1 + floor(position_id / original_max_position_embeddings))`.

Generation/cache ops:

- DynamicCache compatible KV cache, per layer.
- Cache update after RoPE and query scaling does not affect cached K/V. Cached keys are stored after RoPE, values are unrotated.
- Cache length controls default position ids.
- `logits_to_keep` supports integer tail slice or tensor indices.

Quantized/packed weight metadata ops:

- FP8 static quantized linear weights may appear in Instruct non-BF16 official configs. Converter maps `qscale_act` to `.activation_scale` and `qscale_weight` to `.weight_scale_inv`.
- Dense fallback should accept BF16/Base/Reasoning configs first; FP8 admission should require explicit provider support or pre-dequantized weights.

Optional/deferred heads:

- `Ministral3ForTokenClassification`, `Ministral3ForSequenceClassification`, and `Ministral3ForQuestionAnswering` are generic head wrappers. Defer for causal LM target.

## 5. Layer/block breakdown

Decoder block, repeated `num_hidden_layers` times:

```text
residual = x
x_norm = RMSNorm(x)
q = Linear_Q(x_norm).view(B, S, 32, 128).transpose(1, 2)
k = Linear_K(x_norm).view(B, S, 8, 128).transpose(1, 2)
v = Linear_V(x_norm).view(B, S, 8, 128).transpose(1, 2)
q, k = apply_rope(q, k, cos, sin)
q = q * llama4_position_scale(position_ids)
k, v = cache.update(k, v, layer_idx) if cache enabled
attn = causal_or_windowed_attention(q, k, v)
x = residual + Linear_O(attn.transpose(1, 2).reshape(B, S, 4096))

residual = x
x_norm = RMSNorm(x)
mlp = Linear_down(silu(Linear_gate(x_norm)) * Linear_up(x_norm))
x = residual + mlp
```

Shape notes:

- Attention inner width is `num_attention_heads * head_dim`, not necessarily `hidden_size`.
- O projection input width is always `q_width`; O output is `hidden_size`.
- MLP intermediate differs by model size: 9216, 14336, or 16384.
- All listed linear modules are bias-free in source.

## 6. Attention requirements

Attention type: autoregressive causal self-attention. No cross-attention in `ministral3`.

Head structure:

- MHA/GQA: GQA.
- Query heads: 32.
- KV heads: 8.
- Head dim: 128.
- Query width: 4096 for all official 3B/8B/14B text configs.
- KV width: 1024.
- Repeat factor: 4 if using eager repeated-KV implementation.

Masking:

- `create_causal_mask` when `sliding_window is None`.
- `create_sliding_window_causal_mask` when `sliding_window` is set.
- Eager path adds mask to attention scores before fp32 softmax.
- Source attention backend dispatch supports eager, SDPA, FlashAttention, and flex attention through `ALL_ATTENTION_FUNCTIONS`.

KV cache:

- Per layer cached K shape before repeat: `[B, 8, T_cache, 128]`.
- Per layer cached V shape before repeat: `[B, 8, T_cache, 128]`.
- Attention backend may consume grouped KV directly or logically repeat to `[B, 32, T_cache, 128]`.
- Cache stores keys after RoPE. Query-only `llama4_position_scale` is applied after RoPE and is not cached.
- Decode position ids default to `arange(S_new) + past_key_values.get_seq_length()`.

Packed/varlen support: no explicit packed sequence metadata in `ministral3` source. FlashAttention backends may use their own internals, but the model ABI is regular dense tensors plus mask/cache.

## 7. Position encoding and custom math

RoPE is computed once per forward from `position_ids`, then passed to every layer. YaRN is the default and official configs use `factor=16`, `original_max_position_embeddings=16384`, `beta_fast=32`, `beta_slow=1`, `mscale=1`, `mscale_all_dim=1`. Official 14B configs use `rope_theta=1e9`; 3B and 8B use `1e6`.

Concise source-equivalent math:

```python
def ministral3_rope_and_query_scale(q, k, cos, sin, position_ids, beta, original_max_pos):
    cos = cos[:, None, :, :]
    sin = sin[:, None, :, :]
    q = q * cos + rotate_half(q) * sin
    k = k * cos + rotate_half(k) * sin
    scale = 1 + beta * log(1 + floor(position_ids / original_max_pos))
    q = q * scale[:, None, :, None].to(q.dtype)
    return q, k
```

YaRN inverse frequencies can be precomputed per config/device for static maximum context, but the source computes cos/sin from runtime `position_ids`. For decode, a single-position cos/sin path is enough if cache position is known. The extra query scale depends only on `position_ids`, `llama_4_scaling_beta`, and `original_max_position_embeddings`.

## 8. Preprocessing and input packing

Text-only runtime ABI:

- Inputs: `input_ids [B, S]` or `inputs_embeds [B, S, hidden]`, exactly one required.
- Optional `attention_mask`; converted by Transformers mask utility into additive causal mask.
- Optional `position_ids [B, S]`; otherwise generated from cache length.
- Optional `past_key_values`; if `use_cache` is true and no cache is supplied, source creates a `DynamicCache`.

Tokenizer/generation snapshot from `mistralai/Ministral-3-8B-Instruct-2512`:

- `bos_token_id=1`, `eos_token_id=2`, `pad_token_id=11`, `max_length=262144`.
- Chat template uses `[SYSTEM_PROMPT]`, `[INST]`, tool-call tags, and multimodal `[IMG]` placeholders.
- Tokenizer/processor config has `processor_class="PixtralProcessor"` and image processor settings, but those affect the top-level multimodal wrapper, not text decoder operators.

GPU graph boundary recommendation: keep tokenizer/chat template and multimodal placeholder construction outside the first DinoML text graph. The first graph should accept already tokenized text ids and masks.

## 9. Graph rewrite / lowering opportunities

### Rewrite: split Q/K/V linears -> packed QKV GEMM

Source pattern:

```text
q = Linear_Q(x)
k = Linear_K(x)
v = Linear_V(x)
```

Replacement:

```text
packed = GEMM(x, concat_rows(Wq, Wk, Wv).T)
q, k, v = split(packed, [q_width, kv_width, kv_width])
```

Preconditions:

- All three projections are bias-free.
- Same input tensor and dtype.
- Weight rows remain in all-Q, all-K, all-V order after any required HF conversion.
- Split sizes come from `num_attention_heads * head_dim`, `num_key_value_heads * head_dim`, `num_key_value_heads * head_dim`.

Failure cases: source-coupled FP8 weights without provider support; mixed device residency; checkpoints that have not had Q/K RoPE permutation applied.

Parity test sketch: compare Q/K/V tensors before RoPE for one block across 3B, 8B, 14B shapes.

### Rewrite: GQA repeat_kv elimination

Source pattern:

```text
k = expand/reshape repeat from 8 KV heads to 32 query heads
scores = q @ k.T
```

Replacement: native grouped-query attention kernel consuming `[B, 32, S_q, 128]`, `[B, 8, S_k, 128]`, repeat factor 4.

Preconditions: `num_attention_heads % num_key_value_heads == 0`; identical attention mask semantics; backend preserves fp32 softmax behavior or accepted tolerance.

Failure cases: backend that only accepts repeated dense KV, sliding-window mask not implemented, attention output requests requiring dense attention weights.

### Rewrite: RMSNorm fusion

Source pattern:

```text
x_fp32 = x.to(float32)
variance = mean(x_fp32 * x_fp32, dim=-1, keepdim=True)
y = weight * (x_fp32 * rsqrt(variance + eps)).to(input_dtype)
```

Replacement: fused RMSNorm kernel.

Preconditions: normalize last dimension, one weight vector, no bias, eps from config.

Parity test sketch: random fp32/fp16/bf16 hidden states at hidden sizes 3072, 4096, 5120.

### Rewrite: SwiGLU MLP fusion

Source pattern:

```text
down_proj(silu(gate_proj(x)) * up_proj(x))
```

Replacement: packed gate/up GEMM -> fused SiLU multiply -> down GEMM.

Preconditions: bias-free gate/up, same input, identical intermediate size, activation `silu`.

Failure cases: different activation in future configs, FP8 scales unavailable for packed operation.

### Rewrite: last-token-only logits

Source pattern: `lm_head(hidden_states[:, slice_indices, :])`.

Replacement: when `logits_to_keep=1`, run LM head only on last token hidden state.

Preconditions: caller does not request full logits or tensor index selection.

Failure cases: loss computation, perplexity/evaluation needing all sequence logits, arbitrary tensor `logits_to_keep`.

## 10. Kernel fusion candidates

Highest priority:

- RMSNorm: two per layer plus final norm; required for all runtime paths.
- GQA FlashAttention with RoPE-ready Q/K and KV cache: dominant prefill/decode cost.
- QKV packed projection with split: reduces launch overhead and improves GEMM throughput.
- RoPE plus query scale: small but sits on the critical attention path; preserve position-dependent scale.
- SwiGLU: gate/up packed GEMM plus fused SiLU multiply.

Medium priority:

- Residual add fused with output projection or following norm where memory planning allows.
- LM head last-token path and vocab GEMM profiling, especially for 131072 vocab.
- Sliding-window attention kernel, guarded by non-null `sliding_window`.
- FP8 static linear provider if targeting Instruct non-BF16 official repos.

Lower priority:

- Generic classification/QA/token heads.
- Attention weight materialization for `output_attentions`.
- Full logits for every prefill token outside validation/perplexity use cases.

## 11. Runtime staging plan

Stage 1: parse `Ministral3Config` and nested `Mistral3Config.text_config`; reject full multimodal wrapper unless routed to `mistral3`.

Stage 2: load dense BF16/FP16/FP32 weights for one layer and run block parity without cache. Include 3B/14B projection-width traps.

Stage 3: full prefill parity for text-only `Ministral3ForCausalLM` with eager dense attention and full logits or last-token logits.

Stage 4: decode parity with per-layer KV cache, confirming cached K after RoPE and unrotated V.

Stage 5: replace eager attention with native GQA FlashAttention/SDPA-compatible backend and add sliding-window rejection or support.

Stage 6: add QKV packing, RMSNorm, RoPE/query-scale, SwiGLU, and last-token-logit fusions.

Stage 7: admit FP8 static quantized checkpoints with explicit scale tensor loading and provider support, or route to dense dequant fallback.

Initially stubbable: chat template, sampling, tool-call formatting, multimodal image placeholders, generic classification/QA heads, FP8 provider.

## 12. Parity and validation plan

- Config parsing tests for source defaults, official nested text configs, and omitted `tie_word_embeddings`.
- Weight-shape tests for 3B, 8B, 14B:
  - Q: `[4096, hidden]`.
  - K/V: `[1024, hidden]`.
  - O: `[hidden, 4096]`.
  - gate/up: `[intermediate, hidden]`.
  - down: `[hidden, intermediate]`.
- Unit parity for RMSNorm, YaRN RoPE cos/sin, RoPE apply, query scale, repeat-free GQA attention, and SwiGLU.
- Single-layer parity with random dense weights and masks.
- After-N-layer parity at small sequence lengths using `tiny-random/ministral-3`.
- Prefill logits parity on BF16/FP32 for one official-size config with random weights.
- Decode parity for token-by-token generation against Transformers using identical cache positions.
- Sliding-window tests if admitted: compare masks and logits around window boundaries.
- FP8 loading tests if admitted: verify `activation_scale` and `weight_scale_inv` ownership and dense fallback dequantization.

Suggested tolerances: fp32 `1e-4` absolute for block/logit tests; fp16/bf16 `1e-2` to `3e-2` for attention-heavy paths unless using identical kernels and accumulation order.

## 13. Performance probes

- Prefill throughput by sequence length: 1k, 4k, 16k, 64k, 262k where memory permits.
- Decode tokens/sec by batch and cache length.
- KV cache memory per layer and total memory for 3B/8B/14B.
- Attention backend comparison: eager repeated KV, native GQA, FlashAttention, sliding-window when enabled.
- QKV packed versus split projection launch count and runtime.
- RMSNorm and SwiGLU fusion microbenchmarks by hidden/intermediate size.
- LM head full-sequence versus last-token-only logits with vocab 131072.
- FP8 load/dequant/provider comparison for Instruct non-BF16 configs.
- GGUF or other quantized weight loading probes if using converted community checkpoints; separate from source dense parity.

## 14. Skip/defer list

- Training, labels/loss, dropout, and gradient checkpointing.
- Full multimodal `Mistral3ForConditionalGeneration`, vision tower, projector, image processor, and `[IMG]` embedding stitch.
- Generic token classification, sequence classification, and question answering heads.
- Beam search, speculative decoding, tool-call rendering, and chat-template validation beyond token ids.
- FP8 static execution until scale tensor ABI and provider behavior are explicit.
- Sliding-window attention if no target config sets `sliding_window`.
- Attention weight outputs for production paths.

## 15. Final implementation checklist

- [ ] Parse `Ministral3Config` and nested `Mistral3Config.text_config`.
- [ ] Reject or route top-level multimodal `mistral3` models outside the text-only path.
- [ ] Load embeddings, decoder weights, norm, LM head, and tied embedding aliases.
- [ ] Validate non-hidden attention widths for 3B and 14B.
- [ ] Implement RMSNorm last-dim kernel.
- [ ] Implement bias-free linear/GEMM coverage for Q/K/V/O, MLP, and LM head.
- [ ] Implement YaRN RoPE parameter generation and RoPE apply.
- [ ] Implement `llama_4_scaling_beta` query scale.
- [ ] Implement causal GQA attention prefill.
- [ ] Implement KV cache update and decode attention.
- [ ] Implement optional sliding-window mask or reject non-null `sliding_window`.
- [ ] Implement SwiGLU MLP.
- [ ] Implement `logits_to_keep=1` last-token LM head optimization.
- [ ] Add parity tests for one layer, prefill logits, and decode tokens.
- [ ] Add performance probes for attention, MLP, LM head, and KV cache memory.
- [ ] Add FP8 static quantization admission or explicit rejection for Instruct non-BF16 configs.
