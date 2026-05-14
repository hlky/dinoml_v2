# BitNet Transformers Audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: microsoft/bitnet-b1.58-2B-4T
Config source: HF config.json, tokenizer_config.json, generation_config.json, HF model API metadata
Source files inspected:
- X:/H/transformers/src/transformers/models/bitnet/configuration_bitnet.py
- X:/H/transformers/src/transformers/models/bitnet/modeling_bitnet.py
- X:/H/transformers/src/transformers/models/bitnet/modular_bitnet.py
- X:/H/transformers/src/transformers/integrations/bitnet.py
- X:/H/transformers/src/transformers/quantizers/quantizer_bitnet.py
- X:/H/transformers/src/transformers/utils/quantization_config.py
- X:/H/transformers/src/transformers/modeling_rope_utils.py
- X:/H/transformers/src/transformers/activations.py
Any missing files or assumptions: GGUF repo has no config.json; report uses HF API GGUF metadata and official dense/quantized configs.
```

Primary HF links:

- [microsoft/bitnet-b1.58-2B-4T](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T)
- [microsoft/bitnet-b1.58-2B-4T-bf16](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-bf16)
- [microsoft/bitnet-b1.58-2B-4T-gguf](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-gguf)
- [Transformers BitNet docs](https://huggingface.co/docs/transformers/en/model_doc/bitnet)

Local snapshots written under `agents/plans/transformers/bitnet/_sources/`:

- `microsoft_bitnet-b1.58-2B-4T_config.json`
- `microsoft_bitnet-b1.58-2B-4T-bf16_config.json`
- `microsoft_bitnet-b1.58-2B-4T_generation_config.json`
- `microsoft_bitnet-b1.58-2B-4T_tokenizer_config.json`
- `microsoft_bitnet-b1.58-2B-4T_special_tokens_map.json`
- `api_microsoft_bitnet-b1.58-2B-4T.json`
- `api_microsoft_BitNet-b1.58-2B-4T-gguf.json`
- `api_search_bitnet.json`

`modeling_bitnet.py` is generated from `modular_bitnet.py`; future source edits should target `modular_bitnet.py`. The generated file is still the concrete runtime source inspected for exact forward paths.

## 2. High-level architecture

BitNet in this source basis is a text-only decoder causal LM:

```text
tokenizer/chat template -> input_ids/attention_mask
  -> token embedding
  -> N x decoder block with RMSNorm, GQA causal self-attention, RoPE, gated MLP
  -> final RMSNorm
  -> tied/untied LM head
  -> logits/sampling
```

Primary DinoML target: `BitNetForCausalLM` inference with prefill and autoregressive decode.

Stage decomposition:

- CPU/data pipeline: tokenizer, chat template, BOS/EOS/EOT handling, generation parameters.
- GPU/runtime prefill: embeddings, decoder stack, causal attention with RoPE, logits.
- GPU/runtime decode: one or more new tokens plus per-layer KV cache update.
- Loading/provider stage: BitNet quantizer may replace dense `nn.Linear` modules with BitNet linear variants before weight load.

The native model body is close to Llama/Gemma-style dense decoder code, but BitNet differs in two high-impact places: attention output is RMS-normalized before `o_proj`, and the MLP applies an extra RMSNorm after the gated activation product and before `down_proj`.

## 3. Important config dimensions

Source default dimensions from `BitNetConfig`:

| Field | Default |
|---|---:|
| `model_type` | `bitnet` |
| `vocab_size` | 128256 |
| `hidden_size` | 2560 |
| `intermediate_size` | 6912 |
| `num_hidden_layers` | 30 |
| `num_attention_heads` | 20 |
| `num_key_value_heads` | 5 |
| `head_dim` | inferred as 128 |
| `num_key_value_groups` | 4 |
| `hidden_act` | `relu2` |
| `max_position_embeddings` | 2048 default, 4096 official checkpoint |
| `rope_theta` / default theta | 500000.0 |
| `rms_norm_eps` | 1e-5 |
| `attention_bias` | false |
| `attention_dropout` | 0.0 |
| `use_cache` | true |
| `tie_word_embeddings` | false default, true official checkpoint |
| `bos_token_id` | 128000 |
| `eos_token_id` | 128001 default; generation config also stops on 128009 |

Representative checkpoint sweep:

| Checkpoint | Source | Operator-significant fields |
|---|---|---|
| `microsoft/bitnet-b1.58-2B-4T` | `config.json`, API | `max_position_embeddings=4096`, `torch_dtype=bfloat16`, `tie_word_embeddings=true`, `quantization_config={quant_method=bitnet, linear_class=autobitlinear, quantization_mode=offline}`. API safetensors metadata reports BF16 and U8 tensors. |
| `microsoft/bitnet-b1.58-2B-4T-bf16` | `config.json` | Same dimensions, but `quantization_mode=online`; this implies online/fake weight quantization through `AutoBitLinear` rather than loading fixed packed U8 weights. |
| `microsoft/bitnet-b1.58-2B-4T-gguf` | HF model API | No `config.json` sibling. API GGUF metadata reports `architecture=bitnet-b1.58`, `context_length=4096`, file `ggml-model-i2_s.gguf`, and chat template/BOS/EOS metadata. Treat as a GGUF loading target, not a native HF config target. |

No official small/debug native `model_type=bitnet` checkpoint was found in the HF search snapshot. Many older/community BitNet-named repositories are `llama`, `mistral`, or custom-code models and are out of scope for this native BitNet family audit.

## 3a. Family variation traps

- `modeling_bitnet.py` starts from dense `nn.Linear`; BitNet quantized behavior is injected by the HF quantizer before weight loading.
- Official configs include old `auto_map` remote-code entries, but the inspected in-library source implements `model_type=bitnet` natively. DinoML should scope this report to the native source and reject unknown remote-code divergences until separately audited.
- `head_dim` is not a declared config field in defaults; source computes `hidden_size // num_attention_heads` if absent. Do not infer projection widths only from `hidden_size`; read or compute `head_dim`.
- `num_key_value_heads=5 < num_attention_heads=20`, so this is GQA. KV cache stores 5 heads before repeat, not 20 heads.
- `hidden_act=relu2` is squared ReLU, not SiLU/GELU. The MLP is gated, then sub-normalized.
- Attention has `attn_sub_norm` after attention output reshape and before `o_proj`, unlike Llama/Qwen dense decoders.
- MLP has `ffn_sub_norm` on `act(gate_proj(x)) * up_proj(x)` before `down_proj`.
- `BitNetConfig` default has `tie_word_embeddings=false`, but official checkpoint sets `true`. Weight tying must preserve alias semantics across quantized/dense loading.
- Official safetensors config has `quantization_config.linear_class=autobitlinear`. The quantized storage/runtime path depends on `linear_class` and `quantization_mode`.
- Offline `autobitlinear` can load packed U8 weights and unpack them to dense-like module weights during load; `bitlinear` keeps packed U8 weights resident and unpacks every forward in the Python fallback.
- `BitLinear` packs along output-feature rows in groups of 4. All official out-feature dimensions are divisible by 4; DinoML should still validate this.
- GGUF is a separate source-coupled packed format. The HF GGUF repo has no `config.json`; use GGUF metadata plus a GGUF reader/provider contract.
- RoPE is default full-head RoPE with theta 500000 for the official checkpoint. The generic Transformers RoPE machinery can admit other `rope_parameters`, but no official BitNet config variant requiring longrope/yarn/llama3 was observed.
- No NCHW/NHWC or vision layout translation issues apply; this is a token decoder.

## 4. Operator coverage checklist

Tensor/layout ops:

- Embedding lookup: `[B, S] -> [B, S, 2560]`.
- Views/transposes: Q/K/V reshape to `[B, S, heads, 128]`, transpose to `[B, heads, S, 128]`, attention output transpose back, contiguous/reshape.
- Slice/index for `logits_to_keep`: `hidden_states[:, slice_indices, :]`.
- Cache append/update per layer.
- Causal mask creation/addition, including prefix length from cache.

Neural network primitives:

- RMSNorm over last dim for hidden size 2560 and intermediate size 6912.
- Linear projections:
  - `q_proj`: `Linear(2560 -> 2560)`, no bias by official config.
  - `k_proj`: `Linear(2560 -> 640)`, no bias.
  - `v_proj`: `Linear(2560 -> 640)`, no bias.
  - `o_proj`: `Linear(2560 -> 2560)`, no bias.
  - `gate_proj`: `Linear(2560 -> 6912)`, no bias.
  - `up_proj`: `Linear(2560 -> 6912)`, no bias.
  - `down_proj`: `Linear(6912 -> 2560)`, no bias.
  - `lm_head`: `Linear(2560 -> 128256)`, no bias, tied in official config.
- Elementwise residual add.
- `relu2(x) = relu(x) ** 2`.
- Gated product: `relu2(gate_proj(x)) * up_proj(x)`.

Attention primitives:

- Causal self-attention.
- GQA repeat of K/V from 5 KV heads to 20 query heads.
- RoPE on Q and K before cache update.
- Matmul QK^T scaled by `head_dim ** -0.5`.
- Additive causal/padding mask.
- Softmax in fp32 then cast back to query dtype in eager path.
- Matmul probabilities by V.
- Optional FlashAttention, SDPA, or Flex attention via `ALL_ATTENTION_FUNCTIONS`.

Quantized/packed weight metadata ops:

- BitNet `pack_weights` / `unpack_weights` for ternary values encoded as 2-bit codes in `uint8`.
- `BitLinear.weight`: packed U8 buffer shaped `[out_features / 4, in_features]`.
- `BitLinear.weight_scale`: scalar buffer.
- Optional `BitLinear.bias`: dense buffer, not used by official attention/MLP config but possible if `attention_bias=true`.
- `AutoBitLinear.weight_scale`: scalar buffer in offline mode.
- Offline deserialization detects packed weight if `weight.shape[0] * 4 == module.out_features` and unpacks.
- Activation quantization:
  - `BitLinear.activation_quant`: int8 activation plus per-token scale.
  - `AutoBitLinear.ActQuant`: fake-quantizes activation and returns dequantized dtype tensor.
- Online `AutoBitLinear.WeightQuant`: fake-quantizes dense weights each forward.
- Safe dense fallback: unpack packed U8 to dense ternary weights and run normal GEMM/linear, applying scale exactly per selected class.

Position/rotary ops:

- Default RoPE with `inv_freq = 1 / theta ** (arange(0, head_dim, 2) / head_dim)`.
- Cos/sin computed in float32 and cast to hidden dtype.
- `rotate_half` split-half convention.

Generation/cache ops:

- `DynamicCache` initialization when `use_cache` and no cache provided.
- Cache stores K/V after RoPE and before repeat-kv expansion.
- Position IDs start at `past_key_values.get_seq_length()`.
- Generation config uses sampling defaults: `temperature=0.6`, `top_p=0.9`, `max_length=4096`, EOS ids `[128001, 128009]`.

Preprocessing-coupled ops:

- Tokenizer fast path, no processor/image/audio branch.
- Chat template emits role-prefixed text ending with `<|eot_id|>` and optional `Assistant: ` generation prompt.
- `model_input_names`: `input_ids`, `attention_mask`.

## 5. Layer/block breakdown

Decoder model:

```text
input_ids -> embed_tokens -> hidden_states [B, S, 2560]
position_ids = arange(S) + past_seen_tokens
position_embeddings = rotary_emb(hidden_states, position_ids)
causal_mask = create_causal_mask(...)
repeat 30 decoder layers
final hidden_states = RMSNorm(hidden_states)
lm_head(hidden_states[:, logits_to_keep, :]) -> logits [B, T, 128256]
```

Decoder block, repeated 30 times:

```text
residual = x
x = RMSNorm_2560(x)
q = Linear(2560 -> 2560)(x).view(B, S, 20, 128).transpose(1, 2)
k = Linear(2560 -> 640)(x).view(B, S, 5, 128).transpose(1, 2)
v = Linear(2560 -> 640)(x).view(B, S, 5, 128).transpose(1, 2)
q, k = RoPE(q, k, cos, sin)
k, v = cache.update(k, v, layer_idx)  # when cache is present
attn = causal_attention(q, k, v, mask, scale=1/sqrt(128))
attn = attn.transpose(1, 2).reshape(B, S, 2560)
attn = RMSNorm_2560(attn)
x = residual + Linear(2560 -> 2560)(attn)

residual = x
x = RMSNorm_2560(x)
g = Linear(2560 -> 6912)(x)
u = Linear(2560 -> 6912)(x)
m = RMSNorm_6912(relu(g) ** 2 * u)
x = residual + Linear(6912 -> 2560)(m)
```

All listed projections are bias-free for the official checkpoint. `attention_bias=true` is source-supported for Q/K/V/O only; MLP projections are always bias-free in this source.

## 6. Attention requirements

Required attention variant: causal decoder self-attention with GQA.

| Property | BitNet requirement |
|---|---|
| Causality | causal |
| Cross-attention | none |
| Q heads | 20 |
| KV heads | 5 |
| KV repeat | repeat each KV head 4 times for eager dense attention |
| Head dim | 128 |
| Projection widths | Q/O 2560, K/V 640 |
| RoPE placement | apply to Q/K before cache update |
| Cache storage | per-layer K/V in `[B, 5, cache_seq, 128]` logical shape before repeat |
| Masking | additive causal/padding mask from `create_causal_mask` |
| Softmax | eager path computes softmax with `dtype=torch.float32`, then casts to query dtype |
| Dropout | 0.0 in inference |
| Backends | source advertises FlashAttention, SDPA, Flex attention; eager fallback exists |

Prefill can use a standard GQA causal attention backend if it preserves source math order: RoPE, cache update, backend attention with scale, then attention-sub-RMSNorm before output projection.

Decode cache ABI:

- Key tensor append input per layer: `[B, 5, new_tokens, 128]`.
- Value tensor append input per layer: `[B, 5, new_tokens, 128]`.
- Stored keys are already RoPE-rotated.
- Repeat to 20 heads is an attention-kernel concern, not a cache storage concern.
- `position_ids` derive from current query length plus existing cache length.

## 7. Position encoding and custom math

Default RoPE source math:

```python
def bitnet_default_inv_freq(head_dim=128, rope_theta=500000.0):
    return 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
```

Forward RoPE:

```python
def bitnet_rope(position_ids, inv_freq, x_dtype):
    freqs = (inv_freq[None, :, None].float() @ position_ids[:, None, :].float()).transpose(1, 2)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(x_dtype), emb.sin().to(x_dtype)
```

Apply RoPE:

```python
def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_bitnet_rope(q, k, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)
```

Custom math:

- `RMSNorm(x) = x * rsqrt(mean(x^2, dim=-1, keepdim=True) + eps) * weight`; norm computation upcasts input to fp32 and casts normalized values back before multiplying by weight.
- `relu2(x) = square(relu(x))`.
- Attention output sub-norm and FFN intermediate sub-norm are source-critical, not optional cleanups.
- BitNet quantization math is loading/provider-sensitive. See section 9 rewrites for dense fallback equations.

RoPE cos/sin can be precomputed for static context windows, but dynamic position IDs with cache offsets must be supported. Dynamic RoPE update decorators are present for advanced RoPE types; no official BitNet checkpoint requiring them was found.

## 8. Preprocessing and input packing

Runtime model inputs:

- `input_ids`: `[B, S]` int token IDs.
- `attention_mask`: optional mask consumed by `create_causal_mask`.
- `position_ids`: optional; generated as `[1, S]` if omitted.
- `inputs_embeds`: optional alternative to `input_ids`, mutually exclusive with `input_ids`.

Tokenizer/config coupling:

- BOS token string: `<|begin_of_text|>`, id 128000.
- `special_tokens_map.json` EOS token string: `<|end_of_text|>`.
- `tokenizer_config.json` EOS token string: `<|eot_id|>`, id 128009 in added tokens.
- `generation_config.json` EOS ids: `[128001, 128009]`.
- Chat template formats each message as `Role: content<|eot_id|>` and appends `Assistant: ` when `add_generation_prompt` is true.
- Tokenizer `model_max_length` is a very large sentinel; effective model/generation context is 4096 for official configs and generation config.

No multimodal packing, placeholder scatter, image/audio processor, `cu_seqlens`, token type IDs, or position-grid metadata is required.

## 9. Graph rewrite / lowering opportunities

### Rewrite: dense BitNet block canonicalization

Source pattern:

```text
RMSNorm -> Q/K/V linears -> RoPE -> GQA causal attention -> RMSNorm -> O linear -> residual
RMSNorm -> gate/up linears -> relu2 -> mul -> RMSNorm -> down linear -> residual
```

Replacement pattern:

```text
canonical_decoder_block(bitnet_subnorm_attention=True, bitnet_subnorm_mlp=True)
```

Preconditions:

- `hidden_act == "relu2"`.
- `attention_bias == false` for official fast path; bias variant needs separate coverage.
- `num_attention_heads % num_key_value_heads == 0`.
- `head_dim * num_attention_heads == q_proj.out_features`.
- `o_proj.in_features == q_proj.out_features`.

Failure cases:

- Missing `attn_sub_norm` or `ffn_sub_norm`.
- Alternative activation.
- Non-default attention bias if unsupported.

Parity test sketch: compare one decoder layer with random BF16/FP32 tensors, with and without cache, against HF eager attention.

### Rewrite: QKV projection packing for GQA

Source pattern:

```text
q_proj(x), k_proj(x), v_proj(x)
```

Replacement:

```text
one packed GEMM producing [Q:2560 | K:640 | V:640], then split in that order
```

Preconditions:

- Same input `x`.
- Same dtype/device.
- All three modules have compatible BitNet quantization class and materialization policy.
- Bias either absent for all or packable in the same Q, K, V order.

Weight transform:

```python
packed_weight = concat([q_proj.weight, k_proj.weight, v_proj.weight], dim=0)
packed_bias = None  # official config
```

Quantized failure cases:

- Packed U8 BitNet weights must preserve each source module's scale/metadata.
- A fused quantized provider must know the split row boundaries: 2560, 640, 640.
- Do not concatenate raw packed rows unless provider understands BitNet row-packing and scale ownership.

### Rewrite: BitLinear packed fallback to dense GEMM

Source pattern:

```text
uint8 packed weight [out/4, in], scalar weight_scale, optional bias
```

Dense fallback:

```python
codes = unpack_2bit_along_output_rows(packed)  # codes 0,1,2 => -1,0,1
w = (codes.to(dtype) - 1)
y = linear(activation_quant_or_input, w, bias)
```

Preconditions:

- `out_features % 4 == 0`.
- Packed storage first dimension times 4 equals logical out features.
- Codes should be in `{0,1,2}`; code `3` is not produced by HF packer and should be rejected or treated as invalid.
- Apply scale exactly according to selected class:
  - `BitLinear`: output divides by `input_scale * weight_scale`.
  - offline `AutoBitLinear`: output multiplies by `weight_scale` after dense linear.
  - online `AutoBitLinear`: dense weights are fake-quantized each forward.

Failure cases:

- Unknown `linear_class`, `quantization_mode`, per-channel scale extensions, or GGUF tensor type without a GGUF provider.

### Rewrite: MLP gate/up fusion

Source pattern:

```text
relu2(gate_proj(x)) * up_proj(x) -> RMSNorm_6912 -> down_proj
```

Replacement:

```text
dual GEMM gate/up -> fused relu2_mul -> RMSNorm -> GEMM down
```

Preconditions:

- Both projections share input and shape `[B*S, 2560] -> [B*S, 6912]`.
- Same quantization provider class and compatible scales.
- Preserve RMSNorm after the product; do not fold it away.

Parity test sketch: run MLP-only parity on random inputs over FP32, BF16, and quantized dense fallback.

### Rewrite: last-token-only logits

Source pattern:

```text
logits = lm_head(hidden_states[:, slice_indices, :])
```

Replacement:

```text
when logits_to_keep == 1 during decode, run LM head only on last token
```

Preconditions:

- Caller does not request full sequence logits or loss.
- Slice is trailing and contiguous.
- Weight tying/quantization state for `lm_head` is preserved.

## 10. Kernel fusion candidates

Highest priority:

- BitNet linear provider/load path: packed U8/GGUF ternary metadata, scales, dense fallback, and eventually direct packed GEMM.
- GQA causal attention with RoPE and KV cache: standard decoder bottleneck, but cache stores 5 KV heads while attention consumes 20 query heads.
- RMSNorm kernels for hidden size 2560 and intermediate size 6912, including sub-norm placements.
- MLP dual projection plus `relu2 * up` fusion; this is a repeated block-specific hotspot.

Medium priority:

- QKV packed projection with split `[Q,K,V] = [2560,640,640]`.
- Attention output RMSNorm + output projection scheduling; do not assume Llama ordering.
- Last-token-only LM head, especially because vocab is 128256.
- RoPE precompute/cache by position window and dtype.

Lower priority:

- Training/QAT-specific online quantization kernels.
- Attention bias variants.
- FlexAttention-specific paths unless configs demand them.

## 11. Runtime staging plan

Stage 1: parse native BitNet config and load official BF16/dense-equivalent weights with quantization disabled or fully unpacked to dense tensors. Validate embeddings, one block, and logits.

Stage 2: implement quantization-aware loading contract for `quant_method=bitnet`, including `autobitlinear` offline/online routing and safe dense fallback for packed U8.

Stage 3: run prefill parity for the full decoder using eager/dense attention and unpacked weights.

Stage 4: implement decode with per-layer GQA KV cache storing `[B, 5, T, 128]`.

Stage 5: enable optimized GQA attention backend preserving RoPE-before-cache and attention-sub-RMSNorm-after-attention order.

Stage 6: add BitNet-specific provider paths: packed U8 linear, GGUF `i2_s` loading, and dequantize-before-GEMM or direct ternary GEMM experiments.

Stage 7: add graph fusions for QKV, MLP dual GEMM, last-token logits, and continuous batching.

Initially stub/defer sampling controllers, online QAT, and direct packed kernels by unpacking to dense BF16/FP32 references.

## 12. Parity and validation plan

- Unit test `relu2`, `RMSNorm_2560`, `RMSNorm_6912`, RoPE, and repeat-kv against HF source.
- Unit test BitNet pack/unpack round trip for row counts divisible and non-divisible by 4; official fast path can require divisible by 4.
- Test dense fallback for `BitLinear`, offline `AutoBitLinear`, and online `AutoBitLinear` separately because their scale placement differs.
- Single attention-layer parity:
  - no cache prefill `[B=1,S=7]`;
  - decode with existing cache length and one new token;
  - padding mask plus causal mask.
- Single decoder block parity in FP32 and BF16.
- Full prefill logits parity on short prompts, full-sequence logits and `logits_to_keep=1`.
- Decode token parity for several generated steps with fixed sampling disabled first.
- GGUF load parity: compare GGUF `i2_s` dequantized tensors to safetensors/native output where a trusted converter is available.

Suggested tolerances:

- FP32 dense: `rtol=1e-4`, `atol=1e-5` for block/logits smoke tests.
- BF16 dense/unpacked: `rtol=2e-2`, `atol=2e-2`, tighten per operator where stable.
- Quantized fallback: compare exact intermediate unpacking where possible; logits tolerance should be calibrated against HF quantized execution.

No DinoML tests were run for this docs-only audit.

## 13. Performance probes

- Weight load time and peak memory: offline U8 unpack-to-dense vs packed-resident vs GGUF load.
- Per-linear provider comparison: dense BF16 GEMM, unpack-on-load GEMM, unpack-each-forward fallback, direct ternary kernel if available.
- Prefill throughput sweep: batch size, sequence length 128/512/2048/4096.
- Decode tokens/sec sweep: batch size and active cache length.
- KV cache memory: 30 layers x 2 tensors x `[B, 5, T, 128]` x dtype.
- Attention backend comparison: eager, SDPA, FlashAttention, DinoML GQA kernel.
- MLP fusion comparison: separate gate/up/down vs fused dual projection and fused relu2/mul/RMSNorm.
- Last-token logits probe: full sequence LM head vs trailing-token LM head.
- GGUF probe: `i2_s` dequantize-before-GEMM vs direct packed/ternary linear.

## 14. Skip/defer list

- Training and gradient checkpointing.
- Online QAT behavior as a first optimized path; preserve dense fallback only.
- Beam search and advanced generation controllers.
- FlexAttention unless needed by a concrete deployment.
- Attention bias variants until a checkpoint requires `attention_bias=true`.
- LongRoPE/Yarn/Llama3 RoPE variants unless a native BitNet config requires them.
- CPU/disk device maps for BitNet quantized models; HF quantizer rejects CPU/disk in multi-device maps.
- Community BitNet-named Llama/Mistral/OLMo repositories; audit under their actual model families.

## 15. Final implementation checklist

- [ ] Parse `BitNetConfig`, including legacy `rope_theta` -> `rope_parameters` effective defaults.
- [ ] Load tokenizer/generation metadata for BOS/EOS/EOT and chat-template parity.
- [ ] Load dense/unpacked BF16 weights for first parity.
- [ ] Preserve tied embedding/LM head aliasing for official checkpoint.
- [ ] Implement RMSNorm and `relu2`.
- [ ] Implement BitNet RoPE with theta 500000 and cache-position offsets.
- [ ] Implement GQA causal attention with KV cache stored before repeat expansion.
- [ ] Implement attention output sub-RMSNorm before `o_proj`.
- [ ] Implement MLP gated `relu2 * up` plus intermediate RMSNorm.
- [ ] Implement `quant_method=bitnet` load admission.
- [ ] Implement packed U8 row-unpack contract and validate `out_features % 4 == 0`.
- [ ] Implement offline `AutoBitLinear` dense fallback with `weight_scale` placement.
- [ ] Implement `BitLinear` fallback with int8 activation scale and packed weight scale placement.
- [ ] Add GGUF `bitnet-b1.58`/`i2_s` metadata admission as a separate provider contract.
- [ ] Add one-block, prefill, and decode parity tests.
- [ ] Add performance probes for packed/dequantized linear and GQA decode.
