# OLMo2 Transformers audit for DinoML v2

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model family: olmo2
Primary runtime target: causal language modeling, prefill and autoregressive decode
Config source: official Hugging Face config.json files, snapshotted under agents/plans/transformers/olmo2/_sources/
```

Source files inspected:

- `X:/H/transformers/src/transformers/models/olmo2/modeling_olmo2.py`
- `X:/H/transformers/src/transformers/models/olmo2/configuration_olmo2.py`
- `X:/H/transformers/src/transformers/models/olmo2/modular_olmo2.py`
- `X:/H/transformers/src/transformers/models/olmo2/convert_olmo2_weights_to_hf.py`
- OLMo comparison basis: `src/transformers/models/olmo/modeling_olmo.py`, `configuration_olmo.py`
- Shared infrastructure: `cache_utils.py`, `masking_utils.py`, `integrations/sdpa_attention.py`, `integrations/flash_attention.py`, `integrations/flex_attention.py`, `modeling_layers.py`, `modeling_rope_utils.py`, `configuration_utils.py`

Representative configs inspected:

- `allenai/OLMo-2-1124-7B`
- `allenai/OLMo-2-1124-7B-Instruct`
- `allenai/OLMo-2-1124-13B`
- `allenai/OLMo-2-1124-13B-Instruct`
- `allenai/OLMo-2-0325-32B`
- `allenai/OLMo-2-0325-32B-Instruct`
- `allenai/OLMo-2-1124-7B-RM`

Hugging Face URLs:

- `https://huggingface.co/allenai/OLMo-2-1124-7B/resolve/main/config.json`
- `https://huggingface.co/allenai/OLMo-2-1124-13B/resolve/main/config.json`
- `https://huggingface.co/allenai/OLMo-2-0325-32B/resolve/main/config.json`

Any missing files or assumptions:

- No official tiny/debug OLMo2 checkpoint was found in the inspected official `allenai` set. First DinoML tests should use synthetic configs or one-layer weight fixtures.
- `modeling_olmo2.py` and `configuration_olmo2.py` are generated from `modular_olmo2.py`; the modular file is authoritative for future upstream source edits.
- Tokenizer files were not audited beyond special-token and vocab fields in `config.json`, because tokenizer logic does not change the model graph.

## 2. High-level architecture

OLMo2 is a text-only decoder-only Transformer for causal language modeling. It is close to OLMo/Llama-style decoder code, but with important norm placement differences:

```text
token ids -> token embedding -> repeated OLMo2 decoder blocks -> final RMSNorm -> LM head -> logits
```

Generation dataflow:

```text
CPU tokenizer -> input_ids/attention_mask -> prefill decoder -> KV cache -> single-token decode loop -> logits/sampling
```

Stage decomposition:

- CPU/data pipeline: tokenization, chat template, padding, attention mask creation inputs.
- GPU/runtime prefill: embedding lookup, position id creation or ingestion, RoPE cos/sin generation, decoder blocks, optional all-token logits.
- GPU/runtime decode: one or more new tokens, cached K/V per layer, causal mask handling, last-token logits.
- Independently stageable pieces: embedding plus one block parity, RoPE parity, attention with cache, MLP SwiGLU, final logits.

Primary target should be `Olmo2ForCausalLM`. `Olmo2Model` is required as the base. `Olmo2ForSequenceClassification` is optional/deferred for generation; reward-model configs use it and require last-non-pad pooling plus a scalar score head.

## 3. Important config dimensions

Source defaults from `Olmo2Config`:

| Field | Default | Runtime impact |
| --- | ---: | --- |
| `vocab_size` | 50304 | Embedding and LM head rows |
| `hidden_size` | 4096 | Decoder width |
| `intermediate_size` | 11008 | MLP gate/up width |
| `num_hidden_layers` | 32 | Decoder block count |
| `num_attention_heads` | 32 | Query head count |
| `num_key_value_heads` | `None -> num_attention_heads` | MHA by default; GQA when lower |
| `head_dim` | inferred as `hidden_size // num_attention_heads` unless present | Projection and attention head width |
| `hidden_act` | `silu` | SwiGLU activation |
| `max_position_embeddings` | 2048 | RoPE cache initial length |
| `rope_parameters` | `None` | Standardized from legacy `rope_theta`/`rope_scaling` at config load |
| `attention_bias` | `False` | Q/K/V/O projections normally bias-free |
| `tie_word_embeddings` | `False` | LM head is normally separate from token embedding |
| `rms_norm_eps` | `1e-5` | Checkpoints override to `1e-6` |
| `use_cache` | `True` | Default generation cache behavior |

Checkpoint sweep from official configs:

| Checkpoint | Arch | Hidden | Layers | Q heads | KV heads | Head dim | MLP | Max pos | RoPE theta | Vocab | Dtype | Cache default |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| OLMo-2-1124-7B | CausalLM | 4096 | 32 | 32 | 32 | 128 | 11008 | 4096 | 500000 | 100352 | float32 | true |
| OLMo-2-1124-7B-Instruct | CausalLM | 4096 | 32 | 32 | 32 | 128 | 11008 | 4096 | 500000 | 100352 | bfloat16 | false |
| OLMo-2-1124-13B | CausalLM | 5120 | 40 | 40 | 40 | 128 | 13824 | 4096 | 500000 | 100352 | float32 | true |
| OLMo-2-1124-13B-Instruct | CausalLM | 5120 | 40 | 40 | 40 | 128 | 13824 | 4096 | 500000 | 100352 | bfloat16 | false |
| OLMo-2-0325-32B | CausalLM | 5120 | 64 | 40 | 8 | 128 | 27648 | 4096 | 500000 | 100352 | float32 | true |
| OLMo-2-0325-32B-Instruct | CausalLM | 5120 | 64 | 40 | 8 | 128 | 27648 | 4096 | 500000 | 100352 | bfloat16 | false |
| OLMo-2-1124-7B-RM | SequenceClassification | 4096 | 32 | 32 | 32 | 128 | 11008 | 4096 | 500000 | 100280 | bfloat16 | true |

Config facts:

- 1124 7B/13B are MHA (`num_key_value_heads == num_attention_heads`).
- 0325 32B is GQA (`40` query heads, `8` KV heads, group factor `5`).
- Checkpoint configs use legacy `rope_theta` and `rope_scaling: null`; the inspected config base standardizes these into `config.rope_parameters`.
- Instruct configs set `use_cache: false`, but the source still implements cache support and accepts a runtime `use_cache` override.

## 3a. Family variation traps

- Do not assume OLMo2 is pre-norm. Decoder blocks apply attention, then `post_attention_layernorm`, then residual add; MLP, then `post_feedforward_layernorm`, then residual add.
- Do not reuse OLMo layernorm semantics. OLMo uses weightless `LayerNorm`; OLMo2 uses learned RMSNorm.
- OLMo2 adds learned RMSNorm on projected Q and K before head reshape and RoPE. This changes tensor-parallel constraints and fusion placement.
- 0325 32B requires GQA/MQA-style attention support even though common 1124 checkpoints are MHA.
- Attention projections are separate Q, K, V modules in HF weights. The converter splits original fused `att_proj.weight` in Q, K, V order with row sizes `[hidden, kv_heads * head_dim, kv_heads * head_dim]`.
- MLP conversion splits original fused `ff_proj.weight` into `up_proj` then `gate_proj`, while HF forward computes `silu(gate_proj(x)) * up_proj(x)`.
- `attention_bias` is config-controlled, but official inspected configs set it false.
- `tie_word_embeddings` is false in inspected configs. The class declares tied-weight keys for compatibility, but these official checkpoints use a separate `lm_head.weight`.
- `rope_scaling` appears in configs but is null for inspected checkpoints. Current source can route non-default `rope_parameters["rope_type"]` through shared RoPE helpers; DinoML should reject unvalidated non-default RoPE types until separately tested.
- Reward-model configs use `Olmo2ForSequenceClassification`, a different head and vocab size. This should not be treated as a CausalLM target.
- Layout is text `[batch, seq, hidden]`; no NHWC/channel-last translation is relevant for the core model. Protect attention head axes from generic layout passes.

## 4. Operator coverage checklist

Tensor/layout ops:

- Integer token embedding lookup: `[B, S] -> [B, S, H]`.
- Optional `inputs_embeds` direct input path.
- `arange`, add scalar/cache length, `unsqueeze` for default `position_ids`.
- `view`/reshape from `[B, S, heads * D]` to `[B, S, heads, D]`.
- `transpose(1, 2)` to `[B, heads, S, D]`.
- `contiguous`, reshape back to `[B, S, heads * D]`.
- Slice last `logits_to_keep` tokens before LM head.
- For sequence classification only: last non-pad index via `input_ids != pad_token_id`, integer `argmax`, gather `[B, S, num_labels] -> [B, num_labels]`.

Neural network primitives:

- Bias-free Linear unless `attention_bias=True`.
- CausalLM common shapes:
  - 7B Q/O/LM hidden: `4096 -> 4096`, K/V `4096 -> 4096`, MLP gate/up `4096 -> 11008`, down `11008 -> 4096`, LM head `4096 -> 100352`.
  - 13B Q/O `5120 -> 5120`, K/V `5120 -> 5120`, gate/up `5120 -> 13824`, down `13824 -> 5120`.
  - 32B GQA Q/O `5120 -> 5120`, K/V `5120 -> 1024`, gate/up `5120 -> 27648`, down `27648 -> 5120`.
- RMSNorm with learned weight and fp32 reduction.
- SiLU and gated multiply for SwiGLU.
- Residual add after post-attention norm and post-MLP norm.
- Final RMSNorm.

Attention primitives:

- Causal self-attention only.
- MHA and GQA.
- Matmul QK^T, additive mask, fp32 softmax, dropout only in training, AV matmul for eager fallback.
- SDPA/Flash/Flex compatible dispatch through `ALL_ATTENTION_FUNCTIONS`.

Position/rotary ops:

- Default RoPE over full head dim.
- `inv_freq = 1 / rope_theta ** (arange(0, head_dim, 2) / head_dim)`.
- Cos/sin computed in fp32 under autocast-disabled context.
- `rotate_half` split/concat.

Generation/cache ops:

- Dynamic per-layer KV cache.
- Cache stores K after RoPE and V after projection/reshape, before any repeat expansion.
- Cache tensor shape per layer: `[B, num_key_value_heads, cached_seq, head_dim]` for both K and V.
- Decode needs `past_seen_tokens = past_key_values.get_seq_length()` for default position ids.
- Reorder/reset/static cache behavior belongs to shared Transformers cache support, not OLMo2-specific source.

Distributed/tensor-parallel ops:

- Source declares TP plans. Q/K/V use `colwise_gather_output` because Q/K RMSNorms require gathered full projected vectors before norm. O uses `rowwise_split_input`.
- DinoML single-GPU can ignore TP initially but should not fuse in a way that prevents later Q/K norm placement.

## 5. Layer/block breakdown

Shared variables:

```text
B = batch
S = current query length
T = cached/key length after update
H = hidden_size
A = num_attention_heads
K = num_key_value_heads
D = head_dim
I = intermediate_size
G = A / K
```

Decoder block, repeated `num_hidden_layers` times:

```text
residual = x                                      # [B, S, H]

q = q_proj(x)                                    # [B, S, A*D], bias=config.attention_bias
q = RMSNorm(q)                                   # learned weight [A*D], fp32 variance
k = k_proj(x)                                    # [B, S, K*D]
k = RMSNorm(k)                                   # learned weight [K*D]
v = v_proj(x)                                    # [B, S, K*D]

q = view(q, [B, S, A, D]).transpose(1, 2)        # [B, A, S, D]
k = view(k, [B, S, K, D]).transpose(1, 2)        # [B, K, S, D]
v = view(v, [B, S, K, D]).transpose(1, 2)        # [B, K, S, D]
q, k = RoPE(q, k, cos, sin)
k, v = cache.update(k, v, layer_idx)             # [B, K, T, D] when cache is used

attn = causal_attention(q, k, v, mask, scale=D^-0.5)
attn = reshape(attn, [B, S, A*D])
attn = o_proj(attn)                              # [B, S, H]
attn = post_attention_layernorm(attn)            # RMSNorm after projection
x = residual + attn

residual = x
m = down_proj(silu(gate_proj(x)) * up_proj(x))   # SwiGLU, [B, S, H]
m = post_feedforward_layernorm(m)                # RMSNorm after MLP
x = residual + m
```

Model tail:

```text
x = final RMSNorm(x)
logits = lm_head(x[:, slice_indices, :])         # [B, kept_tokens, vocab]
```

Note the norm placement: there is no input norm before attention and no norm before MLP. Both sublayer norms are after the sublayer projection and before residual add.

## 6. Attention requirements

Attention variant:

- Causal self-attention.
- No cross-attention.
- No sliding-window/local attention in inspected configs.
- No ALiBi or relative bias.
- MHA for 1124 7B/13B and GQA for 0325 32B.
- Head dim is 128 in all inspected official configs.

Projection and shape details:

- Q projection output: `[B, S, A * D]`.
- K/V projection output: `[B, S, K * D]`.
- Q/K RMSNorm happens before reshaping into heads.
- RoPE is applied to Q and K before cache update.
- Cached K shape before repeat: `[B, K, T, D]`.
- Cached V shape before repeat: `[B, K, T, D]`.
- Eager attention expands K/V with `repeat_kv` to `[B, A, T, D]` when `G > 1`.
- SDPA may either use `enable_gqa=True` or explicitly repeat K/V depending on torch/backend/mask conditions.

Masking:

- Source calls shared `create_causal_mask`.
- A 2D attention mask represents padded tokens and can become a backend-specific 4D additive or block mask.
- Eager path adds the mask to attention scores before softmax.
- For SDPA, the shared integration can skip explicit masks for pure causal prefill when allowed.

Math order:

```text
scores = matmul(q, k.transpose(-2, -1)) * head_dim**-0.5
scores += attention_mask if present
probs = softmax(scores, dim=-1, dtype=float32).to(q.dtype)
out = matmul(probs, v)
```

Backend compatibility:

- Source advertises FlashAttention, SDPA, and FlexAttention support.
- `output_attentions=True` forces eager-like behavior for usable attention weights; optimized integrations warn or return no weights.
- Dropout is only nonzero in training. In inference, pass zero dropout.

## 7. Position encoding and custom math

RoPE setup:

- Source uses `config.rope_parameters["rope_type"]`.
- For default RoPE, `rope_theta` comes from standardized config. Inspected official checkpoints use `rope_theta = 500000`.
- `inv_freq` is a non-persistent buffer and can be precomputed per config/head_dim/rope type.
- Cos/sin depends on runtime `position_ids` and batch length, but can be cached by position range for normal decode.

Short reproduction snippets:

```python
def olmo2_rms_norm(x, weight, eps):
    orig_dtype = x.dtype
    y = x.float()
    y = y * torch.rsqrt(y.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (weight * y).to(orig_dtype)
```

```python
def olmo2_default_rope(position_ids, head_dim, rope_theta):
    inv = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    freqs = (inv[None, :, None] @ position_ids[:, None, :].float()).transpose(1, 2)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()
```

```python
def apply_olmo2_rope(q, k, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin
```

Non-default RoPE types:

- The current source can dispatch to shared `ROPE_INIT_FUNCTIONS` for non-default `rope_type`.
- Inspected official configs have `rope_scaling: null`, so first integration can admit only default RoPE with `rope_theta=500000` and full-head rotary.

## 8. Preprocessing and input packing

Runtime inputs:

- `input_ids`: integer tensor `[B, S]`.
- `attention_mask`: optional tensor, usually `[B, total_length]` for padding.
- `position_ids`: optional `[B, S]`; if missing, source builds contiguous positions offset by cache length.
- `inputs_embeds`: optional `[B, S, H]` alternative to `input_ids`. Source requires exactly one of `input_ids` or `inputs_embeds`.
- `past_key_values`: optional shared cache object.

CPU/data-pipeline work:

- Tokenization and chat template construction.
- Padding side and special-token layout are tokenizer responsibilities.
- Official configs use pad token `100277` and eos token `100257` for OLMo2 1124/0325 checkpoints.

GPU/runtime work:

- Embedding lookup.
- Position id creation when not supplied.
- Causal/padding mask materialization or equivalent fused attention metadata.

Generation-controller behavior:

- The modeling source does not implement custom forced decoder ids, suppress-token processors, timestamp logic, image/audio placeholders, or speculative decoding.
- First DinoML integration can rely on a generic text generation loop.

Sequence classification/reward model:

- Optional target only. It applies the base decoder, a bias-free score projection on every token, then pools the rightmost non-pad token using `input_ids != pad_token_id`.

## 9. Graph rewrite / lowering opportunities

### Rewrite: separate Q/K/V projections to fused QKV GEMM

Source pattern:

```text
q = q_proj(x)
k = k_proj(x)
v = v_proj(x)
q = q_norm(q)
k = k_norm(k)
```

Replacement pattern:

```text
qkv = GEMM(x, concat_rows(q_weight, k_weight, v_weight))
split qkv -> q, k, v
q = RMSNorm(q)
k = RMSNorm(k)
```

Preconditions:

- All projections share input `x`.
- Projection bias is either absent or all present and concatenated in Q, K, V order.
- Split sizes are `[A*D, K*D, K*D]`.
- Q and K RMSNorm must remain after the split. Do not norm the packed QKV tensor as one vector.

Weight transform:

```python
packed_weight = torch.cat([q_proj.weight, k_proj.weight, v_proj.weight], dim=0)
packed_bias = None or torch.cat([q_bias, k_bias, v_bias], dim=0)
```

Failure cases:

- Future configs with nonstandard projection packing.
- Tensor-parallel lowering that cannot gather Q/K before RMSNorm.

Parity test sketch:

- Random `[B, S, H]` input, compare separate projections plus norms to packed GEMM plus split plus norms for MHA and GQA.

### Rewrite: Q/K RMSNorm plus reshape

Source pattern:

```text
RMSNorm([B, S, heads*D]) -> view -> transpose
```

Replacement:

```text
head-aware RMSNorm over last packed dimension, then produce [B, heads, S, D]
```

Preconditions:

- Norm is over the full packed Q or K vector, not per head independently.
- Weight shape is `[heads * D]`.

Failure cases:

- Per-head norm interpretation would be wrong because mean is over all Q or K projection channels.

### Rewrite: RoPE plus attention prefill/decode

Source pattern:

```text
q,k = apply_rotary_pos_emb(q,k,cos,sin)
cache.update(k,v)
attention(q,k,v,mask)
```

Replacement:

```text
fused_rope_cache_attention(q, k, v, position_ids, cache, mask)
```

Preconditions:

- Default RoPE with full head dim.
- K stored in cache after RoPE.
- V stored without RoPE.
- Attention backend supports MHA and GQA cache shapes.

Failure cases:

- Non-default RoPE types.
- Requested `output_attentions=True`.
- Packed sequence masks not represented in the fused backend.

### Rewrite: SwiGLU MLP fusion

Source pattern:

```text
down_proj(silu(gate_proj(x)) * up_proj(x))
```

Replacement:

```text
fused_two_gemm_swiglu_down or GEMM(gate/up) -> fused silu_mul -> GEMM(down)
```

Preconditions:

- `hidden_act == "silu"`.
- `gate_proj`, `up_proj`, and `down_proj` are bias-free as in official configs.

Failure cases:

- Any future activation other than SiLU.
- Nonstandard fused source checkpoint where gate/up order is not transformed correctly.

### Rewrite: last-token-only logits

Source pattern:

```text
hidden_states[:, slice_indices, :] -> lm_head
```

Replacement:

```text
gather kept token states before LM-head GEMM
```

Preconditions:

- In decode, only last token logits are needed.
- `logits_to_keep` is positive integer or explicit token index tensor.

Failure cases:

- Loss computation or full-sequence logits requested.

## 10. Kernel fusion candidates

Highest priority:

- RMSNorm. OLMo2 uses it for Q, K, every sublayer output, and final norm; it is on the critical path and has fp32 reduction semantics.
- GQA causal attention with RoPE and KV cache. Required for 0325 32B and for efficient decode.
- QKV projection packing with Q/K RMSNorm split. This cuts launch overhead but must preserve post-projection norm placement.
- SwiGLU MLP. Gate/up projections plus SiLU/multiply dominate per-token compute outside attention.
- Last-token-only logits. Vocab is 100352, so avoiding full-sequence LM head in decode matters.

Medium priority:

- RoPE generation/application fused with cache update.
- Bias-free linear epilogue plus residual/RMSNorm placement. Post-sublayer norm makes the usual pre-norm fusion recipes unsafe.
- Packed prefill attention mask handling through SDPA/Flash-compatible metadata.

Lower priority:

- Sequence classification pooling and score head for RM checkpoints.
- Non-default RoPE scaling support.
- Tensor-parallel plan fidelity.

## 11. Runtime staging plan

Stage 1: config and weight loading

- Parse `olmo2` configs, including legacy `rope_theta`/`rope_scaling` into internal RoPE parameters.
- Load token embedding, per-layer Q/K/V/O, Q/K norms, post-attention/post-FFN norms, MLP weights, final norm, and LM head.
- Support untied LM head first.

Stage 2: one-block parity

- Implement exact OLMo2 block with eager attention and no cache.
- Validate 7B-like MHA and 32B-like GQA synthetic dimensions.

Stage 3: full prefill

- Run all blocks with causal mask and RoPE.
- Produce full logits or `logits_to_keep` slices.

Stage 4: decode with KV cache

- Cache post-RoPE K and raw V per layer as `[B, KV_heads, T, D]`.
- Use default position id offset from cache length.
- Add cache reorder/reset only when generation features require it.

Stage 5: optimized attention

- Add Flash/SDPA-style MHA and GQA paths.
- Preserve eager fallback for parity and `output_attentions`.

Stage 6: graph rewrites and fusions

- Add QKV packing, SwiGLU fusion, RoPE-cache-attention fusion, and last-token logits.

Stage 7: optional heads and production scheduling

- Add sequence classification pooling if RM support is needed.
- Add batching, paged cache, and quantized/offloaded weights after dense parity is stable.

## 12. Parity and validation plan

Custom op tests:

- RMSNorm random tensor tests for fp32, fp16, bf16 storage with fp32 reduction.
- RoPE default tests against Transformers for varied `position_ids`, including cache offset positions.
- `repeat_kv`/GQA shape tests with `A=40`, `K=8`, `D=128`.

Layer tests:

- Single attention module parity with no cache, MHA and GQA.
- Single attention module parity with cache update: prefill then one-token decode.
- Single MLP parity for SiLU gated product.
- Full decoder layer parity verifying post-attention/post-FFN norm placement.

Model tests:

- One-layer synthetic OLMo2 CausalLM logits parity.
- Full small synthetic config prefill logits parity.
- Decode token parity over several generated steps using the same prompt and cache.
- Check `logits_to_keep=1` against slicing full logits.

Optional head tests:

- Sequence classification pooling with right padding and left padding.
- No-pad-token batch-size rejection if supporting RM head exactly.

Recommended tolerances:

- fp32: `rtol=1e-4`, `atol=1e-5` for full block, tighter for isolated ops.
- fp16/bf16: `rtol=1e-2`, `atol=1e-2` for full logits, with op-level tolerances based on accumulation policy.
- Compare pre-softmax attention scores and post-softmax outputs separately when debugging fused attention.

## 13. Performance probes

- Prefill throughput by sequence length: 128, 512, 2048, 4096.
- Decode tokens/sec by batch size and cache length.
- MHA versus GQA attention backend comparison.
- KV cache memory usage for 7B, 13B, and 32B dimensions.
- RoPE plus cache update time as a standalone decode probe.
- MLP GEMM/SwiGLU/down projection time per token.
- LM head full-sequence versus last-token-only GEMM time.
- End-to-end generation latency split into embedding, decoder blocks, LM head, sampler.
- Optional RM score throughput for sequence classification if reward models are targeted.

No benchmark observations are included here; these are source-derived probe recommendations.

## 14. Skip/defer list

Safe to defer for first CausalLM integration:

- Training and gradient checkpointing.
- Dropout behavior beyond forcing zero in inference.
- `output_attentions=True` optimized path; keep eager fallback.
- Sequence classification/reward-model head.
- Non-default RoPE scaling variants.
- Tensor parallel and pipeline parallel execution.
- Quantization, GGUF conversion, CPU/GPU offload policy.
- Beam search and advanced generation controllers beyond generic cache reorder when needed.
- FlexAttention block-mask packed sequence optimizations.
- Remote-code or historical `olmo_1124` model_type configs unless mapped explicitly.

Do not defer:

- Post-sublayer RMSNorm placement.
- Q/K post-projection RMSNorm.
- GQA for 0325 32B.
- RoPE-before-cache semantics.
- Untied LM head.

## 15. Final implementation checklist

- [ ] Parse `Olmo2Config`, including legacy `rope_theta` into normalized RoPE params.
- [ ] Load untied embedding and LM-head weights.
- [ ] Load per-layer Q/K/V/O, Q/K norm, post-attention RMSNorm, post-FFN RMSNorm, and MLP weights.
- [ ] Implement OLMo2 RMSNorm with fp32 reduction and learned scale.
- [ ] Implement default RoPE with `rope_theta=500000` and full-head rotary.
- [ ] Implement attention with MHA and GQA cache shapes.
- [ ] Store cached K after RoPE and V after projection.
- [ ] Implement post-attention norm before residual add.
- [ ] Implement SwiGLU MLP and post-MLP norm before residual add.
- [ ] Implement final RMSNorm and untied LM head.
- [ ] Add `logits_to_keep` lowering for last-token logits.
- [ ] Add eager one-block parity tests.
- [ ] Add prefill logits parity tests.
- [ ] Add decode KV-cache parity tests.
- [ ] Add 0325 32B-shaped GQA synthetic tests.
- [ ] Add QKV packing rewrite guarded by split and norm preconditions.
- [ ] Add SwiGLU fusion rewrite.
- [ ] Benchmark prefill, decode, attention backend, MLP, and LM-head slices.
