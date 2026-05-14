# DinoML Transformers Audit: jais2

## 1. Source basis

Transformers commit/version: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4` from `X:/H/transformers`.

Model id: primary upstream examples are [inceptionai/Jais-2-8B-Chat](https://huggingface.co/inceptionai/Jais-2-8B-Chat) and [inceptionai/Jais-2-70B-Chat](https://huggingface.co/inceptionai/Jais-2-70B-Chat). Both official dense repos, and the matching official GGUF repos, were gated for raw `config.json` access in this environment.

Config source: source defaults from `configuration_jais2.py`; accessible open snapshots under `hf_snapshots/`:

| Snapshot | Source | Notes |
|---|---|---|
| `hf_snapshots/yoriis_JAIS2-IT-0.3/` | [yoriis/JAIS2-IT-0.3](https://huggingface.co/yoriis/JAIS2-IT-0.3) | Open mirror/finetune, `model_type=jais2`. |
| `hf_snapshots/Omaratef3221_jais-2-8b-chat-s1-full-aramed/` | [Omaratef3221/jais-2-8b-chat-s1-full-aramed](https://huggingface.co/Omaratef3221/jais-2-8b-chat-s1-full-aramed) | Open finetune, same architecture values. |
| `hf_snapshots/Omaratef3221_jais-2-8b-chat-s1-full-s2-full-medarabench/` | [Omaratef3221/jais-2-8b-chat-s1-full-s2-full-medarabench](https://huggingface.co/Omaratef3221/jais-2-8b-chat-s1-full-s2-full-medarabench) | Open finetune, config snapshot only. |

Source files inspected:

- `src/transformers/models/jais2/configuration_jais2.py`
- `src/transformers/models/jais2/modeling_jais2.py`
- `src/transformers/models/jais2/modular_jais2.py`
- `src/transformers/models/jais2/__init__.py`
- Supporting source: `activations.py`, `masking_utils.py`, `cache_utils.py`, `integrations/sdpa_attention.py`, `modeling_flash_attention_utils.py`

Any missing files or assumptions: `modeling_jais2.py` and `configuration_jais2.py` are generated from `modular_jais2.py`; future source edits should target the modular file, but this audit treats generated files as the authoritative runtime surface. Official 70B dimensions were not available from raw config access; only HF repo metadata confirmed it is gated, `jais2`, `AutoModelForCausalLM`, and approximately 72B parameters.

## 2. High-level architecture

Jais2 is a text-only decoder-only causal language model.

```text
token ids -> token embedding -> N decoder blocks -> final LayerNorm -> LM head -> logits/sampling
```

Primary DinoML runtime target: `Jais2ForCausalLM` prefill and decode for autoregressive text generation. `Jais2Model` is a useful hidden-state subtarget. Training loss, labels, output attentions, and hidden-state capture are optional/deferred for first integration.

The main architectural deltas from Llama-like defaults are source-derived:

- LayerNorm is used instead of RMSNorm.
- MLP is ungated `Linear -> relu2 -> Linear`, not SwiGLU.
- Attention and MLP projections default to `bias=True`.
- RoPE is used for Q/K.

## 3. Important config dimensions

Source defaults:

| Field | Default | Source/runtime meaning |
|---|---:|---|
| `vocab_size` | 150272 | Token embedding rows and LM head rows. |
| `hidden_size` | 3328 | Decoder hidden width. |
| `intermediate_size` | 26624 | MLP expansion width. |
| `num_hidden_layers` | 32 | Decoder block count. |
| `num_attention_heads` | 26 | Query heads. |
| `num_key_value_heads` | defaults to attention heads | MHA by default; source supports GQA/MQA if smaller. |
| `head_dim` | `hidden_size // num_attention_heads` | 128 for defaults; explicit configs can override. |
| `max_position_embeddings` | 8192 | RoPE original cache length. |
| `hidden_act` | `relu2` | ReLU squared. |
| `layer_norm_eps` | 1e-5 | LayerNorm epsilon. |
| `attention_bias` | true | Bias on q/k/v/o projections. |
| `mlp_bias` | true | Bias on up/down projections. |
| `tie_word_embeddings` | false | LM head is not tied by default/config. |
| `use_cache` | true source default | Accessible finetune configs set `false`; generation can still pass `use_cache=True`. |

Representative checkpoint sweep:

| Repo/config | Availability | hidden | layers | heads/KV | head_dim | MLP | RoPE | dtype | cache |
|---|---|---:|---:|---:|---:|---:|---|---|---|
| source default | local source | 3328 | 32 | 26/26 effective | 128 | 26624 | `rope_parameters` expected, default theta from configs | not set | true |
| `yoriis/JAIS2-IT-0.3` | open mirror/finetune | 3328 | 32 | 26/26 | 128 | 26624 | `default`, theta 500000 | bfloat16 | false |
| `Omaratef3221/...s1-full-aramed` | open finetune | 3328 | 32 | 26/26 | 128 | 26624 | `default`, theta 500000 | bfloat16 | false |
| `Omaratef3221/...s1-full-s2-full-medarabench` | open finetune | 3328 | 32 | 26/26 | 128 | 26624 | `default`, theta 500000 | bfloat16 | false |
| `inceptionai/Jais-2-8B-Chat` | gated official | unknown raw config; HF metadata says `jais2`, ~8.09B params | unknown | unknown | unknown | unknown | unknown | unknown | unknown |
| `inceptionai/Jais-2-70B-Chat` | gated official | unknown raw config; HF metadata says `jais2`, ~72.04B params | unknown | unknown | unknown | unknown | unknown | unknown | unknown |

## 3a. Family variation traps

- Do not assume RMSNorm. Jais2 replaces Llama RMSNorm with `nn.LayerNorm`.
- Do not assume gated MLP. The MLP has only `up_proj` and `down_proj`; `relu2(x) = square(relu(x))`.
- Do not assume bias-free Llama weights. Accessible configs and source defaults use attention and MLP biases.
- Do not infer projection width only from `hidden_size`. Source uses `num_attention_heads * head_dim` and `num_key_value_heads * head_dim`; explicit `head_dim` can make attention width differ from hidden width.
- Source supports GQA/MQA structurally through `num_key_value_heads`, but accessible configs use MHA. DinoML should guard `num_attention_heads % num_key_value_heads == 0`.
- Accessible configs contain `pretraining_tp`, but generated `modeling_jais2.py` does not read it.
- Config class default `num_key_value_heads=None` becomes `num_attention_heads`; checkpoint configs spell out 26.
- Tokenizer `model_max_length` in open snapshots is 65536 while model config `max_position_embeddings` is 8192. Treat tokenizer max length as preprocessing metadata, not proof of long-context graph support.
- Official dense and GGUF repos are gated. GGUF support is a weight-loading/provider contract, not a modeling-source op difference, until metadata is available.

## 4. Operator coverage checklist

Tensor/layout ops:

- Embedding lookup: `[B,T] -> [B,T,H]`.
- View/reshape/transpose: Q/K/V `[B,T,out] -> [B,T,heads,D] -> [B,heads,T,D]`; attention output transpose back to `[B,T,H_attn]`.
- Concatenate along last dim for RoPE frequency duplication and `rotate_half`.
- Slice/index for `logits_to_keep`, last-token-only logits, and optional tensor index selection.

Neural network primitives:

- LayerNorm over last dim, epsilon 1e-5, affine weight/bias.
- Linear projections with optional bias:
  - default Q: `Linear(3328 -> 3328)`
  - default K/V: `Linear(3328 -> 3328)` for MHA; `3328 -> num_key_value_heads * head_dim` if GQA
  - default O: `Linear(3328 -> 3328)`
  - default MLP up: `Linear(3328 -> 26624)`
  - default MLP down: `Linear(26624 -> 3328)`
  - LM head: `Linear(3328 -> 150272)`, no bias
- ReLU squared activation.
- Residual adds after attention and MLP.

Attention primitives:

- Causal self-attention.
- MHA for accessible configs; GQA/MQA admission should be supported or explicitly rejected.
- Softmax over key length in fp32 for eager parity, then cast to query dtype.
- Optional SDPA/FlashAttention backend parity; eager fallback is source-visible.

Position/cache ops:

- RoPE cos/sin generation and application to Q/K.
- Dynamic or static KV cache per layer with keys/values shaped `[B, kv_heads, S, head_dim]`.
- Cache stores keys after RoPE application and values after V projection.

Preprocessing-coupled ops:

- Tokenization is outside the neural graph. Runtime receives `input_ids`, `attention_mask`, optional `position_ids` or `inputs_embeds`.

## 5. Layer/block breakdown

Decoder block, repeated `num_hidden_layers` times:

```text
residual = x
x = LayerNorm(x)
q = Linear_q(x).view(B,T,Hq,D).transpose(1,2)
k = Linear_k(x).view(B,T,Hkv,D).transpose(1,2)
v = Linear_v(x).view(B,T,Hkv,D).transpose(1,2)
q,k = RoPE(q,k, cos, sin)
k,v = cache.update(k,v, layer_idx) when cache is enabled
attn = causal_attention(q,k,v, mask, scale=D^-0.5)
x = residual + Linear_o(attn.transpose(1,2).reshape(B,T,Hq*D))
residual = x
x = LayerNorm(x)
x = Linear_down(relu(Linear_up(x))^2)
x = residual + x
```

For default/open 8B configs: `H=3328`, `heads=26`, `kv_heads=26`, `D=128`, MLP width `26624`.

## 6. Attention requirements

Required attention variant: causal decoder self-attention.

| Property | Requirement |
|---|---|
| Causal/noncausal | Causal. |
| Self/cross | Self-attention only. |
| MHA/GQA/MQA | MHA in accessible configs; source supports GQA through smaller `num_key_value_heads`. |
| Q width | `num_attention_heads * head_dim`. |
| K/V width | `num_key_value_heads * head_dim`. |
| Scaling | `head_dim ** -0.5`. |
| Masking | `create_causal_mask` combines causal and padding masks; eager path adds mask before softmax. |
| RoPE | Applied to Q and K before cache update. |
| Cache | Per-layer K/V cache. K is stored post-RoPE; V is stored after projection. |
| Flash/SDPA | Source declares FlashAttention, SDPA, FlexAttention support; eager attention remains the parity baseline. |

Decode cache shapes:

```text
K_cache[layer], V_cache[layer]: [batch, num_key_value_heads, total_seen_tokens, head_dim]
```

For GQA, eager attention repeats K/V across head groups before `Q @ K^T`; optimized DinoML attention should avoid materializing the repeat when possible.

## 7. Position encoding and custom math

Default/open configs use RoPE with `rope_type="default"` and `rope_theta=500000.0`. Source computes inverse frequencies in fp32:

```python
inv_freq = 1.0 / (theta ** (arange(0, head_dim, 2) / head_dim))
freqs = inv_freq[None, :, None] @ position_ids[:, None, :]
emb = concat(freqs, freqs, dim=-1)
cos, sin = cos(emb), sin(emb)
q = q * cos[:, None, :, :] + rotate_half(q) * sin[:, None, :, :]
k = k * cos[:, None, :, :] + rotate_half(k) * sin[:, None, :, :]
```

`rotate_half(x)` splits the last dimension into two halves and returns `concat(-second_half, first_half)`. Cos/sin can be precomputed for fixed position ranges and dtype-cast at use, but dynamic cache position and batched `position_ids` must be honored.

`relu2` custom activation:

```python
def relu2(x):
    y = relu(x)
    return y * y
```

## 8. Preprocessing and input packing

The neural graph accepts either:

- `input_ids: [B,T]`, then `embed_tokens(input_ids)`, or
- `inputs_embeds: [B,T,H]`, bypassing token lookup.

Open tokenizer snapshots:

- `tokenizer_class`: `TokenizersBackend`
- `bos_token`: `<|begin_of_text|>`, `bos_token_id=0`
- `eos_token`: `<|eot_id|>`, `eos_token_id=150024`
- `pad_token`: `<|endoftext|>`, `pad_token_id=1`
- `model_max_length=65536`

GPU graph inputs for first integration should be `input_ids`, `attention_mask`, optional `position_ids`, and optional cache handles. Chat templates and sampling are generation-controller work outside the compiled block.

## 9. Graph rewrite / lowering opportunities

### Rewrite: Pack QKV projections

Source pattern: three separate linears from the same normalized hidden state.

Replacement: one GEMM producing `[Q | K | V]`, followed by static splits.

Preconditions:

- Same input tensor, dtype, and batch/sequence axes.
- Bias policy preserved for all three projections.
- Packed weight row order is all Q rows, then all K rows, then all V rows.
- Split sizes are `num_attention_heads * head_dim`, `num_key_value_heads * head_dim`, `num_key_value_heads * head_dim`.

Failure cases: mixed quantization policies, missing/extra projection bias, unsupported GQA split sizes, or source weights already stored in a custom packed layout.

Parity sketch: compare q/k/v tensors before RoPE for random `[B,T,H]` inputs.

### Rewrite: MLP relu2 fusion

Source pattern:

```text
up = Linear(x)
act = relu(up) * relu(up)
out = Linear(act)
```

Replacement: fused GEMM epilogue for `relu2` when supported, or generated elementwise `relu2` between two GEMMs.

Preconditions: activation exactly `relu2`; no gate projection; bias flags preserved.

### Rewrite: Last-token-only logits

Source pattern: `lm_head(hidden_states[:, slice_indices, :])`.

Replacement: for decode, run the vocab GEMM only for last token or requested indices.

Preconditions: `logits_to_keep == 1` or known static/tensor positions; output API accepts reduced time dimension.

### Rewrite: RoPE + attention fusion

Fuse Q/K RoPE into attention pre-processing or attention kernel.

Preconditions: default half-rotation convention, cached K stored after RoPE, correct position IDs, no unsupported dynamic RoPE scaling.

## 10. Kernel fusion candidates

Highest priority:

- LayerNorm with affine scale/bias: every block has two LayerNorms plus final norm.
- QKV GEMM packing plus RoPE plus attention prefill/decode.
- Causal MHA/GQA attention with KV cache, avoiding repeated K/V materialization.
- MLP `Linear + relu2 + Linear`, especially for large `26624` intermediate width.
- Last-token-only LM head for decode with 150272 vocab rows.

Medium priority:

- Bias+residual fusion around attention and MLP outputs.
- RoPE cos/sin precompute and cache-position-aware application.
- Bias epilogues for all projection GEMMs.

Lower priority:

- Output attentions materialization.
- Training loss path.
- Gated official GGUF load support until exact GGUF metadata is available.

## 11. Runtime staging plan

1. Parse `Jais2Config`, reject unsupported `rope_type`, unsupported `head_dim` mismatch cases, and non-divisible KV grouping.
2. Load dense 8B-compatible weights and run embedding + one block parity.
3. Implement full prefill without cache using eager-style causal attention.
4. Add Dynamic/Static KV cache ABI for decode; confirm K is cached post-RoPE.
5. Add optimized attention backend for MHA first, then GQA if a checkpoint requires it.
6. Add QKV packing, relu2 MLP fusion, LayerNorm fusion, and last-token logits.
7. Add gated official/GGUF admission once official config and weight metadata are available.

Initially stub/defer: training loss, `output_attentions=True`, tensor-parallel plans, paged cache, FlashAttention-specific packed sequence variants, and official 70B until config is accessible.

## 12. Parity and validation plan

- Config validation tests for open 8B snapshots and source defaults.
- Unit tests for `relu2`, LayerNorm epsilon, RoPE half-rotation, and Q/K cache update ordering.
- Single-block parity with random dense weights in fp32, then bf16/fp16 tolerance.
- Full 32-layer small-shape parity using random weights, short sequence, no cache.
- Prefill logits parity against Transformers for accessible open checkpoint configs.
- Decode parity for one-token and multi-token cache extension; verify cache shapes and position IDs.
- `logits_to_keep` parity for `0`, `1`, and tensor indices.

Suggested tolerances: fp32 `atol=1e-4, rtol=1e-4`; bf16/fp16 block-level `atol=2e-2, rtol=2e-2`, tightened per-op where possible.

## 13. Performance probes

- Prefill tokens/sec across `B in {1,4,8}` and `T in {128,1024,4096,8192}`.
- Decode tokens/sec for `B in {1,8,32}` with cache lengths from 128 to 8192.
- Attention backend comparison: eager reference, SDPA-like, FlashAttention-like, DinoML fused attention.
- MLP throughput with and without fused relu2.
- LayerNorm bandwidth and fusion impact.
- LM head last-token GEMM cost versus all-token logits.
- KV cache memory: `layers * 2 * B * kv_heads * S * head_dim * dtype_bytes`.
- If GGUF becomes accessible: dense load-time dequant versus runtime dequant-before-GEMM for MLP/attention projections.

## 14. Skip/defer list

- Training and loss computation.
- Gradient checkpointing.
- `output_attentions=True` dense attention matrix outputs.
- Tensor parallel and pipeline parallel plans.
- Paged attention/cache unless production scheduling requires it.
- Official 70B execution until raw config/weights are accessible.
- GGUF quantized loading/provider details until gated GGUF metadata is available.
- Long-context beyond model `max_position_embeddings=8192` despite tokenizer `model_max_length=65536`.

## 15. Final implementation checklist

- [ ] Parse `Jais2Config` and load open/gated configs with clear access errors.
- [ ] Validate `hidden_size`, `head_dim`, `num_attention_heads`, and `num_key_value_heads` admission.
- [ ] Load token embeddings, LayerNorm affine params, projection biases, MLP weights, and LM head.
- [ ] Implement LayerNorm parity for Jais2 epsilon.
- [ ] Implement `relu2`.
- [ ] Implement RoPE default theta path and half-rotation convention.
- [ ] Implement causal self-attention MHA; add GQA guard/path.
- [ ] Implement KV cache with post-RoPE K storage.
- [ ] Implement full prefill parity.
- [ ] Implement decode parity with cache extension.
- [ ] Add QKV packing rewrite.
- [ ] Add MLP relu2 fusion or epilogue.
- [ ] Add last-token-only LM head lowering.
- [ ] Add tokenizer/generation-controller metadata handling for BOS/EOS/PAD.
- [ ] Benchmark prefill, decode, MLP, LayerNorm, LM head, and KV memory.
