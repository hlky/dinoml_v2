# Transformers Gemma Operator and Integration Report

## 1. Source basis

```text
Transformers commit/version:
  Local checkout X:/H/transformers at b75feb2af64c3e29cbbc1bd859958c5432cc7ed4.
  Remote upstream: https://github.com/huggingface/transformers.git

Model family / architecture:
  gemma

Primary runtime target:
  Text-only decoder-only causal language-model generation.

Source files inspected:
  X:/H/transformers/src/transformers/models/gemma/modular_gemma.py
  X:/H/transformers/src/transformers/models/gemma/modeling_gemma.py
  X:/H/transformers/src/transformers/models/gemma/configuration_gemma.py
  X:/H/transformers/src/transformers/models/gemma/tokenization_gemma.py
  X:/H/transformers/src/transformers/models/gemma/convert_gemma_weights_to_hf.py
  X:/H/transformers/src/transformers/cache_utils.py
  X:/H/transformers/src/transformers/masking_utils.py
  X:/H/transformers/src/transformers/modeling_rope_utils.py

Pinned source URLs for future review:
  https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/gemma/modular_gemma.py
  https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/gemma/modeling_gemma.py
  https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/gemma/configuration_gemma.py
  https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/gemma/tokenization_gemma.py

Representative configs and tokenizer/generation metadata:
  Official google/gemma-* and google/codegemma-* raw files were gated in this
  environment, returning HTTP 401. Open mirrors and converted/derived repos were
  used for config examples and are labeled as such:
  https://huggingface.co/mfuntowicz/gemma-2b/raw/main/config.json
  https://huggingface.co/alpindale/gemma-2b-it/raw/main/config.json
  https://huggingface.co/milandean/gemma-7b/raw/main/config.json
  https://huggingface.co/OpenVINO/gemma-7b-fp16-ov/raw/main/config.json
  https://huggingface.co/unsloth/codegemma-2b/raw/main/config.json
  https://huggingface.co/unsloth/codegemma-7b/raw/main/config.json
  https://huggingface.co/milandean/gemma-7b/raw/main/tokenizer_config.json
  https://huggingface.co/milandean/gemma-7b/raw/main/generation_config.json
  https://huggingface.co/alpindale/gemma-2b-it/raw/main/tokenizer_config.json
  https://huggingface.co/unsloth/codegemma-2b/raw/main/tokenizer_config.json
  https://huggingface.co/unsloth/codegemma-2b/raw/main/generation_config.json

Any missing files or assumptions:
  modeling_gemma.py and configuration_gemma.py are generated from
  modular_gemma.py. The generated files are the exact runtime source in this
  checkout; modular_gemma.py is authoritative for future Transformers source
  edits. No remote-code files are required for standard Gemma. This report
  assumes inference-only CUDA execution. No DinoML tests were run.
```

## 2. High-level architecture

Gemma is a text-only decoder stack with tied token embeddings and LM head for
causal generation.

```text
tokenizer -> input_ids/attention_mask
  -> scaled token embedding
  -> decoder prefill/decode stack with RoPE self-attention and KV cache
  -> final RMSNorm
  -> lm_head logits, optionally logits_to_keep
  -> generation controller/sampling
```

Stage decomposition:

- CPU/data pipeline: BPE tokenizer with byte fallback, normalizer replacing
  spaces with `▁`, optional chat template outside this source file, and
  left-padding conventions from tokenizer metadata.
- Prefill: full prompt through all decoder layers, causal mask, RoPE, GQA/MQA
  attention, final norm, and logits.
- Decode: one or more new tokens with `DynamicCache`; cached keys are stored
  after RoPE and before any KV repeat expansion.
- Logits/sampling: `GemmaForCausalLM.forward` only produces logits. Beam search,
  temperature/top-k/top-p sampling, and stopping criteria live in Transformers
  generation utilities and are not part of the core module graph.

## 3. Important config dimensions

Source defaults from `GemmaConfig`:

| Field | Source default | Runtime significance |
|---|---:|---|
| `vocab_size` | 256000 | token embedding and LM head width |
| `hidden_size` | 3072 | residual stream width |
| `intermediate_size` | 24576 | gated MLP width |
| `num_hidden_layers` | 28 | decoder block count |
| `num_attention_heads` | 16 | query head count |
| `num_key_value_heads` | 16 | KV head count; may be less than query heads |
| `head_dim` | 256 | explicit Q/K/V head width |
| `hidden_act` | `gelu_pytorch_tanh` | gated MLP activation in current source default |
| `max_position_embeddings` | 8192 | nominal RoPE/generation context |
| `rms_norm_eps` | 1e-6 | RMSNorm epsilon |
| `attention_bias` | false | Q/K/V/O projection bias flag |
| `attention_dropout` | 0.0 | inference path uses zero dropout |
| `use_cache` | true | default cache support |
| `tie_word_embeddings` | true | `lm_head.weight` aliases `embed_tokens.weight` |
| `rope_parameters` | from `rope_theta` or default | standardized by config mixin |
| `use_bidirectional_attention` | null | source-supported noncausal override, not used by sampled generation configs |

Representative checkpoint/config sweep:

| Config source | Scope | H | I | Layers | Q heads | KV heads | D | KV groups | Vocab | Max pos | RoPE theta | Activation | Dtype |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `mfuntowicz/gemma-2b` mirror | base 2B | 2048 | 16384 | 18 | 8 | 1 | 256 | 8 | 256000 | 8192 | 10000 | `gelu` in config | bf16 |
| `alpindale/gemma-2b-it` mirror | instruct 2B | 2048 | 16384 | 18 | 8 | 1 | 256 | 8 | 256000 | 8192 | 10000 | `gelu` in config | bf16 |
| `milandean/gemma-7b` mirror | base 7B | 3072 | 24576 | 28 | 16 | 16 | 256 | 1 | 256000 | 8192 | 10000 | `gelu` in config | bf16 |
| `OpenVINO/gemma-7b-fp16-ov` mirror | converted 7B | 3072 | 24576 | 28 | 16 | 16 | 256 | 1 | 256000 | 8192 | 10000 | `gelu` in config | fp16 |
| `unsloth/codegemma-2b` mirror | CodeGemma 2B | 2048 | 16384 | 18 | 8 | 1 | 256 | 8 | 256000 | 8192 | 10000 | `gelu` | bf16 |
| `unsloth/codegemma-7b` mirror | CodeGemma 7B | 3072 | 24576 | 28 | 16 | 16 | 256 | 1 | 256000 | 8192 | 10000 | `gelu_pytorch_tanh` in newer mirror | bf16 |

Tokenizer/generation metadata observed from open mirrors:

| Source | Tokenizer class | BOS/EOS handling | Padding | Generation notes |
|---|---|---|---|---|
| `milandean/gemma-7b/tokenizer_config.json` | `GemmaTokenizer` | `add_bos_token=true`, `add_eos_token=false` | source class default left padding | generation config only sets BOS/EOS/PAD ids |
| `alpindale/gemma-2b-it/tokenizer_config.json` | `GemmaTokenizer` | `add_bos_token=true`, `add_eos_token=false` | no explicit side in file | generation config only sets BOS/EOS/PAD ids |
| `unsloth/codegemma-2b/tokenizer_config.json` | `GemmaTokenizer` | `add_bos_token=true`, `add_eos_token=false` | `padding_side=left` | generation config has `max_length=8192` |

## 3a. Family variation traps

- Gemma is not Gemma2/Gemma3. This source has no multimodal branch, no SigLIP
  tower, no sliding/full hybrid layer pattern, no Q/K head norms, no
  post-attention/post-FFN extra norms, and no configured logits softcapping.
- The 2B family uses MQA/GQA: `num_key_value_heads=1`, `num_attention_heads=8`.
  The 7B family uses full MHA: `num_key_value_heads=16`,
  `num_attention_heads=16`.
- `hidden_size` can differ from `num_attention_heads * head_dim`. For 2B,
  `2048 == 8 * 256`; for 7B, `3072 != 16 * 256`. Q/O projection width is
  `num_attention_heads * head_dim`, not necessarily `hidden_size`.
- Current source default is `hidden_act="gelu_pytorch_tanh"`, but many older
  configs contain `hidden_act="gelu"` or only the legacy extra field
  `hidden_activation`. DinoML should parse the exact loaded config and reject
  unsupported activation mismatches instead of assuming the source default.
- `rope_theta` from older configs is standardized into `config.rope_parameters`
  by `RotaryEmbeddingConfigMixin`. The runtime source reads
  `config.rope_parameters["rope_type"]` and `["rope_theta"]`.
- The converter reveals original checkpoint QKV packing. Native Gemma
  checkpoints may contain packed `qkv_proj`; HF weights are split into separate
  `q_proj`, `k_proj`, `v_proj`. DinoML should treat HF safetensors as split
  weights unless ingesting original Google checkpoints directly.
- `use_bidirectional_attention` toggles `GemmaAttention.is_causal`, but the mask
  helper checks `config.is_causal`. This field is a source-supported trap for
  non-generation use, not required for standard causal LM parity.
- Token embeddings are scaled by `sqrt(hidden_size)` using a registered buffer
  cast to embedding weight dtype. This differs from LLaMA-style unscaled token
  embeddings.
- RMSNorm stores a zero-centered parameter and multiplies by `1 + weight`.
  Loading or folding norms must preserve that one-plus contract.
- LM head and token embedding are tied by default. Lowering must preserve one
  logical parameter alias, especially with GGUF/encoded constant ownership.
- No tanh softcapping is present in the audited Gemma source. Gemma3 reports
  may mention optional softcapping; do not import that behavior into Gemma.

## 4. Operator coverage checklist

### Tensor/layout ops

- Token embedding gather: `input_ids[B,T] -> [B,T,H]`.
- Embedding scale multiply by `sqrt(H)` in embedding dtype.
- Optional `inputs_embeds` path with mutual exclusion against `input_ids`.
- Position id generation: `arange(T) + past_seen_tokens`, then unsqueeze to
  `[1,T]`.
- Causal mask construction from `attention_mask`, `past_key_values`, and
  `position_ids`.
- Linear input flattening over `[B,T]` leading dims and reshape back.
- Q/K/V projection reshape and transpose:
  - Q: `[B,T,A*D] -> [B,A,T,D]`.
  - K/V: `[B,T,KvH*D] -> [B,KvH,T,D]`.
- RoPE cos/sin broadcast from `[B,T,D]` to `[B,1,T,D]`.
- Eager fallback `repeat_kv`: `[B,KvH,S,D] -> [B,A,S,D]`.
- Attention output transpose/contiguous and reshape to `[B,T,A*D]`.
- Residual adds.
- `logits_to_keep` slicing before LM head: int `0` means keep all logits,
  positive int keeps last N positions, tensor indices select arbitrary positions.

### Neural network primitives

- Bias-free Linear/GEMM for sampled configs:
  - 2B Q `2048 -> 2048`, K/V `2048 -> 256`, O `2048 -> 2048`.
  - 2B MLP gate/up `2048 -> 16384`, down `16384 -> 2048`.
  - 2B LM head `2048 -> 256000`, tied with embedding.
  - 7B Q `3072 -> 4096`, K/V `3072 -> 4096`, O `4096 -> 3072`.
  - 7B MLP gate/up `3072 -> 24576`, down `24576 -> 3072`.
  - 7B LM head `3072 -> 256000`, tied with embedding.
- Gemma RMSNorm with fp32 accumulation and `(1 + weight.float())`.
- Gated MLP: `down(act(gate(x)) * up(x))`.
- Activation dispatch for `hidden_act`, with `gelu` and `gelu_pytorch_tanh`
  both seen in representative configs.
- Final RMSNorm before logits.

### Attention primitives

- Causal decoder self-attention.
- MHA for 7B and MQA/GQA for 2B/CodeGemma 2B.
- RoPE applied to Q and K before cache update.
- `DynamicCache` per-layer K/V update.
- Eager fallback order: matmul QK, multiply by `head_dim**-0.5`, add mask,
  softmax in fp32, cast to query dtype, dropout, matmul V.
- Optimized dispatch through `ALL_ATTENTION_FUNCTIONS` when `_attn_implementation`
  selects FlashAttention, SDPA, or FlexAttention-compatible backends.

### Position/rotary ops

- Default RoPE inverse frequency with `base=rope_theta`, `dim=head_dim`.
- Optional advanced RoPE types through `ROPE_INIT_FUNCTIONS` if configs provide
  non-default `rope_parameters`; not observed in sampled base configs.
- `rotate_half` split at `head_dim // 2`.

### Generation/cache ops

- `DynamicCache(config)` initialization when `use_cache=True` and no cache is
  supplied.
- Per-layer cache tensor logical shape before repeat:
  `[B, num_key_value_heads, past_seq + current_seq, head_dim]`.
- Cached keys are already RoPE-rotated.
- `past_key_values.get_seq_length()` drives default position offsets.
- Last-token or last-N logits path through `logits_to_keep`.

### Preprocessing-coupled ops

- BPE tokenizer with byte fallback and `▁` space normalization.
- Left padding is the source tokenizer class default; mirrors may or may not
  repeat it in tokenizer config.
- BOS is normally added, EOS normally not added, based on tokenizer metadata.
- CodeGemma FIM tokens are tokenizer/controller behavior, not a different core
  graph.

## 5. Layer/block breakdown

Model setup:

```text
inputs_embeds = Embedding(V,H)(input_ids) * sqrt(H)
if use_cache and no past: past_key_values = DynamicCache(config)
position_ids = arange(T) + past_key_values.get_seq_length()
causal_mask = create_causal_mask(config, inputs_embeds, attention_mask, past, position_ids)
cos, sin = rotary_emb(hidden_states, position_ids)
```

Decoder block, repeated `num_hidden_layers` times:

```text
residual = x
y = RMSNorm(input)(x)
q = Linear(H -> A*D, bias=attention_bias)(y).view(B,T,A,D).transpose(1,2)
k = Linear(H -> KvH*D, bias=attention_bias)(y).view(B,T,KvH,D).transpose(1,2)
v = Linear(H -> KvH*D, bias=attention_bias)(y).view(B,T,KvH,D).transpose(1,2)
q,k = apply_rope(q,k,cos,sin)
k,v = cache.update(k,v,layer_idx) if cache exists
a = attention(q,k,v, causal_mask, scale=D**-0.5)
a = a.transpose(1,2).reshape(B,T,A*D)
a = Linear(A*D -> H, bias=attention_bias)(a)
x = residual + a

residual = x
y = RMSNorm(post_attention)(x)
y = Linear(I -> H)(act(Linear(H -> I)(y)) * Linear(H -> I)(y))
x = residual + y
```

LM head:

```text
x = final RMSNorm(x)
logits = Linear(H -> V, bias=False)(x[:, slice_indices, :])
```

There is no attention logit softcap or final logit softcap in this Gemma source.

## 6. Attention requirements

- Type: decoder self-attention.
- Masking: causal by default through `create_causal_mask`; bidirectional mode
  is out-of-scope for first generation parity.
- Head geometry:
  - 2B: Q `[B,8,T,256]`, K/V `[B,1,T,256]`, group size 8.
  - 7B: Q `[B,16,T,256]`, K/V `[B,16,T,256]`, group size 1.
- Scale: `head_dim ** -0.5`.
- RoPE: Q/K only, before cache update.
- Cache shape per layer before repeat: `[B,KvH,S,256]` for both key and value.
- Repeat expansion shape in eager fallback: `[B,A,S,256]`.
- Cache growth: no sliding window in Gemma configs sampled, so full cache grows
  to total seen sequence length. `DynamicCache(config)` falls back to
  `full_attention` layers when no `sliding_window` or `layer_types` exist.
- Packed/varlen: `masking_utils` has packed sequence detection for training-like
  cases when `attention_mask is None` and no cache is present. First inference
  integration can ignore packed training batches and require normal causal
  generation inputs.
- FlashAttention/SDPA compatibility: the source advertises FlashAttention, SDPA,
  FlexAttention, and generic attention backend support. A DinoML optimized
  attention path must preserve native GQA/MQA without materializing repeated
  K/V, RoPE-before-cache, fp32 softmax semantics or documented tolerance, and
  the additive mask order.

Eager math order to preserve for fallback parity:

```text
attn = matmul(q, repeat_kv(k).transpose(-2,-1)) * (D ** -0.5)
attn = attn + additive_mask
prob = softmax(attn, dim=-1, dtype=float32).to(q.dtype)
out = matmul(dropout(prob), repeat_kv(v))
```

## 7. Position encoding and custom math

Default RoPE:

```python
def gemma_inv_freq(config):
    base = config.rope_parameters["rope_theta"]
    dim = config.head_dim
    i = arange(0, dim, 2, dtype=float32)
    return 1.0 / (base ** (i / dim))

def gemma_rope(position_ids, inv_freq, attention_scaling, dtype):
    freqs = (inv_freq[None, :, None].float() @ position_ids[:, None, :].float()).transpose(1, 2)
    emb = concat([freqs, freqs], axis=-1)
    cos = cos(emb) * attention_scaling
    sin = sin(emb) * attention_scaling
    return cos.to(dtype), sin.to(dtype)

def apply_gemma_rope(q, k, cos, sin):
    cos = cos[:, None, :, :]
    sin = sin[:, None, :, :]
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin
```

RMSNorm:

```python
def gemma_rms_norm(x, weight, eps):
    y = x.float()
    y = y * rsqrt(mean(y * y, axis=-1, keepdims=True) + eps)
    y = y * (1.0 + weight.float())
    return y.to(x.dtype)
```

Embedding scaling:

```python
inputs_embeds = embedding(input_ids) * tensor(sqrt(hidden_size)).to(embedding_weight_dtype)
```

Precompute opportunities:

- Default RoPE inv-frequencies are config constants.
- Fixed-context cos/sin tables can be cached by `(position, dtype, device)` for
  default RoPE. Dynamic/non-default RoPE from `ROPE_INIT_FUNCTIONS` needs a
  config guard.
- `(1 + RMSNorm.weight)` can be folded at load time if the artifact records that
  the source parameter was Gemma one-plus RMSNorm.

## 8. Preprocessing and input packing

Text runtime inputs:

- Required: `input_ids[B,T]` or `inputs_embeds[B,T,H]`, exactly one.
- Optional: `attention_mask[B,S]`, `position_ids[1 or B,T]`,
  `past_key_values`, `use_cache`, `logits_to_keep`.
- Tokenizer source class:
  - BPE with byte fallback.
  - Normalizer replaces spaces with `▁`.
  - Pre-tokenizer splits on spaces with merged previous behavior.
  - Decoder reverses `▁`, byte fallback, and fuse.
  - `padding_side = "left"` in the class.
- Mirror tokenizer metadata generally sets `add_bos_token=true` and
  `add_eos_token=false`.
- Generation config metadata in sampled mirrors is minimal: token ids, and in
  one CodeGemma mirror `max_length=8192`.

No image, audio, packed multimodal placeholders, token type ids, or
`cu_seqlens`-style metadata are part of this Gemma family.

## 9. Graph rewrite / lowering opportunities

### Rewrite: GemmaRMSNorm -> RMSNormOnePlusWeight

Source pattern:

```text
x.float() * rsqrt(mean(x.float() ** 2, dim=-1) + eps)
-> multiply by (1 + weight.float())
-> cast to input dtype
```

Replacement:

```text
RMSNorm(axis=-1, eps, fp32_accum=True, scale=(1 + weight))
```

Preconditions:

- Weight shape matches normalized last dimension.
- No bias term.
- Preserve fp32 accumulation and one-plus scale semantics.

Failure cases:

- Generic RMSNorm kernels that multiply by raw `weight` will be wrong unless
  weight folding happens once and is recorded in constant provenance.

Parity test sketch:

- Compare hidden-size norms for 2048/3072 and activation dtypes fp32/fp16/bf16.

### Rewrite: PyTorch Linear -> GEMM_RCR

Source pattern:

```text
nn.Linear(in_features, out_features, bias=False or attention_bias)
```

Replacement:

```text
FlattenLeadingDims -> GEMM_RCR(A, weight[out,in]) -> optional bias -> Reshape
```

Preconditions:

- Activation is dense row-major.
- HF weight layout is `[out_features, in_features]`.
- Bias absent in sampled configs; if `attention_bias=True`, use bias epilogue.

Shape equations:

- Q out = `num_attention_heads * head_dim`.
- K/V out = `num_key_value_heads * head_dim`.
- O in = `num_attention_heads * head_dim`.

Failure cases:

- Assuming `A * D == H` breaks 7B (`4096 != 3072`).

### Rewrite: separate Q/K/V projections -> grouped or concatenated QKV

Source pattern:

```text
q_proj(normed_x), k_proj(normed_x), v_proj(normed_x)
```

Replacement:

```text
GroupedGEMM(q,k,v) or ConcatenatedLinear(H -> (A + 2*KvH) * D) -> split
```

Preconditions:

- Same input tensor and dtype.
- Bias settings match.
- Weight constants have compatible residency/quantization policy.

Weight transform:

```python
w_qkv = concat([w_q, w_k, w_v], axis=0)
```

Failure cases:

- Q/K/V output widths differ for MQA/GQA.
- Original Google packed checkpoints use a source-specific `qkv_proj` layout;
  this rewrite applies to HF split weights, not raw checkpoint ingestion.

### Rewrite: eager repeat_kv attention -> native GQA/MQA attention

Source pattern:

```text
repeat_kv(k, A/KvH), repeat_kv(v, A/KvH), matmul/softmax/matmul
```

Replacement:

```text
GQAAttention(q[B,A,Q,D], k[B,KvH,K,D], v[B,KvH,K,D], group_size=A/KvH)
```

Preconditions:

- `A % KvH == 0`.
- Backend preserves scale, causal mask, RoPE-before-cache, and dtype behavior.

Failure cases:

- Materialized repeat-KV is acceptable for early parity but too memory-heavy for
  production 2B/CodeGemma decode.

### Rewrite: tied embedding/LM head as one logical constant

Source pattern:

```text
model.embed_tokens.weight aliases lm_head.weight
```

Replacement:

```text
One logical constant, two consumers: embedding gather and GEMM/logits projection.
```

Preconditions:

- `tie_word_embeddings=True` and `_tied_weights_keys` matches
  `lm_head.weight -> model.embed_tokens.weight`.

Failure cases:

- Duplicating large vocab weights doubles memory and can break future
  GGUF/offload residency accounting.

## 10. Kernel fusion candidates

Highest priority:

- RMSNormOnePlusWeight: two decoder norms per layer plus final norm; exact
  `(1 + weight)` math is required.
- Native GQA/MQA attention with KV cache: 2B and CodeGemma 2B otherwise pay
  repeat-KV memory and bandwidth.
- RoPE + attention layout staging: every layer rotates Q/K before cache update.
- Gated GELU MLP fusion: `act(gate) * up` is a large bandwidth path.
- Last-token-only logits: source `logits_to_keep` avoids full `[B,T,V]` logits.

Medium priority:

- Grouped Q/K/V projection with unequal output widths.
- Tied embedding/LM-head constant residency and GGUF/offload-aware aliasing.
- Embedding scale folded into embedding output kernel or into a fused gather.
- Fused residual add after attention/MLP where it fits memory planning.

Lower priority:

- Bidirectional attention mode.
- Sequence/token classification heads.
- Advanced RoPE scaling variants not present in sampled base configs.
- Raw Google checkpoint `qkv_proj` conversion inside DinoML runtime.

## 11. Runtime staging plan

Stage 1: Config and weights.

- Parse `GemmaConfig`, including `head_dim`, `num_key_value_heads`,
  `rope_parameters`, `hidden_act`, `attention_bias`, and tied-weight metadata.
- Load HF split weights and preserve embedding/LM-head aliasing.

Stage 2: One-block parity.

- Implement scaled embedding, Gemma RMSNorm, Linear/GEMM, gated MLP, RoPE, and
  eager attention fallback.
- Validate 2B and 7B geometry because 7B has `A*D != H`.

Stage 3: Full prefill parity.

- Build full decoder stack with causal mask and final norm.
- Add `logits_to_keep` and full/last-token logits checks.

Stage 4: Decode with KV cache.

- Implement full dynamic per-layer KV cache shape `[B,KvH,S,D]`.
- Validate position offsets and RoPE-before-cache behavior over several decode
  steps.

Stage 5: Optimized attention and fusion.

- Replace repeat-KV fallback with native GQA/MQA attention.
- Add RMSNorm, RoPE, gated MLP, and last-token logits fusions.

Stage 6: Weight residency/quantization.

- Add tied-weight-aware constant planning and later GGUF/offload integration for
  embedding/LM head and projection weights.

## 12. Parity and validation plan

- Config parser tests for 2B, 7B, and CodeGemma mirror configs.
- Tokenizer metadata smoke tests for BOS/EOS and left padding assumptions.
- RMSNorm unit tests for `(1 + weight)` and fp32 accumulation.
- Embedding scale unit test comparing `Embedding(input_ids) * sqrt(H)`.
- RoPE unit tests for `head_dim=256`, `rope_theta=10000`, and position offsets
  after non-empty cache.
- Projection shape tests:
  - 2B Q/K/V/O with MQA.
  - 7B Q/O width 4096 against hidden 3072.
- Gated MLP activation parity for `gelu` and `gelu_pytorch_tanh` configs.
- Eager attention parity for MHA and MQA/GQA, including fp32 softmax.
- Cache parity across prefill plus 2-4 decode tokens.
- Full one-layer parity, then full prefill logits parity.
- Decode token parity with `logits_to_keep=1`.
- Tied-weight alias test proving embedding and LM head consume one logical
  parameter.
- Suggested tolerances: fp32 custom math `rtol=1e-5, atol=1e-5`; fp16/bf16
  layerwise `rtol=2e-2, atol=2e-2` initially, tightened after attention backend
  math order is fixed.

No DinoML tests were run for this docs-only audit.

## 13. Performance probes

- Prefill tokens/sec by batch size and sequence length for 2B and 7B shapes.
- Decode tokens/sec by batch size and cache length.
- KV cache memory:
  - per layer = `2 * B * num_key_value_heads * S * head_dim * dtype_bytes`.
  - 2B benefits strongly from one KV head; preserve native GQA.
- Native GQA attention versus repeat-KV fallback latency and memory.
- RoPE generation versus cached cos/sin lookup.
- RMSNorm bandwidth and fusion benefit.
- MLP GEMM plus gated activation bandwidth.
- LM head full logits versus `logits_to_keep=1`.
- Embedding/LM-head tied constant residency, including future GGUF/offload
  memory accounting.
- Config-specific activation throughput for `gelu` versus `gelu_pytorch_tanh`.

These are proposed probes, not measurements.

## 14. Skip/defer list

Safe to defer for first generation integration:

- Training, labels/loss, dropout, gradient checkpointing.
- Sequence and token classification heads.
- Bidirectional attention mode.
- Raw Google checkpoint conversion from packed `qkv_proj`; load HF split
  safetensors first.
- Advanced RoPE scaling variants unless selected target configs require them.
- Quantized AWQ/GPTQ/GGUF weight loading beyond DinoML's existing encoded
  constant roadmap.
- Multi-GPU tensor/pipe parallel plans.
- Beam search, speculative decoding, and sampling controller features outside
  core model execution.
- CodeGemma FIM prompt construction as a generation-controller/tokenizer task.

## 15. Final implementation checklist

- [ ] Parse `GemmaConfig` and normalize `rope_parameters`.
- [ ] Load HF split weights and preserve tied embedding/LM-head aliasing.
- [ ] Implement scaled token embedding.
- [ ] Implement Gemma RMSNorm with `(1 + weight)` and fp32 accumulation.
- [ ] Implement Linear/GEMM shapes using `num_heads * head_dim`, not hidden size.
- [ ] Implement gated MLP with config-selected activation.
- [ ] Implement default RoPE and `apply_rotary_pos_emb`.
- [ ] Implement causal mask handling for generation inputs.
- [ ] Implement eager MHA/GQA attention fallback.
- [ ] Implement native GQA/MQA attention optimized path.
- [ ] Implement full dynamic KV cache `[B,KvH,S,D]`.
- [ ] Implement `logits_to_keep`.
- [ ] Add config, RMSNorm, RoPE, projection, MLP, attention, and cache parity tests.
- [ ] Add one-block, full prefill, and cached decode parity tests.
- [ ] Add tied-weight constant residency/alias test.
- [ ] Benchmark prefill, decode, KV memory, MLP, RMSNorm, and LM-head slices.
