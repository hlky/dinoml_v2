# Transformers Phi-3 Operator and Integration Report

## 1. Source basis

```text
Transformers commit/version:
  Local checkout X:/H/transformers at b75feb2af64c3e29cbbc1bd859958c5432cc7ed4.
  Remote upstream: https://github.com/huggingface/transformers.git

Model id:
  Primary worked example: microsoft/Phi-3-mini-4k-instruct.
  Additional native phi3 references: microsoft/Phi-3-mini-128k-instruct,
  microsoft/Phi-3-medium-128k-instruct.
  Related but out-of-scope for this native phi3 report:
  microsoft/Phi-3-small-8k-instruct and microsoft/Phi-3-small-128k-instruct
  because their configs use model_type=phi3small and custom remote-code classes.

Config source:
  https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/raw/main/config.json
  https://huggingface.co/microsoft/Phi-3-mini-128k-instruct/raw/main/config.json
  https://huggingface.co/microsoft/Phi-3-medium-128k-instruct/raw/main/config.json
  https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/raw/main/tokenizer_config.json
  https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/raw/main/generation_config.json

Source files inspected:
  X:/H/transformers/src/transformers/models/phi3/modeling_phi3.py
  X:/H/transformers/src/transformers/models/phi3/modular_phi3.py
  X:/H/transformers/src/transformers/models/phi3/configuration_phi3.py
  X:/H/transformers/src/transformers/modeling_rope_utils.py
  X:/H/transformers/src/transformers/cache_utils.py
  X:/H/transformers/src/transformers/masking_utils.py
  X:/H/transformers/src/transformers/integrations/sdpa_attention.py
  X:/H/transformers/src/transformers/integrations/flash_attention.py
  X:/H/transformers/docs/source/en/model_doc/phi3.md

Any missing files or assumptions:
  modeling_phi3.py is generated from modular_phi3.py; the generated file is the
  exact runtime implementation at the inspected commit, while modular_phi3.py is
  the future edit source. The native phi3 directory has no tokenizer file; the
  model docs and tokenizer_config identify a LlamaTokenizer-style tokenizer with
  Phi-3 special tokens. This report assumes inference-only CUDA execution and
  prioritizes decoder-only text generation with KV cache.
```

## 2. High-level architecture

Phi-3 native `phi3` is a text-only decoder transformer. It is close to Llama/Mistral but has two important packing choices: attention uses one packed `qkv_proj`, and the MLP uses one packed `gate_up_proj`. The generation path is causal LM: token IDs become embeddings, decoder blocks run with RoPE and optional sliding-window masking, final RMSNorm feeds a bias-free LM head, and the generation controller samples from logits.

```text
Llama-style tokenization and chat template
  -> token embedding
  -> decoder-only prefill/decode stack with packed QKV, RoPE/LongRoPE, cache
  -> final RMSNorm
  -> lm_head logits, optionally logits_to_keep
  -> sampling/stopping on Phi-3 special EOS IDs
```

Stage decomposition:

- CPU/data pipeline: tokenizer, left padding, chat template, attention mask construction input.
- GPU/runtime prefill: embedding, packed projections, RoPE, causal or sliding-window attention, MLP, final logits.
- GPU/runtime decode: one or more new tokens, cache-length-derived `position_ids`, cache update, last-token logits.
- Independently validateable regions: RMSNorm, packed QKV split, RoPE/LongRoPE, attention with KV cache, packed SwiGLU MLP, LM head.

## 3. Important config dimensions

Worked native example: `microsoft/Phi-3-mini-4k-instruct`.

| Field | Phi-3 mini 4k value | Source / notes |
|---|---:|---|
| architecture | Phi3ForCausalLM | config.json |
| model_type | phi3 | config.json |
| vocab_size / V | 32064 | config.json |
| hidden_size / H | 3072 | config.json |
| intermediate_size / I | 8192 | config.json |
| num_hidden_layers | 32 | config.json |
| num_attention_heads / A | 32 | config.json |
| num_key_value_heads / KvH | 32 | config.json |
| head_dim / D | 96 inferred | `H // A`; source also accepts explicit `head_dim` |
| GQA group size / G | 1 | `A // KvH` |
| packed QKV width | 9216 | `(A + 2*KvH) * D` |
| packed gate/up width | 16384 | `2 * I` |
| max_position_embeddings | 4096 | config.json |
| original_max_position_embeddings | 4096 | config.json/config default |
| rope_theta | 10000 | config.json normalized into `rope_parameters` |
| RoPE type | default | `rope_scaling: null` becomes default |
| sliding_window | 2047 | config.json; native source selects sliding mask when non-null |
| hidden_act | silu | config.json |
| rms_norm_eps | 1e-5 | config.json |
| attention/residual dropout | 0.0 | config.json; disabled in inference |
| attention_bias | false | config metadata; native source ignores this field and hardcodes bias-free projections |
| tie_word_embeddings | false | config.json; source still records a possible tie key |
| torch_dtype | bfloat16 | config metadata |
| use_cache | true | config.json |

Representative checkpoint sweep:

| Checkpoint | Native class scope | H | I | layers | A | KvH | D | G | V | max pos | original pos | sliding_window | RoPE | dtype |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| microsoft/Phi-3-mini-4k-instruct | yes | 3072 | 8192 | 32 | 32 | 32 | 96 | 1 | 32064 | 4096 | 4096 | 2047 | default theta 10000 | bf16 |
| microsoft/Phi-3-mini-128k-instruct | yes | 3072 | 8192 | 32 | 32 | 32 | 96 | 1 | 32064 | 131072 | 4096 | 262144 | longrope, short/long factor length 48 | bf16 |
| microsoft/Phi-3-medium-128k-instruct | yes | 5120 | 17920 | 40 | 40 | 10 | 128 | 4 | 32064 | 131072 | 4096 | 131072 | config says `type: su`, normalized to longrope | bf16 |
| microsoft/Phi-3-small-8k-instruct | no, phi3small remote code | 4096 | 14336 | 32 | 32 | 8 | 128 | 4 | 100352 | 8192 | omitted | n/a | phi3small custom base 1000000 | bf16 |
| microsoft/Phi-3-small-128k-instruct | no, phi3small remote code | 4096 | 14336 | 32 | 32 | 8 | 128 | 4 | 100352 | 131072 | 8192 | n/a | phi3small custom `su` fields | bf16 |

## 3a. Family variation traps

- Native `phi3` mini 4k is MHA (`A=KvH=32`), while medium 128k is GQA (`A=40`, `KvH=10`, group size 4). DinoML must not assume Phi-3 always has full KV heads.
- `head_dim` can be explicit, but source defaults to `hidden_size // num_attention_heads`. Medium uses `5120 // 40 = 128`; mini uses `96`, which differs from many Llama/Mistral 7B assumptions.
- Q/K/V are packed in one bias-free `qkv_proj` with split order `[all Q rows, all K rows, all V rows]`, not three modules.
- MLP gate/up are packed in one bias-free `gate_up_proj` with split order `[gate, up]`, then `up * activation(gate)`.
- Native source hardcodes projection biases as `False`; sampled configs include `attention_bias: false`, but setting it true would not change this source path.
- Long-context configs use old `rope_scaling` payloads with `"type": "longrope"` or legacy `"type": "su"`. `Phi3Config` converts `"su"` and `"yarn"` to `"longrope"` for native source.
- LongRoPE dynamically switches short versus long inverse frequencies based on max `position_ids`; generation explicitly invalidates the cache once the input crosses `original_max_position_embeddings + 1`.
- `sliding_window` is present even for mini 4k (`2047`) and mini 128k (`262144`, larger than max context). Native source selects `create_sliding_window_causal_mask` for any non-null value.
- `Phi-3-small-*` is not implemented by `src/transformers/models/phi3`; its configs advertise `Phi3SmallForCausalLM`, block-sparse attention, GEGLU, different tokenizer class, and model_type `phi3small`. Route it to a separate audit.
- Tokenizer/generation metadata adds stop IDs beyond config `eos_token_id`: generation_config uses EOS list `[32000, 32001, 32007]`.
- The tokenizer uses left padding and a chat template with `<|system|>`, `<|user|>`, `<|assistant|>`, and `<|end|>` tokens. Prompt construction is not part of the core module graph but is required for end-to-end parity.

## 4. Operator coverage checklist

### Tensor/layout ops

- Token embedding gather: `input_ids[B,T] -> hidden[B,T,H]`.
- Flatten leading dims for Linear/GEMM and restore `[B,T,*]`.
- Packed QKV projection and split:
  - `qkv = Linear(H -> (A + 2*KvH)*D, bias=False)`.
  - Q slice `[0 : A*D] -> view [B,T,A,D] -> transpose [B,A,T,D]`.
  - K slice `[A*D : A*D + KvH*D] -> [B,KvH,T,D]`.
  - V slice remaining -> `[B,KvH,T,D]`.
- RoPE split/pass-through on last dim: rotate first `rotary_dim`, concatenate unrotated tail when partial rotary is configured.
- Packed MLP split: `gate_up_proj[B,T,2I] -> chunk(2, dim=-1)`.
- Attention output transpose `[B,A,T,D] -> [B,T,A,D]`, contiguous/reshape to `[B,T,A*D]`.
- KV repeat fallback when backend lacks native GQA: `[B,KvH,K,D] -> [B,A,K,D]`.
- Causal or sliding-window additive mask construction with caller `attention_mask`.
- Hidden-state slice/select for `logits_to_keep`.

### Neural network primitives

- Bias-free GEMMs:
  - Mini QKV: `3072 -> 9216`; O: `3072 -> 3072`.
  - Mini MLP gate/up: `3072 -> 16384`; down: `8192 -> 3072`.
  - Mini LM head: `3072 -> 32064`.
  - Medium QKV: `5120 -> 7680`; O: `5120 -> 5120`.
  - Medium MLP gate/up: `5120 -> 35840`; down: `17920 -> 5120`.
  - Medium LM head: `5120 -> 32064`.
- RMSNorm with fp32 accumulation, scale only, no bias.
- SiLU activation and elementwise multiply for packed SwiGLU-style MLP.
- Residual adds after attention and after MLP. Dropout modules exist but are zero in sampled inference configs.
- LM head is bias-free. Config says untied embeddings, though source declares a tied-weight key for generic tie handling.

### Attention primitives

- Causal decoder self-attention only.
- MHA and GQA.
- Q scaling by `head_dim ** -0.5`.
- RoPE applied to Q and K before cache update.
- DynamicCache K/V append or sliding cache path, depending on config.
- Eager fallback: repeat K/V, score matmul, additive mask, fp32 softmax, dropout, value matmul.
- Optimized backend dispatch through `ALL_ATTENTION_FUNCTIONS`; class declares FlashAttention, SDPA, and flex-attention support.

### Position/rotary ops

- Default RoPE inverse frequencies from `rope_theta`, `head_dim`, and `partial_rotary_factor`.
- LongRoPE inverse frequencies from `short_factor` or `long_factor`, selected by max position.
- Cos/sin are computed in fp32 with autocast disabled, multiplied by `attention_scaling`, then cast to activation dtype.
- `position_ids` default to `arange(T) + past_seen_tokens`.

### Generation/cache ops

- `DynamicCache(config=self.config)` allocation when `use_cache=True` and no cache is provided.
- Cache shape before repeat: per layer K and V `[B,KvH,cache_T,D]`; for sliding cache, storage may be limited to `min(seq_len, sliding_window)` through the shared cache abstraction.
- Cached K is post-RoPE.
- `prepare_inputs_for_generation` may clear `past_key_values` when crossing `original_max_position_embeddings + 1` for LongRoPE correctness.
- `logits_to_keep` avoids full prefill logits.
- Generation metadata requires multiple EOS IDs.

### Preprocessing-coupled ops

- LlamaTokenizer-style SentencePiece tokenization with added Phi-3 special tokens.
- Left padding and prompt chat template are CPU/data-pipeline responsibilities.
- Runtime graph accepts `input_ids`, optional `attention_mask`, optional `position_ids`, optional `past_key_values`, or direct `inputs_embeds`.

### Distributed/tensor-parallel ops

- `Phi3Config.base_model_tp_plan` marks packed `qkv_proj` and `gate_up_proj` as colwise with gathered output because downstream slicing/chunking needs the complete packed vector.
- `o_proj` and `down_proj` are rowwise split-input.
- `lm_head` is colwise gather output. Single-GPU first integration can defer TP, but packed projections need shard-aware split semantics later.

## 5. Layer/block breakdown

Decoder block, repeated `N` times:

```text
residual = x                                             # [B,T,H]
y = RMSNorm(x)

qkv = Linear(H -> (A + 2*KvH)*D, bias=False)(y)          # [B,T,QKV]
q = qkv[..., :A*D].view(B,T,A,D).transpose(1,2)          # [B,A,T,D]
k = qkv[..., A*D:A*D+KvH*D].view(B,T,KvH,D).transpose(1,2)
v = qkv[..., A*D+KvH*D:].view(B,T,KvH,D).transpose(1,2)
cos,sin = rotary(position_ids)                           # [B,T,rotary_dim]
q,k = apply_phi3_rope(q,k,cos,sin)
k,v = cache.update(layer,k,v) if cache enabled
attn = CausalAttention(q,k,v,mask,scale=D**-0.5,sliding_window)
attn = attn.transpose(1,2).reshape(B,T,A*D)
x = residual + Dropout(Linear(A*D -> H, bias=False)(attn))

residual = x
y = RMSNorm(x)
gate_up = Linear(H -> 2I, bias=False)(y)
gate, up = chunk(gate_up, 2, dim=-1)
mlp = Linear(I -> H, bias=False)(up * SiLU(gate))
x = residual + Dropout(mlp)
```

Model head:

```text
x = embed_tokens(input_ids) or inputs_embeds
position_ids = arange(T) + cache_length unless supplied
mask = causal_mask or sliding_window_causal_mask
position_embeddings = rotary_emb(x, position_ids)
x = decoder_layers(x, mask, position_embeddings, cache)
x = RMSNorm(x)
logits = Linear(H -> V, bias=False)(x[:, slice_indices, :])
```

Other heads in source:

- `Phi3ForCausalLM`: required for this report.
- `Phi3Model`: required as the base decoder.
- `Phi3ForSequenceClassification`: optional/deferred; generic classification wrapper.
- `Phi3ForTokenClassification`: optional/deferred; generic token classification wrapper.

## 6. Attention requirements

- Attention is causal self-attention. There is no cross-attention.
- Native source must cover MHA and GQA:
  - Mini: Q `[B,32,T,96]`, K/V `[B,32,T,96]`, group size 1.
  - Medium 128k: Q `[B,40,T,128]`, K/V `[B,10,T,128]`, group size 4.
- Packed QKV storage split is `[Q all heads][K all KV heads][V all KV heads]`; do not assume interleaved per-head QKV rows.
- Masking:
  - If `config.sliding_window is None`, source uses full causal mask.
  - If non-null, source uses sliding-window causal mask and passes `sliding_window` to the attention backend.
  - Mini 128k has `sliding_window=262144`, which is larger than `max_position_embeddings`; it still selects the sliding-window mask path but behaves effectively full-window for normal max context.
- Cache:
  - K/V are cached after RoPE and before any `repeat_kv`.
  - Cache shape is `[B,KvH,cache_T,D]`; eager attention repeats to `[B,A,cache_T,D]`.
  - `DynamicCache(config)` chooses sliding cache layers whenever the config has non-null `sliding_window`.
- Eager math order: repeat K/V, `matmul(q, k.T) * scale`, add mask, softmax in fp32, cast to query dtype, dropout, matmul with V.
- Optimized backend compatibility:
  - SDPA path can use PyTorch `enable_gqa=True` when supported; otherwise it repeats K/V.
  - FlashAttention path receives Q/K/V after RoPE and the `sliding_window` kwarg.
  - DinoML's useful production target is native GQA FlashAttention/paged attention with sliding-window/local support and a KV cache that stores only KV heads.
- Eager fallback is too slow for production because it materializes repeated K/V for GQA and decomposes attention into unfused matmul/softmax/matmul.

## 7. Position encoding and custom math

Default RoPE and Phi-3 partial-rotary handling:

```python
def phi3_default_inv_freq(head_dim, rope_theta, partial_rotary_factor=1.0):
    dim = int(head_dim * partial_rotary_factor)
    i = arange(0, dim, 2, dtype=float32)
    return 1.0 / (rope_theta ** (i / dim))

def phi3_cos_sin(position_ids, inv_freq, attention_scaling, dtype):
    freqs = (inv_freq[None, :, None].float() @ position_ids[:, None, :].float()).transpose(1, 2)
    emb = concat([freqs, freqs], axis=-1)
    return (cos(emb) * attention_scaling).to(dtype), (sin(emb) * attention_scaling).to(dtype)

def apply_phi3_rope(q, k, cos, sin):
    cos = cos[:, None, :, :]
    sin = sin[:, None, :, :]
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_out = concat([q_rot * cos + rotate_half(q_rot) * sin, q_pass], axis=-1)
    k_out = concat([k_rot * cos + rotate_half(k_rot) * sin, k_pass], axis=-1)
    return q_out, k_out
```

LongRoPE parameters:

```python
def phi3_longrope_inv_freq(head_dim, theta, short_factor, long_factor,
                           original_max_position_embeddings, max_position,
                           max_position_embeddings, partial_rotary_factor=1.0,
                           factor=None, attention_factor=None):
    dim = int(head_dim * partial_rotary_factor)
    factor = factor or (max_position_embeddings / original_max_position_embeddings)
    if attention_factor is None:
        attention_factor = 1.0 if factor <= 1.0 else sqrt(1 + log(factor) / log(original_max_position_embeddings))
    ext = long_factor if max_position + 1 > original_max_position_embeddings else short_factor
    i = arange(0, dim, 2, dtype=float32) / dim
    inv_freq = 1.0 / (tensor(ext, dtype=float32) * theta ** i)
    return inv_freq, attention_factor
```

Precompute:

- Default RoPE `inv_freq` and cos/sin tables can be precomputed per `(theta, D, partial_rotary_factor, max_position, dtype)`.
- LongRoPE has two stable factor sets, short and long. DinoML can precompute both tables or switch tables when the generation span crosses `original_max_position_embeddings`.

Dynamic inputs:

- `position_ids` depends on cache length unless supplied.
- LongRoPE update depends on `torch.max(position_ids) + 1`, not merely the current token count.
- Generation crossing the short/long boundary requires cache invalidation and prefill recomputation for parity with source.

## 8. Preprocessing and input packing

CPU/data-pipeline work:

- Tokenize with LlamaTokenizer-compatible model and Phi-3 added special tokens.
- `tokenizer_config.json` sampled for mini 4k has `add_bos_token=false`, `add_eos_token=false`, `padding_side="left"`, `pad_token=<|endoftext|>`, and `model_max_length=4096`.
- Chat template:

```text
<|system|>
...<|end|>
<|user|>
...<|end|>
<|assistant|>
```

- Special sampled IDs include `<|endoftext|>` 32000, `<|assistant|>` 32001, `<|system|>` 32006, `<|end|>` 32007, and `<|user|>` 32010.

GPU/runtime work:

- Inputs are `input_ids[B,T]` or `inputs_embeds[B,T,H]`, optional `attention_mask[B,T]`, optional `position_ids[B,T]`, optional cache.
- For generation, `generation_config.json` uses EOS list `[32000, 32001, 32007]`, BOS 1, and pad 32000. Sampling and stopping live outside the core module graph.
- No image/audio/video tensors, modality placeholders, packed patch rows, or scatter-stitch ops are required for native `phi3`.

## 9. Graph rewrite / lowering opportunities

### Rewrite: Phi3RMSNorm -> RMSNormScaleOnly

Preconditions:

- Exact source form: cast to fp32, mean of squared values on last dim, multiply by `rsqrt(var + eps)`, cast normalized tensor to input dtype, multiply by `[H]` scale.
- No bias and no mean subtraction.

Replacement:

```text
RMSNorm(x, weight, eps, axis=-1, fp32_accum=True)
```

Shape equations:

- Input/output `[*,H]`; weight `[H]`.

Failure cases:

- Do not rewrite generic LayerNorm or a norm with bias.
- Preserve fp32 accumulation for bf16/fp16 parity.

Parity test sketch:

- Compare random fp32/bf16 tensors for `H=3072` and `H=5120`, eps `1e-5`.

### Rewrite: packed Phi3 QKV -> split GEMM outputs

Preconditions:

- Source module is `Phi3Attention.qkv_proj` with bias false.
- Packed output width is `(A + 2*KvH) * D`.
- Split order is `[Q, K, V]` as contiguous row blocks.

Replacement:

```text
GEMM_RCR(H -> packed_qkv) -> Slice(Q,K,V) -> View/Transpose
```

Weight transform:

```python
w_q = w_qkv[: A * D, :]
w_k = w_qkv[A * D : A * D + KvH * D, :]
w_v = w_qkv[A * D + KvH * D :, :]
```

Failure cases:

- Do not infer K/V width from `hidden_size`; use `KvH * D`.
- Do not rewrite phi3small remote-code projections through this rule without a separate audit.

Parity test sketch:

- Compare packed projection and split tensors before RoPE for mini 4k and medium 128k shapes.

### Rewrite: packed gate_up_proj -> SwiGLU MLP

Preconditions:

- Source module is `Phi3MLP.gate_up_proj`, bias false.
- Output width is `2 * intermediate_size`.
- Split order is `gate, up`.
- Activation is `ACT2FN[hidden_act]`; sampled native phi3 uses `silu`.

Replacement:

```text
GEMM_RCR(H -> 2I) -> Split(gate, up) -> up * SiLU(gate) -> GEMM_RCR(I -> H)
```

Failure cases:

- If a future config changes `hidden_act`, use the configured activation.
- A backend fusion must preserve source order `up * activation(gate)`.

Parity test sketch:

- Compare MLP output for random `[B,T,H]` tensors, including bf16 tolerance.

### Rewrite: LongRoPE table selection

Preconditions:

- `rope_parameters.rope_type == "longrope"`.
- `short_factor` and `long_factor` lengths equal `rotary_dim // 2`.
- Generation scheduler can detect crossing `original_max_position_embeddings`.

Replacement:

```text
if max(position_ids) + 1 <= original_max_position_embeddings:
    use short_rope_table
else:
    invalidate/recompute prefill cache and use long_rope_table
```

Failure cases:

- Unsafe if cache is retained across the short/long factor switch.
- Unsafe if legacy `type: su` is not normalized to longrope before validation.

Parity test sketch:

- For mini 128k, compare cos/sin at positions below 4096 and above 4096; test generation cache reset at token 4097.

### Rewrite: sliding-window mask -> local causal attention

Preconditions:

- `config.sliding_window is not None`.
- Backend implements the same causal local visibility as HF mask utilities.
- Cache layer policy matches the same window semantics.

Replacement:

```text
AdditiveMaskAttention(q,k,v,mask) -> LocalCausalGQAAttention(q,k,v,window=sliding_window)
```

Failure cases:

- Mini 128k has a window larger than max context; local attention should reduce to full causal within normal bounds.
- If a backend ignores the sliding window kwarg, only use it when `sliding_window is None` or proven full-window.

Parity test sketch:

- Use small synthetic windows to compare local attention against HF additive masks for prompt lengths below, equal to, and above the window.

### Rewrite: logits_to_keep -> last-token LM head

Preconditions:

- Inference/generation does not need all token logits.
- `labels is None`.
- `logits_to_keep` is an int or explicit position tensor.

Replacement:

```text
SliceHidden([B,T,H] -> [B,K,H]) -> LMHeadGEMM(H -> V)
```

Failure cases:

- Training/loss and full-sequence scoring require full logits.

Parity test sketch:

- Compare `logits_to_keep=1`, `0`, and explicit tensor indices against source.

## 10. Kernel fusion candidates

Highest priority:

- RMSNorm: two per block plus final norm, fp32 accumulation, bandwidth-sensitive.
- Packed QKV GEMM + split + RoPE layout path: avoids extra materialization and handles both MHA and GQA.
- Native GQA FlashAttention/paged attention with KV cache and sliding-window support.
- Packed SwiGLU MLP fusion: `gate_up_proj -> chunk -> SiLU(gate) * up` is a large activation path.
- Last-token-only LM head via `logits_to_keep`.

Medium priority:

- LongRoPE short/long table management and cache invalidation.
- Sliding-window mask lowering to local attention for mini 4k and any future constrained-window configs.
- KV cache append/pack kernels storing `[B,KvH,T,D]` without repeated KV heads.
- Sharded packed projection lowering following the source TP plan.

Lower priority:

- Sequence and token classification heads.
- Dropout, training loss, and gradient checkpointing.
- Remote-code Phi3Small block-sparse attention, GEGLU, and custom tokenizer. Treat as separate family.

## 11. Runtime staging plan

Stage 1: Parse `Phi3Config`, normalize `rope_scaling` into `rope_parameters`, reject or route non-native `model_type != "phi3"`.

Stage 2: Load embeddings, packed QKV weights, packed gate/up weights, RMSNorm scales, down/O projections, and LM head.

Stage 3: Implement RMSNorm, packed projection splitting, and packed MLP parity on random tensors.

Stage 4: Implement default RoPE and one-layer mini 4k prefill parity without cache.

Stage 5: Add causal/sliding-window mask parity and native/eager attention equivalence.

Stage 6: Add `DynamicCache`-style decode with cached K/V stored post-RoPE and pre-repeat.

Stage 7: Add LongRoPE support for mini/medium 128k, including short/long factor switch and cache reset behavior.

Stage 8: Enable optimized GQA attention, local attention, and last-token LM head.

Stage 9: Add production features: sharded packed projections, paged KV cache, continuous batching, and sampling integration.

Initial stubs:

- Keep tokenizer/chat template in CPU preprocessing.
- Defer classification heads.
- For first parity, materialized eager attention is acceptable before optimized attention lands.

## 12. Parity and validation plan

- Config normalization tests:
  - mini 4k default RoPE.
  - mini 128k `type=longrope`.
  - medium 128k legacy `type=su` normalized to longrope.
  - reject `phi3small` for native phi3 path.
- RMSNorm random tensor tests for fp32/bf16/fp16.
- Packed QKV split tests for MHA and GQA shapes.
- Packed MLP tests for `hidden_act=silu`.
- RoPE tests:
  - default mini positions.
  - LongRoPE positions below and above `original_max_position_embeddings`.
  - partial rotary pass-through if a synthetic config sets `partial_rotary_factor < 1`.
- Attention tests:
  - eager repeat vs native GQA for medium shape.
  - sliding-window mask with small synthetic windows.
  - cache shape and value parity across 2-4 decode steps.
- Single decoder layer parity for mini-like and medium-like dimensions.
- Prefill logits parity with `logits_to_keep=1`.
- LongRoPE generation boundary parity: verify cache is dropped/recomputed when crossing token 4097 for 128k configs.
- End-to-end text generation smoke test with the Phi-3 chat template and EOS list.
- Suggested tolerances: fp32 `rtol=1e-5, atol=1e-6`; bf16/fp16 initially `rtol=2e-2, atol=2e-2`, then tighten after fused kernels are stable.

## 13. Performance probes

- Prefill throughput sweep for mini and medium shapes: `B`, `T`, and context length.
- Decode tokens/sec with cache lengths around 1K, 4K, 32K, and 128K.
- KV cache memory:

```text
bytes = 2 * layers * B * KvH * cache_T * D * dtype_bytes
mini MHA:   2 * 32 * B * 32 * T * 96 * bytes_per_elem
medium GQA: 2 * 40 * B * 10 * T * 128 * bytes_per_elem
```

- Eager repeated-KV memory overhead versus native GQA for medium.
- Sliding-window mask/local attention cost for mini 4k.
- LongRoPE table generation and boundary recompute cost.
- Packed QKV single GEMM versus split GEMM alternatives.
- Packed MLP activation bandwidth and fusion benefit.
- LM head full logits versus `logits_to_keep=1`.
- Prompt preprocessing/tokenization throughput and chat-template cost, kept separate from GPU runtime.

## 14. Skip/defer list

- Training, loss, dropout behavior, and gradient checkpointing.
- Sequence classification and token classification heads.
- Beam search, speculative decoding, and assistant generation.
- Multi-GPU TP/PP for first parity, though packed projection sharding should be planned early.
- Quantization formats and GGUF ingestion for first native phi3 parity.
- Remote-code `phi3small` configs with block-sparse attention and GEGLU.
- Full tokenizer implementation inside DinoML runtime; keep tokenization in the data pipeline.
- Exotic generation processors beyond EOS stopping and basic sampling.

## 15. Final implementation checklist

- [ ] Parse native `Phi3Config` and reject/route `model_type=phi3small`.
- [ ] Normalize `rope_scaling`/`rope_parameters`, including legacy `su -> longrope`.
- [ ] Load embeddings, packed QKV, packed gate/up, norms, projections, and LM head.
- [ ] Implement Phi3 RMSNorm with fp32 accumulation.
- [ ] Implement packed QKV GEMM split with MHA/GQA widths.
- [ ] Implement default RoPE and Phi-3 partial-rotary apply.
- [ ] Implement LongRoPE short/long factor selection.
- [ ] Implement cache reset behavior at the LongRoPE original-context boundary.
- [ ] Implement causal and sliding-window masks.
- [ ] Implement native GQA attention or safe repeat fallback.
- [ ] Implement KV cache storing post-RoPE pre-repeat K/V.
- [ ] Implement packed SiLU MLP.
- [ ] Implement `logits_to_keep` and last-token LM head.
- [ ] Add packed projection, RoPE, attention, MLP, and cache parity tests.
- [ ] Add prefill and decode logits parity tests.
- [ ] Benchmark prefill, decode, KV memory, LongRoPE, MLP, and LM head.
