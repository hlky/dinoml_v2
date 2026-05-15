# Phi Transformers Family Audit

Primary target: `PhiForCausalLM` decoder-only text generation on CUDA. This is a docs-only source/config audit; no DinoML runtime code was edited and no DinoML tests were run.

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model family: phi
Primary runtime target: PhiForCausalLM causal LM prefill, decode, and generation
Dinoml assumptions: inference-only first, CUDA GPU target, preserve Transformers tensor axes, prefer explicit rewrites for partial RoPE, LayerNorm, parallel residual, attention/cache, and GELU MLP.
```

Source files inspected:

- Local authoritative modular source: `transformers/src/transformers/models/phi/modular_phi.py`
- Local generated concrete source: `transformers/src/transformers/models/phi/modeling_phi.py`
- Local config source: `transformers/src/transformers/models/phi/configuration_phi.py`
- Local conversion helper: `transformers/src/transformers/models/phi/convert_phi_weights_to_hf.py`
- Shared source: `transformers/src/transformers/cache_utils.py`, `transformers/src/transformers/masking_utils.py`, `transformers/src/transformers/modeling_rope_utils.py`, `transformers/src/transformers/activations.py`
- Upstream source URL pattern at pinned commit: `https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/phi/...`

Authoritative source note:

- `modeling_phi.py` is generated from `modular_phi.py`; future Transformers source edits should be checked in `modular_phi.py`, while `modeling_phi.py` is the concrete implementation audited for runtime behavior.

Representative Hugging Face configs and metadata inspected:

- `https://huggingface.co/microsoft/phi-1/raw/main/config.json`
- `https://huggingface.co/microsoft/phi-1_5/raw/main/config.json`
- `https://huggingface.co/microsoft/phi-2/raw/main/config.json`
- `https://huggingface.co/microsoft/phi-2/raw/main/generation_config.json`
- `https://huggingface.co/microsoft/phi-2/raw/main/tokenizer_config.json`
- `https://huggingface.co/microsoft/phi-2/raw/main/special_tokens_map.json`
- Historical legacy/remote-code configs visible in Hub history for `microsoft/phi-1` and `microsoft/phi-2`, used only to identify native-vs-remote-code traps.

Missing files or assumptions:

- There is no `tokenization_phi.py` in this Transformers directory. Official Phi repos use `CodeGenTokenizer` tokenizer metadata rather than a Phi-specific tokenizer class.
- No current official `microsoft/phi-1`, `microsoft/phi-1_5`, or `microsoft/phi-2` config requires `trust_remote_code=True` after the Hub configs were updated for native `PhiForCausalLM`.
- Historical configs carried `auto_map` and legacy fields such as `n_embd`, `n_head`, `n_layer`, `rotary_dim`, `activation_function`, and `layer_norm_epsilon`. This report targets the current native `PhiConfig`/`PhiForCausalLM` path, not historical remote-code behavior.
- Parameter counts, license, and model-card task claims are not used as operator facts here. Dimensions are from `config.json` or source defaults.

## 2. High-level architecture

Phi is a text-only decoder-only Transformer for causal language modeling. It uses learned token embeddings, partial RoPE on Q/K, causal self-attention, a non-gated GELU MLP, LayerNorm rather than RMSNorm, and a distinctive parallel residual block: attention and MLP both consume the same normalized hidden state and are added together with the residual.

```text
CodeGen byte-level BPE tokenization -> input_ids/attention_mask
  -> token embedding + embedding dropout
  -> repeated Phi decoder blocks
  -> final LayerNorm
  -> biased LM head logits
  -> generation controller / sampling
```

Generation stage split:

```text
CPU tokenizer -> prefill full prompt -> per-layer K/V cache
              -> decode one or more new tokens with position offset
              -> logits_to_keep / lm_head -> sampler/controller
```

Independently stageable pieces:

- CPU/data pipeline: CodeGen tokenizer, special token handling, padding/attention mask construction.
- GPU prefill: embedding lookup, causal mask, shared RoPE cos/sin, all decoder blocks, optional K/V cache population.
- GPU decode: one-token or chunked new token embedding, position IDs offset by cache length, K/V cache append, attention over prior plus new tokens, last-token logits.
- Block-level validation: Phi's parallel residual block can be validated independently from the tokenizer and generation sampler.

Implemented heads in source:

- Required for target: `PhiForCausalLM`.
- Optional/deferred: base `PhiModel` for hidden states.
- Deferred for first causal-LM integration: sequence classification and token classification via generic heads.

## 3. Important config dimensions

Source defaults from `PhiConfig`:

| Field | Default / behavior | Lowering effect |
| --- | ---: | --- |
| `vocab_size` | 51200 | Token embedding and LM head rows |
| `hidden_size` | 2048 | Hidden width `H` |
| `intermediate_size` | 8192 | MLP width `I` |
| `num_hidden_layers` | 24 | Decoder block count |
| `num_attention_heads` | 32 | Query head count `A` |
| `num_key_value_heads` | `None` -> `num_attention_heads` | KV heads `KvH`; source can express GQA/MQA if set lower |
| `head_dim` | `hidden_size // num_attention_heads` unless an attr exists | Per-head width `D` |
| `hidden_act` | `gelu_new` | tanh GELU approximation |
| `max_position_embeddings` | 2048 | RoPE cache/default max context |
| `rope_parameters` | standardized from old fields; `partial_rotary_factor` defaulted to 0.5 for BC | Partial RoPE dimensions |
| `rope_theta` | effectively 10000.0 in sampled configs | RoPE base |
| `layer_norm_eps` | 1e-5 | All LayerNorm modules |
| `qk_layernorm` | false | Optional Q/K LayerNorm before RoPE |
| `use_cache` | true | Generation cache enabled by default |
| `tie_word_embeddings` | false | LM head usually separate from token embedding |
| `resid_pdrop`, `embd_pdrop`, `attention_dropout` | 0.0 source/default except Phi-2 `resid_pdrop=0.1` | Dropout is identity in eval |

Representative checkpoint sweep:

| Model id | Layers | H | Heads/KV | D | Rotary factor | Rotary dims | MLP | Max pos | Vocab | LN eps | dtype metadata | BOS/EOS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `microsoft/phi-1` | 24 | 2048 | 32 / null -> 32 | 64 | 0.5 | 32 | 8192 | 2048 | 51200 | 1e-5 | float32 in current config | null/null |
| `microsoft/phi-1_5` | 24 | 2048 | 32 / null -> 32 | 64 | 0.5 | 32 | 8192 | 2048 | 51200 | 1e-5 | float16 | null/null |
| `microsoft/phi-2` | 32 | 2560 | 32 / 32 | 80 | 0.4 | 32 | 10240 | 2048 | 51200 | 1e-5 | float16 | 50256/50256 |
| `PhiConfig()` synthetic default | 24 | 2048 | 32 / null -> 32 | 64 | 0.5 | 32 | 8192 | 2048 | 51200 | 1e-5 | source default only | 1/2 |

Generation/tokenizer metadata inspected:

| Repo | Tokenizer class | `model_max_length` | Special tokens | Generation config |
| --- | --- | ---: | --- | --- |
| `microsoft/phi-2` | `CodeGenTokenizer` | 2048 | BOS/EOS/UNK all `<\|endoftext\|>` | `_from_model_config=true`, BOS/EOS 50256 |

Notable derived dimensions:

- Phi-1/Phi-1.5 Q/K/V/O are all `2048 -> 2048`; MLP is `2048 -> 8192 -> 2048`; LM head is `2048 -> 51200` with bias.
- Phi-2 Q/K/V/O are all `2560 -> 2560`; MLP is `2560 -> 10240 -> 2560`; LM head is `2560 -> 51200` with bias.
- The official sampled configs all rotate exactly 32 dimensions per head despite different head widths: `64 * 0.5 = 32` for Phi-1/1.5 and `80 * 0.4 = 32` for Phi-2.

## 3a. Family variation traps

- Phi is not a Llama-style serial residual block. There is only one per-block `input_layernorm`; attention and MLP both consume that normalized tensor, then `hidden = residual + attn + mlp`.
- Phi uses `nn.LayerNorm`, not RMSNorm. Q/K optional normalization also uses affine `LayerNorm` over head dimension.
- MLP is non-gated: `Linear -> gelu_new -> Linear`. Do not lower it as SwiGLU.
- All attention projections and the attention output projection have bias in native source.
- The LM head is `nn.Linear(hidden_size, vocab_size, bias=True)`. Even though `_tied_weights_keys` lists `lm_head.weight` and `model.embed_tokens.weight`, official configs set `tie_word_embeddings=false`; preserve separate LM head weight and bias unless a checkpoint explicitly ties.
- Source supports GQA/MQA through `num_key_value_heads`, but official sampled configs are MHA. Do not assume all future `model_type="phi"` configs are MHA.
- `num_key_value_heads=null` is effective MHA after config post-init. Loader/admission should normalize it before shape planning.
- Partial RoPE rotates only the prefix of the head dimension and concatenates the unrotated tail back. Fused RoPE kernels that assume full-head rotation are wrong for Phi.
- `config.head_dim` is not declared in `PhiConfig`, but `PhiAttention` reads it if present. If an exotic config sets `head_dim != hidden_size // num_attention_heads`, Q projection width becomes `num_attention_heads * head_dim`; Q/K LayerNorm still constructs with `hidden_size // num_attention_heads`, so such configs should be rejected unless validated separately.
- Historical official configs used remote-code `auto_map` plus legacy names (`n_embd`, `n_head`, `n_layer`, `n_positions`, `rotary_dim`, `activation_function`, `layer_norm_epsilon`, `attn_pdrop`, `n_head_kv`, `flash_attn`, `flash_rotary`, `fused_dense`). Current native source does not read most of those names directly. DinoML should either translate known legacy configs to modern `PhiConfig` fields or reject/reroute remote-code configs.
- Historical remote-code flags such as `flash_attn`, `flash_rotary`, and `fused_dense` are not native-source behavior in this commit. Native source dispatches attention through `ALL_ATTENTION_FUNCTIONS` via `config._attn_implementation`.
- No image/audio/layout translation applies. Protect text axes: embeddings `[B,S,H]`, LayerNorm axis `-1`, Q/K/V reshape/transpose axes, RoPE final head dimension, attention softmax `dim=-1`, and logits `[B,S,V]`.

## 4. Operator coverage checklist

Tensor/layout ops:

- `input_ids[int64] [B,S]` or mutually exclusive `inputs_embeds [B,S,H]`.
- Token embedding lookup: `Embedding(vocab_size, H) -> [B,S,H]`.
- Dropout as inference identity for `embed_dropout`, `resid_dropout`, attention dropout.
- Flatten leading dims for GEMM/Linear and restore `[B,S,*]`.
- View/reshape projection outputs:
  - Q: `[B,S,A*D] -> [B,S,A,D] -> [B,A,S,D]`
  - K/V: `[B,S,KvH*D] -> [B,S,KvH,D] -> [B,KvH,S,D]`
- Slices along final head dim for partial RoPE: `rot = [..., :rotary_ndims]`, `pass = [..., rotary_ndims:]`, then concat.
- Optional `repeat_kv` fallback: `[B,KvH,S,D] -> [B,A,S,D]`.
- Cache update/append along sequence axis.
- Residual additions: three-way `residual + attention_output + mlp_output`.
- Slice/gather for `logits_to_keep` before LM head.

Neural network primitives:

- LayerNorm over final hidden axis `H`, epsilon `layer_norm_eps`, affine weight and bias.
- Optional Q/K LayerNorm over per-head `D`, affine weight and bias, before RoPE.
- Biased Linear:
  - Q: `H -> A*D`
  - K: `H -> KvH*D`
  - V: `H -> KvH*D`
  - Attention output `dense`: `A*D -> H`
  - MLP `fc1`: `H -> I`
  - MLP `fc2`: `I -> H`
  - LM head: `H -> vocab_size`
- `gelu_new`: `0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))`.
- Softmax in fp32 in eager attention, then cast to query dtype.

Attention primitives:

- Causal self-attention only.
- MHA for official checkpoints; GQA/MQA possible through config.
- Q/K partial RoPE before cache update.
- Eager fallback: repeat KV, QK matmul, scale by `head_dim ** -0.5`, add causal/padding mask, fp32 softmax, dropout, PV matmul.
- Backend dispatch through `ALL_ATTENTION_FUNCTIONS` for eager/SDPA/Flash/Flex-compatible paths.

Position/rotary/cache/generation ops:

- Position IDs default to `arange(S) + past_key_values.get_seq_length()` and shape `[1,S]`.
- RoPE cos/sin generation in fp32 with autocast disabled, cast back to hidden dtype.
- `DynamicCache(config)` creation when `use_cache=True` and no cache is passed.
- Per-layer cache stores K/V after partial RoPE for K and after V projection for V.
- `logits_to_keep`: `0` keeps all positions; positive int keeps trailing positions; tensor input gathers arbitrary positions.

Preprocessing-coupled ops:

- CodeGen tokenizer CPU pipeline: byte-level BPE files, `CodeGenTokenizer`, `add_prefix_space=false`.
- `attention_mask [B,total_kv]` is optional and consumed by shared mask utilities.
- No multimodal stitch tensors, packed patch descriptors, token type IDs, or `cu_seqlens` are part of native Phi.

Distributed metadata:

- Config declares TP plan: Q/K/V and `fc1` column-wise; attention `dense` and `fc2` row-wise; LM head column-wise gather output. Defer for first single-GPU parity.

## 5. Layer/block breakdown

For hidden width `H`, query heads `A`, KV heads `KvH`, head dim `D`, MLP width `I`, sequence chunk `T`, and total key length `Ktot`:

```text
input_ids [B,T]
inputs_embeds = embed_tokens(input_ids)         # [B,T,H]
position_ids = arange(T) + past_seen_tokens     # [1,T], unless caller supplies [B,T]
causal_mask = create_causal_mask(...)
position_embeddings = rotary_emb(hidden, position_ids)
```

Decoder block, repeated `num_hidden_layers` times:

```text
residual = hidden                               # [B,T,H]
x = LayerNorm(hidden, eps=layer_norm_eps)

q = Linear_bias_q(x)                            # [B,T,A*D]
k = Linear_bias_k(x)                            # [B,T,KvH*D]
v = Linear_bias_v(x)                            # [B,T,KvH*D]
q = view(q, [B,T,A,D]).transpose(1, 2)           # [B,A,T,D]
k = view(k, [B,T,KvH,D]).transpose(1, 2)         # [B,KvH,T,D]
v = view(v, [B,T,KvH,D]).transpose(1, 2)         # [B,KvH,T,D]

if qk_layernorm:
    q = LayerNorm(q, axis=-1, eps=layer_norm_eps)
    k = LayerNorm(k, axis=-1, eps=layer_norm_eps)

q_rot, q_pass = q[..., :rotary_ndims], q[..., rotary_ndims:]
k_rot, k_pass = k[..., :rotary_ndims], k[..., rotary_ndims:]
q_rot, k_rot = partial_rope(q_rot, k_rot, cos, sin)
q = concat(q_rot, q_pass, axis=-1)
k = concat(k_rot, k_pass, axis=-1)

k, v = cache.update(k, v, layer_idx) optional   # [B,KvH,Ktot,D]
attn = causal_attention(q, k, v, causal_mask, scale=D**-0.5)
attn = reshape(attn.transpose(1, 2), [B,T,A*D])
attn = Linear_bias_dense(attn)                  # [B,T,H]

mlp = Linear_bias_fc1(x)                        # [B,T,I]
mlp = gelu_new(mlp)
mlp = Linear_bias_fc2(mlp)                      # [B,T,H]

hidden = residual + dropout(attn) + dropout(mlp)
```

Final causal LM:

```text
hidden = final_layernorm(hidden)                # [B,T,H]
selected = hidden[:, slice_indices, :]          # logits_to_keep
logits = lm_head(selected)                      # [B,T_keep,vocab], bias=True
```

Example shapes:

- Phi-1/Phi-1.5: `H=2048`, `A=KvH=32`, `D=64`, `rotary_ndims=32`, `I=8192`.
- Phi-2: `H=2560`, `A=KvH=32`, `D=80`, `rotary_ndims=32`, `I=10240`.

## 6. Attention requirements

Required attention variant:

- Decoder self-attention, causal.
- Official checkpoints are MHA: `num_key_value_heads == num_attention_heads` after null normalization.
- Native source supports GQA/MQA if a config sets `num_key_value_heads < num_attention_heads`; eager path physically repeats K/V to query heads.
- No cross-attention, no ALiBi, no sliding-window/local attention in native Phi.

Shapes:

- New Q: `[B,A,T,D]`.
- New K/V: `[B,KvH,T,D]`.
- Cache per layer after update:
  - key `[B,KvH,Ktot,D]`
  - value `[B,KvH,Ktot,D]`
- If eager repeat is used, attention math sees K/V as `[B,A,Ktot,D]`.
- Scores: `[B,A,T,Ktot]`.
- Output before projection: `[B,T,A*D]`.

Cache placement:

- K is cached after optional Q/K LayerNorm and partial RoPE.
- V is cached after V projection/reshape, with no positional transform.
- Cache stores KV heads, not repeated query heads.

Masking and math order:

```text
scores = matmul(q, repeat_kv(k).transpose(-1,-2)) * (D ** -0.5)
scores = scores + attention_mask if present
probs = softmax(scores, dim=-1, dtype=float32).to(q.dtype)
probs = dropout(probs) in training only
context = matmul(probs, repeat_kv(v))
```

Backend compatibility:

- Source declares FlashAttention, SDPA, and FlexAttention support through the shared attention interface.
- DinoML should start from eager parity, then lower to native MHA/GQA fused attention under strict guards: causal mask or causal+standard padding mask, no requested attention weights, dropout disabled, and supported partial-RoPE/cache layout.
- Eager repeat is acceptable for parity but should not be the production GQA path if non-MHA Phi configs are admitted.

## 7. Position encoding and custom math

Phi uses partial RoPE. The RoPE table is generated only for `rotary_ndims = int(head_dim * partial_rotary_factor)`, and the remaining head dimensions pass through unchanged.

Source-equivalent default RoPE:

```python
def phi_inv_freq(head_dim, partial_rotary_factor, rope_theta):
    dim = int(head_dim * partial_rotary_factor)
    return 1.0 / (rope_theta ** (arange(0, dim, 2).float() / dim))

def phi_rotary_embedding(position_ids, inv_freq, dtype):
    freqs = (inv_freq[None, :, None] @ position_ids[:, None, :].float()).transpose(1, 2)
    emb = cat([freqs, freqs], dim=-1)
    return cos(emb).to(dtype), sin(emb).to(dtype)

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return cat([-x2, x1], dim=-1)

def apply_phi_partial_rope(q, k, cos, sin, rotary_ndims):
    q_rot, q_pass = q[..., :rotary_ndims], q[..., rotary_ndims:]
    k_rot, k_pass = k[..., :rotary_ndims], k[..., rotary_ndims:]
    cos = cos[:, None, :, :]
    sin = sin[:, None, :, :]
    q_rot = q_rot * cos + rotate_half(q_rot) * sin
    k_rot = k_rot * cos + rotate_half(k_rot) * sin
    return cat([q_rot, q_pass], dim=-1), cat([k_rot, k_pass], dim=-1)
```

Custom math:

```python
def gelu_new(x):
    return 0.5 * x * (1.0 + tanh(sqrt(2.0 / pi) * (x + 0.044715 * x**3)))
```

Precompute opportunities:

- For default RoPE, `inv_freq` is static for `rope_theta`, `head_dim`, and `partial_rotary_factor`.
- Cos/sin can be precomputed up to admitted max position, but only for the rotated prefix width.
- Decode can compute/slice just the new position rows.

Dynamic behavior:

- `position_ids` default depends on current cache length.
- Shared `dynamic_rope_update` is attached to the rotary module for advanced RoPE types. Official sampled Phi configs use default RoPE (`rope_scaling=null`, `rope_theta=10000.0`), so first integration can reject non-default `rope_type`/scaling.

## 8. Preprocessing and input packing

CPU/data-pipeline tokenizer facts from `microsoft/phi-2` metadata:

- Tokenizer class: `CodeGenTokenizer`.
- `model_max_length=2048`.
- `add_prefix_space=false`.
- BOS/EOS/UNK are `<|endoftext|>`.
- The tokenizer metadata contains many added whitespace/tab tokens above ID 50256; the model vocab size is 51200, so these are ordinary tokenizer/model vocabulary entries, not runtime graph structure.

Runtime graph inputs:

- Required: `input_ids [B,S] int64` or `inputs_embeds [B,S,H]`, exactly one.
- Optional: `attention_mask [B,K]` or backend-compatible prepared mask.
- Optional: `position_ids [B,S]`; if omitted, source uses `[1,S]` arange plus cache length.
- Optional: `past_key_values`; if absent and `use_cache=True`, source creates `DynamicCache(config)`.
- No token type IDs, segment IDs, multimodal placeholders, packed image/audio descriptors, or `cu_seqlens` are native Phi inputs.

Generation-controller behavior:

- `microsoft/phi-2` generation metadata has BOS/EOS 50256.
- `phi-1` and `phi-1_5` current configs have null BOS/EOS in `config.json`; generation callers may need tokenizer/controller defaults.
- Sampling, beam search, stopping criteria, and prompt formatting are outside the compiled module graph for first integration.

## 9. Graph rewrite / lowering opportunities

### Rewrite: biased Linear -> GEMM with bias epilogue

Source pattern:

```text
y = nn.Linear(in_features, out_features, bias=True)(x)
```

Replacement:

```text
FlattenLeading -> GEMM_RCR or GEMM_RRR depending on stored weight plan -> BiasAdd -> RestoreLeading
```

Preconditions:

- Input last dimension equals projection input width.
- PyTorch linear weight is stored `[out_features, in_features]`.
- Bias is present for all native Phi projections and LM head.
- Dropout is disabled/eval identity.

Shape equations:

- Flatten `[B,S,K] -> [B*S,K]`; output `[B*S,N] -> [B,S,N]`.

Failure cases:

- Treating Phi projections as bias-free Llama/Qwen-like projections loses required bias.
- Tied LM-head assumptions are wrong for official configs unless explicitly tied by loader.

Parity test sketch:

- Compare Q/K/V/O, MLP up/down, and LM head projections for Phi-1 and Phi-2 shapes, including bias.

### Rewrite: partial RoPE as prefix-only RoPE

Source pattern:

```text
q_rot, q_pass = q[..., :rotary_ndims], q[..., rotary_ndims:]
k_rot, k_pass = k[..., :rotary_ndims], k[..., rotary_ndims:]
q_rot, k_rot = apply_rope(q_rot, k_rot, cos, sin)
q = cat(q_rot, q_pass, dim=-1)
k = cat(k_rot, k_pass, dim=-1)
```

Replacement:

```text
PartialRoPE(q,k, rotary_ndims, cos/sin) -> Q/K with unrotated tail preserved
```

Preconditions:

- `rotary_ndims` is even and equals cos/sin final dimension.
- RoPE table built from `dim=rotary_ndims`, not full `head_dim`.
- Layout is `[B,heads,S,D]` before applying.

Failure cases:

- Full-head RoPE silently corrupts tail dimensions.
- Computing inverse frequencies with denominator `head_dim` instead of `rotary_ndims` changes phase.

Parity test sketch:

- Compare Q/K before and after RoPE for Phi-1 (`D=64,rot=32`) and Phi-2 (`D=80,rot=32`) at positions including decode offsets.

### Rewrite: Phi parallel residual block

Source pattern:

```text
x = LayerNorm(hidden)
attn = self_attn(x)
mlp = fc2(gelu_new(fc1(x)))
hidden = hidden + attn + mlp
```

Replacement:

```text
LayerNorm once -> branch to attention and MLP -> fused/add tree residual + attn + mlp
```

Preconditions:

- Both branches consume the same normalized tensor.
- No second post-attention LayerNorm exists.
- Dropout disabled for inference.

Failure cases:

- Serializing as `hidden = hidden + attn; mlp(LN(hidden))` produces a different model.
- Reusing Llama/Qwen block templates without this branch structure will fail parity.

Parity test sketch:

- One-block random-hidden parity with attention stubbed or real attention, verifying MLP input is the pre-attention normalized tensor.

### Rewrite: native MHA/GQA attention

Source pattern:

```text
repeat_kv(k/v) -> matmul(q,k^T) -> scale -> mask add -> fp32 softmax -> matmul(v)
```

Replacement:

```text
Fused causal attention(q[B,A,T,D], k/v[B,KvH,K,D], group_size=A/KvH)
```

Preconditions:

- `A % KvH == 0`.
- Q/K have already received optional QK LayerNorm and partial RoPE.
- Cache stores post-RoPE K and projected V.
- Attention weights are not requested.
- Dropout disabled.

Failure cases:

- Unsupported masks or advanced RoPE types should fall back or reject.
- Physical KV repeat should not be used for production GQA if memory matters.

Parity test sketch:

- Compare eager output and fused attention for MHA official shapes plus a synthetic GQA config.

### Rewrite: last-token-only biased LM head

Source pattern:

```text
slice_indices = slice(-logits_to_keep, None) if int else logits_to_keep
logits = lm_head(hidden_states[:, slice_indices, :])
```

Replacement:

```text
Slice/Gather selected hidden rows -> GEMM with lm_head.weight and lm_head.bias
```

Preconditions:

- Labels/loss are not requested.
- Generation consumes only final token or a known token subset.
- LM head bias is included.

Failure cases:

- Full-sequence logprobs/loss need all logits.
- Assuming tied embedding weight without checking config/weight aliasing.

Parity test sketch:

- Compare `logits_to_keep=0`, `1`, positive tail counts, and tensor index selection against HF.

## 10. Kernel fusion candidates

Highest priority:

- LayerNorm with affine bias: one per block plus final norm; optional Q/K LayerNorm if admitted.
- Biased GEMM coverage for every projection and LM head.
- Partial RoPE fused with Q/K layout transform, preserving unrotated tail.
- Causal prefill/decode attention with artifact-visible K/V cache.
- Phi parallel residual fusion: branch from one normalized tensor and sum residual, attention, and MLP outputs.

Medium priority:

- MLP fusion around `fc1 -> gelu_new -> fc2`; at minimum fuse `gelu_new` with the activation write.
- Grouped Q/K/V GEMMs with bias; no packed source weights, but same input feeds all three projections.
- Last-token-only logits with biased LM head.
- Cache append/update kernels for `[B,KvH,T,D]`.

Lower priority:

- Q/K LayerNorm path; official sampled configs set `qk_layernorm=false`.
- Non-default RoPE scaling/dynamic rope variants.
- Sequence/token classification heads.
- Tensor parallel sharding.
- Beam cache reorder and advanced generation processors.

## 11. Runtime staging plan

1. Parse modern `PhiConfig`, normalize `num_key_value_heads=null` to `num_attention_heads`, and standardize RoPE parameters.
2. Add config admission: accept default/native Phi fields first; reject or explicitly translate historical remote-code configs with `n_embd`/`rotary_dim`/`auto_map`.
3. Load token embedding, all biased projections, LayerNorm weights/biases, final LayerNorm, and biased LM head; preserve actual tie/alias state instead of assuming tying.
4. Implement primitive parity for affine LayerNorm, `gelu_new`, biased Linear/GEMM, and partial RoPE.
5. Implement one Phi decoder block with eager attention and parallel residual.
6. Implement full prefill logits parity with no cache and with cache population.
7. Implement decode cache append, position offset, and `logits_to_keep=1`.
8. Add optimized MHA attention for official Phi configs.
9. Add synthetic GQA admission/parity only if a selected Phi checkpoint needs `num_key_value_heads < num_attention_heads`.
10. Add fusions: partial RoPE/layout, grouped QKV GEMMs, MLP activation fusion, residual add fusion, and last-token LM head.

Initially stub/defer:

- Training loss, dropout randomness, gradient checkpointing.
- Sequence/token classification heads.
- Historical remote-code flags and custom kernels.
- Non-default RoPE scaling.
- Tokenizer execution inside DinoML runtime.
- Beam/speculative decoding and multi-GPU tensor parallelism.

## 12. Parity and validation plan

Primitive tests:

- Affine LayerNorm over hidden dim for `H=2048` and `H=2560`, fp32/fp16/bf16.
- Optional Q/K LayerNorm synthetic test over `D=64` and `D=80`.
- `gelu_new` random tensor parity against Transformers activation.
- Biased Linear/GEMM tests for `2048 -> 2048`, `2048 -> 8192`, `8192 -> 2048`, `2560 -> 2560`, `2560 -> 10240`, `10240 -> 2560`, and LM head widths.
- Partial RoPE tests for `(D=64,rot=32)` and `(D=80,rot=32)`, including nonzero decode offsets.
- `repeat_kv`/GQA synthetic tests for `A/KvH` groups even though official Phi is MHA.

Model/block tests:

- One-block random-input parity in fp32, verifying parallel residual semantics.
- Two-layer tiny synthetic config parity for hidden states and cache shapes.
- Full prefill logits parity for fixed token IDs on accessible Phi weights or a randomly initialized HF/DinoML paired model.
- Decode parity: prefill, then several one-token decode steps; compare logits and per-layer K/V cache shape/content.
- Mask parity: no padding, standard padding masks, and decode masks with `past_length + T`.
- `logits_to_keep` parity for all logits, last token, tail count, and tensor indices.
- Config admission tests for modern configs and historical legacy/remote-code fields.

Recommended tolerances:

- fp32 eager primitives/block: `rtol=1e-4`, `atol=1e-5`.
- fp16/bf16 optimized attention/logits: start with `rtol=3e-2`, `atol=3e-2`, then tighten after backend-specific comparison.
- Token-level end-to-end greedy parity should be checked after logits parity is stable.

## 13. Performance probes

- Prefill throughput: `B={1,4,16}`, `S={1,128,512,2048}` for Phi-1/Phi-2 shapes.
- Decode tokens/sec: cache lengths `{16,128,512,2048}`, batch sizes `{1,4,16,64}`.
- KV cache memory:
  - Formula: `layers * 2 * B * KvH * max_seq * D * dtype_bytes`.
  - Phi-1/Phi-1.5 fp16 at `B=1,S=2048`: `24*2*32*2048*64*2 ~= 402.7 MB`.
  - Phi-2 fp16 at `B=1,S=2048`: `32*2*32*2048*80*2 ~= 671.1 MB`.
- Attention backend comparison: eager matmul/softmax vs fused MHA, and synthetic GQA if admitted.
- Partial RoPE overhead: table generation/slice vs fused apply in Q/K layout.
- Parallel residual scheduling: materialize both branches vs fused add tree.
- MLP probe: `gelu_new` activation bandwidth and `fc1/fc2` GEMM times.
- LM head probe: full `[B,S,V]` logits vs `logits_to_keep=1`.
- Weight residency/dequant probes for biased projections once DinoML GGUF/runtime-dequant support is applicable.

All probes are proposed; no benchmark observations are included.

## 14. Skip/defer list

- Training, labels/loss, dropout randomness, and gradient checkpointing.
- Sequence classification and token classification heads.
- Returning attention weights/hidden states as a first optimized path.
- Historical remote-code implementation parity and config-only flags (`flash_attn`, `flash_rotary`, `fused_dense`) unless explicitly selected as a separate audit.
- Non-default/dynamic RoPE variants.
- Beam search cache reorder, speculative decoding, and advanced logits processors.
- Tokenizer execution inside the GPU runtime.
- Tensor parallel/pipeline parallel execution.
- Quantized weights and quantized KV cache until dense parity is established.
- Exotic configs with explicit `head_dim` different from `hidden_size // num_attention_heads`.

## 15. Final implementation checklist

- [ ] Parse modern `PhiConfig` and normalize `num_key_value_heads`.
- [ ] Standardize RoPE parameters and admit default partial RoPE first.
- [ ] Add config guard/translator for historical remote-code Phi configs.
- [ ] Load embeddings, decoder weights, affine LayerNorms, and biased LM head.
- [ ] Preserve actual embedding/LM-head tying or separation.
- [ ] Implement affine LayerNorm, including optional Q/K LayerNorm.
- [ ] Implement biased Linear/GEMM lowering for attention, MLP, and LM head.
- [ ] Implement `gelu_new`.
- [ ] Implement partial RoPE with unrotated head tail.
- [ ] Implement Phi parallel residual block exactly.
- [ ] Implement causal MHA attention and cache layout `[B,KvH,T,D]`.
- [ ] Add native GQA attention or reject non-MHA configs until supported.
- [ ] Implement decode cache append and position offset.
- [ ] Implement `logits_to_keep` / last-token biased LM-head lowering.
- [ ] Add primitive, one-block, prefill, decode, mask, and logits-slicing parity tests.
- [ ] Benchmark prefill, decode, KV memory, partial RoPE, MLP, and LM-head slices.
