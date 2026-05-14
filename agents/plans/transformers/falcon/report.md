# Falcon Transformers Family Audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: Falcon family; representative sweep: tiiuae/falcon-rw-1b, tiiuae/falcon-rw-7b, tiiuae/falcon-7b, tiiuae/falcon-40b, tiiuae/falcon-11B
Config source: Official Hugging Face config.json/tokenizer_config.json/generation_config.json where available
Source files inspected:
  X:/H/transformers/src/transformers/models/falcon/configuration_falcon.py
  X:/H/transformers/src/transformers/models/falcon/modeling_falcon.py
  X:/H/transformers/src/transformers/models/falcon/convert_custom_code_checkpoint.py
  X:/H/transformers/tests/models/falcon/test_modeling_falcon.py
  X:/H/transformers/src/transformers/configuration_utils.py
  X:/H/transformers/src/transformers/modeling_rope_utils.py
Any missing files or assumptions:
  The pinned Falcon directory has no local tokenization_falcon.py; tokenizer facts below come from official HF tokenizer metadata.
  Older repo cards still mention trust_remote_code, but current in-library configs use model_type="falcon" and the inspected Falcon source.
  Falcon-180B is gated/licensed; only repo metadata/model card was consulted, not config JSON, so it is not in the dimension table.
```

Source URLs for future review:

- `modeling_falcon.py`: https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/falcon/modeling_falcon.py
- `configuration_falcon.py`: https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/falcon/configuration_falcon.py
- Representative configs: https://huggingface.co/tiiuae/falcon-rw-1b/raw/main/config.json, https://huggingface.co/tiiuae/falcon-rw-7b/raw/main/config.json, https://huggingface.co/tiiuae/falcon-7b/raw/main/config.json, https://huggingface.co/tiiuae/falcon-40b/raw/main/config.json, https://huggingface.co/tiiuae/falcon-11B/raw/main/config.json
- Tokenizer/generation metadata: https://huggingface.co/tiiuae/falcon-7b/raw/main/tokenizer_config.json, https://huggingface.co/tiiuae/falcon-40b/raw/main/tokenizer_config.json, https://huggingface.co/tiiuae/falcon-11B/raw/main/generation_config.json

Primary runtime target: `FalconForCausalLM` decoder-only text generation. Required body is `FalconModel`; sequence classification, token classification, and question answering heads are optional/deferred for first generation integration.

## 2. High-level architecture

Falcon is a text-only decoder-only causal LM. It has learned token embeddings, repeated decoder blocks, a final LayerNorm, and an LM head. Important variation is inside the decoder block: old RefinedWeb checkpoints can use serial attention then MLP with ALiBi, 7B uses parallel attention/MLP with MQA and RoPE, and newer/new-decoder checkpoints use grouped key/value heads plus parallel attention.

```text
tokenizer/input_ids + attention_mask
  -> word embeddings
  -> repeated Falcon decoder blocks
     -> fused query_key_value projection
     -> RoPE or ALiBi causal self-attention with KV cache
     -> attention output projection
     -> parallel or serial MLP
  -> final LayerNorm
  -> tied or untied lm_head
  -> logits/sampling
```

Generation stage split:

```text
CPU tokenizer -> prefill(input_ids, attention_mask, empty DynamicCache)
              -> per-layer K/V cache
              -> decode(new token(s), attention_mask, cache)
              -> logits_to_keep/lm_head
              -> sampler/controller
```

Independently stageable pieces:

- Config and weight loader, including packed `query_key_value` interpretation.
- One-block prefill parity without cache.
- KV-cache update/read parity.
- RoPE path and ALiBi path as separate attention variants.
- New-decoder GQA/MQA packing and old full-MHA ALiBi packing.
- Last-token-only LM head during decode.

## 3. Important config dimensions

Source defaults from `FalconConfig` at this commit: `vocab_size=65024`, `hidden_size=4544`, `num_hidden_layers=32`, `num_attention_heads=71`, `num_kv_heads=None` then normalized to `num_attention_heads`, `alibi=False`, `new_decoder_architecture=False`, `multi_query=True`, `parallel_attn=True`, `bias=False`, `max_position_embeddings=2048`, `activation="gelu"`, `ffn_hidden_size=None` then normalized to `4 * hidden_size`, `use_cache=True`, `bos_token_id=11`, `eos_token_id=11`, `pad_token_id=None`, and `tie_word_embeddings=True`.

| Field | Meaning for lowering |
| --- | --- |
| `hidden_size` | Decoder width `H`; must equal `num_attention_heads * head_dim`. |
| `num_hidden_layers` | Number of decoder blocks. |
| `num_attention_heads` | Query head count. |
| `num_kv_heads` | Source normalizes missing value to query heads, but attention overrides effective KV heads to 1 for old MQA mode. |
| `head_dim` | `hidden_size // num_attention_heads`; source raises if not divisible. |
| `ffn_hidden_size` | MLP expansion width; default `4H`, but Falcon-11B sets 16384 for `H=4096`. |
| `alibi` | If true, no RoPE in attention; runtime ALiBi bias is built from `attention_mask`. |
| `rope_parameters` / `rope_theta` | Current config standardization maps legacy `rope_theta` into `rope_parameters`; default RoPE uses theta and full `head_dim`. Falcon-11B uses `rope_theta=500042.0`. |
| `new_decoder_architecture` | Uses Falcon-40B-style packed GQA layout and always parallel attention semantics in config docs. |
| `multi_query` | Old decoder only: true means one K/V head; false means full MHA. Ignored by new decoder. |
| `parallel_attn` | Old decoder: parallel MLP and attention if true, serial if false. New decoder configs use parallel behavior. |
| `num_ln_in_parallel_attn` | New decoder norm split: `None` becomes 2 in source; Falcon-11B sets 1. |
| `bias` | All `FalconLinear` projections use this flag; old RW checkpoints use bias=true, later 7B/40B/11B use false. |
| `tie_word_embeddings` | Source LM head is tied to embeddings when config says true; Falcon-11B config sets false. |
| `cache` | HF `DynamicCache`; per-layer K/V tensors have shape `[B, effective_kv_heads, T, head_dim]`. |

Representative checkpoint sweep from official `config.json`:

| Model id | Layers | Hidden | Q heads | Effective KV heads | Head dim | MLP width | Pos enc | Block mode | Bias | Vocab | Max pos | dtype/config notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | ---: | ---: | --- |
| `tiiuae/falcon-rw-1b` | 24 | 2048 | 32 | 32 | 64 | 8192 | ALiBi | old serial, `multi_query=false`, `parallel_attn=false` | true | 50304 | source default 2048 | bos=1/eos=2; in-library config now `model_type=falcon`. |
| `tiiuae/falcon-rw-7b` | 36 | 4096 | 64 | 64 | 64 | 16384 | ALiBi | old serial, `multi_query=false`, `parallel_attn=false` | true | 65024 | source default 2048 | RW-style full-MHA ALiBi. |
| `tiiuae/falcon-7b` | 32 | 4544 | 71 | 1 | 64 | 18176 | RoPE default theta | old parallel, `multi_query=true` | false | 65024 | source default 2048 | Fused QKV width is `H + 2D = 4672`, not `3H`. |
| `tiiuae/falcon-40b` | 60 | 8192 | 128 | 8 stored / 128 expanded in `_split_heads` | 64 | 32768 | RoPE default theta | new decoder, `num_kv_heads=8`, parallel | false | 65024 | source default 2048 | Packed groups of 16 Q heads plus one K/V per group. |
| `tiiuae/falcon-11B` | 60 | 4096 | 32 | 8 stored / 32 expanded in `_split_heads` | 128 | 16384 | RoPE theta 500042 | new decoder, `num_ln_in_parallel_attn=1` | false | 65024 | 8192 | `tie_word_embeddings=false`, generation bos/eos 11. |

Falcon-180B repo metadata says it is a 180B causal decoder-only model with multiquery and BF16 tensors, but config access is gated/licensed in the inspected session. Treat it as a follow-up sweep item before claiming exact shapes.

## 3a. Family variation traps

- `query_key_value` is always one packed projection, but the row packing differs by architecture:
  - new decoder: `view(B,T,num_kv_heads, num_heads // num_kv_heads + 2, D)` gives grouped Q heads followed by one K and one V per KV group.
  - old full-MHA: `view(B,T,num_heads,3,D)` gives per-head `[q,k,v]` groups.
  - old MQA: `view(B,T,num_heads + 2,D)` gives all Q heads followed by one K and one V.
- Source `_split_heads` for `new_decoder_architecture` broadcasts K/V to query shape then flattens, but forward later reshapes K/V to `num_heads` for cache storage. This is more like expanded GQA in the source path than a compact KV-cache implementation.
- Config `num_kv_heads` defaults to `num_attention_heads`, but old MQA ignores it and uses effective KV heads 1.
- `alibi=True` disables RoPE. FlashAttention2 path raises if ALiBi is present.
- ALiBi and RoPE differ not only in position math but also attention math order: non-ALiBi eager path uses `scores / sqrt(D)` before `softmax(scores + mask)`, while ALiBi path computes `(scores + alibi) * inv_sqrt(D)` then adds mask.
- `parallel_attn` changes residual/norm schedule. Old RW checkpoints are serial; later 7B/40B/11B are parallel.
- `new_decoder_architecture` can use one or two LayerNorms before parallel attention/MLP. Falcon-40B source default becomes two norms; Falcon-11B explicitly uses one norm.
- Some old configs had `model_type="RefinedWebModel"`, `n_head`, `n_layer`, or remote-code `RWForCausalLM`; the converter maps these to current Falcon names. Current raw configs inspected are already mostly converted.
- Falcon-11B uses `rope_theta=500042.0`, `max_position_embeddings=8192`, and untied LM head; Falcon-7B/40B configs omit these and use source/config defaults.
- `bias` toggles all attention/MLP linears. Do not assume bias-free across RW checkpoints.
- The tokenizer has no pad token in model config by default; batched generation tests explicitly set `tokenizer.pad_token = tokenizer.eos_token` and use left padding.
- Text tensor layout is `[B,T,H]`; cache is `[B,KV,T,D]` in the source. There is no image NCHW/NHWC issue, but head-layout rewrites need no-layout-translation guards around view/transpose/reshape chains.

## 4. Operator coverage checklist

Tensor/layout ops:

- `Embedding(vocab_size, hidden_size)` from `input_ids[int64] -> [B,T,H]`.
- `arange`, `unsqueeze`, shape math for `position_ids`.
- `view`, `reshape`, `transpose`, `permute`, `flatten`, `broadcast_to` for QKV split/merge.
- Slicing for `logits_to_keep`: `hidden_states[:, slice_indices, :]`.
- Concatenate/update for dynamic KV cache growth.
- Mask creation via Transformers causal-mask utility; boolean-to-additive mask conversion for ALiBi cases.

Neural network primitives:

- LayerNorm over last dim `H`, affine, epsilon usually `1e-5`.
- Packed QKV `FalconLinear(H -> qkv_out_dim)`, optional bias.
- Attention output `FalconLinear(H -> H)`, optional bias.
- MLP up `FalconLinear(H -> ffn_hidden_size)`, optional bias.
- GELU activation from `get_activation(config.activation)`.
- MLP down `FalconLinear(ffn_hidden_size -> H)`, optional bias.
- Residual add and attention/hidden dropout. In eval and inspected configs, dropout probability is 0.0.
- Final LayerNorm.
- LM head `Linear(H -> vocab_size, bias=False)`, tied or untied according to config.

Attention primitives:

- Causal self-attention only.
- Old full-MHA ALiBi: Q/K/V `[B,heads,T,D]`; cache `[B,heads,T,D]`.
- Old MQA RoPE: Q `[B,heads,T,D]`, K/V stored `[B,1,T,D]` after split before source reshape; optimized runtime should keep compact KV cache.
- New decoder GQA/RoPE: packed Q groups per KV head; source expands K/V to heads, but optimized runtime should preserve compact `[B,num_kv_heads,T,D]` where parity allows.
- SDPA path for non-ALiBi or ALiBi after mask integration when `config._attn_implementation == "sdpa"` and `output_attentions=False`.
- FlashAttention2 path for RoPE only, with Q/K/V layout `[B,T,heads,D]` and no ALiBi.
- Eager fallback matmul, additive mask, softmax, value matmul.

Position/rotary/relative-bias ops:

- Default RoPE cos/sin generation from `rope_parameters["rope_theta"]` and `head_dim`.
- Optional advanced RoPE types via shared `ROPE_INIT_FUNCTIONS` if configs set non-default `rope_type`.
- ALiBi slope and position generation from `attention_mask.cumsum`.

Generation/cache ops:

- HF `DynamicCache(config)` equivalent.
- Per-layer `cache.update(key_layer, value_layer, layer_idx)`.
- `past_key_values.get_seq_length()` for position IDs and mask length.
- 2D `attention_mask [B,past+T]`, causal mask creation, and `logits_to_keep` support.
- `input_ids` xor `inputs_embeds` validation.

Tokenizer/preprocessing-coupled ops:

- `PreTrainedTokenizerFast`; official tokenizer metadata uses model inputs `input_ids` and `attention_mask`.
- `add_prefix_space=false`.
- Falcon-7B/40B tokenizer metadata has eos token `<|endoftext|>` and `model_max_length=2048`.
- Generation config for Falcon-11B sets bos/eos token id 11.

Optional heads:

- Sequence classification: dense score per token, gather rightmost non-pad token with `argmax` over non-pad mask.
- Token classification: dropout plus `Linear(H -> num_labels)`.
- Question answering: `Linear(H -> 2)`, split/squeeze start/end logits.

## 5. Layer/block breakdown

Top-level CausalLM:

```text
input_ids [B,T] -> word_embeddings [B,T,H]
if use_cache and no cache: cache = DynamicCache(config)
if position_ids absent: arange(T) + past_seen_tokens
if alibi: build ALiBi from attention_mask or all-ones mask
causal_mask = create_causal_mask(...)
position_embeddings = rotary_emb(hidden_states, position_ids)
for each decoder layer:
  hidden = FalconDecoderLayer(hidden, alibi, causal_mask, cache, position_embeddings)
hidden = final LayerNorm(hidden)
logits = lm_head(hidden[:, selected_positions, :])
```

Decoder layer, old serial (`new_decoder_architecture=false`, `parallel_attn=false`; RW 1B/7B):

```text
residual = hidden
attn_in = input_layernorm(hidden)
attn = self_attention(attn_in)
residual = dropout_add(attn, residual)
mlp_in = post_attention_layernorm(residual)
mlp = dense_4h_to_h(GELU(dense_h_to_4h(mlp_in)))
output = dropout_add(mlp, residual)
```

Decoder layer, old parallel (`new_decoder_architecture=false`, `parallel_attn=true`; Falcon-7B):

```text
residual = hidden
normed = input_layernorm(hidden)
attn = self_attention(normed)
mlp = dense_4h_to_h(GELU(dense_h_to_4h(normed)))
output = residual + dropout(attn + mlp)
```

Decoder layer, new decoder with two norms (`num_ln_in_parallel_attn=2`; Falcon-40B effective default):

```text
residual = hidden
attn_in = ln_attn(hidden)
mlp_in = ln_mlp(hidden)
attn = self_attention(attn_in)
mlp = dense_4h_to_h(GELU(dense_h_to_4h(mlp_in)))
output = residual + dropout(attn + mlp)
```

Decoder layer, new decoder with one norm (`num_ln_in_parallel_attn=1`; Falcon-11B):

```text
residual = hidden
normed = input_layernorm(hidden)
attn = self_attention(normed)
mlp = dense_4h_to_h(GELU(dense_h_to_4h(normed)))
output = residual + dropout(attn + mlp)
```

Projection examples:

- Falcon-7B: QKV `4544 -> 4672` because old MQA uses `H + 2D`; MLP `4544 -> 18176 -> 4544`; output projection `4544 -> 4544`.
- Falcon-40B: QKV `8192 -> 9216` because `(8 * 2 + 128) * 64`; MLP `8192 -> 32768 -> 8192`.
- Falcon-11B: QKV `4096 -> 6144` because `(8 * 2 + 32) * 128`; MLP `4096 -> 16384 -> 4096`.
- RW-7B: QKV `4096 -> 12288` because full MHA uses `3H`; MLP `4096 -> 16384 -> 4096`.

## 6. Attention requirements

Required variants:

| Variant | Checkpoints | Q heads | Stored/effective KV heads | Position | Optimized backend notes |
| --- | --- | ---: | ---: | --- | --- |
| Full MHA + ALiBi | `falcon-rw-1b`, `falcon-rw-7b` | 32/64 | 32/64 | ALiBi | SDPA is tested against eager for ALiBi; FA2 rejected. |
| Old MQA + RoPE | `falcon-7b` | 71 | 1 | RoPE | Compact KV cache is important; packed QKV is `Q_all,K,V`. |
| New decoder GQA + RoPE | `falcon-40b`, `falcon-11B` | 128/32 | config 8 but source expands in split | RoPE | Runtime should prefer compact GQA cache and repeat/broadcast in attention kernel. |

Common properties:

- Causal self-attention, no cross-attention, no sliding window, no local attention, no MoE.
- Masking uses `create_causal_mask`; ALiBi forces mask creation and may convert bool masks to additive minimum dtype.
- Cache update happens after RoPE for RoPE paths; cached keys are position-encoded.
- For ALiBi, cached keys are ordinary K tensors; ALiBi is recomputed from mask and not baked into cache.
- Eager non-ALiBi path:

```text
scores = Q @ K.transpose(-1,-2)
scores = scores / sqrt(D)
probs = softmax(scores + attention_mask, dim=-1, dtype=hidden_states.dtype)
context = probs @ V
```

- Eager ALiBi path:

```text
scores = Q @ K.transpose(-1,-2)
if fp16/bf16: scores = scores.float()
logits = (scores + alibi[B,heads,1,K]) * inv_sqrt(D)
probs = softmax(logits + attention_mask, dim=-1, dtype=hidden_states.dtype)
context = probs @ V
```

Cache shapes:

- Source forward before cache update uses `key_layer/value_layer [B,num_kv_heads,T,D]`, where `num_kv_heads = self.num_heads` for new decoder due to expansion, else effective old mode KV heads.
- Source `DynamicCache.update` returns full accumulated K/V in the same shape.
- DinoML optimized target should store old MQA as `[B,1,T,D]` and new GQA as `[B,num_kv_heads,T,D]`, then use repeat-KV inside attention. This differs from current source expansion for new decoder and needs explicit parity tests.
- Cache memory formula for compact target: `2 * layers * B * kv_heads * T * head_dim * dtype_size`. Source-expanded new decoder would multiply KV memory by query heads instead.

Backend compatibility:

- SDPA path is used inside `FalconAttention` for `config._attn_implementation == "sdpa"` and `output_attentions=False`.
- FlashAttention2 path changes layout to `[B,T,heads,D]`, casts accidental fp32 inputs back to projection dtype, and calls `_flash_attention_forward`; it raises on ALiBi.
- Fused attention parity must preserve whether dropout is applied. In inference dropout is zero, but source comments note non-ALiBi eager path does not apply attention dropout while ALiBi eager path does.

## 7. Position encoding and custom math

RoPE path:

```python
def falcon_rope_cos_sin(position_ids, head_dim, rope_theta, dtype):
    inv_freq = 1.0 / (rope_theta ** (arange(0, head_dim, 2).float() / head_dim))
    freqs = (inv_freq[None, :, None] @ position_ids[:, None, :].float()).transpose(1, 2)
    emb = cat([freqs, freqs], dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)

def apply_falcon_rope(q, k, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin
```

Source forces RoPE frequency math to float32 under autocast-disabled context, then casts cos/sin to hidden dtype. Cos/sin depend on runtime `position_ids`; static prefix tables are possible for fixed max positions and default RoPE, but dynamic RoPE update types need runtime logic.

ALiBi path:

```python
def falcon_alibi(attention_mask, num_heads, dtype):
    B, K = attention_mask.shape
    closest = 2 ** floor(log2(num_heads))
    base = 2 ** (-(2 ** -(log2(closest) - 3)))
    slopes = base ** arange(1, closest + 1)
    if closest != num_heads:
        extra_base = 2 ** (-(2 ** -(log2(2 * closest) - 3)))
        extra_count = min(closest, num_heads - closest)
        slopes = cat([slopes, extra_base ** arange(1, 1 + 2 * extra_count, 2)])
    positions = ((attention_mask.cumsum(-1) - 1) * attention_mask)[:, None, :]
    return (slopes[..., None].bfloat16() * positions).reshape(B * num_heads, 1, K).to(dtype)
```

ALiBi slopes are static for a head count, but positions depend on the runtime 2D mask. Left padding changes `cumsum`, so no-padding specializations need a guard.

## 8. Preprocessing and input packing

CPU/data pipeline:

- Official Falcon-7B/40B tokenizer metadata: `PreTrainedTokenizerFast`, `model_input_names=["input_ids","attention_mask"]`, `add_prefix_space=false`, eos token `<|endoftext|>`, `model_max_length=2048`.
- Config token IDs vary: RW checkpoints use bos=1/eos=2; 7B/40B/11B use bos=11/eos=11.
- `pad_token_id` is `None` in source defaults and inspected configs. For batched generation, HF tests set left padding and `pad_token = eos_token`.
- No token type IDs are part of the model forward contract.

GPU/runtime inputs:

- Exactly one of `input_ids [B,T]` or `inputs_embeds [B,T,H]`.
- Optional `attention_mask [B,past+T]`; if absent ALiBi path creates all ones for ALiBi generation.
- Optional `position_ids [B,T]`; if absent source uses `arange(T) + past_seen_tokens`.
- Optional `past_key_values`; source creates `DynamicCache(config)` when `use_cache=True`.

Generation-controller behavior:

- `GenerationMixin` supplies generic sampling/beam search; core graph only needs logits and cache.
- `logits_to_keep` can be int or tensor. For normal decode, use `1` to compute only last-token logits.
- No forced decoder IDs, timestamp processors, multimodal placeholder stitching, or packed varlen descriptors are present.

## 9. Graph rewrite / lowering opportunities

### Rewrite: Falcon packed QKV projection

Source pattern:

```text
fused_qkv = Linear(hidden, W_qkv)
q,k,v = _split_heads(fused_qkv)
```

Replacement:

```text
one GEMM for packed QKV -> metadata views/splits matching config-specific packing
```

Preconditions:

- `hidden_size == num_attention_heads * head_dim`.
- Packing mode is selected from `new_decoder_architecture` and `multi_query`.
- For new decoder, `num_attention_heads % num_kv_heads == 0`.

Shape equations:

- Old MQA: `qkv_out = H + 2D`; rows are `[Q_all, K, V]`.
- Old full-MHA: `qkv_out = 3H`; rows are `[q,k,v]` per head in `view(..., heads, 3, D)`.
- New decoder: `qkv_out = (num_attention_heads + 2 * num_kv_heads) * D`; rows are grouped as `[Q_group, K_group, V_group]`.

Failure cases:

- Treating every packed weight as `[Q_all,K_all,V_all]`.
- Expanding new decoder K/V in weight loading rather than attention lowering.

Parity test sketch: compare DinoML split Q/K/V tensors after projection against `_split_heads` for each of the three modes.

### Rewrite: compact GQA/MQA cache

Source pattern:

```text
new decoder _split_heads broadcasts K/V to query shape, then cache.update stores expanded K/V
old MQA stores one K/V head
```

Replacement:

```text
store compact K/V [B,kv_heads,T,D]; repeat/broadcast in attention kernel
```

Preconditions:

- Attention backend supports MQA/GQA repeat semantics exactly.
- Cache API records logical query heads and physical KV heads.
- Parity tests compare logits, not internal source cache layout, for new decoder.

Failure cases:

- Debug API or exported cache expected to match HF expanded new-decoder cache tensors exactly.
- Attention kernel cannot support `num_heads / num_kv_heads` grouping.

### Rewrite: RoPE + attention prefill/decode

Source pattern:

```text
Q,K = apply_rotary_pos_emb(Q,K,cos,sin)
K,V = cache.update(K,V)
SDPA/eager attention
```

Replacement:

```text
fused_rope_attention_prefill_or_decode(Q,K,V,position_ids,cache,mask)
```

Preconditions:

- `alibi=false`.
- RoPE type supported; start with default and Falcon-11B theta.
- No `output_attentions=True`.
- Attention dropout is zero/eval.

Failure cases:

- Dynamic/linear/yarn RoPE configs without matching kernel math.
- Tensor-valued `position_ids` with non-monotonic positions unless kernel supports gather from cos/sin table.

### Rewrite: ALiBi causal attention

Source pattern:

```text
alibi = build_alibi_tensor(attention_mask)
scores = QK^T
probs = softmax((scores + alibi) / sqrt(D) + mask)
context = probs V
```

Replacement:

```text
fused_causal_attention_alibi(Q,K,V,attention_mask,slopes)
```

Preconditions:

- `alibi=true`.
- Kernel reproduces mask-cumsum positions for left padding.
- Softmax accumulation/upcast behavior is within tolerance.

Failure cases:

- FlashAttention2 backend, because source explicitly rejects ALiBi there.
- Nonstandard additive masks not derivable from 2D attention mask.

### Rewrite: parallel attention/MLP residual fusion

Source pattern:

```text
output = residual + dropout(attention_output + mlp_output)
```

Replacement:

```text
fused add/add/residual epilogue, dropout identity in eval
```

Preconditions:

- `model.eval()` or dropout probability zero.
- `parallel_attn=true` or `new_decoder_architecture=true`.

Failure cases:

- Training mode with dropout randomness.
- Serial RW mode, where MLP consumes post-attention residual norm.

### Rewrite: last-token-only LM head

Source pattern:

```text
lm_head(hidden_states[:, slice_indices, :])
```

Replacement:

```text
slice to [B,1,H] before vocab GEMM for decode
```

Preconditions:

- Inference/generation only.
- `logits_to_keep=1` or equivalent last-position request.

Failure cases:

- Prompt logprob scoring over all positions.
- Tensor `logits_to_keep` requesting arbitrary indices.

## 10. Kernel fusion candidates

Highest priority:

- Packed QKV GEMM plus config-specific split metadata. Falcon is especially easy to get wrong here because packing differs across RW, 7B, and 40B/11B.
- RoPE MQA/GQA attention prefill and decode with compact KV cache. This is central for Falcon-7B/40B/11B generation throughput and memory.
- ALiBi full-MHA attention for RW checkpoints, including left-padding-aware positions.
- LayerNorm kernels for pre-attention/pre-MLP/final normalization.
- Last-token-only LM head for decode.

Medium priority:

- MLP activation fusion: `Linear -> GELU -> Linear`.
- Parallel attention/MLP residual-add fusion for 7B/40B/11B.
- Fused RoPE application into attention Q/K load path.
- GGUF/runtime-dequant GEMM path for large projection weights after dense parity, especially QKV and MLP.

Lower priority:

- FlashAttention2 layout path exactly as HF source, because DinoML can instead lower to its own fused attention once parity is proven.
- Sequence/token classification and QA heads.
- Training dropout/loss paths and gradient checkpointing.
- Source-expanded new-decoder cache compatibility mode if compact cache is accepted as the DinoML runtime contract.

## 11. Runtime staging plan

1. Parse `FalconConfig`, including legacy/standardized RoPE fields, `new_decoder_architecture`, `multi_query`, `parallel_attn`, `num_kv_heads`, `num_ln_in_parallel_attn`, `bias`, and `tie_word_embeddings`.
2. Load weights and validate QKV packing on tiny/random or small official-compatible checkpoints.
3. Implement one decoder block parity in fp32/eval without cache for each block schedule: old serial, old parallel, new one-norm/two-norm.
4. Add full prefill parity for `FalconForCausalLM` with eager attention: first Falcon-7B-style RoPE MQA, then Falcon-RW ALiBi, then new-decoder GQA.
5. Add DynamicCache decode parity with compact KV cache and a compatibility check against HF logits.
6. Add `logits_to_keep=1` decode path and tied/untied LM-head handling.
7. Replace eager attention with fused RoPE MQA/GQA and ALiBi kernels.
8. Add graph fusions for MLP, residual adds, and packed QKV split views.
9. Add quantized/GGUF weight-loading experiments once dense BF16/FP16 parity is stable.

Initial stubs: dropout as identity in eval; no training losses; no classification/QA/token heads; reject unsupported advanced RoPE types until a config requires them.

## 12. Parity and validation plan

- Config normalization tests: source defaults, Falcon-7B, Falcon-40B, Falcon-11B; assert head_dim, ffn width, effective KV heads, RoPE theta, and tie-word behavior.
- Packed QKV split tests for all three modes using random hidden states and loaded/random weights.
- RoPE unit tests for default theta and Falcon-11B `rope_theta=500042.0`, including decode position offset.
- ALiBi unit tests for head counts 32 and 64, unpadded and left-padded masks.
- Single-block parity for old serial ALiBi, old parallel MQA RoPE, new decoder one-norm, and new decoder two-norm.
- Prefill logits parity on short prompts for representative checkpoints or tiny-random analogs.
- Decode parity: prefill plus one-token and multi-token decode with cache; compare logits and cache growth.
- Batched left-padding parity following HF test behavior with `pad_token=eos_token`.
- LM-head parity for tied and untied embeddings, including `logits_to_keep=1` versus full logits slice.
- Recommended tolerances: fp32 `rtol=1e-4, atol=1e-4`; fp16/bf16 end-to-end `rtol=5e-3, atol=5e-3`, with looser checks only around fused attention if justified by backend math.

No DinoML tests were run for this docs-only audit.

## 13. Performance probes

- Prefill throughput sweep by variant: RW ALiBi full-MHA, Falcon-7B MQA RoPE, new-decoder GQA RoPE.
- Decode tokens/sec sweep with cache lengths `{128,512,2048,8192}` where config allows.
- KV cache memory comparison: HF source-expanded new decoder versus DinoML compact GQA cache.
- QKV GEMM profile for unusual output widths, especially Falcon-7B `4544 -> 4672` and Falcon-40B `8192 -> 9216`.
- Attention backend comparison: eager matmul/softmax, SDPA-compatible lowering, DinoML fused attention.
- ALiBi construction overhead with and without left padding.
- Last-token LM head cost versus full prompt logits for large vocab 65024.
- MLP GEMM and activation fusion probe for `H -> 4H -> H`.
- GGUF/runtime-dequant projection probe for QKV, MLP up/down, and LM head weights.

## 14. Skip/defer list

- Training, gradients, dropout randomness, labels/loss paths.
- Beam search specifics and advanced generation processors beyond standard causal LM sampling.
- Sequence classification, token classification, and question answering heads.
- Tokenizer execution inside DinoML runtime.
- FlashAttention2 exact wrapper path; prefer DinoML fused attention after parity.
- Advanced RoPE types not present in inspected configs, except preserving parse/reject diagnostics.
- Multi-GPU/tensor parallel execution and HF quantization wrappers.
- Falcon-180B exact integration until gated config/weights are inspected.
- Exporting HF-identical expanded new-decoder cache tensors, unless an API consumer needs that compatibility.

## 15. Final implementation checklist

- [ ] Parse `FalconConfig` and normalize source defaults/legacy RoPE fields.
- [ ] Load tokenizer/generation metadata for bos/eos/pad behavior.
- [ ] Implement tied and untied LM-head loading.
- [ ] Implement Falcon LayerNorm placements for old serial, old parallel, and new decoder blocks.
- [ ] Implement `FalconLinear` as bias-optional GEMM.
- [ ] Implement packed QKV split for old full-MHA, old MQA, and new-decoder grouped packing.
- [ ] Implement default RoPE and Falcon-11B theta handling.
- [ ] Implement ALiBi slopes and mask-cumsum positions.
- [ ] Implement eager attention parity for RoPE and ALiBi paths.
- [ ] Implement compact MQA/GQA KV cache and cache update/read.
- [ ] Add one-block parity tests for each architecture schedule.
- [ ] Add prefill and decode parity tests with left padding.
- [ ] Add `logits_to_keep=1` and LM-head parity tests.
- [ ] Add fused RoPE MQA/GQA attention lowering.
- [ ] Add fused ALiBi attention lowering.
- [ ] Add packed QKV, residual-add, MLP, and last-token LM-head rewrites.
- [ ] Benchmark prefill, decode, cache memory, ALiBi construction, QKV GEMM, MLP GEMM, and LM-head slicing.
