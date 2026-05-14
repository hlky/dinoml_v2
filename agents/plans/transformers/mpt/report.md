# MPT Transformers Family Audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model family: mpt
Primary runtime target: MptForCausalLM causal LM prefill, decode, and generation
Dinoml assumptions: inference-only first, CUDA GPU target, preserve Transformers tensor axes, prefer explicit ALiBi/attention/cache rewrites before optimized kernels.
```

Source files inspected:

- Local: `X:/H/transformers/src/transformers/models/mpt/configuration_mpt.py`
- Local: `X:/H/transformers/src/transformers/models/mpt/modeling_mpt.py`
- Local: `X:/H/transformers/src/transformers/models/mpt/__init__.py`
- Local shared utilities: `X:/H/transformers/src/transformers/cache_utils.py`, `X:/H/transformers/src/transformers/masking_utils.py`
- Auto mappings: `X:/H/transformers/src/transformers/models/auto/modeling_auto.py`, `X:/H/transformers/src/transformers/models/auto/tokenization_auto.py`
- Upstream source URL pattern at the pinned commit: `https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/mpt/...`

Representative configs inspected:

- Open mirror `gretelai/mpt-7b`: `https://huggingface.co/gretelai/mpt-7b/raw/main/config.json`
- Open mirror `gl198976/mpt-7b-instruct`: `https://huggingface.co/gl198976/mpt-7b-instruct/raw/main/config.json`
- Open mirror `TehVenom/MPT-7b-storywriter-Apache-2.0`: `https://huggingface.co/TehVenom/MPT-7b-storywriter-Apache-2.0/raw/main/config.json`
- Conversion/mirror `michaelfeil/ct2fast-mpt-30b`: `https://huggingface.co/michaelfeil/ct2fast-mpt-30b/raw/main/config.json`
- Matching `tokenizer_config.json`, `special_tokens_map.json`, and `generation_config.json` where present in those repos.

Missing files or assumptions:

- The canonical `mosaicml/mpt-*` repos returned 401 Unauthorized during this audit. The sweep therefore uses open mirrors/conversions and labels them as such.
- The official checkpoints historically used remote-code files with class names like `MPTConfig` and `MPTForCausalLM`. The pinned Transformers source has native `MptConfig` and `MptForCausalLM` classes plus auto mappings for `model_type="mpt"`.
- Config fields advertise several Mosaic/remote-code features that this native modeling file does not implement: learned positional embeddings, MQA, q/k LayerNorm, prefix-LM masks, sequence-id masks, attention backend selection from `attn_config.attn_impl`, `no_bias=False`, `expansion_ratio != 4`, and `logit_scale`.
- No DinoML runtime code was edited and no DinoML tests were run for this docs-only audit.

## 2. High-level architecture

MPT is a text-only decoder-only causal LM. The inspected native HF implementation is a pre-LayerNorm transformer with token embeddings, ALiBi positional bias, full multi-head self-attention, a non-gated GELU MLP, final LayerNorm, and a tied bias-free LM projection.

```text
GPT-NeoX-style BPE tokenizer -> input_ids/attention_mask
  -> token embedding
  -> repeated decoder blocks with ALiBi causal self-attention
  -> final LayerNorm
  -> tied LM head logits
  -> generation controller / sampling
```

Generation stage split:

```text
CPU tokenizer -> prefill(input_ids, attention_mask, empty cache) -> per-layer K/V cache
              -> decode(new token chunk, grown attention_mask, K/V cache)
              -> logits_to_keep/lm_head -> sampler/controller
```

Independently stageable pieces:

- CPU/data pipeline: GPT-NeoX tokenizer files, special tokens, prompt formatting for instruct/chat variants.
- GPU prefill: embeddings, ALiBi materialization, causal/padding mask, all decoder blocks, optional cache population.
- GPU decode: append K/V to per-layer cache, slice static ALiBi bias to current key length, attention over prior cache plus new tokens, last-token logits.
- Optimization boundary: ALiBi construction and fused attention can be validated separately from tokenizer and generation sampling.

## 3. Important config dimensions

Source defaults from `MptConfig` at this commit:

| Field | Default | Lowering effect |
| --- | ---: | --- |
| `d_model` / `hidden_size` | 2048 | Hidden width `H`. |
| `n_heads` / `num_attention_heads` | 16 | MHA head count. |
| `head_dim` | `d_model // n_heads` | Source computes by integer division and assumes exact reshape. |
| `n_layers` / `num_hidden_layers` | 24 | Decoder block count. |
| `expansion_ratio` | 4 | Config field exists, but source MLP hardcodes `4 * hidden_size`. |
| `max_seq_len` | 2048 | ALiBi tensor length; no learned position table is instantiated. |
| `vocab_size` | 50368 | Token embedding and LM head rows. |
| `layer_norm_epsilon` | `1e-5` | All LayerNorms. |
| `attn_config.attn_type` | `multihead_attention` | Native source still implements full MHA only. |
| `attn_config.attn_impl` | `torch` | Native source does eager matmul/softmax attention; this field is not a kernel dispatch. |
| `attn_config.alibi` | `true` | Native source always builds ALiBi, regardless of this field. |
| `attn_config.alibi_bias_max` | 8 | Slope scale in ALiBi helper, but native forward does not pass config value and therefore uses helper default 8. |
| `attn_config.clip_qkv` | `null` | Optional clamp after packed QKV projection. |
| `attn_config.softmax_scale` | `null` | Defaults to `1 / sqrt(head_dim)` when null. |
| `use_cache` | `false` | Config default disables cache unless caller/generation enables it. |
| `tie_word_embeddings` | `true` | LM head tied to token embedding via `_tied_weights_keys`. |
| `tokenizer` | auto mapping | Auto tokenizer for native `mpt` maps to `GPTNeoXTokenizer` when tokenizers is available. |

Representative checkpoint sweep:

| Model id | Provenance | Layers | Hidden | Heads | Head dim | MLP width in source | Max seq | Vocab | Config/operator notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `gretelai/mpt-7b` | open mirror of MPT-7B config | 32 | 4096 | 32 | 128 | 16384 | 2048 | 50432 | `bf16`, ALiBi true, `attn_impl=torch`, `use_cache=false`, old `auto_map` remote-code entries. |
| `gl198976/mpt-7b-instruct` | open mirror | 32 | 4096 | 32 | 128 | 16384 | 2048 | 50432 | Same operator dimensions as 7B; instruct behavior is prompt/tokenizer/controller level. |
| `TehVenom/MPT-7b-storywriter-Apache-2.0` | open storywriter mirror | 32 | 4096 | 32 | 128 | 16384 | 65536 | 50432 | Long-context ALiBi; config has `alibi_bias_max=16` and `clip_qkv=6`, but native source ignores configured bias max in forward. |
| `michaelfeil/ct2fast-mpt-30b` | conversion/mirror | 48 | 7168 | 64 | 112 | 28672 | 8192 | 50432 | 30B-scale dense projections/cache; same native ops if loaded as `model_type=mpt`. |
| `MptConfig()` | source default/debug | 24 | 2048 | 16 | 128 | 8192 | 2048 | 50368 | Useful synthetic smoke shape; not a real checkpoint. |

Tokenizer/generation metadata from inspected open repos:

| Model id | Tokenizer class | `model_max_length` | Special tokens | Generation metadata |
| --- | --- | ---: | --- | --- |
| 7B mirrors | `GPTNeoXTokenizer` | 2048 | `<\|endoftext\|>` for BOS/EOS/UNK | generation config usually `_from_model_config`, `use_cache=false`; one mirror sets `eos_token_id=0`. |
| Storywriter mirror | `GPTNeoXTokenizer` | 65536 | same | `use_cache=false` in generation config. |
| 30B conversion | `GPTNeoXTokenizer` | 8192 | same | `use_cache=false`. |

## 3a. Family variation traps

- The native source has no learned position embedding module even though `learned_pos_emb=True` appears in many configs. Position handling is ALiBi-only in the inspected implementation.
- `attn_config.alibi=False` is not honored by native `MptModel.forward`; it always calls `build_mpt_alibi_tensor`. Likewise, `alibi_bias_max` from config is not forwarded, so source uses default 8 even for configs that specify 16.
- `attn_config.attn_impl` values such as `torch`, `flash`, or `triton` are historical remote-code controls. Native `MptAttention.forward` always uses explicit `torch.matmul`, mask fill, fp32 softmax, and matmul with V.
- `attn_type="multiquery_attention"` is accepted by the strict config type, but native `Wqkv` is always `H -> 3H` and reshapes K/V to `n_heads`, so MQA configs are out-of-scope for this source.
- `prefix_lm=True` and `attn_uses_sequence_id=True` are described in config docs, but native `MptModel.forward` has no `prefix_mask`, `sequence_id`, or `token_type_ids` handling. Shared `create_causal_mask` can handle packed masks/block overlays in other models, but MPT does not pass those inputs.
- `qk_ln=True` is not implemented; there are no Q/K LayerNorm modules.
- `no_bias` is effectively always true for transformer projections in native source: QKV, output, MLP up/down, and LM head are bias-free. LayerNorm bias is explicitly set to `None` for Hub compatibility.
- Source MLP hardcodes `4 * hidden_size`; do not trust `expansion_ratio` as an operator-changing field unless a future source revision uses it.
- `logit_scale`, `embedding_fraction`, `resid_pdrop`, and `emb_pdrop` are config/backward-compatibility fields not used in native inference forward. `attn_pdrop` is used for attention and MLP dropout, but eval inference should treat dropout as identity.
- Historical configs include `auto_map` to repo-local `configuration_mpt.MPTConfig` and `modeling_mpt.MPTForCausalLM`. Loading with `trust_remote_code=True` can execute old Mosaic/LLM-Foundry code with broader behavior than this native audit.
- Long-context Storywriter configs are a separate memory/performance regime: ALiBi tensor shape is `[heads, 1, max_seq_len]`, and KV cache memory scales with 65K context.
- Text tensors are `[B, T, H]`; cache tensors are `[B, heads, T, head_dim]`. There is no NCHW/NHWC concern, but head reshapes, `chunk(3, dim=2)`, `transpose(1, 2)`, softmax `dim=-1`, and LM logits axes should be protected from generic layout translation.

## 4. Operator coverage checklist

Tensor/layout ops:

- `input_ids[int64] [B, T]` or mutually exclusive `inputs_embeds [B, T, H]`.
- Token embedding lookup: `Embedding(vocab_size, H) -> [B, T, H]`.
- Shape/view ops: `reshape`, `transpose(1,2)`, `permute(0,2,1,3)`, `contiguous`, `view`, `chunk(3, dim=2)`, slice/gather for kept logits.
- Boolean causal/padding mask from `create_causal_mask`, converted to bool and consumed by `masked_fill`.
- ALiBi creation: `arange`, `ceil(log2(num_heads))`, `pow`, optional `concat`, broadcast multiply, and final squeeze.
- Optional QKV clamp: `clamp(min=-clip_qkv, max=clip_qkv)`.
- Dynamic cache append/concat or equivalent in-place cache write.

Neural network primitives:

- LayerNorm over last dim `H`, epsilon `1e-5`, weight only in inspected source because bias is set to `None`.
- Packed QKV Linear `H -> 3H`, bias=False. For MPT-7B: `4096 -> 12288`; for 30B: `7168 -> 21504`.
- Attention output Linear `H -> H`, bias=False.
- MLP up Linear `H -> 4H`, bias=False.
- Exact GELU, `nn.GELU(approximate="none")`.
- MLP down Linear `4H -> H`, bias=False.
- Residual add after attention and after MLP; dropout is identity in eval.
- Final LM head Linear `H -> vocab_size`, bias=False, tied to token embeddings when `tie_word_embeddings=True`.

Attention primitives:

- Causal self-attention MHA only.
- Scores: `Q [B,A,Q,D] @ K.transpose(-1,-2) [B,A,D,K] * softmax_scale`.
- Add ALiBi bias with source shape `[A, 1, max_seq_len]`, sliced on key length and broadcast over batch/query.
- Apply boolean mask via `masked_fill(mask, finfo(dtype).min)`.
- Softmax over keys in fp32, cast probabilities to value dtype, dropout, then `P @ V`.

Generation/cache ops:

- HF `DynamicCache` equivalent with per-layer K/V shape `[B, A, T, D]`.
- Cache update takes new K/V `[B,A,T_new,D]`, appends along sequence axis `-2`, and returns full K/V.
- `past_key_values.get_seq_length()` is used for mask construction and ALiBi slicing behavior.
- `logits_to_keep` can be int tail slice or tensor index selection before the LM head.

Preprocessing-coupled ops:

- GPT-NeoX tokenizer metadata: `GPTNeoXTokenizer`, `add_prefix_space=false`, `<|endoftext|>` as BOS/EOS/UNK.
- `attention_mask [B, K]` is a keep-mask input to shared mask utilities; no token type IDs or position IDs enter native MPT.

Packed/varlen sequence metadata:

- Native MPT source does not expose prefix masks, sequence-id masks, `cu_seqlens`, or varlen descriptors.
- If remote-code MPT is admitted later, `prefix_lm` and `attn_uses_sequence_id` need a separate audit against that remote implementation.

## 5. Layer/block breakdown

For hidden width `H`, heads `A`, head dim `D=H/A`, sequence chunk `T`, and total key length `K`:

```text
input_ids [B,T]
hidden = wte(input_ids)                         # [B,T,H]
alibi = build_mpt_alibi_tensor(A, max_seq_len)  # [A,1,max_seq_len]
mask = create_causal_mask(...).to(bool)         # backend-shaped bool mask
```

Decoder block, repeated `n_layers` times:

```text
residual = hidden
x = LayerNorm(hidden, eps=layer_norm_epsilon)   # weight-only LN in native source
qkv = Linear_qkv(x)                             # [B,T,3H], bias=False
qkv = clamp(qkv, -clip_qkv, clip_qkv) optional
q, k_new, v_new = chunk(qkv, 3, dim=2)           # each [B,T,H]
q = reshape(q, [B,T,A,D]).transpose(1, 2)        # [B,A,T,D]
k_new = reshape(k_new, [B,T,A,D]).transpose(1, 2)
v_new = reshape(v_new, [B,T,A,D]).transpose(1, 2)
k, v = cache.update(k_new, v_new, layer_idx) optional
scores = (q @ k.transpose(-1, -2)) * scale       # [B,A,T,K]
scores = scores + alibi[:, :, -K:]              # broadcast [A,1,K]
scores = masked_fill(scores, mask, finfo.min)
p = softmax(scores.float(), dim=-1).to(v.dtype)
context = p @ v                                  # [B,A,T,D]
context = context.permute(0,2,1,3).contiguous().view(B,T,H)
hidden = residual + dropout(out_proj(context))  # out_proj H -> H, bias=False

residual = hidden
x = LayerNorm(hidden, eps=layer_norm_epsilon)
x = Linear_up(x)                                # H -> 4H, bias=False
x = GELU_exact(x)
x = Linear_down(x)                              # 4H -> H, bias=False
hidden = residual + dropout(x)
```

Final causal LM head:

```text
hidden = final LayerNorm(hidden)
selected = hidden[:, slice_indices, :]           # all positions if logits_to_keep=0
logits = lm_head(selected)                       # [B,T_keep,vocab], bias=False
```

For a 7B config, each block has QKV `4096 -> 12288`, output `4096 -> 4096`, MLP up `4096 -> 16384`, and MLP down `16384 -> 4096`. For a 30B config, those become `7168 -> 21504`, `7168 -> 7168`, `7168 -> 28672`, and `28672 -> 7168`.

## 6. Attention requirements

Required variant:

- Causal self-attention, no cross-attention.
- Full MHA: query heads = key heads = value heads = `n_heads`; no native GQA/MQA.
- Head dim: `d_model // n_heads`; representative dims are 128 for 7B and 112 for 30B.
- Prefill shapes: Q/K/V new `[B,A,S,D]`; scores `[B,A,S,S]`; cache after update `[B,A,S,D]` per K/V per layer.
- Decode shapes: Q new `[B,A,T,D]`; K/V cache after append `[B,A,P+T,D]`; scores `[B,A,T,P+T]`.
- Cache stores projected K/V before any positional transform; ALiBi is not baked into cached keys.
- ALiBi shape in native source is `[A, 1, max_seq_len]`, not `[A, max_seq_len, max_seq_len]`. It relies on softmax translation invariance and only adds a key-position-dependent bias.
- Masking uses shared `create_causal_mask`, then native MPT converts the result to bool and calls `masked_fill(..., finfo(dtype).min)`.
- Attention math order to preserve: packed QKV projection; optional clamp; reshape/transpose; cache append; QK matmul scaled by `softmax_scale`; add ALiBi; apply mask; fp32 softmax; cast to V dtype; dropout; PV matmul; output projection.
- Source optimized backend dispatch: none in `MptAttention`. Any FlashAttention/SDPA implementation in DinoML is an optimization rewrite, not a direct mirror of a source backend call.

Packed/varlen, prefix LM, and sequence-id masks:

- Native source has no `prefix_mask`, `sequence_id`, or token-type input path. Config docs mention those features, but they are remote-code/config compatibility traps for this audit.
- Shared `masking_utils` contains packed/block mask helpers, but MPT does not pass the required arguments. Do not advertise packed sequence support for native MPT without a source change or a separate remote-code target.

Likely eager fallback cost:

- Source eager attention materializes `[B,A,Q,K]` scores and probabilities. For long Storywriter context this is too expensive for prefill without fused/streaming attention, but it is the parity reference.

## 7. Position encoding and custom math

MPT native source uses ALiBi only. There are no learned position embeddings, no RoPE, and no relative position tables in `MptModel`.

Short source-equivalent ALiBi:

```python
def mpt_alibi(num_heads, max_seq_len, alibi_bias_max=8):
    pos = arange(1 - max_seq_len, 1).view(1, 1, 1, max_seq_len)
    pow2 = 2 ** ceil(log2(num_heads))
    base = arange(1, pow2 + 1).float() * (alibi_bias_max / pow2)
    slopes = (1.0 / (2 ** base)).view(1, pow2, 1, 1)
    if pow2 != num_heads:
        slopes = cat([slopes[:, 1::2], slopes[:, ::2]], dim=1)[:, :num_heads]
    return (pos * slopes).squeeze(0)  # [heads, 1, max_seq_len]
```

Runtime slicing inside attention:

```python
key_length = key_states.shape[-2]
bias = alibi[:, :, -key_length:]
scores = scores + bias
```

Precompute opportunity:

- For a fixed `max_seq_len`, `num_heads`, and effective `alibi_bias_max`, the full `[A,1,max_seq_len]` ALiBi tensor can be precomputed or generated once per module/session.
- The source creates it inside every model forward, so an optimized runtime should cache it explicitly while preserving dtype/device behavior.

Dynamic inputs:

- ALiBi does not depend on `attention_mask`, unlike BLOOM. Padding affects only the boolean causal/padding mask.
- Decode key length comes from cache growth, so the ALiBi slice depends on current cache length.

Custom math:

- MLP activation is exact GELU (`approximate="none"`), not GPT-2/BLOOM tanh GELU and not SwiGLU.

## 8. Preprocessing and input packing

CPU/data pipeline:

- Inspected open MPT repos point to `GPTNeoXTokenizer` and `tokenizer_name="EleutherAI/gpt-neox-20b"`.
- Tokenizer metadata uses `add_prefix_space=false`.
- Special tokens map BOS/EOS/UNK to `<|endoftext|>`; one inspected generation config explicitly gives `eos_token_id=0`.
- Instruct/chat behavior is prompt-template/controller work; it does not change the native MPT module graph.

GPU/runtime inputs:

- `input_ids [B,T] int64` or `inputs_embeds [B,T,H]`; passing both is rejected.
- Optional `attention_mask [B,K]`, normally `K = past_length + T`.
- Optional `past_key_values`; if `use_cache=True` and absent, source creates `DynamicCache(config)`.
- No `position_ids`, `token_type_ids`, `prefix_mask`, `sequence_id`, image/audio placeholders, or multimodal stitch tensors in native MPT.

Generation-controller behavior:

- Config and generation metadata commonly set `use_cache=false`, but generation callers may pass `use_cache=True`. DinoML should treat cache as an admitted runtime mode, not as guaranteed by config defaults.
- `logits_to_keep` controls LM-head work. `0` keeps all positions; positive int keeps the trailing positions; tensor input indexes arbitrary sequence positions.
- Beam search cache reordering comes from generic `GenerationMixin`/cache behavior, not model-specific MPT code, and can be deferred for greedy/sampling parity.

## 9. Graph rewrite / lowering opportunities

### Rewrite: packed QKV projection

Source pattern:

```text
qkv = Linear(H -> 3H, bias=False)(x)
q, k, v = qkv.chunk(3, dim=2)
q/k/v = reshape([B,T,A,D]).transpose(1,2)
```

Replacement:

```text
single GEMM_RRR/RCR -> logical split into Q/K/V views -> attention layout [B,A,T,D]
```

Preconditions:

- Native MPT full-MHA config: `attn_type == "multihead_attention"` or no attempt to load MQA weights.
- `hidden_size == n_heads * head_dim`.
- Weight rows are all-Q, all-K, all-V blocks because source chunks the last projection dimension into three equal blocks.
- Optional QKV clamp is either absent or represented between projection and split.

Shape equations:

- Input `[B,T,H]`, packed output `[B,T,3H]`, each Q/K/V `[B,A,T,D]`.

Failure cases:

- Historical remote-code MQA or alternative packed layouts.
- Loader transposes or splits packed weights without recording the split order.

Parity test sketch:

- Compare projected Q/K/V tensors after reshape/transpose against HF for random inputs and both `clip_qkv=None` and `clip_qkv=6`.

### Rewrite: ALiBi eager attention to fused causal attention

Source pattern:

```text
scores = (Q @ K^T) * scale
scores += alibi[:, :, -K:]
scores = masked_fill(scores, bool_mask, finfo.min)
P = softmax(scores.float(), dim=-1).to(V.dtype)
out = P @ V
```

Replacement:

```text
fused_mha_alibi_prefill(Q,K,V,mask,slopes_or_bias)
fused_mha_alibi_decode(Q,K_cache,V_cache,mask,slopes_or_bias)
```

Preconditions:

- Full MHA; no MQA/GQA.
- ALiBi bias generated with native MPT slope ordering and effective bias max.
- Fused kernel supports key-position ALiBi and standard causal/padding masks.
- Softmax accumulation/upcast behavior is within admitted tolerance.

Failure cases:

- Remote-code prefix-LM or sequence-id masks.
- Long context exceeding precomputed ALiBi/cache capacity.
- Source revision begins honoring `alibi_bias_max` or `alibi=False`; config parsing must catch that.

Parity test sketch:

- Unit-test ALiBi tensor for head counts 16, 32, 64 and lengths 1, 2048, 8192; then compare single-block attention output for padding/no-padding and decode cache lengths.

### Rewrite: dynamic cache concat to session-owned cache

Source pattern:

```text
key_states, value_states = past_key_values.update(k_new, v_new, layer_idx)
```

Replacement:

```text
write k_new/v_new into preallocated [layer, 2, B, A, max_T, D] cache at offset P
return views covering [0:P+T]
```

Preconditions:

- Max cache length admitted at compile/session creation.
- Cache layout is artifact-visible.
- Batch size and beam reorder policy are explicit.

Failure cases:

- Generic HF offloaded/sliding/hybrid cache classes; native MPT does not need sliding cache but `DynamicCache(config)` can select layer classes based on future config fields.

Parity test sketch:

- Prefill, then N single-token decode steps; compare per-layer K/V tensors and logits to HF.

### Rewrite: last-token-only LM head

Source pattern:

```text
logits = lm_head(hidden_states[:, slice_indices, :])
```

Replacement:

```text
slice/gather kept hidden positions before vocab GEMM
```

Preconditions:

- No labels/loss path.
- Generation only requires last token or known kept positions.
- Tied embedding/LM head alias is preserved as one logical parameter.

Failure cases:

- Prompt scoring/full-sequence logprobs.
- Tensor `logits_to_keep` with arbitrary positions unless gather is implemented.

Parity test sketch:

- Compare `logits_to_keep=0`, `1`, `8`, and tensor-index variants against HF output shapes/values.

## 10. Kernel fusion candidates

Highest priority:

- Bias-free GEMM coverage for MPT projections, including packed QKV and large MLP up/down matrices.
- LayerNorm over hidden dim with weight-only affine. MPT has two LayerNorms per block plus final norm.
- ALiBi causal attention prefill and decode with `[B,A,T,D]` cache, fp32 softmax, and key-position bias.
- Last-token-only tied LM head for decode.

Medium priority:

- Packed QKV projection plus split/layout metadata to avoid materializing three separate tensors unnecessarily.
- Exact GELU MLP fusion around `up_proj -> GELU -> down_proj`; at minimum fuse GELU with the intermediate activation write where possible.
- QKV clamp fusion for Storywriter-style `clip_qkv=6`.
- Precomputed/cached ALiBi tensor generation for long contexts.
- Residual-add/dropout-elision fusion in eval mode.

Lower priority:

- Classification, token classification, and QA heads.
- Beam cache reorder and advanced generation processors.
- Remote-code-only prefix/sequence-id masks and MQA until a separate source target is admitted.
- Tokenizer execution inside the GPU runtime.

## 11. Runtime staging plan

1. Parse native `MptConfig` and reject or clearly flag source-unsupported config combinations: MQA, `qk_ln`, prefix LM, sequence-id masks, `alibi=False`, non-default `alibi_bias_max` if exact native source parity is required, and non-4 expansion ratio.
2. Load dense weights with tied `transformer.wte.weight` / `lm_head.weight` alias preservation and bias-free projection contracts.
3. Implement primitive parity for LayerNorm weight-only affine, exact GELU, packed QKV projection, optional QKV clamp, and MPT ALiBi.
4. Run one-block eager parity with no cache, all-ones mask, fp32 first.
5. Add full prefill parity for `MptForCausalLM`, including `create_causal_mask` semantics and final LM head.
6. Add dynamic-cache decode parity with per-layer `[B,A,T,D]` K/V buffers and last-token logits.
7. Replace eager attention with fused ALiBi attention under strict mask/cache guards.
8. Add long-context probes for Storywriter-like `max_seq_len=65536` only after dense 2K/8K parity is stable.
9. Add GGUF/quantized weight loading after dense parity; MPT projection shapes are good candidates for DinoML's runtime-dequant GEMM work.

Initial stubs/deferred behavior:

- Dropout as identity in eval.
- No training loss, no gradient checkpointing.
- No remote-code prefix-LM, sequence-id masks, MQA, or q/k LayerNorm.
- No classification/QA/token-classification heads for the causal LM target.

## 12. Parity and validation plan

- Config admission tests: verify native-supported configs load and unsupported remote-code features fail clearly.
- ALiBi unit tests: heads 16/32/64, lengths 1/128/2048/8192, non-power-of-two head count synthetic case, and Storywriter bias-max decision documented.
- QKV primitive tests: random `[B,T,H]`, compare packed output, Q/K/V splits, reshape/transpose, and optional clamp against HF.
- LayerNorm tests: weight-only affine and `bias=None` handling.
- Exact GELU tests: compare `nn.GELU(approximate="none")` for fp32/fp16/bf16.
- Single-block fp32 parity: random hidden states and attention mask, no cache.
- Prefill LM parity: short prompts for a 7B-shape or smaller synthetic config; compare hidden states and logits.
- Decode parity: prefill then one-token and chunked decode with `use_cache=True`; compare cache shapes and logits.
- Mask parity: no padding, right/left padding where tokenizer/controller emits it, and decode masks with `past_length + T`.
- `logits_to_keep` parity: all logits, last token, trailing N, and tensor-index forms.

Recommended tolerances:

- fp32 eager primitives/blocks: `rtol=1e-4`, `atol=1e-5`.
- fp16/bf16 fused attention/logits: start `rtol=5e-3`, `atol=5e-3`, with stricter primitive tests where deterministic.

## 13. Performance probes

- Prefill throughput sweep: `B={1,4,8}`, `S={128,512,2048,8192}`; add `S=65536` only for long-context stress after kernel memory behavior is known.
- Decode tokens/sec sweep: cache lengths `{128,512,2048,8192}` and batch sizes `{1,4,16}`.
- KV cache memory: `layers * 2 * B * heads * max_T * head_dim * dtype_size`. For 7B bf16 at `B=1,T=2048`: about `32*2*1*32*2048*128*2 = 1.07 GB`; for Storywriter 65K it is about 32x larger.
- ALiBi overhead: per-forward tensor construction versus cached/session-owned ALiBi.
- Attention backend comparison: eager matmul/softmax baseline versus fused ALiBi prefill/decode.
- GEMM probes: QKV, attention output, MLP up/down, and LM head separately for 7B and 30B shapes.
- QKV clamp overhead for Storywriter configs.
- Last-token LM head probe: full `[B,T,V]` logits versus kept-position logits.
- Weight residency/dequant probes for large dense projections once GGUF/encoded constants are admitted.

All probes are proposed; this report includes no benchmark measurements.

## 14. Skip/defer list

- Training, losses, dropout randomness, and gradient checkpointing.
- Remote-code-only `prefix_lm`, `attn_uses_sequence_id`, q/k LayerNorm, MQA, and Triton/Flash config-specific attention implementations.
- Classification, token classification, and question-answering heads.
- Beam search cache reorder and advanced generation processors for first greedy/sampling parity.
- Tokenizer implementation inside DinoML runtime.
- Multi-GPU tensor parallelism and distributed checkpoint sharding.
- Quantized weights and quantized KV cache until dense parity is established.
- Native support for `alibi=False`, learned position embeddings, or non-default `alibi_bias_max` unless a future source revision implements them.

## 15. Final implementation checklist

- [ ] Parse `MptConfig` and `MptAttentionConfig`.
- [ ] Add config admission checks for native-source-supported MPT only.
- [ ] Load GPT-NeoX tokenizer metadata or define CPU tokenizer boundary.
- [ ] Preserve tied `wte` / `lm_head` weight aliasing.
- [ ] Load bias-free QKV, output, MLP up/down, and LM head weights.
- [ ] Implement weight-only LayerNorm with `bias=None`.
- [ ] Implement exact GELU.
- [ ] Implement packed QKV projection and split order `[Q_all, K_all, V_all]`.
- [ ] Implement optional QKV clamp.
- [ ] Implement MPT ALiBi generation and cache/slicing policy.
- [ ] Implement eager causal attention with ALiBi, bool mask fill, fp32 softmax, and `[B,A,T,D]` cache.
- [ ] Implement dynamic decode cache append with artifact-visible cache layout.
- [ ] Implement `logits_to_keep` slicing/gather before LM head.
- [ ] Add one-block, prefill, decode, ALiBi, QKV, mask, and logits-slicing parity tests.
- [ ] Add fused ALiBi prefill/decode attention rewrite under strict guards.
- [ ] Add performance probes for ALiBi materialization, GEMMs, prefill, decode, KV memory, QKV clamp, and LM head slicing.
