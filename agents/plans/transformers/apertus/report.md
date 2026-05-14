# Apertus Transformers Family Audit

## 1. Source basis

```text
Transformers commit/version:
  X:/H/transformers @ b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id:
  swiss-ai/Apertus-8B-2509
  swiss-ai/Apertus-8B-Instruct-2509
  swiss-ai/Apertus-70B-2509
  swiss-ai/Apertus-70B-Instruct-2509
Config source:
  Hugging Face Hub config/generation/tokenizer metadata snapshots saved beside this report.
Source files inspected:
  X:/H/transformers/src/transformers/models/apertus/configuration_apertus.py
  X:/H/transformers/src/transformers/models/apertus/modeling_apertus.py
  X:/H/transformers/src/transformers/models/apertus/modular_apertus.py
  X:/H/transformers/src/transformers/activations.py
  X:/H/transformers/src/transformers/modeling_rope_utils.py
  X:/H/transformers/src/transformers/masking_utils.py
Any missing files or assumptions:
  No Apertus-specific tokenizer source exists in the model family directory.
  modeling_apertus.py and configuration_apertus.py are generated from modular_apertus.py; future upstream source edits should inspect modular_apertus.py first, while DinoML parity should match generated imported code.
  HF connector found official Apertus repos and did not report gated access.
```

Snapshot files in this folder:

- `swiss-ai__Apertus-8B-2509__config.json`
- `swiss-ai__Apertus-8B-Instruct-2509__config.json`
- `swiss-ai__Apertus-70B-2509__config.json`
- `swiss-ai__Apertus-70B-Instruct-2509__config.json`
- matching `generation_config.json`, `tokenizer_config.json`, and `special_tokens_map.json` snapshots

Primary DinoML runtime target for this report: `ApertusForCausalLM` text-generation inference, with prefill and decode support. `ApertusModel` is required as the decoder body. `ApertusForTokenClassification` is optional/deferred for this target.

## 2. High-level architecture

Apertus is a decoder-only causal language model with Llama-like attention plumbing, GQA, RMSNorm, per-head Q/K RMSNorm, Llama-3-style RoPE scaling, and a custom xIELU feed-forward activation. It has no encoder, vision branch, audio branch, MoE, cross-attention, or multimodal embedding stitch in the inspected in-library source.

```text
tokenizer/input ids -> token embedding -> repeated decoder blocks -> final RMSNorm -> LM head -> logits/sampling
```

Runtime stages:

- CPU/data pipeline: fast tokenizer, BOS/EOS/pad handling, attention mask construction, and any caller-owned prompt/chat formatting.
- GPU prefill: embedding, full decoder stack over prompt length, causal attention, optional KV cache creation.
- GPU decode: one or more new tokens, RoPE position offset from cache length, KV cache update per layer, last-token logits.
- Generation controller: stopping on checkpoint-specific EOS ids, sampling/greedy policy, optional chat formatting.

Independently validatable pieces: RMSNorm, xIELU MLP, Llama-3 RoPE table/function, one attention layer with GQA and Q/K norm, one full decoder block, full prefill logits, cached decode token parity.

## 3. Important config dimensions

Source defaults from `ApertusConfig`:

| Field | Default |
| --- | ---: |
| `vocab_size` | 131072 |
| `hidden_size` | 4096 |
| `intermediate_size` | 14336 |
| `num_hidden_layers` | 32 |
| `num_attention_heads` | 32 |
| `num_key_value_heads` | defaults to `num_attention_heads` if omitted |
| `hidden_act` | `xielu` |
| `max_position_embeddings` | 65536 |
| `rms_norm_eps` | 1e-5 |
| `attention_bias` | false |
| `attention_dropout` | 0.0 |
| `tie_word_embeddings` | false |
| default RoPE | `llama3`, theta 12000000, factor 8, original max 8192, low/high factors 1/4 |

Representative checkpoint sweep from saved `config.json` snapshots:

| Checkpoint | Params from HF metadata | Layers | Hidden | Heads | KV heads | Head dim | MLP dim | Context | Dtype | Cache default | EOS in config | EOS in generation config |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| `swiss-ai/Apertus-8B-2509` | 8.05B | 32 | 4096 | 32 | 8 | 128 inferred | 21504 | 65536 | bf16 | true | 2 | 2 |
| `swiss-ai/Apertus-8B-Instruct-2509` | 8.05B | 32 | 4096 | 32 | 8 | 128 inferred | 21504 | 65536 | bf16 | false | 68 | 2, 68, 72 |
| `swiss-ai/Apertus-70B-2509` | 70.6B | 80 | 8192 | 64 | 8 | 128 inferred | 43008 | 65536 | bf16 | true | 2 | 2 |
| `swiss-ai/Apertus-70B-Instruct-2509` | 70.6B | 80 | 8192 | 64 | 8 | 128 inferred | 43008 | 65536 | bf16 | false | 68 | 2, 68, 72 |

The checkpoint configs use `rope_scaling` plus `rope_theta`; the current Transformers config path standardizes this to `rope_parameters` before RoPE computation. `head_dim` is omitted in all four configs, so the source computes `hidden_size // num_attention_heads = 128`.

Tokenizer metadata:

| Variant | Tokenizer class | BOS | EOS | PAD | Adds BOS | Adds EOS | Chat template |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Base | `PreTrainedTokenizerFast` | `<s>` | `</s>` | `<pad>` | true | false | no |
| Instruct | `PreTrainedTokenizerFast` | `<s>` | `<\|assistant_end\|>` | `<pad>` | true | false | not present in saved tokenizer config |

## 3a. Family variation traps

- GQA is mandatory for official checkpoints: 8B uses 32 query heads / 8 KV heads, 70B uses 64 query heads / 8 KV heads.
- Official configs omit `head_dim`; do not infer projection widths only from `hidden_size` without applying source logic. Here the inferred head dim is 128.
- `hidden_size == num_attention_heads * head_dim` for inspected checkpoints, but the source allows explicit `head_dim`; future configs could diverge.
- Source default `num_key_value_heads=None` means MHA if omitted, but official checkpoints explicitly use GQA.
- Checkpoint JSON advertises `qk_norm=true`, `mlp_bias=false`, and `post_norm=false`. The inspected source implements Q/K RMSNorm, bias-free MLP, pre-norm blocks, and ignores those flags as switches.
- `attention_bias=false` in official configs. If true, source adds bias to Q/K/V/O projections.
- `hidden_act=xielu` is a custom activation with learned scalar parameters `alpha_p` and `alpha_n`, not SwiGLU/GEGLU.
- No `gate_proj` exists in the Apertus MLP; do not apply common Llama SwiGLU rewrites.
- Base and instruct checkpoints differ in EOS semantics and `use_cache` config defaults. Generation config for instruct stops on `[2, 68, 72]` even though model config has `eos_token_id=68`.
- The model source supports attention backend selection (`eager`, `sdpa`, FlashAttention variants, paged variants through Transformers interfaces). DinoML should own a deterministic backend choice instead of inheriting runtime dispatch.
- No sliding-window, local, block-sparse, ALiBi, MoE, tensor-parallel collectives, or quantized/packed weight format is required by the inspected native source.
- Tensor layouts in modeling code are token-major dense PyTorch layouts: hidden states `[B, S, H]`, attention states `[B, heads, S, D]`. There is no NHWC/channel-last vision region.

## 4. Operator coverage checklist

Tensor/layout ops:

- token embedding lookup: `input_ids [B,S] -> [B,S,H]`
- view/reshape: projections to `[B,S,heads,D]` or `[B,S,kv_heads,D]`
- transpose between `[B,S,heads,D]` and `[B,heads,S,D]`
- contiguous after attention output transpose
- slice/indexing for `logits_to_keep`; common decode path should lower last-token-only logits
- residual add, dtype casts around RMSNorm and softmax

Neural network primitives:

- RMSNorm over last dim for hidden states, eps `1e-5`, fp32 variance, output cast to input dtype
- RMSNorm over per-head Q/K dim `D=128`
- Linear projections:
  - 8B Q: `4096 -> 4096`, K/V: `4096 -> 1024`, O: `4096 -> 4096`, MLP up/down: `4096 -> 21504 -> 4096`, LM head `4096 -> 131072`
  - 70B Q: `8192 -> 8192`, K/V: `8192 -> 1024`, O: `8192 -> 8192`, MLP up/down: `8192 -> 43008 -> 8192`, LM head `8192 -> 131072`
- xIELU activation with learned `alpha_p`, `alpha_n`, buffers `beta=0.5`, `eps=-1e-6`

Attention primitives:

- causal self-attention
- GQA repeat or backend-native GQA
- Q/K RoPE after Q/K RMSNorm and before cache update
- additive mask for eager attention
- softmax over key dimension in fp32, cast back to query dtype
- dropout is zero for inference
- KV cache update per layer, storing post-RoPE keys and unrotated values

Position/rotary ops:

- Llama-3 RoPE parameter standardization from checkpoint `rope_scaling`
- position ids default to `[past_seen, ..., past_seen+S-1]`
- cos/sin generated in fp32 and cast to hidden dtype

Generation/cache ops:

- Dynamic cache or equivalent per-layer KV cache
- cache length query for position id offset and mask sizing
- cache reorder only if beam search is later admitted; defer for first integration
- checkpoint-specific EOS handling in generation controller

Preprocessing-coupled ops:

- tokenizer and chat template are CPU/data-pipeline work
- attention mask is optional dense `[B,total_len]` padding mask from tokenizer/caller

Optional/deferred:

- token classification head from `ApertusForTokenClassification`
- training loss, labels, dropout, gradient checkpointing
- Hub-provided RMSNorm / rotary / xIELU kernels as optional provider replacements, not required semantic source

## 5. Layer/block breakdown

Decoder body:

```text
input_ids [B,S] -> embed_tokens -> x [B,S,H]
position_ids = arange(S) + cache_len
cos,sin = rotary_emb(x, position_ids) -> [B,S,D]
repeat N layers:
  residual = x
  x_norm = RMSNorm_H(x)
  q = Linear_q(x_norm).view(B,S,QH,D).transpose(1,2)
  k = Linear_k(x_norm).view(B,S,KVH,D).transpose(1,2)
  v = Linear_v(x_norm).view(B,S,KVH,D).transpose(1,2)
  q = RMSNorm_D(q)
  k = RMSNorm_D(k)
  q,k = RoPE(q,k,cos,sin)
  if cache: k,v = cache.update(k,v,layer_idx)
  attn = causal_attention(q,k,v,mask,scale=D**-0.5)
  attn = attn.transpose(1,2).reshape(B,S,QH*D)
  x = residual + Linear_o(attn)
  residual = x
  x = RMSNorm_H(x)
  x = Linear_down(xIELU(Linear_up(x)))
  x = residual + x
final = RMSNorm_H(x)
logits = Linear_lm(final[:, selected_positions, :])
```

All official checkpoints are bias-free for attention projections and MLP projections. LM head has no bias. Token embeddings and LM head are not tied (`tie_word_embeddings=false`), despite `_tied_weights_keys` metadata existing for possible tying in inherited loading machinery.

## 6. Attention requirements

Required attention variant: causal decoder self-attention with GQA.

- Query heads: 32 for 8B, 64 for 70B.
- KV heads: 8 for both official sizes.
- Head dim: 128 inferred.
- Query projection width: `num_attention_heads * head_dim` = hidden size for inspected configs.
- Key/value projection width: `num_key_value_heads * head_dim` = 1024 for both 8B and 70B.
- Query length: prompt length in prefill, usually 1 in decode.
- KV length: prompt length in prefill; cache length plus decode length in decode.
- Masking: causal mask plus optional padding mask. Eager path adds a float mask before softmax.
- Packed/varlen: generic Transformers mask utilities can detect packed position ids, but Apertus source does not add family-specific packed metadata. First DinoML target can require ordinary padded/unpacked batches.
- Sliding window/local attention: not configured or implemented for Apertus.
- Cross-attention: not present.
- Cache: per-layer K shape `[B, KVH, T, D]`, V shape `[B, KVH, T, D]`. K is cached after Q/K norm and RoPE; V is cached after V projection reshape/transpose and before any repeat.
- GQA expansion: eager path repeats K/V to `[B,QH,T,D]`; optimized backends may use native GQA.
- Backend compatibility: source declares FlashAttention, SDPA, flex attention, and paged attention support through generic Transformers dispatch. DinoML parity should implement a dense causal/GQA reference first, then optimized FlashAttention-style prefill/decode.

Attention math order:

```text
q_proj/k_proj/v_proj -> q_norm/k_norm -> RoPE(q,k) -> cache update -> attention(q,k,v,mask,scale) -> o_proj
```

## 7. Position encoding and custom math

RoPE is computed once per model forward and passed to every layer.

```python
def aperture_llama3_inv_freq(head_dim=128, theta=12000000.0, factor=8.0, old_context=8192,
                             low_freq_factor=1.0, high_freq_factor=4.0):
    inv_freq = 1.0 / (theta ** (arange(0, head_dim, 2).float() / head_dim))
    wavelen = 2 * pi / inv_freq
    low = old_context / low_freq_factor
    high = old_context / high_freq_factor
    inv_low = where(wavelen > low, inv_freq / factor, inv_freq)
    smooth = (old_context / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
    inv_mid = (1 - smooth) * inv_low / factor + smooth * inv_low
    return where((wavelen >= high) & (wavelen <= low), inv_mid, inv_low)
```

Application:

```python
def apply_rope(q, k, cos, sin):
    cos = cos[:, None, :, :]
    sin = sin[:, None, :, :]
    rotate = lambda x: cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)
    return q * cos + rotate(q) * sin, k * cos + rotate(k) * sin
```

Precomputable: inverse frequencies and optionally cos/sin tables up to the admitted max context. Runtime-dependent: `position_ids`, cache offset, batch expansion, dtype/device.

xIELU source behavior:

```python
alpha_p = softplus(alpha_p_param)
alpha_n = beta + softplus(alpha_n_param)
y = where(x > 0,
          alpha_p * x * x + beta * x,
          (expm1(min(x, eps)) - x) * alpha_n + beta * x)
```

The xIELU module has learned parameters, so lowering must load `alpha_p` and `alpha_n` weights, not treat the activation as parameter-free.

## 8. Preprocessing and input packing

Text preprocessing is owned by `PreTrainedTokenizerFast` metadata in the checkpoint repo.

- Model graph input: `input_ids [B,S]`, optional `attention_mask [B,total_len]`, optional `position_ids [B,S]`, optional `inputs_embeds [B,S,H]`.
- Exactly one of `input_ids` and `inputs_embeds` must be supplied.
- Default position ids are generated from cache length and current sequence length.
- Base tokenizer EOS token is `</s>`; instruct tokenizer EOS token is `<|assistant_end|>`.
- Instruct generation config stops on `[2, 68, 72]`; first integration should route stopping rules through generation metadata rather than the neural graph.
- `add_bos_token=true`, `add_eos_token=false` in tokenizer configs.
- No modality placeholder tokens, scatter, image/audio/video packing, cu_seqlens, or external feature extractor is required.

The tokenizer metadata reports an effectively unbounded `model_max_length`; the neural context guard should come from `max_position_embeddings=65536`, plus any DinoML deployment-specific admission limit.

## 9. Graph rewrite / lowering opportunities

### Rewrite: QKV projections as grouped GEMM, not packed-weight QKV

Source pattern:

```text
q = Linear_q(x)
k = Linear_k(x)
v = Linear_v(x)
```

Replacement:

```text
three independent GEMMs, or grouped GEMM with three output buffers
```

Preconditions:

- Same input tensor and batch/sequence flattening.
- Bias handling matches `attention_bias`; official checkpoints have no bias.
- Preserve output split widths: Q width `QH*D`, K/V width `KVH*D`.

Failure cases:

- Do not assume a single packed QKV source weight; source weights are separate modules.
- If future config has `attention_bias=true`, include three separate bias vectors.

Parity test sketch: compare Q/K/V tensors immediately before Q/K RMSNorm for random hidden states.

### Rewrite: last-token-only logits

Source pattern:

```text
logits = lm_head(hidden_states[:, slice_indices, :])
```

Replacement:

```text
decode path: gather final hidden token -> GEMM(H -> vocab)
prefill scoring path: optional selected positions only
```

Preconditions:

- `logits_to_keep` is an integer or compile-known index tensor.
- Generation only needs last-token logits.

Failure cases:

- Training/loss or full-logit export requires all positions.

Parity test sketch: compare full logits last position against optimized last-token logits.

### Rewrite: GQA repeat elimination

Source pattern:

```text
k = repeat_kv(k, num_key_value_groups)
v = repeat_kv(v, num_key_value_groups)
attn = softmax(q @ k.T) @ v
```

Replacement:

```text
GQA attention kernel maps query head h to kv head h // groups without materializing repeated K/V
```

Preconditions:

- `num_attention_heads % num_key_value_heads == 0`.
- No downstream consumer requires materialized repeated K/V.

Failure cases:

- Debug output of attention internals expecting repeated tensors.

Parity test sketch: compare eager repeated-KV attention with grouped-head kernel for prefill and cached decode.

### Rewrite: xIELU fused MLP epilogue

Source pattern:

```text
up = GEMM(x, W_up)
y = xIELU(up)
out = GEMM(y, W_down)
```

Replacement:

```text
GEMM + xIELU elementwise fusion before down GEMM
```

Preconditions:

- xIELU parameters are loaded and broadcast as scalars.
- Activation input is dense contiguous in last dimension.

Failure cases:

- Missing `expm1`, `min`, or dtype parity in reduced precision.

Parity test sketch: random tensors across negative/near-zero/positive values; compare fp32 and bf16 tolerances.

## 10. Kernel fusion candidates

Highest priority:

- RMSNorm for hidden states and per-head Q/K. It appears twice per block plus two head-dim norms in attention.
- GQA FlashAttention-style prefill/decode with RoPE-applied cached keys. This dominates runtime and memory.
- xIELU MLP activation fusion. Apertus MLP is a wide `up -> xIELU -> down` path with no gate; fusing the activation avoids a large intermediate round trip.
- Last-token-only LM head for decode. Vocab is 131072, so avoiding full sequence logits matters.

Medium priority:

- Q/K/V grouped GEMM launch scheduling for separate projection weights.
- RoPE generation/application fusion near attention.
- KV cache layout optimized for `[layer, batch, kv_head, seq, head_dim]` or DinoML's cache ABI.
- Bias-free GEMM specialization for all official projections.

Lower priority:

- Token classification head.
- Beam cache reorder.
- Hub kernel compatibility for xIELU/RMSNorm; useful as a reference/provider option, not required for first parity.
- Tensor parallel plans from config metadata.

## 11. Runtime staging plan

Stage 1: parse Apertus config and load weights for 8B shape metadata. Admit bf16 weights with fp32 reference path where needed.

Stage 2: implement/validate scalar primitives: RMSNorm, xIELU, Llama-3 RoPE parameter generation, RoPE application.

Stage 3: one-block parity without cache using random weights/tensors, eager dense causal attention, no tokenizer.

Stage 4: full 8B prefill parity against Transformers for short prompts, full logits or last-token logits.

Stage 5: cached decode parity with dynamic cache ABI, position offset, and per-layer K/V update.

Stage 6: optimized attention and GEMM lowering: GQA no-repeat attention, last-token logits, MLP fusion, optional profile-guided GEMM candidates.

Stage 7: production generation wrapper: tokenizer metadata, instruct EOS set, chat template outside graph, batching and cache scheduling.

Stage 8: scale to 70B dimensions and evaluate GGUF/quantized loading as a separate provider contract if requested. No quantized storage is required by native Apertus source.

Initially stub/defer: token classification, training loss, dropout, gradient checkpointing, beam-search cache reorder, tensor parallel collectives, Hub custom kernels.

## 12. Parity and validation plan

- xIELU unit tests: compare source formula over fp32/bf16 inputs, including negative values near `eps=-1e-6`, zero, and large positives.
- RMSNorm unit tests: hidden-size and head-dim variants, fp32 variance, bf16 output.
- RoPE unit tests: compare inv_freq/cos/sin for positions `[0, 1, 8191, 8192, 65535]`.
- Attention unit tests: 8B dimensions with `B=1..4`, `S=1, 7, 128`, GQA 32/8; compare eager attention before and after repeat-elimination.
- Cache tests: prefill then single-token decode equals full forward on concatenated sequence within tolerance.
- Single decoder layer parity with random weights.
- Full small-sequence model parity on a real checkpoint if weights are available locally; compare final hidden and logits.
- Generation metadata parity: instruct checkpoints stop on any of `[2,68,72]`.

Suggested tolerances:

- fp32 primitives: `rtol=1e-5`, `atol=1e-6`
- bf16 end-to-end logits: start with `rtol=3e-2`, `atol=3e-2`, tighten per kernel as DinoML bf16 paths mature
- attention probabilities in fp32 softmax reference: compare outputs rather than probabilities for optimized kernels

## 13. Performance probes

- Tokenizer and caller-owned prompt-formatting throughput outside DinoML graph.
- Prefill latency/throughput sweep: batch `{1,4,8}`, sequence `{128,1024,4096,8192,65536 admitted cap}`.
- Decode tokens/sec sweep with KV cache: batch `{1,8,32}`, context `{128,4096,32768,65536}`.
- Attention backend comparison: eager dense reference, SDPA-like, FlashAttention/GQA no-repeat.
- KV cache memory usage for 8B and 70B: layers * 2 * B * KVH * T * 128 * dtype bytes.
- MLP throughput: `H -> intermediate -> H` with xIELU fusion on/off.
- LM head last-token vs full-sequence logits.
- Weight load time and residency memory for bf16 8B/70B.
- Optional future quantized/GGUF load and dequant probes; label as provider work, not native source requirement.

## 14. Skip/defer list

- Training, labels/loss, dropout behavior, gradient checkpointing.
- Token classification head.
- Beam search and cache reorder until greedy/sampling decode parity is stable.
- Tensor parallel and pipeline parallel plans.
- FlashAttention/paged attention exact backend parity beyond semantic dense attention.
- Hub custom kernels for RMSNorm/rotary/xIELU.
- Quantized or packed weight formats; native configs are bf16 dense safetensors.
- Any multimodal, vision, audio, sparse/local attention, or MoE path.

## 15. Final implementation checklist

- [ ] Parse Apertus config, including legacy `rope_scaling` -> `rope_parameters` standardization.
- [ ] Load bf16 dense weights and preserve untied embedding/LM-head parameters.
- [ ] Implement/tokenize CPU-side metadata handling for BOS/EOS/PAD and instruct stop ids.
- [ ] Implement RMSNorm over hidden dim and head dim.
- [ ] Implement xIELU with learned `alpha_p`/`alpha_n` parameters.
- [ ] Implement Llama-3 RoPE inv_freq/cos/sin and apply order.
- [ ] Implement bias-free Q/K/V/O and MLP GEMMs for 8B/70B dimensions.
- [ ] Implement GQA causal attention reference with optional padding mask.
- [ ] Implement KV cache ABI storing post-RoPE K and V per layer.
- [ ] Add one-block random parity tests.
- [ ] Add prefill logits parity against Transformers.
- [ ] Add cached decode parity against full forward.
- [ ] Add last-token-only logits rewrite and parity test.
- [ ] Add GQA repeat-elimination rewrite and parity test.
- [ ] Benchmark prefill, decode, MLP, LM head, and KV memory.
