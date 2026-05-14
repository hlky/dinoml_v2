# GPT-2 Transformers Audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model family: gpt2
Primary runtime target: GPT2LMHeadModel causal LM prefill, decode, and generation
Dinoml assumptions: inference-only first, CUDA GPU target, preserve PyTorch/Transformers tensor axes, prefer graph rewrites that canonicalize Conv1D/attention/LayerNorm/GELU into explicit runtime primitives.
```

Source files inspected:

- Local: `X:/H/transformers/src/transformers/models/gpt2/modeling_gpt2.py`
- Local: `X:/H/transformers/src/transformers/models/gpt2/configuration_gpt2.py`
- Local: `X:/H/transformers/src/transformers/models/gpt2/tokenization_gpt2.py`
- Local shared utilities: `X:/H/transformers/src/transformers/pytorch_utils.py`, `X:/H/transformers/src/transformers/cache_utils.py`, `X:/H/transformers/src/transformers/masking_utils.py`, `X:/H/transformers/src/transformers/activations.py`
- Upstream source URL pattern at the pinned commit: `https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/gpt2/...`

Representative Hugging Face configs fetched from official Hub raw files:

- `https://huggingface.co/sshleifer/tiny-gpt2/raw/main/config.json`
- `https://huggingface.co/distilbert/distilgpt2/raw/main/config.json`
- `https://huggingface.co/openai-community/gpt2/raw/main/config.json`
- `https://huggingface.co/openai-community/gpt2-medium/raw/main/config.json`
- `https://huggingface.co/openai-community/gpt2-large/raw/main/config.json`
- `https://huggingface.co/openai-community/gpt2-xl/raw/main/config.json`
- Matching `tokenizer_config.json` and available `generation_config.json` files from the same repos.

Missing files or assumptions:

- No remote-code files are required for these checkpoints.
- The source tree contains no separate `tokenization_gpt2_fast.py`; this commit uses `GPT2Tokenizer(TokenizersBackend)` with a fast tokenizers backend built from BPE, ByteLevel pre-tokenizer, and ByteLevel decoder.
- `sshleifer/tiny-gpt2` has no `generation_config.json`; generation token IDs come from `config.json` and Transformers defaults.
- Dtype/parameter counts are not asserted from Hub metadata in this report. All dimensions below are from `config.json` or source defaults.

## 2. High-level architecture

GPT-2 is a text-only decoder-only Transformer with learned token embeddings, learned absolute position embeddings, pre-LayerNorm decoder blocks, causal self-attention, a non-gated GELU MLP, final LayerNorm, and a tied bias-free LM projection.

```text
byte-level BPE text preprocessing -> input_ids/attention_mask
  -> token embedding + learned absolute position embedding
  -> repeated decoder blocks
  -> final LayerNorm
  -> tied LM head logits
  -> generation controller / sampling
```

Stage decomposition:

- CPU/data pipeline: byte-level BPE tokenization, padding, attention mask creation, optional special-token handling.
- GPU prefill: embed full prompt `[B, S]`, build positions `[1 or B, S]`, run all decoder layers with causal self-attention over `S`, and optionally populate per-layer K/V cache.
- GPU decode: embed new token chunk `[B, T_decode]`, positions offset by cache length, append new K/V to cache, attend over `[past + T_decode]`, produce logits usually for the last token only.
- Generation controller: outside the core module graph; handles BOS/EOS, max length, sampling/beam logic, stopping, and cache reordering.

Optimization can validate tokenization, one block, prefill logits, decode logits, cache append, and logits slicing independently.

## 3. Important config dimensions

`GPT2Config` source defaults:

| Field | Default | Source / effect |
| --- | ---: | --- |
| `model_type` | `gpt2` | config class |
| `vocab_size` | 50257 | token embedding and LM head rows |
| `n_positions` / `max_position_embeddings` | 1024 | learned absolute position table rows |
| `n_embd` / `hidden_size` | 768 | hidden width |
| `n_layer` / `num_hidden_layers` | 12 | decoder block count |
| `n_head` / `num_attention_heads` | 12 | MHA heads |
| `head_dim` | `n_embd / n_head` | source requires exact divisibility |
| `n_inner` | null | MLP width defaults to `4 * n_embd` |
| `activation_function` | `gelu_new` | tanh GELU approximation |
| `layer_norm_epsilon` | `1e-5` | all LayerNorms |
| `scale_attn_weights` | `true` | attention scale by `head_dim ** -0.5` |
| `scale_attn_by_inverse_layer_idx` | `false` | optional extra scale by `1/(layer_idx+1)` |
| `reorder_and_upcast_attn` | `false` | optional eager-only fp32 baddbmm attention path |
| `use_cache` | `true` | generation cache enabled by default |
| `add_cross_attention` | `false` | encoder-decoder style cross-attn is optional/deferred |
| `tie_word_embeddings` | `true` | LM head tied to token embedding |
| `bos_token_id`, `eos_token_id` | 50256 | generation/tokenizer end-of-text token |
| `pad_token_id` | null | no native pad token in base GPT-2 |

Representative checkpoint sweep:

| Model id | Layers | Hidden | Heads | Head dim | MLP width | Positions | Vocab | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `sshleifer/tiny-gpt2` | 2 | 2 | 2 | 1 | 8 | 1024 | 50257 | Debug checkpoint; same ops, pathological tiny head_dim |
| `distilbert/distilgpt2` | 6 | 768 | 12 | 64 | 3072 | 1024 | 50257 | Distilled GPT-2, fewer layers |
| `openai-community/gpt2` | 12 | 768 | 12 | 64 | 3072 | 1024 | 50257 | Common base checkpoint |
| `openai-community/gpt2-medium` | 24 | 1024 | 16 | 64 | 4096 | 1024 | 50257 | Extra config fields `n_special`, `predict_special_tokens` appear but do not change source ops |
| `openai-community/gpt2-large` | 36 | 1280 | 20 | 64 | 5120 | 1024 | 50257 | Larger dense/cache memory |
| `openai-community/gpt2-xl` | 48 | 1600 | 25 | 64 | 6400 | 1024 | 50257 | Extra `output_past` legacy field; current source uses `use_cache` |

Effective defaults often omitted from checkpoint configs: `n_inner=None`, `scale_attn_weights=True`, `scale_attn_by_inverse_layer_idx=False`, `reorder_and_upcast_attn=False`, `add_cross_attention=False`, `tie_word_embeddings=True`, `pad_token_id=None`, and current attention implementation selection via `config._attn_implementation`.

## 3a. Family variation traps

- GPT-2 uses MHA, not GQA/MQA: `num_key_value_heads == num_attention_heads` by architecture, but there is no explicit `num_key_value_heads` config field.
- All standard OpenAI GPT-2 sizes use `head_dim=64`; tiny GPT-2 uses `head_dim=1`, which is useful for smoke tests but not representative for kernel tuning.
- `Conv1D` is semantically a linear layer but stores weights as `[in_features, out_features]`. Dinoml weight loading must either preserve an RRR-style matmul using the stored layout or transpose into a normal `[out, in]` linear constant.
- `c_attn` is a fused QKV projection for self-attention with output `3 * hidden`; split is along hidden dim into Q, K, V. Cross-attention uses separate `q_attn` plus a fused KV `c_attn`, but cross-attention is not required for causal LM GPT-2 checkpoints.
- Position encoding is learned absolute embedding, not RoPE/ALiBi. Position IDs are generated from `past_key_values.get_seq_length()` when omitted.
- Token type IDs, if supplied, are embedded through the same token embedding table `wte` and added to hidden states. They are optional and uncommon for generation.
- Padding is not built into GPT-2 tokenizer/config by default; batched generation with padding needs explicit tokenizer/model pad policy. Sequence classification has a hard batch-size trap when no pad token is configured, but this is deferred for LM.
- `reorder_and_upcast_attn=True` changes attention math order in eager mode: fp32 `baddbmm`, scaling in the GEMM alpha, softmax fp32, then downcast. Standard fetched configs omit it and therefore use the normal attention interface.
- No layout translation is needed for text tensors. Protect embedding, split/view/transpose, LayerNorm `dim=-1`, softmax `dim=-1`, and LM head logits axes with a conceptual no-layout-translation guard.

## 4. Operator coverage checklist

Tensor/layout ops:

- Integer input IDs `[B, S]`; optional attention mask `[B, S_total]`.
- Embedding lookup for `wte`: `[vocab_size, H]`, output `[B, S, H]`.
- Embedding lookup for `wpe`: `[n_positions, H]`, `position_ids` `[1 or B, S]`, output broadcast/add to `[B, S, H]`.
- Optional token type embedding lookup through `wte(token_type_ids)`.
- Add, residual add, dropout as inference identity, view/reshape, transpose, split along last dim, contiguous/materialization when needed.
- Final logits slice for `logits_to_keep`: either all sequence positions or selected tail/index positions.

Neural network primitives:

- LayerNorm over hidden dim, epsilon `1e-5`: `ln_1`, `ln_2`, `ln_f`.
- Conv1D-as-linear with bias:
  - Self-attention `c_attn`: `[B, S, H] -> [B, S, 3H]`, stored weight `[H, 3H]`, bias `[3H]`.
  - Attention output `c_proj`: `[B, S, H] -> [B, S, H]`, stored weight `[H, H]`, bias `[H]`.
  - MLP `c_fc`: `[B, S, H] -> [B, S, 4H]`, stored weight `[H, 4H]`, bias `[4H]`.
  - MLP `c_proj`: `[B, S, 4H] -> [B, S, H]`, stored weight `[4H, H]`, bias `[H]`.
- `gelu_new`: `0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))`.
- LM head: bias-free `Linear(H -> vocab_size)` tied to `wte.weight`; normal `nn.Linear` expects weight `[vocab_size, H]`.

Attention primitives:

- Causal self-attention over `q [B, heads, Tq, D]`, `k/v [B, heads, Tkv, D]`.
- Mask addition before softmax for eager path; SDPA/Flash backends may receive `None` mask and use backend causal handling when legal.
- Softmax over key dimension `dim=-1`, dropout disabled for inference, then attention-value matmul.
- Cache update/append per layer.

Generation/cache ops:

- `DynamicCache` allocation or caller-supplied cache.
- Per-layer cache append along sequence axis `-2`.
- `get_seq_length()` for position offset and mask length.
- Beam search cache reorder via `index_select(0, beam_idx)` is needed for beam generation but can be deferred for greedy/sampling.

Preprocessing-coupled ops:

- Byte-level BPE tokenization with `vocab.json` and `merges.txt`.
- Attention mask semantics: 1/true means keep, 0/false means masked/pad before conversion to backend mask/bias.

## 5. Layer/block breakdown

For hidden width `H`, heads `A`, head dim `D=H/A`, MLP width `F=n_inner or 4H`, sequence chunk `T`, current K/V length `Ktot`:

```text
Input:
  input_ids [B, T]
  inputs_embeds = wte(input_ids) [B, T, H]
  position_ids = arange(T) + past_seen_tokens [1, T]
  hidden = inputs_embeds + wpe(position_ids) [B, T, H]
```

Decoder block, repeated `n_layer` times:

```text
residual = hidden
x = LayerNorm(hidden, eps=1e-5)
qkv = Conv1D_c_attn(x)                         # [B, T, 3H], weight [H, 3H]
q, k_new, v_new = split(qkv, H, dim=-1)
q = view(q, [B, T, A, D]).transpose(1, 2)       # [B, A, T, D]
k_new = view(k_new, [B, T, A, D]).transpose(1, 2)
v_new = view(v_new, [B, T, A, D]).transpose(1, 2)
k, v = cache_append_or_identity(k_new, v_new)  # [B, A, Ktot, D]
attn = causal_attention(q, k, v, mask, scale=D**-0.5)
attn = transpose/reshape(attn, [B, T, H])
hidden = residual + Conv1D_c_proj(attn)         # weight [H, H]

residual = hidden
x = LayerNorm(hidden, eps=1e-5)
x = Conv1D_c_fc(x)                              # [B, T, F], weight [H, F]
x = gelu_new(x)
x = Conv1D_mlp_c_proj(x)                        # [B, T, H], weight [F, H]
hidden = residual + x
```

Final:

```text
hidden = LayerNorm(hidden, eps=1e-5)
logits = hidden[:, slice_indices, :] @ wte.weight.T  # [B, T_keep, vocab]
```

All GPT-2 Conv1D projections have bias. The LM head has no bias and is tied to token embeddings.

## 6. Attention requirements

Required for target:

- Causal self-attention only.
- MHA: query heads = key heads = value heads = `n_head`; no repeat-kv.
- Head dim: source computes `hidden_size // num_attention_heads` and raises if not exact.
- Prefill shapes:
  - Q/K/V new: `[B, A, S, D]`
  - Cache after layer update when `use_cache=True`: keys `[B, A, S, D]`, values `[B, A, S, D]`
  - Attention scores: `[B, A, S, S]`
- Decode shapes for token chunk `T` and prior cache length `P`:
  - Q new `[B, A, T, D]`
  - K/V new `[B, A, T, D]`
  - Cache after append `[B, A, P+T, D]`
  - Attention scores `[B, A, T, P+T]`
- Cached keys are stored after projection and reshape/transpose. There is no RoPE/position transform on Q/K, so the cache stores plain projected K/V.
- Masking:
  - `GPT2Model.forward` reshapes a rank-2 attention mask to `[B, S_total]`.
  - `create_causal_mask` combines causal lower-triangular visibility with padding and backend-specific representation.
  - Eager attention receives an additive mask/bias, adds it to scores, then softmaxes.
  - SDPA/Flash paths can receive `None` when the mask can be represented by backend `is_causal` and there is no padding/packing complication.
- Attention math order in eager default:
  - `scores = matmul(q, k.transpose(-1,-2)) * scaling`
  - add mask if present
  - softmax in last dimension
  - cast attention weights to value dtype
  - dropout, then matmul with V
- Optional eager `reorder_and_upcast_attn`:
  - reshapes Q/K to `[B*A, Tq, D]` and `[B*A, D, Tk]`
  - uses fp32 `baddbmm` with `alpha=scaling`
  - softmax remains fp32 before downcast to V dtype

Optimized backend source path:

- `GPT2PreTrainedModel` declares support for FlashAttention and SDPA.
- `GPT2Attention.forward` dispatches through `ALL_ATTENTION_FUNCTIONS` based on `config._attn_implementation`, falling back to eager. Dinoml parity should start from eager math, then lower to fused prefill/decode attention under strict mask/cache guards.

Deferred attention variants:

- Cross-attention from `add_cross_attention=True` uses `q_attn` plus KV projection over encoder hidden states and an `EncoderDecoderCache`; not needed for standard GPT2LMHeadModel checkpoints.
- Packed/block custom masks are supported in shared masking utilities but not a GPT-2 checkpoint requirement.

## 7. Position encoding and custom math

GPT-2 uses learned absolute position embeddings:

```python
if position_ids is None:
    past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
    position_ids = torch.arange(T, device=device) + past_seen_tokens
    position_ids = position_ids.unsqueeze(0)
hidden_states = wte(input_ids) + wpe(position_ids)
```

Precomputable:

- `wpe.weight [n_positions, H]` is a normal constant.
- For fixed max sequence, Dinoml can precompute/arange position IDs and gather rows, but the start offset depends on current cache length in decode.

Dynamic inputs:

- Caller-supplied `position_ids` must override the default.
- Decode offset depends on `past_key_values.get_seq_length()`.

Custom math required:

```python
def gelu_new(x):
    return 0.5 * x * (1.0 + tanh(sqrt(2.0 / pi) * (x + 0.044715 * x**3)))
```

No RoPE, ALiBi, relative position bias, sliding window, or convolutional positional embeddings are required for standard GPT-2.

## 8. Preprocessing and input packing

Tokenizer coupling:

- GPT-2 tokenizer is byte-level BPE over `vocab.json` and `merges.txt`.
- The tokenizer uses `pre_tokenizers.ByteLevel(add_prefix_space=add_prefix_space)` and `decoders.ByteLevel()`.
- Default special tokens are all `"<|endoftext|>"`: unknown, BOS, and EOS map to token ID 50256 in standard configs.
- Default `add_prefix_space=False`. Leading spaces affect token IDs; this is model-coupled behavior and must stay in the CPU tokenizer pipeline.
- `model_input_names = ["input_ids", "attention_mask"]`.
- `tokenizer_config.json` for inspected checkpoints reports `model_max_length=1024`.

Runtime graph inputs:

- Required first integration: `input_ids [B, S]`, optional `attention_mask [B, S_total]`, optional `past_key_values`, optional `position_ids`.
- `token_type_ids` are optional. If present, the model embeds them through `wte` and adds them to token+position embeddings. This can be deferred for generation-first parity if the loader rejects or ignores absent token type IDs only.
- Padding is not native: configs have `pad_token_id=None`. For batched padded generation, the caller/tokenizer must define a pad token, often EOS by convention outside this source.

Generation controller facts from fetched configs:

- `bos_token_id=50256`, `eos_token_id=50256`.
- `task_specific_params.text-generation` commonly says `do_sample=true`, `max_length=50`; this is Hub config metadata, not core graph structure.
- Available generation configs are `_from_model_config=true` with BOS/EOS IDs and old Transformers version tags.

No multimodal placeholders, packed patch descriptors, `cu_seqlens`, audio features, or image processors are involved.

## 9. Graph rewrite / lowering opportunities

### Rewrite: GPT-2 Conv1D -> explicit Linear/GEMM

Source pattern:

```text
y = torch.addmm(bias, x.view(-1, in_features), weight_stored)
weight_stored shape = [in_features, out_features]
y.view(..., out_features)
```

Replacement:

```text
FlattenLeading -> GEMM_RRR(x_2d [M, K], weight_stored [K, N]) -> BiasAdd -> RestoreLeading
```

Preconditions:

- Source op is `transformers.pytorch_utils.Conv1D`.
- Last dimension of input is exactly `nx`.
- Stored weight shape is `[nx, nf]`; bias shape `[nf]`.
- No training dropout semantics required.

Shape equations:

- `M = B * T` or product of leading dims.
- `K = nx`, `N = nf`.
- Output shape is original leading dims plus `nf`.

Weight transform:

- Preferred Dinoml lowering: consume stored `[K, N]` directly with `gemm_rrr_bias`.
- Alternative loader transform for normal linear ABI: `linear_weight = conv1d.weight.T` giving `[N, K]`; then use RCR/Linear conventions. Record provenance to avoid double transpose.

Failure cases:

- Accidentally treating Conv1D weight as `[out, in]` silently produces wrong outputs or shape mismatches.
- Quantized or packed loaders must preserve the chosen logical layout in manifests.

Parity test sketch:

- Compare each projection (`c_attn`, `c_proj`, `c_fc`, `mlp.c_proj`) against HF Conv1D for random `[B, T, H]`, including tiny GPT-2 and base GPT-2 shapes.

### Rewrite: self-attention QKV Conv1D split -> fused QKV projection

Source pattern:

```text
q, k, v = Conv1D(H -> 3H)(x).split(H, dim=2)
q/k/v = view([B, T, A, D]).transpose(1, 2)
```

Replacement:

```text
FusedQKVLinear -> split logical Q/K/V -> layout transform to [B, A, T, D]
```

Preconditions:

- `add_cross_attention=False`.
- `hidden_size == num_heads * head_dim`.
- `c_attn` output exactly `3H`.
- Split order is Q, K, V.

Shape equations:

- Input `[B, T, H]`; QKV `[B, T, 3H]`; each projected tensor `[B, A, T, D]`.

Weight transform:

- If using stored Conv1D layout, c_attn weight is `[H, 3H]`, with contiguous Q/K/V column ranges.
- For three separate linear kernels, split columns into `[H, H]` chunks and transpose only if the target linear ABI requires `[out, in]`.

Failure cases:

- Cross-attention mode uses different projections and should not hit this rewrite.
- Non-contiguous output layout assumptions around `view` and `transpose` must be captured before fused attention.

Parity test sketch:

- Validate Q/K/V tensors before attention against HF for base and tiny configs.

### Rewrite: prefill MHA -> fused causal attention

Source pattern:

```text
scores = q @ k.T * scale
scores += additive_mask
weights = softmax(scores, dim=-1)
out = weights @ v
```

Replacement:

```text
Flash/cutlass-style causal attention(q, k, v, optional padding mask, scale)
```

Preconditions:

- Self-attention, MHA, no cross-attention.
- No requested attention weights output.
- Dropout disabled (`eval()`).
- Mask is pure causal or causal plus standard padding mask representable by the backend.
- No `reorder_and_upcast_attn` unless fused kernel explicitly matches fp32 score/softmax behavior.

Shape equations:

- Prefill: `q,k,v [B,A,S,D] -> out [B,A,S,D]`.
- Decode: `q [B,A,T,D]`, `k/v [B,A,P+T,D] -> out [B,A,T,D]`.

Failure cases:

- Packed/block masks, custom mask functions, or explicit attention outputs.
- Backend numerical behavior may differ from eager in fp16/bf16; tolerances must reflect chosen path.

Parity test sketch:

- Prefill logits and per-block hidden-state comparison for `S={1,16,128,1024}` and padding/no-padding cases.

### Rewrite: decode attention + KV append

Source pattern:

```text
k_cache = cat(k_cache, k_new, dim=-2)
v_cache = cat(v_cache, v_new, dim=-2)
attn(q_new, k_cache, v_cache)
```

Replacement:

```text
In-place cache write at cache_position -> fused decode attention over cache
```

Preconditions:

- Static maximum cache length is known/admitted by runtime session.
- Cache tensor layout is `[layer, 2, B, A, max_seq, D]` or equivalent artifact-visible layout with per-layer views.
- Position offset equals previous cache length unless caller supplied compatible `position_ids`.

Shape equations:

- Write `k_new/v_new [B,A,T,D]` into slots `[P:P+T]`.
- Attend over `P+T` keys.

Failure cases:

- DynamicCache source uses `torch.cat`; Dinoml should not literally reallocate in production.
- Beam cache reorder requires batch-axis gather on all per-layer K/V tensors.

Parity test sketch:

- Run one prefill then N single-token decode steps; compare logits and final cache tensors to HF.

### Rewrite: last-token-only logits

Source pattern:

```text
slice_indices = slice(-logits_to_keep, None) if int else logits_to_keep
logits = lm_head(hidden_states[:, slice_indices, :])
```

Replacement:

```text
Gather/slice hidden positions -> tied embedding GEMM only for kept positions
```

Preconditions:

- Generation only needs last token or a known index set.
- Labels/loss are not requested.
- Tied embedding constant is available as `[vocab, H]`; GEMM uses `hidden [M,H] @ embedding.T [H,V]`.

Failure cases:

- Full-sequence logprob/loss or speculative/assistant paths may need more than last token.

Parity test sketch:

- Compare `logits_to_keep=1`, `0`, and tensor indices against HF output shapes/values.

## 10. Kernel fusion candidates

Highest priority:

- Conv1D/GEMM loader and lowering: every GPT-2 projection depends on the `[in, out]` stored layout.
- LayerNorm over last dim: three per layer plus final LayerNorm; pre-LN blocks make this latency-critical.
- Fused QKV projection with bias and split metadata: removes extra materialization and feeds attention layout directly.
- Fused causal attention prefill and decode with artifact-visible KV cache: dominant runtime cost for long prompts and generation.
- Last-token-only tied LM head: avoids `[B,S,V]` logits in decode and long prompt continuation.

Medium priority:

- Bias + GELU_new + projection scheduling in MLP: `c_fc -> gelu_new -> c_proj`; activation fusion and epilogue choices matter.
- Residual add fusion around attention/MLP projections where memory bandwidth dominates.
- Position ID generation + position embedding gather for decode offset.
- Mask canonicalization: distinguish pure causal, causal+padding, and unsupported custom masks early.

Lower priority:

- Token type embedding add.
- Cross-attention blocks for custom GPT-2-as-decoder configs.
- Classification/QA heads.
- Beam cache reorder and generation processors beyond greedy/sampling.

## 11. Runtime staging plan

Stage 1: config and weight loading

- Parse `GPT2Config`, reject unsupported `add_cross_attention=True` for the first LM target.
- Load `wte`, `wpe`, LayerNorm weights, Conv1D weights with explicit `[in,out]` layout metadata, and tied LM head.

Stage 2: single-block eager parity

- Implement embeddings, LayerNorm, Conv1D-as-GEMM, GELU_new, residuals, and eager causal attention for one block.
- Validate tiny GPT-2 and base GPT-2 shapes.

Stage 3: full prefill parity

- Run all layers for `GPT2LMHeadModel` with no cache and with cache population.
- Support attention masks for no-padding and standard padding.

Stage 4: decode with KV cache

- Replace dynamic cat with session-owned cache storage.
- Implement position offset, in-place K/V append, decode attention, and last-token logits.

Stage 5: optimized attention

- Add fused prefill/decode attention under pure-causal and causal+padding guards.
- Preserve eager fallback for unsupported masks or attention-output requests.

Stage 6: generation controller surface

- Minimal greedy/sampling loop using BOS/EOS from config/generation_config.
- Add cache reorder only when beam search is admitted.

Stage 7: fusions and production batching

- Add QKV+layout fusion, residual/LayerNorm scheduling, logits slicing, continuous batching/cache-page policy.

Initially stub/defer labels/loss, training dropout, output attentions/hidden states, beam search, cross-attention, and non-LM heads.

## 12. Parity and validation plan

- Tokenizer smoke: byte-level BPE encodes `"Hello world"` and `" Hello world"` differently; EOS/BOS token ID is 50256.
- Conv1D primitive: random tensors for shapes from tiny, base, medium, XL; compare direct HF Conv1D to Dinoml GEMM lowering.
- GELU_new primitive: random fp32/fp16 inputs; compare tanh approximation exactly to source formula.
- Single decoder block: compare hidden output and cache K/V for `B=1/2`, `S=1/16`, no padding.
- Attention mask tests: prefill with left/right padding masks and decode with prior cache length; compare logits.
- Full prefill logits: `sshleifer/tiny-gpt2` for cheap exact checks, `openai-community/gpt2` for representative shapes.
- Decode token parity: prefill a prompt, then feed one token at a time with cache; compare each-step logits and sampled greedy token.
- Last-token logits: compare `logits_to_keep=1`, full logits, and tensor index selection.
- Cache reorder later: beam index gather on batch axis for all layers.

Recommended tolerances:

- fp32 eager: `rtol=1e-4`, `atol=1e-5` for hidden/logits, tighter for primitive GEMMs if deterministic.
- fp16/bf16 optimized attention: start with `rtol=5e-2`, `atol=5e-2` for logits, then tune per backend; compare greedy token equality as an end-to-end sanity check.

## 13. Performance probes

- Tokenizer throughput: texts/sec and tokens/sec for byte-level BPE, separate from GPU runtime.
- Prefill throughput: tokens/sec for `S={1,16,128,512,1024}`, `B={1,4,16}`.
- Decode throughput: generated tokens/sec for cache lengths `{16,128,512,1024}` and batch sizes `{1,4,16,64}`.
- KV cache memory: bytes = `layers * 2 * B * heads * max_seq * head_dim * dtype_size`; for GPT-2 base fp16 at `B=1,S=1024`: `12*2*1*12*1024*64*2 ~= 37.7 MB`.
- Attention backend comparison: eager matmul/softmax vs SDPA/Flash-equivalent prefill and decode.
- GEMM probe: Conv1D QKV, MLP up/down, and LM head separately, with stored `[K,N]` layout vs transposed loader path.
- Last-token logits probe: full `[B,S,V]` logits vs kept-position logits.
- Batch scheduling probe: continuous decode with heterogeneous prompt lengths and cache offsets.

All probes above are proposed; no benchmark observations are included.

## 14. Skip/defer list

- Training, losses, dropout randomness, gradient checkpointing.
- `output_attentions=True` and full hidden-state recording.
- Beam search and cache reorder for first greedy/sampling milestone.
- Cross-attention / `add_cross_attention=True`.
- `GPT2DoubleHeadsModel`, sequence classification, token classification, and QA heads.
- `reorder_and_upcast_attn=True` optimized parity; support with eager fallback first.
- Quantized weights and quantized KV cache.
- Tensor parallel/multi-GPU sharding.
- Long-context extrapolation beyond learned `n_positions=1024` unless a checkpoint explicitly changes the learned table.

## 15. Final implementation checklist

- [ ] Parse GPT-2 config and representative Hub generation/tokenizer metadata.
- [ ] Load `wte`, `wpe`, LayerNorm, Conv1D, and tied LM head weights with explicit layout metadata.
- [ ] Implement Conv1D-as-GEMM with stored `[in_features, out_features]` handling.
- [ ] Implement learned absolute position embedding with decode offset from cache length.
- [ ] Implement `gelu_new` tanh approximation.
- [ ] Implement GPT-2 decoder block eager parity.
- [ ] Implement causal MHA prefill with mask canonicalization.
- [ ] Implement artifact-visible K/V cache `[B, heads, seq, head_dim]` per layer.
- [ ] Implement decode cache append and fused decode-attention candidate.
- [ ] Implement tied LM head and `logits_to_keep` last-token lowering.
- [ ] Add byte-level BPE tokenizer integration or define CPU pipeline boundary.
- [ ] Add primitive parity tests for Conv1D, GELU_new, LayerNorm, attention masks, and cache append.
- [ ] Add one-block, full-prefill, and multi-step decode parity tests against Transformers.
- [ ] Benchmark Conv1D GEMMs, prefill attention, decode attention, KV memory, and LM head slicing.
