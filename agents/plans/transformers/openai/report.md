# OpenAI GPT (`openai`) Transformers Family Audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: openai-community/openai-gpt
Config source: Hugging Face config.json snapshots saved beside this report
Source files inspected:
  transformers/src/transformers/models/openai/configuration_openai.py
  transformers/src/transformers/models/openai/modeling_openai.py
  transformers/src/transformers/models/openai/tokenization_openai.py
  transformers/src/transformers/pytorch_utils.py (Conv1D weight layout)
Any missing files or assumptions:
  No tokenization_openai_fast.py exists in this checkout.
  No processor/image/audio files apply.
  No gated model links were needed; representative configs were public.
```

Primary runtime target for this report: `OpenAIGPTLMHeadModel` causal language model, inference-only. `OpenAIGPTModel` is required as the base. `OpenAIGPTDoubleHeadsModel` and `OpenAIGPTForSequenceClassification` are optional/deferred heads.

Primary source URLs:

- [modeling_openai.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/openai/modeling_openai.py)
- [configuration_openai.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/openai/configuration_openai.py)
- [tokenization_openai.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/openai/tokenization_openai.py)
- [openai-community/openai-gpt config](https://huggingface.co/openai-community/openai-gpt/blob/main/config.json)

Local snapshots:

- `config_openai-community_openai-gpt.json`
- `config_CoffeeAddict93_gpt1-modest-proposal.json`
- `config_instruct-generalize_gpt-1.json`
- `config_hf-tiny_OpenAIGPTLMHeadModel.json`
- `config_hf-tiny_OpenAIGPTForSequenceClassification.json`

## 2. High-level architecture

Legacy GPT-1-style text-only decoder stack with learned token embeddings, learned absolute positional embeddings, optional token-type embeddings using the same token embedding table, post-attention LayerNorm, post-MLP LayerNorm, and tied LM head.

```text
BPE/lowercase tokenization -> input_ids/attention_mask/token_type_ids
  -> token + position + optional token-type embeddings
  -> repeated causal self-attention + MLP blocks
  -> tied LM projection
  -> logits/sampling
```

There is no encoder branch, no cross-attention, no RoPE/ALiBi, no MoE, no multimodal projector, and no source-implemented KV cache. Generation through Transformers recomputes the full prefix because `prepare_inputs_for_generation` just forwards `input_ids` and kwargs; the model forward does not accept or return `past_key_values`.

## 3. Important config dimensions

Source defaults come from `OpenAIGPTConfig`; checkpoint values come from saved `config.json` files.

| Field | Default/source | Primary checkpoint | Notes |
|---|---:|---:|---|
| `model_type` | `openai-gpt` | `openai-gpt` | Config class model type. |
| `vocab_size` | 40478 | 40478 | Token/LM vocab. |
| `n_positions` | 512 | 512 | Learned position table and causal mask size. |
| `n_embd` / hidden | 768 | 768 | Residual width. |
| `n_layer` | 12 | 12 | Decoder block count. |
| `n_head` | 12 | 12 | MHA heads. |
| `head_dim` | inferred 64 | inferred 64 | `n_embd // n_head`; source requires divisibility. |
| `intermediate_size` | inferred 3072 | inferred 3072 | MLP uses `4 * n_embd`. |
| activation | `gelu` | `gelu` | Source maps `gelu` to `gelu_new`, not exact GELU. |
| attention type | causal dense MHA | causal dense MHA | Static lower-triangular mask plus optional padding mask. |
| cache support | none in source | none in source | No `use_cache` path in model forward. |
| dtype | source agnostic | no dtype in primary config | Community/debug configs may set `torch_dtype=float32`. |
| tied embeddings | `True` default | implicit via class | LM head tied to token embeddings by `_tied_weights_keys`. |

Representative checkpoint sweep:

| Model | Source type | Architecture | Vocab | Layers | Heads | Hidden | Positions | Operator-significant variation |
|---|---|---|---:|---:|---:|---:|---:|---|
| `openai-community/openai-gpt` | canonical public | LM head | 40478 | 12 | 12 | 768 | 512 | Main target. Config includes legacy `n_ctx`, `n_special`, `predict_special_tokens`; current source does not read those fields. |
| `CoffeeAddict93/gpt1-modest-proposal` | community fine-tune | LM head | 40478 | 12 | 12 | 768 | 512 | Same operator shape; `torch_dtype=float32` metadata. |
| `instruct-generalize/gpt-1` | community fine-tune | LM head | 40478 | 12 | 12 | 768 | 512 | Same operator shape; `torch_dtype=float32` metadata. |
| `hf-tiny-model-private/tiny-random-OpenAIGPTLMHeadModel` | public debug | LM head | 1407 | 5 | 4 | 32 | 512 | Useful tiny parity shape; `pad_token_id=1406`, `is_decoder=true` is config metadata only for this source. |
| `hf-tiny-model-private/tiny-random-OpenAIGPTForSequenceClassification` | public debug | sequence classification | 1407 | 5 | 4 | 32 | 512 | Head changes pooling/indexing and output shape, not transformer block ops. |

## 3a. Family variation traps

- `Conv1D` is not ordinary `nn.Linear` storage: weights are `[in_features, out_features]`, and forward uses `torch.addmm(bias, x_flat, weight)`. Weight import should either keep this orientation or transpose into DinoML's standard GEMM layout.
- `afn="gelu"` selects `gelu_new` in source. The config docs say `"gelu"`, but parity should use the tanh approximation implementation used by Transformers here.
- Attention is post-norm by block sublayer: `ln_1(x + attn(x))`, then `ln_2(n + mlp(n))`. Do not translate as modern pre-norm GPT-2.
- No KV cache exists in the inspected source. Autoregressive decode parity should initially be full-prefix recompute unless DinoML adds a guarded graph rewrite that changes the runtime contract.
- `token_type_ids`, if supplied, are embedded through `tokens_embed`, not a separate segment table.
- `position_ids` default to `0..seq_len-1` and do not account for cached prefix length.
- `attention_mask` is `[B,S] -> [B,1,1,S]` and converted to `(1-mask) * torch.finfo(dtype).min`; the causal mask uses a separate lower-triangular buffer and `-1e4`.
- `n_ctx`, `n_special`, `predict_special_tokens`, and `is_decoder` appear in some configs but are not read by this modeling source.
- `OpenAIGPTForSequenceClassification` rejects batch size > 1 when `pad_token_id` is absent; this is head-specific admission behavior.
- Layout translation is not relevant for image/video axes. Preserve token layout `[B,S,H]`; attention uses explicit `view`/`permute` contracts around heads.

## 4. Operator coverage checklist

Tensor/layout ops:

- Int token inputs, optional `attention_mask`, optional `token_type_ids`, optional `position_ids`.
- Embedding lookup for token IDs `[B,S] -> [B,S,H]`.
- Embedding lookup for learned positions `[1,S] or [B,S] -> [*,S,H]`.
- Optional token-type embedding lookup using token embedding table.
- Elementwise add for embeddings and residuals.
- `view`/flatten input IDs to `[-1,S]`; restore hidden state to original leading dims plus hidden.
- Split QKV packed projection output along last dim in order `[query, key, value]`.
- Reshape/permute heads:
  - Q/V: `[B,S,H] -> [B,heads,S,head_dim]`
  - K: `[B,S,H] -> [B,heads,head_dim,S]`
  - merge: `[B,heads,S,head_dim] -> [B,S,H]`
- Last-token or tensor-index logits slicing for `logits_to_keep`.
- Optional hidden-state/attention tuple accumulation can be deferred for first runtime.

Neural network primitives:

- `Conv1D(768 -> 2304)` packed QKV with bias for primary config.
- `Conv1D(768 -> 768)` attention output projection with bias.
- LayerNorm over hidden dim with epsilon `1e-5`.
- MLP:
  - `Conv1D(768 -> 3072)` with bias.
  - `gelu_new` activation for primary config.
  - `Conv1D(3072 -> 768)` with bias.
- LM head `Linear(768 -> 40478, bias=False)` tied to token embeddings.
- Dropout modules are present but should be identity in inference.

Attention primitives:

- Dense causal self-attention MHA, no GQA/MQA.
- QK matmul: `[B,heads,S,head_dim] x [B,heads,head_dim,S] -> [B,heads,S,S]`.
- Score scale by `1 / sqrt(head_dim)`.
- Lower-triangular mask crop to runtime sequence length.
- Optional padding mask add.
- Softmax on last dimension.
- Attention-value matmul: `[B,heads,S,S] x [B,heads,S,head_dim]`.

Position/relative-bias ops:

- Learned absolute position embedding only. No RoPE, ALiBi, relative position bias, or rotary cache.

Generation/cache ops:

- Full-prefix prefill/logits path required.
- Source has no per-layer KV cache ABI. A future optimized decode cache would be a DinoML extension, not direct source parity.

Optional head ops:

- Double-head multiple choice: sequence summary with `cls_index` gather and optional projection/tanh/dropouts.
- Sequence classification: per-token linear score, last non-pad index by `argmax(token_indices * non_pad_mask)`, gather pooled logits.

## 5. Layer/block breakdown

Primary config: `B=batch`, `S<=512`, `H=768`, `heads=12`, `D=64`, `I=3072`, `V=40478`.

Decoder block, repeated 12 times:

```text
x: [B,S,H]
qkv = Conv1D_H_to_3H(x) + bias              # weight [H,3H], split [Q,K,V]
q = view(q, [B,S,heads,D]).permute(0,2,1,3)
k = view(k, [B,S,heads,D]).permute(0,2,3,1)
v = view(v, [B,S,heads,D]).permute(0,2,1,3)
scores = matmul(q, k) / sqrt(D)
scores = scores * causal_mask + -1e4 * (1 - causal_mask)
scores = scores + optional_padding_mask
probs = softmax(scores, dim=-1)
context = matmul(probs, v)
attn = merge_heads(context)
attn = Conv1D_H_to_H(attn) + bias
n = LayerNorm(x + attn)
m = Conv1D_H_to_4H(n) + bias
m = gelu_new(m)
m = Conv1D_4H_to_H(m) + bias
h = LayerNorm(n + m)
```

LM head:

```text
logits = hidden[:, slice_indices, :] @ tokens_embed.weight.T
```

The source computes `slice_indices = slice(-logits_to_keep, None)` for integer `logits_to_keep`; with `logits_to_keep=0`, Python `-0` yields `0`, so all logits are kept.

## 6. Attention requirements

- Type: causal self-attention only.
- Heads: MHA, `num_key_value_heads == num_attention_heads`; no GQA/MQA.
- Widths: Q, K, V all `H`; per-head Q/K/V width `D=H/heads`.
- Sequence shape: square self-attention over current full sequence. No rectangular cross-attention.
- Masking:
  - Built-in lower triangular causal mask buffer `[1,1,n_positions,n_positions]`, cropped to `[1,1,S,S]`.
  - Padding mask, if provided, starts `[B,S]`, expands to `[B,1,1,S]`, casts to model dtype, and uses dtype min for masked positions.
  - Causal mask uses multiplicative keep plus `-1e4`; padding mask is additive after causal mask.
- Packed/varlen: none in source.
- Sliding/local/block sparse: none.
- Position interaction: learned absolute embeddings are added before QKV projection. No attention-time positional transform.
- KV cache: none. Cached keys/values are not accepted or returned.
- FlashAttention/SDPA compatibility: mathematically compatible with dense causal MHA plus padding mask, but source mask ordering and `-1e4` causal fill should be parity-tested if using fused attention.

## 7. Position encoding and custom math

Position encoding is a learned embedding table:

```python
if position_ids is None:
    position_ids = position_ids_buffer[None, :seq_len]
hidden = token_embed(input_ids) + position_embed(position_ids) + token_type_embed_or_0
```

No dynamic RoPE tables or position-dependent attention kernels are required. `position_ids` can be caller-supplied; default IDs depend only on runtime sequence length.

Custom activation parity:

```python
def openai_gpt_activation(x, afn):
    # Source ACT_FNS maps "gelu" to gelu_new.
    if afn == "gelu":
        return gelu_new_tanh_approx(x)
    if afn in ("silu", "swish"):
        return x * sigmoid(x)
    if afn == "relu":
        return max(x, 0)
```

## 8. Preprocessing and input packing

Tokenizer source uses `tokenizers` BPE with:

- `vocab.json`, `merges.txt`, or `tokenizer.json`.
- BERT normalizer with lowercasing.
- BERT pre-tokenizer.
- BPE decoder suffix `</w>`.
- Model input names: `input_ids`, `attention_mask`.

CPU/data pipeline should own tokenization, lowercasing, BPE, padding, and attention mask construction. GPU/runtime graph consumes:

- `input_ids`: integer `[B,S]`.
- `attention_mask`: optional numeric/bool-like `[B,S]`, with 1 for keep and 0 for mask.
- `token_type_ids`: optional integer `[B,S]`; source uses token embedding table for these.
- `position_ids`: optional integer `[1,S]` or `[B,S]`; otherwise generated from static buffer.

There are no modality placeholders, scatter embedding stitches, `cu_seqlens`, image grids, audio features, or postprocessing requirements for the primary target.

## 9. Graph rewrite / lowering opportunities

### Rewrite: Conv1D to GEMM/Linear

Source pattern:

```text
y = addmm(bias, x.reshape(-1, in_features), weight[in_features, out_features])
y = y.reshape(*x.shape[:-1], out_features)
```

Replacement:

```text
FlattenLeading -> GEMM_RRR/GEMM_RCR -> BiasAdd -> RestoreLeading
```

Preconditions:

- Weight is a Transformers `Conv1D` parameter with storage `[in,out]`.
- Input is contiguous or lowered with an explicit contiguous/stride guard.
- Bias exists and has shape `[out]`.

Weight transform:

```python
# If DinoML GEMM expects Linear-style [out,in], transpose once at import.
linear_weight = conv1d_weight.T
```

Failure cases: treating `Conv1D.weight` as `[out,in]` silently transposes every projection.

Parity test sketch: compare QKV projection, attention output projection, and both MLP projections on random `[B,S,H]` tensors.

### Rewrite: Packed QKV projection

Source pattern:

```text
qkv = Conv1D(H -> 3H)(x)
q, k, v = split(qkv, H, dim=-1)
```

Replacement:

```text
single GEMM with packed output -> three view/slice aliases -> head reshape
```

Preconditions:

- Split order is exactly Q, K, V.
- Output width exactly `3 * H`.
- All heads share `D = H / n_head`.

Failure cases: do not reuse for model families with interleaved per-head QKV packing or separate K/V widths.

### Rewrite: Inference dropout removal

Source pattern: dropout after embeddings, attention probabilities, attention projection, and MLP projection.

Replacement: identity in eval/inference mode.

Preconditions: model is in inference mode and no training parity is requested.

### Rewrite: Last-token-only logits

Source pattern:

```text
logits = lm_head(hidden_states[:, slice_indices, :])
```

Replacement: if generation only needs the final token, compute `hidden[:, -1:, :] @ W.T`.

Preconditions:

- `labels is None`.
- `logits_to_keep == 1` or caller only consumes final-step logits.
- No API path requests full logits.

Failure cases: primary config task-specific params use generation but source default `logits_to_keep=0` returns all logits.

### Rewrite: Full-prefix decode to KV cache

This is an optimization extension, not source parity. It would require adding a DinoML cache ABI and proving equivalence under learned absolute positions and padding masks.

Preconditions:

- Decode appends tokens monotonically.
- Position IDs are contiguous or externally supplied in a cache-compatible form.
- Cached K values are stored after the K head permute, shape `[B,heads,D,T]`; cached V values `[B,heads,T,D]`.
- Padding mask semantics for prior tokens are fixed.

Failure cases: arbitrary caller-supplied `position_ids`, arbitrary `inputs_embeds`, or non-monotonic generation should fall back to full-prefix.

## 10. Kernel fusion candidates

Highest priority:

- LayerNorm + residual patterns for post-attention and post-MLP blocks. This model is post-norm, so fusions differ from modern pre-norm GPTs.
- Packed QKV GEMM + split/head reshape. This is the largest repeated projection and has simple packed storage.
- Dense causal attention prefill. S=512 is small enough for a straightforward fused attention path, but exact mask fill ordering needs parity checks.
- MLP GEMM + `gelu_new` + GEMM, especially bias activation fusion.

Medium priority:

- Token + position + token-type embedding add fusion.
- Attention output projection + residual + LayerNorm.
- MLP output projection + residual + LayerNorm.
- Last-token-only LM head for generation.

Lower priority:

- Optional sequence summary/classification pooling kernels.
- Output attentions/hidden state materialization.
- Full-prefix-to-KV-cache rewrite. Useful for decode performance but not source-native.

## 11. Runtime staging plan

Stage 1: config and weight loading.

- Parse `OpenAIGPTConfig`.
- Load token embeddings, position embeddings, all `Conv1D` weights with correct orientation, LayerNorm params, and tied LM head.
- Reject unsupported `afn`, `summary_type="attn"`, and `n_embd % n_head != 0`.

Stage 2: one-block parity.

- Run embedding add, one attention block, one MLP block against Transformers.
- Use `eval()` so dropout is identity.

Stage 3: full base model parity.

- Run all blocks and return last hidden state.
- Support optional `attention_mask`, `token_type_ids`, and `position_ids`.

Stage 4: LM head prefill parity.

- Add tied LM projection.
- Validate full logits and `logits_to_keep` slices.

Stage 5: generation controller integration.

- Start with full-prefix recompute.
- Later add guarded DinoML KV-cache extension as an optimized alternate path.

Stage 6: optional heads.

- Add double-head multiple choice and sequence classification only after base/LM parity is stable.

Stage 7: fusions/performance.

- Introduce packed QKV, fused attention, residual LayerNorm, MLP activation fusions, and last-token logits.

## 12. Parity and validation plan

- Random tensor tests for `Conv1D` import orientation across QKV, output projection, and MLP projections.
- Activation parity for `afn in {"gelu", "relu", "silu", "swish"}`; primary checkpoint only requires `gelu`.
- Attention unit tests:
  - no padding mask.
  - padding mask with right padding.
  - explicit causal mask parity for `S < n_positions`.
  - fp32 and fp16 mask behavior if reduced precision is enabled.
- Single-block parity with fixed random weights and inputs, tolerance: fp32 `1e-5`/`1e-4`, fp16 `1e-2` depending on attention backend.
- Full 12-layer `openai-community/openai-gpt` hidden-state parity on short prompts.
- LM logits parity for full logits and `logits_to_keep=1`.
- Tokenization smoke test in CPU pipeline for lowercasing/BPE, but keep tokenizer outside GPU graph.
- Optional head tests:
  - multiple-choice `mc_token_ids` gather.
  - sequence classification with and without `pad_token_id`; batch > 1 without pad should reject.

## 13. Performance probes

- Tokenization throughput separately from runtime.
- Prefill throughput over `S={16,64,128,256,512}` and `B={1,4,16}`.
- Decode full-prefix recompute tokens/sec over growing prefix length.
- Optional DinoML KV-cache decode tokens/sec if the extension is added.
- Attention backend comparison: unfused matmul/softmax/matmul vs fused causal attention.
- MLP GEMM throughput and `gelu_new` fusion benefit.
- LM head cost for full logits vs last-token-only logits.
- Memory probe for activations and optional attention-output materialization.

## 14. Skip/defer list

- Training losses and dropout randomness.
- Gradient checkpointing and backward pass.
- Beam search/sampling policy beyond consuming logits.
- Native source KV cache parity, because no such source path exists.
- Optional output attentions and all hidden states for first optimized path.
- Multiple-choice and sequence-classification heads for the initial LM target.
- General `inputs_embeds` generation with arbitrary external position IDs for optimized cached decode.
- Quantized/packed weights; configs inspected use dense weights.
- Multi-GPU/tensor parallelism.

## 15. Final implementation checklist

- [ ] Parse `OpenAIGPTConfig` and reject unsupported config combinations.
- [ ] Load embeddings, position table, LayerNorm weights, and `Conv1D` weights with `[in,out]` storage handling.
- [ ] Preserve LM head/token embedding tied-weight alias.
- [ ] Implement embedding add with optional token-type embeddings.
- [ ] Implement dense causal MHA with source mask order and scale.
- [ ] Implement `gelu_new` activation for `afn="gelu"`.
- [ ] Implement post-attention and post-MLP LayerNorm residual order.
- [ ] Implement full-prefix `OpenAIGPTLMHeadModel` logits.
- [ ] Add `logits_to_keep` lowering and last-token optimization guard.
- [ ] Add one-block, full-model, and LM-logits parity tests.
- [ ] Add optional attention-mask and token-type parity tests.
- [ ] Benchmark prefill, full-prefix decode, attention backend, MLP, and LM head.
- [ ] Defer KV-cache optimization until full-prefix parity is stable.
