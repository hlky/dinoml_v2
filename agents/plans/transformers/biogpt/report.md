# BioGPT Transformers Audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: microsoft/biogpt family, with large and PubMedQA variants
Config source: HF config.json files plus BioGptConfig source defaults
Source files inspected:
  transformers/src/transformers/models/biogpt/modeling_biogpt.py
  transformers/src/transformers/models/biogpt/modular_biogpt.py
  transformers/src/transformers/models/biogpt/configuration_biogpt.py
  transformers/src/transformers/models/biogpt/tokenization_biogpt.py
  transformers/src/transformers/masking_utils.py
  transformers/src/transformers/cache_utils.py
Any missing files or assumptions:
  BioGPT has no processor/image/audio files. Current modeling_biogpt.py is generated
  from modular_biogpt.py; future source edits should target modular_biogpt.py, while
  this report uses generated modeling_biogpt.py for concrete lowering behavior.
```

Pinned upstream source URLs:

- [modeling_biogpt.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/biogpt/modeling_biogpt.py)
- [modular_biogpt.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/biogpt/modular_biogpt.py)
- [configuration_biogpt.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/biogpt/configuration_biogpt.py)
- [tokenization_biogpt.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/biogpt/tokenization_biogpt.py)

Representative checkpoint config snapshots are recorded in `_sources/config_sweep.json`.
Open HF configs inspected:

- [microsoft/biogpt](https://huggingface.co/microsoft/biogpt)
- [microsoft/BioGPT-Large](https://huggingface.co/microsoft/BioGPT-Large)
- [microsoft/BioGPT-Large-PubMedQA](https://huggingface.co/microsoft/BioGPT-Large-PubMedQA)

The following Microsoft task fine-tunes returned HTTP 401 for `config.json` and the model API without credentials: [BioGPT-Large-BC5CDR](https://huggingface.co/microsoft/BioGPT-Large-BC5CDR), [BioGPT-Large-KD-DTI](https://huggingface.co/microsoft/BioGPT-Large-KD-DTI), and [BioGPT-Large-DDI](https://huggingface.co/microsoft/BioGPT-Large-DDI). Access with an authorized HF token would resolve the missing checkpoint configs.

Primary runtime target for this report: `BioGptForCausalLM` causal language modeling, prefill plus decode. `BioGptModel` is required as the body. Token and sequence classification heads are optional/deferred for a first LM target.

## 2. High-level architecture

BioGPT is a text-only decoder-only Transformer. It uses learned token embeddings, learned absolute positional embeddings, stacked pre-norm decoder blocks, a final LayerNorm, and a tied LM projection.

```text
Moses+BPE tokenization -> input_ids/attention_mask
-> scaled token embedding + learned absolute position embedding
-> N x decoder self-attention/MLP block
-> final LayerNorm
-> tied LM projection
-> logits/sampling
```

Stage decomposition:

- CPU/data pipeline: Moses tokenizer, BPE merges, special-token construction, padding, attention mask.
- GPU/runtime prefill: embeddings, causal mask, all decoder blocks, final norm, logits.
- GPU/runtime decode: one or more new tokens, append/update per-layer KV cache, last-token logits when `logits_to_keep=1`.
- Independently optimizable stages: tokenizer/vocab loading, decoder block parity, attention backend, last-token-only LM head.

## 3. Important config dimensions

Source defaults from `BioGptConfig`:

| Field | Default | Runtime meaning |
|---|---:|---|
| `vocab_size` | 42384 | token embedding rows and LM head rows |
| `hidden_size` | 1024 | model width |
| `num_hidden_layers` | 24 | decoder block count |
| `num_attention_heads` | 16 | MHA query/key/value heads |
| `head_dim` | 64 inferred | `hidden_size // num_attention_heads`; source rejects non-divisible configs |
| `intermediate_size` | 4096 | MLP `fc1/fc2` width |
| `hidden_act` | `gelu` | ungated FFN activation |
| `max_position_embeddings` | 1024 | learned absolute position table, plus source offset of 2 |
| `layer_norm_eps` | `1e-12` | all LayerNorms |
| `scale_embedding` | `true` | multiply token embeddings by `sqrt(hidden_size)` |
| `use_cache` | `true` | DynamicCache KV cache by default |
| `tie_word_embeddings` | `true` default | LM head aliases token embedding weight |

Checkpoint sweep:

| Model | Source | Layers | Hidden | Heads | Head dim | MLP | Max pos | Vocab | Dtype | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| `microsoft/biogpt` | `config.json` | 24 | 1024 | 16 | 64 | 4096 | 1024 | 42384 | omitted | base causal LM |
| `microsoft/BioGPT-Large` | `config.json` | 48 | 1600 | 25 | 64 | 6400 | 2048 | 57717 | `float32` | larger PubMed model |
| `microsoft/BioGPT-Large-PubMedQA` | `config.json` | 48 | 1600 | 25 | 64 | 6400 | 2048 | 57726 | `float32` | QA fine-tune, slightly different vocab |
| task fine-tunes listed above | HF HTTP 401 | unknown | unknown | unknown | unknown | unknown | unknown | unknown | unknown | needs authorized access |

## 3a. Family variation traps

- `vocab_size` changes materially: 42384 for base, 57717 for large, 57726 for PubMedQA. Tokenizer/model weights must be paired by checkpoint; do not reuse a base vocab with large weights.
- `num_attention_heads=25` in large models is unusual but valid because `1600 / 25 = 64`. Do not assume head count is a power of two.
- There is no `num_key_value_heads`: BioGPT is full MHA, not GQA/MQA.
- Position encoding is learned absolute positions with a hard offset of 2, not RoPE/ALiBi. Long-context extension is not a RoPE-theta problem.
- Token embeddings are scaled by `sqrt(hidden_size)` when `scale_embedding=True`; GPT-2 style imports may miss this.
- Q/K/V are three separate `nn.Linear` modules with PyTorch weight layout `[out_features, in_features]`; this differs from GPT-2 `Conv1D` packed projection layout.
- Versus OPT-style decoders, BioGPT is close in learned-position/cache shape, but checkpoint coupling differs: BioGPT uses a Moses+BPE biomedical tokenizer, separate BioGPT vocab sizes, and the current source is generated from the BioGPT modular file rather than reusing OPT modules directly.
- MLP is ungated `Linear -> GELU -> Linear`; no SwiGLU/GEGLU.
- All Q/K/V/O and MLP linear projections include bias. LM head has no bias.
- LM head weight is tied to token embedding through `_tied_weights_keys`; lowering must preserve one logical parameter alias.
- The tokenizer builds a single sequence as `</s> X`, using `sep_token_id`/`eos_token_id` at the front rather than a BOS token.
- `token_type_ids` is accepted by token classification but ignored by the BioGPT body.
- No NCHW/NHWC layout translation applies; this is text-only `[batch, sequence, hidden]`.

## 4. Operator coverage checklist

Tensor/layout ops:

- integer token IDs and attention mask inputs, shape `[B, S]`
- embedding lookup for tokens `[vocab, hidden]`
- embedding lookup for positions `[max_position_embeddings + 2, hidden]`
- scalar multiply of token embeddings by `sqrt(hidden_size)` when enabled
- broadcast/add token and position embeddings
- reshape/view `[B, S, hidden] -> [B, S, heads, head_dim]`
- transpose to attention layout `[B, heads, S, head_dim]`
- contiguous/reshape back to `[B, S, hidden]`
- residual adds and dropout elision for inference
- logits slicing via `logits_to_keep` for last-token or selected-token logits

Neural network primitives:

- `LayerNorm(hidden, eps=1e-12)` before attention, before MLP, and after all layers
- biased `Linear(hidden -> hidden)` for q, k, v, out
- biased `Linear(hidden -> intermediate)`, GELU, biased `Linear(intermediate -> hidden)`
- bias-free tied `Linear(hidden -> vocab)` LM projection

Attention primitives:

- causal self-attention only
- MHA with `heads = num_attention_heads`, `head_dim = hidden / heads`
- score matmul `q @ k.transpose(-2, -1) * head_dim**-0.5`
- additive causal/padding mask before softmax
- softmax over source sequence dimension
- value matmul and output projection
- optional HF backend dispatch through eager, SDPA, FlashAttention, or FlexAttention interfaces

Position/cache/tokenizer ops:

- learned absolute position IDs with offset 2
- `DynamicCache` per-layer key/value append along sequence dimension
- cache reorder/select for beam or batch changes if generation uses beams
- Moses tokenization plus BPE merge table in CPU pipeline

Optional heads:

- token classification: dropout plus `Linear(hidden -> num_labels)`, optional active-token masking for loss only
- sequence classification: bias-free `Linear(hidden -> num_labels)`, gather last non-pad token logits

## 5. Layer/block breakdown

For base `microsoft/biogpt`, repeated 24 times with `H=1024`, `I=4096`, `A=16`, `D=64`. For large variants, repeated 48 times with `H=1600`, `I=6400`, `A=25`, `D=64`.

```text
input_ids [B,S]
attention_mask [B,S_total]
tok = Embedding(vocab,H,padding_idx=1)(input_ids) * sqrt(H)
pos = LearnedPosition(max_pos+2,H)(position_ids + 2)
x = dropout(tok + pos)               # dropout disabled for inference

Decoder block:
  residual = x
  y = LayerNorm(H, eps=1e-12)(x)
  q = Linear(H -> H, bias=True)(y).view(B,S,A,D).transpose(1,2)
  k = Linear(H -> H, bias=True)(y).view(B,S,A,D).transpose(1,2)
  v = Linear(H -> H, bias=True)(y).view(B,S,A,D).transpose(1,2)
  k,v = cache.update(k,v, layer_idx)  # if use_cache
  a = Attention(q,k,v, causal_mask, scale=D**-0.5)
  a = Linear(H -> H, bias=True)(a)
  x = residual + a

  residual = x
  y = LayerNorm(H, eps=1e-12)(x)
  y = Linear(H -> I, bias=True)(y)
  y = GELU(y)
  y = Linear(I -> H, bias=True)(y)
  x = residual + y

x = final LayerNorm(H, eps=1e-12)(x)
logits = tied Linear(H -> vocab, bias=False)(x[:, selected_positions, :])
```

Training-only `layerdrop`, activation dropout, hidden dropout, losses, and gradient checkpointing can be ignored for inference parity.

## 6. Attention requirements

BioGPT requires autoregressive causal self-attention. There is no encoder, no cross-attention in the model body, no sliding window, no local attention, no ALiBi, no RoPE, and no packed/varlen sequence metadata in the BioGPT source.

Attention shape contract:

- Input hidden states: `[B, S_q, H]`
- Q/K/V projected shape before transpose: `[B, S, heads, head_dim]`
- Q/K/V attention shape: `[B, heads, S, head_dim]`
- Dynamic cache stores per-layer keys and values as `[B, heads, S_cached, head_dim]`
- After cache update, returned keys/values cover `[B, heads, S_cached + S_q, head_dim]`
- Attention output after transpose/reshape: `[B, S_q, H]`

Masking:

- If `attention_mask` is absent, source creates ones of shape `[B, past_length + S_q]`.
- The model calls `create_causal_mask(config, inputs_embeds, attention_mask, past_key_values)`.
- Eager attention receives an additive mask and adds it to attention scores before softmax.
- SDPA/Flash/Flex paths may skip materializing a mask when safe, but semantic parity is causal masking plus optional padding masking.

Cache layout and update:

- `BioGptModel` creates `DynamicCache(config)` when `use_cache=True` and no cache is provided.
- `DynamicLayer.update` appends new keys and values with `torch.cat(..., dim=-2)`.
- Cached keys are stored after projection and after reshape/transpose. There is no position-encoding transform on Q/K, so there is no pre/post-RoPE distinction.
- Beam reorder uses batch-dimension `index_select(0, beam_idx)`.

FlashAttention/SDPA compatibility:

- Source advertises `_supports_flash_attn`, `_supports_sdpa`, and `_supports_flex_attn`.
- Fused attention must preserve scale `head_dim**-0.5`, mask-before-softmax order, and output layout.

## 7. Position encoding and custom math

BioGPT uses learned absolute positional embeddings with an offset of 2.

```python
def biogpt_positions(attention_mask, past_len, position_ids=None):
    if position_ids is None:
        # Used by the embedding module if caller does not pass position_ids.
        pos = attention_mask.cumsum(dim=1)
        pos = (pos * attention_mask - 1).long()
        pos = pos[:, past_len:]
    else:
        # Current BioGptModel constructs arange(seq_len) + past_len and passes it.
        pos = position_ids
    return learned_position_embedding(pos + 2)
```

What can be precomputed:

- Position embedding weights are constants.
- For fixed max decode length, position IDs can be generated by sequence offset.

What depends on runtime inputs:

- `past_key_values_length` changes decode position IDs.
- If a caller invokes the embedding layer without explicit `position_ids`, padding-aware cumsum positions depend on `attention_mask`.

There is no RoPE, M-RoPE, ALiBi, relative position bias, or attention score softcap.

## 8. Preprocessing and input packing

BioGPT uses `BioGptTokenizer`, a Python tokenizer with `sacremoses` Moses tokenization followed by BPE over checkpoint-specific `vocab.json` and `merges.txt`.

Runtime graph inputs:

- `input_ids`: `[B, S]`, integer token IDs
- `attention_mask`: `[B, S_total]`, 1 for tokens to attend to, 0 for padding
- optional `position_ids`: `[1 or B, S]`; source default is contiguous arange plus cache length
- optional `inputs_embeds`: `[B, S, H]`, mutually exclusive with `input_ids`

Special-token behavior from tokenizer source:

- Single sequence is built as `</s> X`.
- Pair sequence is built as `</s> A </s> B`.
- `bos_token="<s>"`, `eos_token="</s>"`, `sep_token="</s>"`, `pad_token="<pad>"`, `unk_token="<unk>"`.

No image/audio/video preprocessing, placeholder token expansion, scatter stitching, token type embeddings, or packed sequence descriptors are required for the LM target.

## 9. Graph rewrite / lowering opportunities

### Rewrite: separate Q/K/V Linear -> packed QKV GEMM

Source pattern:

```text
q = Linear(H,H)(x)
k = Linear(H,H)(x)
v = Linear(H,H)(x)
```

Replacement:

```text
qkv = Linear(H, 3H)(x)
split qkv as [q, k, v] along last dimension
```

Preconditions:

- same input tensor and dtype
- all three projections have bias or all bias handling is represented
- packed weight rows are concatenated in source order `[q_proj.weight; k_proj.weight; v_proj.weight]`
- packed bias is `[q_bias; k_bias; v_bias]`

Failure cases:

- partial quantized loading that stores projections separately and cannot expose a packed logical weight
- debugging/export modes that require original module names

Parity test sketch:

- Compare q/k/v tensors before attention for random hidden states and real checkpoint weights.

### Rewrite: pre-norm residual attention block fusion

Source pattern:

```text
LayerNorm -> QKV -> causal attention -> out_proj -> residual add
```

Replacement:

```text
LayerNorm kernel + packed QKV GEMM + fused attention + output GEMM + residual add
```

Preconditions:

- inference mode
- dropout disabled
- dense causal mask or backend-supported padding mask
- no requested attention weights output for the optimized path

Failure cases:

- `output_attentions=True` requiring materialized attention probabilities
- unusual attention implementation with incompatible mask representation

### Rewrite: MLP GELU block fusion

Source pattern:

```text
LayerNorm -> Linear(H,I) -> GELU -> Linear(I,H) -> residual add
```

Replacement:

```text
LayerNorm + GEMM, fused GELU, GEMM, residual epilogue
```

Preconditions:

- `hidden_act == "gelu"`
- no activation dropout
- inference mode

### Rewrite: last-token-only logits

Source pattern:

```text
logits = output_projection(hidden_states[:, slice_indices, :])
```

Replacement:

```text
gather selected hidden states -> tied vocab GEMM only for selected positions
```

Preconditions:

- generation caller requests `logits_to_keep=1` or explicit token indices
- loss is not being computed
- hidden-state outputs do not require full logits

Weight transform:

- none; preserve tied `output_projection.weight is embed_tokens.weight`.

## 10. Kernel fusion candidates

Highest priority:

- LayerNorm with `eps=1e-12`: appears three times per block path plus final norm.
- Packed QKV GEMM: removes three separate GEMM launches and enables attention-friendly layout.
- Causal MHA prefill/decode with KV cache: core runtime cost; full MHA with `head_dim=64`.
- Last-token-only tied LM head: vocab GEMM is large, especially `vocab_size` 57k in large checkpoints.

Medium priority:

- GELU MLP fusion: large `H -> 4H -> H` feed-forward path dominates FLOPs.
- Attention output projection plus residual add epilogue.
- Token + position embedding add and scale: cheap but useful for launch count reduction.

Lower priority:

- Optional classifier heads.
- Beam cache reorder/select kernels unless beam search is in first scope.
- Materialized attention probabilities for diagnostics.

## 11. Runtime staging plan

1. Parse `BioGptConfig` and tokenizer metadata; reject unsupported non-MHA or non-GELU variants if encountered.
2. Load base checkpoint weights and preserve tied embedding/LM-head aliasing.
3. Implement one decoder block parity without cache using eager attention and dense masks.
4. Add full prefill parity for `BioGptForCausalLM`.
5. Add `DynamicCache` decode parity with per-layer `[B, heads, S, head_dim]` K/V buffers.
6. Add optimized attention backend for prefill and decode.
7. Add packed QKV, MLP, residual, and last-token logits rewrites.
8. Add large and PubMedQA checkpoint sweeps; add gated task fine-tunes once credentials are available.

Initial stubs allowed:

- Dropout, layerdrop, training losses, gradient checkpointing.
- Token/sequence classification heads.
- Beam search and generation processors beyond returning logits.

## 12. Parity and validation plan

- Config parser tests for base and large dimensions, including large `num_attention_heads=25`.
- Tokenizer coupling test: confirm vocab size and special-token IDs match checkpoint config/tokenizer files.
- Position embedding test: compare position IDs and `+2` offset with and without cache.
- Single-block fp32 parity with random hidden states, attention masks, and no cache.
- Prefill logits parity against Transformers for short biomedical prompts.
- Decode parity: run prefill, append one token with `past_key_values`, compare next-token logits and cache lengths.
- Tied-weight test: verify LM head and embedding are one logical weight.
- Optional: last-token-only logits parity for `logits_to_keep=1`.

Recommended tolerances:

- fp32 eager: `rtol=1e-4`, `atol=1e-5` for logits after one block; looser after full model.
- fp16/bf16 optimized attention: start with `rtol=5e-2`, `atol=5e-2` for logits, then tighten per backend.

## 13. Performance probes

- tokenizer throughput for Moses+BPE separately from GPU runtime
- prefill tokens/sec by batch and sequence length
- decode tokens/sec by batch and cache length
- KV cache memory: `layers * 2 * B * heads * S * head_dim * dtype_size`
- QKV packed versus separate GEMM timing
- attention backend comparison: eager, SDPA/Flash-style fused, DinoML fused
- MLP GEMM/GELU/GEMM timing by base and large dimensions
- tied LM head throughput for full sequence versus last-token-only
- vocab-size sensitivity: 42k base versus 57k large/PubMedQA

## 14. Skip/defer list

- Training, losses, gradient checkpointing, dropout, layerdrop.
- Token classification and sequence classification heads for first LM target.
- Beam search and cache reorder until greedy/sampling decode works.
- Gated task fine-tunes until HF access is available.
- Quantized or packed checkpoint loading; the inspected source uses normal dense PyTorch weights.
- Multi-GPU tensor parallelism.
- RoPE/ALiBi/sliding-window support for this family; not present in BioGPT source.

## 15. Final implementation checklist

- [ ] Parse `BioGptConfig` and checkpoint tokenizer metadata.
- [ ] Load dense weights and preserve `output_projection.weight` / `embed_tokens.weight` alias.
- [ ] Implement scaled token embeddings and learned absolute positions with offset 2.
- [ ] Implement BioGPT pre-norm decoder block with biased MHA and GELU MLP.
- [ ] Implement causal mask and padding mask semantics for prefill.
- [ ] Implement per-layer DynamicCache K/V buffers shaped `[B, heads, S, head_dim]`.
- [ ] Add prefill logits parity for `microsoft/biogpt`.
- [ ] Add decode one-token parity with cache.
- [ ] Add large checkpoint config/shape admission for `BioGPT-Large`.
- [ ] Add PubMedQA vocab-size admission/parity smoke when weights are available.
- [ ] Add packed QKV rewrite guarded by exact weight/bias concatenation.
- [ ] Add fused attention backend and fallback path.
- [ ] Add last-token-only tied LM head optimization.
- [ ] Benchmark tokenizer, prefill, decode, KV memory, and LM-head cost.
