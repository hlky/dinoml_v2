# Transformers Audit: nomic_bert

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: nomic-ai/nomic-embed-text-v1.5 as the native-source reference; older/variant configs swept separately.
Config source: local Transformers config plus HF config.json snapshots in _sources/.
Source files inspected:
- transformers/src/transformers/models/nomic_bert/configuration_nomic_bert.py
- transformers/src/transformers/models/nomic_bert/modeling_nomic_bert.py
- transformers/src/transformers/models/nomic_bert/modular_nomic_bert.py
- transformers/src/transformers/modeling_rope_utils.py
- transformers/src/transformers/masking_utils.py
- transformers/src/transformers/integrations/sdpa_attention.py
- transformers/src/transformers/integrations/flash_attention.py
Any missing files or assumptions: no tests/imports were run. The generated modeling file says future source edits should be made in modular_nomic_bert.py; this report treats the generated modeling file as the runtime truth and the modular file as the edit source.
```

HF snapshots saved under `_sources/`:

- [nomic-ai/nomic-embed-text-v1.5 config](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5/blob/main/config.json)
- [nomic-ai/nomic-embed-text-v1 config](https://huggingface.co/nomic-ai/nomic-embed-text-v1/blob/main/config.json)
- [nomic-ai/nomic-bert-2048 config](https://huggingface.co/nomic-ai/nomic-bert-2048/blob/main/config.json)
- [nomic-ai/nomic-embed-text-v1-unsupervised config](https://huggingface.co/nomic-ai/nomic-embed-text-v1-unsupervised/blob/main/config.json)
- [nomic-ai/nomic-embed-text-v2-moe-unsupervised config](https://huggingface.co/nomic-ai/nomic-embed-text-v2-moe-unsupervised/blob/main/config.json)
- SentenceTransformers pooling/module metadata for `nomic-embed-text-v1` and `nomic-embed-text-v1.5`.
- Remote-code reference files from `nomic-ai/nomic-bert-2048` were downloaded only to classify native-vs-remote divergence.

## 2. High-level architecture

Primary DinoML target: text encoder embeddings, not autoregressive generation.

Dataflow:

```text
tokenizer / task prefix / attention mask
-> word + token-type embeddings
-> embedding LayerNorm
-> repeated bidirectional RoPE encoder blocks
-> last_hidden_state
-> external embedding pooling / optional model CLS pooler / optional MLM or classification head
```

Native `NomicBertModel` is a BERT-like bidirectional encoder. It has no decoder, no KV cache, no causal prefill/decode loop, and no cross-attention. The model body can be validated independently from embedding-service postprocessing:

- CPU/data pipeline: tokenization, task prefixes such as `search_query:` and `search_document:`, padding/truncation, attention mask.
- GPU/runtime body: encoder and optional in-model heads.
- External embedding ABI: SentenceTransformers mean pooling over non-pad tokens, optional v1.5 Matryoshka `layer_norm -> slice -> l2 normalize`.

## 3. Important config dimensions

Native source defaults from `NomicBertConfig`:

| Field | Native default | Runtime impact |
|---|---:|---|
| `vocab_size` | 30528 | word embedding and MLM decoder width |
| `hidden_size` | 768 | token width |
| `num_hidden_layers` | 12 | encoder block count |
| `num_attention_heads` | 12 | MHA heads |
| `head_dim` | 64 if omitted | Q/K/V per-head width |
| `intermediate_size` | 3072 | gated MLP width |
| `hidden_act` | `silu` | SwiGLU-style `silu(gate) * up` |
| `max_position_embeddings` | 2048 | position buffer and RoPE original cache length |
| `type_vocab_size` | 2 | segment/token-type embedding table |
| `layer_norm_eps` | `1e-12` | all LayerNorms |
| `rope_parameters` | required dict in practice | `rope_theta`, `rope_type`, optional scaling |
| `default_theta` | 1000.0 | source default for Nomic RoPE theta |

Representative config sweep:

| Checkpoint | Native-safe? | Shape fields | RoPE fields | Head/embedding contract | Notes |
|---|---|---|---|---|---|
| `nomic-embed-text-v1.5` | Partly | `hidden_size=768`, `layers=12`, `heads=12`, `head_dim=64`, `intermediate=3072`, `max_position_embeddings=2048`, `n_positions=8192` | `rope_type=default`, `rope_theta=1000`, `rotary_scaling_factor=null` | `NomicBertModel`; SentenceTransformers mean pooling; MRL postprocess in README | Native source reads `max_position_embeddings`, not legacy `n_positions`; 8192 claim needs an admission decision. |
| `nomic-embed-text-v1` | Yes for current config snapshot | same 768/12/12/64/3072, `max_position_embeddings=8192` | `rope_type=dynamic`, `factor=2.0`, `rope_theta=1000` | `NomicBertModel`; mean pooling | Long-context dynamic NTK path is source-relevant. |
| `nomic-bert-2048` | Remote-code legacy | legacy `n_embd=768`, `n_layer=12`, `n_head=12`, `n_inner=3072`, `n_positions=2048` | legacy `rotary_emb_base=1000`, `rotary_emb_fraction=1.0` | `NomicBertForPreTraining` remote class | Native source does not implement `NomicBertForPreTraining`; use native `NomicBertForMaskedLM` only after weight/key audit. |
| `nomic-embed-text-v1-unsupervised` | Not native-safe as-is | legacy fields only, `n_positions=8192` | legacy `rotary_scaling_factor=2` but no native `rope_parameters` snapshot | `NomicBertModel`; mean pooling | Must map legacy config or route to remote-code audit. |
| `nomic-embed-text-v2-moe-unsupervised` | Reject for this audit | legacy fields, `vocab_size=250048`, `type_vocab_size=1` | legacy `rotary_emb_base=10000` | MoE flags: `num_experts=8`, `moe_every_n_layers=2`, `moe_top_k=2`; GELU; projection/MLP biases true | Native source ignores MoE and bias flags; separate remote-code MoE audit required. |

## 3a. Family variation traps

- Native source uses separate `q_proj`, `k_proj`, `v_proj`, `o_proj`, all biasless. Remote-code checkpoints used packed `Wqkv` and config flags like `qkv_proj_bias`.
- Native source implements MHA only: `num_key_value_heads`/GQA is absent.
- Native MLP is always gated with `gate_proj`, `up_proj`, `down_proj`, all biasless. Historical `activation_function`, `mlp_fc1_bias`, and `mlp_fc2_bias` fields are not native runtime controls unless mapped to native `hidden_act` and module construction.
- Native `NomicBertForPreTraining`, multiple-choice, QA, remote MoE, remote vision/pooling classes, fused dropout-add-layernorm flags, and remote packed projection layouts are out of scope for the in-library native model.
- `n_positions` and `rotary_scaling_factor` are legacy remote-code fields. Native long-context behavior must come through `max_position_embeddings` plus `rope_parameters`.
- The native encoder is post-norm at the block level: attention residual then LayerNorm, MLP residual then LayerNorm. Do not assume Llama-style RMSNorm/pre-norm.
- Embeddings do not include absolute position embeddings; `position_ids` are used for RoPE and for gathering default token-type ids.
- Text embedding parity usually needs external mean pooling and normalization; the in-model pooler is CLS-only and disabled by default.

## 4. Operator coverage checklist

Tensor/layout ops:

- `arange` for default `position_ids`.
- `gather` for default `token_type_ids` from buffered zeros using `position_ids`.
- `view/reshape`, `transpose`, `contiguous`, `cat`, `unsqueeze`, broadcast expand.
- `slice/select`: CLS `hidden_states[:, 0]`, Matryoshka vector slice, MLM/classification losses deferred.

Neural network primitives:

- Embedding lookup: word `[vocab_size, 768]`, token type `[type_vocab_size, 768]`.
- LayerNorm over 768 with eps from config: embeddings, post-attention, post-MLP, MLM transform.
- Linear no-bias: Q/K/V `768 -> 12 * 64`, O `768 -> 768`, MLP gate/up `768 -> 3072`, down `3072 -> 768`.
- Linear with bias: optional CLS pooler `768 -> 768`, sequence classifier `768 -> num_labels`, token classifier `768 -> num_labels`, MLM transform `768 -> 768`, MLM decoder `768 -> vocab_size`.
- Activations: `silu` for native default MLP, `tanh` for pooler, configured `hidden_act` for MLM transform and MLP.
- Dropout is present in source but can be compile-erased for inference.

Attention primitives:

- Bidirectional dense self-attention.
- Q/K RoPE before attention.
- Mask add with 4D additive mask for eager attention; SDPA/Flash mask interfaces may return `None` for all-visible batches.
- Softmax over key dimension, then attention-value matmul.

Position/rotary ops:

- RoPE inverse frequency generation and `cos/sin` in fp32.
- `rotate_half` with split-half convention, not interleaved.
- Dynamic NTK RoPE if `rope_parameters.rope_type` contains `dynamic`.

Pooling/embedding-head ops:

- SentenceTransformers mean pooling: `sum(last_hidden_state * attention_mask) / clamp(sum(mask), min=1e-9)`.
- v1.5 MRL postprocess from README: `layer_norm(embedding) -> embedding[:, :matryoshka_dim] -> l2_normalize`.
- Optional in-model CLS pooler: first token, dense, tanh.

## 5. Layer/block breakdown

Embeddings:

```text
input_ids [B,S] -> word_embeddings [B,S,768]
token_type_ids [B,S] -> token_type_embeddings [B,S,768]
x = word + token_type
x = LayerNorm(x)
```

Encoder block, repeated `num_hidden_layers`:

```text
residual = x
q = Linear(768 -> H*D, bias=False)(x).view(B,S,H,D).transpose(1,2)
k = Linear(768 -> H*D, bias=False)(x).view(B,S,H,D).transpose(1,2)
v = Linear(768 -> H*D, bias=False)(x).view(B,S,H,D).transpose(1,2)
q,k = RoPE(q,k, cos[B,S,D], sin[B,S,D])
attn = BidirectionalAttention(q,k,v, mask, scale=D**-0.5)
x = residual + Linear(H*D -> 768, bias=False)(attn.transpose/reshape)
x = LayerNorm(x)
residual = x
mlp = Linear(3072 -> 768, bias=False)(act(Linear(768 -> 3072)(x)) * Linear(768 -> 3072)(x))
x = LayerNorm(residual + mlp)
```

MLM head:

```text
hidden -> Linear(768 -> 768) -> activation -> LayerNorm -> Linear(768 -> vocab_size)
decoder.weight is tied to word_embeddings.weight; decoder bias aliases predictions.bias.
```

Classification heads:

- Sequence classification uses native CLS pooler output, dropout, then `Linear(768 -> num_labels)`.
- Token classification uses per-token sequence output, dropout, then `Linear(768 -> num_labels)`.

## 6. Attention requirements

- Type: noncausal bidirectional self-attention.
- Heads: native MHA, `num_attention_heads=12`, no separate KV head count.
- Head dim: explicit `head_dim` if present, otherwise `hidden_size // num_attention_heads`.
- Projection width: `num_attention_heads * head_dim`; allow the report-level guard that this can differ from `hidden_size` if a future config sets nonstandard `head_dim`.
- Masking: `attention_mask` is a 2D padding mask or already-prepared 4D mask. `create_bidirectional_mask` dispatches by `config._attn_implementation`.
- Backends advertised by native source: eager, SDPA, FlashAttention, FlexAttention.
- No KV cache, decode cache, sliding-window, local attention, ALiBi, cross-attention, or packed varlen metadata is implemented in native `nomic_bert`.
- FlashAttention path transposes `[B,H,S,D] -> [B,S,H,D]`, rejects zero dimensions, and may cast fp32 queries to module weight dtype/autocast dtype.

## 7. Position encoding and custom math

Native RoPE:

```python
inv_freq = 1.0 / (theta ** (arange(0, head_dim, 2) / head_dim))
freqs = (inv_freq[None, :, None] @ position_ids[:, None, :]).transpose(1, 2)
emb = cat([freqs, freqs], dim=-1)
cos = cos(emb) * attention_scaling
sin = sin(emb) * attention_scaling

def rotate_half(x):
    return cat([-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]], dim=-1)

q = q * cos[:, None, :, :] + rotate_half(q) * sin[:, None, :, :]
k = k * cos[:, None, :, :] + rotate_half(k) * sin[:, None, :, :]
```

Default Nomic theta is `1000.0`, not Llama's common `10000.0`. Dynamic NTK configs recompute `inv_freq` when `max(position_ids)+1` exceeds the cached sequence length, using the shared Transformers formula:

```text
base = theta * ((factor * seq_len / max_position_embeddings) - (factor - 1)) ** (dim / (dim - 2))
```

Guard: legacy remote fields `rotary_emb_base`, `rotary_emb_fraction`, `rotary_emb_interleaved`, and `rotary_scaling_factor` are not the native ABI unless converted to `rope_parameters`.

## 8. Preprocessing and input packing

Model graph inputs:

- `input_ids [B,S]` or `inputs_embeds [B,S,768]`, exactly one required.
- Optional `attention_mask [B,S]`; 1 means visible token in standard HF usage.
- Optional `token_type_ids [B,S]`; omitted path gathers/expands a zero buffer by `position_ids`.
- Optional `position_ids [1,S]` or `[B,S]`; omitted path is `arange(S)[None, :]`.

Embedding-service ABI:

- Task prefixes are tokenizer/text protocol, not model ops.
- SentenceTransformers metadata composes `Transformer` plus `Pooling`; pooling config sets mean-token pooling only.
- v1.5 README postprocess for Matryoshka embeddings is external to `NomicBertModel`: normalize layer-wise over embedding width, slice to requested dimension such as 512/256/128/64, then L2-normalize.

No image/audio/video tensors, modality placeholders, scatter stitching, or packed sequence descriptors are native to this source basis.

## 9. Graph rewrite / lowering opportunities

### Rewrite: Separate Q/K/V linears -> Packed QKV GEMM

Preconditions:

- Same input tensor and dtype for Q/K/V.
- All three projections are biasless.
- `out_features == num_attention_heads * head_dim` for all three.

Replacement:

```text
x @ Wq.T, x @ Wk.T, x @ Wv.T -> x @ concat_rows(Wq, Wk, Wv).T -> split [q,k,v]
```

Weight transform: concatenate Q, K, V weights along output rows in source order `[q, k, v]`.

Failure cases: future biasful configs, remote packed `Wqkv` checkpoints needing different state-key mapping, GQA/MQA, nonstandard projection widths.

Parity test sketch: compare per-layer Q/K/V tensors before RoPE for random fp32/fp16 inputs and masks.

### Rewrite: Gated MLP -> Fused SwiGLU/GEGLU epilogue

Preconditions:

- `hidden_act` is supported by DinoML fused elementwise.
- `gate_proj` and `up_proj` share input and output width.
- Down projection consumes exactly `act(gate) * up`.

Replacement:

```text
GEMM_gate + GEMM_up -> activation_multiply -> GEMM_down
```

Weight transform: optional pack gate/up into one `Linear(768 -> 2*3072)` with split `[gate, up]`.

Failure cases: legacy remote GELU ungated MLP, biasful MoE/MLP configs, remote fused-bias modules.

### Rewrite: Mean pooling to reductions

Preconditions:

- `attention_mask` is dense `[B,S]`.
- Pooling mode is only mean tokens.
- Padding mask values are 0/1 or bool.

Replacement:

```text
masked = hidden * mask[..., None]
sum = reduce_sum(masked, axis=1)
den = clamp(reduce_sum(mask, axis=1), min=1e-9)
embedding = sum / den[..., None]
```

Failure cases: weighted mean, CLS/last-token pooling, external prompt logic omitted from inputs.

### Rewrite: Inference dropout removal

Preconditions: `model.eval()` / inference-only artifact.

Replacement: erase dropout nodes and keep residual-add/LayerNorm order unchanged.

### Layout guard

The source is text sequence layout `[B,S,C]`. A channel-last/NHWC-style layout pass is not relevant. Protect attention reshape/transposes, LayerNorm `dim=-1`, softmax `dim=-1`, mean pooling `axis=1`, and Matryoshka slice `axis=1` from image-layout translation.

## 10. Kernel fusion candidates

Highest priority:

- LayerNorm over hidden width 768, especially residual-add + LayerNorm post-attention and post-MLP.
- Packed QKV GEMM + split for encoder blocks.
- RoPE + attention prefill for bidirectional full-sequence encoding.
- Mean pooling + L2 normalization for embedding-serving parity.

Medium priority:

- Gated MLP fused activation multiply.
- MLM transform `Linear -> activation -> LayerNorm`.
- Matryoshka postprocess `LayerNorm -> slice -> L2 normalize`.

Lower priority:

- CLS pooler dense+tanh and classification heads.
- Token classification dense head.
- Eager attention output-attentions path.

## 11. Runtime staging plan

Stage 1: native config/weight loader for `NomicBertConfig` fields only, with explicit rejection of unmapped legacy remote fields.

Stage 2: one-block encoder parity with embeddings, RoPE, bidirectional mask, attention, gated MLP, post-norm residuals.

Stage 3: full encoder `last_hidden_state` parity for `nomic-embed-text-v1.5`-style native config at short sequence lengths.

Stage 4: embedding-service output parity: mean pooling, optional MRL truncation, L2 normalization, similarity matrix orientation `[queries, documents] = query_emb @ doc_emb.T`.

Stage 5: long-context admission: support native `rope_parameters.rope_type in {"default","dynamic"}` and validate v1 dynamic 8192 config.

Stage 6: optional heads: MLM, sequence classification, token classification.

Stage 7: optimized kernels/fusions: packed QKV, fused residual LayerNorm, fused MLP, attention backend selection.

Stub initially: losses, dropout, output attentions, hidden-state collection, training flags, remote-code MoE and vision classes.

## 12. Parity and validation plan

- Config parser tests: accept native `hidden_size`/`rope_parameters`; reject or require mapping for legacy-only `n_embd`, `rotary_scaling_factor`, `num_experts`.
- Random tensor op tests: RoPE default and dynamic NTK, `rotate_half`, 2D/4D bidirectional masks, mean pooling with all-pad guard.
- Single-layer parity: compare embeddings, Q/K after RoPE, attention output, post-attention LayerNorm, MLP output.
- Full encoder parity: fp32 tolerance about `1e-4`; fp16/bf16 tolerance model-wide about `2e-2` until attention and LayerNorm accumulation policies are fixed.
- Embedding parity: compare pooled normalized vectors and cosine similarity against Transformers/SentenceTransformers for fixed tokenizer outputs.
- Head parity: MLM logits with tied decoder weights; classifier logits for synthetic configs.
- Long-context parity: sequence lengths around 2048, 4096, 8192 for dynamic RoPE configs.

## 13. Performance probes

- Encoder throughput by batch and sequence length: `S=128/512/2048/8192`.
- Attention backend comparison: eager decomposition, SDPA, DinoML fused attention.
- RoPE generation cost vs precomputed cos/sin cache for fixed sequence buckets.
- Packed QKV vs three GEMMs.
- Gated MLP packed gate/up vs separate GEMMs.
- Mean pooling/postprocess overhead relative to encoder runtime.
- Memory probes for activation temporaries at long sequence lengths.
- Optional MLM head cost: full `[B,S,V]` logits vs masked-position-only gather+GEMM rewrite if a future task needs it.

## 14. Skip/defer list

- Training losses and gradient checkpointing.
- Autoregressive generation and KV cache.
- Cross-attention and encoder-decoder modes.
- Remote-code `NomicBertForPreTraining` exact key/layout compatibility until a separate migration audit.
- Remote MoE `nomic-embed-text-v2-moe-*`.
- Remote vision classes and multimodal embedding alignment.
- General fused dropout-add-layernorm training kernels.
- General packed/varlen FlashAttention metadata unless a serving path proves it is needed.

## 15. Final implementation checklist

- [ ] Parse native `NomicBertConfig` fields and reject legacy-only configs without a mapper.
- [ ] Load word/token-type embeddings, LayerNorms, Q/K/V/O, gated MLP, and optional head weights.
- [ ] Preserve MLM decoder/word embedding tied-weight alias.
- [ ] Implement/default `position_ids` and `token_type_ids` handling.
- [ ] Implement default and dynamic RoPE with theta `1000.0`.
- [ ] Implement bidirectional attention mask creation for 2D and 4D masks.
- [ ] Implement one encoder block parity.
- [ ] Implement full encoder parity.
- [ ] Add SentenceTransformers mean pooling and v1.5 MRL postprocess as an external embedding head.
- [ ] Add packed QKV rewrite with strict bias/layout guards.
- [ ] Add gated MLP fusion with activation guards.
- [ ] Add long-context dynamic RoPE validation.
- [ ] Add MLM, sequence classification, and token classification only after base embeddings are stable.
- [ ] Benchmark sequence-length and batch-size sweeps.

## Gated gaps for DinoML

- `nomic-embed-text-v2-moe-unsupervised` is not covered by native `nomic_bert`; it needs MoE routing, expert GEMM, biasful projections/MLPs, GELU, and legacy config mapping.
- `nomic-bert-2048` remote `NomicBertForPreTraining` is not the same class as native `NomicBertForMaskedLM`; weight-key migration and packed projection layout must be audited before claiming parity.
- Legacy-only configs with `n_positions`/`rotary_scaling_factor` must not silently load through native defaults; they need an explicit conversion to `max_position_embeddings`/`rope_parameters`.
- Embedding parity requires external pooling and normalization metadata; `NomicBertModel` alone is not a complete sentence-embedding runtime.
