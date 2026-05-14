# EuroBERT Transformers Family Audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: EuroBERT/EuroBERT-210m primary; EuroBERT/EuroBERT-610m and EuroBERT/EuroBERT-2.1B swept
Config source: HF Hub config.json snapshots saved in this folder
Source files inspected:
- X:/H/transformers/src/transformers/models/eurobert/configuration_eurobert.py
- X:/H/transformers/src/transformers/models/eurobert/modeling_eurobert.py
- X:/H/transformers/src/transformers/models/eurobert/modular_eurobert.py
- X:/H/transformers/src/transformers/masking_utils.py
Any missing files or assumptions:
- No processor/image/audio files; this is text-only.
- Official Hub repos are public and not gated.
- Official configs carry remote-code auto_map entries. This report scopes DinoML to native in-library source at the pinned commit. Native source does not implement the config-advertised QA head.
```

Auxiliary snapshots in this folder:

- `config_sweep.tsv`
- `source_snapshot.md`
- `EuroBERT_EuroBERT-210m.config.json`
- `EuroBERT_EuroBERT-610m.config.json`
- `EuroBERT_EuroBERT-2.1B.config.json`
- two fine-tuned sequence-classification configs

## 2. High-level architecture

EuroBERT is a text-only, encoder-style masked language model built from Llama-like pre-norm blocks with bidirectional self-attention instead of causal decoding.

```text
tokenizer/data pipeline -> input_ids/attention_mask -> token embedding -> bidirectional RoPE encoder -> final RMSNorm -> task head
```

Primary DinoML runtime target: `EuroBertForMaskedLM`, producing `[B, S, vocab_size]` logits for fill-mask. Base encoder and sequence/token classification heads are independently stageable. Autoregressive decode, KV-cache production serving, vision/audio preprocessing, and multimodal packing are not part of the primary target.

## 3. Important config dimensions

| Checkpoint | Task/head | Hidden | Layers | Heads | KV heads | Head dim | Intermediate | Max pos | Vocab |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `EuroBERT/EuroBERT-210m` | masked LM | 768 | 12 | 12 | 12 | 64 inferred | 3072 | 8192 | 128256 |
| `EuroBERT/EuroBERT-610m` | masked LM | 1152 | 26 | 18 | 6 | 64 inferred | 4096 | 8192 | 128256 |
| `EuroBERT/EuroBERT-2.1B` | masked LM | 2304 | 32 | 18 | 6 | 128 inferred | 6144 | 8192 | 128256 |
| `Ludo33/eurobert210m_Mobilite_v1` | seq cls | 768 | 12 | 12 | 12 | 64 | 3072 | 8192 | 128256 |
| `luissattelmayer/EuroBERT-immigration-binary-german` | seq cls | 768 | 12 | 12 | 12 | 64 | 3072 | 8192 | 128256 |

Common source/config facts: `hidden_act="silu"`, `attention_bias=false`, `mlp_bias=false` by default, `rms_norm_eps=1e-5`, `tie_word_embeddings=false`, `mask_token_id=128002`, `pad/eos_token_id=128001`, `bos_token_id=128000`, `rope_theta=250000` in Hub configs. The pinned native config stores RoPE through `rope_parameters`; older configs expose `rope_theta`/`rope_scaling`, so loading must use Transformers config normalization or reproduce that mapping.

## 3a. Family variation traps

- `610m` and `2.1B` use GQA: `num_key_value_heads < num_attention_heads`, requiring KV repeat or a native GQA attention kernel.
- `2.1B` has `head_dim=128` inferred from `2304 / 18`; do not hard-code 64.
- `intermediate_size` is not uniformly `4 * hidden_size` for larger checkpoints.
- Native source reads `classifier_pooling`; some fine-tuned configs contain historical `clf_pooling`. Normalize deliberately or reject with a clear error.
- Official `auto_map` advertises `EuroBertForQuestionAnswering`; pinned native source has no QA class, so QA is remote-code-only/out of scope here.
- `_tied_weights_keys` declares `lm_head.weight` tied to embeddings, but `tie_word_embeddings=false` in configs. Treat loaded checkpoint aliasing as a weight-loader contract, not an assumption from config alone.
- No channel layout translation is relevant; all model tensors are sequence-major `[B, S, H]`.

## 4. Operator coverage checklist

Tensor/layout ops: embedding lookup, arange/default position IDs, view/reshape, transpose, contiguous, unsqueeze, expand, concat for RoPE rotate-half, indexed row selection for BOS pooling, sequence reductions for mean/late pooling.

Neural primitives: RMSNorm over last dim, dense Linear/GEMM, residual add, SiLU, elementwise multiply, GELU for sequence classification head, optional classifier Linear.

Attention primitives: bidirectional MHA/GQA, Q/K RoPE, fp32 softmax over key length, additive padding mask, `repeat_kv` or GQA-native attention, dropout can be omitted for inference.

Position math: RoPE with theta `250000`, cos/sin computed in fp32 and cast to hidden dtype.

Heads: masked-LM Linear `hidden_size -> vocab_size`; sequence classification dense `H -> H`, GELU, classifier `H -> num_labels`; token classification `H -> num_labels`.

Generation/cache ops: no autoregressive decode required. `past_key_values` is accepted by the attention method but the model uses a bidirectional mask and primary inference should reject cache/decode for first integration.

Preprocessing-coupled ops: tokenizer supplies `input_ids`, `attention_mask`, optional `position_ids`; no model-coupled processor.

## 5. Layer/block breakdown

Encoder block, repeated `num_hidden_layers`:

```text
x0 = hidden_states
x = RMSNorm(x0)
q = Linear(H -> num_heads * head_dim, bias=attention_bias)(x)
k = Linear(H -> kv_heads * head_dim, bias=attention_bias)(x)
v = Linear(H -> kv_heads * head_dim, bias=attention_bias)(x)
q,k,v = view [B,S,heads,D] / [B,S,kv_heads,D] then transpose to [B,heads,S,D]
q,k = RoPE(q,k, cos[position_ids], sin[position_ids])
attn = bidirectional_attention(q,k,v, padding_mask)
x = x0 + Linear(num_heads * D -> H, bias=attention_bias)(attn)
y0 = x
y = RMSNorm(y0)
y = Linear(intermediate -> H, bias=mlp_bias)(SiLU(Linear(H -> intermediate)(y)) * Linear(H -> intermediate)(y))
hidden_states = y0 + y
```

Final encoder output applies `RMSNorm(H)`. Masked LM applies `Linear(H -> vocab_size)` at every token.

## 6. Attention requirements

Attention is noncausal bidirectional self-attention. It is rectangular only in the sense that padding masks may produce `[B, 1, Q, KV]`; normal encoder self-attention uses `Q = KV = S`.

| Field | Requirement |
|---|---|
| Causality | noncausal/bidirectional |
| Attention type | self-attention |
| MHA/GQA | both: 210m MHA, 610m/2.1B GQA |
| Head counts | see config table |
| Q width | `num_attention_heads * head_dim` |
| K/V width | `num_key_value_heads * head_dim` |
| Masking | padding mask via `create_bidirectional_mask`; full attention when no padding |
| Sliding/local | none in inspected configs/source |
| KV cache | not required for primary masked-LM encoder target |
| Backend compatibility | source advertises FlashAttention, SDPA, and flex attention support |

Eager math order: repeat KV groups, compute `Q @ K^T`, multiply by `head_dim ** -0.5`, add mask, softmax in fp32, cast to query dtype, then `P @ V`.

## 7. Position encoding and custom math

RoPE is shared across layers and computed once per model forward from `position_ids`.

```python
def eurobert_rope(q, k, position_ids, inv_freq, attention_scaling=1.0):
    freqs = (inv_freq[None, :, None].float() @ position_ids[:, None, :].float()).transpose(1, 2)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos() * attention_scaling
    sin = emb.sin() * attention_scaling
    cos = cos[:, None, :, :]
    sin = sin[:, None, :, :]
    rotate = lambda x: torch.cat((-x[..., x.shape[-1] // 2 :], x[..., : x.shape[-1] // 2]), dim=-1)
    return (q * cos) + (rotate(q) * sin), (k * cos) + (rotate(k) * sin)
```

`inv_freq` can be precomputed per config/head_dim/theta. Cos/sin depend on runtime `position_ids` and sequence length; default position IDs are `[0..S-1]`.

## 8. Preprocessing and input packing

Tokenizer/data pipeline owns text normalization and tokenization. Runtime graph inputs are:

- `input_ids`: `[B, S]` integer token IDs.
- `attention_mask`: optional `[B, S]`, with 1/true for valid tokens and 0/false for padding.
- `position_ids`: optional `[B, S]`; source default is arange over sequence length with batch dimension 1.
- `inputs_embeds`: optional alternative to `input_ids`; first DinoML target can reject it.

Special token IDs from configs: BOS `128000`, EOS/PAD `128001`, MASK `128002`. No segment/token type IDs, packed sequence descriptors, cu_seqlens, placeholder scatter, image/audio features, or layout metadata are required.

## 9. Graph rewrite / lowering opportunities

### Rewrite: QKV projection packing

Source pattern: three bias-free Linear ops from the same normalized hidden state.

Replacement: one packed GEMM producing `[Q, K, V]`, then split by widths.

Preconditions: identical input tensor, same dtype, all three projections present, bias flags match and are supported, split sizes exactly `[num_heads * D, kv_heads * D, kv_heads * D]`.

Failure cases: mixed bias, remote/fine-tuned custom modules, tensor-parallel sharded weights not normalized.

Parity sketch: compare individual Q/K/V tensors before RoPE for 210m and GQA configs.

### Rewrite: GQA attention without materialized repeat

Source pattern: `repeat_kv(k/v, num_key_value_groups)` followed by dense attention.

Replacement: GQA attention kernel that maps query head to KV head by integer division.

Preconditions: `num_attention_heads % num_key_value_heads == 0`, no requested attention weights requiring repeated materialization.

Failure cases: unsupported attention backend, non-divisible head counts.

Parity sketch: run eager attention with materialized repeat against GQA kernel for random masks.

### Rewrite: masked-LM selected-token logits

Source pattern: LM head over all `[B,S,H]` states.

Replacement: gather masked positions first, then GEMM `H -> vocab` only for selected tokens.

Preconditions: caller only needs mask predictions and supplies mask positions; output ABI changes from dense logits to selected logits.

Failure cases: generic `AutoModelForMaskedLM` parity requiring full logits.

### Rewrite: classifier late pooling fusion

Source pattern: `Linear -> GELU -> Linear` over every token, then masked mean of logits.

Replacement: fuse token classifier head and masked reduction where classification output only is needed.

Preconditions: `classifier_pooling == "late"`, inference only, no hidden states requested.

Failure cases: `bos`/`mean` pooling modes or callers requesting per-token intermediate logits.

## 10. Kernel fusion candidates

Highest priority:

- RMSNorm: every block has two RMSNorms plus final RMSNorm.
- QKV GEMM packing and RoPE: reduces launches and preserves split order.
- Bidirectional MHA/GQA attention: 610m/2.1B need efficient GQA without materialized KV repeat.
- SwiGLU MLP: fuse SiLU and multiply before down projection.

Medium priority:

- Bias-free residual GEMM epilogues for attention output and MLP down projection.
- Full masked-LM logits GEMM with tied/aliased embedding handling where present.
- Sequence-classification pooling reductions for fine-tuned checkpoints.

Lower priority:

- Token-classification head fusion.
- Selected-token masked-LM logits ABI optimization.
- Remote-code QA head, only after a separate remote-code scope decision.

## 11. Runtime staging plan

Stage 1: parse native EuroBERT config, normalize older `rope_theta`/`rope_scaling` into RoPE parameters, load 210m weights.

Stage 2: implement base encoder block parity with random weights for MHA 210m dimensions.

Stage 3: add GQA coverage for 610m/2.1B dimensions, including native GQA attention or guarded KV repeat.

Stage 4: run full encoder plus masked-LM head with full logits.

Stage 5: add sequence and token classification heads; support `classifier_pooling` and explicitly handle/normalize historical `clf_pooling`.

Stage 6: add optimized attention, QKV packing, RMSNorm/SwiGLU fusion, and selected-mask logits as optional graph rewrites.

Stub initially: losses, training dropout, gradient checkpointing, returned attentions/hidden states, remote-code QA.

## 12. Parity and validation plan

- Unit parity for RMSNorm fp32/fp16/bf16 with fp32 variance.
- RoPE parity for default `position_ids`, custom batch position IDs, head_dim 64 and 128.
- Attention parity for MHA and GQA with no mask, all-valid mask, and padded mask.
- One-block parity with random weights for 210m, 610m, and 2.1B shapes at short sequence lengths.
- Full encoder parity on 210m checkpoint for masked positions and full logits.
- Fine-tuned sequence-classification parity for `late` pooling and masked mean denominator.
- Recommended tolerances: fp32 `1e-4` absolute/relative for block outputs, fp16/bf16 `5e-3` to `1e-2` depending on attention backend and accumulation policy.

## 13. Performance probes

- Encoder throughput by sequence length: 128, 512, 2048, 8192.
- Compare MHA 210m vs GQA 610m/2.1B attention memory and latency.
- Full logits GEMM cost versus selected-mask logits cost.
- RMSNorm/SwiGLU fusion launch count and bandwidth.
- Batch-size sweep for fill-mask throughput.
- Padding-ratio sweep to measure mask handling cost.
- Weight loading and optional GGUF/dense provider experiments separately from source-required behavior.

## 14. Skip/defer list

- Training losses and dropout behavior.
- Gradient checkpointing.
- Autoregressive generation and decode KV cache.
- Remote-code QA head.
- `inputs_embeds` first-class ABI.
- Tensor parallel and pipeline parallel execution plans.
- Returning dense attention weights in optimized attention path.
- Quantization or packed weight formats; no source-coupled quantization is present.

## 15. Final implementation checklist

- [ ] Parse EuroBERT native config and normalize legacy RoPE fields.
- [ ] Load token embedding, per-layer projections, RMSNorm weights, MLP weights, final norm, and heads.
- [ ] Implement/source-map RMSNorm.
- [ ] Implement bidirectional MHA and GQA attention with padding mask.
- [ ] Implement RoPE theta `250000` and dynamic `position_ids`.
- [ ] Implement SwiGLU MLP.
- [ ] Implement masked-LM full logits head.
- [ ] Add sequence-classification pooling modes and `clf_pooling` compatibility decision.
- [ ] Add token-classification head.
- [ ] Add one-block and full-model parity tests for 210m.
- [ ] Add GQA parity tests for 610m and 2.1B dimensions.
- [ ] Benchmark encoder-only and masked-LM logits paths.
- [ ] Gate or reject remote-code-only QA explicitly.
