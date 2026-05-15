# I-BERT (`ibert`) Transformers Family Audit

## 1. Source basis

```text
Transformers commit/version: local checkout transformers at b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: primary kssteven/ibert-roberta-base; sweep includes base, large, large-mnli, and open fine-tunes
Config source: Hugging Face config.json files plus current IBertConfig defaults
Source files inspected:
  transformers/src/transformers/models/ibert/configuration_ibert.py
  transformers/src/transformers/models/ibert/modeling_ibert.py
  transformers/src/transformers/models/ibert/quant_modules.py
  transformers/src/transformers/models/ibert/__init__.py
Any missing files or assumptions: tokenizer coupling inspected from config/tokenizer_config only; no imports/tests run; base-mnli/base-squad2/large-squad2 config URLs returned 401 at checked paths.
```

Primary runtime target for DinoML: encoder-only masked-LM and classification inference for RoBERTa-shaped I-BERT checkpoints. The first useful target should be dense `quant_mode=false` parity, with `quant_mode=true` integer-only behavior admitted later behind explicit quantization-buffer and integer-kernel gates.

See `_sources/source_notes.md` and `_sources/config_sweep.md` for source and config snapshots.

## 2. High-level architecture

I-BERT in Transformers is a RoBERTa/BERT-style bidirectional text encoder with optional task heads. The source docstring inherits decoder wording, but this implementation constructs only embeddings, encoder self-attention layers, and optional pooler; `IBertEncoder` explicitly sets cross-attentions to `None`.

```text
tokenized text -> word/token-type/position embeddings -> embedding add + layernorm
  -> N encoder blocks (MHA + residual LN + FFN + residual LN)
  -> sequence output
  -> masked-LM / sequence classification / token classification / QA / multiple-choice head
```

Quantized mode does not change topology, but it changes many primitives: embeddings and linears return `(float_value, scaling_factor)` pairs representing integer-simulated values, residual adds are done through fixed-point rescale, and GELU/softmax/layernorm can use integer approximation paths unless forced to dequantize.

## 3. Important config dimensions

| Field | Source default | Public RoBERTa base | Public RoBERTa large | Operator impact |
|---|---:|---:|---:|---|
| `hidden_size` | 768 | 768 | 1024 | Encoder width, projection input/output. |
| `num_hidden_layers` | 12 | 12 | 24 | Repeated encoder blocks. |
| `num_attention_heads` | 12 | 12 | 16 | MHA head count. |
| `head_dim` | 64 inferred | 64 | 64 | `hidden_size / heads`; source rejects non-divisible hidden size. |
| `intermediate_size` | 3072 | 3072 | 4096 | FFN expansion. |
| `vocab_size` | 30522 | 50265 | 50265 | Embedding and masked-LM decoder rows. |
| `max_position_embeddings` | 512 | 514 | 514 | Position embedding rows; RoBERTa uses pad offset. |
| `type_vocab_size` | 2 | 1 | 1 | Token-type embedding rows. |
| `hidden_act` | `gelu` | `gelu` | `gelu` | Source raises for non-`gelu`. |
| `layer_norm_eps` | `1e-12` | `1e-5` | `1e-5` | LN parity. |
| `quant_mode` | false | false | false | Enables integer simulation modules when true. |
| `force_dequant` | `none` | omitted -> `none` | omitted -> `none` | Selectively routes nonlinear quant modules back to float. |
| cache support | none | none | none | Encoder-only; no KV cache. |

Representative checkpoint sweep:

| Model id | Architecture | Hidden/Layers/Heads | Task | Quant fields |
|---|---|---:|---|---|
| `kssteven/ibert-roberta-base` | `IBertForMaskedLM` | 768 / 12 / 12 | fill-mask | `quant_mode=false`, `force_dequant` omitted. |
| `kssteven/ibert-roberta-large` | `IBertForMaskedLM` | 1024 / 24 / 16 | fill-mask | `quant_mode=false`, `force_dequant` omitted. |
| `kssteven/ibert-roberta-large-mnli` | `IBertForSequenceClassification` | 1024 / 24 / 16 | classification | `quant_mode=false`, 3 labels. |
| `DunnBC22/ibert-roberta-base-finetuned-WikiNeural` | `IBertForTokenClassification` | 768 / 12 / 12 | token classification | `quant_mode=false`, `force_dequant=none`, 9 labels. |
| `VitaliiVrublevskyi/ibert-roberta-base-finetuned-mrpc` | `IBertForSequenceClassification` | 768 / 12 / 12 | classification | `quant_mode=false`, `problem_type=single_label_classification`. |

## 3a. Family variation traps

- Public configs mostly advertise `quant_mode=false`; DinoML should not infer integer-only behavior from the family name alone.
- `quant_mode=true` is source-supported and operator-significant: forward returns and consumes scale tensors between nearly every submodule.
- `force_dequant` can disable only GELU, only softmax, only layernorm, or all nonlinear integer approximations. This creates mixed integer/float regions.
- Current source ignores historical `position_embedding_type=absolute` fields seen in fine-tuned configs; it always uses absolute learned position embeddings.
- `hidden_act` must be `gelu`; reject other activations for this family.
- RoBERTa checkpoints use `type_vocab_size=1`, so token-type IDs other than 0 would be out of range.
- `resize_token_embeddings()` is not supported; embedding/LM-head aliasing must be preserved for masked-LM.
- Quantized activation range buffers (`x_min`, `x_max`) are training/calibration state. In inference, they must be loaded and treated as constants, not recomputed.
- Layout passes should preserve `[B, S, H]` sequence layout. There is no image/video layout work; transposes are attention-head reshapes only.

## 4. Operator coverage checklist

Tensor/layout ops:

- `arange`, `expand`, `zeros`, `ones`, `ne`, `int`, `cumsum`, `long`, pad-aware position ID construction.
- Shape/view ops: `view`, `reshape`, `transpose(1, 2)`, `transpose(-1, -2)`, `permute(0, 2, 1, 3)`, `contiguous`, `split`, `squeeze`, flatten/unflatten for multiple choice.
- Indexing: first-token pooling `hidden_states[:, 0]`.

Neural primitives:

- Embedding lookup for word, token type, and position tables.
- Dense linear/GEMM with bias: Q/K/V, attention output, FFN up/down, pooler/head linears.
- LayerNorm over hidden dimension.
- GELU for FFN and masked-LM head.
- Tanh for pooler and classification head.
- Dropout is disabled in eval but appears in source.

Attention primitives:

- Dense noncausal self-attention only.
- Q/K/V projection width: `hidden_size -> hidden_size`.
- Scores: `[B, heads, S, S] = Q @ K^T / sqrt(head_dim)`.
- Additive extended attention mask broadcast over heads/query positions.
- Softmax over last dimension.
- Context: attention probabilities @ V, then head merge.

Quantized/packed metadata ops:

- Symmetric quantization: `round(x / scale)`, clamp to signed range.
- Per-channel weight scale for encoder `QuantLinear`: scale shape `[out_features]`.
- Global activation scale for `QuantAct`: scale shape `[1]`.
- Bias integer quantization scale: `weight_scale * input_activation_scale`.
- Fixed-point rescale/add with mantissa/exponent from `np.frexp`.
- Integer GELU polynomial approximation.
- Integer softmax polynomial-exp approximation and fixed `1 / 2**output_bit` output scale.
- Integer LayerNorm mean/variance/sqrt/rescale path, with loaded `shift` buffer and lazily computed `dim_sqrt`.

Preprocessing-coupled ops:

- RoBERTa tokenizer contract: `RobertaTokenizer`, `<s>`/`</s>`/`<pad>`/`<mask>`, `model_max_length=512`, pad id 1.
- `input_ids` or `inputs_embeds` mutually exclusive.
- Optional caller-provided `attention_mask`, `token_type_ids`, and `position_ids`.

## 5. Layer/block breakdown

Embedding stage:

```text
word = QuantEmbedding(vocab_size -> H)(input_ids)
tok = QuantEmbedding(type_vocab_size -> H)(token_type_ids)
x = QuantAct(16)(word + tok)
pos = QuantEmbedding(max_position_embeddings -> H)(position_ids)
x = QuantAct(16)(x + pos)
x = IntLayerNorm(H, eps)(x)
x = Dropout(x)
x = QuantAct(8)(x)
```

Encoder block, repeated `num_hidden_layers` times:

```text
q = QuantLinear(H -> H, bias, int8 weight, int32 bias, per-channel)(x)
k = QuantLinear(H -> H, bias, int8 weight, int32 bias, per-channel)(x)
v = QuantLinear(H -> H, bias, int8 weight, int32 bias, per-channel)(x)
q,k,v = QuantAct(8)(q/k/v)
q,k,v = view [B,S,heads,head_dim] -> [B,heads,S,head_dim]
scores = q @ k.transpose(-1,-2) / sqrt(head_dim)
scores = scores + extended_attention_mask
prob = IntSoftmax(8 or float softmax)(scores)
ctx = prob @ v
ctx = permute/contiguous/view -> [B,S,H]
ctx = QuantAct(8)(ctx)
attn_out = QuantLinear(H -> H)(ctx)
attn_resid = QuantAct(22)(attn_out + x)
x = IntLayerNorm(H)(attn_resid)
x = QuantAct(8)(x)
ff = QuantLinear(H -> intermediate)(x)
ff = IntGELU(ff)
ff = QuantAct(8)(ff)
ff = QuantLinear(intermediate -> H)(ff)
ff_resid = QuantAct(22)(ff + x)
x = IntLayerNorm(H)(ff_resid)
x = QuantAct(8)(x)
```

Heads:

- Masked LM: dense `H -> H`, float GELU, float LayerNorm, decoder `H -> vocab_size`; decoder weight tied to word embedding.
- Sequence classification: first token, dropout, dense `H -> H`, tanh, dropout, out projection `H -> num_labels`.
- Multiple choice: flatten `[B, C, S]` to `[B*C, S]`, base model with pooler, classifier `H -> 1`, reshape `[B, C]`.
- Token classification: sequence dropout, classifier `H -> num_labels`.
- QA: classifier `H -> num_labels`, split last dim into start/end, squeeze to `[B, S]`.

## 6. Attention requirements

I-BERT requires encoder self-attention only for the primary target.

| Property | Requirement |
|---|---|
| Causal | No; bidirectional encoder. |
| Cross-attention | Not implemented in this source. |
| MHA/MQA/GQA | Standard MHA; no GQA/MQA. |
| Heads/head dim | Base 12 x 64, large 16 x 64. |
| Query/key/value width | All equal `hidden_size`. |
| Masking | Additive extended attention mask from `PreTrainedModel.get_extended_attention_mask`. |
| Packed/varlen | Not implemented. |
| Sliding/local | Not implemented. |
| Relative/RoPE/ALiBi | None. |
| KV cache | None. |
| FlashAttention/SDPA | Source uses eager matmul + softmax + matmul; a fused dense encoder attention kernel is an optimization only. |

Quantized attention needs either exact source integer simulation or an admitted approximation. The score scale is `query_scale * key_scale / sqrt(head_dim)`, then mask is added in float-valued tensor form before `IntSoftmax`.

## 7. Position encoding and custom math

Position encoding is learned absolute embedding. If `position_ids` is omitted with `input_ids`, source uses pad-aware cumulative positions:

```python
mask = input_ids.ne(padding_idx).int()
incremental = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
position_ids = incremental.long() + padding_idx
```

For RoBERTa configs with `padding_idx=1`, non-pad positions begin at 2. If `inputs_embeds` are supplied, positions are sequential from `padding_idx + 1` with no pad inference.

Integer-specific math snippets:

```python
# Symmetric quantization
n = 2 ** (num_bits - 1) - 1
scale = max(abs(min), abs(max)).clamp(min=1e-8) / n
x_int = clamp(round(x / scale), -n, n - 1)
```

```python
# Attention score scale in quant mode
score_scale = query_scale * key_scale / sqrt(head_dim)
```

```python
# IntSoftmax sketch
x_int = x / score_scale
x_int = x_int - max(x_int, dim=-1)
exp_int = polynomial_exp_with_floor_and_clamp(x_int)
sum_int = sum(exp_int, dim=-1)
prob_int = floor(exp_int * floor(2**32 / sum_int) / 2**24)
prob_scale = 1 / 2**8
```

## 8. Preprocessing and input packing

The neural graph consumes text tensors:

- `input_ids`: `[B, S]` int64 token IDs.
- `attention_mask`: optional `[B, S]`, promoted by Transformers helper to additive broadcast mask.
- `token_type_ids`: optional `[B, S]`; default zeros.
- `position_ids`: optional `[B, S]`; otherwise generated as above.
- `inputs_embeds`: optional `[B, S, H]`, mutually exclusive with `input_ids`.

Tokenizer coupling is RoBERTa-like for public checkpoints: vocab/merges tokenizer, `<s>` as CLS/BOS id 0, `</s>` as SEP/EOS id 2, `<pad>` id 1, `<mask>` token configured with left-strip behavior. Tokenization and padding belong in the CPU/data pipeline for first integration. Position ID construction can be CPU-side or graph-side; graph-side parity needs `ne`, `cumsum`, masked multiply, and pad-offset addition.

No multimodal packing, placeholder scatter, image/audio preprocessing, or varlen metadata exists.

## 9. Graph rewrite / lowering opportunities

### Rewrite: dense I-BERT linear to GEMM epilogue

Source pattern: `QuantLinear(..., quant_mode=false)` or head `nn.Linear`.

Replacement: row-major GEMM with bias epilogue.

Preconditions:

- `quant_mode=false`, or quantized path has been dequantized to dense values intentionally.
- Input is contiguous logical `[B, S, in]` or flattened to `[B*S, in]`.
- Weight layout is PyTorch linear `[out, in]`; GEMM RHS can use transposed/column-major handling.

Failure cases: `quant_mode=true` without a quantization-aware lowering plan; non-contiguous views not normalized.

Parity test: compare one encoder block and one head against Transformers eval mode for fixed random weights.

### Rewrite: Q/K/V projection packing

Source pattern: three independent `QuantLinear(H -> H)` for query, key, value.

Replacement: packed QKV GEMM `H -> 3H`, split in `[q, k, v]` order.

Preconditions:

- All three projections have same input, dtype, quantization mode, and bias presence.
- For quantized mode, either identical activation input scale and supported per-channel output scales are preserved per output row, or rewrite is disabled.
- Packed weight layout is row concatenation `[Wq; Wk; Wv]`, bias `[bq; bk; bv]`.

Failure cases: mixed quant/dequant projections, missing scale buffer support, weight tying assumptions.

### Rewrite: encoder attention to fused dense MHA

Source pattern: reshape heads, `matmul`, divide by sqrt, additive mask, softmax last dim, dropout eval no-op, `matmul`, merge heads.

Replacement: fused noncausal dense attention.

Preconditions:

- Eval mode dropout disabled.
- No output attentions requested, or fused kernel can optionally materialize probabilities.
- Mask is additive and broadcast-compatible.
- No quantized integer softmax requirement unless fused kernel reproduces source approximation.

Failure cases: `output_attentions=True` fast path without attention output, `quant_mode=true` exact parity target, dynamic masks with unsupported rank.

### Rewrite: RoBERTa position IDs as precomputed input

Source pattern: graph-side `ne -> cumsum -> multiply -> add padding_idx`.

Replacement: CPU tokenizer/padding pipeline emits `position_ids`.

Preconditions:

- Input IDs and padding policy controlled by DinoML pipeline.
- `past_key_values_length=0` for this encoder target.

Failure cases: caller supplies `inputs_embeds` without pad information; custom caller-provided `position_ids`.

### Rewrite: quantized weights as encoded constants

Source pattern: runtime computes `weight_integer` and scales from float weights during forward.

Replacement: compile/load-time quantization or direct load of integer buffers plus scales as explicit constants.

Preconditions:

- Calibration/range buffers are fixed and loaded from checkpoint or a quantization artifact.
- Symmetric quantization formula, bit widths, and per-channel axis match source.
- Dense fallback remains available for public `quant_mode=false` configs.

Failure cases: missing activation min/max buffers, training-mode dynamic stats, force-dequant mixed regions, checkpoints that only carry float weights and no calibrated activation ranges.

## 10. Kernel fusion candidates

Highest priority:

- Dense encoder GEMM + bias for projections and FFN. This dominates base dense parity and maps to existing CUTLASS work.
- LayerNorm over `[B, S, H]`, both float and future integer path.
- Dense noncausal encoder attention with additive mask.
- QKV packed projection for dense mode.

Medium priority:

- Residual add + LayerNorm fusion for attention and FFN outputs.
- GELU + FFN down projection scheduling; full GELU fusion is useful but less critical than GEMM.
- Masked-LM head dense + GELU + LN + decoder path, including last-mask-token or selected-token logits when a pipeline can avoid full `[B,S,V]`.
- Sequence/token classification head fusions for deployment tasks.

Lower priority:

- Exact integer GELU/softmax polynomial kernels.
- Integer LayerNorm sqrt/rescale kernel.
- Fixed-point rescale/add kernel for `QuantAct` identity paths.
- Attention probability materialization for `output_attentions=True`.

## 11. Runtime staging plan

Stage 1: parse configs and load dense weights for `quant_mode=false` public base/large checkpoints. Reject `quant_mode=true` with a clear gated gap.

Stage 2: implement base encoder parity in eval mode: embeddings, position IDs, additive attention mask, encoder blocks, optional pooler.

Stage 3: add task heads: masked LM first, then sequence classification, token classification, QA, and multiple choice.

Stage 4: add graph rewrites/fusions for dense inference: GEMM epilogues, packed QKV, fused attention, residual+LN.

Stage 5: design quantized I-BERT admission. Require explicit quantization manifest for weights, activation ranges, scale tensors, bit widths, and `force_dequant`.

Stage 6: implement bounded quantized mode: quantized embedding/linear constants and fixed-point residual rescale, with nonlinear force-dequant fallback.

Stage 7: add exact integer nonlinear kernels for `IntGELU`, `IntSoftmax`, and `IntLayerNorm` where performance justifies it.

## 12. Parity and validation plan

- Config parser tests for source defaults versus RoBERTa checkpoint overrides.
- Position ID parity for padded and unpadded `input_ids`; include `inputs_embeds` sequential fallback.
- Single operator tests: embedding lookup, LayerNorm, GELU, additive mask, dense attention, first-token pooling.
- Single encoder block parity for base dimensions in fp32 eval mode.
- After-N-layer parity for 1, 2, 12 layers with random deterministic weights.
- Masked-LM logits parity against `kssteven/ibert-roberta-base` on short text with mask token.
- Sequence classification parity for large MNLI config/head.
- Token classification parity for WikiNeural fine-tune config/head.
- QA head shape/parity with synthetic `num_labels=2`.
- Quantized future tests: exact `SymmetricQuantFunction`, per-channel `QuantLinear`, `FixedPointMul`, `IntSoftmax`, `IntGELU`, `IntLayerNorm` against source eager implementation.

Recommended tolerances: dense fp32 `rtol=1e-4, atol=1e-5` for full encoder; fp16/bf16 only after dense path is stable and tolerances are measured. Quantized path should target bit-exact integer intermediates for micro-ops or explicitly document any approximation.

## 13. Performance probes

- Encoder-only throughput by batch and sequence length: S in 16, 64, 128, 512.
- Dense attention backend comparison: eager matmul/softmax versus fused encoder attention.
- QKV packed versus three-GEMM projection timing.
- GEMM candidate profile for base and large FFN/projection shapes.
- Masked-LM full vocabulary projection cost versus selected-token logits.
- Classification/token-classification head overhead after encoder.
- Position ID and tokenizer-side preprocessing throughput.
- Quantized future probes: load-time quantization cost, encoded int8 weight memory footprint, dequant/provider comparison, exact integer nonlinear kernel throughput, and activation scale buffer traffic.

## 14. Skip/defer list

- Training, calibration, and dynamic activation range updates.
- Gradient checkpointing and dropout behavior outside eval.
- KV cache, decode, causal generation, beam search.
- Cross-attention/decoder behavior implied by generic docstrings but absent from this implementation.
- `resize_token_embeddings`.
- `output_attentions=True` optimized path; allow slow/materialized fallback first.
- Exact integer nonlinear kernels for first dense integration.
- Unsupported private/gated checkpoint configs until accessible.

## 15. Final implementation checklist

- [ ] Parse `IBertConfig`, including `quant_mode`, `force_dequant`, RoBERTa overrides, and `hidden_act == "gelu"` rejection.
- [ ] Load/tie word embeddings and masked-LM decoder weights without cloning logical aliases.
- [ ] Implement pad-aware position ID generation or require CPU-side `position_ids`.
- [ ] Implement dense embeddings + encoder block + pooler for `quant_mode=false`.
- [ ] Implement additive encoder attention mask semantics.
- [ ] Add dense masked-LM, sequence classification, token classification, QA, and multiple-choice heads.
- [ ] Add QKV packing rewrite with `[q, k, v]` row concatenation guard.
- [ ] Add fused dense encoder attention rewrite with `output_attentions` guard.
- [ ] Gate `quant_mode=true` behind explicit scale/buffer/operator manifest.
- [ ] Implement quantized `QuantLinear`/`QuantEmbedding` encoded constants after dense parity.
- [ ] Implement or intentionally force-dequant `IntGELU`, `IntSoftmax`, and `IntLayerNorm`.
- [ ] Add parity tests for base/large configs and at least one classification/token head.
- [ ] Benchmark encoder, attention, GEMM, and full masked-LM/head projections separately.

## Gated gaps for DinoML

- `quant_mode=true` is not just int8 weights; it is a scale-carrying ABI between modules. Admit only when scales, bit widths, activation ranges, and `force_dequant` policy are artifact-visible.
- Exact I-BERT integer nonlinear behavior requires custom floor/round/clamp/polynomial/sqrt kernels. Dense fallback is safe for public `quant_mode=false` checkpoints but is not integer-only I-BERT parity.
- Quantized source mutates buffers during forward in training/calibration. DinoML inference must reject training-style dynamic updates and require fixed loaded buffers.
- `IntLayerNorm` divides by `self.weight` to form a bias term; zero or malformed LN weights need validation even if official checkpoints are well-formed.
- Public configs with `position_embedding_type` should not trigger alternate position encoding; current source does not read that field.
