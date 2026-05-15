# ModernBERT Transformers audit for DinoML v2

## 1. Source basis

Transformers commit/version:

- Local source checkout: `transformers`
- Commit inspected: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`
- In-library source basis: `src/transformers/models/modernbert`
- `modeling_modernbert.py` is generated from `modular_modernbert.py`; future upstream source edits should target the modular file.

Model id:

- Primary production checkpoints: `answerdotai/ModernBERT-base`, `answerdotai/ModernBERT-large`
- Tiny/debug checkpoints: `hf-internal-testing/tiny-random-ModernBertModel`, `hf-internal-testing/tiny-random-ModernBertForMaskedLM`, `hf-internal-testing/tiny-random-ModernBertForSequenceClassification`, `hf-internal-testing/tiny-random-ModernBertForTokenClassification`
- Head variant inspected from tests/config: `netique/ModernBertForMultipleChoice`

Config source:

- HF raw `config.json` files downloaded from the model repos above.
- Tokenizer coupling snapshot: `answerdotai/ModernBERT-base/tokenizer_config.json`.

Source files inspected:

- `transformers/src/transformers/models/modernbert/configuration_modernbert.py`
- `transformers/src/transformers/models/modernbert/modeling_modernbert.py`
- `transformers/src/transformers/models/modernbert/modular_modernbert.py`
- `transformers/src/transformers/masking_utils.py`
- `transformers/src/transformers/modeling_flash_attention_utils.py`
- `transformers/src/transformers/integrations/flash_attention.py`
- `transformers/src/transformers/integrations/sdpa_attention.py`
- `transformers/tests/models/modernbert/test_modeling_modernbert.py`

Local snapshots written under `_sources/`:

- `configuration_modernbert.py`
- `modeling_modernbert.py`
- `modular_modernbert.py`
- representative `*.config.json` files
- `answerdotai__ModernBERT-base.tokenizer_config.json`

Any missing files or assumptions:

- No custom remote code is required for the audited in-library source.
- This report targets inference for the base encoder and masked LM first. Sequence/token classification, question answering, and multiple choice heads are optional follow-up surfaces.
- Config fields such as `position_embedding_type`, `layer_norm_eps`, `reference_compile`, and `repad_logits_with_grad` appear in some checkpoint configs but are not read by the inspected ModernBERT modeling path.

## 2. High-level architecture

ModernBERT is a text-only encoder. It is not an autoregressive decoder and has no KV cache requirement for the primary target.

Dataflow:

```text
tokenizer -> input_ids/attention_mask/optional position_ids
-> token embedding + LayerNorm
-> repeated bidirectional encoder layers with full or sliding-window attention
-> final LayerNorm
-> base hidden states or task head logits
```

Stage decomposition:

- CPU/data pipeline: tokenizer, special-token insertion, padding/truncation to `model_max_length=8192`.
- GPU/runtime stage 1: embedding lookup and embedding LayerNorm.
- GPU/runtime stage 2: encoder block stack. Full attention and sliding attention layers are independently testable but share QKV/MLP structure.
- GPU/runtime stage 3: task heads. Masked LM, token classification, QA, sequence classification, and multiple choice all consume the final encoder states.

Independently optimizable regions:

- Embedding + LayerNorm.
- Per-layer QKV projection + RoPE + attention + output projection.
- GLU MLP.
- Dense MLM prediction head and decoder projection.
- Classification pooling and small projection heads.

## 3. Important config dimensions

Source defaults from `ModernBertConfig`:

| Field | Default | Runtime meaning |
|---|---:|---|
| `vocab_size` | 50368 | Token embedding rows and MLM decoder output width |
| `hidden_size` | 768 | Encoder width |
| `intermediate_size` | 1152 | GLU half-width; `Wi` projects to `2 * intermediate_size` |
| `num_hidden_layers` | 22 | Encoder layer count |
| `num_attention_heads` | 12 | MHA head count; no GQA/MQA in current source |
| `head_dim` | `hidden_size // num_attention_heads` | 64 for base/large/tiny configs inspected |
| `max_position_embeddings` | 8192 | RoPE cache/source max length, not learned absolute embeddings |
| `local_attention` | 128 | Total local attention window config; `sliding_window = local_attention // 2` |
| `global_attn_every_n_layers` | 3 if `layer_types` omitted | Layers where `i % 3 == 0` use full attention |
| `global_rope_theta` | 160000.0 | Full-attention RoPE base when old config fields are used |
| `local_rope_theta` | 10000.0 | Sliding-attention RoPE base when old config fields are used |
| `hidden_activation` | `gelu` | MLP gate activation |
| `attention_bias` | false | QKV and attention output projection bias |
| `mlp_bias` | false | MLP projection bias |
| `norm_bias` | false | LayerNorm bias |
| `decoder_bias` | true | MLM decoder bias |
| `classifier_pooling` | `cls` default, `mean` in official MLM configs | Classification pooling mode |
| `tie_word_embeddings` | true | MLM decoder weight aliases token embedding weight |
| `sparse_prediction` | false | Only affects MLM path when labels are supplied |

Representative checkpoint sweep:

| Model/config | Architecture | Hidden | Layers | Heads | MLP intermediate | Max pos | Local attention | Full/sliding layers | Pooling | Vocab |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| `answerdotai/ModernBERT-base` | `ModernBertForMaskedLM` | 768 | 22 | 12 | 1152 | 8192 | 128 | 8/14 | mean | 50368 |
| `answerdotai/ModernBERT-large` | `ModernBertForMaskedLM` | 1024 | 28 | 16 | 2624 | 8192 | 128 | 10/18 | mean | 50368 |
| `hf-internal-testing/tiny-random-ModernBertForMaskedLM` | `ModernBertForMaskedLM` | 32 | 4 | 2 | 64 | 512 | 128 | 2/2 | mean | 50368 |
| `hf-internal-testing/tiny-random-ModernBertForSequenceClassification` | `ModernBertForSequenceClassification` | 32 | 4 | 2 | 64 | 512 | 128 | 2/2 | mean | 50368 |
| `hf-internal-testing/tiny-random-ModernBertForTokenClassification` | `ModernBertForTokenClassification` | 32 | 4 | 2 | 64 | 8192 | 128 | 2/2 | mean | 50368 |
| `netique/ModernBertForMultipleChoice` | `ModernBertForMultipleChoice` | 768 | 22 | 12 | 1152 | 8192 | 128 | 8/14 | cls | 50368 |

The full/sliding counts above are derived from `global_attn_every_n_layers=3` where `layer_types` is absent.

## 3a. Family variation traps

- `layer_types` is the real schedule if present. If it is absent, the config creates `full_attention` for layer indices divisible by `global_attn_every_n_layers` and `sliding_attention` otherwise.
- "Global attention" means full-sequence attention layers, not Longformer-style global tokens.
- Sliding attention is bidirectional local attention, not causal local attention.
- The config exposes `sliding_window = local_attention // 2`; ModernBERT attention passes `config.sliding_window + 1` to attention kernels so FlashAttention's inclusive window behavior yields symmetric local attention.
- Source uses `nn.LayerNorm`, not RMSNorm, despite older config doc text saying "rms normalization layers".
- RoPE is used for both full and sliding layers. `position_embedding_type: "absolute"` in configs is ignored by the inspected source.
- There is no `num_key_value_heads`; Q, K, and V all use `num_attention_heads`.
- QKV is a single packed `Linear(hidden_size -> 3 * hidden_size)` then reshaped to `[B, S, 3, heads, head_dim]` and unbound in Q/K/V order.
- `hidden_size` must be divisible by `num_attention_heads`; source raises otherwise.
- `attention_bias`, `mlp_bias`, `norm_bias`, `decoder_bias`, and `classifier_bias` are independent. In source, `classifier_bias` controls the dense layer inside `ModernBertPredictionHead`; the final task classifier `nn.Linear` layers use PyTorch's default bias.
- `classifier_pooling="cls"` and `"mean"` are both source behavior. Multiple choice has a special `cls` path using `attention_mask.argmax(dim=-1)` for the first non-pad token after flattening choices.
- `sparse_prediction=True` only filters hidden states when `labels is not None`. Plain inference without labels still returns dense logits.
- FlashAttention can enter unpad/varlen paths. Padding masks trigger unpadding; non-monotonic flattened `position_ids` with batch size 1 are treated as packed sequences.
- SDPA/eager paths materialize or skip masks through `masking_utils`; mask semantics differ by backend but should be parity-equivalent.
- Output attentions require eager attention for real tensors. SDPA/Flash warn and return no weights.

## 4. Operator coverage checklist

Tensor/layout ops:

- Integer token input and embedding lookup: `[B, S] -> [B, S, H]`.
- Optional direct `inputs_embeds` path: `[B, S, H]`.
- Reshape/view for QKV: `[B, S, 3H] -> [B, S, 3, heads, head_dim]`.
- Unbind/chunk/split: QKV unbind on dimension `-3`; MLP splits `2 * intermediate_size`; QA splits logits into start/end.
- Transpose: Q/K/V `[B, S, heads, D] <-> [B, heads, S, D]`; attention output transpose back.
- `contiguous`, reshape flatten/unflatten, multiple-choice flatten `[B, C, S] -> [B*C, S]`.
- Gather/indexing: first token `[:, 0]`; MC first non-pad token `[row_index, attention_mask.argmax(-1)]`.
- Mean pooling with mask sum/divide.

Neural network primitives:

- `Embedding(vocab_size, hidden_size, padding_idx=pad_token_id)`.
- LayerNorm over hidden dimension, `eps=norm_eps`, optional bias.
- Linear projections:
  - QKV: `hidden_size -> 3 * hidden_size`, optional bias.
  - Attention output: `hidden_size -> hidden_size`, optional bias.
  - MLP `Wi`: `hidden_size -> 2 * intermediate_size`, optional bias.
  - MLP `Wo`: `intermediate_size -> hidden_size`, optional bias.
  - Prediction head dense: `hidden_size -> hidden_size`, `classifier_bias`.
  - MLM decoder: `hidden_size -> vocab_size`, `decoder_bias`, weight tied to embeddings when enabled.
  - Classification heads: `hidden_size -> num_labels` or `hidden_size -> 1`.
- GELU and elementwise multiply for GLU MLP: `act(input) * gate`.
- Dropout modules are inference no-ops but must preserve training-disabled parity.

Attention primitives:

- Bidirectional full MHA.
- Bidirectional sliding-window MHA with symmetric local window.
- Dense eager fallback: `matmul(q, k.T) * head_dim**-0.5`, additive float mask, fp32 softmax, dropout, `matmul(weights, v)`.
- SDPA backend: `torch.nn.functional.scaled_dot_product_attention` with bool/None attention mask.
- FlashAttention backend: transposes to `[B, S, heads, D]`, optional unpadding/varlen, optional sliding-window kernel argument.

Sparse/local attention pattern ops:

- Full attention layers: no local restriction, still respect padding mask.
- Sliding layers: allow token pair `(q, kv)` only if `abs(q_idx - kv_idx) <= config.sliding_window` in mask creation; FlashAttention receives `sliding_window=config.sliding_window + 1`, which becomes window tuple `(sliding_window - 1, sliding_window - 1)`.
- No random/block-sparse/hash attention.

Position/rotary ops:

- Per-layer-type RoPE cos/sin computed from `position_ids`, in fp32, cast back to hidden dtype.
- Separate theta values for `full_attention` and `sliding_attention` by default.
- `apply_rotary_pos_emb` upcasts Q/K to fp32 for rotation then casts to original dtype.

Generation/cache ops:

- Not applicable for the primary target. ModernBERT is encoder-only and has no decode KV cache.

Packed/varlen sequence metadata ops:

- FlashAttention padding path needs nonzero indices, `cu_seqlens_q/k`, max sequence lengths, unpad, varlen attention, and pad back.
- FlashAttention packed path can derive `cu_seqlens` from flattened non-monotonic `position_ids` when `batch_size == 1`.
- SDPA/eager do not use FlashAttention varlen tensors; they use masks.

Preprocessing-coupled ops:

- Tokenizer emits `input_ids` and usually `attention_mask`. Token type IDs are not part of the ModernBERT model forward.
- Special token IDs from config: `cls_token_id=50281`, `sep/eos_token_id=50282`, `pad_token_id=50283`; tokenizer config names `[CLS]`, `[SEP]`, `[PAD]`, `[MASK]`.

## 5. Layer/block breakdown

Embedding:

```text
if inputs_embeds:
  x = inputs_embeds
else:
  x = Embedding(input_ids)
x = Dropout(LayerNorm(x))
```

Encoder layer, repeated `num_hidden_layers` times:

```text
attn_input = Identity(x) for layer 0 else LayerNorm(x)
qkv = Linear(attn_input)                         # [B, S, 3H]
qkv = view(qkv, [B, S, 3, heads, head_dim])
q, k, v = unbind(qkv, dim=-3)
q, k, v = transpose to [B, heads, S, head_dim]
q, k = RoPE(q, k, cos_sin_for_layer_type)
attn = full_or_sliding_attention(q, k, v, mask)
attn = Linear(reshape(attn, [B, S, H]))
x = x + attn
mlp_input = LayerNorm(x)
input, gate = chunk(Linear(mlp_input), 2, dim=-1)
mlp = Linear(Dropout(GELU(input) * gate))
x = x + mlp
```

Final encoder:

```text
last_hidden_state = final LayerNorm(x)
```

Masked LM head:

```text
h = LayerNorm(GELU(Linear(last_hidden_state)))
logits = Linear(h)  # hidden_size -> vocab_size
```

Sequence classification:

```text
pooled = last_hidden_state[:, 0]                         # cls mode
pooled = masked_sum(last_hidden_state) / mask_sum         # mean mode
pooled = prediction_head(pooled)
logits = Linear(Dropout(pooled))                         # hidden_size -> num_labels
```

Token classification:

```text
logits = Linear(Dropout(prediction_head(last_hidden_state)))
```

Question answering:

```text
logits = Linear(Dropout(prediction_head(last_hidden_state)))  # hidden_size -> num_labels
start_logits, end_logits = split(logits, 1, dim=-1)
squeeze last dim
```

Multiple choice:

```text
flatten [B, choices, S] inputs to [B * choices, S]
run encoder
pool cls or mean
logits = Linear(Dropout(prediction_head(pooled)))  # hidden_size -> 1
reshape to [B, choices]
```

## 6. Attention requirements

Attention type:

- Noncausal encoder self-attention only.
- MHA, not MQA/GQA.
- Full bidirectional layers and local bidirectional sliding layers coexist in one stack.

Shapes:

- Hidden states: `[B, S, H]`.
- Q/K/V after projection and transpose: `[B, num_attention_heads, S, head_dim]`.
- Base: heads `12`, head_dim `64`.
- Large: heads `16`, head_dim `64`.
- Tiny configs inspected: heads `2`, head_dim `16`.

Masking style:

- Input `attention_mask`: 2D `[B, S]`, where valid tokens are truthy.
- Full attention mask: bidirectional plus padding.
- Sliding attention mask: bidirectional local plus padding.
- Eager mask: float additive mask, `0` for allowed and dtype min for masked.
- SDPA mask: bool mask or `None`; full bidirectional mask can be skipped when no padding and no local constraint.
- FlashAttention mask: `None` when no padding; 2D mask for unpadding when padding exists.

Packed/varlen support:

- FlashAttention unpads Q/K/V when a padding mask exists, computes cumulative sequence lengths, runs varlen attention, then pads output back.
- FlashAttention also treats batch-size-1 non-monotonic `position_ids` as packed flattened sequences and derives `cu_seqlens` from positions equal to zero.
- DinoML should initially reject packed `position_ids` unless it implements the same `cu_seqlens` derivation and varlen attention path.

Sliding-window/local attention:

- Config `local_attention=128` means `config.sliding_window=64`.
- Mask function uses `abs(q_idx - kv_idx) <= 64`.
- ModernBERT attention sets module `sliding_window=65` for FlashAttention compatibility; FlashAttention processing converts this to `(64, 64)`.

Flash/SDPA/eager differences:

- Eager returns attention weights and uses fp32 softmax before casting back.
- SDPA and FlashAttention do not support `output_attentions=True` in the source path.
- FlashAttention requires nonzero dimensions and may cast query/key/value back to the projection weight dtype if upstream layers silently produced fp32.
- FlashAttention transposes `[B, heads, S, D]` to `[B, S, heads, D]` before calling the common flash utility.

KV cache:

- Not required. Do not build autoregressive cache plumbing for this family unless a separate decoder variant is audited.

## 7. Position encoding and custom math

ModernBERT uses RoPE for encoder attention. There are separate parameter sets by layer type.

Default inverse-frequency computation:

```python
def modernbert_inv_freq(config, layer_type):
    base = config.rope_parameters[layer_type]["rope_theta"]
    dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    return 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
```

Cos/sin computation:

```python
def modernbert_rope(inv_freq, position_ids, dtype):
    freqs = (inv_freq[None, :, None].float() @ position_ids[:, None, :].float()).transpose(1, 2)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)
```

Application:

```python
def apply_modernbert_rope(q, k, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    qf, kf = q.float(), k.float()
    q_rot = qf * cos + rotate_half(qf) * sin
    k_rot = kf * cos + rotate_half(kf) * sin
    return q_rot.to(q.dtype), k_rot.to(k.dtype)
```

Precompute opportunities:

- For standard monotonic positions, cos/sin can be cached per layer type, sequence length, dtype, and device.
- For explicit `position_ids`, especially packed sequences, cos/sin gather/compute depends on runtime input and cannot be assumed contiguous.
- Advanced RoPE types can be routed through `ROPE_INIT_FUNCTIONS` if `rope_parameters[layer_type]["rope_type"] != "default"`. The inspected representative configs use old theta fields that standardize to default RoPE.

## 8. Preprocessing and input packing

Text input contract:

- `input_ids`: `[B, S]` integer token IDs.
- `attention_mask`: optional `[B, S]`, truthy for non-pad tokens.
- `position_ids`: optional `[B, S]`. If omitted, source creates `torch.arange(S).unsqueeze(0)` and relies on broadcasting in RoPE.
- `inputs_embeds`: optional `[B, S, H]`, mutually exclusive with `input_ids`.
- No `token_type_ids` in the source forward signature.

Tokenizer/config coupling:

- `answerdotai/ModernBERT-base` tokenizer config has `model_max_length=8192`.
- Special token names in tokenizer config include `[CLS]`, `[SEP]`, `[PAD]`, `[MASK]`.
- Config IDs: `bos/cls=50281`, `eos/sep=50282`, `pad=50283`.

Packed sequence behavior:

- The model itself accepts `position_ids`; the FlashAttention utility interprets non-monotonic batch-size-1 `position_ids` as packed sequence boundaries.
- For first integration, treat packed flattened sequences as a separate feature gate. A normal padded batch can use `attention_mask`.

CPU/data-pipeline versus GPU/runtime:

- Tokenization, string normalization, special-token construction, and truncation are CPU/data-pipeline work.
- Embeddings, masks, RoPE, attention, and heads are GPU/runtime work.
- For SDPA/eager parity, mask construction may be represented as generated runtime mask ops or folded into a fused attention kernel's mask predicate.

## 9. Graph rewrite / lowering opportunities

### Rewrite: packed QKV Linear to three logical projections

Source pattern:

```text
qkv = Linear(H -> 3H)(x)
qkv.view(B, S, 3, heads, D).unbind(dim=-3)
```

Replacement pattern:

```text
Q = Linear(H -> H)
K = Linear(H -> H)
V = Linear(H -> H)
```

Preconditions:

- Weight is stored in output-major PyTorch Linear layout `[3H, H]`.
- Bias, if enabled, is `[3H]`.
- Split order is Q, K, V.

Weight transform:

```python
wq, wk, wv = Wqkv.weight.view(3, H, H).unbind(0)
bq, bk, bv = Wqkv.bias.view(3, H).unbind(0) if bias else (None, None, None)
```

Failure cases:

- Do not apply if a checkpoint has nonstandard packed projection layout.
- Do not clone tied or shared weights elsewhere; this QKV weight is not tied.

Parity test sketch:

- Compare Q/K/V tensors before RoPE for random hidden states with bias on and off.

### Rewrite: QKV + RoPE + attention backend

Source pattern:

```text
Linear -> view/unbind -> transpose -> RoPE -> attention -> transpose/reshape -> output Linear
```

Replacement pattern:

```text
FusedModernBertAttention(layer_type, theta, local_window)
```

Preconditions:

- `hidden_size % num_attention_heads == 0`.
- `attention_bias` known.
- Layer type is exactly `full_attention` or `sliding_attention`.
- For optimized local attention, sequence length and mask admission must be compatible with the selected backend.
- Packed sequence path rejected unless varlen metadata is implemented.

Shape equations:

- `D = hidden_size / num_attention_heads`
- Q/K/V `[B, heads, S, D]`
- Output `[B, S, hidden_size]`

Failure cases:

- `output_attentions=True` requires eager fallback.
- Advanced/non-default RoPE types require their own parity gate.
- Padded FlashAttention requires unpad/varlen/pad-back support.

Parity test sketch:

- Per-layer compare attention output before residual for full and sliding layer indices, with no padding, right padding, and explicit position IDs.

### Rewrite: GLU MLP fusion

Source pattern:

```text
input, gate = Linear(H -> 2I)(x).chunk(2, dim=-1)
out = Linear(I -> H)(GELU(input) * gate)
```

Replacement pattern:

```text
GatedMLP_GELU(H, I)
```

Preconditions:

- `hidden_activation == "gelu"` or mapped activation has a DinoML implementation.
- Chunk split is exactly half along last dimension.
- `mlp_bias` known.

Failure cases:

- Non-GELU activations from custom configs need separate activation parity.

Parity test sketch:

- Random tensor compare after `Wo`, with fp32 and reduced precision.

### Rewrite: MLM decoder tied embedding

Source pattern:

```text
decoder.weight aliases model.embeddings.tok_embeddings.weight
```

Replacement pattern:

```text
single logical parameter used by embedding lookup and final vocab projection
```

Preconditions:

- `tie_word_embeddings=True` or `_tied_weights_keys` applied during load.

Failure cases:

- Untied custom checkpoints must load separate decoder weight.

Parity test sketch:

- Verify storage identity or equivalent one-logical-constant manifest and compare logits.

### Rewrite: classifier mean pooling

Source pattern:

```text
(hidden * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
```

Replacement pattern:

```text
MaskedMeanPool(axis=sequence)
```

Preconditions:

- `attention_mask` is present or synthesized all-ones.
- Denominator is nonzero for every batch row.

Failure cases:

- All-pad rows are undefined by source division behavior and should be rejected.

Parity test sketch:

- Compare pooled vectors for left padding, right padding, and no padding.

## 10. Kernel fusion candidates

Highest priority:

- LayerNorm over `[B, S, H]`: appears at embedding, layer pre-norms, final norm, and prediction heads.
- QKV projection + RoPE preparation: dominant per-layer pre-attention path and shape-sensitive.
- Full bidirectional attention: required by every third layer in production configs.
- Bidirectional sliding-window attention: required by most layers; dense fallback is wasteful at long sequence length.
- GLU MLP (`Linear -> GELU -> multiply -> Linear`): repeated every layer and fuses well.
- MLM decoder `H -> vocab_size`: required for masked LM; tied embedding alias must be preserved.

Medium priority:

- FlashAttention unpad/varlen and pad-back: necessary for padded throughput parity with HF FlashAttention.
- Mask construction/fusion for SDPA/eager fallback: needed for sliding-window correctness before optimized local attention exists.
- Prediction head fusion: `Linear -> GELU -> LayerNorm` appears in all task heads.
- Masked mean pooling: required for official configs using `classifier_pooling="mean"`.

Lower priority:

- Eager attention weights output: useful for debugging, not first inference target.
- Sparse MLM prediction with labels: primarily training/evaluation with labels, not normal inference.
- Multiple-choice first-non-pad `cls` gather: head-specific and can be staged after base/MLM.

## 11. Runtime staging plan

Stage 1: config and weights

- Parse `ModernBertConfig`, normalize `layer_types`, normalize old theta fields into `rope_parameters`.
- Load embeddings, packed QKV, output projections, MLP weights, LayerNorm weights, and task head weights.
- Preserve MLM tied embedding/decoder alias when `tie_word_embeddings=True`.

Stage 2: one-block parity

- Implement embedding + one full-attention layer + one sliding-attention layer in fp32.
- Use eager dense attention masks first for correctness.

Stage 3: full base encoder

- Run all layers for tiny and base configs with `attn_implementation="eager"` or an equivalent deterministic reference.
- Add final LayerNorm and base model output parity.

Stage 4: masked LM

- Add prediction head and decoder projection.
- Validate dense logits. Defer `sparse_prediction=True` label-filtered output.

Stage 5: classification heads

- Add sequence/token classification, QA, and multiple choice as optional heads.
- Cover `cls` and `mean` pooling.

Stage 6: optimized attention

- Lower full attention to a fused SDPA/Flash-style provider.
- Add local bidirectional sliding-window provider.
- Add unpadding/varlen only after dense padded parity is stable.

Stage 7: packed sequence support

- Admit packed flattened FlashAttention behavior via explicit `position_ids` and `cu_seqlens` metadata.
- Keep this behind a feature gate until parity is proven.

## 12. Parity and validation plan

Random tensor tests:

- RoPE cos/sin for full and sliding theta values.
- `apply_rotary_pos_emb` upcast/downcast behavior.
- GLU MLP split order and activation multiply.
- Bidirectional sliding mask predicate.
- QKV packed weight split.

Single-layer parity:

- One full-attention layer and one sliding-attention layer from a tiny config.
- Cases: no padding, right padding, left padding, explicit position IDs.

After-N-layer parity:

- Tiny 4-layer model from `hf-internal-testing/tiny-random-ModernBertModel`.
- Production base first 2/4/22 layers if partial graph extraction is available.

Encoder/projector parity:

- `ModernBertModel` hidden states for a short tokenizer example. HF tests use `"Hello World!"` with expected shape `[1, 5, 768]`.

Masked LM parity:

- `ModernBertForMaskedLM` logits shape `[B, S, 50368]`; HF integration checks base model values with `attn_implementation="sdpa"`.

Head parity:

- Token classification tiny checkpoint logits `[B, S, 2]`.
- Sequence classification tiny checkpoint logits `[B, 2]`.
- QA start/end logits `[B, S]` after split/squeeze.
- Multiple choice logits `[B, choices]`.

Recommended tolerances:

- fp32 eager reference: `rtol=1e-4`, `atol=1e-4` for short sequences, matching HF integration style.
- fp16/bf16 optimized attention: start with `rtol=5e-3`, `atol=5e-3`, tighten per backend after attention parity is characterized.

## 13. Performance probes

- Tokenizer throughput at sequence lengths 128, 512, 2048, 8192.
- Encoder-only throughput for base and large.
- Sequence-length sweep for full-only layers versus sliding layers: 128, 512, 1024, 4096, 8192.
- Sliding-window sweep if custom configs change `local_attention`.
- Batch-size sweep for padded batches and no-padding batches.
- Attention backend comparison: eager dense, SDPA mask, full FlashAttention, sliding FlashAttention/local provider.
- Unpad/varlen overhead probe: padded density 25%, 50%, 75%, 100%.
- MLM decoder cost probe: full vocab logits versus last/masked-token-only projection if a future sparse inference mode is desired.
- Classification head overhead: mean pooling and prediction head relative to encoder.
- Memory probe: attention activation footprint for full layers at 8192 tokens; no KV cache memory probe is needed.

## 14. Skip/defer list

- Training losses and gradient checkpointing.
- `sparse_prediction=True` label-filtered MLM path.
- `output_attentions=True` except eager debug fallback.
- Packed flattened sequences via non-monotonic `position_ids`.
- FlashAttention varlen unpad/pad-back optimization, until dense/masked parity is stable.
- Advanced RoPE scaling variants not present in representative configs.
- Multi-GPU/tensor parallel.
- Quantized checkpoints.
- Any autoregressive generation/KV-cache support.
- Layout translation passes; this is a text encoder with `[B, S, H]` semantics, so no NHWC/NCHW rewrite is relevant.

## 15. Final implementation checklist

- [ ] Parse `ModernBertConfig`, including `layer_types`, `global_attn_every_n_layers`, and old theta fields.
- [ ] Load token embedding and preserve MLM decoder tie when enabled.
- [ ] Implement embedding `Embedding -> LayerNorm`.
- [ ] Implement LayerNorm with optional bias.
- [ ] Implement packed QKV projection split in Q/K/V order.
- [ ] Implement ModernBERT RoPE for full and sliding layer types.
- [ ] Implement bidirectional full attention mask.
- [ ] Implement bidirectional sliding-window attention mask.
- [ ] Add dense eager attention reference path.
- [ ] Add SDPA/Flash-style optimized full attention path.
- [ ] Add optimized local bidirectional sliding attention path.
- [ ] Implement GLU MLP fusion or primitive composition.
- [ ] Implement residual block order, including layer-0 identity attention norm.
- [ ] Implement final encoder LayerNorm.
- [ ] Implement masked LM prediction head and tied decoder projection.
- [ ] Implement optional sequence classification pooling modes.
- [ ] Implement optional token classification head.
- [ ] Implement optional QA split/squeeze head.
- [ ] Implement optional multiple-choice flatten/pool/reshape head.
- [ ] Add one-layer full/sliding parity tests.
- [ ] Add tiny full-model parity tests.
- [ ] Add base masked-LM short-prompt parity test.
- [ ] Add backend/performance probes for full versus sliding attention.
