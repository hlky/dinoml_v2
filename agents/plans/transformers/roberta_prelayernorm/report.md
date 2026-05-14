# DinoML Transformers audit: roberta_prelayernorm

## 1. Source basis

Transformers commit/version:

- Local checkout `X:/H/transformers` at commit `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`.
- Source doc says the model was added to Transformers on 2022-12-19.

Model id:

- Primary representative checkpoint: [`andreasmadsen/efficient_mlm_m0.40`](https://huggingface.co/andreasmadsen/efficient_mlm_m0.40).
- Additional open configs inspected are recorded under `_sources/`.

Config source:

- Local config class: `X:/H/transformers/src/transformers/models/roberta_prelayernorm/configuration_roberta_prelayernorm.py`.
- HF raw `config.json` snapshots fetched on 2026-05-13 for efficient-MLM, downstream classification, and smaller open variants.

Source files inspected:

- `configuration_roberta_prelayernorm.py`
- `modeling_roberta_prelayernorm.py`
- `convert_roberta_prelayernorm_original_pytorch_checkpoint_to_pytorch.py`
- `docs/source/en/model_doc/roberta-prelayernorm.md`
- `tests/models/roberta_prelayernorm/test_modeling_roberta_prelayernorm.py`

Any missing files or assumptions:

- No tokenizer files were downloaded. Tokenizer coupling is RoBERTa-style text IDs plus `attention_mask`, `token_type_ids`, and special token IDs from config.
- No model weights were inspected. Weight tying is inferred from source `_tied_weights_keys` and config/default `tie_word_embeddings`.
- First useful DinoML runtime target for this report: base encoder plus masked-LM head. Causal LM and encoder-decoder use are implemented in source but should be later-stage because they add cache, causal masks, and optional cross-attention.

## 2. High-level architecture

RoBERTa-PreLayerNorm is a text-only Transformer encoder by default. It is architecturally close to RoBERTa/BERT MHA encoder blocks, but each block normalizes before attention and before FFN, then adds residuals after the output projections. The base model also applies a final `LayerNorm` after all encoder layers.

Dataflow:

```text
token ids / optional token_type_ids / optional position_ids
-> word + token-type + learned absolute-position embeddings
-> embedding LayerNorm
-> repeated pre-LayerNorm encoder blocks
-> final LayerNorm
-> optional pooler or task head
```

Stage decomposition:

- CPU/data pipeline: tokenizer, padding, `attention_mask`, optional token type IDs, optional explicit position IDs.
- GPU/runtime: embeddings, position-ID-derived gathers when defaults are used, dense encoder stack, final norm, and selected head.
- Independently stageable outputs: base last hidden state `[B, S, H]`, pooled first-token state `[B, H]`, masked-LM logits `[B, S, vocab]`, classification logits, token logits, QA start/end logits.
- Later-stage generation path: source can run as a causal decoder when `is_decoder=True`; this is not required for masked-LM parity.

## 3. Important config dimensions

Source defaults from `RobertaPreLayerNormConfig`:

| Field | Default | Runtime effect |
| --- | ---: | --- |
| `vocab_size` | 50265 | Word embedding rows and LM decoder output rows |
| `hidden_size` | 768 | Encoder width |
| `num_hidden_layers` | 12 | Repeated pre-LN block count |
| `num_attention_heads` | 12 | MHA head count |
| `head_dim` | `hidden_size / num_attention_heads` | Source requires divisibility unless `embedding_size` exists |
| `intermediate_size` | 3072 | FFN expansion |
| `hidden_act` | `gelu` | FFN activation |
| `max_position_embeddings` | 512 | Learned absolute position rows |
| `type_vocab_size` | 2 | Token type embedding rows |
| `layer_norm_eps` | `1e-12` | All LayerNorm eps unless checkpoint overrides |
| `is_decoder` | false | Enables causal mask/cache path |
| `add_cross_attention` | false | Adds cross-attention only when decoder |
| `use_cache` | true | Effective only for decoder mode |
| `tie_word_embeddings` | true | LM decoder weight tied to token embedding by default |

Representative checkpoint sweep:

| Checkpoint | Arch | H | Layers | Heads | Head dim | FFN | Vocab | Max pos | Type vocab | LN eps | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `andreasmadsen/efficient_mlm_m0.15` | MaskedLM | 1024 | 24 | 16 | 64 | 4096 | 50265 | 514 | 1 | `1e-5` | Large efficient-MLM shape |
| `andreasmadsen/efficient_mlm_m0.20` | MaskedLM | 1024 | 24 | 16 | 64 | 4096 | 50265 | 514 | 1 | `1e-5` | Same operator shape as m0.15 |
| `andreasmadsen/efficient_mlm_m0.40` | MaskedLM | 1024 | 24 | 16 | 64 | 4096 | 50265 | 514 | 1 | `1e-5` | Primary integration target |
| `andreasmadsen/efficient_mlm_m0.80` | MaskedLM | 1024 | 24 | 16 | 64 | 4096 | 50265 | 514 | 1 | `1e-5` | Same operator shape |
| `ThomasLI/efficient_mlm_m0.40-finetuned-classification` | SequenceClassification | 1024 | 24 | 16 | 64 | 4096 | 50265 | 514 | 1 | `1e-5` | Adds classifier head |
| `cambridge-climb/baseline-roberta_pre_layer_norm-model` | MaskedLM | 500 | 10 | 10 | 50 | 2000 | 8192 | 512 | 2 | `1e-5` | Smaller, untied config snapshot says `tie_word_embeddings=false` |
| `mist-models/mist-28M-ti624ev1` | MaskedLM | 512 | 8 | 8 | 64 | 2048 | 165 | 2048 | 2 | `1e-12` | Long max positions, tiny vocab |

## 3a. Family variation traps

- The in-library source uses `model_type = "roberta-prelayernorm"`, while the Python directory is `roberta_prelayernorm`. Admission should accept the config model type, not infer from directory spelling.
- Efficient-MLM checkpoints omit `is_decoder`, `add_cross_attention`, and `tie_word_embeddings`; source defaults make them encoder-only with tied LM weights.
- Source default `max_position_embeddings=512`, but efficient-MLM configs use 514. Position guard must use checkpoint config, not class default.
- `type_vocab_size` varies between 1 and 2. Default generated token types are zeros, but explicit caller token type IDs must be range-checked against this field.
- `layer_norm_eps` varies between `1e-5` and `1e-12`; fused LayerNorm must preserve eps.
- Smaller open configs change `hidden_size`, `num_heads`, `head_dim`, vocab, and position table size; do not hard-code RoBERTa-large dimensions.
- `hidden_size` must be divisible by `num_attention_heads` for current source. No GQA/MQA exists; KV heads equal query heads.
- Pre-LayerNorm ordering is the defining trap: each encoder layer has attention-LN before QKV and FFN-LN before first FFN linear, plus a final model-level LN. A post-LN BERT rewrite is wrong.
- Causal LM path is optional and only correct when `config.is_decoder=True`; source warns if `RobertaPreLayerNormForCausalLM` is used standalone without decoder mode.
- Decoder cross-attention exists only when both `is_decoder=True` and `add_cross_attention=True`; encoder-only masked-LM configs do not instantiate cross-attention weights.
- `chunk_size_feed_forward` is read from inherited config. For inference integration, require `chunk_size_feed_forward == 0` initially or lower chunked FFN as explicit static sequence chunks.
- Source advertises FlashAttention, SDPA, and flex attention support through Transformers attention backend dispatch. DinoML should still canonicalize to its own attention ABI with masks and layout guards.

## 4. Operator coverage checklist

Tensor/layout ops:

- Shape/rank guards for `input_ids [B,S]` or `inputs_embeds [B,S,H]`, exactly one required.
- Embedding gather for word, token type, and position tables.
- `ne(input_ids, pad)`, cast to int, `cumsum(dim=1)`, multiply by mask, add padding offset for default position IDs.
- `arange`, `unsqueeze`, `expand`, and optional `gather(dim=1)` for default token type buffer indexed by position IDs.
- `view`/`reshape`, `transpose(1,2)`, final `contiguous` after attention layout `[B, heads, S, head_dim] -> [B,S,H]`.
- Slice/index for first-token pooling/classification: `hidden_states[:, 0, :]`.
- Split/squeeze for QA logits: `[B,S,2] -> [B,S] + [B,S]`.
- Multiple-choice flatten/unflatten: `[B,C,S] -> [B*C,S]`, logits `[B*C,1] -> [B,C]`.

Neural network primitives:

- Dense linear with bias for all projections and heads.
- LayerNorm over last dim with checkpoint eps.
- GELU for FFN and LM head.
- Tanh for pooler and classification head.
- Dropout is training-only; inference should compile as identity.
- Residual add after attention output projection and FFN output projection.

Attention primitives:

- Dense MHA self-attention, noncausal for primary encoder target.
- Optional causal self-attention with KV cache for decoder mode.
- Optional dense cross-attention when decoder plus encoder hidden states.
- Attention mask addition before softmax.
- Softmax over key dimension.
- Attention backend may be eager, SDPA, FlashAttention, or flex in Transformers; DinoML can lower to a single explicit attention primitive plus backend-specific kernels.

Position/custom math:

- Learned absolute position embeddings only; no RoPE, ALiBi, or relative bias.
- Position IDs depend on padding when `input_ids` are used.

Generation/cache ops:

- Defer for first masked-LM target.
- Decoder mode needs per-layer self-attention K/V cache shaped `[B, heads, T, head_dim]`.
- Encoder-decoder mode needs separate cross-attention cache and `is_updated[layer_idx]` flags.

Parameter aliasing:

- LM head decoder weight is tied to `roberta_prelayernorm.embeddings.word_embeddings.weight`.
- LM head decoder bias aliases `lm_head.bias` through `_tied_weights_keys`.
- If a config explicitly disables word embedding tying, DinoML must preserve distinct logical parameters.

## 5. Layer/block breakdown

Embedding block:

```text
if position_ids absent and input_ids present:
  mask = input_ids != pad_token_id
  position_ids = (cumsum(mask, dim=1) + past_length) * mask + pad_token_id
elif position_ids absent and inputs_embeds present:
  position_ids = arange(pad+1, pad+1+S).expand([B,S])

x = word_embedding[input_ids] or inputs_embeds
x = x + token_type_embedding[token_type_ids or zeros]
x = x + position_embedding[position_ids]
x = LayerNorm(x, eps)
x = DropoutIdentity(x)
```

Encoder block, repeated `num_hidden_layers`:

```text
attn_input = LayerNorm(x, eps)
q = Linear(H -> H, bias)(attn_input).view(B,S,heads,head_dim).transpose(1,2)
k = Linear(H -> H, bias)(attn_input).view(B,S,heads,head_dim).transpose(1,2)
v = Linear(H -> H, bias)(attn_input).view(B,S,heads,head_dim).transpose(1,2)
context = Attention(q, k, v, mask, scale=head_dim**-0.5)
context = context.transpose(1,2).reshape(B,S,H)
x = Linear(H -> H, bias)(context) + x

ffn_input = LayerNorm(x, eps)
h = Linear(H -> intermediate_size, bias)(ffn_input)
h = GELU(h)
x = Linear(intermediate_size -> H, bias)(h) + x
```

Base model output:

```text
sequence_output = final LayerNorm(x, eps)
pooler_output = tanh(Linear(H -> H)(sequence_output[:, 0, :])) if pooler enabled
```

Masked-LM head:

```text
h = Linear(H -> H)(sequence_output)
h = GELU(h)
h = LayerNorm(h, eps)
logits = Linear(H -> vocab_size)(h)  # tied decoder weight by default
```

Task heads:

- Sequence classification: first token, dropout identity, `Linear(H -> H)`, tanh, dropout identity, `Linear(H -> num_labels)`.
- Token classification: dropout identity, `Linear(H -> num_labels)` per token.
- Multiple choice: base model with pooler, dropout identity, `Linear(H -> 1)`, reshape to `[B,C]`.
- QA: `Linear(H -> num_labels)` per token, usually `num_labels=2`, split into start/end logits.

## 6. Attention requirements

Primary target attention:

- Type: encoder self-attention.
- Causality: bidirectional/noncausal.
- Heads: MHA, `num_key_value_heads == num_attention_heads`.
- Head dim: `hidden_size / num_attention_heads`.
- Q/K/V widths: each `H -> H`, bias enabled.
- Runtime tensor layout in source: Q/K/V become `[B, heads, S, head_dim]`.
- Masking: Transformers creates an additive bidirectional mask from `attention_mask`; eager path adds mask to scores before softmax.
- Packed/varlen: no source-specific packed sequence metadata.
- Local/sliding/block sparse: none.
- RoPE/ALiBi/relative bias: none.
- FlashAttention/SDPA compatibility: source routes through `ALL_ATTENTION_FUNCTIONS` and declares support for FlashAttention, SDPA, flex attention, and generic attention backend. Parity should be tested against eager math first, then fused attention.

Optional decoder attention:

- Self-attention becomes causal when `config.is_decoder=True`.
- `past_key_values` is a Transformers `DynamicCache` or the self-attention part of `EncoderDecoderCache`.
- Cache update stores K/V after linear projection and reshape/transpose; no RoPE is applied.
- Cross-attention is only instantiated when `add_cross_attention=True`; Q comes from decoder hidden states, K/V from `encoder_hidden_states`.
- Cross-attention cache stores projected encoder K/V once per layer and reuses them when `is_updated[layer_idx]` is true.

## 7. Position encoding and custom math

This family uses learned absolute position embeddings. The only custom position math is RoBERTa-style padding-aware position IDs.

```python
def roberta_prelayernorm_position_ids(input_ids, pad_token_id, past_len=0):
    mask = (input_ids != pad_token_id).to_int()
    incremental = (cumsum(mask, dim=1) + past_len) * mask
    return incremental.to_long() + pad_token_id
```

When `inputs_embeds` are supplied, padding cannot be inferred:

```python
def position_ids_from_embeds(batch, seq, pad_token_id):
    ids = arange(pad_token_id + 1, seq + pad_token_id + 1)
    return expand(ids[None, :], [batch, seq])
```

Precomputable:

- Position embedding table.
- Default all-zero token type ID buffer.
- Static arange for fixed sequence buckets.

Dynamic:

- Padding-aware `cumsum` if callers rely on default position IDs from `input_ids`.
- `past_len` offset in decoder cache mode.

## 8. Preprocessing and input packing

Text input contract:

- `input_ids`: `[B,S]` integer token IDs.
- `attention_mask`: optional `[B,S]`, converted by Transformers masking utilities into additive attention masks.
- `token_type_ids`: optional `[B,S]`; if absent, source uses a registered all-zero buffer and gathers by position IDs before expanding to `[B,S]`.
- `position_ids`: optional `[B,S]`; if absent, generated from `input_ids` or `inputs_embeds`.
- `inputs_embeds`: optional `[B,S,H]`; mutually exclusive with `input_ids`.

Special-token behavior:

- Config defaults: `pad_token_id=1`, `bos_token_id=0`, `eos_token_id=2`.
- Some open configs use different BOS/EOS values; token semantics are tokenizer-owned, not graph ops.
- Pooling/classification uses the first token position, conventionally `<s>`.

CPU/data-pipeline work:

- Tokenization, padding, truncation, and task-specific label handling.
- Generation controller for causal LM is outside the first target.

GPU/runtime work:

- Embedding lookups, default position ID math if not supplied, default token type handling if not supplied, attention mask application.

No image/audio/video/packed varlen metadata exists for this family.

## 9. Graph rewrite / lowering opportunities

### Rewrite: QKV linear packing

Source pattern:

```text
q = Linear(H -> H)(x_norm)
k = Linear(H -> H)(x_norm)
v = Linear(H -> H)(x_norm)
```

Replacement:

```text
qkv = Linear(H -> 3H)(x_norm)
split last dim into q, k, v
```

Preconditions:

- Same input tensor and dtype.
- All three projections have bias.
- Output split order must be `[q, k, v]`.
- Weight transform concatenates output rows in Q, K, V order for a PyTorch-style linear weight `[out, in]`.

Failure cases:

- Quantized/packed weights that require separate materialization.
- Any future config/source variant with projection-specific dimensions.

Parity test sketch:

- Random `x_norm`, compare separate linears versus packed linear plus split for fp32/fp16 tolerances.

### Rewrite: Linear + residual as GEMM epilogue

Source pattern:

```text
x = Linear(H -> H)(context) + residual
x = Linear(FFN -> H)(hidden) + residual
```

Replacement:

```text
GEMM bias add with residual add epilogue
```

Preconditions:

- Residual tensor shape exactly `[B,S,H]`.
- Flatten leading dims to `M = B*S`; `K` and `N` match projection.
- Dense row-major contiguous or a proven layout contract.

Failure cases:

- Non-contiguous source views not materialized/guarded.
- Residual broadcast beyond exact shape.

Parity test sketch:

- Compare flattened GEMM residual epilogue to PyTorch linear plus add for encoder output and FFN output shapes.

### Rewrite: LM head last-token-only for causal LM

Source pattern:

```text
logits = lm_head(hidden_states[:, slice_indices, :])
```

Replacement:

```text
slice hidden first, then apply LM head only to retained positions
```

Preconditions:

- `logits_to_keep` is int or static tensor indices accepted by DinoML.
- Causal LM mode only; masked-LM target needs all positions.

Failure cases:

- Training loss path with full labels.
- Dynamic index tensors not representable in first integration.

Parity test sketch:

- Compare logits for `logits_to_keep=1`, `0`, and static index tensor.

### Rewrite: attention layout fusion

Source pattern:

```text
Linear -> view(B,S,heads,D) -> transpose(1,2)
Attention -> transpose(1,2) -> reshape(B,S,H)
```

Replacement:

```text
Projection directly produces attention backend layout, backend returns [B,S,H] or fused output projection input layout
```

Preconditions:

- Source semantic axes remain `[batch, sequence, hidden]`.
- Only the local attention region is translated.
- Output projection receives the exact same logical `[B,S,H]`.

Layout guards:

- Do not globally reinterpret text tensors as channel-last; this is sequence layout, not image layout.
- `transpose(1,2)` is head/sequence axis movement and must not be confused with NCHW/NHWC rewrites.

Parity test sketch:

- Compare eager attention scores/output before and after layout-eliding lowering with masks enabled.

## 10. Kernel fusion candidates

Highest priority:

- LayerNorm over `[B,S,H]`, because this family uses embedding LN, two pre-LNs per layer, final model LN, and LM-head LN.
- Dense MHA attention with additive mask, because the encoder is attention-dominated and DinoML attention is currently a major gated gap.
- GEMM bias plus residual epilogues for attention output and FFN output projections.
- Linear bias GELU for FFN first projection and LM head dense.

Medium priority:

- Packed QKV projection.
- Embedding sum plus LayerNorm for fixed token type/position inputs.
- Masked-LM head `Linear -> GELU -> LayerNorm -> tied vocab GEMM`.
- Pooler/classifier `Linear -> tanh -> Linear` for classification workloads.

Lower priority:

- Default position ID `cumsum` fusion; useful only when callers do not precompute `position_ids`.
- Multiple-choice flatten/unflatten specialization.
- Decoder cache update kernels, since decoder mode is not the first target.

## 11. Runtime staging plan

Stage 1: config and weight loader.

- Parse `roberta-prelayernorm` configs.
- Admit encoder-only masked-LM checkpoints first: `is_decoder` false or missing, `add_cross_attention` false or missing.
- Preserve tied LM decoder/embedding aliasing.

Stage 2: one-block encoder parity.

- Implement embeddings, LayerNorm, MHA eager-equivalent attention, FFN, final LN.
- Stub dropout as identity.

Stage 3: full base encoder parity.

- Run all layers, return last hidden state.
- Add pooler optionally.

Stage 4: masked-LM head parity.

- Add LM head and tied vocab projection.
- Validate primary `andreasmadsen/efficient_mlm_m0.40` shape.

Stage 5: task heads.

- Add sequence classification, token classification, multiple choice, and QA as thin heads.

Stage 6: optimized lowering.

- Enable QKV packing, fused attention, LayerNorm kernels, and GEMM residual epilogues under guarded parity tests.

Stage 7: optional decoder modes.

- Add causal masks, self-attention KV cache, cross-attention, and `logits_to_keep`.

## 12. Parity and validation plan

- Config parsing tests for source defaults versus fetched configs, especially omitted `is_decoder`, omitted `tie_word_embeddings`, `type_vocab_size`, `max_position_embeddings`, and `layer_norm_eps`.
- Position ID tests matching source regression cases: padded input IDs preserve pad position, `inputs_embeds` generate sequential IDs.
- Embedding block parity with explicit and omitted `token_type_ids`/`position_ids`.
- Single LayerNorm parity for eps `1e-5` and `1e-12`.
- Single encoder block parity with random weights and masks, fp32 first.
- Full encoder parity against Transformers for a tiny config and one representative large config.
- Masked-LM logits parity for `andreasmadsen/efficient_mlm_m0.40`; source test expects output shape `[1,11,50265]`.
- Task head shape/parity tests for classification, token classification, multiple choice, and QA.
- Optional decoder parity later: cached versus no-cache next-token hidden states, matching Transformers test pattern.

Recommended tolerances:

- fp32: `rtol=1e-4`, `atol=1e-4` for full model logits/hidden slices.
- fp16/bf16: start with `rtol=5e-3`, `atol=5e-3`; tighten per fused-kernel behavior.

## 13. Performance probes

- Embedding plus position-ID preprocessing throughput with caller-supplied versus runtime-generated `position_ids`.
- Encoder-only throughput across `B` and `S` sweeps for 24-layer 1024-wide checkpoints.
- Attention backend comparison: eager GEMM/softmax/GEMM, SDPA-style fused attention, and future FlashAttention-style kernels.
- LayerNorm kernel timing by `(B*S, H)` for `H=500`, `512`, `1024`.
- FFN GEMM probes for `1024 -> 4096 -> 1024`, `512 -> 2048 -> 512`, and `500 -> 2000 -> 500`.
- Masked-LM head throughput and vocab projection cost for vocab 50265 versus 8192 versus 165.
- Task-head overhead for first-token classification and QA split/squeeze.
- Memory probes for activations and temp layout copies with and without attention layout fusion.
- Optional decoder later: prefill throughput, decode token/sec, and KV cache memory.

## 14. Skip/defer list

Safe to defer for first masked-LM/base-encoder integration:

- Training losses and label handling.
- Dropout randomness.
- Gradient checkpointing and output-capture hooks.
- Causal LM generation and `GenerationMixin`.
- KV cache and `logits_to_keep`.
- Encoder-decoder cross-attention.
- Beam search, sampling, and generation controller behavior.
- Flex attention backend specifics; use source-eager parity first.
- Chunked FFN when `chunk_size_feed_forward != 0`; reject initially unless explicitly lowered.
- Quantized or packed weight formats; none are source-coupled in this family.

## 15. Final implementation checklist

- [ ] Parse `RobertaPreLayerNormConfig` with source defaults and checkpoint overrides.
- [ ] Add admission guard for encoder-only first target: reject `is_decoder=True`, `add_cross_attention=True`, and nonzero `chunk_size_feed_forward`.
- [ ] Load embedding, position, token type, encoder, final LN, and LM head weights.
- [ ] Preserve tied LM decoder and word embedding aliasing.
- [ ] Implement embedding gather/sum plus padding-aware position IDs.
- [ ] Implement or enable LayerNorm over last dimension with checkpoint eps.
- [ ] Implement dense MHA encoder attention with additive bidirectional mask.
- [ ] Implement pre-LayerNorm encoder block ordering.
- [ ] Implement final model LayerNorm.
- [ ] Implement masked-LM head.
- [ ] Add optional pooler and task heads after masked-LM parity.
- [ ] Add QKV packing rewrite with `[q,k,v]` split-order tests.
- [ ] Add residual-GEMM epilogue rewrite for attention and FFN outputs.
- [ ] Add attention layout fusion guarded to the local attention region.
- [ ] Add parity tests for position IDs, one block, full encoder, and masked-LM logits.
- [ ] Add performance probes for LayerNorm, attention, FFN GEMMs, and vocab projection.

## Gated DinoML gaps

- LayerNorm is still a required runtime gap for this family; no encoder block can be correct without it.
- Attention is gated on a faithful dense MHA/additive-mask path before optimized SDPA/FlashAttention lowering.
- Embedding gather and integer position-ID generation need explicit coverage, including `cumsum(dim=1)`.
- Layout lowering must preserve text axes. Only the local QKV/head attention region is a layout-fusion candidate.
- Decoder mode is gated behind cache ABI, causal masks, and optional cross-attention caches; it should not block masked-LM encoder parity.
