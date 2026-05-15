# RoBERTa Transformers Audit

## 1. Source basis

Transformers commit/version:

- Local checkout `transformers` at commit `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`.

Model id:

- Primary: `FacebookAI/roberta-base`.
- Representative config sweep: `FacebookAI/roberta-base`, `FacebookAI/roberta-large`, `distilbert/distilroberta-base`, `hf-internal-testing/tiny-random-roberta`, `microsoft/codebert-base`.

Config source:

- `https://huggingface.co/FacebookAI/roberta-base/raw/main/config.json`
- `https://huggingface.co/FacebookAI/roberta-large/raw/main/config.json`
- `https://huggingface.co/distilbert/distilroberta-base/raw/main/config.json`
- `https://huggingface.co/hf-internal-testing/tiny-random-roberta/raw/main/config.json`
- `https://huggingface.co/microsoft/codebert-base/raw/main/config.json`
- Tokenizer metadata checked from corresponding `tokenizer_config.json`; `special_tokens_map.json` exists for tiny-random and CodeBERT but is absent on the FacebookAI and DistilRoBERTa repos checked here.

Source files inspected:

- `transformers/src/transformers/models/roberta/modeling_roberta.py`
- `transformers/src/transformers/models/roberta/modular_roberta.py`
- `transformers/src/transformers/models/roberta/configuration_roberta.py`
- `transformers/src/transformers/models/roberta/tokenization_roberta.py`
- `transformers/src/transformers/models/roberta/tokenization_roberta_old.py`
- Shared helpers: `transformers/src/transformers/masking_utils.py`, `transformers/src/transformers/integrations/sdpa_attention.py`, `transformers/src/transformers/integrations/flash_attention.py`.
- BERT comparison points: `transformers/src/transformers/models/bert/modeling_bert.py`, `configuration_bert.py`, `tokenization_bert.py`.

Any missing files or assumptions:

- `modeling_roberta.py` is generated from `modular_roberta.py`; future upstream edits should target the modular source, but this report uses the generated file because it contains the expanded runtime path.
- Primary target is `RobertaModel` encoder plus `RobertaForMaskedLM` inference. Classification, QA, multiple-choice, decoder/causal-LM, cross-attention, training losses, and generation are optional/deferred.
- No DinoML tests were run, per task scope.

## 2. High-level architecture

RoBERTa is a text-only bidirectional Transformer encoder with learned token, learned absolute position, and learned token-type embeddings. For the primary masked-LM target, the encoder output is passed through an MLM head:

```text
Byte-level BPE tokenization -> input_ids/attention_mask
-> word + token_type + absolute position embeddings
-> embedding LayerNorm/dropout
-> N bidirectional encoder layers
-> MLM dense + GELU + LayerNorm + tied vocab projection
-> logits [batch, seq, vocab]
```

The base encoder output shape is `[B, S, H]`. For inference, dropout is disabled by `eval()`, but the graph still contains dropout modules in the source. The model supports alternate decoder/cross-attention modes through config flags, but standard RoBERTa checkpoints use `is_decoder=false` and `add_cross_attention=false`.

## 3. Important config dimensions

Source defaults from `RobertaConfig`:

| Field | Default | Runtime meaning |
|---|---:|---|
| `vocab_size` | 50265 | Word embedding rows and MLM decoder output classes |
| `hidden_size` | 768 | Encoder width `H` |
| `num_hidden_layers` | 12 | Repeated encoder blocks |
| `num_attention_heads` | 12 | MHA heads |
| `head_dim` | 64 | Inferred as `hidden_size / num_attention_heads` |
| `intermediate_size` | 3072 | FFN expansion |
| `hidden_act` | `gelu` | FFN and MLM-head activation |
| `max_position_embeddings` | 512 | Source default only; common checkpoints use 514 |
| `type_vocab_size` | 2 | Source default only; common checkpoints use 1 |
| `pad_token_id` | 1 | Padding token and position embedding padding index |
| `bos_token_id` / `eos_token_id` | 0 / 2 | Also `cls` / `sep` in tokenizer convention |
| `layer_norm_eps` | `1e-12` | Source default only; common checkpoints use `1e-5` |
| `use_cache` | true | Only relevant to decoder mode; ignored for encoder target |
| `tie_word_embeddings` | true | MLM decoder weight tied to input word embeddings |

Representative checkpoint sweep:

| Model | Arch in config | Layers | H | Heads | Head dim | FFN | Vocab | Max pos | Token types | LN eps | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `FacebookAI/roberta-base` | `RobertaForMaskedLM` | 12 | 768 | 12 | 64 | 3072 | 50265 | 514 | 1 | `1e-5` | Common production/base MLM |
| `FacebookAI/roberta-large` | `RobertaForMaskedLM` | 24 | 1024 | 16 | 64 | 4096 | 50265 | 514 | 1 | `1e-5` | Same operators, bigger GEMMs |
| `distilbert/distilroberta-base` | `RobertaForMaskedLM` | 6 | 768 | 12 | 64 | 3072 | 50265 | 514 | 1 | `1e-5` | Same RoBERTa class, fewer layers |
| `hf-internal-testing/tiny-random-roberta` | omitted | 5 | 32 | 4 | 8 | 37 | 1000 | 512 | 16 | `1e-12` | Debug config; token-type table differs |
| `microsoft/codebert-base` | `RobertaModel` | 12 | 768 | 12 | 64 | 3072 | 50265 | 514 | 1 | `1e-5` | Encoder-only architecture in config |

Config-derived tokenizer notes:

- Common RoBERTa tokenizer `model_max_length` is 512 while common model `max_position_embeddings` is 514.
- IDs used by common configs: `<s>`/CLS/BOS `0`, `<pad>` `1`, `</s>`/SEP/EOS `2`, `<unk>` usually `3`, `<mask>` usually `50264`.

## 3a. Family variation traps

- Position ids are not BERT-style `0..S-1` slices. When `input_ids` are provided, RoBERTa computes non-pad positions by cumulative count and adds `pad_token_id`, so first real token gets position `2` when `pad_token_id=1`; pad tokens keep position `1`.
- Common checkpoints need 514 learned position rows for 512 tokenizer tokens because positions `0` and `1` are effectively reserved/unused for ordinary non-pad tokens, with `1` used as padding index.
- Tokenizer model inputs omit `token_type_ids`; `RobertaTokenizerFast.create_token_type_ids_from_sequences` returns all zeros. Common checkpoints set `type_vocab_size=1`, so passing BERT-like segment ids containing `1` is invalid.
- Source `RobertaConfig` defaults differ from common checkpoint configs: `max_position_embeddings=512`, `type_vocab_size=2`, `layer_norm_eps=1e-12`. Load checkpoint config, do not instantiate bare defaults for parity.
- `tiny-random-roberta` has `type_vocab_size=16`, `max_position_embeddings=512`, and `layer_norm_eps=1e-12`; it is useful for shape/debug testing but is not representative of production RoBERTa tokenizer coupling.
- Attention implementation is configurable through `config._attn_implementation`: eager, SDPA, FlashAttention, and flex masks are wired through shared Transformers interfaces. For initial parity, eager or an explicit bidirectional mask path is easier to match.
- `RobertaForCausalLM` and cross-attention are implemented, but standard RoBERTa MLM checkpoints are encoder-only. Do not let generation/cache requirements leak into the first masked-LM target.

## 4. Operator coverage checklist

Tensor/layout ops:

- Integer token inputs: `input_ids [B, S]`, optional `attention_mask [B, S]`, optional `position_ids [B, S]`, optional `token_type_ids [B, S]`.
- Embedding lookup for word, position, and token-type tables.
- Elementwise add of three embedding tensors.
- `cumsum(mask, dim=1)` for default RoBERTa position ids if position generation is in graph.
- `ne(input_ids, pad_token_id)`, cast bool/int/long, multiply by mask, add scalar.
- `view/reshape`, `transpose(1, 2)`, `transpose(2, 3)`, `contiguous` around attention.
- Slice `features[:, 0, :]` for classification head only, deferred.
- For masked LM, full `[B, S, V]` vocab logits. Optional optimization can gather masked positions only if the caller contract changes.

Neural network primitives:

- LayerNorm over last dim with checkpoint eps, usually `1e-5`.
- Dense/GEMM with bias for Q, K, V, attention output, FFN up/down, MLM dense, MLM decoder.
- GELU exact/Transformers activation for FFN and MLM head.
- Residual add plus LayerNorm after attention output and FFN output. This is post-norm BERT/RoBERTa style.
- Tanh pooler only for `RobertaModel(add_pooling_layer=True)` and some optional heads; not required for `RobertaForMaskedLM`.

Attention primitives:

- Bidirectional self-attention, MHA only. No GQA/MQA.
- Q/K/V projections `[H -> H]` with bias.
- Attention scores `[B, heads, S, S] = Q @ K^T * head_dim^-0.5`.
- Additive float mask for eager path or boolean SDPA mask. Padding mask is key-side only for normal encoder attention.
- Softmax over last dim and attention/value matmul.

Position/special-token ops:

- RoBERTa absolute learned positions with padding-aware position ids.
- No RoPE, ALiBi, relative bias, sliding window, or local attention in the standard encoder.
- Special token sequence for pair inputs is `<s> A </s></s> B </s>`, with token-type ids still all zeros.

Preprocessing-coupled ops:

- Byte-level BPE tokenizer with prefix-space sensitivity.
- Padding id must be `1` for common checkpoints.
- Attention mask uses 1/true for keep and 0/false for pad in tokenizer output before model mask conversion.

## 5. Layer/block breakdown

Embedding block:

```text
input_ids: [B, S]
mask = input_ids != pad_token_id
position_ids = cumsum(mask, dim=1) * mask + pad_token_id
token_type_ids = zeros([B, S]) unless explicitly supplied

x = word_embedding[input_ids]              # [B, S, H]
x += token_type_embedding[token_type_ids]  # common table shape [1, H]
x += position_embedding[position_ids]      # common table shape [514, H]
x = LayerNorm(x, eps=config.layer_norm_eps)
x = dropout(x)                             # no-op in inference eval mode
```

Encoder block, repeated `num_hidden_layers` times:

```text
q = Linear(H -> H, bias=True)(x).view(B, S, heads, head_dim).transpose(1, 2)
k = Linear(H -> H, bias=True)(x).view(B, S, heads, head_dim).transpose(1, 2)
v = Linear(H -> H, bias=True)(x).view(B, S, heads, head_dim).transpose(1, 2)

attn = softmax((q @ k.transpose(-2, -1)) * head_dim^-0.5 + mask, dim=-1)
ctx = (attn @ v).transpose(1, 2).reshape(B, S, H)
x = LayerNorm(Linear(H -> H, bias=True)(ctx) + x)

ff = GELU(Linear(H -> intermediate_size, bias=True)(x))
x = LayerNorm(Linear(intermediate_size -> H, bias=True)(ff) + x)
```

Masked-LM head:

```text
h = Linear(H -> H, bias=True)(x)
h = GELU(h)
h = LayerNorm(h, eps=config.layer_norm_eps)
logits = Linear(H -> vocab_size, bias=True)(h)
```

The MLM decoder weight is tied to `roberta.embeddings.word_embeddings.weight`; the source also creates an LM-head bias parameter of shape `[vocab_size]`.

For base/large concrete shapes:

- Base: Q/K/V/output are `768 -> 768`, FFN `768 -> 3072 -> 768`, MLM decoder `768 -> 50265`.
- Large: Q/K/V/output are `1024 -> 1024`, FFN `1024 -> 4096 -> 1024`, MLM decoder `1024 -> 50265`.

## 6. Attention requirements

Required for primary target:

- Noncausal bidirectional self-attention.
- Multi-head attention with `num_key_value_heads == num_attention_heads`; no KV repeat.
- Head dim inferred by integer division. Source rejects `hidden_size % num_attention_heads != 0` unless an `embedding_size` escape hatch exists.
- Query scaling is passed as `head_dim ** -0.5`; eager path multiplies scores before mask addition.
- Eager path mask is additive float `[B, 1, S, S]` containing `0` for keep and `torch.finfo(dtype).min` for masked positions.
- SDPA path can skip mask creation for fully unpadded bidirectional attention; otherwise it creates boolean `[B, 1, S, S]` and passes it to `scaled_dot_product_attention`.
- FlashAttention path receives a 2D padding mask or `None`; standard RoBERTa has no causal flag for encoder mode.

Optional/deferred:

- Decoder mode creates causal masks and updates a DynamicCache. Cache tensor shape is `[B, heads, seen_tokens, head_dim]` for keys and values because RoBERTa uses full MHA. Cached keys are plain projected keys; there is no position rotation.
- Cross-attention is available only when `is_decoder=true` and `add_cross_attention=true`; not part of standard MLM checkpoints.
- `output_attentions=True` requires eager for exact attention weights. Optimized backends return `None` and warn.

## 7. Position encoding and custom math

RoBERTa uses learned absolute position embeddings with padding-aware ids. This is the most important non-BERT parity point:

```python
def roberta_position_ids(input_ids, padding_idx, past_key_values_length=0):
    mask = (input_ids != padding_idx).int()
    incremental = (cumsum(mask, dim=1) + past_key_values_length) * mask
    return incremental.long() + padding_idx
```

For common checkpoints with `pad_token_id=1`, non-padding tokens start at position id `2`; pads remain `1`. With tokenizer max length 512, a fully non-pad sequence uses positions `2..513`, which requires `max_position_embeddings=514`.

If `inputs_embeds` are supplied instead of `input_ids`, source cannot infer padding and creates sequential ids from `padding_idx + 1` to `padding_idx + S`.

No RoPE, ALiBi, relative position bias, sinusoidal encoding, or convolutional positional encoding is used in the standard RoBERTa encoder.

## 8. Preprocessing and input packing

Tokenizer/runtime contract:

- Tokenizer is byte-level BPE and is sensitive to leading spaces. `"Hello"` and `" Hello"` produce different first word pieces unless `add_prefix_space=True`.
- Fast tokenizer model inputs are `["input_ids", "attention_mask"]`; token type ids are not emitted by default.
- Single sequence special-token layout: `<s> tokens </s>`.
- Pair sequence layout: `<s> A </s></s> B </s>`.
- `create_token_type_ids_from_sequences` returns all zeros for both single and pair inputs.
- Common tokenizer configs checked here set `model_max_length=512`.
- The `<mask>` token in fast tokenizer is an added token with `lstrip=True`, so fill-mask text preprocessing can absorb the preceding space.

GPU/runtime inputs for initial DinoML integration:

- Required: `input_ids [B, S]` int64/int32 equivalent, `attention_mask [B, S]` bool/int keep mask.
- Optional first integration simplification: precompute `position_ids [B, S]` and all-zero `token_type_ids [B, S]` on CPU/data pipeline, then feed them as inputs or constants. This avoids requiring integer `cumsum` in the first graph.
- If DinoML owns tokenizer output, enforce or validate that `S <= tokenizer model_max_length` and position ids do not exceed `max_position_embeddings - 1`.

## 9. Graph rewrite / lowering opportunities

### Rewrite: all-zero token types -> constant embedding add

Source pattern:

```text
token_type_ids omitted -> all zeros -> token_type_embedding[token_type_ids]
```

Replacement:

```text
broadcast token_type_embedding[0] over [B, S, H], or fold into embedding add
```

Preconditions:

- `token_type_ids` is absent or statically known all-zero.
- All runtime token type ids are in range; for common checkpoints `type_vocab_size == 1`.

Failure cases:

- Debug or custom configs may use `type_vocab_size > 1` and caller-supplied nonzero ids.

Parity test sketch:

- Compare embedding output with omitted `token_type_ids` and explicit zeros for base and tiny-random configs.

### Rewrite: RoBERTa position ids in data pipeline

Source pattern:

```text
position_ids = cumsum(input_ids != pad_id, dim=1) * mask + pad_id
```

Replacement:

```text
CPU/tokenizer-side position_ids input -> position_embedding lookup
```

Preconditions:

- The graph boundary explicitly accepts `position_ids`.
- Padding id and max sequence contract come from loaded config/tokenizer.

Failure cases:

- `inputs_embeds` path uses sequential ids instead of pad-aware ids.
- Decoder mode adds `past_key_values_length`.

Parity test sketch:

- Padded batch with different sequence lengths; verify pad tokens use position `1` and first non-pad token uses `2`.

### Rewrite: QKV separate linears -> packed QKV projection

Source pattern:

```text
q = Linear(H,H)(x); k = Linear(H,H)(x); v = Linear(H,H)(x)
```

Replacement:

```text
qkv = Linear(H, 3H)(x) with packed weights/bias, then split last dim
```

Preconditions:

- Same input tensor `x`.
- All three projections have bias and same output shape `H`.
- Weight packing preserves source order `[q, k, v]`.

Failure cases:

- Cross-attention computes query from decoder hidden states and K/V from encoder hidden states; do not apply this rewrite there.

Parity test sketch:

- Random block weights, compare q/k/v tensors before attention.

### Rewrite: post-attention residual add + LayerNorm fusion

Source pattern:

```text
y = Linear(ctx)
x = LayerNorm(y + residual)
```

Replacement:

```text
fused bias/residual/LayerNorm, optionally after GEMM epilogue
```

Preconditions:

- Same dtype/shape `[B, S, H]`.
- LayerNorm axis is last dim.
- Inference dropout is disabled.

Failure cases:

- Training mode dropout between linear and residual changes semantics.

Parity test sketch:

- Compare fused and unfused block output for fp32 and fp16 at RoBERTa eps `1e-5`.

### Rewrite: full logits -> masked-position logits

Source pattern:

```text
logits = MLMHead(sequence_output)  # [B, S, V]
```

Replacement:

```text
gather masked positions -> MLMHead -> [num_masked, V]
```

Preconditions:

- Caller only needs logits at known masked positions.
- Public output contract is allowed to differ from `RobertaForMaskedLM`.

Failure cases:

- Hugging Face parity for `RobertaForMaskedLM` requires full `[B, S, V]` logits.

Parity test sketch:

- Compare gathered rows from full HF logits to optimized masked-position logits.

## 10. Kernel fusion candidates

Highest priority:

- Embedding sum + LayerNorm: removes multiple memory passes over `[B, S, H]`; must preserve RoBERTa position ids and token-type behavior.
- Packed QKV GEMM and attention layout handling: three `H -> H` projections dominate each layer; packing is safe for self-attention.
- Bidirectional encoder attention with padding mask: standard MHA shape, no RoPE/cache, good first FlashAttention/SDPA-style target.
- GEMM + bias + GELU for FFN up projection, and GEMM + bias + residual + LayerNorm for down/output projections.
- MLM head dense + GELU + LayerNorm, plus large vocab projection. For fill-mask serving, masked-position-only logits are the main bandwidth reduction if the API allows it.

Medium priority:

- Position id generation kernel if DinoML chooses to keep `cumsum` in graph.
- Token-type embedding fold for `type_vocab_size=1`.
- Attention mask creation/skip path: avoid materializing `[B, 1, S, S]` for unpadded bidirectional batches.

Lower priority:

- Pooler and classification heads.
- Decoder cache/causal LM support.
- Cross-attention.
- Training losses.

## 11. Runtime staging plan

Stage 1: config/tokenizer contract and weights

- Parse `RobertaConfig` from checkpoint, not source defaults.
- Load word, position, token-type, encoder, and MLM-head weights.
- Enforce `pad_token_id`, `max_position_embeddings`, `type_vocab_size`, and tokenizer max length consistency.

Stage 2: embedding parity

- Accept `input_ids`, `attention_mask`, and either generated or supplied `position_ids`.
- First implementation can precompute `position_ids` and all-zero token types outside the graph.

Stage 3: one-block encoder parity

- Implement/fuse post-norm MHA and FFN block for `[B, S, H]`.
- Validate base and tiny-random configs.

Stage 4: full encoder parity

- Run all layers with eager-equivalent bidirectional padding mask.
- Add base and large shape tests.

Stage 5: masked-LM head

- Implement full `[B, S, vocab]` logits for Hugging Face parity.
- Optionally add masked-position-only rewrite behind an explicit API or graph rewrite guard.

Stage 6: optimized attention/fusions

- Add no-padding mask skip, packed QKV, fused LayerNorm/residual, and optimized MHA kernels.

Stage 7: optional heads

- Add sequence/token classification, QA, multiple choice, pooler, and decoder/cache only if a target requires them.

## 12. Parity and validation plan

- Position-id unit tests:
  - unpadded `[0, ..., 2]` sequence produces positions starting at `2`.
  - right-padded and left-padded batches keep pad positions at `1`.
  - `inputs_embeds` path uses sequential positions and does not inspect padding.
- Tokenizer coupling tests:
  - special token layouts for single and pair inputs.
  - token type ids are all zeros.
  - leading-space tokenization changes ids.
- Embedding parity:
  - compare word + token-type + position + LayerNorm output against HF for base config.
- Attention parity:
  - one layer with no padding and with mixed padding; compare eager outputs.
  - test mask skip path separately from masked path.
- Full encoder parity:
  - tiny-random exact-ish fp32 smoke test.
  - roberta-base first N layers and full 12-layer hidden-state parity.
- Masked-LM parity:
  - compare `[B, S, V]` logits for fixed input with one or more `<mask>` tokens.
  - if masked-position rewrite is added, compare gathered full logits to optimized logits.

Suggested tolerances:

- fp32: `atol=1e-4`, `rtol=1e-4` for full encoder/logits after many layers; tighter for isolated ops.
- fp16/bf16: start around `atol=2e-2`, `rtol=2e-2` for end-to-end logits, then tighten per kernel.

## 13. Performance probes

- Tokenizer throughput: byte-level BPE plus special-token/padding generation on CPU.
- Embedding throughput: with graph-side versus precomputed position ids.
- Encoder throughput sweep: `B in {1, 8, 32}`, `S in {16, 64, 128, 512}` for base and large.
- Attention backend comparison: eager materialized mask, SDPA-style mask, no-padding skip, and fused custom attention.
- Padding sensitivity: same token count with packed short sequences versus padded batch.
- MLM head cost: full vocab logits over all positions versus masked-position-only logits.
- Memory bandwidth probes for residual + LayerNorm fusion.
- Large checkpoint GEMM probe: `H=1024`, FFN `4096`, vocab projection `1024 -> 50265`.

## 14. Skip/defer list

Safe to defer for first encoder/masked-LM integration:

- Training losses and label handling.
- Dropout randomness.
- Gradient checkpointing and chunked feed-forward execution.
- Sequence classification, token classification, QA, multiple choice, and pooler parity.
- `RobertaForCausalLM`, decoder cache, cross-attention, beam search, and generation helpers.
- `output_attentions=True` optimized-backend parity.
- FlashAttention/flex-attention specific mask internals beyond a faithful bidirectional padding mask.
- Quantization and multi-GPU/tensor parallel behavior.

## 15. Final implementation checklist

- [ ] Parse checkpoint `RobertaConfig` and reject bare-default mismatch hazards.
- [ ] Load tokenizer metadata needed for `pad_token_id`, special-token layout, and max length.
- [ ] Load/tie word embedding and MLM decoder weights.
- [ ] Implement RoBERTa padding-aware position id generation or accept precomputed `position_ids`.
- [ ] Implement word, position, and token-type embedding lookup plus embedding LayerNorm.
- [ ] Add all-zero token-type fast path for `type_vocab_size=1`.
- [ ] Implement post-norm encoder block with Q/K/V linears, bidirectional MHA, residuals, LayerNorm, FFN GELU.
- [ ] Implement full MLM head logits.
- [ ] Add packed QKV rewrite with self-attention-only guard.
- [ ] Add fused residual + LayerNorm kernels for attention and FFN outputs.
- [ ] Add attention mask parity tests for padded and unpadded batches.
- [ ] Add position-id parity tests for RoBERTa-specific padding behavior.
- [ ] Add one-block, full-encoder, and masked-LM parity tests against HF.
- [ ] Benchmark encoder and MLM-head bottlenecks separately.
