# Flaubert Transformers audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: flaubert family; representative ids: flaubert/flaubert_small_cased, flaubert/flaubert_base_uncased, flaubert/flaubert_base_cased, flaubert/flaubert_large_cased
Config source: official Hugging Face config.json URLs fetched 2026-05-13; compact snapshot in representative_configs.json
Source files inspected:
  X:/H/transformers/src/transformers/models/flaubert/modeling_flaubert.py
  X:/H/transformers/src/transformers/models/flaubert/configuration_flaubert.py
  X:/H/transformers/src/transformers/models/flaubert/tokenization_flaubert.py
Any missing files or assumptions:
  special_tokens_map.json returned 404 for the representative official repos; source tokenizer defaults are used for special-token behavior.
  No remote-code files are required for the in-library Flaubert source.
```

Primary source URLs at the inspected commit:

- [modeling_flaubert.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/flaubert/modeling_flaubert.py)
- [configuration_flaubert.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/flaubert/configuration_flaubert.py)
- [tokenization_flaubert.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/flaubert/tokenization_flaubert.py)

Representative config URLs:

- [flaubert/flaubert_small_cased](https://huggingface.co/flaubert/flaubert_small_cased/raw/main/config.json)
- [flaubert/flaubert_base_uncased](https://huggingface.co/flaubert/flaubert_base_uncased/raw/main/config.json)
- [flaubert/flaubert_base_cased](https://huggingface.co/flaubert/flaubert_base_cased/raw/main/config.json)
- [flaubert/flaubert_large_cased](https://huggingface.co/flaubert/flaubert_large_cased/raw/main/config.json)

## 2. High-level architecture

Flaubert is a text-only XLM-style Transformer encoder. The useful first DinoML runtime target should be encoder + masked-LM logits through `FlaubertWithLMHeadModel`, because all official representative checkpoints declare that architecture and `causal=false`.

Dataflow:

```text
Moses/BPE tokenization -> input_ids/attention_mask/langs/token_type_ids/position_ids
  -> token + position + optional language + optional token-type embeddings
  -> embedding LayerNorm + dropout + padding zeroing
  -> repeated encoder block
  -> last hidden state
  -> optional LM/classification/token/QA/multiple-choice head
```

The neural graph is rank-3 sequence-first hidden state work, `[batch, seq, hidden]`. There is no vision/audio layout work and no NHWC/NCHW tensor region. Layout guidance: protect the whole model from image-style layout translation; only local matmul/attention internal layouts such as `[B,S,H] -> [B,heads,S,D]` should be compiler-owned rewrites.

## 3. Important config dimensions

Source defaults from `FlaubertConfig`:

| Field | Default | Source/runtime effect |
|---|---:|---|
| `vocab_size` / `n_words` | 30145 | token embedding rows and LM projection classes |
| `emb_dim` / `hidden_size` | 2048 | hidden width |
| `n_layers` | 12 | encoder block count |
| `n_heads` | 16 | MHA head count |
| `head_dim` | `emb_dim // n_heads` | asserted integral in attention |
| FFN width | `4 * emb_dim` | computed in `FlaubertModel`, not a config field |
| `max_position_embeddings` | 512 | learned/sinusoidal position table length |
| `dropout`, `attention_dropout` | 0.1 | inference disabled |
| `gelu_activation` | true | otherwise ReLU FFN |
| `sinusoidal_embeddings` | false | if true, initialization fills fixed sinusoidal table |
| `causal` | false | if true, triangular self-attention mask |
| `pre_norm` | false | changes block norm/residual order |
| `layer_norm_eps` | 1e-12 | LayerNorm epsilon |
| `asm` | false | if true, LM head uses `AdaptiveLogSoftmaxWithLoss` |
| `n_langs`, `use_lang_emb` | 1, true | language embeddings only created when `n_langs > 1 and use_lang_emb` |

Representative checkpoint sweep:

| Model id | Layers | Hidden | Heads | Head dim | FFN | Vocab | Pos | Norm order | LN eps | LayerDrop | Cased |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---|
| `flaubert_small_cased` | 6 | 512 | 8 | 64 | 2048 | 68729 | 512 | pre-norm | 1e-6 | 0.2 | yes |
| `flaubert_base_uncased` | 12 | 768 | 12 | 64 | 3072 | 67542 | 512 | post-norm | 1e-12 | 0.0 | no |
| `flaubert_base_cased` | 12 | 768 | 12 | 64 | 3072 | 68729 | 512 | post-norm | 1e-12 | 0.0 | yes |
| `flaubert_large_cased` | 24 | 1024 | 16 | 64 | 4096 | 68729 | 512 | pre-norm | 1e-6 | 0.2 | yes |

Config-only historical/training fields such as `amp`, `fp16`, `bptt`, `word_mask`, `word_shuffle`, `max_batch_size`, and `tokens_per_batch` are not read by the inspected inference modeling path.

## 3a. Family variation traps

- `pre_norm` changes the block graph: small and large official configs are pre-norm; base cased/uncased are post-norm.
- `layer_norm_eps` is `1e-6` for small/large and `1e-12` for base variants; parity tests must use the checkpoint value.
- `gelu_activation=false` switches FFN activation to ReLU, though the representative configs use GELU.
- `sinusoidal_embeddings=true` is implemented only as position-table initialization, not dynamic sinusoidal generation during forward.
- `causal=true` is implemented with a triangular mask, but inspected official Flaubert checkpoints are non-causal encoders.
- `is_encoder=false` is rejected by source construction with `NotImplementedError`; DinoML should reject decoder Flaubert configs for this family report.
- `asm=true` routes the LM head through `nn.AdaptiveLogSoftmaxWithLoss`; official representative configs use `asm=false`, so first integration should reject `asm=true`.
- `n_langs > 1 and use_lang_emb=true` adds `lang_embeddings(langs)`. Representative configs are monolingual French (`n_langs=1`), so `langs` is normally unused by the base checkpoints.
- `token_type_ids` are embedded through the same token embedding matrix, not a separate segment embedding table. This is XLM/Flaubert-specific and must not be translated as BERT token-type embeddings.
- `attention_mask` has source convention 1=keep, 0=mask. If omitted, lengths are derived from `input_ids != pad_index`.
- `summary_type="attn"` is explicitly not implemented for sequence summary heads.
- No NHWC/NCHW conversion applies. Any generic layout pass must keep sequence axis `dim=1` reductions/means and `dim=-1` softmax/logit axes intact.

## 4. Operator coverage checklist

Tensor/layout ops:

- Rank/shape validation for `input_ids [B,S]` or `inputs_embeds [B,S,H]`.
- Embedding lookup: token table `[V,H]`, position table `[P,H]`, optional language table `[n_langs,H]`.
- Broadcast/add for embeddings: `[B,S,H] + [B,S,H]`.
- Optional token-type embedding lookup through token embedding table with `token_type_ids [B,S]`.
- Padding zeroing: `tensor *= mask.unsqueeze(-1).to(dtype)`.
- Reshape/view/transpose/contiguous for attention: `[B,S,H] -> [B,S,heads,D] -> [B,heads,S,D]`, then back.
- Gather/index/select for sequence summary `first`, `last`, `mean`, `cls_index`; QA heads also use `gather`, `expand`, `topk`.
- Multiple-choice flatten: `[B,C,S] -> [B*C,S]`, then logits reshape `[B*C,1] -> [B,C]`.

Neural network primitives:

- LayerNorm over last dim with checkpoint epsilon.
- Dense Linear with bias for all attention projections, FFN projections, LM projection, and heads.
- GELU or ReLU FFN activation.
- Tanh in optional sequence/QA heads.
- Dropout is present in source but disabled for inference.

Attention primitives:

- Dense MHA self-attention, no GQA/MQA: Q/K/V each `Linear(H -> H)` with bias.
- Matmul QK: `[B,heads,Q,D] x [B,heads,D,K] -> [B,heads,Q,K]`.
- Scale before score matmul by `1 / sqrt(head_dim)`.
- Boolean mask expansion and masked fill with dtype minimum.
- Softmax explicitly computed as `softmax(scores.float(), dim=-1).type_as(scores)`.
- Matmul AV: `[B,heads,Q,K] x [B,heads,K,D] -> [B,heads,Q,D]`.

Position/custom math:

- Learned absolute positions by default.
- Optional fixed sinusoidal table at init.
- No RoPE, ALiBi, relative bias, sliding window, or block-sparse attention.

Generation/cache ops:

- Source imports `GenerationMixin` and has cache-aware attention, but `FlaubertModel` rejects decoder construction and official configs are encoder-only. Treat cache/pseudo-generation as optional/deferred for first parity.
- `prepare_inputs_for_generation` appends one mask token and creates `langs` filled with `lang_id`. This is masked-token generation scaffolding, not causal decode parity.

Preprocessing-coupled ops:

- Python tokenizer uses sacremoses normalization/tokenization plus BPE merges, with optional lowercasing from tokenizer config.
- Special token layout from tokenizer source: single sequence `<s> X </s>`; pair sequence `<s> A </s> B </s>`.

## 5. Layer/block breakdown

Embeddings:

```text
input_ids: [B,S] int64
lengths = sum(input_ids != pad_index, dim=1) unless supplied
mask, attn_mask = get_masks(S, lengths, causal, padding_mask=attention_mask)
position_ids = arange(S)[None, :].expand(B,S) unless supplied
x = token_embedding(input_ids)                         # [B,S,H]
x = x + position_embedding(position_ids)               # [B,S,H]
if langs is active: x = x + lang_embedding(langs)
if token_type_ids: x = x + token_embedding(token_type_ids)
x = LayerNorm(x, eps=layer_norm_eps)
x = dropout(x)
x = x * mask[..., None]
```

Post-norm block, repeated `n_layers` times when `pre_norm=false`:

```text
q = Linear(H -> H, bias)(x).view(B,S,heads,D).transpose(1,2)
k = Linear(H -> H, bias)(x).view(B,S,heads,D).transpose(1,2)
v = Linear(H -> H, bias)(x).view(B,S,heads,D).transpose(1,2)
attn = dense_attention(q, k, v, attn_mask)
x = LayerNorm(x + dropout(Linear(H -> H, bias)(attn)), eps)
x = LayerNorm(x + FFN(x), eps)
x = x * mask[..., None]
```

Pre-norm block, repeated `n_layers` times when `pre_norm=true`:

```text
y = LayerNorm(x, eps)
attn = dense_attention(qkv(y), attn_mask)
x = x + dropout(out_proj(attn))
y = LayerNorm(x, eps)
x = x + FFN(y)
x = x * mask[..., None]
```

FFN:

```text
FFN(x):
  y = Linear(H -> 4H, bias)(x)
  y = GELU(y) or ReLU(y)
  y = Linear(4H -> H, bias)(y)
  y = dropout(y)
```

Masked LM head:

```text
logits = Linear(H -> vocab_size, bias)(x)
```

The LM output weight is tied to `transformer.embeddings.weight` for the normal `asm=false` head. Preserve this as one logical parameter alias even if the linear lowering uses transposed GEMM storage.

Optional heads:

- Sequence classification: `FlaubertSequenceSummary` picks first/last/mean/cls-index state, optional projection to `num_labels`, optional activation/dropout.
- Token classification: dropout + `Linear(H -> num_labels)` at each token.
- Simple QA: `Linear(H -> num_labels)`, split start/end logits.
- XLNet-style QA: start linear, top-k beam, gather/expand start states, concat hidden/start, tanh, LayerNorm, end linear, answer-class linear path.
- Multiple choice: flatten choices, sequence summary, `Linear(num_labels -> 1)`, reshape to `[B,num_choices]`. The `logits_proj` input width follows the sequence-summary projection output, usually `num_labels`.

## 6. Attention requirements

Primary target requires non-causal dense encoder self-attention:

| Property | Flaubert source behavior |
|---|---|
| Causal | `config.causal`; official sweep false |
| Attention kind | self-attention in active encoder path |
| Cross-attention | class supports `kv`, but Flaubert decoder/cross-attn modules are commented/rejected |
| Heads | MHA, `n_heads`; no GQA/MQA |
| Head dim | `emb_dim // n_heads`, asserted integral |
| Q/K/V width | all `H`; output width `H` |
| Biases | all Q/K/V/O projections have bias |
| Mask | noncausal `[B,S]` keep mask or causal `[B,S,S]` lower-triangular keep mask |
| Mask fill | `(mask == 0)` then in-place fill with `torch.finfo(dtype).min` |
| Softmax | upcast to fp32, softmax over last dim, cast back |
| Packed/varlen | none in source |
| Sliding/local/block sparse | none |
| Position interaction | absolute position embeddings are added before attention; no per-head position math |
| KV cache | present in copied attention helper, but first Flaubert target should not rely on it |
| FlashAttention/SDPA | source uses eager matmul/softmax/matmul; FlashAttention replacement is safe only with fp32 softmax and mask semantics preserved |

Cache caveat: `FlaubertModel.forward` creates an `EncoderDecoderCache` when `cache is None`, then slices inputs by `cache.get_seq_length()` when `input_ids` is not `None`. For the encoder-first target, DinoML should disable cache semantics and run full sequence parity. If later supporting `causal=true`, cache keys/values are stored after K/V projection and reshape as `[B,heads,past_S,D]`; positions are absolute IDs before projection.

## 7. Position encoding and custom math

Default behavior is learned absolute position embeddings:

```text
position_ids = [0, 1, ..., S-1] expanded to [B,S]
x += position_embeddings(position_ids)
```

Optional sinusoidal initialization:

```python
def flaubert_sinusoidal_table(n_pos, dim):
    enc[pos, j] = pos / (10000 ** (2 * (j // 2) / dim))
    table[:, 0::2] = sin(enc[:, 0::2])
    table[:, 1::2] = cos(enc[:, 1::2])
    return table
```

This is a load/init-time table transform. It can be precomputed and treated as a constant embedding table; no runtime sin/cos op is required for normal inference.

Mask creation:

```python
def flaubert_masks(S, lengths, causal, attention_mask=None):
    arange = torch.arange(S)
    mask = attention_mask if attention_mask is not None else (arange[None, :] < lengths[:, None])
    if causal:
        attn_mask = arange[None, None, :] <= arange[None, :, None]  # [B,S,S] after repeat
    else:
        attn_mask = mask                                            # [B,S]
    return mask, attn_mask
```

Attention score math order:

```text
q = q / sqrt(head_dim)
scores = q @ k^T
scores.masked_fill(mask == 0, finfo(dtype).min)
weights = softmax(scores.float(), dim=-1).to(scores.dtype)
context = weights @ v
```

## 8. Preprocessing and input packing

CPU/data-pipeline:

- Flaubert tokenizer is Python BPE with sacremoses. It normalizes Unicode punctuation, applies Moses punctuation normalization and tokenization for French by default, optionally lowercases, then applies BPE merges.
- Representative tokenizer configs set `model_max_length=512`; uncased sets `do_lowercase=true`, cased variants false.
- Special token source defaults: `bos="<s>"`, `sep="</s>"`, `pad="<pad>"`, `cls="</s>"`, `mask="<special1>"`, `additional_special_tokens=<special0>...<special9>`.
- Build special-token sequence: single `<s> X </s>`; pair `<s> A </s> B </s>`.

GPU/runtime graph inputs:

- `input_ids [B,S]` or `inputs_embeds [B,S,H]`.
- `attention_mask [B,S]` optional, 1=valid token, 0=masked token.
- `lengths [B]` optional; ignored for multiple choice if supplied.
- `position_ids [B,S]` optional; default arange.
- `langs [B,S]` optional; only consumed if `n_langs > 1 and use_lang_emb`.
- `token_type_ids [B,S]` optional; source maps them through the token embedding table.

There is no image/audio processor, no placeholder scatter, no cu_seqlens/packed sequence metadata, and no NHWC/NCHW concern.

## 9. Graph rewrite / lowering opportunities

### Rewrite: separate Q/K/V linears -> packed QKV projection

Source pattern:

```text
q = Linear_q(x); k = Linear_k(x); v = Linear_v(x)
reshape each to [B,heads,S,D]
```

Replacement:

```text
qkv = Linear_packed(x, W=[Wq; Wk; Wv], b=[bq; bk; bv])
split last dim in order q, k, v
reshape/transpose each
```

Preconditions:

- Self-attention only (`kv is None`).
- All three projections are `Linear(H -> H)` with bias.
- Packed weight layout must preserve PyTorch linear convention: output rows concatenate Q rows, then K rows, then V rows. GEMM lowering may store transposed constants, but logical split order remains Q/K/V.
- No cross-attention cache reuse path.

Failure cases:

- Future decoder/cross-attention path with `kv` separate from query.
- Any checkpoint with nonstandard projection sizes, not present in inspected source.

Parity test sketch: compare q/k/v tensors before attention for random `[B,S,H]` across base and large shapes.

### Rewrite: Flaubert attention -> fused dense encoder attention

Source pattern:

```text
scale q, q @ k.T, mask fill, fp32 softmax, cast, dropout, weights @ v
```

Replacement:

```text
FusedMHA(q, k, v, keep_mask, scale=1/sqrt(D), causal=config.causal, softmax_accum=float32)
```

Preconditions:

- Inference mode, dropout probability ignored.
- Mask convention exactly 1=keep, 0=mask.
- Noncausal `[B,S]` mask or causal lower-triangular mask.
- Return attentions not requested.
- Backend can preserve fp32 softmax accumulation and dtype-min masking.

Failure cases:

- `output_attentions=True`, training dropout, cross-attention helper path, or cache/update semantics.

Parity test sketch: random hidden and masks, compare post-attention output and edge cases with all padding rejected or guarded.

### Rewrite: embedding additions + LayerNorm

Source pattern:

```text
token_embed + position_embed + optional lang_embed + optional token_type_embed
LayerNorm
mask multiply
```

Replacement:

```text
FusedEmbeddingAddLayerNormMask
```

Preconditions:

- Static active embedding set known from config and provided inputs.
- Token-type embedding uses token table, not segment table.
- Padding row in token embedding remains zeroed by loading/init contract.

Failure cases:

- `inputs_embeds` path bypasses token embedding.
- Multilingual configs require active `langs` input and valid range guards.

### Rewrite: LM head tied embedding -> GEMM with embedding table

Source pattern:

```text
logits = Linear(H -> V, bias)(hidden)
proj.weight aliases token_embedding.weight
```

Replacement:

```text
logits = hidden @ embedding_weight.T + bias
```

Preconditions:

- `asm=false`.
- Preserve logical weight alias; updating/loading one parameter updates both.
- Optional last-token-only is not valid for masked-LM full-sequence scoring unless caller asks for selected positions.

Failure cases:

- `asm=true` adaptive log softmax.
- Classification heads with independent projections.

### Rewrite: sequence summary canonicalization

Source pattern:

```text
first/last/mean/cls_index gather -> optional linear -> optional activation
```

Replacement:

```text
SelectOrReduceSequence(summary_type) -> Linear/Tanh as configured
```

Preconditions:

- `summary_type != "attn"`.
- For `mean`, preserve axis `dim=1`; no layout translation.
- For `cls_index`, index shape/range guards.

## 10. Kernel fusion candidates

Highest priority:

- LayerNorm with checkpoint epsilon: appears at embeddings and twice per block; eps varies across checkpoints.
- Dense MHA prefill kernel: QKV pack, mask fill, fp32 softmax, AV matmul for `[B,S,H]` encoder workloads.
- GEMM + bias for QKV, O, FFN, and LM head; Flaubert is mostly dense linear algebra.
- FFN Linear + GELU/ReLU + Linear: important because FFN width is `4H`.

Medium priority:

- Embedding add + LayerNorm + padding mask multiply, especially for short 512-token sequences.
- LM head full-vocab GEMM, including tied-weight handling and optional selected-position logits for masked-token workloads.
- QA top-k/gather path only if extractive QA is a target.

Lower priority:

- Inference dropout elimination/canonicalization.
- Sequence summary head fusions.
- Causal/cache path support; not needed for official encoder-only checkpoints.
- Adaptive softmax (`asm=true`); representative configs do not require it.

## 11. Runtime staging plan

Stage 1: config and weight loading. Parse `FlaubertConfig`, reject `is_encoder=false`, reject `asm=true` initially, derive `head_dim=emb_dim/n_heads` and `ffn_dim=4*emb_dim`, preserve tied LM weight alias.

Stage 2: embedding + one block parity. Implement token/position embedding, optional token-type and language additions, LayerNorm, mask creation, one post-norm and one pre-norm block.

Stage 3: full encoder parity. Run full sequence `[B,S] -> [B,S,H]` for small/base/large shape families with noncausal masks.

Stage 4: masked LM parity. Add tied LM projection and full-sequence logits. Stub training loss and adaptive softmax.

Stage 5: optional task heads. Add sequence classification, token classification, simple QA, multiple choice. Defer XLNet-style QA beam head unless needed.

Stage 6: optimized lowering. Add QKV packing, fused attention, FFN and embedding fusions. Keep source-faithful fallback.

Stage 7: optional causal/cache experiments. Only if a real Flaubert causal checkpoint is targeted; otherwise keep rejected/deferred.

## 12. Parity and validation plan

- Config parsing tests for the four representative configs, checking derived `head_dim`, FFN width, norm order, vocab size, and LN epsilon.
- Embedding parity with and without `attention_mask`, `token_type_ids`, and `inputs_embeds`.
- One-block random parity for post-norm base and pre-norm small/large, fp32 tolerance around `1e-5`.
- Full encoder parity against Transformers for `flaubert_small_cased` and `flaubert_base_uncased` with short and max-ish sequences, masks including left/right padding patterns.
- Mask edge tests: supplied `attention_mask` versus length-derived mask; causal mask separately if admitted.
- LM head parity: logits `[B,S,V]`, tied weight identity, cased/uncased vocab differences.
- Optional head parity: sequence summary first/last/mean/cls-index, token classification logits, simple QA split/squeeze, multiple-choice flatten/reshape.
- fp16/bf16 optimized-kernel tests should compare against Transformers eager with fp32 softmax accumulation; suggested fp16 tolerance `atol=2e-2, rtol=2e-2` after full encoder, tighter for single ops.

## 13. Performance probes

- Encoder throughput sweep by model size: small/base/large.
- Batch-size sweep at `S=32,128,512`.
- Sequence-length sweep for attention scaling and mask overhead.
- QKV packed versus three-GEMM projection timing.
- Fused attention versus eager GEMM/softmax/GEMM timing, with fp32 softmax.
- FFN GEMM + activation fusion timing.
- Embedding/LayerNorm/mask fusion timing for short sequences.
- LM head full-vocab GEMM timing, separated from encoder.
- Tokenizer throughput in CPU pipeline; sacremoses can dominate short-input latency.
- Memory probes for `[B,heads,S,S]` attention scores at base/large.

## 14. Skip/defer list

- Training losses, dropout, LayerDrop stochastic behavior, gradient checkpointing.
- `asm=true` adaptive softmax.
- Decoder/cross-attention path and autoregressive cache decode; source rejects decoder Flaubert.
- `output_attentions=True` optimized path; support only fallback if needed.
- `summary_type="attn"` because source raises `NotImplementedError`.
- XLNet-style QA beam/top-k head unless extractive QA parity is a first target.
- Multilingual `n_langs > 1` language embeddings unless a concrete checkpoint requires it.
- Quantization and packed-weight formats; none are source-coupled in inspected Flaubert.
- Any NHWC/NCHW layout translation; not applicable.

## 15. Final implementation checklist

- [ ] Parse `FlaubertConfig` and derive `head_dim`, `ffn_dim=4*emb_dim`.
- [ ] Reject initially unsupported `is_encoder=false`, `asm=true`, and `summary_type="attn"`.
- [ ] Load token, position, optional language, attention, FFN, LayerNorm, and LM head weights.
- [ ] Preserve tied `pred_layer.proj.weight` / `transformer.embeddings.weight` alias.
- [ ] Implement embedding add + LayerNorm + padding mask multiply.
- [ ] Implement Flaubert mask creation with 1=keep attention masks.
- [ ] Implement post-norm and pre-norm encoder blocks.
- [ ] Implement dense MHA with fp32 softmax accumulation and dtype-min mask fill.
- [ ] Implement FFN `Linear -> GELU/ReLU -> Linear`.
- [ ] Implement masked-LM projection for `asm=false`.
- [ ] Add optional sequence classification, token classification, simple QA, and multiple-choice heads.
- [ ] Add QKV packing rewrite with Q/K/V split-order tests.
- [ ] Add fused attention rewrite with mask/causal guards.
- [ ] Add parity tests for small cased, base uncased/cased, and large cased configs.
- [ ] Benchmark encoder-only, LM-head-only, QKV packing, attention, FFN, and tokenizer throughput.
