# XLM Transformers Audit

## 1. Source basis

```text
Transformers commit/version:
  Local checkout X:/H/transformers at b75feb2af64c3e29cbbc1bd859958c5432cc7ed4.

Model family:
  xlm

Primary runtime target:
  XLMModel encoder hidden states plus XLMWithLMHeadModel masked/causal-language-model logits.
  First DinoML target should be encoder/MLM inference. Causal generation is source-present
  but should be admitted as a separate follow-up because native decoder construction is blocked.

Config source:
  https://huggingface.co/FacebookAI/xlm-mlm-en-2048/raw/main/config.json
  https://huggingface.co/FacebookAI/xlm-mlm-100-1280/raw/main/config.json
  https://huggingface.co/FacebookAI/xlm-clm-enfr-1024/raw/main/config.json
  https://huggingface.co/FacebookAI/xlm-mlm-tlm-xnli15-1024/raw/main/config.json
  https://huggingface.co/hf-internal-testing/tiny-random-xlm/raw/main/config.json
  Snapshots are under agents/plans/transformers/xlm/_sources/configs/.

Source files inspected:
  X:/H/transformers/src/transformers/models/xlm/configuration_xlm.py
  X:/H/transformers/src/transformers/models/xlm/modeling_xlm.py
  X:/H/transformers/src/transformers/models/xlm/tokenization_xlm.py

Any missing files or assumptions:
  No remote code is required for native XLM. No DinoML imports, model execution,
  tests, or commits were run. The report is source-first and treats training
  losses/dropout as deferred unless they change inference output shape.
```

## 2. High-level architecture

XLM is a text-only Transformer encoder implementation with optional causal masks, optional learned language embeddings, learned or initialized sinusoidal absolute positions, post-residual LayerNorm, and an LM prediction layer that can be either dense tied projection or adaptive log softmax.

```text
Python tokenizer/data pipeline -> input_ids/attention_mask/langs/lengths
-> token + position + optional language embeddings
-> N x Transformer encoder block
-> optional LM / classification / QA / token / multiple-choice head
```

The useful first runtime is encoder plus LM logits. The tokenizer and language-id selection belong in the CPU/data pipeline. GPU/runtime starts at embedding gather, but may also own position-id arange, padding masks, causal masks, and optional `langs` embedding gather.

## 3. Important config dimensions

Source defaults from `XLMConfig`:

| Field | Default | Runtime meaning |
|---|---:|---|
| `vocab_size` / `n_words` | 30145 | token embedding rows and dense LM projection width |
| `emb_dim` / `hidden_size` | 2048 | hidden width `H` |
| `n_layers` | 12 | Transformer block count |
| `n_heads` | 16 | MHA head count |
| `head_dim` | 128 | inferred as `emb_dim / n_heads`; source asserts divisibility |
| FFN hidden | `4 * emb_dim` | source does not expose `intermediate_size`; uses fixed 4x expansion |
| `gelu_activation` | true | FFN activation: GELU if true, ReLU if false |
| `sinusoidal_embeddings` | false | initializes position table with sinusoid values and freezes via `requires_grad=False` |
| `causal` | false | selects noncausal padding mask vs triangular causal mask |
| `asm` | false | dense LM projection vs adaptive log softmax |
| `n_langs` | 1 | language embedding table rows when `use_lang_emb` and `n_langs > 1` |
| `use_lang_emb` | true | enables addition of `lang_embeddings(langs)` when present |
| `max_position_embeddings` | 512 | absolute position table length |
| `layer_norm_eps` | `1e-12` | embedding and block LayerNorm epsilon |
| `is_encoder` | true | native source raises if false |

Representative checkpoint sweep:

| Model id | Arch | Layers | H | Heads x D | FFN | Vocab | Langs | Lang emb | Causal | ASM | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---|---|---|---|
| `FacebookAI/xlm-mlm-en-2048` | `XLMWithLMHeadModel` | 12 | 2048 | 16 x 128 | 8192 | 30145 | 1 | gated unused | false | false | monolingual MLM; largest H in sampled configs |
| `FacebookAI/xlm-mlm-100-1280` | `XLMWithLMHeadModel` | 16 | 1280 | 16 x 80 | 5120 | 200000 | 100 | false | false | false | 100-language checkpoint, huge vocab, `use_lang_emb=false` |
| `FacebookAI/xlm-clm-enfr-1024` | `XLMWithLMHeadModel` | 6 | 1024 | 8 x 128 | 4096 | 64139 | 2 | true | false in config | false | model card says CLM, but current config says noncausal encoder |
| `FacebookAI/xlm-mlm-tlm-xnli15-1024` | `XLMWithLMHeadModel` | 12 | 1024 | 8 x 128 | 4096 | 95000 | 15 | true | false | false | TLM/MLM multilingual path with language embeddings |
| `hf-internal-testing/tiny-random-xlm` | unspecified | 5 | 32 | 4 x 8 | 128 | 30145 | 2 | true | false | false | debug config, `summary_type=last` |

The sampled official configs all set `asm=false` and `is_encoder=true`. No sampled config exercises `sinusoidal_embeddings=true`, `gelu_activation=false`, or `causal=true`, but the native source implements those gates.

## 3a. Family variation traps

- Native `XLMModel` rejects `is_encoder=false` at construction, even though the config class documents encoder/decoder mode and attention contains cross-attention/cache plumbing.
- `causal=true` is still meaningful in encoder mode: `get_masks` builds a triangular `[B,S,S]` mask instead of a padding-only `[B,S]` mask. DinoML should not equate "encoder class" with noncausal attention.
- `FacebookAI/xlm-clm-enfr-1024` is named/card-described as CLM, but its fetched config has `causal=false` and `is_encoder=true`. Treat the config as source of truth for native execution and record this as a checkpoint metadata trap.
- Language embeddings are conditional on both `n_langs > 1`, `use_lang_emb=true`, and runtime `langs` being provided. Some multilingual configs have many `lang2id` entries but `use_lang_emb=false`.
- `token_type_ids` are not BERT segment embeddings. Source adds `self.embeddings(token_type_ids)`, reusing the word embedding table, so admitting nonzero token type ids is a compatibility path that can accidentally read ordinary word rows.
- Position ids default to plain `arange(S)` sliced from a registered buffer; they are not padding-aware like RoBERTa. Padding is zeroed later by multiplying hidden states by `mask`.
- `asm=true` requires `config.asm_cutoffs` and `config.asm_div_value`, but these are not declared as strict fields in `XLMConfig` defaults and were absent from sampled public configs. Treat adaptive softmax as a gated config gap until a real checkpoint requires it.
- `sinusoidal_embeddings=true` is an initialization-time behavior for the position embedding table, not a per-forward sinusoid op.
- The attention module always uses dense MHA with `num_key_value_heads == num_attention_heads`; no GQA/MQA/sliding-window/local attention.
- The source uses `scores.float()` before softmax, then casts back to score dtype. Fused attention should preserve fp32 softmax accumulation behavior.
- `logits_to_keep` can slice hidden states before the LM projection. This is important for large-vocab CLM-like serving.
- `summary_type="attn"` raises `NotImplementedError`; classification configs with that setting must be rejected.

## 4. Operator coverage checklist

Tensor/layout ops:

- `Embedding(input_ids)`: `[B,S] -> [B,S,H]`.
- `Embedding(position_ids)`: `[B,S] -> [B,S,H]`; default `position_ids = arange(S)`.
- Optional `Embedding(langs)`: `[B,S] -> [B,S,H]`, table `[n_langs,H]`.
- Optional `Embedding(token_type_ids)` using the word embedding table.
- Mask construction from `lengths` or `attention_mask`: noncausal `[B,S]`; causal `[B,S,S]`.
- Elementwise add, dropout-as-noop in inference, hidden-state multiply by mask.
- Attention reshape/transpose: `[B,S,H] -> [B,A,S,D]` and back.
- Optional head ops: first/last/mean/`cls_index` gather, multiple-choice flatten/reshape, QA gather/expand/topk.

Neural network primitives:

- LayerNorm over last dim, epsilon usually `1e-12`.
- Bias linear/GEMM for Q/K/V/O: `Linear(H -> H)` each.
- FFN: `Linear(H -> 4H)`, GELU or ReLU, `Linear(4H -> H)`.
- Dense LM projection: `Linear(H -> vocab_size)` with weight tied to `transformer.embeddings.weight` when `asm=false`.
- Adaptive log softmax path when `asm=true`: `nn.AdaptiveLogSoftmaxWithLoss.log_prob(x)` for logits-equivalent output.

Attention primitives:

- Dense MHA self-attention.
- Optional cross-attention code path in `MultiHeadAttention(kv=...)`, but no active native XLM decoder body.
- Score order: project, reshape, `q / sqrt(D)`, `q @ k^T`, additive fill with dtype minimum where mask is false, fp32 softmax, `weights @ v`.

Position/custom math:

- Learned absolute position table by default.
- Optional sinusoidal initialization of the position table.
- No RoPE, ALiBi, relative bias, convolutional positions, or learned segment-relative attention.

Generation/cache ops:

- Non-primary. Attention can update `DynamicCache` or `EncoderDecoderCache` with K/V tensors shaped `[B,A,T,D]`.
- `XLMWithLMHeadModel.prepare_inputs_for_generation` appends one mask token and optionally fills `langs` with `config.lang_id`, then drops external masks/positions/token types.

Preprocessing-coupled ops:

- XLMTokenizer Python path: Moses punctuation normalization/tokenization, optional lowercase/accent stripping, BPE merges.
- Custom tokenizers for `zh`, `ja`, and `th` through optional external packages.
- `lang2id` / `id2lang` mapping from tokenizer/config must align with runtime `langs`.

## 5. Layer/block breakdown

Let `B=batch`, `S=sequence length`, `H=emb_dim`, `A=n_heads`, `D=H/A`, and `I=4H`.

Embedding block:

```text
input_ids[B,S] or inputs_embeds[B,S,H]
lengths = sum(input_ids != pad_index) unless supplied
mask, attn_mask = get_masks(S, lengths, causal, attention_mask)
position_ids = arange(S)[None, :] unless supplied

x = token_embedding[input_ids] + position_embedding[position_ids]
if langs is supplied and use_lang_emb and n_langs > 1:
    x += lang_embedding[langs]
if token_type_ids is supplied:
    x += token_embedding[token_type_ids]
x = LayerNorm(x)
x = x * mask[..., None]
```

Encoder block, repeated `n_layers` times:

```text
q = Linear(H -> H, bias=True)(x).view(B,S,A,D).transpose(1,2)
k = Linear(H -> H, bias=True)(x_or_kv).view(B,K,A,D).transpose(1,2)
v = Linear(H -> H, bias=True)(x_or_kv).view(B,K,A,D).transpose(1,2)
scores = (q / sqrt(D)) @ k.transpose(-2, -1)
scores.masked_fill_(mask == 0, finfo(dtype).min)
weights = softmax(scores.float(), dim=-1).type_as(scores)
ctx = weights @ v
attn = Linear(H -> H, bias=True)(ctx.transpose/reshape)
x = LayerNorm(x + attn)
x = LayerNorm(x + FFN(x))
x = x * mask[..., None]
```

LM head:

```text
selected = hidden_states[:, slice_indices, :]   # supports logits_to_keep
if asm is false:
    logits = Linear(H -> vocab_size, bias=True, tied_weight=token_embedding)(selected)
else:
    logits = AdaptiveLogSoftmaxWithLoss.log_prob(selected)
```

Concrete sampled shapes:

- `xlm-mlm-en-2048`: Q/K/V/O `2048 -> 2048`, FFN `2048 -> 8192 -> 2048`, LM `2048 -> 30145`.
- `xlm-mlm-100-1280`: Q/K/V/O `1280 -> 1280`, FFN `1280 -> 5120 -> 1280`, LM `1280 -> 200000`.
- `xlm-mlm-tlm-xnli15-1024`: Q/K/V/O `1024 -> 1024`, FFN `1024 -> 4096 -> 1024`, LM `1024 -> 95000`.

## 6. Attention requirements

Primary encoder/MLM attention:

- Self-attention only for first target.
- MHA, not GQA/MQA: K/V heads equal query heads.
- Noncausal if `causal=false`: mask is padding-shaped `[B,S]`, broadcast to `[B,A,Q,K]`.
- Causal if `causal=true`: mask is triangular `[B,S,S]`, then broadcast to `[B,A,Q,K]`.
- Query length can be shorter than key length when cache is used; ordinary encoder parity can start with `Q=K=S`.
- No packed/varlen ABI, no sliding window, no relative bias, no RoPE.
- Mask fill uses `torch.finfo(scores.dtype).min`, not a fixed `-1e4`.
- Softmax is explicitly performed in fp32 and cast back.

Cache path:

- `MultiHeadAttention` can update a `DynamicCache` with projected K/V per layer.
- Self-attention cache stores K/V after linear projection and head transpose, shape `[B,A,T,D]`.
- Cross-attention cache support exists in the attention module, but `XLMModel` has decoder/cross-attention blocks commented out and rejects `is_encoder=false`.
- `XLMModel.forward` does not return `cache` in `BaseModelOutput`; cache admission should be explicit rather than inferred from normal encoder outputs.

FlashAttention/SDPA compatibility:

- Dense noncausal and causal masks are compatible with standard SDPA/Flash-style kernels if mask semantics and fp32 softmax are preserved.
- `output_attentions=True` requires materializing full attention weights `[B,A,Q,K]`.

## 7. Position encoding and custom math

Default position ids are simple absolute positions:

```python
def xlm_position_ids(batch_size, seq_len):
    return arange(seq_len)[None, :].expand(batch_size, seq_len)
```

Mask construction equivalent:

```python
def xlm_masks(seq_len, lengths, causal, padding_mask=None):
    alen = arange(seq_len)
    mask = padding_mask if padding_mask is not None else (alen[None, :] < lengths[:, None])
    if causal:
        attn_mask = alen[None, None, :] <= alen[None, :, None]
    else:
        attn_mask = mask
    return mask, attn_mask
```

Sinusoidal initialization, when enabled:

```python
angle = pos / (10000 ** (2 * floor(j / 2) / dim))
table[:, 0::2] = sin(angle[:, 0::2])
table[:, 1::2] = cos(angle[:, 1::2])
```

This table can be materialized at load time as a constant. It is not recomputed per input.

## 8. Preprocessing and input packing

CPU/data pipeline:

- `XLMTokenizer` uses `vocab.json` plus `merges.txt`.
- Most languages use Moses punctuation normalization, non-printing character removal, Moses tokenization, optional lowercase/accent stripping, then BPE.
- Chinese, Japanese, and Thai have custom tokenizer branches requiring optional packages (`jieba`, `Mykytea`, `pythainlp`) or externally pre-tokenized input through `bypass_tokenizer`.
- Special-token layout is:
  - single: `<s> tokens </s>`
  - pair: `<s> A </s> B </s>`
- Language ids are tokenizer/config metadata. If `use_lang_emb=true`, the runtime graph needs `langs[B,S]`.

GPU/runtime inputs:

- Required for first target: `input_ids[B,S]`.
- One of `attention_mask[B,S]` or `lengths[B]`; if both are absent source derives lengths from `input_ids != pad_index`.
- Optional: `langs[B,S]`, `position_ids[B,S]`, `inputs_embeds[B,S,H]`.
- `token_type_ids[B,S]` should be rejected or explicitly guarded for first integration because source reuses token embeddings rather than a segment table.

Generation-controller behavior:

- `prepare_inputs_for_generation` appends `mask_token_id` to the sequence.
- If `config.lang_id` is not `None`, it creates a full `langs` tensor filled with that id.
- It removes externally supplied attention masks, token type ids, and position ids, relying on `XLMModel.forward` to recreate them.

## 9. Graph rewrite / lowering opportunities

### Rewrite: QKV linears to packed projection

Source pattern:

```text
q = Linear(H,H)(x); k = Linear(H,H)(x); v = Linear(H,H)(x)
```

Replacement:

```text
Linear(H,3H) -> split [q,k,v] -> reshape heads
```

Preconditions:

- Self-attention only; `kv is None`.
- All three projections have bias.
- Weight layout transform concatenates output rows in `[q,k,v]` order.

Failure cases:

- Cross-attention path, if ever admitted, has Q from decoder states and K/V from encoder states.
- Cache update must preserve projected K/V layout.

Parity test sketch:

- Compare separate Q/K/V tensors before head reshape for random hidden states and checkpoint weights.

### Rewrite: language embedding fold for single-language/no-language checkpoints

Source pattern:

```text
if langs is not None and use_lang_emb and n_langs > 1:
    x += lang_embedding[langs]
```

Replacement:

```text
omit language add, or add a broadcast constant row for fixed lang_id
```

Preconditions:

- `n_langs == 1`, or `use_lang_emb=false`, or serving API fixes one `lang_id`.

Failure cases:

- TLM/multilingual workloads with token-level mixed languages require full `langs[B,S]` gather.

### Rewrite: attention mask specialization

Source pattern:

```text
lengths -> padding mask -> broadcast additive mask
```

Replacement:

```text
all-valid fast path, padding-only SDPA mask, or causal triangular mask
```

Preconditions:

- `attention_mask` values are boolean/0-1 and match `lengths`.
- `causal` is known at compile/config time.

Failure cases:

- Cache path with shorter query slice needs key-length-aware mask handling.

### Rewrite: large-vocab logits_to_keep

Source pattern:

```text
hidden_states[:, slice_indices, :] -> LM projection
```

Replacement:

```text
last-token or selected-token hidden gather before vocab GEMM
```

Preconditions:

- API requests `logits_to_keep > 0` or a tensor of positions.
- Labels/loss are not needed for inference.

Failure cases:

- Full MLM parity requires `[B,S,V]` logits.
- Adaptive softmax path has a different provider contract.

### Rewrite: adaptive softmax admission gate

Source pattern:

```text
asm=true -> AdaptiveLogSoftmaxWithLoss.log_prob(x)
```

Replacement:

```text
reject initially, or route to dense fallback only if weights can be materialized equivalently
```

Preconditions:

- First integration: `asm=false`.

Failure cases:

- Real adaptive-softmax checkpoints require cutoffs/div value and multiple projection clusters.

## 10. Kernel fusion candidates

Highest priority:

- Bias GEMM coverage for Q/K/V/O, FFN up/down, and dense LM projection.
- Large-vocab logits with `logits_to_keep` support, especially `V=200000`.
- Dense MHA with padding/causal mask specialization and fp32 softmax.
- Embedding sum + LayerNorm + mask multiply, with optional language embedding.

Medium priority:

- Packed QKV projection.
- Residual add + LayerNorm after attention and FFN.
- GEMM + GELU/ReLU for FFN up projection.
- Mask generation kernels for `lengths` and causal masks when not precomputed.

Lower priority:

- Adaptive softmax provider.
- Full QA top-k/gather/classifier head.
- Multiple-choice flatten/summary convenience path.
- Cache/generation path for CLM-style serving.

## 11. Runtime staging plan

Stage 1: config and weights.

- Parse `XLMConfig`, require `is_encoder=true`, `emb_dim % n_heads == 0`, and initially `asm=false`.
- Load embeddings, position table, optional language table, encoder blocks, and dense LM head.
- Preserve LM head / token embedding tied-weight alias.

Stage 2: embedding and masks.

- Implement `input_ids`, `lengths`/`attention_mask`, position ids, optional `langs`.
- Reject `token_type_ids` except under an explicit compatibility flag.

Stage 3: one encoder block.

- Implement dense MHA and FFN with post-residual LayerNorm.
- Validate noncausal first, then causal mask if a target config requires it.

Stage 4: full encoder and LM head.

- Run all layers and dense LM logits.
- Add `logits_to_keep` selected-token projection for serving.

Stage 5: multilingual parity.

- Add language-id ABI tests for `xlm-mlm-tlm-xnli15-1024`.
- Add no-language-embedding path for `xlm-mlm-100-1280`.

Stage 6: optional heads.

- Sequence classification and token classification are simple follow-ups.
- QA full head requires top-k/gather path and can wait.

Stage 7: gated follow-ups.

- Causal/cache generation path.
- Adaptive softmax.
- Sinusoidal position-table loading if a checkpoint requires it.

## 12. Parity and validation plan

- Config parsing tests for all sampled snapshots, including source-default omitted fields.
- Tokenizer metadata tests for special-token layout and language-id mappings.
- Mask tests:
  - `lengths`-derived padding masks.
  - explicit `attention_mask`.
  - causal triangular mask.
- Embedding parity:
  - no language embedding.
  - language embedding enabled with constant and mixed `langs`.
  - reject/guard `token_type_ids`.
- One-layer parity for `H=32` tiny config and one production shape.
- Full encoder parity on short sequences and `S=512`.
- LM head parity for full logits and `logits_to_keep`.
- Tied-weight alias test: LM projection weight and input embedding remain one logical parameter.
- Optional head parity:
  - sequence summary `first`, `last`, `mean`, `cls_index`.
  - token classification.
  - simple QA first, full SQuAD head later.

Suggested tolerances:

- fp32: start with `rtol=1e-4`, `atol=1e-5` for block-level parity.
- fp16/bf16: start with `rtol=5e-2`, `atol=5e-2` end-to-end, then tighten after softmax/LayerNorm accumulation policy is fixed.

## 13. Performance probes

- Tokenizer throughput by language class: Moses path vs `zh`/`ja`/`th` custom path.
- Encoder throughput sweep: `B in {1, 8, 32}`, `S in {32, 128, 512}`.
- Attention breakdown: QKV GEMM, mask handling, softmax, value matmul, output GEMM.
- Causal vs noncausal attention mask overhead.
- Language embedding overhead for `n_langs=15` and no-op path for `use_lang_emb=false`.
- FFN GEMM/activation time for H=1024, 1280, 2048.
- LM projection cost for `V=30145`, `64139`, `95000`, `200000`.
- Full logits vs `logits_to_keep=1` selected logits.
- Dense LM projection vs adaptive-softmax fallback if an `asm=true` checkpoint is later identified.
- Cache memory/projection overhead only after causal generation is admitted.

## 14. Skip/defer list

Safe to defer for first encoder/MLM integration:

- Training losses, dropout randomness, and gradient checkpointing/chunking.
- `is_encoder=false` decoder mode because native source raises.
- Cache/generation path and beam/sampling controllers.
- Adaptive softmax (`asm=true`) until a real checkpoint plus cutoffs is selected.
- `token_type_ids` compatibility path.
- Full SQuAD-style QA top-k head.
- `summary_type="attn"` because source raises.
- Remote-code/non-native variants.
- Tensor parallelism and quantization.

## 15. Final implementation checklist

- [ ] Parse `XLMConfig` and validate `is_encoder=true`, `emb_dim % n_heads == 0`.
- [ ] Initially admit `asm=false`; document/reject adaptive softmax configs.
- [ ] Load token, position, optional language, encoder, LayerNorm, and dense LM weights.
- [ ] Preserve tied embedding / LM projection weight alias.
- [ ] Implement `lengths` and `attention_mask` mask construction.
- [ ] Implement noncausal and causal dense MHA mask semantics.
- [ ] Implement embedding sum with optional `langs` and final mask multiply.
- [ ] Reject or explicitly guard `token_type_ids`.
- [ ] Implement post-residual LayerNorm encoder block with GELU/ReLU FFN.
- [ ] Implement dense LM head and `logits_to_keep`.
- [ ] Add packed QKV rewrite with self-attention-only guard.
- [ ] Add language-embedding fold/no-op rewrites.
- [ ] Add config, mask, embedding, one-block, full-encoder, and LM parity tests.
- [ ] Add multilingual `langs` parity for XNLI15/TLM-style configs.
- [ ] Benchmark attention, FFN, LayerNorm, language embedding, and LM projection separately.
