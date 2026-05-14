# CamemBERT Transformers Audit

## 1. Source basis

Transformers commit/version:

- Local checkout `X:/H/transformers` at commit `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`.

Model id:

- Primary: `almanach/camembert-base`.
- Representative configs: `almanach/camembert-base`, `camembert/camembert-large` / API id `almanach/camembert-large`, `camembert/camembert-base-ccnet` / API id `almanach/camembert-base-ccnet`, `hf-internal-testing/tiny-random-camembert`, `qanastek/pos-french-camembert`, `Jean-Baptiste/camembert-ner`.

Config source:

- `https://huggingface.co/almanach/camembert-base/raw/main/config.json`
- `https://huggingface.co/camembert/camembert-large/raw/main/config.json`
- `https://huggingface.co/camembert/camembert-base-ccnet/raw/main/config.json`
- `https://huggingface.co/hf-internal-testing/tiny-random-camembert/raw/main/config.json`
- `https://huggingface.co/qanastek/pos-french-camembert/raw/main/config.json`
- `https://huggingface.co/Jean-Baptiste/camembert-ner/raw/main/config.json`
- Snapshots are under `agents/plans/transformers/camembert/_sources/`.

Source files inspected:

- `X:/H/transformers/src/transformers/models/camembert/modeling_camembert.py`
- `X:/H/transformers/src/transformers/models/camembert/modular_camembert.py`
- `X:/H/transformers/src/transformers/models/camembert/configuration_camembert.py`
- `X:/H/transformers/src/transformers/models/camembert/tokenization_camembert.py`
- Comparison points: RoBERTa and XLM-RoBERTa modeling/config/tokenizer files at the same commit.

Any missing files or assumptions:

- `modeling_camembert.py` is generated from `modular_camembert.py`. Runtime behavior is easiest to audit in the generated file; future upstream source edits should target the modular file.
- [hf-internal-testing/tiny-random-CamembertModel](https://huggingface.co/hf-internal-testing/tiny-random-CamembertModel) returned `401 Unauthorized`; access would only improve debug checkpoint coverage. The public `hf-internal-testing/tiny-random-camembert` config was used instead.
- Primary DinoML target here is encoder inference plus `CamembertForMaskedLM`. Token/sequence classification and QA are optional heads. `CamembertForCausalLM` exists in source but is a non-primary decoder variant.
- No DinoML tests were run, per task scope.

## 2. High-level architecture

CamemBERT is a text-only RoBERTa-style bidirectional Transformer encoder: learned token embeddings, learned absolute position embeddings with padding-aware offsets, learned token-type embeddings that are normally all-zero, post-residual LayerNorm blocks, and optional task heads.

```text
SentencePiece/Unigram tokenizer -> input_ids/attention_mask
-> word + token_type + learned absolute position embeddings
-> embedding LayerNorm
-> N bidirectional encoder layers
-> MLM transform + tied vocab projection
-> logits [B, S, V]
```

The useful first runtime contract is `input_ids [B,S]`, `attention_mask [B,S]`, optional `position_ids [B,S]`, and optional all-zero `token_type_ids [B,S]`. Tokenization is CPU/data-pipeline work; encoder and heads are GPU/runtime work.

## 3. Important config dimensions

Source defaults from `CamembertConfig`:

| Field | Default | Runtime meaning |
|---|---:|---|
| `vocab_size` | 30522 | Source default only; production CamemBERT uses 32005 |
| `hidden_size` | 768 | Encoder width `H` |
| `num_hidden_layers` | 12 | Encoder block count |
| `num_attention_heads` | 12 | MHA heads |
| `head_dim` | 64 | Inferred as `hidden_size / num_attention_heads` |
| `intermediate_size` | 3072 | FFN expansion |
| `hidden_act` | `gelu` | FFN and MLM transform activation |
| `max_position_embeddings` | 512 | Source default only; production checkpoints use 514 |
| `type_vocab_size` | 2 | Source default only; production checkpoints use 1 |
| `layer_norm_eps` | `1e-12` | Source default only; production checkpoints use `1e-5` |
| `pad_token_id` / `bos_token_id` / `eos_token_id` | `1` / `0` / `2` | Padding and special-token ids |
| `use_cache` | true | Effective only when `is_decoder=true`; forced off for encoder target |
| `tie_word_embeddings` | true | MLM decoder weight aliases input word embeddings |

Representative checkpoint sweep:

| Model | Arch / auto model | Layers | H | Heads | D | FFN | Vocab | Max pos | Type vocab | LN eps | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `almanach/camembert-base` | `CamembertForMaskedLM` | 12 | 768 | 12 | 64 | 3072 | 32005 | 514 | 1 | `1e-5` | Primary fill-mask checkpoint |
| `camembert/camembert-large` | `AutoModel` metadata | 24 | 1024 | 16 | 64 | 4096 | 32005 | 514 | 1 | `1e-5` | Larger encoder geometry |
| `camembert/camembert-base-ccnet` | `AutoModel` metadata | 12 | 768 | 12 | 64 | 3072 | 32005 | 514 | 1 | `1e-5` | Same operator surface, alternate pretraining corpus |
| `hf-internal-testing/tiny-random-camembert` | `CamembertModel` | 5 | 32 | 4 | 8 | 37 | 1000 | 512 | 16 | `1e-12` | Debug-sized, not tokenizer-representative |
| `qanastek/pos-french-camembert` | token classification | 12 | 768 | 12 | 64 | 3072 | 32005 | 514 | 1 | `1e-5` | POS head over encoder |
| `Jean-Baptiste/camembert-ner` | token classification | 12 | 768 | 12 | 64 | 3072 | 32005 | 514 | 1 | `1e-5` | NER head over encoder |

## 3a. Family variation traps

- CamemBERT's generated modeling body is effectively RoBERTa with CamemBERT names and config class. The modular source inherits RoBERTa classes and swaps base-model wiring.
- Production configs differ from `CamembertConfig()` defaults: `vocab_size=32005`, `max_position_embeddings=514`, `type_vocab_size=1`, `layer_norm_eps=1e-5`.
- Position ids are RoBERTa/fairseq-style, not BERT-style. With `pad_token_id=1`, non-pad tokens start at position id `2`; pad tokens stay at `1`.
- The 514 position rows are required for 512 tokenizer tokens because ids `0` and `1` are not ordinary non-pad positions.
- Tokenizer model inputs are only `input_ids` and `attention_mask`; token type ids are not emitted. Common checkpoints have `type_vocab_size=1`, so BERT-like segment id `1` is out of range.
- `CamembertTokenizer` is SentencePiece-like Unigram via `tokenizers`, not byte-level BPE like RoBERTa. It applies whitespace replacement/strip normalization plus Metaspace; XLM-R uses a different pre-tokenizer sequence and normally a much larger multilingual vocab.
- `CamembertForCausalLM` is implemented and supports decoder caches if `is_decoder=true`, but primary CamemBERT checkpoints are encoder/MLM or encoder task-head models. Do not require KV cache for first integration.
- The public tiny-random config uses `type_vocab_size=16`, `max_position_embeddings=512`, and `layer_norm_eps=1e-12`; it is useful for small shape tests but is not production-like.

## 4. Operator coverage checklist

Tensor/layout ops:

- Integer gather from embedding tables: `input_ids [B,S] -> [B,S,H]`, `position_ids [B,S] -> [B,S,H]`, `token_type_ids [B,S] -> [B,S,H]`.
- Position id generation if not precomputed: `ne`, int cast, `cumsum(dim=1)`, multiply, scalar add.
- Elementwise add of three embedding tensors.
- Attention reshapes/transposes: `[B,S,H] -> [B,A,S,D]`, attention output back to `[B,S,H]`.
- Mask creation/broadcast through Transformers `create_bidirectional_mask` for encoder mode.
- Optional head indexing: `features[:,0,:]` for sequence classification, flatten choices for multiple choice, split/squeeze for QA.

Neural network primitives:

- LayerNorm over last dimension with checkpoint eps.
- Bias Linear/GEMM for Q/K/V/O, FFN up/down, MLM transform, MLM decoder, and task heads.
- GELU in FFN and MLM transform; tanh in pooler/sequence classification.
- Residual add + LayerNorm after attention output and FFN output.
- Tied output projection for MLM/causal LM: `lm_head.decoder.weight` aliases `roberta.embeddings.word_embeddings.weight`; LM bias is a separate `[V]` parameter tied to decoder bias key.

Attention primitives:

- Bidirectional self-attention, MHA only. No GQA/MQA.
- Q/K/V projections all `Linear(H -> H, bias=True)`.
- Scores are `q @ k^T * head_dim^-0.5`, then mask add, softmax, dropout in training, and `attn @ v`.
- Source dispatches through `ALL_ATTENTION_FUNCTIONS`, with eager fallback and SDPA/Flash/Flex support metadata.

Preprocessing-coupled ops:

- SentencePiece/Unigram tokenization through `sentencepiece.bpe.model` or `tokenizer.json`.
- Special token layout: `<s> A </s>` and `<s> A </s> </s> B </s>`.
- Metaspace prefix-space behavior; default source tokenizer has `add_prefix_space=True`.

Generation/cache ops:

- None required for encoder/MLM. Optional decoder mode uses `DynamicCache` or `EncoderDecoderCache` with full MHA key/value tensors `[B,A,T,D]`.

## 5. Layer/block breakdown

Embedding block:

```text
input_ids: [B,S]
mask = input_ids != pad_token_id
position_ids = cumsum(mask, dim=1) * mask + pad_token_id
token_type_ids = zeros([B,S]) unless explicitly supplied

x = word_embedding[input_ids]             # [B,S,H]
x += token_type_embedding[token_type_ids] # common table [1,H]
x += position_embedding[position_ids]     # common table [514,H]
x = LayerNorm(x, eps=config.layer_norm_eps)
```

Encoder block, repeated `N` times:

```text
q = Linear(H -> H, bias=True)(x).view(B,S,A,D).transpose(1,2)
k = Linear(H -> H, bias=True)(x).view(B,S,A,D).transpose(1,2)
v = Linear(H -> H, bias=True)(x).view(B,S,A,D).transpose(1,2)

ctx = Attention(q, k, v, bidirectional_padding_mask)
ctx = ctx.transpose(1,2).reshape(B,S,H)
x = LayerNorm(Linear(H -> H, bias=True)(ctx) + x)

ff = GELU(Linear(H -> I, bias=True)(x))
x = LayerNorm(Linear(I -> H, bias=True)(ff) + x)
```

Masked-LM head:

```text
h = Linear(H -> H, bias=True)(x)
h = GELU(h)
h = LayerNorm(h, eps=config.layer_norm_eps)
logits = Linear(H -> V, bias=True, tied_weight=word_embeddings)(h)
```

Concrete production shapes:

- Base: Q/K/V/O `768 -> 768`, FFN `768 -> 3072 -> 768`, MLM decoder `768 -> 32005`.
- Large: Q/K/V/O `1024 -> 1024`, FFN `1024 -> 4096 -> 1024`, MLM decoder `1024 -> 32005`.

## 6. Attention requirements

Primary target:

- Noncausal bidirectional self-attention.
- Self-attention only; no cross-attention unless decoder mode is explicitly admitted.
- MHA with `num_key_value_heads == num_attention_heads`; no repeat-KV path.
- Head dim is inferred and source rejects non-divisible `hidden_size % num_attention_heads` unless an `embedding_size` escape hatch exists.
- Eager attention order is matmul, scale, additive mask, softmax over keys, dropout, value matmul.
- Encoder mode calls `create_bidirectional_mask`; fully unpadded optimized paths can avoid materializing full `[B,1,S,S]` masks if parity is preserved.
- No sliding-window, local, sparse, RoPE, ALiBi, or relative bias.

Optional decoder/causal branch:

- `CamembertModel.forward` forces `use_cache=False` when `config.is_decoder` is false.
- If `is_decoder=true`, the source creates a `DynamicCache`; with encoder states it creates an `EncoderDecoderCache`.
- Self-attention cache stores projected K/V after the linear projection and before attention, shape `[B,A,T,D]`.
- Cross-attention cache stores encoder-projected K/V once and reuses it via `is_updated[layer_idx]`.
- This branch should be a separate DinoML target; it is not required for encoder/MLM CamemBERT parity.

## 7. Position encoding and custom math

CamemBERT uses learned absolute positions with RoBERTa-style padding offsets:

```python
def camembert_position_ids(input_ids, padding_idx=1, past_len=0):
    mask = (input_ids != padding_idx).int()
    incremental = (cumsum(mask, dim=1) + past_len) * mask
    return incremental.long() + padding_idx
```

For common checkpoints, pads have position `1`; first real token has position `2`; a 512-token non-pad sequence reaches `513`, hence `max_position_embeddings=514`. If `inputs_embeds` are supplied, source cannot detect padding and generates sequential ids from `padding_idx + 1` to `padding_idx + S`.

No RoPE, M-RoPE, ALiBi, relative position bias, or convolutional positional embedding is used in the primary encoder path.

## 8. Preprocessing and input packing

CPU/data-pipeline:

- `CamembertTokenizer` uses a Unigram model loaded from `sentencepiece.bpe.model` or `tokenizer.json`.
- Normalization replaces repeated whitespace/newlines/tabs with a single space and strips only the right side.
- Metaspace uses replacement `U+2581` and defaults to prepending a space, so leading-space behavior matters.
- Special tokens are `<s>`, `<pad>`, `</s>`, `<unk>`, `<mask>`, plus common unused tokens such as `<s>NOTUSED` and `</s>NOTUSED`.
- Pair inputs use the RoBERTa-style double separator.

GPU/runtime:

- Required tensors: `input_ids [B,S]`, `attention_mask [B,S]`.
- First integration may accept precomputed `position_ids` and omit graph-side `cumsum`.
- Token type ids can be omitted or all zeros for production configs; if supplied, validate `< type_vocab_size`.
- Enforce `S <= 512` for common tokenizer configs or at least `max(position_ids) < max_position_embeddings`.

## 9. Graph rewrite / lowering opportunities

### Rewrite: all-zero token type embedding fold

Source pattern:

```text
token_type_ids omitted -> gather all-zero registered buffer -> token_type_embedding[0]
```

Replacement:

```text
broadcast token_type_embedding[0] over [B,S,H], or fold into embedding-sum kernel
```

Preconditions:

- `type_vocab_size == 1` or caller guarantees all token type ids are zero.
- Position ids are in range because source gathers the zero buffer using `position_ids`.

Failure cases:

- Tiny/random or custom configs with `type_vocab_size > 1` and nonzero token type ids.

Parity test sketch:

- Compare omitted `token_type_ids` against explicit zeros for base and tiny-random configs.

### Rewrite: position ids outside graph

Source pattern:

```text
cumsum(input_ids != pad_id, dim=1) * mask + pad_id
```

Replacement:

```text
CPU/tokenizer-side position_ids -> position embedding gather
```

Preconditions:

- Graph boundary includes `position_ids`.
- Data pipeline uses checkpoint `pad_token_id` and respects `max_position_embeddings`.

Failure cases:

- `inputs_embeds` path uses simple sequential ids.
- Optional decoder branch adds `past_key_values_length`.

Parity test sketch:

- Right-padded and left-padded batches: pads stay `1`, first non-pad token is `2`.

### Rewrite: QKV linears -> packed self-attention projection

Source pattern:

```text
q = Linear(H,H)(x); k = Linear(H,H)(x); v = Linear(H,H)(x)
```

Replacement:

```text
Linear(H,3H) -> split [q,k,v]
```

Weight transform:

```python
w_qkv = concat([w_q, w_k, w_v], axis=0)
b_qkv = concat([b_q, b_k, b_v], axis=0)
```

Preconditions:

- Self-attention only; same `hidden_states` input for all three projections.
- Bias is present for all projections.

Failure cases:

- Cross-attention computes Q from decoder states and K/V from encoder states.

Parity test sketch:

- Compare projected q/k/v tensors before reshape for random block weights.

### Rewrite: full MLM logits -> masked-position logits

Source pattern:

```text
MLMHead(sequence_output) -> [B,S,V]
```

Replacement:

```text
Gather masked hidden states -> MLMHead -> [num_masked,V]
```

Preconditions:

- Serving API only needs fill-mask positions and accepts changed output shape.

Failure cases:

- Hugging Face `CamembertForMaskedLM` parity requires full `[B,S,V]` logits.

Parity test sketch:

- Gather HF full logits at mask positions and compare against optimized head output.

## 10. Kernel fusion candidates

Highest priority:

- Embedding sum + LayerNorm, preserving CamemBERT position ids and all-zero token type behavior.
- Packed QKV projection plus attention layout handling.
- Bidirectional MHA with padding-mask fast path.
- Bias GEMM + GELU for FFN up projection; bias GEMM + residual + LayerNorm for attention/FFN outputs.
- MLM transform and large vocab projection; masked-position-only logits if API allows.

Medium priority:

- Position id generation kernel if not precomputed.
- Token-type embedding fold for `type_vocab_size=1`.
- Attention mask creation/skip path for all-ones masks.

Lower priority:

- Pooler and classification/QA heads.
- Decoder cache and CausalLM support.
- Training losses and dropout.

## 11. Runtime staging plan

Stage 1: config/tokenizer/weights

- Parse checkpoint config, not source defaults.
- Load word, position, token-type, encoder, and MLM-head weights.
- Preserve tied `lm_head.decoder.weight` and word embedding alias.

Stage 2: embedding parity

- Accept `input_ids`, `attention_mask`, and precomputed or generated `position_ids`.
- Fold all-zero token types only behind guards.

Stage 3: one encoder block

- Implement post-norm MHA and FFN block for `[B,S,H]`.
- Validate base and tiny-random geometry.

Stage 4: full encoder

- Run all layers with eager-equivalent bidirectional padding mask.
- Scale to base and large shapes.

Stage 5: masked LM

- Implement full `[B,S,32005]` logits for HF parity.
- Add masked-position-only rewrite later.

Stage 6: optional task heads

- Token classification first because public CamemBERT checkpoints use it; then sequence classification, QA, multiple choice.

Stage 7: optional decoder

- Admit CausalLM/cache only as a separate non-primary target.

## 12. Parity and validation plan

- Config parsing tests: production defaults vs source defaults, especially vocab, max positions, token types, eps.
- Tokenizer coupling tests: special-token layout, pair double separator, Metaspace leading-space behavior, mask token.
- Position-id tests with padded batches and `inputs_embeds` path.
- Embedding parity against HF for `almanach/camembert-base`.
- One-layer parity for no padding and mixed padding.
- Full encoder parity for tiny-random and first/last hidden states on base.
- MLM parity for `[B,S,V]` logits, plus masked-position rewrite parity if added.
- Token-classification head parity for one public NER/POS checkpoint when heads are admitted.

Suggested tolerances:

- fp32: `atol=1e-4`, `rtol=1e-4` for full encoder/logits; tighter for isolated linears and LayerNorm.
- fp16/bf16: start around `atol=2e-2`, `rtol=2e-2` end-to-end, then tighten per kernel.

## 13. Performance probes

- Tokenizer throughput for SentencePiece/Unigram plus special-token insertion.
- Position-id generation in graph vs CPU precompute.
- Encoder throughput sweep for base and large: `B in {1,8,32}`, `S in {16,64,128,512}`.
- Attention backend comparison: eager mask, SDPA-style mask, unpadded skip, fused bidirectional attention.
- FFN GEMM and GELU time for base vs large.
- LayerNorm/residual bandwidth probes.
- MLM full logits `[B,S,32005]` vs masked-position-only logits.
- Token-classification head overhead over encoder-only runtime.

## 14. Skip/defer list

Safe to defer for first encoder/MLM integration:

- Training losses, dropout randomness, gradient checkpointing, and chunked feed-forward.
- Sequence classification, token classification, QA, and multiple choice unless a target checkpoint requires the head.
- `CamembertForCausalLM`, decoder cache, cross-attention, beam search, and generation helpers.
- Optimized-backend exact `output_attentions=True` parity.
- Quantization, tensor parallelism, and remote-code handling.

## 15. Final implementation checklist

- [ ] Parse `CamembertConfig` from checkpoint and reject source-default mismatch hazards.
- [ ] Load tokenizer metadata needed for SentencePiece/Unigram, special tokens, `pad_token_id`, and max length.
- [ ] Load/tie word embedding and MLM decoder weights.
- [ ] Implement CamemBERT/RoBERTa padding-aware position ids or accept precomputed `position_ids`.
- [ ] Implement word, position, and token-type embedding lookup plus embedding LayerNorm.
- [ ] Add all-zero token-type fast path for `type_vocab_size=1`.
- [ ] Implement post-norm encoder block with bias Q/K/V/O linears, bidirectional MHA, residuals, LayerNorm, and GELU FFN.
- [ ] Implement full MLM head logits for `V=32005`.
- [ ] Add packed QKV rewrite with self-attention-only guard.
- [ ] Add fused residual + LayerNorm candidates.
- [ ] Add position-id and tokenizer-coupling parity tests.
- [ ] Add one-block, full-encoder, and MLM parity tests against HF.
- [ ] Benchmark encoder, attention, FFN, LayerNorm, and MLM head separately.
