# RoCBert Transformers audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: weiweishi/roc-bert-base-zh, plus representative fine-tunes listed below
Config source: Hugging Face config.json snapshots saved beside this report
Source files inspected:
- transformers/src/transformers/models/roc_bert/configuration_roc_bert.py
- transformers/src/transformers/models/roc_bert/modeling_roc_bert.py
- transformers/src/transformers/models/roc_bert/tokenization_roc_bert.py
Any missing files or assumptions: no image/audio processor; tokenizer side dictionaries are runtime input-pipeline assets. No gated/401 config gaps were hit for the sampled repos.
```

Primary source URLs at the inspected commit:

- https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/roc_bert/configuration_roc_bert.py
- https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/roc_bert/modeling_roc_bert.py
- https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/roc_bert/tokenization_roc_bert.py

Saved snapshots:

- `weiweishi__roc-bert-base-zh.config.json`
- `hf-internal-testing__tiny-random-RoCBertForMaskedLM.config.json`
- `daze-unlv__weiweishi-roc-bert-base-zh.config.json`
- `TiffanyH__Philosophy_ROCBert.config.json`
- `twn39__roc-bert-base-zh-finetune-dianping.config.json`
- `weiweishi__roc-bert-base-zh.tokenizer_config.json`
- `weiweishi__roc-bert-base-zh.special_tokens_map.json`

## 2. High-level architecture

RoCBert is a BERT-derived text encoder with extra per-token shape and pronunciation embeddings. The first useful DinoML target should be encoder or masked-LM/sequence-classification inference. The source also exposes a causal-LM wrapper and optional decoder/cross-attention mode inherited from BERT, but the official checkpoint is bidirectional pretraining/fill-mask, not an autoregressive production decoder.

```text
CPU tokenizer -> input_ids + input_shape_ids + input_pronunciation_ids + token_type_ids + attention_mask
  -> word/shape/pronunciation/token/position embeddings
  -> optional concat + Linear map to hidden_size
  -> LayerNorm
  -> N x bidirectional Transformer encoder blocks
  -> pooled CLS output and/or MLM/classification/QA/token heads
```

Independently stageable parts:

- CPU/data pipeline: WordPiece tokenizer plus `word_shape.json` and `word_pronunciation.json` lookup tables.
- Embedding front-end: gather three token-index tables, concatenate or average, add segment and position embeddings.
- Encoder body: standard MHA + post-LN residual blocks.
- Heads: MLM tied decoder, pooled sequence/multiple-choice classifiers, token classifier, QA split head.

## 3. Important config dimensions

| Field | Official `weiweishi/roc-bert-base-zh` | Source default | Notes |
|---|---:|---:|---|
| `vocab_size` | 21128 | 30522 | Official Chinese vocab differs from source default. |
| `hidden_size` | 768 | 768 | Encoder width. |
| `num_hidden_layers` | 12 | 12 | Repeated encoder blocks. |
| `num_attention_heads` | 12 | 12 | MHA only; no GQA/MQA field. |
| `head_dim` | 64 | 64 | Computed as `hidden_size / num_attention_heads`. |
| `intermediate_size` | 3072 | 3072 | FFN expansion. |
| `max_position_embeddings` | 512 | 512 | Learned absolute table. |
| `type_vocab_size` | 2 | 2 | Tiny checkpoint uses 16. |
| `enable_shape` / `enable_pronunciation` | true / true | true / true | Controls side embedding use. |
| `shape_embed_dim` / `shape_vocab_size` | 512 / 24858 | 512 / 24858 | Side token table. |
| `pronunciation_embed_dim` / `pronunciation_vocab_size` | 768 / 910 | 768 / 910 | Side token table. |
| `concat_input` | true | true | If true, concat side embeddings then project to hidden. |
| `hidden_act` | `gelu` | `gelu` | ACT2FN lookup. |
| `layer_norm_eps` | 1e-12 | 1e-12 | BERT-style LayerNorm. |
| `use_cache` | true | true | Effective only when `is_decoder=True`; encoder disables cache. |
| `torch_dtype` | float32 | not fixed | Config metadata; source layers are ordinary torch modules. |

Representative checkpoint sweep:

| Model | Architecture | Hidden/layers/heads | FFN | Vocab | Shape/pron dims | Labels/head variation |
|---|---|---:|---:|---:|---|---|
| `weiweishi/roc-bert-base-zh` | `RoCBertForPreTraining` | 768 / 12 / 12 | 3072 | 21128 | 512 / 768 | MLM + training-only contrastive path. |
| `hf-internal-testing/tiny-random-RoCBertForMaskedLM` | `RoCBertForMaskedLM` | 32 / 5 / 4 | 37 | 21128 | 32 / 32 | Tiny debug shape, `type_vocab_size=16`. |
| `daze-unlv/weiweishi-roc-bert-base-zh` | `RoCBertForMultipleChoice` | 768 / 12 / 12 | 3072 | 21128 | 512 / 768 | Multiple-choice flatten/reshape head. |
| `TiffanyH/Philosophy_ROCBert` | `RoCBertForSequenceClassification` | 768 / 12 / 12 | 3072 | 21128 | 512 / 768 | Sequence classifier; config says single-label. |
| `twn39/roc-bert-base-zh-finetune-dianping` | `RoCBertForSequenceClassification` | 768 / 12 / 12 | 3072 | 21128 | 512 / 768 | 5-label classifier from `id2label`. |

## 3a. Family variation traps

- `vocab_size` in source defaults is 30522, but the official and sampled checkpoints use 21128.
- `hidden_size` must be divisible by `num_attention_heads`; source rejects otherwise unless an `embedding_size` attr exists.
- `head_dim` is computed, not separately configured.
- `concat_input=True` changes the embedding graph to `cat([word, shape?, pronunciation?], -1) -> Linear(input_dim, hidden_size)`. For the official checkpoint that input width is `768 + 512 + 768 = 2048`.
- `concat_input=False` changes semantics: word/token/position embeddings are LayerNorm/dropout first, then side embeddings are added and divided by the number of present sources. This path needs a separate parity target.
- If shape/pronunciation IDs are omitted and `concat_input=True`, the source fills them with zeros. If `concat_input=False`, side embeddings are used only when the corresponding IDs are provided.
- Historical config fields such as `directionality`, `enable_cls`, `pooler_fc_size`, `pooler_num_attention_heads`, `pooler_num_fc_layers`, `pooler_size_per_head`, `pooler_type`, and `position_embedding_type` are present in checkpoints but not read by this RoCBert modeling source. Treat them as ignored for this source basis.
- `is_decoder=True` and `add_cross_attention=True` enable inherited decoder/cross-attention behavior, but sampled RoCBert checkpoints are encoder-style. First integration should reject or defer decoder configs unless explicitly targeted.
- No NHWC/NCHW tensor layout exists in the neural graph. All model tensors are text sequences `[B, S, H]`; attention temporarily uses `[B, heads, S, head_dim]`.

## 4. Operator coverage checklist

Tensor/layout ops:

- Integer embedding gather for `input_ids`, `input_shape_ids`, `input_pronunciation_ids`, `token_type_ids`, `position_ids`.
- `torch.cat(..., dim=-1)` for concat embedding path.
- Dense `Linear(2048 -> 768)` for official concat embedding map; generally `hidden + enabled side dims -> hidden`.
- Add, clone/copy semantics, in-place-like accumulation lowering, scalar divide for non-concat embedding path.
- `view`, `reshape`, `transpose(1, 2)`, `transpose(2, 3)`, `contiguous`, `split`, `squeeze`, first-token slice `hidden[:, 0]`.
- Gather of default `token_type_ids` from the registered zero buffer by `position_ids` when token type IDs are omitted.

Neural network primitives:

- LayerNorm over hidden dim with epsilon 1e-12.
- Dropout modules become identity for inference.
- Linear layers with bias throughout Q/K/V, attention output, FFN, pooler, heads.
- GELU default activation via ACT2FN; keep ACT2FN-configurable activation admission.
- Tanh for pooler.
- L2 normalize and pairwise matmul only for pretraining contrastive loss; defer for inference.

Attention primitives:

- Bidirectional dense self-attention for encoder target.
- Optional causal self-attention and optional encoder-decoder cross-attention for decoder configs.
- Attention mask addition before softmax; mask creation is delegated to `create_bidirectional_mask` or `create_causal_mask`.
- SDPA/Flash/Flex dispatch is source-supported through `ALL_ATTENTION_FUNCTIONS`; eager fallback math is `Q @ K^T * head_dim^-0.5 + mask -> softmax -> dropout -> @ V`.

Position/relative-bias ops:

- Learned absolute position embedding table only. No RoPE, ALiBi, or relative position bias in this source.

Generation/cache ops:

- Encoder target: no KV cache.
- Decoder target: `EncoderDecoderCache(DynamicCache, DynamicCache)` with per-layer self-attention KV and optional cross-attention KV.
- Causal LM supports `logits_to_keep` as int or tensor slice before LM head.
- `prepare_inputs_for_generation` trims `input_shape_ids` and `input_pronunciation_ids` to the last token when past cache is present.

Preprocessing-coupled ops:

- CPU tokenizer emits `input_shape_ids` and `input_pronunciation_ids` from JSON token maps. Unknown/special IDs are controlled by tokenizer files, not learned by the model graph.
- Special token layout is BERT style: `[CLS] X [SEP]` or `[CLS] A [SEP] B [SEP]`.

Quantized/packed weight metadata ops:

- None in source. All weights are ordinary dense PyTorch tensors.

Parameter sharing:

- MLM/pretraining/causal LM decoder weight is tied to `roc_bert.embeddings.word_embeddings.weight`.
- LM decoder has an output-only bias of shape `[vocab_size]`; `_tied_weights_keys` also aliases decoder bias to predictions bias.

## 5. Layer/block breakdown

Embedding front-end, official concat path:

```text
word = Embedding(vocab_size=21128, hidden=768)(input_ids)                    -> [B,S,768]
shape = Embedding(shape_vocab=24858, dim=512)(input_shape_ids or zeros)      -> [B,S,512]
pron = Embedding(pron_vocab=910, dim=768)(input_pronunciation_ids or zeros) -> [B,S,768]
x = cat([word, shape, pron], dim=-1)                                        -> [B,S,2048]
x = Linear(2048 -> 768)(x)
x = x + token_type_embedding[token_type_ids] + position_embedding[position_ids]
x = LayerNorm(x, eps=1e-12)
x = Dropout(x)  # identity in inference
```

Non-concat path:

```text
x = word + token_type + position
x = LayerNorm(x)
x = Dropout(x)
y = clone(x)
if shape ids provided: y += shape_embedding
if pronunciation ids provided: y += pronunciation_embedding
y = y / denominator
```

Encoder block, repeated `num_hidden_layers` times:

```text
q = Linear(H -> H)(x).view(B,S,heads,head_dim).transpose(1,2)
k = Linear(H -> H)(x).view(B,S,heads,head_dim).transpose(1,2)
v = Linear(H -> H)(x).view(B,S,heads,head_dim).transpose(1,2)
a = Attention(q, k, v, mask)                         -> [B,heads,S,head_dim]
a = a.transpose(1,2).reshape(B,S,H).contiguous()
x = LayerNorm(Linear(H -> H)(a) + residual)
m = ACT2FN[hidden_act](Linear(H -> intermediate)(x))
x = LayerNorm(Linear(intermediate -> H)(m) + residual)
```

Pool/head variants:

- Pooler: select `hidden[:, 0]`, then `Linear(768 -> 768)`, `tanh`.
- MLM/pretraining/causal LM head: `Linear(768 -> 768) -> activation -> LayerNorm -> Linear(768 -> vocab_size)`, with decoder weight tied to input word embedding.
- Sequence classification: pooler output -> `Linear(768 -> num_labels)`.
- Multiple choice: flatten `[B,C,S]` to `[B*C,S]`, encode, classifier `Linear(768 -> 1)`, reshape logits to `[B,C]`.
- Token classification: per-token `Linear(768 -> num_labels)`.
- QA: per-token `Linear(768 -> num_labels)`, then `split(1, dim=-1)` and squeeze to start/end logits. For standard QA, `num_labels=2`.

## 6. Attention requirements

Primary encoder target:

- Noncausal bidirectional self-attention.
- MHA only: `num_attention_heads=12`, KV heads equal query heads, `head_dim=64` for base.
- Query/key/value width all equal hidden size.
- Attention input layout after projection is `[B, heads, S, head_dim]`.
- Eager score shape is `[B, heads, S, S]`.
- Mask is additive and broadcastable to attention scores. Mask creation utilities convert user `attention_mask` into backend-specific additive masks.
- Dropout is present but inference uses probability 0.
- FlashAttention/SDPA/Flex source dispatch is allowed by `_supports_*` flags and `ALL_ATTENTION_FUNCTIONS`; DinoML should first implement source-equivalent eager dense MHA, then route to fused attention when mask/backend constraints match.

Decoder/cross-attention optional target:

- `is_decoder=True` makes self-attention causal and cache-enabled.
- `add_cross_attention=True` inserts cross-attention after self-attention. Query comes from decoder hidden states; key/value come from `encoder_hidden_states`.
- Self KV cache stores per-layer tensors after projection and transpose, shape `[B, heads, cached_S, head_dim]`.
- Cross-attention cache stores projected encoder K/V once per layer and reuses it when `is_updated[layer_idx]` is true.
- `prepare_inputs_for_generation` slices side IDs to `[:, -1:]` when cache is present, so input packing must keep side IDs aligned with token IDs.

No local/window/block/sparse/hash attention is implemented.

## 7. Position encoding and custom math

RoCBert uses learned absolute positions. The source registers a `[1, max_position_embeddings]` position-id buffer and slices it by `past_key_values_length`:

```python
def default_position_ids(position_ids_buffer, seq_length, past_len):
    return position_ids_buffer[:, past_len : seq_length + past_len]
```

Token type defaults use the zero `token_type_ids` buffer and gather by position IDs:

```python
def default_token_type_ids(token_type_buffer, position_ids, batch_size, seq_length):
    buffered = token_type_buffer.expand(position_ids.shape[0], -1)
    gathered = torch.gather(buffered, dim=1, index=position_ids)
    return gathered.expand(batch_size, seq_length)
```

Embedding merge has the only RoCBert-specific math:

```python
def concat_embed(word, shape, pron, map_linear, token_type, pos, ln):
    x = torch.cat((word, shape, pron), dim=-1)
    x = map_linear(x)
    return ln(x + token_type + pos)
```

All position and token-type buffers can be precomputed for static sequence buckets. Dynamic decode depends on `past_key_values_length`.

## 8. Preprocessing and input packing

CPU/data pipeline:

- `RoCBertTokenizer` is a slow WordPiece tokenizer with Chinese character splitting enabled by default.
- Required files are `vocab.txt`, `word_shape.json`, and `word_pronunciation.json`.
- The tokenizer returns `input_ids`, `input_shape_ids`, `input_pronunciation_ids`, optional `token_type_ids`, and optional `attention_mask`.
- `tokenizer_config.json` for the official checkpoint sets `model_max_length=512`, `do_basic_tokenize=true`, `do_lower_case=true`, and `tokenize_chinese_chars=true`.
- Special tokens are `[CLS]`, `[SEP]`, `[PAD]`, `[MASK]`, `[UNK]`.

Packing rules:

- Single sequence: `[CLS] tokens [SEP]`.
- Pair sequence: `[CLS] A [SEP] B [SEP]`.
- Shape/pronunciation IDs are built with the same special-token structure, using the `[UNK]` entry from each side dictionary for inserted special tokens in `prepare_for_model`.
- Padding also pads `input_shape_ids` and `input_pronunciation_ids` with each dictionary's pad token value.
- For caller-provided `inputs_embeds`, input token IDs are absent, but shape/pronunciation IDs may still be supplied and participate in embedding merge depending on config.

GPU/runtime graph inputs for first integration should be tensors:

```text
input_ids: int64/int32 [B,S]
input_shape_ids: int64/int32 [B,S]
input_pronunciation_ids: int64/int32 [B,S]
token_type_ids: int64/int32 [B,S], optional default zero
position_ids: int64/int32 [1,S] or [B,S], optional generated
attention_mask: bool/int/float [B,S], optional all-valid
```

## 9. Graph rewrite / lowering opportunities

### Rewrite: concat embedding map to one fused gather-cat-GEMM

Source pattern:

```text
Embedding(word) + optional Embedding(shape/pron as separate tensors)
cat(..., dim=-1)
Linear(input_dim -> H)
add token_type + position
LayerNorm
```

Replacement:

```text
Fused embedding gather into packed temporary -> GEMM/Linear -> Add2 -> LayerNorm
```

Preconditions:

- `concat_input=True`.
- Enabled side embeddings and their dimensions match config.
- All input ID tensors share `[B,S]`.
- Missing side IDs are filled with zeros before gather.

Shape equations:

- `concat_dim = H + (enable_shape ? shape_embed_dim : 0) + (enable_pronunciation ? pronunciation_embed_dim : 0)`.
- Official base: `[B,S,2048] @ [2048,768] -> [B,S,768]`.

Failure cases:

- `concat_input=False`; use average-add path.
- Caller supplies `inputs_embeds` with incompatible hidden width.
- Side IDs use unsupported integer dtype or out-of-range values.

Parity test sketch:

- Compare fused gather-cat-linear-LN against Transformers for random IDs, omitted side IDs, and explicit zero side IDs.

### Rewrite: QKV separate projections to packed QKV GEMM

Source pattern:

```text
query = Linear(H -> H)(x)
key = Linear(H -> H)(x)
value = Linear(H -> H)(x)
```

Replacement:

```text
Linear(H -> 3H) with packed weights [Wq; Wk; Wv] and packed bias [bq; bk; bv],
then split in Q,K,V order.
```

Preconditions:

- Self-attention, not cross-attention with different key/value source.
- All three projections have bias and identical input/output widths.
- Packed split order must be Q, K, V.

Shape equations:

- Base: `[B*S,768] @ [768,2304] + [2304] -> split three `[B,S,768]` tensors.

Failure cases:

- Cross-attention needs Q from decoder states and K/V from encoder states, so only K/V can be packed together.
- Weight tying or external checkpoint format does not preserve ordinary dense `nn.Linear` layout.

Parity test sketch:

- Pack weights from a loaded block and compare q/k/v tensors before transpose.

### Rewrite: encoder attention to fused SDPA/FlashAttention

Source pattern:

```text
q,k,v [B,heads,S,D] -> matmul scale mask softmax matmul -> [B,S,H]
```

Replacement:

```text
Fused dense bidirectional attention
```

Preconditions:

- Additive mask is representable by the backend.
- No attention output weights are requested.
- Inference dropout is zero.
- Head dimension and dtype supported by provider.

Failure cases:

- Need to return dense attention weights.
- Nonstandard attention backend keyword path not modeled.
- Decoder cache or cross-attention target not yet admitted.

### Rewrite: MLM last/selected-token logits

Source pattern:

```text
hidden[:, slice_indices, :] -> prediction head -> logits
```

Replacement:

```text
Gather selected hidden rows before prediction transform and vocab GEMM.
```

Preconditions:

- Causal LM or application only needs masked/selected token logits.
- `slice_indices` is static int tail or bounded index tensor.

Failure cases:

- Full-sequence MLM fill-mask needs all masked positions or caller-selected positions, not necessarily last token.

### Layout guidance

No NCHW/NHWC rewrite applies. Treat `[B,S,H]` as row-major sequence layout and protect attention head transforms with a no-layout-translation guard unless the whole QKV -> attention -> output projection region is fused. Axis-sensitive operations include `cat(dim=-1)`, `softmax(dim=-1)`, `normalize(dim=-1)` in training-only pretraining contrastive loss, `split(dim=-1)`, and `hidden[:, 0]` pooling.

## 10. Kernel fusion candidates

Highest priority:

- LayerNorm + residual add for attention output and FFN output. This appears twice per layer and is latency/bandwidth sensitive.
- Packed QKV projection + head reshape for self-attention. Reduces three GEMM launches/reads to one projection path.
- Fused bidirectional attention for `[B,heads,S,D]`, especially `S <= 512`, `D=64`.
- FFN `Linear -> GELU -> Linear`, with optional activation fusion and residual/LN pairing.

Medium priority:

- Fused RoCBert embedding front-end: side embedding gathers, concat, map linear, token/position add, LayerNorm.
- MLM prediction head `Linear -> GELU -> LayerNorm -> vocab GEMM`, with selected-token lowering for fill-mask or causal decoding.
- Pooler `first-token gather -> Linear -> tanh` for classifier workloads.

Lower priority:

- Multiple-choice flatten/reshape and classifier head.
- QA split/squeeze head.
- Training-only contrastive normalize + similarity matrices.

## 11. Runtime staging plan

Stage 1: config/weight loader.

- Parse RoCBertConfig including side vocab dimensions and ignored historical fields.
- Load dense weights and preserve LM decoder/input embedding tying.
- Reject decoder/cross-attention configs for the first encoder target.

Stage 2: embedding parity.

- Implement official concat embedding path with side IDs.
- Validate omitted side IDs default to zero.
- Add separate non-concat average-add path only after official base passes.

Stage 3: one encoder block parity.

- Lower LayerNorm, packed or separate Q/K/V, dense attention, residual output, FFN.
- Compare intermediate hidden states against Transformers.

Stage 4: full encoder and heads.

- Add pooler, sequence classification, token classification, QA, MLM.
- Multiple-choice can reuse encoder after flattening `[B,C,S] -> [B*C,S]`.

Stage 5: optimized attention and GEMM fusions.

- Introduce fused dense bidirectional attention and QKV packing under guards.
- Add selected-token logits rewrite for causal/MLM use cases.

Stage 6: optional decoder target.

- Admit `is_decoder=True`, causal masks, DynamicCache, and `prepare_inputs_for_generation` side-ID slicing.
- Add cross-attention cache only if a real checkpoint/use case needs it.

## 12. Parity and validation plan

- Tokenizer snapshot test: verify CPU tokenizer emits aligned `input_ids`, `input_shape_ids`, `input_pronunciation_ids`, `token_type_ids`, and `attention_mask` for single and pair sequences.
- Embedding tests: official base dimensions, missing side IDs, explicit zeros, custom `position_ids`, custom `token_type_ids`.
- One-layer fp32 parity: compare embedding output, q/k/v before attention, attention output, post-attention LN, FFN output with tolerances around `1e-5` fp32.
- Full encoder parity: compare last hidden state and pooler output for batch sizes 1 and >1, sequence lengths including padding.
- MLM parity: compare logits for all tokens and selected masked positions. Preserve tied decoder weight.
- Classification parity: sequence and token classification heads from sampled fine-tunes; multiple-choice reshape test for `[B,C,S]`.
- Optional decoder parity: prefill logits and one-step cached decode only after decoder admission.
- Suggested tolerances: fp32 `atol=1e-4, rtol=1e-4`; fp16/bf16 initially `atol=5e-3, rtol=5e-3`, tighten per backend.

## 13. Performance probes

- CPU tokenizer throughput, including side dictionary lookups.
- Embedding front-end throughput with gather-cat-linear-LN separated from encoder.
- Encoder-only throughput for `S={16,32,64,128,256,512}` and batch sweep.
- Attention backend comparison: eager BMM/softmax/BMM versus fused dense attention.
- QKV separate versus packed projection.
- FFN GEMM/activation/GEMM throughput.
- MLM full-sequence vocab GEMM versus selected-token logits.
- Classification end-to-end requests/sec for sequence and multiple-choice heads.
- Memory probe for side embedding tables and concat temporary `[B,S,2048]` in the official path.

## 14. Skip/defer list

- Training losses, contrastive pretraining branch, and gradient checkpointing.
- Decoder/cross-attention/cache mode unless a real RoCBert decoder checkpoint is targeted.
- Returning attention weights in optimized attention path.
- Non-concat embedding path can wait behind official checkpoint parity.
- Tokenizer implementation inside GPU runtime; keep it CPU/data-pipeline owned.
- Quantization and packed weight formats; none are source-required.
- Multi-GPU/tensor parallel.

## 15. Final implementation checklist

- [ ] Parse `RoCBertConfig` including shape/pronunciation fields.
- [ ] Load tokenizer side assets as CPU/data-pipeline metadata.
- [ ] Load dense weights and preserve LM decoder/input embedding tying.
- [ ] Implement concat embedding path with side ID default-zero behavior.
- [ ] Implement learned position and token-type embedding defaults.
- [ ] Implement dense bidirectional MHA with additive mask.
- [ ] Implement BERT post-LN residual attention and FFN blocks.
- [ ] Implement pooler, MLM, sequence classification, token classification, QA, and multiple-choice heads.
- [ ] Add QKV packing rewrite with Q,K,V split-order tests.
- [ ] Add fused attention rewrite with mask/dropout/output-attention guards.
- [ ] Add selected-token logits rewrite.
- [ ] Add tokenizer/input packing parity tests.
- [ ] Add one-block and full-encoder parity tests.
- [ ] Benchmark embedding front-end, attention, FFN, and selected-token logits.
