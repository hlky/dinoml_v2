# Transformers MPNet Operator and Integration Report

## 1. Source basis

```text
Transformers commit/version:
  Local checkout transformers at b75feb2af64c3e29cbbc1bd859958c5432cc7ed4.

Model id:
  Primary source/config target: microsoft/mpnet-base.
  Representative sentence embedding targets:
    sentence-transformers/all-mpnet-base-v2
    sentence-transformers/paraphrase-mpnet-base-v2

Config source:
  https://huggingface.co/microsoft/mpnet-base/raw/main/config.json
  https://huggingface.co/microsoft/mpnet-base/raw/main/tokenizer_config.json
  https://huggingface.co/sentence-transformers/all-mpnet-base-v2/raw/main/config.json
  https://huggingface.co/sentence-transformers/all-mpnet-base-v2/raw/main/1_Pooling/config.json
  https://huggingface.co/sentence-transformers/all-mpnet-base-v2/raw/main/modules.json
  https://huggingface.co/sentence-transformers/all-mpnet-base-v2/raw/main/sentence_bert_config.json
  https://huggingface.co/sentence-transformers/paraphrase-mpnet-base-v2/raw/main/config.json
  https://huggingface.co/sentence-transformers/paraphrase-mpnet-base-v2/raw/main/1_Pooling/config.json
  https://huggingface.co/sentence-transformers/paraphrase-mpnet-base-v2/raw/main/modules.json

Source files inspected:
  transformers/src/transformers/models/mpnet/configuration_mpnet.py
  transformers/src/transformers/models/mpnet/modeling_mpnet.py
  transformers/src/transformers/models/mpnet/tokenization_mpnet.py
  transformers/src/transformers/models/auto/modeling_auto.py
  transformers/src/transformers/modeling_utils.py

Any missing files or assumptions:
  No MPNet remote code is required for the inspected checkpoints. The current
  Transformers source has PyTorch MPNet only; no family-specific SDPA,
  FlashAttention, generation cache, or remote-code alternate kernels were found.
  Source/config snapshots are under agents/plans/transformers/mpnet/_sources/.
```

HF repo shas inspected:

- [microsoft/mpnet-base](https://huggingface.co/microsoft/mpnet-base): `6996ce1e91bd2a9c7d7f61daec37463394f73f09`
- [sentence-transformers/all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2): `e8c3b32edf5434bc2275fc9bab85f82640a19130`
- [sentence-transformers/paraphrase-mpnet-base-v2](https://huggingface.co/sentence-transformers/paraphrase-mpnet-base-v2): `6cc9279c672dc57f94445ef259b28a1b736fec8f`

## 2. High-level architecture

MPNet in Transformers is a text-only bidirectional encoder with learned token
and absolute position embeddings, a shared learned relative-position bucket bias
added inside every self-attention layer, BERT-style post-norm residual blocks,
and optional task heads.

```text
WordPiece tokenization + special tokens
  -> token ids / attention mask
  -> token embedding + position embedding + LayerNorm
  -> repeated encoder blocks with shared relative position bias
  -> last hidden state
  -> optional first-token pooler / MLM head / classification / token / QA heads
```

Sentence-transformers MPNet uses the same encoder branch, then applies wrapper
pooling outside `modeling_mpnet.py`:

```text
MPNet last_hidden_state + attention_mask -> masked mean pooling -> optional L2 normalize -> embedding
```

Stages that can be validated independently:

- CPU/tokenizer: WordPiece, `<s> A </s>` and `<s> A </s></s> B </s>` packing.
- GPU encoder: embeddings, position-id construction, relative-position buckets,
  dense bidirectional self-attention, FFN, LayerNorm.
- Heads: masked-LM tied decoder, first-token classifier/pooler, token/QA heads,
  sentence-transformers masked mean pooling and normalization.

## 3. Important config dimensions

Primary `microsoft/mpnet-base` dimensions:

| Field | Value | Source |
|---|---:|---|
| `vocab_size` | 30527 | HF config |
| `hidden_size` | 768 | HF config |
| `num_hidden_layers` | 12 | HF config |
| `num_attention_heads` | 12 | HF config |
| `head_dim` | 64 | source-derived `hidden_size / num_attention_heads` |
| `intermediate_size` | 3072 | HF config |
| `hidden_act` | `gelu` | HF config |
| `max_position_embeddings` | 514 | HF config |
| `relative_attention_num_buckets` | 32 | HF config |
| relative bucket `max_distance` | 128 | source default in `relative_position_bucket` |
| `layer_norm_eps` | `1e-05` | HF config; source default is `1e-12` |
| `pad_token_id` | 1 | HF config/source embedding padding index |
| `bos_token_id` / `eos_token_id` | 0 / 2 | HF config |
| tokenizer `model_max_length` | 512 | tokenizer config |
| cache support | none | source-derived |

Representative checkpoint sweep:

| Model | Architecture field | Layers | Hidden | Heads | FFN | Max positions | Buckets | Extra wrapper behavior |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `microsoft/mpnet-base` | `MPNetForMaskedLM` | 12 | 768 | 12 | 3072 | 514 | 32 | fill-mask/MLM |
| `sentence-transformers/all-mpnet-base-v2` | `MPNetForMaskedLM` | 12 | 768 | 12 | 3072 | 514 | 32 | mean pooling + L2 normalize; wrapper max seq 384 |
| `sentence-transformers/paraphrase-mpnet-base-v2` | `MPNetModel` | 12 | 768 | 12 | 3072 | 514 | 32 | mean pooling; no Normalize module in inspected `modules.json` |

## 3a. Family variation traps

- Source `MPNetConfig` defaults `max_position_embeddings=512` and
  `layer_norm_eps=1e-12`, but accessible base checkpoints use `514` and
  `1e-05`. Do not rely on config-class defaults when loading real weights.
- `MPNetEmbeddings` hardcodes `padding_idx = 1`; admission should reject
  incompatible configs or document that only pad id 1 is source-compatible.
- `hidden_size` must be divisible by `num_attention_heads`; this family has no
  GQA/MQA and no explicit `num_key_value_heads`.
- Relative-position bias is one shared embedding table at encoder scope, shaped
  `[relative_attention_num_buckets, num_attention_heads]`, reused by all layers.
- `compute_position_bias` accepts a `position_ids` argument but encoder forward
  does not pass user `position_ids`; runtime parity for normal source execution
  should use arange-relative buckets by sequence length, not token-derived
  absolute position ids.
- MLM output has tied weight aliases:
  `lm_head.decoder.weight -> mpnet.embeddings.word_embeddings.weight` and
  `lm_head.decoder.bias -> lm_head.bias`. Preserve alias identity.
- Sentence-transformers checkpoints add pooling/normalization modules outside
  Transformers MPNet. DinoML should treat those as wrapper graph nodes, not as
  MPNet core architecture.
- The tokenizer pair template has two separator tokens between A and B:
  `<s> A </s></s> B </s>`. Token type ids are not a model input.
- No causal mask, KV cache, RoPE, ALiBi, sliding window, sparse attention,
  packed varlen attention, MoE, convolution, or multimodal stitch path exists in
  this source basis.

## 4. Operator coverage checklist

Tensor/layout ops:

- `Embedding(input_ids[B,S]) -> [B,S,768]` for word embeddings.
- `Embedding(position_ids[B,S]) -> [B,S,768]` for absolute position embeddings.
- Elementwise add for token + position embeddings.
- `LayerNorm(768, eps)` on `[B,S,768]`.
- Dropout is identity in inference; training dropout may be deferred.
- `view`, `transpose`, `permute`, `contiguous`, `expand`, `squeeze`, `split`,
  first-token slice, and masked mean reduction for sentence-transformers.
- Integer ops for position ids: `ne`, `int`, `cumsum(dim=1)`, multiply, cast to
  long, add `padding_idx`.

Neural network primitives:

- Dense linears with bias:
  `q/k/v/o: Linear(768 -> 768)`,
  `intermediate: Linear(768 -> 3072)`,
  `output: Linear(3072 -> 768)`.
- GELU FFN activation.
- Residual add + LayerNorm after attention output and after FFN output.
- Tanh pooler/classification activation.

Attention primitives:

- Noncausal dense self-attention, MHA, `12` heads, head dim `64`.
- Batched QK matmul `[B,12,S,64] @ [B,12,64,S] -> [B,12,S,S]`.
- Scale by `1 / sqrt(64)`.
- Add relative-position bias `[B,12,S,S]`.
- Add extended padding mask `[B,1,1,S]` or supplied `[B,1,S,S]`.
- Softmax over key axis, then attention-probs @ V.
- Optional attention output tensor `[B,12,S,S]` if requested.

Position/relative-bias ops:

- Relative-distance matrix construction from sequence arange.
- MPNet/T5-style bidirectional bucketization with exact small buckets and
  log-spaced large buckets.
- Relative bias embedding lookup and axis reorder
  `[S,S,H] -> [1,H,S,S] -> [B,H,S,S]`.

Task and wrapper heads:

- MLM head: `Linear(768 -> 768) -> GELU -> LayerNorm(768) -> tied Linear(768 -> vocab_size)`.
- Pooler: first token `[B,768] -> Linear(768 -> 768) -> Tanh`.
- Sequence classification: first token -> dropout -> `Linear(768 -> 768)` ->
  Tanh -> dropout -> `Linear(768 -> num_labels)`.
- Multiple choice: flatten `[B,C,S]` to `[B*C,S]`, pooler, dropout,
  `Linear(768 -> 1)`, reshape `[B,C]`.
- Token classification: dropout, `Linear(768 -> num_labels)` over `[B,S,768]`.
- QA: `Linear(768 -> num_labels)`, usually `num_labels=2`, split start/end and
  squeeze to `[B,S]`.
- Sentence-transformers: masked mean pooling over tokens using attention mask,
  optional L2 normalize over embedding dimension.

Generation/cache ops:

- Not applicable. MPNet is an encoder/fill-mask/sentence-embedding family here.

## 5. Layer/block breakdown

Embedding stage:

```text
if input_ids:
  position_ids = cumsum(input_ids != 1) * (input_ids != 1) + 1
else:
  position_ids = arange(2, S + 2).expand(B,S)
x = word_embedding(input_ids) or inputs_embeds
x = x + position_embedding(position_ids)
x = LayerNorm(x)
```

Encoder setup:

```text
relative_position = arange(S)[None, :] - arange(S)[:, None]
bucket = relative_position_bucket(relative_position, num_buckets=32, max_distance=128)
position_bias = relative_attention_bias(bucket)    # [S,S,12]
position_bias = permute to [1,12,S,S], expand to [B,12,S,S]
```

Encoder block, repeated 12 times:

```text
q = Linear(768 -> 768)(x).view(B,S,12,64).transpose(1,2)
k = Linear(768 -> 768)(x).view(B,S,12,64).transpose(1,2)
v = Linear(768 -> 768)(x).view(B,S,12,64).transpose(1,2)
scores = q @ k.transpose(-1,-2) / sqrt(64)
scores = scores + position_bias + extended_attention_mask
p = softmax(scores, dim=-1)
attn = p @ v
attn = attn.transpose/contiguous/view to [B,S,768]
x_attn = LayerNorm(dropout(Linear(768 -> 768)(attn)) + x)
h = GELU(Linear(768 -> 3072)(x_attn))
x = LayerNorm(dropout(Linear(3072 -> 768)(h)) + x_attn)
```

All projections in the inspected source have bias. There is no pre-norm variant
or gated MLP variant.

## 6. Attention requirements

- Type: encoder-only, noncausal, bidirectional self-attention.
- Head pattern: standard MHA, `num_key_value_heads == num_attention_heads == 12`
  by construction; no KV sharing/repeat expansion.
- Query/key/value width: all `hidden_size=768`, split into `12 x 64`.
- Sequence shape: square self-attention over `[S,S]`; no cross-attention.
- Masking: source accepts a 2D padding mask `[B,S]` or 3D attention mask
  `[B,S,S]` through the shared `get_extended_attention_mask`. Values are
  converted to additive mask with `0.0` for attend and `torch.finfo(dtype).min`
  for masked positions.
- Relative bias: additive `[B,H,S,S]` before the padding mask and softmax.
- Packed/varlen: not present. A first integration can require padded dense
  batches and contiguous row-major hidden states.
- Sliding/local/block sparse: not present.
- KV cache/decode: not present.
- FlashAttention/SDPA compatibility: possible optimization only if the backend
  can accept an additive per-head relative bias plus padding mask. Vanilla
  dense attention is the parity path.

## 7. Position encoding and custom math

MPNet uses both absolute position embeddings and relative-position bucket bias.
For normal `input_ids`, absolute position ids skip pad tokens and start at
`padding_idx + 1 == 2`; pads map to id `1`.

Source-equivalent position id helper:

```python
def mpnet_position_ids(input_ids, padding_idx=1):
    mask = (input_ids != padding_idx).int()
    incremental = cumsum(mask, dim=1).type_as(mask) * mask
    return incremental.long() + padding_idx
```

Relative buckets:

```python
def mpnet_relative_position_bucket(relative_position, num_buckets=32, max_distance=128):
    ret = 0
    n = -relative_position
    half = num_buckets // 2
    ret += (n < 0).long() * half
    n = abs(n)
    max_exact = half // 2
    large = max_exact + (log(n.float() / max_exact) /
                         log(max_distance / max_exact) *
                         (half - max_exact)).long()
    large = minimum(large, full_like(large, half - 1))
    return ret + where(n < max_exact, n, large)
```

Precompute opportunities:

- For fixed or bucketed `S`, bucket indices `[S,S]` can be precomputed.
- The bias values depend on the learned `relative_attention_bias` weights and
  batch expansion. For static `S`, the gathered `[H,S,S]` bias can be cached per
  artifact/session, but changing weights or sequence length invalidates it.
- Absolute position ids depend on input padding unless caller supplies
  `inputs_embeds`, so they are normally runtime integer work.

## 8. Preprocessing and input packing

Tokenizer/source contract:

- WordPiece tokenizer with BERT normalizer/pre-tokenizer.
- Default special tokens: `<s>` id 0, `<pad>` id 1, `</s>` id 2, `[UNK]`,
  `<mask>`.
- Single sequence template: `<s> A </s>`.
- Pair template: `<s> A </s></s> B </s>`.
- `model_input_names = ["input_ids", "attention_mask"]`; no token type ids.
- `microsoft/mpnet-base` tokenizer config uses `model_max_length=512` and
  `do_lower_case=true`; sentence-transformers wrappers may override effective
  wrapper settings such as `max_seq_length=384`.

GPU graph inputs:

- `input_ids[B,S]` int64/int32 acceptable after frontend normalization.
- `attention_mask[B,S]` with 1 for valid tokens and 0 for pads.
- Optional `position_ids[B,S]` and optional `inputs_embeds[B,S,768]`; source
  rejects providing both `input_ids` and `inputs_embeds`.

Sentence embedding wrapper:

- Mean pooling must use the attention mask, not a raw average over padded tokens.
- `all-mpnet-base-v2` then applies vector L2 normalization; `paraphrase` modules
  inspected did not include `2_Normalize`.

CPU/data-pipeline work can own tokenization, lowercasing, truncation, and special
token packing. GPU/runtime parity must own embedding lookup, position-id math if
not precomputed by the caller, attention mask conversion, and pooling.

## 9. Graph rewrite / lowering opportunities

### Rewrite: fused QKV projection

Source pattern:

```text
q = Linear(H -> H)(x)
k = Linear(H -> H)(x)
v = Linear(H -> H)(x)
reshape each to [B,Hd,S,D]
```

Replacement:

```text
packed_qkv = Linear(H -> 3H)(x)
split packed_qkv as [q, k, v] along last dim
reshape/split heads
```

Preconditions:

- All three inputs are the same `hidden_states` tensor.
- Q/K/V linears have independent dense weights and biases with identical
  input/output width.
- Weight packing order must be all-Q rows, all-K rows, all-V rows:
  `W_packed = concat([Wq, Wk, Wv], dim=0)`, `b_packed = concat([bq,bk,bv])`
  for PyTorch linear storage `[out_features, in_features]`.
- No consumer reads intermediate Q/K/V tensors other than attention.

Failure cases:

- Debug/export mode requiring separate named Q/K/V outputs.
- Nonstandard checkpoints with quantized or packed source weights that already
  impose a different layout.

Parity test sketch:

- Compare separate and packed projections for random `[B,S,768]`, including
  non-multiple batch/sequence buckets and fp32/fp16 tolerances.

### Rewrite: relative-bias precompute by sequence bucket

Source pattern:

```text
arange -> relative_position -> bucket -> embedding lookup -> permute -> expand
```

Replacement:

```text
precomputed_bucket[S,S] -> gather relative_attention_bias -> [H,S,S]
```

Preconditions:

- `S` belongs to a known compile/profile bucket or a session cache key.
- `relative_attention_num_buckets` and `max_distance=128` match source.
- Shared bias weight is not mutated after artifact load.

Failure cases:

- Dynamic arbitrary sequence lengths with no runtime cache path.
- Caller-visible custom `position_ids` in a future encoder path.

### Rewrite: dense attention with additive bias to provider attention

Source pattern:

```text
scores = q @ k.T * scale + relative_bias + additive_padding_mask
softmax(scores) @ v
```

Replacement:

```text
bias-aware attention provider(q, k, v, additive_bias=relative_bias + mask)
```

Preconditions:

- Provider supports noncausal MHA, per-head additive bias, and padding mask with
  source-equivalent dtype/min-value behavior.
- No dropout in inference.
- Output attentions are not requested, or provider can return probabilities.

Failure cases:

- Flash/SDPA provider that cannot consume arbitrary `[B,H,S,S]` additive bias.
- `output_attentions=True` without probability materialization support.

### Rewrite: sentence-transformers masked mean pooling

Source wrapper pattern:

```text
emb = last_hidden_state
mask = attention_mask[..., None]
pooled = sum(emb * mask, dim=1) / clamp(sum(mask, dim=1), min=eps)
optional normalize(pooled, dim=-1)
```

Replacement:

```text
fused masked row reduction + reciprocal + optional L2 normalize
```

Preconditions:

- Mask uses 1 for valid tokens and 0 for pads.
- Pooling config has `mean_tokens=true` and no CLS/max/sqrt pooling enabled.

Failure cases:

- Multiple pooling modes enabled or custom prompt-token exclusion in a newer
  sentence-transformers wrapper not represented by inspected configs.

## 10. Kernel fusion candidates

Highest priority:

- Bias-aware dense MHA for noncausal encoder attention. Relative bias is the
  main family-specific blocker for dropping into a generic flash attention path.
- QKV packed GEMM + reshape/transposes. This removes three separate linears per
  layer and gives a stable ABI for attention providers.
- LayerNorm + residual/dropout identity fusion for inference around attention
  output and FFN output.
- FFN fused epilogue: `Linear(768 -> 3072) + GELU` and
  `Linear(3072 -> 768) + residual + LayerNorm`.

Medium priority:

- Relative-bias gather/precompute cache by sequence bucket.
- MLM head dense + GELU + LayerNorm + tied decoder path for fill-mask.
- Masked mean pooling + L2 normalization for sentence-transformers embeddings.

Lower priority:

- Training dropout and losses.
- Output-attention probability materialization optimizations.
- Multiple-choice flatten/unflatten convenience lowering.

## 11. Runtime staging plan

Stage 1: config and weights.

- Parse MPNet config and reject unsupported traps: pad id not 1, hidden not
  divisible by heads, non-GELU activation if not supported.
- Load base encoder weights and preserve MLM tied aliases when present.

Stage 2: single-block parity.

- Implement embeddings, position ids, relative bucket bias, one encoder layer,
  and dense attention reference.

Stage 3: full encoder parity.

- Run all 12 layers with padding masks and dynamic sequence buckets.
- Validate `last_hidden_state` and optional pooler output.

Stage 4: heads.

- Add MLM, sequence classification, token classification, QA, multiple-choice
  first-token/pooler heads.
- Add sentence-transformers masked mean pooling and optional normalization.

Stage 5: optimized attention/rewrite path.

- Add QKV packing, relative-bias caching, and bias-aware attention provider
  admission. Keep dense fallback mandatory until provider supports additive
  per-head bias.

Stage 6: production polish.

- Add sequence-length/profile sweeps, output-attention fallback, and wrapper
  metadata ingestion for sentence-transformers repos.

Training losses and dropout can be stubbed or deferred for inference-first
integration.

## 12. Parity and validation plan

- Unit test `create_position_ids_from_input_ids` for pad/non-pad mixtures,
  all-pad rows, no-pad rows, and `inputs_embeds` sequential ids.
- Unit test `relative_position_bucket` against source for sequence lengths
  `1, 2, 8, 32, 129, 512`, including positive/negative distances and saturation.
- Single-layer parity with random weights for `[B,S]` cases such as
  `[1,1]`, `[2,7]`, `[3,128]`, and padded masks.
- Full encoder parity for `microsoft/mpnet-base` on short text batches, checking
  last hidden state and pooler output.
- Masked-LM parity for known masked-token examples, checking logits at masked
  positions and tied-weight output.
- Sentence-transformers parity for `all-mpnet-base-v2`: encoder output,
  masked mean pooled vector, normalized embedding, and cosine similarity matrix.
- Head parity for token classification and QA using random initialized heads or
  accessible fine-tuned checkpoints.

Recommended inference tolerances:

- fp32: `atol=1e-5`, `rtol=1e-4` for encoder/head outputs.
- fp16/bf16 optimized paths: start at `atol=2e-2`, `rtol=2e-2`; tighten per
  provider after attention and LayerNorm accumulation policy is fixed.

## 13. Performance probes

- Tokenization throughput separately from encoder throughput.
- Encoder-only latency/throughput for batch sweep `B=1, 8, 32, 128`.
- Sequence sweep `S=8, 32, 128, 384, 512` with and without padding.
- Relative-bias construction cost versus cached bucket/gather path.
- Separate Q/K/V GEMM versus packed QKV GEMM.
- Dense attention versus bias-aware provider attention once available.
- FFN GEMM and LayerNorm/residual fusion contribution by layer.
- MLM logits cost for full sequence versus selected masked positions if a
  product only needs mask-token logits.
- Sentence-transformers end-to-end embeddings/sec, including masked pooling and
  L2 normalize.

Benchmark observations are not included; the above are source-derived probes.

## 14. Skip/defer list

- Training losses, dropout behavior, and gradient checkpointing.
- Permuted-language-model pretraining workflow beyond the existing MLM head.
- Beam search, decode loop, autoregressive KV cache, speculative decoding.
- Sparse/local attention and packed varlen attention.
- Quantized/packed checkpoint formats unless a specific MPNet deployment needs
  them; dense safetensors/PyTorch weights are the safe first path.
- Output attentions on optimized provider path; keep dense fallback.
- Tokenizer implementation in GPU runtime; keep CPU preprocessing first.

## 15. Final implementation checklist

- [ ] Parse MPNet config and sentence-transformers wrapper metadata.
- [ ] Load embeddings, encoder, relative bias, and head weights.
- [ ] Preserve MLM decoder/bias tied aliases.
- [ ] Implement MPNet position-id generation with pad id 1 guard.
- [ ] Implement relative-position bucket op and bias gather.
- [ ] Implement dense noncausal MHA with additive per-head relative bias.
- [ ] Implement post-norm attention and FFN blocks.
- [ ] Add first-token pooler and task heads.
- [ ] Add sentence-transformers masked mean pooling and optional L2 normalize.
- [ ] Add QKV packing rewrite with exact weight/bias concatenation tests.
- [ ] Add relative-bias cache/precompute by sequence bucket.
- [ ] Gate optimized attention on provider support for `[B,H,S,S]` additive bias.
- [ ] Add single-layer, full-encoder, MLM, and sentence-embedding parity tests.
- [ ] Benchmark sequence/batch sweeps and relative-bias overhead.

Gated gaps for DinoML:

- Additive per-head relative bias is required for optimized attention; generic
  FlashAttention/SDPA lowering is unsafe without this ABI.
- Runtime integer position-id generation and relative-bucket math need explicit
  ops or a precompute/cache contract.
- Sentence-transformers integration is a wrapper graph with masked mean pooling
  and optional normalization, not just `MPNetModel`.
- Tied MLM decoder aliases must be preserved by the loader/lowering path.
- Layout translation around attention axes needs guards: source uses
  `[B,S,H] -> [B,heads,S,head_dim] -> [B,S,H]`, and softmax is always over the
  key axis `-1`.
