# Transformers Funnel Audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4 from local checkout transformers
Model id: funnel-transformer/{small,small-base,medium,intermediate,large,xlarge}
Config source: HF config.json snapshots saved under agents/plans/transformers/funnel/_sources/
Source files inspected:
  transformers/src/transformers/models/funnel/configuration_funnel.py
  transformers/src/transformers/models/funnel/modeling_funnel.py
  transformers/src/transformers/models/funnel/tokenization_funnel.py
  transformers/docs/source/en/model_doc/funnel.md
  transformers/tests/models/funnel/test_modeling_funnel.py
  transformers/src/transformers/activations.py
Any missing files or assumptions:
  No remote code is required for the sampled checkpoints. This report scopes DinoML first target to inference for encoder outputs and task heads, not training losses.
```

The authoritative implementation for runtime behavior is `modeling_funnel.py`; there is no modular generated source variant in this family. The docs explicitly distinguish full checkpoints from `-base` checkpoints: full checkpoints are intended for `FunnelModel`, masked LM, pretraining, token classification, and QA; `-base` checkpoints are intended for `FunnelBaseModel`, sequence classification, and multiple choice.

## 2. High-level architecture

Funnel is a bidirectional text encoder with sequence-length reduction between encoder blocks. The full model adds an upsampling decoder so token-level heads regain first-block sequence length.

```text
WordPiece input ids + token type ids + attention mask
  -> token embedding -> LayerNorm -> Dropout
  -> Funnel encoder blocks with relative attention and sequence pooling
  -> either downsampled encoder ABI (FunnelBaseModel)
  -> or repeat-interleave upsample + residual from first block + decoder layers (FunnelModel)
  -> task head
```

Stage decomposition:

- CPU/data pipeline: Funnel WordPiece tokenizer with `<cls>` token type id 2, segment A type id 0, segment B type id 1, and separator ids from tokenizer vocabulary.
- GPU/runtime encoder: dense embeddings, relative multi-head self-attention, sequence pooling before later blocks, FFN, LayerNorm.
- Optional full-model decoder: upsample reduced hidden state by sequence repeat, add first-block hidden state, run `num_decoder_layers` unpooled Funnel layers.
- Heads: classification heads can use the base downsampled encoder and pool by `last_hidden_state[:, 0]`; token-level heads require full upsampled `FunnelModel`.

## 3. Important config dimensions

Source default dimensions from `FunnelConfig`:

| field | default | operator impact |
| --- | ---: | --- |
| `vocab_size` | 30522 | embedding rows and LM projection rows |
| `block_sizes` | [4, 4, 4] | encoder layers per block; pooling begins at block index > 0 |
| `block_repeats` | [1, 1, 1] if omitted | repeats each layer application without cloning layer weights |
| `num_decoder_layers` | 2 | full-model upsampling decoder layers |
| `d_model` / hidden size | 768 | hidden width |
| `n_head` | 12 | MHA heads |
| `d_head` | 64 | per-head Q/K/V width |
| `d_inner` | 3072 | FFN hidden width |
| `hidden_act` | `gelu_new` | tanh GELU approximation |
| `layer_norm_eps` | 1e-9 | LayerNorm epsilon |
| `pooling_type` | `mean` | mean or max pooling; checkpoints use mean |
| `attention_type` | `relative_shift` | relative shift or factorized relative position attention |
| `separate_cls` | true | keeps CLS separate during pooling |
| `truncate_seq` | true | drops last non-CLS token before pooling when needed |
| `pool_q_only` | true | first layer after pooling attends pooled Q to unpooled K/V |

Representative checkpoint sweep from saved `config.json` snapshots:

| checkpoint | architecture | blocks | repeats | encoder applications | decoder layers | d_model | heads x d_head | d_inner | attention |
| --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | --- |
| funnel-transformer/small | FunnelModel | [4,4,4] | [1,1,1] | 12 | 2 | 768 | 12 x 64 | 3072 | relative_shift |
| funnel-transformer/small-base | FunnelBaseModel | [4,4,4] | [1,1,1] | 12 | config has 2 but unused | 768 | 12 x 64 | 3072 | relative_shift |
| funnel-transformer/medium | FunnelModel | [6,3,3] | [1,2,2] | 18 | 2 | 768 | 12 x 64 | 3072 | relative_shift |
| funnel-transformer/intermediate | FunnelModel | [6,6,6] | [1,1,1] | 18 | 2 | 768 | 12 x 64 | 3072 | relative_shift |
| funnel-transformer/large | FunnelModel | [8,8,8] | [1,1,1] | 24 | 2 | 1024 | 16 x 64 | 4096 | relative_shift |
| funnel-transformer/xlarge | FunnelModel | [10,10,10] | [1,1,1] | 30 | 2 | 1024 | 16 x 64 | 4096 | relative_shift |

All sampled configs include historical `rel_attn_type="factorized"`, but the current source reads `attention_type`, not `rel_attn_type`. DinoML should treat `rel_attn_type` as ignored for this source basis.

## 3a. Family variation traps

- Full versus base checkpoints are not interchangeable for all heads. `FunnelBaseModel` returns a reduced sequence; full `FunnelModel` restores token length through the decoder.
- `block_repeats` repeats the same physical layer module application. Lowering must preserve weight sharing within repeated applications.
- `pool_q_only=True` creates rectangular attention at the first layer of each pooled block: pooled query length attends over the previous unpooled key/value length.
- `separate_cls` and `truncate_seq` change pooling and upsampling shape equations around token 0.
- `attention_type="factorized"` is implemented but not used by sampled configs. It replaces relative-shift gather with two additional einsum paths over factorized sinusoidal matrices.
- `type_vocab_size` appears in configs/tests but the model has no token-type embedding table. Token types only enter relative segment attention through equality matrices and a learned `[2, n_head, d_head]` segment embedding.
- `max_position_embeddings` appears in configs but position embeddings are generated from runtime `seq_len`; there is no learned position table in this source.
- There is no autoregressive decode or KV cache. All attention is bidirectional encoder-style self-attention.
- Source tensor layout is `[B, T, H]` for hidden states and `[B, heads, Q, K]` for attention scores. No NHWC/channel-last layout pass is relevant.

## 4. Operator coverage checklist

Tensor/layout ops:

- Embedding lookup for `input_ids -> [B,T,d_model]`.
- Reshape/view for Q/K/V `[B,T,n_head*d_head] -> [B,T,n_head,d_head]`.
- Flatten/reshape for multiple-choice `[B,C,T] -> [B*C,T]` and logits `[B*C,1] -> [B,C]`.
- Slice/index for `hidden[:, 0]`, CLS separation, suffix truncation, and first-block hidden selection.
- Concatenate along sequence axis for CLS-preserving pooling and upsampling.
- Repeat interleave along sequence axis for full-model decoder upsample.
- Pad on sequence axis for `truncate_seq=True` upsampling.
- Split/squeeze/contiguous for QA logits.

Neural network primitives:

- `LayerNorm(d_model, eps=1e-9)` after embeddings, attention residual, and FFN residual.
- Linear Q: `d_model -> n_head*d_head`, bias false.
- Linear K/V: `d_model -> n_head*d_head`, bias true.
- Linear output projection: `n_head*d_head -> d_model`, bias true.
- FFN: `Linear(d_model -> d_inner)` + `gelu_new` + `Linear(d_inner -> d_model)`.
- Dropout is present in source but no-op for inference.
- Classification head: `Linear(d_model -> d_model)` + tanh + `Linear(d_model -> num_labels or 1)`.
- Pretraining discriminator head: `Linear(d_model -> d_model)` + `gelu_new` + `Linear(d_model -> 1)` + squeeze.
- LM head: `Linear(d_model -> vocab_size)`, tied to token embeddings by `_tied_weights_keys`.
- Token classification / QA heads: `Linear(d_model -> num_labels)`, QA expects `num_labels=2`.

Attention primitives:

- Dense noncausal MHA with rectangular Q/K/V support.
- Attention score math:
  - content score `einsum("bind,bjnd->bnij", q + r_w_bias, k)`.
  - relative positional score, either relative-shift or factorized.
  - relative token-type score from equality mask and learned segment embeddings.
  - additive mask `score -= 1e6 * (1 - attention_mask)`.
  - upcast score to fp32 before mask, softmax on last dim with output dtype requested as original score dtype.
  - value matmul `einsum("bnij,bjnd->bind", prob, v)`.

Position/relative-bias ops:

- Runtime sinusoidal relative vectors; no learned position table.
- Gather of relative position rows for `relative_shift`.
- `_relative_shift_gather` reshape/slice/reshape crop pattern.
- Optional factorized relative attention path with sin/cos construction and two einsums.

Pooling/downsampling/upscaling ops:

- Sequence pooling by reshaping `[B,T]` or `[B,T,H]` to rank-4 and applying `avg_pool2d` / `max_pool2d` with kernel `(2,1)`, stride `(2,1)`, `ceil_mode=True`.
- Min pooling for attention masks implemented as negative max pooling over the float mask.
- Stride slicing for token type and CLS masks along query/key axes.
- Full-model decoder upsample by repeat-interleave with CLS preservation and optional pad/crop.

Preprocessing-coupled ops:

- Funnel tokenizer emits token type id 2 for `<cls>`, 0 for sequence A, 1 for sequence B.
- GPU graph consumes `input_ids`, optional `attention_mask`, optional `token_type_ids`, or direct `inputs_embeds`.

## 5. Layer/block breakdown

Embedding:

```text
x = Embedding(input_ids) or inputs_embeds
x = LayerNorm(x, eps=1e-9)
x = Dropout(x)
```

Encoder block `b`, repeated over configured physical layers and `block_repeats[b]`:

```text
if b > 0 and current length can still pool:
  pooled_hidden = pool_tensor(hidden, mean/max, stride=2)
  if pool_q_only:
    first layer query = pooled_hidden
    first layer key/value = previous hidden
    token_type_mat and cls_mask are stride-pooled only on query axis before attention
  else:
    query/key/value = pooled_hidden
    attention mask is min-pooled

for each layer application:
  q = Linear(query, bias=False).view(B, Q, N, D)
  k = Linear(key, bias=True).view(B, K, N, D)
  v = Linear(value, bias=True).view(B, K, N, D)
  score = content_relative_segment_scores(q, k, token_type_ids, relative_positions)
  score = mask(score, attention_mask)
  attn = softmax(score, dim=-1)
  context = einsum(attn, v).reshape(B, Q, N*D)
  hidden = LayerNorm(query + Dropout(Linear(context)))
  hidden = LayerNorm(hidden + Dropout(Linear(gelu_new(Linear(hidden)))))
  after first rectangular pooled layer, pool key-side masks/attention inputs
```

Full-model decoder:

```text
u = upsample(final_encoder_hidden, stride=2 ** (num_blocks - 1), target_len=first_block_hidden.T)
hidden = u + first_block_hidden
repeat num_decoder_layers:
  run normal square FunnelLayer without sequence pooling
```

For sampled full checkpoints, the decoder upsample stride is `4` because there are three blocks and pooling happens before blocks 1 and 2.

## 6. Attention requirements

Funnel attention is noncausal encoder self-attention. There is no cross-attention, no sliding window, no GQA/MQA, no KV cache, and no autoregressive prefill/decode mode.

Required attention shape cases:

- Block 0 and normal layers: `Q == K == current sequence length`.
- First layer of pooled block when `pool_q_only=True`: `Q == pooled length`, `K == previous length`.
- Heads: `n_head`, with `d_head`; sampled configs keep `d_model == n_head*d_head`, but source should not infer that for validation unless checked.
- Attention mask is `[B,K]`; token-type matrix is `[B,Q,K]` after stride pooling.

Masking and math order:

```text
q = q_linear(query).view(B,Q,N,D) * (1/sqrt(D))
k = k_linear(key).view(B,K,N,D)
content = einsum(q + r_w_bias * scale, k)
score = content + positional_attn + token_type_attn
score_fp32 = score.float()
score_fp32 -= 1e6 * (1 - attention_mask[:,None,None].float())
prob = softmax(score_fp32, dim=-1, dtype=score.dtype)
out = einsum(prob, v)
```

FlashAttention/SDPA compatibility is limited. The content attention resembles standard dense attention, but the source adds a precomputed per-layer relative positional score and a token-type score before softmax. A first integration can lower this as explicit score tensor construction plus softmax plus BMM. A fused attention provider would need an ABI for additive relative/segment bias with rectangular Q/K and CLS masks.

## 7. Position encoding and custom math

Funnel uses relative sinusoidal position math. For sampled configs, `attention_type="relative_shift"` is the active path.

Concise relative-shift score sketch:

```python
def funnel_relative_shift_attention(q_head, r_kernel, r_r_bias, r, context_len, shift, scale):
    # q_head: [B, Q, N, D], r: [R, H], r_kernel: [H, N, D]
    r_head = einsum("td,dnh->tnh", r, r_kernel)
    raw = einsum("binh,tnh->bnit", q_head + r_r_bias * scale, r_head)
    return relative_shift_gather(raw, context_len, shift)
```

`relative_shift_gather` is:

```python
def relative_shift_gather(x, context_len, shift):
    B, N, Q, R = x.shape
    x = x.reshape(B, N, R, Q)
    x = x[:, :, shift:, :]
    x = x.reshape(B, N, Q, R - shift)
    return x[..., :context_len]
```

Position vectors are generated from runtime `seq_len` using `arange`, inverse frequencies, sin/cos, and gather indices derived from pooled positions. These can be precomputed per admitted sequence bucket and config because they depend on sequence length, dtype, device, block index, `separate_cls`, and `truncate_seq`, but not on token values.

Token-type relative score:

```python
same_segment = token_type_ids[:, :, None] == token_type_ids[:, None]
cls = token_type_ids == 2
token_type_mat = same_segment | cls[:, :, None] | cls[:, None]
bias = einsum("bind,snd->bnis", q_head + r_s_bias * scale, seg_embed)
score = where(token_type_mat[:, None], same_bias, diff_bias)
```

## 8. Preprocessing and input packing

Tokenizer coupling matters because `<cls>` receives token type id 2, not the usual segment id 0:

```text
single: <cls>:2 A:0 <sep>:0
pair:   <cls>:2 A:0 <sep>:0 B:1 <sep>:1
```

Runtime inputs:

- `input_ids`: `[B,T]` int token ids, mutually exclusive with `inputs_embeds`.
- `inputs_embeds`: `[B,T,d_model]` dense embeddings, bypassing embedding lookup.
- `attention_mask`: optional `[B,T]`, default ones. Converted to embedding dtype inside encoder for pooling.
- `token_type_ids`: optional `[B,T]`, default zeros. Used to create a dense `[B,T,T]` equality matrix with CLS override.
- `position_ids` is accepted by `FunnelBaseModel.forward` but not used by the source.

No image/audio/video processor, packed varlen metadata, or generation controller exists.

## 9. Graph rewrite / lowering opportunities

### Rewrite: QKV projection canonicalization

Source pattern:

```text
q = Linear(hidden, Wq, no bias)
k = Linear(hidden, Wk, bias)
v = Linear(hidden, Wv, bias)
```

Replacement:

```text
For square self-attention layers, optional packed QKV GEMM -> split [q,k,v].
For rectangular pooled layers, keep Q separate from packed KV because query input differs from key/value input.
```

Preconditions:

- `query is key is value` for QKV packing.
- Preserve Q no-bias while K/V have bias. Packed bias must include zero Q bias.
- Packed weight split order must be `[q, k, v]`.
- Do not apply to the first pooled layer with `pool_q_only=True`.

Parity test sketch: one block with no pooling and one block with pooling; compare packed and unpacked scores before softmax.

### Rewrite: sequence pool as specialized 1D pool

Source pattern:

```text
[B,T,H] -> [B,1,T,H] -> avg_pool2d/max_pool2d(kernel=(2,1), stride=(2,1), ceil_mode=True) -> [B,T2,H]
```

Replacement:

```text
Specialized SequencePool2(stride=2, mode=mean|max|min, ceil=true, preserve_cls, truncate_seq)
```

Preconditions:

- Pooling is only along sequence axis.
- Kernel and stride are exactly 2.
- Preserve source CLS pre-concat and truncation behavior.
- Min pooling is `-max_pool(-x)`.

Failure cases:

- General `avg_pool2d` without `ceil_mode` is not parity.
- Layout passes must not reinterpret `[B,T,H]` as image layout.

### Rewrite: decoder upsample as indexed row copy

Source pattern:

```text
repeat_interleave(x[:,1:], repeats=stride, dim=1), optional pad, crop, concat CLS
```

Replacement:

```text
Generate output token i from source floor((i-1)/stride)+1, with CLS copied from source 0.
```

Preconditions:

- `stride == 2 ** (num_blocks - 1)`.
- `target_len` equals first-block hidden length.
- Guard `separate_cls` and `truncate_seq`.

### Rewrite: relative position cache per sequence bucket

Source pattern:

```text
arange -> sin/cos -> gather relative rows per block and shift
```

Replacement:

```text
Compile or runtime-cache bucketed position tables keyed by seq_len, dtype, block index, attention_type, separate_cls, truncate_seq.
```

Failure cases:

- Dynamic sequence lengths not in admitted buckets need fallback construction or rejection.
- Factorized attention has a different table ABI.

### Rewrite: explicit score-bias attention

Source pattern:

```text
content_score + positional_attn + token_type_attn + mask -> softmax -> value matmul
```

Replacement:

```text
DenseAttentionWithAdditiveBias(q,k,v,bias,mask)
```

Preconditions:

- Bias tensor shape is `[B,N,Q,K]` or broadcastable equivalent after materialization.
- Mask uses source `-1e6` convention.
- Upcast-before-softmax behavior is preserved.

## 10. Kernel fusion candidates

Highest priority:

- LayerNorm residual blocks: Funnel is LayerNorm-heavy and DinoML currently lists LayerNorm as unported.
- SequencePool2 with CLS/truncation guards: required for encoder ABI and not covered by current fixed floor-mode pooling.
- Relative attention score construction: content GEMM/BMM plus relative positional and segment biases dominate complexity.
- Softmax + value BMM over `[B,N,Q,K]`, including rectangular Q/K.

Medium priority:

- FFN GEMM + `gelu_new` + GEMM fusion for `d_model -> d_inner -> d_model`.
- Full-model upsample indexed copy plus residual add.
- Packed QKV for non-pooled square attention layers.
- Token-type matrix construction and segment-bias `where` lowering.

Lower priority:

- Factorized attention path; implemented by source but absent in sampled official configs.
- Output attentions and hidden-state tuple materialization; useful for debugging but not first runtime target.
- Training losses and dropout.

## 11. Runtime staging plan

Stage 1: config and weight loading.

- Parse `FunnelConfig`.
- Reject unsupported `attention_type` except `relative_shift` initially.
- Preserve repeated physical layer applications from `block_repeats`.
- Preserve tied LM head and embedding weight alias for masked LM.

Stage 2: base encoder classification path.

- Target `FunnelBaseModel` plus sequence classification/multiple choice.
- Implement embedding, LayerNorm, dense attention with explicit relative/segment bias, FFN, and sequence pooling.
- Validate downsampled output length and `last_hidden_state[:,0]` pooling.

Stage 3: full model token-level path.

- Add decoder upsample and decoder layers.
- Enable masked LM, token classification, QA, and pretraining discriminator inference.

Stage 4: optimized lowering.

- Bucket/cache relative position tables.
- Add specialized sequence pooling kernel.
- Add attention-bias fusion and packed QKV where safe.

Stage 5: broader variants.

- Admit `attention_type="factorized"`, `pooling_type="max"`, `pool_q_only=False`, and `separate_cls=False` only with direct parity coverage.

## 12. Parity and validation plan

Recommended source parity tests:

- Relative position table construction for odd/even `T`, `separate_cls` true/false, `truncate_seq` true/false.
- `pool_tensor` parity for `[B,T]` masks and `[B,T,H]` hidden states with odd lengths, including mean/max/min modes and `ceil_mode=True`.
- `_relative_shift_gather` random tensor parity across square and rectangular attention.
- Single `FunnelRelMultiheadAttention` parity with token type ids containing CLS id 2.
- Single `FunnelLayer` and one full block parity in fp32.
- Full `FunnelBaseModel` output shape parity: e.g. local tests expect input length 7 to reduce to 2 with default truncate and to 3 with `truncate_seq=False`.
- Full `FunnelModel` output shape parity: restored `[B,T,d_model]`.
- Head parity for sequence classification, multiple choice, masked LM, token classification, and QA logits.

Tolerances:

- fp32: start with `rtol=1e-4, atol=1e-4`, matching local integration test style.
- fp16/bf16: use wider tolerances after attention upcast/mask/softmax ordering is matched; validate logits rather than only hidden means.

## 13. Performance probes

- Sequence length sweep over odd/even lengths: isolate pooling shape and relative table construction overhead.
- Encoder-only base throughput for classification path.
- Full-model throughput with decoder upsample for token-level path.
- Rectangular pooled attention versus square attention timing by block.
- Relative position table cache hit/miss overhead.
- Token-type segment-bias path with all-zero token types versus pair inputs with CLS id 2.
- FFN GEMM throughput for 768/3072 and 1024/4096 variants.
- Memory footprint for explicit `[B,N,Q,K]` score and bias tensors before any fused attention provider.

## 14. Skip/defer list

Safe to defer for first integration:

- Training losses and dropout randomness.
- `output_attentions=True` and `output_hidden_states=True` materialization, except where full-model decoder needs the first-block hidden internally.
- `attention_type="factorized"` unless a target config requires it.
- `pooling_type="max"` and `pool_q_only=False` unless a target config requires them.
- TensorFlow checkpoint conversion.
- Slow tokenizer implementation inside DinoML runtime; tokenization can remain CPU/data-pipeline work.
- Generic image/layout NHWC work; this is a text-only `[B,T,H]` model.

## 15. Final implementation checklist

- [ ] Parse `FunnelConfig` and reject unsupported historical/ignored config combinations.
- [ ] Load full and `-base` checkpoint weights with correct architecture admission.
- [ ] Preserve `block_repeats` physical weight sharing.
- [ ] Implement or admit `LayerNorm`.
- [ ] Implement `gelu_new` tanh approximation.
- [ ] Implement sequence pooling with CLS/truncate and `ceil_mode=True`.
- [ ] Implement runtime/token-bucket relative position table generation for `relative_shift`.
- [ ] Implement `_relative_shift_gather`.
- [ ] Implement token-type matrix with CLS id 2 override.
- [ ] Implement dense MHA with additive relative/segment bias and rectangular Q/K.
- [ ] Implement full-model repeat-interleave upsample and residual add.
- [ ] Implement base sequence classification and multiple-choice heads.
- [ ] Implement full masked LM, token classification, QA, and discriminator heads.
- [ ] Add single-op parity tests for pooling, relative shift, token-type bias, and upsample.
- [ ] Add one-block and full-model parity tests against Transformers.
- [ ] Benchmark encoder-only, full-model, attention, pooling, and relative table cache paths.

## Gated DinoML gaps

- `LayerNorm` is a hard gate for every Funnel path.
- Current DinoML pooling coverage is floor-mode `avg_pool1d/avg_pool2d` and `max_pool2d`; Funnel requires sequence-axis pooling equivalent to `avg_pool2d/max_pool2d(..., ceil_mode=True)` plus min-pool by negated max.
- Attention lowering must support explicit additive relative and token-type bias before softmax; plain BMM-softmax-BMM is insufficient.
- Rectangular attention for pooled blocks must be admitted: `Q` length can be pooled while `K/V` length remains unpooled.
- `arange`, `sin`, `cos`, `gather`, reshape/slice relative-shift machinery must be available or replaced by bucketed precomputed tables.
- `repeat_interleave` exists in DinoML's checklist, but Funnel's full-model decoder needs a guarded upsample-by-repeat with CLS/pad/crop semantics.
- Full token-level heads require the full `FunnelModel`; base checkpoints intentionally expose a reduced sequence and should not be routed to token-level heads.
