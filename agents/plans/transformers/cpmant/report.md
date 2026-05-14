# CPMAnt Transformers Family Audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model family: cpmant
Primary runtime target: CpmAntForCausalLM text generation, with first-stage focus on prefill/next-token logits before optimized cache decode.
Dinoml assumptions: inference-only first, CUDA GPU target, preserve Transformers/PyTorch tensor axes, and treat the internal prompt/segment-relative-bias path as part of the model ABI rather than tokenizer-only behavior.
```

Source files inspected:

- Local: `X:/H/transformers/src/transformers/models/cpmant/configuration_cpmant.py`
- Local: `X:/H/transformers/src/transformers/models/cpmant/modeling_cpmant.py`
- Local: `X:/H/transformers/src/transformers/models/cpmant/tokenization_cpmant.py`
- Local shared utilities: `X:/H/transformers/src/transformers/cache_utils.py`, `X:/H/transformers/src/transformers/generation/utils.py`
- Auto mappings: `X:/H/transformers/src/transformers/models/auto/modeling_auto.py`, `X:/H/transformers/src/transformers/models/auto/tokenization_auto.py`
- Upstream source URL pattern at the pinned commit: `https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/cpmant/...`

Model configs and snapshots inspected:

- Official Hub repo: [openbmb/cpm-ant-10b](https://huggingface.co/openbmb/cpm-ant-10b)
- Config: `https://huggingface.co/openbmb/cpm-ant-10b/raw/main/config.json`; local snapshot: `openbmb_cpm-ant-10b_config_snapshot.json`
- Generation config: `https://huggingface.co/openbmb/cpm-ant-10b/raw/main/generation_config.json`
- Special tokens map: `https://huggingface.co/openbmb/cpm-ant-10b/raw/main/special_tokens_map.json`
- Hub/weight-index summary snapshot: `openbmb_cpm-ant-10b_hub_snapshot.md`

Missing files or assumptions:

- Hub search found only one native `model_type="cpmant"` checkpoint. No 3-5 checkpoint sweep is available without inventing non-native or unrelated CPM families.
- `openbmb/cpm-ant-10b` is public and not gated according to the Hugging Face Hub API. `tokenizer_config.json` returned 404.
- The checkpoint config contains generation defaults (`num_beams`, `repetition_penalty`, `max_new_tokens`) that are controller policy, not neural graph operators.
- `CpmAntIntermediate` and `CpmAntOutput` are copied BERT helper classes in the file but are not instantiated by `CpmAntModel` or `CpmAntForCausalLM`; first integration should ignore them.
- No DinoML runtime code was edited and no tests were run for this docs-only audit.

## 2. High-level architecture

CPMAnt is a text-only Transformer family exposed as a causal LM, but the inspected native source is not a plain GPT-style decoder. It prepends learned prompt-token IDs inside `CpmAntModel.forward`, adds learned segment embeddings, uses segment-aware relative attention bias, and builds an all-context attention mask for valid tokens. For generation parity, the internal prompt insertion and position-bias construction are model-owned GPU/runtime work.

```text
rjieba + wordpiece tokenizer -> input_ids
  -> internal learned prompt token prefix + segment ids
  -> token/prompt embedding + segment embedding
  -> segment-aware relative position bias + valid-token attention mask
  -> repeated pre-LN MHA + gated GELU FFN blocks
  -> final RMSNorm
  -> tied LM head over vocab + prompt rows
  -> generation controller / sampling
```

Stage decomposition:

- CPU/data pipeline: `CPMAntTokenizer` requires `rjieba`, loads `vocab.txt`, maps special space/newline placeholder tokens to literal `" "` and `"\n"`, and left-pads by default.
- GPU prefill: prepend `prompt_length` learned prompt token IDs from the third prompt block, construct segment/context/span/position tensors, embed token plus segment, materialize segment-relative attention bias, run 48 blocks, drop prompt rows from returned hidden states, and compute LM logits.
- GPU decode: source advertises `use_cache=True` and calls `DynamicCache.update`, but no family-specific `prepare_inputs_for_generation` handles the internally prepended prompt. DinoML should validate cache decode against Transformers before assuming generic decode slicing is correct at this pinned commit.
- Generation controller: beam count, repetition penalty, max length/new tokens, BOS/EOS/PAD handling, and sampling are outside the neural graph.

Independently stageable pieces are tokenizer ABI, prompt-prefix embedding construction, position-bias bucket math, one transformer block, full prefill logits, cache append behavior, and controller parity.

## 3. Important config dimensions

Source defaults from `CpmAntConfig`:

| Field | Default | Lowering effect |
| --- | ---: | --- |
| `model_type` | `cpmant` | Native class dispatch. |
| `vocab_size` | 30720 | Base vocabulary rows. |
| `prompt_types` | 32 | Learned prompt row groups appended to embedding table. |
| `prompt_length` | 32 | Internal prompt prefix length; source uses the third prompt group. |
| Effective embedding rows | 31744 | `vocab_size + prompt_types * prompt_length`. |
| `hidden_size` | 4096 | Hidden width `H`. |
| `num_hidden_layers` | 48 | Transformer block count. |
| `num_attention_heads` | 32 | Full MHA head count. |
| `dim_head` | 128 | Per-head width; `32 * 128 == 4096` for public config. |
| `dim_ff` | 10240 | Gated FFN intermediate width. |
| `dropout_p` | 0.0 | Dropout modules exist but are inactive for public inference. |
| `position_bias_num_buckets` | 512 | Relative-position bucket count before segment-pair buckets. |
| `position_bias_max_distance` | 2048 | Log bucket saturation distance. |
| `segment_types` | 32 | Segment embedding rows and segment-pair bias combinations. |
| Relative bias rows | 1536 | `segment_types * segment_types + position_bias_num_buckets`. |
| `eps` | `1e-6` | RMSNorm epsilon. |
| `init_std` | 1.0 | Position-bias init only. |
| `use_cache` | `true` | DynamicCache path enabled by default. |
| `tie_word_embeddings` | `true` | LM head weight is tied to input embedding. |

Representative checkpoint sweep:

| Checkpoint | Source | Layers | Hidden | Heads x dim | FFN | Vocab rows | Prompt | Position bias | Dtype | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| `openbmb/cpm-ant-10b` | `config.json` | 48 | 4096 | 32 x 128 | 10240 | 30720 base / 31744 effective | 32 x 32 | 512 buckets, max distance 2048, 32 segment types | `float32` in config | Only native cpmant checkpoint found; generation config has beam/repetition defaults. |

## 3a. Family variation traps

- The effective embedding and LM-head width is `vocab_size + prompt_types * prompt_length`, not `vocab_size`.
- `input_embedding.weight` and `lm_head.weight` are one logical tied parameter. The tied rows include the prompt-token rows even though logits for prompt rows are not normally desired.
- Source prepends prompt token IDs internally on every `forward` call; graph import cannot treat tokenizer output length as model sequence length.
- The source builds `segment = 2` for nonzero input IDs and `segment = 0` for padding and internal prompt IDs. This makes `pad_token_id == 0` a model-graph condition.
- Attention is full MHA, not MQA/GQA. There are no attention projection biases.
- Position encoding is learned relative bias with segment-pair buckets, not RoPE or ALiBi.
- The attention mask is all-to-all over valid tokens when `context` is all ones; do not silently replace it with a standard lower-triangular causal mask for prefill parity.
- `attention_mask` accepted by `CpmAntForCausalLM.forward` is a dummy text-generation-pipeline parameter and is not forwarded into `CpmAntModel`.
- `CpmAntModel.forward` casts `input_ids` to `int32`; embedding indices and generated aranges follow that dtype.
- Cache decode has a source-risk: generic generation slicing plus internal prompt prepending may produce non-obvious `past_length` behavior. Treat cache decode as validation-gated.
- Checkpoint field `mask_modules` is present but not read by native source.

## 4. Operator coverage checklist

Tensor/layout ops:

- `to(dtype=int32)`, `torch.where(input_ids != 0, 2, 0)`, reductions for sequence length, `cat`, `arange`, `repeat`, `full`, `view`, `permute`, `contiguous`, slicing, comparison masks, boolean `and/or/not`.
- Shape-sensitive prompt transform: user `[B, S]` becomes internal `[B, S + 32]` before the encoder.

Neural network primitives:

- Embedding lookup: `input_embedding` rows `31744 x 4096`; `segment_embedding` rows `32 x 4096`.
- RMSNorm over last axis with fp32 variance accumulation and learned scale.
- Bias-free Linear Q/K/V: each `4096 -> 4096`.
- Bias-free attention output Linear: `4096 -> 4096`.
- Gated GELU FFN: `w_0: 4096 -> 10240`, `w_1: 4096 -> 10240`, elementwise GELU/mul, `w_out: 10240 -> 4096`.
- Residual adds after attention and FFN.
- Bias-free tied LM projection: `4096 -> 31744`.

Attention primitives:

- Dense self-attention matmul/softmax/matmul with shape `[B, 32, Q, K]`.
- Scale by `1 / sqrt(dim_head)`.
- Add precomputed position bias `[B, 32, Q, K]`.
- Boolean mask fill to `-inf`, softmax, second mask fill to `0`, optional dropout.
- DynamicCache update for keys/values shaped `[B, 32, K, 128]`.

Position/relative-bias ops:

- Segment-pair bucket index `query_segment * segment_types + key_segment + num_buckets`.
- Bidirectional log relative-position bucket for same-segment token pairs.
- `F.embedding(bucket_ids, relative_attention_bias)` where bias table is `[1536, 32]`, then permute to `[B, 32, Q, K]`.

Preprocessing-coupled ops:

- Tokenizer depends on `rjieba` segmentation plus greedy wordpiece fallback.
- Left padding is assumed by the model-side valid-token mask, which counts nonzero IDs and builds a reversed length mask.
- BOS/EOS/PAD/UNK IDs come from vocab/special-token files; config sets BOS 6, EOS 7, PAD 0.

Generation/cache ops:

- Dynamic per-layer K/V append/reorder support is needed if cache decode is admitted.
- Last-token-only logits can use `logits_to_keep=1`; default `0` computes all hidden positions because `slice(-0, None)` is `slice(0, None)`.

## 5. Layer/block breakdown

Input construction:

```text
tokens = input_ids.to(int32)                                  # [B, S]
segment_user = where(tokens != 0, 2, 0)                       # [B, S]
length = count_nonzero(segment_user)                          # [B]
prompt_ids = arange(vocab_size + 2*prompt_length,
                    vocab_size + 3*prompt_length)             # [32]
internal_ids = concat(prompt_ids.repeat(B,1), tokens)          # [B, S+32]
segment = concat(zeros([B,32]), segment_user)                 # [B, S+32]
hidden = input_embedding(internal_ids) + segment_embedding(segment)
```

Transformer block, repeated 48 times for `openbmb/cpm-ant-10b`:

```text
a = RMSNorm(x)
q = Linear(a; 4096 -> 4096, bias=False).view(B,Q,32,128).permute(B,32,Q,128)
k = Linear(a; 4096 -> 4096, bias=False).view(B,K,32,128).permute(B,32,K,128)
v = Linear(a; 4096 -> 4096, bias=False).view(B,K,32,128).permute(B,32,K,128)
k,v = cache.update(k,v) if cache is present
s = (q @ k^T) / sqrt(128)
s = s + position_bias
p = softmax(mask_fill(s, invalid, -inf), dim=-1)
p = mask_fill(p, invalid, 0)
a_out = (p @ v).permute(B,Q,32,128).reshape(B,Q,4096)
x = x + Linear(a_out; 4096 -> 4096, bias=False)
m = RMSNorm(x)
g = GELU(Linear(m; 4096 -> 10240, bias=False))
u = Linear(m; 4096 -> 10240, bias=False)
x = x + Linear(g * u; 10240 -> 4096, bias=False)
```

Output:

```text
x = final RMSNorm(x)
if first prefill: x = x[:, prompt_length:, :]
logits = tied_lm_head(x[:, slice_indices, :])                 # [B, kept_Q, 31744]
```

All block projections are bias-free. Physical weights are distinct per layer; no ALBERT-style layer sharing is present.

## 6. Attention requirements

Required attention variant:

- Dense self-attention over internal prompt plus user tokens.
- Source behavior is all-to-all over valid tokens during prefill because `context` is filled with ones. It is not the normal lower-triangular GPT causal mask in the inspected code.
- Full MHA: `num_heads=32`, `num_key_value_heads=32` by inference from separate K/V heads, `head_dim=128`, Q/K/V width `4096`.
- Position bias is added after QK scaling and before mask fill/softmax.
- Mask is boolean `[B, Q, K]`, expanded to `[B, 1, Q, K]`.
- Cached keys are stored after linear projection and reshape, before position bias. There is no RoPE mutation of K.
- Per-layer cache tensor shape after prefill is expected to be `[B, 32, S_internal, 128]` for both K and V, where `S_internal = prompt_length + user_seq_len`.
- No sliding window, block sparse, hash/sort/bucket attention, cross-attention, or packed varlen attention is implemented.

FlashAttention/SDPA compatibility:

- The dense math can map to standard fused attention only if the runtime admits an arbitrary additive per-batch/per-head bias `[B,H,Q,K]` and boolean mask with full noncausal visibility.
- The source applies a second post-softmax invalid-mask fill to zero. This matters for rows with masked positions and should be covered by parity tests.
- Because prefill is noncausal in source, a causal FlashAttention path is unsafe unless the mask-building ABI is changed and validated.

Cache caution:

- Native source has no `prepare_inputs_for_generation` override. Generic GenerationMixin may slice `input_ids` on subsequent decode steps, while `CpmAntModel.forward` prepends prompt tokens and slices by `past_length`. DinoML should initially admit prefill/next-token logits and only enable cache decode after reproducing Transformers generation behavior for this exact commit.

## 7. Position encoding and custom math

CPMAnt uses segment-aware relative attention bias. The `key_pos` and `query_pos` inputs are checked for shape but the bucket math uses fresh `arange` tensors for relative positions.

```python
def cpmant_bucket(query_segment, key_segment, querylen, keylen,
                  num_buckets=512, max_distance=2048, segment_types=32):
    seg_bucket = query_segment[..., None] * segment_types + key_segment[:, None, :] + num_buckets
    rel = arange(keylen)[None, :] - arange(querylen)[:, None]
    half = num_buckets // 2
    side = (rel > 0).int() * half
    dist = abs(rel)
    max_exact = half // 2
    large = max_exact + (
        log(dist.float() / max_exact) / log(max_distance / max_exact)
        * (half - max_exact)
    ).int()
    large = min(large, half - 1)
    abs_bucket = side + where(dist < max_exact, dist.int(), large)
    return where(query_segment[..., None] == key_segment[:, None, :], abs_bucket, seg_bucket)
```

The relative-bias table can be preloaded as a weight. Bucket IDs depend on batch segment IDs, query/key lengths, and decode cache length. For same-segment pairs, absolute buckets can be cached per `(Q,K)` length; segment-pair replacement still depends on the segment matrix.

## 8. Preprocessing and input packing

Tokenizer ABI:

- `CpmAntTokenizer` is a Python tokenizer, not a fast tokenizer in this source directory.
- It requires `rjieba` and applies `rjieba.cut(text, False)`, then a greedy wordpiece tokenizer over `vocab.txt`.
- It joins decoded tokens directly with `""`.
- It maps vocab placeholders `"</_>"` and `"</n>"` to literal `" "` and `"\n"` in the encoder.
- Default `padding_side` is left.
- `model_input_names` are `input_ids` and `attention_mask`, but native model forward ignores the user attention mask.

Model-owned packing:

- Internal prompt IDs are not tokenizer outputs. They are generated as `vocab_size + 2*prompt_length` through `vocab_size + 3*prompt_length - 1`.
- Segment IDs are generated from token ID zero/nonzero status and prompt prefix zeros.
- `context` is filled with ones and `span` with zeros in the native source.
- The valid-token mask assumes left padding and counts nonzero user tokens; right padding would not match source intent.

No vision/audio/video/image-code preprocessing, placeholder scatter, packed `cu_seqlens`, token type IDs from the caller, or external feature extractors are present.

## 9. Graph rewrite / lowering opportunities

### Rewrite: tied embedding/LM projection alias

Source pattern:

```text
input_embedding: [31744, 4096]
lm_head: Linear(4096 -> 31744, bias=False), tied to input_embedding.weight
```

Replacement:

```text
Embedding(weight=W) and MatMul(hidden, W.T) share one logical weight object
```

Preconditions:

- `tie_word_embeddings == true`.
- Effective rows equal `vocab_size + prompt_types * prompt_length`.
- Loader preserves aliasing or explicit shared storage metadata.

Failure cases:

- Untied converted checkpoints or prompt-row-pruned deployments.

Parity test sketch:

- Verify logits from `hidden @ input_embedding.weight.T` match HF `lm_head`.

### Rewrite: separate Q/K/V linears -> packed QKV GEMM

Source pattern:

```text
q = x @ Wq.T
k = x @ Wk.T
v = x @ Wv.T
```

Replacement:

```text
qkv = x @ concat_rows(Wq, Wk, Wv).T
split qkv into [Q, K, V] row-contiguous blocks
```

Preconditions:

- Same input tensor for self-attention Q/K/V.
- All three projections are bias-free.
- Widths are equal: each `num_heads * dim_head`.
- Split order is all-Q rows, then all-K rows, then all-V rows. This is not per-head interleaved.

Failure cases:

- Cross-attention variants, projection bias, or converted packed weights with different storage order.

### Rewrite: gated GELU FFN fusion

Source pattern:

```text
GELU(x @ W0.T) * (x @ W1.T) -> out @ Wout.T
```

Replacement:

```text
dual GEMM + fused GELU/mul epilogue, then output GEMM
```

Preconditions:

- Activation is exact PyTorch `torch.nn.GELU()` default, not approximate tanh unless PyTorch default is configured that way by backend.
- Both input projections are bias-free and output width is `dim_ff`.

Failure cases:

- Approximate-GELU substitutions without tolerance validation.

### Rewrite: segment-relative-bias precompute

Source pattern:

```text
bucket_ids = where(same_segment, absolute_position_bucket, segment_pair_bucket)
bias = embedding(bucket_ids, relative_attention_bias).permute(0,3,1,2)
```

Replacement:

```text
precompute absolute bucket matrix for fixed Q,K; generate segment-pair matrix by small integer kernel; gather bias table once per layer stack input and share across layers
```

Preconditions:

- Same `position_bias` is passed to all layers in a forward call.
- Segment IDs are identical for all layers.
- Bias table is shared globally, not per-layer.

Failure cases:

- Runtime mutates segment IDs per layer, or future source moves bias inside layers.

### Rewrite: internal prompt prefix materialization

Source pattern:

```text
cat(fixed_prompt_ids.repeat(B,1), input_ids) -> embedding
```

Replacement:

```text
lookup fixed prompt embedding rows once, batch-broadcast, concat with user token embeddings
```

Preconditions:

- Prompt IDs are exactly the third prompt block.
- Batch broadcast preserves row order.
- Segment embedding for prompt segment zero is added.

Failure cases:

- Config changes prompt group selection or caller supplies custom prompt tokens.

## 10. Kernel fusion candidates

Highest priority:

- RMSNorm with fp32 variance accumulation and scale multiply; used twice per block plus final norm.
- Packed QKV projection plus reshape/transpose for `[B,S,4096] -> [B,32,S,128]`.
- Dense attention with arbitrary additive bias and boolean mask; CPMAnt cannot use a simple causal-only kernel for source parity.
- Gated GELU FFN fusion for `4096 -> 10240 -> 4096`.
- Last-token-only logits with tied embedding matrix for generation.

Medium priority:

- Segment-relative-position bias materialization/gather; table is shared but output is large `[B,32,Q,K]`.
- Prompt embedding broadcast plus concat elimination.
- Mask generation from left-padding length, context, and span.

Lower priority:

- Dropout elimination for `dropout_p=0.0`.
- `output_attentions` and `output_hidden_states` collection paths.
- Loss path for labels.

## 11. Runtime staging plan

Stage 1: parse `CpmAntConfig`, load weight index, and preserve tied embedding/LM-head aliasing.

Stage 2: implement tokenizer-adjacent ABI checks: left padding, PAD ID zero, ignored caller attention mask, internal prompt prefix generation.

Stage 3: implement source-default prefill graph without cache: embeddings, segment embeddings, mask, position bias, one block, final norm, prompt-row drop, logits.

Stage 4: one-block and full-model parity against Transformers for short CPU/GPU fp32 prompts, including left padding.

Stage 5: implement `logits_to_keep=1` optimized LM head and validate next-token logits.

Stage 6: investigate and validate native Transformers cache decode behavior at this commit. Admit cache only after reproducing generic GenerationMixin plus internal prompt behavior.

Stage 7: add packed-QKV, RMSNorm, gated-GELU, attention-bias, and prompt-prefix rewrites behind guards.

Stage 8: generation-controller parity for `generation_config.json` policies can be layered after neural logits/cache parity.

Initially stub or defer beam search, repetition penalty, loss, attention/hidden-state returns, and cache decode if validation shows source/generic generation mismatch.

## 12. Parity and validation plan

- Config/load test: verify effective embedding rows `31744`, bias table rows `1536`, and 48 layer names match the checkpoint index.
- Custom op tests: RMSNorm fp32 accumulation, CPMAnt bucket function across same/different segments, and left-padding mask construction.
- Single-block parity: random hidden states plus source-built mask/bias, fp32 first; then bf16/fp16 tolerances after kernels exist.
- Prefill parity: compare `CpmAntModel` hidden states after prompt-row drop for short prompts with and without left padding.
- LM parity: compare `CpmAntForCausalLM` logits for `logits_to_keep=0` and `logits_to_keep=1`.
- Cache investigation: run HF `generate` or manual two-step forward for one prompt, inspect `past_key_values.get_seq_length()`, internal sequence lengths, and logits equality versus full recompute before admitting DinoML cache.
- Tokenizer parity: verify `rjieba` segmentation, space/newline token mapping, left padding, and BOS/EOS/PAD IDs using the official vocab.

Suggested tolerances: fp32 `atol=1e-4, rtol=1e-4` for block/logit parity; fp16/bf16 should use looser layerwise tolerances and final token-rank checks because attention softmax plus large bias tensors can amplify differences.

## 13. Performance probes

- Tokenizer throughput with `rjieba` separated from GPU graph time.
- Position-bias materialization time and memory for `(B,S) = (1,128), (1,1024), (8,1024)`.
- Prefill-only latency/throughput by sequence length, including prompt overhead.
- Attention backend comparison: eager matmul versus fused dense attention supporting arbitrary additive bias.
- Gated FFN GEMM utilization and fusion benefit.
- LM-head throughput for full logits versus last-token-only logits.
- KV cache memory if decode is admitted: `layers * 2 * B * heads * seq_internal * head_dim * dtype_size`.
- Batch-size sweep with left padding to expose mask/bias generation overhead.

## 14. Skip/defer list

- Training loss and `CrossEntropyLoss`.
- Dropout behavior for nonzero `dropout_p`.
- Beam search, repetition penalty, and sampling processors until neural logits parity is stable.
- `output_attentions` and `output_hidden_states`.
- Cache decode optimization until generic GenerationMixin behavior is validated for this source.
- Quantization and packed-weight loading; checkpoint uses ordinary PyTorch shards and no source-coupled quantized format.
- Multi-GPU/tensor parallel lowering.
- The unused copied BERT helper classes.

## 15. Final implementation checklist

- [ ] Parse `CpmAntConfig` including `dim_head`, `dim_ff`, prompt fields, segment fields, and position-bias fields.
- [ ] Load official weight names and preserve `input_embedding.weight` / `lm_head.weight` aliasing.
- [ ] Implement internal prompt token prefix and segment embedding addition.
- [ ] Implement left-padding valid-token mask from PAD ID zero and nonzero length counts.
- [ ] Implement CPMAnt segment-aware relative-position bucket and shared bias gather.
- [ ] Implement RMSNorm with fp32 variance accumulation.
- [ ] Implement bias-free MHA with additive `[B,H,Q,K]` bias and boolean mask.
- [ ] Implement DynamicCache-compatible K/V append only after validation.
- [ ] Implement gated GELU FFN with exact activation parity.
- [ ] Implement prompt-row drop before returning hidden states on first prefill.
- [ ] Implement tied LM head over `vocab_size + prompt_types * prompt_length` rows and `logits_to_keep`.
- [ ] Add tokenizer ABI checks for `rjieba`, left padding, space/newline mapping, and special IDs.
- [ ] Add one-block, full-prefill, next-token-logit, and cache-investigation parity tests.
- [ ] Benchmark tokenizer, position bias, prefill attention, FFN, logits, and possible decode cache memory.
