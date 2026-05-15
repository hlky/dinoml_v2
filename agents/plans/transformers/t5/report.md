# Transformers T5 Operator and Integration Report

## 1. Source basis

```text
Transformers commit/version:
  Local checkout transformers at b75feb2af64c3e29cbbc1bd859958c5432cc7ed4.
  Remote upstream: https://github.com/huggingface/transformers.git

Model id:
  Primary worked example: google-t5/t5-small.
  Additional sizing references: google-t5/t5-{base,large,3b,11b},
  google/t5-v1_1-{small,base,large,xl,xxl}, google/flan-t5-{small,base,large,xl,xxl}.

Config source:
  https://huggingface.co/google-t5/t5-small/raw/main/config.json
  https://huggingface.co/google-t5/t5-small/raw/main/tokenizer_config.json
  Additional configs fetched from the Hugging Face model repos listed above.
  HF plugin metadata confirmed google-t5/t5-small as transformers architecture t5,
  AutoModelForSeq2SeqLM, about 60.5M parameters.

Source files inspected:
  transformers/src/transformers/models/t5/modeling_t5.py
  transformers/src/transformers/models/t5/configuration_t5.py
  transformers/src/transformers/models/t5/tokenization_t5.py

Any missing files or assumptions:
  No remote-code files are required for standard T5. The report assumes
  inference-only CUDA GPU execution and prioritizes text-to-text generation.
  Original T5 configs omit some fields that current T5Config supplies by
  default, especially feed_forward_proj="relu", use_cache=True,
  relative_attention_max_distance=128, and num_decoder_layers=num_layers.
```

## 2. High-level architecture

T5 is a text-only encoder-decoder transformer. The encoder is bidirectional self-attention. The decoder has causal self-attention, encoder-decoder cross-attention, and an LM projection to vocabulary. Token embeddings are shared between encoder, decoder, and LM head by tied weights.

```text
SentencePiece/Unigram tokenization
  -> shared token embedding
  -> encoder stack
  -> decoder prefill/decode with self-attention cache and cross-attention cache
  -> optional decoder output scale by d_model**-0.5
  -> tied lm_head projection
  -> logits/sampling
```

## 3. Important config dimensions

The first table is the worked example. Shape symbols used below: `B=batch`, `S=encoder/source length`, `T=decoder target length`, `H=d_model`, `A=num_heads`, `D=d_kv`, `I=d_ff`, `V=vocab_size`, `K=A*D`.

| Field | google-t5/t5-small value | Notes |
|---|---:|---|
| architecture | T5ForConditionalGeneration | Encoder-decoder LM |
| vocab_size / V | 32128 | Shared embedding and LM projection |
| d_model / H | 512 | Hidden width |
| num_layers | 6 | Encoder layers; decoder defaults to same |
| num_decoder_layers | 6 inferred | Config omits it; T5Config defaults to num_layers |
| num_heads / A | 8 | MHA, not GQA/MQA |
| d_kv / D | 64 | Per-head Q/K/V dim |
| inner attention K=A*D | 512 | Equal to H for small/base/large; not always equal in 3B/11B |
| d_ff / I | 2048 | ReLU FFN intermediate |
| feed_forward_proj | relu inferred | Original T5; T5 v1.1/FLAN use gated-gelu |
| relative_attention_num_buckets | 32 | Learned per-head relative bias buckets |
| relative_attention_max_distance | 128 inferred | T5Config default where absent |
| n_positions / tokenizer max | 512 | Relative bias can generalize beyond this, but configs/tokenizer use 512 |
| layer_norm_epsilon | 1e-6 | T5 RMS-style norm, no bias |
| dropout_rate | 0.1 | Disabled for inference |
| use_cache | True inferred | Decoder-only cache path |
| dtype | F32 checkpoint metadata for t5-small | Runtime should support fp32/fp16/bf16 weights/activations |

Representative real checkpoint ranges:

| Family/checkpoint | H | A | D | K=A*D | I | encoder/decoder layers | FFN |
|---|---:|---:|---:|---:|---:|---:|---|
| google-t5/t5-small | 512 | 8 | 64 | 512 | 2048 | 6/6 | relu |
| google-t5/t5-base | 768 | 12 | 64 | 768 | 3072 | 12/12 | relu |
| google-t5/t5-large | 1024 | 16 | 64 | 1024 | 4096 | 24/24 | relu |
| google-t5/t5-3b | 1024 | 32 | 128 | 4096 | 16384 | 24/24 | relu |
| google-t5/t5-11b | 1024 | 128 | 128 | 16384 | 65536 | 24/24 | relu |
| google/flan-t5-small | 512 | 6 | 64 | 384 | 1024 | 8/8 | gated-gelu |
| google/flan-t5-base | 768 | 12 | 64 | 768 | 2048 | 12/12 | gated-gelu |
| google/flan-t5-large | 1024 | 16 | 64 | 1024 | 2816 | 24/24 | gated-gelu |
| google/flan-t5-xl | 2048 | 32 | 64 | 2048 | 5120 | 24/24 | gated-gelu |
| google/flan-t5-xxl | 4096 | 64 | 64 | 4096 | 10240 | 24/24 | gated-gelu |

Important inference: Dinoml should not assume `A*D == H`. Original 3B/11B use much wider attention projections than hidden size.

## 4. Operator coverage checklist

### Tensor/layout ops

- Token embedding gather: `input_ids[B,S] -> hidden[B,S,H]`, shared for encoder and decoder.
- Reshape/view: projection output `[B,L,K] -> [B,L,A,D]`.
- Transpose/permute: `[B,L,A,D] <-> [B,A,L,D]`.
- Contiguous/reshape after attention output: `[B,A,L,D] -> [B,L,K]`.
- Slice masks to key length for cache-backed attention.
- Optional select/gather last-token logits for decode optimization.
- Tied weight aliasing for shared embedding/lm_head.

### Neural network primitives

- Bias-free Linear:
  - T5-small attention: Q/K/V `Linear(512 -> 512)`, O `Linear(512 -> 512)`.
  - T5-small FFN ReLU: `Linear(512 -> 2048)`, ReLU, `Linear(2048 -> 512)`.
  - FLAN/T5 v1.1 gated FFN: two parallel `Linear(H -> I)`, GELU-new on one path, multiply, `Linear(I -> H)`.
  - LM head: `Linear(H -> 32128)`, bias-free, tied to embedding weight.
- T5LayerNorm/RMSNorm-style norm: mean of squares over hidden dim, fp32 accumulation, scale only, no bias, no mean subtraction.
- Elementwise add residuals.
- Elementwise multiply for gated FFN.
- Activation: ReLU and GELU-new.
- Optional decoder output scale: multiply by `H**-0.5` before LM head for original tied embeddings.
- Dropout is present in source but should be compiled away for inference.
- Clamp-inf paths are training/fp16 safety behavior and can be deferred for inference parity unless matching HF eager exactly in fp16 overflow cases.

### Attention primitives

- Encoder noncausal self-attention with relative position bias and padding mask.
- Decoder causal self-attention with relative position bias, causal mask, padding mask, and KV cache.
- Decoder cross-attention over encoder hidden states, noncausal source mask, zero relative bias by default, and reusable cross-attention KV cache.
- Softmax in fp32 then cast back to score dtype.
- Batched GEMM score: `[B,A,Q,D] x [B,A,D,Kv] -> [B,A,Q,Kv]`.
- Batched GEMM value: `[B,A,Q,Kv] x [B,A,Kv,D] -> [B,A,Q,D]`.

### Position/relative-bias ops

- Relative position bucket math with integer arange, compare, abs/min, log, clamp/min, where.
- Learned embedding lookup from bucket IDs to per-head bias `[1,A,Q,Kv]`.
- Bias sharing across all layers in a stack after first layer computes it.
- Decoder bias uses `past_seen_tokens` offset during cached decode.

### Generation/cache ops

- EncoderDecoderCache with separate self-attention and cross-attention DynamicCache objects.
- Per-layer self-attention K/V append for decoder decode.
- Cross-attention K/V compute once per layer and reuse after first generated token.
- Cache length query per layer for relative bias offset.
- Decoder start token uses `pad_token_id=0`; label path uses `_shift_right`.

### Preprocessing-coupled ops

- T5Tokenizer uses Unigram/SentencePiece-style model with pad/eos/unk IDs 0/1/2 and 100 sentinel `<extra_id_N>` tokens.
- Tokenizer post-processing appends `</s>` to single and pair sequences.
- Task prompts such as `summarize: ` or `translate English to German: ` are data-pipeline concerns, not runtime graph ops.

### Distributed/tensor-parallel ops

- No explicit tensor-parallel source path in inspected T5 code. Large original T5-11B shapes imply useful future support for sharded Q/K/V/O, FFN, and LM head GEMMs, but this is not required for single-GPU first parity.

## 5. Layer/block breakdown

Encoder stack:

```text
x = shared_embedding(input_ids)                         # [B,S,H]
x = dropout(x)                                          # inference no-op
position_bias = None
for layer in 0..Nenc-1:
  y = T5LayerNorm(x)                                    # scale-only RMS norm
  q = Linear(H -> A*D, bias=False)(y) -> [B,A,S,D]
  k = Linear(H -> A*D, bias=False)(y) -> [B,A,S,D]
  v = Linear(H -> A*D, bias=False)(y) -> [B,A,S,D]
  if layer == 0: position_bias = relative_bias(S,S,bidirectional=True)
  scores = MatMul(q, k^T) + position_bias + padding_mask
  p = Softmax(scores.float, dim=-1).astype(scores.dtype)
  attn = MatMul(p, v) -> [B,S,A*D]
  x = x + Linear(A*D -> H, bias=False)(attn)
  z = T5LayerNorm(x)
  if relu FFN:
    z = Linear(H -> I, bias=False)(z)
    z = ReLU(z)
    z = Linear(I -> H, bias=False)(z)
  if gated-gelu FFN:
    a = GELU_new(Linear(H -> I, bias=False)(z))
    b = Linear(H -> I, bias=False)(z)
    z = Linear(I -> H, bias=False)(a * b)
  x = x + z
x = final T5LayerNorm(x)
```

Decoder block:

```text
x = decoder_embedding(decoder_input_ids)                # [B,T,H]
self_position_bias = None
cross_position_bias = None
for layer in 0..Ndec-1:
  y = T5LayerNorm(x)
  q = Linear(H -> A*D)(y)
  k,v = Linear(H -> A*D)(y), Linear(H -> A*D)(y)
  k,v = append_to_self_cache(layer, k, v)
  if layer == 0: self_position_bias = relative_bias(T,Kv,bidirectional=False,past_seen_tokens)
  x = x + Linear(A*D -> H)(Attention(q,k,v,self_position_bias,causal_mask))

  y = T5LayerNorm(x)
  q = Linear(H -> A*D)(y)
  k,v = Linear(H -> A*D)(encoder_hidden), Linear(H -> A*D)(encoder_hidden)
  k,v = reuse_or_set_cross_cache(layer, k, v)
  cross_position_bias = zeros([1,A,T,S]) plus encoder padding mask
  x = x + Linear(A*D -> H)(Attention(q,k,v,cross_position_bias))

  x = x + FFN(T5LayerNorm(x))
x = final T5LayerNorm(x)
if scale_decoder_outputs: x = x * H**-0.5
logits = Linear(H -> V, bias=False, tied_weight=shared_embedding)(x)
```

## 6. Attention requirements

- Encoder self-attention: noncausal MHA, `A` query heads and `A` KV heads, head dim `D`, relative bias is bidirectional. Mask shape is broadcast-compatible with `[B,A,S,S]`.
- Decoder self-attention: causal MHA, same head count and KV head count, relative bias is unidirectional. Mask shape is `[B,1,T,Kv]` or broadcast-compatible with `[B,A,T,Kv]`. Cached decode usually has `T=1`, `Kv=past+1`.
- Decoder cross-attention: noncausal MHA from decoder queries to encoder K/V. No learned relative bias in source; when `position_bias` is absent, source creates zeros `[1,A,T,S]` and adds encoder mask.
- There is no MQA/GQA in the T5 source; all Q/K/V use `A` heads.
- There is no RoPE, ALiBi, sliding-window, or local attention.
- Eager fallback path is explicit matmul-softmax-matmul and will be too slow for large checkpoints and long sequences. Dinoml should lower to fused attention kernels once relative bias and cache semantics are supported.
- FlashAttention/SDPA compatibility: required additive bias is `[1,A,Q,K]` plus masks. A fused backend must accept arbitrary additive per-head bias. Plain causal FlashAttention without additive relative bias is insufficient for parity.
- Cache requirements: store self-attention K/V per decoder layer with shape `[B,A,Kv,D]`. Store cross-attention K/V per decoder layer with shape `[B,A,S,D]` after first use; this is static across decode tokens for a fixed encoder output.

## 7. Position encoding and custom math

T5 uses learned relative attention bias, not absolute position embeddings or RoPE. Only blocks with `has_relative_attention_bias=True` own the embedding table; `T5Stack` builds that only for `i == 0` in each stack and shares the computed bias through later blocks.

Concise implementation sketch:

```python
def t5_relative_position_bucket(relative_position, bidirectional, num_buckets=32, max_distance=128):
    buckets = 0
    if bidirectional:
        num_buckets //= 2
        buckets += (relative_position > 0).long() * num_buckets
        relative_position = abs(relative_position)
    else:
        relative_position = -minimum(relative_position, 0)

    max_exact = num_buckets // 2
    is_small = relative_position < max_exact
    large = max_exact + (
        log(relative_position.float() / max_exact)
        / log(max_distance / max_exact)
        * (num_buckets - max_exact)
    ).long()
    large = minimum(large, num_buckets - 1)
    return buckets + where(is_small, relative_position, large)

def t5_compute_bias(query_length, key_length, table, bidirectional, past_seen_tokens=0):
    context = arange(query_length)[:, None] + past_seen_tokens
    memory = arange(key_length)[None, :]
    rel = memory - context
    bucket = t5_relative_position_bucket(rel, bidirectional)
    values = embedding(table, bucket)        # [Q,K,A]
    return values.permute(2, 0, 1)[None]    # [1,A,Q,K]
```

Precompute opportunities:

- Encoder full-sequence bias for a fixed `S` can be precomputed per stack/head in the active dtype.
- Decoder prefill bias for fixed `T` can be precomputed, but decode bias depends on `past_seen_tokens`.
- Bucket IDs are integer-only and can be cached per `(Q,K,past_seen_tokens,bidirectional)`.
- The learned table shape is `[relative_attention_num_buckets, A]`.

## 8. Preprocessing and input packing

CPU/data-pipeline work:

- Tokenization uses a Unigram tokenizer with metaspace behavior and `spiece.model`/`tokenizer.json`.
- `pad_token_id=0`, `eos_token_id=1`, `unk_token_id=2`.
- Tokenizer appends EOS to sequences.
- 100 sentinel tokens `<extra_id_0>` to `<extra_id_99>` are part of common T5 denoising/task usage.
- Task prefix strings are outside the runtime graph.

GPU/runtime graph work:

- Accept `input_ids[B,S]`, `attention_mask[B,S]`, `decoder_input_ids[B,T]`, and optional `decoder_attention_mask[B,T]`.
- For generation, first step runs encoder and decoder prefill; subsequent steps can pass only last decoder token with cache.
- `_shift_right(labels)` is training/helper behavior. For inference, generation should provide decoder start token `0` and then sampled tokens.
- No image/audio/video packing, placeholder tokens, token type IDs, or grid metadata are present.

## 9. Graph rewrite / lowering opportunities

### Rewrite: T5LayerNorm -> RMSNormScaleOnly

Preconditions:

- Norm is exactly `x * rsqrt(mean(float32(x)^2, axis=-1, keepdim=True) + eps) * weight`.
- No mean subtraction and no bias.
- Reduction axis is the last hidden dimension.
- Accumulation is fp32 for fp16/bf16 inputs.

Replacement:

```text
RMSNorm(x, weight, eps, axis=-1, bias=None, fp32_accum=True)
```

Shape equations:

- Input `[*, H]`, weight `[H]`, output `[*, H]`.

Failure cases:

- Do not rewrite generic LayerNorm with mean subtraction.
- Do not fuse if weight dtype/cast behavior must preserve unusual quantized module paths.

Parity test sketch:

- Random fp32/fp16/bf16 tensors across `H={512,768,1024,2048,4096}`; compare to HF T5LayerNorm.

### Rewrite: bias-free Linear -> GEMM_RCR

Preconditions:

- Source is `nn.Linear(in_features, out_features, bias=False)`.
- Input is dense row-major `[M,in_features]` after flattening leading dims.
- Weight stored as `[out_features,in_features]`.

Replacement:

```text
FlattenLeadingDims -> GEMM_RCR(A=[M,K], B=[N,K]) -> Reshape([*,N])
```

Shape equations:

- `M = product(input.shape[:-1])`, `K=in_features`, `N=out_features`.

Weight transform:

```python
B = linear.weight  # already [N,K], consumed as column-major/logical transposed RHS for RCR
```

Failure cases:

- Runtime non-contiguous inputs without an admitted layout view.
- Tied LM head should preserve alias/provenance with shared embedding.

Parity test sketch:

- Compare Q/K/V/O, FFN, and LM head projections independently for all representative `H,I,K,V`.

### Rewrite: separate Q/K/V projections -> QKV projection group

Preconditions:

- Same input tensor for q/k/v in self-attention.
- All three projections are bias-free and have identical input dim `H` and output dim `A*D`.
- Weight order is preserved as q, k, v.
- Cross-attention does not qualify for combined qkv because q input is decoder hidden and k/v input is encoder hidden.

Replacement:

```text
GroupedGEMM or ConcatenatedLinear(H -> 3*A*D) -> Split(q,k,v)
```

Weight transform:

```python
w_qkv = concat([w_q, w_k, w_v], axis=0)
```

Failure cases:

- Original T5 3B/11B where `A*D` is larger than `H` is still valid but must not use `H` as projection output.
- Do not apply if weights are separately quantized with incompatible storage policies.

Parity test sketch:

- Compare split outputs against three independent HF linear calls.

### Rewrite: relative bias compute -> cached bias tensor

Preconditions:

- `query_length`, `key_length`, `past_seen_tokens`, bidirectionality, bucket count, max distance, dtype, and device are known for the run/decode step.
- Relative bias embedding table is unchanged.

Replacement:

```text
RelativeBucketIds -> BiasEmbedding -> BiasTensor cache
```

Shape equations:

- Bucket IDs `[Q,K]`, table `[Bkt,A]`, bias `[1,A,Q,K]`.

Failure cases:

- Dynamic decode with changing `past_seen_tokens` needs per-step cache or small on-device recompute.

Parity test sketch:

- Exhaustive small `Q,K` plus long positions past max distance; compare bucket IDs and bias values.

### Rewrite: cross-attention K/V reuse

Preconditions:

- Encoder hidden states are fixed for the request.
- Cross-attention layer index has already populated K/V cache.
- Attention mask for encoder source is unchanged.

Replacement:

```text
CrossKVCached(layer) instead of Linear(k/v)(encoder_hidden_states) each decode token
```

Failure cases:

- New source sequence, changed encoder output, or changed source mask invalidates cache.

Parity test sketch:

- Decode two tokens with and without cache; compare logits and cross-attention K/V tensors.

## 10. Kernel fusion candidates

Highest priority:

- RMSNormScaleOnly: used before every attention and FFN plus final stack norm; cheap but memory-bandwidth sensitive.
- Bias-free GEMM family coverage: all projections are bias-free Linear; large T5-11B has extreme `1024 -> 16384` attention projections and `1024 -> 65536` FFN.
- Attention with additive relative bias and cache: required for parity and throughput; plain causal attention without bias is not enough.
- Gated GELU FFN fusion for FLAN/T5 v1.1: two input GEMMs, GELU-new, multiply, output GEMM; common in current usage.
- Last-token-only LM logits: full `[B,T,V]` logits during decode waste bandwidth; generation usually needs `[B,1,V]`.

Medium priority:

- QKV grouped projection for self-attention prefill.
- Cross-attention K/V precompute and cache packing.
- Relative bucket/bias precompute kernels or host-side cached tensors.
- LM head tied-weight GEMM with optional shard/top-k path for large vocab.
- ReLU FFN activation fusion for original T5.

Lower priority:

- Dropout removal/canonicalization for inference graphs.
- Clamp-inf exact eager fallback for fp16 overflow parity.
- Beam-search-specific cache reorder and gather kernels.
- Classification and QA heads.

## 11. Runtime staging plan

Stage 1: Parse T5Config and load shared embedding, encoder, decoder, relative bias, and LM head aliases. Stub generation helpers and training heads.

Stage 2: Implement standalone custom op parity for T5LayerNorm and relative position bucket/bias.

Stage 3: Run one encoder block parity with dense fp32 tensors and padding mask.

Stage 4: Run full encoder parity for `google-t5/t5-small`, then one gated FFN checkpoint such as `google/flan-t5-small`.

Stage 5: Implement decoder prefill without cache, including causal mask, self relative bias, cross-attention, output scale, and logits.

Stage 6: Implement decode with EncoderDecoderCache semantics: self K/V append and cross K/V reuse.

Stage 7: Add optimized attention backend with additive relative bias and cached decode.

Stage 8: Add graph rewrites/fusions for RMSNorm, QKV grouping, gated FFN, and last-token logits.

Stage 9: Scale validation to larger shapes, especially original `t5-3b/t5-11b` where `A*D != H` and FLAN XL/XXL where `H` is large.

## 12. Parity and validation plan

- Random tensor tests for T5LayerNorm against fp32/fp16/bf16 HF output. Suggested tolerances: fp32 `rtol=1e-5, atol=1e-6`; fp16/bf16 `rtol=2e-2, atol=2e-2` initially.
- Relative bucket tests for bidirectional and unidirectional modes with positions below/above `max_exact` and beyond `max_distance`.
- Relative bias tests comparing `[1,A,Q,K]` values for encoder and decoder with `past_seen_tokens`.
- Single attention parity for encoder self-attn, decoder self-attn, and decoder cross-attn.
- Single block parity for ReLU T5 and gated-gelu FLAN/T5 v1.1.
- Full encoder parity on `google-t5/t5-small`.
- Prefill logits parity for prompt + decoder prefix.
- Decode token parity for 2-4 generated steps with cache on/off.
- End-to-end text output parity for greedy generation, allowing exact token match in fp32 and small logit drift in reduced precision.
- Cache parity: inspect self K/V growth shape `[B,A,past,D]` and cross K/V reuse.

## 13. Performance probes

- Tokenization/preprocessing throughput separately from runtime.
- Encoder-only throughput over `B` and `S`.
- Decoder prefill throughput over `B`, `S`, and `T`.
- Decode-only tokens/sec with cache over `B` and source length `S`.
- Attention backend comparison: eager matmul-softmax-matmul vs fused additive-bias attention.
- Relative bias compute/cache overhead for prefill and decode.
- FFN GEMM sweep for ReLU vs gated-gelu variants.
- LM head logits cost and last-token-only optimization impact.
- KV cache memory usage: self cache `2 * Ndec * B * A * T * D * dtype_bytes`; cross cache `2 * Ndec * B * A * S * D * dtype_bytes`.
- Large-shape probes for `t5-11b` attention projection widths and FLAN-XXL hidden width.

## 14. Skip/defer list

- Training and loss computation.
- Dropout and gradient checkpointing.
- Sequence classification, token classification, and QA heads.
- Beam search and sampling policies beyond greedy single-step logits.
- Multi-GPU tensor parallel and pipeline parallel.
- Quantization-specific int8 branches.
- Remote-code support.
- Exact fp16 clamp-inf fallback unless a parity target exposes it.
- Speculative decoding and continuous batching.

## 15. Final implementation checklist

- [ ] Parse T5Config, including defaulted fields missing from older configs.
- [ ] Load and alias shared embedding, encoder embedding, decoder embedding, and LM head weights.
- [ ] Implement T5 RMSNormScaleOnly with fp32 accumulation.
- [ ] Implement relative position bucket IDs.
- [ ] Implement relative bias embedding and cache/precompute path.
- [ ] Implement bias-free Linear -> GEMM_RCR lowering for projection and FFN weights.
- [ ] Implement encoder bidirectional self-attention with additive relative bias.
- [ ] Implement decoder causal self-attention with additive relative bias and KV cache.
- [ ] Implement decoder cross-attention with cross K/V reuse.
- [ ] Implement ReLU FFN path for original T5.
- [ ] Implement gated GELU FFN path for T5 v1.1/FLAN.
- [ ] Implement decoder output scaling guard.
- [ ] Implement tied LM head logits, with last-token-only decode option.
- [ ] Add one-layer parity tests for encoder, decoder self-attn, cross-attn, and FFN.
- [ ] Add full encoder parity for `google-t5/t5-small`.
- [ ] Add prefill logits and cached decode parity.
- [ ] Benchmark encoder, prefill, decode, FFN, LM head, and KV cache memory.
