# UMT5 Transformers audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: google/umt5-small, google/umt5-base, google/umt5-xl, google/umt5-xxl
Config source: Hugging Face config.json snapshots under
  agents/plans/transformers/umt5/_sources/
Source files inspected:
  transformers/src/transformers/models/umt5/configuration_umt5.py
  transformers/src/transformers/models/umt5/modeling_umt5.py
  transformers/src/transformers/models/t5/tokenization_t5.py
  transformers/src/transformers/models/auto/tokenization_auto.py
Any missing files or assumptions:
  No remote-code files are required for official UMT5 checkpoints. Official
  configs were accessible; no gated/401 gaps were encountered. UMT5 has no
  family-local tokenizer file: AutoTokenizer maps model_type "umt5" to
  T5Tokenizer, and the checkpoint tokenizer_config selects T5Tokenizer.
  Primary runtime target: UMT5ForConditionalGeneration text-to-text inference
  on CUDA, with encoder-only and task heads treated as optional/deferred.
```

Primary source URLs at the pinned commit:

- `configuration_umt5.py`: https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/umt5/configuration_umt5.py
- `modeling_umt5.py`: https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/umt5/modeling_umt5.py
- `tokenization_t5.py`: https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/t5/tokenization_t5.py
- `tokenization_auto.py`: https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/auto/tokenization_auto.py

Official config links:

- https://huggingface.co/google/umt5-small/raw/main/config.json
- https://huggingface.co/google/umt5-base/raw/main/config.json
- https://huggingface.co/google/umt5-xl/raw/main/config.json
- https://huggingface.co/google/umt5-xxl/raw/main/config.json
- https://huggingface.co/google/umt5-small/raw/main/tokenizer_config.json
- https://huggingface.co/google/umt5-small/raw/main/special_tokens_map.json

## 2. High-level architecture

UMT5 is a text-only encoder-decoder transformer. The encoder uses bidirectional
self-attention. The decoder uses causal self-attention, dense encoder-decoder
cross-attention, and a vocabulary LM projection. The source is mostly copied
from T5-style modules, but UMT5 is not a pure alias: its config defaults,
checkpoint vocab/tokenizer coupling, forced tied embeddings, and per-layer
relative bias behavior need separate admission.

```text
T5Tokenizer/SentencePiece-style tokenization
-> shared token embedding
-> UMT5 encoder stack
-> decoder prefill/decode with self-attention cache and cross-attention cache
-> d_model**-0.5 decoder-output scale
-> tied lm_head projection
-> logits/sampling
```

Independently stageable pieces:

- CPU/data pipeline: T5Tokenizer, sentinel tokens, padding/eos handling, decoder start token construction.
- Encoder runtime: embedding, bidirectional attention with relative bias, RMS-style norm, gated FFN.
- Decoder runtime: causal self-attention cache, encoder-decoder cross-attention cache, LM head.
- Optional encoder-only target: `UMT5EncoderModel`.
- Deferred heads: sequence classification, token classification, and QA heads.

## 3. Important config dimensions

Shape symbols: `B=batch`, `S=source length`, `T=decoder length`, `H=d_model`,
`A=num_heads`, `D=d_kv`, `K=A*D`, `I=d_ff`, `V=vocab_size`.

| Field | Source default | Runtime significance |
|---|---:|---|
| `model_type` | `umt5` | AutoConfig routes to UMT5Config. |
| `vocab_size` / `V` | 250112 | Official Google configs override to 256384. |
| `d_model` / `H` | 512 | Hidden width. |
| `d_kv` / `D` | 64 | Per-head Q/K/V width. |
| `num_heads` / `A` | 6 | MHA only; no GQA/MQA. |
| `inner_dim` / `K` | `A * D = 384` | Projection width can differ from `H`; small/default uses 384 != 512. |
| `d_ff` / `I` | 1024 | FFN intermediate width. |
| `num_layers` | 8 | Encoder layer count. |
| `num_decoder_layers` | defaults to `num_layers` | Decoder can differ by config, though official sweep keeps symmetry. |
| `relative_attention_num_buckets` | 32 | Learned relative bias rows. |
| `relative_attention_max_distance` | 128 | Log bucket saturation distance. |
| `feed_forward_proj` | `gated-gelu` | Config converts this to `dense_act_fn="gelu_new"` and `is_gated_act=True`. |
| `tie_word_embeddings` | forced `True` | `UMT5Config` discards raw `tie_word_embeddings` and forces true. |
| `use_cache` | true | Decoder cache support through `EncoderDecoderCache`. |
| tokenizer class | `T5Tokenizer` in configs | UMT5 has no family-local tokenizer; AutoTokenizer maps to T5Tokenizer. |

Representative checkpoint sweep, from official `config.json` snapshots:

| Model/config | H | A | D | K=A*D | I | Enc/dec layers | Vocab | FFN | Config `tie_word_embeddings` | Effective source tie |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| `google/umt5-small` | 512 | 6 | 64 | 384 | 1024 | 8/8 | 256384 | gated-gelu | false | true |
| `google/umt5-base` | 768 | 12 | 64 | 768 | 2048 | 12/12 | 256384 | gated-gelu | false | true |
| `google/umt5-xl` | 2048 | 32 | 64 | 2048 | 5120 | 24/24 | 256384 | gated-gelu | false | true |
| `google/umt5-xxl` | 4096 | 64 | 64 | 4096 | 10240 | 24/24 | 256384 | gated-gelu | false | true |

Checkpoint-only or historical fields observed: `dense_act_fn`, `is_gated_act`,
`output_past`, `scalable_attention`, `max_new_tokens`, and
`transformers_version`. The current UMT5 modeling source does not read
`scalable_attention` or `output_past`; do not admit a separate scalable-attention
runtime based on those fields alone.

## 3a. Family variation traps

- UMT5 source forces `tie_word_embeddings=True` even though official configs say `false`. DinoML should preserve the logical alias between `shared.weight`, encoder/decoder embeddings, and `lm_head.weight`, plus the decoder output scale before logits.
- Do not assume `H == A * D`. `google/umt5-small` projects `512 -> 384` for Q/K/V and `384 -> 512` for O.
- UMT5 self-attention has learned relative-position bias in every self-attention layer. This differs from current T5/MT5 source, where only block 0 owns the bias and later layers reuse a passed `position_bias`.
- Cross-attention has no learned relative bias; it creates a zero bias tensor and only adds the encoder attention mask.
- All attention and FFN projection linears are bias-free. Classification and QA heads are the main biased linears.
- Attention scores are `q @ k.T` with no `1/sqrt(D)` scale in forward. The initializer is chosen to avoid that scale.
- `feed_forward_proj="gated-gelu"` maps to `gelu_new` and two input projections (`wi_0`, `wi_1`) multiplied before `wo`.
- Text tensors are `[B, sequence, hidden]`; no NHWC/NCHW layout translation applies. Reshape/transpose axes in attention are layout-sensitive and should be guarded from generic layout rewrites.
- Tokenizer coupling matters: official UMT5 uses T5Tokenizer with 300 `<extra_id_*>` sentinel tokens and a much larger multilingual vocab than classic T5.

## 4. Operator coverage checklist

Tensor/layout ops:

- Embedding gather: `input_ids[B,S] -> [B,S,H]` and decoder `input_ids[B,T] -> [B,T,H]`.
- View/reshape: `[B,L,K] -> [B,L,A,D]`.
- Transpose/permute: `[B,L,A,D] <-> [B,A,L,D]`.
- Contiguous plus view after attention: `[B,A,L,D] -> [B,L,K]`.
- Mask expansion: encoder mask `[B,S] -> [B,1,1,S]`; decoder causal mask from `create_causal_mask`; cross mask from `invert_attention_mask`.
- Relative-position bucket construction: `arange`, subtract, abs/min/max, comparison, log, cast to long, embedding gather.
- Cache update/read for self and cross attention.
- Tied parameter aliases for shared embedding and LM head.

Neural network primitives:

- Bias-free attention linears: Q/K/V `Linear(H -> K)`, O `Linear(K -> H)`.
- Bias-free gated FFN: `Linear(H -> I)` twice, `gelu_new`, multiply, `Linear(I -> H)`.
- UMT5LayerNorm/RMSNorm: fp32 `mean(x*x)` over last dim, `rsqrt`, learned scale only, no bias and no mean subtraction.
- Residual add after self-attention, cross-attention, and FFN.
- LM head: `Linear(H -> V)`, bias-free, tied to embedding.
- Decoder output multiplier: `sequence_output * H**-0.5` before LM head.
- Inference dropout elision.
- fp16 inf-clamp after attention/FFN appears in block forward. It is mainly a training/overflow safety path; exact eager fp16 parity may need it if overflow cases are tested.

Attention primitives:

- Encoder bidirectional MHA with per-layer relative bias and padding mask.
- Decoder causal self-attention with per-layer relative bias, causal mask, optional padding mask, and KV cache.
- Decoder cross-attention with rectangular queries `T` over encoder keys `S`, no relative bias, encoder padding mask, and cross-attention cache reuse.
- Softmax upcasts scores to fp32 and casts back to score dtype.
- No SDPA/FlashAttention dispatch in source; implementation is explicit matmul-softmax-matmul.

Preprocessing-coupled ops:

- T5Tokenizer Unigram/SentencePiece-style tokenization with pad id 0, eos id 1, unk id 2.
- Single/pair template appends `</s>`.
- 300 additional sentinel tokens in official UMT5 tokenizer configs.
- `_shift_right(labels)` or `_shift_right(input_ids)` for decoder starts: first token is `decoder_start_token_id=0`, and shifted `-100` becomes pad id.

## 5. Layer/block breakdown

Encoder block, repeated `num_layers` times:

```text
x0: [B,S,H]
n = UMT5LayerNorm(x0)
q,k,v = Linear(H -> K, bias=False)(n)
q,k,v -> [B,A,S,D]
scores = q @ k.transpose(-1,-2)                         # [B,A,S,S], no scale
bias = relative_attention_bias(bucket(memory-query))     # [1,A,S,S]
scores = scores + bias + encoder_padding_mask
p = softmax(scores.float(), dim=-1).to(scores.dtype)
a = p @ v                                                # [B,A,S,D]
a = Linear(K -> H, bias=False)(merge_heads(a))
x1 = x0 + a
f = UMT5LayerNorm(x1)
ff = Linear(I -> H, bias=False)(gelu_new(wi_0(f)) * wi_1(f))
out = x1 + ff
```

Decoder block, repeated `num_decoder_layers` times:

```text
y0: [B,T,H]
n = UMT5LayerNorm(y0)
self q,k,v = Linear(H -> K, bias=False)(n)
self k/v update or read cache as [B,A,total_T,D]
self scores = q @ k.T + causal_mask + per-layer relative_bias
y1 = y0 + Linear(K -> H)(softmax(scores.float()).to(dtype) @ v)

c = UMT5LayerNorm(y1)
cross q = Linear(H -> K)(c)
cross k/v = Linear(H -> K)(encoder_hidden[B,S,H]) or read cross cache
cross scores = q @ k.T + encoder_padding_mask            # zero position bias
y2 = y1 + Linear(K -> H)(softmax(cross_scores.float()).to(dtype) @ v)

f = UMT5LayerNorm(y2)
ff = Linear(I -> H)(gelu_new(wi_0(f)) * wi_1(f))
out = y2 + ff
```

LM head:

```text
sequence_output = decoder_final_norm_output * (H ** -0.5)
logits = sequence_output @ shared.weight.T
```

## 6. Attention requirements

UMT5 uses dense MHA, not GQA/MQA. Q, K, and V each have width `K=A*D`; value
width equals key width. The source does not enforce `K == H`.

Encoder self-attention:

- Noncausal, bidirectional, self-attention.
- Query/key/value shape after projection: `[B,A,S,D]`.
- Scores shape: `[B,A,S,S]`.
- Mask: padding mask is converted to additive finfo-min values and added to position bias.
- Relative bias: per layer, learned table `[num_buckets, A]`.
- No KV cache.

Decoder self-attention:

- Causal self-attention with optional decoder padding mask.
- Query shape for decode can be `[B,A,1,D]`; key/value cache grows to `[B,A,past+1,D]`.
- Cached keys are stored after linear projection and head transpose; there is no RoPE. Relative bias is recomputed with `past_seen_tokens` offset and not stored in cache.
- Cache object: `EncoderDecoderCache(self_attention_cache=DynamicCache, cross_attention_cache=DynamicCache)`.

Decoder cross-attention:

- Rectangular dense attention: decoder queries `[B,A,T,D]` attend to encoder keys `[B,A,S,D]`.
- Cross K/V are cached separately. `EncoderDecoderCache.is_updated[layer_idx]` marks when cross K/V for a layer have been populated and can be reused.
- No learned cross relative bias. A zero `[1,A,T,S]` tensor is created then encoder mask is added.

FlashAttention compatibility:

- A fused backend can cover the matmul-softmax-matmul region if it supports additive per-head relative bias and mask, scale=1.0, fp32 softmax accumulation, and separate self/cross cache semantics.
- For first parity, compose GEMM/BMM + bias + softmax + BMM. Add FlashAttention only after relative-bias and cache tests pass.

## 7. Position encoding and custom math

UMT5 uses T5-style learned relative position buckets, not absolute embeddings,
RoPE, or ALiBi. The bucket function differs for encoder and decoder because it
uses `self.is_decoder`.

```python
def umt5_relative_bucket(relative_position, is_decoder, num_buckets=32, max_distance=128):
    buckets = 0
    if not is_decoder:
        num_buckets //= 2
        buckets += (relative_position > 0).long() * num_buckets
        relative_position = abs(relative_position)
    else:
        relative_position = -min(relative_position, 0)
    max_exact = num_buckets // 2
    is_small = relative_position < max_exact
    large = max_exact + (
        log(relative_position.float() / max_exact) / log(max_distance / max_exact)
        * (num_buckets - max_exact)
    ).long()
    large = min(large, num_buckets - 1)
    return buckets + where(is_small, relative_position, large)
```

`compute_bias(query_length, key_length, past_seen_tokens)` creates query
positions as `arange(query_length) + past_seen_tokens` and key positions as
`arange(key_length)`, embeds buckets, then permutes to `[1,A,Q,K]`. For static
sequence buckets this can be precomputed per layer/head/table and sliced. For
decode, the query offset depends on the current cache length.

## 8. Preprocessing and input packing

UMT5 is text-only. There is no image/audio processor, no packed varlen metadata,
and no multimodal embedding stitch.

Tokenizer/runtime boundary:

- `AutoTokenizer` maps `model_type="umt5"` to T5Tokenizer.
- Official configs set `tokenizer_class="T5Tokenizer"`.
- T5Tokenizer uses `spiece.model`/`tokenizer.json` Unigram data when loaded from the repo.
- Special IDs: `<pad>=0`, `</s>=1`, `<unk>=2`.
- Official tokenizer config has 300 `<extra_id_*>` additional special tokens.
- T5Tokenizer post-processing appends `</s>` to single sequences and to each side of pairs.
- Padding side is tokenizer/data-pipeline behavior; the model supports left or right padding because relative positions are used and masks are additive.

GPU graph inputs for generation:

- Encoder `input_ids[B,S]` and `attention_mask[B,S]`.
- Decoder `decoder_input_ids[B,T]` or generated one-token decode input.
- Optional precomputed `encoder_outputs`; this is an independently cacheable encoder boundary.
- Decoder start token is pad id 0. For label-driven calls, `_shift_right` produces decoder inputs and replaces `-100` labels with pad id.

## 9. Graph rewrite / lowering opportunities

### Rewrite: bias-free Linear to GEMM

Source pattern:

```text
Linear(in -> out, bias=False)(x[B,L,in])
```

Replacement:

```text
Flatten leading dims -> GEMM_RCR(x2d, weight[out,in]) -> Reshape[B,L,out]
```

Preconditions:

- Weight is dense row-major `[out_features, in_features]`.
- No bias.
- Preserve `K=A*D` independently from `H`.
- Runtime shape may flatten `B*L`; `in_features` and `out_features` are static.

Failure cases: non-dense/quantized weights need separate loading/provider policy.

Parity sketch: compare Q/K/V/O, FFN, and LM projections for small/base shapes,
including `H=512,K=384` small case.

### Rewrite: gated FFN fusion

Source pattern:

```text
gelu_new(wi_0(norm_x)) * wi_1(norm_x) -> wo
```

Replacement:

```text
dual GEMM or two GEMMs -> fused gelu_new*multiply -> GEMM
```

Preconditions:

- `feed_forward_proj` starts with `gated-`.
- Both `wi_0` and `wi_1` are bias-free with identical `[I,H]` storage.
- Activation exactly `gelu_new` for official configs.

Failure cases: source validation allows other `gated-{ACT_FN}` strings if
`ACT2FN` contains them; only admit audited activations.

### Rewrite: relative bias precompute

Source pattern:

```text
bucket(arange(K) - (arange(Q) + past)) -> Embedding([buckets,A]) -> [1,A,Q,K]
```

Replacement:

```text
precompute bucket index matrix per (is_decoder,Q,K,past bucket) or compute once per decode step,
then gather per-layer bias table
```

Preconditions:

- `relative_attention_num_buckets` and `max_distance` static.
- Per-layer bias tables are distinct; do not share the table across layers.
- Decode path must include `past_seen_tokens` in query positions.

Failure cases: dynamic `Q/K` beyond precomputed table needs fallback bucket
kernel or larger cache.

### Rewrite: last-token logits for decode

Source pattern:

```text
lm_head(decoder_hidden[B,T,H])
```

Replacement:

```text
slice decoder_hidden[:, -1:, :] -> scale -> tied lm_head
```

Preconditions:

- Generation only needs next-token logits.
- Loss/training and full-sequence logits are not requested.
- Preserve `H**-0.5` scale before projection.

## 10. Kernel fusion candidates

Highest priority:

- RMSNorm scale-only kernel with fp32 accumulation.
- Bias-free GEMM coverage for Q/K/V/O, FFN, and LM head, including `K != H`.
- Relative-bias + mask + softmax path for dense attention with scale=1.0.
- Decoder KV cache ABI for self cache and cross cache.

Medium priority:

- Gated GELU FFN fusion: dual input projection plus `gelu_new * linear`.
- Decode last-token LM projection with tied embedding weight.
- Relative-bucket precompute/gather for encoder and prefill.

Lower priority:

- Exact fp16 inf-clamp parity path.
- Classification/QA head fusions.
- FlashAttention backend once relative bias and cache behavior are proven.

## 11. Runtime staging plan

Stage 1: parse UMT5Config and tokenizer metadata. Normalize effective
`tie_word_embeddings=True`, reject unsupported `scalable_attention` assumptions,
and preserve `K=A*D`.

Stage 2: load weights with alias tracking for `shared.weight`,
`encoder.embed_tokens.weight`, `decoder.embed_tokens.weight`, and
`lm_head.weight`.

Stage 3: single encoder block parity with per-layer relative bias and RMSNorm.

Stage 4: full encoder parity and cacheable encoder output ABI.

Stage 5: decoder prefill parity with causal self-attention, cross-attention,
relative bias, and tied LM head scale.

Stage 6: autoregressive decode with `EncoderDecoderCache`: self K/V append,
cross K/V one-time update/reuse, cache reorder support for generation.

Stage 7: optimize: relative-bias precompute, gated FFN fusion, last-token logits,
and optionally FlashAttention-compatible fused attention.

Initially stubbable: task heads, dropout, losses, gradient checkpointing, output
attentions/hidden states, and exact fp16 overflow clamp behavior.

## 12. Parity and validation plan

- Config tests: official small/base/xl/xxl configs parse to effective
  `tie_word_embeddings=True`; `K=A*D` is preserved; ignored fields are logged.
- Tokenizer metadata smoke: pad/eos/unk IDs, 300 sentinel tokens, and vocab size
  agree with checkpoint files.
- Custom op tests: relative bucket indices for encoder and decoder, including
  positive/negative distances and saturation beyond 128.
- RMSNorm tests: fp32 accumulation for fp16/bf16 inputs, no mean subtraction, no bias.
- Single-layer parity: encoder block and decoder block for random tensors.
- Full encoder parity: `UMT5EncoderModel` hidden states for small config.
- Prefill parity: `UMT5ForConditionalGeneration` logits with masks and varying source/target lengths.
- Decode parity: one-token incremental decode equals full prefill slice within tolerance; verify self cache growth and cross cache reuse.
- Recommended tolerances: fp32 `rtol=1e-4, atol=1e-5`; fp16/bf16 start at `rtol=5e-2, atol=5e-2` until fused softmax/RMSNorm policies are calibrated.

## 13. Performance probes

- Tokenizer throughput separately from GPU runtime.
- Encoder-only throughput over `B` and `S`.
- Decoder prefill throughput over `B`, `S`, and `T`.
- Decode tokens/sec with fixed encoder outputs and growing self cache.
- Cross-cache memory and self-cache memory per layer:
  `2 * B * A * length * D * dtype_size`, separately for self and cross caches.
- Relative-bias overhead: eager bucket/gather versus precomputed/sliced tables.
- FFN GEMM/activation breakdown for gated GELU.
- LM head projection cost for large vocab `V=256384`, full sequence versus last token.
- Attention backend comparison: composed BMM/softmax/BMM versus fused attention with additive bias.

## 14. Skip/defer list

- Training losses and gradient checkpointing.
- Dropout behavior in training mode.
- Sequence classification, token classification, and question answering heads.
- Beam search/controller details beyond cache reorder and token logits.
- Output attentions/hidden states unless needed for debugging.
- `inputs_embeds`-only paths for first generation integration.
- Exact fp16 inf-clamp behavior unless parity tests exercise overflow.
- Quantized/packed checkpoints; no official UMT5-specific packed format was inspected here.

## 15. Final implementation checklist

- [ ] Parse UMT5Config and apply effective source defaults/overrides.
- [ ] Preserve `K=A*D` separately from `H`.
- [ ] Load T5Tokenizer metadata and sentinel-token contract.
- [ ] Load shared embedding/LM head as one logical tied parameter.
- [ ] Implement UMT5 RMSNorm.
- [ ] Implement relative-position bucket and per-layer bias gather.
- [ ] Implement encoder self-attention with additive padding mask.
- [ ] Implement decoder causal self-attention with `EncoderDecoderCache` self K/V.
- [ ] Implement decoder cross-attention with one-time cross K/V cache update.
- [ ] Implement gated GELU FFN.
- [ ] Implement decoder output scale before tied LM projection.
- [ ] Add single-block, encoder, prefill, and decode parity tests.
- [ ] Add performance probes for encoder, prefill, decode, relative bias, FFN, and LM head.
