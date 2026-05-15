# Transformers audit: exaone4_5

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: LGAI-EXAONE/EXAONE-4.5-33B, plus FP8 and AWQ variants
Config source: HF config.json / processor_config.json snapshots under _sources
Source files inspected:
  transformers/src/transformers/models/exaone4_5/configuration_exaone4_5.py
  transformers/src/transformers/models/exaone4_5/modeling_exaone4_5.py
  transformers/src/transformers/models/exaone4_5/modular_exaone4_5.py
  transformers/src/transformers/models/exaone4_5/processing_exaone4_5.py
  transformers/src/transformers/models/exaone4/configuration_exaone4.py
  transformers/src/transformers/models/exaone4/modeling_exaone4.py
Any missing files or assumptions: video_preprocessor_config.json returned 404; video processor data came from processor_config.json.
```

`modeling_exaone4_5.py` is generated from `modular_exaone4_5.py`. The wrapper delegates the language body to `AutoModel.from_config(config.text_config)`, which resolves to `exaone4`, so the decoder audit includes the `exaone4` source dependency.

## 2. High-level architecture

Primary runtime target: multimodal conditional generation, staged as text decoder first, then image/video prefix encoding plus decoder prefill/decode.

```text
text/images/videos -> processor token/patch packing -> vision encoder + patch merger
                 -> masked scatter into token embeddings -> exaone4 decoder prefill
                 -> KV-cache decode -> lm_head logits/sampling
```

The model is a VLM wrapper around:

- A Qwen2-VL-like vision transformer: 3D patch embedding, 2D vision RoPE, packed variable-length vision attention, window attention with periodic full attention, patch merger to text hidden width.
- An EXAONE4 causal decoder: RMSNorm, GQA self-attention, QK-norm, hybrid sliding/full causal attention, SwiGLU MLP, final RMSNorm, untied LM head.
- Processor-owned placeholder expansion for `<|image_pad|>` and `<|video_pad|>` tokens; model-owned `masked_scatter` replaces those token embeddings with vision features.

Independently stageable pieces: processor shape/token-count validation, vision encoder, embedding stitch, text prefill, cache decode, and logits-only head.

## 3. Important config dimensions

Representative dense 33B config:

| Field | Text decoder | Vision encoder |
|---|---:|---:|
| dtype | bfloat16 | bfloat16 |
| hidden_size | 5120 | 2048 |
| layers/depth | 64 | 28 |
| attention heads | 40 | 32 |
| KV heads | 8 | 8 |
| head_dim | 128 | 64 |
| intermediate_size | 27392 | 5120 |
| activation | silu | silu |
| max_position_embeddings | 262144 | n/a |
| sliding_window | 4096 | window_size 112 pixels |
| layer pattern | `LLLG` repeated | full blocks `[6, 13, 20, 27]` |
| vocab_size | 153600 | n/a |
| patch / merge | n/a | patch 14, temporal patch 2, spatial merge 2 |
| output width | 5120 logits input | 5120 merger output |

Checkpoint sweep:

| Checkpoint | Architecture variation | Quant metadata | Notable source issue |
|---|---|---|---|
| `EXAONE-4.5-33B` | 64 decoder layers, 48 sliding + 16 full, 28 vision layers | none | historical `text_config.model_type="exaone4_5_text"` and `rope_scaling` |
| `EXAONE-4.5-33B-FP8` | same dimensions, but `layer_types` length is 65 for 64 layers | compressed-tensors FP8: Linear weights per channel, dynamic token activations | extra trailing layer type is unused by source loop; normalize/reject |
| `EXAONE-4.5-33B-AWQ` | same dimensions as dense | compressed-tensors int4 pack quant, group size 128 | dense visual and selected language/MTP modules are ignored by quant config |

## 3a. Family variation traps

- The text decoder is not implemented in `exaone4_5`; it is delegated to `exaone4`.
- `hidden_size == num_attention_heads * head_dim` for 33B text, but DinoML should read `head_dim`/derive it from config rather than assume future variants.
- GQA is required: 40 query heads, 8 KV heads, repeat factor 5.
- Text full-attention layers use global NoPE when `sliding_window` is set; sliding layers apply RoPE and use a sliding-window mask.
- Dense config uses `rope_scaling`, newer quant configs use `rope_parameters`; loader must normalize this before constructing text RoPE.
- FP8 config has one extra `layer_types` entry. Admission should require at least `num_hidden_layers` entries and either reject extras or log normalization.
- Vision QKV is packed as one biased `Linear(hidden, q_dim + 2 * kv_dim)` and split `[Q, K, V]`.
- Text Q/K/V projections are separate bias-free linears; safe fusion must preserve separate weight names and QK norm between projection and RoPE/attention.
- `masked_scatter` is broad in source, but processor expansion gives a stricter ordered placeholder-token contract that DinoML can guard.
- Layout-sensitive vision code starts from channels-first patch tensors and uses a `Conv3d` over `[C,T,H,W]` patches. NHWC/channel-last is an optimization only inside a controlled rewrite.

## 4. Operator coverage checklist

Tensor/layout ops:

- Embedding lookup, boolean/equality masks, masked scatter or guarded indexed row copy, cat/split, reshape/view, transpose/permute, argsort, gather/indexing, repeat/repeat_interleave, cumsum, pad, unique_consecutive, arange, stack.
- Vision packing: `Conv3d(3 -> 2048, kernel=stride=(2,14,14), bias=False)`, window indexing and reverse indexing.

Neural primitives:

- RMSNorm over width 5120, head_dim 128, vision width 2048, vision head_dim 64.
- Dense GEMM/Linear: text Q 5120->5120, K/V 5120->1024, O 5120->5120, MLP gate/up 5120->27392, down 27392->5120, LM head 5120->153600.
- Vision packed QKV 2048->3072 with bias, projection 2048->2048 with bias, MLP gate/up/down 2048<->5120 with bias, patch merger `8192->8192->5120`.
- SiLU/SwiGLU, GELU in patch merger, residual adds.

Attention primitives:

- Text causal GQA, full and sliding-window variants, QK norm, optional FlashAttention/SDPA/FlexAttention backend dispatch.
- Vision noncausal GQA over packed variable-length sequences, with windowed and full blocks selected by `cu_seqlens`.

Position/rotary ops:

- Text Llama3 RoPE parameters: theta 1,000,000, factor 16, low/high factors 1/4, original max position 8192.
- Global NoPE on full text layers in hybrid configs.
- Vision 2D RoPE over height/width patch positions; temporal affects token packing, not vision RoPE frequencies.

Generation/cache ops:

- Dynamic KV cache with per-layer K/V shaped before repeat as `[batch, 8, seq, 128]` for text.
- Cache update happens after optional RoPE in sliding layers; full NoPE layers cache unrotated QK-norm keys.
- `logits_to_keep` slices hidden states before LM head.

Quantized/packed weight metadata:

- FP8 and AWQ configs use `compressed-tensors` metadata; the modeling source itself does not dequantize. Treat as loader/provider work with dense fallback or explicit reject.

## 5. Layer/block breakdown

Text decoder block, repeated 64 times:

```text
q = Linear(hidden 5120 -> 40 * 128, bias=False)
k = Linear(hidden 5120 -> 8 * 128, bias=False)
v = Linear(hidden 5120 -> 8 * 128, bias=False)
q = RMSNorm(q over head_dim)
k = RMSNorm(k over head_dim)
if sliding layer or no sliding_window: q,k = RoPE(q,k)
else full layer with hybrid sliding_window: NoPE(q,k)
k,v = cache.update(k,v, layer_idx) when cache enabled
attn = causal GQA(q,k,v, mask=full or sliding, scale=1/sqrt(128))
x = residual + RMSNorm(Linear(attn 5120 -> 5120, bias=False))
mlp = Linear(SiLU(gate_proj(x)) * up_proj(x), 27392 -> 5120)
x = residual + RMSNorm(mlp)
```

Vision block, repeated 28 times:

```text
patches = Conv3d(C=3 -> 2048, kernel=stride=(2,14,14), bias=False)
tokens are reordered by window_index
q,k,v = packed Linear(2048 -> 2048 + 512 + 512, bias=True), split Q,K,V
q,k = 2D vision RoPE(q,k)
attn = noncausal GQA over cu_seqlens (window blocks except full indexes)
x = residual + Linear(attn, 2048 -> 2048, bias=True)
x = residual + SwiGLU MLP(2048 -> 5120 -> 2048, bias=True)
merged = RMSNorm + reshape 4 spatial-merge tokens -> Linear 8192 -> 8192 -> GELU -> Linear 8192 -> 5120
reverse argsort restores original merged-token order
```

## 6. Attention requirements

Text attention:

- Causal self-attention only; no cross-attention.
- GQA: 40 query heads, 8 KV heads, repeat factor 5.
- Full layers: causal full mask, NoPE in the hybrid config.
- Sliding layers: causal sliding-window mask with `sliding_window=4096`, RoPE applied, backend receives `sliding_window=4096`.
- Masking source builds a dict with `"full_attention"` and `"sliding_attention"` masks, then selects per layer from `config.layer_types`.
- Cache stores K/V before KV repeat. Source cache update is after the RoPE/NoPE branch, so cache parity must preserve per-layer positional treatment.
- Eager fallback repeats KV, does `Q @ K^T * scale`, adds additive mask, softmax in fp32, dropout, then `P @ V`.

Vision attention:

- Noncausal self-attention over packed image/video patch sequences.
- GQA: 32 query heads, 8 KV heads, head_dim 64.
- FlashAttention path uses `cu_seq_lens_q/k` and max sequence length. Non-flash path splits packed tensors by lengths and loops per segment.
- Window attention uses generated `cu_window_seqlens`; full blocks use image/video-level `cu_seqlens`.

## 7. Position encoding and custom math

Text RoPE/NoPE:

```python
q = q_norm(q_proj(x).view(B, S, Hq, D).transpose(1, 2))
k = k_norm(k_proj(x).view(B, S, Hkv, D).transpose(1, 2))
if sliding_window is None or layer_type == "sliding_attention":
    q, k = rope(q, k, cos[position_ids], sin[position_ids])
# else: full_attention layer in hybrid model uses NoPE
```

Vision 2D RoPE:

```python
h_ids, w_ids = merged_grid_position_ids(grid_thw, spatial_merge_size)
freqs = rotary_emb(max_grid_size)[stack([h_ids, w_ids])].flatten(1)
emb = cat([freqs, freqs], dim=-1)
q, k = vision_rope(q, k, emb.cos(), emb.sin())
```

Precompute opportunities: text inverse frequencies and vision inverse frequencies are static per config. Text cos/sin depend on runtime position ids and long-context RoPE scaling. Vision cos/sin depend on `grid_thw` and window/reorder metadata.

## 8. Preprocessing and input packing

Processor outputs include `input_ids`, `attention_mask`, optional `mm_token_type_ids`, `pixel_values`, `pixel_values_videos`, `image_grid_thw`, `video_grid_thw`, and `second_per_grid_ts`.

Image/video preprocessing is Qwen2VL-style:

- channels-first output, RGB conversion, resize/rescale/normalize with CLIP-like mean/std.
- patch size 14, temporal patch size 2, merge size 2.
- min pixels 3136, max pixels 3211264.
- video config from `processor_config.json`: min frames 4, max frames 768, `do_sample_frames=false`.

Placeholder expansion:

- Text contains `<|image_pad|>` and `<|video_pad|>`.
- Processor replaces one placeholder with `grid_thw.prod() // merge_size**2` repeated special tokens.
- Model validates feature/token element counts, then uses `masked_scatter` to write flattened image/video features into those token positions.
- DinoML can lower this to ordered indexed row copy if it guards placeholder token count, order, and feature flatten order.

## 9. Graph rewrite / lowering opportunities

### Rewrite: vision non-overlap Conv3d patch embed -> Linear

Source pattern: `Conv3d(3, 2048, kernel=stride=(2,14,14), bias=False)` after a view into patch blocks.

Replacement: flatten each `[3,2,14,14]` patch to 1176 values, GEMM with `weight.reshape(2048, 1176).T`, reshape to token rows.

Preconditions: kernel equals stride, padding/dilation default, groups 1, input already packed into complete patches, channels-first flatten order preserved, no bias.

Failure cases: dynamic or incomplete patch tiles, layout-translated tensors without a verified weight permutation, future grouped/bias patch embed.

Parity sketch: compare patch embed output for random bf16/fp32 images over varied `grid_thw`.

### Rewrite: masked_scatter placeholder stitch -> indexed row copy

Source pattern: boolean mask expanded across hidden dim and `inputs_embeds.masked_scatter(mask, vision_features)`.

Replacement: compute placeholder row indices from `input_ids == image_token_id/video_token_id`, verify count equals feature rows, copy rows in source flatten order.

Preconditions: processor-expanded placeholders are ordered, features are `[sum_mm_tokens, hidden]`, no mixed hidden dtype mismatch, token counts match exactly.

Failure cases: caller supplies arbitrary `inputs_embeds` without `input_ids`, noncontiguous or interleaved feature order not matching placeholders, duplicate custom placeholder embeddings.

### Rewrite: separate text Q/K/V projections -> grouped GEMM with post-split QK norm

Source pattern: independent bias-free linears for Q, K, V.

Replacement: optional packed weight GEMM producing `[Q,K,V]`, then split, reshape, QK RMSNorm, RoPE/NoPE, attention.

Preconditions: preserve source split order Q then K then V; QK norm remains after split; K/V widths are 1024 each; packed weight is compile-time transformed and provenance tracked.

Failure cases: quantized variants whose packed metadata cannot be transformed safely, tensor-parallel sharding, future biases.

### Rewrite: sliding/full attention dispatch

Source pattern: per-layer mask lookup by `layer_types`.

Replacement: static layer metadata in DinoML graph with two attention families: sliding causal GQA and full causal NoPE GQA.

Preconditions: `layer_types[i]` known at compile time; sliding window positive; full layers keep NoPE; cache manifest records layer family.

Failure cases: malformed layer_types length, unsupported backend that silently treats sliding as full, cache code that assumes all keys are RoPE-rotated.

## 10. Kernel fusion candidates

Highest priority:

- Text RMSNorm and head-dim Q/K RMSNorm: occurs twice per block plus per attention Q/K.
- GQA FlashAttention with hybrid layer manifest: essential for 256K prefill and 4096-window layers.
- Decode KV cache update + sliding/full attention: cache storage differs by layer RoPE/NoPE behavior.
- SwiGLU MLP fusion: `SiLU(gate) * up -> down` dominates decoder GEMM traffic.
- Last-token-only logits using `logits_to_keep`: avoids full-sequence vocab GEMM.

Medium priority:

- Vision patch Conv3d-to-GEMM and patch merger fusion.
- Vision packed QKV + 2D RoPE + packed varlen attention.
- Placeholder indexed-copy rewrite to avoid general boolean scatter.
- Quantized Linear provider admission for compressed-tensors FP8/AWQ with dense fallback.

Lower priority:

- Beam expansion of visual tensors.
- Training losses and gradient checkpointing.
- Returning dense attention weights.

## 11. Runtime staging plan

Stage 1: parse dense config, normalize `exaone4_5_text -> exaone4`, reject malformed layer metadata, load dense bf16 weights.

Stage 2: text-only one-block and full decoder prefill parity with eager full/sliding attention.

Stage 3: KV-cache decode with per-layer cache manifest: 48 sliding RoPE layers and 16 full NoPE layers.

Stage 4: optimized attention backend for GQA full/sliding, then last-token logits.

Stage 5: vision encoder standalone parity: patch embed, window index, 2D RoPE, packed varlen attention, patch merger.

Stage 6: multimodal stitch and end-to-end image-text prefill/decode.

Stage 7: admit FP8/AWQ configs only after compressed-tensors loader/provider path or route to explicit dense materialization.

## 12. Parity and validation plan

- Unit parity: RMSNorm, RoPE, QK norm, repeat_kv, sliding mask, full NoPE branch.
- Single decoder layer with fixed random weights, both `sliding_attention` and `full_attention`.
- Cache parity: prefill logits vs incremental decode for mixed layer types; verify cached K position encoding per layer.
- Vision unit tests: patch embed rewrite, window indexing/reverse indexing, 2D RoPE, packed varlen attention segments.
- Stitch tests: processor placeholder counts, indexed-copy rewrite vs `masked_scatter`.
- End-to-end: text-only logits, image+text logits, video+text logits if video processor inputs are available.
- Suggested tolerances: fp32 `1e-4` to `1e-5`; bf16/fp16 attention/MLP `1e-2` relative or model-local tolerance after backend choice.

## 13. Performance probes

- Text prefill throughput by sequence length: 4K, 32K, 128K, 256K.
- Decode tokens/sec with batch and cache length sweeps; split by sliding vs full layer time.
- KV-cache memory: 64 layers of `[B, 8, S, 128]` K and V, noting sliding layers may have bounded attention but source DynamicCache still grows unless backend truncates.
- Vision encoder throughput by image pixels and video frames; separately time patch embed, window indexing, attention, merger.
- Placeholder stitch bandwidth and overhead vs indexed copy.
- LM head time with `logits_to_keep=1` vs full sequence.
- Quantized load/dequant/provider comparison for dense, FP8, and AWQ variants.

## 14. Skip/defer list

- Training, labels/loss, gradient checkpointing.
- Sequence/token/question-answering heads from text dependency.
- Beam search visual tensor expansion beyond basic generation compatibility.
- FP8/AWQ execution until compressed-tensors provider metadata is admitted.
- MTP fields (`_num_mtp_layers`, `mtp_*`, ignored load keys) because current wrapper ignores `mtp.*` unexpected keys for the audited conditional-generation path.
- General boolean scatter; prefer guarded indexed-copy lowering.
- Full video ingest/decode ownership; processor output tensors can be the initial ABI.

## 15. Final implementation checklist

- [ ] Parse `exaone4_5` wrapper config and normalize text config aliases.
- [ ] Reject or normalize `layer_types` length mismatches.
- [ ] Load dense bf16 text and vision weights with untied LM head.
- [ ] Implement EXAONE4 RMSNorm, QK norm, SwiGLU, and dense linear coverage.
- [ ] Implement hybrid causal GQA: sliding RoPE layers plus full NoPE layers.
- [ ] Implement KV-cache manifest per layer type.
- [ ] Add text-only prefill and decode parity tests.
- [ ] Implement vision patch embed, 2D RoPE, packed varlen attention, and patch merger.
- [ ] Replace multimodal `masked_scatter` with guarded indexed row copy.
- [ ] Add image-text end-to-end logits parity.
- [ ] Add FP8/AWQ admission path or explicit unsupported-config diagnostics.
- [ ] Benchmark prefill, decode, vision encoder, stitch, LM head, and quantized loading.

