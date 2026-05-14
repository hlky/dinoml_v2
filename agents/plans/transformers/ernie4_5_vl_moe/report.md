# DinoML audit: Transformers `ernie4_5_vl_moe`

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model family: ernie4_5_vl_moe
Primary runtime target: multimodal image/video-to-text conditional generation
Source root: X:/H/transformers/src/transformers/models/ernie4_5_vl_moe
```

Source files inspected:

- `configuration_ernie4_5_vl_moe.py`
- `modeling_ernie4_5_vl_moe.py`
- `modular_ernie4_5_vl_moe.py`
- `processing_ernie4_5_vl_moe.py`
- `image_processing_ernie4_5_vl_moe.py`
- `image_processing_pil_ernie4_5_vl_moe.py`
- `video_processing_ernie4_5_vl_moe.py`
- `convert_ernie4_5_vl_moe_to_hf.py`

The generated modeling/config/image-processing files state that they are generated from `modular_ernie4_5_vl_moe.py`; for future Transformers-source edits the modular file is authoritative, while the generated file is still the exact runtime code inspected here.

Representative config snapshots saved under `_sources/`:

- `baidu/ERNIE-4.5-VL-28B-A3B-Base-PT`
- `baidu/ERNIE-4.5-VL-28B-A3B-Paddle`
- `baidu/ERNIE-4.5-VL-28B-A3B-Thinking`
- `baidu/ERNIE-4.5-VL-424B-A47B-Paddle`
- `tiny-random/ernie-4.5-vl-moe`

Important source gap: the official Baidu configs inspected are legacy/remote-code configs with `model_type: "ernie4_5_moe_vl"`, not native nested `model_type: "ernie4_5_vl_moe"` configs. The native conversion script maps top-level fields into `text_config`, `vision_config`, token IDs, `rope_parameters`, and `mlp_layer_types`. DinoML should require either a converted native config or run an explicit conversion/admission step.

## 2. High-level architecture

ERNIE 4.5 VL MoE is a multimodal generation model:

```text
CPU processor
  -> flattened image/video patch rows + grid metadata + expanded placeholder tokens
  -> vision transformer encoder
  -> variable-resolution resampler/projector
  -> masked-scatter stitch into text token embeddings
  -> MoE causal decoder prefill
  -> cached autoregressive decode
  -> lm_head logits/sampling
```

Stage decomposition:

- CPU/data pipeline: image/video decode, resize, rescale, normalize, patch flattening, placeholder expansion, `mm_token_type_ids`, `moe_mm_token_type_ids`, `image_grid_thw`, `video_grid_thw`.
- Vision encoder: packed noncausal variable-length vision transformer over flattened patch rows; independently cacheable per prompt image/video.
- Resampler/projector: spatial merge, hard-coded temporal merge size 2, GELU/LayerNorm projection into text hidden width; independently testable before text stitch.
- Prefix construction: token embedding lookup plus image/video feature copy into placeholder-token rows.
- Prefill: full multimodal prefix through causal MoE decoder with M-RoPE and KV cache creation.
- Decode: text-only token steps using cached K/V and cached `rope_deltas`; pixel inputs and multimodal token type IDs are dropped after first iteration when cache is used.

## 3. Important config dimensions

Native source defaults:

| Field | Native default / source behavior |
| --- | --- |
| text hidden size | 2560 |
| text layers | 28 |
| text heads / KV heads | 20 / 4 |
| default head_dim | `hidden_size // num_attention_heads` = 128 unless config has `head_dim` |
| vocab size | 103424 |
| max positions | 131072 |
| text RoPE | only `rope_type="default"` accepted; default theta 500000; M-RoPE section `[22, 22, 20]` |
| text MLP | SwiGLU: gate/up/down, `silu`, optional bias via `use_bias` |
| MoE | default `moe_k=6`, `moe_num_experts=64`, shared experts=2, text/vision expert intermediate sizes `[1536, 512]` |
| native `mlp_layer_types` | default first layer dense, remaining layers sparse |
| vision hidden/depth/heads | 1280 / 32 / 16 |
| vision patch/merge | patch 14, spatial merge 2, temporal merge 2 |
| vision intermediate | 4 * 1280 = 5120 |
| cache | `use_cache=True`; standard Transformers `Cache`/`DynamicCache` |
| dtype in official configs | bfloat16 from config metadata |

Representative checkpoint sweep:

| Config source | Source status | Text dims | MoE | Vision dims | Notes |
| --- | --- | --- | --- | --- | --- |
| `baidu/ERNIE-4.5-VL-28B-A3B-Base-PT` | legacy remote-code config, convertible | H=2560, L=28, Q=20, KV=4, vocab=103424 | top-k 6, experts `[64,64]`, shared 2, interm `[1536,512]` | H=1280, depth=32, heads=16, patch=14, merge=2 | includes `rope_scaling` type default with `[22,22,20]`; `tie_word_embeddings=true` |
| `baidu/ERNIE-4.5-VL-28B-A3B-Paddle` | legacy remote-code config, convertible | H=2560, L=28, Q=20, KV=4 | same as 28B Base | same as 28B Base | omits some token/max-position fields present in Base; conversion/source defaults fill some fields |
| `baidu/ERNIE-4.5-VL-28B-A3B-Thinking` | legacy remote-code config, convertible | same as 28B Base | same as 28B Base | same as 28B Base | tokenizer chat template differs for thinking mode; neural graph unchanged by native source |
| `baidu/ERNIE-4.5-VL-424B-A47B-Paddle` | legacy remote-code config, convertible with care | H=8192, L=54, Q=64, KV=8, vocab=103424 | top-k 8, experts `[64,64]`, interm `[3584,1536]`, shared omitted in snapshot | H=1280, depth=32, heads=16 | `tie_word_embeddings=false`; `moe_layer_start_index=3` scalar in snapshot, while conversion uses `min(...)`/`max(...)` and expects iterable-like legacy shape |
| `tiny-random/ernie-4.5-vl-moe` | open tiny mirror, legacy-style config | H=8, L=2, Q=4, KV=4 | top-k 8, experts `[32,32]`, shared 0, interm `[32,32]` | H=16, depth=2, heads=1 | many training/fusion flags are not read by native source |

## 3a. Family variation traps

- Native `model_type` differs from many official checkpoints. Do not admit `ernie4_5_moe_vl` directly without converting or using remote code.
- Official legacy configs store text fields at top level; native code expects `config.text_config` and `config.vision_config`.
- Legacy `moe_num_experts` is a two-element list for text/vision modality-isolated experts, but native converted text config uses a scalar count.
- MoE intermediate size is a two-element list: text-routed experts use index 0; vision-routed experts use index 1. Shared experts use text intermediate size times shared-expert count.
- `mlp_layer_types` controls dense vs sparse per layer. Native default is `[dense, sparse, ..., sparse]`; converted configs derive it from legacy layer start/end/interval.
- Text attention is GQA when `num_key_value_heads < num_attention_heads`.
- `head_dim` can be explicit; do not infer projection widths from hidden size alone.
- Text RoPE rejects non-default rope types even though the decorator references dynamic RoPE update infrastructure.
- Vision path consumes flattened patch rows, not NCHW feature maps. Layout passes must not reinterpret it as raw image tensor compute after preprocessing.
- Resampler is hard-coded for `temporal_merge_size == 2`; native video processor also rejects other temporal patch sizes.
- Source uses broad `masked_scatter` for multimodal stitch, but processor guarantees a stricter placeholder expansion pattern if text comes from the processor.
- `moe_mm_token_type_ids` includes image/video start/end tokens for modality-isolated MoE; plain `mm_token_type_ids` marks only placeholder tokens for M-RoPE.
- Several legacy fields such as `fuse_rope`, `use_sparse_flash_attn`, `moe_capacity`, `moe_multimodal_dispatch_use_allgather`, and training/recompute flags are not read by inspected native modeling code.

## 4. Operator coverage checklist

Tensor/layout ops:

- Embedding lookup for `input_ids`.
- Reshape/view/transpose/permute/flatten/contiguous.
- `cat`, `split`, `repeat`, `repeat_interleave`, `index_select`, `where`, boolean masking, `masked_scatter`, `index_add_`.
- `topk`, `gather`, `one_hot`, `nonzero`, `cumsum`, `pad`.
- Runtime shape checks for image/video placeholder count and grid divisibility.

Neural primitives:

- Linear/GEMM with optional bias; many text projections are biasless when `use_bias=false`.
- RMSNorm for text/resampler output.
- LayerNorm for vision blocks and resampler MLPs.
- Activations: `silu`, `quick_gelu`, `gelu`.
- Vision patch projection: `Linear(3 * 14 * 14 -> vision_hidden)` over pre-flattened patch rows.
- Resampler: `Linear(vision_hidden * 4 -> same) -> GELU -> Linear -> LayerNorm`, then temporal concatenation to width `vision_hidden * 8`, second GELU/LayerNorm MLP, final `Linear(vision_hidden * 4 -> text_hidden) -> RMSNorm`.

Attention primitives:

- Text causal self-attention, GQA/MQA-capable, RoPE before cache update.
- Vision noncausal packed varlen self-attention over per-frame/per-image patch sequences, with `cu_seqlens`.
- Attention backend dispatch through `ALL_ATTENTION_FUNCTIONS`; FlashAttention path uses `cu_seq_lens_q/k`, eager path splits by lengths.

MoE:

- FP32 router linear: `[tokens, hidden] x [num_experts, hidden]`.
- Softmax over experts, correction bias add, `topk`, gather, renormalize by clamped selected-weight sum.
- Per-expert SwiGLU GEMM: packed `gate_up_proj[expert]` with shape `[2 * intermediate, hidden]`, split order `(gate, up)`, then `down_proj[expert]` `[hidden, intermediate]`.
- Weighted expert accumulation by `index_add_`.
- Modality isolation: text tokens and vision tokens route through separate expert parameter sets.

Position/rotary:

- Text M-RoPE with 3D position IDs shape `[3, batch, seq]`, pre-rotated frequency order, recomposition to head dimension.
- Vision 2D RoPE from `grid_thw`, spatial-merge-aware position reorder, duplicated H/W embedding.

Generation/cache:

- Per-layer self-attention KV cache, stored after RoPE application.
- `rope_deltas` cached on the model object to shift text decode positions after multimodal prefill.
- `logits_to_keep` last-token or indexed-logit slicing before `lm_head`.
- Generation input expansion has special handling for visual tensors because their leading dimension is total patch rows, not batch.

Preprocessing-coupled ops:

- Smart resize to multiples of `patch_size * merge_size`.
- Rescale/normalize with OpenAI CLIP mean/std.
- Image patch packing to `[sum_images, grid_h * grid_w, 3 * patch^2]`.
- Video sampling, optional timestamp drawing, even-frame padding, patch packing to `[sum_videos, grid_t * grid_h * grid_w, 3 * patch^2]`.
- Placeholder expansion by `grid.prod() // merge^2` for images and `grid.prod() // (merge^2 * temporal_patch_size)` for videos.

## 5. Layer/block breakdown

Vision patch preprocessing, per image:

```text
RGB image/video frame -> resize to H,W divisible by 28
-> rescale/normalize
-> flatten 14x14 patches in spatial-merge-aware order
-> pixel_values rows [num_patch_rows, 588]
-> image_grid_thw/video_grid_thw rows [T, H/14, W/14]
```

Vision tower, repeated `vision_depth` times:

```text
x = Linear(588 -> vision_hidden, bias=False)(patch_rows)
for block:
  residual = x
  x = LayerNorm(x)
  q,k,v = Linear(vision_hidden -> 3 * vision_hidden, bias=True)
  q,k = vision_rope(q,k, grid_thw)
  x = packed_noncausal_attention(q,k,v, cu_seqlens)
  x = residual + Linear(vision_hidden -> vision_hidden)(x)
  x = x + Linear(intermediate -> vision_hidden)(quick_gelu(Linear(vision_hidden -> intermediate)(LayerNorm(x))))
x = LayerNorm(x)
```

Resampler/projector:

```text
x = reshape groups of spatial_merge_size^2 tokens into width vision_hidden * 4
x = spatial MLP + LayerNorm
x = temporal_slicing: concat even/odd temporal slices; images duplicate the single temporal slice
x = temporal MLP + LayerNorm over width vision_hidden * 4
x = Linear(vision_hidden * 4 -> text_hidden)
x = RMSNorm(x)
```

Text decoder layer, repeated `text_layers`:

```text
res = x
x = RMSNorm(x)
q = Linear(hidden -> num_heads * head_dim, bias=use_bias)
k = Linear(hidden -> num_kv_heads * head_dim, bias=use_bias)
v = Linear(hidden -> num_kv_heads * head_dim, bias=use_bias)
q,k = text_mrope(q,k, position_ids)
k,v = cache.update(k,v, layer_idx) if cache enabled
x = causal_attention(q,k,v, mask)
x = Linear(num_heads * head_dim -> hidden, bias=use_bias)(x) + res

res = x
x = RMSNorm(x)
if dense layer:
  x = down(silu(gate(x)) * up(x))
else:
  x = modality_isolated_moe(x, moe_mm_token_type_ids)
x = x + res
```

LM head:

```text
logits = Linear(text_hidden -> vocab_size, bias=False)(hidden[:, logits_to_keep, :])
```

Weight alias: `lm_head.weight` is tied to `model.language_model.embed_tokens.weight` in the native conditional generation class. Some legacy configs set `tie_word_embeddings=false`; DinoML should preserve the actual loaded parameter aliasing or explicit untied weights from the converted artifact.

## 6. Attention requirements

Text attention:

- Causal self-attention only for primary generation target.
- GQA: query heads = `num_attention_heads`, KV heads = `num_key_value_heads`, repeat factor = Q/KV.
- Projection widths: Q/O use `num_attention_heads * head_dim`; K/V use `num_key_value_heads * head_dim`.
- RoPE is applied to Q/K before cache update; cached keys are post-RoPE.
- Mask is standard causal/additive mask from Transformers cache/mask utilities.
- Cache shape conceptually `[batch, num_kv_heads, cache_seq, head_dim]` per layer before backend repeat. Exact concrete cache class is Transformers `Cache`.
- Source attention backend is configurable through `_attn_implementation`; eager fallback materializes attention weights, optimized backends should use SDPA/FlashAttention-compatible path.

Vision attention:

- Noncausal self-attention.
- Full MHA, no GQA: Q/K/V all `vision_hidden`, heads = `vision_num_heads`.
- Input is a single packed sequence of all visual patch rows.
- `cu_seqlens = pad(cumsum(repeat_interleave(grid_h * grid_w, grid_t)), left=0)`.
- FlashAttention path uses varlen metadata. Non-Flash path splits Q/K/V into chunks by `cu_seqlens` lengths and runs ordinary noncausal attention per chunk.
- No KV cache for the vision encoder; image/video features can be cached as prompt-prefix embeddings outside the decoder cache.

## 7. Position encoding and custom math

Text M-RoPE:

```python
def ernie_text_mrope_inv_freq(config):
    dim = config.head_dim or config.hidden_size // config.num_attention_heads
    inv = 1.0 / (theta ** (arange(0, dim, 2) / dim))
    h, w, t = config.mrope_section  # default [22, 22, 20]
    out[:h+w] = concat([inv[:-t][0::2], inv[:-t][1::2]])
    out[-t:] = inv[-t:]
    return out

def ernie_text_mrope(position_ids_3d):
    # position_ids_3d: [3, batch, seq] for temporal, height, width
    freqs = inv_freq[None, None, :, None] @ position_ids_3d[:, :, None, :]
    cos, sin = cos(freqs).transpose(2, 3), sin(freqs).transpose(2, 3)
    # split [h,w,t], rotate source order, interleave h/w, append t, repeat each scalar twice
    return recomposed_cos, recomposed_sin
```

Text 3D position IDs:

- Text runs use monotonic 1D positions expanded to all 3 axes.
- Vision runs use temporal/height/width positions from `grid_thw`, divided by temporal/spatial merge sizes.
- After each image/video group, `current_pos` advances by `max(grid_h, grid_w) // spatial_merge_size`, not by token count.
- `rope_deltas = max(position_ids) + 1 - nonpad_sequence_length`; decode uses this cached delta.

Vision RoPE:

- Builds H/W indices from grid shape with spatial-merge-aware reorder.
- Looks up a 1D `outer(arange(max_grid_size), inv_freq)` table for H and W, flattens H/W embeddings, then duplicates to form cos/sin width for Q/K.

Precomputable:

- Base inverse-frequency tables per dtype/device and maximum grid size.
- Processor-derived grid metadata and placeholder counts per prompt.

Dynamic:

- M-RoPE position IDs depend on exact token order, `mm_token_type_ids`, `attention_mask`, image/video grids, and cached decode length.

## 8. Preprocessing and input packing

Image processor ABI:

- Accepts PIL/NumPy/Torch images, channel-first or channel-last.
- Converts to RGB, bicubic resize, rescale by `1/255`, normalize by OpenAI CLIP mean/std.
- Resize uses `smart_resize` with factor `patch_size * merge_size = 28`, min pixels 3136, and source default max pixels `14 * 14 * 4 * 1280`; saved official preprocessors use max pixels 4816896.
- Emits:
  - `pixel_values`: `[sum_images, grid_h * grid_w, 588]`
  - `image_grid_thw`: `[num_images, 3]` with rows `[1, grid_h, grid_w]`

Video processor ABI:

- Owns decode/sample unless caller passes tensors plus metadata through Transformers video utilities.
- Defaults: sample frames enabled, min 16, max 180, temporal patch/merge size 2, optional timestamp drawing enabled.
- If frame count is odd, duplicates the last frame.
- Accepts channel-last videos but converts to `[T, C, H, W]`.
- Emits:
  - `pixel_values_videos`: `[sum_videos, grid_t * grid_h * grid_w, 588]`
  - `video_grid_thw`: `[num_videos, 3]`
- Official video processor JSON was not available at probed URLs, so these are source defaults.

Text/multimodal processor ABI:

- Tokenizer-owned special strings: image/video start, placeholder, and end tokens.
- Placeholder expansion:
  - image placeholder count = `image_grid_thw[i].prod() // merge_size**2`
  - video placeholder count = `video_grid_thw[i].prod() // (merge_size**2 * temporal_patch_size)`
- Emits `mm_token_type_ids`: text 0, image placeholder 1, video placeholder 2.
- Emits `moe_mm_token_type_ids`: same, but image/video start/end tokens are also marked as vision tokens for MoE routing.

Placeholder stitch:

- Source computes image/video features, concatenates per-modality feature lists in processor order, expands boolean masks to hidden width, checks `mask.numel == features.numel`, then calls `inputs_embeds.masked_scatter(mask, features)`.
- Processor guarantees the stricter pattern: placeholder tokens appear in contiguous runs where the chat template placed each image/video marker, and expanded counts match grid-derived resampler outputs. DinoML can lower this to guarded row-copy by ordered placeholder indices instead of admitting arbitrary boolean scatter.

## 9. Graph rewrite / lowering opportunities

### Rewrite: pre-flattened patch projection to GEMM

Source pattern:

```text
pixel_values [Npatch, 588] -> Linear(588 -> vision_hidden, bias=False)
```

Replacement: GEMM RRR/RCR with static K=588.

Preconditions:

- Processor or caller already emitted flattened patch rows in source order.
- `patch_size=14`, `in_channels=3`; K equals `3 * 14 * 14`.
- No NHWC/NCHW reinterpretation inside model graph.

Failure cases: raw images passed into graph, different patch size/channel count, or processor variant with different flatten order.

### Rewrite: placeholder `masked_scatter` to indexed row copy

Source pattern:

```text
inputs_embeds = inputs_embeds.masked_scatter(input_ids == image_token_id expanded to hidden, image_embeds)
```

Replacement:

```text
positions = nonzero(input_ids == token_id) in row-major order
copy rows image_embeds[i] -> inputs_embeds[positions[i]]
```

Preconditions:

- `input_ids` available.
- Count of placeholder rows equals feature rows.
- Placeholder positions are ordered by processor expansion and do not overlap image/video masks.
- Hidden width matches text hidden size.

Failure cases: caller provides only `inputs_embeds` and no `input_ids`, duplicate embedding-value equality ambiguity, mismatched counts, arbitrary user-built nonprocessor placeholders.

### Rewrite: MoE expert loop to grouped expert GEMM

Source pattern:

```text
topk router -> per-expert token gather -> gate_up GEMM -> silu*up -> down GEMM -> weighted index_add
```

Replacement: sort/group tokens by selected expert, run grouped GEMM per expert, then scatter-add weighted outputs.

Preconditions:

- Inference-only; no router aux loss outputs required.
- Expert weights use source layout `gate_up_proj[E, 2I, H]`, split order `(gate, up)`, and `down_proj[E, H, I]`.
- Preserve FP32 router and selected-weight normalization.
- Preserve modality-isolated expert pools for `moe_mm_token_type_ids`.

Failure cases: needing exact router logits output for diagnostics/training, dynamic expert capacity policies from legacy remote code, distributed all-to-all behavior.

### Rewrite: text QKV projections to fused projection

Source pattern: separate Q/K/V linears.

Replacement: one packed projection only if weights are packed as `[Q rows, K rows, V rows]` by DinoML during load.

Preconditions:

- Bias flags match; source default has no bias.
- Split widths are explicit: Q=`num_heads*head_dim`, K/V=`num_kv_heads*head_dim`.
- RoPE still applies before cache update.

Failure cases: explicit `head_dim` causing hidden-size mismatch assumptions, tensor-parallel sharded weights, quantized per-projection metadata.

### Rewrite: vision packed varlen attention to FlashAttention

Source pattern: `cu_seqlens` packed noncausal attention.

Replacement: varlen FlashAttention with Q/K/V `[total_tokens, heads, head_dim]`.

Preconditions:

- `cu_seqlens` int32 or backend-compatible.
- No attention mask.
- Noncausal; dropout 0 in inference.
- Vision RoPE applied first.

Failure cases: backend lacks varlen API; ONNX/tracing dtype compatibility path; output attentions requested.

### Layout guard

Do not globally translate image tensors to NHWC inside the model: the model proper receives rank-3 flattened patch rows. Layout optimization belongs in the processor-to-patch-packing region only, with a guard that the flatten order exactly matches source `view`/`permute`/`reshape`.

## 10. Kernel fusion candidates

Highest priority:

- RMSNorm and LayerNorm: used in every text/vision block and resampler.
- Text GQA attention with M-RoPE and KV cache: core prefill/decode path.
- MoE grouped expert GEMM: dominant text decoder cost, with modality-isolated expert pools.
- Placeholder indexed row-copy: avoids broad boolean scatter admission.
- Resampler spatial/temporal projection: small but model-specific and necessary for multimodal parity.

Medium priority:

- Fused QKV projection + M-RoPE preparation for prefill.
- SwiGLU dense MLP and expert MLP epilogues.
- Vision varlen FlashAttention with `cu_seqlens`.
- Last-token-only logits via `logits_to_keep`.
- Processor patch flattening on GPU for controlled image pipelines, after CPU parity.

Lower priority:

- Beam-search visual tensor expansion optimization.
- Router aux loss and router-logit collection.
- Timestamp drawing acceleration; this is CPU/preprocessing and not required for neural graph parity.
- Distributed/tensor-parallel MoE all-to-all behavior from legacy configs.

## 11. Runtime staging plan

1. Config admission and conversion: accept native `ernie4_5_vl_moe` configs; for official legacy configs require conversion using the source mapping or reject with a clear message.
2. Text-only decoder parity: embeddings, M-RoPE with ordinary 1D positions, GQA attention, dense first layer, sparse MoE layers, LM head.
3. Decode cache parity: per-layer KV cache, post-RoPE cached keys, `rope_deltas` behavior for text-only and multimodal prefixes.
4. Vision encoder parity: flattened patch rows + `grid_thw` into vision tower output.
5. Resampler parity: spatial merge, temporal slicing/duplication, projector, RMSNorm.
6. Multimodal stitch parity: processor placeholder expansion, indexed-copy lowering, M-RoPE position IDs with mixed text/image/video groups.
7. Optimized prefill/decode: FlashAttention/SDPA, grouped MoE, fused normalizations, last-token logits.
8. Optional production features: video decode/sampling ownership, beam expansion of visual tensors, quantized weight loading, tensor parallel.

Initially stub:

- Training losses, router aux loss, output attentions/hidden states.
- Video timestamp drawing; accept preprocessed frames or disable drawing.
- Beam search beyond greedy/sampling.

## 12. Parity and validation plan

- Unit parity for `smart_resize`, image patch flattening, video patch flattening, and placeholder-count expansion.
- Unit parity for text M-RoPE inv-frequency pre-rotation/recomposition using synthetic 3D position IDs.
- Unit parity for `get_rope_index` with mixed text/image/video token-type runs and attention masks.
- Unit parity for resampler `_temporal_slicing` for image `T=1`, even video `T`, and odd-frame-after-processor duplication.
- Single vision block parity in fp32/bf16.
- Single decoder layer parity for dense layer and sparse MoE layer, including modality-isolated MoE masks.
- End-to-end prefill logits for text-only, one image, and image+video prompts.
- Decode token parity from a multimodal prefix, checking that pixel inputs are ignored after first cached iteration.
- Suggested tolerances: fp32 `rtol=1e-4/atol=1e-5`; bf16/fp16 block-level `rtol=5e-2/atol=5e-2` unless accumulation is matched more closely.

## 13. Performance probes

- Processor throughput: resize/normalize/patch packing for image resolution sweep.
- Video pipeline: decode/sample/timestamp/patch packing split by frame count and resolution.
- Vision tower throughput by total packed patch tokens and number of `grid_thw` segments.
- Resampler throughput by image/video token count.
- Prefill tokens/sec for text-only vs image/video prefix.
- Decode tokens/sec with KV cache for batch size and sequence length sweeps.
- MoE routing distribution and grouped GEMM utilization by top-k and active experts.
- KV cache memory by layers, KV heads, head_dim, dtype, and sequence length.
- Placeholder stitch copy bandwidth and CPU/GPU synchronization overhead.
- Last-token logits vs full-sequence logits.
- Quantized or GGUF load/dequant probes only after dense parity is stable.

## 14. Skip/defer list

- Training, losses, gradient checkpointing, recompute flags.
- Router aux loss unless `output_router_logits=True` is a required product feature.
- Remote-code-only distributed MoE capacity/all-to-all behavior.
- Beam search and visual tensor expansion beyond minimal generation.
- Output attentions/hidden states.
- Timestamp drawing in the first GPU runtime.
- Non-default text RoPE types; native source rejects them.
- Alternate temporal patch sizes; native video processor rejects values other than 2.
- General boolean `masked_scatter`; use guarded indexed row copy for processor-generated placeholders.
- Global NCHW/NHWC layout translation inside the model graph.

## 15. Final implementation checklist

- [ ] Add config admission for native `ernie4_5_vl_moe` and explicit legacy conversion/rejection for `ernie4_5_moe_vl`.
- [ ] Preserve source-derived token IDs and tokenizer/processor special-token ABI.
- [ ] Implement or bind image/video patch packing parity tests.
- [ ] Implement text M-RoPE pre-rotation/recomposition and 3D position ID construction.
- [ ] Implement GQA causal attention with post-RoPE KV cache update.
- [ ] Implement dense SwiGLU MLP.
- [ ] Implement modality-isolated MoE router and grouped expert GEMM lowering.
- [ ] Implement vision patch projection and packed varlen noncausal attention.
- [ ] Implement variable-resolution resampler including temporal slicing.
- [ ] Lower placeholder stitch to guarded indexed row copy.
- [ ] Add single-block text, vision, resampler, and full prefill parity tests.
- [ ] Add cached decode parity with multimodal `rope_deltas`.
- [ ] Benchmark processor, vision, prefill, decode, MoE, and cache memory separately.
