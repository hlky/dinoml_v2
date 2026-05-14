# Transformers audit: qwen3_vl_moe

## 1. Source basis

```text
Transformers commit/version:
  Local checkout X:/H/transformers at b75feb2af64c3e29cbbc1bd859958c5432cc7ed4.
  Representative configs report transformers_version 4.57.0.dev0.

Model id:
  Primary: Qwen/Qwen3-VL-30B-A3B-Instruct and Qwen/Qwen3-VL-235B-A22B-Instruct.
  Variants inspected: matching Thinking variants, Qwen/Qwen3-VL-30B-A3B-Instruct-FP8,
  tiny-random/qwen3-vl-moe.

Config source:
  HF raw config snapshots saved under
  agents/plans/transformers/qwen3_vl_moe/_sources/.

Source files inspected:
  X:/H/transformers/src/transformers/models/qwen3_vl_moe/configuration_qwen3_vl_moe.py
  X:/H/transformers/src/transformers/models/qwen3_vl_moe/modular_qwen3_vl_moe.py
  X:/H/transformers/src/transformers/models/qwen3_vl_moe/modeling_qwen3_vl_moe.py
  X:/H/transformers/src/transformers/models/qwen3_vl/processing_qwen3_vl.py
  X:/H/transformers/src/transformers/models/qwen3_vl/video_processing_qwen3_vl.py
  X:/H/transformers/src/transformers/models/qwen2_vl/image_processing_qwen2_vl.py

Any missing files or assumptions:
  qwen3_vl_moe has no local processor file. Processor ABI comes from Qwen3VLProcessor,
  with Qwen2VLImageProcessorFast named in official preprocessor configs and
  Qwen3VLVideoProcessor in library source. The generated modeling file says
  modular_qwen3_vl_moe.py is authoritative for future Transformers edits, but
  the generated modeling file contains the full expanded implementation.
```

No remote-code-only behavior was required for the inspected official configs. `processor_config.json` was absent for the official repos; `preprocessor_config.json` was available and identifies `processor_class: Qwen3VLProcessor`.

## 2. High-level architecture

Primary DinoML target: multimodal causal generation, with image/video prefill followed by text decode.

Dataflow:

```text
CPU/image/video processor
  -> packed patch rows + image_grid_thw/video_grid_thw + mm_token_type_ids
  -> vision encoder with packed variable-length noncausal attention
  -> merged visual embeddings + DeepStack visual embeddings
  -> token embedding placeholder stitch + M-RoPE position ids
  -> MoE text decoder prefill
  -> cached autoregressive decode
  -> logits/sampling
```

Stage decomposition:

- CPU/data pipeline: image resize/rescale/normalize, video frame sampling, timestamp prompt expansion, tokenization, placeholder expansion, `mm_token_type_ids`.
- Vision runtime: packed patch rows to Conv3d/linear patch embedding, absolute position interpolation, 27-layer ViT-style encoder, merger, three DeepStack merger outputs.
- Prefix construction: replace `<|image_pad|>` / `<|video_pad|>` token embeddings using mask/count-validated visual embeddings; compute 3D M-RoPE ids from token-type runs and THW grids.
- Text prefill: causal GQA decoder with Q/K RMSNorm, M-RoPE, KV cache, MoE MLP in every production layer.
- Decode: pixels are not forwarded after first cached step; positions are generated from cached `rope_deltas`; only text token embeddings and cache update remain.

Independently validatable units: image/video processor ABI, vision encoder output and merger shapes, placeholder stitch, M-RoPE ids, one decoder block, MoE routing/expert GEMM, prefill logits, one-token decode.

## 3. Important config dimensions

Source defaults differ materially from production checkpoints. DinoML should treat checkpoint config as authoritative.

| Field | Source default | 30B-A3B configs | 235B-A22B configs | Notes |
| --- | ---: | ---: | ---: | --- |
| text hidden_size | 2048 | 2048 | 4096 | Vision `out_hidden_size` matches text hidden. |
| num_hidden_layers | 24 | 48 | 94 | Production overrides source default. |
| num_attention_heads | 16 | 32 | 64 | |
| num_key_value_heads | 16 | 4 | 4 | Production uses GQA; source default is MHA-like. |
| head_dim | hidden/heads default | 128 | 128 | Explicit in configs; hidden != heads * source default assumption is possible. |
| intermediate_size | 5632 | 6144 | 12288 | Dense MLP width for `mlp_only_layers`, normally unused in inspected prod configs. |
| moe_intermediate_size | 1408 | 768 | 1536 | Expert gate/up width. |
| num_experts | 60 | 128 | 128 | |
| num_experts_per_tok | 4 | 8 | 8 | Top-k router requirement. |
| decoder_sparse_step | 1 | 1 | 1 | With empty `mlp_only_layers`, every layer is MoE. |
| vocab_size | 151936 | 151936 | 151936 | |
| max_position_embeddings | 128000 default | 262144 | 262144 | |
| rope_theta | 500000 default | 5000000 | 5000000 | Checkpoints store `rope_theta`; source modeling reads normalized rope params. |
| M-RoPE section | `[24,20,20]` | `[24,20,20]` | `[24,20,20]` | Interleaved M-RoPE. |
| attention_bias | false | false | false | Q/K/V/O no bias. |
| tie_word_embeddings | top-level false | false | false | LM head is separate from embeddings. |
| vision depth | 27 | 27 | 27 | |
| vision hidden/heads | 1152 / 16 | same | same | head_dim 72; rotary uses half head dim. |
| vision patch / temporal patch / merge | 16 / 2 / 2 | same | same | Processor and model must agree. |
| vision out_hidden_size | 3584 default | 2048 | 4096 | Matches text hidden in checkpoint. |
| DeepStack indexes | `[8,16,24]` | same | same | Three early decoder injections. |
| dtype | source unspecified/default | `bfloat16` in text config | `bfloat16` | Config-derived. |
| cache support | true | true | true | DynamicCache if `use_cache` and no cache supplied. |

Representative checkpoint sweep:

| Repo | Operator-significant variation |
| --- | --- |
| `Qwen/Qwen3-VL-30B-A3B-Instruct` | 48-layer text MoE, hidden 2048, 32 Q heads, 4 KV heads, top-8 of 128 experts, bf16. |
| `Qwen/Qwen3-VL-30B-A3B-Thinking` | Same neural shape as Instruct; generation/chat policy differs outside graph. |
| `Qwen/Qwen3-VL-235B-A22B-Instruct` | 94-layer text MoE, hidden 4096, 64 Q heads, 4 KV heads, top-8 of 128 experts, larger expert width. |
| `Qwen/Qwen3-VL-235B-A22B-Thinking` | Same neural shape as 235B Instruct; generation policy differs outside graph. |
| `Qwen/Qwen3-VL-30B-A3B-Instruct-FP8` | Same 30B topology plus `quantization_config`: FP8 e4m3, dynamic activation scheme, block size `[128,128]`, many ignored dense layers. |
| `tiny-random/qwen3-vl-moe` | Tiny structural test config, not performance representative; useful only for parser and ABI smoke tests. |

## 3a. Family variation traps

- Production checkpoints use GQA (`num_key_value_heads=4`) even though source defaults set KV heads equal to attention heads.
- Production configs set `num_experts=128` and `top_k=8`; source defaults are 60 and 4.
- Every production decoder layer is MoE because `decoder_sparse_step=1` and `mlp_only_layers=[]`.
- `head_dim` is explicit and should not be inferred from hidden size alone.
- Checkpoint config stores `rope_scaling` plus `rope_theta`; modeling uses normalized `config.rope_parameters`. Loader/config normalization must be preserved.
- Vision config `out_hidden_size` must match text hidden; source default 3584 is not representative of the inspected official checkpoints.
- FP8 checkpoint advertises quantization metadata and ignored layers. This is a weight-loading/provider contract, not a normal dtype-only graph change.
- Processor expands each image/video placeholder into many repeated placeholder tokens. The model only validates counts, not contiguity or processor provenance.
- Video prompt expansion inserts timestamp text plus a `<|vision_start|>...<|vision_end|>` span per temporal grid row/frame.
- `mm_token_type_ids` is required for correct multimodal M-RoPE when grids are passed; missing it raises.
- Vision tensors are already packed rows, not NCHW images, by the model boundary. NHWC should be considered only inside the processor-to-patch or Conv3d-to-linear rewrite region.
- Vision packed attention may use FlashAttention varlen `cu_seqlens`; non-flash fallback splits and loops per packed segment.

## 4. Operator coverage checklist

Tensor/layout ops:

- Shape/view: `view`, `reshape`, `transpose`, `permute`, `contiguous`, `flatten`, `chunk`, `split`, `cat`, `stack`, `expand`, `repeat`, `repeat_interleave`, `cumsum`, `pad`, `roll`, boolean masks.
- Indexed/scatter: `where`, `nonzero`, `one_hot`, `topk`, `searchsorted`, `index_add_`, masked assignment, `masked_scatter`, boolean read/write, `torch.split` by runtime lengths.
- Runtime validation: count/numel equality checks for placeholder tokens and visual features.

Neural primitives:

- Embedding lookup for text tokens and vision absolute position table.
- Linear/GEMM, bias and no-bias.
- Conv3d patch embedding with `kernel=stride=(temporal_patch_size, patch_size, patch_size)`.
- RMSNorm for text hidden and Q/K head-dim norms.
- LayerNorm for vision blocks and merger.
- SiLU-gated expert MLP: `silu(gate) * up -> down`.
- GELU tanh approximation for vision MLP, plain GELU in merger.
- Residual adds.
- Final LM head `Linear(hidden -> vocab_size)`, no bias, optionally `logits_to_keep`.

Attention primitives:

- Text causal self-attention, GQA/MQA-style KV repeat, Q/K RMSNorm before RoPE, RoPE before cache update.
- Vision noncausal self-attention over packed image/video segments; varlen FlashAttention path consumes `cu_seqlens`.
- Eager attention fallback: matmul, add mask, fp32 softmax, dropout disabled in inference, matmul V.

Position/rotary/custom math:

- Text M-RoPE with 3 axes (temporal, height, width), interleaved section rewrite `[24,20,20]`.
- Extra text-position channel for causal mask: model passes `[4,B,S]` position ids, then uses channel 0 for text mask and channels 1: for M-RoPE.
- Vision rotary from row/column coordinates and absolute pos embedding interpolation over a 2D learned table.
- `rope_deltas` cache for decode positions after multimodal prefill.

Preprocessing-coupled ops:

- Image/video resize/rescale/normalize in channels-first processor.
- Patch packing to row tensors shaped `[sum_tokens_before_merge, C * temporal_patch_size * patch_size * patch_size]`.
- Grid metadata `image_grid_thw` / `video_grid_thw` as `[num_items,3]`.
- `mm_token_type_ids` values: text 0, image 1, video 2, inferred by processor from special-token spans.
- Video frame sampling and timestamp text generation.

Scatter/indexed update ops for multimodal embedding stitch:

- `inputs_embeds.masked_scatter(image_mask, image_embeds)` and same for video.
- DeepStack additive stitch: clone hidden states, gather visual positions, add visual embeddings into early decoder layers.
- Mixed image+video DeepStack joins create a temporary visual-order buffer and assign image/video deepstack features by masks.

Quantized/packed weight metadata ops:

- Official FP8 config uses `quant_method=fp8`, `fmt=e4m3`, dynamic activation scheme, `weight_block_size=[128,128]`, with many ignored visual/router/lm_head layers. DinoML should reject or route this checkpoint until an explicit FP8 provider/loading contract exists.

Distributed/tensor-parallel metadata:

- Config source advertises TP/EP plans: Q/K/V colwise, O rowwise, MoE router/expert grouped GEMM. DinoML single-GPU first integration can ignore distributed execution but should preserve logical packed expert tensors.

## 5. Layer/block breakdown

Vision processor to patch embedding:

```text
image/video frames -> channels-first resize/rescale/normalize
  -> pack rows in grid order
  -> pixel_values row width = 3 * temporal_patch_size * patch_size * patch_size
  -> view rows as [-1, 3, Tpatch, P, P]
  -> Conv3d(3 -> vision_hidden, kernel=stride=(Tpatch,P,P), bias=True)
  -> [total_patch_tokens, vision_hidden]
```

Vision block, repeated 27 times:

```text
x = x + VisionAttention(LayerNorm(x), cu_seqlens, vision_rope)
x = x + Linear(gelu_tanh(Linear(LayerNorm(x))))
```

Vision attention:

```text
qkv = Linear(vision_hidden -> 3 * vision_hidden, bias=True)
q,k,v = reshape [seq, 3, heads, head_dim]
q,k = vision_rope(q,k)
attn = noncausal attention per packed image/video segment
x = Linear(attn, vision_hidden -> vision_hidden, bias=True)
```

Vision merger:

```text
x = LayerNorm(x)                  # before or after spatial merge reshape depending merger
x = view[-1, vision_hidden * merge_size^2]
x = Linear -> GELU -> Linear(out_hidden_size)
```

Text decoder block, repeated N times:

```text
res = x
x = RMSNorm(x)
q = RMSNorm_head(Linear(hidden -> n_q_heads * head_dim, bias=False).view(...))
k = RMSNorm_head(Linear(hidden -> n_kv_heads * head_dim, bias=False).view(...))
v = Linear(hidden -> n_kv_heads * head_dim, bias=False).view(...)
q,k = M-RoPE(q,k)
k,v = cache.update(k,v, layer)
x = causal GQA attention(q,k,v, mask)
x = res + Linear(n_q_heads * head_dim -> hidden, bias=False)

res = x
x = RMSNorm(x)
x = MoE(x)  # production checkpoints
x = res + x
```

MoE block:

```text
h = x.reshape(B*S, hidden)
router_logits = Linear(h, router_weight [num_experts, hidden])
router_probs = softmax(router_logits, fp32)
scores, experts = topk(router_probs, top_k)
scores = scores / scores.sum(-1, keepdim=True)
for hit expert:
  gate, up = Linear(h_for_expert, gate_up_proj[expert]).chunk(2, -1)
  y = Linear(silu(gate) * up, down_proj[expert])
  output.index_add_(token_idx, y * score)
```

Production dimensions:

- 30B: text `hidden=2048`, Q width `32*128=4096`, KV width `4*128=512`, expert gate/up packed `128 x 1536 x 2048`, expert down `128 x 2048 x 768`.
- 235B: text `hidden=4096`, Q width `64*128=8192`, KV width `4*128=512`, expert gate/up packed `128 x 3072 x 4096`, expert down `128 x 4096 x 1536`.

## 6. Attention requirements

Text attention:

- Causal self-attention.
- GQA in production: 30B uses 32 query heads, 4 KV heads, repeat factor 8; 235B uses 64 query heads, 4 KV heads, repeat factor 16.
- Head dim 128. Query output width is `num_attention_heads * head_dim`; K/V output width is `num_key_value_heads * head_dim`.
- Q and K get RMSNorm over `head_dim` after projection and before RoPE.
- M-RoPE is applied before cache update. Cached keys are stored after RoPE.
- Mask comes from `create_causal_mask`, using text position ids from channel 0 of `[4,B,S]` when supplied.
- Eager math order: repeat KV, QK matmul, multiply by `head_dim ** -0.5`, add mask, fp32 softmax, cast to query dtype, dropout, matmul V.
- Source supports FlashAttention, SDPA, and FlexAttention through `ALL_ATTENTION_FUNCTIONS`. DinoML can start with dense causal attention and add a GQA cache-aware FlashAttention provider later.

Vision attention:

- Noncausal self-attention over packed image/video sequences.
- QKV is one fused linear with bias, split order `q,k,v` from reshape `[seq,3,heads,head_dim]`.
- FlashAttention path uses `cu_seq_lens_q=cu_seqlens`, `cu_seq_lens_k=cu_seqlens`, max lengths from adjacent differences.
- Non-flash path splits Q/K/V per packed segment and concatenates outputs.
- No KV cache in vision branch; vision outputs are independently cacheable across decode after prefill.

Cache/generation:

- If `use_cache` and no cache is supplied, text model creates `DynamicCache`.
- `prepare_inputs_for_generation` removes `pixel_values` and `pixel_values_videos` after the first cached iteration.
- `_prepare_position_ids_for_generation` returns `[4,B,S]` positions for first multimodal prefill and `[3,B,S]` delta-adjusted positions for decode continuation when `past_length != 0`.
- Beam expansion has custom logic for packed visual tensors because pixel rows and grid rows are not batch-major tensors.

## 7. Position encoding and custom math

Text M-RoPE:

```python
def text_mrope(inv_freq, position_ids_3, mrope_section=(24, 20, 20)):
    # position_ids_3: [3, batch, seq] for temporal, height, width
    freqs = matmul(inv_freq[None, None, :, None], position_ids_3[:, :, None, :]).transpose(2, 3)
    freqs_t = freqs[0].clone()
    for dim, offset in ((1, 1), (2, 2)):
        length = mrope_section[dim] * 3
        freqs_t[..., offset:length:3] = freqs[dim, ..., offset:length:3]
    emb = cat([freqs_t, freqs_t], dim=-1)
    return cos(emb), sin(emb)
```

`get_rope_index` groups each sequence by contiguous `mm_token_type_ids`. Text runs consume monotonic positions. Image/video runs consume `get_vision_position_ids`, with H/W divided by `spatial_merge_size`; current position advances by `max(grid_h, grid_w) // spatial_merge_size`, not by visual token count. For videos, `video_grid_thw` is repeated by temporal grid count and converted to per-frame grids with `T=1` because timestamp text separates frames.

Vision position math:

- `rot_pos_emb(grid_thw)` builds row/column coordinate pairs in spatial-merge block order, repeats per temporal grid, looks up a rotary frequency table up to `max(H,W)`, flattens row/column rotary embeddings, then duplicates into cos/sin.
- `fast_pos_embed_interpolate(grid_thw)` bilinearly interpolates a learned absolute 2D table of size `sqrt(num_position_embeddings)^2`; default/checkpoint table side is 48. It repeats across temporal grid and permutes into the same merge-block token order.

Precompute opportunities:

- Text `inv_freq` and vision rotary frequency table are static by config/dtype/device.
- Vision absolute interpolation indices/weights depend on `grid_thw` and can be cached by `(T,H,W)`.
- Full M-RoPE `position_ids` depends on prompt tokenization, `mm_token_type_ids`, grids, attention mask, and cached `rope_deltas`.

## 8. Preprocessing and input packing

Image processor ABI:

- Official preprocessor config names `Qwen2VLImageProcessorFast`, with `patch_size=16`, `temporal_patch_size=2`, `merge_size=2`, mean/std `[0.5,0.5,0.5]`.
- Images are channels-first at processor output and packed into rows, not passed as NCHW images to modeling.
- Output:
  - `pixel_values`: `[sum_i grid_h_i * grid_w_i, 3 * 2 * 16 * 16]`.
  - `image_grid_thw`: `[num_images, 3]`, rows `[1, grid_h, grid_w]`.
- Placeholder count per image: `prod(image_grid_thw[i]) // merge_size**2`.

Video processor ABI:

- Qwen3VL video processor defaults: `fps=2`, `min_frames=4`, `max_frames=768`, temporal patch size 2, merge size 2.
- Processor samples frame indices, resizes by temporal/spatial pixel budget, pads frames to temporal-patch divisibility by repeating the last frame, packs rows.
- Output:
  - `pixel_values_videos`: `[sum_v grid_t_v * grid_h_v * grid_w_v, 3 * 2 * 16 * 16]`.
  - `video_grid_thw`: `[num_videos, 3]`, rows `[grid_t, grid_h, grid_w]`.
- Qwen3VL processor expands one video placeholder into timestamp text plus one vision span per `grid_t` temporal unit. The model then repeats `video_grid_thw` by `grid_t` for M-RoPE.

Placeholder and modality ids:

- Token ids from config: `vision_start=151652`, `vision_end=151653`, `image=151655`, `video=151656`.
- Processor defaults return `mm_token_type_ids=True`; values are text 0, image 1, video 2.
- The model validates placeholder count by comparing `inputs_embeds[mask].numel()` to feature `numel()`.
- Source uses broad `masked_scatter`; under processor-generated prompts this can be lowered to ordered indexed row copy with guards:
  - all image/video masks select whole hidden rows,
  - selected row count equals feature rows,
  - mask order is row-major token order,
  - feature order is concatenated image/video grid order from processor.
- Mixed image+video DeepStack stitch also needs order guards for `visual_pos_masks`, `image_mask_joint`, and `video_mask_joint`.

## 9. Graph rewrite / lowering opportunities

### Rewrite: packed Conv3d patch embed -> Linear

Source pattern:

```text
pixel_values rows -> view[-1, C, Tpatch, P, P] -> Conv3d(C,out,kernel=(Tpatch,P,P),stride=same)
```

Replacement:

```text
Linear(row_width -> vision_hidden, bias=True)
```

Preconditions:

- Input comes from Qwen processor packed rows with vector order matching the model `view(-1,C,T,P,P)`.
- `kernel_size == stride == (temporal_patch_size, patch_size, patch_size)`.
- `groups == 1`, `dilation == 1`, `padding == 0`.
- Row width equals `C*Tpatch*P*P`; dtype conversion to conv weight dtype is preserved.

Weight transform:

```python
w_linear = conv.weight.reshape(out_channels, in_channels * temporal_patch_size * patch_size * patch_size)
b_linear = conv.bias
```

Failure cases: raw NCHW/NCTHW images at model boundary, changed processor packing order, non-default patch geometry, grouped/dilated/padded conv.

Parity test: compare patch_embed output for random packed rows and real processor rows.

### Rewrite: placeholder masked_scatter -> indexed row copy

Source pattern:

```text
mask = input_ids == image_token_id or video_token_id
inputs_embeds = inputs_embeds.masked_scatter(mask[...,None].expand_as(inputs_embeds), features)
```

Replacement:

```text
row_indices = nonzero(mask)
inputs_embeds[row_indices, :] = features.reshape(-1, hidden)
```

Preconditions:

- Mask is row-complete over hidden dim.
- `mask.sum() == features.shape[0]`.
- Feature dtype cast to `inputs_embeds.dtype` is applied.
- Processor-generated placeholder order is accepted; arbitrary user masks should fall back or reject.

Failure cases: `inputs_embeds` mode where special-token embeddings are compared by value, duplicate/equal embeddings causing ambiguous masks, count mismatch.

### Rewrite: MoE fallback loop -> grouped expert GEMM

Source pattern:

```text
topk router -> per-expert token gather -> gate_up GEMM -> silu*up -> down GEMM -> weighted index_add
```

Replacement:

```text
router topk -> token/expert bucketing -> grouped GEMM gate_up -> fused SwiGLU -> grouped GEMM down -> weighted scatter-add
```

Preconditions:

- `top_k=8`, `num_experts=128` for production configs.
- Expert weights stored as `[E, 2*moe_intermediate, hidden]` and `[E, hidden, moe_intermediate]`.
- Top-k probabilities are renormalized after topk.
- Deterministic topk tie behavior is either matched or guarded as numerically irrelevant for production.

Failure cases: `mlp_only_layers` non-empty, `decoder_sparse_step` not 1, source/hub optimized expert implementation with different ABI.

### Rewrite: vision packed varlen attention -> segmented attention provider

Source pattern: concatenated patch sequence plus `cu_seqlens`.

Replacement: FlashAttention varlen or segmented dense attention.

Preconditions:

- `cu_seqlens = pad(cumsum(repeat_interleave(grid_h*grid_w, grid_t)), left=0)`.
- Attention is noncausal and has no mask.
- Segment order matches processor packed order.

Failure cases: non-flash fallback split loop must remain available for unsupported dtypes/shapes.

### Layout guard: channels-first processor and packed rows

Initial translation should preserve source axes. NHWC/channel-last optimization is only safe inside controlled regions:

- Processor resize/normalize may use NHWC internally only if final packed row order matches source.
- Patch embed linear rewrite is layout-independent at model boundary if row vector order is preserved.
- Vision transformer works on `[seq, hidden]`; no NHWC translation should cross into token sequence code.

## 10. Kernel fusion candidates

Highest priority:

- RMSNorm and head-dim Q/K RMSNorm. These appear in every text block and affect attention parity.
- GQA FlashAttention with RoPE-applied cached K/V. Prefill and decode bottleneck, especially 262k context.
- MoE routing and grouped expert GEMM. Production checkpoints are all-MoE and top-8 of 128 experts.
- Placeholder row-copy stitch. Avoid admitting general boolean scatter to run Qwen3-VL.
- Packed Conv3d-to-linear patch embed. Simple, high-confidence rewrite.

Medium priority:

- Vision varlen noncausal attention from `cu_seqlens`.
- Vision absolute position interpolation cache by grid shape.
- Vision merger/deepstack merger Linear-GELU-Linear fusion.
- Last-token-only logits via `logits_to_keep`.
- M-RoPE cos/sin generation fused or cached for common grids.

Lower priority:

- Full beam expansion for packed visual tensors.
- FP8 e4m3 blocked weight provider.
- Auxiliary router loss and router logits capture; training/eval diagnostics, not first inference target.

## 11. Runtime staging plan

Stage 1: config and ABI admission.

- Parse qwen3_vl_moe config, normalize `rope_scaling`/`rope_theta` into the fields used by modeling.
- Reject unsupported FP8 quantization, non-default processor patch/merge values, or missing `mm_token_type_ids` when multimodal grids are present.

Stage 2: vision encoder parity.

- Implement packed patch-row input and Conv3d-to-linear rewrite.
- Run vision block and merger parity on tiny/random and one production-shaped synthetic grid.
- Cache vision outputs as prefill-only artifacts.

Stage 3: placeholder stitch and M-RoPE.

- Lower masked_scatter to guarded indexed copy.
- Implement `get_rope_index`, `get_vision_position_ids`, `rope_deltas`, and text M-RoPE.
- Validate image-only, video-only, and mixed image+video prefix construction.

Stage 4: text decoder dense skeleton.

- Implement one block with GQA attention, Q/K RMSNorm, cache update, final RMSNorm, logits.
- Stub MoE with dense reference or call out unsupported until grouped expert path lands.

Stage 5: MoE production path.

- Add topk router, token bucketing, grouped GEMMs, weighted scatter-add.
- Validate all-MoE 30B/235B layer shapes.

Stage 6: decode.

- Drop visual tensors after first cached step.
- Use cached `rope_deltas` for positions.
- Validate one-token decode against Transformers with cached K/V.

Stage 7: performance.

- Add GQA FlashAttention, varlen vision attention, grouped expert scheduling, last-token logits, optional FP8/GGUF weight-loading paths.

## 12. Parity and validation plan

- Processor ABI snapshots: for fixed image sizes and video frame counts, assert `pixel_values` row counts, row width 1536, `*_grid_thw`, placeholder counts, and `mm_token_type_ids`.
- Patch embed parity: random packed rows through Conv3d source vs transformed Linear.
- Vision position parity: `fast_pos_embed_interpolate`, `rot_pos_emb`, and `cu_seqlens` for several grids including non-square H/W.
- Placeholder stitch parity: image-only, video-only, mixed; include count mismatch rejection.
- M-RoPE parity: compare `get_rope_index`, `compute_3d_position_ids`, and `rope_deltas` for padded/unpadded prompts.
- One vision block parity in fp32/bf16 tolerances.
- One text block parity with cache disabled and enabled.
- MoE parity: router logits/topk/scores, expert grouped output, tie-sensitive deterministic cases.
- Prefill logits parity with tiny-random checkpoint.
- Decode parity: first decode token with cache and no pixel inputs.

Suggested tolerances: fp32 custom math `1e-5` absolute where possible; bf16/fp16 block/logit comparisons should use wider relative tolerances and layer-by-layer drift checks. For top-k routing, compare selected expert ids exactly on deterministic inputs before numeric output comparisons.

## 13. Performance probes

- Processor throughput: images/sec and video frames/sec, split resize/normalize/packing/tokenization.
- Vision encoder throughput by total visual tokens and number of packed segments.
- Vision varlen attention backend comparison: split eager vs SDPA/FlashAttention-style.
- Prefill throughput by text length, visual token count, and batch size.
- Decode tokens/sec by batch, cache length, and with/without `logits_to_keep=1`.
- KV cache memory: layers * KV heads * sequence * head_dim for 30B and 235B.
- MoE routing time, token bucketing time, grouped GEMM time, scatter-add time.
- Expert load balance distribution under real prompts.
- FP8/dense/GGUF load and dequant probes once provider contracts exist.
- Grid-shape cache hit rate for vision position interpolation.

## 14. Skip/defer list

Safe to defer for first inference target:

- Training, labels loss, router auxiliary loss, gradient checkpointing.
- Router logits capture except for MoE parity debugging.
- Beam search packed-visual expansion, after greedy/sampling decode works.
- FP8 checkpoint execution until blocked-FP8 provider/loading is designed.
- Tensor/expert parallel execution plans.
- General boolean scatter; use guarded row-copy for processor-generated placeholders.
- Arbitrary `inputs_embeds` multimodal mode based on embedding-value comparison.
- Non-default processor geometry, if not in inspected official configs.
- Full video decode ownership; first integration can accept processor-produced frames/metadata or prepacked tensors.

## 15. Final implementation checklist

- [ ] Parse `Qwen3VLMoeConfig` and normalize checkpoint `rope_scaling`/`rope_theta`.
- [ ] Admit official dense bf16 30B/235B configs; reject FP8 until provider contract exists.
- [ ] Load tied/untied embedding and LM head weights with top-level untied behavior.
- [ ] Implement packed image/video tensor ABI: `pixel_values`, `pixel_values_videos`, `image_grid_thw`, `video_grid_thw`, `mm_token_type_ids`.
- [ ] Add Conv3d patch embed to Linear rewrite with packing-order tests.
- [ ] Implement vision absolute position interpolation and vision rotary position generation.
- [ ] Implement packed vision noncausal attention with `cu_seqlens` or segmented fallback.
- [ ] Implement vision merger and DeepStack merger outputs.
- [ ] Lower placeholder `masked_scatter` to guarded indexed row copy.
- [ ] Implement DeepStack additive hidden-state stitch.
- [ ] Implement multimodal `get_rope_index`, `get_vision_position_ids`, `rope_deltas`, and M-RoPE.
- [ ] Implement text Q/K RMSNorm, GQA attention, RoPE-before-cache, and DynamicCache ABI.
- [ ] Implement MoE router top-k normalization and grouped expert GEMMs.
- [ ] Implement weighted expert scatter-add.
- [ ] Implement last-token/logits-to-keep LM head path.
- [ ] Add one-block, vision-encoder, prefill-logit, and decode-token parity tests.
- [ ] Benchmark processor, vision, prefill, decode, MoE routing/GEMM, and KV memory.

## Gated DinoML gaps

- General MoE operator/provider maturity: top-k router, token bucketing, grouped expert GEMM, weighted scatter-add.
- GQA cache-aware attention with Q/K head RMSNorm and RoPE-before-cache.
- M-RoPE position ABI: `[4,B,S]` prefill positions, `[3,B,S]` decoder continuation, `rope_deltas` state.
- Processor/grid ABI: packed row tensors and `mm_token_type_ids`; missing or inconsistent metadata must reject.
- Placeholder stitch: needs guarded indexed row-copy lowering rather than broad masked scatter.
- Vision packed varlen attention and `cu_seqlens`.
- Conv3d patch embed rewrite requires strict packed-row order guards.
- NHWC/channel-last optimization must stay inside processor/patch regions; token-sequence regions are layout-neutral and should be protected from image-axis rewrites.
- FP8 checkpoints require explicit blocked-FP8 loading/provider support or rejection.
