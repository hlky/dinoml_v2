# Transformers Audit: glm4v_moe

## 1. Source basis

Transformers commit/version: local `X:/H/transformers` at `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`; HF configs report `transformers_version: 4.57.1`.

Model id: primary `zai-org/GLM-4.5V`; quantized variant `zai-org/GLM-4.5V-FP8`; comparison/out-of-scope shared vision family `zai-org/GLM-4.1V-9B-Thinking`.

Config source: public HF `config.json`, `generation_config.json`, and `preprocessor_config.json` snapshots saved under `_sources/`.

Source files inspected:

- `src/transformers/models/glm4v_moe/modular_glm4v_moe.py`
- `src/transformers/models/glm4v_moe/modeling_glm4v_moe.py`
- `src/transformers/models/glm4v_moe/configuration_glm4v_moe.py`
- Shared GLM-4V processor sources: `processing_glm4v.py`, `image_processing_glm4v.py`, `video_processing_glm4v.py`

Any missing files or assumptions: `processor_config.json` for `zai-org/GLM-4.5V` returned 404; `preprocessor_config.json` names `Glm4vProcessor`. No code tests or imports were run. `modeling_glm4v_moe.py` and `configuration_glm4v_moe.py` are generated from `modular_glm4v_moe.py`; use modular source for future upstream edits.

## 2. High-level architecture

Primary runtime target: multimodal causal generation with image/video prefix processing, text prefill, and autoregressive decode.

Architecture:

```text
CPU/data pipeline image/video resize+patch packing
  -> vision transformer over packed patch rows
  -> spatial merge/project to text hidden width
  -> replace image placeholder token embeddings
  -> MoE text decoder prefill with M-RoPE and KV cache
  -> decode with cached KV and cached rope_deltas
  -> logits/sampling
```

Stage decomposition:

- CPU/data pipeline: image/video decode, frame sampling, resize, normalize, patch flattening, placeholder expansion, `mm_token_type_ids`.
- Vision encoder/projector: independently cacheable per image/video input until text prompt changes. It consumes packed patch rows plus `grid_thw`.
- Prefix construction: token embeddings are overwritten at image-token positions using `masked_scatter`; this can be lowered to ordered indexed row copy under processor guards.
- Text prefill: causal GQA decoder with 3-axis multimodal RoPE, routed/shared MoE MLPs, and `DynamicCache`.
- Decode: image/video tensors are dropped after first generation iteration; `rope_deltas` and KV cache drive text-only incremental steps.

## 3. Important config dimensions

Primary GLM-4.5V dimensions from HF `config.json`:

| Field | Value |
|---|---:|
| text hidden size | 4096 |
| text layers | 46 |
| attention heads / KV heads | 96 / 8 |
| head_dim | 128 |
| attention projection widths | Q 12288, K 1024, V 1024, O 12288 -> 4096 |
| dense MLP intermediate | 10944 |
| MoE intermediate per expert | 1408 |
| routed experts / experts per token | 128 / 8 |
| shared experts | 1 |
| dense layers before MoE | 1 |
| vocab size | 151552 |
| max positions | 65536 |
| RoPE | default theta 10000, partial_rotary_factor 0.5, mrope_section [8, 12, 12] |
| dtype | bfloat16 |
| vision hidden / heads / layers | 1536 / 12 / 24 |
| vision patch / temporal patch / spatial merge | 14 / 2 / 2 |
| vision out hidden | 4096 |
| image/video placeholder IDs | 151363 / 151364 in GLM-4.5V config |
| cache | `use_cache: true`, `DynamicCache` by default |

Representative checkpoint sweep:

| Model | Scope | Text body | Vision body | Operator-significant difference |
|---|---|---|---|---|
| `zai-org/GLM-4.5V` | primary | `glm4v_moe`, 46 layers, 128 routed experts, top-8 | 24-layer GLM-4V vision | Dense first layer, MoE after layer 0; bf16 dense weights |
| `zai-org/GLM-4.5V-FP8` | loading follow-up | same config dimensions | same config dimensions | Adds `compressed-tensors` FP8 metadata for Linear weights/activations and a long ignore list |
| `zai-org/GLM-4.1V-9B-Thinking` | out of scope for MoE | `glm4v`, 40 dense layers, 32 heads, 2 KV heads | same patch/merge topology | Shares processor/vision ABI but lacks MoE routing and uses different image/video placeholder IDs |

## 3a. Family variation traps

- `hidden_size != num_attention_heads * head_dim`: GLM-4.5V has `4096 != 96 * 128`; Q projection expands to 12288 and O projects back to 4096.
- GQA is extreme: 96 query heads and 8 KV heads, with repeat factor 12 in eager attention.
- `partial_rotary_factor=0.5`; only the first 64 dims of each 128-dim head are rotary, and M-RoPE splits those rotary frequencies across temporal/height/width sections.
- Video and image placeholders both use the image placeholder token in source masks when `input_ids` are present; modality disambiguation comes from `mm_token_type_ids` and begin/end video tags.
- `position_ids` can be `[4, batch, seq]`: text positions are prepended for packed mask construction, while the last three planes feed M-RoPE.
- Vision input to the model is already flattened patch rows, not raw NCHW images. Raw image/video processing is CPU/data-pipeline work unless DinoML explicitly owns that processor region.
- Vision attention is packed variable-length noncausal self-attention over images/frames using `cu_seqlens` for FlashAttention.
- First text decoder layer is dense MLP; later layers are MoE. Do not instantiate routed expert state for layer 0.
- FP8 checkpoint metadata is not read as normal dense dtype behavior by `modeling_glm4v_moe.py`; admit it as a separate quantized loading/provider task.

## 4. Operator coverage checklist

Tensor/layout ops:

- `view`, `reshape`, `transpose`, `permute`, `contiguous`, `cat`, `split`, `repeat`, `repeat_interleave`, `expand`, `cumsum`, `arange`, `stack`.
- Boolean masks, `masked_fill`, `masked_scatter` or a bounded indexed-copy replacement.
- `topk`, `gather`, `one_hot`, `where`, `index_add_`, `nonzero` for eager MoE fallback.
- Packed sequence descriptor construction: `cu_seqlens = pad(cumsum(repeat_interleave(grid_h * grid_w, grid_t)))`.

Neural primitives:

- Embedding lookup, tied logical embedding/LM-head identity key is declared but config has `tie_word_embeddings=false`.
- RMSNorm over hidden dim with fp32 variance.
- Linear/GEMM for Q/K/V/O, MLP, expert, router, LM head.
- SiLU gated MLP: `down(silu(gate(x)) * up(x))`.
- Vision Conv3d patch embed with kernel=stride `[2,14,14]`.
- Vision Conv2d downsample with kernel=stride `2`.
- LayerNorm in vision patch merger, GELU, SiLU gated merger.
- Bicubic `grid_sample` for adapted 2D learned vision positional embeddings.

Attention primitives:

- Text causal self-attention, GQA, RoPE-before-cache, optional FlashAttention/SDPA/flex backend.
- Vision noncausal self-attention over packed per-image/per-frame sequences, MHA with packed varlen FlashAttention option.

Position/rotary:

- Text M-RoPE with 3 position planes and `mrope_section`.
- Vision rotary over merged spatial order, from `rot_pos_emb(grid_thw)`.
- Learned vision 2D position table sampled with normalized `[-1,1]` coordinates.

MoE:

- Router Linear `4096 -> 128` in fp32.
- Sigmoid router scores, correction bias, group top-2 scoring, top group selection, masked top-8 experts.
- Routed expert packed weights: `gate_up_proj [128, 2816, 4096]`, `down_proj [128, 4096, 1408]`.
- Shared expert dense SwiGLU with intermediate `1408 * n_shared_experts`.

Generation/cache:

- `DynamicCache` per layer, storing post-RoPE K/V as `[batch, kv_heads, cache_seq, head_dim]`.
- Beam expansion must repeat multimodal side tensors by per-sample visual lengths, not by batch dimension alone.
- `logits_to_keep` last-token logits slicing.

Preprocessing-coupled ABI:

- `pixel_values`: `[sum_images, grid_t * grid_h * grid_w, 3 * temporal_patch_size * patch_size^2]`.
- `pixel_values_videos`: same row format over sampled video clips.
- `image_grid_thw` / `video_grid_thw`: `[num_items, 3]`, values are patch-grid temporal, height, width before spatial merge.
- `mm_token_type_ids`: 0 text, 1 image placeholders, 2 video-frame image placeholders inside video tags.

## 5. Layer/block breakdown

Vision patch and encoder:

```text
processor emits flattened rows [Npatch, 1176]
patch_embed: Conv3d(3 -> 1536, kernel=stride [2,14,14]) over row-restored patches
post_conv RMSNorm(1536)
add sampled learned 2D position embedding
repeat 24x:
  x = x + noncausal packed MHA(RMSNorm(x), vision RoPE, cu_seqlens)
  x = x + SwiGLU(RMSNorm(x), 1536 -> 4096 -> 1536)
post RMSNorm(1536)
reshape groups of 2x2 spatial-merge tokens
Conv2d downsample 1536 -> 4096, kernel=stride 2
patch merger: Linear(4096 -> 4096) + LayerNorm + GELU + SwiGLU(4096 -> 10944 -> 4096)
```

Text decoder, layer 0:

```text
x = x + Attention(RMSNorm(x))
x = x + dense SwiGLU(RMSNorm(x), 4096 -> 10944 -> 4096)
```

Text decoder, layers 1..45:

```text
x = x + Attention(RMSNorm(x))
router_logits = Linear(fp32 x, 4096 -> 128)
topk_indices, topk_weights = grouped sigmoid top-k(router_logits)
routed = sum_experts topk_weight * down_i(silu(gate_i(x)) * up_i(x))
shared = shared_down(silu(shared_gate(x)) * shared_up(x))
x = x + routed + shared
```

Projection bias: text Q/K/V have bias; O has no bias. Vision QKV has no bias by config; vision attention output has no bias. MLPs are bias-free in source for text and most vision paths.

## 6. Attention requirements

Text attention:

- Causal self-attention with GQA: 96 query heads, 8 KV heads, head dim 128, repeat factor 12 for eager attention.
- Q width is 12288; K/V width is 1024 each; output projection is 12288 -> 4096.
- Rotary is applied to Q/K before cache update, so cached keys are post-RoPE.
- Eager path computes `softmax((Q @ K^T) * head_dim^-0.5 + mask)` in fp32 then casts to query dtype.
- `create_causal_mask` may use text-only positions from `[4,b,s]` packed positions when no attention mask is supplied.
- FlashAttention/SDPA/flex are advertised by source. DinoML should first implement an explicit GQA causal path and use backend-specific packed masking only after parity.

Vision attention:

- Noncausal self-attention, 12 heads, head dim 128.
- Packed varlen ABI is `hidden_states [total_patch_tokens, 1536]` plus `cu_seqlens [num_sequences+1]`.
- FlashAttention path passes `cu_seq_lens_q/k`, `max_length_q/k`, and `is_causal=False`.
- Non-Flash path splits Q/K/V into chunks from `cu_seqlens`, attends per chunk, then concatenates.

KV cache:

- Text cache per layer stores K/V after RoPE as `[batch, 8, cache_seq, 128]`.
- At decode, image/video tensors are removed from model inputs when `is_first_iteration` is false and cache is used.
- `rope_deltas` is a model-side state used to offset future text position IDs after multimodal prefill; this is not ordinary KV data and should be part of the generation ABI.

## 7. Position encoding and custom math

Text rotary:

```python
def glm4v_mrope(freqs, mrope_section):
    chunks = freqs.split(mrope_section, dim=-1)
    return cat([chunk[i % 3] for i, chunk in enumerate(chunks)], dim=-1)
```

`freqs` is built from `inv_freq [rotary_dim/2]` and `position_ids [3,b,s]`, producing temporal/height/width frequency planes before section interleave. `partial_rotary_factor=0.5` gives rotary dim 64 for head dim 128.

Vision rotary:

```text
for each grid [t,h,w]:
  produce h/w ids in spatial-merge-major order
  repeat ids over t
  rotary_pos_emb = rotary_table[max(h,w)][[h_ids,w_ids]].flatten(1)
  cos/sin = cat(rotary_pos_emb, rotary_pos_emb).cos/sin
```

Multimodal RoPE index:

- Text spans increment position by text length.
- Image/video spans use 3D position IDs over temporal/height/width grids after spatial merge.
- For videos, `video_grid_thw` is first repeated into per-frame rows `[1,h,w]`; timestamps are represented in prompt text, while the model's 3D positions treat frames as separated grids.
- `rope_delta = max(multimodal_position) + 1 - unmasked_sequence_length` is cached for decode.

Vision learned position interpolation:

- The source uses `F.grid_sample(..., mode="bicubic", align_corners=False, padding_mode="border")` over a learned square 2D table.
- Coordinates are normalized from patch h/w ids with `(coord + 0.5) / target_size * 2 - 1`.

## 8. Preprocessing and input packing

Image processor:

- Resizes each image so H/W are multiples of `patch_size * merge_size = 28`, with min/max pixel guards from preprocessor config.
- Converts to RGB, rescales by `1/255`, normalizes with CLIP mean/std.
- If image tensor is `[B,C,H,W]`, inserts `T=1`; then pads frames to a multiple of temporal patch size by repeating the last frame.
- Emits flattened rows in a spatial-merge-aware order:

```text
[B,T,C,H,W]
-> [B, grid_t, tp, C, gh/merge, merge, ph, gw/merge, merge, pw]
-> permute [B, grid_t, gh/merge, gw/merge, merge_h, merge_w, C, tp, ph, pw]
-> [B, grid_t * gh * gw, C * tp * ph * pw]
```

Video processor:

- Owns frame sampling when metadata is available: default `fps=2`, `max_duration=300`, even frame count by duplicating the last sampled frame when needed.
- Uses the same resize/normalize/flatten layout as images.
- Emits `pixel_values_videos` and `video_grid_thw`.

Processor placeholder ABI:

- For each image placeholder, processor expands `<|image|>` into `prod(image_grid_thw) // merge_size^2` repeated image tokens.
- For each video placeholder, processor expands into per-frame structures: `<|begin_of_image|><|image|><|end_of_image|>{timestamp}` and then expands each frame image token to `prod(video_grid_thw) // merge_size^2 // grid_t` repeated tokens.
- `create_mm_token_type_ids` marks image token IDs inside video begin/end spans as type 2 and standalone image token IDs as type 1.
- Model source checks feature element count equals placeholder mask element count, then uses `masked_scatter`.

Safe DinoML stitch lowering:

- Admit a bounded ordered row-copy only when processor invariants are present: placeholder positions in token order, feature rows concatenated in the same item order, placeholder count equals feature rows, and hidden width equals text hidden size.
- Reject arbitrary `inputs_embeds` placeholder detection for first integration; comparing dense embedding vectors to special token embeddings is fragile and broader than needed.

## 9. Graph rewrite / lowering opportunities

### Rewrite: processor-packed patch rows -> Conv3d patch projection as Linear

Source pattern: processor emits rows shaped `[tokens, C * temporal_patch_size * patch_size * patch_size]`; model immediately reshapes each row to `[C,tp,ph,pw]` and applies Conv3d with kernel=stride equal to the full patch.

Replacement: `Linear(1176 -> 1536)` with weight `conv.weight.reshape(1536, 1176)` plus bias if present.

Preconditions:

- Input is exactly the GLM-4V processor row order.
- `kernel_size == stride == [temporal_patch_size, patch_size, patch_size]`.
- No padding, dilation, or groups.
- Row flatten order matches source `view(-1,C,tp,ph,pw)`.

Failure cases: raw NCHW/NCTHW tensors entering the model, changed temporal/patch sizes, nonstandard processor order.

Parity sketch: compare patch embed output for random packed rows against Conv3d source for several grid sizes.

### Rewrite: placeholder `masked_scatter` -> ordered indexed row copy

Source pattern: `inputs_embeds.masked_scatter(mask.expand_as(inputs_embeds), cat(image_embeds))`.

Replacement: gather placeholder row indices from `input_ids == image_token_id`, then copy feature rows into those embedding rows.

Preconditions:

- `input_ids` path, not arbitrary `inputs_embeds`.
- Processor-generated placeholders and `mm_token_type_ids`.
- Number of placeholder rows equals feature rows.
- Image stitch and video stitch are not both applied to the same placeholder rows in one pass unless source's "image token inside video" convention is explicitly reproduced.

Failure cases: user-supplied embeddings, mismatched feature count, custom prompts with image token IDs outside processor control.

### Rewrite: MoE eager loop -> grouped routed GEMM

Source pattern: router top-k, per-expert token selection, expert SwiGLU, weighted `index_add_`.

Replacement: sort or bucket token-expert pairs by expert, run grouped GEMM for gate/up and down, multiply by top-k weights, scatter-add to token rows.

Preconditions:

- Fixed `n_routed_experts=128`, top-k=8 for primary checkpoint.
- No training aux loss required for first inference target.
- Router correction bias remains fp32.

Failure cases: `output_router_logits` parity mode, non-default grouping fields, quantized experts without materialization support.

### Rewrite: last-token logits

Source pattern: `lm_head(hidden_states[:, slice_indices, :])` with `logits_to_keep`.

Replacement: for decode and prefill sampling, compute only last token or requested logits rows.

Preconditions: generation controller does not need full prompt logits or loss.

## 10. Kernel fusion candidates

Highest priority:

- RMSNorm: appears twice per text layer, twice per vision block, and around vision patching.
- GQA RoPE attention: Q/K/V projection, partial M-RoPE, cache update, and causal attention dominate prefill/decode.
- MoE routing plus grouped expert GEMM: 45 MoE layers with top-8 over 128 experts; eager loops are not viable.
- Placeholder indexed copy: required to avoid admitting general boolean scatter.
- Packed vision attention: varlen noncausal attention over image/frame patch chunks.

Medium priority:

- Processor-packed patch Linear rewrite and vision Conv2d downsample rewrite.
- Vision position interpolation or cached interpolation table per grid shape.
- SwiGLU fusion for dense/shared/expert/vision MLPs.
- Last-token-only logits.

Lower priority:

- Auxiliary router loss and router logits output.
- Beam expansion of visual side tensors.
- FP8 compressed-tensors direct execution; first stage can require dense materialization.

## 11. Runtime staging plan

Stage 1: parse config and load dense bf16 GLM-4.5V weights; reject FP8/compressed-tensors unless a dense materialization path is selected.

Stage 2: implement one text dense layer and one MoE layer parity with fixed random tensors, including partial M-RoPE and GQA cache update.

Stage 3: implement processor-owned image ABI and vision encoder parity for packed rows, first without video.

Stage 4: implement image placeholder row-copy stitch and multimodal prefill logits.

Stage 5: add decode with KV cache and `rope_deltas`; drop pixel tensors after first iteration.

Stage 6: add video ABI: frame sampling in CPU pipeline, video prompt expansion, per-frame `grid_thw` flattening, and video placeholder stitch.

Stage 7: replace eager MoE with grouped/provider-backed routing kernels and add production attention backends.

## 12. Parity and validation plan

- Config parser tests for GLM-4.5V, GLM-4.5V-FP8 rejection/materialization policy, and GLM-4.1V out-of-scope rejection for this family.
- Custom op tests: RMSNorm, partial M-RoPE, `get_vision_position_ids`, `get_rope_index`, placeholder row-copy, grouped top-k router.
- Vision patch tests: packed row Linear rewrite against source Conv3d for image and video rows.
- Single text layer parity: layer 0 dense MLP and layer 1 MoE separately.
- Vision block parity: one packed chunk and multiple chunks with `cu_seqlens`.
- Prefill parity: image-only prompt logits against HF source for bf16 tolerance.
- Decode parity: one-token and multi-token generation with existing KV and `rope_deltas`.
- Suggested tolerances: fp32 custom math `1e-5` absolute/relative where applicable; bf16/fp16 end-to-end compare with `1e-2` class tolerances and token-level exactness where logits margin is large.

## 13. Performance probes

- Processor throughput: resize/normalize/patch flatten for image resolution sweep.
- Video sampling and patch packing throughput by duration/fps.
- Vision encoder throughput by total patch rows and number of packed chunks.
- Prefill tokens/sec split into text-only, image-prefill, video-prefill.
- Decode tokens/sec with cache length sweep.
- KV cache memory: 46 layers * 2 tensors * 8 KV heads * head_dim 128 * sequence length * dtype.
- MoE router distribution and grouped GEMM occupancy by batch/sequence length.
- Placeholder stitch overhead versus indexed row-copy.
- FP8 load/dequant/materialization cost for `GLM-4.5V-FP8` if admitted later.

## 14. Skip/defer list

- Training, gradient checkpointing, aux router loss as a runtime output.
- `output_router_logits` except debug parity mode.
- Arbitrary `inputs_embeds` multimodal placeholder detection.
- General boolean `masked_scatter`.
- Beam search visual expansion beyond a correctness-preserving host-side implementation.
- FP8 compressed-tensors native execution.
- Multi-GPU tensor parallel and pipeline parallel plans.
- Remote/custom processor variants not matching saved GLM-4V processor ABI.

## 15. Final implementation checklist

- [ ] Parse `Glm4vMoeConfig`, including `rope_scaling`/`rope_theta` normalization to text `rope_parameters`.
- [ ] Reject non-`glm4v_moe` configs for this family audit.
- [ ] Load dense bf16 weights and preserve logical weight aliases.
- [ ] Implement GLM partial M-RoPE and `rope_deltas` ABI.
- [ ] Implement GQA causal attention with post-RoPE KV cache.
- [ ] Implement RMSNorm and SiLU gated dense MLP.
- [ ] Implement MoE router top-k and grouped expert GEMM path.
- [ ] Implement packed image/video input ABI with `grid_thw`.
- [ ] Add packed patch Linear rewrite under processor-order guards.
- [ ] Implement vision packed noncausal attention with `cu_seqlens`.
- [ ] Implement placeholder stitch as ordered indexed row copy with count guards.
- [ ] Add prefill parity for image prompts.
- [ ] Add decode parity with cached KV and cached `rope_deltas`.
- [ ] Add performance probes for vision, prefill, decode, MoE, and FP8 materialization.
