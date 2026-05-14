# Transformers Audit: `qwen3_vl`

## 1. Source basis

```text
Transformers commit/version:
  X:/H/transformers @ b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
  Sampled checkpoint configs report transformers_version 4.56.0 or 4.57.0.dev0.

Model id:
  Primary dense source basis: Qwen/Qwen3-VL-4B-Thinking, Qwen/Qwen3-VL-4B-Thinking-FP8,
  Qwen/Qwen3-VL-32B-Thinking-FP8.
  Variation-only MoE configs: Qwen/Qwen3-VL-30B-A3B-Instruct,
  Qwen/Qwen3-VL-235B-A22B-Instruct-FP8.

Config source:
  Local config class plus public Hugging Face raw config/preprocessor/tokenizer/generation
  snapshots saved under agents/plans/transformers/qwen3_vl/_sources/.

Source files inspected:
  X:/H/transformers/src/transformers/models/qwen3_vl/configuration_qwen3_vl.py
  X:/H/transformers/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py
  X:/H/transformers/src/transformers/models/qwen3_vl/processing_qwen3_vl.py
  X:/H/transformers/src/transformers/models/qwen3_vl/video_processing_qwen3_vl.py
  X:/H/transformers/src/transformers/models/qwen3_vl/modular_qwen3_vl.py
  X:/H/transformers/src/transformers/models/qwen2_vl/image_processing_qwen2_vl.py
  X:/H/transformers/src/transformers/models/auto/{image_processing,processing,video_processing,modeling}_auto.py

Any missing files or assumptions:
  No local qwen3_vl image processor exists; AutoImageProcessor maps qwen3_vl to
  Qwen2VLImageProcessor/Qwen2VLImageProcessorPil. processor_config.json was absent
  for sampled public repos. MoE checkpoints use qwen3_vl_moe and are gated behind
  a separate source-family audit.
```

Hugging Face URLs used:

- [Qwen/Qwen3-VL-4B-Thinking](https://huggingface.co/Qwen/Qwen3-VL-4B-Thinking)
- [Qwen/Qwen3-VL-4B-Thinking-FP8](https://huggingface.co/Qwen/Qwen3-VL-4B-Thinking-FP8)
- [Qwen/Qwen3-VL-32B-Thinking-FP8](https://huggingface.co/Qwen/Qwen3-VL-32B-Thinking-FP8)
- [Qwen/Qwen3-VL-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct)
- [Qwen/Qwen3-VL-235B-A22B-Instruct-FP8](https://huggingface.co/Qwen/Qwen3-VL-235B-A22B-Instruct-FP8)

## 2. High-level architecture

Qwen3-VL dense checkpoints are multimodal causal decoders: a vision encoder converts flattened image/video patch tensors into text-hidden-size visual embeddings, those embeddings replace placeholder token embeddings, then a Qwen-style decoder runs causal prefill/decode and emits LM logits.

```text
processor text/image/video packing
  -> flattened patch tensors + image_grid_thw/video_grid_thw + mm_token_type_ids
  -> vision encoder Conv3d patch embed + learned 2D pos interpolation + vision RoPE attention
  -> patch merger + deepstack visual features
  -> placeholder masked_scatter into token embeddings
  -> causal text decoder with 3D M-RoPE, GQA KV cache, deepstack additions
  -> last-token or sliced logits -> sampling
```

Stage decomposition:

- CPU/data pipeline: image/video resize, rescale, normalize, patch flattening, token placeholder expansion, video timestamp prompt construction, `mm_token_type_ids`.
- Independently cacheable vision work: `pixel_values`/`pixel_values_videos` plus `*_grid_thw` through the vision model into final visual embeddings and three deepstack feature tensors.
- Prefix construction: concatenate text embeddings with visual embeddings by placeholder scatter; compute 3D position ids and `rope_deltas`.
- Prefill: text decoder over full multimodal sequence with causal mask and cache allocation.
- Decode: text-only token steps; image/video tensors are dropped after the first iteration when `use_cache=True`; positions use cached `rope_deltas`.

## 3. Important config dimensions

Dense source defaults in `configuration_qwen3_vl.py`:

| Field | Default |
|---|---:|
| text hidden_size | 4096 |
| text layers | 32 |
| text attention heads / KV heads | 32 / 32 |
| head_dim | 128 |
| text intermediate_size | 22016 |
| vocab_size | 151936 |
| max_position_embeddings | 128000 |
| rope theta default | 500000.0 |
| attention_bias / dropout | false / 0.0 |
| activation | text `silu`, vision `gelu_pytorch_tanh` |
| cache | `use_cache=True`, Transformers `Cache`/`DynamicCache` |
| vision hidden/depth/heads | 1152 / 27 / 16 |
| vision intermediate_size | 4304 |
| vision patch/temporal/merge | 16 / 2 / 2 |
| vision output hidden | 3584 |
| deepstack indexes | [8, 16, 24] |
| placeholder ids | image 151655, video 151656, start 151652, end 151653 |

Representative checkpoint sweep:

| Model | Body | Text hidden | Layers | Heads/KV | FFN/MoE | Max pos | Vision hidden/depth/out | Tie embeddings | Quant |
|---|---|---:|---:|---:|---|---:|---|---|---|
| Qwen3-VL-4B-Thinking | dense | 2560 | 36 | 32/8 | 9728 dense | 262144 | 1024/24/2560 | true | none |
| Qwen3-VL-4B-Thinking-FP8 | dense | 2560 | 36 | 32/8 | 9728 dense | 262144 | 1024/24/2560 | top false, text true | fp8 e4m3 dynamic |
| Qwen3-VL-32B-Thinking-FP8 | dense | 5120 | 64 | 64/8 | 25600 dense | 262144 | 1152/27/5120 | false | fp8 e4m3 dynamic |
| Qwen3-VL-30B-A3B-Instruct | MoE, separate source | 2048 | 48 | 32/4 | 128 experts, top-8, 768 expert FFN | 262144 | 1152/27/2048 | false | none |
| Qwen3-VL-235B-A22B-Instruct-FP8 | MoE, separate source | 4096 | 94 | 64/4 | 128 experts, top-8, 1536 expert FFN | 262144 | 1152/27/4096 | false | fp8 e4m3 dynamic |

Processor snapshots use `Qwen2VLImageProcessorFast` for images and `Qwen3VLVideoProcessor` for videos, both with patch size 16, temporal patch size 2, merge size 2, mean/std `[0.5, 0.5, 0.5]`. Sampled image size limits are `shortest_edge=65536`, `longest_edge=16777216`; video limits are `shortest_edge=4096`, `longest_edge=25165824`.

## 3a. Family variation traps

- `model_type=qwen3_vl_moe` is not the dense `qwen3_vl` source family. MoE configs need router/expert audit, expert-parallel/provider planning, and separate operator gates.
- Checkpoints use `rope_scaling` with `mrope_interleaved=true`, `mrope_section=[24,20,20]`, and `rope_theta=5000000`, while the local text config class calls this field `rope_parameters` and default theta is lower.
- Dense checkpoints use GQA (`num_key_value_heads < num_attention_heads`) even when source defaults are MHA.
- `head_dim` is explicit; do not infer only from `hidden_size / num_attention_heads`.
- FP8 checkpoints include `quantization_config` and long ignored-layer lists that exclude vision merger/patch/pos/deepstack modules and `lm_head`. Treat this as a loader/provider contract, not just dtype.
- Image preprocessing is source-coupled through Qwen2-VL image processor auto mapping; qwen3-specific audit must still validate Qwen2 patch order.
- Video text placeholders are expanded per temporal frame with timestamp strings and repeated `<|vision_start|>...<|vision_end|>` spans; `video_grid_thw` is later repeated per frame for M-RoPE.
- `mm_token_type_ids` are required when multimodal grids are passed and position ids are not supplied.
- DeepStack injects visual features into early text decoder layers by indexed add at visual token positions. This is more than a one-time placeholder stitch.
- `tie_word_embeddings` may differ between top-level and text config in FP8 snapshots; preserve actual weight aliasing from loaded weights.
- Vision source is packed sequence-first with `cu_seqlens`, not a dense `[B,H,W,C]` ViT batch.

## 4. Operator coverage checklist

Tensor/layout ops:

- `view`/`reshape`/`permute`/`transpose`/`contiguous`, `split`, `cat`, `stack`, `repeat`, `repeat_interleave`, `expand`, `flatten`, `unsqueeze`, `cumsum`, `pad`, `searchsorted`, `diff`, boolean masks.
- Placeholder `masked_scatter` and indexed update/add: `hidden_states[visual_pos_masks, :] += visual_embeds`.
- Gather/embedding lookup for token embeddings, learned 2D vision position table, rotary frequency table indexing.

Neural network primitives:

- Text `Embedding(vocab_size -> hidden)`, RMSNorm over hidden and per-head `head_dim`, dense GEMM linears, SwiGLU `down(silu(gate(x))*up(x))`.
- Vision `Conv3d(in=3,out=vision_hidden,kernel=stride=(2,16,16),bias=True)` over flattened patch records after a source-specific reshape.
- Vision LayerNorm, MLP `Linear(hidden -> intermediate) -> gelu_pytorch_tanh -> Linear(intermediate -> hidden)`.
- Vision merger: reshape groups of `spatial_merge_size^2` patches, optional LayerNorm over merged width for deepstack, `Linear(4*vision_hidden -> 4*vision_hidden) -> GELU -> Linear(... -> text_hidden)`.

Attention primitives:

- Text causal GQA with q/k per-head RMSNorm before RoPE, cache update after RoPE, scale `head_dim ** -0.5`, causal mask from `create_causal_mask`, eager/SDPA/FlashAttention backend interface.
- Vision noncausal packed self-attention with qkv packed as `[q,k,v]` from one `Linear(hidden -> 3*hidden, bias=True)`, variable-length chunks from `cu_seqlens`, and optional FlashAttention varlen path.

Position/rotary ops:

- Text 3D M-RoPE cos/sin from `(3, batch, seq)` position ids, `mrope_section`, and interleaved frequency replacement.
- Vision 2D rotary table per grid plus learned position interpolation from a square learned table.

Generation/cache ops:

- Transformers `DynamicCache`/`Cache` per layer storing post-RoPE keys and raw values with shape `[batch, num_kv_heads, seq, head_dim]`.
- `prepare_inputs_for_generation` drops `pixel_values` and `pixel_values_videos` after first cached iteration.
- Beam/sample expansion must repeat image/video flattened tensors by sample-level grid lengths, not by batch dimension blindly.

Preprocessing-coupled ops:

- Qwen2-VL image smart resize, bicubic resize, RGB conversion, rescale/normalize, patch flatten layout.
- Qwen3-VL video frame sampling, temporal padding by last frame, smart resize with temporal factor, patch flatten layout, timestamp placeholder strings.

Quantized/packed weight metadata ops:

- FP8 configs require `quant_method=fp8`, `fmt=e4m3`, dynamic activation scheme, and ignored-layer admission. Dense fallback should load non-FP8 checkpoints first.

Layout guards:

- Processor emits flattened patch tensors shaped `[sum_items, grid_t*grid_h*grid_w, C*temporal_patch*patch*patch]`, not NHWC images.
- The vision Conv3d source immediately reshapes to `[-1, C, temporal_patch, patch, patch]`; a Conv3d-to-linear rewrite is only legal with the exact flatten order preserved.
- Protect Qwen2 image patch `permute(0,2,5,3,6,1,4,7)` and Qwen3 video patch `permute(0,1,4,7,5,8,3,2,6,9)` from generic NHWC/NCHW layout translation unless all downstream axes are rewritten.

## 5. Layer/block breakdown

Vision preprocessing:

```text
image pixels [N,3,H,W]
  -> resize to H,W divisible by patch_size*merge_size
  -> normalize
  -> flatten_patches [N, grid_h*grid_w, 3*temporal_patch_size*patch_size^2]
  -> image_grid_thw rows [1, grid_h, grid_w]

video pixels [N,T,3,H,W]
  -> sample/pad frames so T divisible by temporal_patch_size
  -> resize H,W divisible by patch_size*merge_size
  -> normalize
  -> flatten_patches [N, grid_t*grid_h*grid_w, 3*temporal_patch_size*patch_size^2]
  -> video_grid_thw rows [grid_t, grid_h, grid_w]
```

Vision encoder, repeated `vision.depth` times:

```text
x = Conv3dPatchEmbed(flattened_patches)          # [total_patches, vision_hidden]
x = x + interpolated_learned_2d_pos(grid_thw)
rotary = vision_rot_pos_emb(grid_thw)
cu_seqlens = cumsum(repeat_interleave(grid_h*grid_w, grid_t))
x = x + Attention(LayerNorm(x), cu_seqlens, rotary)
x = x + VisionMLP(LayerNorm(x))
if layer_idx in deepstack_indexes: save VisionPatchMerger(x, postshuffle_norm=True)
vision_embeds = VisionPatchMerger(x, postshuffle_norm=False)
```

Text decoder block, repeated `num_hidden_layers`:

```text
res = x
x = RMSNorm(x)
q = RMSNorm_head(Linear(hidden -> n_heads*head_dim, bias=attention_bias)(x))
k = RMSNorm_head(Linear(hidden -> n_kv_heads*head_dim, bias=attention_bias)(x))
v = Linear(hidden -> n_kv_heads*head_dim, bias=attention_bias)(x)
q,k = M-RoPE(q,k, position_embeddings)
k,v = cache.update(k,v, layer_idx) if cache enabled
x = res + Linear(n_heads*head_dim -> hidden, bias=attention_bias)(Attention(q,k,v, causal_mask))
res = x
x = res + Linear(intermediate -> hidden)(silu(gate_proj(x)) * up_proj(x))
if early deepstack layer: x[visual_pos_masks] += deepstack_visual_embeds[layer_idx]
```

LM head:

```text
x = final RMSNorm(x)
logits = Linear(hidden -> vocab_size, bias=False)(x[:, slice_indices, :])
```

## 6. Attention requirements

Text attention:

- Causal self-attention, GQA/MHA depending on config.
- Query heads `num_attention_heads`; key/value heads `num_key_value_heads`; head dim explicit.
- q/k/v tensors after transpose are `[batch, heads_or_kv_heads, q_len, head_dim]`.
- KV cache stores encoded keys after M-RoPE and values before any repeat expansion. Repeat to query head count happens in eager attention only; optimized kernels should use native GQA where possible.
- Mask is a causal mask built from attention mask, cache, and text position ids. Multimodal position ids do not make attention noncausal.
- FlashAttention/SDPA compatible through Transformers backend interface; eager fallback repeats KV and materializes attention weights.
- q/k RMSNorm over `head_dim` before RoPE is mandatory.

Vision attention:

- Noncausal self-attention over packed image/video sequences.
- Source uses `cu_seqlens` for varlen FlashAttention. Non-flash path splits q/k/v per packed segment and concatenates outputs.
- qkv projection is a single packed `Linear(hidden -> 3*hidden, bias=True)` split order `[q, k, v]`.
- No KV cache; vision outputs may be cached outside the decoder as independent prefix assets.

Generation:

- Prefill may include multimodal tensors and computes `rope_deltas`.
- Decode is text-only when `use_cache=True`; `position_ids = text_positions + rope_deltas`.
- Beam expansion must split visual tensors by per-sample grid-derived flattened lengths; video uses a `searchsorted` recovery because qwen3 video placeholders are per frame.

## 7. Position encoding and custom math

Text M-RoPE:

```python
def qwen3vl_mrope_cos_sin(inv_freq, position_ids_3, mrope_section, scale=1.0):
    # position_ids_3: [3, batch, seq] for temporal, height, width
    freqs = matmul(inv_freq[None, None, :, None], position_ids_3[:, :, None, :]).transpose(2, 3)
    freqs_t = freqs[0].clone()
    for dim, offset in [(1, 1), (2, 2)]:
        freqs_t[..., offset : mrope_section[dim] * 3 : 3] = freqs[dim, ..., offset : mrope_section[dim] * 3 : 3]
    emb = concat([freqs_t, freqs_t], dim=-1)
    return cos(emb) * scale, sin(emb) * scale
```

The first dimension of `position_ids` can be 4 in the text model: row 0 is text positions for causal mask, rows 1..3 are visual/text M-RoPE positions. If only 2D positions are supplied, source expands them to all axes.

Vision position math:

- Learned 2D absolute table has `num_position_embeddings` entries interpreted as a square grid.
- For each input grid, source linearly maps `h` and `w` indices into that square table and bilinearly blends four embedding rows.
- It then reorders by spatial merge grouping before addition to patch embeddings.
- Vision RoPE builds row/column ids in merge-block order, repeats them for temporal frames, looks up frequency rows, flattens row/column frequencies, and applies ordinary rotate-half RoPE to q/k.

M-RoPE position id construction:

- Text runs receive monotonic positions.
- Image/video runs receive temporal/height/width positions from `get_vision_position_ids`.
- For vision runs, `current_pos` advances by `max(grid_h, grid_w) // spatial_merge_size`, not by token count.
- For videos, `video_grid_thw` is repeated `grid_t` times and each repeated row becomes `[1, grid_h, grid_w]` because timestamps split video frames in the prompt.
- `rope_deltas = max(position_ids) + 1 - unpadded_sequence_length` is cached on the model object.

## 8. Preprocessing and input packing

Processor output ABI:

| Tensor/key | Shape | Owner |
|---|---|---|
| `input_ids` | `[batch, seq]` | tokenizer/chat template |
| `attention_mask` | `[batch, seq]` | tokenizer |
| `mm_token_type_ids` | `[batch, seq]`, text 0/image 1/video 2 | processor, required for multimodal M-RoPE |
| `pixel_values` | `[sum_images, grid_h*grid_w, 3*2*16*16]` | Qwen2-VL image processor |
| `image_grid_thw` | `[num_images, 3]`, rows `[1, grid_h, grid_w]` | image processor |
| `pixel_values_videos` | `[sum_videos, grid_t*grid_h*grid_w, 3*2*16*16]` | Qwen3-VL video processor |
| `video_grid_thw` | `[num_videos, 3]`, rows `[grid_t, grid_h, grid_w]` | video processor |

Placeholder stitch:

- Image text placeholders: each `<|image_pad|>` expands to `prod(image_grid_thw[i]) // merge_size^2` repeated image pad tokens.
- Video placeholders: each `<|video_pad|>` expands into per-frame timestamp text plus `<|vision_start|>` and `frame_seqlen = grid_h*grid_w // merge_size^2` video pad tokens and `<|vision_end|>`.
- Runtime checks require placeholder token count times hidden width to equal visual feature element count.
- The model replaces placeholders with `inputs_embeds.masked_scatter(mask, visual_embeds)`.
- DeepStack then adds visual features at the same visual positions for early decoder layers.

CPU/data pipeline can be stubbed initially by accepting already-tokenized `input_ids`, `mm_token_type_ids`, flattened patch tensors, and grid tensors. End-to-end parity needs the exact Qwen2 image and Qwen3 video processor behavior.

## 9. Graph rewrite / lowering opportunities

### Rewrite: vision Conv3d patch embed -> Linear

Source pattern:

```text
hidden_states.view(-1, C, Tpatch, P, P)
Conv3d(C -> vision_hidden, kernel=stride=(Tpatch,P,P), bias=True)
view(-1, vision_hidden)
```

Replacement:

```text
Linear(C*Tpatch*P*P -> vision_hidden, bias=True)
```

Preconditions:

- Input is already the processor-flattened patch vector with the exact Qwen patch order.
- Conv3d kernel and stride both equal `(temporal_patch_size, patch_size, patch_size)`.
- No padding, dilation, or groups.
- Weight flatten must match PyTorch Conv3d storage `[out, in, kt, kh, kw]` over the source `view`.

Failure cases:

- Raw NCHW/NCTHW image/video tensors.
- Any alternative image processor layout, patch size, temporal patch size, or channel order.
- NHWC layout pass that changes the flattened vector semantic order.

Parity test sketch: compare source `patch_embed` and rewritten linear for random flattened processor patches across image (`grid_t=1`) and video (`grid_t>1`) cases.

### Rewrite: QKV packed vision projection split

Preconditions:

- `qkv.weight` is `[3*hidden, hidden]`, bias `[3*hidden]`.
- Split order is `[q, k, v]` after `reshape(seq, 3, num_heads, head_dim).permute(1,0,2,3).unbind(0)`.

Replacement: one GEMM with packed output, followed by view/split, or three GEMMs only if weight rows are split exactly.

### Rewrite: placeholder stitch -> indexed copy/update

Preconditions:

- Boolean mask count equals feature rows.
- Feature rows are already concatenated in processor/grid order.
- No duplicate writes except source-defined visual positions.

Replacement: `index_select/nonzero(mask) -> indexed_copy` for initial visual embeds; `indexed_add` for DeepStack.

Failure cases: `inputs_embeds` path where placeholder masks are inferred by exact embedding-vector equality should be gated or normalized to `input_ids`.

### Rewrite: last-token-only logits

Source already accepts `logits_to_keep`; DinoML should lower integer `1` decode to `hidden[:, -1:, :] @ lm_head.T` and avoid full-sequence vocab GEMM.

### Layout guard: processor patch order

The image and video flatten `permute` orders are semantic. Introduce a conceptual no-layout-translation guard around processor-flattened patch tensors and vision patch embed unless the pass rewrites every axis use, Conv3d/linear weight flattening, position interpolation order, and merger grouping.

## 10. Kernel fusion candidates

Highest priority:

- Text RMSNorm and per-head q/k RMSNorm: every decoder layer uses them, and q/k norm placement affects attention parity.
- GQA FlashAttention prefill/decode with M-RoPE applied before cache update.
- SwiGLU MLP fused activation multiply plus GEMM epilogue opportunities.
- Placeholder indexed copy and DeepStack indexed add to avoid materializing expanded boolean masks.
- Vision patch embed Linear rewrite once processor ABI is frozen.

Medium priority:

- Vision packed qkv + RoPE + varlen attention path using `cu_seqlens`.
- Vision merger MLP over merged `4*vision_hidden` rows.
- Learned 2D position interpolation precompute/cache per distinct `grid_thw`.
- Last-token-only logits.

Lower priority:

- Processor GPU resize/normalize/patchify; useful for throughput but initially can stay CPU-side.
- FP8 provider path for sampled FP8 checkpoints.
- Beam expansion helpers for visual tensors.

## 11. Runtime staging plan

Stage 1: dense text-only decoder parity for a small dense config. Stub all multimodal tensors; validate RMSNorm, q/k norm, GQA, M-RoPE default text expansion, cache update, and sliced logits.

Stage 2: processor ABI loader. Accept `pixel_values`, `pixel_values_videos`, `image_grid_thw`, `video_grid_thw`, `mm_token_type_ids`; add shape guards and placeholder count validation.

Stage 3: vision encoder parity. Start with image-only flattened patches, then video flattened patches. Validate Conv3d/Linear patch embed, pos interpolation, vision RoPE, packed varlen attention, merger, and deepstack outputs independently.

Stage 4: multimodal prefill parity. Implement placeholder scatter, 3D position ids, `rope_deltas`, DeepStack additions, and prefill logits.

Stage 5: decode parity with KV cache. Drop visual tensors after first iteration, use cached `rope_deltas`, and validate token-by-token decode.

Stage 6: optimized lowering. Add Conv3d-to-linear rewrite, indexed copy/update kernels, fused RMSNorm/SwiGLU, GQA FlashAttention, and last-token logits.

Stage 7: optional surfaces. Add FP8 loading/provider support, MoE source-family support, beam expansion, and processor acceleration.

## 12. Parity and validation plan

- Unit-test M-RoPE cos/sin against source for text-only, image+text, video+text, and mixed image/video prompts. Include cached decode with nonzero `rope_deltas`.
- Unit-test processor placeholder counts from `image_grid_thw.prod() // 4` and per-frame video expansion.
- Single vision block parity with random flattened patch tensors and small grids.
- Full vision encoder parity for one image grid and one video grid; compare final merged embeds and three deepstack tensors.
- Placeholder stitch parity: exact equality of scatter positions and failure on mismatched token/feature counts.
- One text decoder layer parity, then N-layer parity with cache off/on.
- Prefill logits parity for image-only, video-only, and mixed prompts.
- Decode token parity for at least 4 generated steps after multimodal prefill.
- Suggested tolerances: fp32 `rtol=1e-4, atol=1e-4`; bf16/fp16 `rtol=2e-2, atol=2e-2` initially, tightened per kernel; FP8 only after a dedicated quantization path.

No tests/imports were run for this audit, per request.

## 13. Performance probes

- Processor throughput: image resize/patchify and video sampling/patchify separately.
- Vision encoder throughput by total visual tokens and number of packed segments.
- Position interpolation and M-RoPE construction overhead for dynamic grids.
- Prefill tokens/sec over text-only, image-only, video-only, mixed prompts.
- Decode tokens/sec with and without last-token-only logits.
- KV cache memory by layers, KV heads, head dim, and sequence length.
- Scatter/indexed update cost for visual embedding stitch and DeepStack.
- Attention backend comparison: eager/SDPA/FlashAttention for text GQA and vision varlen.
- FP8 load/dequant/provider comparison for FP8 checkpoints after admission.

## 14. Skip/defer list

- Training, gradients, and gradient checkpointing.
- MoE `qwen3_vl_moe` model body.
- FP8 quantized execution for first dense parity.
- Beam search and visual tensor expansion, unless generation parity requires it.
- Full chat template/text timestamp generation in compiled graph; keep in CPU processor/controller.
- GPU resize/rescale/normalize/patchify acceleration.
- Speculative decoding and multi-GPU tensor parallel.

## 15. Final implementation checklist

- [ ] Parse dense `qwen3_vl` config and reject/route `qwen3_vl_moe`.
- [ ] Load dense weights with correct tied/untied embedding handling.
- [ ] Implement Qwen3-VL text RMSNorm, q/k head RMSNorm, SwiGLU, GQA attention, cache update, and sliced logits.
- [ ] Implement M-RoPE position-id construction, interleaved cos/sin, and `rope_deltas` cache ABI.
- [ ] Add processor ABI guards for `pixel_values`, `pixel_values_videos`, `image_grid_thw`, `video_grid_thw`, and `mm_token_type_ids`.
- [ ] Implement vision patch embed, learned 2D pos interpolation, vision RoPE, varlen noncausal attention, merger, and deepstack mergers.
- [ ] Implement placeholder indexed copy and DeepStack indexed add.
- [ ] Add Conv3d patch embed to Linear rewrite with patch-order guards.
- [ ] Add no-layout-translation guards around flattened patch tensors and axis-sensitive merger/position math.
- [ ] Add single-op parity tests for M-RoPE, vision position interpolation, placeholder stitch, and cache decode positions.
- [ ] Add vision encoder parity and multimodal prefill/decode logits parity.
- [ ] Benchmark processor, vision encoder, prefill, decode, scatter/update, and KV memory.
- [ ] Separately audit/admit FP8 checkpoints and `qwen3_vl_moe`.

## Gated gaps for DinoML

- Multimodal admission requires `mm_token_type_ids`; without them source raises for multimodal grids because M-RoPE cannot be computed correctly.
- Processor patch order is ABI, not an optimization detail. Any NHWC/channel-last or Conv rewrite must prove exact flatten-order parity.
- Vision attention needs packed varlen sequence support or a segmented fallback keyed by `cu_seqlens`.
- Placeholder stitch and DeepStack need indexed copy/add kernels with source count checks.
- Decode needs a model/session-owned `rope_deltas` state tied to prefill cache lifetime.
- FP8 configs require explicit quantization provider semantics and ignored-layer handling.
- MoE configs are a different source family and must not be admitted through the dense `qwen3_vl` path.
