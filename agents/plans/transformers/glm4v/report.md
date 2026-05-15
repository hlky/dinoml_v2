# Transformers family audit: `glm4v`

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4 from local checkout transformers
Model id: primary zai-org/GLM-4.1V-9B-Thinking; small fixture tiny-random/glm-4v
Config source: local source defaults plus downloaded HF raw config/preprocessor snapshots in _sources/
Source files inspected:
  transformers/src/transformers/models/glm4v/configuration_glm4v.py
  transformers/src/transformers/models/glm4v/modeling_glm4v.py
  transformers/src/transformers/models/glm4v/modular_glm4v.py
  transformers/src/transformers/models/glm4v/processing_glm4v.py
  transformers/src/transformers/models/glm4v/image_processing_glm4v.py
  transformers/src/transformers/models/glm4v/video_processing_glm4v.py
Any missing files or assumptions:
  processor_config.json was not present for the inspected official repos; preprocessor_config.json was available for GLM-4.1V and GLM-4.5V.
  modeling_glm4v.py is generated from modular_glm4v.py; future Transformers edits should target modular_glm4v.py, but runtime behavior here is taken from generated modeling_glm4v.py.
```

Config snapshots inspected:

- [zai-org/GLM-4.1V-9B-Thinking config](https://huggingface.co/zai-org/GLM-4.1V-9B-Thinking/raw/main/config.json), saved as `_sources/zai-org_GLM-4.1V-9B-Thinking_config.json`.
- [zai-org/GLM-4.1V-9B-Thinking preprocessor](https://huggingface.co/zai-org/GLM-4.1V-9B-Thinking/raw/main/preprocessor_config.json), saved as `_sources/zai-org_GLM-4.1V-9B-Thinking_preprocessor_config.json`.
- [tiny-random/glm-4v config](https://huggingface.co/tiny-random/glm-4v/raw/main/config.json), saved as `_sources/tiny-random_glm-4v_config.json`.
- [zai-org/GLM-4.5V config](https://huggingface.co/zai-org/GLM-4.5V/raw/main/config.json), saved for comparison only because it is `model_type="glm4v_moe"`.
- [zai-org/glm-4v-9b config](https://huggingface.co/zai-org/glm-4v-9b/raw/main/config.json), saved for comparison only because it is older `model_type="chatglm"` with remote-code mappings.

Primary runtime target: multimodal causal generation with image/video prefix encoding, text prefill, and autoregressive decode. First DinoML target should be image+text prefill and decode for native `model_type="glm4v"` only.

## 2. High-level architecture

GLM4V is a vision/video encoder plus causal text decoder:

```text
CPU processor
  -> image/video resize, normalize, patch flatten, token placeholder expansion, mm_token_type_ids
  -> vision Conv3d patch embed + vision transformer + spatial merge/projector
  -> masked placeholder stitch into token embeddings
  -> multimodal 3D/text position IDs and M-RoPE
  -> causal decoder prefill
  -> KV-cache decode
  -> lm_head logits
```

Stage decomposition:

- CPU/data pipeline: image/video loading, RGB conversion, resize with patch/merge divisibility, rescale/normalize, patch flattening, video frame sampling, prompt placeholder expansion, `mm_token_type_ids`.
- Vision encoder/projector: consumes already-flattened patch rows shaped `[sum_patches, C * temporal_patch * patch_h * patch_w]`; can be cached per image/video before decoder prefill.
- Prefix construction: text embeddings are created, then image/video features are stitched at placeholder token positions.
- Prefill: full multimodal sequence goes through causal text decoder with M-RoPE position IDs.
- Decode: no image/video tensors are forwarded after the first iteration when `use_cache=True`; decode uses KV cache plus stored `rope_deltas`.

Independently useful validation slices:

- Vision processor ABI and patch row order.
- Vision encoder/projector output count per `grid_thw`.
- Placeholder stitch count/order.
- Text-only decoder block parity.
- Multimodal M-RoPE position IDs.
- Prefill logits and one-step decode logits with cache.

## 3. Important config dimensions

Native production checkpoint, from `zai-org/GLM-4.1V-9B-Thinking_config.json`:

| Field | Value |
|---|---:|
| text hidden size | 4096 |
| text layers | 40 |
| text attention heads | 32 |
| text KV heads | 2 |
| text head dim | 128, inferred from 4096 / 32 |
| text intermediate | 13696 |
| vocab size | 151552 |
| max position embeddings | 65536 |
| hidden activation | `silu` |
| attention bias | text projections have bias in source; config carries `attention_bias=true` |
| MLP bias | false |
| norm | RMSNorm eps `1e-5` |
| dtype | `bfloat16` in config text subconfig |
| cache | `use_cache=true`; Transformers `DynamicCache` |
| RoPE | `rope_theta=10000`, `partial_rotary_factor=0.5`, `mrope_section=[8,12,12]` |
| vision hidden size | 1536 |
| vision depth | 24 |
| vision heads | 12 |
| vision head dim | 128 |
| vision patch | 14 |
| temporal patch | 2 |
| spatial merge | 2 |
| vision out hidden size | 4096 |
| vision intermediate | 13696 |
| image token ID | 151343 |
| video token ID | 151344 |
| image start/end IDs | 151339 / 151340 |
| video start/end IDs | 151341 / 151342 |

Representative sweep:

| Checkpoint | Scope | model_type | Text layers | Hidden | Heads/KV | Inter. | Vision | Tokens | Notes |
|---|---|---|---:|---:|---:|---:|---|---|---|
| `tiny-random/glm-4v` | in scope as debug fixture | `glm4v` | 2 | 64 | 2/1 | 128 | 2 layers, hidden 64, out 64 | image 151343, video 151344 | Small native ABI fixture; `mrope_section=[2,2,4]`. |
| `zai-org/GLM-4.1V-9B-Thinking` | primary | `glm4v` | 40 | 4096 | 32/2 | 13696 | 24 layers, hidden 1536, out 4096 | image 151343, video 151344 | Main native target. |
| `zai-org/GLM-4.5V` | out of scope for this report | `glm4v_moe` | 46 | 4096 | 96/8 | dense 10944, MoE 1408 | similar vision dimensions | image 151363, video 151364 | Requires separate `glm4v_moe` audit: MoE, different token IDs, more heads/layers. |
| `zai-org/glm-4v-9b` | out of scope | `chatglm` | 40 | 4096 | 32/2 | 13696 | older vision config, 63 layers, image size 1120 | `boi/eoi` IDs | Remote-code ChatGLM architecture; should not be routed to native `glm4v`. |

Preprocessor config for GLM-4.1V:

| Field | Value |
|---|---:|
| shortest edge pixels | 12544 |
| longest edge pixels | 9633792 |
| patch size | 14 |
| temporal patch size | 2 |
| merge size | 2 |
| mean/std | OpenAI CLIP mean/std |
| image processor | `Glm4vImageProcessor` |
| processor class | `Glm4vProcessor` |

## 3a. Family variation traps

- `glm4v` is not older `glm-4v-9b` from `zai-org/glm-4v-9b`: that config is `chatglm` with remote-code mappings and different vision topology.
- `GLM-4.5V` is `glm4v_moe`, not native `glm4v`. It shares processor-like settings but adds MoE routing and different placeholder token IDs. DinoML should route it to a separate audit.
- Text attention is GQA/MQA-like: `num_key_value_heads` can be far smaller than `num_attention_heads` (`2` vs `32` in GLM-4.1V).
- Source text attention always constructs `q_proj`, `k_proj`, and `v_proj` with `bias=True`; do not infer all attention bias behavior from `Glm4vTextConfig.attention_bias`.
- `hidden_size == num_attention_heads * head_dim` for inspected native configs, but `glm4v_moe` has explicit `head_dim=128` with `num_heads=96`, so separate audits must not infer attention output width from `hidden_size`.
- The text MLP uses packed `gate_up_proj` with split order `[gate, up]`, then `up * silu(gate)`.
- Vision attention uses one fused `qkv` projection with split order `[q, k, v]` after reshaping to `[seq, 3, heads, head_dim]`.
- Vision input to the model is not NCHW image tensors. The processor emits flattened patch rows; the model immediately views them into `[N, C, temporal_patch, patch, patch]` for Conv3d.
- Vision uses both rotary 2D spatial embeddings for attention and interpolated absolute 2D position embeddings via `grid_sample`.
- Video placeholders use the same normal image placeholder token ID inside video spans; `mm_token_type_ids` distinguishes image tokens (`1`) from video image-frame tokens (`2`).
- Source stitch uses broad `masked_scatter`, but processor expansion guarantees ordered placeholder tokens. DinoML should lower to guarded indexed row copy, not admit arbitrary boolean scatter as the first path.
- `position_ids` can be `[4, B, S]` during generation preparation: first plane is text positions for mask packing, remaining three are temporal/height/width M-RoPE IDs.
- Packed/varlen vision attention uses `cu_seqlens`; non-FlashAttention path loops over split sequences and is a fallback, not the production shape to optimize first.
- Layout translation must be guarded. Vision patch processor and patch embed have strict flatten order; `grid_sample` coordinates and Conv2d downsample expect source axes exactly.

## 4. Operator coverage checklist

Tensor/layout ops:

- `view`, `reshape`, `permute`, `transpose`, `contiguous`, `flatten`, `unsqueeze`, `squeeze`.
- `cat`, `split`, `chunk`, `repeat`, `repeat_interleave`, `expand`, `cumsum`, `pad`, `arange`, `stack`, `tolist`-derived static split planning.
- Boolean masks, equality comparisons against token IDs, mask expansion to embedding shape.
- Guarded `masked_scatter` replacement with indexed row copy.

Neural primitives:

- Embedding lookup: token embeddings `[vocab, hidden]`.
- LM head: `Linear(hidden -> vocab)`, no bias; source advertises tied weight key but top config has `tie_word_embeddings=false`, so preserve alias metadata only if weights are tied in an actual checkpoint.
- RMSNorm over last dim with fp32 variance and output cast back to input dtype.
- Linear/GEMM for text Q/K/V/O, packed gated MLP, down projection, vision QKV/O, vision MLP, patch merger.
- SiLU, GELU, elementwise multiply, residual add.
- Conv3d patch embed with kernel=stride=`[temporal_patch_size, patch_size, patch_size]`.
- Conv2d spatial downsample with kernel=stride=`spatial_merge_size`.
- LayerNorm in vision patch merger after projection.
- `grid_sample` bicubic, `align_corners=False`, `padding_mode="border"` for interpolated absolute vision position embeddings.

Attention primitives:

- Text causal self-attention, GQA: Q heads `32`, KV heads `2`, head dim `128` for GLM-4.1V.
- Vision noncausal packed self-attention with `cu_seqlens` and no attention mask.
- Eager fallback repeats KV heads before `matmul`; optimized path should use GQA FlashAttention/SDPA without materializing repeats where possible.
- Softmax is fp32 in eager path, then cast back.

Position/rotary ops:

- Text M-RoPE with partial rotary factor `0.5`.
- Vision 2D rotary position generation based on `grid_thw` and spatial merge ordering.
- Multimodal `get_rope_index` from `input_ids`, `mm_token_type_ids`, `image_grid_thw`, `video_grid_thw`, and optional `attention_mask`.
- Decode position update using stored `rope_deltas`.

Generation/cache ops:

- Transformers `DynamicCache`; per-layer K/V tensors are stored after RoPE application.
- Cache reorder/expand comes from GenerationMixin plus GLM4V visual tensor expansion override.
- First iteration carries image/video tensors; later cached decode sets `pixel_values` and `pixel_values_videos` to `None`.
- `logits_to_keep` last-token logits slice should be supported to avoid full-sequence LM head.

Preprocessing-coupled ops:

- CPU image/video resize with patch*merge divisibility and min/max pixel guards.
- Rescale/normalize using CLIP mean/std.
- Patch flatten ordering from processor:
  `[B,T,C,H,W] -> view(B, grid_t,tp,C, gh/m, m, ph, gw/m, m, pw) -> permute(0,1,4,7,5,8,3,2,6,9) -> [B, grid_t*grid_h*grid_w, C*tp*ph*pw]`.
- Video frame sampling at default `fps=2`, max duration `300s`, even frame count by repeating final frame.
- Prompt placeholder expansion: one `<|image|>` per post-merge vision token.
- `mm_token_type_ids`: `0=text`, `1=image`, `2=video`.

Packed/varlen metadata:

- `image_grid_thw` and `video_grid_thw`, shape `[num_images_or_videos, 3]`.
- Vision `cu_seqlens`, int32 for FlashAttention unless tracing.
- Video `video_grid_thw` is repeated into per-frame grids `[1,h,w]` for visual encoding, then split back by original video token counts.

## 5. Layer/block breakdown

Vision preprocessing and patch embedding:

```text
processor emits pixel_values: [sum_media, grid_t*grid_h*grid_w, C*tp*ph*pw]
patch_embed:
  x.view(-1, C, tp, ph, pw)
  Conv3d(C -> vision_hidden, kernel=[tp,ph,pw], stride=[tp,ph,pw])
  x.view(-1, vision_hidden)
  RMSNorm
```

Vision block, repeated `depth` times:

```text
res = x
x_norm = RMSNorm(x)
q,k,v = Linear(vision_hidden -> 3*vision_hidden, bias=config.attention_bias)
q,k = 2D vision RoPE(q,k)
x_attn = noncausal attention(q,k,v, cu_seqlens)
x_attn = Linear(vision_hidden -> vision_hidden, bias=False)
x = res + x_attn
res = x
x = RMSNorm(x)
x = Linear(vision_hidden -> out_hidden_size, bias=False)
x = SiLU(gate) * up
x = Linear(out_hidden_size -> vision_hidden, bias=False)
x = res + x
```

Vision postprocessing:

```text
x = RMSNorm(x)
x.view(-1, merge, merge, vision_hidden)
x.permute(0, 3, 1, 2)
x = Conv2d(vision_hidden -> out_hidden_size, kernel=merge, stride=merge)
x.view(-1, out_hidden_size)
pooler = Linear(out_hidden_size -> out_hidden_size)
pooler = GELU(LayerNorm(pooler))
pooler = SwiGLU Linear(out_hidden_size -> intermediate -> out_hidden_size)
```

Text decoder layer, repeated `num_hidden_layers` times:

```text
res = x
x = RMSNorm(x)
q = Linear(hidden -> heads*head_dim, bias=True)
k = Linear(hidden -> kv_heads*head_dim, bias=True)
v = Linear(hidden -> kv_heads*head_dim, bias=True)
q,k = text M-RoPE(q,k, position_ids)
k,v = cache.update(k,v, layer_idx) if cache enabled
x = causal GQA attention(q,k,v, mask)
x = Linear(heads*head_dim -> hidden, bias=False)
x = RMSNorm(x)
x = res + x
res = x
x = RMSNorm(x)
gate_up = Linear(hidden -> 2*intermediate, bias=False)
gate, up = chunk(gate_up, 2, dim=-1)
x = up * SiLU(gate)
x = Linear(intermediate -> hidden, bias=False)
x = RMSNorm(x)
x = res + x
```

The post-attention and post-MLP RMSNorms are unusual versus many Llama-family blocks; preserve ordering for parity.

## 6. Attention requirements

Text attention:

- Causal self-attention only for primary target.
- GQA/MQA-like: query heads and KV heads differ.
- GLM-4.1V: Q shape `[B, Hq=32, S, D=128]`; K/V shape `[B, Hkv=2, S, D=128]` before backend repeat/group handling.
- Q/K RoPE is applied before cache update, so cached keys are post-RoPE.
- Masking uses `create_causal_mask`; when packed 4-plane positions are present, text position plane is passed for mask construction.
- Eager path computes `matmul(q, k^T) * head_dim^-0.5`, adds mask, softmax in fp32, dropout, then `matmul` with V.
- FlashAttention/SDPA are advertised by source. DinoML should target causal GQA FlashAttention for prefill/decode and keep eager reference parity.

Vision attention:

- Noncausal self-attention over each image/frame sequence chunk.
- Packed variable-length path passes `cu_seq_lens_q`, `cu_seq_lens_k`, `max_length_q`, `max_length_k`, `is_causal=False`.
- Non-FlashAttention fallback splits Q/K/V by `cu_seqlens` and loops over media chunks.
- No cross-attention between vision and text; fusion happens by embedding stitch before text decoder.

Cache:

- Text decoder uses one KV cache pair per text layer.
- Cache stores `[B, kv_heads, cached_seq, head_dim]` K/V before backend repeat expansion.
- Decode path should carry `rope_deltas` alongside KV cache because future position IDs depend on multimodal prefill positions.
- Vision encoder/projector outputs can be cached independently across repeated generation calls, but this is not a Transformers KV cache.

## 7. Position encoding and custom math

Text M-RoPE:

```python
def glm4v_text_mrope(inv_freq, position_ids, mrope_section):
    # position_ids: [3, B, S] for temporal, height, width
    freqs = (inv_freq[None, None, :, None].expand(3, B, -1, 1)
             @ position_ids[:, :, None, :].float()).transpose(2, 3)
    chunks = freqs.split(mrope_section, dim=-1)
    freqs = torch.cat([chunk[i % 3] for i, chunk in enumerate(chunks)], dim=-1)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos() * attention_scaling, emb.sin() * attention_scaling
```

Then source interleaves cos/sin for the rotated half:

```python
cos = cos[..., : cos.shape[-1] // 2].repeat_interleave(2, dim=-1)
sin = sin[..., : sin.shape[-1] // 2].repeat_interleave(2, dim=-1)
q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
q = cat(q_rot * cos + rotate_even_odd(q_rot) * sin, q_pass)
k = cat(k_rot * cos + rotate_even_odd(k_rot) * sin, k_pass)
```

`position_ids` construction:

- Text spans increment positions by token length.
- Image/video spans derive 3D IDs from grid shape after spatial merge.
- For each vision span, `current_pos` advances by `max(grid_h, grid_w) // spatial_merge_size`, not by token count.
- Video grids are split into one-frame grids before position construction because timestamps separate frames in prompt text.
- `rope_deltas = max(position_ids)+1 - effective_input_length`; decode adds this delta to text positions.

Vision RoPE:

- Builds h/w position IDs in spatial-merge-friendly order.
- Looks up a 1D rotary table up to `max(grid_h, grid_w)`.
- Indexes by `[hpos, wpos]`, flattens the two axes into head rotary dim, then uses normal half-rotate on vision Q/K.

Precomputable:

- RoPE inverse frequencies and static image absolute position embedding table.
- For fixed image/video grids, vision RoPE indices, `cu_seqlens`, and multimodal position IDs can be cached.

Dynamic:

- `grid_thw`, placeholder positions, `mm_token_type_ids`, `attention_mask`, and `rope_deltas` are request dependent.

## 8. Preprocessing and input packing

Image ABI:

- Input image is converted to RGB, resized so height/width are multiples of `patch_size * merge_size` with min/max pixel guards.
- Processor emits `pixel_values` as flattened patch rows, not image tensors.
- Shape is `[sum_images, grid_t * grid_h * grid_w, 3 * temporal_patch_size * patch_size * patch_size]`; for still images, temporal dimension is padded to `temporal_patch_size`, so `grid_t=1`.
- `image_grid_thw` is `[num_images, 3]`, values `[grid_t, grid_h, grid_w]`.

Video ABI:

- Video decode and frame sampling are processor/data-pipeline work.
- Default sampling is `fps=2`, `max_duration=300`; missing metadata is an error when sampling by fps.
- Odd sampled frame count is made even by repeating the last frame.
- Processor emits `pixel_values_videos` with the same flattened patch row width.
- `video_grid_thw` is `[num_videos, 3]` before model entry; model expands it into per-frame `[1,h,w]` rows for visual encoding.

Text and placeholders:

- Processor replaces each `<|image|>` with enough repeated `<|image|>` placeholder tokens to match `prod(grid_thw) // merge_size^2`.
- Video prompt expansion emits `<|begin_of_image|><|image|><|end_of_image|>{timestamp}` for each selected frame, then expands each frame image token.
- `create_mm_token_type_ids` marks placeholder image token IDs inside `<|begin_of_video|> ... <|end_of_video|>` spans as `2`, and standalone image placeholders as `1`.
- Model requires `mm_token_type_ids` when multimodal grids are passed and `position_ids` is not provided.

Placeholder stitch:

- Source uses `inputs_embeds.masked_scatter(image_mask, image_embeds)` and then a second masked scatter for video.
- With `input_ids`, both image and video masks compare against `config.image_token_id`; video separation relies on the caller not passing both image and video paths with ambiguous masks unless `mm_token_type_ids` and prompt structure are well-formed.
- Count validation checks `inputs_embeds[mask].numel() == features.numel()`, so it validates total scalar count, not a stronger per-sample segment mapping.

DinoML lowering should require:

- Placeholder count equals feature rows for each modality.
- Placeholder positions are in processor order.
- For video, placeholder positions are inside video spans as indicated by `mm_token_type_ids`.
- Reject arbitrary user-provided `inputs_embeds` stitch until a stronger embedding-token equivalence path is implemented.

## 9. Graph rewrite / lowering opportunities

### Rewrite: processor patch rows + Conv3d patch embed -> Linear

Source pattern:

```text
pixel_values rows already flatten [C, temporal_patch, patch_h, patch_w]
view(-1, C, tp, ph, pw)
Conv3d(C -> hidden, kernel=[tp,ph,pw], stride=[tp,ph,pw])
view(-1, hidden)
```

Replacement:

```text
MatMul(pixel_values, conv3d.weight.reshape(hidden, C*tp*ph*pw).T) + bias
```

Preconditions:

- Input is exactly processor-emitted flattened patch rows.
- Conv3d kernel equals stride and spans the whole patch volume.
- `groups == 1`, padding `0`, dilation `1`.
- Flatten order matches processor `permute(0,1,4,7,5,8,3,2,6,9)`.

Failure cases:

- Raw image/video tensors are passed directly.
- Different processor patch order, temporal patch size, channel order, or grouped convolution.

Parity test sketch:

- Random flattened patch rows for several `grid_thw`; compare Conv3d source path to Linear rewrite fp32/fp16/bf16.

### Rewrite: spatial merge Conv2d -> block Linear

Source pattern:

```text
x.view(-1, merge, merge, hidden)
x.permute(0, 3, 1, 2)
Conv2d(hidden -> out_hidden, kernel=merge, stride=merge)
view(-1, out_hidden)
```

Replacement:

```text
BlockFlatten(merge, merge, hidden) -> MatMul(weight_flat.T) + bias
```

Preconditions:

- Input sequence is still ordered in processor spatial merge order.
- Conv2d kernel=stride=`spatial_merge_size`, padding `0`, dilation `1`, groups `1`.
- `grid_h` and `grid_w` are divisible by merge.

Failure cases:

- Any layout pass changes the `view/permute` interpretation.
- Non-divisible grids.

### Rewrite: masked_scatter multimodal stitch -> indexed row copy

Source pattern:

```text
mask = input_ids == image_token_id
inputs_embeds = inputs_embeds.masked_scatter(mask[...,None].expand_as(inputs_embeds), features)
```

Replacement:

```text
positions = nonzero(input_ids == image_token_id and optional mm_token_type_ids == modality)
inputs_embeds[positions] = features[row_order]
```

Preconditions:

- `input_ids` path, not arbitrary `inputs_embeds` token-equivalence path.
- Processor-expanded placeholders are in row-major feature order.
- Feature row count equals placeholder count per modality.
- For video, positions are filtered by `mm_token_type_ids == 2` or explicit video span guards.

Failure cases:

- User supplies custom `inputs_embeds`.
- Same image token ID appears in unmodeled contexts.
- Both image and video passed without unambiguous modality token types.

### Rewrite: packed gate/up projection -> fused SwiGLU GEMM

Source pattern:

```text
gate_up = Linear(hidden -> 2*intermediate, bias=False)
gate, up = chunk(gate_up, 2, dim=-1)
out = down_proj(up * silu(gate))
```

Replacement:

```text
GEMM packed [gate, up] -> fused silu_multiply -> GEMM down
```

Preconditions:

- Weight rows are first gate then up.
- Activation is `silu`.
- Last dimension chunk is exactly 2 equal parts.

### Layout guard: vision absolute position interpolation

`grid_sample` uses normalized `(w,h)` coordinates, bicubic mode, `align_corners=False`, and border padding. Do not apply NHWC/NCHW axis rewrites through this region unless the grid coordinate order and position table layout are rewritten together. First integration should preserve source layout.

## 10. Kernel fusion candidates

Highest priority:

- RMSNorm with fp32 variance and cast-back.
- Text GQA FlashAttention with post-RoPE KV cache.
- M-RoPE position embedding and Q/K apply, including partial rotary and interleaved even/odd rotation.
- Packed `gate_up_proj` + SiLU multiply for text MLP.
- Last-token-only LM head using `logits_to_keep`.
- Indexed row-copy multimodal stitch.

Medium priority:

- Vision Conv3d patch embed as Linear when input is processor-flattened.
- Vision packed varlen noncausal attention using `cu_seqlens`.
- Vision spatial merge Conv2d as Linear/block GEMM.
- Vision absolute position `grid_sample` precompute/cache for common grids.
- Q/K/V projection packing for text and vision, preserving split orders.

Lower priority:

- Full GPU preprocessing for resize/normalize/patch flatten.
- Beam expansion of visual tensors.
- Dynamic RoPE variants beyond inspected default.
- MoE routing for GLM-4.5V; separate family.

## 11. Runtime staging plan

Stage 1: config and admission.

- Admit only `model_type="glm4v"` and `architectures=["Glm4vForConditionalGeneration"]`.
- Reject `chatglm`, `glm4v_moe`, remote-code `auto_map`, missing `mm_token_type_ids` for multimodal inputs, and unsupported processor variants.

Stage 2: text-only decoder parity.

- Load text weights, implement decoder layer ordering, M-RoPE in text-only 3-plane fallback, causal GQA attention, KV cache, and LM head.

Stage 3: processor ABI and placeholder stitch.

- Parse processor outputs from CPU pipeline, validate `pixel_values`, `image_grid_thw`, `mm_token_type_ids`, placeholder counts, and indexed row copy.

Stage 4: vision encoder/projector parity.

- Implement patch embed, vision rotary, absolute position interpolation, packed vision attention, downsample, and patch merger.

Stage 5: multimodal prefill.

- Implement `get_rope_index`, `rope_deltas`, image/video feature stitch, and prefill logits.

Stage 6: decode with cache.

- Ensure later generation iterations omit pixel tensors, reuse KV cache and `rope_deltas`, and compute last-token logits only.

Stage 7: optimized kernels/fusions.

- Enable GQA FlashAttention, fused RMSNorm, fused SwiGLU, patch Conv rewrites, and vision varlen attention.

Initial stubs:

- CPU/data pipeline can remain external if it produces the exact processor ABI tensors.
- Video can be deferred after image+text; the decoder and image path share most mechanics.
- Beam search can use single-beam first.

## 12. Parity and validation plan

- Processor shape tests: image sizes with min/max resize, divisibility by `patch_size*merge_size`, and expected token count `grid_t*grid_h*grid_w/merge^2`.
- Patch flatten order test: synthetic image/video tensor with known values, compare processor output rows to expected flatten order.
- Vision patch embed rewrite test: Conv3d path vs Linear path.
- Vision one-block parity: random `pixel_values` and `grid_thw`, compare after one block.
- Vision full encoder parity for small fixture.
- M-RoPE unit tests: compare `get_rope_index`, `rope_deltas`, cos/sin, and Q/K application for text-only, image, and video prompts.
- Placeholder stitch tests: single image, multiple images, image+video, and mismatched placeholder count rejection.
- Text decoder single-layer parity with cache disabled/enabled.
- Prefill logits parity on tiny fixture.
- Decode parity: prefill with image then one token decode; verify pixel tensors are not reprocessed and logits match.
- Beam expansion smoke test for visual tensors if generation expansion is in scope.

Recommended tolerances:

- fp32 custom math: `rtol=1e-4`, `atol=1e-5`.
- bf16/fp16 full-block parity: `rtol=2e-2`, `atol=2e-2`, with tighter per-op fp32 references where possible.
- Attention backend parity should compare fp32 eager first, then reduced precision FlashAttention/SDPA with backend-specific tolerance.

## 13. Performance probes

- CPU preprocessing throughput: images/sec and video frames/sec for resize/normalize/patch flatten.
- Vision encoder throughput by `sum_patches`, `grid_h`, `grid_w`, and number of media items.
- Vision varlen attention probe: packed FlashAttention vs split-loop fallback.
- Prefill throughput by text length and image token count.
- Decode tokens/sec with and without last-token-only LM head.
- KV cache memory by batch, layers, KV heads, and sequence length.
- Placeholder stitch overhead for indexed row copy vs scatter fallback.
- Patch embed Linear rewrite vs Conv3d on flattened rows.
- Spatial merge Conv2d rewrite vs source Conv2d.
- End-to-end request latency split into preprocessing, vision, prefill, and decode.

## 14. Skip/defer list

- Training loss and gradient checkpointing.
- GLM-4.5V / `glm4v_moe` MoE routing.
- Older remote-code `chatglm` GLM-4V.
- Full GPU image/video preprocessing.
- Beam search and visual tensor expansion beyond a smoke test.
- Arbitrary user-provided `inputs_embeds` multimodal stitch.
- Dynamic/non-default RoPE variants not present in inspected native configs.
- Multi-GPU tensor parallel plans.
- Quantized/packed weight formats; none are source-coupled in native `glm4v` code.

## 15. Final implementation checklist

- [ ] Parse native `glm4v` config and reject `chatglm`/`glm4v_moe`.
- [ ] Parse processor/preprocessor settings and validate patch/merge ABI.
- [ ] Load text, vision, and LM-head weights with alias metadata for tied-key checks.
- [ ] Implement GLM4V RMSNorm ordering and decoder residual structure.
- [ ] Implement text GQA attention with post-RoPE KV cache.
- [ ] Implement M-RoPE cos/sin and `get_rope_index`/`rope_deltas`.
- [ ] Implement packed text SwiGLU with `[gate, up]` split.
- [ ] Implement vision patch embed, vision RoPE, and absolute position `grid_sample`.
- [ ] Implement packed vision varlen attention with `cu_seqlens`.
- [ ] Implement vision spatial merge and patch merger.
- [ ] Implement guarded indexed row-copy placeholder stitch.
- [ ] Add Conv3d patch embed -> Linear rewrite.
- [ ] Add Conv2d spatial merge -> Linear rewrite.
- [ ] Add last-token-only LM-head lowering.
- [ ] Add text-only tiny parity tests.
- [ ] Add image+text prefill parity tests.
- [ ] Add one-step decode cache parity tests.
- [ ] Add processor ABI and mismatch rejection tests.
- [ ] Benchmark preprocessing, vision encoder, prefill, decode, and KV cache memory.

## Gated gaps for DinoML

- No native admission for `glm4v_moe` or `chatglm` GLM-4V under this report.
- No arbitrary `masked_scatter`; first path needs processor-ordered indexed row copy with placeholder guards.
- No layout translation through processor patch flatten, vision `grid_sample`, or spatial merge until axis rewrites are explicitly proven.
- No multimodal generation without `mm_token_type_ids` unless caller supplies trusted `position_ids`.
- No decode parity without carrying `rope_deltas` beside KV cache.
- No optimized vision path without `cu_seqlens` support for packed noncausal attention or a validated split-loop fallback.
