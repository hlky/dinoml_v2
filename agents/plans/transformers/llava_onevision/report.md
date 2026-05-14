# LLaVA-OneVision Transformers Audit

## 1. Source basis

```text
Transformers commit/version:
  b75feb2af64c3e29cbbc1bd859958c5432cc7ed4 from X:/H/transformers.

Model id:
  Family: llava_onevision.
  Primary representative checkpoint: llava-hf/llava-onevision-qwen2-7b-ov-hf.
  Additional configs inspected: llava-hf/llava-onevision-qwen2-0.5b-ov-hf,
  llava-hf/llava-onevision-qwen2-7b-ov-chat-hf,
  llava-hf/llava-onevision-qwen2-72b-ov-hf,
  llava-hf/llava-onevision-qwen2-7b-si-hf.

Config source:
  HF raw config/preprocessor/processor/tokenizer/generation snapshots saved under
  agents/plans/transformers/llava_onevision/_sources/.

Source files inspected:
  X:/H/transformers/src/transformers/models/llava_onevision/configuration_llava_onevision.py
  X:/H/transformers/src/transformers/models/llava_onevision/modeling_llava_onevision.py
  X:/H/transformers/src/transformers/models/llava_onevision/modular_llava_onevision.py
  X:/H/transformers/src/transformers/models/llava_onevision/processing_llava_onevision.py
  X:/H/transformers/src/transformers/models/llava_onevision/image_processing_llava_onevision.py
  X:/H/transformers/src/transformers/models/llava_onevision/image_processing_pil_llava_onevision.py
  X:/H/transformers/src/transformers/models/llava_onevision/video_processing_llava_onevision.py
  X:/H/transformers/src/transformers/models/qwen2/configuration_qwen2.py
  X:/H/transformers/src/transformers/models/qwen2/modeling_qwen2.py
  X:/H/transformers/src/transformers/models/siglip/configuration_siglip.py
  X:/H/transformers/src/transformers/models/siglip/modeling_siglip.py
  X:/H/transformers/src/transformers/models/llava_next/modeling_llava_next.py
  X:/H/transformers/src/transformers/models/llava_next/processing_llava_next.py
  X:/H/transformers/src/transformers/models/vipllava/modeling_vipllava.py
  X:/H/transformers/src/transformers/models/vipllava/configuration_vipllava.py

Any missing files or assumptions:
  modeling_llava_onevision.py and image_processing_llava_onevision.py are generated from
  modular_llava_onevision.py; use the modular file as the future edit authority, but
  the generated file is the exact imported runtime source at this commit.
  No custom remote-code files were required for the inspected llava-hf checkpoints.
  This report targets inference-only multimodal generation on CUDA.
```

Primary source URLs:

- https://github.com/huggingface/transformers/tree/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/llava_onevision
- https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/qwen2/modeling_qwen2.py
- https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/siglip/modeling_siglip.py

## 2. High-level architecture

LLaVA-OneVision is a multimodal projector plus autoregressive text decoder:

```text
CPU/data preprocessing
  -> anyres image patch/video frame tensors
  -> SigLIP vision tower
  -> 2-layer multimodal projector
  -> image/video placeholder embedding stitch
  -> Qwen2 causal decoder prefill
  -> Qwen2 decode with KV cache
  -> lm_head logits/sampling
```

Stage decomposition:

- CPU/data pipeline: chat template/tokenization, image anyres resize/pad/patch extraction, video frame resize/normalize, placeholder-token expansion.
- Vision encoder: SigLIP vision model over NCHW `pixel_values`; no cross-attention into the decoder. Image/video features can be computed independently before text prefill.
- Projector: `Linear(vision_hidden * selected_layers -> text_hidden) -> GELU -> Linear(text_hidden -> text_hidden)`, bias controlled by config and true in inspected configs.
- Prefix construction: source computes token embeddings, replaces `<image>` and `<video>` placeholder positions by projected visual vectors with `masked_scatter`, then calls the Qwen2 language model using `inputs_embeds`.
- Decode: visual inputs are forwarded only on the first generation iteration, or when `use_cache=False`; later decode steps rely on Qwen2 KV cache that already contains the multimodal prefix.

Independently stageable validation units are image processor token-count parity, SigLIP encoder parity, projector parity, multimodal stitch parity, Qwen2 prefill logits, and Qwen2 cached decode.

## 3. Important config dimensions

Shared multimodal defaults from inspected configs:

| Field | Value |
|---|---:|
| `model_type` | `llava_onevision` |
| image token id | `151646` |
| video token id | `151647` |
| `vision_feature_layer` | `-1` |
| `vision_feature_select_strategy` | `full` |
| projector act | `gelu` |
| projector bias | effective `true` from source default; inspected configs omit this field |
| `vision_aspect_ratio` | `anyres_max_9` |
| image processor size | `384 x 384` |
| processor `num_image_tokens` | `729` |
| vision tower | `siglip_vision_model` |
| vision hidden/layers/heads/intermediate | `1152 / 26 / 16 / 4304` |
| vision patch/image size | `14 / 384` |
| vision sequence length | `27 * 27 = 729`, no CLS for SigLIP |

Representative checkpoint sweep:

| Checkpoint | Text hidden | Layers | Heads | KV heads | Effective head dim | Intermediate | Vocab | RoPE theta | Top dtype |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `llava-onevision-qwen2-0.5b-ov-hf` | 896 | 24 | 14 | 2 | 64 | 4864 | 152000 | 1,000,000 | `float16` |
| `llava-onevision-qwen2-7b-ov-hf` | 3584 | 28 | 28 | 4 | 128 | 18944 | 152128 | 1,000,000 | `float16` |
| `llava-onevision-qwen2-7b-ov-chat-hf` | 3584 | 28 | 28 | 4 | 128 | 18944 | 152128 | 1,000,000 | `float16` |
| `llava-onevision-qwen2-7b-si-hf` | 3584 | 28 | 28 | 4 | 128 | 18944 | 152128 | 1,000,000 | `float16` |
| `llava-onevision-qwen2-72b-ov-hf` | 8192 | 80 | 64 | 8 | 128 | 29568 | 152128 | 1,000,000 | `float16` |

Several Qwen2 fields are omitted by checkpoint `text_config` and supplied by `Qwen2Config` defaults: `max_position_embeddings=32768`, `use_cache=True`, `hidden_act="silu"`, `rms_norm_eps=1e-6`, `attention_dropout=0.0`, `use_sliding_window=False`, and `sliding_window=None` after post-init. The checkpoints set `rope_theta` directly, not `rope_parameters`; the effective default RoPE type is the Qwen2 default with base `1e6`.

## 3a. Family variation traps

- Qwen2 uses GQA: `num_key_value_heads < num_attention_heads` for all inspected configs. Cache/storage uses KV heads, not expanded query heads.
- `head_dim` is not explicitly present in these configs; source computes `hidden_size // num_attention_heads`.
- Image token count is dynamic for single-image anyres inputs. `processor.num_image_tokens=729` is only the base 384x384 SigLIP patch count.
- Multi-image samples intentionally skip anyres patching per image. They are padded to square and emitted as one 384x384 patch each, with `num_image_tokens + 1` placeholders per image because of the newline embedding.
- Video placeholders use `num_frames * ceil(sqrt(num_image_tokens)/2)^2 + 1`; with 729 base tokens this is `num_frames * 14 * 14 + 1`.
- `vision_feature_select_strategy="default"` drops the first feature token before projection. For SigLIP OneVision configs the default is `full`, and SigLIP has no CLS token, so assuming LLaVA-NeXT CLS behavior is unsafe.
- `vision_feature_layer` may be a list; projector input width becomes `vision_hidden * len(layers)`.
- `vision_aspect_ratio="anyres_max_9"` changes image feature downsampling in `pack_image_features` when unpadded patch grids exceed the max-patch budget.
- Source supports both image and video tokens in the same forward, with independent masked scatters and count checks.
- `logits_to_keep` can slice logits to last tokens only; first integration should preserve the full hidden state but can optimize the final projection.
- `tie_word_embeddings=False` in inspected configs, but config post-init can inherit `text_config.tie_word_embeddings` if present.
- Layout trap: modeling and processors use NCHW tensors for vision, then sequence-major patch tokens. NHWC is only an optimization candidate inside bounded Conv2d/resize/interpolate regions.

## 4. Operator coverage checklist

Tensor/layout ops:

- NCHW image/video tensors, `view`, `reshape`, `flatten`, `transpose`, `permute`, `contiguous`.
- Dynamic `torch.cat`, `torch.split`, `torch.stack`, zero padding on the patch dimension.
- Boolean masks from token ids, `unsqueeze`, `expand_as`, `masked_scatter`.
- Shape-derived checks comparing placeholder count and visual feature element count.
- Dynamic sequence slicing for `logits_to_keep`.

Vision preprocessing-coupled ops:

- Resize preserving aspect ratio, center padding, square padding, image-grid resolution selection, non-overlapping patch division.
- Rescale and normalize using OpenAI CLIP mean/std.
- Bilinear interpolation in image feature downsampling and video pooling.
- Bicubic interpolation for SigLIP position embeddings if non-384 inputs reach the tower; current processor normalizes all patches to 384.

Neural network primitives:

- SigLIP patch embedding: `Conv2d(3 -> 1152, kernel=14, stride=14, padding=valid)` over NCHW.
- SigLIP encoder repeated 26 times: LayerNorm, noncausal MHA, MLP `Linear(1152 -> 4304) -> GELU -> Linear(4304 -> 1152)`, residual adds.
- Projector: `Linear(1152 * selected_layers -> text_hidden) -> GELU -> Linear(text_hidden -> text_hidden)`, usually `1152 -> 3584 -> 3584` for 7B.
- Qwen2 embeddings and causal decoder blocks.
- Qwen2 RMSNorm, GQA projections, SwiGLU MLP, LM head.

Attention primitives:

- SigLIP encoder noncausal MHA, no KV cache, head dim `1152 / 16 = 72`.
- Qwen2 causal self-attention with RoPE before cache update, GQA repeat for eager path, SDPA/FlashAttention-compatible backend dispatch.
- Causal masks and optional sliding-window causal masks from Qwen2 source, although inspected configs disable sliding attention.

Position/rotary ops:

- SigLIP learned 2D patch position embedding flattened to sequence; optional bicubic interpolation.
- Qwen2 RoPE over query/key with base `rope_theta=1e6`; cos/sin computed in float32 and cast to model dtype.

Generation/cache ops:

- Qwen2 `DynamicCache` or provided `Cache`.
- Per-layer KV cache shape before repeat: key/value `[batch, num_key_value_heads, cached_seq, head_dim]`.
- Cache stores RoPE-applied keys because source applies RoPE before `past_key_values.update`.
- Multimodal visual inputs are omitted after first generation iteration when cache is enabled.

Packed/varlen multimodal metadata:

- `image_sizes`: original image `(height, width)` per image.
- `batch_num_images`: number of images per text sample, needed to distinguish single-image anyres from multi-image square path.
- `pixel_values`: either 5D `[num_images, max_num_patches, 3, 384, 384]` or 4D stacked patches after source trimming.
- `pixel_values_videos`: `[batch, frames, 3, 384, 384]`.

Parameter aliasing:

- `lm_head.weight` is declared tie-compatible with `model.language_model.embed_tokens.weight`, but inspected configs set `tie_word_embeddings=false`; do not force aliasing unless the loaded config/weights require it.

## 5. Layer/block breakdown

Image path:

```text
processor:
  choose best grid from 36 pinpoints
  resize original preserving aspect ratio
  pad to chosen resolution
  divide into 384x384 patches
  prepend resized original 384x384 image
  normalize to NCHW float tensor

vision tower:
  patch_embeds = Conv2d(3, 1152, kernel=14, stride=14)(pixel_values)
  tokens = flatten_hw_transpose + learned_pos
  repeat 26 times:
    x = x + MHA(LayerNorm(x))
    x = x + MLP(LayerNorm(x))
  SigLIP model also computes a post-layernorm `last_hidden_state`; OneVision
  selects `image_outputs.hidden_states[vision_feature_layer]` as returned by HF
  hidden-state capture, so parity should compare that exact tensor.
```

Projector and image packing:

```text
selected = hidden_states[vision_feature_layer]       # or concat selected layers
if strategy == "default": selected = selected[:, 1:]
projected = linear_2(gelu(linear_1(selected)))
split projected by per-image num_patches
for each image:
  base = first patch feature
  if extra patches:
    view grid -> permute to [C, grid_h * h, grid_w * w]
    unpad to original aspect
    optionally bilinear downsample for anyres_max_N
    append image_newline as an extra width column
    flatten spatial to token sequence
    concat base tokens before packed anyres tokens
  else:
    append one newline token
```

Video path:

```text
pixel_values_videos [B, F, 3, 384, 384]
  -> view [B*F, 3, 384, 384]
  -> SigLIP tower
  -> select features and project
  -> view each frame as 27x27 token grid
  -> NCHW permute
  -> bilinear interpolate to ceil(27/2)=14 by 14
  -> flatten per frame
  -> reshape [B, F*196, text_hidden]
  -> append one image_newline token per video
```

Qwen2 decoder block, repeated `num_hidden_layers`:

```text
residual = x
x = RMSNorm(x)
q = Linear(hidden -> num_heads * head_dim, bias=True)(x)
k = Linear(hidden -> num_kv_heads * head_dim, bias=True)(x)
v = Linear(hidden -> num_kv_heads * head_dim, bias=True)(x)
q,k = RoPE(q,k)
k,v = cache.update(k,v) if cache exists
x = Attention(q,k,v, causal_mask, scaling=head_dim**-0.5, sliding_window=maybe_none)
x = residual + Linear(num_heads * head_dim -> hidden, bias=False)(x)
residual = x
x = RMSNorm(x)
x = residual + down_proj(silu(gate_proj(x)) * up_proj(x))
```

## 6. Attention requirements

SigLIP vision attention:

- Noncausal self-attention.
- MHA, 16 heads, head dim 72 for inspected configs.
- Sequence length is 729 per 384x384 patch.
- No KV cache. This branch is independently cacheable at the output-feature level, not as decoder KV.

Qwen2 text attention:

- Causal self-attention.
- GQA: query heads and KV heads differ. For 7B, Q heads 28, KV heads 4, head dim 128, repeat factor 7 in eager attention.
- Q/K/V are separate biased Linear layers. Output projection has no bias.
- RoPE is applied to Q and K before cache update; cached K is already position-encoded.
- Eager attention order is `matmul(q, k.T) * scale`, add mask, softmax in float32, cast to query dtype, dropout, matmul V.
- Source dispatches through Transformers attention interfaces; `_supports_flash_attn`, `_supports_sdpa`, flex attention, and attention backend support are enabled by the wrapper.
- Sliding-window fields exist in Qwen2, but inspected configs omit/disable `use_sliding_window`; effective layer types are all full attention.
- Decode cache per layer stores `[B, num_key_value_heads, T, head_dim]` K and V. Expanded repeated KV heads are a compute view, not cache storage.

## 7. Position encoding and custom math

Qwen2 RoPE:

```python
def qwen2_rope(q, k, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)
```

Cos/sin generation uses `inv_freq = 1 / rope_theta ** (arange(0, head_dim, 2) / head_dim)`, computes `freqs = inv_freq @ position_ids` in float32, concatenates duplicated frequencies, applies `cos`/`sin`, then casts to the model dtype. The inspected checkpoints set `rope_theta=1_000_000`.

SigLIP position embedding:

```text
patch_embedding -> flatten HW -> add learned position_embedding[position_ids]
if interpolate_pos_encoding=True or dynamic tracing forces it:
  reshape learned positions to [1, C, sqrt(N), sqrt(N)]
  bicubic interpolate to new patch grid
  flatten back to [1, new_tokens, C]
```

Image unpad and downsample math in `pack_image_features` is model-specific and shape-sensitive. The unpadding compares original and padded aspect ratios, removes symmetric padding, and then, for `anyres_max_9`, downsamples if `sqrt(curr_h * curr_w / (9 * base_grid_h**2)) > 1.1`.

## 8. Preprocessing and input packing

Image processor contract:

- Inputs are converted to RGB, resized/padded, rescaled by `1/255`, and normalized by OpenAI CLIP mean/std.
- For a single image in a sample, the processor chooses the best resolution from the 36 default pinpoints ranging from `384x384` to `2304x2304`, creates 384x384 tiles, and prepends a 384x384 resized original image.
- For multi-image samples, `need_patching=False`; each image is padded to square and processed as one patch.
- Returned fields are `pixel_values`, `image_sizes`, and `batch_num_images`.
- `pixel_values` is padded on the patch-count axis for batching, so model-side code trims each image to `image_num_patches` before concatenating.

Processor placeholder expansion:

- Text `<image>` tokens are expanded before tokenization into exactly the number of placeholders expected by the packed image features.
- Single-image token count is `base_features + unpadded_features + newline_features`, then minus one if `vision_feature_select_strategy=="default"`.
- Base features for OneVision are `num_image_tokens`, not `patches_h * patches_w + extra_cls`; this differs from LLaVA-NeXT because SigLIP has no CLS.
- Multi-image token count per image is `num_image_tokens + 1` for the newline.
- `<video>` token count is `(num_frames * ceil(sqrt(num_image_tokens)/2)^2) + 1`.

Runtime stitch:

```text
inputs_embeds = token_embedding(input_ids)
image_mask = input_ids == image_token_id
video_mask = input_ids == video_token_id
assert inputs_embeds[expanded_mask].numel() == features.numel()
inputs_embeds = inputs_embeds.masked_scatter(expanded_image_mask, image_features)
inputs_embeds = inputs_embeds.masked_scatter(expanded_video_mask, video_features)
```

For first integration, visual feature tensors can be precomputed and cached independently from decoder KV. Once stitched, the decoder cache contains the multimodal prefix just like text tokens.

## 9. Graph rewrite / lowering opportunities

### Rewrite: SigLIP patch Conv2d -> Linear

Source pattern:

```text
Conv2d(3 -> 1152, kernel=14, stride=14, padding=valid) -> flatten(2).transpose(1,2)
```

Replacement:

```text
WindowFlatten(NCHW, 14x14 non-overlap) -> MatMul(weight_flat.T) -> BiasAdd -> token sequence
```

Preconditions:

- `kernel_size == stride == patch_size`.
- `padding == "valid"`, dilation 1, groups 1.
- Input height/width divisible by patch size.
- Preserve source NCHW flatten order unless a local layout pass rewrites both window flatten and weight layout.

Weight transform:

```python
w = conv.weight.reshape(out_channels, in_channels * kh * kw)
```

Failure cases:

- Dynamic non-divisible image sizes, interpolated position embeddings with nonstandard tower inputs, grouped conv.

Parity sketch:

- Compare Conv2d path and lowered GEMM on random normalized `[B,3,384,384]` patches before and after position add.

### Rewrite: multimodal projector as two GEMMs

Source pattern:

```text
Linear(V -> H, bias=maybe) -> GELU -> Linear(H -> H, bias=maybe)
```

Preconditions:

- `vision_feature_layer` count fixed at compile/load time.
- Activation exactly `gelu` from `ACT2FN`.
- Projector bias flag matches loaded weights.

Failure cases:

- Config changes to a list of feature layers without widening `linear_1`.

Parity sketch:

- Compare projector outputs for selected SigLIP hidden states in fp32/fp16.

### Rewrite: image packing as explicit shape program

Source pattern:

```text
view grid -> permute -> flatten -> unpad -> optional interpolate -> newline concat -> flatten/transposed sequence
```

Replacement:

```text
Shape-derived gather/copy packer with optional bilinear resize and newline column append
```

Preconditions:

- `vision_aspect_ratio` parseable as `anyres_max_N`.
- `image_sizes` available and match processor original sizes.
- `image_grid_pinpoints` match config/processor.

Failure cases:

- Placeholder counts generated by a different processor revision, multi-image path mistaken for anyres path.

Parity sketch:

- Fixed image sizes spanning square, wide, tall, and max-grid cases; compare packed lengths and values.

### Rewrite: last-token-only logits

Source pattern:

```text
logits = lm_head(hidden_states[:, slice_indices, :])
```

Replacement:

```text
Gather last K hidden states -> GEMM to vocab
```

Preconditions:

- `logits_to_keep` is a positive int or static index tensor.
- Loss computation is absent.

Failure cases:

- Training/loss path, caller requests full sequence logits.

## 10. Kernel fusion candidates

Highest priority:

- Qwen2 RMSNorm: every decoder layer uses two RMSNorms plus final norm; source upcasts internally to float32.
- GQA FlashAttention with RoPE and KV cache: dominant prefill/decode cost; cache ABI must store post-RoPE K with KV-head count.
- Qwen2 SwiGLU MLP: `silu(gate) * up -> down` is large and repeated.
- SigLIP encoder LayerNorm + attention + MLP: image/video prefill cost is sizeable because anyres can multiply patch count.
- Placeholder stitch/prefix builder: needs a correct dynamic masked scatter or a deterministic indexed-copy lowering.

Medium priority:

- SigLIP patch Conv2d-to-GEMM rewrite.
- Multimodal projector fused GELU MLP.
- Image packer kernels for unpad/newline/flatten and optional bilinear downsample.
- Video 27x27 to 14x14 bilinear pooling plus flatten.
- Last-token-only LM head GEMM for decode.

Lower priority:

- SigLIP position embedding interpolation for non-processor inputs.
- Training loss and label shift.
- Sliding-window Qwen2 masks, because inspected OneVision configs disable them.

## 11. Runtime staging plan

1. Parse `LlavaOnevisionConfig`, nested SigLIP/Qwen2 configs, processor metadata, token ids, and image-grid pinpoints.
2. Load weights for SigLIP vision tower, projector, Qwen2 decoder, embeddings, LM head, and `image_newline`.
3. Implement a CPU/data-pipeline-compatible processor shim or ingest HF-processor outputs exactly: `input_ids`, `attention_mask`, `pixel_values`, `image_sizes`, `batch_num_images`, optional `pixel_values_videos`.
4. Validate SigLIP patch + encoder output for one 384x384 patch.
5. Validate projector and image/video packing against HF for representative image sizes.
6. Implement multimodal embedding stitch as indexed copy/masked scatter with strict feature/token count checks.
7. Run Qwen2 prefill parity from `inputs_embeds` with full logits.
8. Add cached decode with visual inputs omitted after first iteration.
9. Enable optimized attention/RMSNorm/SwiGLU/GEMM rewrites.
10. Add processor-aware batching and optional independent visual feature cache.

Stubs acceptable in the first milestone: HF processor outside DinoML runtime, video path, list-valued `vision_feature_layer`, full-sequence logits during decode, and non-default attention backends.

## 12. Parity and validation plan

- Processor token-count tests for square, wide, tall, high-resolution, multi-image, and video examples; assert placeholder counts equal packed feature lengths.
- SigLIP single-layer and full-encoder parity on normalized random/image patches, fp32 tolerance around `1e-5`, fp16/bf16 around `1e-2` depending on attention backend.
- Projector parity on random selected hidden states for each checkpoint hidden size.
- Image packer parity for known `image_sizes`: compare packed lengths, newline positions, and values.
- Video packer parity for several frame counts: expected tokens `frames * 196 + 1`.
- Multimodal stitch parity: compare `inputs_embeds` after masked scatter with HF.
- Qwen2 one-block parity from stitched embeddings.
- Prefill logits parity for 0.5B first, then 7B.
- Decode parity for one or two generated tokens with cache; verify no vision tensors are consumed after the first cached step.
- End-to-end image prompt and video prompt parity using HF processor outputs.

## 13. Performance probes

- HF processor throughput split by image size and batch shape.
- SigLIP encoder throughput for patch counts 1, 2, 5, 10, 17, 37.
- Image packer cost for unpad/downsample/newline on wide/tall/max-resolution cases.
- Video tower and pooling throughput by frame count.
- Qwen2 prefill throughput by multimodal-expanded sequence length.
- Decode tokens/sec with and without visual prefix cache.
- KV cache memory by model size: 0.5B, 7B, 72B.
- Attention backend comparison: eager/SDPA/FlashAttention where available.
- Projector GEMM throughput for variable visual token counts.
- Last-token-only versus full-sequence LM head cost.

## 14. Skip/defer list

- Training, loss, gradients, and gradient checkpointing.
- Beam search and advanced generation controllers beyond normal causal decode.
- Sliding-window Qwen2 attention until a checkpoint that enables it is in scope.
- Remote-code variants and non-Qwen2 text backbones.
- Multi-GPU tensor parallel, despite Qwen2 config carrying a TP plan.
- Processor implementation inside GPU runtime; first stage can consume HF processor outputs.
- Non-default list-valued `vision_feature_layer` unless a checkpoint requires it.
- Quantization/packed weight formats; inspected source uses normal dense PyTorch modules.
- SigLIP contrastive heads and classification heads; OneVision consumes `SiglipVisionModel` hidden states only.

## 15. Final implementation checklist

- [ ] Parse `LlavaOnevisionConfig` and nested `SiglipVisionConfig`/`Qwen2Config`.
- [ ] Parse processor configs: token strings, token ids, `num_image_tokens`, grid pinpoints, `vision_aspect_ratio`.
- [ ] Load dense weights and preserve optional embedding/LM-head aliasing only when config requires it.
- [ ] Implement/bridge image preprocessing outputs: `pixel_values`, `image_sizes`, `batch_num_images`.
- [ ] Implement SigLIP patch embedding, position add, encoder layers, and post layernorm.
- [ ] Implement multimodal projector with config-derived input width and bias flag.
- [ ] Implement OneVision image packer: anyres grid reshape, unpad, `anyres_max_N` downsample, newline append, base patch concat.
- [ ] Implement video path: frame flatten, SigLIP/projector, 2x bilinear pooling, newline append.
- [ ] Implement placeholder count validation and deterministic embedding stitch.
- [ ] Implement Qwen2 decoder from `inputs_embeds`: RMSNorm, GQA attention, RoPE, SwiGLU, LM head.
- [ ] Implement Qwen2 KV cache storing post-RoPE KV heads.
- [ ] Add prefill and decode parity tests for 0.5B and 7B configs.
- [ ] Add image/video processor-token-count parity tests.
- [ ] Add rewrite tests for SigLIP patch Conv2d-to-GEMM and last-token logits.
- [ ] Benchmark processor, vision tower, projector/packing, prefill, decode, and KV memory separately.
