# LLaVA-NeXT-Video Transformers Audit

## 1. Source basis

Transformers commit/version: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4` in `transformers`.

Model id: family `llava_next_video`; representative checkpoints inspected from Hugging Face:

- `llava-hf/LLaVA-NeXT-Video-7B-hf`
- `llava-hf/LLaVA-NeXT-Video-7B-32K-hf`
- `llava-hf/LLaVA-NeXT-Video-34B-hf`
- `llava-hf/LLaVA-NeXT-Video-34B-DPO-hf`
- `katuni4ka/tiny-random-llava-next-video`

Config source: checkpoint `config.json`, plus processor/preprocessor JSON where available. Small snapshots are under `agents/plans/transformers/llava_next_video/_sources/`. Raw Hub URLs used include:

- `https://huggingface.co/llava-hf/LLaVA-NeXT-Video-7B-hf/raw/main/config.json`
- `https://huggingface.co/llava-hf/LLaVA-NeXT-Video-7B-hf/raw/main/processor_config.json`
- `https://huggingface.co/llava-hf/LLaVA-NeXT-Video-7B-hf/raw/main/preprocessor_config.json`
- `https://huggingface.co/llava-hf/LLaVA-NeXT-Video-7B-32K-hf/raw/main/config.json`
- `https://huggingface.co/llava-hf/LLaVA-NeXT-Video-34B-hf/raw/main/config.json`
- `https://huggingface.co/llava-hf/LLaVA-NeXT-Video-34B-DPO-hf/raw/main/config.json`
- `https://huggingface.co/katuni4ka/tiny-random-llava-next-video/raw/main/config.json`

Source files inspected:

- `src/transformers/models/llava_next_video/configuration_llava_next_video.py`
- `src/transformers/models/llava_next_video/modeling_llava_next_video.py`
- `src/transformers/models/llava_next_video/modular_llava_next_video.py`
- `src/transformers/models/llava_next_video/processing_llava_next_video.py`
- `src/transformers/models/llava_next_video/video_processing_llava_next_video.py`
- Comparison sources: `llava_next/modeling_llava_next.py`, `llava_next/processing_llava_next.py`, `llava_onevision/modeling_llava_onevision.py`, `llava_onevision/processing_llava_onevision.py`.
- Delegated tower/decoder sources: `clip/modeling_clip.py`, `clip/configuration_clip.py`, `llama/modeling_llama.py`, and `mistral/modeling_mistral.py`.

Any missing files or assumptions: `modeling_llava_next_video.py` and `configuration_llava_next_video.py` are generated from `modular_llava_next_video.py`; the generated files are the exact runtime source basis, while modular is the future edit source. `llava-hf/LLaVA-NeXT-Video-DPO-7B-hf` returned 401 for `config.json`, so it is not included. No remote-code files were required for the inspected in-library class. The report targets inference-time image/video/text generation on CUDA.

## 2. High-level architecture

LLaVA-NeXT-Video is a CLIP-style vision encoder plus multimodal projector plus causal language model. It inherits the LLaVA-NeXT AnyRes image packing path and adds a video path that treats frames as independent CLIP images, spatially pools patch tokens, projects them to text hidden width, then stitches them into repeated `<video>` placeholders.

Dataflow:

```text
image/video/text processor -> expanded input_ids + pixel tensors
images -> AnyRes CLIP patches -> selected vision hidden states -> projector -> spatial pack/unpad/newline
videos -> per-frame CLIP -> selected vision hidden states -> spatial pool -> projector -> frame-token sequence
input_ids -> token embeddings
image/video placeholder masks -> masked_scatter multimodal features into embeddings
stitched embeddings -> text decoder prefill -> logits + KV cache
cached decode -> text decoder only -> logits/sampling
```

Stage decomposition:

- CPU/data-pipeline: image RGB conversion, AnyRes best-resolution selection, resize/pad/patch extraction, CLIP normalization, video frame decoding/sampling outside this model, video resize/crop/normalize, tokenizer/chat template, and placeholder expansion.
- Independently cacheable image prefix: `pixel_values + image_sizes -> vision_tower -> feature selection -> projector -> pack_image_features`. Cache key must include image sizes, grid pinpoints, `vision_feature_layer`, feature selection, projector weights, and `image_newline`.
- Independently cacheable video prefix: `pixel_values_videos -> vision_tower -> feature selection -> spatial pool -> projector -> flatten frames`. Cache key must include frame count/order, video processor settings, `spatial_pool_mode`, `spatial_pool_stride`, and feature selection.
- Prefix construction: image/video features are inserted with `masked_scatter` into the token embedding stream at repeated placeholder token positions.
- Prefill: the delegated causal LM consumes the full multimodal embedding sequence and creates the text decoder KV cache.
- Decode: `prepare_inputs_for_generation` forwards `pixel_values`, `pixel_values_videos`, and `image_sizes` only on the first generation iteration, or when cache is disabled.

## 3. Important config dimensions

Source defaults from `LlavaNextVideoConfig`: `image_token_index=32001`, `video_token_index=32000`, `projector_hidden_act="gelu"`, `vision_feature_select_strategy="default"`, `vision_feature_layer=-2`, `multimodal_projector_bias=True`, `tie_word_embeddings=False`, `spatial_pool_mode="average"`, `spatial_pool_stride=2`, `image_seq_length=576`, `video_seq_length=288`, and default AnyRes pinpoints `[[336,672],[672,336],[672,672],[1008,336],[336,1008]]`. If omitted, `vision_config` defaults to CLIP vision hidden 1024, 24 layers, 16 heads, patch 14, image 336. If omitted, `text_config` defaults to Llama.

| Checkpoint | Text model | Text dims | KV heads | Context / RoPE / cache | Vision tower | Video/image settings | Dtype source |
|---|---:|---:|---:|---:|---:|---:|---|
| tiny-random | Llama | hidden 16, 2 layers, 4 heads, head_dim 4, MLP 64 | 4 | max pos 2048, rope theta 10000, `use_cache=true` | CLIP hidden 32, 2 layers, 4 heads, patch 2, image 32 | image token 32001, video token 32000, image_seq 225, video_seq 49, pool average stride 2 | `config.json` float32 |
| 7B | Llama/Vicuna | config omits hidden/layers/heads; effective current Llama defaults should be resolved at load time | effective Llama default unless resolved from weights/config class | max pos 4096, `rope_scaling={type:linear,factor:2.5}` | CLIP hidden 1024, 24 layers, 16 heads, patch 14, image 336 | image token 32001, video token 32000, pool average stride 2 out channels 1024 | top `config.json` bfloat16; text float16 |
| 7B-32K | Mistral | config omits hidden/layers/heads except MLP 14336; effective current Mistral defaults are hidden 4096, 32 heads, 32 layers unless overridden by class | 8 | max pos 32768, rope theta 1000000, `sliding_window=null` | CLIP hidden 1024, 24 layers, 16 heads, patch 14, image 336 | same token ids, pool average stride 2 out channels 1024 | bfloat16 |
| 34B | Llama/Yi-style | hidden 7168, 60 layers, 56 heads, head_dim 128, MLP 20480 | 8 | max pos 4096, rope theta 5000000, text `use_cache=false` in config | CLIP hidden 1024, 24 layers, 16 heads, patch 14, image 336 | image token 64004, video token 64003, pool average stride 2 | bfloat16 |
| 34B-DPO | Llama/Yi-style | same shape as 34B in fetched config | 8 | same as 34B | same as 34B | same token ids/settings as 34B | bfloat16 |

Processor settings from `LLaVA-NeXT-Video-7B-hf`: `processor_class="LlavaNextVideoProcessor"`, `image_token="<image>"`, `video_token="<video>"`, `num_additional_image_tokens=1`, `patch_size=14`, and `vision_feature_select_strategy="default"`. Image preprocessor uses `LlavaNextImageProcessor`, BICUBIC resize, `size.shortest_edge=336`, `crop_size=336x336`, `do_pad=true`, RGB conversion, rescale by `1/255`, and OpenAI CLIP mean/std. Video processor source defaults use `size.shortest_edge=224`, `crop_size=224x224`, BICUBIC, CLIP mean/std, center crop, RGB conversion, and `do_sample_frames=False` for backward compatibility.

## 3a. Family variation traps

- Text model is delegated. This family can wrap Llama/Vicuna, Mistral, and Yi-style Llama configs; decoder operator coverage and cache ABI follow the delegated `text_config`.
- Some configs omit text dimensions. DinoML should instantiate the delegated config class or read resolved model config, not infer from model name alone.
- GQA varies: 34B uses 56 query heads and 8 KV heads; 7B-32K Mistral uses 32 query heads and 8 KV heads after default resolution.
- Token IDs vary. 7B uses video/image IDs 32000/32001; 34B uses 64003/64004.
- `image_seq_length` and `video_seq_length` are nominal config fields. Actual placeholder counts are processor-derived and depend on image sizes, frame count, patch size, and pooling stride.
- Image path and video path use different spatial sizes in the standard processor: images use 336, videos default to 224. The same CLIP tower with `image_size=336` can still receive video frames as pixel tensors; the source uses selected hidden states length dynamically.
- Video token expansion assumes average-pool stride 2 behavior with `num_video_tokens = (H/patch)*(W/patch)//4*num_frames`. This formula is exact for the default 224/14 grid and stride 2, but would be wrong for `spatial_pool_mode="conv"` with changed out channels or non-divisible grids unless processor and model are updated together.
- `spatial_pool_mode` can be `"average"`, `"max"`, or `"conv"`. Conv mode changes runtime ops and may change channel count before the projector if `spatial_pool_out_channels` differs from `vision_config.hidden_size`; the projector source still expects `vision_config.hidden_size * num_feature_layers`, so mismatched conv out channels should be rejected unless source changes.
- `vision_feature_layer` may be an int or list. A list concatenates hidden states on the last dimension before pooling/projector.
- `vision_feature_select_strategy="default"` removes token 0 before image packing and before video pooling. `"full"` keeps token 0; video pooling then requires a square token count, so `"full"` with CLIP CLS is likely invalid unless a CLS-free tower is used.
- Image AnyRes packing is inherited from `llava_next`; video does not use AnyRes unpadding, image newline, or image sizes.
- `inputs_embeds` mode detects placeholder positions by exact equality to special-token embeddings. That is fragile in compiled runtimes and should be a later feature.
- `use_image_newline_parameter` appears in checkpoint configs but the inspected source always creates `self.image_newline` and uses it for images only; video path in this family does not append newline tokens.
- Difference from `llava_next`: adds `video_token_id`, `pixel_values_videos`, `vision_resampler`, video placeholder checking, and first-iteration video forwarding.
- Difference from `llava_onevision`: OneVision defaults to SigLIP/Qwen-style token IDs, `vision_feature_select_strategy="full"`, `vision_aspect_ratio="anyres_max_9"`, optional multi-image batch metadata, bilinear video pooling after projection, and appends a video newline token. LLaVA-NeXT-Video pools before projection and does not append video newline.

## 4. Operator coverage checklist

Tensor/layout ops:

- Embedding lookup for `input_ids -> inputs_embeds`, plus optional equality-based placeholder lookup for `inputs_embeds` mode.
- Image tensors: `pixel_values` as `[batch_or_images, num_patches, C, H, W]` or flattened `[total_patches, C, H, W]`; split/concat by `image_size_to_num_patches`.
- Video tensors: `pixel_values_videos` as `[batch, frames, C, H, W]`; reshape to `[batch*frames, C, H, W]`; split back into per-video groups.
- Feature selection: hidden-state tuple indexing, optional `cat(..., dim=-1)`, optional token-0 slice.
- Image pack/unpad: `view`, `permute`, `contiguous`, spatial flatten, aspect-ratio unpadding, learned newline column append, flatten to `[tokens, hidden]`, concat base-image tokens.
- Video pool: view `[B*F, seq, D] -> [B*F, H, W, D]`, permute to NCHW, `AvgPool2d`/`MaxPool2d`/`Conv2d`, flatten spatial, transpose to sequence.
- Placeholder stitch: boolean masks for image and video token IDs, `unsqueeze`, `expand_as`, count/numel check, `masked_scatter`.
- Logits: `logits_to_keep` slice and `Linear(text_hidden -> vocab_size, bias=False)`.

Neural network primitives:

- CLIP vision tower: `Conv2d(3 -> vision_hidden, kernel=stride=patch, bias=False)`, learned class token, learned position embeddings, LayerNorm, full self-attention, Linear Q/K/V/O with bias, QuickGELU or configured activation MLP, residual adds.
- Projector: `Linear(vision_hidden * num_feature_layers -> text_hidden)`, GELU by default, then `Linear(text_hidden -> text_hidden)`, bias controlled by `multimodal_projector_bias`.
- Video resampler: default `AvgPool2d(kernel=2,stride=2)` over CLIP patch grid; optional max pool or conv.
- Text decoder: RMSNorm, Q/K/V/O projections, RoPE, causal attention, GQA repeat for eager math, gated SiLU MLP, residual adds, final norm, LM head.

Attention primitives:

- Vision full noncausal MHA over CLIP tokens.
- Text causal self-attention with compact KV cache, RoPE on Q/K before cache update, optional GQA, optional Mistral sliding-window causal mask when delegated config has a non-null `sliding_window`.
- Source wrapper advertises FlashAttention, SDPA, flex attention, and generic attention backend support; actual dispatch is delegated to the text and vision models.

Position/rotary ops:

- CLIP learned absolute position embeddings for the vision tower.
- Llama/Mistral RoPE with config-dependent `rope_theta` and `rope_scaling`.

Generation/cache ops:

- Dynamic text cache allocation in delegated decoder when `use_cache=True` and no cache is provided.
- Per-layer compact KV cache shape `[batch, num_key_value_heads, kv_seq_len, head_dim]`; repeat to query heads only for eager attention math.
- `prepare_inputs_for_generation` keeps image/video tensors only on first iteration or when cache is disabled.
- Last-token-only logits via `logits_to_keep`.

Preprocessing-coupled ops:

- Image AnyRes best-resolution selection, padding, patch extraction, image-size metadata.
- Video frame decoding/sampling is outside the model; processor accepts already provided video or sampled frames.
- Video placeholder expansion from frame count and processed frame grid.

Scatter/indexed update ops:

- Separate image and video placeholder masks must each match the flattened feature tensor element count exactly.
- Stitched embeddings preserve original text token positions; multimodal tokens are not prepended as a separate prefix object.

Quantized/packed weight metadata ops: none in this in-library source. Quantized checkpoints should be treated as external loading/provider concerns.

## 5. Layer/block breakdown

Image preprocessing and AnyRes patch creation:

```text
image [H,W,3]
best = select_best_resolution((H,W), image_grid_pinpoints)
resized = aspect-preserving resize
padded = pad to best resolution
patches = divide padded image into 336x336 crops
base = resized/cropped 336x336 image
per crop: RGB -> rescale -> normalize -> NCHW
pixel_values image item = [base] + patches
```

Video preprocessing:

```text
video frames [F,H,W,3]
processor: optional frame sampling outside model defaults, RGB, resize shortest edge 224,
center crop 224x224, rescale, normalize
pixel_values_videos = [B,F,3,224,224] for default processor
```

CLIP vision block, repeated `vision_config.num_hidden_layers`:

```text
pixel_values [N,3,H,W]
patch = Conv2d(3 -> vision_hidden, kernel=stride=patch_size)
tokens = flatten spatial -> transpose              # [N, patches, vision_hidden]
tokens = concat(class_token, tokens) + positions
x = pre_layernorm(tokens)
for block:
  y = LayerNorm(x)
  q,k,v = Linear(vision_hidden -> vision_hidden, bias=True)
  y = full self-attention(q,k,v)
  x = x + Linear(vision_hidden -> vision_hidden, bias=True)
  y = LayerNorm(x)
  y = Linear(vision_hidden -> intermediate) -> activation -> Linear(intermediate -> vision_hidden)
  x = x + y
```

Image feature select/project/pack:

```text
selected = hidden_states[layer] or cat(hidden_states[layers], dim=-1)
if strategy == "default": selected = selected[:, 1:]
projected = Linear(... -> text_hidden) -> GELU -> Linear(text_hidden -> text_hidden)
split projected by per-image patch count
base = projected[0]
patch_grid = projected[1:].view(num_patch_h, num_patch_w, grid_h, grid_w, text_hidden)
patch_grid = permute/flatten to [text_hidden, feature_h, feature_w]
patch_grid = unpad_image(patch_grid, original_image_size)
patch_grid = cat learned image_newline as extra feature column
image_tokens = cat(base, flattened patch_grid)
```

Video feature select/pool/project:

```text
pixel_values_videos [B,F,C,H,W] -> [B*F,C,H,W]
vision hidden = CLIP(pixel_values_videos, output_hidden_states=True)
selected = hidden_states[layer] or cat(hidden_states[layers], dim=-1)
if strategy == "default": selected = selected[:, 1:]
pooled = view selected as square grid -> NCHW -> AvgPool2d/MaxPool2d/Conv2d -> sequence
projected = Linear(... -> text_hidden) -> GELU -> Linear(text_hidden -> text_hidden)
split projected into B tensors of length F
flatten each video [F, pooled_tokens, text_hidden] -> [F*pooled_tokens, text_hidden]
```

Text decoder block, repeated according to delegated config:

```text
x = RMSNorm(x)
q = Linear(hidden -> num_attention_heads * head_dim)
k = Linear(hidden -> num_key_value_heads * head_dim)
v = Linear(hidden -> num_key_value_heads * head_dim)
q,k = RoPE(q,k)
k,v = cache.update(k,v, layer_idx)
attn = causal_attention(q,k,v, mask, scaling=head_dim**-0.5)
x = residual + o_proj(attn)
y = RMSNorm(x)
y = down_proj(silu(gate_proj(y)) * up_proj(y))
x = residual + y
```

Projection bias details: multimodal projector bias defaults to enabled. CLIP attention/MLP linears use bias in standard source. Mistral text projections are bias-free. Llama text projections follow `attention_bias` and `mlp_bias`, false for the representative 34B config.

## 6. Attention requirements

Vision attention:

- Full noncausal self-attention, no KV cache.
- CLIP default 336 image path has 24x24 patch tokens plus CLS, so source hidden states have length 577 before default CLS removal.
- Default 224 video processor with patch 14 yields 16x16 patch tokens plus CLS, so hidden states have length 257 before default CLS removal.
- Attention math comes from CLIP: Q/K/V linear projections, scale by `head_dim**-0.5`, backend attention interface, output projection.

Text attention:

- Causal self-attention inherited from Llama or Mistral.
- Query tensor shape before attention: `[batch, num_attention_heads, q_len, head_dim]`.
- Cached K/V tensor shape before repeat expansion: `[batch, num_key_value_heads, kv_seq_len, head_dim]`.
- Cached keys are stored after RoPE, because the source applies RoPE before `cache.update`.
- Mistral passes `sliding_window=getattr(config, "sliding_window", None)` to the attention backend and switches mask creation to sliding-window only when non-null. The 7B-32K config has `sliding_window=null`.
- Prefill sequence includes text plus expanded image/video placeholders. Decode is text-only once multimodal embeddings have been merged into the cache.

Optimized attention compatibility: first DinoML support should target compact GQA KV cache for Llama/Mistral-style decoders and full CLIP encoder attention. Fused attention parity must preserve RoPE placement, mask semantics, backend scaling, and compact cache storage.

## 7. Position encoding and custom math

CLIP vision uses learned absolute position embeddings. For default image AnyRes, each crop is resized to the tower image size before CLIP, so no high-resolution CLIP position interpolation is needed in the standard path. For video, the source passes processed frames through the same tower; if frame size differs from `vision_config.image_size`, position behavior depends on delegated CLIP support and should be validated directly.

Llama/Mistral RoPE is inherited. DinoML should implement or compose the delegated decoder family rather than duplicating a LLaVA-specific RoPE.

Custom image unpadding:

```python
def unpad_image_like_source(tensor, original_h, original_w):
    current_h, current_w = tensor.shape[1:]
    if original_w / original_h > current_w / current_h:
        new_h = int(round(original_h * (current_w / original_w), 7))
        pad = (current_h - new_h) // 2
        return tensor[:, pad : current_h - pad, :]
    new_w = int(round(original_w * (current_h / original_h), 7))
    pad = (current_w - new_w) // 2
    return tensor[:, :, pad : current_w - pad]
```

Custom video pooling shape:

```python
def llava_next_video_pool(tokens, stride=2):
    # tokens: [batch_frames, square_patches, channels], CLS already removed
    side = int(math.sqrt(tokens.shape[1]))
    x = tokens.view(tokens.shape[0], side, side, tokens.shape[-1]).permute(0, 3, 1, 2)
    x = avg_pool2d_or_configured_pool(x, kernel_size=stride, stride=stride)
    return x.flatten(2).transpose(1, 2).contiguous()
```

Precomputable: CLIP position embeddings, text RoPE inverse frequencies for fixed RoPE type, image/video placeholder counts for known sizes/frame counts, and multimodal projected features for static media. Dynamic inputs: `image_sizes`, frame count, selected best resolution, position IDs during decode, cache positions inside delegated generation logic.

## 8. Preprocessing and input packing

Processor contract:

- Text may contain `<image>` and `<video>` markers. Processor expands each marker into repeated special tokens before tokenization.
- Image output includes `pixel_values`, `image_sizes`, `input_ids`, and `attention_mask`.
- Video output includes `pixel_values_videos`, `input_ids`, and `attention_mask`; this family does not pass `image_sizes_videos` to the model.
- The processor checks special multimodal token counts after tokenization.

Image placeholder count:

```text
best_h,best_w = select_best_resolution(original_h, original_w, image_grid_pinpoints)
scale_h = best_h / processed_h
scale_w = best_w / processed_w
patches_h = processed_h / patch_size
patches_w = processed_w / patch_size
unpadded_features,newline_features = aspect-ratio unpadding count
base_features = patches_h * patches_w + num_additional_image_tokens
num_image_tokens = unpadded_features + newline_features + base_features
if strategy == "default": num_image_tokens -= 1
```

Video placeholder count:

```text
height,width = processed frame size
num_frames = pixel_values_videos[0].shape[0]
num_image_tokens = (height // patch_size) * (width // patch_size)
num_video_tokens = (num_image_tokens // 4) * num_frames
```

For the default video processor and CLIP patch size, this is `((224/14)*(224/14)/4)*F = 64*F` video tokens. The model's config default `video_seq_length=288` therefore should not be treated as a universal count; an 8-frame default processed video produces 512 tokens, and 32 frames produces 2048 tokens.

Runtime stitch:

```text
inputs_embeds = token_embedding(input_ids)
image_features = cat(projected packed image features)
video_features = cat(flattened projected per-video features)
image_mask = (input_ids == image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
video_mask = (input_ids == video_token_id).unsqueeze(-1).expand_as(inputs_embeds)
inputs_embeds = masked_scatter(inputs_embeds, image_mask, image_features)
inputs_embeds = masked_scatter(inputs_embeds, video_mask, video_features)
```

Generation controller behavior: no special forced decoder IDs are implemented in this wrapper. End-to-end parity still needs tokenizer/chat-template parity and generation sampling/beam settings from standard Transformers generation.

## 9. Graph rewrite / lowering opportunities

### Rewrite: CLIP patch Conv2d -> Linear

Source pattern: `Conv2d(C -> hidden, kernel_size=patch_size, stride=patch_size, padding=0, bias=False)` followed by flatten and transpose.

Replacement:

```text
PatchExtract(NCHW, patch, patch) -> FlattenPatch(C*patch*patch) -> MatMul(weight_flat.T)
```

Preconditions:

- `kernel_size == stride == patch_size`
- `padding == 0`, `dilation == 1`, `groups == 1`
- Input H/W divisible by patch size after preprocessing
- Preserve NCHW flatten order

Weight transform:

```python
w_linear = conv.weight.reshape(out_channels, in_channels * patch * patch)
```

Failure cases: non-CLIP towers, non-divisible dynamic frame sizes, altered conv bias/groups/dilation, or layout pass that changes channel/spatial axes without rewriting extraction order. Parity test: compare CLIP embedding output before position add for random NCHW images.

### Rewrite: video AvgPool2d over patch grid -> token block pooling

Source pattern: sequence `[B*F, H*W, D] -> view [B*F,H,W,D] -> NCHW -> AvgPool2d(2,2) -> sequence`.

Replacement:

```text
Reshape tokens to [B*F,H/2,2,W/2,2,D] -> mean over 2x2 block -> flatten spatial
```

Preconditions:

- `spatial_pool_mode == "average"`
- `spatial_pool_stride == 2`
- CLS already removed
- Token count is a perfect square and H/W divisible by 2
- No layout translation across the axis-sensitive view/permute unless axes are rewritten exactly

Failure cases: max/conv pool modes, odd grids, `"full"` strategy with CLS included, or non-square token grids.

### Rewrite: multimodal projector as GEMM-GELU-GEMM

Source pattern: two dense linears with activation.

Replacement:

```text
GEMM(features, linear_1.T, bias) -> GELU -> GEMM(hidden, linear_2.T, bias)
```

Preconditions: dense projector weights, known `vision_feature_layer` count, projector bias flag honored. Failure cases: conv pool out channels not equal to expected projector input.

### Rewrite: placeholder masked_scatter -> indexed copy

Source pattern: boolean mask expanded to embedding width followed by `masked_scatter`.

Replacement:

```text
indices = where(input_ids == token_id)
ScatterRows(inputs_embeds, indices, feature_rows)
```

Preconditions:

- `input_ids` available, not `inputs_embeds`-only placeholder equality mode
- Number of indices equals feature row count for each modality
- Row-major feature order matches processor token expansion order

Failure cases: mixed image/video ordering bugs, padded batches with missing placeholders, or feature/token count mismatch. Parity test: construct prompts with image then video, video then image, and multiple videos/images.

### Layout guard: AnyRes image pack/unpad

This region should be protected by `no_layout_translation()` initially. The source uses shape views and axis-specific `permute(4,0,2,1,3)`, `flatten(1,2)`, `flatten(2,3)`, and channel-first unpadding. A layout pass would need exact rewrites for all axes and for newline concatenation on the last spatial width axis.

## 10. Kernel fusion candidates

Highest priority:

- Text decoder RMSNorm + QKV/GQA attention + RoPE + cache update for Llama/Mistral delegated models. This dominates prefill/decode once vision features are cached.
- Compact GQA FlashAttention for prefill and decode. 34B and Mistral variants require KV-head compact storage and repeat-free kernels.
- Multimodal projector GEMM-GELU-GEMM. Image/video prefixes can be large; this is an easy isolated parity target.
- Video spatial pooling over token grids. Default 32-frame video can produce large frame-token batches before pooling.

Medium priority:

- CLIP patch embedding Conv2d-to-GEMM and CLIP encoder LayerNorm/attention/MLP fusions.
- AnyRes pack/unpad/indexed newline kernel. Useful for reducing CPU/GPU synchronization if image features are produced on GPU.
- Placeholder indexed row copy. It is small but correctness-critical and cleaner than boolean `masked_scatter`.
- Last-token-only LM head for decode via `logits_to_keep`.

Lower priority:

- Max/conv video pool modes. They are config-supported but not the common representative path.
- `inputs_embeds` placeholder equality mode.
- Full OneVision-style video newline/interpolation behavior; it belongs to `llava_onevision`, not this family.

## 11. Runtime staging plan

Stage 1: parse `LlavaNextVideoConfig`, load dense weights, and instantiate delegated CLIP plus Llama/Mistral/Yi decoder configs. Reject unsupported text model families initially.

Stage 2: implement and validate video-only feature path with random tensors: CLIP selected hidden state stub or real tiny CLIP, default CLS removal, average pool, projector, flatten by frame.

Stage 3: reuse/compose LLaVA-NeXT image path for AnyRes images, including unpadding and image newline. Validate against existing `llava_next` audit tests where possible.

Stage 4: implement placeholder expansion/stitch for image, video, and mixed prompts using `input_ids` masks only. Defer `inputs_embeds` equality mode.

Stage 5: run full prefill parity on tiny-random checkpoint: image-only, video-only, and mixed image+video prompts.

Stage 6: enable cached decode with delegated text cache. Confirm multimodal tensors are consumed only before cache creation or when cache is disabled.

Stage 7: optimize CLIP/projector/pooling and text attention with guarded rewrites/fusions.

Stage 8: expand admission to 34B/Yi and Mistral 32K variants, including GQA and RoPE/rope-scaling coverage.

## 12. Parity and validation plan

- Config resolution tests: source defaults, tiny-random, 7B, 7B-32K, 34B, and 34B-DPO; verify token IDs, pool mode/stride, delegated decoder dimensions, and feature-selection policy.
- Processor tests: image placeholder counts for square, wide, and tall images; video placeholder counts for 1, 8, and 32 frames at default 224; mixed image/video prompts.
- Custom op tests: `unpad_image`, video average pooling rewrite, feature split/flatten order, and indexed-copy replacement for `masked_scatter`.
- Vision/projector parity: tiny-random real CLIP/projector for image and video features, tolerances fp32 `1e-5` absolute/relative, fp16/bf16 `5e-2` for full path unless tighter local evidence supports it.
- Single decoder layer parity: delegated Llama/Mistral layer with stitched embeddings and attention mask.
- Prefill logits parity: tiny-random image-only, video-only, and mixed multimodal prompts, checking `logits_to_keep=0` and last-token-only.
- Decode parity: prefill with media, then generate one token with `past_key_values`; verify no vision tensors are required on the second cached step.
- End-to-end smoke: one representative 7B config in reduced precision with known media and deterministic greedy decode, after operator parity is stable.

## 13. Performance probes

- Processor throughput: image AnyRes preprocessing, video decode/frame sampling outside model, video resize/crop/normalize.
- CLIP vision throughput separately for image patches and video frames.
- Video pool/projector throughput versus frame count and processed frame size.
- AnyRes image pack/unpad/newline throughput and CPU/GPU transfer behavior.
- Placeholder stitch cost versus multimodal token count.
- Prefill-only latency by expanded sequence length: text-only, image-only, 8-frame video, 32-frame video, mixed media.
- Decode tokens/sec with existing multimodal KV cache.
- KV cache memory by delegated decoder family and multimodal prefix length.
- Attention backend comparison: eager/SDPA/FlashAttention for CLIP and text decoder.
- GQA cache memory/performance sweep for Mistral 7B-32K and 34B.
- Pool mode sweep if non-average modes are admitted.

## 14. Skip/defer list

- Training, loss parity beyond basic shifted LM loss smoke, gradient checkpointing, and dropout behavior.
- `inputs_embeds`-only placeholder equality mode.
- Conv video pool with `spatial_pool_out_channels != vision_config.hidden_size` until source-compatible configs are identified.
- Non-default `vision_feature_select_strategy="full"` for CLIP-with-CLS video unless token square/pooling behavior is proven.
- Remote-code or `llava_next_video2`/Qwen2 variants; they are separate model families/classes.
- Quantized loading/provider-specific formats.
- Multi-GPU tensor parallel and speculative decoding.
- OneVision-specific video newline and `vision_aspect_ratio` downsampling behavior.

## 15. Final implementation checklist

- [ ] Parse `LlavaNextVideoConfig` and resolve delegated `vision_config`/`text_config`.
- [ ] Load dense CLIP, projector, text decoder, LM head, and `image_newline` weights.
- [ ] Reuse/adapt LLaVA-NeXT AnyRes image preprocessing and pack/unpad path.
- [ ] Implement video processor contract for `[B,F,C,H,W]` pixel tensors and frame-count placeholder expansion.
- [ ] Implement CLIP selected hidden-state extraction with int/list `vision_feature_layer`.
- [ ] Implement `vision_feature_select_strategy="default"` CLS removal and reject unsafe `"full"` video cases initially.
- [ ] Implement default video average pooling over square patch-token grids.
- [ ] Implement multimodal projector GEMM-GELU-GEMM.
- [ ] Implement image/video placeholder count checks and indexed row copy into embeddings.
- [ ] Compose delegated Llama/Mistral/Yi text decoder prefill with compact KV cache.
- [ ] Implement `prepare_inputs_for_generation` media forwarding semantics.
- [ ] Add tiny-random parity tests for video features, image features, mixed stitch, prefill logits, and one-step decode.
- [ ] Add config sweep/admission tests for 7B, 7B-32K, 34B, and 34B-DPO.
- [ ] Benchmark processor, CLIP, video pool/projector, prefill, decode, and KV memory separately.
