# Aya Vision Transformers Audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model family: aya_vision
Primary runtime target: multimodal image+text conditional generation
Config source: local Transformers source defaults plus HF config/processor snapshots. Official CohereForAI/aya-vision-* repos returned 401 from this environment, so production configs were inspected through open mirrors and one hf-internal-testing namespace mirror.
Source files inspected:
- transformers/src/transformers/models/aya_vision/configuration_aya_vision.py
- transformers/src/transformers/models/aya_vision/modeling_aya_vision.py
- transformers/src/transformers/models/aya_vision/modular_aya_vision.py
- transformers/src/transformers/models/aya_vision/processing_aya_vision.py
- transformers/src/transformers/models/got_ocr2/image_processing_got_ocr2.py
- transformers/src/transformers/models/got_ocr2/image_processing_pil_got_ocr2.py
- transformers/src/transformers/models/siglip/configuration_siglip.py
- transformers/src/transformers/models/siglip/modeling_siglip.py
- transformers/src/transformers/models/cohere/modeling_cohere.py
- transformers/src/transformers/models/cohere2/modeling_cohere2.py
Snapshots written under: agents/plans/transformers/aya_vision/_sources/
Any missing files or assumptions: modeling_aya_vision.py is generated from modular_aya_vision.py; future source edits should target the modular file. No remote-code files were required for the inspected native source path.
```

Primary source URLs:
- [configuration_aya_vision.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/aya_vision/configuration_aya_vision.py)
- [modeling_aya_vision.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/aya_vision/modeling_aya_vision.py)
- [modular_aya_vision.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/aya_vision/modular_aya_vision.py)
- [processing_aya_vision.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/aya_vision/processing_aya_vision.py)
- [GotOcr2 image processor](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/got_ocr2/image_processing_got_ocr2.py)
- [SigLIP modeling](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/siglip/modeling_siglip.py)

Config snapshots:
- `hf-internal-testing/namespace-CohereForAI-repo_name_aya-vision-8b`
- `unsloth/aya-vision-8b`
- `mlx-community/aya-vision-8b-{4bit,6bit,8bit}`
- `unsloth/aya-vision-32b`
- `unsloth/aya-vision-32b-bnb-4bit`

## 2. High-level architecture

Aya Vision is a SigLIP vision encoder plus a custom downsampling SwiGLU multimodal projector plus a delegated causal text decoder. The wrapper owns the final LM head; it does not instantiate `CohereForCausalLM` or `Cohere2ForCausalLM`.

```text
image/text processor -> tiled pixel_values + structured input_ids
pixel_values -> SigLIP vision tower hidden states -> feature selection
selected vision tokens -> pixel-shuffle downsample projector -> text-width image features
input_ids -> token embeddings
IMG_PATCH placeholder mask -> masked_scatter(image features into embeddings)
stitched inputs_embeds -> delegated text decoder prefill -> Aya lm_head -> logits/sampling
decode steps -> delegated text decoder only with KV cache -> Aya lm_head -> logits/sampling
```

Stage decomposition:
- CPU/data pipeline: image loading, optional tiled crop packing, resize, rescale, normalize, tokenizer/chat template, and `<image>` replacement with structured image tokens.
- Cacheable vision prefix: SigLIP tower plus projector for each image tile can be computed independently of text decode when tile order and feature-layer policy are fixed.
- Prefix construction: only `<|IMG_PATCH|>` token positions are overwritten by image features; start/end/tile-label tokens stay normal text tokens.
- Prefill: delegated `cohere2` or `cohere` decoder consumes the stitched embedding sequence and builds a text KV cache over image and text tokens.
- Decode: `prepare_inputs_for_generation` forwards `pixel_values` only on the first generation iteration, or when cache is disabled.

Independently optimizable pieces are the GotOcr2-style image packer, SigLIP vision tower, projector pixel shuffle, indexed placeholder stitch, delegated text prefill, delegated text decode, and last-token LM head.

## 3. Important config dimensions

Source defaults from `AyaVisionConfig`:

| Field | Source default | Runtime significance |
|---|---:|---|
| `vision_config.model_type` | `siglip_vision_model` | Delegates vision ops to SigLIP vision. |
| `vision_config.hidden_size` | 1152 | Projector input channel per raw patch token. |
| `vision_config.image_size` | 384 | Source default only; inspected checkpoints use 364. |
| `vision_config.patch_size` | 14 | SigLIP Conv2d patch stride. |
| `vision_config.num_hidden_layers` | 26 | Source default; inspected checkpoints use 27. |
| `text_config.model_type` | `cohere2` | Delegated decoder unless checkpoint overrides to `cohere`. |
| `vision_feature_layer` | -1 | Hidden-state layer selected from SigLIP. |
| `vision_feature_select_strategy` | `full` | Keeps all SigLIP patch tokens; no CLS token exists for SigLIP anyway. |
| `downsample_factor` | 2 | Projector converts 26x26 SigLIP tokens into 13x13 text image tokens. |
| `adapter_layer_norm_eps` | 1e-6 | Projector LayerNorm epsilon. |
| `image_token_index` | 255036 | Model placeholder id for `<|IMG_PATCH|>` in 8B configs. |
| `tie_word_embeddings` | true | Aya wrapper ties `lm_head.weight` to delegated text embeddings when enabled. |

Representative config sweep:

| Checkpoint/config source | Basis | Text decoder | Layers | Hidden | Heads/KV | MLP | RoPE theta | Vision | Projector | Image token | Dtype | Notes |
|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|---|---|
| `hf-internal-testing/...aya-vision-8b` | open internal mirror | `cohere2` | 32 | 4096 | 32/8 | 14336 | 50000 | SigLIP 1152, 27L, patch 14, image 364 | 28672 | 255036 | fp16 | Has `max_position_embeddings=8192`; no quant metadata. |
| `unsloth/aya-vision-8b` | open mirror of 8B | `cohere2` | 32 | 4096 | 32/8 | 14336 | 50000 | SigLIP 1152, 27L, patch 14, image 364 | 28672 | 255036 | bf16 | Native dense mirror; `sliding_window=4096` present. |
| `mlx-community/aya-vision-8b-4bit/6bit/8bit` | open MLX mirrors | `cohere2` | 32 | 4096 | 32/8 | 14336 | 50000 | SigLIP 1152, 27L, patch 14, image 364, `skip_vision=true` in config | 28672 | 255036 | fp16 | `quantization` is MLX loader metadata, not native HF module behavior. |
| `unsloth/aya-vision-32b` | open mirror of 32B | `cohere` | 40 | 8192 | 64/8 | 24576 | 4000000 | SigLIP 1152, 27L, patch 14, image 364 | 49152 | 255022 | bf16 | 32B uses `cohere`, not `cohere2`; image token id differs. |
| `unsloth/aya-vision-32b-bnb-4bit` | bitsandbytes mirror | `cohere` | 40 | 8192 | 64/8 | 24576 | 4000000 | same as 32B | 49152 | 255022 | bf16 compute | `quantization_config` is loading/provider work. |

Processor/preprocessor fields from inspected snapshots:

| Field | Value | Runtime significance |
|---|---:|---|
| `processor_config.patch_size` | 28 | Placeholder token grid stride after projector downsample, not SigLIP Conv2d patch size. |
| `processor_config.img_size` | 364 | Per-tile image side; `364 / 14 = 26`, then projector downsample gives `13 * 13 = 169` image feature tokens per tile. |
| `preprocessor_config.image_processor_type` | `GotOcr2ImageProcessor` | Aya uses GotOcr2-style resize/tiled crop processor through AutoImageProcessor mapping. |
| `preprocessor_config.size` | 364x364 | Pixel tensor tile shape. |
| `preprocessor_config.crop_to_patches` | false in files | AyaVisionProcessor default `images_kwargs` sets `crop_to_patches=True`, so normal processor calls tile by default unless overridden. |
| `image_mean/std` | `[0.5,0.5,0.5]` | Config-derived normalization, not source GotOcr2 default. |

## 3a. Family variation traps

- 8B and 32B do not share the same decoder family. 8B mirrors use `cohere2`; 32B mirrors use `cohere`.
- Aya wrapper owns the LM head and does not apply delegated Cohere/Cohere2 `logit_scale`. The inspected `AyaVisionForConditionalGeneration.forward` computes `lm_head(hidden_states)` only.
- `image_token_index` differs: 8B uses 255036 and 32B uses 255022. Do not hard-code the 8B id.
- Projector input width is `vision_hidden * downsample_factor^2`: 1152 * 4 = 4608. `alignment_intermediate_size` is 28672 for 8B and 49152 for 32B; it is split in half by SwiGLU before `linear_2`.
- The source config class does not declare `alignment_intermediate_size`, `alignment_activation_fn`, `projector_hidden_act`, or `max_splits_per_img`, but the modeling source reads `alignment_intermediate_size` via `getattr`. Other listed fields are ignored by native Aya modeling at this commit.
- Processor placeholder count is coupled to projector pixel shuffle. For 364x364, SigLIP emits 26x26 = 676 tokens per tile and projector emits 13x13 = 169 tokens per tile.
- The processor emits start/end and tile label tokens around image patch placeholders. Only `<|IMG_PATCH|>` positions are replaced by image features.
- GotOcr2 `preprocessor_config.crop_to_patches=false` is overridden by `AyaVisionProcessorKwargs` default `crop_to_patches=True` for normal Aya processor calls.
- Tiled images include local crops first and a thumbnail/global tile last when more than one tile is selected. Prompt token order mirrors that tile order.
- `vision_feature_layer` may be an int or list. A list concatenates selected SigLIP hidden states on the feature dimension before projector, so `linear_1` input must scale by `len(layers)`.
- SigLIP source is NCHW through Conv2d patch embedding, then token-major `[B,T,C]`. NHWC/channel-last optimization should be restricted to the image preprocessing/patch embedding region with explicit axis rewrites.
- Cohere2 source has sliding/full layer types. In the inspected source, `apply_rotary_pos_emb` is called only when a layer has `sliding_window is not None`; full-attention layers in `cohere2` skip the attention-local RoPE call.
- Cohere and Cohere2 use an interleaved/even-odd RoPE variant that differs from LLaMA.
- Quantized mirrors include MLX or bitsandbytes metadata. Native Aya source still declares dense `nn.Linear`, `nn.Embedding`, Conv2d, and norm modules; quant metadata is loader/provider scope.

## 4. Operator coverage checklist

Tensor/layout ops:
- Image preprocessing: RGB conversion, optional tile-crop packing, bicubic resize, rescale, normalize, NCHW batch tensor construction.
- Tile flattening: image processor may return `pixel_values [sum_tiles, 3, 364, 364]` for a batch with multiple images.
- SigLIP patch embedding: `Conv2d(3 -> 1152, kernel=stride=14, padding="valid")`, flatten spatial axes, transpose to `[tiles, 676, 1152]`.
- Hidden-state selection: tuple indexing, optional `[:, 1:]` if strategy is `"default"`, concat on last dim for multi-layer selection.
- Projector pixel shuffle: reshape/permute sequence grid from `[tiles, 676, 1152]` to `[tiles, 13, 13, 4608]` for downsample factor 2.
- Projector output flattening is implicit for `masked_scatter`: output is `[tiles, 13, 13, text_hidden]`, flattened in row-major order when scattered.
- Text embedding lookup: `input_ids [B,S] -> [B,S,H_text]`.
- Placeholder mask: equality on `input_ids == image_token_id`, `unsqueeze`, `expand_as`, count/numel validation.
- Embedding stitch: `masked_scatter` of image features into token embeddings.
- Logits slice: `hidden_states[:, slice_indices, :]` where `logits_to_keep` may be int tail count or tensor indices.

Neural network primitives:
- SigLIP LayerNorm, full noncausal MHA, Linear Q/K/V/O with bias, GELU/tanh-like configured MLP, residual adds.
- Projector LayerNorm over 4608, `Linear(4608 -> alignment_intermediate_size, bias=True)`, split/chunk on last dim, SiLU, multiply, `Linear(alignment_intermediate_size/2 -> text_hidden, bias=True)`.
- Delegated Cohere/Cohere2 LayerNorm, biasless decoder MLP, causal attention, residual adds, final norm.
- Aya LM head: `Linear(text_hidden -> vocab_size, bias=False)`, tied to text embeddings when `tie_word_embeddings=true`.

Attention primitives:
- Vision: noncausal full self-attention over 676 patch tokens per tile, no KV cache.
- Text: delegated causal self-attention with MHA/GQA; 8B/32B inspected configs both use GQA.
- Cohere2 may require both sliding-window and full causal masks according to `layer_types`.
- Eager attention math uses QK matmul, additive mask, fp32 softmax, dropout only in training, and AV matmul.

Position/rotary ops:
- SigLIP learned patch position embedding with optional bicubic interpolation for nonstandard spatial sizes.
- Cohere/Cohere2 interleaved RoPE for text decoder, with even/odd `rotate_half`.
- Position IDs default to `arange(seq_len) + past_key_values.get_seq_length()`.

Generation/cache ops:
- Delegated DynamicCache creation when `use_cache` and no cache is supplied.
- Per-layer text cache shapes before GQA repeat: `[B, num_key_value_heads, T_cache, head_dim]`.
- Generation preparation drops `pixel_values` after first cached iteration.
- Image features/prefix embeddings are independently cacheable, but they are not KV caches until text prefill runs.

Preprocessing-coupled ops:
- GotOcr2 tiling grid selection, crop extraction, thumbnail insertion.
- Structured placeholder string creation: BOI, local `TILE_i`, global `TILE_GLOBAL`, repeated `<|IMG_PATCH|>`, EOI.
- Optional `mm_token_type_ids` creation by processor; Aya model forward does not consume it.

Quantized/packed metadata:
- MLX `quantization` and bitsandbytes `quantization_config` should be rejected or routed to explicit loader/provider support. They are not core native Aya graph ops.

## 5. Layer/block breakdown

SigLIP vision tower, repeated `vision_layers`:

```text
pixel_values [N_tiles, 3, 364, 364]
patch = Conv2d(3 -> 1152, kernel=stride=14, bias=True)  # [N_tiles,1152,26,26]
x = patch.flatten(2).transpose(1, 2)                    # [N_tiles,676,1152]
x = x + learned_position_embedding[676]
for layer:
  y = LayerNorm(x)
  q,k,v = Linear(1152 -> 1152, bias=True)
  y = full_self_attention(q,k,v)
  x = x + Linear(1152 -> 1152, bias=True)(y)
  y = LayerNorm(x)
  y = Linear(1152 -> 4304) -> activation -> Linear(4304 -> 1152)
  x = x + y
x = post_layernorm(x)
```

Aya projector for normal 8B/32B SigLIP features:

```text
image_features [N_tiles, 676, 1152]
grid = reshape to [N_tiles, 26, 26, 1152]
grid = pixel_shuffle_downsample_factor_2(grid)          # [N_tiles,13,13,4608]
y = LayerNorm(4608, eps=1e-6)
y = Linear(4608 -> 28672 or 49152, bias=True)
x, gate = chunk(y, 2, dim=-1)
y = silu(gate) * x                                      # [N_tiles,13,13,14336 or 24576]
y = Linear(14336 -> 4096 or 24576 -> 8192, bias=True)
```

Embedding stitch:

```text
inputs_embeds = text_embedding(input_ids)                # [B,S,H_text]
mask = (input_ids == config.image_token_id)              # [B,S]
mask = mask.unsqueeze(-1).expand_as(inputs_embeds)
assert inputs_embeds[mask].numel() == image_features.numel()
inputs_embeds = inputs_embeds.masked_scatter(mask, image_features)
```

Delegated decoder:
- 8B uses `Cohere2Model`: parallel residual input norm, GQA attention, SwiGLU MLP, final norm; layer mask alternates sliding/full according to `layer_types`.
- 32B uses `CohereModel`: parallel residual input norm, GQA attention, SwiGLU MLP, final norm.

Aya conditional-generation head:

```text
hidden = delegated_decoder(inputs_embeds, attention_mask, position_ids, cache)
logits = lm_head(hidden[:, logits_to_keep, :])           # no Aya-level logit_scale multiply
```

## 6. Attention requirements

Vision attention:

| Requirement | Aya/SigLIP behavior |
|---|---|
| Causal/noncausal | Noncausal full self-attention. |
| Cache | None. Vision outputs can be cached as encoder/projector features, not KV. |
| Heads/head dim | 16 heads, 1152 hidden, head dim 72 in inspected configs. |
| Mask | Normally no mask for fixed image patches. |
| Backend | SigLIP supports eager/SDPA/Flash/Flex through `ALL_ATTENTION_FUNCTIONS`. |

Text attention:

| Requirement | 8B `cohere2` | 32B `cohere` |
|---|---|---|
| Causal/noncausal | Causal | Causal |
| Self/cross | Self-attention only | Self-attention only |
| Heads/KV/head_dim | 32/8/128 | 64/8/128 |
| MHA/GQA | GQA | GQA |
| Masking | `create_causal_mask` plus `create_sliding_window_causal_mask` by layer type | causal mask |
| Sliding/local | Yes when layer type is `sliding_attention`, window 4096 in dense 8B mirror | No Cohere-family sliding in inspected 32B config |
| RoPE | Cohere2 interleaved/even-odd, source applies inside attention only for sliding-window layers | Cohere interleaved/even-odd |
| Cache layout | `[B, 8, T, 128]` K/V before repeat | `[B, 8, T, 128]` K/V before repeat |

Cache distinctions:
- Image tile features from SigLIP/projector are independently cacheable prefix features.
- Text prefill cache is the delegated decoder autoregressive KV cache after image/text embedding stitch.
- Decode does not re-run vision when cache is used.

Eager fallback is fine for parity but too slow for production. A production path needs compact-GQA causal attention that can handle Cohere/Cohere2 RoPE and, for 8B, layer-specific sliding-window masks.

## 7. Position encoding and custom math

Projector pixel shuffle is custom and parity-sensitive:

```python
def aya_pixel_shuffle(image_features, downsample_factor=2):
    # image_features: [B, S, D], S must be a square grid.
    b, s, d = image_features.shape
    h = w = int(s ** 0.5)
    x = image_features.reshape(b, w, h, d)
    x = x.reshape(b, w, h // downsample_factor, d * downsample_factor)
    x = x.permute(0, 2, 1, 3)
    x = x.reshape(b, h // downsample_factor, w // downsample_factor, -1)
    x = x.permute(0, 2, 1, 3)
    return x
```

For 364x364 and SigLIP patch 14, `S=676`, `h=w=26`, and the output has spatial shape 13x13 with 4608 channels. The function returns rank-4 `[tiles, 13, 13, C]`, and source relies on later flattening behavior in `masked_scatter`.

Cohere/Cohere2 RoPE uses interleaved frequencies and even/odd rotation:

```python
def cohere_rotate_half(x):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack([-x2, x1], dim=-1).flatten(-2)

def cohere_apply_rope(q, k, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    qf, kf = q.float(), k.float()
    return ((qf * cos) + (cohere_rotate_half(qf) * sin)).to(q.dtype), \
           ((kf * cos) + (cohere_rotate_half(kf) * sin)).to(k.dtype)
```

Precompute candidates:
- SigLIP learned positions are weights for the configured 26x26 grid; interpolation depends on runtime height/width only if nonstandard sizes are admitted.
- Projector shuffle shape factors are static for normal checkpoints.
- RoPE inverse frequencies depend on text config `rope_theta` and `head_dim`; cos/sin rows depend on runtime `position_ids` and cache length.

## 8. Preprocessing and input packing

Processor output contract:
- `input_ids [B,S_expanded]`, `attention_mask [B,S_expanded]`.
- `pixel_values [sum_image_tiles, 3, 364, 364]` after image preprocessing.
- Optional `mm_token_type_ids` if requested; model ignores it.

Image tiling:
- `AyaVisionProcessor` fetches/loads images, flattens image lists, then calls GotOcr2ImageProcessor with default `crop_to_patches=True`.
- GotOcr2 tiling chooses a `(num_columns, num_rows)` grid between `min_patches=1` and `max_patches=12`, resizes the image to that tiled canvas, slices local 364x364 tiles, and adds a thumbnail/global tile when there is more than one local tile.
- `num_patches` returned by the image processor includes that thumbnail/global tile.

Placeholder expansion:

```text
img_patches_per_tile = (img_size // processor_patch_size)^2 = (364 // 28)^2 = 169
if num_patches > 1:
  for idx in 1..num_patches-1:
    append "TILE_{idx}" + 169 * "<|IMG_PATCH|>"
append "TILE_GLOBAL" + 169 * "<|IMG_PATCH|>"
wrap with "<|START_OF_IMG|>" and "<|END_OF_IMG|>"
```

Token count per image:

```text
num_image_tokens = 2 + num_patches * (169 + 1)
```

This counts BOI/EOI plus one tile-label token per tile plus 169 patch placeholders per tile. Only the 169 * `num_patches` `<|IMG_PATCH|>` positions are replaced by image features.

Stitching:
- Model computes `image_features = get_image_features(pixel_values).pooler_output`.
- It casts image features to `inputs_embeds` dtype/device.
- It builds a placeholder mask from `input_ids == config.image_token_id`; if only `inputs_embeds` are supplied, it compares embeddings against the image-token embedding. DinoML should defer the embeddings-only detection path unless required.
- `masked_scatter` requires feature count and placeholder count to match exactly.

Differences from LLaVA-style families:
- LLaVA usually expands one `<image>` into a flat run of identical placeholder tokens. Aya inserts structured start/end and tile/global tokens around patch placeholders.
- Aya's projector downsampling is a spatial pixel-shuffle-style packing from 26x26x1152 to 13x13x4608 before SwiGLU, not LLaVA's typical Linear-GELU-Linear projector over patch tokens.
- Aya's image processor uses GotOcr2-style dynamic tiling and thumbnail insertion rather than a single fixed square crop in normal calls.

CPU/data-pipeline work can own image tiling and tokenizer string construction initially. GPU/runtime work should own SigLIP, projector, indexed copy/stitch, prefill, decode, and logits.

## 9. Graph rewrite / lowering opportunities

### Rewrite: SigLIP non-overlap Conv2d patch embed -> Linear

Source pattern:

```text
Conv2d(3 -> 1152, kernel_size=14, stride=14, padding="valid")
flatten(2).transpose(1, 2)
```

Replacement:

```text
NCHW WindowFlatten([14,14,3]) -> MatMul(weight_flat.T) -> BiasAdd -> [B,676,1152]
```

Preconditions:
- `kernel_size == stride == 14`.
- `padding == "valid"`/zero, `dilation == 1`, `groups == 1`.
- Input is NCHW or an explicit NHWC pass has transformed both input and weight layout.
- Runtime height/width produce the expected floor grid or the position embedding interpolation path is implemented.

Weight transform:

```python
w = conv.weight.reshape(out_channels, in_channels * kh * kw)
```

Failure cases:
- Nonstandard image sizes without position interpolation.
- Layout pass changes flatten order.
- Future vision configs with groups, padding, or different channels.

Parity test sketch: compare Conv2d+flatten+transpose to rewritten GEMM for 364x364 and representative dtype policies.

### Rewrite: projector pixel shuffle + LayerNorm as explicit downsample-pack

Source pattern: reshape/permute chain in `AyaVisionMultiModalProjector.pixel_shuffle`.

Replacement:

```text
[tiles, 26, 26, 1152] -> DownsamplePack2x2Channels -> [tiles, 13, 13, 4608] -> LayerNorm(last_dim)
```

Preconditions:
- Sequence length is a perfect square.
- Height and width are divisible by `downsample_factor`.
- Consumer is projector LayerNorm/Linear over last dim.
- Preserve source ordering exactly; do not substitute PyTorch image `pixel_shuffle`, which has a different semantic direction.

Failure cases:
- `vision_feature_layer` list changes channel width.
- Non-square or odd-sized grids.

Parity test sketch: random `[N,676,1152]`, compare source shuffle output elementwise.

### Rewrite: Aya placeholder `masked_scatter` -> indexed copy

Source pattern:

```text
mask = input_ids == image_token_id
inputs_embeds = inputs_embeds.masked_scatter(mask[...,None].expand_as(inputs_embeds), image_features)
```

Replacement:

```text
indices = nonzero(input_ids == image_token_id)
copy image_features.reshape(-1, H) into inputs_embeds[indices]
```

Preconditions:
- Placeholder count equals `image_features.numel() / H`.
- Feature flatten order is source row-major over `[tile, y, x, hidden]`.
- Batch image ordering from processor is preserved.

Failure cases:
- `inputs_embeds`-only image-token detection.
- Mismatched tokenizer/config image token id.
- Dynamic tile counts without explicit per-sample descriptors in the runtime boundary.

Parity test sketch: one image, two images, and multi-tile image prompts; compare stitched embeddings.

### Rewrite: cacheable vision/projector prefix

Source pattern:

```text
pixel_values -> SigLIP hidden_states[-1] -> Aya projector -> masked scatter before decoder
```

Replacement: split into a vision-prefix graph returning image features plus a prefill graph that accepts image features and token positions.

Preconditions:
- Same processor tile order and same `vision_feature_layer`/strategy.
- Same projector weights and dtype.
- Text prompt positions are known after tokenization.

Failure cases:
- Runtime varies `vision_feature_layer`.
- Processor emits different crop/tile count for same image due to kwargs.

### Layout opportunity: guarded NCHW -> NHWC in image-local region

Candidate region: resize/rescale/normalize, Conv2d patch embed, and flatten-to-token. Axis-sensitive rewrites include channel normalization axis, crop/slice spatial axes, Conv2d input layout, and flatten/transpose consumer. The text/projector tensor sequence is token-major or NHWC-like rank-4 after shuffle; protect decoder and placeholder scatter with a conceptual `no_layout_translation()` guard unless a dedicated layout proof exists.

## 10. Kernel fusion candidates

Highest priority:
- Aya projector downsample-pack + LayerNorm + first Linear/SwiGLU: this is unique to Aya and directly controls image-token parity.
- Indexed placeholder copy: small but required for end-to-end multimodal correctness.
- Delegated GQA causal attention with KV cache: 8B and 32B both use KV heads fewer than query heads.
- Cohere/Cohere2 LayerNorm and interleaved RoPE: parity-sensitive and reused in every decoder block.
- Last-token-only Aya LM head: avoids full sequence-vocab GEMM during decode.

Medium priority:
- SigLIP patch Conv2d lowered to GEMM.
- SigLIP vision encoder LayerNorm/attention/MLP kernels for 676-token tile batches.
- Cohere2 sliding-window attention mask support for 8B.
- Packed QKV projections in delegated text decoder, with exact Q/K/V split order.
- Cacheable vision-prefix subgraph for serving.

Lower priority:
- GotOcr2 image tiling on GPU; CPU/data pipeline can own it first.
- Position-embedding interpolation for nonstandard vision sizes.
- Quantized mirror support for MLX/bitsandbytes metadata.
- Output attentions and hidden-state tuple memory optimization.

## 11. Runtime staging plan

Stage 1: parse `AyaVisionConfig`, nested SigLIP/text configs, processor configs, and image token ids. Reject unsupported quantized mirrors unless an explicit provider path is selected.

Stage 2: projector-only parity with random `[tiles,676,1152]` features for 8B and 32B dimensions. Validate pixel shuffle, LayerNorm, SwiGLU split order, and output flattening.

Stage 3: placeholder stitch parity from `input_ids`, text embeddings, and projected image features. Include one image, two images, and multi-tile prompts.

Stage 4: SigLIP vision tower parity for fixed 364x364 tiles and `vision_feature_layer=-1`, `full` strategy.

Stage 5: delegated text prefill from already-stitched `inputs_embeds`; first target 8B `cohere2`, then 32B `cohere`.

Stage 6: decode with delegated KV cache and `prepare_inputs_for_generation` behavior that omits `pixel_values` after first cached step.

Stage 7: optimize attention, projector, Conv2d-as-GEMM, indexed stitch, and vision-prefix caching behind strict guards.

Initially stub: tokenizer/chat template, image fetching, CPU image tiling internals, generation sampling policy, training loss, `inputs_embeds`-only placeholder detection, quantized loaders, and nonstandard dynamic vision sizes.

## 12. Parity and validation plan

- Config tests: parse 8B `cohere2`, 32B `cohere`, image token id difference, `alignment_intermediate_size`, and ignored historical fields.
- Processor token-count tests: verify `num_image_tokens = 2 + n_tiles * 170` and patch placeholder count `n_tiles * 169`.
- Image processor tests: known image sizes through GotOcr2 tile-count calculation; verify local tiles then thumbnail/global tile order.
- Projector custom op tests: pixel shuffle output for fp32/fp16/bf16; compare source module for 8B and 32B dimensions.
- SigLIP vision parity: 364x364 tile through patch embedding, selected hidden state, post layer norm.
- Stitch tests: mask count checks, mismatched token/features error, indexed copy parity with `masked_scatter`.
- Text decoder tests: reuse Cohere/Cohere2 single-layer, prefill, and decode cache parity; specifically include 8B sliding/full layer mask behavior.
- End-to-end prefill logits: fixed processor outputs and small prompt with one image; compare `logits_to_keep=1` and full logits.
- Decode parity: first cached generation step includes image features in cache; subsequent steps omit `pixel_values`.
- Tolerances: fp32 custom ops around `rtol=1e-4, atol=1e-5`; fp16/bf16 full-model hidden/logits around `rtol=1e-2, atol=1e-2` unless accumulation policy is matched more tightly.

No DinoML tests or benchmarks were run for this docs-only audit.

## 13. Performance probes

- CPU preprocessing throughput by image size, crop-to-patches flag, and max tile count.
- Tile count distribution and resulting expanded prompt lengths for real workloads.
- SigLIP vision throughput for `[N_tiles,3,364,364]` with N tile sweep.
- Projector throughput for 8B and 32B dimensions: 676 input tokens -> 169 output tokens per tile.
- Stitch latency versus sequence length, batch size, and image token count.
- Prefill tokens/sec including expanded image placeholder tokens.
- Decode tokens/sec with 8B `cohere2` and 32B `cohere` cache shapes.
- KV cache memory: `layers * 2 * B * kv_heads * T * 128 * dtype_size`.
- Attention backend comparison for Cohere2 sliding/full layers and Cohere GQA.
- Last-token-only logits versus full logits.
- End-to-end requests/hour with and without cached vision/projector prefixes.
- Loader/dequant time if quantized mirrors are admitted later.

## 14. Skip/defer list

- Training, labels/loss, gradients, and gradient checkpointing.
- Beam search internals beyond delegated cache reorder.
- Output attentions and full hidden-state capture optimization.
- `inputs_embeds`-only placeholder detection by embedding equality.
- Dynamic/non-364 vision sizes and SigLIP position interpolation.
- GPU implementation of image fetching/tiling; keep in CPU/data pipeline first.
- MLX, bitsandbytes, GGUF, or other quantized mirror loading unless a provider path is explicitly in scope.
- Tensor parallel and pipeline parallel plans.
- Remote-code or non-native variants.

## 15. Final implementation checklist

- [ ] Parse `AyaVisionConfig` and nested `vision_config`/`text_config`.
- [ ] Admit `cohere2` 8B and `cohere` 32B decoder variants through separate delegated text paths.
- [ ] Load SigLIP, projector, delegated text model, and tied Aya LM head weights.
- [ ] Preserve checkpoint-specific `image_token_index`.
- [ ] Implement/validate GotOcr2 tile-count and Aya structured placeholder contract in data pipeline fixtures.
- [ ] Implement SigLIP 364x364 vision tower path.
- [ ] Implement Aya pixel-shuffle downsample pack.
- [ ] Implement projector LayerNorm -> Linear -> SwiGLU -> Linear.
- [ ] Implement image feature flatten/order contract for scatter.
- [ ] Implement placeholder count validation and indexed-copy stitch.
- [ ] Implement delegated text prefill from stitched `inputs_embeds`.
- [ ] Implement delegated decode with compact GQA KV cache.
- [ ] Implement Cohere/Cohere2 interleaved RoPE and Cohere2 sliding-window masks as required by selected checkpoint.
- [ ] Implement Aya LM head without delegated `logit_scale`.
- [ ] Add projector, stitch, vision, prefill, and decode parity tests.
- [ ] Add Conv2d patch-embed-to-GEMM rewrite behind strict layout guards.
- [ ] Add cacheable vision/projector prefix graph boundary.
- [ ] Benchmark preprocessing, vision, projector, stitch, prefill, decode, logits, and KV memory separately.
