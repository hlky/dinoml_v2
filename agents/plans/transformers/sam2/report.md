# Transformers SAM2 Operator and Integration Report

## 1. Source basis

```text
Transformers commit/version:
  Local checkout transformers at b75feb2af64c3e29cbbc1bd859958c5432cc7ed4.

Model id:
  Primary image-path sizing reference: facebook/sam2.1-hiera-tiny.
  Representative sweep: facebook/sam2.1-hiera-tiny, facebook/sam2.1-hiera-small,
  facebook/sam2.1-hiera-base-plus, facebook/sam2.1-hiera-large,
  facebook/sam2-hiera-tiny, facebook/sam2-hiera-large.

Config source:
  https://huggingface.co/facebook/sam2.1-hiera-tiny/raw/main/config.json
  https://huggingface.co/facebook/sam2.1-hiera-small/raw/main/config.json
  https://huggingface.co/facebook/sam2.1-hiera-base-plus/raw/main/config.json
  https://huggingface.co/facebook/sam2.1-hiera-large/raw/main/config.json
  https://huggingface.co/facebook/sam2-hiera-tiny/raw/main/config.json
  https://huggingface.co/facebook/sam2-hiera-large/raw/main/config.json
  Matching preprocessor_config.json and processor_config.json files were also
  downloaded where present.

Source files inspected:
  transformers/src/transformers/models/sam2/configuration_sam2.py
  transformers/src/transformers/models/sam2/modeling_sam2.py
  transformers/src/transformers/models/sam2/processing_sam2.py
  transformers/src/transformers/models/sam2/image_processing_sam2.py
  transformers/src/transformers/models/sam2/modular_sam2.py
  transformers/src/transformers/models/sam2/convert_sam2_to_hf.py

Any missing files or assumptions:
  The public Facebook SAM2/SAM2.1 checkpoints sampled are model_type
  "sam2_video" with architecture "Sam2VideoModel". This report is scoped to the
  image promptable segmentation path implemented by Sam2Model in modeling_sam2.py.
  The shared image encoder, prompt encoder, mask decoder, processor coordinate
  packing, and image processor are source-derived. Video memory attention,
  memory encoder, object pointers, and temporal tracking fields are out of scope
  for this sam2 report and should be handled by the separate sam2_video audit.
  hf-internal-testing/tiny-random-Sam2Model returned 401:
  [hf-internal-testing/tiny-random-Sam2Model](https://huggingface.co/hf-internal-testing/tiny-random-Sam2Model).
```

Authoritative source note: `modeling_sam2.py` is generated from `modular_sam2.py`; future source edits should inspect the modular file, while this report cites the generated modeling file for exact runtime code.

## 2. High-level architecture

SAM2 image inference is promptable image segmentation:

```text
RGB/resize/normalize image preprocessing
  -> Hiera image encoder + FPN neck
  -> cacheable image embeddings/high-res features
  -> point/box/mask prompt encoder
  -> two-way mask decoder
  -> low-res masks, IoU scores, object-score logits
  -> mask resize/threshold/postprocess
```

Stage decomposition:

- CPU/data pipeline: convert RGB, resize to `1024 x 1024`, rescale/normalize, optional segmentation-map resize to `256 x 256`, normalize points and boxes into the target image coordinate space, and pad ragged point prompts with label `-10`.
- Image encoder: NCHW `pixel_values` enter a `Conv2d(3 -> hidden, k=7, s=4, p=3)` patch stem, then Hiera blocks operate channel-last on `[B,H,W,C]` with window/global spatial attention and query pooling.
- FPN neck: channel-last Hiera stage outputs are permuted to NCHW, projected to `256` channels, top-down merged with nearest upsample, and reduced to three feature levels `[B,256,256,256]`, `[B,256,128,128]`, `[B,256,64,64]`.
- Cacheable image state: `get_image_features()` preprojects the first two high-res levels with `conv_s0/conv_s1` and returns flattened `[HW,B,C]`; `get_image_embeddings()` returns NCHW image embeddings for repeated prompt clicks.
- Prompt encoder: sparse points/boxes become `[B,point_batch,T,256]`; masks become dense `[B,256,64,64]`; missing masks use an expanded learned no-mask embedding.
- Mask decoder: learned object/IoU/mask tokens plus sparse prompts attend bidirectionally with the image grid, then transposed convolutions and per-mask hypernetwork MLPs emit low-res masks.

## 3. Important config dimensions

Worked example: `facebook/sam2.1-hiera-tiny` image subgraph, using config subfields from an official `sam2_video` config.

| Field | Value | Source |
|---|---:|---|
| image size | `1024 x 1024` | config/preprocessor |
| image input layout | NCHW `[B,3,1024,1024]` | source |
| patch stem | `Conv2d(3 -> 96, k=7, s=4, p=3)` | config/source |
| initial patch grid | `256 x 256` | inferred from source/config |
| Hiera stage dims | `[96,192,384,768]` | config |
| Hiera blocks | `[1,2,7,2]`, total 12 | config |
| Hiera heads | `[1,2,4,8]` | config |
| head dim | `96` in every stage for tiny | inferred from `dim_out / heads` |
| window sizes | `[8,4,14,7]` | config |
| global attention blocks | `[5,7,9]` | config |
| query pooling | stride `[2,2]` at first block of stages 1-3 | source/config |
| FPN hidden size | 256 | config |
| returned FPN feature sizes | `256^2`, `128^2`, `64^2` | config/source |
| prompt hidden size | 256 | config |
| prompt image grid | `64 x 64` | `image_size / patch_size` |
| prompt mask input size | `256 x 256` | `4 * image_size / patch_size` |
| mask decoder layers | 2 two-way blocks | config |
| mask decoder heads/head dim | 8 / 32 self-attn; 8 / 16 cross-attn | source/config |
| mask decoder MLP dim | 2048 | config |
| mask tokens | 4 mask tokens = 1 single + 3 multimask | source/config |
| low-res mask output | `[B,point_batch,1 or 3,256,256]` | source |
| dtype | `torch_dtype: float32` in sampled configs | config |
| cache support | image embeddings/high-res features are cacheable; no autoregressive KV cache | source |

Representative checkpoint sweep:

| Checkpoint | Version | Stage dims | Blocks | Heads | Windows | Global blocks | Pos bg | FPN/prompt/decoder |
|---|---|---:|---:|---:|---|---|---|---:|
| `facebook/sam2.1-hiera-tiny` | 2.1 | 96,192,384,768 | 1,2,7,2 | 1,2,4,8 | 8,4,14,7 | 5,7,9 | 7,7 | 256 |
| `facebook/sam2.1-hiera-small` | 2.1 | 96,192,384,768 | 1,2,11,2 | 1,2,4,8 | 8,4,14,7 | 7,10,13 | 7,7 | 256 |
| `facebook/sam2.1-hiera-base-plus` | 2.1 | 112,224,448,896 | 2,3,16,3 | 2,4,8,16 | 8,4,14,7 | 12,16,20 | 14,14 | 256 |
| `facebook/sam2.1-hiera-large` | 2.1 | 144,288,576,1152 | 2,6,36,4 | 2,4,8,16 | 8,4,16,8 | 23,33,43 | 7,7 | 256 |
| `facebook/sam2-hiera-tiny` | 2.0 | 96,192,384,768 | 1,2,7,2 | 1,2,4,8 | 8,4,14,7 | 5,7,9 | 7,7 | 256 |
| `facebook/sam2-hiera-large` | 2.0 | 144,288,576,1152 | 2,6,36,4 | 2,4,8,16 | 8,4,16,8 | 23,33,43 | 7,7 | 256 |

Preprocessor sweep: sampled official repos use `size={"height":1024,"width":1024}`, `mask_size={"height":256,"width":256}`, ImageNet mean/std, RGB conversion, rescale and normalize enabled, and SAM2 disables the SAM1 padding logic (`do_pad=None` in source defaults).

## 3a. Family variation traps

- Public official SAM2 repos are video configs. For the `sam2` image audit, use only `vision_config`, `prompt_encoder_config`, `mask_decoder_config`, image processor, and `Sam2Model` source. Do not treat memory encoder or object-pointer fields as required for image-only parity.
- Source layout is mixed: patch stem consumes NCHW, Hiera transformer blocks use NHWC, FPN and mask decoder use NCHW, and `get_image_features()` flattens NCHW feature maps to `[HW,B,C]`.
- Hiera patch stem is overlapping `Conv2d(k=7,s=4,p=3)`, so it is not a safe non-overlap Conv-to-Linear rewrite.
- Head dim is not equal to `hidden_size / heads` for all variants if inferred globally; use per-stage `embed_dim_per_stage[stage] / num_attention_heads_per_stage[stage]`.
- Window attention includes dynamic padding/cropping. Large uses window `[8,4,16,8]`; tiny/small/base-plus use `[8,4,14,7]`.
- `Sam2MultiScaleAttention` computes an eager `attn_weights` before optional query pooling, then calls the selected attention backend again. This pre-backend softmax result is discarded by source and is a DCE/perf trap for faithful graph capture.
- Prompt coordinate labels have two special negatives: `-1` becomes the learned not-a-point embedding, while `-10` is processor padding and is zeroed.
- If both points and boxes are supplied, `input_points.shape[1]` must equal `input_boxes.shape[1]`; boxes cannot be ragged-padded across images by the processor.
- Mask prompts must include channel dimension for the Conv2d path. The practical tensor is `[B,1,256,256]`, even where docstrings say `[B,image_size,image_size]`.
- `attention_similarity` is an additive decoder attention mask/bias. FlashAttention is explicitly bypassed for that call in favor of SDPA.
- `multimask_output=False` can still inspect all four masks internally and choose a fallback by stability if `dynamic_multimask_via_stability=True` and not training.
- SAM2 differs from prior SAM audit: Hiera replaces ViT; patch stem is overlapping; FPN returns high-res features; mask decoder adds object-score token/head and high-res skip features; image processor does not use SAM1 pad/crop metadata.

## 4. Operator coverage checklist

### Tensor/layout ops

- NCHW image input and segmentation-map/mask tensors.
- NCHW -> NHWC after patch stem; NHWC LayerNorm over final channel.
- NHWC window partition/unpartition with pad, view, permute, contiguous, and crop.
- Query pooling via `max_pool2d` after NHWC -> NCHW permute and back.
- NHWC -> NCHW FPN lateral projection and NCHW nearest upsample/add.
- Flatten/permute feature maps: `[B,C,H,W] -> [HW,B,C]` and rehydrate to `[B,C,H,W]`.
- Prompt packing: pad, cat, stack, repeat, repeat_interleave, expand, slice, gather, `torch.where`.
- Hypernetwork matmul: `[B,P,M,32] @ [B,P,32,256*256] -> [B,P,M,256,256]`.
- Postprocess variable-length list outputs for per-image masks.

### Neural network primitives

- Patch stem `Conv2d(3 -> stage0_dim, k=7, s=4, p=3, bias=True)`.
- Hiera QKV `Linear(dim -> 3*dim_out, bias=True)` with split order `[q,k,v]`; tiny stage examples: `96 -> 288`, `96 -> 576` at first stage-1 transition, `192 -> 1152`, `384 -> 2304`.
- Hiera attention output `Linear(dim_out -> dim_out, bias=True)`.
- Stage transition residual projection `Linear(dim -> dim_out)` plus query pooling.
- Hiera MLP `Linear(dim -> 4*dim) -> GELU -> Linear(4*dim -> dim)`.
- FPN lateral `Conv2d(backbone_channel -> 256,k=1,s=1,p=0)` for reversed channel list.
- Prompt mask embed: `Conv2d(1 -> 4,k=2,s=2) -> channels_first LayerNorm -> GELU -> Conv2d(4 -> 16,k=2,s=2) -> channels_first LayerNorm -> GELU -> Conv2d(16 -> 256,k=1)`.
- Decoder self-attn projections `Linear(256 -> 256)`; decoder cross-attn projections `Linear(256 -> 128)` when `attention_downsample_rate=2`; output `Linear(internal -> 256)`.
- Decoder MLP `Linear(256 -> 2048) -> ReLU -> Linear(2048 -> 256)`.
- Upscaler `ConvTranspose2d(256 -> 64,k=2,s=2) -> channels_first LayerNorm -> GELU -> ConvTranspose2d(64 -> 32,k=2,s=2) -> GELU`.
- High-res feature projections `conv_s0: Conv2d(256 -> 32,k=1)`, `conv_s1: Conv2d(256 -> 64,k=1)`.
- Four hypernetwork MLPs `Linear(256 -> 256) -> ReLU -> Linear(256 -> 256) -> ReLU -> Linear(256 -> 32)`.
- IoU head `Linear(256 -> 256) -> ReLU -> Linear(256 -> 256) -> ReLU -> Linear(256 -> 4) -> sigmoid`.
- Object score head `Linear(256 -> 256) -> ReLU -> Linear(256 -> 256) -> ReLU -> Linear(256 -> 1)`.

### Attention primitives

- Noncausal Hiera spatial self-attention over NHWC image grids, either local window or full-grid global.
- Query-pooled attention at stage transitions: Q length is downsampled while K/V length remain the pre-pool window/grid length.
- Noncausal decoder sparse-token self-attention.
- Rectangular token-to-image cross-attention: query length is output+prompt tokens, key/value length is `64*64` for the main image embedding.
- Rectangular image-to-token cross-attention: query length is `64*64`, key/value length is token count.
- Additive attention bias path via `attention_similarity`; no causal mask, no RoPE, no ALiBi, no KV cache, no varlen packed sequence metadata.
- SDPA/FlashAttention backend dispatch exists, but target-guided additive masks force SDPA for that decoder call.

### Position/custom math ops

- Hiera learned background/window positional embeddings with bicubic interpolation and tiled window embedding.
- FPN sine/cosine 2D positional encoding.
- Prompt random Fourier positional embedding for points, boxes, and image-wide decoder grid.
- Coordinate scaling, `sin`, `cos`, threshold-based stability scores, `argmax`, `gather`, `where`.

### Preprocessing/postprocessing ops

- RGB conversion, bilinear resize to `1024 x 1024`, rescale, ImageNet normalize.
- Segmentation maps/masks use nearest preprocessing resize to `256 x 256`.
- Point/box coordinate scale from original `(H,W)` into target `1024 x 1024`.
- Low-res mask bilinear resize to each original image size, optional non-overlap suppression, optional threshold.
- Automatic mask-generation helpers include crop boxes, point grids, stability filtering, mask-to-box, RLE conversion, and NMS.

## 5. Layer/block breakdown

Hiera patch stem:

```text
pixel_values: [B,3,1024,1024] NCHW
x = Conv2d(3 -> C0, k=7,s=4,p=3)(pixel_values)
x = permute NCHW -> NHWC                         # [B,256,256,C0]
x = x + bicubic/background/window position       # NHWC
```

Hiera block, repeated per configured stage:

```text
residual = x                                      # [B,H,W,C_in] NHWC
y = LayerNorm(C_in)(x)
if dim change:
  residual = max_pool2d(Linear(C_in -> C_out)(y), stride=2)
if local:
  y = window_partition(y, window_size)            # [B*nW,win,win,C_in]
q,k,v = Linear(C_in -> 3*C_out)(y).split(q,k,v)
if stage transition:
  q = max_pool2d(q, stride=2)                     # Q length shrinks only
y = Attention(q,k,v, noncausal)
y = Linear(C_out -> C_out)(y)
if local:
  y = window_unpartition(y, crop_to_residual_hw)
x = residual + y
x = x + MLP(LayerNorm(C_out)(x))                  # MLP hidden 4*C_out
```

Vision neck:

```text
stage outputs: [(256,256,C0),(128,128,C1),(64,64,C2),(32,32,C3)] NHWC
for low-to-high stage order:
  lateral = permute NHWC -> NCHW
  lateral = Conv2d(Ci -> 256,k=1)(lateral)
  if top_down_level:
    lateral = lateral + nearest_upsample(prev, scale=2)
  pos = sine_2d_position(lateral.shape)
return selected reversed levels: [B,256,256,256], [B,256,128,128], [B,256,64,64]
```

Prompt encoder:

```text
points = points + 0.5; optional append pad point if no boxes
point_pe = random_fourier(points / 1024)
point_pe[label == -1] = not_a_point_embed
point_pe[label == -10] = 0
point_pe += point_label_embedding[label>=0]

boxes = boxes + 0.5 -> reshape corners [B,P,2,2] -> append pad corner
box_pe = random_fourier(corners / 1024)
box_pe[...,0] += point_embed[2]; box_pe[...,1] += point_embed[3]
box_pe[...,2] = not_a_point_embed

mask_prompt [B,1,256,256] -> Conv/LayerNorm/GELU stack -> [B,256,64,64]
no mask -> expand learned no_mask_embed to [B,256,64,64]
```

Mask decoder:

```text
tokens = cat([object_token, iou_token, 4 mask_tokens], sparse_prompts)
image = image_embedding[-1] + dense_prompt_embedding       # [B,256,64,64]
repeat image and positions by point_batch_size

for 2 two-way blocks:
  tokens = tokens + self_attention(tokens)
  tokens = tokens + cross_attention(tokens -> image_grid, optional additive bias)
  tokens = tokens + MLP(tokens)
  image_grid = image_grid + cross_attention(image_grid -> tokens)
final tokens = tokens + cross_attention(tokens -> image_grid)

image_grid -> [B*P,256,64,64]
up = ConvTranspose2d(256->64) + high_res_128
up = LayerNorm/GELU
up = ConvTranspose2d(64->32) + high_res_256
up = GELU
hyper = 4 MLP(mask_token_out)                         # [B,P,4,32]
masks = hyper @ up.flatten_hw                         # [B,P,4,256,256]
iou = sigmoid(IoUHead(iou_token))
object_score_logits = ObjectHead(object_token)
slice masks to 3 multimask outputs or 1 single/dynamic output
```

Projection bias: source `nn.Linear` and `nn.Conv2d`/`ConvTranspose2d` defaults imply bias unless explicitly absent; Hiera QKV/proj, MLPs, FPN convs, prompt mask convs, decoder projections, hypernetwork, IoU, and object heads all need bias handling.

## 6. Attention requirements

Hiera image attention:

- Noncausal self-attention over 2D spatial tokens.
- MHA, no GQA/MQA. Per-stage head counts from config; head dim is `dim_out / heads`.
- Local windows use dynamic padding if `H` or `W` is not divisible by window size, followed by unpartition crop.
- Global attention uses `window_size=0` for configured block indexes.
- At stage transitions, Q is pooled by `query_stride=[2,2]`; K/V remain unpooled. Dinoml attention ABI must allow rectangular `q_len != kv_len`.
- No masks in standard Hiera attention. No KV cache. Flash/SDPA compatible if backend supports rectangular noncausal attention after pooling and local-window batching.

Two-way decoder attention:

- Noncausal token self-attention over sparse/output tokens with `q_len=kv_len=T`.
- Token-to-image cross-attention has `q_len=T`, `kv_len=4096`, hidden width 256, internal q/k/v width 128 for default cross-attn.
- Image-to-token cross-attention has `q_len=4096`, `kv_len=T`.
- Final token-to-image cross-attention repeats token-to-image after the blocks.
- `attention_similarity` is an additive mask/bias passed to token-to-image cross-attention. Source falls back from FlashAttention to SDPA when it is present.
- No autoregressive decoding and no KV cache. Cacheable artifacts are image embeddings/high-res features and image-wide positional embeddings.

Eager fallback to avoid: the source eager path upcasts softmax to fp32 and returns dense attention weights. For production, prefer fused noncausal attention for Hiera/decoder and avoid materializing attention weights unless explicitly requested.

## 7. Position encoding and custom math

Hiera positional embedding:

```python
def sam2_hiera_pos(pos_embed, pos_embed_window, hw):
    h, w = hw
    pos = interpolate_bicubic(pos_embed, size=(h, w))
    tile = pos_embed_window.tile([x // y for x, y in zip(pos.shape, pos_embed_window.shape)])
    return (pos + tile).permute(0, 2, 3, 1)
```

Prompt random Fourier embedding:

```python
def sam2_prompt_pe(coords_xy, gaussian_matrix, input_hw=None):
    coords = coords_xy.clone()
    if input_hw is not None:
        coords[..., 0] = coords[..., 0] / input_hw[1]
        coords[..., 1] = coords[..., 1] / input_hw[0]
    coords = 2 * coords - 1
    phases = 2 * pi * (coords @ gaussian_matrix)
    return concat(sin(phases), cos(phases), dim=-1)
```

FPN sine position:

```python
def sam2_sine_2d(shape, num_pos_feats, temperature=10000):
    mask = zeros([B, H, W], bool)
    y = cumsum(~mask, dim=1); x = cumsum(~mask, dim=2)
    y = y / (y[:, -1:, :] + 1e-6) * (2*pi)
    x = x / (x[:, :, -1:] + 1e-6) * (2*pi)
    # divide by temperature bands, interleave sin/cos, concat y then x, return NCHW
```

Dynamic multimask stability:

```python
def stability(mask_logits, delta=0.05):
    inside = sum(mask_logits.flatten(-2) > delta, dim=-1)
    union = sum(mask_logits.flatten(-2) > -delta, dim=-1)
    return where(union > 0, inside.float() / union.float(), 1.0)
```

Precomputable: image-wide prompt positional embedding for fixed `64 x 64`, FPN sine encodings for fixed feature sizes, and Hiera positional tables after interpolation for fixed image size. Dynamic: point/box prompt Fourier encodings, window padding when spatial sizes vary, and stability/gather mask selection.

## 8. Preprocessing and input packing

`Sam2ImageProcessor`:

- Emits `pixel_values` as NCHW, `original_sizes` as per-image `(H,W)`.
- Defaults: bilinear resize to `1024 x 1024`, rescale `1/255`, ImageNet normalization, RGB conversion.
- Segmentation maps use nearest resize to `mask_size={"height":256,"width":256}`, no rescale/normalize, and return `labels` as int64.
- SAM2 source disables SAM1 padding fields; postprocess directly upsamples masks to `original_sizes`.

`Sam2Processor` prompt packing:

- Requires either `images` or `original_sizes`; with cached image embeddings, callers can pass `original_sizes` without images.
- Points format: `[image, object/point_batch, point, xy]`; labels `[image, object/point_batch, point]`.
- Boxes format: `[image, box_batch, xyxy]`; boxes are scaled but not ragged-padded across images.
- Coordinates scale as `x *= target_size / old_w`, `y *= target_size / old_h`, with `target_size=1024` from processor config.
- Point padding value is `-10`; preserved during coordinate normalization and zeroed in prompt embedding.
- If labels are omitted but points are present, model fills them with positive label `1`.
- If no points and no boxes are provided, model injects a dummy point `[0,0]` with label `-1`.
- Input masks are resized inside the model to `[256,256]` with bilinear `align_corners=False, antialias=True` if needed, then consumed by Conv2d as `[B,1,256,256]`.

Postprocessing:

- `post_process_masks(masks, original_sizes)` iterates per image, bilinear-interpolates each mask tensor to original `(H,W)`, optionally applies non-overlap argmax suppression across object dimension, and optionally thresholds at `mask_threshold`.
- Automatic mask-generation helpers are CPU/data-pipeline or postprocess work for first integration: crop generation, mask filtering by IoU/stability, mask-to-box, RLE, and NMS.

## 9. Graph rewrite / lowering opportunities

### Rewrite: FPN 1x1 Conv2d -> Per-Pixel Linear

Source pattern:

```text
NHWC stage output -> permute NCHW -> Conv2d(Cin -> 256,k=1) -> top-down add
```

Replacement:

```text
NHWC PerPixelMatMul(Cin -> 256) -> optional layout materialization for NCHW consumers
```

Preconditions:

- `kernel_size == 1`, `stride == 1`, `padding == 0`, `groups == 1`.
- Consumer layout is either kept NHWC through a fused FPN region or a single final NCHW materialization is inserted.
- Top-down upsample/add axes are rewritten correctly for NHWC if fused.

Weight transform:

```python
w_linear = conv.weight.reshape(out_channels, in_channels)
b = conv.bias
```

Failure cases: any non-1x1 FPN conv, external consumer expecting NCHW, or incomplete axis rewrites for upsample/add/position generation.

Parity sketch: compare isolated FPN level outputs before/after rewrite for random `[B,H,W,C]` stages across tiny and large channel widths.

### Rewrite: cache image features for repeated prompts

Source pattern:

```text
pixel_values -> get_image_features -> conv_s0/conv_s1 -> flattened features
prompt loop repeatedly calls mask_decoder
```

Replacement:

```text
EncodeImageOnce -> cache [high_res_256, high_res_128, image_64, positions] -> PromptDecodeMany
```

Preconditions:

- Same image, same resized `pixel_values`, same encoder weights, same dtype.
- Cache includes `no_memory_embedding` addition on the last feature map and the preprojected high-res levels expected by mask decoder.

Failure cases: image augmentations/crops change coordinates, video memory path, or target embeddings/attention similarity tied to a different feature cache.

Parity sketch: run source `pixel_values` path and cached `image_embeddings` path for identical prompts; compare masks and IoU scores.

### Rewrite: decoder ConvTranspose2d k2/s2 -> pixel shuffle or tiled GEMM

Source pattern:

```text
ConvTranspose2d(Cin -> Cout,k=2,s=2,p=0) + high_res_skip
```

Replacement:

```text
PerPixelLinear(Cin -> Cout*4) -> depth_to_space(2) -> skip add
```

Preconditions:

- `kernel_size == stride == 2`, `padding == 0`, `output_padding == 0`, `dilation == 1`, `groups == 1`.
- Weight transform exactly matches transposed-conv spatial ordering.
- High-res skip tensor layout and shape match after upsample.

Failure cases: different transposed-conv params or provider depth-to-space ordering mismatch.

Parity sketch: random ConvTranspose2d weights/input against transformed path for both upscaler layers.

### Rewrite: remove dead pre-backend attention softmax in Hiera attention

Source pattern:

```text
attn_weights = softmax((q * scale) @ k.T)  # result unused
if query_stride: q = pool(q)
attention_backend(q,k,v)
```

Replacement:

```text
if query_stride: q = pool(q)
attention_backend(q,k,v)
```

Preconditions:

- `output_attentions` does not require the discarded local tensor.
- The graph compiler can prove no side effects and no NaN/exception behavior is relied upon.

Failure cases: debugging mode that records this intermediate, strict eager exception parity, or backend path changes attention math order.

Parity sketch: compare block output and requested attention outputs; only enable rewrite when the dead tensor is not part of outputs.

### Guard: no global layout translation across prompt/mask decoder

Source pattern:

```text
NCHW image embeddings -> flatten/permute -> two-way transformer -> view back NCHW -> ConvTranspose2d
```

Guard:

```text
with no_layout_translation():
  prompt mask Conv2d stack
  image_embeddings flatten/permute ABI
  high_res feature skip adds
  ConvTranspose2d upscaler
```

Reason: axes encode public cache ABI and prompt decoder shape contracts. NHWC optimization is possible only inside tightly fused conv/norm/activation regions with all consumers controlled.

## 10. Kernel fusion candidates

Highest priority:

- Hiera window attention with query pooling: dominates encoder cost and needs efficient local-window batching plus rectangular Q/KV support at stage transitions.
- Image feature cache + prompt decoder path: SAM2 is often used with repeated clicks; avoiding re-encoding is a first-order throughput win.
- Decoder cross-attention token-to-image/image-to-token: small token length against 4096 image keys is latency-sensitive and repeated per prompt.
- ConvTranspose upscaler + high-res skip + LayerNorm/GELU: directly feeds mask logits and is fixed-shape enough for a specialized path.

Medium priority:

- NHWC Hiera block fusion: LayerNorm + QKV + attention + MLP in channel-last layout can avoid ping-pong around the backbone.
- FPN 1x1 conv/top-down nearest upsample/add: image encoder tail is simple and layout-sensitive.
- Prompt mask Conv2d stack: required for mask prompts but small; fuse Conv/LayerNorm/GELU if mask prompts are common.
- Hypernetwork MLP stack + batched mask matmul: four fixed MLPs and a `[32 x 65536]` matmul per prompt batch.

Lower priority:

- Processor/postprocess GPU kernels for crop generation, RLE, and NMS; useful for automatic mask generation but not needed for first promptable segmentation.
- Non-overlap mask suppression; source default is off.
- Attention weight materialization paths; keep as debug/optional.

## 11. Runtime staging plan

1. Parse `Sam2Config` image subconfigs and reject/route `model_type="sam2_video"` fields outside the image subgraph unless explicitly running the video audit.
2. Load weights and validate image encoder one-block parity with NCHW input and NHWC Hiera internals.
3. Implement Hiera encoder + FPN parity, including local/global attention, query pooling, position embeddings, and selected FPN feature ABI.
4. Implement image-feature cache ABI: `get_image_features()` flattened form and `get_image_embeddings()` NCHW form.
5. Implement prompt encoder for points/boxes/no-mask; then add mask prompt Conv2d path.
6. Implement mask decoder from cached image embeddings with `multimask_output=True`.
7. Add `multimask_output=False` dynamic stability fallback, object score logits, target `attention_similarity`, and target embedding.
8. Add processor coordinate normalization and mask postprocess parity.
9. Add optimized local-window attention, decoder cross-attention, and guarded layout/fusion passes.
10. Defer automatic mask generation helpers and all video-memory behavior until core image segmentation is stable.

## 12. Parity and validation plan

- Random tensor tests for `window_partition/window_unpartition`, including non-divisible sizes and query-pooled transitions.
- Random tests for Hiera positional interpolation/tile math at `256`, `128`, and variant window sizes.
- Single Hiera block parity for local block and global block in fp32, then bf16/fp16 tolerances.
- Encoder parity after each stage and after FPN level selection.
- Prompt encoder parity for positive/negative points, `-1`, `-10`, boxes, no boxes, and no prompts.
- Mask prompt embedding parity for `[B,1,256,256]` and model-internal resize from another size.
- Two-way transformer parity with synthetic `[B,P,T,256]` tokens and `[B,256,64,64]` image embeddings.
- Full mask decoder parity from cached embeddings for `multimask_output=True` and dynamic single-mask mode.
- End-to-end image path parity against a sampled official checkpoint after routing video config subfields to the image subgraph.
- Postprocess parity for bilinear resize, thresholding, and optional non-overlap suppression.

Recommended tolerances: fp32 `atol=1e-4, rtol=1e-4` for isolated ops and `1e-3` for end-to-end masks; fp16/bf16 start at `atol=2e-2, rtol=2e-2`, with stricter checks on logits before thresholding where possible.

## 13. Performance probes

- Image preprocessing throughput: resize/normalize to `1024 x 1024`.
- Hiera encoder throughput by variant: tiny/small/base-plus/large.
- Local-window attention sweep by window size and query-pooling stage.
- FPN throughput and memory layout comparison: source mixed layout vs guarded NHWC fused regions.
- Prompt decoder latency with cached image embeddings for point batch sizes `1,4,16,64`.
- Mask prompt path overhead with and without `input_masks`.
- End-to-end first-click latency: preprocessing + encoder + prompt decoder + postprocess.
- Repeated-click throughput: cached image features + prompt decoder only.
- Memory footprint of cached high-res/image embeddings at batch sizes `1,2,4`.
- Attention backend comparison: eager, SDPA, FlashAttention where compatible; include `attention_similarity` fallback case.
- Postprocess throughput for mask resize/threshold and automatic mask generation helpers separately.

## 14. Skip/defer list

- Training and gradient checkpointing.
- `sam2_video` memory encoder, memory attention, temporal/object-pointer tracking, and video-specific processors.
- Automatic mask generation crop pyramid, RLE, and NMS for first promptable segmentation.
- PerSAM target-guided `attention_similarity` and `target_embedding` can be stubbed after basic prompts unless required by a product path.
- Non-overlap postprocess constraints, max-hole, and max-sprinkle cleanup; current source has TODOs around connected components.
- Multi-GPU tensor parallelism and quantization.
- Materialized attention outputs except debug parity.

## 15. Final implementation checklist

- [ ] Parse `Sam2Config`, `Sam2VisionConfig`, `Sam2HieraDetConfig`, `Sam2PromptEncoderConfig`, and `Sam2MaskDecoderConfig`.
- [ ] Route official `sam2_video` checkpoints to image-only subgraph or defer to `sam2_video` audit.
- [ ] Load Hiera, FPN, prompt encoder, and mask decoder weights with bias handling.
- [ ] Implement overlapping patch stem and NHWC Hiera blocks.
- [ ] Implement local/global Hiera attention with window pad/crop and query pooling.
- [ ] Implement Hiera positional interpolation/tiled window embedding.
- [ ] Implement FPN neck and selected feature-level ABI.
- [ ] Implement image embedding cache paths.
- [ ] Implement point/box/no-mask prompt encoder with `-1` and `-10` label semantics.
- [ ] Implement mask prompt Conv2d embedding path.
- [ ] Implement two-way mask decoder attention and MLPs.
- [ ] Implement high-res skip upscaler, hypernetwork mask matmul, IoU head, and object-score head.
- [ ] Implement dynamic multimask stability fallback.
- [ ] Implement processor coordinate normalization and mask postprocess.
- [ ] Add guarded FPN/layout rewrites and no-layout-translation guards around decoder ABI.
- [ ] Add parity tests for encoder, prompt encoder, decoder, cached embeddings, and postprocess.
- [ ] Benchmark first-click and repeated-click paths separately.
