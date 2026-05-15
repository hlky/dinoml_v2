# SegGPT Transformers audit for DinoML v2

## 1. Source basis

Transformers commit/version: local checkout `transformers` at `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`.

Model id: primary and only official in-library checkpoint found: `BAAI/seggpt-vit-large`.

Config source:

- `https://huggingface.co/BAAI/seggpt-vit-large/raw/main/config.json`
- `https://huggingface.co/BAAI/seggpt-vit-large/raw/main/preprocessor_config.json`
- Hugging Face model API reported repo sha `75cfa51ad31561cd2a8fb02964e93aeb9a544b45`, open access, `model.safetensors`, `F32` parameter count `370,723,203`.

Source files inspected:

- `transformers/src/transformers/models/seggpt/configuration_seggpt.py`
- `transformers/src/transformers/models/seggpt/modeling_seggpt.py`
- `transformers/src/transformers/models/seggpt/image_processing_seggpt.py`
- `transformers/src/transformers/models/seggpt/image_processing_pil_seggpt.py`
- `transformers/src/transformers/models/seggpt/convert_seggpt_to_hf.py`
- `transformers/tests/models/seggpt/test_modeling_seggpt.py`
- `transformers/tests/models/seggpt/test_image_processing_seggpt.py`
- `transformers/docs/source/en/model_doc/seggpt.md`

Any missing files or assumptions:

- No remote-code files are required for the official checkpoint.
- Only one official HF checkpoint/config was found for this model family. There is no small/debug checkpoint or structurally different checkpoint sweep available in-library.
- The conversion script references the original `BAAI/SegGPT` PyTorch checkpoint, but the DinoML audit scope is the current in-library Transformers implementation.
- Primary DinoML runtime target should be `SegGptForImageSegmentation` inference. `SegGptModel` encoder output is a useful intermediate parity target. Training loss is deferred.

## 2. High-level architecture

SegGPT is a prompt-conditioned image segmentation model. It uses an image processor plus a ViT-like encoder with decomposed relative position bias and a convolutional image decoder. It is not an autoregressive language model and has no token generation or KV cache.

Dataflow:

```text
CPU image/mask preprocessing -> NCHW pixel_values, prompt_pixel_values, prompt_masks
  -> height concat into two 896x448 NCHW images
  -> Conv2d patch embedding -> NHWC 56x28 patch grids
  -> prompt mask-token blend + segment/type/position embeddings
  -> batch concat input/prompt grids -> ViT encoder with merge after layer 2
  -> collect normalized layers 5/11/17/23
  -> concat features -> Linear patch decoder -> pixel-shuffle-like reshape to NCHW
  -> 3x3 Conv2d + channels-first LayerNorm + GELU + 1x1 Conv2d
  -> pred_masks [B, 3, 896, 448] -> CPU/PyTorch postprocess to semantic map
```

Stage decomposition:

- CPU/data pipeline: RGB conversion, resizing to `448x448`, rescale by `1/255`, ImageNet mean/std normalize, optional class palette mapping for 2D prompt masks.
- Input packing: source model concatenates prompt and target image along height, and separately concatenates prompt mask with either itself or labels along height.
- Encoder stem: non-overlapping patch `Conv2d(3 -> 1024, kernel=16, stride=16)` on two packed NCHW images, then NHWC internal layout.
- Encoder: 24 pre-LN transformer layers; first three layers operate on `2B` batch rows until `merge_index=2` averages input/prompt halves back to `B`.
- Decoder: four intermediate NHWC feature maps concatenate on channel, project each patch to `16*16*64`, reshape to NCHW image features, then small conv head.
- Postprocess: slice lower height half, unnormalize and scale to `[0,255]`, optional nearest resize, then either nearest palette color via squared distance/argmin or channel mean.

Independently validatable units: processor tensors, forward input packing and default `bool_masked_pos`, patch embedding, one attention block with relative position bias, layer-2 merge behavior, full encoder intermediate states, decoder reshape, conv head, and postprocessing.

## 3. Important config dimensions

| Field | Source default | `BAAI/seggpt-vit-large` | Operator significance |
|---|---:|---:|---|
| architecture | n/a | `SegGptForImageSegmentation` | Runtime target includes decoder head |
| `image_size` | `(896, 448)` | `[896, 448]` | Model input after height packing, not processor single-image size |
| processor size | `448x448` | `448x448` | Each input/prompt/mask image before model packing |
| `patch_size` | 16 | 16 | Patch grid `56x28`, patch area 256 |
| `num_channels` | 3 | 3 | RGB images and RGB-coded masks |
| `hidden_size` | 1024 | 1024 | Encoder width |
| `num_hidden_layers` | 24 | 24 | Transformer depth |
| `num_attention_heads` | 16 | 16 | MHA heads |
| `head_dim` | inferred 64 | inferred 64 | `hidden_size // heads` |
| `mlp_dim` | `hidden_size * 4` | 4096 | FFN expansion |
| `hidden_act` | `gelu` | `gelu` | FFN and decoder activation |
| `qkv_bias` | true | true | QKV packed projection has bias |
| `layer_norm_eps` | `1e-6` | `1e-6` | Encoder and decoder norms |
| `drop_path_rate` | 0.1 | 0.1 | Inference identity because model is eval |
| `hidden_dropout_prob` | 0.0 | 0.0 | Dropout no-op for official config |
| `pretrain_image_size` | 224 | 224 | Absolute position table starts as `14x14 + 1` |
| `decoder_hidden_size` | 64 | 64 | Decoder image feature channels |
| `use_relative_position_embeddings` | true | true | Decomposed relative attention bias required |
| `merge_index` | 2 | 2 | Batch-half merge after encoder layer 2 |
| `intermediate_hidden_state_indices` | `[5,11,17,23]` | `[5,11,17,23]` | Four features for decoder concat |
| `beta` | 0.01 | 0.01 | Training smooth-L1 only |
| `torch_dtype` | source default | `float32` | Hub config/model storage reports F32 |

Representative checkpoint sweep:

| Model id | Access | Config variation | Processor variation | Notes |
|---|---|---|---|---|
| `BAAI/seggpt-vit-large` | open | ViT-large style, 24 layers, 1024 hidden, 16 heads, rel pos on | resize/normalize to `448x448`, ImageNet mean/std | Only official in-library checkpoint found |
| `BAAI/SegGPT` | open repo/model-card plus original checkpoint referenced by converter | Original `.pth` source for conversion, not a Transformers `model_type=seggpt` config basis | n/a | Use only as provenance for conversion, not as current runtime config |

Config fields in the Hub config that the inspected source does not read structurally: `mlp_ratio` is present in `config.json` and tests pass it into `SegGptConfig`, but modeling code reads `config.mlp_dim`, not `mlp_ratio`. Treat `mlp_ratio` as historical/ignored for this source basis unless config validation changes.

## 3a. Family variation traps

- Public model tensors are NCHW, but most encoder math is NHWC after patch embedding. Do not globally translate NCHW to NHWC without guarding the packing, decoder, and postprocess axes.
- `config.image_size` is the packed model size `(896,448)`. The image processor emits individual `448x448` tensors; the model doubles height internally.
- Default `bool_masked_pos` length is `num_patches=56*28=1568`; first half is false and second half is true. This relies on row-major patch order after height concatenation.
- The encoder batch dimension changes from `2B` to `B` after `merge_index=2`. Intermediate features for the decoder are collected only after this merge for the official indices.
- `feature_ensemble=True` adds shape-dependent split/mean/expand logic inside each layer. First integration can reject or stage this separately.
- `embedding_type` must be `"instance"` or `"semantic"` and selects one of two learned type tokens. It is a runtime branch but not a new operator family.
- Relative position parameters are per layer and shaped by the configured patch grid: `rel_pos_h [111,64]`, `rel_pos_w [55,64]` for the official grid. Source still calls linear interpolation in `get_rel_pos`.
- Attention is dense noncausal MHA over a rectangular 2D patch grid, with decomposed H/W relative bias added before softmax.
- Decoder reconstruction is a pixel-shuffle-like reshape/permute from NHWC patch features to NCHW image features. This is axis-sensitive and should be guarded.
- `SegGptLayerNorm(data_format="channels_first")` implements channels-first normalization by permuting to NHWC, applying `LayerNorm(C)`, and permuting back.
- Postprocess is not a simple logits argmax. It treats output masks as RGB images, unnormalizes them, and maps to class IDs either by palette nearest color or channel mean.
- No causal decode, no text tokenizer, no RoPE, no ALiBi, no MoE, no quantized/packed weights, no KV cache.

## 4. Operator coverage checklist

Tensor/layout ops:

- NCHW input validation for `pixel_values`, `prompt_pixel_values`, `prompt_masks`: `[B,3,448,448]` from the processor for the official checkpoint.
- Concatenate along NCHW height axis `dim=2`: image pair and mask/label pair become `[B,3,896,448]`.
- Non-overlapping `Conv2d(3 -> 1024, kernel=16, stride=16, padding=0)` patch projection.
- Permute patch projection output `[B,1024,56,28] -> [B,56,28,1024]`.
- Broadcast learned tokens `[1,1,1,1024]` over `[B,56,28,1024]`.
- Mask-token blend: `prompt = prompt * (1 - w) + mask_token * w`, with `w` reshaped from `[B,1568]` to `[B,56,28,1]`.
- Add learned segment/type/absolute position tensors with NHWC broadcasting.
- Concatenate input and prompt embeddings along batch axis: `[B,H,W,C] + [B,H,W,C] -> [2B,H,W,C]`.
- Reshape/permute for QKV split: `[B,H,W,1024] -> [3,B,16,H*W,64]`.
- Matrix multiplies for attention scores and values.
- Split along spatial width-like axis inside feature ensemble, mean reductions, expand, concat.
- Layer-2 merge: slice first/second halves of batch, add, multiply by `0.5`.
- Concatenate four intermediate hidden states along channel: `[B,56,28,4*1024]`.
- Decoder linear and pixel shuffle: `[B,56,28,4096] -> [B,56,28,16384] -> [B,64,896,448]`.
- Channels-first LayerNorm implemented through NCHW/NHWC permutes.
- Postprocess slice lower height half, NCHW/NHWC permutes, clip, optional nearest resize, palette distance reshape/view, argmin.

Neural network primitives:

- `Linear(1024 -> 3072, bias=qkv_bias)` packed QKV, output split order is Q, K, V contiguous blocks.
- `Linear(1024 -> 1024, bias=True)` attention output projection.
- `LayerNorm(1024, eps=1e-6)` before attention, before MLP, and on collected intermediate states.
- `Linear(1024 -> 4096)`, GELU, `Linear(4096 -> 1024)` MLP.
- `Linear(4096 -> 16384)` decoder patch projection.
- `Conv2d(64 -> 64, kernel=3, padding=1)`, channels-first LayerNorm, GELU, `Conv2d(64 -> 3, kernel=1)`.
- Inference DropPath is identity. Training stochastic depth can be deferred.

Attention primitives:

- Dense noncausal self-attention over `S=Hpatch*Wpatch=1568` tokens per image grid.
- MHA with `heads=16`, `head_dim=64`, query/key/value widths all 1024.
- Score math: `(query * head_dim**-0.5) @ key.transpose(-2,-1)`.
- Decomposed relative position bias before softmax using two gathered/interpolated tables and two `einsum` patterns.
- Softmax computes in fp32 then casts back to query dtype.

Position/relative-bias ops:

- Absolute learned patch position table `position_embeddings[:,1:]` with bicubic interpolation from `14x14` to `56x28` for the official runtime shape.
- Per-layer decomposed relative bias tables with linear interpolation and integer gather.

Preprocessing-coupled ops:

- Optional mask palette build in CPU pipeline.
- 2D segmentation prompt masks may become RGB via palette or channel repeat before resize/rescale/normalize.
- Processor uses nearest resampling for masks and bicubic for images/prompt images.

Structured-output postprocessing ops:

- Lower-half crop, unnormalize, clamp to `[0,255]`, optional nearest resize to target image size.
- Palette nearest-color map: squared distance to `[num_labels+1,3]` palette and `argmin(dim=-1)`.
- If `num_labels` is absent, channel mean and integer cast.

## 5. Layer/block breakdown

Input packing for official inference:

```text
pixel_values:        [B,3,448,448]
prompt_pixel_values: [B,3,448,448]
prompt_masks:        [B,3,448,448]

x_img  = cat(prompt_pixel_values, pixel_values, dim=height) -> [B,3,896,448]
x_mask = cat(prompt_masks, prompt_masks, dim=height)         -> [B,3,896,448]
```

Embeddings:

```text
input_embeddings  = Conv2d(3->1024,k16,s16)(x_img).permute(N,H,W,C)  -> [B,56,28,1024]
prompt_embeddings = Conv2d(3->1024,k16,s16)(x_mask).permute(N,H,W,C) -> [B,56,28,1024]
prompt_embeddings = masked_blend(prompt_embeddings, mask_token, bool_masked_pos)
pos_embed = bicubic_interpolate([1,14,14,1024] -> [1,56,28,1024])
input_embeddings += segment_token_input + type_token + pos_embed
prompt_embeddings += segment_token_prompt + type_token + pos_embed
hidden = cat(input_embeddings, prompt_embeddings, dim=batch) -> [2B,56,28,1024]
```

Encoder block, repeated 24 times:

```text
y = LayerNorm(hidden)
qkv = Linear(1024 -> 3072, bias=True)(y)
q,k,v = reshape/split qkv to [batch*16, 1568, 64]
scores = (q * 0.125) @ k.T
scores += decomposed_relative_position_bias(q, rel_pos_h, rel_pos_w, grid=56x28)
probs = softmax(scores, dim=-1, fp32).to(q.dtype)
attn = probs @ v
attn = reshape/permute to [batch,56,28,1024]
attn = Linear(1024 -> 1024)(attn)
hidden = hidden + attn
residual = hidden
hidden = LayerNorm(hidden)
hidden = Linear(1024 -> 4096) -> GELU -> Linear(4096 -> 1024)
hidden = residual + hidden
```

Layer-specific control:

```text
if feature_ensemble:
    optional split/mean/expand/cat on attention_output
if layer_index == 2:
    hidden = (hidden[:B] + hidden[B:]) * 0.5
if layer_index in [5, 11, 17, 23]:
    save LayerNorm(hidden)
```

Decoder:

```text
features = cat(saved_features, dim=-1) -> [B,56,28,4096]
patches = Linear(4096 -> 16384)(features)
image_features = reshape/permute patches -> [B,64,896,448]
pred_masks = Conv2d(64->64,k3,p1) -> LayerNorm(C=64 over channels) -> GELU -> Conv2d(64->3,k1)
```

## 6. Attention requirements

SegGPT requires encoder-style dense self-attention only.

- Causality: noncausal/bidirectional.
- Self/cross: self-attention only; prompt conditioning is packed in batch/spatial inputs rather than cross-attention.
- Heads: MHA, `num_attention_heads=16`; no GQA/MQA.
- Head dim: `64`; Q/K/V each project from hidden width 1024.
- Sequence shape: rectangular patch grid `56x28`; flattened length `1568`.
- Batch shape: layers 0-2 attend over `2B` grids, layers 3-23 over `B` grids after merge.
- Masking: no attention mask in source forward. Masked image area is handled before encoder by replacing prompt mask patches with `mask_token`.
- Relative bias: decomposed H/W relative position bias is mandatory for the official checkpoint.
- Packed/varlen: none.
- Sliding/local/block attention: none.
- KV cache: not applicable.
- FlashAttention/SDPA compatibility: plain dense attention can map to fused attention only if the decomposed relative bias is materialized or fused before softmax. A pure FlashAttention path without custom bias support is not parity-complete.

## 7. Position encoding and custom math

Absolute patch position interpolation:

```python
def seggpt_abs_pos(position_embeddings, height, width):
    patch_pos = position_embeddings[:, 1:]       # [1, 196, C]
    p = int(patch_pos.shape[1] ** 0.5)           # 14
    x = patch_pos.reshape(1, p, p, -1).permute(0, 3, 1, 2)
    x = interpolate(x, size=(height, width), mode="bicubic", align_corners=False)
    return x.permute(0, 2, 3, 1)                # [1, H, W, C]
```

Decomposed relative position bias:

```python
def add_rel_pos(scores, q, rel_h, rel_w, H, W):
    rh = get_rel_pos(H, H, rel_h)                # [H, H, head_dim]
    rw = get_rel_pos(W, W, rel_w)                # [W, W, head_dim]
    q2 = q.reshape(batch_heads, H, W, head_dim)
    bias_h = einsum("bhwc,hkc->bhwk", q2, rh)
    bias_w = einsum("bhwc,wkc->bhwk", q2, rw)
    scores = scores.reshape(batch_heads, H, W, H, W)
    scores = scores + bias_h[:, :, :, :, None] + bias_w[:, :, :, None, :]
    return scores.reshape(batch_heads, H * W, H * W)
```

`get_rel_pos` linearly interpolates the table to `2*max(q_size,k_size)-1`, builds float-scaled coordinate grids, casts gathered indices to long, and returns a table indexed by relative coordinates. For the official fixed grid the output sizes are stable, so DinoML can precompute the integer index maps and table interpolation results per compiled grid, but the final bias depends on runtime query values.

## 8. Preprocessing and input packing

Processor contract:

- Inputs: `images`, `prompt_images`, and `prompt_masks`; at least one must be provided by the generic processor, but full model inference needs all three model tensors.
- Image and prompt image: prepared as RGB-like image tensors, resized to `448x448` with bicubic, rescaled by `1/255`, normalized by ImageNet mean/std, returned as NCHW.
- Prompt masks: either 2D segmentation maps converted to RGB by palette or repeated channels, or already-RGB masks when `do_convert_rgb=False`. Masks resize with nearest interpolation, then rescale/normalize like images.
- Output tensors for the official checkpoint: `pixel_values`, `prompt_pixel_values`, `prompt_masks`, each `[B,3,448,448]`.

GPU/runtime input packing:

- Model casts image tensors to patch-conv weight dtype.
- Model height-concats images and masks to `[B,3,896,448]`.
- Default inference `bool_masked_pos` is created on the model device as `[1,1568]`, with false for the first 784 patches and true for the second 784 patches. For `B > 1`, source relies on broadcasting through reshape/masking behavior; DinoML should validate or explicitly expand to `[B,1568]`.
- `embedding_type` controls a learned type-token branch: default `"instance"`, optional `"semantic"`.

Postprocessing:

- Raw `pred_masks` are `[B,3,896,448]`.
- Source keeps only lower half along height: `[B,3,448,448]`.
- Unnormalize in channel-last form, return to NCHW, multiply by 255, clamp.
- Optional target resize uses nearest interpolation.
- With `num_labels`, map each RGB pixel to the nearest palette color by squared distance and `argmin`; without `num_labels`, output `mean(channel).int()`.

## 9. Graph rewrite / lowering opportunities

### Rewrite: patch Conv2d -> Linear

Source pattern:

```text
Conv2d(3 -> 1024, kernel=16, stride=16, padding=0)(NCHW) -> permute to NHWC
```

Replacement:

```text
WindowFlatten(NCHW or guarded NHWC, 16x16 non-overlap) -> Linear(768 -> 1024) -> [B,56,28,1024]
```

Preconditions:

- `kernel_size == stride == patch_size`.
- `padding == 0`, `dilation == 1`, `groups == 1`.
- Input height/width equal config image size and divisible by patch size.
- Preserve PyTorch flatten order: channels, patch_h, patch_w from NCHW convolution.

Weight transform:

```python
w_linear = conv.weight.reshape(out_channels, in_channels * patch_h * patch_w)
b_linear = conv.bias
```

Failure cases: tuple patch sizes need normalized handling; dynamic image sizes need position/rel-pos recompilation or runtime interpolation support.

Parity test sketch: compare patch embedding output after the source `.permute(0,2,3,1)` on random `[B,3,896,448]`.

### Rewrite: packed QKV Linear -> GEMM plus split

Source pattern:

```text
Linear(1024 -> 3072)(NHWC) -> reshape [B,S,3,16,64] -> split Q,K,V
```

Replacement: one GEMM over flattened `[B*Hpatch*Wpatch,1024]` plus view/split into Q/K/V. Split order is Q then K then V contiguous output blocks.

Preconditions: hidden last layout, contiguous packed output, `hidden_size == heads * head_dim`, qkv bias present or explicitly absent.

Parity test sketch: one attention block through QKV split with source weights.

### Rewrite: relative-position index precompute

Source pattern: per-forward `arange`, scale, cast, and gather in `get_rel_pos`.

Replacement: for fixed grid `56x28`, precompute H and W integer index matrices and only gather/interpolate tables at load/compile time when shape is static.

Preconditions: static q/k sizes and fixed `rel_pos_h/w` shapes. If dynamic grid is admitted, cache by `(q_h,q_w,k_h,k_w)`.

Failure cases: tracing path or non-official image sizes can require interpolation to different lengths.

### Rewrite: decoder patch projection -> pixel shuffle

Source pattern:

```text
Linear(4096 -> 16*16*64) on [B,56,28,C]
reshape [B,56,28,16,16,64]
permute [B,64,56,16,28,16]
reshape [B,64,896,448]
```

Replacement: model as a deterministic depth-to-space/pixel-shuffle variant with patch grid axes, then lower the following conv head normally.

Preconditions: `patch_size=16`, `decoder_hidden_size=64`, row-major patch grid order, no layout translation across the reshape unless all axes are rewritten.

Failure cases: any NHWC conv-head optimization must preserve the exact spatial interleave order.

### Rewrite: guarded NHWC encoder region

Source pattern: after patch projection, encoder hidden states are `[B,H,W,C]`.

Replacement: keep encoder in NHWC and lower LayerNorm/Linear/attention over last dimension directly. This is already source-faithful and should be the initial translation.

No-layout-translation guards:

- Public NCHW inputs and height concat `dim=2`.
- `bool_masked_pos` reshape to `[B,patch_h,patch_w,1]`.
- Relative-position H/W axes.
- Decoder pixel-shuffle reshape/permute.
- Channels-first decoder conv head and channels-first LayerNorm.
- Postprocess lower-half height crop and palette mapping.

If a later pass admits end-to-end NHWC images, required axis rewrites include NCHW height concat `dim=2 -> dim=1`, NCHW channel normalization axes, postprocess `masks.shape[2] // 2 -> masks.shape[1] // 2`, and channel reductions from `dim=0/1` depending on rank.

## 10. Kernel fusion candidates

Highest priority:

- Patch Conv2d as GEMM/window-flatten: large first projection and currently blocked by missing general Conv2d.
- LayerNorm over NHWC hidden width 1024: appears twice per block plus intermediate collection norms.
- Packed QKV GEMM + reshape/split: 24 layers, dominant projection path.
- Relative-position bias generation/add fused with attention score preparation: avoids materializing repeated `1568x1568` bias intermediates separately.
- Dense attention with additive decomposed bias: main memory/compute bottleneck at `S=1568`, `heads=16`.
- MLP GELU block: `1024 -> 4096 -> 1024` repeated 24 times, classic GEMM+activation fusion candidate.

Medium priority:

- Decoder `Linear(4096 -> 16384)` plus pixel-shuffle reshape: large output projection with deterministic layout transform.
- Decoder conv head: `3x3 Conv2d(64->64)` plus channels-first LayerNorm/GELU and `1x1 Conv2d`.
- Default mask-token blend: simple fused elementwise select/blend over `[B,56,28,1024]`.
- Postprocess palette nearest-color mapping if end-to-end segmentation maps are in runtime scope.

Lower priority:

- Feature ensemble branch: only needed for few-shot batch ensembling and can be staged.
- Training loss patchify/unpatchify and smooth-L1: not needed for inference-first.
- Output attentions reshaping: diagnostic output, not first production path.

## 11. Runtime staging plan

Stage 1: parse config and load official weights. Admit only `BAAI/seggpt-vit-large`-style configs: `image_size=[896,448]`, `patch_size=16`, `hidden_size=1024`, `heads=16`, `use_relative_position_embeddings=True`, `intermediate_hidden_state_indices=[5,11,17,23]`.

Stage 2: CPU/data-pipeline parity for processor and postprocessor. Keep palette mapping and target resize outside the first compiled graph if needed.

Stage 3: patch embedding and embedding-pack parity. Include default `bool_masked_pos`, mask-token blend, type-token branch, position interpolation to `56x28`, and batch concat.

Stage 4: one encoder block parity. Implement NHWC LayerNorm, QKV, decomposed relative bias, dense attention, output projection, MLP, residuals.

Stage 5: full encoder parity through merge and intermediate feature collection. Validate layer 2 batch merge and saved features.

Stage 6: decoder parity. Lower feature concat, decoder linear, pixel-shuffle reshape, conv head, and raw `pred_masks`.

Stage 7: optimized kernels and layout rewrites. Add patch-conv GEMM rewrite, fused attention/bias, MLP fusions, and guarded decoder layout optimizations.

Initially stub/defer: `feature_ensemble=True`, labels/loss, output attentions, arbitrary image sizes, arbitrary checkpoint variants, postprocess palette mapping inside compiled runtime.

## 12. Parity and validation plan

- Processor parity: compare `pixel_values`, `prompt_pixel_values`, and `prompt_masks` for official example images against Transformers test tolerances (`rtol=1e-4`, `atol=1e-4`).
- Patch embedding parity: random packed `[B,3,896,448]`, compare source conv+permute to DinoML lowering.
- Position interpolation parity: compare absolute position tensor `[1,56,28,1024]` and relative H/W gathered tables.
- One-layer parity: run a single `SegGptLayer` with fixed random NHWC input at both `2B` and `B` batch shapes.
- Merge parity: after layer index 2, verify `(first_half + second_half) * 0.5` and output shape `[B,56,28,1024]`.
- Full encoder parity: compare `last_hidden_state` and four `intermediate_hidden_states`.
- Decoder parity: feed saved feature tensors and compare raw `pred_masks [B,3,896,448]`.
- End-to-end raw mask parity: compare `SegGptForImageSegmentation.pred_masks` on the official sample. Recommended fp32 tolerance `atol=1e-4` initially; fp16/bf16 should use looser image-output tolerances after attention softmax and interpolation are validated.
- Postprocess parity: compare lower-half crop, unnormalize/clip, nearest resize, palette argmin, and no-palette mean/int path.

Do not mark runtime parity complete unless raw mask and postprocessed semantic map parity are both covered for at least one official example.

## 13. Performance probes

- Processor throughput: image resize/normalize and mask palette conversion separately from model runtime.
- Patch embedding throughput: Conv2d path versus window-flatten GEMM rewrite for `[B,3,896,448]`.
- Encoder block microbench: attention versus MLP time at `S=1568`, `C=1024`, `heads=16`.
- Relative-position bias overhead: materialized eager bias add versus fused/custom bias inside attention score path.
- Full encoder throughput sweep over batch size, including the `2B` first-three-layer behavior.
- Decoder throughput: large decoder linear plus pixel-shuffle reshape versus conv head.
- End-to-end raw mask throughput for `B=1,2,4` if memory allows.
- Postprocess throughput with and without palette nearest-color mapping; sweep `num_labels`.
- Memory probes: attention score/probability tensors at `B*heads*S*S`, intermediate feature retention for four layers, and decoder linear output.

## 14. Skip/defer list

- Training loss and labels path.
- Gradient checkpointing and DropPath training behavior.
- `feature_ensemble=True` few-shot ensembling branch.
- `output_attentions=True` diagnostic attention tensors.
- Video segmentation claims from the paper/model card; the inspected HF model is an image model wrapper.
- Arbitrary image sizes or non-`448x448` processor sizes.
- In-graph palette postprocessing and target-size resize for the first compiled model; keep as CPU/PyTorch postprocess until raw masks are stable.
- Quantization, tensor parallelism, multi-GPU, and remote-code variants.

## 15. Final implementation checklist

- [ ] Parse `SegGptConfig` and admit a bounded `seggpt-vit-large` config.
- [ ] Load official weights, preserving shared patch embedding module used for image and mask streams.
- [ ] Implement/compose processor parity or define the CPU preprocessor boundary.
- [ ] Lower NCHW height input packing and default `bool_masked_pos`.
- [ ] Implement patch Conv2d or guarded patch Conv2d-to-Linear rewrite.
- [ ] Implement NHWC token additions and mask-token blend.
- [ ] Implement bicubic absolute position interpolation or precompute for `56x28`.
- [ ] Implement dense MHA with decomposed relative position bias.
- [ ] Implement NHWC LayerNorm, QKV/output projections, GELU MLP, and residuals.
- [ ] Implement layer-2 batch merge and intermediate state collection.
- [ ] Implement decoder feature concat, `Linear(4096 -> 16384)`, pixel-shuffle reshape, and conv head.
- [ ] Add raw `pred_masks` parity tests against Transformers.
- [ ] Add postprocess parity tests for palette and no-palette semantic maps.
- [ ] Add performance probes for patch embedding, attention, MLP, decoder, and postprocess.
- [ ] Add guarded NHWC/NCHW layout rewrite tests before enabling optimized layout passes.
