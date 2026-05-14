# EoMT Transformers Audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: tue-mps/coco_panoptic_eomt_large_640 as the main reference; sweep also covers small/base/large/giant/7B, ADE20K, Cityscapes, COCO instance/panoptic variants.
Config source: public Hugging Face config.json and preprocessor_config.json files, plus source defaults from EomtConfig.
Source files inspected: src/transformers/models/eomt/{modular_eomt.py,modeling_eomt.py,configuration_eomt.py,image_processing_eomt.py,image_processing_pil_eomt.py,convert_eomt_to_hf.py}; tests/models/eomt/*; docs/source/en/model_doc/eomt.md.
Any missing files or assumptions: no gated/401 EoMT DINOv2 links found. DINOv3 EoMT checkpoints use model family eomt_dinov3 and are out of scope. modeling_eomt.py/configuration_eomt.py are generated; modular_eomt.py is authoritative for future source edits.
```

Small notes/config snapshots are in `agents/plans/transformers/eomt/source_notes.md` and `agents/plans/transformers/eomt/config_sweep.md`.

## 2. High-level architecture

EoMT is an encoder-only image segmentation model: DINOv2-style ViT patch encoder with CLS/register tokens, learned segmentation query tokens injected only in the final encoder blocks, a light convolutional upscale path, a query MLP mask head, and class/mask outputs for semantic, instance, or panoptic postprocessing.

```text
CPU image preprocessing -> NCHW pixel_values -> Conv2d patch embed -> add absolute patch embeddings + CLS/register tokens
-> ViT encoder blocks without queries -> concatenate learned queries before final N blocks
-> final query/image joint self-attention blocks, optionally mask-constrained
-> final LayerNorm -> class logits + mask embeddings x upscaled image features
-> semantic/instance/panoptic postprocess
```

Stage decomposition:

- CPU/data pipeline: RGB conversion, resize, rescale by `1/255`, ImageNet normalization, optional square pad, or semantic sliding-window patch split with patch offsets.
- Core GPU graph: fixed-resolution NCHW image batch to `class_queries_logits` and `masks_queries_logits`.
- Postprocess: CPU or GPU tensor logic that resizes/crops/stitches masks, combines class probabilities and mask probabilities, thresholds objects, and emits variable-length records.
- Cacheability: no autoregressive KV cache. Preprocessed image patches and encoder hidden states before query injection are independently stageable for experiments, but source forward recomputes all blocks.

## 3. Important config dimensions

| Field | Source/default | Runtime effect |
|---|---:|---|
| `hidden_size` | 384, 768, 1024, 1536, 4096 observed | ViT channel width, query width, mask head width |
| `num_hidden_layers` | 12, 24, 32, 40 observed | full encoder depth |
| `num_attention_heads` | 6, 12, 16, 24, 32 observed | dense self-attention heads |
| `head_dim` | `hidden_size / heads`; 64 except 7B uses 128 | attention scale and QKV reshape |
| `image_size` | 512, 640, 1024, 1280 observed | fixed patch-position table and mask grid |
| `patch_size` | 16 observed | Conv2d patch embed stride/kernel |
| `num_register_tokens` | 4 observed/default | prefix tokens after CLS |
| `num_queries` | 100 semantic, 200 COCO instance/panoptic | query token count and class/mask output count |
| `num_blocks` | 3, 4, 5 observed | number of final blocks after query injection and mask predictions |
| `num_upscale_blocks` | 2 observed/default | mask feature map upsample factor is 4x over patch grid |
| `mlp_ratio` | 4 observed/default | vanilla MLP hidden width is `4 * hidden_size` |
| `use_swiglu_ffn` | false for small/base/large, true for giant/7B | changes FFN operator and weight layout |
| `num_labels` | 19, 80, 133, 150 observed | class predictor output is `num_labels + 1` |

Representative sweep: see `config_sweep.md`. For main `coco_panoptic_eomt_large_640`: patch grid is `40x40`, prefix before queries is `1 + 4 + 1600 = 1605`, final sequence after query injection is `1805`, mask logits are `[B, 200, 160, 160]`, class logits are `[B, 200, 134]`.

## 3a. Family variation traps

- The current source is not an `AutoBackbone` wrapper despite DINOv2 heritage; it owns the ViT operators directly.
- `image_size` is used to build `grid_size = image_size // patch_size`; source assumes square scalar `image_size` in `EomtForUniversalSegmentation.__init__`.
- Position embeddings are a learned table over patch positions only. There is no interpolation path in EoMT; mismatched runtime H/W against config patch grid should be rejected unless a separate resize/split route guarantees the configured size.
- Query tokens are inserted at `num_hidden_layers - num_blocks`, so attention sequence length changes inside the encoder.
- Final blocks build a dense 4D attention mask from predicted masks. In eval, `attn_mask_probs` defaults to all ones, so this path is active.
- Giant and 7B use SwiGLU (`Linear(C -> 2H)`, chunk, `silu(x1) * x2`, `Linear(H -> C)`) with `H = ceil((C * mlp_ratio * 2 / 3) / 8) * 8`; small/base/large use GELU MLP.
- Semantic checkpoints use `do_split_image=True`, no padding, shortest-edge resize, and patch offset stitching. Instance/panoptic use square pad and no split by default.
- NCHW is the semantic source layout for model Conv2d/ConvTranspose2d/depthwise Conv2d/postprocess. `LayerNorm2d` temporarily permutes NCHW to NHWC and back; this is a local channel-last candidate but a layout pass must guard all surrounding conv/einsum axes.
- DINOv3-named EoMT checkpoints have different `model_type` and should route to a separate audit.

## 4. Operator coverage checklist

Tensor/layout ops:

- NCHW Conv2d patch embed: `[B, 3, H, W] -> [B, C, H/16, W/16]`, kernel=stride=`16`, no padding.
- Flatten spatial, transpose, reshape, concat along sequence, expand learned CLS/register/query tokens, Embedding lookup for absolute patch position ids.
- Sequence slicing: query tokens, prefix/image tokens after `num_queries + 1 + num_register_tokens`.
- NCHW/NHWC permutes for `LayerNorm2d`; NCHW interpolate/crop/pad in processor/postprocess.

Neural primitives:

- LayerNorm over sequence channel and over 2D channels via NHWC.
- Linear Q/K/V/out projections, all biased.
- Linear class predictor `C -> num_labels + 1`.
- Vanilla MLP `C -> 4C -> C` with config activation, usually GELU.
- SwiGLU FFN for giant/7B: `C -> 2H -> H -> C`.
- LayerScale elementwise multiply by learned `[C]`.
- ConvTranspose2d `C -> C`, kernel=2, stride=2, bias=true.
- Depthwise Conv2d `C -> C`, kernel=3, padding=1, groups=C, bias=false.
- Mask head MLP: three biased `C -> C` linears with activation after first two.
- Einsum mask product: `bqc,bchw->bqhw`.

Attention primitives:

- Noncausal dense self-attention over sequence; no KV cache.
- Q/K/V width equals hidden size; MHA only in source, no GQA/MQA.
- Masked final-block attention requires boolean comparisons, indexed mask update, expand to `[B, heads, S, S]`, float mask fill with `-1e9`, and attention backend compatibility.

Pre/postprocess-coupled ops:

- Resize with bilinear for images/masks logits, nearest for training segmentation maps, rescale/normalize, square zero pad, sliding-window split/stitch, sigmoid/softmax, argmax, thresholding, per-query scoring, variable-length segment list assembly.
- No NMS in source postprocessing.

Training-only/deferred ops:

- Hungarian matching, point sampling with `grid_sample`, dice/BCE/CE losses, distributed `num_masks` reduction, random point sampling, and DropPath randomness are not needed for first inference parity.

## 5. Layer/block breakdown

Patch embedding:

```text
pixel_values: [B,3,H,W] NCHW
patches = Conv2d(3 -> C, kernel=16, stride=16)(pixel_values)  # [B,C,G,G]
tokens = flatten spatial then transpose                         # [B,G*G,C]
tokens += Embedding(position_ids[0:G*G])
hidden = concat(CLS[1], register[R], tokens, dim=1)             # [B,1+R+G*G,C]
```

Encoder block, repeated `num_hidden_layers` times, with queries inserted before the final `num_blocks`:

```text
x_norm = LayerNorm(x)
q,k,v = Linear(C -> C) each, reshape [B,S,Hd,Nhd] -> [B,Hd,S,Nhd]
attn = noncausal attention(q,k,v, optional 4D mask)
x = x + LayerScale(Linear(C -> C)(attn))
y = LayerNorm(x)
y = MLP(y) or SwiGLU(y)
x = x + LayerScale(y)
```

Prediction path used before each final masked-attention block and after the last block:

```text
query_tokens = logits[:, :num_queries, :]                       # [B,Q,C]
class_logits = Linear(C -> num_labels+1)(query_tokens)
image_tokens = logits[:, Q+1+R:, :].transpose(1,2).reshape(B,C,G,G)
mask_embed = MLP(C -> C -> C -> C)(query_tokens)                # [B,Q,C]
mask_features = upscale_block(image_tokens)                     # [B,C,4G,4G]
mask_logits = einsum("bqc,bchw->bqhw", mask_embed, mask_features)
```

Upscale block, repeated `num_upscale_blocks=2` in observed checkpoints:

```text
NCHW -> ConvTranspose2d(k=2,s=2) -> activation -> depthwise Conv2d(k=3,p=1) -> channel LayerNorm
```

## 6. Attention requirements

EoMT uses encoder-style dense self-attention, not generation attention.

- Causal: no.
- Self/cross: self-attention only, but final blocks include learned query tokens in the same sequence and can mask query-to-image attention based on prior mask logits.
- MHA/GQA/MQA: MHA only; `num_key_value_heads` is absent.
- Heads: observed `(6,12,16,24,32)`; head dim is 64 except 7B uses 128.
- Mask: final-block predicted mask creates `[B,S,S]` boolean mask, sets query rows and image-token columns from interpolated mask logits `> 0`, expands to `[B,H,S,S]`, then converts disallowed entries to `-1e9`.
- Packed/varlen: none in source.
- Sliding/local attention: none; semantic sliding windows are preprocessing crops, not attention windows.
- Backend: `_supports_sdpa=True`; source dispatches via `ALL_ATTENTION_FUNCTIONS`, with eager fallback softmax upcast to fp32.
- Cache: no KV cache or decode state.

## 7. Position encoding and custom math

Absolute learned patch positions are added before CLS/register concatenation. CLS/register/query tokens do not receive position embeddings in source.

```python
patches = conv2d(pixel_values).flatten(2).transpose(1, 2)
patches = patches + position_embeddings(position_ids)
hidden = cat([cls.expand(B), register.expand(B), patches], dim=1)
```

Mask-constrained attention setup:

```python
mask = ones(B, S, S, dtype=bool)
mask[:, :Q, Q + 1 + R :] = interpolate(mask_logits, size=(G, G)).view(B, Q, G * G) > 0
mask = mask[:, None].expand(-1, heads, -1, -1).float().masked_fill(~mask, -1e9)
```

Inference should set or freeze `attn_mask_probs` to one for source-default parity. Any value below one introduces random query mask disabling.

## 8. Preprocessing and input packing

Processor output is channels-first `pixel_values`. Default math: convert/prepare image, resize, rescale by `0.00392156862745098`, normalize by ImageNet mean/std, optional split or pad, return `BatchFeature`.

Instance/panoptic checkpoints:

- Resize to `shortest_edge == longest_edge == image_size`, then zero-pad to target square if needed.
- Model batch generally matches input image batch.
- Postprocess unpads logits to resized content size and upsamples to original `target_sizes`.

Semantic checkpoints:

- Resize with `shortest_edge=image_size`, `longest_edge=None`.
- Split along the longer side into overlapping crops of size `image_size`; emit `patch_offsets = [image_index, start, end]`.
- Model batch is number of patches, not number of original images.
- Postprocess averages overlapping segmentation logits by `patch_counts`, then resizes to original size.

Postprocess ABI:

- Semantic: interpolate mask logits to processor target size, `softmax(class)[..., :-1]`, `sigmoid(mask)`, `einsum("bqc,bqhw->bchw")`, stitch/resize, final `argmax` per pixel.
- Panoptic: interpolate/unpad, `softmax(class).max(-1)`, drop null class and low scores, per-pixel `argmax(score * sigmoid(mask))`, validity checks, merge stuff classes if provided.
- Instance: interpolate/unpad, drop null class before `max`, binary masks from `mask_pred > 0`, mask score from average sigmoid over positive pixels, threshold, sequential segment ids. No NMS.

## 9. Graph rewrite / lowering opportunities

### Rewrite: non-overlap patch Conv2d -> GEMM

Preconditions:

- `kernel_size == stride == patch_size`, `padding == 0`, `dilation == 1`, `groups == 1`.
- Input NCHW and H/W divisible by patch size.
- Flatten order must match PyTorch Conv2d then `flatten(2).transpose(1,2)`.

Replacement:

```text
NCHW WindowFlatten([ph,pw]) -> GEMM([3*ph*pw] x C) + bias -> [B,Gh*Gw,C]
```

Weight transform: `conv.weight.reshape(C, 3 * ph * pw)` with PyTorch linear using `weight.T`.

Failure cases: non-divisible runtime size, alternate layout without axis rewrite, non-square config paths that hit scalar `grid_size`.

### Rewrite: QKV projection packing

Source pattern: three independent biased linears `q_proj`, `k_proj`, `v_proj` from `[B,S,C]` to `[B,S,C]`.

Replacement: one packed GEMM `C -> 3C`, split in Q,K,V order.

Preconditions: same input, same dtype/device, all biases present, no hooks/recorded intermediate outputs required. Weight layout should be `[q_rows, k_rows, v_rows]` for an output-row-major packed linear.

### Rewrite: local NCHW `LayerNorm2d` region

Source pattern: `permute NCHW->NHWC`, `layer_norm(C)`, `permute NHWC->NCHW`.

Replacement: channel LayerNorm over NCHW with stride-aware kernel, or keep NHWC internally for the immediately surrounding upsample/depthwise-conv region only if conv providers support it.

Guards: source conv ops are NCHW; do not globally translate the whole model without rewriting Conv2d/ConvTranspose2d/depthwise axes and einsum operands.

### Rewrite: mask logits einsum -> batched GEMM

Source pattern: `einsum("bqc,bchw->bqhw")`.

Replacement: flatten spatial features to `[B,C,HW]`, run batched GEMM `[B,Q,C] x [B,C,HW] -> [B,Q,HW]`, reshape to `[B,Q,H,W]`.

Preconditions: dense contiguous or known strides; preserve dtype accumulation policy and output layout.

### Rewrite: semantic patch postprocess accumulation

Source pattern: loop over patch offsets, add logits and counts, divide by clamped counts, resize.

Replacement: deterministic scatter-add over non-overlapping/overlapping long-axis ranges with fixed offsets from processor.

Preconditions: patch offsets are trusted processor output; reject arbitrary offsets for first integration.

## 10. Kernel fusion candidates

Highest priority:

- Patch Conv2d-to-GEMM plus position add and prefix concat, because it is the first high-volume image ingestion path.
- ViT block fusions: LayerNorm, packed QKV GEMM, attention, output projection, residual LayerScale add.
- Dense MHA/SDPA for long image sequences: 640 large uses sequence 1605 then 1805; 1280 large uses 6405 then 6605.
- Mask head einsum as BMM/GEMM, because `[B,Q,C] x [B,C,4G*4G]` is a dominant segmentation-head matmul.

Medium priority:

- Upscale block NCHW kernels: ConvTranspose2d, activation, depthwise Conv2d, channel LayerNorm.
- SwiGLU FFN fusion for giant/7B variants.
- Final-block attention mask construction from mask logits, including interpolate to patch grid and threshold.
- Postprocess softmax/sigmoid/einsum/interpolate for semantic output when end-to-end segmentation latency matters.

Lower priority:

- Training-only losses and Hungarian matcher.
- DropPath/random mask disabling; inference should avoid random paths.
- PIL/Torchvision CPU preprocessing acceleration inside DinoML runtime; this can remain data-pipeline-owned initially.

## 11. Runtime staging plan

Stage 1: parse EoMT config and load one small/base checkpoint; enforce square scalar `image_size`, `patch_size=16`, NCHW input, no training labels.

Stage 2: single-block and whole-encoder parity without query insertion using random weights/tensors.

Stage 3: full forward parity for fixed padded panoptic/instance path, including query insertion and mask logits/class logits.

Stage 4: add final-block predicted attention mask construction; verify eval parity with `attn_mask_probs == 1`.

Stage 5: postprocess parity for panoptic and instance outputs. Keep variable-length record assembly in host code at first.

Stage 6: semantic sliding-window ABI: processor-owned patch split, batched patch inference, patch-offset stitch/resize.

Stage 7: optimize: QKV packing, GEMM patch embedding, SDPA, mask einsum BMM, upsample/depthwise conv providers, optional NHWC-local guarded fusions.

Stubs allowed initially: losses, Hungarian matcher, training segmentation map preprocessing, random mask disabling, DINOv3 EoMT, and 7B if memory/provider coverage is not ready.

## 12. Parity and validation plan

- Random op tests: patch Conv2d rewrite, `LayerNorm2d`, ConvTranspose2d block, depthwise Conv2d, mask-head BMM rewrite, final-block attention mask construction.
- Single EomtLayer parity in fp32 with fixed shape `[B,S,C]` and no mask, then with `[B,H,S,S]` mask.
- Whole model random-weight parity for small config at `image_size=64` or source test config, checking logits shape and max error.
- Checkpoint parity for `tue-mps/coco_panoptic_eomt_large_640`: expected shapes `[1,200,134]` and `[1,200,160,160]`.
- Semantic checkpoint parity for `tue-mps/ade20k_semantic_eomt_large_512`: representative source test expects two patches for a COCO sample and shapes `[2,100,151]`, `[2,100,128,128]`.
- Postprocess parity: semantic argmax map shape, panoptic/instance `segments_info` ordering and thresholds.
- Tolerances: fp32 `rtol=1e-4, atol=1e-4` for isolated ops; fp16/bf16 `rtol=1e-2, atol=1e-2` for checkpoint logits, matching source integration tolerance style.

## 13. Performance probes

- Processor throughput split by padded instance/panoptic versus semantic sliding-window split.
- Encoder throughput by image size: 512, 640, 1024, 1280.
- Final query-block overhead: compare sequence length before and after query injection.
- Attention backend comparison for `S=1029/1229`, `1605/1805`, `4101/4201`, `6405/6605`.
- Upscale/mask-head throughput: ConvTranspose/depthwise path plus `Q x C x HW` mask BMM.
- End-to-end postprocess latency for semantic stitching and panoptic/instance variable-length loops.
- Memory probes for attention masks: `[B,H,S,S]` is especially large for 1024/1280 variants.
- SwiGLU versus vanilla MLP throughput for giant/7B.

## 14. Skip/defer list

- Training losses, Hungarian matching, point sampling, auxiliary loss accounting.
- Random DropPath and random query-mask disabling.
- DINOv3 EoMT checkpoints; separate model family.
- Non-source remote-code variants or configs with `backbone` fields that current native source ignores/removes during conversion.
- NMS, boxes, and detection ABI; source is segmentation mask classification only.
- General NHWC translation; only local guarded layout regions should be optimized first.
- Multi-GPU/tensor parallel and quantized/packed checkpoint loading; source has no family-specific quantized runtime path.

## 15. Final implementation checklist

- [ ] Parse `EomtConfig` and reject unsupported `model_type`/DINOv3 variants.
- [ ] Load NCHW `pixel_values` ABI and processor metadata for padded versus split inference.
- [ ] Implement patch Conv2d embedding and absolute patch position add.
- [ ] Implement CLS/register/query token expansion and insertion at `layers - num_blocks`.
- [ ] Implement EoMT ViT block with LayerNorm, MHA/SDPA, LayerScale, vanilla MLP.
- [ ] Implement SwiGLU FFN for giant/7B admission.
- [ ] Implement predicted final-block attention mask construction.
- [ ] Implement mask head MLP, upscale block, and `bqc,bchw->bqhw`.
- [ ] Implement class predictor with null class output.
- [ ] Implement semantic postprocess including patch stitch.
- [ ] Implement panoptic and instance postprocess without NMS.
- [ ] Add guarded Conv2d patch embedding -> GEMM rewrite.
- [ ] Add QKV packing rewrite and parity tests.
- [ ] Add local `LayerNorm2d`/layout guard tests.
- [ ] Benchmark attention, mask BMM, upscale block, and postprocess separately.
