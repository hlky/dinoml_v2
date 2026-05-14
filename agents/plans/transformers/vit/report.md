# Transformers ViT Operator and Integration Report

## 1. Source basis

```text
Transformers commit/version:
  Local checkout X:/H/transformers at b75feb2af64c3e29cbbc1bd859958c5432cc7ed4.
  Remote upstream: https://github.com/huggingface/transformers.git

Model id:
  Primary worked example: google/vit-base-patch16-224.
  Additional sizing references: google/vit-base-patch16-224-in21k,
  google/vit-large-patch16-224, google/vit-large-patch32-384,
  WinKawaks/vit-small-patch16-224.

Config source:
  https://huggingface.co/google/vit-base-patch16-224/raw/main/config.json
  https://huggingface.co/google/vit-base-patch16-224/raw/main/preprocessor_config.json
  Additional configs and preprocessors fetched from Hugging Face repos listed above.
  HF plugin metadata confirmed common ViT repos as `vit`, mostly image-classification
  or image-feature-extraction.

Source files inspected:
  X:/H/transformers/src/transformers/models/vit/modeling_vit.py
  X:/H/transformers/src/transformers/models/vit/configuration_vit.py
  X:/H/transformers/src/transformers/models/vit/image_processing_vit.py

Any missing files or assumptions:
  No remote-code files are required for standard ViT. This report prioritizes
  image classification and base image-feature extraction. Masked image modeling
  is documented as optional/deferred. Dinoml assumption: prefer NHWC/channel-last
  layout for vision tensors and eliminate source NCHW permutes where legal.
```

## 2. High-level architecture

ViT is a vision encoder transformer. The processor resizes/rescales/normalizes images, the model patchifies with a non-overlapping Conv2d, prepends a learned CLS token, adds learned absolute position embeddings, runs bidirectional transformer encoder blocks, then uses the CLS token for classification.

```text
image preprocessing -> patch embedding -> CLS + position embeddings -> ViT encoder -> CLS classifier/logits
```

Stage decomposition:

- CPU/data pipeline: resize, rescale, normalize, arrange `pixel_values`.
- Patch stage: source NCHW Conv2d patch embed; Dinoml should prefer NHWC patch flatten -> GEMM.
- Encoder stage: token-sequence transformer, layout independent after patch tokens are `[B,N,H]`.
- Head stage: CLS select and classifier GEMM, independently testable.

## 3. Important config dimensions

Worked example: `google/vit-base-patch16-224`.

| Field | Value | Source |
|---|---:|---|
| primary runtime target | image classification | HF repo metadata |
| hidden_size / H | 768 | config.json |
| num_hidden_layers | 12 | config.json |
| num_attention_heads / A | 12 | config.json |
| head_dim / D | 64 | inferred `H/A` |
| intermediate_size / I | 3072 | config.json |
| image_size | 224 | config.json/preprocessor |
| patch_size | 16 | config.json |
| num_channels | 3 | config.json |
| patches | 196 | `(224/16)^2` |
| sequence length | 197 | patches + CLS |
| qkv_bias | true | config.json |
| hidden_act | gelu | config.json |
| layer_norm_eps | 1e-12 | config.json |
| processor mean/std | [0.5]*3 / [0.5]*3 | preprocessor_config |
| dtype | not specified | config; source casts pixel values to patch weight dtype |

Representative checkpoint sweep:

| Checkpoint | H | I | layers | heads | image | patch | patches+CLS | task metadata |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| WinKawaks/vit-small-patch16-224 | 384 | 1536 | 12 | 6 | 224 | 16 | 197 | image-classification |
| google/vit-base-patch16-224 | 768 | 3072 | 12 | 12 | 224 | 16 | 197 | image-classification |
| google/vit-base-patch16-224-in21k | 768 | 3072 | 12 | 12 | 224 | 16 | 197 | image-feature-extraction |
| google/vit-large-patch16-224 | 1024 | 4096 | 24 | 16 | 224 | 16 | 197 | image-classification |
| google/vit-large-patch32-384 | 1024 | 4096 | 24 | 16 | 384 | 32 | 145 | image-classification |

## 3a. Family variation traps

- Source `pixel_values` layout is NCHW `[B,C,H,W]`; Dinoml should prefer NHWC/channel-last internally for vision memory behavior.
- Patch Conv2d has `kernel_size == stride == patch_size`, so it is an exact non-overlap patch projection and a prime Conv2d -> GEMM rewrite.
- Position embedding interpolation is optional and only used when `interpolate_pos_encoding=True` or tracing; fixed-size inference can use the static position table.
- Classification and feature-extraction checkpoints share the encoder but differ in head needs. Masked image modeling has a decoder with Conv2d + PixelShuffle and should be a separate optional target.
- Patch size 16 vs 32 changes sequence length and patch projection K dimension.

## 4. Operator coverage checklist

### Tensor/layout ops

- Source NCHW image input `[B,C,H,W]`; preferred Dinoml NHWC `[B,H,W,C]`.
- Patch flatten/window extraction for non-overlapping patches.
- Conv2d patch embed source output `[B,Hid,Hp,Wp] -> flatten(2).transpose(1,2) -> [B,N,Hid]`.
- CLS token expand and concat.
- Position embedding add.
- Optional bool masked position replacement for masked image modeling.
- CLS token select `sequence[:,0,:]`.

### Neural network primitives

- Patch Conv2d:
  - base: `Conv2d(3 -> 768, kernel=16, stride=16, bias=True)` equivalent to `Linear(768 input? no: 3*16*16=768 -> 768)`.
  - large patch32: `Linear(3*32*32=3072 -> 1024)`.
- Linear with bias for Q/K/V/O, MLP, pooler, classifier.
- LayerNorm with bias/scale.
- GELU.
- Tanh pooler activation for `ViTModel` when pooling layer is enabled.
- Dropout is zero in common configs/source inference.

### Attention primitives

- Bidirectional noncausal MHA.
- No GQA/MQA, no RoPE, no cache.
- Source dispatch uses `ALL_ATTENTION_FUNCTIONS`; eager fallback is matmul + optional mask + fp32 softmax + matmul.
- Attention mask is optional and made bidirectional from input masks.

### Position/rotary/relative-bias ops

- Learned absolute position table `[1,N+1,H]`.
- Optional bicubic interpolation of patch position embeddings for different image sizes.

### Preprocessing-coupled ops

- Image processor resize to configured size, rescale, normalize.
- Processor outputs `pixel_values`; source expects NCHW, but Dinoml frontend can accept NHWC and lower consistently.

### Distributed/tensor-parallel ops

- No explicit TP plan in ViT config. Large ViT may benefit from sharded MLP/attention GEMMs but not required for first integration.

## 5. Layer/block breakdown

Patch/embedding path:

```text
source: pixel_values [B,3,H,W] NCHW
patch = Conv2d(3 -> Hdim, kernel=P, stride=P)(pixel_values)
patch = flatten spatial and transpose -> [B,(H/P)*(W/P),Hdim]
tokens = concat(cls_token[B,1,Hdim], patch)
tokens = tokens + position_embeddings
```

Preferred Dinoml NHWC lowering:

```text
pixel_values [B,H,W,3] NHWC
patches = nonoverlap_window_flatten_NHWC(pixel_values, P, P)  # [B*Hp*Wp, P*P*3]
patch = GEMM_RCR_Bias(patches, weight_reordered)             # [B*Hp*Wp,Hdim]
tokens = reshape [B,Hp*Wp,Hdim] and concat CLS
```

Encoder block, repeated `N` times:

```text
residual = x
y = LayerNorm(x)
q,k,v = Linear(H -> H, bias=qkv_bias)(y)
y = bidirectional_attention(q,k,v)
y = Linear(H -> H, bias=True)(y)
x = residual + y

residual = x
y = LayerNorm(x)
y = Linear(H -> I, bias=True)(y)
y = GELU(y)
y = Linear(I -> H, bias=True)(y)
x = residual + y
```

Head:

```text
x = final LayerNorm(x)
cls = x[:,0,:]
logits = Linear(H -> num_labels, bias=True)(cls)
```

## 6. Attention requirements

- Noncausal self-attention only.
- MHA: base `A=12,D=64`; large `A=16,D=64`; small `A=6,D=64`.
- Optional additive bidirectional mask.
- No KV cache, no causal mask, no sliding window, no RoPE/relative bias.
- SDPA/Flash-style noncausal attention is compatible. Sequence lengths are small for 224/16 (197) and 384/32 (145), so patch/MLP/head costs may matter as much as attention.

## 7. Position encoding and custom math

Static path:

```python
tokens = tokens + position_embeddings  # [1, num_patches + 1, hidden]
```

Interpolation path for higher resolution:

```python
cls = pos[:, :1]
patch = pos[:, 1:].reshape(1, sqrt_n, sqrt_n, H).permute(0, 3, 1, 2)
patch = bicubic_interpolate(patch, size=(new_h, new_w), align_corners=False)
patch = patch.permute(0, 2, 3, 1).reshape(1, new_h * new_w, H)
pos = concat([cls, patch], dim=1)
```

Precompute:

- Static position embeddings for fixed configured image size.
- Interpolated position embeddings per resolution bucket.

## 8. Preprocessing and input packing

CPU/data-pipeline:

- Resize to preprocessor size, commonly 224 or 384.
- Rescale and normalize with mean/std from preprocessor config, commonly `[0.5,0.5,0.5]`.

GPU/runtime:

- Source contract is NCHW `pixel_values`.
- Dinoml should prefer NHWC input or normalize to NHWC immediately, then lower patch projection without NCHW permute.
- No modality placeholders, token type IDs, grid metadata, or packed varlen descriptors.

## 9. Graph rewrite / lowering opportunities

### Rewrite: non-overlap patch Conv2d -> NHWC Linear

Preconditions:

- `Conv2d(in=C,out=Hid,kernel=(P,P),stride=(P,P),padding=0,dilation=1,groups=1)`.
- Input H/W divisible by P.
- Source pattern is `Conv2d -> flatten(2) -> transpose(1,2)`.
- Dinoml internal activation is NHWC or can prove the input layout.

Replacement:

```text
NHWCWindowFlatten(P,P) -> GEMM_RCR_Bias -> Reshape[B,N,Hid]
```

Weight transform:

```python
# source conv weight [O, C, P, P], activation flatten NHWC order [kh, kw, c]
w = conv.weight.permute(0, 2, 3, 1).reshape(O, P * P * C)
```

Failure cases:

- Non-divisible image dimensions without interpolation/bucketing guard.
- If source NCHW is preserved, use `[C,kh,kw]` flatten order instead.

Parity test sketch:

- Compare Conv2d+flatten+transpose against NHWC flatten GEMM for fp32/fp16/bf16.

### Rewrite: QKV projections -> packed QKV

Preconditions:

- Same LayerNorm output feeds q/k/v.
- q/k/v have equal output width and same bias setting.

Replacement:

```text
PackedLinear(H -> 3H, bias=True) -> Split(q,k,v)
```

Weight transform:

```python
w_qkv = concat([w_q, w_k, w_v], axis=0)
b_qkv = concat([b_q, b_k, b_v], axis=0)
```

### Rewrite: position interpolation cache

Preconditions:

- Resolution bucket is known.
- Position table is constant.

Replacement:

```text
BicubicPositionInterpolation -> CachedPositionTable[resolution]
```

Failure cases:

- Dynamic arbitrary image sizes without bucketization.

### Rewrite: CLS-only classifier head

Preconditions:

- Task is classification and only CLS logits are required.

Replacement:

```text
FinalLayerNorm -> SelectCLS -> ClassifierGEMM
```

Failure cases:

- Feature extraction or masked image modeling needs all tokens.

## 10. Kernel fusion candidates

Highest priority:

- Patch Conv2d -> NHWC GEMM with compile-time weight permutation; removes NCHW Conv2d/flatten/transpose pattern.
- LayerNorm + residual scheduling for transformer blocks.
- Packed QKV projection with bias.
- GELU MLP activation fusion.

Medium priority:

- Noncausal SDPA/Flash attention for short ViT sequences.
- Embedding concat/add fusion for CLS + position.
- Position interpolation cache per image-size bucket.
- CLS select + classifier GEMM.

Lower priority:

- Masked image modeling decoder Conv2d + PixelShuffle.
- Pooler dense+tanh if only `ViTModel` pooler output is targeted.

## 11. Runtime staging plan

Stage 1: Parse config and processor config; load patch, encoder, norm, classifier weights.

Stage 2: Implement preprocessing-compatible NHWC input and patch projection parity.

Stage 3: Implement one encoder block parity.

Stage 4: Full encoder and image classification logits parity.

Stage 5: Add position interpolation buckets for non-default resolutions.

Stage 6: Add NHWC Conv2d->GEMM rewrite/fusion and packed QKV.

Stage 7: Optionally add masked image modeling decoder.

## 12. Parity and validation plan

- Processor output parity or documented handoff contract for `pixel_values`.
- Patch projection parity for NCHW source and NHWC rewritten path.
- Position embedding static and interpolated parity.
- One attention and one block parity.
- Full `ViTModel` last hidden state and CLS parity.
- Image classification logits parity for base/small/large.
- Suggested tolerances: fp32 `rtol=1e-5, atol=1e-6`; fp16/bf16 `rtol=2e-2, atol=2e-2` initially.

## 13. Performance probes

- Image processor throughput separately from runtime.
- Patch projection NCHW Conv2d vs NHWC GEMM.
- Encoder throughput over batch size and resolution.
- Attention vs MLP time for sequence lengths 145 and 197.
- Position interpolation cost with/without cache.
- End-to-end images/sec for classification.

## 14. Skip/defer list

- Training/loss.
- Masked image modeling decoder and PixelShuffle.
- Arbitrary dynamic image sizes without buckets.
- Multi-GPU sharding.
- Quantization.

## 15. Final implementation checklist

- [ ] Parse ViTConfig and preprocessor config.
- [ ] Define NHWC runtime image layout contract.
- [ ] Load patch Conv2d, CLS, position, encoder, norm, and classifier weights.
- [ ] Implement patch Conv2d source parity.
- [ ] Add NHWC patch Conv2d -> GEMM rewrite with weight permutation.
- [ ] Implement CLS/position embedding path.
- [ ] Implement noncausal MHA and MLP blocks.
- [ ] Add full classifier parity.
- [ ] Add position interpolation cache/parity.
- [ ] Benchmark patch projection, encoder, attention, MLP, and end-to-end images/sec.
