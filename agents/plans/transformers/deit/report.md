# DeiT Transformers family audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: primary native examples are facebook/deit-base-distilled-patch16-224 and facebook/deit-base-distilled-patch16-384
Config source: official Hugging Face config/preprocessor JSON downloaded 2026-05-13
Source files inspected: src/transformers/models/deit/{configuration_deit.py,image_processing_deit.py,image_processing_pil_deit.py,modeling_deit.py,modular_deit.py}
Any missing files or assumptions: no gated/401/403 gaps. `modeling_deit.py` is generated from `modular_deit.py`; future Transformers source edits should target `modular_deit.py`, but runtime behavior here is read from generated `modeling_deit.py`.
```

Pinned source URLs:

- [modeling_deit.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/deit/modeling_deit.py)
- [modular_deit.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/deit/modular_deit.py)
- [configuration_deit.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/deit/configuration_deit.py)
- [image_processing_deit.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/deit/image_processing_deit.py)
- [image_processing_pil_deit.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/deit/image_processing_pil_deit.py)

Local snapshots and representative configs are under `agents/plans/transformers/deit/_sources/`.

## 2. High-level architecture

DeiT is a vision-only encoder transformer for image classification and optional masked image modeling. Primary DinoML target: inference for `DeiTForImageClassificationWithTeacher` on distilled checkpoints, plus base encoder and single-head classification as lower-risk subtargets.

Dataflow:

```text
CPU/GPU image processor -> NCHW pixel_values -> non-overlap Conv2d patch embedding
-> flatten to token sequence -> prepend CLS + distillation tokens
-> add learned absolute position embeddings, optionally bicubic-interpolated
-> repeated encoder blocks -> final LayerNorm
-> classification head(s) on token 0 and optionally token 1 -> logits
```

Stage decomposition:

- CPU/data-pipeline: RGB conversion if requested by caller, resize, optional center crop, rescale, normalize, output `pixel_values`.
- Encoder: patch `Conv2d`, token concat, absolute position add, noncausal self-attention/MLP stack.
- Heads: base `DeiTModel` can pool token 0 through dense+tanh; classification models skip pooler and read token 0; teacher variant reads tokens 0 and 1 and averages two classifier outputs.
- Feature/backbone outputs: no nested `backbone_config`, AutoBackbone, or image-like feature pyramid. The only feature contract is the token sequence `last_hidden_state` shaped `[B, 2 + num_patches, hidden_size]`, plus optional hidden-state/attention captures from the encoder.
- Deferred for first vision inference: masked image modeling decoder and training losses.

## 3. Important config dimensions

Source defaults from `DeiTConfig`:

| Field | Default | Runtime effect |
|---|---:|---|
| `hidden_size` | 768 | token width and classifier input |
| `num_hidden_layers` | 12 | encoder block count |
| `num_attention_heads` | 12 | MHA head count |
| effective `head_dim` | `hidden_size // num_attention_heads` unless config has `head_dim` | Q/K/V per-head width |
| `intermediate_size` | 3072 | MLP expansion width |
| `hidden_act` | `gelu` | MLP activation |
| `image_size` | 224 | static image guard when not interpolating positions |
| `patch_size` | 16 | Conv2d kernel and stride |
| `num_channels` | 3 | `pixel_values` channel guard |
| `qkv_bias` | `True` | Q/K/V Linear bias; output projection always has bias |
| `layer_norm_eps` | `1e-12` | all LayerNorm eps |
| `hidden_dropout_prob` / `attention_probs_dropout_prob` | 0.0 | inactive in eval; still present in source |
| `encoder_stride` | 16 | masked image modeling decoder only |
| `pooler_output_size` | defaults to `hidden_size` | base `DeiTModel` pooler only |

Representative checkpoint sweep:

| Checkpoint | Native source scope | Architecture in config | H | Layers | Heads | MLP | Image | Patch | Labels | Processor notes |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `hf-internal-testing/tiny-random-DeiTModel` | in-scope debug | `DeiTModel` | 32 | 5 | 4 | 37 | 30 | 2 | default | `DeiTImageProcessor`, resize/crop 30, mean/std 0.5 |
| `facebook/deit-base-distilled-patch16-224` | in-scope | `DeiTForImageClassificationWithTeacher` | 768 | 12 | 12 | 3072 | 224 | 16 | 1000 | legacy `DeiTFeatureExtractor`, resize 256, center crop 224, ImageNet mean/std |
| `facebook/deit-base-distilled-patch16-384` | in-scope | `DeiTForImageClassificationWithTeacher` | 768 | 12 | 12 | 3072 | 384 | 16 | 1000 | resize 438, center crop 384, ImageNet mean/std |
| `facebook/deit-tiny-patch16-224` | route to ViT unless aliased | `ViTForImageClassification`, `model_type: vit` | 192 | 12 | 3 | 768 | 224 | 16 | 1000 | resize 224, mean/std 0.5, no center-crop field |
| `facebook/deit-small-patch16-224` | route to ViT unless aliased | `ViTForImageClassification`, `model_type: vit` | 384 | 12 | 6 | 1536 | 224 | 16 | 1000 | resize 224, mean/std 0.5 |
| `facebook/deit-base-patch16-224` | route to ViT unless aliased | `ViTForImageClassification`, `model_type: vit` | 768 | 12 | 12 | 3072 | 224 | 16 | 1000 | resize 224, mean/std 0.5 |

Downloaded distilled configs omit `qkv_bias`, `encoder_stride`, `pooler_output_size`, and `pooler_act`; under the inspected `DeiTConfig` their effective defaults are `qkv_bias=True`, `encoder_stride=16`, `pooler_output_size=hidden_size`, and `pooler_act="tanh"`. Those fields are source defaults, not explicit facts from the older checkpoint JSON.

## 3a. Family variation traps

- Native `deit` embeddings always include two special tokens: CLS at sequence index 0 and distillation at index 1. Non-distilled official DeiT-named checkpoints above have `model_type: "vit"` and should be handled by the ViT audit unless DinoML adds an explicit compatibility alias.
- `head_dim` is not declared in `DeiTConfig`, but `DeiTAttention` honors a config-provided `head_dim` if present. Admission should reject or test `hidden_size != num_attention_heads * head_dim` because projection output width becomes `num_heads * head_dim`.
- `qkv_bias` controls Q/K/V bias only. Attention output projection and MLP/classification Linear layers have bias.
- The static position table has `num_patches + 2` rows. Interpolation preserves the first two special-token rows and bicubic-resizes only patch rows.
- Source tensors are NCHW through image processing and patch `Conv2d`, then `[B, S, H]` token-major. NHWC is an optimization candidate only inside a guarded patch-embedding/layout pass.
- Processor configs differ materially: distilled checkpoints use ImageNet mean/std and 256/438 resize before crop; some legacy non-distilled configs use mean/std 0.5 and omit center crop.

## 4. Operator coverage checklist

Tensor/layout ops:

- NCHW image tensor validation and optional dtype cast to patch projection weight dtype.
- Conv output flatten over spatial dims: `[B, Hhidden, Gh, Gw] -> [B, Hhidden, Gh*Gw]`.
- Transpose/contiguous/view/reshape around tokens and attention heads.
- Expand CLS/distillation parameters to batch, concat on sequence dim 1.
- Slice/index tokens: `[:, 0, :]`, `[:, 1, :]`, and masked-image `[:, 2:]`.
- Optional bicubic interpolate of position embeddings on NCHW-like `[1, hidden, grid_h, grid_w]`.

Neural network primitives:

- `Conv2d(C -> H, kernel=patch, stride=patch, padding=0, groups=1, bias=True)`.
- LayerNorm over hidden dim, eps usually `1e-12`.
- Linear Q/K/V: `H -> num_heads * head_dim`, Q/K/V bias controlled by `qkv_bias`.
- Linear attention output: `num_heads * head_dim -> H`, bias true.
- MLP: `Linear(H -> I)`, GELU or `ACT2FN[hidden_act]`, `Linear(I -> H)`.
- Classification heads: one or two `Linear(H -> num_labels)`, bias true, then optional average.
- Base pooler: token 0, `Linear(H -> pooler_output_size)`, tanh by default.

Attention primitives:

- Noncausal encoder self-attention only, MHA, no KV cache.
- Eager path: QK matmul, additive mask if present, softmax in fp32 then cast back, dropout, AV matmul.
- Source advertises SDPA/Flash/Flex support through `ALL_ATTENTION_FUNCTIONS`; DinoML can start with dense noncausal attention and later map to fused attention.

Preprocessing-coupled ops:

- Resize, center crop, rescale by 1/255 when enabled, channel-wise normalize, output `pixel_values`.
- Processor returns channels-first tensors for the model contract.

Deferred for first classification target:

- Mask token replacement, masked image modeling 1x1 Conv2d + PixelShuffle decoder, L1 reconstruction loss.

## 5. Layer/block breakdown

Patch embedding:

```text
pixel_values: [B, C, Himg, Wimg]
x = Conv2d(C -> Hhidden, kernel=(P_h,P_w), stride=(P_h,P_w))(pixel_values)
x = flatten spatial dims -> [B, Hhidden, Gh*Gw]
x = transpose(1, 2) -> [B, Gh*Gw, Hhidden]
```

Embeddings:

```text
cls = expand([1,1,Hhidden] -> [B,1,Hhidden])
dist = expand([1,1,Hhidden] -> [B,1,Hhidden])
x = concat([cls, dist, patches], dim=1) -> [B, 2 + Gh*Gw, Hhidden]
x = x + position_embeddings or interpolated_position_embeddings
```

Encoder block, repeated `num_hidden_layers`:

```text
residual = x
y = LayerNorm(x)
q = Linear(H -> heads*head_dim, bias=qkv_bias)(y).view(B,S,heads,head_dim).transpose(1,2)
k = same
v = same
attn = softmax((q @ k^T) * head_dim**-0.5 + mask, dim=-1)
y = (attn @ v).transpose(1,2).reshape(B,S,heads*head_dim)
y = Linear(heads*head_dim -> H, bias=True)(y)
x = residual + Dropout(y)
residual = x
y = LayerNorm(x)
y = Linear(H -> intermediate, bias=True)(y)
y = GELU(y)
y = Linear(intermediate -> H, bias=True)(y)
x = residual + Dropout(y)
```

Output/head:

```text
sequence = LayerNorm(x)
classification: logits = classifier(sequence[:, 0, :])
teacher: cls_logits = cls_head(sequence[:, 0, :]); dist_logits = dist_head(sequence[:, 1, :]); logits = (cls_logits + dist_logits) / 2
```

## 6. Attention requirements

- Type: bidirectional/noncausal self-attention in encoder blocks.
- Heads: MHA only in official configs; no GQA/MQA fields in `DeiTConfig`.
- Shapes: hidden `[B,S,H]`; Q/K/V `[B,heads,S,head_dim]`; attention scores `[B,heads,S,S]`; output `[B,S,H]`.
- Masking: `DeiTModel.forward` calls `create_bidirectional_mask`. With no padding/custom attention mask, optimized backends may receive `None`; explicit masks are additive/broadcasted according to Transformers mask utilities.
- Packed/varlen/sliding/local: not used by native source.
- RoPE/ALiBi/relative bias: none.
- KV cache: not applicable. This is an encoder-style image model; no prefill/decode state ABI.
- Backend parity: eager path upcasts softmax to fp32. Fused attention should preserve scale-before-mask, additive mask before softmax, fp32 softmax accumulation where required, and no causal flag.

## 7. Position encoding and custom math

DeiT uses learned absolute position embeddings with two special rows.

```python
def deit_interpolate_pos_encoding(position_embeddings, embeddings, height, width, patch_size):
    num_patches = embeddings.shape[1] - 2
    num_positions = position_embeddings.shape[1] - 2
    if num_patches == num_positions and height == width:
        return position_embeddings
    special = position_embeddings[:, :2]
    patch = position_embeddings[:, 2:]
    dim = embeddings.shape[-1]
    old = int(num_positions ** 0.5)
    new_h, new_w = height // patch_size, width // patch_size
    patch = patch.reshape(1, old, old, dim).permute(0, 3, 1, 2)
    patch = bicubic_interpolate(patch, size=(new_h, new_w), align_corners=False)
    patch = patch.permute(0, 2, 3, 1).reshape(1, new_h * new_w, dim)
    return concat([special, patch], dim=1)
```

For fixed-size 224 or 384 inference, position embeddings can be loaded as constants and added directly. Interpolation depends on runtime `height,width` only when `interpolate_pos_encoding=True`; DinoML can initially reject dynamic image sizes without that flag and add bicubic interpolation later.

## 8. Preprocessing and input packing

Native `DeiTImageProcessor` defaults: bicubic resize to 256x256, center crop 224x224, rescale, normalize with ImageNet standard mean/std, output `pixel_values`. The PIL variant has the same defaults but CPU/PIL backend. Actual downloaded checkpoint preprocessor configs override these:

- Distilled 224: resize 256, center crop 224, mean/std `[0.485,0.456,0.406]` / `[0.229,0.224,0.225]`.
- Distilled 384: resize 438, center crop 384, same ImageNet mean/std.
- Legacy non-distilled ViT-routed checkpoints: resize 224 and mean/std `[0.5,0.5,0.5]`.
- Tiny random: resize/crop 30, rescale factor `1/255`, mean/std 0.5.

Model input contract is `pixel_values` shaped `[B, C, H, W]`, usually float, channel-first. The model casts `pixel_values` to the patch projection weight dtype before embedding.

No tokenizer, placeholder tokens, packed sequence metadata, or multimodal stitching is present. `bool_masked_pos` is only for masked image modeling and has shape `[B, num_patches]`.

## 9. Graph rewrite / lowering opportunities

### Rewrite: non-overlap Conv2d patch embedding to Linear/GEMM

Source pattern:

```text
Conv2d(C -> H, kernel=patch, stride=patch, padding=0).flatten(2).transpose(1,2)
```

Replacement:

```text
WindowFlatten(NCHW, patch_h, patch_w, row-major spatial order) -> MatMul(weight_flat.T) -> BiasAdd -> [B, Gh*Gw, H]
```

Preconditions:

- `kernel_size == stride == patch_size`
- `padding == 0`, `dilation == 1`, `groups == 1`
- input `Himg,Wimg` divisible by patch dims
- source spatial order matches PyTorch Conv2d output order: `Gh` major then `Gw`
- weight transform: `w_flat = conv.weight.reshape(out_channels, in_channels * patch_h * patch_w)`
- bias optional but current Conv2d includes bias by default

Failure cases: overlapping patches, dynamic non-divisible sizes, nonstandard groups/dilation/padding, NHWC layout without a verified window flatten permutation.

Parity sketch: compare patch embedding output before token concat for random NCHW inputs and official weights.

### Rewrite: Q/K/V Linear fusion

Source pattern: three independent `Linear(H -> heads*head_dim)` projections from the same normalized hidden states.

Replacement: one packed GEMM `H -> 3*heads*head_dim`, split in source order `[q, k, v]`.

Preconditions: same input tensor, all projections same dtype/device, same bias policy, no observers/hooks, output split order preserved.

Weight transform: concatenate rows/out_features as `[q_proj.weight; k_proj.weight; v_proj.weight]`, same for biases when present.

### Rewrite: special-token concat as constant prefix

Source pattern: expand `cls_token` and `distillation_token`, concat with patch tokens, then add position embeddings.

Replacement: materialize prefix rows through broadcasted constant views and a sequence concat or fused prefix-copy kernel.

Preconditions: token count fixed or known after interpolation, no masked image modeling token replacement in the same fused region.

### Rewrite: guarded NCHW to NHWC patch region

Candidate only for patch embedding and preprocessing-adjacent kernels. If using NHWC/channel-last internally:

- Rewrite Conv2d input channel axis from dim 1 to dim -1.
- Transform Conv weights from `[out,in,kh,kw]` to the provider's expected NHWC/filter layout.
- Preserve output token order after flatten.
- Insert `no_layout_translation()` guard before token sequence ops and attention, which are `[B,S,H]` and not image-layout tensors.

Do not translate LayerNorm, softmax, concat sequence dim, or token indexing as image layout operations.

### Rewrite: teacher head average fusion

Source pattern: two classifier GEMMs followed by elementwise add and multiply by 0.5.

Replacement: keep as two GEMMs plus fused average, or if weights are static and both heads share shape, concatenate heads in one GEMM then split/average.

Preconditions: both heads present, same `num_labels`, no need to expose `cls_logits`/`distillation_logits` separately for the optimized path unless outputs are requested.

## 10. Kernel fusion candidates

Highest priority:

- Patch Conv2d to GEMM or dedicated non-overlap patch kernel: first large image-side op and layout boundary.
- Encoder LayerNorm + QKV projection: repeated 12 times and feeds attention.
- Dense noncausal attention over short ViT sequences: 197 tokens for 224 and 578 tokens for 384 in distilled configs.
- MLP GEMM + GELU + GEMM scheduling/fusion, especially `H=768, I=3072`.

Medium priority:

- QKV packed projection and attention output projection candidate selection.
- Residual add after attention/MLP with dropout elided in eval.
- Final LayerNorm + token gather for classification heads.
- Teacher two-head averaging.

Lower priority:

- Base pooler dense+tanh, because classification heads do not use it.
- Bicubic position interpolation for non-default image sizes.
- Masked image modeling decoder Conv2d + PixelShuffle.

## 11. Runtime staging plan

1. Parse `DeiTConfig` and reject `model_type: vit` legacy checkpoints unless routed to ViT.
2. Load weights for `DeiTModel` and distilled teacher classification head.
3. Implement patch embedding and embedding assembly parity for fixed image size.
4. Implement one encoder block parity with eager noncausal attention.
5. Run full encoder parity and final LayerNorm.
6. Add `DeiTForImageClassificationWithTeacher` heads and averaged logits.
7. Add single-head `DeiTForImageClassification`.
8. Add guarded optimizations: Conv2d-to-GEMM, QKV fusion, fused attention, MLP fusion.
9. Later: position interpolation, base pooler, masked image modeling.

Initial stubs: dropout as identity in eval, no training losses, no `output_attentions`, no dynamic image-size interpolation unless explicitly enabled.

## 12. Parity and validation plan

- Processor contract: compare `pixel_values` shape/range for one PIL image against Transformers processor for 224 and 384 distilled configs.
- Patch embedding parity: random `[B,3,H,W]` tensors, compare Conv2d+flatten+transpose.
- Position add parity: fixed-size table add and interpolated 224->384 or 224->larger path.
- Single-layer parity: one `DeiTLayer` with random weights/inputs, eager attention.
- After-N-layer parity: checkpoints at 1, 6, 12 layers for `last_hidden_state`.
- Head parity: teacher `cls_logits`, `distillation_logits`, and averaged `logits`; also single-head classification.
- End-to-end image parity: compare ImageNet logits for `facebook/deit-base-distilled-patch16-224` and 384 variant.
- Tolerances: fp32 `rtol=1e-4, atol=1e-5`; fp16/bf16 start with `rtol=5e-3, atol=5e-3`, tightening per backend.

## 13. Performance probes

- Image preprocessing throughput separately from model runtime.
- Patch embedding throughput for Conv2d vs GEMM rewrite, 224 and 384.
- Encoder-only throughput by batch size: 1, 8, 32, 64.
- Attention backend comparison for sequence lengths 197 and 578.
- MLP GEMM throughput and activation fusion impact.
- End-to-end logits throughput for distilled 224 and 384.
- Memory footprint for activations at 384, especially attention scores if using unfused eager attention.
- NHWC guarded patch-region benchmark vs faithful NCHW translation.

## 14. Skip/defer list

- Training, gradient checkpointing, dropout behavior in train mode, and loss functions.
- Masked image modeling and PixelShuffle decoder for first classification target.
- Dynamic arbitrary-resolution inference without `interpolate_pos_encoding` support.
- Returning full attention matrices and hidden-state capture unless needed for debugging parity.
- Legacy `model_type: vit` DeiT-named checkpoints in this family report; route to ViT or create an alias plan.
- Quantization, multi-GPU/tensor parallel, speculative/generation features.

## 15. Final implementation checklist

- [ ] Parse `DeiTConfig` and processor config.
- [ ] Reject or route `model_type: vit` DeiT-named checkpoints.
- [ ] Load patch embedding Conv2d, CLS token, distillation token, and `num_patches + 2` position embeddings.
- [ ] Implement NCHW patch embedding and `[B,S,H]` token assembly.
- [ ] Implement optional learned position interpolation with two special-token rows.
- [ ] Implement LayerNorm, MHA, GELU MLP encoder block.
- [ ] Implement bidirectional attention mask handling or admit maskless image inference first.
- [ ] Implement final LayerNorm and teacher classification heads.
- [ ] Add Conv2d patch embedding to GEMM rewrite with layout guards.
- [ ] Add QKV packed projection rewrite.
- [ ] Add parity tests for embeddings, one block, full encoder, and logits.
- [ ] Add performance probes for 224 and 384 distilled checkpoints.
