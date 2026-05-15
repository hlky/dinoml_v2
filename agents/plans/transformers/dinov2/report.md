# DinoV2 Transformers Audit

## 1. Source basis

Transformers commit/version: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4` in local checkout `transformers`.

Model family: `dinov2`

Primary runtime target: vision encoder/image feature extraction. `Dinov2ForImageClassification` is optional; `Dinov2Backbone` is useful for dense feature maps but can follow encoder parity.

Source files inspected:

- `transformers/src/transformers/models/dinov2/configuration_dinov2.py`
- `transformers/src/transformers/models/dinov2/modeling_dinov2.py`
- `transformers/src/transformers/models/dinov2/convert_dinov2_to_hf.py` only for source inventory
- `transformers/src/transformers/models/bit/image_processing_bit.py`, because official configs use `BitImageProcessor`
- `transformers/src/transformers/image_processing_backends.py` and `image_processing_utils.py` for the shared processor tensor contract
- `transformers/src/transformers/models/auto/image_processing_auto.py`, confirming `dinov2 -> BitImageProcessor`
- `transformers/src/transformers/models/dinov2_with_registers/*`, as a sibling variant check only

Representative configs/preprocessor configs fetched from official Hugging Face repos:

- `facebook/dinov2-small`
- `facebook/dinov2-base`
- `facebook/dinov2-large`
- `facebook/dinov2-giant`
- `facebook/dinov2-base-imagenet1k-1-layer`
- `facebook/dinov2-with-registers-base` and `facebook/dinov2-with-registers-giant` as out-of-family register-token traps

Pinned source URLs for future review:

- `https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/dinov2/modeling_dinov2.py`
- `https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/dinov2/configuration_dinov2.py`
- `https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/bit/image_processing_bit.py`

Missing files or assumptions:

- There is no `image_processing_dinov2.py` in the pinned `dinov2` directory. `AutoImageProcessor` maps `model_type="dinov2"` to `BitImageProcessor`.
- Plain `dinov2` has no register tokens. Register checkpoints use `model_type="dinov2_with_registers"` and a separate generated modeling file.
- This is docs-only. No DinoML tests or Transformers model execution were run.

## 2. High-level architecture

DinoV2 is a ViT-style image encoder:

```text
image preprocessing -> NCHW pixel_values -> non-overlap Conv2d patch embedding
  -> flatten patches to token sequence -> prepend CLS -> add interpolated learned absolute positions
  -> repeated pre-norm Transformer encoder blocks -> final LayerNorm
  -> pooled CLS embedding / patch features / optional classification logits
```

Base model outputs:

- `last_hidden_state`: `[B, 1 + Npatch, H]`
- `pooler_output`: final normalized CLS token `[B, H]`

Optional heads:

- Image classification: concatenate final CLS token with mean of final patch tokens, then `Linear(2H -> num_labels)`.
- Backbone: select hidden states by stage, optional final `LayerNorm`, remove CLS, reshape patch tokens to `[B, H, image_h / patch, image_w / patch]`.

Stage decomposition:

- CPU/data pipeline: RGB conversion, resize shortest edge, center crop, rescale, normalize, batch tensor assembly.
- Encoder GPU graph: patch embedding, token assembly, positional interpolation/add, Transformer blocks, final norm.
- Independently cacheable pieces: model weights, learned class token, learned absolute positional table, optional mask token. Interpolated positional table can be cached per `(height, width, dtype)` if input sizes repeat.
- Classification head: small optional tail after encoder parity.
- Backbone feature extraction: optional output formatting path with axis-sensitive reshape/permute.

## 3. Important config dimensions

Source defaults from `Dinov2Config`:

| Field | Default | Runtime significance |
| --- | ---: | --- |
| `hidden_size` | 768 | token width |
| `num_hidden_layers` | 12 | encoder block count |
| `num_attention_heads` | 12 | MHA head count |
| `head_dim` | 64 | inferred as `hidden_size / num_attention_heads` |
| `mlp_ratio` | 4 | standard MLP hidden width `H * 4` |
| `hidden_act` | `gelu` | standard MLP activation |
| `layer_norm_eps` | `1e-6` | all LayerNorms |
| `image_size` | 224 | used to size learned pos table at init; official configs use 518 |
| `patch_size` | 14 | official configs use non-overlap 14x14 patch conv |
| `num_channels` | 3 | processor emits RGB |
| `qkv_bias` | true | Q/K/V projections have optional bias |
| `layerscale_value` | 1.0 | per-channel learned scale after attention and MLP |
| `use_swiglu_ffn` | false | giant variants use SwiGLU |
| `use_mask_token` | true | pretraining-only optional patch mask replacement |

Representative checkpoint sweep:

| Model id | Architecture | H | Layers | Heads | Head dim | MLP | SwiGLU | Image/patch | Seq at 224 crop | Pos table from config | Processor |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- | ---: | ---: | --- |
| `facebook/dinov2-small` | `Dinov2Model` | 384 | 12 | 6 | 64 | `4H=1536` | no | 518 / 14 | 257 | `1 + 37*37 = 1370` | `BitImageProcessor` |
| `facebook/dinov2-base` | `Dinov2Model` | 768 | 12 | 12 | 64 | `4H=3072` | no | 518 / 14 | 257 | 1370 | `BitImageProcessor` |
| `facebook/dinov2-large` | `Dinov2Model` | 1024 | 24 | 16 | 64 | `4H=4096` | no | 518 / 14 | 257 | 1370 | `BitImageProcessor` |
| `facebook/dinov2-giant` | `Dinov2Model` | 1536 | 40 | 24 | 64 | SwiGLU hidden 4096 | yes | 518 / 14 | 257 | 1370 | `BitImageProcessor` |
| `facebook/dinov2-base-imagenet1k-1-layer` | `Dinov2ForImageClassification` | 768 | 12 | 12 | 64 | `4H=3072` | no | 518 / 14 | 257 | 1370 | `BitImageProcessor` |
| `facebook/dinov2-with-registers-base` | sibling `dinov2_with_registers` | 768 | 12 | 12 | 64 | `4H=3072` | no | 518 / 14 | 261 | 1370 plus 4 registers | `BitImageProcessor` |
| `facebook/dinov2-with-registers-giant` | sibling `dinov2_with_registers` | 1536 | 40 | 24 | 64 | SwiGLU hidden 4096 | yes | 518 / 14 | 261 | 1370 plus 4 registers | `BitImageProcessor` |

The official processor resizes to shortest edge 256 and center-crops 224x224, so normal image classification uses 16x16 patch tokens plus CLS. The model config still declares `image_size=518`, so the learned position table has 37x37 patch positions and is interpolated down to 16x16 for default processor output.

## 3a. Family variation traps

- Plain `dinov2` and `dinov2_with_registers` are separate model types. Do not load register checkpoints through `Dinov2Model`.
- Register variants insert `num_register_tokens=4` after CLS and before patch tokens. Classification/backbone patch slicing changes from `[:, 1:]` to `[:, 1 + num_register_tokens:]`.
- Giant uses SwiGLU FFN. Its hidden width is `round_up_to_8(int(H * mlp_ratio * 2 / 3))`, not `4H`; for `H=1536`, the gated half is 4096 and `weights_in` is `Linear(1536 -> 8192)`.
- Source attention is non-causal MHA, not GQA/MQA. `hidden_size` must divide `num_attention_heads`.
- Dropout and stochastic depth are configured but official inference configs use zero rates. DinoML can initially lower dropout/drop-path as identity for inference.
- Absolute position interpolation is part of the model graph for any runtime image size that differs from the stored position grid. It uses bicubic interpolation in fp32 and casts back.
- Source tensor layout is NCHW for `pixel_values`, patch Conv2d, and backbone feature-map output. NHWC is an optimization candidate, not a semantic default.
- Axis-sensitive layout points: Conv2d reads channel axis 1; `flatten(2).transpose(1, 2)` assumes `[B, H, Gh, Gw]`; classification averages patch tokens on sequence axis 1; backbone reshapes token sequence to `[B, Gh, Gw, H]` then permutes to `[B, H, Gh, Gw]`.
- `bool_masked_pos` is pretraining-only but present in `Dinov2Model.forward`. Its shape is `[B, Npatch]` and it replaces patch embeddings before CLS insertion.
- Backbone source comments preserve an original implementation quirk around height/width ordering; translation should exactly preserve `reshape(batch, height // patch, width // patch, -1)` then `permute(0,3,1,2)`.

## 4. Operator coverage checklist

Tensor/layout ops:

- NCHW input validation and channel count check.
- Conv2d non-overlap patch embedding: `Conv2d(3 -> H, kernel=14, stride=14, padding=0, groups=1)`.
- `flatten(2)`, `transpose(1,2)`, `view/reshape`, `permute`, `contiguous`.
- Token concatenation on sequence axis: CLS prepend, optional register insertion in sibling family.
- Broadcast add for position embeddings `[1, S, H]` onto `[B, S, H]`.
- Slice/select: CLS `[:, 0, :]`, patch tokens `[:, 1:, :]`, backbone stage hidden states.
- `mean(dim=1)` for classification patch-token mean.
- Optional `torch.where(bool_masked_pos[..., None], mask_token, patch_embedding)`.

Neural network primitives:

- Dense Linear/GEMM with bias for Q, K, V, attention output, MLPs, classifier.
- LayerNorm over last dim with eps `1e-6`.
- GELU activation for standard MLP.
- SiLU plus elementwise multiply for SwiGLU giant.
- Per-channel LayerScale multiply `[H]`.
- Residual adds.
- Inference dropout/drop-path identity.

Attention primitives:

- Non-causal self-attention over full image token sequence.
- MHA with Q/K/V shapes `[B, heads, S, 64]`.
- Softmax over key sequence axis.
- Optional backend dispatch through Transformers attention interfaces; eager parity is matmul-scale-mask-softmax-dropout-matmul.

Position/custom math:

- Learned absolute position table split into CLS and patch table.
- Bicubic interpolation on `[1, H, sqrt(Npos), sqrt(Npos)]` in fp32, `align_corners=False`, then back to original dtype.
- Register variants use antialiasing in interpolation; plain `dinov2` does not pass `antialias=True`.

Preprocessing-coupled ops:

- RGB conversion, resize shortest edge 256 with bicubic resampling, center crop 224x224, rescale by `1/255`, normalize by ImageNet mean/std, output channels-first `pixel_values`.

## 5. Layer/block breakdown

Embeddings:

```text
pixel_values: [B, 3, image_h, image_w]
patch = Conv2d(3 -> H, kernel=P, stride=P)(pixel_values)  # [B, H, Gh, Gw]
patch = flatten spatial then transpose                         # [B, Gh*Gw, H]
if bool_masked_pos: patch = where(mask[..., None], mask_token, patch)
tokens = concat(cls_token.expand(B,1,H), patch, dim=1)          # [B, 1+N, H]
tokens = tokens + interpolate_pos_encoding(tokens, image_h, image_w)
```

Encoder block, repeated `num_hidden_layers`:

```text
y = LayerNorm(x)
q = Linear(H -> H, bias=qkv_bias)(y).view(B, S, heads, 64).transpose(1, 2)
k = Linear(H -> H, bias=qkv_bias)(y).view(B, S, heads, 64).transpose(1, 2)
v = Linear(H -> H, bias=qkv_bias)(y).view(B, S, heads, 64).transpose(1, 2)
a = Attention(q, k, v, noncausal, scale=1/sqrt(64))
a = reshape/transposed context to [B, S, H]
a = Linear(H -> H, bias=True)(a)
x = x + LayerScale(a)
y = LayerNorm(x)
y = MLP(y) or SwiGLUFFN(y)
x = x + LayerScale(y)
```

Standard MLP:

```text
Linear(H -> int(H * mlp_ratio), bias=True) -> GELU -> Linear(int(H * mlp_ratio) -> H, bias=True)
```

SwiGLU FFN:

```text
hidden = round_up_to_8(int(H * mlp_ratio * 2 / 3))
z = Linear(H -> 2 * hidden, bias=True)(x)
x1, x2 = chunk(z, 2, dim=-1)
out = Linear(hidden -> H, bias=True)(silu(x1) * x2)
```

Model tail:

```text
sequence = LayerNorm(encoder_last)
pooler_output = sequence[:, 0, :]
```

Classification head:

```text
cls = sequence[:, 0]
patch = sequence[:, 1:]  # plain dinov2 only
logits = Linear(2H -> num_labels)(concat(cls, patch.mean(dim=1), dim=1))
```

Backbone output:

```text
hidden_state = optional LayerNorm(stage_hidden_state)
patch_state = hidden_state[:, 1:]
feature = patch_state.reshape(B, image_h // P, image_w // P, H)
feature = feature.permute(0, 3, 1, 2).contiguous()
```

## 6. Attention requirements

- Variant: self-attention only.
- Causality: non-causal.
- Heads: MHA, no KV head sharing.
- Head dim: 64 for all official swept configs.
- Sequence length: `1 + floor(image_h / patch_h) * floor(image_w / patch_w)` for plain `dinov2`; register sibling adds `num_register_tokens`.
- Masking: no attention mask in model forward. Eager attention accepts `attention_mask` but `Dinov2SelfAttention` passes `None`.
- Cache: no KV cache.
- Sliding window/local attention: none.
- Position interaction: learned absolute positions are added before attention; no RoPE, ALiBi, relative bias, or convolutional position encoding.
- Source math order: project Q/K/V separately, reshape to `[B, heads, S, head_dim]`, compute `q @ k.transpose(-2,-1) * head_dim**-0.5`, add mask if present, softmax along `-1`, dropout during training, then `attn @ v`.
- Optimized backend compatibility: source advertises SDPA, FlashAttention, and FlexAttention support through `ALL_ATTENTION_FUNCTIONS`. For parity, implement eager math first; then replace with non-causal flash/SDPA when attention dropout is zero and no mask is present.

## 7. Position encoding and custom math

Plain DinoV2 position interpolation:

```python
def dinov2_pos_embed(pos, tokens, height, width, patch_size):
    n_patches = tokens.shape[1] - 1
    n_pos = pos.shape[1] - 1
    if n_patches == n_pos and height == width:
        return pos
    cls = pos[:, :1]
    patch = pos[:, 1:]
    dim = tokens.shape[-1]
    grid = int(n_pos ** 0.5)
    patch = patch.reshape(1, grid, grid, dim).permute(0, 3, 1, 2)
    patch = bicubic_interpolate_fp32(
        patch, size=(height // patch_size, width // patch_size), align_corners=False
    ).to(pos.dtype)
    patch = patch.permute(0, 2, 3, 1).view(1, -1, dim)
    return concat([cls, patch], dim=1)
```

Precompute/cache opportunity:

- For common 224x224 processor output and official `patch_size=14`, interpolated patch grid is 16x16.
- Because interpolation depends only on learned position table, target image size, patch size, and dtype, DinoML can cache the resulting `[1, 1+N, H]` tensor per shape.

Important parity details:

- Plain `dinov2` bicubic interpolation uses fp32 intermediate and `align_corners=False`.
- It skips interpolation only when not tracing, number of patch tokens matches stored patch positions, and `height == width`.
- Official configs store a 37x37 patch position grid from `image_size=518`; default processor crop is 16x16 patches, so interpolation is normally exercised.

## 8. Preprocessing and input packing

Official preprocessor config:

```text
image_processor_type: BitImageProcessor
do_convert_rgb: true
do_resize: true
size.shortest_edge: 256
resample: bicubic
do_center_crop: true
crop_size: 224x224
do_rescale: true
rescale_factor: 1/255
do_normalize: true
image_mean: [0.485, 0.456, 0.406]
image_std: [0.229, 0.224, 0.225]
```

Processor tensor contract:

- Input may be PIL/NumPy/torch image-like data.
- Shared image-processing utilities prepare images in channels-first working format.
- Output key is `pixel_values`.
- With `return_tensors="pt"` and default config, expected runtime tensor is float-like `[B, 3, 224, 224]`.
- Model forward requires `pixel_values`; no `attention_mask`, grid metadata, token type IDs, text tokens, or packed sequence descriptors exist.

CPU/data pipeline versus GPU/runtime:

- Treat RGB conversion, resize, center crop, rescale, and normalize as data-pipeline work first.
- GPU runtime starts at normalized `pixel_values`.
- Later optimization could fuse rescale/normalize into an input staging kernel if DinoML owns image ingestion.

## 9. Graph rewrite / lowering opportunities

### Rewrite: non-overlap Conv2d patch embedding -> patch GEMM

Source pattern:

```text
Conv2d(C -> H, kernel=P, stride=P, padding=0, dilation=1, groups=1)
-> flatten(2) -> transpose(1,2)
```

Replacement:

```text
WindowFlattenNCHW([B,C,Himg,Wimg], window=P, stride=P)
-> MatMul([B*Npatch, C*P*P] x [C*P*P, H])
-> BiasAdd
-> Reshape([B, Npatch, H])
```

Preconditions:

- `kernel_size == stride == patch_size`
- `padding == 0`
- `dilation == 1`
- `groups == 1`
- channel count equals config `num_channels`
- `image_h` and `image_w` are at least one patch; exact source Conv2d floors partial trailing pixels
- flatten order must match PyTorch NCHW Conv2d weight layout and `flatten(2).transpose(1,2)`

Weight transform:

```python
w_gemm = conv.weight.reshape(hidden_size, num_channels * patch_h * patch_w).T
b = conv.bias
```

Layout constraints:

- Faithful source graph is NCHW.
- NHWC optimization requires either NHWC image staging plus NHWC window flatten, or a local NCHW->NHWC conversion before patch GEMM. Weight flatten order must be audited with a parity test.

Failure cases:

- overlapping patches, padding, dilation, grouped conv, non-RGB channels, or consumers requiring native Conv2d behavior.

Parity test sketch:

- Random `[B,3,224,224]`, `[B,3,518,518]`, and non-square divisible sizes; compare Conv2d path to WindowFlatten+GEMM in fp32 and fp16.

### Rewrite: separate Q/K/V projections -> fused QKV projection

Source pattern:

```text
query = Linear(H -> H)
key = Linear(H -> H)
value = Linear(H -> H)
view/transposes to [B, heads, S, 64]
```

Replacement:

```text
Linear(H -> 3H) with concatenated weights/biases -> split last dim -> reshape heads
```

Preconditions:

- All Q/K/V have same input `hidden_states`.
- Same dtype/device and all biases either present according to `qkv_bias` or absent.
- Split order must be query, key, value to match source.

Weight transform:

```python
w_qkv = torch.cat([w_q, w_k, w_v], dim=0)
b_qkv = torch.cat([b_q, b_k, b_v], dim=0)
```

Failure cases:

- Partial weight override, quantized per-projection policies that need separate metadata, or debugging paths that capture individual projection outputs.

### Rewrite: final patch tokens -> NCHW backbone feature map

Source pattern:

```text
hidden[:, 1:].reshape(B, Gh, Gw, H).permute(0,3,1,2).contiguous()
```

Replacement:

```text
TokenToFeatureMap(sequence_without_cls, output_layout=NCHW)
```

Preconditions:

- Plain `dinov2`: remove exactly 1 CLS token.
- Register sibling: remove `1 + num_register_tokens`.
- `Gh = pixel_values.shape[2] // patch_size`, `Gw = pixel_values.shape[3] // patch_size`.

Layout constraints:

- If DinoML keeps NHWC feature maps internally, downstream backbone consumers must accept NHWC or a final NHWC->NCHW conversion must be emitted. Do not silently change public output layout.

### Rewrite: position interpolation cache

Source pattern:

```text
reshape position table -> bicubic interpolate -> reshape -> concat cls
```

Replacement:

```text
Shape-specialized cached constant pos_embed_for_(Gh,Gw,dtype)
```

Preconditions:

- Inference-only weights immutable.
- Runtime `height`, `width`, and dtype known at compile time or cache lookup time.
- Plain `dinov2` uses non-antialias interpolation; register sibling uses antialiasing.

Failure cases:

- Weight mutation, training, tracing semantics that intentionally always interpolate, or dynamic arbitrary image sizes without cache support.

## 10. Kernel fusion candidates

Highest priority:

- Patch embedding Conv2d-to-GEMM/window-flatten. It is the first large image-specific op and also the cleanest layout boundary.
- LayerNorm over `[B,S,H]` with eps `1e-6`, including final norm. Every block uses two norms.
- Fused QKV projection plus reshape/split. Reduces three GEMM launches and materialization overhead.
- Non-causal FlashAttention/SDPA for image-token MHA. Official dropout is zero and no attention mask is used.
- MLP GEMM + GELU + GEMM, plus SwiGLU path for giant.

Medium priority:

- LayerScale multiply plus residual add.
- Positional add fused with token assembly or first norm input staging.
- Classification head concat/mean/linear for ImageNet checkpoints.
- Backbone token-to-feature-map layout conversion.

Lower priority:

- Training-only dropout/drop-path.
- `bool_masked_pos` mask-token replacement for pretraining.
- Dynamic bicubic interpolation on GPU for arbitrary image sizes; cache common shapes first.
- Register-token sibling support, unless chosen as a separate target.

## 11. Runtime staging plan

Stage 1: config and processor metadata

- Parse `Dinov2Config`.
- Load `BitImageProcessor` settings into a data-pipeline contract.
- Accept normalized NCHW `pixel_values` as the compiled model input.

Stage 2: patch embedding and position parity

- Implement Conv2d patch embedding faithfully.
- Implement or precompute position interpolation for 224x224 and 518x518.
- Validate embeddings before encoder blocks.

Stage 3: one encoder block parity

- Implement LayerNorm, Q/K/V, non-causal attention, output projection, LayerScale, MLP/GELU, residuals.
- Validate small/base block in fp32.

Stage 4: full encoder

- Run all layers and final LayerNorm.
- Validate `last_hidden_state` and CLS `pooler_output`.

Stage 5: optional heads

- Add classification head for `Dinov2ForImageClassification`.
- Add backbone feature-map extraction with source NCHW output.

Stage 6: optimization passes

- Add Conv2d-to-GEMM rewrite.
- Fuse QKV.
- Enable non-causal flash attention.
- Add cached position interpolation.

Stage 7: variant expansion

- Add giant SwiGLU.
- Audit and implement `dinov2_with_registers` as a separate family/report or explicit extension.

## 12. Parity and validation plan

Recommended tests:

- Processor contract smoke: representative PIL/NumPy input through HF processor yields `[B,3,224,224]`; DinoML data pipeline or fixture matches rescale/normalize values.
- Patch embedding parity: compare Conv2d output and lowered patch GEMM for small/base shapes.
- Position interpolation parity: compare cached/interpreted position tensors at 224x224, 518x518, and one non-square divisible size.
- One-block parity: random hidden state `[B,257,H]` through a single encoder block with copied weights.
- Full encoder parity: `facebook/dinov2-small` and `facebook/dinov2-base` on one or two fixed images.
- Giant parity: specifically cover SwiGLU dimensions and SiLU multiply.
- Classification head parity: base ImageNet checkpoint logits for fixed image.
- Backbone parity: selected stages produce `[B,H,16,16]` for 224 crop.

Tolerances:

- fp32: use tight absolute/relative tolerance around `1e-4` for block outputs, with looser tolerance around bicubic interpolation if implemented outside PyTorch.
- fp16/bf16: start with `1e-2` class/feature tolerance and inspect layerwise drift; attention softmax and LayerNorm accumulation precision matter.

No DinoML tests were run for this report.

## 13. Performance probes

- Processor throughput: images/sec for RGB/resize/crop/normalize separately from model.
- Patch embedding throughput: Conv2d native versus WindowFlatten+GEMM for 224, 518, and batched inputs.
- Encoder-only throughput: small/base/large/giant at batch sizes 1, 8, 32.
- Attention backend comparison: eager matmul/softmax versus SDPA/FlashAttention for sequence lengths 257 and 1370.
- MLP throughput: standard GELU MLP versus SwiGLU giant.
- Position interpolation overhead: dynamic bicubic every run versus cached per shape.
- Backbone output overhead: token-to-NCHW feature map conversion cost.
- End-to-end image classification latency/throughput with and without CPU preprocessing included.

These are proposed probes, not measurements.

## 14. Skip/defer list

Safe to defer for first encoder/image-feature target:

- Training, gradients, gradient checkpointing.
- Dropout/drop-path stochastic behavior.
- `bool_masked_pos` pretraining mask path, unless masked image modeling is targeted.
- Image classification loss computation.
- Multi-label/regression loss branching.
- Arbitrary dynamic image sizes without cached position interpolation.
- Register-token checkpoints, because they are a sibling model family.
- Quantization.
- Multi-GPU/tensor parallel execution.
- Remote-code support; official swept repos use local Transformers classes.

## 15. Final implementation checklist

- [ ] Parse `Dinov2Config` and checkpoint metadata.
- [ ] Load base encoder weights, CLS token, mask token, position embeddings, LayerScale weights.
- [ ] Represent processor contract for normalized NCHW `pixel_values`.
- [ ] Implement faithful NCHW patch Conv2d path.
- [ ] Add guarded Conv2d patch embedding -> WindowFlatten+GEMM rewrite.
- [ ] Implement position interpolation or shape-specialized cached position tensors.
- [ ] Implement LayerNorm eps `1e-6`.
- [ ] Implement MHA Q/K/V projections, non-causal attention, output projection.
- [ ] Add optional fused QKV rewrite.
- [ ] Implement LayerScale multiply and residual adds.
- [ ] Implement GELU MLP.
- [ ] Implement SwiGLU FFN for giant.
- [ ] Implement final LayerNorm and CLS pooler output.
- [ ] Add optional image classification head.
- [ ] Add optional backbone feature-map output preserving NCHW public layout.
- [ ] Add patch embedding parity tests.
- [ ] Add position interpolation parity tests.
- [ ] Add one-block and full-encoder parity tests.
- [ ] Add classification/backbone parity tests when heads are in scope.
- [ ] Benchmark processor, patch embedding, attention, MLP, and full encoder.
