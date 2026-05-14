# DINOv2 With Registers Transformers Audit

## 1. Source basis

Transformers commit/version: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4` in local checkout `X:/H/transformers`.

Model family: `dinov2_with_registers`

Primary runtime target: DINOv2-with-registers vision encoder/backbone inference. Image classification is optional but implemented in source. Training, pretraining masks, and loss paths are deferred for first DinoML integration.

Config source:

- Source defaults: `X:/H/transformers/src/transformers/models/dinov2_with_registers/configuration_dinov2_with_registers.py`
- Official Hugging Face configs saved under `agents/plans/transformers/dinov2_with_registers/_sources/`
- Representative model URLs: [small](https://huggingface.co/facebook/dinov2-with-registers-small), [base](https://huggingface.co/facebook/dinov2-with-registers-base), [large](https://huggingface.co/facebook/dinov2-with-registers-large), [giant](https://huggingface.co/facebook/dinov2-with-registers-giant), and ImageNet 1-layer classifier variants for each size.

Source files inspected:

- `X:/H/transformers/src/transformers/models/dinov2_with_registers/modeling_dinov2_with_registers.py`
- `X:/H/transformers/src/transformers/models/dinov2_with_registers/configuration_dinov2_with_registers.py`
- `X:/H/transformers/src/transformers/models/dinov2_with_registers/modular_dinov2_with_registers.py`
- `X:/H/transformers/src/transformers/models/dinov2_with_registers/convert_dinov2_with_registers_to_hf.py`
- `X:/H/transformers/src/transformers/models/dinov2/modeling_dinov2.py`, for delegation/difference checks
- `X:/H/transformers/src/transformers/models/bit/image_processing_bit.py`, because checkpoints use `BitImageProcessor`
- Auto mappings for model, backbone, classifier, and image processor registration.

Pinned source URLs:

- https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/dinov2_with_registers/modeling_dinov2_with_registers.py
- https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/dinov2_with_registers/configuration_dinov2_with_registers.py
- https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/dinov2_with_registers/modular_dinov2_with_registers.py
- https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/bit/image_processing_bit.py

Any missing files or assumptions:

- No gated/401/403 gaps were observed while fetching representative configs and preprocessor configs.
- `modeling_dinov2_with_registers.py` and `configuration_dinov2_with_registers.py` are generated from `modular_dinov2_with_registers.py`. The generated files are authoritative for runtime behavior at the pinned commit; future source edits should target the modular file.
- The modular source delegates most blocks to `dinov2`; the generated modeling file contains a standalone copy with `Dinov2WithRegisters*` classes. DinoML should compare generated runtime behavior, not only the modular inheritance.
- A local import sanity check could not run because the Python environment has an incompatible `huggingface_hub` package (`is_offline_mode` missing). This report is source/config-derived; no DinoML tests or Transformers execution were run.

## 2. High-level architecture

DINOv2-with-registers is a ViT-style image encoder with extra learned register tokens inserted after CLS and before patch tokens.

```text
image preprocessing -> NCHW pixel_values
  -> non-overlap Conv2d patch embedding -> flatten/transpose to token sequence
  -> prepend CLS -> add learned absolute positions to CLS+patch tokens
  -> insert register tokens after CLS
  -> repeated pre-norm encoder blocks
  -> final LayerNorm -> pooled CLS / full token sequence / backbone feature maps / optional logits
```

Stage decomposition:

- CPU/data pipeline: RGB conversion, resize shortest edge to 256, center crop 224x224, rescale by `1/255`, normalize with ImageNet mean/std, assemble NCHW `pixel_values`.
- Encoder graph: patch projection, token concatenation, positional interpolation/add, register insertion, noncausal self-attention blocks, final LayerNorm.
- Backbone ABI: hidden states are selected by `out_features`/`out_indices`; each selected state can be LayerNormed and reshaped to NCHW image-like maps after removing CLS plus register tokens.
- Classification ABI: final normalized CLS token is concatenated with mean over patch tokens only, explicitly excluding register tokens, then passed through `Linear(2H -> num_labels)`.
- Independently cacheable constants: CLS token, register tokens, learned position table, patch projection weights, per-layer weights. Interpolated position tables can be cached per `(input_height, input_width, dtype)` when dynamic image sizes are admitted.

## 3. Important config dimensions

Source defaults from `Dinov2WithRegistersConfig`:

| Field | Default | Runtime significance |
| --- | ---: | --- |
| `hidden_size` | 768 | token width `H` |
| `num_hidden_layers` | 12 | encoder block count |
| `num_attention_heads` | 12 | MHA heads |
| `head_dim` | 64 | inferred as `hidden_size / num_attention_heads`; source rejects non-divisible configs |
| `mlp_ratio` | 4 | ungated MLP width `4H`; SwiGLU rounded width for giant |
| `hidden_act` | `gelu` | activation for non-SwiGLU MLP |
| `image_size` | 224 | source default only; official checkpoints use 518 |
| `patch_size` | 16 | source default only; official checkpoints use 14 |
| `num_channels` | 3 | input channel count |
| `qkv_bias` | true | separate Q/K/V Linear layers include bias |
| `layerscale_value` | 1.0 | per-channel post-attn/post-MLP scale vectors |
| `num_register_tokens` | 4 | learned tokens inserted after CLS |
| `apply_layernorm` | true | backbone feature-map postprocessing |
| `reshape_hidden_states` | true | backbone returns NCHW maps instead of token sequences |
| `_supports_sdpa/_flash_attn/_flex_attn` | true in model class | noncausal encoder attention can route through shared attention backends |

Representative checkpoint sweep, from official `config.json` snapshots:

| Checkpoint | Architecture | H | Layers | Heads | Head dim | Patch | Registers | MLP | Classifier |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `facebook/dinov2-with-registers-small` | `Dinov2WithRegistersModel` | 384 | 12 | 6 | 64 | 14 | 4 | GELU, `4H=1536` | no |
| `facebook/dinov2-with-registers-base` | `Dinov2WithRegistersModel` | 768 | 12 | 12 | 64 | 14 | 4 | GELU, `4H=3072` | no |
| `facebook/dinov2-with-registers-large` | `Dinov2WithRegistersModel` | 1024 | 24 | 16 | 64 | 14 | 4 | GELU, `4H=4096` | no |
| `facebook/dinov2-with-registers-giant` | `Dinov2WithRegistersModel` | 1536 | 40 | 24 | 64 | 14 | 4 | SwiGLU rounded hidden `4096` | no |
| `*-imagenet1k-1-layer` | `Dinov2WithRegistersForImageClassification` | same as size | same | same | 64 | 14 | 4 | same | 1000-label linear head, inferred from label maps |

All fetched configs use `image_size=518`, `torch_dtype="float32"`, `qkv_bias=true`, `attention_probs_dropout_prob=0.0`, and `hidden_dropout_prob=0.0`.

Checkpoint config notes:

- Official configs include `interpolate_antialias` and `interpolate_offset`, but this pinned modeling source does not read them. It hardcodes bicubic interpolation with `align_corners=False` and `antialias=True`.
- Official large and giant configs serialize `stage_names` only through `stage12` despite `num_hidden_layers` being 24 or 40. Source `__post_init__` computes `["stem"] + stage1..stageN` from `num_hidden_layers`; DinoML should prefer source-derived stage names and treat serialized `stage_names` as suspect metadata unless validated by Transformers config loading.
- `num_labels` is not an explicit scalar in the fetched ImageNet configs, but `id2label`/`label2id` have 1000 entries. Transformers config behavior normally derives `num_labels` from label maps.

## 3a. Family variation traps

- This is not plain `dinov2`: sequence length is `1 + num_register_tokens + Npatch`, and patch-token slices begin at index `1 + num_register_tokens`.
- Position embeddings are stored for CLS plus patch tokens only: shape `[1, 1 + Npatch_pretrain, H]`. Register tokens do not receive position embeddings directly; they are inserted after the positional add.
- Modular source mostly delegates to `dinov2`, but generated source is standalone. Audit/runtime parity should use generated `modeling_dinov2_with_registers.py`.
- Giant uses SwiGLU FFN. Its hidden width is `round_up_to_multiple_of_8(int((H * mlp_ratio) * 2 / 3)) = 4096`, with `weights_in: Linear(1536 -> 8192)` split as `[x1, x2]`, then `silu(x1) * x2`, then `weights_out: Linear(4096 -> 1536)`.
- Small/base/large use ordinary GELU MLPs: `Linear(H -> 4H)`, GELU, `Linear(4H -> H)`.
- Q/K/V are three independent Linear modules, not a packed QKV module in the HF runtime. The converter splits original packed timm QKV weights in Q, K, V order.
- Backbone reshaping is axis-sensitive: after dropping CLS+registers, source reshapes to `[B, height // patch, width // patch, H]`, then permutes to NCHW. The source comment says it copies an original implementation bug around height/width order; do not silently rewrite this path without parity tests.
- Source input is NCHW. NHWC/channel-last is an optimization candidate only for local Conv2d/reshape regions with explicit axis rewrites.
- `bool_masked_pos` triggers `torch.where` replacement of patch embeddings before CLS/register insertion, but it is pretraining-oriented and can be deferred for inference-only image feature extraction.

## 4. Operator coverage checklist

Tensor/layout ops:

- NCHW input validation and dtype cast to patch projection weight dtype.
- Conv2d output flatten from `[B, H, Hp, Wp]` to `[B, H, Hp*Wp]`, transpose to `[B, Hp*Wp, H]`.
- `expand` for CLS/register tokens, `cat` along token axis, slicing on token axis, `view`/`reshape`, `permute`, `contiguous`.
- Backbone token-to-map reshape: `[B, Npatch, H] -> [B, Hp, Wp, H] -> [B, H, Hp, Wp]`.

Neural network primitives:

- `Conv2d(C -> H, kernel=patch_size, stride=patch_size, padding=0, groups=1)` with bias.
- LayerNorm over last dimension with `eps=1e-6`.
- Linear Q/K/V/O and MLP projections, all with bias.
- GELU for small/base/large MLP.
- SiLU, chunk, multiply for giant SwiGLU.
- Elementwise residual add and per-channel LayerScale multiply.
- Mean reduction over patch-token axis for classifier.

Attention primitives:

- Noncausal self-attention only.
- MHA with `num_key_value_heads == num_attention_heads`; no GQA/MQA.
- Q/K/V shape `[B, heads, S, head_dim]`, score shape `[B, heads, S, S]`.
- No attention mask in model forward for normal encoder inference.
- No KV cache, no autoregressive generation.
- Source can dispatch through Transformers eager/SDPA/FlashAttention/FlexAttention attention interfaces.

Position/custom math:

- Learned absolute position table for CLS+patch tokens.
- Bicubic interpolation in float32 on NCHW-like temporary `[1, H, sqrt(N), sqrt(N)]`, then cast back to original dtype.
- Hardcoded `antialias=True`, `align_corners=False`.

Preprocessing-coupled ops:

- `BitImageProcessor`: RGB conversion, resize shortest edge, center crop, rescale, normalize.
- Official preprocessor configs use ImageNet mean/std, not the `BitImageProcessor` source defaults from CLIP mean/std.

Optional classifier/backbone ops:

- Classifier: patch-token mean excludes register tokens, then `Linear(2H -> num_labels)`.
- Backbone: `filter_output_hidden_states`, optional final LayerNorm on each selected hidden state, optional NCHW reshape.

## 5. Layer/block breakdown

Embedding path for input `pixel_values [B, 3, IH, IW]`:

```text
patch = Conv2d(3 -> H, kernel=P, stride=P)(pixel_values)
patch = flatten(2).transpose(1, 2)                  # [B, Hp*Wp, H]
if bool_masked_pos: patch = where(mask, mask_token, patch)
tokens = cat([cls.expand(B,1,H), patch], dim=1)      # [B, 1+N, H]
tokens = tokens + interpolate_pos_encoding(tokens, IH, IW)
tokens = cat([tokens[:, :1], registers.expand(B,R,H), tokens[:, 1:]], dim=1)
```

Encoder block, repeated `L` times:

```text
y = LayerNorm(x)
q = Linear(H -> H, bias=qkv_bias)(y).view(B,S,heads,64).transpose(1,2)
k = Linear(H -> H, bias=qkv_bias)(y).view(B,S,heads,64).transpose(1,2)
v = Linear(H -> H, bias=qkv_bias)(y).view(B,S,heads,64).transpose(1,2)
a = Attention(q, k, v, mask=None, causal=False, scale=1/sqrt(64))
a = reshape_to_tokens(a) -> Linear(H -> H, bias=True)
x = x + LayerScale(a)
y = LayerNorm(x)
y = MLP_or_SwiGLU(y)
x = x + LayerScale(y)
```

Final model output:

```text
sequence_output = LayerNorm(last_hidden_state)       # [B, 1+R+N, H]
pooler_output = sequence_output[:, 0, :]             # [B, H]
```

Classifier output:

```text
cls = sequence_output[:, 0]
patch = sequence_output[:, 1 + num_register_tokens :]
logits = Linear(2H -> num_labels)(cat([cls, mean(patch, dim=1)], dim=1))
```

Backbone feature output:

```text
hidden_state = selected_hidden_state
if apply_layernorm: hidden_state = final_layernorm(hidden_state)
if reshape_hidden_states:
    hidden_state = hidden_state[:, 1 + R :]
    hidden_state = hidden_state.reshape(B, IH // P, IW // P, H)
    hidden_state = hidden_state.permute(0, 3, 1, 2).contiguous()
```

## 6. Attention requirements

Attention is encoder self-attention:

- Noncausal, full dense attention.
- Self-attention only, no cross-attention.
- MHA only: Q heads = K heads = V heads.
- Head dim is 64 for all official checkpoints.
- No runtime attention mask for image inference.
- No packed/varlen support in source.
- No RoPE, ALiBi, relative bias, sliding window, or local/block sparse pattern.
- No KV cache. This is not a generation model.
- Dropout is 0.0 in official configs and should be disabled for inference.
- Eager math order: `matmul(q, k.transpose) * scale`, optional mask add, softmax over last dim, dropout, `matmul(weights, v)`, transpose/contiguous, reshape to `[B,S,H]`.
- SDPA/Flash/Flex backend parity must preserve noncausal dense attention, scale `1/sqrt(head_dim)`, and no mask.

For a 224x224 processor output and patch size 14, `Hp=Wp=16`, `Npatch=256`, `R=4`, so attention sequence length is `S=261`. For pretraining/native image size 518, `Hp=Wp=37`, `Npatch=1369`, `S=1374`.

## 7. Position encoding and custom math

Position encoding is learned absolute CLS+patch embeddings, interpolated when input patch grid differs from the learned grid.

Short implementation sketch:

```python
def interpolate_pos(pos, embeddings, image_h, image_w, patch_size):
    n_patches = embeddings.shape[1] - 1
    n_pos = pos.shape[1] - 1
    if n_patches == n_pos and image_h == image_w:
        return pos
    cls = pos[:, 0]                  # [1, H]
    patch = pos[:, 1:]               # [1, N, H]
    side = int(n_pos ** 0.5)
    patch = patch.reshape(1, side, side, H).permute(0, 3, 1, 2)
    patch = bicubic_interpolate(
        patch.float(),
        size=(image_h // patch_size, image_w // patch_size),
        align_corners=False,
        antialias=True,
    ).to(pos.dtype)
    patch = patch.permute(0, 2, 3, 1).view(1, -1, H)
    return cat([cls.unsqueeze(0), patch], dim=1)
```

Precomputable:

- Original position table is constant.
- Interpolated position table is deterministic for `(IH, IW, patch_size, dtype)` and can be cached per admitted dynamic image shape.

Dynamic inputs:

- `IH`/`IW` from `pixel_values.shape`.
- Interpolation output grid must be `(IH // patch_size, IW // patch_size)`.
- Source assumes square pretraining position grid because it computes `sqrt(num_positions)`.

## 8. Preprocessing and input packing

Official `preprocessor_config.json` contract:

| Field | Value |
| --- | --- |
| `image_processor_type` | `BitImageProcessor` |
| `do_convert_rgb` | true |
| `do_resize` | true |
| `size.shortest_edge` | 256 |
| `do_center_crop` | true |
| `crop_size` | 224x224 |
| `do_rescale` | true |
| `rescale_factor` | `1/255` |
| `do_normalize` | true |
| `image_mean` | `[0.485, 0.456, 0.406]` |
| `image_std` | `[0.229, 0.224, 0.225]` |
| output layout | `pixel_values [B, 3, 224, 224]` by default |

CPU/data-pipeline work:

- Image decoding, RGB conversion, resize, crop, rescale, normalize.

GPU/runtime work:

- Source model expects already-formed NCHW `pixel_values`.
- Optional `bool_masked_pos [B, Npatch]` enters the embedding graph only for masked pretraining behavior; defer for first inference target.

There is no tokenizer, input packing metadata, placeholder token expansion, or postprocessing required for base encoder/backbone inference.

## 9. Graph rewrite / lowering opportunities

### Rewrite: non-overlap Conv2d patch embedding to GEMM

Source pattern:

```text
Conv2d(C -> H, kernel=P, stride=P, padding=0, dilation=1, groups=1)
  -> flatten spatial -> transpose to tokens
```

Replacement:

```text
WindowFlatten_NCHW_to_tokens([B,C,IH,IW], P) -> MatMul(weight_flat.T) -> BiasAdd
```

Preconditions:

- `kernel_size == stride == patch_size`
- `padding == 0`
- `dilation == 1`
- `groups == 1`
- `IH % patch_size == 0` and `IW % patch_size == 0`
- Input source layout is NCHW unless a guarded channel-last region is proven.

Shape equations:

- `Hp = IH // P`, `Wp = IW // P`, `N = Hp * Wp`
- Output `[B, N, H]`
- Official 224 path: `[B,3,224,224] -> [B,256,H]`
- Official 518 learned-grid path: `[B,3,518,518] -> [B,1369,H]`

Weight transform:

```python
w_flat = conv.weight.reshape(H, C * P * P)
patch_tokens = windows @ w_flat.T + conv.bias
```

Failure cases:

- Non-divisible dynamic dimensions require fallback or explicit admission reject.
- NHWC input requires different window flatten order or weight permutation.

Parity test sketch:

- Compare Conv2d+flatten+transpose against WindowFlatten+GEMM for random NCHW tensors at 224 and 518.

### Rewrite: separate Q/K/V Linears to packed QKV GEMM

Source pattern:

```text
q = Linear(H -> H)(x)
k = Linear(H -> H)(x)
v = Linear(H -> H)(x)
```

Replacement:

```text
qkv = Linear(H -> 3H)(x); split last dim as [q, k, v]
```

Preconditions:

- Same input tensor and dtype for all three projections.
- All projections have matching `H -> H`, all biases present or all absent according to `qkv_bias`.
- Packed row order must be Q, K, V, matching converter split order.

Weight transform:

```python
packed_w = cat([q.weight, k.weight, v.weight], dim=0)
packed_b = cat([q.bias, k.bias, v.bias], dim=0)
```

Failure cases:

- Loading must preserve original separate parameter names or explicitly materialize a derived packed constant with provenance.

### Rewrite: backbone token-to-map layout pass

Source pattern:

```text
slice tokens -> reshape(B, Hp, Wp, H) -> permute(0,3,1,2).contiguous()
```

Optimized candidate:

```text
emit NHWC feature map internally, optionally materialize NCHW only at ABI boundary
```

Preconditions:

- Consumer accepts NHWC or a layout manifest says the consumer is inside the same controlled layout region.
- Axis rewrites are explicit: LayerNorm remains last-dim over `H`; mean reductions over token axis are not channel reductions; backbone ABI default is NCHW.

Failure cases:

- External `BackboneOutput.feature_maps` parity expects NCHW when `reshape_hidden_states=True`.
- Any selected hidden state with `reshape_hidden_states=False` remains token sequence `[B,S,H]`.

### Guarded no-layout-translation regions

- Attention token sequence `[B,S,H]` and Q/K/V `[B,heads,S,64]` should not be treated as image layout tensors.
- Position interpolation temporarily uses NCHW-like `[1,H,side,side]` for bicubic interpolation; axis changes alter math.
- Classifier patch-token mean uses token axis `dim=1`, not spatial/channel axis after a layout rewrite.

## 10. Kernel fusion candidates

Highest priority:

- Conv patch embedding lowered to GEMM or fused im2col+GEMM for `P=14`, `C=3`, `H in {384,768,1024,1536}`.
- LayerNorm over last dim with `eps=1e-6`.
- Packed QKV projection plus reshape/transpose.
- Dense noncausal attention for short-to-medium vision sequences, especially `S=261` and `S=1374`.
- GEML/GELU/Linear MLP fusion for small/base/large.
- SwiGLU `Linear -> chunk -> SiLU*multiply -> Linear` for giant.

Medium priority:

- LayerScale multiply fused into residual add.
- Final LayerNorm plus CLS slice for base encoder output.
- Classifier tail: patch-token mean, concat, small linear.
- Position interpolation cache by image shape to avoid repeated bicubic work.

Lower priority:

- Dropout and DropPath can be omitted/identity in inference for official configs.
- `bool_masked_pos` `where` support for pretraining-style masked inputs.
- Backbone NCHW materialization fusion with downstream detection/segmentation consumers.

## 11. Runtime staging plan

Stage 1: parse config and load base/register weights.

- Admit `model_type="dinov2_with_registers"`, `num_register_tokens`, patch size, source-derived stage names, and ignored/historical config fields.

Stage 2: embedding parity.

- Implement Conv2d patch embedding, CLS/register insertion, and position add/interpolation for fixed 224x224 first.

Stage 3: one-block encoder parity.

- LayerNorm, separate or packed QKV, dense attention, output projection, LayerScale, MLP/SwiGLU.

Stage 4: full encoder parity.

- Validate small/base first, then large, then giant for SwiGLU and layer count.

Stage 5: output ABIs.

- Base model `last_hidden_state`/`pooler_output`; backbone selected feature maps with register-token stripping; optional classifier logits.

Stage 6: optimized rewrites.

- Conv-to-GEMM, packed QKV, fused attention, fused MLP/SwiGLU, NHWC-local backbone materialization.

Stage 7: dynamic image sizes.

- Guard shape divisibility and interpolate/cached positional embeddings for non-224 admitted sizes.

Initially stub/defer:

- Training losses, dropout/drop path randomness, `bool_masked_pos`, gradient checkpointing, and alternate attention output tensors.

## 12. Parity and validation plan

Concrete tests:

- Patch embedding Conv2d vs GEMM rewrite at `[1,3,224,224]`, `[B,3,224,224]`, and `[1,3,518,518]`.
- Position interpolation parity for 224 and 518 inputs, including dtype cast behavior around float32 bicubic interpolation.
- Register placement test: sequence length equals `1 + R + Npatch`; register slice is `[:, 1:1+R, :]`; patch slice begins at `1+R`.
- Single block fp32 parity against Transformers for random hidden states.
- Full encoder parity for `small` and `base` at 224x224.
- Large parity for 24 layers and giant parity for SwiGLU 40-layer path.
- Backbone ABI parity for `out_features=["stage2","stage5","stage8","stage11"]`: feature maps are NCHW and register tokens are excluded.
- Classifier parity on ImageNet 1-layer configs: logits shape `[B,1000]`, patch mean excludes register tokens.

Recommended tolerances:

- fp32: `atol=1e-4`, `rtol=1e-4` for full encoder; tighter for isolated ops except bicubic interpolation.
- fp16/bf16 optimized paths: start with `atol=2e-2`, `rtol=2e-2` for full encoder, then tighten per kernel.

Validation not run in this audit:

- No DinoML tests.
- No Transformers execution due local `huggingface_hub` import mismatch.

## 13. Performance probes

- Processor throughput: image decode/resize/crop/normalize to NCHW.
- Patch embedding only: Conv2d path vs WindowFlatten+GEMM for 224 and 518.
- Position interpolation cache hit/miss cost.
- Encoder block throughput by size: small/base/large/giant.
- Attention backend comparison for `S=261` and `S=1374`, batch sweep `[1, 2, 4, 8, 16]`.
- MLP vs SwiGLU block timing.
- Backbone output materialization cost: token sequence only vs NCHW feature maps.
- End-to-end encoder throughput for batch and image-resolution sweeps.
- Memory bandwidth/activation footprint for keeping selected hidden states for backbone outputs.

## 14. Skip/defer list

- Training and loss computation.
- Gradient checkpointing.
- Dropout/DropPath stochastic behavior.
- `bool_masked_pos` pretraining mask support.
- Attention probability outputs unless explicitly requested.
- Dynamic arbitrary image sizes beyond guarded `IH % patch == 0`, `IW % patch == 0`.
- Quantization and packed weights.
- Multi-GPU/tensor parallel.
- End-to-end downstream detector/segmenter postprocessing; this report only owns encoder/backbone feature contracts.

## 15. Final implementation checklist

- [ ] Parse `Dinov2WithRegistersConfig`, including `num_register_tokens`, source-derived `stage_names`, and ignored historical interpolation fields.
- [ ] Load official weights and preserve separate parameter names for Q/K/V or record packed-QKV derived constants with provenance.
- [ ] Implement NCHW patch Conv2d and/or guarded Conv-to-GEMM rewrite.
- [ ] Implement CLS/register token insertion: `[CLS, registers, patches]`.
- [ ] Implement learned absolute position add with bicubic float32 interpolation cache.
- [ ] Implement noncausal dense MHA with head dim 64 and no KV cache.
- [ ] Implement LayerNorm, LayerScale, residual adds, GELU MLP, and giant SwiGLU.
- [ ] Implement base model outputs: `last_hidden_state [B,1+R+N,H]`, `pooler_output [B,H]`.
- [ ] Implement backbone output selection, optional LayerNorm, register-token stripping, and NCHW map ABI.
- [ ] Implement optional ImageNet classifier head with patch mean excluding registers.
- [ ] Add parity tests for embedding/register placement, one block, full encoder, backbone ABI, and classifier logits.
- [ ] Add performance probes for patch embedding, attention, MLP/SwiGLU, position interpolation, and backbone materialization.
