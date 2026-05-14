# Transformers Audit: pvt_v2

## 1. Source basis

Transformers commit/version: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4` from local checkout `X:/H/transformers`.

Model id: representative official checkpoints under `OpenGVLab`: [`pvt_v2_b0`](https://huggingface.co/OpenGVLab/pvt_v2_b0), [`pvt_v2_b1`](https://huggingface.co/OpenGVLab/pvt_v2_b1), [`pvt_v2_b2`](https://huggingface.co/OpenGVLab/pvt_v2_b2), [`pvt_v2_b2_linear`](https://huggingface.co/OpenGVLab/pvt_v2_b2_linear), [`pvt_v2_b3`](https://huggingface.co/OpenGVLab/pvt_v2_b3), [`pvt_v2_b4`](https://huggingface.co/OpenGVLab/pvt_v2_b4), [`pvt_v2_b5`](https://huggingface.co/OpenGVLab/pvt_v2_b5).

Config source: `src/transformers/models/pvt_v2/configuration_pvt_v2.py` plus representative `config.json` files from the model repos above.

Source files inspected:

- [`modeling_pvt_v2.py`](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/pvt_v2/modeling_pvt_v2.py)
- [`configuration_pvt_v2.py`](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/pvt_v2/configuration_pvt_v2.py)
- [`convert_pvt_v2_to_pytorch.py`](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/pvt_v2/convert_pvt_v2_to_pytorch.py)
- Shared processor: [`models/pvt/image_processing_pvt.py`](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/pvt/image_processing_pvt.py)
- Test shape references: [`tests/models/pvt_v2/test_modeling_pvt_v2.py`](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/tests/models/pvt_v2/test_modeling_pvt_v2.py)

Any missing files or assumptions: there is no `image_processing_pvt_v2.py`; PVTv2 uses the shared `PvtImageProcessor`. No remote code was required. No model execution or DinoML tests were run.

## 2. High-level architecture

PVTv2 is a hierarchical vision encoder with four pyramid stages. The primary useful DinoML targets are:

- Required first target: `PvtV2ForImageClassification`, image classification.
- Also important: `PvtV2Backbone`, feature-map backbone for detection/segmentation frameworks.
- Base encoder `PvtV2Model` is required because both heads compose it.

Dataflow:

```text
CPU image resize/rescale/normalize -> NCHW pixel_values
-> stage1 overlap Conv2d patch embedding -> token LayerNorm -> blocks -> final stage norm -> NCHW feature map
-> stage2/3/4 repeat pyramid downsample + token blocks
-> classification: NHWC view -> flatten spatial -> mean over tokens -> Linear logits
-> backbone: selected NCHW feature maps stage1..stage4
```

The model repeatedly crosses layout domains:

- Conv2d regions consume/produce NCHW `[B,C,H,W]`.
- Attention, MLP, and LayerNorm regions operate on token layout `[B,H*W,C]`.
- Each stage ends by reshaping tokens back to NCHW for the next overlap patch embedding.

## 3. Important config dimensions

Source defaults from `PvtV2Config`:

| Field | Default |
|---|---:|
| image_size | `224` normalized to `(224,224)` |
| num_channels | 3 |
| num_encoder_blocks | 4 |
| depths | `[2,2,2,2]` |
| hidden_sizes | `[32,64,160,256]` |
| patch_sizes | `[7,3,3,3]` |
| strides | `[4,2,2,2]` |
| sr_ratios | `[8,4,2,1]` |
| num_attention_heads | `[1,2,5,8]` |
| mlp_ratios | `[8,8,4,4]` |
| hidden_act | `gelu` |
| qkv_bias | `true` |
| layer_norm_eps | `1e-6` |
| linear_attention | `false` |

Representative checkpoint sweep, from official `config.json` files:

| Checkpoint | Depths | Hidden sizes | Heads | MLP ratios | SRA | Linear SRA | Drop path |
|---|---|---|---|---|---|---:|---:|
| `pvt_v2_b0` | `2,2,2,2` | `32,64,160,256` | `1,2,5,8` | `8,8,4,4` | `8,4,2,1` | false | 0.0 |
| `pvt_v2_b1` | `2,2,2,2` | `64,128,320,512` | `1,2,5,8` | `8,8,4,4` | `8,4,2,1` | false | 0.0 |
| `pvt_v2_b2` | `3,4,6,3` | `64,128,320,512` | `1,2,5,8` | `8,8,4,4` | `8,4,2,1` | false | 0.0 |
| `pvt_v2_b2_linear` | `3,4,6,3` | `64,128,320,512` | `1,2,5,8` | `8,8,4,4` | ignored | true | 0.0 |
| `pvt_v2_b3` | `3,4,18,3` | `64,128,320,512` | `1,2,5,8` | `8,8,4,4` | `8,4,2,1` | false | 0.0 |
| `pvt_v2_b4` | `3,8,27,3` | `64,128,320,512` | `1,2,5,8` | `8,8,4,4` | `8,4,2,1` | false | 0.3 |
| `pvt_v2_b5` | `3,6,40,3` | `64,128,320,512` | `1,2,5,8` | `4,4,4,4` | `8,4,2,1` | false | 0.3 |

For 224x224 inputs, stage map shapes are normally:

| Stage | Conv in channels | Out channels | Spatial | Tokens |
|---|---:|---:|---:|---:|
| 1 | 3 | `C1` | 56x56 | 3136 |
| 2 | `C1` | `C2` | 28x28 | 784 |
| 3 | `C2` | `C3` | 14x14 | 196 |
| 4 | `C3` | `C4` | 7x7 | 49 |

## 3a. Family variation traps

- `linear_attention=True` changes the attention reducer from strided Conv2d with `kernel=stride=sr_ratio` to `AdaptiveAvgPool2d(7)` followed by 1x1 Conv2d, LayerNorm, GELU. It also inserts a ReLU before the MLP depthwise convolution.
- Published configs include `reshape_last_stage=true`, but the inspected source does not read it. DinoML should treat it as ignored for this source basis.
- The modeling source uses separate `query`, `key`, and `value` Linear modules. The conversion script documents original checkpoints storing K/V as one packed matrix named `attn.kv`, split in key-then-value order during conversion.
- `qkv_bias` only controls Q/K/V Linear bias. Attention output projection, MLP projections, and classifier use normal biased Linear modules in the inspected source.
- `num_attention_heads=[1,2,5,8]` means stage 3 uses 5 heads. Do not assume powers of two.
- Head dimension remains regular for published configs: B0 is `[32,32,32,32]`; B1-B5 are `[64,64,64,64]`.
- Backbone outputs are NCHW feature maps, not token sequences. `out_features`/`out_indices` select ordered stage maps.
- Layout optimization needs guards around every NCHW <-> token transition. The semantic graph should preserve PyTorch axis numbers first; NHWC/channel-last is only a local optimization opportunity.
- Classification uses spatial average pooling implemented as `permute(0,2,3,1) -> reshape(B,-1,C) -> mean(dim=1)`, not a class token.

## 4. Operator coverage checklist

Tensor/layout ops:

- NCHW Conv2d input/output.
- Flatten spatial dimensions: `[B,C,H,W] -> [B,C,H*W]`.
- Transpose/permute: `[B,C,N] <-> [B,N,C]`, `[B,N,C] -> [B,C,H,W]`, `[B,C,H,W] -> [B,H,W,C]`.
- Reshape/view with static or guarded dynamic `H,W`.
- Contiguous materialization after stage token-to-map permute.
- Mean reduction over token axis for classification.
- Backbone tuple assembly for selected feature maps.

Neural network primitives:

- Overlap patch Conv2d: stage 1 `kernel=7,stride=4,pad=3`; stages 2-4 `kernel=3,stride=2,pad=1`.
- Depthwise Conv2d in every MLP: `kernel=3,stride=1,pad=1,groups=hidden_features`.
- Spatial reduction Conv2d in attention for SRA stages with `sr_ratio > 1`: `kernel=stride=sr_ratio`, no padding.
- Linear projections: Q/K/V with bias when `qkv_bias=true`; attention output, MLP dense1/dense2, and classifier with bias.
- LayerNorm over last token channel dimension with `eps=1e-6`.
- GELU; optional ReLU only for `linear_attention=True`.
- Dropout and DropPath are inference identities.

Attention primitives:

- Noncausal encoder self-attention only.
- Per-stage MHA with rectangular attention when SRA reduces K/V length.
- Softmax over key length.
- Matmul QK^T and attention-probability times V.
- No causal mask, no KV cache, no packed varlen metadata.

Preprocessing-coupled ops:

- `PvtImageProcessor`: resize to 224x224 by default, rescale by `1/255`, normalize by ImageNet mean/std, output `pixel_values` NCHW.

## 5. Layer/block breakdown

Stage `s`, repeated for 4 stages:

```text
input map: [B,C_in,H_in,W_in]
patch = Conv2d(C_in -> C_s, kernel=patch_size_s, stride=stride_s, pad=floor(kernel/2))
tokens = flatten_spatial(patch).transpose(1,2)       # [B,N_s,C_s]
tokens = LayerNorm(C_s)(tokens)

for each block in depths[s]:
  a = LayerNorm(C_s)(tokens)
  q = Linear(C_s -> C_s, bias=qkv_bias)(a)
  if linear_attention:
    kv_src = AdaptiveAvgPool2d(7)(a as [B,C_s,H_s,W_s])
    kv_src = Conv2d(C_s -> C_s, kernel=1)(kv_src)
    kv_src = GELU(LayerNorm(C_s)(kv_src as [B,49,C_s]))
  elif sr_ratio_s > 1:
    kv_src = Conv2d(C_s -> C_s, kernel=sr_ratio_s, stride=sr_ratio_s)(a as [B,C_s,H_s,W_s])
    kv_src = LayerNorm(C_s)(kv_src as [B,N_kv,C_s])
  else:
    kv_src = a
  k = Linear(C_s -> C_s, bias=qkv_bias)(kv_src)
  v = Linear(C_s -> C_s, bias=qkv_bias)(kv_src)
  attn = softmax((q @ k.T) / sqrt(C_s / heads_s), dim=-1)
  tokens = tokens + Linear(C_s -> C_s)(attn @ v)

  m = LayerNorm(C_s)(tokens)
  m = Linear(C_s -> C_s * mlp_ratio_s)(m)
  if linear_attention: m = ReLU(m)
  m = DepthwiseConv2d(kernel=3,pad=1,groups=C_s * mlp_ratio_s)(m as map)
  m = GELU(m)
  m = Linear(C_s * mlp_ratio_s -> C_s)(m)
  tokens = tokens + m

tokens = final stage LayerNorm(C_s)(tokens)
output map = tokens.reshape(B,H_s,W_s,C_s).permute(0,3,1,2).contiguous()
```

Classification head:

```text
last map [B,C4,H4,W4] -> [B,H4,W4,C4] -> [B,H4*W4,C4]
pooled = mean(tokens, dim=1)
logits = Linear(C4 -> num_labels)(pooled)
```

## 6. Attention requirements

Attention is encoder-only, noncausal self-attention. No generation, no KV cache, no RoPE, no ALiBi, no relative-position bias, and no attention mask is consumed by `PvtV2Model.forward`.

For normal SRA:

- Query length: `N_q = H_s * W_s`.
- Key/value length: `N_kv = floor((H_s - sr_ratio_s) / sr_ratio_s + 1) * floor((W_s - sr_ratio_s) / sr_ratio_s + 1)` for `sr_ratio_s > 1`, because the Conv2d has no padding.
- If `sr_ratio_s == 1`, `N_kv = N_q`.
- Published 224x224 B1-B5 SRA lengths by stage: `3136->49`, `784->49`, `196->49`, `49->49`.

For Linear SRA:

- K/V source is always pooled to 7x7, so `N_kv=49` for every stage, independent of input resolution after adaptive pooling.
- The config doc says `sr_ratio` is ignored; the source still stores it but branches on `linear_attention`.

FlashAttention/SDPA compatibility: the dense attention math is standard enough for fused attention if DinoML accepts rectangular noncausal attention and preserves Q/K/V head layout `[B,heads,N,head_dim]`. The expensive source-specific work is the K/V spatial reducer before K/V projection.

## 7. Position encoding and custom math

PVTv2 has no absolute positional table or RoPE. Positional information comes from convolutions:

- Overlapping patch embeddings include padded Conv2d.
- The MLP includes a 3x3 depthwise Conv2d over the token map.
- Attention SRA uses spatial Conv2d or adaptive pooling to make K/V spatially aware.

Custom shape/layout sketch:

```python
def token_to_nchw(x, height, width):
    b, n, c = x.shape
    assert n == height * width
    return x.transpose(1, 2).reshape(b, c, height, width)

def nchw_to_token(x):
    b, c, h, w = x.shape
    return x.flatten(2).transpose(1, 2), h, w
```

Precompute opportunity: none for position tables; convolution weights are normal learned constants.

## 8. Preprocessing and input packing

The shared `PvtImageProcessor` is CPU/data-pipeline work for first integration:

- Resize enabled, default size `{height:224,width:224}`.
- Bicubic resampling in source defaults; HF preprocessor configs store `resample: 2`.
- Rescale enabled with factor `1/255`.
- Normalize enabled with ImageNet mean `[0.485,0.456,0.406]` and std `[0.229,0.224,0.225]`.
- Model input is `pixel_values` in NCHW `[B,3,H,W]`.

No token packing, masks, placeholder scatter, box metadata, or postprocessor is required for image classification. Backbone consumers such as DETR/MaskFormer own downstream task-specific preprocessing/postprocessing.

## 9. Graph rewrite / lowering opportunities

### Rewrite: overlap patch Conv2d to im2col/GEMM

Source pattern: fixed Conv2d patch embedding with padding and overlapping windows.

Replacement: `Im2Col(NCHW windows) -> GEMM(weight_flat.T) -> BiasAdd -> reshape NCHW`.

Preconditions:

- Static or guarded `kernel`, `stride`, `padding`, `dilation=1`, `groups=1`.
- Input channel count matches config or previous stage output.
- Output spatial formula matches PyTorch Conv2d exactly.

Failure cases: not a non-overlap patchify rewrite; padding and overlap are required, so a simple reshape-window flatten is unsafe.

Parity test sketch: compare each stage patch embedding output before LayerNorm for random NCHW inputs at 224 and one non-square guarded size.

### Rewrite: SRA Conv2d K/V reducer to pooled GEMM/conv provider

Source pattern:

```text
tokens [B,N,C] -> NCHW [B,C,H,W] -> Conv2d(C->C,k=sr,stride=sr) -> tokens -> LayerNorm -> K/V Linear
```

Replacement: lower Conv2d through a provider-backed conv path or im2col/GEMM, then fuse tokenization and LayerNorm where layout permits.

Preconditions:

- `linear_attention=false`.
- `sr_ratio > 1`; stage 4 skips reducer for normal configs.
- No padding in the SRA Conv2d.

Failure cases: `linear_attention=true` must route to adaptive average pool + 1x1 Conv2d instead.

### Rewrite: Linear SRA adaptive pool path

Source pattern:

```text
NCHW -> AdaptiveAvgPool2d(7) -> Conv2d 1x1 -> flatten/transposed tokens -> LayerNorm -> GELU
```

Replacement: specialized `adaptive_avg_pool2d_output_7x7` plus 1x1 GEMM/Conv2d. For fixed 224 inputs, all stages already have output 7x7 after normal pyramid stages, but adaptive pooling still matters for general resolution parity.

Preconditions:

- `linear_attention=true`.
- Output size exactly 7x7.
- Preserve PyTorch adaptive-pool bin boundaries for non-224 inputs.

### Rewrite: token Linear as GEMM

Source pattern: Linear on `[B,N,C]`.

Replacement: flatten to `[B*N,C] -> GEMM_RCR(weight [out,C]) -> bias -> reshape`.

Preconditions:

- Dense contiguous or layout-described token tensor.
- Bias present for Q/K/V and MLP/classifier as source defaults imply.

Weight transform: PyTorch Linear weight is `[out_features,in_features]`, matching RCR RHS column-major/dense-transposed logical use if DinoML uses existing GEMM RCR conventions.

### Rewrite: layout-transition elimination in controlled regions

Source pattern: `NCHW -> flatten/transpose -> LayerNorm/Linear -> transpose/view -> NCHW` around MLP depthwise conv and SRA reducer.

Replacement: a guarded layout pass can keep channel-last/NHWC internally for LayerNorm-adjacent regions or keep token-major for adjacent GEMMs, but only if Conv2d/depthwise Conv2d providers and axis-sensitive reductions are rewritten.

Required axis rewrites:

- LayerNorm normalized axis remains channels: token `dim=-1`, NHWC `dim=-1`, NCHW would be `dim=1` and is not source-equivalent for a vanilla LayerNorm op.
- Classification mean is over spatial tokens; in NHWC map form it is reduction over `H,W`, not channel.
- Backbone ABI must return NCHW feature maps unless a separate consumer contract opts into NHWC.

## 10. Kernel fusion candidates

Highest priority:

- Conv2d coverage for overlap patch embeddings and SRA reducers. PVTv2 cannot run without general Conv2d plus depthwise Conv2d.
- LayerNorm over `[B,N,C]` with `eps=1e-6`. It appears before attention, before MLP, after patch embedding, after K/V spatial reduction, and at each stage output.
- Token Linear/GEMM + bias for Q/K/V, attention projection, and MLP projections.
- Rectangular noncausal attention for SRA: stage 1 attention uses large Q length with only 49 K/V tokens.

Medium priority:

- Depthwise Conv2d 3x3 in MLP, ideally fused with token-to-map/map-to-token layout plumbing.
- SRA reducer Conv2d + LayerNorm fusion, especially for stages 1-3 where all K/V lengths collapse to 49 on 224 inputs.
- Linear SRA path: adaptive average pool 7x7 + 1x1 Conv2d + LayerNorm + GELU.
- Classification spatial mean + Linear head.

Lower priority:

- DropPath/dropout training behavior; inference treats these as identities.
- Output attentions materialization. Useful for debugging, not required for classification/backbone fast path.

## 11. Runtime staging plan

Stage 1: parse `PvtV2Config`, reject unsupported config combinations explicitly, and load converted HF weights.

Stage 2: implement one stage with patch Conv2d, token LayerNorm, one attention block using dense rectangular attention, MLP depthwise Conv2d, and stage map output.

Stage 3: run full `PvtV2Model` encoder parity for B0 normal SRA at 224x224.

Stage 4: add classification head parity for `OpenGVLab/pvt_v2_b0`.

Stage 5: add backbone ABI with `out_features`/`out_indices` selection and NCHW feature map outputs.

Stage 6: broaden checkpoint configs to B1/B2/B5 and then `b2_linear` after adaptive pooling is available.

Stage 7: add layout/fusion passes for token/map transitions, SRA reduction, depthwise Conv2d, and attention.

Can stub initially: labels/losses, training dropout/DropPath, gradient checkpointing, output attentions.

## 12. Parity and validation plan

- Config tests: parse every representative `OpenGVLab` config and verify effective source defaults for omitted fields.
- Patch embedding parity: random NCHW inputs, per-stage Conv2d + flatten + LayerNorm.
- SRA reducer parity: stages with `sr_ratio` 8/4/2 and stage 4 no-reducer branch.
- Linear SRA parity: `b2_linear`, including adaptive pool 7x7 and MLP pre-ReLU.
- Single block parity: compare block output after attention residual and after MLP residual.
- Full encoder parity: B0 first, then B2/B5 for depth stress.
- Classification parity: logits for a fixed preprocessed image; source tests use tolerance `1e-4` for fp32.
- Backbone parity: selected feature maps shapes and values for `out_features=["stage1","stage2","stage3","stage4"]`.
- Reduced precision smoke: fp16 model/input path for classification once Conv/LayerNorm/attention support tolerances are established.

Recommended tolerances: fp32 `rtol=1e-4, atol=1e-4` for fixed-image integration; fp16 start around `rtol=5e-3, atol=5e-3` and tighten per op.

## 13. Performance probes

- Processor throughput: PIL/torchvision resize+normalize outside the GPU graph.
- Stage-by-stage latency and memory: isolate stage 1 large-Q attention versus later stages.
- Conv2d provider comparison: overlap patch, SRA reducer, depthwise MLP, 1x1 linear SRA.
- Attention shape sweep: `[N_q,N_kv]` of `[3136,49]`, `[784,49]`, `[196,49]`, `[49,49]`.
- Batch-size sweep for B0/B2/B5 at 224x224.
- Resolution sweep for non-224 inputs, especially normal SRA output lengths and Linear SRA adaptive pooling.
- Layout-pass probe: source-faithful NCHW/token transitions versus guarded channel-last fusion.
- Backbone throughput returning all feature maps versus final stage only.

## 14. Skip/defer list

- Training losses and label handling.
- Dropout and DropPath stochastic behavior.
- Gradient checkpointing.
- Output attentions unless needed for debugging parity.
- Non-image-classification downstream detector/segmenter postprocessing; PVTv2 backbone only emits feature maps.
- Remote-code or non-OpenGVLab configs.
- Quantized/packed weights; no inspected source path requires them.
- NHWC public ABI. Keep it an internal guarded optimization only.

## 15. Final implementation checklist

- [ ] Parse `PvtV2Config` and representative OpenGVLab config fields.
- [ ] Implement/route NCHW Conv2d for overlap patch embeddings.
- [ ] Implement LayerNorm over token channel axis with `eps=1e-6`.
- [ ] Lower token Linear layers to GEMM+bias.
- [ ] Implement SRA Conv2d reducer for `sr_ratio > 1`.
- [ ] Implement rectangular noncausal MHA for encoder self-attention.
- [ ] Implement depthwise Conv2d 3x3 in MLP.
- [ ] Implement classification spatial mean and classifier Linear.
- [ ] Implement backbone NCHW feature-map output selection.
- [ ] Gate `linear_attention=True` until adaptive average pool 7x7 path is implemented.
- [ ] Add B0 single-stage, full-encoder, classification, and backbone parity tests.
- [ ] Add B2/B5 depth stress tests.
- [ ] Add `b2_linear` parity once Linear SRA is supported.
- [ ] Benchmark stage attention shapes and Conv2d/depthwise Conv2d provider choices.
