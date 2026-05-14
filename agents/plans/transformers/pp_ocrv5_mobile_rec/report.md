# Transformers Family Audit: pp_ocrv5_mobile_rec

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: PaddlePaddle/PP-OCRv5_mobile_rec_safetensors
Config source: HF config.json and preprocessor_config.json, saved in _sources/
Source files inspected: pp_ocrv5_mobile_rec configuration/modeling/modular, pp_ocrv5_server_rec image processor/modeling, pp_lcnet_v3 backbone
Any missing files or assumptions: no tokenizer file exists; recognition decode is image-processor postprocess. Language-specific HF repos inspected are Paddle inference configs, not native Transformers configs.
```

Primary URLs:

- `https://huggingface.co/PaddlePaddle/PP-OCRv5_mobile_rec_safetensors`
- `https://huggingface.co/PaddlePaddle/en_PP-OCRv5_mobile_rec`
- `https://huggingface.co/PaddlePaddle/cyrillic_PP-OCRv5_mobile_rec`
- `https://huggingface.co/PaddlePaddle/latin_PP-OCRv5_mobile_rec`

Generated-file note: `modeling_pp_ocrv5_mobile_rec.py`, `configuration_pp_ocrv5_mobile_rec.py`, and the server-rec processor/modeling files are generated from modular files. Future source edits should target the modular files.

## 2. High-level architecture

Runtime target: `PPOCRV5MobileRecForTextRecognition`, an OCR text-line recognizer.

Dataflow:

```text
image preprocessing -> NCHW PP-LCNetV3 backbone -> avg_pool2d -> SVTR conv/sequence encoder -> per-timestep Linear -> softmax probabilities -> CTC-style duplicate/blank removal
```

Stages:

- CPU/data pipeline: RGB conversion, resize to height 48 with batch-widest aspect ratio, rescale, ImageNet normalization, pad to at least width 320.
- Backbone: PP-LCNetV3 feature extractor returning `feature_maps[-1]` in `NCHW`.
- Recognition head: height-collapsed feature map, local convs, 2 noncausal self-attention blocks, classifier, softmax over character classes.
- Postprocess: argmax over class axis, remove adjacent duplicate ids, drop blank id 0, map ids through `character_list`, average selected probabilities.

There is no autoregressive decode, no KV cache, and no text tokenizer ABI.

## 3. Important config dimensions

Native safetensors config:

| Field | Value | Source |
| --- | ---: | --- |
| `model_type` | `pp_ocrv5_mobile_rec` | HF `config.json` |
| `backbone_config.model_type` | `pp_lcnet_v3` | HF `config.json` |
| `backbone_config.scale` | `0.95` | HF `config.json` |
| `backbone_config.divisor` | `16` | HF `config.json` |
| `backbone_config.out_features` | `stage2..stage5` | HF `config.json` |
| `hidden_size` | `120` | HF `config.json` |
| `mlp_ratio` | `2.0` | HF `config.json` |
| `depth` | `2` SVTR blocks | HF `config.json` |
| `num_attention_heads` | `8` | HF `config.json` |
| `head_dim` | `15` | source-derived, `120 / 8` |
| `head_out_channels` | `18385` | HF `config.json` |
| `conv_kernel_size` | `[1, 3]` | HF `config.json` |
| `qkv_bias` | `true` | HF `config.json` |
| `attention_dropout` | `0.0` | HF `config.json` |
| `layer_norm_eps` | `1e-6` | source default |
| processor size | height 48, width 320 | HF `preprocessor_config.json` |
| processor max width | 3200 | HF `preprocessor_config.json` |
| character entries | 18385, first entry `blank` | HF `preprocessor_config.json` |

Representative config sweep:

| Repo | Config kind | Operator-significant fields |
| --- | --- | --- |
| `PP-OCRv5_mobile_rec_safetensors` | native Transformers | PP-LCNetV3 scale 0.95, SVTR hidden 120, 8 heads, 18385 output classes |
| same repo preprocessor | Transformers processor | `pixel_values` resize/pad ABI, 18385 class dictionary with blank at id 0 |
| `en_PP-OCRv5_mobile_rec` | Paddle inference | CTC dictionary has 436 entries; TRT dynamic shapes `[1,3,48,160]`, `[1,3,48,320]`, `[8,3,48,3200]` |
| `cyrillic_PP-OCRv5_mobile_rec` | Paddle inference | CTC dictionary has 850 entries; same dynamic shape hints |
| `latin_PP-OCRv5_mobile_rec` | Paddle inference | CTC dictionary has 836 entries; same dynamic shape hints |

The Paddle configs are useful variation evidence, but DinoML should not load them as native Transformers configs without a separate Paddle-export adapter.

## 3a. Family variation traps

- The mobile class composes `pp_lcnet_v3`; server-rec defaults compose `hgnet_v2`. Do not accidentally route mobile checkpoints through server-rec backbone assumptions.
- `configuration_pp_ocrv5_mobile_rec.py` in the generated checkout calls `consolidate_backbone_kwargs_to_config` twice, first with mobile `pp_lcnet_v3`, then with inherited server `hgnet_v2`. The observed native checkpoint supplies an explicit `pp_lcnet_v3` backbone and should be admitted only when that resolved backbone is PP-LCNetV3 for this audit.
- `head_out_channels` must match the processor `character_list` length for native Transformers end-to-end decode. Language-specific Paddle repos have different dictionaries and no native Transformers `head_out_channels`.
- `qkv_bias=True` creates a packed QKV bias where K bias is zeros; `qkv_bias=False` removes QKV bias entirely.
- Input layout is semantically `NCHW` for the model graph. NHWC is only a guarded layout optimization.
- Height is assumed to collapse to 1 before `squeeze(2).transpose(1, 2)`. With the observed 48-pixel processor height and safetensors strides, this holds; other heights need a guard or a separate shape derivation.
- Width is dynamic. Processor may emit widths from 320 up to 3200; Paddle shape hints include 160, 320, and 3200. Sequence length after the observed safetensors stride/pool path is approximately `W / 8` for divisible widths: 320 -> 40, 3200 -> 400.
- Attention is encoder-style noncausal MHA over image-column tokens, not text generation.
- The model returns probabilities after softmax, not raw logits, in `last_hidden_state`.

## 4. Operator coverage checklist

Tensor/layout ops:

- `reshape`, `view`, `flatten(start_dim=2)`, `transpose`, `permute`, `contiguous`
- `squeeze(dim=2)` with height-is-one guard
- `cat(dim=1)` for NCHW channel concatenation

Neural network primitives:

- `Conv2d` NCHW, including standard 3x3/5x5, 1x1 pointwise, and depthwise grouped conv where `groups=in_channels`
- `BatchNorm2d` inference
- `LayerNorm` over last sequence dimension, eps `1e-6`
- `Linear(120 -> 360)` packed QKV, `Linear(120 -> 120)`, `Linear(120 -> 240)`, `Linear(240 -> 120)`, `Linear(120 -> 18385)`
- Activations: `hardswish`, `silu`, `relu`, `hardsigmoid`
- `AdaptiveAvgPool2d(1)` in PP-LCNetV3 squeeze-excitation
- `avg_pool2d(kernel=(3,2), stride=(3,2))`
- Elementwise add, multiply, scalar affine `scale * x + bias`

Attention primitives:

- Noncausal self-attention with `B,H,T,D` Q/K/V, `H=8`, `D=15`
- `MatMul(Q, K^T) * head_dim^-0.5`, softmax on last dim, `MatMul(prob, V)`
- Optional Transformers SDPA/Flash/Flex dispatch must preserve noncausal no-mask semantics.

Preprocessing-coupled ops:

- Resize to height 48 and dynamic width derived from aspect ratio and batch-widest sample
- Rescale and ImageNet normalization
- Right/bottom padding to at least 48x320 when target width is below 320

Discrete decode/postprocess:

- `argmax/max(dim=-1)` over class axis
- Adjacent duplicate suppression
- Blank id 0 suppression
- Per-sample mean confidence over selected positions
- Character id lookup through processor `character_list`

## 5. Layer/block breakdown

PP-LCNetV3 backbone, `NCHW`:

```text
stem: Conv2d(3 -> 16, k=3, stride=2, pad=1) -> BN -> identity activation
for each stage block:
  depthwise LearnableRepLayer:
    optional identity BN when stride=1 and channels match
    optional 1x1 Conv+BN branch
    4 parallel kxk depthwise Conv+BN branches
    sum branches -> scalar learnable affine -> hardswish-affine unless stride == 2
  optional SE:
    AdaptiveAvgPool2d(1) -> Conv1x1 -> ReLU -> Conv1x1 -> Hardsigmoid -> multiply residual
  pointwise LearnableRepLayer:
    1x1 Conv+BN branches -> sum -> scalar learnable affine -> hardswish-affine
```

Observed safetensors mobile channel path after `make_divisible(..., divisor=16)` is approximately:

```text
16 -> 32 -> 64 -> 128 -> 240 -> 480
```

Recognition head:

```text
hidden = backbone(pixel_values).feature_maps[-1]
hidden = avg_pool2d(hidden, kernel=(3,2), stride=(3,2))
residual = hidden
hidden = Conv2d(C=480 -> 60, k=(1,3), pad=(0,1)) -> BN -> SiLU
hidden = Conv2d(60 -> 120, k=1) -> BN -> SiLU
hidden: [B,120,1,T] -> flatten(2).transpose(1,2) = [B,T,120]
repeat depth=2:
  y = LayerNorm(120)
  qkv = Linear(120 -> 360, bias=[q_bias, zeros, v_bias])
  qkv -> reshape [B,T,3,8,15] -> permute [3,B,8,T,15]
  y = noncausal self-attention -> Linear(120 -> 120)
  hidden = hidden + y
  y = LayerNorm(120) -> Linear(120 -> 240) -> SiLU -> Linear(240 -> 120)
  hidden = hidden + y
hidden = LayerNorm(120)
hidden -> view [B,1,T,120] -> permute [B,120,1,T]
hidden = Conv2d(120 -> 480, k=1) -> BN -> SiLU
hidden = cat(residual, hidden, dim=1) = [B,960,1,T]
hidden = Conv2d(960 -> 60, k=(1,3), pad=(0,1)) -> BN -> SiLU
hidden = Conv2d(60 -> 120, k=1) -> BN -> SiLU
hidden = squeeze(2).transpose(1,2) = [B,T,120]
probs = Linear(120 -> 18385) -> softmax(dim=2, dtype=float32) -> cast back
```

## 6. Attention requirements

Required attention variant:

- Noncausal self-attention only.
- MHA, not MQA/GQA: `num_attention_heads=8`, `num_key_value_heads` not present.
- `head_dim=15`, so fused kernels must accept non-power-of-two head dims.
- No attention mask is used; the forward passes `attention_mask=None`.
- No RoPE, ALiBi, relative bias, cache, packed varlen metadata, or generation decode.
- Source eager path does not upcast attention weights to fp32 before softmax. It uses `softmax` in the active tensor dtype unless the selected attention backend changes that behavior.
- `_supports_flash_attn`, `_supports_sdpa`, and `_supports_flex_attn` are true, but first DinoML parity can use explicit BMM/softmax/BMM.

Shape for sequence width `T`:

```text
q,k,v: [B, 8, T, 15]
attn_scores: [B, 8, T, T]
attn_output before merge: [B, T, 8, 15]
```

## 7. Position encoding and custom math

There is no positional encoding in the inspected source. The SVTR blocks attend over flattened spatial order without explicit learned or sinusoidal position embeddings.

Custom math to preserve:

```python
def pp_ocrv5_mobile_rec_ctc_decode(probs, character_list):
    pred_prob, pred_id = probs.max(dim=-1)
    keep = torch.ones_like(pred_id, dtype=torch.bool)
    keep[:, 1:] = pred_id[:, 1:] != pred_id[:, :-1]
    keep &= pred_id != 0
    # Per sample: text = ''.join(character_list[i] for i in pred_id[b][keep[b]])
    # score = pred_prob[b][keep[b]].mean()
```

Guard: empty decoded selections need a defined score policy. The source calls `.mean()` on an empty tensor, which yields `nan` in PyTorch.

## 8. Preprocessing and input packing

Transformers image processor ABI:

- Input images are converted to RGB by default.
- Resize target height is 48. Target width is based on the widest image in the batch, aspect ratio, default 320x48 ratio, and `max_image_width=3200`.
- Rescale and normalize use ImageNet standard mean/std.
- If target width is below `pad_size.width=320`, pad to width 320.
- Output consumed by the model is `pixel_values` in `NCHW`, `[B,3,48,W]`.
- Width is batch-coupled because all grouped images resize to the batch target size.

Paddle inference configs add useful deployment hints:

- Dynamic shapes: min `[1,3,48,160]`, opt `[1,3,48,320]`, max `[8,3,48,3200]`.
- Preprocess names include `DecodeImage`, `RecResizeImg`, and `KeepKeys`.
- Postprocess name is `CTCLabelDecode`.

No tokenizer or text input enters the neural graph.

## 9. Graph rewrite / lowering opportunities

### Rewrite: fold Conv2d + BatchNorm2d for inference

Preconditions:

- Module is in eval/inference mode.
- BatchNorm running mean/var, gamma, beta, eps are constants.
- Conv branch has known weight and optional bias.

Replacement:

```text
Conv2d(weight, bias) -> BatchNorm2d -> activation
becomes
Conv2d(folded_weight, folded_bias) -> activation
```

Failure cases: training mode, mutable BN state, missing running stats.

Parity test sketch: random NCHW inputs for each standard/depthwise/1x1 branch, compare folded vs unfused fp32 and fp16 tolerances.

### Rewrite: reparameterize LearnableRepLayer branch sums

Preconditions:

- Inference mode.
- All branch Conv+BN weights are constants.
- Same stride, padding, groups, output channels.
- 1x1 branch can be padded into kxk center.
- Identity BN branch can be converted to grouped identity convolution when present.

Replacement:

```text
sum(identity_bn?, conv1x1_bn?, convkxk_bn x N) -> scalar affine -> activation
becomes
single Conv2d(fused_kxk_weight, fused_bias) -> scalar affine -> activation
```

Weight transform: fold each branch BN, zero-pad 1x1 kernels to kxk, add kernels/biases elementwise.

Failure cases: nonconstant branch weights, training mode, mismatched groups/stride/padding, dynamic branch count not known.

### Rewrite: 1x1 Conv2d -> GEMM

Preconditions:

- `kernel_size=(1,1)`, `stride=1`, `padding=0`, `dilation=1`.
- NCHW input is contiguous or locally layout-controlled.

Replacement:

```text
NCHW -> flatten B*H*W rows -> GEMM(Cin -> Cout) -> restore NCHW
```

Layout constraints: in a channel-last optimized region, use NHWC row-major channels and rewrite consumers accordingly.

### Rewrite: height-1 conv `(1,3)` as 1D width convolution

Preconditions:

- Input height is exactly 1.
- Kernel is `(1,3)`, padding `(0,1)`, stride 1.
- Groups are 1.

Replacement:

```text
[B,C,1,T] -> [B,C,T] Conv1d(k=3,pad=1) -> [B,C,1,T]
```

This may simplify a local sequence-head lowering path. Preserve NCHW semantics outside the guarded region.

### Rewrite: explicit attention chain to BMM/softmax/BMM

Preconditions:

- Noncausal self-attention, no mask, dropout disabled.
- `head_dim=15`, `scale=head_dim^-0.5`.

Replacement:

```text
reshape/permute QKV -> BMM(Q, K^T) -> scale -> softmax(last) -> BMM(P, V) -> merge heads
```

Failure cases: future masked attention, training dropout, backend that changes precision semantics.

### Layout rewrite: NCHW backbone to NHWC/channel-last candidate

Candidate region: PP-LCNetV3 conv-only backbone and height-1 head convolutions.

Required axis rewrites:

- Conv/BN/channel activations: channel axis `1` becomes last axis.
- `cat(dim=1)` becomes `cat(dim=-1)`.
- `flatten(2).transpose(1,2)` over `[B,C,H,W]` must become a controlled reshape from `[B,H,W,C]` to `[B,H*W,C]`.
- `view(batch,height,width,channels).permute(0,3,1,2)` can be eliminated only if the sequence-to-image restoration is tracked as NHWC.
- `squeeze(2).transpose(1,2)` over NCHW must be replaced by `[B,1,T,C] -> [B,T,C]` in NHWC.

Guarded no-layout-translation boundaries:

- Initial semantic import should preserve NCHW.
- Processor output ABI remains `NCHW`.
- SVTR sequence blocks are `[B,T,C]` and should be protected from image-layout rewrites.

## 10. Kernel fusion candidates

Highest priority:

- Conv2d+BatchNorm folding and LearnableRepLayer branch reparameterization: removes many branch convolutions/adds in the PP-LCNetV3 backbone.
- Depthwise and pointwise Conv2d provider path: backbone is convolution-dominated.
- LayerNorm + packed QKV Linear for `120 -> 360`: small but repeated over dynamic sequence length.
- Attention BMM/softmax/BMM for `D=15`: width can reach `T=400`, making attention visible.
- Final classifier `Linear(120 -> 18385)` plus softmax: class dimension is large and dominates the head.

Medium priority:

- SiLU/hardswish/hardsigmoid elementwise fusion after convs.
- 1x1 Conv2d as GEMM in channel-last local regions.
- CTC argmax/dedup/blank suppression as a compact postprocess kernel for batched output.

Lower priority:

- Training dropout and gradient checkpointing.
- Exotic attention backend parity before the explicit eager chain is validated.

## 11. Runtime staging plan

Stage 1: parse native Transformers config and preprocessor config; reject non-`pp_lcnet_v3` mobile backbones for this audit scope.

Stage 2: implement/validate PP-LCNetV3 backbone as NCHW with static height 48 and dynamic width buckets.

Stage 3: implement backbone-to-head pooling and height-collapse guards.

Stage 4: implement SVTR head with explicit BMM/softmax/BMM attention and final classifier probabilities.

Stage 5: implement CTC-style postprocess from probability tensor to text/score outside the compiled core.

Stage 6: add inference rewrites: Conv+BN folding, LearnableRepLayer reparameterization, 1x1 Conv->GEMM, guarded NHWC conv region.

Stage 7: benchmark width/batch buckets and decide whether final classifier softmax needs a specialized kernel.

Stub initially: OCR detection/cropping, text-line extraction, language-specific Paddle repo adapters, and full image processor on GPU.

## 12. Parity and validation plan

- Processor parity: fixed input images with aspect ratios below 320, at 320, and near max width 3200; verify `pixel_values` shape and numeric normalization.
- Backbone unit parity: random `[B,3,48,W]` for `W in {160,320,640,3200}` against Transformers PP-LCNetV3.
- Reparameterization parity: compare unfused LearnableRepLayer against fused single-conv form per stage.
- SVTR block parity: random `[B,T,120]` for `T in {20,40,400}`, fp32 first; include `head_dim=15`.
- End-to-end neural parity: compare `last_hidden_state` probabilities for the safetensors checkpoint.
- Postprocess parity: crafted probability tensors for duplicate ids, blank id 0, all-blank output, and mixed confidence.

Suggested tolerances:

- fp32: `rtol=1e-4`, `atol=1e-5` for unfused graph; slightly looser after Conv+BN folding.
- fp16/bf16: start with `rtol=1e-2`, `atol=1e-2`, isolate softmax/classifier drift.

## 13. Performance probes

- CPU image preprocessing throughput by width bucket and batch size.
- Backbone-only latency/throughput for `[B,3,48,W]`, `B in {1,8}`, `W in {160,320,640,3200}`.
- Fused vs unfused LearnableRepLayer branch count impact.
- SVTR attention time by `T in {20,40,100,400}`.
- Final classifier + softmax time for `[B,T,18385]`.
- End-to-end text-line recognition requests/sec with detection/cropping excluded.
- NCHW vs guarded NHWC conv-region comparison.
- Memory footprint for probability output at max width: `[8,400,18385]` is large and may dominate output bandwidth.

## 14. Skip/defer list

- Training, dropout, gradient checkpointing.
- OCR detection, text-line cropping, document layout orchestration.
- Language-specific Paddle inference repos as first native Transformers targets.
- General tokenizer integration; this family uses processor-owned CTC decode.
- KV cache, autoregressive generation, beam search.
- Server-rec `hgnet_v2` backbone and detection families unless separately audited.
- Broad NHWC translation until NCHW parity and axis guards are validated.

## 15. Final implementation checklist

- [ ] Parse `PPOCRV5MobileRecConfig` and paired preprocessor config.
- [ ] Reject or separately route non-`pp_lcnet_v3` backbone configs.
- [ ] Load PP-LCNetV3 and recognition-head weights with constant alias checks.
- [ ] Implement NCHW Conv2d, depthwise Conv2d, BatchNorm2d inference, adaptive avg pool, avg pool.
- [ ] Implement hardswish, hardsigmoid, silu, scalar affine, elementwise add/mul.
- [ ] Implement LayerNorm last-dim and Linear/GEMM paths for SVTR.
- [ ] Implement noncausal MHA with `head_dim=15`, no mask, no cache.
- [ ] Implement final classifier and fp32 softmax over class axis.
- [ ] Implement CTC-style postprocess using processor `character_list`.
- [ ] Add height-collapse and dynamic-width guards.
- [ ] Add Conv+BN folding rewrite.
- [ ] Add LearnableRepLayer reparameterization rewrite.
- [ ] Add guarded NHWC/channel-last conv-region experiment.
- [ ] Validate processor, backbone, SVTR block, end-to-end probabilities, and CTC postprocess parity.
- [ ] Benchmark preprocessing, backbone, SVTR attention, classifier/softmax, and output bandwidth.

