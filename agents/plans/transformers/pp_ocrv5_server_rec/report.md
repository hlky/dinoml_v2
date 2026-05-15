# Transformers Audit: pp_ocrv5_server_rec

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: PaddlePaddle/PP-OCRv5_server_rec_safetensors
Config source: HF config.json and preprocessor_config.json, plus source defaults
Source files inspected:
  transformers/src/transformers/models/pp_ocrv5_server_rec/configuration_pp_ocrv5_server_rec.py
  transformers/src/transformers/models/pp_ocrv5_server_rec/image_processing_pp_ocrv5_server_rec.py
  transformers/src/transformers/models/pp_ocrv5_server_rec/modeling_pp_ocrv5_server_rec.py
  transformers/src/transformers/models/pp_ocrv5_server_rec/modular_pp_ocrv5_server_rec.py
  transformers/src/transformers/models/hgnet_v2/configuration_hgnet_v2.py
  transformers/src/transformers/models/hgnet_v2/modeling_hgnet_v2.py
Any missing files or assumptions:
  No code tests/imports were run. No safetensors weight metadata was inspected.
  The generated pp_ocrv5_server_rec files are derived from modular_pp_ocrv5_server_rec.py.
  HGNetV2 backbone is treated as a composed nested family whose operator surface is needed here.
```

HF artifacts checked:

| Artifact | Status | Notes |
| --- | --- | --- |
| `PaddlePaddle/PP-OCRv5_server_rec_safetensors/config.json` | accessible | Transformers-native model config. |
| `PaddlePaddle/PP-OCRv5_server_rec_safetensors/preprocessor_config.json` | accessible | Processor config and full CTC character list. |
| `PaddlePaddle/PP-OCRv5_server_rec/config.json` | accessible | Paddle-style deployment metadata, useful for OCR pipeline hints but not the in-library model config. |
| `PaddlePaddle/PP-OCRv5_server_rec/preprocessor_config.json` | missing | HF returned entry-not-found during audit. |

## 2. High-level architecture

This is an OCR text recognizer, not a language generator. The first useful DinoML target is
`PPOCRV5ServerRecForTextRecognition`: image preprocessing plus CNN/SVTR recognizer inference and
greedy CTC-style postprocess.

```text
CPU/image pipeline:
  input image(s) -> RGB conversion -> dynamic-width resize to height 48 -> rescale/normalize
  -> optional right/bottom pad to at least 48x320 -> NCHW pixel_values

GPU/model graph:
  pixel_values [B,3,48,W] -> HGNetV2 backbone -> feature_maps[-1]
  -> avg_pool2d(kernel=(3,2)) -> SVTR conv/attention encoder
  -> Linear(hidden_size -> vocab/classes) -> softmax(dim=2) -> probabilities [B,T,V]

CPU/postprocess:
  probabilities -> argmax over V -> remove adjacent duplicates -> drop blank id 0
  -> character_list lookup -> text + mean retained probability
```

Independently stageable pieces:

- Image preprocessing can be validated separately from the neural graph.
- HGNetV2 backbone can be audited as a CNN backbone, but pp_ocrv5_server_rec depends on its NCHW
  feature map ABI.
- SVTR head can be validated from `feature_maps[-1]` input onward.
- CTC-like greedy decode is a postprocess ABI, not a neural op requirement.

## 3. Important config dimensions

| Field | Value | Provenance | Runtime impact |
| --- | ---: | --- | --- |
| `model_type` | `pp_ocrv5_server_rec` | HF config | Model admission key. |
| `backbone_config.model_type` | `hgnet_v2` | HF config/source default | Nested CNN backbone. |
| `backbone_config.arch` | `L` | HF config | Informational in inspected source; source does not branch on `arch`. |
| `hidden_size` | 120 | HF config/source default | SVTR token width and classifier input. |
| `mlp_ratio` | 2.0 | HF config/source default | SVTR MLP hidden width 240. |
| `depth` | 2 | HF config/source default | Number of SVTR transformer-style blocks. |
| `head_out_channels` | 18385 | HF config | Vocabulary/class count, matches preprocessor character list length. |
| `conv_kernel_size` | `[1,3]` | HF config/source default | SVTR conv padding is `[0,1]`; width-local, height-preserving. |
| `num_attention_heads` | 8 | HF config/source default | Noncausal MHA heads. |
| `head_dim` | 15 | Inference from source | `120 / 8`; non power-of-two head dim. |
| `qkv_bias` | true | HF config/source default | Q and V bias present, K bias is a zero middle slice. |
| `attention_dropout` | 0.0 | HF config/source default | Dropout inactive for inference. |
| `layer_norm_eps` | 1e-6 | Source default | LayerNorm epsilon. |
| `processor.size` | 48x320 | HF preprocessor | Base resize height/width. |
| `processor.pad_size` | 48x320 | HF preprocessor | Pad target when resized width is below 320. |
| `processor.max_image_width` | 3200 | HF preprocessor | Dynamic maximum width. |
| `character_list` length | 18385 | HF preprocessor | Index table for decode; class id 0 is blank. |

Representative checkpoint sweep:

| Checkpoint/artifact | Native config? | Processor? | Operator variation found |
| --- | --- | --- | --- |
| `PaddlePaddle/PP-OCRv5_server_rec_safetensors` | yes | yes | Main target: HGNetV2 L backbone, SVTR depth 2, vocab 18385. |
| `PaddlePaddle/PP-OCRv5_server_rec` | no, Paddle deployment metadata | no separate preprocessor file | Same named product, contains CTC dictionary and TRT shape hints. Use only as pipeline provenance. |
| Source defaults without HF config | yes | source defaults only | Same OCR head defaults, with nested HGNetV2 defaults if no config provided. |

The audit found only one Transformers-native checkpoint config for this exact family. Broader
PP-OCR variants should be audited separately rather than inferred from this source.

## 3a. Family variation traps

- This is not an autoregressive text model. There is no tokenizer-controlled decode, KV cache,
  causal mask, or logits sampling ABI.
- The model returns probabilities after `softmax(dim=2)`, not raw logits. DinoML should either match
  that ABI or explicitly split a pre-softmax internal artifact from the public HF-compatible output.
- `head_dim=15` is unusual. Attention kernels must not assume head dimensions are multiples of 8,
  16, or 32 unless a fallback path exists.
- `qkv_bias=True` creates a packed bias with Q bias, zero K bias, and V bias. Packed-QKV rewrites must
  preserve the middle zero-bias behavior.
- The SVTR sequence is `height * width` tokens before it is reshaped back to NCHW. Layout rewrites
  must preserve row-major flatten order.
- `hidden_states.squeeze(2)` in the SVTR head requires height axis 2 to equal 1 after convs. This is a
  hard graph guard.
- Source semantic layout is NCHW for image/model tensors. NHWC/channel-last is only a guarded
  optimization region, mostly around local conv blocks where all consumers are controlled.
- Axis-sensitive ops: `avg_pool2d`, `flatten(2)`, `transpose(1,2)`, `view(B,H,W,C)`,
  `permute(0,3,1,2)`, `cat(..., dim=1)`, `squeeze(2)`, `softmax(dim=2)`,
  postprocess `max(dim=-1)`.
- Processor target width is batch-coupled: all images resize using the widest original image in the
  batch. Different batching can change padded width and sequence length.
- The non-safetensors repo's Paddle config reports TensorRT max shape `[8,3,48,3200]`; this is useful
  for staging buckets but is not a Transformers config field.
- `backbone_config.arch="L"` is present in HF config, but inspected HGNetV2 source does not consume
  `arch` to select channel tables. Effective channel fields come from source defaults unless the full
  config supplies explicit stage fields.

## 4. Operator coverage checklist

### Tensor/layout ops

- NCHW image tensors, contiguous dense tensors.
- Resize bilinear and optional pad in preprocessing.
- `F.pad` with `(0,1,0,1)` in HGNetV2 stem.
- `torch.cat` along channel axis.
- `flatten(2)`, `transpose(1,2)`, `view`, `permute`, `contiguous`, `squeeze(2)`.
- Shape guards for channel count 3, height 48 at model input, dynamic width up to 3200, and SVTR
  pre-output height exactly 1.

### Neural network primitives

- Conv2d with bias false, BatchNorm2d, activation (`relu` in HGNetV2 defaults, `silu` in OCR SVTR
  conv/MLP path).
- Depthwise Conv2d in HGNetV2 downsample and light blocks (`groups=in_channels` or
  `groups=out_channels`).
- MaxPool2d kernel 2, stride 1, `ceil_mode=True`.
- AvgPool2d kernel `(3,2)` in OCR wrapper.
- LayerNorm over token width 120.
- Linear: QKV `120 -> 360`, projection `120 -> 120`, MLP `120 -> 240 -> 120`, head
  `120 -> 18385`.
- Softmax: attention weights over source tokens and classifier probabilities over class axis.

### Attention primitives

- Noncausal dense self-attention over SVTR flattened spatial sequence.
- MHA: 8 heads, head dim 15, Q/K/V width 120.
- No attention mask is used by source forward; `attention_mask` is accepted but passed as `None`.
- Backend can be eager, SDPA, FlashAttention, or FlexAttention through Transformers attention
  interface, but parity must preserve the same noncausal math and no-mask behavior.

### Preprocessing-coupled ops

- Batch grouping by original image shape before resize/normalize.
- Target width computed from widest image in batch with height fixed to 48, minimum ratio equivalent
  to 320/48, maximum width 3200.
- ImageNet mean/std rescale and normalize.
- Pad to width 320 only when resized target width is below 320.

### Postprocess ops

- `argmax` and `max` over vocabulary axis.
- Adjacent duplicate suppression along time axis.
- Blank id 0 removal.
- Character table lookup.
- Mean confidence over retained positions. Empty selection needs an explicit DinoML policy because
  source would compute a mean over an empty tensor.

## 5. Layer/block breakdown

Top-level forward:

```text
pixel_values [B,3,48,W]
backbone_outputs = HGNetV2Backbone(pixel_values)
x = backbone_outputs.feature_maps[-1]                 # NCHW, stage4 feature map
x = avg_pool2d(x, kernel=(3,2))                        # still NCHW
x = PPOCRV5ServerRecHead(x)
return probabilities [B,T,18385]
```

SVTR OCR head:

```text
residual = x                                           # [B,C,H,W], C from HGNetV2 stage4
x = ConvBNAct(C -> C/8, kernel=(1,3), pad=(0,1))
x = ConvBNAct(C/8 -> 120, kernel=(1,1))
B, Ctok=120, H, W = x.shape
x = flatten spatial row-major -> transpose             # [B,H*W,120]
repeat depth=2:
  r = x
  x = LayerNorm(120, eps=1e-6)
  qkv = Linear(120 -> 360)
  q,k,v = reshape to [B,8,T,15]
  x = noncausal attention(q,k,v) -> Linear(120 -> 120)
  x = r + x
  r = x
  x = LayerNorm(120, eps=1e-6)
  x = Linear(120 -> 240) -> SiLU -> Linear(240 -> 120)
  x = r + x
x = LayerNorm(120, eps=1e-6)
x = view [B,H,W,120] -> permute [B,120,H,W]
x = ConvBNAct(120 -> C, kernel=(1,1))
x = cat([residual, x], dim=1)                          # [B,2C,H,W]
x = ConvBNAct(2C -> C/8, kernel=(1,3), pad=(0,1))
x = ConvBNAct(C/8 -> 120, kernel=(1,1))
x = squeeze height axis 2 -> transpose                 # [B,W,120]
x = Linear(120 -> 18385)
x = softmax(dim=2, dtype=float32).to(original_dtype)
```

HGNetV2 backbone summary:

```text
NCHW image -> stem convs with pad, maxpool, channel concat
-> four stages of downsample and HGNetV2BasicLayer blocks
-> each basic layer concatenates intermediate conv outputs along channel axis
-> 1x1 aggregation convs
-> configured out_features ["stage1","stage2","stage3","stage4"]
```

## 6. Attention requirements

- Required: dense noncausal self-attention in SVTR blocks.
- Not required: causal attention, cross-attention, KV cache, packed/varlen attention, sliding window,
  ALiBi, RoPE, relative position bias, generation masks.
- Shape:
  - Input `hidden_states`: `[B,T,120]`.
  - QKV projection output: `[B,T,360]`.
  - Packed reshape/permute: `[3,B,8,T,15]`, then Q/K/V each `[B,8,T,15]`.
  - Scores: `[B,8,T,T]`, scale `15 ** -0.5`.
  - Output before projection: `[B,T,120]`.
- Masking: source ignores `attention_mask` in the attention call and passes `attention_mask=None`.
- Backend: Transformers marks support for FlashAttention, SDPA, and FlexAttention. DinoML can use
  fused attention when head dim 15 and no mask are supported; otherwise a GEMM/softmax/GEMM fallback
  is required.

## 7. Position encoding and custom math

No RoPE, ALiBi, learned absolute position embedding, or relative bias is implemented in the inspected
source. Spatial order is encoded implicitly by convolutional features and row-major flattening.

Greedy CTC-like decode parity:

```python
def decode(prob, character_list):
    # prob: [T, V], already softmaxed by model
    conf, ids = prob.max(dim=-1)
    keep = torch.ones_like(ids, dtype=torch.bool)
    keep[1:] = ids[1:] != ids[:-1]
    keep &= ids != 0
    text = "".join(character_list[i] for i in ids[keep])
    score = conf[keep].mean()
    return text, score
```

DinoML should decide how to represent empty decoded text. The source does not special-case the empty
selection before `mean()`.

## 8. Preprocessing and input packing

Processor source contract:

- Converts to RGB by default.
- Uses bilinear resize through the Torchvision backend.
- Groups images by original shape for batched processing, but target size is computed once from the
  widest image shape in the batch.
- Target height is always 48.
- Target width is roughly `ceil(48 * original_width / original_height)`, clamped by a lower effective
  ratio of 320/48 and upper maximum 3200.
- If target width is below 320, pads to `[48,320]`.
- Emits `pixel_values`, expected by model as NCHW `[B,3,48,W]`.

Paddle deployment metadata from the non-safetensors repo advertises dynamic shape hints:

```text
min: [1,3,48,160]
opt: [1,3,48,320]
max: [8,3,48,3200]
```

Those are not read by the Transformers source, but they are good initial DinoML profiling buckets:
width 160, 320, and 3200, with batch 1 and 8.

## 9. Graph rewrite / lowering opportunities

### Rewrite: inference Conv2d + BatchNorm2d -> Conv2d with folded bias

Preconditions:

- Inference mode only.
- BatchNorm running mean/var, gamma, beta, eps are frozen and available.
- Conv bias is absent in source convs; folded bias must be generated.
- Preserve NCHW semantics or apply a complete channel-last transform for the whole local region.

Replacement:

```text
Conv2d(weight, bias=False) -> BatchNorm2d -> activation
=> Conv2d(folded_weight, folded_bias=True) -> activation
```

Failure cases:

- Training mode, unfrozen BN stats, missing running stats, or mixed precision policy that changes
  BN rounding beyond tolerance.

Parity sketch:

- Compare one HGNetV2 stem block and one SVTR conv block before/after folding in fp32 and fp16.

### Rewrite: 1x1 Conv2d -> channel GEMM

Preconditions:

- Kernel `(1,1)`, stride 1, padding 0, dilation 1, groups 1.
- Input is dense NCHW or a channel-last layout pass owns producer and consumer.
- Flatten spatial positions in a stable order.

Replacement:

```text
[B,C,H,W] -> reshape [B*H*W,C] -> GEMM(C -> Cout) -> reshape [B,Cout,H,W]
```

Weight transform:

```python
w2d = conv.weight.reshape(out_channels, in_channels)
```

Failure cases:

- Non-contiguous or partially translated layouts without explicit stride support.

### Rewrite: static local Conv2d `(1,3)` -> width-neighborhood kernel or im2col GEMM

Preconditions:

- Kernel `(1,3)`, stride 1, padding `(0,1)`, dilation 1, groups 1.
- Height axis is not mixed by the kernel.
- NCHW semantic axes are preserved, or an NHWC pass rewrites all axis references.

Replacement:

```text
width-local 3-tap channel convolution, optionally lowered by im2col over W
```

Failure cases:

- Any layout rewrite that swaps height/width without updating padding and squeeze guards.

### Rewrite: packed QKV Linear -> single GEMM plus split

Preconditions:

- `hidden_size % num_attention_heads == 0`.
- Packed weight order is all-Q, all-K, all-V in the last dimension after `Linear(120 -> 360)`.
- Bias order is Q bias, zero K bias, V bias when `qkv_bias=True`.

Replacement:

```text
GEMM [B*T,120] x [120,360] -> reshape [B,T,3,8,15] -> split Q,K,V
```

Failure cases:

- Assumptions that K has a learned bias or that `head_dim` is vector-aligned.

### Rewrite: public output softmax/postprocess split

Preconditions:

- HF-compatible public output must remain probabilities.
- Optimized recognizer-only deployment may accept a separate internal pre-softmax ABI if documented.

Replacement:

```text
Internal: classifier scores [B,T,V]
Public HF parity: softmax(scores, dim=2) -> probabilities
Postprocess: argmax(probabilities)
```

Failure cases:

- Using logits in postprocess score without changing confidence semantics.

### Layout opportunity: guarded NCHW <-> NHWC conv islands

Candidate regions:

- HGNetV2 stem/stages and SVTR conv blocks can be internally channel-last if all conv, BN, activation,
  cat, pool, and pad axes are rewritten together.

No-layout-translation guard boundaries:

- Model input ABI `pixel_values` is NCHW.
- HGNetV2 output `feature_maps[-1]` is consumed as NCHW.
- `avg_pool2d`, `flatten(2)`, `view(B,H,W,C)`, `permute(0,3,1,2)`, `cat(dim=1)`, and `squeeze(2)`
  are axis-sensitive. Do not translate across these boundaries without a full proof of axis rewrites.

## 10. Kernel fusion candidates

Highest priority:

- Conv2d+BatchNorm+activation fusion for HGNetV2 and SVTR convs. This dominates the CNN backbone and
  removes many small ops.
- Depthwise Conv2d support for HGNetV2 downsample/light blocks.
- QKV GEMM + reshape/split + attention fallback for head dim 15. This is required for SVTR parity.
- Classifier `Linear(120 -> 18385)` plus softmax over `V`. The vocabulary axis is large.

Medium priority:

- 1x1 Conv2d as GEMM for aggregation and SVTR channel projection blocks.
- LayerNorm + Linear fusion in SVTR blocks.
- Softmax/argmax postprocess separation so deployments can optionally return text records without
  materializing probabilities on host.
- Shape-specialized width buckets: 160, 320, 640, 1280, 3200.

Lower priority:

- Fully fused CTC greedy decode kernel. Useful for production output latency, but can begin as CPU
  postprocess.
- NHWC/channel-last conv islands. Good for performance after parity, but unsafe as an initial semantic
  translation unless guard boundaries are explicit.

## 11. Runtime staging plan

1. Parse config and preprocessor metadata; reject configs without `model_type=pp_ocrv5_server_rec`
   and nested `backbone_config.model_type=hgnet_v2`.
2. Load weights and build an inference-only NCHW graph with folded Conv+BN where possible.
3. Validate processor output shape policy on fixed images, especially dynamic width and pad behavior.
4. Implement HGNetV2 backbone parity for the exact `stage4` feature map consumed here.
5. Implement OCR head from `feature_maps[-1]`: avg pool, SVTR conv blocks, two attention blocks,
   classifier, and softmax.
6. Add greedy CTC-like postprocess as a host-side first pass.
7. Add optimized kernels/fusions: ConvBNAct, depthwise conv, 1x1 GEMM, packed-QKV attention, large
   vocabulary classifier/softmax.
8. Add guarded NHWC/channel-last islands only after NCHW parity is stable.

Stub initially:

- Training-only labels and losses.
- Hidden-state capture unless needed for parity debugging.
- GPU postprocess; CPU decode is enough for first end-to-end output parity.

## 12. Parity and validation plan

- Processor parity:
  - Single image at narrow, default, and very wide aspect ratios.
  - Batch with mixed original widths to confirm widest-image target width behavior.
  - Confirm NCHW output, ImageNet normalization, and pad-to-320 rule.
- Operator parity:
  - ConvBNAct folding for fp32 with tight tolerance, then fp16 with relaxed tolerance.
  - HGNetV2 stem including `F.pad`, `MaxPool2d(ceil_mode=True)`, and channel concat.
  - Depthwise conv stage blocks.
  - SVTR block with random `[B,T,120]`, head dim 15.
  - Classifier+softmax over 18385 classes.
- Subgraph parity:
  - Backbone `feature_maps[-1]` against Transformers.
  - OCR head from saved/passed backbone feature map.
  - Full model probabilities `[B,T,18385]`.
- Postprocess parity:
  - Duplicate class sequence, blank id 0, all-blank/empty output, and known mixed Unicode character
    IDs from the HF `character_list`.
- Suggested tolerances:
  - fp32: `rtol=1e-4`, `atol=1e-5` for subgraphs; softmax probability comparisons may need
    `rtol=1e-4`, `atol=1e-6`.
  - fp16: `rtol=1e-2`, `atol=1e-3`, with logits/probabilities checked carefully around argmax ties.

## 13. Performance probes

- Image preprocessing throughput by image aspect ratio and batch size.
- Backbone-only throughput for widths 160, 320, 640, 1280, 3200.
- OCR head-only throughput from `feature_maps[-1]`.
- Attention backend comparison for sequence lengths induced by width buckets.
- Classifier+softmax time and memory bandwidth for `[B,T,18385]`.
- End-to-end latency and throughput for batch 1 and batch 8.
- CPU postprocess throughput and host transfer cost if probabilities are copied back.
- Conv layout comparison: NCHW baseline versus guarded channel-last conv islands.

## 14. Skip/defer list

- Training, labels, loss, gradient checkpointing.
- Autoregressive generation, beam search, sampling, KV cache.
- General tokenizer integration. The recognizer uses an image processor character list, not a text
  tokenizer.
- Broad PP-OCR family variants outside this exact `pp_ocrv5_server_rec` source.
- GPU CTC decode until model parity and host postprocess are stable.
- Unproven global NHWC translation across SVTR flatten/view/squeeze boundaries.
- Dynamic batching policy beyond width buckets and max width 3200.

## 15. Final implementation checklist

- [ ] Parse `PPOCRV5ServerRecConfig` and nested `HGNetV2Config`.
- [ ] Parse `PPOCRV5ServerRecImageProcessor` metadata and `character_list`.
- [ ] Admit NCHW `pixel_values [B,3,48,W]` with width guard `W <= 3200`.
- [ ] Implement or compose HGNetV2 backbone operators needed for `feature_maps[-1]`.
- [ ] Implement `avg_pool2d(kernel=(3,2))` after the backbone.
- [ ] Implement SVTR conv blocks with Conv2d, BatchNorm2d, SiLU, concat, squeeze height guard.
- [ ] Implement SVTR noncausal self-attention with head dim 15 and no mask.
- [ ] Implement packed QKV bias rule: Q bias, zero K bias, V bias.
- [ ] Implement MLP `120 -> 240 -> 120` with SiLU.
- [ ] Implement classifier `120 -> 18385` and public `softmax(dim=2, dtype=float32)`.
- [ ] Implement CTC-style greedy postprocess: argmax, duplicate removal, blank id 0 removal,
      character lookup, confidence mean.
- [ ] Add parity tests for processor, backbone, OCR head, full probabilities, and postprocess records.
- [ ] Add width-bucket performance probes for 160, 320, 640, 1280, and 3200.
- [ ] Add guarded rewrite tests for ConvBN folding, 1x1 Conv-to-GEMM, packed QKV, and no-layout
      translation boundaries.
