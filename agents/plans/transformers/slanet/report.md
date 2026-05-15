# DinoML Transformers Audit: `slanet`

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: PaddlePaddle/SLANet_plus_safetensors; PaddlePaddle/SLANet_safetensors
Config source: HF config.json for the two public checkpoints plus source defaults
Source files inspected:
  transformers/src/transformers/models/slanet/configuration_slanet.py
  transformers/src/transformers/models/slanet/modeling_slanet.py
  transformers/src/transformers/models/slanet/modular_slanet.py
  transformers/src/transformers/models/pp_lcnet/configuration_pp_lcnet.py
  transformers/src/transformers/models/pp_lcnet/modeling_pp_lcnet.py
  transformers/src/transformers/models/slanext/image_processing_slanext.py
  transformers/tests/models/slanet/test_modeling_slanet.py
Any missing files or assumptions:
  modeling_slanet.py/configuration_slanet.py are generated from modular_slanet.py.
  SLANet reuses SLANeXtImageProcessor through auto image processing.
  Only two public native SLANet checkpoints were found; no gated repos were needed.
```

Evidence snapshot: `agents/plans/transformers/slanet/evidence_snapshot.md`.

## 2. High-level architecture

Primary runtime target: image table-structure recognition. This is not a text Transformer and has no KV cache. It is a PP-LCNet CNN backbone, a CSP-PAN feature pyramid/fusion neck, and an autoregressive GRU attention head that emits HTML table-structure token probabilities.

```text
image preprocessing -> NCHW pixel_values -> PP-LCNet backbone feature maps
  -> CSP-PAN fusion -> [B, S, 96] image token sequence
  -> recurrent SLA head greedy loop -> [B, T, 50] structure probabilities
  -> processor post_process_table_recognition -> HTML token list + score
```

Independently stageable pieces:

- CPU/data pipeline: resize-to-max-side, pad to 488x488, rescale/normalize, channel-first packing.
- Vision graph: PP-LCNet and CSP-PAN are feed-forward CNN regions.
- Decoder graph: recurrent greedy loop with attention over fixed image tokens, one-hot previous token input, `GRUCell`, and early EOS break.
- Postprocessing: argmax/max score scan to convert class ids into HTML structure tokens.

## 3. Important config dimensions

| Field | Source default | Public checkpoint value | Operator significance |
|---|---:|---:|---|
| `post_conv_out_channels` | 96 | 96 | CSP-PAN output channel width and attention context width. |
| `out_channels` | 50 | 50 | Structure vocabulary and one-hot width. EOS is `49`. |
| `hidden_size` | 256 | 256 | GRU hidden width and generator hidden width. |
| `max_text_length` | 500 | 500 | Loop upper bound is `max_text_length + 1`. |
| `hidden_act` | `hardswish` | omitted, default `hardswish` | Conv/BN activation. |
| `csp_kernel_size` | 5 | omitted, default `5` | CSP bottleneck and downsample depthwise kernels. |
| `csp_num_blocks` | 1 | omitted, default `1` | One bottleneck per CSP block. |
| `backbone_config.model_type` | `pp_lcnet` | `pp_lcnet` | Nested backbone owner. |
| `backbone_config.scale` | 1 | 1 | PP-LCNet channel multiplier. |
| `backbone_config.out_features` | stage2-5 | stage2-5 | CSP-PAN consumes four image maps. |
| `preprocessor size/pad_size` | 512 source default | 488 | Checkpoint ABI fixes common input to `[B,3,488,488]`. |

Representative checkpoint sweep:

| Checkpoint | Config dims | Preprocessor | Safetensors notes | Variation |
|---|---|---|---|---|
| `PaddlePaddle/SLANet_plus_safetensors` | 96/50/256/500, PP-LCNet x1 | resize+pad 488 | 385 tensors, key shapes match config | Plus model name in `inference.yml`; TensorRT max batch example 1. |
| `PaddlePaddle/SLANet_safetensors` | same | same | same header tensor shapes | TensorRT max batch example 8. |
| Unit-test tiny config | 16/1/16/1, custom PP-LCNet channels | random `[B,3,488,488]` | source-only test config | Useful for one-block/parity scaffolding but not production. |

For `[B,3,488,488]` and default PP-LCNet x1, feature map sizes are inferred from source Conv2d padding/stride:

| Stage | Channels | Spatial |
|---|---:|---:|
| stem | 16 | 244 x 244 |
| stage1 | 32 | 244 x 244 |
| stage2 | 64 | 122 x 122 |
| stage3 | 128 | 61 x 61 |
| stage4 | 256 | 31 x 31 |
| stage5 | 512 | 16 x 16 |
| CSP-PAN output | 96 | 16 x 16 |
| SLA head input | 96 | sequence `S=256` |

## 3a. Family variation traps

- `SLANetConfig` composes a nested `backbone_config`; DinoML should admit only audited PP-LCNet backbones for first integration.
- The public configs omit `hidden_act`, `csp_kernel_size`, and `csp_num_blocks`; effective defaults come from source.
- `modeling_slanet.py` is generated. Future Transformers source edits should inspect `modular_slanet.py`.
- The source head returns structure probabilities only. Paddle `inference.yml` mentions table boxes, but the native HF model graph has no location head output.
- The decoder uses dynamic early stop based on EOS across the batch. Fixed-shape runtimes can first unroll all `max_text_length + 1` steps and defer early exit.
- `F.softmax(..., dtype=torch.float32).to(original dtype)` appears in attention and final structure probabilities; fp32 softmax accumulation matters.
- Source CNN graph is NCHW. NHWC/channel-last is an optimization candidate only for local conv/BN/activation regions, not the semantic graph ABI.
- Axis-sensitive NCHW ops: `BatchNorm2d` channel dim 1, `torch.cat(..., dim=1)` in CSP/PAN, `flatten(2).transpose(1,2)` to create `[B,S,C]`, and `F.interpolate(..., size=low_level_feature.shape[-2:])`.
- Processor parity trap: `inference.yml` lists BGR decode, while HF image processor defaults to RGB conversion. DinoML should scope parity to HF processor behavior unless a Paddle compatibility mode is explicitly added.

## 4. Operator coverage checklist

Tensor/layout ops:

- NCHW contiguous Conv2d layout, depthwise groups, BatchNorm2d inference affine, Hardswish/ReLU/Hardsigmoid primitives, nearest upsample, concat on channel dim, flatten from dim 2, transpose 1/2, stack along time, squeeze/unsqueeze, one-hot, zeros creation.

Neural network primitives:

- PP-LCNet Conv2d+BatchNorm2d+Hardswish blocks.
- Depthwise Conv2d: weights `[C,1,k,k]`, `groups=C`.
- Pointwise Conv2d: 1x1 dense convolution.
- CSP-PAN Conv2d/BN/Hardswish, bottleneck residual branch by concat not add.
- Linear projections: `96 -> 256` no bias, `256 -> 256` with bias, `256 -> 1` no bias, generator `256 -> 256 -> 50`, GRUCell input `146`, hidden `256`.

Attention/recurrent primitives:

- Additive attention over image tokens, not Transformer MHA.
- `softmax(dim=1)` over source sequence `[B,S,1]`.
- Batched matmul `[B,1,S] @ [B,S,96] -> [B,1,96]`.
- GRUCell gate math with `weight_ih [768,146]`, `weight_hh [768,256]`.

Preprocessing-coupled ops:

- Fixed-point bilinear resize in `SLANeXtImageProcessor._resize`, pad to 488x488, rescale by 1/255, normalize by ImageNet mean/std, channel-first tensor output.

Postprocessing ops:

- Argmax and max over class dim, EOS/ignored token filtering, mean confidence over emitted non-special tokens, HTML wrapper insertion.

## 5. Layer/block breakdown

PP-LCNet backbone, default x1:

```text
pixel_values: [B,3,H,W] NCHW
stem: Conv2d(3 -> 16, k=3, s=2, p=1, bias=False) -> BatchNorm2d -> Hardswish
stage1: DWConv(16,k=3,s=1) -> BN -> Hardswish -> PWConv(16 -> 32) -> BN -> Hardswish
stage2: DW/PW blocks 32 -> 64 with first stride 2, then 64 -> 64 stride 1
stage3: DW/PW blocks 64 -> 128 with first stride 2, then 128 -> 128 stride 1
stage4: DW/PW blocks 128 -> 256 stride 2, then five 256 -> 256 k=5 stride 1 blocks
stage5: DW/PW blocks 256 -> 512 stride 2 with SE in generic PP-LCNet, but SLANet generated source disables SE in its own post-CSP depthwise blocks only
```

CSP-PAN:

```text
input maps: [stage2, stage3, stage4, stage5] = channels [64,128,256,512]
project each: Conv1x1(Ci -> 96) -> BN -> Hardswish
top-down:
  upsample nearest high-level feature to low-level H,W
  concat dim=1 => [B,192,H,W]
  CSP block: Conv1x1(192->48), Conv1x1(192->48), bottleneck(s), concat 96, Conv1x1(96->96)
bottom-up:
  depthwise separable downsample 96 -> 96, stride 2, k=5
  concat with next pyramid feature => [B,192,H,W]
  CSP block => [B,96,H,W]
final: flatten(2).transpose(1,2) => [B,H*W,96]
```

SLA head per recurrent step:

```text
features_0 = zeros([B,256], fp32)
predicted_chars_0 = zeros([B], int64)  # BOS id is effectively 0
char_onehot = one_hot(predicted_chars, 50).float()       # [B,50]
batch_hidden_proj = Linear(96 -> 256, bias=False)(tokens) # [B,S,256]
prev_hidden_proj = Linear(256 -> 256)(features).unsqueeze(1)
scores = Linear(256 -> 1, bias=False)(tanh(batch_hidden_proj + prev_hidden_proj))
attn = softmax(scores, dim=1, fp32).transpose(1,2)       # [B,1,S]
context = matmul(attn, tokens).squeeze(1)                # [B,96]
gru_input = concat([context, char_onehot], dim=1)         # [B,146]
features = GRUCell(146, 256)(gru_input, features)
logits = Linear(256 -> 256) -> Linear(256 -> 50)
predicted_chars = argmax(logits, dim=1)
```

## 6. Attention requirements

No Transformer self-attention, causal attention, cross-attention, RoPE, ALiBi, FlashAttention, or KV cache is required for `slanet`.

Required attention is additive recurrent attention:

- Query source: previous GRU hidden state `[B,256]`.
- Key/value source: fixed image token sequence `[B,S,96]`, normally `S=256` for 488x488 public checkpoints.
- Projection widths: keys `96 -> 256`, query `256 -> 256`, score `256 -> 1`.
- Masking: no source mask in native code.
- Softmax axis: `dim=1`, over source sequence, with fp32 softmax.
- Cache opportunity: `input_to_hidden(batch_hidden)` can be precomputed once per image because source recomputes it every step but it is invariant across decode steps. This is a feature-cache, not KV cache.

## 7. Position encoding and custom math

Native `slanet` has no explicit position encoding in the SLANet-owned graph. Spatial position is implicit in convolutional feature maps and flatten order.

Critical custom math:

```python
attn_scores = score(tanh(input_to_hidden(tokens) + hidden_to_hidden(prev).unsqueeze(1)))
attn = softmax(attn_scores, dim=1, dtype=float32).to(attn_scores.dtype)
context = matmul(attn.transpose(1, 2), tokens).squeeze(1)
```

The HF image processor implements a fixed-point bilinear resize path using integer weights scaled by 2048 and a final right shift by 22. Treat exact resize parity as CPU/data-pipeline work unless DinoML owns preprocessing.

## 8. Preprocessing and input packing

HF processor runtime tensors:

- Input images are converted to tensors by the Transformers image backend.
- Public checkpoints set resize target and pad target to 488x488.
- `_resize` scales the longer side to `max(size.height, size.width)`, preserving aspect ratio.
- Padding then produces batchable `[B,3,488,488]` `pixel_values`.
- Rescale/normalize uses ImageNet mean/std.
- Model consumes NCHW `pixel_values`; no attention masks, token type ids, boxes, or packed metadata enter the model graph.

Postprocessing:

- `outputs.last_hidden_state` shape `[B,T,50]` is already softmax probabilities.
- `post_process_table_recognition` currently only uses `outputs[0:1]`, so multi-image batch postprocessing is not fully general.
- It ignores BOS/EOS ids, stops at EOS after position 0, converts ids to the fixed HTML-token dictionary, computes mean max probability over emitted tokens, and wraps with `<html><body><table>...</table></body></html>`.

## 9. Graph rewrite / lowering opportunities

### Rewrite: Conv2d + BatchNorm2d + Hardswish inference fusion

Source pattern: `Conv2d(bias=False) -> BatchNorm2d -> Hardswish`.

Replacement: fold BN affine into Conv2d weight/bias, then either fuse Hardswish epilogue or emit elementwise.

Preconditions:

- Model in eval mode with frozen BN running mean/var/weight/bias.
- NCHW source semantics preserved.
- Activation exactly `hardswish`.

Failure cases: training mode, unfrozen BN stats, alternate activation, unknown memory format.

### Rewrite: 1x1 Conv2d -> per-pixel Linear/GEMM

Preconditions:

- `kernel_size=1`, `stride=1`, `padding=0`, `groups=1`, `dilation=1`.
- Input is dense NCHW or a controlled NHWC-translated region.

Replacement:

```text
NCHW -> flatten spatial as M=B*H*W -> GEMM(Cin -> Cout) -> restore spatial
```

Weight transform: `conv.weight.reshape(out_channels, in_channels)`.

Layout constraints: for NHWC, rewrite channel axis from dim 1 to dim -1 and ensure downstream concat/BN/activation also translated.

### Rewrite: depthwise separable conv block

Source pattern: depthwise Conv2d groups `C`, BN, Hardswish, optional identity SE in SLANet post-CSP blocks, pointwise Conv2d, BN, Hardswish.

Replacement: keep depthwise as depthwise provider op; lower pointwise to GEMM if profitable.

Failure cases: generic PP-LCNet stage5 SE blocks from the nested backbone are real SE modules when `use_squeeze_excitation=True`; do not erase them.

### Rewrite: precompute additive attention keys

Source pattern per decode step recomputes `input_to_hidden(batch_hidden)`.

Replacement:

```text
key_proj = Linear(96 -> 256, bias=False)(tokens) once
loop step uses key_proj + hidden_to_hidden(prev).unsqueeze(1)
```

Preconditions: image tokens immutable across recurrent loop and dtype behavior preserved. Keep softmax fp32.

### Guarded layout rewrite: NCHW conv/CSP regions -> NHWC

Candidate region: PP-LCNet + CSP-PAN conv/BN/activation/concat/nearest-upsample until `flatten(2).transpose(1,2)`.

Required axis rewrites:

- `BatchNorm2d` channel axis `1 -> -1` or use channel-last BN equivalent.
- `torch.cat(..., dim=1) -> dim=-1`.
- `flatten(2).transpose(1,2)` changes to a direct `[B,H*W,C]` reshape if tensor is already NHWC.
- Conv weights require OIHW -> HWIO or provider-specific transform.

No-layout-translation guards:

- Processor/model ABI remains NCHW unless DinoML owns the full processor-to-backbone boundary.
- The SLA head expects `[B,S,C]`; only enter it after a proven equivalent flatten order.
- Dynamic odd spatial sizes must preserve source padding/stride output shape.

## 10. Kernel fusion candidates

Highest priority:

- Conv2d+BN+Hardswish for PP-LCNet and CSP-PAN.
- Depthwise Conv2d+BN+Hardswish and pointwise Conv2d+BN+Hardswish.
- SLA additive attention key precompute plus score/tanh/softmax/matmul loop.

Medium priority:

- 1x1 Conv2d to GEMM for channel projectors and CSP convs.
- Nearest upsample + concat + following 1x1 projections in PAN.
- GRUCell fused gate GEMMs and elementwise gate update.

Lower priority:

- Exact fixed-point processor resize on GPU.
- Final `stack -> softmax` if full unrolled output is required.
- HTML-token postprocessing acceleration; it is small and CPU-friendly.

## 11. Runtime staging plan

Stage 1: parse config and load weights for `SLANetForTableRecognition`; admit only `pp_lcnet` backbone x1 and `out_channels=50`.

Stage 2: implement/evaluate PP-LCNet backbone feature parity on `[1,3,488,488]`, with NCHW semantic graph.

Stage 3: add CSP-PAN parity and verify final image tokens `[B,256,96]` for public preprocessor shape.

Stage 4: implement SLA head with full fixed unroll of 501 steps; ignore early break initially but preserve probabilities.

Stage 5: add greedy early-exit controller and processor postprocess parity.

Stage 6: introduce optimized conv/BN/activation fusion and optional guarded NHWC local layout pass.

Stage 7: precompute additive-attention keys and fuse GRU/attention hot loop.

## 12. Parity and validation plan

- Unit test processor output shape and normalization for a known image: `[1,3,488,488]`.
- Single Conv2d+BN+Hardswish block parity in fp32, then fp16/bf16 if supported.
- PP-LCNet hidden-state parity for stem and stages 2-5.
- CSP-PAN parity: each projected and fused feature map, final `[B,S,96]`.
- SLA single-step parity from random tokens/hidden/prev char.
- Full recurrent parity with fixed unroll and with early stop enabled.
- End-to-end checkpoint parity against `PaddlePaddle/SLANet_plus_safetensors` expected table structure from Transformers slow test.
- Suggested tolerances: fp32 `rtol=1e-4, atol=1e-5`; fp16/bf16 compare final probabilities more loosely and keep softmax accumulation fp32.

## 13. Performance probes

- Processor resize+pad throughput separately from model graph.
- PP-LCNet backbone latency by stage and batch size.
- CSP-PAN latency for feature sizes from 488 input and dynamic-shape examples.
- SLA head tokens/sec for fixed unroll versus early stop.
- Impact of precomputing additive attention keys.
- NCHW provider path versus guarded NHWC/channel-last conv path.
- Batch-size sweep: 1 and 8, matching public `inference.yml` examples.
- Resolution sweep: `[32,32]`, `[64,448]`, `[488,488]`.
- Memory footprint for storing all `structure_preds_list` across up to 501 steps.

## 14. Skip/defer list

- Training, losses, gradients, and checkpointing.
- Paddle `TableBoxEncode`/bbox outputs; native HF `slanet` does not expose a location head.
- General arbitrary `backbone_config`; first admit PP-LCNet x1 public configs.
- Full GPU preprocessing and exact fixed-point resize unless end-to-end preprocessing parity becomes product-critical.
- Beam search or sampling; source uses greedy argmax only.
- Transformer attention/KV-cache work; not applicable.
- Generic dynamic shape support beyond documented public preprocessor/dynamic examples.

## 15. Final implementation checklist

- [ ] Parse `SLANetConfig` plus nested `PPLCNetConfig` defaults.
- [ ] Load `model.safetensors` weights and verify key tensor shapes.
- [ ] Implement/evaluate NCHW Conv2d, depthwise Conv2d, BatchNorm2d inference, Hardswish.
- [ ] Implement nearest upsample and channel concat for CSP-PAN.
- [ ] Implement `flatten(2).transpose(1,2)` token packing.
- [ ] Implement additive recurrent attention with fp32 softmax.
- [ ] Implement `GRUCell(146,256)` and generator `Linear(256->256->50)`.
- [ ] Add greedy argmax loop and EOS early-exit controller.
- [ ] Add processor/postprocessor parity harness for HF behavior.
- [ ] Add guarded Conv+BN+Hardswish fusion.
- [ ] Add optional guarded NHWC/channel-last rewrite for fully controlled conv regions.
- [ ] Add attention-key precompute rewrite.
- [ ] Benchmark processor, backbone, CSP-PAN, and SLA loop separately.
