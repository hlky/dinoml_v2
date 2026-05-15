# Transformers Audit: slanext

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: PaddlePaddle/SLANeXt_wired_safetensors, PaddlePaddle/SLANeXt_wireless_safetensors
Config source: HF config.json and preprocessor_config.json for the two native safetensors checkpoints
Source files inspected:
- transformers/src/transformers/models/slanext/modular_slanext.py
- transformers/src/transformers/models/slanext/configuration_slanext.py
- transformers/src/transformers/models/slanext/modeling_slanext.py
- transformers/src/transformers/models/slanext/image_processing_slanext.py
- transformers/tests/models/slanext/test_modeling_slanext.py
- transformers/tests/models/slanext/test_image_processing_slanext.py
- transformers/src/transformers/models/got_ocr2/modeling_got_ocr2.py, as modular inheritance source for the vision encoder pattern
Any missing files or assumptions: only two native model_type=slanext safetensors checkpoints were found. Older PaddlePaddle/SLANeXt_wired and PaddlePaddle/SLANeXt_wireless repos are PaddleOCR inference artifacts, not native Transformers safetensors checkpoints.
```

`configuration_slanext.py`, `modeling_slanext.py`, and `image_processing_slanext.py` are generated from `modular_slanext.py`; future Transformers source edits should be made in the modular file. This report uses the generated files as the exact in-library runtime surface because they contain the expanded classes imported by AutoModel/AutoImageProcessor.

Evidence snapshot: `agents/plans/transformers/slanext/evidence/config_sweep.json`.

## 2. High-level architecture

Primary runtime target: image table-structure recognition through `SLANeXtForTableRecognition`. This is not a text LLM and has no KV-cache decode path. It is a vision encoder plus an autoregressive GRU-style table-token head.

```text
image decode/resize/pad/normalize -> pixel_values [B,3,512,512]
-> patch Conv2d -> NHWC vision transformer blocks with window/global attention
-> NCHW neck convs -> stride-2 post conv -> feature sequence [B,256,512]
-> recurrent attention-GRU token head -> table token probabilities [B,T,50]
-> table-token postprocessing -> HTML-like structure tokens
```

Stage decomposition:

- CPU/data pipeline: image decode, optional RGB conversion, custom aspect-preserving bilinear resize, rescale/normalize, pad to 512 x 512, CHW tensor emission.
- Vision encoder: independently validatable from `pixel_values` to feature map `[B,256,32,32]`.
- Backbone post-conv/flatten: `[B,256,32,32] -> [B,512,16,16] -> [B,256,512]`.
- SLA head: recurrent loop up to `max_text_length + 1` steps, early-stops when every batch item has emitted EOS.
- Postprocessing: argmax/max-score decode, ignore SOS/EOS, wrap predicted tokens in `html/body/table`.

## 3. Important config dimensions

Default/native checkpoint dimensions:

| Field | Value | Source |
|---|---:|---|
| `vision_config.image_size` | 512 | HF config |
| `vision_config.num_channels` | 3 | HF config |
| `vision_config.patch_size` | 16 | HF config |
| Patch grid | 32 x 32 | inferred from config |
| `vision_config.hidden_size` | 768 | HF config |
| `vision_config.num_hidden_layers` | 12 | HF config |
| `vision_config.num_attention_heads` | 12 | HF config |
| Vision `head_dim` | 64 | source computes `hidden_size // heads` |
| `vision_config.mlp_dim` | 3072 | HF config |
| `vision_config.output_channels` | 256 | HF config |
| `vision_config.window_size` | 14 | HF config |
| `vision_config.global_attn_indexes` | [2, 5, 8, 11] | HF config |
| `vision_config.qkv_bias` | true | HF config |
| `vision_config.use_abs_pos` | true | HF config |
| `vision_config.use_rel_pos` | true | HF config |
| `post_conv_in_channels` | 256 | HF config |
| `post_conv_out_channels` | 512 | HF config |
| Head `hidden_size` | 512 | HF config |
| `out_channels` / vocabulary | 50 | HF config |
| `max_text_length` | 500 | HF config |
| `loc_reg_num` | 8 | HF config, ignored by native modeling source |
| Safetensors parameters | 91,191,610 total; mostly F16, 36,864 F32 | HF repo metadata |

Representative checkpoint sweep:

| Checkpoint | Native Transformers? | Operator-significant dimensions | Notes |
|---|---|---|---|
| `PaddlePaddle/SLANeXt_wired_safetensors` | yes | 512 image, 12 layers, 12 heads, 14 window, 4 global layers, vocab 50 | Integration test target in Transformers. |
| `PaddlePaddle/SLANeXt_wireless_safetensors` | yes | same as wired | No config/preprocessor delta observed. Different weights/task flavor only. |
| `PaddlePaddle/SLANeXt_wired` | no, PaddleOCR artifact | inference.yml declares NCHW `[1,3,512,512]` TRT shapes and same preprocess/postprocess concepts | Contains `inference.pdiparams`, not native safetensors. Route through PaddleOCR audit or conversion, not this native source report. |
| `PaddlePaddle/SLANeXt_wireless` | no, PaddleOCR artifact | same as wired PaddleOCR artifact | Not a native Transformers checkpoint. |
| Local tiny test config | synthetic only | hidden 1, layers 1, heads 1, mlp 4, out_channels 1, max_text_length 1 | Useful for unit shape smoke, not representative performance. |

## 3a. Family variation traps

- `hidden_size` must be divisible by `num_attention_heads`; source silently uses integer floor for `head_dim`, so DinoML should reject non-divisible configs.
- Window attention pads the 32 x 32 patch grid to 42 x 42 for `window_size=14`, producing 9 windows per image; global layers do dense 1024-token attention.
- `global_attn_indexes` changes attention tensor sizes and whether `rel_pos_h/w` length is 27 or 63.
- The vision body is NHWC between patch embedding and neck, then switches back to NCHW for conv neck/post-conv. Layout translation must be guarded and region-local.
- `loc_reg_num` appears in checkpoint config and PaddleOCR YAML, but the inspected native modeling source does not read it and has no location/bbox regression head. Treat as ignored for this source basis.
- The processor warns that arbitrary `resample` is unsupported and uses its own fixed-point bilinear resize. End-to-end parity depends on this, but first GPU graph parity can consume already prepared `pixel_values`.
- The head loop is data-dependent because EOS can stop before `max_text_length + 1`; static-runtime staging may need fixed upper-bound output plus reported length or an initial fixed 501-step path.
- `_keep_in_fp32_modules_strict` keeps `structure_attention_cell` and `structure_generator` in fp32 on reload for reduced-precision models.
- The postprocessor can fail on an empty `score_list` if a sequence only emits ignored tokens; not central to neural graph lowering, but an end-to-end parity edge case.

## 4. Operator coverage checklist

Tensor/layout ops:

- NCHW input validation `[B,3,512,512]`.
- `Conv2d` patch embedding, kernel/stride 16, `3 -> 768`, bias true, output `[B,768,32,32]`.
- `permute(0,2,3,1)` to NHWC `[B,32,32,768]`.
- Window pad NHWC with channel-last pad tuple `(0,0,0,pad_w,0,pad_h)`.
- `reshape`, `permute`, `contiguous`, slicing for window partition/unpartition.
- `flatten(2)` and `transpose(1,2)` after post-conv.
- `stack` over dynamic recurrent step list; `cat` `[context, one_hot]` along feature dim.
- `argmax`, equality, `any(-1)`, `all()` for early stop.

Neural primitives:

- `Linear(768 -> 2304, bias=qkv_bias)` for packed QKV.
- `Linear(768 -> 768, bias=true)` attention output.
- `LayerNorm(768, eps=1e-6)` on NHWC.
- MLP block: `Linear(768 -> 3072)`, GELU, `Linear(3072 -> 768)`.
- Neck: `Conv2d(768 -> 256, 1x1, bias=false)`, channel-first LayerNorm over C, `Conv2d(256 -> 256, 3x3, pad=1, bias=false)`, channel-first LayerNorm.
- Post conv: `Conv2d(256 -> 512, 3x3, stride=2, pad=1, bias=false)`.
- Attention-GRU score path: `Linear(512 -> 512, bias=false)`, `Linear(512 -> 512, bias=true)`, `tanh`, `Linear(512 -> 1, bias=false)`.
- GRUCell input width `512 + 50 = 562`, hidden width 512; requires sigmoid/tanh gates and two bias vectors.
- Structure generator: `Linear(512 -> 512)`, `Linear(512 -> 50)`, identity activation.

Attention primitives:

- Dense noncausal 2D self-attention over local windows `[B*9,14,14,768]` and global feature maps `[B,32,32,768]`.
- Packed QKV split order: source linear output is reshaped as `[B, H*W, 3, heads, head_dim]`, then unbound in Q, K, V order.
- Attention matmul shapes: window `[B*9*12,196,64] @ [B*9*12,64,196]`; global `[B*12,1024,64] @ [B*12,64,1024]`.
- Softmax over key dimension in fp32, cast back to query dtype.
- Add decomposed 2D relative position bias before softmax.
- Additive recurrent attention in the SLA head over 256 visual positions: scores `[B,256,1]`, softmax over dim 1, context matmul `[B,1,256] @ [B,256,512]`.

Position/custom math:

- Absolute position table `[1,32,32,768]` added in NHWC.
- Relative position tables per attention layer: window layers `[27,64]` each axis, global layers `[63,64]` each axis for default config.
- Relative position interpolation path exists for non-matching q/k sizes via 1D linear interpolate.

Preprocessing-coupled ops:

- Aspect-preserving resize to make the longer side 512, using source's fixed-point bilinear math.
- Rescale by `1/255`, normalize by ImageNet mean/std, pad to `[3,512,512]`.
- Decoder vocabulary construction and postprocess token wrapping.

No generation/cache ops:

- No Transformer KV cache, RoPE, ALiBi, causal mask, beam search, language model logits, or tokenizer-driven text generation.

## 5. Layer/block breakdown

Patch and position:

```text
pixel_values: [B,3,512,512] NCHW
x = Conv2d(3 -> 768, kernel=16, stride=16, bias=true) -> [B,768,32,32]
x = permute NCHW -> NHWC -> [B,32,32,768]
x = x + pos_embed[1,32,32,768] if use_abs_pos
```

Vision block, repeated 12 times:

```text
residual = x  # [B,H,W,768]
y = LayerNorm(768)(x)
if window layer:
    y = pad NHWC to [B,42,42,768]
    y = window_partition -> [B*9,14,14,768]
y, attn = 2D self-attention(y)
if window layer:
    y = window_unpartition -> [B,32,32,768]
x = residual + y
y = LayerNorm(768)(x)
y = Linear(768 -> 3072) -> GELU -> Linear(3072 -> 768)
x = x + y
```

Vision attention:

```text
qkv = Linear(768 -> 2304, bias=true)(x)
q,k,v = reshape/split to [Bwin*heads, tokens, 64]
scores = (q * 0.125) @ transpose(k)
scores += decomposed_rel_pos(q, rel_h, rel_w)
probs = softmax(scores, dim=-1, dtype=float32).to(q.dtype)
out = probs @ v
out = reshape/permutation back to [Bwin,H,W,768]
out = Linear(768 -> 768)(out)
```

Backbone tail:

```text
x = permute NHWC -> NCHW
x = Conv2d(768 -> 256, 1x1, bias=false)
x = LayerNorm over C through NCHW->NHWC->NCHW
x = Conv2d(256 -> 256, 3x3, padding=1, bias=false)
x = LayerNorm over C through NCHW->NHWC->NCHW
x = Conv2d(256 -> 512, 3x3, stride=2, padding=1, bias=false)
x = flatten spatial -> [B,512,256]
x = transpose -> [B,256,512]
```

SLA recurrent head:

```text
features = zeros([B,512], fp32)
predicted_chars = zeros([B], int64)  # SOS id is also 0
for step in range(max_text_length + 1):
    one_hot = one_hot(predicted_chars, 50).float()  # [B,50]
    token_proj = Linear(512 -> 512, bias=false)(hidden_states.float())  # [B,256,512]
    state_proj = Linear(512 -> 512, bias=true)(features).unsqueeze(1)
    scores = Linear(512 -> 1, bias=false)(tanh(token_proj + state_proj))
    weights = softmax(scores, dim=1, dtype=float32).to(scores.dtype)
    context = matmul(transpose(weights,1,2), hidden_states.float()).squeeze(1)
    features = GRUCell([context, one_hot], features)
    logits = Linear(512 -> 512) -> Linear(512 -> 50)
    predicted_chars = argmax(logits, dim=1)
    stop if every batch item has emitted EOS id 49
probs = softmax(stack(logits_per_step, dim=1), dim=-1, dtype=float32)
```

## 6. Attention requirements

Vision attention:

- Noncausal self-attention only.
- MHA, not GQA/MQA: 12 query heads, 12 key/value heads, head_dim 64.
- Rectangular Q/K lengths are not used for default forward; relative-position helper supports q/k size mismatch through interpolation.
- No attention mask in the vision encoder.
- No packed varlen metadata. Window partition changes effective batch from B to `B * num_windows`.
- Window layers: 8 layers with 196-token dense attention per window for default global indexes.
- Global layers: 4 layers with 1024-token dense attention per image.
- Relative position bias is query-dependent via einsum, not a static pairwise bias table.
- Dropout call remains in source, but inference uses `training=False`, so dropout is identity.
- FlashAttention/SDPA compatibility: possible for the pure QK/softmax/V part, but relative position bias is dynamic and must be added before softmax. Window/global batching and bias materialization need a dedicated wrapper.

SLA head attention:

- Additive attention over fixed visual sequence length 256.
- Query source is recurrent hidden state `[B,512]`; K/V source is backbone sequence `[B,256,512]`.
- No KV cache. `input_to_hidden(batch_hidden)` can be cached across recurrent steps because `batch_hidden` is constant after the backbone for one image.
- Output context is `[B,512]` per step.

## 7. Position encoding and custom math

Absolute position encoding is a learned NHWC tensor added after patch embedding. Relative position is decomposed into height and width terms and added to attention logits.

Concise source-equivalent custom math:

```python
def slanext_rel_pos(q_size, k_size, rel_pos):
    max_rel_dist = 2 * max(q_size, k_size) - 1
    table = interpolate_1d(rel_pos, length=max_rel_dist)  # [max_rel_dist, head_dim]
    q = arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k = arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    idx = (q - k) + (k_size - 1) * max(q_size / k_size, 1.0)
    return table[idx.long()]  # [q_size, k_size, head_dim]

def slanext_decomposed_rel_pos(query, rel_h, rel_w, H, W):
    q = query.reshape(batch_heads, H, W, head_dim)
    rh = slanext_rel_pos(H, H, rel_h)
    rw = slanext_rel_pos(W, W, rel_w)
    bias_h = einsum("bhwc,hkc->bhwk", q, rh)
    bias_w = einsum("bhwc,wkc->bhwk", q, rw)
    return bias_h[:, :, :, :, None] + bias_w[:, :, :, None, :]
```

For default fixed 32 x 32 and 14 x 14 regions, relative index maps can be precomputed. The einsum result still depends on runtime query values.

## 8. Preprocessing and input packing

Processor input can be PIL, NumPy, or Torch image. Runtime tensor contract after processing:

```text
pixel_values: [B,3,512,512], channel-first, float after rescale/normalize
mean/std: ImageNet default
resize: longer side -> 512, aspect-preserving
pad: to 512 x 512
```

The source `_resize` operates on `[B,C,H,W]`, flattens to `[B*C,H,W]`, computes target integer coordinate tables, gathers four neighbors, applies fixed-point bilinear weights scaled by 2048, rounds through `(interp + (1 << 21)) >> 22`, clamps to uint8, then restores dtype. This is not ordinary torchvision bilinear parity.

PaddleOCR YAML for legacy repos includes BGR decode, HWC normalize, `ToCHWImage`, label/box encoders, and bbox side inputs. Native Transformers `SLANeXtForTableRecognition.forward` consumes only `pixel_values`; no `bboxes`, masks, or target labels are read by the native inference graph.

Postprocessing:

- Takes probabilities `[B,T,50]`.
- Uses argmax token ids and max probabilities.
- Ignores SOS id 0 and EOS id 49.
- Stops per sequence at EOS after position 0.
- Wraps structure tokens with `<html>`, `<body>`, `<table>`, and closing tags.
- No NMS, box conversion, OCR, or cell-coordinate output in native source.

## 9. Graph rewrite / lowering opportunities

### Rewrite: non-overlap patch Conv2d -> Linear

Source pattern: `Conv2d(3 -> 768, kernel=16, stride=16, padding=0, groups=1)` followed by NCHW->NHWC permute.

Replacement:

```text
NCHW WindowFlatten over 16x16 patches -> MatMul([768, 3*16*16].T) -> BiasAdd -> NHWC [B,32,32,768]
```

Preconditions:

- `kernel_size == stride == patch_size`.
- `padding == 0`, `dilation == 1`, `groups == 1`.
- input H/W exactly match config image size and are divisible by patch size.
- flatten order matches PyTorch Conv2d NCHW kernel layout `[out,in,kh,kw]`.

Failure cases: non-divisible image size, dynamic shape, non-unit dilation, grouped conv, or if a layout pass changes patch flatten order.

Parity test sketch: compare patch conv output before/after rewrite on random `[1,3,512,512]` fp32 and fp16 inputs with fixed weights.

### Rewrite: vision packed QKV linear split

Source pattern:

```text
Linear(768 -> 2304) -> reshape [B,T,3,H,D] -> permute -> unbind Q,K,V
```

Replacement: one packed GEMM with views into Q/K/V row blocks, or three GEMMs after splitting weight rows `[Q;K;V]`.

Preconditions:

- output width is exactly `3 * hidden_size`.
- split order remains Q, K, V.
- `hidden_size == num_heads * head_dim`.
- bias split follows the same row order when `qkv_bias=true`.

### Rewrite: window partition/unpartition as metadata plus local copy

Source pattern: NHWC pad, reshape, permute, contiguous, reshape to `[B*nwin,14,14,C]`, then inverse.

Replacement: guarded window-layout view/copy primitive that emits window-major contiguous tiles and inverse scatter/copy.

Preconditions:

- local region owns all consumers between partition and unpartition.
- H/W, pad_h/pad_w, and `window_size` are known or guarded.
- no external consumer observes padded tokens.

Layout constraints: this is an NHWC region. Do not rewrite axes as NCHW unless the entire block, relative-position axes, LayerNorm axes, and attention reshape consumers are rewritten together.

### Rewrite: cache recurrent `input_to_hidden(batch_hidden)`

Source recomputes `Linear(512 -> 512, bias=false)` over `[B,256,512]` every decoding step.

Replacement: compute once after backbone and reuse `[B,256,512]` in all SLA steps.

Preconditions:

- inference only.
- `batch_hidden` immutable during head loop.
- dtype/cast behavior preserved: source passes `hidden_states.float()` into the cell.

Failure cases: training, gradient capture, or if future source adds step-dependent visual features.

### Rewrite: fixed-step recurrent unroll with dynamic stop metadata

Source pattern: Python loop with early break after EOS for all batch items.

Replacement: for first integration, unroll exactly `max_text_length + 1 = 501` and report/consume EOS length in postprocess, or implement a bounded while loop with a scalar all-done predicate.

Preconditions:

- output buffer can represent either fixed `[B,501,50]` or dynamic reported T.
- argmax/equality/all semantics match PyTorch.

### Rewrite: guarded NCHW/NHWC conv-neck layout fusion

Source pattern:

```text
NHWC -> permute NCHW -> Conv1x1 -> LN over C via NCHW->NHWC->NCHW -> Conv3x3 -> LN -> Conv stride2 -> flatten
```

Candidate replacement: keep channel-last conv/layernorm kernels through neck and post-conv, then flatten from NHWC.

Required axis rewrites:

- Conv weight layout remains `[out,in,kh,kw]` unless transformed for NHWC provider.
- LayerNorm channel axis changes from source permuted NHWC last dim to optimized channel-last last dim; preserve `normalized_shape=C`.
- Flatten after post-conv must map spatial order identically: source NCHW flatten makes sequence index `h * W + w` with channel as feature.

Failure cases: any external consumer expects source NCHW feature map, provider lacks NHWC conv parity, or dynamic strides are not represented in metadata.

## 10. Kernel fusion candidates

Highest priority:

- Patch Conv2d -> GEMM/implicit GEMM, because it is the first large image op and a clean non-overlap rewrite.
- LayerNorm over NHWC `[*,768]` and channel-first LayerNorm over `[B,C,H,W]` via NHWC conversion, because every vision block and neck uses it.
- Packed QKV projection + reshape/split for `768 -> 2304`.
- Dense attention with additive relative bias for 196-token windows and 1024-token global layers.
- GRUCell(562,512) and additive attention step, because the head can run up to 501 iterations.

Medium priority:

- Window partition/unpartition copy fusion.
- GELU MLP block `Linear -> GELU -> Linear`.
- Conv neck/post-conv library path for 1x1, 3x3 same-pad, and 3x3 stride-2 convs.
- Recurrent precompute of `input_to_hidden(batch_hidden)`.

Lower priority:

- Processor fixed-point resize on GPU. Useful for end-to-end throughput, but CPU/data-pipeline preprocessing can be accepted initially.
- Output softmax `[B,T,50]`; small compared with encoder/head linear work.
- Postprocess token decode in runtime; can remain host-side.

## 11. Runtime staging plan

Stage 1: config/weights admission and processor-free graph stub. Load native safetensors, reject non-native PaddleOCR repos, and require fixed `[B,3,512,512]` prepared inputs.

Stage 2: vision patch/neck conv parity. Implement patch embedding, absolute pos add, neck convs, post conv, and flatten with random/tiny configs before full attention.

Stage 3: one vision block parity. Cover LayerNorm, packed QKV, relative-position math, local/global attention, MLP, and window partition for one layer.

Stage 4: full vision encoder parity. Validate hidden states and final `[B,256,32,32]` neck output, then `[B,256,512]` backbone output.

Stage 5: SLA head fixed unroll. Start with fixed 501 steps and no early break in compiled graph; host postprocess can stop at EOS.

Stage 6: add dynamic early-stop/reporting. Add bounded loop or output length reporting once recurrent parity is stable.

Stage 7: optimize layout and kernels. Add guarded NHWC conv/LN attention fusions, QKV packed lowering, and recurrent projection caching.

Stage 8: end-to-end processor/postprocess parity. Reproduce or compose the source fixed-point resize and table decode.

## 12. Parity and validation plan

- Processor parity: compare `_resize` against Transformers for selected aspect ratios, including tall, wide, square, and edge dimensions near 512.
- Patch embedding parity: random weights, `[1,3,512,512]`, fp32/fp16.
- Relative-position parity: default window 14 and global 32, plus a q/k mismatch case to exercise interpolate.
- Single vision attention parity: compare logits before softmax, softmax probabilities, and projected output.
- Window partition/unpartition parity: default 32 -> 42 pad and non-padded synthetic 28 x 28 case.
- One full vision layer parity for window and global layer variants.
- Full encoder parity against `SLANeXtVisionEncoder`.
- Backbone sequence parity `[B,256,512]`.
- SLA cell parity for one step, then N steps with teacher-forced predicted char inputs if a harness is added.
- End-to-end table-recognition parity on the Transformers integration image and both native safetensors checkpoints.

Recommended tolerances:

- fp32: `atol=1e-5`, `rtol=1e-4` for most ops; tighter for pure layout.
- fp16/bf16: `atol=5e-3`, `rtol=5e-2` for full model, with fp32 softmax/GRU head behavior preserved.
- Validate token sequence exact match separately from probability closeness.

No DinoML tests were run for this audit by request.

## 13. Performance probes

- Processor throughput by image aspect ratio and batch grouping behavior.
- Patch embedding Conv2d/GEMM throughput.
- Window-attention layer throughput for `[B*9,196,768]`.
- Global-attention layer throughput for `[B,1024,768]`.
- Relative-position bias materialization/einsum cost for window and global layers.
- Full vision encoder latency and memory for batch sizes 1, 2, 4, 8.
- Backbone post-conv/flatten throughput.
- SLA head tokens/sec for fixed 501 steps versus EOS early-stop lengths.
- Benefit of caching `input_to_hidden(batch_hidden)` in recurrent head.
- NCHW source path versus guarded NHWC/channel-last conv-neck path.
- End-to-end requests/sec split into preprocessing, encoder, head, and postprocess.

## 14. Skip/defer list

- Training, loss, gradient checkpointing behavior.
- PaddleOCR native `inference.pdiparams` loading; route through a separate PaddleOCR/import audit.
- Location/bbox regression from `loc_reg_num`; not implemented in inspected native source.
- Beam search or generic text generation APIs.
- Dynamic image sizes in the neural graph; native patch embedding requires exact config image size.
- Processor GPU implementation for first graph parity.
- General boolean scatter, tokenizer language control, KV cache, RoPE, MoE, quantized packed weights, multi-GPU tensor parallel.

## 15. Final implementation checklist

- [ ] Admit native `model_type=slanext` configs and reject/route PaddleOCR artifact repos.
- [ ] Parse and validate vision dimensions, including `hidden_size % num_attention_heads == 0`.
- [ ] Load safetensors weights with fp32-preserved SLA modules.
- [ ] Implement fixed prepared-input ABI: `pixel_values [B,3,512,512]`.
- [ ] Implement patch Conv2d and optional Conv2d->Linear rewrite.
- [ ] Implement NHWC LayerNorm, residual add, GELU MLP, and packed QKV split.
- [ ] Implement window partition/unpartition with padding guards.
- [ ] Implement decomposed 2D relative-position math.
- [ ] Implement dense noncausal attention for window and global layers.
- [ ] Implement neck/post Conv2d stack and flatten/transpose to `[B,256,512]`.
- [ ] Implement additive recurrent attention cell and GRUCell.
- [ ] Implement `one_hot`, `argmax`, EOS predicates, and fixed or bounded dynamic recurrent loop.
- [ ] Implement structure generator and final softmax.
- [ ] Add processor/postprocess parity or document host-side composition.
- [ ] Add one-block, full-encoder, head-step, and end-to-end parity tests.
- [ ] Benchmark encoder, head loop, relative-position bias, and layout rewrite variants.
