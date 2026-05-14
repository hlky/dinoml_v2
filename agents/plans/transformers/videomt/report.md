# VidEoMT Transformers Audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4 in local checkout X:/H/transformers
Model id: tue-mps/videomt-dinov2-large-ytvis2019 as the main shape reference; sweep also covers small/base YTVIS, large OVIS, and large VSPW.
Config source: public Hugging Face config.json files, source defaults from VideomtConfig, and conversion registry in convert_videomt_to_hf.py.
Source files inspected: src/transformers/models/videomt/{modular_videomt.py,modeling_videomt.py,configuration_videomt.py,video_processing_videomt.py,convert_videomt_to_hf.py}; src/transformers/video_processing_utils.py for inherited video processor layout/sampling.
Any missing files or assumptions: sampled preprocessor_config.json files returned 404, so processor settings come from source defaults. No remote code is required. modeling_videomt.py and configuration_videomt.py are generated from modular_videomt.py; generated files are the runtime source at this commit, while future source edits should target modular_videomt.py.
```

Pinned source URLs:

- https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/videomt/modeling_videomt.py
- https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/videomt/modular_videomt.py
- https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/videomt/configuration_videomt.py
- https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/videomt/video_processing_videomt.py
- https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/video_processing_utils.py

Small config evidence is in `agents/plans/transformers/videomt/config_sweep.md`.

## 2. High-level architecture

VidEoMT is a video universal segmentation model. It is not an autoregressive text model. The useful first DinoML target is inference for `VideomtForUniversalSegmentation`: video frames in, per-frame class-query logits and mask logits out. Semantic, instance, and panoptic postprocessing can be staged separately.

```text
video decode/frame selection -> resize/RGB/rescale/normalize -> pixel_values_videos[B,T,C,H,W]
  -> flatten frames to [B*T,C,H,W]
  -> Conv2d patch embedding + learned absolute patch positions + CLS/register tokens
  -> early noncausal ViT encoder blocks over each frame independently
  -> regroup [B,T,S,C]
  -> for each frame: prepend learned/propagated query tokens
  -> final noncausal self-attention blocks
  -> LayerNorm -> class predictor + mask head + convolutional upscale -> mask logits
  -> optional semantic/instance/panoptic postprocess
```

Stage decomposition:

- CPU/data pipeline: video URL decode or caller-supplied frame tensor/list, optional uniform frame sampling if enabled by caller, RGB conversion, resize to configured square size, rescale by `1/255`, ImageNet normalization, output channel-first `pixel_values_videos`.
- Core GPU graph: NCTHW-like ABI `[B,T,C,H,W]` with frames flattened to ordinary NCHW images before patch embedding. There is no 3D convolution or tubelet embedding.
- Independently stageable encoder: patch embedding and the first `num_hidden_layers - num_blocks` blocks run per frame and can be validated before query propagation.
- Per-frame query stage: each frame uses learned query tokens; frames after the first add `query_updater(propagated_query)` from the previous frame. This is a fixed-size recurrent video state, not a KV cache.
- Heads: class logits `[B*T,num_queries,num_labels+1]`, mask logits `[B*T,num_queries,H_mask,W_mask]`, and last hidden states `[B*T,num_queries+prefix+patches,C]`.
- Postprocess: resize masks to target frame sizes, combine class probabilities and mask probabilities, threshold/filter, and emit segmentation maps plus variable-length segment records.

## 3. Important config dimensions

Source defaults from `VideomtConfig`:

| Field | Default | Runtime effect |
|---|---:|---|
| `hidden_size` | 1024 | token/query/mask feature width `C` |
| `num_hidden_layers` | 24 | total ViT block count |
| `num_attention_heads` | 16 | dense MHA heads |
| `head_dim` | `hidden_size / heads` | source requires divisibility |
| `image_size` | 640 | fixed patch-position table and square `grid_size` in head |
| `patch_size` | 16 | Conv2d patch kernel/stride |
| `num_channels` | 3 | input channel guard |
| `mlp_ratio` | 4 | vanilla MLP hidden width `4C` |
| `use_swiglu_ffn` | false | if true, switches FFN to SwiGLU rounded hidden width |
| `num_register_tokens` | 4 | prefix tokens after CLS |
| `num_queries` | 200 | object/query token count |
| `num_blocks` | 4 | final blocks that run after query insertion |
| `num_upscale_blocks` | 2 | mask feature map is upsampled by `2 ** num_upscale_blocks` |
| `hidden_act` | `gelu` | MLP and upscale activation |
| `layer_norm_eps` | `1e-6` | LayerNorm epsilon |
| `dtype` | checkpoint configs say `float32` | config metadata; source casts pixels to patch weight dtype |

Representative checkpoint sweep:

| Model id | Hidden | Layers | Heads | Image | Patch grid | Frames | Final query blocks | Labels |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `tue-mps/videomt-dinov2-small-ytvis2019` | 384 | 12 | 6 | 640 | 40x40 | 2 | 3 | 40 |
| `tue-mps/videomt-dinov2-base-ytvis2019` | 768 | 12 | 12 | 640 | 40x40 | 2 | 3 | 40 |
| `tue-mps/videomt-dinov2-large-ytvis2019` | 1024 | 24 | 16 | 640 | 40x40 | 2 | 4 | 40 |
| `tue-mps/videomt-dinov2-large-ovis` | 1024 | 24 | 16 | 640 | 40x40 | 2 | 4 | 25 |
| `tue-mps/videomt-dinov2-large-vspw` | 1024 | 24 | 16 | 1280 | 80x80 | 2 | 4 | 124 |

For the 640 large checkpoints with `B=1`: no-query sequence is `1 + 4 + 40*40 = 1605`; query-stage sequence is `200 + 1605 = 1805`; mask logits are `[2,200,160,160]` after the two 2x upscale blocks.

## 3a. Family variation traps

- `num_frames` appears in checkpoint configs and conversion metadata, but `VideomtConfig` source defaults do not declare it. The model forward accepts any runtime `T` that the processor/caller supplies; first parity should guard to checkpoint `num_frames=2` unless a recurrent-frame sweep is tested.
- Runtime video layout is `[B,T,C,H,W]`. The processor can accept channel-last input videos and immediately permutes each video to channel-first `[T,C,H,W]`. Treat NTHWC/channel-last as a processor-side convenience, not the model semantic graph.
- The model flattens frames into `[B*T,C,H,W]`; all patch embedding and early blocks are image-style 2D operations. There is no temporal attention before query propagation.
- The final `num_blocks` are applied independently per frame, but query tokens after frame 0 depend on the previous frame through `query_updater`. This is a per-video recurrent state ABI with shape `[B,num_queries,C]`.
- `image_size` is assumed scalar and square in `self.grid_size = (image_size // patch_size, image_size // patch_size)`. Tuple image sizes from the config class would not work in the current head without source changes.
- Absolute position embeddings cover patch tokens only. No interpolation path is present in the model; runtime H/W must match config-derived patch grid or `prefix_tokens.reshape(..., *grid_size)` will fail/mis-map tokens.
- Small/base/large public configs use vanilla GELU MLP. Source supports SwiGLU if `use_swiglu_ffn=True`, but sampled VidEoMT configs do not enable it.
- `hidden_size == heads * head_dim` for sampled configs. Source rejects non-divisible variants.
- Class head width depends on `num_labels + 1`; dataset variation affects only final class predictor and postprocess labels, not backbone operators.
- `attn_mask_probs` is initialized and carried from EoMT, but VidEoMT `_disable_attention_mask` raises and the generated forward never passes masks into layers for inference.
- Training with 5D video inputs is explicitly rejected. Loss, Hungarian matching, point sampling, and scipy dependency are out of scope for first inference.
- Generated modeling/config files contain copied EoMT blocks; modular source inherits EoMT. DinoML should audit generated runtime behavior for this family and share implementation only after matching EoMT parity constraints.

## 4. Operator coverage checklist

Tensor/layout ops:

- Video ABI guard: `pixel_values_videos` rank 5, `[B,T,C,H,W]`, `C == config.num_channels`.
- Reshape flatten frames `[B,T,C,H,W] -> [B*T,C,H,W]`.
- Conv2d patch embedding output `[B*T,C_hidden,G,G]`; flatten spatial and transpose to `[B*T,G*G,C_hidden]`.
- Expand learned CLS `[1,1,C]`, register `[1,R,C]`, query `[Q,C]`, and add learned absolute patch position embeddings.
- Concatenate along sequence: `[CLS, registers, patches]`; later `[queries, frame_tokens]`.
- View regroup `[B*T,S,C] -> [B,T,S,C]`; frame slicing over `T`.
- Prefix/image token slicing after query stage: queries `[:Q]`; patch tokens `[Q + 1 + R :]`.
- Transpose and reshape patch tokens `[B,P,C] -> [B,C,G,G]`.
- Final concatenation over frames along batch dimension: list of per-frame `[B,...]` tensors to `[B*T,...]`.

Neural primitives:

- Conv2d patch embed: `Conv2d(3 -> C, kernel=patch_size, stride=patch_size, bias=true)`.
- Biased Linear Q/K/V/out projections: each `Linear(C -> C)`.
- LayerNorm over last dimension `C`, epsilon `1e-6`.
- LayerScale elementwise multiply by learned `[C]`.
- Vanilla MLP: `Linear(C -> 4C)`, GELU, `Linear(4C -> C)`.
- Optional SwiGLU source path: `Linear(C -> 2H)`, chunk on last dim, `silu(x1) * x2`, `Linear(H -> C)`, where `H = ceil((C * mlp_ratio * 2 / 3) / 8) * 8`.
- Query updater: `Linear(C -> C)`.
- Class predictor: `Linear(C -> num_labels + 1)`.
- Mask head: `Linear(C -> C)`, activation, `Linear(C -> C)`, activation, `Linear(C -> C)`.
- Upscale block repeated `num_upscale_blocks`: `ConvTranspose2d(C -> C, kernel=2, stride=2, bias=true)`, activation, depthwise `Conv2d(C -> C, kernel=3, padding=1, groups=C, bias=false)`, channel LayerNorm2d.
- Mask product: `einsum("bqc,bchw->bqhw")`, equivalently per-frame/query batched dot over `C`.

Attention primitives:

- Dense noncausal self-attention only.
- MHA only; no GQA/MQA.
- Q/K/V all project to `C`; shape `[B_or_BT,S,C] -> [B_or_BT,heads,S,head_dim]`.
- Eager attention math: `matmul(q,k^T) * head_dim**-0.5`, optional mask add, softmax in float32, dropout in training only, `matmul(prob,v)`, transpose/contiguous, output projection.
- `_supports_sdpa = True`; first DinoML parity can target eager math then optimize to SDPA/FlashAttention-style dense noncausal attention.

Pre/postprocess-coupled ops:

- Processor: optional uniform frame sampling, decode/fetch outside the model graph, RGB conversion, bilinear resize, rescale, normalize, channel-last to channel-first permute.
- Postprocess semantic: softmax over classes excluding null, sigmoid masks, `matmul([F,labels,Q] x [F,Q,H*W])`, reshape, bilinear upsample, argmax.
- Postprocess instance/panoptic: mask resizing, class softmax/max, sigmoid, thresholding, boolean masks, per-query score calculation, `argmax` over query ownership, variable-length segment records. No NMS is present.

Training-only/deferred:

- Hungarian matching via scipy, point sampling with `grid_sample`, random point generation, topk uncertainty sampling, BCE/CE/Dice losses, distributed `num_masks` reduction, DropPath randomness.

## 5. Layer/block breakdown

Patch and prefix embedding:

```text
pixel_values_videos: [B,T,3,H,W]
flat = reshape(pixel_values_videos, [B*T,3,H,W])
patch = Conv2d(3 -> C, kernel=P, stride=P)(flat)       # [B*T,C,G,G]
tokens = flatten(patch, start_dim=2).transpose(1,2)   # [B*T,G*G,C]
tokens = tokens + position_embedding[position_ids]    # [B*T,G*G,C]
hidden = cat(cls.expand, reg.expand, tokens, dim=1)   # [B*T,1+R+G*G,C]
```

Encoder block, repeated first `num_hidden_layers - num_blocks` times over flattened frames:

```text
x1 = LayerNorm(x)
q,k,v = Linear(C -> C)(x1), reshape to [B*T,heads,S,head_dim]
a = DenseSelfAttention(q,k,v, causal=False)
x = x + LayerScale(Linear(C -> C)(a))
x2 = LayerNorm(x)
m = Linear(C -> 4C)(x2) -> GELU -> Linear(4C -> C)
x = x + LayerScale(m)
```

Per-frame query block:

```text
hidden = view(hidden, [B,T,S,C])
propagated_query = None
for frame in T:
    frame_tokens = hidden[:, frame]                    # [B,S,C]
    if first frame:
        query = learned_query.expand(B,Q,C)
    else:
        query = query_updater(propagated_query) + learned_query
    frame_tokens = cat(query, frame_tokens, dim=1)     # [B,Q+S,C]
    repeat final num_blocks encoder blocks
    sequence_output = LayerNorm(frame_tokens)
    mask_logits, class_logits = predict(sequence_output)
    propagated_query = frame_tokens[:, :Q, :]
```

Prediction head:

```text
query_tokens = logits[:, :Q, :]                        # [B,Q,C]
class_logits = Linear(C -> num_labels+1)(query_tokens)
prefix = logits[:, Q + 1 + R :, :].transpose(1,2)      # [B,C,G*G]
prefix = reshape(prefix, [B,C,G,G])
query_tokens = MLP(C -> C -> C -> C)(query_tokens)
prefix = upscale_block(prefix)                         # [B,C,G*4,G*4] for two blocks
mask_logits = einsum("bqc,bchw->bqhw", query_tokens, prefix)
```

## 6. Attention requirements

- Causal or noncausal: noncausal.
- Self-attention or cross-attention: self-attention only; query tokens attend jointly with image/prefix tokens in final blocks, but this is still self-attention over a concatenated sequence.
- MHA/MQA/GQA: MHA.
- Heads/head dim: sampled configs use `(6,64)`, `(12,64)`, `(16,64)`.
- Query/key/value widths: all `C`.
- Rectangular attention: not required; `Q_len == KV_len == S` per block.
- Masking style: no inference attention mask is passed by VidEoMT forward. Eager helper accepts additive masks, but current family path supplies `None`.
- Packed/varlen: none.
- Sliding/local/block sparse: none.
- RoPE/ALiBi/relative bias: none.
- KV cache: none. The video query recurrence is fixed-size `[B,Q,C]` and updated once per frame; it does not store per-layer K/V tensors and does not grow with sequence length.
- SDPA/FlashAttention compatibility: dense noncausal attention with dropout `0.0` in eval is a clean candidate. Preserve float32 softmax accumulation from eager attention for parity.

For 640 large, early attention sequence is `S=1605`; final query attention sequence is `S=1805`. For VSPW 1280, early `S=6405`, final `S=6605`, which is the major scaling risk.

## 7. Position encoding and custom math

VidEoMT uses a learned absolute patch-position embedding table:

```python
tokens = tokens + position_embeddings(position_ids)  # position_ids = arange(num_patches).expand(1, -1)
```

There is no interpolation, rotary encoding, ALiBi, or relative bias in the inspected source. Position IDs and embedding lookup are constant for a fixed config and can be folded into a cached `[1,G*G,C]` tensor. Runtime image shapes should be guarded to `H == W == image_size` and divisibility by `patch_size`, because the final head reshapes patch tokens to the fixed `grid_size`.

Custom recurrent query math:

```python
if propagated_query is None:
    query = learned_query.expand(batch, -1, -1)
else:
    query = query_updater(propagated_query) + learned_query
```

This depends on the previous frame's final-block query tokens and is deterministic in eval.

## 8. Preprocessing and input packing

Video decode and frame sampling are processor/data-pipeline work. `VideomtVideoProcessor` inherits `BaseVideoProcessor`:

- Default class settings: `size={"height":640,"width":640}`, bilinear resize, RGB conversion, rescale `1/255`, ImageNet mean/std normalization, no center crop, `do_sample_frames=False`.
- If `do_sample_frames=True`, `BaseVideoProcessor.sample_frames` uniformly samples either a fixed `num_frames` or a target `fps`, but the sampled public processor config was absent. First integration should require caller-provided frames or an explicit DinoML data-pipeline policy.
- Input arrays may be channel-last `[T,H,W,C]`; `_prepare_input_videos` permutes to `[T,C,H,W]`.
- Batched output key is `pixel_values_videos`, usually tensorized as `[B,T,C,H,W]`.

Model-coupled input packing:

- No placeholder tokens, text tokens, modality token IDs, `cu_seqlens`, or scatter stitching.
- Temporal order is preserved by frame index. The model flattens `B*T` for early blocks, then restores `[B,T,S,C]` before recurrent query propagation.
- First useful runtime guard: admit fixed `T=2` public checkpoints, fixed square image size, `C=3`, NCHW frame layout. Broader `T` can be admitted after query recurrence parity is tested.

Structured output postprocessing:

- Semantic segmentation returns one `[target_h,target_w]` class-index map per frame. It excludes the null class before class/mask fusion.
- Instance segmentation thresholds combined class/mask scores, writes integer instance IDs, and returns variable-length `segments_info`; no NMS.
- Panoptic segmentation filters null/low-score queries, computes per-pixel winning query by `pred_score * sigmoid(mask)`, validates area overlap, optionally fuses configured stuff labels, and returns segment IDs plus records.

## 9. Graph rewrite / lowering opportunities

### Rewrite: non-overlap Conv2d patch embed -> Linear

Preconditions:

- Input is channel-first NCHW `[N,3,H,W]`.
- `kernel_size == stride == patch_size`.
- `padding == 0`, `dilation == 1`, `groups == 1`.
- `H` and `W` are divisible by patch size and equal config image size for first integration.

Replacement:

```text
WindowFlatten(NCHW, kh=P, kw=P, stride=P) -> MatMul(weight_flat.T) -> BiasAdd -> Reshape [N,G*G,C]
```

Weight transform:

```python
w = conv.weight.reshape(out_channels, in_channels * patch_h * patch_w)
```

Layout constraints: if source NCHW is translated to NHWC, the window flatten order and weight transform must be rewritten together. Failure cases: dynamic non-square shapes, padding/dilation/groups, or mismatched position table.

Parity sketch: compare patch tokens before position add for random fp32 and checkpoint weights at `640x640` and `1280x1280`.

### Rewrite: `einsum("bqc,bchw->bqhw")` -> batched GEMM

Preconditions:

- Query tokens contiguous/logical `[B,Q,C]`.
- Prefix feature map logical `[B,C,H,W]`.
- Reduction axis is channel `C`.

Replacement:

```text
prefix_flat = reshape(prefix, [B,C,H*W])
mask_flat = BMM(query, prefix_flat)  # [B,Q,H*W]
mask = reshape(mask_flat, [B,Q,H,W])
```

Layout constraints: source prefix is NCHW. NHWC would require reducing the last axis and replacing the flatten order.

Parity sketch: random tensors across `C={384,768,1024}`, `Q=200`, grids `160x160` and `320x320`.

### Rewrite: local LayerNorm2d channel-last island

Source pattern:

```text
NCHW -> permute NHWC -> layer_norm over C -> permute NCHW
```

Replacement: a native NCHW channel LayerNorm kernel or a fused depthwise-conv + channel-LN region.

Preconditions: normalized shape is exactly `[C]`; consumers and producers remain NCHW; epsilon and affine weights match PyTorch layer norm.

Layout constraints: do not globally translate the surrounding ConvTranspose2d/depthwise Conv2d/head einsum unless every axis is rewritten. This region is a safe local layout/fusion candidate.

### Rewrite: recurrent frame loop as static scan

Preconditions:

- Fixed `T` known at compile time, initially `T=2`.
- Inference only, no dropout/drop path.
- Per-frame operations share weights.

Replacement: unroll `T` frame iterations with one fixed state tensor `[B,Q,C]`. For `T=2`, this is a simple two-call schedule: frame0 learned query, frame1 updated query.

Failure cases: variable `T` without dynamic loop support, batching videos with different frame counts, or exposing intermediate hidden states with exact Python tuple semantics.

## 10. Kernel fusion candidates

Highest priority:

- Dense noncausal attention for long ViT sequences, especially VSPW `S=6605`.
- LayerNorm + QKV projection preparation. Source has separate Q/K/V linears; a packing pass can combine weights as `[Q;K;V]` only if state dict loading preserves original aliases and bias order.
- MLP GEMM + GELU + GEMM for `C -> 4C -> C`.
- Mask head and mask `BMM` product, because `[Q,C] x [C,H*W]` is large at 160x160 or 320x320.

Medium priority:

- Conv2d patch embedding lowered to GEMM for static square inputs.
- ConvTranspose2d + activation + depthwise Conv2d + LayerNorm2d in the upscale path.
- Query updater + query add for frame recurrence.
- Postprocess semantic class/mask fusion `matmul` and bilinear mask resize if GPU end-to-end postprocess is desired.

Lower priority:

- DropPath and training loss kernels are unnecessary for inference.
- Boolean panoptic segment assembly is variable-length and can stay CPU/postprocess initially.

## 11. Runtime staging plan

Stage 1: parse config and load weights for one public 640 checkpoint; reject training labels, non-5D inputs, non-square image sizes, and unsupported `use_swiglu_ffn=True`.

Stage 2: implement/validate embeddings plus one early encoder block on flattened frames `[B*T,S,C]`.

Stage 3: run all early encoder blocks and restore `[B,T,S,C]`; validate against Transformers for `T=2`.

Stage 4: implement fixed `T=2` recurrent query stage with final blocks and `query_updater` state.

Stage 5: implement prediction heads: class predictor, mask head, upscale block, and mask BMM/einsum.

Stage 6: add semantic postprocess parity; defer instance/panoptic variable-length records until raw logits are stable.

Stage 7: optimize attention, patch embedding, mask product, and layout islands; then consider VSPW 1280 shape admission.

Stub initially: training losses, hidden-state/attention output capture, instance/panoptic segment records, dynamic `T`, `use_swiglu_ffn=True`, and channel-last model input.

## 12. Parity and validation plan

- Processor parity: given a small synthetic list of frames, verify output tensor key, layout `[B,T,C,H,W]`, resize, rescale, and normalization against Transformers.
- Patch embedding parity: compare tokens after Conv2d/flatten/transpose and after position/CLS/register concat.
- Single block parity: fp32 one early block and one query-stage block, including LayerScale.
- Early encoder parity: all `num_hidden_layers - num_blocks` blocks over flattened frames.
- Recurrent query parity: frame0 and frame1 final query tokens separately; ensure frame1 changes when frame0 propagated query is perturbed.
- Head parity: class logits and mask logits for `B=1,T=2`.
- End-to-end raw output parity: `masks_queries_logits`, `class_queries_logits`, and `last_hidden_state`.
- Postprocess parity: semantic map for fixed target sizes; later instance/panoptic records with deterministic thresholds.

Suggested tolerances: fp32 max abs/relative around `1e-4` for block/head tests; fp16/bf16 tolerances should be set after attention backend choice, with looser masks near threshold boundaries.

## 13. Performance probes

- Processor/decode throughput split from model throughput.
- Patch embedding throughput for `640` and `1280`.
- Early encoder attention time at `S=1605` and `S=6405`.
- Final query-stage attention time at `S=1805` and `S=6605`.
- Frame-loop overhead and query state update cost for `T={1,2,4,8}` after fixed `T=2` parity.
- Mask head/upscale/mask-BMM cost at output grids `160x160` and `320x320`.
- Memory probes for attention logits/probabilities at VSPW shape; dense attention may dominate.
- Postprocess latency for semantic versus instance/panoptic paths.
- Batch-size sweep over videos and effective flattened frame batch `B*T`.

## 14. Skip/defer list

- Training with `mask_labels`/`class_labels`; source rejects it for 5D video inputs.
- Hungarian matcher, scipy dependency, point sampling, `grid_sample`, random/topk uncertainty sampling, losses.
- Dynamic frame counts and dynamic loops; start with checkpoint `num_frames=2`.
- Runtime images that do not match config `image_size`.
- `use_swiglu_ffn=True` until a checkpoint needing it is found for VidEoMT.
- Instance/panoptic variable-length postprocess records for first raw-logit parity.
- Output attentions/hidden states capture unless debugging parity.
- Global NHWC/channel-last graph translation; only local guarded layout rewrites first.
- Quantization, tensor parallelism, and distributed training helpers.

## 15. Final implementation checklist

- [ ] Parse `VideomtConfig`, including checkpoint `num_frames` metadata even though source defaults omit it.
- [ ] Load weights and map generated VidEoMT names.
- [ ] Implement video input ABI guard `[B,T,C,H,W]` and fixed `T=2` admission.
- [ ] Implement NCHW Conv2d patch embedding or guarded Conv2d-to-Linear rewrite.
- [ ] Implement CLS/register/position embedding assembly.
- [ ] Implement noncausal MHA block with LayerNorm and LayerScale.
- [ ] Implement vanilla GELU MLP path; add guarded SwiGLU support later.
- [ ] Implement recurrent query state `[B,Q,C]` and `query_updater`.
- [ ] Implement class predictor, mask head, upscale block, and mask BMM.
- [ ] Add local `LayerNorm2d` NCHW/NHWC parity or native channel-LN kernel.
- [ ] Add raw-logit parity tests for small 640 and large 640 checkpoints.
- [ ] Add VSPW 1280 memory/performance probe before admitting it by default.
- [ ] Add semantic postprocess parity, then instance/panoptic postprocess parity.
- [ ] Document layout guards for NTHWC processor input versus model NCTHW/NCHW graph.
