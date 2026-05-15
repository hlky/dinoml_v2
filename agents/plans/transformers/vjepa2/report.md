# DinoML Transformers Audit: vjepa2

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: facebook/vjepa2-vitl-fpc64-256 primary; sweep also covers vjepa2-vith-fpc64-256, vjepa2-vitg-fpc64-256, vjepa2-vitg-fpc64-384, vjepa2-vitl-fpc16-256-ssv2, vjepa2-vitg-fpc64-384-ssv2
Config source: official Hugging Face config.json and video_preprocessor_config.json files
Source files inspected:
- transformers/src/transformers/models/vjepa2/configuration_vjepa2.py
- transformers/src/transformers/models/vjepa2/modeling_vjepa2.py
- transformers/src/transformers/models/vjepa2/video_processing_vjepa2.py
- transformers/src/transformers/models/vjepa2/__init__.py
Any missing files or assumptions: no modular source file found for this family. Conversion scripts were not used as behavioral source. Model cards were read only for sampling/preprocessor context, not operator claims.
```

Primary source URLs:
- Transformers source at commit: [models/vjepa2](https://github.com/huggingface/transformers/tree/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/vjepa2)
- Configs: [vitl-fpc64-256](https://huggingface.co/facebook/vjepa2-vitl-fpc64-256/blob/main/config.json), [vith-fpc64-256](https://huggingface.co/facebook/vjepa2-vith-fpc64-256/blob/main/config.json), [vitg-fpc64-256](https://huggingface.co/facebook/vjepa2-vitg-fpc64-256/blob/main/config.json), [vitg-fpc64-384](https://huggingface.co/facebook/vjepa2-vitg-fpc64-384/blob/main/config.json), [vitl-fpc16-256-ssv2](https://huggingface.co/facebook/vjepa2-vitl-fpc16-256-ssv2/blob/main/config.json), [vitg-fpc64-384-ssv2](https://huggingface.co/facebook/vjepa2-vitg-fpc64-384-ssv2/blob/main/config.json)
- Preprocessors: same repos, `video_preprocessor_config.json`.

Report target: inference-only CUDA runtime for video feature extraction and video classification. Masked predictor parity is documented but should be staged after encoder/classification parity.

## 2. High-level architecture

V-JEPA 2 in Transformers is a video ViT-style encoder with 3D tubelet patch embedding, noncausal full self-attention, 3-axis rotary position embedding over temporal/height/width token coordinates, GELU MLP blocks, and optional heads:

```text
CPU/video decode + frame sampling
  -> VJEPA2VideoProcessor resize/rescale/normalize/center-crop
  -> pixel_values_videos [B,T,C,H,W]
  -> permute to [B,C,T,H,W]
  -> Conv3d tubelet patch embedding
  -> encoder transformer blocks
  -> final LayerNorm
  -> feature tokens [B,N,H]
```

Classification adds:

```text
encoder tokens -> 3 self-attention pooler layers -> learned 1-token cross-attention pooler -> Linear(num_labels)
```

Masked prediction adds:

```text
encoder tokens + context_mask/target_mask
  -> gather context tokens
  -> Linear(hidden -> pred_hidden)
  -> gather/repeat learned mask tokens for targets
  -> concat context+target tokens
  -> argsort/gather by original patch positions
  -> predictor transformer blocks with RoPE position_mask
  -> unsort/gather predicted target tokens
  -> Linear(pred_hidden -> hidden)
```

Independently stageable pieces:
- CPU/data pipeline: video decode, frame selection, resize/crop/rescale/normalize.
- GPU encoder: patch embedding plus transformer stack; cacheable as video features.
- GPU classifier head: attentive pooler plus classifier; depends only on encoder tokens.
- Masked predictor: useful for V-JEPA-style prediction, but index/sort/gather-heavy and not needed for first classification/feature-extraction parity.

## 3. Important config dimensions

Source defaults from `configuration_vjepa2.py`:

| Field | Default | Operator impact |
| --- | ---: | --- |
| `crop_size` | 256 | spatial grid `crop_size / patch_size` |
| `frames_per_clip` | 64 | temporal grid `frames_per_clip / tubelet_size` for configured position ids |
| `tubelet_size` | 2 | Conv3d temporal kernel/stride |
| `patch_size` | 16 | Conv3d spatial kernel/stride |
| `in_chans` | 3 | video channels |
| `hidden_size` | 1024 | encoder width |
| `num_hidden_layers` | 24 | encoder block count |
| `num_attention_heads` | 16 | encoder/pooler heads |
| `mlp_ratio` | 4.0 | encoder MLP hidden width |
| `qkv_bias` | true | encoder Q/K/V Linear bias |
| `layer_norm_eps` | 1e-6 | LayerNorm parity |
| `hidden_act` | gelu | MLP activation |
| `pred_hidden_size` | 384 | predictor-only width |
| `pred_num_attention_heads` | 12 | predictor-only heads |
| `pred_num_hidden_layers` | 12 | predictor-only block count |
| `pred_num_mask_tokens` | 10 | learned mask token table |
| `num_pooler_layers` | 3 | classifier pooler self-attention layers |

Representative checkpoint sweep:

| Checkpoint | Arch | Crop | Frames | Tokens `N` | Hidden | Heads | Head dim | Layers | MLP width | Predictor | Classifier |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `facebook/vjepa2-vitl-fpc64-256` | `VJEPA2Model` | 256 | 64 | 8192 | 1024 | 16 | 64 | 24 | 4096 | 384 x 12 layers | no |
| `facebook/vjepa2-vith-fpc64-256` | `VJEPA2Model` | 256 | 64 | 8192 | 1280 | 16 | 80 | 32 | 5120 | 384 x 12 layers | no |
| `facebook/vjepa2-vitg-fpc64-256` | `VJEPA2Model` | 256 | 64 | 8192 | 1408 | 22 | 64 | 40 | 6144 | 384 x 12 layers | no |
| `facebook/vjepa2-vitg-fpc64-384` | `VJEPA2Model` | 384 | 64 | 18432 | 1408 | 22 | 64 | 40 | 6144 | 384 x 12 layers | no |
| `facebook/vjepa2-vitl-fpc16-256-ssv2` | `VJEPA2ForVideoClassification` | 256 | 16 | 2048 | 1024 | 16 | 64 | 24 | 4096 | present but skipped by classifier | 174 labels |
| `facebook/vjepa2-vitg-fpc64-384-ssv2` | `VJEPA2ForVideoClassification` | 384 | 64 | 18432 | 1408 | 22 | 64 | 40 | 6144 | present but skipped by classifier | 174 labels |

Preprocessor sweep:

| Crop | Resize shortest edge | Output data format | Normalization |
| ---: | ---: | --- | --- |
| 256 | 292 | `channels_first` | ImageNet mean/std, rescale 1/255 |
| 384 | 438 | `channels_first` | ImageNet mean/std, rescale 1/255 |

## 3a. Family variation traps

- Video ABI is `[B,T,C,H,W]` at model entry even though preprocessor `data_format` is `channels_first` per frame. The model immediately permutes to `[B,C,T,H,W]` for `Conv3d`.
- `frames_per_clip` affects configured position grid and token count, but source comments say it "does not impact inference"; actual runtime token count comes from input frames after tubelet Conv3d. DinoML should guard frame counts against the configured rotary grid unless extrapolation parity is intentionally tested.
- If runtime `num_frames < tubelet_size`, the model repeats frames along temporal dimension before Conv3d. That special path must be preserved or rejected.
- ViT-G uses `mlp_ratio=4.363636363636363`, giving MLP width 6144, not `4 * hidden_size`.
- 3D RoPE divides each head into temporal/height/width even dimensions plus a leftover tail. For head dim 64: 20+20+20 rotated, 4 unrotated. For head dim 80: 26+26+26 rotated, 2 unrotated.
- Encoder Q/K/V are separate Linear modules with optional `qkv_bias`; no packed QKV source weight layout.
- Attention is noncausal MHA, not GQA/MQA and no KV cache.
- Predictor is not needed for `VJEPA2ForVideoClassification`, because classifier calls `self.vjepa2(..., skip_predictor=True)`.
- Masked predictor requires `gather`, `cat`, `argsort`, reverse `argsort`, and position-index-aware RoPE. It should be admitted separately from the encoder.
- Classification pooler cross-attention has no output projection in `VJEPA2PoolerCrossAttention`; the residual add happens directly with the query token.
- Historical config fields `hidden_dropout_prob`, `image_size`, `use_SiLU`, and `wide_SiLU` appear in sampled configs but are not read by the inspected modeling source for runtime behavior.
- Layout optimization is tempting around Conv3d, but semantic source axes are fixed: processor/model entry `[B,T,C,H,W]`, Conv3d `[B,C,T,H,W]`, output tokens `[B,N,H]`. NHWC/NTHWC should be a guarded local rewrite, not a default translation.

## 4. Operator coverage checklist

Tensor/layout ops:
- `permute(0,2,1,3,4)` from `[B,T,C,H,W]` to `[B,C,T,H,W]`.
- `to(dtype=conv.weight.dtype)` on video input.
- `repeat` for `num_frames < tubelet_size`, learned query token repeat, context repeat, mask token repeat.
- `flatten(2)`, `transpose(1,2)`, `view`, `reshape`, `contiguous`, `squeeze(1)`.
- `cat` along token axis and batch axis.
- `arange`, integer floor division/modulo/subtract for token coordinates.
- `gather` with expanded index tensors for masks and token sorting.
- `argsort` for predictor sort/unsort.

Neural primitives:
- `Conv3d(in_chans -> hidden, kernel=(tubelet,patch,patch), stride=same, bias=True)` for tubelet patches.
- `LayerNorm(hidden, eps=1e-6)` and `LayerNorm(pred_hidden, eps=1e-6)`.
- `Linear(hidden -> hidden)` Q/K/V/out projections with bias in encoder and pooler.
- `Linear(hidden -> int(hidden * mlp_ratio))`, GELU, `Linear(mlp -> hidden)` with bias.
- Predictor `Linear(hidden -> pred_hidden)`, predictor blocks, and `Linear(pred_hidden -> hidden)`.
- Classifier `Linear(hidden -> num_labels)` with bias.

Attention primitives:
- Dense noncausal self-attention over all video tokens, shape `[B, heads, N, head_dim]`.
- Dense noncausal self-attention in pooler over encoder tokens.
- Dense noncausal cross-attention with learned query shape `[B,1,H]` attending to encoder tokens `[B,N,H]`.
- Attention backend path goes through `ALL_ATTENTION_FUNCTIONS`; eager fallback is matmul -> softmax(dtype fp32) -> dropout -> matmul.

Position/custom math:
- 3-axis RoPE applied to Q and K before attention.
- Position ids are derived from token indices or predictor `position_mask`: frame `id // (grid*grid)`, height `(id % tokens_per_frame) // grid`, width residual.

Preprocessing-coupled ops:
- Video resize shortest edge, center crop, rescale by 1/255, ImageNet normalize.
- Frame sampling/decode is external to model source; model card examples sample indices manually.

Structured output/postprocessing:
- Classification logits `[B,num_labels]`; top-k/softmax/id2label are controller/postprocessing, not model graph.

Not required:
- Causal decode, text tokenizer, logits sampling, KV cache, MoE, quantized weights, NMS, recurrent state.

## 5. Layer/block breakdown

Patch embedding:

```text
pixel_values_videos: [B,T,3,H,W]
x = permute(pixel_values_videos, [0,2,1,3,4])       # [B,3,T,H,W]
if T < tubelet_size: x = repeat temporal to tubelet
x = Conv3d(3 -> hidden, kernel=stride=(tubelet,16,16))(x)
x = flatten spatial/temporal dims, transpose          # [B,N,hidden]
```

Encoder block, repeated `num_hidden_layers` times:

```text
residual = x
x_norm = LayerNorm(x)
q = Linear(hidden -> hidden, bias=qkv_bias)(x_norm).view([B,N,heads,head_dim]).transpose(1,2)
k = Linear(hidden -> hidden, bias=qkv_bias)(x_norm).view(...).transpose(1,2)
v = Linear(hidden -> hidden, bias=qkv_bias)(x_norm).view(...).transpose(1,2)
q,k = 3_axis_rope(q,k, token_positions)
a = dense_noncausal_attention(q,k,v, scale=head_dim**-0.5)
a = transpose/reshape to [B,N,hidden]
x = residual + Linear(hidden -> hidden)(a)
residual = x
x = residual + Linear(mlp_width -> hidden)(GELU(Linear(hidden -> mlp_width)(LayerNorm(x))))
```

Final encoder output:

```text
last_hidden_state = LayerNorm(x)  # [B,N,hidden]
```

Predictor-only block:
- Applies the same `VJEPA2Layer` implementation at `pred_hidden_size` and `pred_num_attention_heads`.
- Uses `position_mask` from concatenated context/target patch ids, so RoPE positions are gathered positions, not dense `arange(N)`.

Classification head:

```text
x = encoder_tokens [B,N,H]
repeat 3 times: x = LN -> self-attn -> residual -> LN -> MLP -> residual
q = learned query_tokens.repeat(B,1,1)                # [B,1,H]
y = cross_attention(q, LN(x), LN(x))                  # [B,1,H], no o_proj
y = q + y
y = y + MLP(LN(y))
pooled = squeeze(y, dim=1)                            # [B,H]
logits = Linear(H -> num_labels)(pooled)
```

## 6. Attention requirements

Encoder attention:
- Noncausal self-attention.
- MHA, not GQA/MQA.
- `heads = num_attention_heads`, `kv_heads = heads`.
- Head dim is `hidden_size / heads`; source rejects non-divisible hidden size.
- Q/K/V widths equal `hidden_size`.
- Query length equals key/value length `N = (T/tubelet) * (H/patch) * (W/patch)` after Conv3d.
- No attention mask in encoder path.
- RoPE is applied before attention to Q and K.
- No KV cache. This is an encoder feature model, not autoregressive generation.
- FlashAttention/SDPA compatibility is advertised by `_supports_sdpa = True` and `_supports_flash_attn = True`; parity still needs the same noncausal dense semantics and no dropout in inference.

Predictor attention:
- Same dense noncausal MHA, but sequence length is `N_context + N_target`.
- Position ids come from sorted patch ids. Admission should require integer masks with valid patch indices.
- Sort/gather means the source attention order is position-sorted, then unsorted before target extraction.

Pooler attention:
- Self-attention layers operate over encoder token length `N`.
- Cross-attention has query length 1 and key/value length `N`; no output projection.
- No cache; learned query can be treated as a constant parameter.

Masking:
- The eager attention helper accepts `attention_mask`, but encoder and pooler pass `None` in observed paths. Pooler layer signatures retain mask plumbing but current classifier path uses no mask.

## 7. Position encoding and custom math

3-axis RoPE uses dynamic token ids and recomputes frequencies inside `rotate_queries_or_keys`. DinoML can precompute `omega` per head slice and dtype, but positions may be a dynamic mask in predictor.

Concise source-equivalent math:

```python
def rotate_half_pairs(x):
    y = x.reshape(*x.shape[:-1], -1, 2)
    y1, y2 = y.unbind(-1)
    return torch.stack((-y2, y1), dim=-1).flatten(-2)

def rotate_axis(x_axis, pos):
    d = x_axis.shape[-1]
    omega = 1.0 / (10000 ** (torch.arange(d // 2, dtype=x_axis.dtype) / (d / 2.0)))
    freq = pos[..., None] * omega
    sin = freq.sin().repeat_interleave(2, dim=-1)
    cos = freq.cos().repeat_interleave(2, dim=-1)
    return x_axis * cos + rotate_half_pairs(x_axis) * sin

def vjepa2_rope(qk, frame_pos, height_pos, width_pos, head_dim):
    axis = 2 * ((head_dim // 3) // 2)
    qkd = rotate_axis(qk[..., 0:axis], frame_pos)
    qkh = rotate_axis(qk[..., axis:2*axis], height_pos)
    qkw = rotate_axis(qk[..., 2*axis:3*axis], width_pos)
    return cat([qkd, qkh, qkw, qk[..., 3*axis:]], dim=-1)
```

Precomputable:
- `omega` for each axis slice and dtype.
- Dense `frame/height/width` position vectors for the unmasked encoder at fixed configured grid.

Dynamic:
- Predictor `position_mask` values, `argsort`, and gathered positions.
- Runtime token count if input frame count or crop changes. First integration should prefer fixed buckets matching checkpoint configs.

## 8. Preprocessing and input packing

`VJEPA2VideoProcessor` defaults:
- Resize shortest edge to `int(crop_size * 256 / 224)`: 292 for crop 256, 438 for crop 384.
- Center crop to square crop.
- Rescale by `0.00392156862745098`.
- Normalize with ImageNet mean `[0.485,0.456,0.406]` and std `[0.229,0.224,0.225]`.
- Output `pixel_values_videos` uses per-frame channels-first format and model input shape `[B,T,C,H,W]`.

Video decode and frame sampling are caller/data-pipeline owned. Model cards show manual frame indices via TorchCodec, and SSV2 preprocessor configs include nullable `do_sample_frames`, `fps`, and `num_frames`, but the inspected `VJEPA2VideoProcessor` class only derives resize size from crop size. DinoML should not infer a built-in frame sampler from the model graph.

Image-as-video path:
- Model card examples repeat an image to multiple frames outside the model. The model itself also repeats only when `num_frames < tubelet_size`.

Masks:
- Default `context_mask` and `target_mask` are full `arange(N)` per batch when both are omitted.
- `apply_masks` expects list entries shaped `[B,num_kept]`; it expands to `[B,num_kept,D]` and gathers along token axis, then concatenates list elements along batch.

## 9. Graph rewrite / lowering opportunities

### Rewrite: non-overlap Conv3d tubelet patch embedding -> Linear

Source pattern:

```text
[B,T,C,H,W] -> permute [B,C,T,H,W] -> Conv3d(kernel=stride=(tubelet,patch,patch)) -> flatten(2).transpose(1,2)
```

Replacement:

```text
WindowFlatten3D in source token order -> GEMM(weight_flat.T) -> BiasAdd -> [B,N,hidden]
```

Preconditions:
- `kernel_size == stride == (tubelet_size, patch_size, patch_size)`.
- `padding == 0`, `dilation == 1`, `groups == 1`.
- `T`, `H`, and `W` divisible by tubelet/patch after any frame-repeat path.
- Input layout and flatten order exactly match PyTorch Conv3d output order: temporal grid, height grid, width grid.
- Bias preserved.

Weight transform:

```python
w = conv.weight.reshape(hidden, in_chans * tubelet * patch * patch)
b = conv.bias
```

Layout constraints:
- Source entry is `[B,T,C,H,W]`; Conv3d consumes `[B,C,T,H,W]`.
- An NTHWC optimization would need a fused window extractor with equivalent flatten order and a weight permutation. Guard the region from generic NHWC axis rewrites unless all consumers remain token-sequence consumers.

Failure cases:
- Non-divisible dynamic input.
- `T < tubelet_size` unless the repeat path is modeled before rewrite.
- Future configs with non-square or list patch size need separate guards.

Parity test sketch:
- Random fp32/fp16 video, compare Conv3d path to WindowFlatten3D+GEMM for 256/16 and 384/16, including bias and exact token order.

### Rewrite: separate Q/K/V Linear -> packed projection

Preconditions:
- Same input tensor, same hidden width, all three projections present.
- Bias packing respects `q, k, v` order.

Replacement:

```text
GEMM(X, concat([Wq, Wk, Wv]).T) -> split last dim into q,k,v
```

Weight layout:
- PyTorch Linear weight is `[out_features, in_features]`.
- Packed weight rows should be `[Q rows, K rows, V rows]`; packed bias same order.

Failure cases:
- Predictor and pooler widths differ; pack only within one module instance.
- Attention backend requiring separate memory layout may prefer late split after reshape.

### Rewrite: 3-axis RoPE precompute for dense encoder positions

Preconditions:
- No predictor `position_mask`; dense encoder token ids are `arange(N)`.
- Fixed crop/frame bucket.

Replacement:
- Precompute frame/height/width sin/cos tables per bucket and apply vectorized pair rotations.

Failure cases:
- Predictor path with dynamic masks.
- Runtime frames/crop not matching precomputed bucket.

### Rewrite: learned-query cross-attention specialized q_len=1

Preconditions:
- Classifier pooler cross-attention, query length exactly 1.
- No attention mask.

Replacement:
- Specialized `[B,heads,1,D] x [B,heads,N,D]` attention kernel and output `[B,1,H]`.

Failure cases:
- Any future pooler with more query tokens or masks.

### Layout rewrite: local NCTHW/NTHWC fusion around patch embedding

Candidate:
- Keep public model ABI faithful as `[B,T,C,H,W]`.
- Internally fuse processor output staging, normalization, and patch extraction in an NTHWC-friendly kernel, then emit tokens `[B,N,H]`.

Required axis rewrites:
- Conv3d channel axis changes from source dim 1 after permute to last channel in optimized window.
- Flatten order must remain temporal-major, then height, then width.
- RoPE token id math depends on that token order.

No-layout-translation guards:
- Predictor token masks and RoPE position ids.
- Any external caller-provided `context_mask`/`target_mask`.
- Classification logits and label postprocessing.

## 10. Kernel fusion candidates

Highest priority:
- Tubelet patch embedding as Conv3d or WindowFlatten3D+GEMM. This is the first large video-specific op and determines token order.
- LayerNorm + Linear for Q/K/V inputs. Every block starts with LayerNorm and three same-input projections.
- Dense noncausal attention with 3-axis RoPE. Token counts are large: 8192 at 256/64 and 18432 at 384/64.
- MLP GELU block: `Linear -> GELU -> Linear`, width 4096/5120/6144 depending variant.

Medium priority:
- Packed QKV projection for encoder/pooler.
- RoPE table precompute and vectorized 3-axis application for fixed buckets.
- Classifier pooler q_len=1 cross-attention.
- Last small classifier GEMM `hidden -> 174`.

Lower priority:
- Predictor gather/sort/unsort kernels.
- DropPath training path; inference should reduce to identity.
- Output attention materialization; omit unless explicitly requested.

## 11. Runtime staging plan

Stage 1: Config/weight loader and encoder skeleton.
- Parse `VJEPA2Config`.
- Load Conv3d, LayerNorm, Linear weights.
- Reject unsupported `patch_size` list/tuple initially unless implemented.

Stage 2: Patch embedding parity.
- Implement source-faithful `[B,T,C,H,W]` -> `[B,N,H]`.
- Support fixed 256/64 and 256/16 first; add 384/64 bucket after.

Stage 3: One encoder block parity.
- LayerNorm, Q/K/V, 3-axis RoPE, dense noncausal attention, output projection, MLP.

Stage 4: Full encoder feature extraction.
- Final LayerNorm and `get_vision_features` output `[B,N,H]`.
- Stub predictor by requiring `skip_predictor=True` for initial runtime.

Stage 5: Classification head.
- Add attentive pooler self-attention layers, learned-query cross-attention, classifier.
- Validate SSV2 `174` logits.

Stage 6: Optimized kernels.
- Conv3d-to-GEMM rewrite, packed QKV, RoPE+attention fusion, q_len=1 cross-attention specialization.

Stage 7: Predictor admission.
- Implement `apply_masks`, mask token generation, sort/unsort, position-mask RoPE.
- Gate on mask shape/range and list length.

## 12. Parity and validation plan

Focused tests:
- Patch embedding: random `[B,T,3,H,W]` against PyTorch for T=16/64, crop=256, and crop=384; include `T < tubelet_size` repeat path.
- RoPE: compare DinoML 3-axis RoPE against `rotate_queries_or_keys` for head_dim 64 and 80, with dense ids and gathered predictor masks.
- Attention block: one block parity in fp32 with dropout disabled.
- Full encoder: compare `last_hidden_state` for small synthetic config first, then representative checkpoint bucket.
- Classification head: compare logits for `facebook/vjepa2-vitl-fpc16-256-ssv2` with fixed preprocessed input.
- Predictor deferred parity: default full masks should produce predictor output with `N` targets; custom context/target masks should validate gather/sort/unsort.

Recommended tolerances:
- fp32: `rtol=1e-4`, `atol=1e-4` for block/full encoder; tighter for isolated linear/LayerNorm.
- fp16/bf16: `rtol=5e-2`, `atol=5e-2` initially for full encoder attention; tune after fused attention parity is known.

End-to-end:
- Processor output parity on a fixed decoded video frame tensor can be CPU-side.
- Model graph parity should consume already-preprocessed `pixel_values_videos` to isolate runtime ops from video decode.

## 13. Performance probes

- Processor throughput: decode/frame sample/resize/crop/normalize frames/sec, separate from GPU.
- Patch embedding throughput for Conv3d path vs WindowFlatten3D+GEMM.
- Encoder block time by token count: 2048, 8192, 18432.
- Attention backend comparison: eager dense, SDPA/FlashAttention-compatible path, DinoML fused RoPE+attention.
- MLP GEMM throughput for widths 1024->4096, 1280->5120, 1408->6144.
- End-to-end encoder throughput by batch size and crop/frame bucket.
- Classification pooler cost separated from encoder.
- Memory probes for attention temporary size at 8192 and 18432 tokens; dense attention materialization is likely prohibitive if attention weights are requested.
- Predictor gather/sort overhead once predictor is admitted.

## 14. Skip/defer list

Safe to defer for first integration:
- Training, loss functions, gradient checkpointing, DropPath stochastic behavior.
- Masked predictor path, unless the product target is V-JEPA prediction instead of feature/classification inference.
- Output attentions and hidden-state recording.
- Built-in video decode/frame sampling; keep CPU/data pipeline owned.
- Dynamic arbitrary crop/frame support beyond configured buckets.
- General `argsort` lowering outside the predictor admission path.
- Quantization, tensor parallel, multi-GPU sharding.
- Any text generation, KV cache, tokenizer, or sampling controller work; not applicable.

Do not defer:
- Correct `[B,T,C,H,W]` input ABI.
- Correct NCTHW Conv3d or equivalent patch order.
- 3-axis RoPE.
- Dense noncausal attention.
- LayerNorm epsilon and GELU parity.

## 15. Final implementation checklist

- [ ] Parse `VJEPA2Config` and reject unsupported historical config fields as ignored.
- [ ] Load official HF weights for Conv3d, LayerNorm, Linear, learned query/mask tokens, classifier.
- [ ] Implement video tensor ABI `[B,T,C,H,W]` with guarded permute/rewrite to Conv3d input.
- [ ] Implement Conv3d tubelet patch embedding or guarded WindowFlatten3D+GEMM replacement.
- [ ] Implement LayerNorm eps `1e-6`.
- [ ] Implement separate and optional packed Q/K/V projections.
- [ ] Implement 3-axis RoPE with temporal/height/width slices and leftover tail.
- [ ] Implement dense noncausal MHA for large video token counts.
- [ ] Implement GELU MLP block with config-derived MLP width.
- [ ] Add full encoder final LayerNorm and `get_vision_features` parity.
- [ ] Gate first runtime to `skip_predictor=True`.
- [ ] Implement attentive pooler self-attention and learned q_len=1 cross-attention for classification.
- [ ] Add classifier logits parity for SSV2 checkpoints.
- [ ] Add patch/rope/block/full-encoder parity tests across L/H/G and 256/384 buckets.
- [ ] Benchmark patch embedding, attention, MLP, full encoder, and classifier pooler separately.
- [ ] Later: admit predictor masks with `gather`, `cat`, `argsort`, `unsort`, and position-mask RoPE guards.
