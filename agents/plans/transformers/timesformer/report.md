# TimeSformer Transformers Audit

## 1. Source basis

Transformers commit/version: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4` from `X:/H/transformers`.

Model id: primary target `facebook/timesformer-base-finetuned-k400`; representative public configs also inspected for K600, Something-Something-v2, and high-resolution variants.

Config source: Hugging Face `config.json` and `preprocessor_config.json` from the public `facebook/timesformer-*` repos, plus `TimesformerConfig` defaults. Snapshots are under `_sources/`.

Source files inspected:

- `src/transformers/models/timesformer/configuration_timesformer.py`
- `src/transformers/models/timesformer/modeling_timesformer.py`
- `src/transformers/models/timesformer/convert_timesformer_to_pytorch.py`
- `src/transformers/models/videomae/image_processing_videomae.py` because TimeSformer checkpoint preprocessors use `VideoMAEImageProcessor`.

Any missing files or assumptions:

- The in-library TimeSformer directory has no dedicated image/video processor file; preprocessing is delegated to VideoMAE image processing.
- `facebook/timesformer-large-finetuned-{k400,k600,ssv2}` returned 401 Unauthorized for `config.json` and `preprocessor_config.json` in this environment. Links: [large-k400](https://huggingface.co/facebook/timesformer-large-finetuned-k400), [large-k600](https://huggingface.co/facebook/timesformer-large-finetuned-k600), [large-ssv2](https://huggingface.co/facebook/timesformer-large-finetuned-ssv2). Access would resolve exact dimensions/preprocessor settings. The conversion script source-derived fallback says `"large"` sets `num_frames = 96`, but that is not a fetched config fact here.

## 2. High-level architecture

TimeSformer is a video encoder for classification. It is a ViT-like patch encoder over video clips, with optional attention modes. The official public checkpoints use `attention_type="divided_space_time"`.

Dataflow:

```text
decoded video frames -> VideoMAEImageProcessor -> pixel_values[B,T,C,H,W]
-> per-frame Conv2d patch embedding
-> spatial position embedding + temporal embedding
-> repeated TimeSformer blocks
-> final LayerNorm -> CLS token -> Linear classifier logits
```

Stage decomposition:

- CPU/data pipeline: decode/sample frames, resize shortest edge, center crop, rescale, normalize, stack frames.
- GPU/runtime graph: patch projection, embedding additions/interpolation if dynamic sizes are admitted, encoder blocks, classifier.
- Independently validatable pieces: preprocessing output contract, patch embedding, one divided-attention block, full encoder hidden states, classification head logits.

Primary runtime target: `TimesformerForVideoClassification`. `TimesformerModel` without the classifier is useful as an encoder-only target. Training losses are not needed for inference.

## 3. Important config dimensions

Source defaults:

| Field | Default / source |
| --- | --- |
| `model_type` | `timesformer` |
| `image_size` | `224` |
| `patch_size` | `16` |
| `num_channels` | `3` |
| `num_frames` | `8` |
| `hidden_size` | `768` |
| `num_hidden_layers` | `12` |
| `num_attention_heads` | `12` |
| `head_dim` | `64` inferred as `hidden_size // num_attention_heads` |
| `intermediate_size` | `3072` |
| `hidden_act` | `gelu` |
| `qkv_bias` | `true` |
| `layer_norm_eps` | `1e-6` |
| `attention_type` | `divided_space_time`; source also implements `space_only`, `joint_space_time` |
| dropout/drop path | defaults `0.0` / `0` |
| cache support | none; encoder-only non-causal video model |

Representative checkpoint sweep:

| Checkpoint | Config source | Task labels | Image / crop | Frames | Tokens after embedding | Attention type |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `facebook/timesformer-base-finetuned-k400` | fetched config | 400 | 224 | 8 | `1 + 14*14*8 = 1569` | divided space-time |
| `facebook/timesformer-base-finetuned-k600` | fetched config | 600 | 224 | 8 | 1569 | divided space-time |
| `facebook/timesformer-base-finetuned-ssv2` | fetched config | 174 | 224 | 8 | 1569 | divided space-time |
| `facebook/timesformer-hr-finetuned-k400` | fetched config | 400 | 448 | 16 | `1 + 28*28*16 = 12545` | divided space-time |
| `facebook/timesformer-hr-finetuned-k600` | fetched config | 600 | 448 | 16 | 12545 | divided space-time |
| `facebook/timesformer-hr-finetuned-ssv2` | fetched config | 174 | 448 | 16 | 12545 | divided space-time |
| `facebook/timesformer-large-finetuned-*` | gated/401; source fallback only | 400/600/174 by script naming | unknown fetched | source conversion sets 96 | unknown | likely divided space-time, not verified from config |

## 3a. Family variation traps

- `attention_type` changes operator structure. `divided_space_time` has temporal attention plus spatial attention in every block. `space_only` and `joint_space_time` use only one attention module per block over the incoming token sequence.
- The encoder layer uses `config.num_frames` and `config.image_size // config.patch_size` to derive reshape sizes. Although embeddings can interpolate time/spatial embeddings for mismatched inputs, DinoML should initially require runtime `T == config.num_frames`, square integer `image_size`, scalar patch size, and `H == W == config.image_size`.
- `image_size` and `patch_size` are annotated as `int | list | tuple`, but layer code performs `config.image_size // config.patch_size`. Non-scalar configs should be rejected or separately audited.
- The model input is semantic `B,T,C,H,W` NCTHW. Processor inputs are frame lists and end as per-video tensors `[T,C,H,W]`. A layout pass may optimize internal convs, but initial translation must preserve these axes.
- Divided attention repeatedly reshapes between token orderings: patch-major temporal order and frame-major spatial order. These regions need a no-layout-translation guard unless every reshape/permute consumer is rewritten together.
- Time and spatial positional embeddings are resizeable with nearest interpolation in source. First integration can avoid this path by requiring matching `num_frames` and image size.
- Classification head uses the final normalized CLS token only: `outputs[0][:, 0] -> Linear(hidden_size, num_labels)`.
- QKV is a single packed `Linear(hidden_size -> 3*hidden_size)` with split order `q, k, v`.
- There is no causal mask, KV cache, RoPE, ALiBi, relative bias, GQA/MQA, MoE, or tokenizer-controlled behavior.

## 4. Operator coverage checklist

Tensor/layout ops:

- Input validation for `pixel_values[B,T,C,H,W]`.
- Reshape `B*T,C,H,W`, flatten spatial patches, transpose to token-major.
- `cat` for CLS token insertion.
- `expand` and `repeat` for CLS token.
- Reshape/permute/view chains for temporal and spatial packing.
- Mean reduction over frame axis for divided-attention CLS aggregation.
- Optional nearest interpolation for spatial and time embeddings.

Neural network primitives:

- Non-overlap `Conv2d(3 -> hidden_size, kernel=patch_size, stride=patch_size, bias=True)` patch projection.
- LayerNorm epsilon `1e-6`.
- Packed QKV Linear `768 -> 2304` with bias in public configs.
- Attention output Linear `768 -> 768`.
- MLP Linear `768 -> 3072`, GELU, Linear `3072 -> 768`.
- Dropout and stochastic depth are identity in eval for fetched public configs.
- Classifier Linear `768 -> num_labels`.

Attention primitives:

- Dense noncausal MHA only.
- Temporal attention in divided mode: batch is `B*Hp*Wp`, sequence length `T`, heads `12`, head dim `64`.
- Spatial attention in divided mode: batch is `B*T`, sequence length `1 + Hp*Wp`, heads `12`, head dim `64`.
- Joint mode: batch `B`, sequence `1 + T*Hp*Wp`.
- Space-only mode: source applies regular attention to the provided sequence; first integration can reject until a checkpoint requires it.

Preprocessing-coupled ops:

- Resize shortest edge, center crop, rescale by `1/255`, normalize with mean/std `[0.45, 0.45, 0.45]` and `[0.225, 0.225, 0.225]` for fetched checkpoints.
- Stack frames into `pixel_values`.

## 5. Layer/block breakdown

Patch and embeddings:

```text
pixel_values[B,T,C,H,W]
-> reshape[B*T,C,H,W]
-> Conv2d(C -> D, kernel=P, stride=P)
-> flatten(2).transpose(1,2) = patch_tokens[B*T, Hp*Wp, D]
-> cat(cls[B*T,1,D], patch_tokens)
-> add spatial position embedding[1, 1+Hp*Wp, D]
-> for divided/joint modes:
   remove cls, reshape B,T,Hp*Wp,D -> B*Hp*Wp,T,D
   add time embedding[1,T,D]
   reshape back B,T*Hp*Wp,D and prepend one clip-level cls[B,1,D]
```

Divided space-time block, repeated `num_hidden_layers`:

```text
patches = hidden[:, 1:, :]
temporal = reshape patches -> [B*Hp*Wp, T, D]
temporal = temporal + temporal_dense(TemporalMHA(LN(temporal)))
spatial = reshape temporal -> [B*T, Hp*Wp, D]
spatial = cat(repeated_cls[B*T,1,D], spatial)
spatial_residual = SpatialMHA(LN(spatial))
cls = mean(spatial_residual[:,0,:].reshape[B,T,D], dim=1, keepdim=True)
patch_residual = reshape spatial_residual[:,1:,:] -> [B,T*Hp*Wp,D]
hidden = cat(initial_cls, temporal) + cat(cls, patch_residual)
hidden = hidden + MLP(LN(hidden))
```

Output:

```text
sequence = final LayerNorm(hidden)
logits = Linear(sequence[:, 0, :])
```

All projection Linear modules have bias in source except `classifier` may be `Identity` when `num_labels <= 0`.

## 6. Attention requirements

Required attention is encoder-style noncausal self-attention. There is no generation decode, no KV cache, and no attention mask in the TimeSformer forward path.

Divided mode requirements:

- MHA, not GQA/MQA.
- Q/K/V widths all equal `hidden_size`; `head_dim = hidden_size // num_attention_heads`.
- Temporal attention has rectangular-free sequence `T x T` per spatial patch.
- Spatial attention has sequence `(1 + Hp*Wp) x (1 + Hp*Wp)` per frame.
- Attention math is eager source order: packed QKV Linear, reshape/permute to `[3,B,H,S,head_dim]`, matmul scores, scale by `head_dim**-0.5`, softmax over last axis, dropout, matmul with V, transpose/reshape, output Linear.
- No masks, no packed/varlen support, no sliding window, no relative position bias.
- FlashAttention/SDPA compatibility is straightforward for eval if no attention weights are requested and dropout is disabled; temporal and spatial attention should be separate calls because their batch and sequence axes differ.

## 7. Position encoding and custom math

Position encoding is learned absolute spatial position plus learned temporal embedding. No RoPE/ALiBi.

Source-equivalent embedding resize sketch:

```python
def resize_spatial_pos(pos, hp, wp):
    cls = pos[:, :1, :]
    patch = pos[:, 1:, :].transpose(1, 2)
    old = int(patch.shape[-1] ** 0.5)
    patch = patch.reshape(1, -1, old, old)
    patch = interpolate(patch, size=(hp, wp), mode="nearest")
    return cat([cls, patch.flatten(2).transpose(1, 2)], dim=1)

def resize_time_pos(time_pos, frames):
    return interpolate(time_pos.transpose(1, 2), size=frames, mode="nearest").transpose(1, 2)
```

For first integration, prefer static admission that avoids interpolation. If dynamic frame/image sizes are admitted later, the nearest interpolation path must be parity-tested exactly, including source axis order.

## 8. Preprocessing and input packing

The fetched preprocessor configs use `VideoMAEImageProcessor`:

- input: one video as a list of frames or batch of videos as list of frame lists.
- resize shortest edge to `224` or `448` depending on checkpoint.
- center crop to square `224x224` or `448x448`.
- rescale by `0.00392156862745098`.
- normalize by mean/std `[0.45,0.45,0.45]` / `[0.225,0.225,0.225]`.
- output: `pixel_values` as a list of tensors `[T,C,H,W]`, and with tensor conversion as `[B,T,C,H,W]`.

Frame sampling is not implemented by the processor or model. Example code samples fixed indices with PyAV before preprocessing. DinoML should treat decode and frame sampling as caller/data-pipeline responsibilities for first integration.

## 9. Graph rewrite / lowering opportunities

### Rewrite: non-overlap Conv2d patch embed -> Linear/GEMM

Source pattern:

```text
reshape[B*T,C,H,W] -> Conv2d(C,D,kernel=P,stride=P,padding=0) -> flatten patches
```

Replacement:

```text
WindowFlatten[B*T, Hp*Wp, C*P*P] -> MatMul(weight_flat.T) -> BiasAdd
```

Preconditions:

- `kernel_size == stride == patch_size`.
- `padding == 0`, `dilation == 1`, `groups == 1`.
- `H % P == 0`, `W % P == 0`.
- Source NCHW patch pixel order is preserved in the flattened window.
- Bias is included.

Weight transform:

```python
w = conv.weight.reshape(hidden_size, num_channels * patch_h * patch_w)
```

Failure cases: non-scalar patch config, non-contiguous or translated layout without a matching window flatten rewrite, dynamic image size without guards.

Parity test sketch: compare patch token tensor before positional additions for random `B,T,C,H,W` and fetched checkpoint dimensions.

### Rewrite: packed QKV Linear -> fused QKV provider

Preconditions:

- Source weight is `[3*D, D]` in PyTorch Linear convention.
- Split order is `q, k, v`.
- `D % num_heads == 0`.
- Bias shape is `[3*D]` when `qkv_bias=True`.

Replacement: one fused QKV GEMM producing three logical tensors or one packed tensor with manifest-visible split metadata.

Failure cases: output attentions requested may still be compatible, but dropout/training is out of first inference scope.

### Rewrite: eval dropout/drop-path removal

Preconditions:

- Inference/eval mode.
- Dropout probabilities may be nonzero in config, but source `Dropout` and `DropPath` are identity outside training.

Replacement: remove dropout and stochastic-depth nodes from inference graph.

### Layout guard: divided-attention reshape island

Source pattern alternates:

```text
[B, T*Hp*Wp, D] <-> [B*Hp*Wp, T, D] <-> [B*T, Hp*Wp, D]
```

Preconditions for any layout rewrite:

- Rewrite the entire island together.
- Preserve token order: source stores patch tokens grouped by frame after embeddings, then repacks to patch-major for temporal attention.
- Rewrite all concat, repeat, mean, and reshape axes together.

Failure cases: local NHWC conversion of only the Conv2d or only one attention side can silently scramble time/space token order.

## 10. Kernel fusion candidates

Highest priority:

- Patch embedding Conv2d-to-GEMM/window flatten for throughput and reuse of GEMM providers.
- LayerNorm + packed QKV projection around temporal and spatial attention.
- SDPA/FlashAttention for temporal and spatial dense MHA separately.
- MLP fusion: Linear + GELU + Linear, with bias.

Medium priority:

- CLS repeat/cat plus spatial packing specialization for divided attention.
- Temporal/spatial reshape-permute elimination inside a layout-aware graph pass.
- Final LayerNorm + CLS slice + classifier Linear for classification-only artifacts.

Lower priority:

- Nearest positional/time embedding interpolation kernels, because first integration can statically avoid them.
- Attention-output materialization optimizations for `output_attentions=True`; this is optional and memory-heavy.

## 11. Runtime staging plan

Stage 1: parse config/preprocessor metadata and load weights for `facebook/timesformer-base-finetuned-k400`; require static `B,T,C,H,W` with `T=8`, `H=W=224`.

Stage 2: implement patch embedding and embeddings parity, including CLS/position/time additions without interpolation.

Stage 3: implement one divided-attention block parity with explicit reshape/permute operations and dense MHA.

Stage 4: run full encoder parity and classifier logits parity for base public checkpoints.

Stage 5: add high-resolution static admission for `448/16` checkpoints after memory/performance probes.

Stage 6: enable graph rewrites: Conv2d patch-to-GEMM, fused QKV, SDPA/FlashAttention, MLP fusion.

Stage 7: decide whether to admit dynamic frame/image interpolation and non-divided attention modes.

Initially stub or reject training losses, `output_attentions=True`, dynamic interpolation, non-scalar image/patch configs, `space_only`, `joint_space_time`, and gated large checkpoints.

## 12. Parity and validation plan

- Processor parity: given a small fixed list of RGB frames, compare `pixel_values` shape and numeric values against `VideoMAEImageProcessor`.
- Patch embedding parity: random `pixel_values[B,T,3,224,224]`, compare Conv2d output after flatten/transpose.
- Embedding parity: compare full embeddings after CLS, spatial position, and time embeddings.
- One-block parity: compare hidden states after layer 0 for divided mode with fp32 tolerance around `1e-4` absolute / `1e-4` relative.
- Full encoder parity: compare `last_hidden_state` for base K400 fp32.
- Classification parity: compare logits and argmax for fetched public checkpoint configs.
- High-resolution parity: repeat for HR K400 after memory guard is validated.
- Negative admission tests: reject wrong frame count, wrong input axis order, non-square/non-scalar config, inaccessible large configs unless supplied by user.

For fp16/bf16 optimized paths, compare against PyTorch fp32 source with looser tolerances after attention/MLP fusion, e.g. `1e-2` range depending on backend accumulation.

## 13. Performance probes

- Preprocessing throughput: resize/crop/normalize/frame-stack frames per second.
- Patch embedding throughput by resolution: `224x8` and `448x16`.
- Temporal attention probe: batch `B*Hp*Wp`, sequence `T`.
- Spatial attention probe: batch `B*T`, sequence `1+Hp*Wp`.
- Full block throughput by batch size and resolution.
- Encoder-only throughput and memory peak for base vs HR.
- Classifier-only overhead after encoder.
- Attention backend comparison: eager matmul/softmax vs SDPA/FlashAttention for temporal and spatial separately.
- Conv patch lowering comparison: Conv2d vs im2col/GEMM provider.

## 14. Skip/defer list

- Training losses and gradient checkpointing.
- Dropout/stochastic depth training behavior.
- `output_attentions=True` as a required production ABI.
- Dynamic positional/time interpolation.
- `space_only` and `joint_space_time` until a real checkpoint target needs them.
- Gated large checkpoints until configs/preprocessors are accessible.
- Multi-GPU/tensor parallel and quantization.
- End-to-end video decode and frame sampling inside DinoML runtime.

## 15. Final implementation checklist

- [ ] Parse `TimesformerConfig` with static scalar image/patch admission.
- [ ] Parse `VideoMAEImageProcessor` metadata for public checkpoints.
- [ ] Load weights, preserving packed QKV split order `q,k,v`.
- [ ] Implement `pixel_values[B,T,C,H,W]` input contract.
- [ ] Implement patch Conv2d embedding and optional Conv2d-to-GEMM rewrite.
- [ ] Implement CLS, spatial position, and time embeddings.
- [ ] Implement divided temporal attention packing.
- [ ] Implement divided spatial attention packing and CLS frame-mean aggregation.
- [ ] Implement LayerNorm, dense MHA, GELU MLP, residual adds, final LayerNorm.
- [ ] Implement classifier head from final CLS token.
- [ ] Add one-block, full-encoder, and logits parity tests.
- [ ] Add admission rejects for wrong frame/image/axis/config variants.
- [ ] Benchmark preprocessing, patch embed, temporal attention, spatial attention, full encoder.
