# VideoMAE Transformers family audit

## 1. Source basis

Transformers commit/version: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`.

Model family: `videomae`.

Primary DinoML runtime target for this report: `VideoMAEForVideoClassification`, with `VideoMAEModel` as the independently stageable video encoder. `VideoMAEForPreTraining` and its reconstruction decoder are documented but deferred for first inference parity.

Config sources inspected:

| Model id | Config link | Local snapshot |
|---|---|---|
| `MCG-NJU/videomae-base` | [config.json](https://huggingface.co/MCG-NJU/videomae-base/resolve/main/config.json) | `_sources/MCG-NJU__videomae-base__config.json` |
| `MCG-NJU/videomae-base-finetuned-kinetics` | [config.json](https://huggingface.co/MCG-NJU/videomae-base-finetuned-kinetics/resolve/main/config.json) | `_sources/MCG-NJU__videomae-base-finetuned-kinetics__config.json` |
| `MCG-NJU/videomae-base-finetuned-ssv2` | [config.json](https://huggingface.co/MCG-NJU/videomae-base-finetuned-ssv2/resolve/main/config.json) | `_sources/MCG-NJU__videomae-base-finetuned-ssv2__config.json` |
| `MCG-NJU/videomae-large` | [config.json](https://huggingface.co/MCG-NJU/videomae-large/resolve/main/config.json) | `_sources/MCG-NJU__videomae-large__config.json` |
| `MCG-NJU/videomae-large-finetuned-kinetics` | [config.json](https://huggingface.co/MCG-NJU/videomae-large-finetuned-kinetics/resolve/main/config.json) | `_sources/MCG-NJU__videomae-large-finetuned-kinetics__config.json` |
| `MCG-NJU/videomae-huge-finetuned-kinetics` | [config.json](https://huggingface.co/MCG-NJU/videomae-huge-finetuned-kinetics/resolve/main/config.json) | `_sources/MCG-NJU__videomae-huge-finetuned-kinetics__config.json` |
| `MCG-NJU/videomae-small-finetuned-kinetics` | [config.json](https://huggingface.co/MCG-NJU/videomae-small-finetuned-kinetics/resolve/main/config.json) | `_sources/MCG-NJU__videomae-small-finetuned-kinetics__config.json` |

Source files inspected:

- `src/transformers/models/videomae/configuration_videomae.py`
- `src/transformers/models/videomae/modeling_videomae.py`
- `src/transformers/models/videomae/image_processing_videomae.py`
- `src/transformers/models/videomae/image_processing_pil_videomae.py`
- `src/transformers/models/videomae/video_processing_videomae.py`
- `src/transformers/models/videomae/convert_videomae_to_pytorch.py`, for official variant dimensions and conversion expectations
- `src/transformers/video_processing_utils.py`, for inherited `BaseVideoProcessor` frame sampling/layout behavior

Source snapshots were written under `_sources/`. Official `preprocessor_config.json` files were accessible and snapshotted. `video_preprocessor_config.json` returned 404 for the inspected repos; current `VideoMAEVideoProcessor` falls back through image/preprocessor config loading, so this is a compatibility note, not gated access.

Any missing files or assumptions: no remote code is required. No gated/401 repos were encountered. This audit did not execute imports or model tests.

## 2. High-level architecture

VideoMAE is a video-only ViT-style encoder with a 3D tubelet patch embedding. The first inference target is video classification.

Dataflow:

```text
video decode/frame selection -> resize/center-crop/rescale/normalize -> pixel_values[B,T,C,H,W]
  -> permute to [B,C,T,H,W]
  -> non-overlap Conv3d tubelet patch embedding
  -> fixed sin/cos absolute position add
  -> noncausal transformer encoder
  -> mean-pool or first-token pool
  -> optional LayerNorm
  -> classification Linear -> logits[B,num_labels]
```

Stage decomposition:

- CPU/data pipeline: video decode or caller-supplied frames, optional frame sampling, RGB conversion, resize, center crop, rescale, normalize, output `pixel_values`.
- GPU/runtime encoder: fixed-shape `pixel_values[B,16,3,224,224]` for the representative configs, tubelet embedding, encoder layers, pooling.
- Head: classification `LayerNorm + Linear` for fine-tuned checkpoints.
- Deferred stage: masked-autoencoder pretraining decoder, mask indexing, pixel reconstruction labels, and MSE loss.

## 3. Important config dimensions

Source defaults from `VideoMAEConfig`:

| Field | Default |
|---|---:|
| `image_size` | 224 |
| `patch_size` | 16 |
| `num_channels` | 3 |
| `num_frames` | 16 |
| `tubelet_size` | 2 |
| `hidden_size` | 768 |
| `num_hidden_layers` | 12 |
| `num_attention_heads` | 12 |
| `head_dim` | `hidden_size / num_attention_heads` |
| `intermediate_size` | 3072 |
| `hidden_act` | `gelu` |
| `qkv_bias` | `True` |
| `hidden_dropout_prob` | 0.0 |
| `attention_probs_dropout_prob` | 0.0 |
| `layer_norm_eps` | 1e-12 |
| `use_mean_pooling` | `True` |
| decoder | 4 layers, hidden 384, 6 heads, intermediate 1536 |
| `norm_pix_loss` | `True` |

Representative checkpoint sweep:

| Model id | Architecture | Hidden | Layers | Heads | Head dim | MLP | Frames | Tubelet | Patches | Labels | Pooling |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `videomae-base` | pretraining | 768 | 12 | 12 | 64 | 3072 | 16 | 2 | 1568 | n/a | first token, via `use_mean_pooling=False` |
| `videomae-base-finetuned-kinetics` | classification | 768 | 12 | 12 | 64 | 3072 | 16 | 2 | 1568 | 400 from `id2label` | mean + `fc_norm` |
| `videomae-base-finetuned-ssv2` | classification | 768 | 12 | 12 | 64 | 3072 | 16 | 2 | 1568 | 174 from `id2label` | mean + `fc_norm` |
| `videomae-large` | pretraining | 1024 | 24 | 16 | 64 | 4096 | 16 | 2 | 1568 | n/a | first token, via `use_mean_pooling=False` |
| `videomae-large-finetuned-kinetics` | classification | 1024 | 24 | 16 | 64 | 4096 | 16 | 2 | 1568 | 400 from `id2label` | mean + `fc_norm` |
| `videomae-huge-finetuned-kinetics` | classification | 1280 | 32 | 16 | 80 | 5120 | 16 | 2 | 1568 | 400 from `id2label` | mean + `fc_norm` |
| `videomae-small-finetuned-kinetics` | classification | 384 | 12 | 16 | 24 | 1536 | 16 | 2 | 1568 | 400 from `id2label` | mean + `fc_norm` |

Preprocessor sweep: inspected official repos use legacy `preprocessor_config.json` with resize/crop/normalize enabled, size `224`, ImageNet mean/std, and no `video_preprocessor_config.json`. Source defaults for the modern video processor are `size={"shortest_edge":224}`, `crop_size={"height":224,"width":224}`, `do_convert_rgb=True`, `do_sample_frames=False`, and output name `pixel_values`.

## 3a. Family variation traps

- `pixel_values` semantic layout is `[batch, frames, channels, height, width]`. Modeling immediately permutes to `[batch, channels, frames, height, width]` for `Conv3d`.
- `num_frames` is effectively fixed by the positional embedding length. For default configs, `16 / tubelet_size=8` temporal tubelets and `14 * 14` spatial patches give `1568` tokens. Supplying a different frame count can make patch embeddings fail when adding fixed position embeddings.
- There is no CLS token in the current source. `use_mean_pooling=False` selects `sequence_output[:, 0]`, which is the first tubelet patch token, not a class token.
- Fine-tuned checkpoints use `use_mean_pooling=True`: mean over all encoded tokens, then `fc_norm`, then classifier.
- Pretraining checkpoints use masked visible-token selection before the encoder and a decoder that concatenates visible and mask tokens with shuffled position embeddings. That path is not needed for classification inference.
- `hidden_size % num_attention_heads` is required. Small has head dim 24; huge has head dim 80, so do not hardcode 64.
- `qkv_bias` is config-controlled and true in inspected configs.
- The code advertises SDPA, FlashAttention, and FlexAttention backend support through the generic attention interface, but VideoMAE passes no mask and sets `is_causal=False`.
- `preprocessor_config.json` exists; `video_preprocessor_config.json` does not in inspected official repos. AutoVideoProcessor may load image processor config as fallback.
- NTHWC/NHWC opportunities are optimizations only. The faithful model input remains `[B,T,C,H,W]`, with source axes preserved around `permute`, `Conv3d`, `flatten(2)`, `transpose(1,2)`, attention, and mean pooling.

## 4. Operator coverage checklist

Tensor/layout ops:

- Input shape checks for rank-5 `pixel_values`.
- `permute(0,2,1,3,4)` from `[B,T,C,H,W]` to `[B,C,T,H,W]`.
- Non-overlap 3D convolution output flatten: `Conv3d -> flatten(start_dim=2) -> transpose(1,2)`.
- Fixed position embedding add with broadcast `[1,N,H]`.
- Mean over token axis `dim=1` for classification when `use_mean_pooling=True`.
- First-token slice `[:,0]` for non-mean-pooling variants.
- Optional boolean masked token selection and reshape for pretraining.
- `cat(dim=1)` for pretraining visible/mask decoder input.
- Pretraining label patchification: view, permute, contiguous, view, var/mean over patch-pixel axis.

Neural network primitives:

- `Conv3d(C -> hidden_size, kernel=(tubelet,patch,patch), stride=(tubelet,patch,patch), bias=True)`.
- Encoder `LayerNorm(hidden_size, eps=1e-12)`.
- Linear Q/K/V: `Linear(hidden_size -> hidden_size, bias=qkv_bias)`.
- Attention output: `Linear(hidden_size -> hidden_size, bias=True)`.
- MLP: `Linear(hidden_size -> intermediate_size)`, GELU, `Linear(intermediate_size -> hidden_size)`.
- Classification head: optional `LayerNorm(hidden_size)` then `Linear(hidden_size -> num_labels)`.
- Pretraining bridge: `Linear(hidden_size -> decoder_hidden_size, bias=False)`.
- Pretraining decoder layers with decoder hidden/intermediate/head counts.
- Reconstruction head: `Linear(decoder_hidden_size -> num_channels * tubelet_size * patch_size^2)`.

Attention primitives:

- Dense noncausal self-attention over token sequence.
- MHA only: Q, K, V heads all equal `num_attention_heads`; no MQA/GQA.
- No attention mask in source forward.
- Optional backend dispatch through Transformers attention interface.

Position encoding:

- Fixed non-learned sinusoidal absolute table `[1,num_patches,hidden_size]`.
- No RoPE, ALiBi, relative bias, learned table interpolation, or temporal-specific factorization.

Preprocessing-coupled ops:

- Video decode/fetch is processor-owned.
- Optional frame sampling is processor-owned, disabled by default for VideoMAE.
- RGB conversion, resize shortest edge, center crop, rescale by `1/255`, normalize by ImageNet mean/std.

Postprocess/classification:

- Runtime graph returns logits only.
- End-to-end classifier uses caller/generation-side `argmax` and `id2label`; no NMS, boxes, masks, temporal smoothing, or softmax is applied by the model.

## 5. Layer/block breakdown

Patch embedding:

```text
pixel_values: [B,T,C,H,W]
assert C == config.num_channels
assert H,W == config.image_size
x = pixel_values.permute(0,2,1,3,4)              # [B,C,T,H,W]
x = Conv3d(C -> hidden, kernel=stride=(ts,ps,ps))
x = x.flatten(2).transpose(1,2)                  # [B,N,hidden]
x = x + sinusoid_pos[1,N,hidden]
if bool_masked_pos: x = x[~mask].reshape(B,N_visible,hidden)
```

Encoder block, repeated `num_hidden_layers`:

```text
attn_in = LayerNorm(x)
q = Linear(hidden -> hidden, bias=qkv_bias)(attn_in).view(B,N,heads,head_dim).transpose(1,2)
k = Linear(hidden -> hidden, bias=qkv_bias)(attn_in).view(B,N,heads,head_dim).transpose(1,2)
v = Linear(hidden -> hidden, bias=qkv_bias)(attn_in).view(B,N,heads,head_dim).transpose(1,2)
ctx = softmax((q @ k^T) * head_dim^-0.5, dim=-1) @ v
ctx = ctx.transpose(1,2).reshape(B,N,hidden)
x = x + Linear(hidden -> hidden)(ctx)
mlp_in = LayerNorm(x)
mlp = Linear(hidden -> intermediate)(mlp_in)
mlp = GELU(mlp)
x = x + Linear(intermediate -> hidden)(mlp)
```

Classification head:

```text
tokens = VideoMAEModel(pixel_values).last_hidden_state
if use_mean_pooling:
    pooled = LayerNorm(tokens.mean(dim=1))
else:
    pooled = tokens[:, 0]
logits = Linear(hidden -> num_labels)(pooled)
```

Pretraining decoder, deferred:

```text
visible = encoder_to_decoder(encoded_visible)
pos_visible = pos[~mask].reshape(B,N_visible,Dd)
pos_mask = pos[mask].reshape(B,N_mask,Dd)
x_full = cat([visible + pos_visible, mask_token + pos_mask], dim=1)
x = decoder_transformer(x_full)
x = x[:, -N_mask:]
logits = Linear(decoder_hidden -> C * tubelet * patch^2)(LayerNorm(x))
loss = MSE(logits, patchified_masked_pixel_targets)
```

## 6. Attention requirements

VideoMAE attention is encoder-only, dense, noncausal self-attention.

| Field | Requirement |
|---|---|
| Causal | No |
| Self/cross | Self-attention only |
| Heads | MHA, `num_attention_heads` Q/K/V heads |
| Head dim | `hidden_size / num_attention_heads` |
| Q/K/V widths | all equal `hidden_size` |
| Query/key lengths | square attention over token count `N`, typically 1568 for unmasked classification |
| Masking | none for classification; pretraining removes masked tokens before encoder |
| Packed/varlen | not used |
| Sliding/local | not used |
| Position interaction | position added before attention |
| KV cache | not applicable |
| Flash/SDPA | compatible as noncausal, no-mask dense attention if backend accepts shape/head dim |

Important source math order: scale is `head_dim ** -0.5`, applied to QK scores before softmax. Dropout is zero in eval and config defaults are zero.

## 7. Position encoding and custom math

The source builds a fixed sin/cos table with NumPy during module construction:

```python
def videomae_sincos(n_position, d_hid):
    table[pos, j] = pos / (10000 ** (2 * floor(j / 2) / d_hid))
    table[:, 0::2] = sin(table[:, 0::2])
    table[:, 1::2] = cos(table[:, 1::2])
    return table[None, :, :]
```

This can be precomputed as a constant for a fixed config. It depends on `num_patches` and hidden size, not on batch contents. For pretraining, the decoder has a separate fixed table with `decoder_hidden_size`.

There is no position interpolation in current source. DinoML should reject mismatched `num_frames`, `image_size`, or `patch_size` unless it implements an explicit positional-table regeneration policy.

## 8. Preprocessing and input packing

Model-coupled runtime tensor:

```text
pixel_values: float tensor [batch, num_frames, num_channels, image_size, image_size]
representative configs: [B,16,3,224,224]
```

Processor behavior:

- `VideoMAEImageProcessor` accepts a single video as a list of frames or a batch as list of lists.
- Torchvision path groups frames by shape, resizes, center crops, rescales, normalizes, then stacks frames per video to `[T,C,H,W]`.
- PIL path processes frames one-by-one and lets `BatchFeature` tensor conversion pack the nested output.
- `VideoMAEVideoProcessor` inherits `BaseVideoProcessor`, renames `pixel_values_videos` to `pixel_values`, and sets `model_input_names=["pixel_values"]`.
- Modern video processor default has `do_sample_frames=False`; a caller must provide the correct number of frames or explicitly enable sampling with `num_frames` or `fps`.
- If input video tensors are channels-last, `BaseVideoProcessor` converts them to channels-first per frame.

First integration recommendation: keep preprocessing in CPU/data pipeline and require already packed `pixel_values[B,T,C,H,W]` at the DinoML graph boundary. Later, add an optional video processor compatibility shim.

## 9. Graph rewrite / lowering opportunities

### Rewrite: non-overlap Conv3d tubelet embedding -> Linear/GEMM

Source pattern:

```text
[B,T,C,H,W] -> permute [B,C,T,H,W]
Conv3d(kernel=stride=(tubelet,patch,patch), groups=1)
flatten(2).transpose(1,2)
```

Replacement:

```text
TubeletWindowFlatten [B,N,tubelet*patch*patch*C]
MatMul(flattened_windows, conv.weight.reshape(hidden, -1).T)
BiasAdd
```

Preconditions:

- `kernel_size == stride == (tubelet_size, patch_h, patch_w)`.
- `padding == 0`, `dilation == 1`, `groups == 1`.
- `T % tubelet_size == 0`, `H % patch_h == 0`, `W % patch_w == 0`.
- Flatten order must match source `Conv3d` over `[C,T,H,W]` after the source permute.
- Weight transform: `W_linear = conv.weight.reshape(hidden_size, C * tubelet * patch_h * patch_w)`.

Failure cases: nonstandard padding/dilation/groups, dynamic sizes without divisibility guards, or layout pass changing flatten order.

Parity test sketch: compare Conv3d path and lowered GEMM for random `[2,16,3,224,224]` and small synthetic sizes with non-square image/patch tuple configs.

### Rewrite: QKV projections fuse as packed GEMM

Source pattern: three independent `Linear(hidden -> hidden)` on the same normalized input.

Replacement: one packed GEMM producing `[q,k,v]`, then split in source order `query`, `key`, `value`.

Preconditions:

- All three projections have same input/output width.
- Bias policy matches `qkv_bias`.
- Weight pack order must be Q then K then V.

Failure cases: config with disabled bias requires no packed bias; future source changes to fused storage must be audited separately.

### Rewrite: classification mean-pool + LayerNorm + Linear

Source pattern:

```text
tokens.mean(dim=1) -> LayerNorm(hidden) -> Linear(hidden,num_labels)
```

Replacement: fused row reduction plus optional LayerNorm/classifier epilogue.

Preconditions: `use_mean_pooling=True`, no hidden-state consumers requiring token output.

Failure cases: callers requesting hidden states, attentions, or base encoder output must preserve full tokens.

### Layout opportunity: NTHWC/NHWC guarded patch region

Opportunity: avoid materializing `[B,T,C,H,W] -> [B,C,T,H,W]` if a tubelet embedding kernel directly consumes `[B,T,H,W,C]` or `[B,T,C,H,W]`.

Required guards:

- Preserve semantic input ABI unless the whole processor-to-embedding region is controlled.
- Rewrite Conv3d/window-flatten axes explicitly.
- Ensure downstream token layout remains `[B,N,H]`.

This is a guarded fusion/layout pass only, not a default graph translation.

## 10. Kernel fusion candidates

Highest priority:

- Tubelet patch embedding as fused window-flatten + GEMM or direct non-overlap Conv3d kernel. This dominates input bandwidth and fixes the tokenization contract.
- Encoder LayerNorm + QKV packed projection. This repeats in every layer and maps cleanly to existing GEMM work.
- Dense noncausal attention backend for long-ish vision sequences (`N=1568`): SDPA/FlashAttention parity with no mask and no cache.
- MLP GELU block: `Linear -> GELU -> Linear`, with GEMM epilogue opportunities.

Medium priority:

- Mean-pool + `fc_norm` + classifier for classification-only artifacts.
- Position-add fused into patch embedding output write.
- Dropout elimination in inference, since checkpoint configs use zero dropout and eval disables it.

Lower priority:

- Pretraining boolean mask gather/scatter and decoder path.
- Patchified reconstruction label generation and normalized pixel loss.
- Processor GPU kernels for resize/crop/normalize, because first integration can keep preprocessing outside the runtime graph.

## 11. Runtime staging plan

Stage 1: parse config and load classification weights for one fine-tuned base checkpoint. Validate shapes and reject unsupported frame/image sizes.

Stage 2: implement encoder-only parity with faithful `[B,T,C,H,W]` input, Conv3d tubelet projection, fixed position add, dense encoder blocks, and full token output.

Stage 3: add classification head parity for `use_mean_pooling=True` Kinetics/SSV2 checkpoints.

Stage 4: add optimized tubelet embedding rewrite and packed QKV rewrite behind exact guards.

Stage 5: route dense noncausal attention through DinoML's best available SDPA/FlashAttention path, with eager GEMM/softmax fallback for parity.

Stage 6: add small/large/huge sweep, dynamic batch, fp16/bf16 tests if weights/dtypes require it.

Stage 7: optionally add pretraining `VideoMAEForPreTraining` decoder and masked reconstruction path.

Stubs acceptable initially: video decode/frame sampling, image processor pipeline, training losses, hidden-state/attention recording, and pretraining decoder.

## 12. Parity and validation plan

- Config parser tests for base, small, large, huge, Kinetics, and SSV2 configs.
- Patch embedding parity against PyTorch for random inputs and converted weights, including flattened token order.
- Single encoder layer parity in fp32 for small synthetic config.
- Full encoder parity for base shape `[1,16,3,224,224]`, comparing final hidden states.
- Classification parity for `videomae-base-finetuned-kinetics`: logits `[B,400]`; for SSV2: logits `[B,174]`.
- Pooling parity for both `use_mean_pooling=True` and `False`.
- Attention backend parity: eager GEMM/softmax vs selected SDPA/Flash backend with no mask.
- Layout rewrite parity: Conv3d path vs window-flatten GEMM path with exact flatten-order checks.
- Preprocessor compatibility smoke: processor-produced `pixel_values` has `[B,T,C,H,W]`, normalized channels, and correct frame count.

Recommended tolerances: fp32 `atol=1e-4, rtol=1e-4` for layer/logit parity; fp16/bf16 use model-output tolerances derived from one-block and full-model reference drift, starting around `atol=1e-2, rtol=1e-2`.

## 13. Performance probes

- CPU video preprocessing throughput: decode, resize/crop, normalize, and pack separately.
- Tubelet embedding kernel latency and memory bandwidth for `[B,16,3,224,224]`.
- Encoder throughput by batch size for base/large/huge.
- Attention backend comparison at `N=1568`, head dims 24, 64, and 80.
- QKV packed projection vs three-GEMM baseline.
- MLP GEMM/GELU throughput by hidden/intermediate sizes.
- Classification head overhead with and without returning full hidden states.
- End-to-end requests/sec with preprocessing outside runtime vs prepacked tensors.
- Memory use for full token activations across base/large/huge and batch sweep.

## 14. Skip/defer list

- Training and gradient checkpointing.
- `VideoMAEForPreTraining` masked decoder, reconstruction labels, and MSE loss.
- Processor-owned video decode and frame sampling inside DinoML runtime.
- Returning attentions/hidden states for optimized classification-only artifacts.
- Non-default image sizes, patch sizes, tubelet sizes, and frame counts until guarded positional and embedding shape support exists.
- Multi-GPU/tensor parallel execution.
- Quantization or packed-weight formats; current source uses ordinary dense PyTorch modules.

## 15. Final implementation checklist

- [ ] Parse `VideoMAEConfig` and reject unsupported source variants.
- [ ] Load dense Conv3d, Linear, LayerNorm, mask token, and classifier weights.
- [ ] Implement faithful `[B,T,C,H,W]` input contract and shape guards.
- [ ] Implement non-overlap Conv3d tubelet embedding or guarded equivalent.
- [ ] Precompute fixed sin/cos position constants.
- [ ] Implement encoder LayerNorm, QKV MHA, attention output, residuals, GELU MLP.
- [ ] Implement classification pooling modes and classifier head.
- [ ] Add base/small/large/huge config-shape tests.
- [ ] Add patch embedding parity tests.
- [ ] Add single-layer and full-encoder parity tests.
- [ ] Add classification logits parity for Kinetics and SSV2 checkpoints.
- [ ] Add guarded Conv3d-to-GEMM tubelet rewrite.
- [ ] Add packed QKV projection rewrite.
- [ ] Benchmark preprocessing, tubelet embedding, attention, MLP, and full encoder.
