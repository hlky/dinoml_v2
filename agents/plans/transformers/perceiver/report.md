# DinoML Transformers Audit: perceiver

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: deepmind/language-perceiver, deepmind/vision-perceiver-{learned,fourier,conv}, deepmind/optical-flow-perceiver, deepmind/multimodal-perceiver
Config source: HF config.json/preprocessor/tokenizer snapshots under _sources/hf_configs
Source files inspected:
- X:/H/transformers/src/transformers/models/perceiver/modeling_perceiver.py
- X:/H/transformers/src/transformers/models/perceiver/configuration_perceiver.py
- X:/H/transformers/src/transformers/models/perceiver/image_processing_perceiver.py
- X:/H/transformers/src/transformers/models/perceiver/image_processing_pil_perceiver.py
- X:/H/transformers/src/transformers/models/perceiver/tokenization_perceiver.py
Any missing files or assumptions: no gated/401/403 files found. Some official repos omit preprocessor/tokenizer files because that modality does not use them. Primary DinoML target recommended: image classification with conv/Fourier preprocessing, then language MLM and multimodal autoencoding as separate staged targets.
```

Source URLs:

- `https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/perceiver/modeling_perceiver.py`
- `https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/perceiver/configuration_perceiver.py`
- `https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/perceiver/image_processing_perceiver.py`
- `https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/perceiver/tokenization_perceiver.py`

## 2. High-level architecture

Perceiver is a modality-flexible encoder plus optional task decoder, not an autoregressive decoder. The shared core is:

```text
modality preprocessing -> latent array -> input-to-latent cross-attention -> repeated latent self-attention -> optional latent-to-query cross-attention decoder -> task logits/postprocessing
```

Stage decomposition:

- CPU/data pipeline: byte tokenizer for language; image center-crop/resize/rescale/normalize for vision; optical-flow patch extraction is expected before the model example; multimodal input assembly provides image/audio/label tensors.
- GPU/runtime preprocessing: text embeddings/positions, image conv or patch/pixel flattening, Fourier/trainable position encodings, modality padding/masking.
- Encoder: learned latents `[B, num_latents, d_latents]`, one input cross-attention, then `num_blocks * num_self_attends_per_block` latent self-attention layers.
- Decoder: classification, masked LM, optical-flow, or multimodal autoencoding query construction followed by another cross-attention from decoder queries to latents.
- No autoregressive prefill/decode path and no KV cache.

## 3. Important config dimensions

Representative checkpoint sweep:

| checkpoint | architecture | num_latents | d_latents | input d_model / preproc channels | blocks x self-attends | self heads | cross heads | qk/v config | task surface |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| `deepmind/language-perceiver` | `PerceiverForMaskedLM` | 256 | 1280 | 768 | 1 x 26 | 8 | 8 | qk=256, v=1280 | byte MLM, 2048 positions, vocab 262 |
| `deepmind/vision-perceiver-learned` | image learned | 512 | 1024 | 512 | 8 x 6 | 8 | 1 | qk/v null -> source defaults | ImageNet classification, learned 224x224 positions |
| `deepmind/vision-perceiver-fourier` | image Fourier | 512 | 1024 | 261 | 8 x 6 | 8 | 1 | qk/v null -> source defaults | ImageNet classification, pixel + 2D Fourier |
| `deepmind/vision-perceiver-conv` | image conv | 512 | 1024 | 322 | 8 x 6 | 8 | 1 | qk/v null -> source defaults | ImageNet classification, conv stem + 2D Fourier |
| `deepmind/optical-flow-perceiver` | optical flow | 2048 | 512 | 322 | 1 x 24 | 16 | 1 | qk/v null -> source defaults | dense flow `[B,368,496,2]` |
| `deepmind/multimodal-perceiver` | multimodal autoencoding | 784 | 512 | 704 | 1 x 8 | 8 | 1 | qk/v null -> source defaults | image/audio/label reconstruction |

Important source defaults and derived dimensions:

| field | value / rule | source impact |
|---|---|---|
| `hidden_act` | `gelu` | FFN activation |
| `layer_norm_eps` | `1e-12` | all Perceiver LayerNorms |
| `attention_probs_dropout_prob` | 0.1 in configs | dropout is source behavior but disabled by eval |
| `cross_attention_shape_for_attention` | `kv` | cross-attention q/k width defaults to input channel width when `qk_channels=None` |
| `self_attention_widening_factor` | 1 in sampled configs | FFN is `Linear(d -> d)` then `Linear(d -> d)` |
| `cross_attention_widening_factor` | 1 | same for cross-attention FFN |
| image Fourier position width | `dims + 2*dims*num_bands` | 2D with 64 bands gives 258; plus RGB gives 261 |
| conv image preproc channels | Conv output 64 + Fourier 258 = 322 | source `d_model=322` |
| learned image preproc channels | Conv1x1 3->256 plus projected pos 256 = 512 | source `d_model=512` |
| multimodal audio samples | `num_frames * audio_samples_per_frame = 30720` | audio patches `30720 / 16 = 1920` |
| cache support | none | encoder/decoder cross-attention only, no generation cache ABI |

## 3a. Family variation traps

- The same config class covers language, image, optical-flow, and multimodal autoencoding. Do not infer the runtime graph from `model_type="perceiver"` alone.
- `d_model` is the preprocessor output channel count, not always a raw hidden size. For image Fourier it is `3 + 2 + 2*2*64 = 261`; for conv it is `64 + 258 = 322`.
- With `qk_channels=None`, source cross-attention q/k width is selected from `kv_dim` because `cross_attention_shape_for_attention="kv"`. Self-attention q/k/v width defaults to `d_latents`.
- Language explicitly overrides q/k width to 256 and value width to 1280 in the encoder config; masked-LM decoder separately uses q/k width 256 and value width 768.
- Decoder cross-attention is noncausal and query-driven; it is not a text generation decoder.
- Layout is mixed: image processors/model entry are NCHW or BTCHW; internal Perceiver token streams are channel-last `[B, N, C]`. NHWC translation should be a guarded local fusion/layout pass, not a global semantic rewrite.
- Multimodal preprocessor and decoder sort modality names before concatenation. With `audio`, `image`, `label`, the flat sequence order is alphabetical, not construction order.
- Multimodal training-style masking uses `torch.bernoulli`; for inference parity it should be disabled or deterministic admission should reject nonzero mask probabilities unless the checkpoint/task requires them. The official multimodal class masks label with probability 1.0 by source construction.

## 4. Operator coverage checklist

Tensor/layout ops:

- Expand learned latents `[num_latents,d_latents] -> [B,num_latents,d_latents]`.
- View/reshape/permute/contiguous for attention head split/merge.
- NCHW `Conv2d` input to NHWC/channel-last token stream conversion after conv/pixels.
- `space_to_depth` for image/video patches: NCHW `[B,C,H,W] -> [B,H/s,W/s,s*s*C]`, BTCHW `[B,T,C,H,W] -> [B,T/t,H/s,W/s,t*s*s*C]`.
- Flatten spatial/temporal index dims to `[B, prod(index_dims), C]`.
- Modality padding, sorted concatenation, and `restructure` slicing by `modality_sizes`.

Neural network primitives:

- `Embedding(vocab_size=262, d_model=768)` and `Embedding(max_position_embeddings=2048, d_model=768)` for language.
- `Linear(q_dim -> qk_channels)`, `Linear(kv_dim -> qk_channels)`, `Linear(kv_dim -> v_channels)` with bias.
- Attention output `Linear(v_channels -> output_channels)` with bias.
- FFN `Linear(d -> widening*d) -> GELU -> Linear(widening*d -> d)` with bias.
- LayerNorm over last dim, eps `1e-12`.
- Image conv path: same-padded `Conv2d(3 -> 64, kernel=7, stride=2, bias=False)`, BatchNorm2d, ReLU, MaxPool2d(kernel=3,stride=2).
- Image learned path: `Conv2d(3 -> 256, kernel=1, stride=1)`.
- Patch optical flow: `Linear(54 -> 64)` after `space_to_depth`.
- Postprocessors: `Linear(512 -> 16)` audio patches, `Linear(512 -> 3)` image projection, `Linear(512 -> num_labels)` label classification.

Attention primitives:

- Noncausal input-to-latent cross-attention: queries are latents, keys/values are preprocessed inputs.
- Noncausal latent self-attention over fixed latent sequence.
- Noncausal decoder cross-attention: queries are task output positions or preprocessed inputs, keys/values are latents.
- MHA only; no MQA/GQA, no RoPE, no ALiBi, no local/sliding window, no FlashAttention dispatch in source.

Position/custom math ops:

- Trainable position embeddings, optional bicubic interpolation for learned image positions.
- Fourier feature generation with linear frequencies, sin/cos, optional raw position concat.
- Coordinate grid generation with `torch.meshgrid(indexing="ij")` over `[-1, 1]`.
- Subsampled decoder query coordinate mapping for multimodal output points.

Generation/cache ops:

- None required. There is no autoregressive KV cache, beam reorder, or sampling controller.

Preprocessing-coupled ops:

- Perceiver byte tokenizer: `[CLS] bytes [SEP]`, byte IDs offset by 6 special tokens, model max length 2048.
- Vision image processor: custom center crop based on `size/crop_size * min_dim`, bicubic resize to 224, rescale/normalize with ImageNet mean/std.
- Optical-flow dense output reshape and divide by 100.

Distributed/tensor-parallel ops:

- None in source.

## 5. Layer/block breakdown

Shared encoder:

```text
inputs_preprocessed: [B, N_in, C_in]
latents = learned_latents.expand(B)                         # [B, L, D]

Input cross-attention:
  q = LayerNorm(D)(latents) -> Linear(D -> qk)
  k = LayerNorm(C_in)(inputs) -> Linear(C_in -> qk)
  v = LayerNorm(C_in)(inputs) -> Linear(C_in -> v)
  a = softmax((q @ k.T) / sqrt(qk_per_head) + input_mask)
  x = Linear(v -> D)(a @ v)
  x = x + latents                                           # if use_query_residual
  y = LayerNorm(D)(x)
  y = Linear(D -> widening*D) -> GELU -> Linear(widening*D -> D)
  latents = x + y

Latent self-attention, repeated num_blocks * num_self_attends_per_block:
  q,k,v = LayerNorm(D)(latents) through Linear(D -> qk/qk/v)
  a = softmax((q @ k.T) / sqrt(qk_per_head))
  x = Linear(v -> D)(a @ v) + latents
  y = LayerNorm(D)(x)
  y = MLP(D -> widening*D -> D)
  latents = x + y
```

Classification image decoder:

```text
query = trainable_pos([B,1,D])
decoded = cross_attention(query, z=latents)                  # q/k/v default D for image configs
logits = Linear(D -> num_labels)(decoded)[:,0,:]
```

Masked-LM decoder:

```text
text embeddings: [B,2048,768]
query = learned output positions [B,2048,768]
decoded = cross_attention(query, z=latents; qk=256, v=768, heads=8, final_project=False)
logits = decoded @ token_embedding.weight.T                  # [B,2048,262]
```

Optical-flow decoder:

```text
input patches: [B,2,27,368,496]
space_to_depth temporal=2, spatial=1 -> [B,368,496,54]
linear patch projection 54 -> 64, concat Fourier 258 -> C_in=322
query = preprocessed inputs with position encoding           # [B,368*496,322]
decoded = cross_attention(query, z=latents) -> Linear(... -> 2)
flow = reshape([B,368,496,2]) / 100
```

Multimodal autoencoding:

```text
audio: waveform [B,30720] -> reshape [B,1920,16] + 1D Fourier(385) => 401
image: [B,16,3,224,224] -> space_to_depth spatial=4 => [B,16,56,56,48] + 3D Fourier(195) => 243
label: one-hot [B,num_labels] -> [B,1,num_labels]
pad each modality to max_channels + 4, sorted concat along sequence
encoder latents -> modality query construction -> shared decoder cross-attention
postprocess slices to audio/image/label outputs
```

## 6. Attention requirements

All attention is dense, noncausal MHA.

Encoder cross-attention:

- Query sequence length `L=num_latents`.
- Key/value length `N_in` from preprocessor: language 2048, vision Fourier 224*224, conv 56*56, optical-flow 368*496, multimodal concatenated audio+image+label.
- Head count from `num_cross_attention_heads`. Common vision configs use 1 cross head; language uses 8.
- Masking: `attention_mask` is inverted to additive mask and applied only to input cross-attention. No causal mask.

Latent self-attention:

- Length is fixed `num_latents`.
- Head count from `num_self_attention_heads`.
- For image conv/Fourier/learned: D=1024, 8 heads, head dim 128.
- For language: qk=256, 8 heads -> q/k head dim 32; v=1280, 8 heads -> v head dim 160.

Decoder cross-attention:

- Query length depends on task: 1 for classification, 2048 for MLM, 368*496 for optical flow, or subsampled multimodal queries.
- Keys/values are latent outputs.
- Source passes `query_mask` but decoder ignores it because latents are the `inputs` side and `inputs_mask=None`.

No packed/varlen support, no sliding/local attention, no RoPE/ALiBi/relative bias, no KV cache. A fused attention implementation must preserve source math order: LayerNorm before projections, scale by `sqrt(qk_channels_per_head)`, additive mask before softmax, dropout after softmax, then value matmul and output projection.

## 7. Position encoding and custom math

Fourier features are source-critical for image, optical-flow, audio, and multimodal paths.

```python
def perceiver_fourier_positions(index_dims, num_bands, max_resolution, concat_pos=True):
    pos = meshgrid_ij(index_dims, low=-1.0, high=1.0)          # [prod(index_dims), dims]
    freq = [linspace(1.0, res / 2, num_bands) for res in max_resolution]
    x = pos[:, :, None] * stack(freq)[None, :, :]              # [N, dims, bands]
    x = reshape(x, [N, dims * num_bands])
    fourier = concat([sin(pi * x), cos(pi * x)], dim=-1)
    return concat([pos, fourier], dim=-1) if concat_pos else fourier
```

Trainable positions are plain learned tables. For learned image classification, `interpolate_pos_encoding=True` bicubic-interpolates a square learned table in NCHW form and returns flattened channel-last positions. This is optional at forward time and should be guarded by input spatial shape.

Subsampled decoder queries map flat indices to coordinates:

```python
coord = unravel_index(subsampled_points, output_index_dims)
pos = -1 + 2 * coord / tensor(output_index_dims)
```

Precomputable: static Fourier grids for fixed resolution and dtype, trainable position tables, modality padding vectors. Dynamic: batch expansion, optional learned-position interpolation, subsampled query positions, stochastic multimodal masks.

## 8. Preprocessing and input packing

Language:

- Tokenizer uses raw UTF-8 bytes plus six special tokens. Actual tokenizer `vocab_size` property is 256, while model config vocabulary is 262 including special tokens.
- Single sequence layout: `[CLS] byte_ids [SEP]`; pair: `[CLS] A [SEP] B [SEP]`.
- GPU graph receives `input_ids [B,2048]` and optional `attention_mask [B,2048]`; token and position embeddings are summed.

Vision classification:

- CPU/image processor emits `pixel_values [B,3,224,224]` after custom center crop, bicubic resize, rescale, ImageNet normalization.
- Learned model: Conv1x1 keeps NCHW, then permutes to `[B,224,224,256]`, flattens to `[B,50176,256]`, concatenates projected learned position `[B,50176,256]`.
- Fourier model: raw pixels permute to `[B,224,224,3]`, flatten, concatenate Fourier `[B,50176,258]`.
- Conv model: `Conv2dSamePadding(7,stride=2)` + BN + ReLU + maxpool produce `[B,64,56,56]`, then channel-last/flatten and Fourier `[B,3136,258]`.

Optical flow:

- Source example expects precomputed patches `[B,2,27,368,496]`.
- `space_to_depth` with temporal block 2 creates `[B,368,496,54]`; optional linear `54 -> 64`; Fourier over train size `[368,496]`; final output reshapes to `[B,368,496,2]` and scales by 0.01.

Multimodal:

- Audio is reshaped to non-overlapping `[B,1920,16]` patches; no STFT/mel frontend is in source.
- Image/video input uses BTCHW and `space_to_depth` to `[B,16,56,56,48]`.
- Label input becomes `[B,1,num_labels]`.
- Each modality is padded to common channel width and concatenated in sorted modality order. Decoder/postprocessor uses `modality_sizes` to slice the flat outputs back into modality records.

## 9. Graph rewrite / lowering opportunities

### Rewrite: static Fourier grid precompute

Source pattern: `meshgrid -> linspace -> multiply -> sin/cos -> concat`.

Replacement: load or cache a constant `[prod(index_dims), pos_dim]` grid, cast to runtime dtype, expand batch.

Preconditions: fixed `index_dims`, `num_bands`, `max_resolution`, `concat_pos`, `sine_only=False`; no custom `pos`; no subsampled points. Shape equation: `pos_dim = dims + 2*dims*num_bands` when `concat_pos=True`. Failure cases: dynamic input resolution, subsampled decoder points, custom `pos`, learned interpolation.

Parity sketch: compare generated grid and precomputed grid for fixed 224x224 and 56x56 in fp32, then cast tests for fp16/bf16.

### Rewrite: same-padded conv stem as optimized local layout region

Source pattern: `ZeroPad2d -> Conv2d(7x7,stride=2,bias=False) -> BatchNorm2d -> ReLU -> MaxPool2d(3,stride=2) -> permute NCHW to NHWC`.

Replacement: an NHWC/local channel-last conv+BN+ReLU+pool kernel or fused conv stem ending in channel-last `[B,56,56,64]`.

Preconditions: exact conv path, static kernel/stride/padding, eval-mode BN folded or implemented identically, all consumers are the immediate flatten/position concat. Axis rewrites: BatchNorm/pool channel axis changes from dim 1 to dim -1 if translated. Failure cases: exposing intermediate NCHW features, training BN, altered pooling padding semantics.

Parity sketch: compare full preprocessor token output `[B,3136,322]`, not just conv stem.

### Rewrite: space_to_depth to window flatten

Source pattern: NCHW/BTCHW view/permute/contiguous/view for non-overlapping patches.

Replacement: `WindowFlatten` that emits channel-last patch vectors directly.

Preconditions: spatial and temporal dimensions divisible by block sizes; channel-first source inputs; no overlapping patches. Shape: `[B,C,H,W] -> [B,H/s,W/s,s*s*C]`; `[B,T,C,H,W] -> [B,T/t,H/s,W/s,t*s*s*C]`. Failure cases: non-divisible dims, different memory layout, downstream consumers expecting NCHW.

Parity sketch: random integer tensor test to preserve element order exactly.

### Rewrite: modality pad/concat metadata

Source pattern: per-modality preprocess -> learned padding concat on channel -> sorted concat on sequence -> later `restructure`.

Replacement: explicit packed multimodal descriptor `{order, lengths, channel_width}` plus fused pad+concat.

Preconditions: modality set fixed, sorted ordering preserved, mask probabilities deterministic/disabled or exactly represented. Failure cases: stochastic masking, missing modalities, dynamic class-label width.

Parity sketch: compare final packed tensor and sliced postprocessor inputs for all modalities.

## 10. Kernel fusion candidates

Highest priority:

- Dense MHA for input-to-latent cross-attention with asymmetric sequence lengths. This is the dominant Perceiver operation for long image/flow inputs because `N_in` can be 50k-182k while query length is latent-bound.
- Latent self-attention + MLP block fusion over fixed latent lengths. Image configs run 48 self-attention layers (`8*6`), language runs 26, so launch overhead and LayerNorm/GELU traffic matter.
- Fourier grid precompute/cache. Avoids expensive meshgrid/sin/cos in every inference when resolution is static.

Medium priority:

- Conv/BN/ReLU/pool preprocessing fused into channel-last token output for `vision-perceiver-conv`.
- `space_to_depth` + optional `Linear(54 -> 64)` for optical flow and multimodal image.
- LayerNorm + QKV projections for self-attention where q/k/v all consume the same normalized tensor.

Lower priority:

- Classification decoder cross-attention, because query length is only 1.
- Embedding decoder matmul for language MLM unless MLM is a first-class target.
- Multimodal postprocessor projections; useful after encoder/decoder parity is stable.

## 11. Runtime staging plan

1. Parse PerceiverConfig and reject unsupported task classes explicitly.
2. Load weights and run source-faithful latent encoder with direct `[B,N,C]` synthetic inputs.
3. Add image Fourier classification end-to-end: image processor contract, Fourier grid, encoder, classification decoder.
4. Add conv preprocessing with guarded NCHW-to-channel-last local layout optimization.
5. Add language MLM: byte tokenizer contract, embeddings, decoder, tied embedding projection.
6. Add optical-flow patches and dense decoder reshape/scale.
7. Add multimodal autoencoding with modality packing and deterministic mask handling.
8. Add optimized attention and preprocessing fusions.

Initially stub: training losses, stochastic multimodal masks, subsampled multimodal output points, optional learned-position interpolation for nonstandard resolutions.

## 12. Parity and validation plan

- Custom op tests: Fourier grid generation, trainable-position interpolation, `space_to_depth`, modality pad/concat/restructure.
- Attention tests: cross-attention with `qk_channels != v_channels`, cross-attention with `qk_channels=None`, self-attention with language qk/v asymmetry.
- Single-layer parity: one `PerceiverLayer` in cross and self modes with fixed synthetic tensors.
- Encoder parity: after input cross-attention, after one self-attend, and final latent output.
- Task parity: image logits for each of learned/Fourier/conv checkpoints; MLM logits at masked byte positions; optical-flow output shape and scale; multimodal output dict shapes.
- End-to-end tolerances: fp32 absolute/relative around `1e-4`; fp16/bf16 around `5e-3` to `1e-2`, with Fourier generated in fp32 then cast if source parity requires.

No DinoML tests were run for this audit by request.

## 13. Performance probes

- Image processor throughput: crop/resize/normalize separated from GPU preprocessor.
- Fourier generation versus cached-grid expansion.
- Input cross-attention sweep over `N_in`: 3136, 50176, 182528.
- Latent self-attention sweep over `num_latents`: 256, 512, 784, 2048.
- End-to-end image classification throughput for learned/Fourier/conv variants.
- MLM throughput for sequence length 2048.
- Optical-flow decode throughput and memory for `[368,496]` dense query output.
- Multimodal packing/postprocessing overhead versus encoder/decoder attention time.
- Layout-pass comparison: source NCHW conv stem plus permute versus fused local channel-last stem.

## 14. Skip/defer list

- Training losses and optical-flow training path.
- Autoregressive generation, beam search, speculative decoding, and KV cache.
- Stochastic multimodal masking unless explicitly targeting the autoencoding training-style behavior.
- Dynamic arbitrary-resolution learned-position interpolation for first image-classification parity.
- Subsampled multimodal output points.
- Multi-GPU/tensor parallelism.
- Quantization/packed weights; no source-coupled quantized format is present.

## 15. Final implementation checklist

- [ ] Parse PerceiverConfig and task architecture.
- [ ] Load learned latent parameters and all Linear/LayerNorm/Embedding weights.
- [ ] Implement dense noncausal MHA with separate q/k head dim and v head dim.
- [ ] Implement latent cross-attention and repeated latent self-attention blocks.
- [ ] Implement Fourier position grid generation and static-grid cache.
- [ ] Implement trainable position embeddings with optional bicubic interpolation.
- [ ] Implement Perceiver byte tokenizer contract for MLM.
- [ ] Implement image processor contract and image preprocessors: pixels, conv1x1, conv, patches.
- [ ] Implement `space_to_depth` with exact element ordering.
- [ ] Implement classification, MLM, optical-flow, and multimodal decoders as staged heads.
- [ ] Add guarded rewrite for conv stem to local channel-last output.
- [ ] Add guarded rewrite for static Fourier grid precompute.
- [ ] Add parity tests for attention qk/v asymmetry and modality packing.
- [ ] Benchmark input cross-attention, latent self-attention, preprocessing, and task decoders separately.
