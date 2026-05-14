# XCodec Transformers Audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: hf-audio/xcodec-* representative checkpoints
Config source: official Hugging Face config.json and preprocessor_config.json
Source files inspected:
  X:/H/transformers/src/transformers/models/xcodec/configuration_xcodec.py
  X:/H/transformers/src/transformers/models/xcodec/modeling_xcodec.py
  X:/H/transformers/src/transformers/models/xcodec/convert_xcodec_weights_to_hf.py
  X:/H/transformers/src/transformers/models/dac/configuration_dac.py
  X:/H/transformers/src/transformers/models/dac/modeling_dac.py
  X:/H/transformers/src/transformers/models/dac/feature_extraction_dac.py
  X:/H/transformers/src/transformers/models/hubert/modeling_hubert.py
  X:/H/transformers/src/transformers/models/wavlm/modeling_wavlm.py
Any missing files or assumptions:
  No tests/imports were run. Safetensors metadata and original ZhenYe234 YAML
  checkpoints were not downloaded. HuBERT, WavLM, and DAC are treated as
  composed in-library families, not fully re-audited here.
```

Representative configs inspected:

- `hf-audio/xcodec-hubert-librispeech`
- `hf-audio/xcodec-hubert-general`
- `hf-audio/xcodec-hubert-general-balanced`
- `hf-audio/xcodec-wavlm-mls`
- `hf-audio/xcodec-wavlm-more-data`

All five public repos expose `config.json`, `model.safetensors`, and
`preprocessor_config.json`. No gated access was encountered for those files.

## 2. High-level architecture

XCodec is a neural audio tokenizer/codec, not an autoregressive text model. The
first useful DinoML target should be codec parity: waveform encode to discrete
audio codes and decode from audio codes back to waveform.

Dataflow:

```text
raw mono waveform -> DacFeatureExtractor padding -> XCodec encode
  -> semantic backbone HuBERT/WavLM hidden-state stack mean
  -> semantic Conv1d adapter
  -> DAC acoustic Conv1d encoder
  -> concat acoustic+semantic channels -> Linear -> residual vector quantizer
  -> integer audio_codes

audio_codes -> residual codebook embedding sum -> Linear -> DAC acoustic decoder
  -> waveform -> crop to original input length in forward()
```

Stage decomposition:

- CPU/data pipeline: sampling-rate guard, mono shape guard, right padding to a
  `hop_length` multiple, optional caller-side truncation.
- Independently cacheable semantic branch: HuBERT or WavLM `AutoModel` run with
  `output_hidden_states=True`, then all hidden states are stacked and averaged.
- Acoustic branch: DAC encoder/decoder submodules copied from `DacModel`, with
  an XCodec-specific decoder adjustment.
- Tokenizer core: local semantic Conv1d adapters, concat/projection, RVQ encode,
  RVQ decode, and codebook ABI.
- End-to-end forward: encode unless `audio_codes` are supplied, decode, crop to
  original sample length.

The semantic backbone can be staged separately from the codec head: DinoML can
first support decode-only from codes, then encode with precomputed semantic
features, then full waveform-to-code encode.

## 3. Important config dimensions

Representative public checkpoints share the XCodec wrapper dimensions:

| Field | Value / source |
| --- | --- |
| `sample_rate` | 16000 from `config.json` |
| preprocessor | `DacFeatureExtractor`, mono, right padding, `hop_length=320` |
| acoustic model | DAC, `downsampling_ratios=[8,5,4,2]`, `upsampling_ratios=[8,5,4,2]` |
| acoustic hidden | encoder hidden 64, decoder hidden 1024, latent hidden 256 |
| semantic hidden | 768 for HuBERT base or WavLM base-plus |
| XCodec `hidden_size` | 1024 = 256 acoustic + 768 semantic |
| `codebook_size` | 1024 |
| `codebook_dim` | 1024 in representative configs |
| `codebook_nbits` | 10, derived as `ceil(log2(1024))` |
| `hop_length` | 320, derived from DAC downsampling product |
| `frame_rate` | 50, derived as `ceil(16000 / 320)` |
| max `num_quantizers` | 8, derived from max bandwidth 4 kbps |
| semantic adapter `strides` | `[1, 1]` |
| semantic adapter `channel_ratios` | `[1, 1]` |
| `block_dilations` | `[1, 1]` |
| dtype | `torch_dtype=float32` in inspected configs |

Checkpoint sweep:

| Model id | Semantic backbone | Semantic layers/heads | Relative bias | Main variation |
| --- | --- | ---: | --- | --- |
| `hf-audio/xcodec-hubert-librispeech` | `facebook/hubert-base-ls960` | 12 / 12 | no | HuBERT semantic source |
| `hf-audio/xcodec-hubert-general` | `ZhenYe234/hubert_base_general_audio` | 12 / 12 | no | HuBERT weights trained for general audio |
| `hf-audio/xcodec-hubert-general-balanced` | `ZhenYe234/hubert_base_general_audio` | 12 / 12 | no | same structure, different checkpoint data |
| `hf-audio/xcodec-wavlm-mls` | `microsoft/wavlm-base-plus` | 12 / 12 | yes, WavLM bucketed relative bias | WavLM semantic source |
| `hf-audio/xcodec-wavlm-more-data` | `microsoft/wavlm-base-plus` | 12 / 12 | yes, WavLM bucketed relative bias | same structure, more training data |

Bandwidth to quantizer count with the inspected dimensions:

| Requested bandwidth | Source admission | Active quantizers |
| ---: | --- | ---: |
| 0.5 kbps | must exactly match `target_bandwidths` | 2 |
| 1.0 kbps | must exactly match `target_bandwidths` | 4 |
| 1.5 kbps | must exactly match `target_bandwidths` | 6 |
| 2.0 kbps | must exactly match `target_bandwidths` | 8 |
| 4.0 kbps | must exactly match `target_bandwidths` | 8 |

The 2 kbps and 4 kbps values map to the same active quantizer count because the
configured model only instantiates eight quantizers.

## 3a. Family variation traps

- `semantic_model_config.model_type` changes real operator coverage. HuBERT is
  standard bidirectional MHA with convolutional positional embeddings; WavLM adds
  bucketed relative position bias and a gated relative-position projection path.
- The XCodec wrapper requires all hidden states from the semantic model, not just
  the final hidden state. It stacks them along a new layer axis and computes a
  simple mean.
- `input_values` must be mono `[B, 1, T]`. The source raises on channels other
  than one.
- The feature extractor pads input lengths to a multiple of `hop_length`, but
  `forward()` crops decoded audio to the original `input_values.shape[-1]`.
- Encode conditionally pads the DAC acoustic path by `hop_length // 2` only when
  the computed Conv1d output length would not match the semantic branch length.
  Do not turn this into unconditional padding without a parity guard.
- Public `audio_codes` layout is `[B, Q, L]`; internal RVQ iteration layout is
  `[Q, B, L]`.
- Codebook embeddings are buffers, not `nn.Embedding` parameters, in XCodec RVQ:
  `quantizer.quantizers.i.codebook.embed` has shape `[codebook_size, codebook_dim]`.
- The acoustic decoder is not vanilla HF DAC. XCodec changes every DAC
  `ConvTranspose1d.output_padding` to `stride % 2` and replaces final `Tanh`
  with `Identity`.
- XCodec's local semantic decoder modules are constructed but not used by
  `encode()`, `decode()`, or `forward()` in the inspected source. They should be
  loaded for weight parity but can be deferred for runtime graph parity unless a
  future source path consumes them.
- Historical config fields such as `encoder_channels`, `decoder_channels`,
  `input_channels`, and `output_channels` appear in checkpoint configs but are
  not read by the inspected `XcodecConfig` or `XcodecModel`.
- Training-only semantic config fields such as SpecAugment probabilities are not
  used for inference in XCodec because `semantic_model.eval()` is used and
  `mask_time_indices` are not passed.

## 4. Operator coverage checklist

Tensor/layout ops:

- 1D right padding on waveform and optional acoustic-path padding by
  `hop_length // 2`.
- Transpose `[B,T,C] <-> [B,C,T]` around semantic adapters and Linear layers.
- Concatenate acoustic and semantic channels: `[B,256,L] + [B,768,L] -> [B,1024,L]`.
- Stack semantic hidden states: tuple of layer tensors to `[B,num_layers+1,L,768]`.
- Mean reduction over hidden-state layer axis.
- Crop waveform tail: `audio_values[..., :original_length]`.

Neural primitives:

- Conv1d NCL with bias/no-bias, stride, dilation, groups=1 for local XCodec and
  DAC paths.
- ConvTranspose1d NCL for DAC decoder and unused local semantic decoder.
- Linear on last dimension: `1024 -> 1024`, `1024 -> 768`, `1024 -> 256`.
- ELU in local semantic residual blocks.
- DAC `Snake1d`: `x + reciprocal(alpha + 1e-9) * sin(alpha * x)^2`.
- DAC residual crop after dilated Conv1d if output time is shorter.
- LayerNorm, GroupNorm, GELU, dropout-as-identity for inference in semantic
  backbones.

Attention primitives:

- HuBERT/WavLM semantic branch requires encoder self-attention only, no KV cache.
- HuBERT: bidirectional MHA, convolutional positional embedding, optional
  attention mask conversion if mask is supplied.
- WavLM: MHA with relative position bucket embedding and gated relative bias.

Discrete codebook / tokenizer ops:

- Euclidean nearest-neighbor codebook lookup:
  `argmax(-(||x||^2 - 2 x E^T + ||E||^2))`.
- RVQ residual update: `residual = residual - decode(indices)` per active
  quantizer.
- RVQ code stack and transpose between public and internal layouts.
- Decode-side `F.embedding(codes, embed)` and sum across quantizers.
- Integer code ABI with code range `[0, codebook_size)`.

Preprocessing-coupled ops:

- Sampling-rate check at 16 kHz.
- Mono channel validation.
- Right padding to `hop_length=320` multiple in `DacFeatureExtractor`.
- Optional `padding_mask` emitted by preprocessor, but XCodec forward does not
  accept or consume it.

Quantized/packed weight metadata ops:

- No source-coupled packed weight format is used for model weights.
- The "quantizer" is the audio tokenizer codebook, not a weight quantization
  provider. Treat codebook buffers as normal dense constants.

## 5. Layer/block breakdown

Encode path for representative configs:

```text
input_values: [B,1,T], mono float waveform

semantic branch:
  mono = input_values[:,0,:]                       # [B,T]
  sem_in = pad(mono, left=160, right=160)          # [B,T+320]
  hidden_states = HuBERT/WavLM(sem_in, output_hidden_states=True)
  e = mean(stack(hidden_states, dim=1), dim=1)     # [B,Ls,768]
  e_semantic = SemanticEncoder(e.transpose(1,2))   # [B,768,Ls]

acoustic branch:
  if conv_length(T, acoustic_encoder) != Ls:
      acoustic_in = pad(input_values, (160,160))   # [B,1,T+320]
  else:
      acoustic_in = input_values
  e_acoustic = DAC.encoder(acoustic_in)            # [B,256,Ls]

fusion and codes:
  embeddings = cat([e_acoustic, e_semantic], dim=1) # [B,1024,Ls]
  embeddings = fc(embeddings.transpose(1,2)).transpose(1,2)
  codes_qbl = RVQ.encode(embeddings, bandwidth)     # [Q,B,Ls]
  audio_codes = codes_qbl.transpose(0,1)             # [B,Q,Ls]
```

Local `SemanticEncoder`:

```text
Conv1d(768 -> 768, kernel=3, stride=1, padding=1, bias=False)
repeat over strides [1,1]:
  repeat over block_dilations [1,1]:
    ELU -> Conv1d(C -> C, kernel=3, dilation=1, padding=1, bias=False)
    ELU -> Conv1d(C -> C, kernel=1, bias=False)
    residual add
  Conv1d(C -> 768, kernel=3, stride=1, padding=1, bias=True)
```

DAC acoustic encoder, representative config:

```text
Conv1d(1 -> 64, kernel=7, padding=3)
for strides [8,5,4,2], channel dims 64->128->256->512->1024:
  three residual units with dilation 1,3,9:
    Snake1d -> Conv1d(kernel=7, dilation=d, padding=3*d)
    Snake1d -> Conv1d(kernel=1)
    centered crop if needed -> residual add
  Snake1d -> Conv1d(kernel=2*stride, stride=stride, padding=ceil(stride/2))
Snake1d -> Conv1d(1024 -> 256, kernel=3, padding=1)
```

Decode path:

```text
audio_codes: [B,Q,L]
codes_qbl = audio_codes.transpose(0,1)
quantized = sum_i embedding(codes_qbl[i], codebook_i).transpose(1,2) # [B,1024,L]
quantized_acoustic = fc2(quantized.transpose(1,2)).transpose(1,2)    # [B,256,L]
audio_values = adjusted DAC.decoder(quantized_acoustic)              # [B,1,T']
forward() crops to original T when input_values is available
```

Adjusted DAC decoder:

```text
Conv1d(256 -> 1024, kernel=7, padding=3)
for upsampling strides [8,5,4,2]:
  Snake1d -> ConvTranspose1d(kernel=2*stride, stride=stride,
                             padding=ceil(stride/2),
                             output_padding=stride % 2)
  residual units with Snake1d/Conv1d as above
Snake1d -> Conv1d(64 -> 1, kernel=7, padding=3)
Identity instead of DAC final Tanh
```

## 6. Attention requirements

XCodec itself has no autoregressive generation attention and no KV cache. The
only attention lives inside the semantic encoder branch used during encode.

HuBERT semantic branch:

- Noncausal encoder self-attention.
- MHA with 12 heads, hidden 768, head dim 64 for inspected configs.
- Convolutional feature extractor first, then feature projection, convolutional
  positional embedding, LayerNorm/dropout, and 12 encoder layers.
- Attention masks are supported by HuBERT source, but XCodec does not pass the
  preprocessor `padding_mask` into `semantic_model`, so first parity should
  match the unmasked XCodec call.
- No KV cache, no sliding window, no cross-attention.

WavLM semantic branch:

- Noncausal encoder self-attention.
- MHA with 12 heads, hidden 768, head dim 64.
- Relative position bias is computed from bucketed distances, then gated by a
  projection of hidden states before attention.
- No KV cache. The relative bias may be cached per shape for a single encode
  request, but it is not an autoregressive cache.

Decode-only from `audio_codes` requires no attention.

## 7. Position encoding and custom math

HuBERT/WavLM use convolutional positional embeddings over `[B,T,H]` hidden
states:

```python
def conv_pos(hidden_states):
    x = hidden_states.transpose(1, 2)
    x = grouped_conv1d_weight_norm(x)
    x = same_pad_trim_if_even_kernel(x)
    x = gelu(x)
    return x.transpose(1, 2)
```

WavLM additionally uses bucketed relative position bias. DinoML should reuse the
separately audited WavLM primitive rather than treating XCodec as owning this
operator.

DAC Snake activation:

```python
def snake1d(x, alpha):
    flat = x.reshape(x.shape[0], x.shape[1], -1)
    y = flat + (alpha + 1e-9).reciprocal() * torch.sin(alpha * flat).pow(2)
    return y.reshape_as(x)
```

XCodec RVQ nearest code:

```python
def xcodec_codebook_encode(x, embed):
    # x: [B,L,D], embed: [K,D]
    dist = -(x.reshape(-1, D).pow(2).sum(1, keepdim=True)
             - 2 * x.reshape(-1, D) @ embed.t()
             + embed.pow(2).sum(1, keepdim=True).t())
    return dist.argmax(dim=-1).view(B, L)
```

Codebook norms can be precomputed per codebook; the input norm and matrix
multiply are dynamic.

## 8. Preprocessing and input packing

`DacFeatureExtractor` owns the public waveform preprocessing:

- Expected audio sample rate: 16000 Hz.
- Mono only for representative XCodec configs (`feature_size=1`).
- Raw audio accepted as NumPy/list, converted to float32.
- Default `padding=True`; right-pads batch to the longest length and to a
  multiple of `hop_length=320`.
- Emits `input_values` shaped `[B,1,T_padded]` after tensor conversion.
- Emits `padding_mask` if padding is used, but the XCodec model source does not
  accept or consume it.
- Stereo is explicitly rejected by the feature extractor.

GPU/runtime graph inputs for first integration:

- Encode: `input_values` `[B,1,T]`, float32, T preferably already padded to a
  multiple of 320 for batch ergonomics. Preserve exact original length if using
  `forward()` crop parity.
- Decode: `audio_codes` `[B,Q,L]`, integer, where `Q <= config.num_quantizers`
  and code values are in `[0,1024)`.
- Bandwidth is scalar control metadata, not a tensor op. It selects active
  quantizer count during encode only.

## 9. Graph rewrite / lowering opportunities

### Rewrite: Decode-only RVQ Embedding Sum

Source pattern:

```text
audio_codes [B,Q,L] -> transpose [Q,B,L]
for i in Q: F.embedding(codes[i], embed_i).permute(0,2,1)
sum quantized vectors -> fc2 -> acoustic decoder
```

Replacement pattern:

```text
CodebookGatherSum(codes, codebook_stack) -> Linear(1024 -> 256) -> DAC decoder
```

Preconditions:

- `codes` rank 3, public layout `[B,Q,L]`.
- `Q <= config.num_quantizers`.
- Each code value is `0 <= code < codebook_size`.
- Codebooks share `[1024,1024]` shape and dtype.

Shape equations:

- Gather output before transpose: `[B,L,1024]`.
- Gather-sum output after transpose: `[B,1024,L]`.

Failure cases:

- Out-of-range codes, non-integer codes, or `Q > num_quantizers` must reject.
- Do not silently pad missing quantizers; source decode sums exactly supplied
  codebook rows.

Parity test sketch:

- Random code tensor for Q in `{1,2,4,8}`, compare fused gather-sum to Python
  `F.embedding` loop before `fc2`.

### Rewrite: RVQ Nearest-Neighbor Encode as Batched GEMM

Source pattern:

```text
dist = -(||x||^2 - 2*x@E.T + ||E||^2)
indices = argmax(dist)
residual -= E[indices]
```

Replacement pattern:

```text
RowNorm(x) + GEMM(x, E.T) + precomputed_code_norms -> ArgMax -> Gather -> ResidualSub
```

Preconditions:

- Euclidean codebook only.
- Static codebook constants.
- Residual dtype and codebook dtype match or explicit cast policy is chosen.
- Argmax tie behavior must match PyTorch first-index behavior.

Failure cases:

- Any source variant that normalizes codebook/inputs like DAC VQ is not this
  XCodec RVQ path.
- Approximate nearest-neighbor search is not parity-safe for tokenizer output.

Parity test sketch:

- Compare code indices and decoded residual update for one and all quantizers
  against source RVQ for random embeddings and real codebooks.

### Rewrite: 1x1 Conv1d / Last-Dim Linear Canonicalization

Source pattern:

```text
Conv1d(Cin -> Cout, kernel=1) on [B,C,T]
Linear(Cin -> Cout) on [B,T,C] wrapped by transpose pairs
```

Replacement pattern:

```text
GEMM over flattened [B*T,Cin] -> [B,T,Cout]
```

Preconditions:

- Dense contiguous NCL or NTC layout known.
- No dilation/stride/padding for Conv1d kernel 1.
- Bias handling preserved.

Failure cases:

- Do not fold dilated `kernel=3/7` temporal Conv1d into this rewrite.

### Rewrite: Static Conv1d Lowering to GEMM/Im2Col

Preconditions:

- Fixed kernel, stride, dilation, padding from config.
- NCL source layout preserved.
- Dynamic time dimension has max-shape workspace or tiled lowering.

Replacement:

```text
TemporalWindowExtract(NCL) -> GEMM(weight_flat.T) -> BiasAdd -> NCL output
```

Layout constraints:

- Source is NCL. A channel-last optimized pass may use NTC internally only inside
  a fully controlled region and must rewrite all Conv1d, transpose, residual,
  and normalization axes consistently.

Failure cases:

- DAC residual units crop the residual path if Conv1d output length shrinks.
  A lowering pass must preserve that centered crop.

### Guard: Acoustic Decoder Adjustment

Source pattern:

```text
Auto DAC decoder -> mutate ConvTranspose1d.output_padding = stride % 2
replace final Tanh with Identity
```

Admission rule:

- XCodec must not reuse an unmodified DAC decoder graph. Reject or rewrite the
  decoder metadata before lowering.

## 10. Kernel fusion candidates

Highest priority:

- Decode-only codebook gather-sum plus `fc2`: This is the smallest useful
  tokenizer runtime and avoids per-codebook Python-style loops.
- DAC Conv1d/ConvTranspose1d blocks with Snake activation: Decode waveform
  synthesis is dominated by temporal convolution and activation bandwidth.
- RVQ encode GEMM + argmax + gather: Required for waveform-to-code tokenization;
  exact nearest-neighbor parity matters.

Medium priority:

- Semantic hidden-state stack mean: Avoid materializing all layer outputs when
  the semantic backbone is under DinoML control by accumulating a running mean.
- Conv1d residual block fusion: ELU/Snake + Conv1d + residual add regions are
  repeated often and have stable static kernels.
- HuBERT/WavLM semantic encoder reuse: Compose separately optimized audio
  encoder support instead of introducing XCodec-specific attention.

Lower priority:

- Full encode end-to-end fusion across semantic and acoustic branches. Branches
  are independently useful and easier to validate separately.
- Local `SemanticDecoder`: present in source but unused by runtime forward.
- Bandwidth specialization: compile separate small-Q encode graphs only after
  the generic active-quantizer loop is correct.

## 11. Runtime staging plan

Stage 1: Config and ABI admission.

- Parse XCodec config plus nested DAC and HuBERT/WavLM configs.
- Admit only mono 16 kHz, DAC acoustic model, HuBERT/WavLM base semantic model,
  `codebook_size=1024`, `codebook_dim=1024`, `hop_length=320`.
- Reject unsupported semantic model types or XCodec variants that read fields
  not covered by this report.

Stage 2: Decode-only parity from `audio_codes`.

- Load XCodec codebook buffers, `fc2`, and adjusted DAC decoder.
- Implement `[B,Q,L]` code ABI and waveform output.
- Stub encode and semantic branch.

Stage 3: Tokenizer core parity with precomputed embeddings.

- Accept fused `[B,1024,L]` embeddings or separate acoustic/semantic embeddings.
- Implement RVQ encode exactly and bandwidth control.

Stage 4: Acoustic encoder parity.

- Implement DAC encoder with XCodec conditional padding and Conv1d output-length
  matching.

Stage 5: Semantic branch composition.

- Compose separately audited HuBERT and WavLM inference graphs with
  `output_hidden_states=True`.
- Add hidden-state mean and local `SemanticEncoder`.

Stage 6: Full `forward()` parity.

- End-to-end waveform encode/decode and crop.
- Preserve no-mask semantic behavior unless source changes to consume
  `padding_mask`.

Stage 7: Optimizations.

- Add gather-sum, RVQ GEMM/argmax, Conv1d block fusions, and layout-local
  rewrites behind parity tests.

## 12. Parity and validation plan

- Config-derived shape tests: verify `hop_length=320`, `frame_rate=50`,
  `num_quantizers=8`, and bandwidth-to-active-Q mapping.
- Codebook gather tests: random `[B,Q,L]` integer codes for Q in `{1,2,4,8}`,
  compare decode embedding sum before `fc2`.
- Decode-only single checkpoint test: real `audio_codes` through DinoML decode
  compared to Transformers `model.decode()`.
- DAC decoder adjustment test: prove output differs from vanilla DAC metadata if
  `output_padding` and final `Identity` are not applied.
- RVQ encode unit test: random embeddings and fixed codebooks, exact code index
  match quantizer-by-quantizer.
- Acoustic encoder branch test: compare output length and values for lengths
  divisible and not divisible by hop length.
- Semantic branch test: compare HuBERT and WavLM hidden-state mean plus local
  `SemanticEncoder`.
- Full forward test: waveform `[B,1,T]` at 16 kHz, compare `audio_codes` exactly
  and waveform with numeric tolerance after crop.

Recommended tolerances:

- Code indices: exact.
- fp32 branch tensors: `rtol=1e-4`, `atol=1e-5` initially, tighten where Conv1d
  lowering is deterministic.
- fp16/bf16 optimized paths should be later opt-ins; inspected checkpoints are
  float32.

## 13. Performance probes

- Preprocessor throughput: padding and batch collation for variable-length audio.
- Decode-only throughput: `[B,Q,L] -> waveform`, sweep B, Q, and L.
- RVQ encode throughput: nearest-neighbor GEMM/argmax/gather per quantizer,
  sweep active Q and codebook size.
- DAC encoder and decoder temporal Conv1d throughput separately.
- Semantic backbone throughput separately for HuBERT and WavLM.
- Full encode throughput with and without semantic hidden-state materialization.
- End-to-end codec real-time factor at 16 kHz.
- Memory probes for storing all semantic hidden states versus running mean.
- Layout probe: NCL Conv1d baseline versus guarded local NTC/channel-last
  optimized regions.

## 14. Skip/defer list

- Training losses, straight-through estimator behavior, codebook EMA/cluster
  updates, and SpecAugment.
- Autoregressive generation, KV cache, beam search, and sampling; not part of
  this model family.
- Stereo audio; feature extractor rejects it for now.
- Non-DAC acoustic models or non-HuBERT/WavLM semantic models.
- Local `SemanticDecoder` runtime use unless a future source path consumes it.
- Weight quantization/provider work; XCodec codebooks are tokenizer constants,
  not packed model weights.
- Approximate nearest-neighbor code search.
- Layout translation across semantic backbones until HuBERT/WavLM audits define
  their own safe axis rewrites.

## 15. Final implementation checklist

- [ ] Parse XCodec config and nested DAC/HuBERT/WavLM configs.
- [ ] Load XCodec weights with codebook buffers as dense constants.
- [ ] Enforce mono 16 kHz waveform ABI and `[B,Q,L]` integer code ABI.
- [ ] Apply XCodec DAC decoder adjustment before lowering.
- [ ] Implement decode-only codebook gather-sum.
- [ ] Implement `fc2` plus adjusted DAC decoder.
- [ ] Add decode-only parity against Transformers.
- [ ] Implement RVQ encode exact nearest-neighbor loop.
- [ ] Add bandwidth-to-active-quantizer admission and tests.
- [ ] Implement DAC acoustic encoder branch and conditional padding guard.
- [ ] Compose HuBERT semantic branch and WavLM semantic branch as separate audited families.
- [ ] Implement hidden-state stack mean and local `SemanticEncoder`.
- [ ] Add full `encode()` parity for HuBERT and WavLM checkpoints.
- [ ] Add full `forward()` crop parity.
- [ ] Benchmark decode-only, RVQ encode, semantic branch, and end-to-end real-time factor.
