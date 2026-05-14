# Transformers family audit: `higgs_audio_v2_tokenizer`

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: bosonai/higgs-audio-v2-tokenizer
Config source: https://huggingface.co/bosonai/higgs-audio-v2-tokenizer/raw/main/config.json
Source files inspected:
  X:/H/transformers/src/transformers/models/higgs_audio_v2_tokenizer/configuration_higgs_audio_v2_tokenizer.py
  X:/H/transformers/src/transformers/models/higgs_audio_v2_tokenizer/modeling_higgs_audio_v2_tokenizer.py
  X:/H/transformers/src/transformers/models/higgs_audio_v2_tokenizer/modular_higgs_audio_v2_tokenizer.py
  X:/H/transformers/src/transformers/models/higgs_audio_v2_tokenizer/convert_higgs_audio_v2_tokenizer_to_hf.py
  X:/H/transformers/src/transformers/models/xcodec/*
  X:/H/transformers/src/transformers/models/dac/configuration_dac.py
  X:/H/transformers/src/transformers/models/dac/modeling_dac.py
  X:/H/transformers/src/transformers/models/dac/feature_extraction_dac.py
  X:/H/transformers/src/transformers/models/hubert/configuration_hubert.py
  X:/H/transformers/src/transformers/models/hubert/modeling_hubert.py
Any missing files or assumptions:
  The in-tree modeling file is generated from modular_higgs_audio_v2_tokenizer.py.
  Public HF repo exposes one current official checkpoint and one legacy config revision; a docs example id was not fetchable.
```

This report targets inference for the audio tokenizer/codec ABI: waveform to discrete audio codes, codes to waveform, and round-trip `forward`. Training losses, codec codebook updates, and speech generation LLM stages are out of scope.

## 2. High-level architecture

`higgs_audio_v2_tokenizer` is a composite neural audio codec, not a text decoder. It combines:

```text
raw mono waveform -> CPU feature extraction/padding
  -> semantic branch: 24 kHz -> 16 kHz resample -> HuBERT hidden-state average -> small Conv1d semantic encoder
  -> acoustic branch: DAC Conv1d encoder
  -> concat acoustic+semantic channels -> Linear fusion -> residual vector quantizer -> audio codes
audio codes -> RVQ embedding sum -> Linear to DAC acoustic width -> DAC ConvTranspose1d decoder -> waveform crop
```

Stageable pieces:

- CPU/data pipeline: sample-rate validation, mono checks, right padding to hop multiple, optional `padding_mask`.
- Semantic feature branch: torchaudio resample plus HuBERT encoder. This can be cached for a fixed waveform chunk before quantization.
- Acoustic encoder/decoder: DAC-style causal-looking but implemented as padded Conv1d/ConvTranspose1d stack.
- RVQ tokenizer: codebook nearest-neighbor lookup and public code packing.
- Postprocess: crop decoded audio to original input length in `forward`.

## 3. Important config dimensions

| Field | Current official config | Source/default note |
|---|---:|---|
| `sample_rate` | 24000 | tokenizer waveform rate |
| `semantic_sample_rate` | 16000 | semantic branch resample target |
| acoustic `downsampling_ratios` | `[8,5,4,2,3]` | product/hop length 960 |
| acoustic hidden size | 256 | DAC encoder output channels |
| semantic hidden size | 768 | HuBERT base |
| fused hidden size | 1024 | 256 + 768 |
| `codebook_size` | 1024 | 10 bits/code |
| `codebook_dim` | 64 | RVQ projected code vector width |
| `target_bandwidths` | `[0.5,1,1.5,2]` | source class default includes 4 kbps |
| frame rate | 25 | `ceil(24000 / 960)` |
| max quantizers | 8 | `floor(1000*2/(25*10))` |
| semantic HuBERT layers | 12 | all hidden states are averaged |
| semantic HuBERT heads | 12 | MHA, head dim 64 |
| semantic HuBERT MLP | 3072 | GELU FFN |
| HuBERT feature conv | dims `[512]*7`, strides `[5,2,2,2,2,2,2]`, kernels `[10,3,3,3,3,2,2]` | total stride 320 |

Representative config sweep:

| Source | Status | Operator-significant values |
|---|---|---|
| `bosonai/higgs-audio-v2-tokenizer@main` | current official | HF-native config, hop 960, 2 kbps max, 8 RVQ quantizers |
| `bosonai/...@d2e09d95...` | current-style revision | same operator-significant config as `main` |
| `bosonai/...@2d3c8d2...` | legacy raw config | non-HF fields: `ratios=[8,5,4,2,3]`, `bins=1024`, `n_q=8`, `semantic_techer=hubert_base_general`; route through converter or reject as direct HF config |
| `hf-audio/higgs_audio_v2_tokenizer-hubert-librispeech` | source gap | docs example id returned an access/auth error while fetching config |

## 3a. Family variation traps

- Source defaults are not the same as the official checkpoint. With no HF config, acoustic ratios default to `[8,5,4,2]`, target bandwidth defaults include 4 kbps, and derived quantizer count changes.
- Public code packing is batch-major `[B, Q, T]`; internal RVQ loops are quantizer-major `[Q, B, T]`.
- `bandwidth` selects a prefix of quantizers at runtime. DinoML must either specialize per allowed bandwidth or emit a bounded dynamic prefix loop with `Q in {2,4,6,8}` for the official 25 Hz/10-bit config.
- `semantic_downsample_factor` is derived from hop length, sample rates, and `downsample_factor`; for the official config it is `2`.
- The semantic branch pads `(160,160)`, not `self.pad`, despite a source TODO. Treat that as source behavior.
- DAC decoder is mutated after construction: ConvTranspose1d output padding becomes `stride % 2`, and final tanh is removed.
- HuBERT `apply_spec_augment=True` in config is inert for this path because the tokenizer sets the semantic model to eval and calls with `torch.no_grad()`, while `mask_time_prob=0.0`.
- Weight norm exists as helper/conversion concern. Runtime inference should load already materialized weights or explicitly fold weight norm before lowering.
- Layout is NCL for conv/audio tensors and NTC around HuBERT/linear layers. Do not globally translate Conv1d to channel-last without guarding every transpose/linear/codebook consumer.

## 4. Operator coverage checklist

Tensor/layout ops:

- `pad` for last-axis audio, fixed `(160,160)` and dynamic `hop_length//2` acoustic pad.
- `transpose`/`permute` between `[B,C,T]` and `[B,T,C]`.
- `reshape`/flatten codebook distance inputs from `[B,T,D]` to `[B*T,D]`.
- `cat` along channel axis for acoustic+semantic features.
- `stack` hidden states and quantizer code outputs.
- `slice`/strided gather for semantic downsample `x[:, ::factor, :]`.
- final crop `[..., :length]`.

Neural primitives:

- Conv1d, ConvTranspose1d, GroupNorm, LayerNorm, BatchNorm1d optional in HuBERT positional conv path, GELU, ELU, Snake1d, Linear, residual add.
- HuBERT feature encoder and Transformer encoder: noncausal MHA, FFN GELU, convolutional positional embedding.
- DAC encoder/decoder Conv1d stack with Snake1d activation.

Discrete codebook / tokenizer ops:

- `Embedding(indices, codebook)` for decode.
- Euclidean nearest codebook search: `argmax(-(x^2 - 2*x@E.T + E^2))`.
- Residual update per quantizer: `residual = residual - decoded(indices)`.
- Prefix quantizer loop selected by bandwidth.
- Public audio code ABI `[B,Q,T]`, integer dtype.

Preprocessing-coupled ops:

- DacFeatureExtractor sample-rate guard, mono guard, right padding to multiple of 960.
- torchaudio resample 24000 -> 16000 in the semantic branch.

Postprocess ops:

- decoded waveform crop to original input length for `forward`.

## 5. Layer/block breakdown

Encode path for official config:

```text
input_values: [B,1,L] at 24 kHz
semantic_input = resample(input_values, 24000, 16000)[:,0,:]
semantic_input = pad(semantic_input, (160,160))
hubert_hidden_states = HubertModel(..., output_hidden_states=True).hidden_states
semantic_features = mean(stack(hubert_hidden_states, dim=1), dim=1)       # [B,T_h,768]
semantic_features = semantic_features[:, ::2, :]
e_semantic = SemanticEncoder(transpose(semantic_features, 1, 2))          # [B,768,T_code]

e_acoustic = DACEncoder(input or pad(input, (480,480)))                  # [B,256,T_code]
embeddings = concat([e_acoustic, e_semantic], dim=1)                     # [B,1024,T_code]
embeddings = Linear(1024 -> 1024)(transpose to [B,T,1024])
embeddings = transpose back to [B,1024,T]
audio_codes = RVQ.encode(embeddings, bandwidth).transpose(0,1)           # [B,Q,T_code]
```

RVQ quantizer:

```text
for q in selected_quantizers:
  h = transpose(residual, [B,C,T] -> [B,T,C])
  z = Linear(1024 -> 64)(h)
  indices = nearest_codebook(z, embed[1024,64])                          # [B,T]
  decoded = Embedding(indices, embed) -> Linear(64 -> 1024) -> [B,1024,T]
  residual = residual - decoded
stack(indices)                                                           # [Q,B,T]
```

Decode path:

```text
audio_codes [B,Q,T] -> transpose [Q,B,T]
quantized = sum_q(project_out(embedding_q(indices_q)))                   # [B,1024,T]
quantized_acoustic = Linear(1024 -> 256)(transpose to [B,T,1024])
audio_values = DACDecoder(transpose back to [B,256,T])                   # [B,1,L_dec]
```

Semantic encoder/decoder blocks use ELU -> Conv1d -> ELU -> 1x1 Conv1d residual units. The semantic decoder exists in the module but is not used by `decode`; first integration can defer it unless full module export requires all weights.

## 6. Attention requirements

Attention is required only inside the HuBERT semantic encoder. It is encoder-style, bidirectional, noncausal self-attention with no KV cache, no autoregressive decode, no RoPE, and no cross-attention.

Official HuBERT settings: 12 layers, 12 heads, hidden size 768, head dim 64, FFN size 3072. Masks are bidirectional masks derived inside HuBERT when `attention_mask` is supplied; the tokenizer `_extract_semantic_features` does not pass the feature extractor `padding_mask`, so the first faithful path can run unmasked full padded sequences unless the outer integration intentionally wires masks and validates parity.

FlashAttention/SDPA compatibility is straightforward for full bidirectional self-attention, but hidden-state output collection is mandatory because the tokenizer averages every HuBERT hidden state, not just the final output.

## 7. Position encoding and custom math

HuBERT uses convolutional positional embedding: transpose `[B,T,C]` to `[B,C,T]`, grouped Conv1d over time, same-pad trimming, activation, transpose back. No RoPE/ALiBi.

Codec-specific math:

```python
def snake1d(x, alpha):
    return x + (1.0 / (alpha + 1e-9)) * sin(alpha * x) ** 2

def rvq_nearest(z, embed):
    # z: [N, D], embed: [K, D]
    scores = -(z.pow(2).sum(1, keepdim=True) - 2 * z @ embed.T + embed.pow(2).sum(1))
    return scores.argmax(dim=-1)
```

Precomputeable: codebook squared norms, folded weight-norm weights, static Conv1d shape formulas, and bandwidth-to-quantizer-count table. Dynamic: waveform length, padded length, semantic/acoustic length comparison, and final crop.

## 8. Preprocessing and input packing

`DacFeatureExtractor` owns raw audio preparation. It accepts mono arrays, validates sampling rate when supplied, pads right with 0.0 to a multiple of `hop_length=960`, and returns `input_values` shaped `[B,1,L_pad]` plus `padding_mask` when padding is enabled. Stereo is explicitly unsupported.

The model itself still performs semantic resampling from 24 kHz to 16 kHz with `torchaudio.functional.resample`, then drops to first channel and pads 160 samples on both sides. This resampling is better treated as CPU/data-pipeline work for first DinoML integration. A GPU/runtime resampler would require a separate audio-op admission and parity suite.

Audio token packing:

- Encode public output: `audio_codes: int64 [B,Q,T_code]`.
- Decode public input: same `[B,Q,T_code]`.
- `Q` is selected by requested bandwidth. For official config, bandwidths map approximately to `{0.5:2, 1:4, 1.5:6, 2:8}`.
- `T_code` follows the acoustic hop/code frame length and must match semantic encoder output length after semantic feature downsample. Source conditionally pads acoustic input if the acoustic conv length would otherwise differ.

## 9. Graph rewrite / lowering opportunities

### Rewrite: RVQ nearest-codebook to GEMM plus rowwise argmax

Source pattern:

```text
dist = -(sum(x*x) - 2*x@embed.T + sum(embed*embed))
indices = argmax(dist, dim=-1)
```

Replacement:

```text
score = GEMM(x, embed.T)
score = 2*score - row_norm(x) - codebook_norm(embed)
indices = ArgMax(score, axis=-1)
```

Preconditions: Euclidean codebook, fixed dense codebook `[1024,64]`, no codebook normalization in this family, inference only. Failure cases: any alternate quantizer or learned update path.

Parity test sketch: compare indices for random `[B,T,64]` against PyTorch implementation, including ties if a deterministic first-index argmax policy is adopted.

### Rewrite: 1x1 Conv1d and timewise Linear canonicalization

Source pattern: `Conv1d(Cin,Cout,kernel=1)` on `[B,C,T]` or `Linear(Cin,Cout)` on `[B,T,C]`.

Replacement: reshape/stride-preserving batched GEMM over `B*T` rows.

Preconditions: dense contiguous input, no groups for Conv1d, bias handling explicit, layout known. Failure cases: non-contiguous layout, grouped/depthwise conv, dynamic axis translation without guards.

### Rewrite: static Conv1d padding and length formulas

Source pattern: repeated Conv1d with fixed kernel/stride/dilation/padding.

Replacement: lower to provider Conv1d or im2col+GEMM for static kernel families.

Preconditions: NCL layout, fixed stride/kernel/dilation, exact PyTorch padding/crop behavior. For DAC residual units, preserve center crop before residual add when output length shrinks.

### Rewrite: folded weight norm

Source pattern: converted original checkpoints may contain `weight_g` and `weight_v`.

Replacement:

```text
weight = weight_g * weight_v / norm(weight_v, dims=all_except_out)
```

Preconditions: conversion-time materialization only. Failure cases: live parametrized module at runtime; reject or fold before compile.

### Guard: no global audio layout translation

NCL Conv1d regions can be optimized internally, but HuBERT and Linear/codebook regions use `[B,T,C]`. Any channel-last pass must rewrite transpose axes, concat axis, semantic downsample axis, and final crop assumptions together. First integration should keep source axes faithful.

## 10. Kernel fusion candidates

Highest priority:

- RVQ GEMM+argmax+embedding decode loop. This is the tokenizer bottleneck unique to this family and avoids materializing a full `[B*T,1024]` distance tensor repeatedly where possible.
- DAC Conv1d/Snake1d residual blocks. Many small Conv1d operations and custom activation dominate codec decode/encode.
- HuBERT hidden-state accumulation. Avoid storing every hidden state if DinoML can stream an accumulated sum while preserving exact output tuple semantics for this model path.

Medium priority:

- Conv1d feature extractor/positional conv kernels for HuBERT.
- LayerNorm + Linear + GELU FFN fusion in HuBERT.
- ConvTranspose1d decoder provider path with exact output-padding mutation.

Lower priority:

- Semantic encoder ELU+Conv1d residual fusion.
- End-to-end resampling on GPU. Useful later, but first parity can keep it in preprocessing.

## 11. Runtime staging plan

Stage 1: config and ABI loader. Parse nested Higgs/DAC/HuBERT configs, derive hop length, frame rate, codebook bits, quantizer count, and reject legacy non-HF configs unless routed through a converter.

Stage 2: decode-only parity. Accept `[B,Q,T]` codes, run RVQ embedding sum, `fc2`, and DAC decoder. This avoids HuBERT and resampling while proving audio-code ABI.

Stage 3: encode acoustic-only scaffold. Run DAC encoder and RVQ on synthetic fused embeddings or with semantic branch stubbed, only for shape plumbing.

Stage 4: full encode parity with HuBERT semantic features and fixed CPU resample/preprocessor.

Stage 5: round-trip `forward` with final crop and bandwidth variants.

Stage 6: add optimized RVQ, Conv1d/ConvTranspose1d, and HuBERT attention/FFN fusions.

Stub initially: training-only losses, semantic decoder, feature extractor mask wiring, GPU resampling, original checkpoint conversion.

## 12. Parity and validation plan

- Config derivation tests: official config yields hop 960, frame rate 25, max quantizers 8, semantic downsample 2.
- Feature extractor tests: mono input, stereo rejection, right padding to multiple of 960, `padding_mask` shape.
- Custom op tests: Snake1d, RVQ nearest-codebook indices, RVQ encode/decode packing, ConvTranspose1d output length with mutated output padding.
- Decode-only parity: random valid codes `[B,Q,T]`, compare waveform to Transformers for fp32. Recommended tolerances: fp32 `1e-5` to `1e-4`; fp16 only after provider-specific baselines.
- Encode branch parity: fixed short waveform, compare `audio_codes` exactly for fp32 where deterministic argmax ties are not present.
- End-to-end parity: `forward(input_values)` should return codes and cropped audio length equal to original input length.
- Bandwidth tests: official values 0.5/1/1.5/2 select 2/4/6/8 quantizers; unsupported bandwidth rejects.

No DinoML tests or imports were run for this audit.

## 13. Performance probes

- CPU preprocessing throughput: feature extraction padding plus 24 kHz to 16 kHz resample.
- HuBERT semantic branch throughput by padded sample length and batch size.
- DAC encoder-only and decoder-only throughput by code frames.
- RVQ loop latency by `Q`, `B*T`, and codebook size 1024.
- Decode-only requests/sec for fixed `[B,8,T]` code tensors.
- End-to-end round-trip latency split into resample, HuBERT, acoustic encoder, RVQ, decoder.
- Memory probes: hidden-state stack for 13 HuBERT states, RVQ distance matrix, ConvTranspose1d temporaries.
- Provider comparison: generic Conv1d vs im2col+GEMM for small audio convs; dense RVQ GEMM+argmax fused vs unfused.

## 14. Skip/defer list

- Training, losses, commitment/codebook updates, quantizer dropout.
- Semantic decoder unless a caller needs hidden semantic reconstruction.
- Original `model.pth` conversion inside DinoML runtime.
- GPU resampling and streaming chunk boundary handling.
- HuBERT CTC or sequence-classification heads.
- General masked attention output materialization; tokenizer only needs hidden states.
- General boolean scatter/gather; this family does not use multimodal placeholder scatter.
- Multi-GPU/tensor parallel execution.

## 15. Final implementation checklist

- [ ] Parse `HiggsAudioV2TokenizerConfig` plus nested DAC/HuBERT configs.
- [ ] Implement feature extractor ABI or require pre-padded `[B,1,L]` tensors.
- [ ] Add source-faithful Conv1d/ConvTranspose1d coverage for DAC/HuBERT NCL tensors.
- [ ] Implement Snake1d.
- [ ] Implement HuBERT encoder path with hidden-state collection or accumulated hidden-state average.
- [ ] Implement semantic feature resample/pad/downsample policy, initially in CPU preprocessing.
- [ ] Implement RVQ nearest-codebook encode with public `[B,Q,T]` packing.
- [ ] Implement RVQ decode and DAC decoder path.
- [ ] Add bandwidth-to-quantizer-count guards.
- [ ] Fold or reject weight-norm-parametrized checkpoint tensors before compile.
- [ ] Add decode-only parity tests.
- [ ] Add full encode and round-trip parity tests.
- [ ] Benchmark preprocessing, HuBERT, RVQ, and DAC decoder separately.
