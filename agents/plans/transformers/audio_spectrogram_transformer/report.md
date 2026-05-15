# Audio Spectrogram Transformer Transformers Family Audit

Primary target: audio-only spectrogram encoder plus audio classification on CUDA. This report treats raw waveform fbank extraction as a CPU/data-pipeline concern first, then targets DinoML parity for the fixed-shape AST encoder and classifier head. AST is not a generation model: there is no decoder, token sampling, or autoregressive KV cache.

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model family: audio_spectrogram_transformer
Primary model id: MIT/ast-finetuned-audioset-10-10-0.4593
Primary task: raw mono waveform -> log-mel fbank -> AST encoder -> audio-classification logits
Config source: Hugging Face config.json and preprocessor_config.json files fetched from model repos listed below.
Source files inspected:
- transformers/src/transformers/models/audio_spectrogram_transformer/configuration_audio_spectrogram_transformer.py
- transformers/src/transformers/models/audio_spectrogram_transformer/feature_extraction_audio_spectrogram_transformer.py
- transformers/src/transformers/models/audio_spectrogram_transformer/modeling_audio_spectrogram_transformer.py
- transformers/src/transformers/models/audio_spectrogram_transformer/modular_audio_spectrogram_transformer.py
- transformers/src/transformers/models/audio_spectrogram_transformer/convert_audio_spectrogram_transformer_original_to_pytorch.py
- transformers/tests/models/audio_spectrogram_transformer/test_modeling_audio_spectrogram_transformer.py
- transformers/tests/models/audio_spectrogram_transformer/test_feature_extraction_audio_spectrogram_transformer.py
Any missing files or assumptions: no remote-code files are needed for inspected MIT checkpoints. `modeling_audio_spectrogram_transformer.py` is generated from `modular_audio_spectrogram_transformer.py`; inspect the generated file for runtime behavior and edit the modular file for future Transformers source changes. Two apparent MIT repos (`MIT/ast-finetuned-esc50`, `MIT/ast-finetuned-audioset-10-10-0.4593-finetuned-speech-commands-v2`) returned 401 and are not included as primary configs.
```

Representative checkpoint configs inspected:

| Model id | Source | Why included |
|---|---|---|
| `MIT/ast-finetuned-audioset-10-10-0.4593` | HF `config.json`, `preprocessor_config.json` | common AudioSet baseline and Transformers integration-test target |
| `MIT/ast-finetuned-audioset-10-10-0.448` | HF `config.json`, `preprocessor_config.json` | same structure, alternate AudioSet checkpoint |
| `MIT/ast-finetuned-audioset-12-12-0.447` | HF `config.json`, `preprocessor_config.json` | stride variation changes patch grid/sequence length |
| `MIT/ast-finetuned-audioset-14-14-0.443` | HF `config.json`, `preprocessor_config.json` | stride variation changes attention size |
| `MIT/ast-finetuned-audioset-16-16-0.442` | HF `config.json`, `preprocessor_config.json` | coarsest inspected AudioSet patch grid |
| `MIT/ast-finetuned-speech-commands-v2` | HF `config.json`, `preprocessor_config.json` | short-window task with different `max_length`, labels, mean/std |

Relevant source anchors at this commit:

- `ASTConfig` defaults: `configuration_audio_spectrogram_transformer.py:24`.
- Feature extractor fbank path: `feature_extraction_audio_spectrogram_transformer.py:36`, `:68`, `:104`, `:155`, `:158`.
- Patch embedding and token/position setup: `modeling_audio_spectrogram_transformer.py:39`, `:64`.
- Encoder attention, MLP, blocks: `modeling_audio_spectrogram_transformer.py:102`, `:130`, `:179`, `:195`.
- Base model pooling and classifier head: `modeling_audio_spectrogram_transformer.py:257`, `:309`, `:327`.
- Conversion script stride variants and label counts: `convert_audio_spectrogram_transformer_original_to_pytorch.py:33`.

## 2. High-level architecture

AST is an audio-only, ViT-like encoder over log-mel spectrogram patches:

```text
CPU raw mono waveform fbank extraction
-> fixed [B, max_length, num_mel_bins] log-mel tensor
-> unsqueeze/transpose to NCHW-like [B, 1, num_mel_bins, max_length]
-> strided Conv2d patch embedding
-> flatten patches to tokens, prepend CLS + distillation token, add learned absolute positions
-> noncausal Transformer encoder
-> final LayerNorm
-> average CLS/distillation tokens
-> LayerNorm + Linear classifier
```

Stage decomposition:

| Stage | Runtime boundary | Notes |
|---|---|---|
| Audio loading/resampling | CPU/data pipeline | Source expects mono waveform samples at the feature extractor sampling rate, normally 16 kHz. It rejects batched numpy rank > 2 as non-mono. |
| Fbank extraction | CPU/data pipeline by default | TorchAudio Kaldi fbank when available; otherwise numpy/STFT mel path. Output is fixed `[B, max_length, num_mel_bins]`. |
| Patch embedding | GPU graph target | `Conv2d(1 -> hidden_size, kernel=(patch, patch), stride=(frequency_stride, time_stride))` after layout-sensitive transpose. |
| Encoder | GPU graph target | Pre-LN bidirectional MHA + GELU MLP, repeated `num_hidden_layers`. No causal mask or cache. |
| Pool/classifier | GPU graph target | Pool is `(sequence[:,0] + sequence[:,1]) / 2`, then classifier LayerNorm and `Linear(hidden_size -> num_labels)`. |
| Top-k/label mapping | Controller/postprocess | Not in model forward. Multi-label thresholding policy is application-level, not encoded in source forward. |

Validation can be split into processor fbank parity, patch embedding token parity, single block parity, full encoder/pool parity, and classifier logit parity.

## 3. Important config dimensions

Source defaults from `ASTConfig`:

| Field | Default | Operator significance |
|---|---:|---|
| `hidden_size` | 768 | Encoder width and classifier input width. |
| `num_hidden_layers` | 12 | Number of repeated encoder blocks. |
| `num_attention_heads` | 12 | MHA heads. |
| `head_dim` | absent | Source computes `hidden_size // num_attention_heads` unless a config adds `head_dim`. |
| `intermediate_size` | 3072 | MLP expansion. |
| `hidden_act` | `gelu` | Activation in FFN. |
| `hidden_dropout_prob` | 0.0 | Dropout is inactive in inference. |
| `attention_probs_dropout_prob` | 0.0 | Attention dropout is inactive in inference. |
| `layer_norm_eps` | 1e-12 | LayerNorm epsilon. |
| `patch_size` | 16 | Conv2d kernel height and width. Source annotation allows list/tuple, but model code uses it as a scalar in shape math. |
| `qkv_bias` | true | Q/K/V Linear bias switch; output projection always has bias. |
| `frequency_stride` | 10 | Conv2d stride over mel-frequency axis after transpose. |
| `time_stride` | 10 | Conv2d stride over time axis after transpose. |
| `max_length` | 1024 | Spectrogram frame count expected by position embedding shape. |
| `num_mel_bins` | 128 | Spectrogram mel-bin count and Conv2d input height. |
| `cache support` | none | Encoder-only non-generation runtime. |

For an input feature tensor `[B, T, F]`:

```text
F_out = floor((F - patch_size) / frequency_stride) + 1
T_out = floor((T - patch_size) / time_stride) + 1
seq_len = F_out * T_out + 2
```

Representative checkpoint sweep:

| Model id | Task labels | `max_length` | `patch` | `stride(f,t)` | Patch grid `F_out x T_out` | `seq_len` | Feature mean/std |
|---|---:|---:|---:|---|---:|---:|---|
| `MIT/ast-finetuned-audioset-10-10-0.4593` | 527 | 1024 | 16 | 10 x 10 | 12 x 101 | 1214 | -4.2677393 / 4.5689974 |
| `MIT/ast-finetuned-audioset-10-10-0.448` | 527 | 1024 | 16 | 10 x 10 | 12 x 101 | 1214 | -4.2677393 / 4.5689974 |
| `MIT/ast-finetuned-audioset-12-12-0.447` | 527 | 1024 | 16 | 12 x 12 | 10 x 85 | 852 | -4.2677393 / 4.5689974 |
| `MIT/ast-finetuned-audioset-14-14-0.443` | 527 | 1024 | 16 | 14 x 14 | 9 x 73 | 659 | -4.2677393 / 4.5689974 |
| `MIT/ast-finetuned-audioset-16-16-0.442` | 527 | 1024 | 16 | 16 x 16 | 8 x 64 | 514 | -4.2677393 / 4.5689974 |
| `MIT/ast-finetuned-speech-commands-v2` | 35 | 128 | 16 | 10 x 10 | 12 x 12 | 146 | -6.845978 / 5.5654526 |

Config fields commonly omitted by checkpoints but supplied by source/default config include `head_dim` absent with effective `64` for inspected 768/12 configs, `hidden_dropout_prob=0.0`, `attention_probs_dropout_prob=0.0`, `layer_norm_eps=1e-12`, and no generation/cache fields.

## 3a. Family variation traps

- `max_length`, `frequency_stride`, and `time_stride` change the position-embedding parameter shape and attention sequence length. Do not load a 10x10 checkpoint into a 12x12/14x14/16x16 config without weight conversion.
- There is no runtime 2D position interpolation in `ASTModel.forward`. Position embeddings must already match `F_out * T_out + 2`; any resizing is an offline conversion/checkpoint-preparation issue.
- Source patch layout is axis-sensitive: input features are `[B, time, mel]`, then `unsqueeze(1)` gives `[B, 1, time, mel]`, then `transpose(2, 3)` gives `[B, 1, mel, time]` before Conv2d.
- `frequency_stride` maps to Conv2d height stride after transpose; `time_stride` maps to Conv2d width stride. A layout pass must rewrite these axes explicitly.
- `patch_size` is typed as `int | list | tuple`, but `get_shape` subtracts it from integers. DinoML should initially admit scalar integer `patch_size` only unless a tuple path is verified.
- The modeling code supports `config.head_dim` if present, allowing `num_heads * head_dim != hidden_size`. Inspected checkpoints omit it. A robust loader should require `q/k/v` output width to match stored weights.
- `qkv_bias` controls only Q/K/V bias. `o_proj`, MLP, Conv2d patch projection, and classifier dense all have bias.
- `attention_mask` is optional and mostly unused by shipped preprocessors (`return_attention_mask=false`), but source can build a bidirectional additive mask from it.
- `ASTFeatureExtractor` has two fbank implementations: TorchAudio Kaldi fbank when speech/torchaudio is available, otherwise a numpy `spectrogram(...)` path with explicit STFT/mel parameters. Exact CPU preprocessing parity should pin which path is used.
- AudioSet and Speech Commands use different `max_length`, `num_labels`, and normalization statistics while sharing the same encoder width/depth.
- Training loss selection in `ASTForAudioClassification` is irrelevant to first inference integration. Do not lower labels/loss unless training is in scope.

## 4. Operator coverage checklist

Tensor/layout ops:

- Rank checks for fixed `[B, T=max_length, F=num_mel_bins]` input features.
- `unsqueeze(dim=1)`, `transpose(2,3)`, Conv2d input layout `[B,1,F,T]`.
- `flatten(start_dim=2)`, `transpose(1,2)`, `contiguous`, `reshape/view`.
- `expand` CLS/distillation tokens from `[1,1,H]` to `[B,1,H]`.
- `cat(dim=1)` for two special tokens plus patch tokens.
- Add learned position embedding `[1, seq_len, H]`.
- Slice/index `sequence[:, 0]`, `sequence[:, 1]`, add, scalar multiply/divide by 2.

Neural network primitives:

- Conv2d patch projection: common baseline `Conv2d(1 -> 768, kernel=16x16, stride=10x10, bias=True)`.
- LayerNorm over last dim `H`, epsilon `1e-12`.
- Linear Q/K/V: `Linear(768 -> 768, bias=qkv_bias)` for common checkpoints.
- Linear attention output: `Linear(768 -> 768, bias=True)`.
- MLP: `Linear(768 -> 3072) -> GELU -> Linear(3072 -> 768)`.
- Classifier: `LayerNorm(768) -> Linear(768 -> num_labels)`, with `num_labels=527` or `35` for inspected checkpoints.
- Dropout nodes should fold to identity for inference.

Attention primitives:

- Noncausal self-attention only.
- MHA with `num_heads=12`, `head_dim=64` for inspected checkpoints.
- Q/K/V reshape to `[B, seq_len, heads, head_dim]`, transpose to `[B, heads, seq_len, head_dim]`.
- Attention score matmul, scale by `head_dim ** -0.5`, optional additive bidirectional mask, fp32 softmax over keys, cast back, value matmul.
- SDPA/Flash/Flex attention interfaces are declared supported; eager math remains the parity reference.

Preprocessing-coupled ops:

- CPU fbank extraction: Kaldi fbank or numpy STFT/mel/log path.
- Pad/truncate fbank frames to `max_length`.
- Normalize `(x - mean) / (std * 2)`.
- Optional feature tensor conversion to numpy/PyTorch.

Generation/cache ops:

- Not applicable. No autoregressive prefill/decode, no logits sampling, no KV cache.

## 5. Layer/block breakdown

Patch embedding:

```text
input_values: [B, T, F]
x = unsqueeze(input_values, dim=1)              # [B, 1, T, F]
x = transpose(x, 2, 3)                          # [B, 1, F, T]
x = Conv2d(1 -> H, kernel=(P,P), stride=(Sf,St), bias=True)
  # [B, H, F_out, T_out]
x = flatten(x, start_dim=2).transpose(1, 2)     # [B, F_out*T_out, H]
x = concat(cls_token, distillation_token, x)    # [B, seq_len, H]
x = x + position_embeddings                     # [1, seq_len, H]
```

Encoder block, repeated `num_hidden_layers`:

```text
residual = x
y = LayerNorm(x, eps=1e-12)
q = Linear(H -> heads*head_dim, bias=qkv_bias)(y)
k = Linear(H -> heads*head_dim, bias=qkv_bias)(y)
v = Linear(H -> heads*head_dim, bias=qkv_bias)(y)
q,k,v = view/transpose to [B, heads, S, head_dim]
a = Attention(q, k, v, bidirectional optional mask)
a = transpose/reshape to [B, S, heads*head_dim]
a = Linear(heads*head_dim -> H, bias=True)(a)
x = residual + a
residual = x
y = LayerNorm(x, eps=1e-12)
y = Linear(H -> intermediate)(y)
y = GELU(y)
y = Linear(intermediate -> H)(y)
x = residual + y
```

Base output and classifier:

```text
sequence = LayerNorm(x, eps=1e-12)
pooled = (sequence[:, 0] + sequence[:, 1]) / 2
logits = Linear(H -> num_labels)(LayerNorm(pooled))
```

## 6. Attention requirements

AST requires encoder-style bidirectional self-attention:

| Attribute | Requirement |
|---|---|
| Causal? | No. `ASTAttention.is_causal = False`. |
| Type | Self-attention only; no cross-attention. |
| Heads | MHA. Common configs: 12 query heads and 12 KV heads. |
| Head dim | Common effective `64`; source allows config `head_dim` override. |
| Masking | Optional bidirectional additive mask produced by `create_bidirectional_mask`; shipped preprocessors default to no attention mask. |
| Packed/varlen | Not required. Inputs are padded/truncated to fixed fbank length. |
| Sliding/local | Not present. |
| Position interaction | None inside attention; positions are learned absolute embeddings added before blocks. |
| KV cache | Not applicable. |
| Backend | Source can dispatch through eager, SDPA, FlashAttention, or FlexAttention interfaces. Eager parity uses fp32 softmax then casts to query dtype. |

There is no decoder cache. The only cache-like optimization worth separating is offline or data-pipeline caching of fbank feature tensors, and possibly reusable encoder outputs for applications that repeatedly classify the same clip.

## 7. Position encoding and custom math

AST uses learned absolute position embeddings with two leading special-token positions:

```python
def ast_patch_grid(max_length, num_mel_bins, patch_size, f_stride, t_stride):
    f_out = (num_mel_bins - patch_size) // f_stride + 1
    t_out = (max_length - patch_size) // t_stride + 1
    return f_out, t_out, f_out * t_out + 2
```

There is no RoPE, ALiBi, relative bias, convolutional positional embedding, or runtime interpolation. The source constructs `position_embeddings` as `[1, num_patches + 2, hidden_size]` during module initialization and adds it directly in `ASTEmbeddings.forward`.

Important interpolation note: the current in-library AST model does not expose ViT-style `interpolate_pos_encoding` in forward. DinoML should reject mismatched input `T/F` or mismatched position-embedding shape instead of silently resizing. Any 2D interpolation for converted checkpoints should be modeled as an offline weight-conversion step, not a runtime graph requirement for this report.

Precomputable:

- Position embeddings are constants.
- CLS and distillation tokens are constants expanded by batch.
- Patch grid and sequence length are static for admitted configs.

Dynamic:

- Batch size only, unless DinoML later admits multiple fixed `max_length` buckets as separate compiled variants.

## 8. Preprocessing and input packing

Feature extractor runtime contract:

| Field | Default / inspected values | Notes |
|---|---|---|
| Input waveform | mono `float32` array/list | Batched numpy rank > 2 is rejected. Stereo is not supported. |
| Sampling rate | 16000 | Passing a mismatched `sampling_rate` raises. Omitting it warns. |
| TorchAudio path | `torchaudio.compliance.kaldi.fbank` | Uses `sample_frequency`, `window_type="hanning"`, `num_mel_bins`; source comment says waveform should not be normalized first. |
| Numpy fallback | STFT/mel spectrogram | `frame_length=400`, `hop_length=160`, `fft_length=512`, `power=2.0`, `center=False`, `preemphasis=0.97`, `remove_dc_offset=True`, log mel floor `1.192092955078125e-07`. |
| Mel filters fallback | 257 frequency bins, min 20 Hz, max sampling_rate/2, Kaldi mel, triangularized in mel space | Built only when TorchAudio speech dependency is unavailable. |
| Window fallback | Hann length 400, `periodic=False` | Note spelling: TorchAudio path uses `"hanning"`. |
| Padding/truncation | zero-pad or truncate frames to `max_length` | Shape after extraction is `[max_length, num_mel_bins]`. |
| Normalization | `(x - mean) / (std * 2)` | Mean/std differ by checkpoint family. |
| Attention mask | Config field exists, default false in inspected preprocessors | Feature extractor `__call__` does not add attention masks itself beyond model input naming; no variable-length mask is needed for fixed fbank tensor. |

CPU/data-pipeline work:

- Audio decode, resampling, mono conversion.
- Fbank extraction, padding/truncation, normalization.
- Optional caching of `[B, T, F]` feature tensors.

GPU/runtime work:

- Starts at `input_values` `[B, max_length, num_mel_bins]`.
- No special placeholder tokens, token type IDs, packed sequence descriptors, or `cu_seqlens`.

## 9. Graph rewrite / lowering opportunities

### Rewrite: AST patch Conv2d -> WindowFlatten + GEMM

Source pattern:

```text
[B,T,F] -> unsqueeze -> transpose to [B,1,F,T]
-> Conv2d(1,H,kernel=(P,P),stride=(Sf,St),padding=0,dilation=1,groups=1)
-> flatten(2).transpose(1,2)
```

Replacement:

```text
ExtractSlidingWindows over [F,T] in source Conv2d order
-> flatten each P*P window
-> MatMul(weight_flat.T) + bias
-> tokens [B, F_out*T_out, H]
```

Preconditions:

- `in_channels == 1`, `groups == 1`.
- `padding == 0`, `dilation == 1`.
- Kernel is square `(patch_size, patch_size)` and stride is `(frequency_stride, time_stride)` after the AST transpose.
- Input feature shape matches config: `T == max_length`, `F == num_mel_bins`.
- Weight flatten preserves PyTorch Conv2d memory order `[out_channels, in_channels, kh, kw]`; with one input channel, flatten `[H, P*P]`.
- Token order must match `Conv2d(...).flatten(2)`: frequency output index outer, time output index inner for contiguous NCHW output flattening.

Failure cases:

- Tuple/list `patch_size` not verified.
- Any nonzero padding/dilation/groups.
- Layout pass that treats source `[B,T,F]` as `[B,F,T]` without the explicit transpose.

Parity test sketch:

- Generate random `[B,T,F]`, random Conv2d weights/bias, compare PyTorch patch embedding output to WindowFlatten+GEMM for 10x10 and 16x16 stride configs.

### Rewrite: Fold inference Dropout

Source pattern: dropout after embeddings, attention output, and MLP output.

Replacement: identity in eval/inference.

Preconditions:

- Model is in inference/eval mode.
- No training loss or stochastic tracing target.

### Rewrite: Attention QKV packing

Source pattern: three independent `Linear(H -> heads*head_dim)` operations.

Replacement:

```text
Linear(H -> 3*heads*head_dim) -> split Q,K,V
```

Preconditions:

- Same input tensor and dtype.
- Same bias policy for Q/K/V (`qkv_bias` true or all absent).
- Weight rows concatenate in source order Q, K, V.
- Output split produces three `[B,S,heads*head_dim]` blocks before view/transpose.

Failure cases:

- Config/weights with different per-projection dims.
- Debug output capture requiring separate named projection tensors.

### Rewrite: Pool special tokens as small fused op

Source pattern:

```text
pooled = (sequence[:, 0] + sequence[:, 1]) / 2
```

Replacement: fused gather-add-scale over two token rows.

Preconditions:

- Distillation token is present. AST always prepends exactly CLS and distillation tokens in inspected source.
- Consumer is classifier head or exported pooler output.

### Layout guard: protect spectrogram axis semantics

Candidate optimized layout:

- Keep feature tensors as `[B,T,F]` at API boundary for parity.
- Internally fuse `unsqueeze + transpose + Conv2d + flatten + transpose` into a patch GEMM without materializing NCHW.

Required axis rewrites:

- Source `transpose(2,3)` changes time/frequency.
- Conv2d height corresponds to frequency, width to time.
- Flatten order is `[frequency_patch_index, time_patch_index]`, not `[time, frequency]`.

Failure cases:

- NHWC/channel-last pass that rewrites Conv2d but forgets to preserve patch token ordering or stride axis mapping.
- Downstream learned position embedding assumes original patch order; any layout-translated token order requires a matching position embedding permutation, which should be avoided for first integration.

## 10. Kernel fusion candidates

Highest priority:

- Patch Conv2d-to-GEMM or direct patch embedding kernel. This is the AST-specific front door and determines sequence layout correctness.
- LayerNorm + Linear around Q/K/V and MLP. Encoder is dominated by repeated LayerNorm, GEMM, GELU, residual.
- Fused scaled dot-product attention for noncausal MHA with fp32 softmax accumulation. Long AudioSet 10x10 sequence is 1214 tokens, so attention is expensive.

Medium priority:

- QKV packed projection. Removes three GEMM launches per layer when weight packing is acceptable.
- GELU FFN fusion or GEMM bias+GELU epilogue for `Linear(H -> 4H)`.
- Classifier LayerNorm + Linear for small final head.
- Special-token pool fused gather/add/scale.

Lower priority:

- Fbank extraction on GPU. Useful only if end-to-end raw waveform throughput becomes a bottleneck; first integration can own fixed fbank tensors.
- Dropout elimination pass. Simple but not performance-dominant once inference graph lowering already folds it.

## 11. Runtime staging plan

1. Parse AST config and checkpoint metadata; admit scalar `patch_size`, fixed `max_length`, fixed `num_mel_bins`, no generation.
2. Load weights for patch projection, CLS/distillation tokens, position embeddings, encoder blocks, final LayerNorm, and optional classifier.
3. Implement/fuse patch embedding parity from `[B,T,F]` to token sequence.
4. Run one encoder block parity with eager attention and fp32 softmax reference.
5. Run full `ASTModel` parity for pooler output on random fbank tensors.
6. Add `ASTForAudioClassification` classifier parity for AudioSet and Speech Commands label counts.
7. Add optional attention-mask path only after unmasked fixed-fbank parity is stable.
8. Add optimized SDPA/FlashAttention and QKV packing under strict shape/layout guards.
9. Keep raw waveform fbank extraction as a CPU reference/preprocessor path; consider GPU preprocessing later as an optimization, not part of the first neural graph.

Stub initially:

- Training losses and labels.
- Runtime position interpolation.
- Raw waveform GPU preprocessing.
- Attention output capture unless needed for debugging.

## 12. Parity and validation plan

- Feature extractor parity: compare HF `ASTFeatureExtractor` output for short, exact, and overlong mono waveforms. Pin TorchAudio availability because TorchAudio and numpy fallback are separate implementations.
- Patch embedding parity: random `[B,max_length,num_mel_bins]` features for 10x10, 12x12, 14x14, 16x16, and Speech Commands 128-frame configs.
- Single-layer parity: one `ASTLayer` with eager attention, fp32 tolerance `rtol=1e-5`, `atol=1e-5`; fp16/bf16 looser after LayerNorm/softmax validation.
- Full encoder parity: `last_hidden_state` and `pooler_output` for random fbank input, batch sizes 1 and >1.
- Classification parity: compare logits for `MIT/ast-finetuned-audioset-10-10-0.4593`; Transformers test expects logits shape `[1,527]` and has a known first-3-logit slice.
- Layout parity: explicit test that a synthetic one-hot/monotonic spectrogram produces the same patch token order after any Conv2d lowering.
- Rejection tests: mismatched `max_length`, mismatched `num_mel_bins`, tuple `patch_size`, position embedding length mismatch, unsupported stereo/rank-3 raw input in preprocessing.

Recommended tolerances:

- fp32: `rtol=1e-4`, `atol=1e-4` end-to-end; tighter for isolated GEMMs.
- fp16/bf16: compare against HF reduced precision with `rtol=1e-2`, `atol=1e-2`, with attention softmax accumulation checked separately.

## 13. Performance probes

- CPU preprocessing throughput: waveforms/sec for TorchAudio fbank and numpy fallback separately.
- Patch embedding throughput: `[B,1024,128]` to `[B,1212,768]` tokens for stride 10 and shorter grids for strides 12/14/16.
- Encoder-only throughput: sequence-length sweep `146`, `514`, `659`, `852`, `1214`.
- Attention backend comparison: eager decomposition vs SDPA vs FlashAttention for noncausal MHA.
- Batch-size sweep: `B=1,2,4,8,16` with fixed AudioSet 10x10 shape.
- Classifier-only probe: pool + head cost, mostly to confirm it is not the bottleneck.
- Memory probe: activation/temp memory vs sequence length; no KV cache memory is expected.
- End-to-end requests/hour split into preprocessing time and neural runtime time.

## 14. Skip/defer list

- Training, labels, regression/classification loss computation.
- Gradient checkpointing.
- Runtime 2D position interpolation or arbitrary input resolutions.
- Tuple/list `patch_size` until source behavior is clarified.
- Raw waveform GPU fbank extraction.
- Stereo/multi-channel audio preprocessing.
- Generation controllers, beam search, sampling, KV cache.
- Quantization-specific loading or fused quantized AST kernels.
- Attention output/hidden-state capture for production path, unless needed for debugging parity.

## 15. Final implementation checklist

- [ ] Parse `ASTConfig` and feature-extractor config.
- [ ] Admit fixed `[B, max_length, num_mel_bins]` fbank input tensors.
- [ ] Reject unsupported tuple/list `patch_size` and position-length mismatches.
- [ ] Load patch Conv2d, special tokens, position embeddings, encoder, final LayerNorm, and classifier weights.
- [ ] Implement AST patch embedding with faithful time/frequency transpose and patch token order.
- [ ] Implement LayerNorm epsilon `1e-12`.
- [ ] Implement noncausal MHA with optional additive bidirectional mask and fp32 softmax parity.
- [ ] Implement GELU MLP blocks and residuals.
- [ ] Implement pooler `(CLS + distillation) / 2`.
- [ ] Implement classifier `LayerNorm -> Linear`.
- [ ] Add Conv2d-to-GEMM rewrite guarded by AST patch preconditions.
- [ ] Add QKV packing rewrite guarded by identical input and Q/K/V dims.
- [ ] Add feature extractor parity tests for TorchAudio and/or pinned numpy fallback.
- [ ] Add random patch/block/full-model parity tests for 10x10 and Speech Commands configs.
- [ ] Benchmark preprocessing, patch embedding, encoder sequence-length sweep, and attention backend variants.
