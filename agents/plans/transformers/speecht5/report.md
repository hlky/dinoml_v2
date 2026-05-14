# SpeechT5 Transformers Family Audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: microsoft/speecht5_asr, microsoft/speecht5_tts, microsoft/speecht5_vc, microsoft/speecht5_hifigan
Config source: local snapshots under agents/plans/transformers/speecht5/_sources/
Source files inspected:
  X:/H/transformers/src/transformers/models/speecht5/modeling_speecht5.py
  X:/H/transformers/src/transformers/models/speecht5/configuration_speecht5.py
  X:/H/transformers/src/transformers/models/speecht5/feature_extraction_speecht5.py
  X:/H/transformers/src/transformers/models/speecht5/processing_speecht5.py
  X:/H/transformers/src/transformers/models/speecht5/tokenization_speecht5.py
  X:/H/transformers/src/transformers/models/speecht5/number_normalizer.py
Any missing files or assumptions:
  The official SpeechT5 tokenizer vocab is a SentencePiece file named spm_char.model, not vocab.json. I did not snapshot the binary tokenizer model because the runtime graph audit only needs token id contracts.
  SpeechT5HifiGanConfig in source declares model_type="speecht5_hifigan", while microsoft/speecht5_hifigan config.json has model_type="hifigan"; route by architecture/config class rather than only model_type.
```

Primary source URLs at the pinned commit:

- `modeling_speecht5.py`: https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/speecht5/modeling_speecht5.py
- `configuration_speecht5.py`: https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/speecht5/configuration_speecht5.py
- `feature_extraction_speecht5.py`: https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/speecht5/feature_extraction_speecht5.py
- `processing_speecht5.py`: https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/speecht5/processing_speecht5.py
- `tokenization_speecht5.py`: https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/speecht5/tokenization_speecht5.py

This report targets inference. Training-only SpecAugment, guided attention loss, spectrogram loss, LayerDrop, gradient checkpointing, and dropout randomness are out of runtime scope except where the source applies dropout in evaluation in the speech decoder prenet.

## 2. High-level architecture

SpeechT5 is an encoder-decoder family with pluggable modality-specific prenets and postnets:

```text
ASR: raw waveform -> speech conv/prenet -> encoder -> text decoder with KV cache -> tied LM head -> token logits/generation
TTS: text ids -> text prenet -> encoder -> mel decoder with speaker embeddings and KV cache -> mel postnet -> optional HiFi-GAN vocoder
VC: raw waveform -> speech conv/prenet -> encoder -> mel decoder with speaker embeddings and KV cache -> mel postnet -> optional HiFi-GAN vocoder
HiFi-GAN: log-mel spectrogram -> Conv1d/ConvTranspose1d/resblocks -> tanh waveform
```

Stage decomposition:

- CPU/data pipeline: tokenization with SentencePiece for text; waveform padding and optional zero-mean/unit-variance normalization for speech input; optional log-mel extraction for training labels/audio targets.
- Speech encoder prenet: raw waveform `[B, T_audio]` to conv features `[B, T_feat, hidden]`, including conv length reduction and attention-mask downsampling.
- Text prenet: token ids `[B, T_text]` to embeddings plus scaled sinusoidal position encoding.
- Shared encoder: noncausal self-attention with learned relative-position bias and FFN, repeated `encoder_layers`.
- Text decoder: causal self-attention, cross-attention, FFN, sinusoidal token positions, dynamic KV cache, LM head.
- Speech decoder: autoregressive mel loop controlled outside the normal `GenerationMixin`, using speaker embeddings, reduction-factor frames, stop probabilities, postnet, and optional vocoder.
- Vocoder: separately stageable convolutional network; consumes completed mel spectrograms and has no attention/cache.

The speech encoder, text encoder, text decoder, mel decoder, postnet, and vocoder can be validated independently. TTS/VC first integration can stub the vocoder and return mel spectrograms before waveform parity.

## 3. Important config dimensions

Common production SpeechT5 configs:

| Field | ASR `microsoft/speecht5_asr` | TTS `microsoft/speecht5_tts` | VC `microsoft/speecht5_vc` |
|---|---:|---:|---:|
| architecture | `SpeechT5ForSpeechToText` | `SpeechT5ForTextToSpeech` | `SpeechT5ForSpeechToSpeech` |
| vocab_size | 81 | 81 | 81 |
| hidden_size | 768 | 768 | 768 |
| encoder_layers | 12 | 12 | 12 |
| decoder_layers | 6 | 6 | 6 |
| encoder_attention_heads | 12 | 12 | 12 |
| decoder_attention_heads | 12 | 12 | 12 |
| head_dim | 64 | 64 | 64 |
| encoder_ffn_dim | 3072 | 3072 | 3072 |
| decoder_ffn_dim | 3072 | 3072 | 3072 |
| hidden_act | `gelu` | `gelu` | `gelu` |
| conv_dim | 7 x 512 | 7 x 512 | 7 x 512 |
| conv_stride | `5,2,2,2,2,2,2` | same | same |
| conv_kernel | `10,3,3,3,3,2,2` | same | same |
| input/logit ratio | 320 | 320 | 320 |
| max_speech_positions | 4000 | 1876 | 1876 |
| max_text_positions | 450 | 600 | 450 |
| num_mel_bins | 80 | 80 | 80 |
| reduction_factor | 2 | 2 | 2 |
| speaker_embedding_dim | 512 | 512 | 512 |
| use_cache | true | true | true |
| tie_word_embeddings | true | true | true |

Representative checkpoint sweep:

| Checkpoint | Scope | Operator-significant variation |
|---|---|---|
| `hf-internal-testing/tiny-random-SpeechT5Model` | tiny/debug base model | `hidden_size=24`, `encoder_layers=4`, `decoder_layers=4`, `heads=2`, `ffn_dim=4`, `conv_dim=32/32/32`, `conv_stride=4/4/4`, `conv_kernel=8/8/8`, `num_mel_bins=20`; useful for small parity but not production shapes. |
| `microsoft/speecht5_asr` | speech-to-text | Speech conv encoder plus text decoder/LM head. Uses standard token generation and tied LM head. |
| `microsoft/speecht5_tts` | text-to-speech | Text encoder plus speech decoder/postnet; requires speaker embeddings and custom mel generation loop. |
| `microsoft/speecht5_vc` | voice conversion | Speech encoder plus speech decoder/postnet; generation defaults missing speaker embeddings to zeros only in `generate_speech`, while shared `_generate_speech` requires speaker embeddings. |
| `microsoft/speecht5_hifigan` | vocoder | Separate conv-only config: `model_in_dim=80`, `upsample_initial_channel=512`, `upsample_rates=4/4/4/4`, `upsample_kernel_sizes=8/8/8/8`, resblock kernels `3/7/11`, dilations `1/3/5`, `normalize_before=true`. |

Processor/feature extractor configs for ASR/TTS/VC all use `sampling_rate=16000`, `feature_size=1`, `padding_value=0`, `do_normalize=false`, `num_mel_bins=80`, `hop_length=16 ms`, `win_length=64 ms`, `fmin=80`, `fmax=7600`, `mel_floor=1e-10`, `return_attention_mask=true`.

## 3a. Family variation traps

- ASR, TTS, and VC share config fields but instantiate different prenet/postnet combinations. Do not infer runtime inputs from `model_type="speecht5"` alone.
- ASR uses `GenerationMixin` text token generation. TTS/VC use `_generate_speech`, a custom loop over mel frames with stop probabilities and reduction factor.
- Speech input is raw waveform for the speech encoder. Mel spectrograms are decoder inputs/targets and vocoder inputs, not the encoder input.
- The speech decoder prenet deliberately applies dropout even when evaluating. Deterministic parity requires controlling RNG or stubbing the prenet dropout policy for first-pass graph parity.
- `reduction_factor` changes decoder target sequence handling: training shifts labels by taking every `r`th frame, and inference predicts `r` mel frames per decode step but feeds back only the last predicted frame.
- Speaker embeddings are L2-normalized, expanded across decoder time, concatenated with hidden states, then projected by `Linear(hidden_size + speaker_embedding_dim -> hidden_size)` plus ReLU.
- Encoder self-attention has a learned relative position embedding added through query-dependent matmul. Decoder self-attention and cross-attention do not use that relative bias.
- Attention is eager `bmm -> softmax -> dropout -> bmm`, not SDPA/FlashAttention dispatch in source.
- All attention is MHA, not GQA/MQA: `KV heads == query heads`.
- Speech conv/pre/post/vocoder code is channel-first NCL. Any channel-last optimization must be local and guarded because LayerNorm, GroupNorm, BatchNorm, Conv1d, ConvTranspose1d, and transpose consumers are axis-sensitive.
- HiFi-GAN checkpoint config has historical `model_type="hifigan"` while the source config class says `speecht5_hifigan`.
- `tie_word_embeddings=true` means ASR decoder token embeddings and LM head must remain one logical tied weight.

## 4. Operator coverage checklist

Tensor/layout ops:

- `view`, `reshape`, `flatten`, `squeeze`, `unsqueeze`, `transpose`, `contiguous`, `cat`, `stack`, `pad_sequence`, `where`, `cumsum`, `flip`, boolean comparisons, mask expansion, dynamic sequence-length slicing.
- Axis-sensitive NCL/NLC transposes around Conv1d and spectrogram postnet/vocoder.

Neural network primitives:

- Embedding lookup for token ids.
- Linear with bias for Q/K/V/O, FFN, feature projection, decoder prenet, speaker projection, mel head, stop-prob head.
- Linear without bias for ASR LM head.
- Conv1d speech feature extractor: production layers `1->512 k10 s5`, then six `512->512` layers with kernels `3,3,3,3,2,2` and strides `2`.
- GroupNorm with `num_groups=out_channels` on first speech conv when `feat_extract_norm="group"`.
- Optional LayerNorm after every conv when `feat_extract_norm="layer"`; production uses group path.
- LayerNorm over hidden/conv channels.
- BatchNorm1d in speech decoder postnet.
- Conv1d postnet: `80->256`, `256->256` x3, `256->80`, kernel 5, stride 1, padding 2, no bias.
- ConvTranspose1d in HiFi-GAN: `512->256`, `256->128`, `128->64`, `64->32`, kernel 8, stride 4, padding 2.
- HiFi-GAN Conv1d pre/post and residual Conv1d blocks with dilation `1,3,5`.
- Activations: GELU, ReLU, tanh, leaky ReLU, sigmoid, softmax.
- Dropout can be compiled away for most inference paths, except speech decoder prenet's source behavior applies dropout unconditionally.

Attention primitives:

- MHA self-attention and cross-attention with `B*H` batched matmuls.
- Causal masks for decoder self-attention; bidirectional masks for encoder and cross-attention.
- Learned relative position bias for encoder self-attention.
- Dynamic `EncoderDecoderCache` for decoder self and cross-attention.

Preprocessing-coupled ops:

- Waveform mono validation, padding/truncation, optional normalization.
- STFT/log-mel feature extraction for `audio_target` labels and vocoder inputs when generated by processor; this belongs in CPU/data pipeline initially.
- SentencePiece tokenization and optional English number normalization.

## 5. Layer/block breakdown

Speech encoder prenet:

```text
input_values [B, T_audio]
x = unsqueeze channel -> [B, 1, T_audio]
for conv layer i:
  x = Conv1d(Cin -> Cout, kernel=conv_kernel[i], stride=conv_stride[i], bias=conv_bias)
  x = GroupNorm(Cout groups) only first layer for production feat_extract_norm="group"
  x = GELU(x)
x = transpose -> [B, T_feat, conv_dim[-1]]
x = LayerNorm(512)
x = Linear(512 -> 768) + dropout
x = optional SpecAugment only during training or explicit mask_time_indices
x = x + weight-norm grouped Conv1d positional embedding over hidden channels
x = x + sinusoidal positional embedding derived from padding mask
```

Text encoder prenet:

```text
input_ids [B, T_text]
x = Embedding(vocab_size, 768, padding_idx=1)
x = x + alpha * sinusoidal_pe[:T_text]
x = dropout(x)
```

Encoder block, repeated 12 production layers:

```text
position_bias = Embedding(clipped_relative_position + max_relative_position)
x = input LayerNorm + dropout before first layer
attn = MHA(q,k,v=Linear(768 -> 768), heads=12, head_dim=64, bias=True)
attn_scores = bmm(q * 1/sqrt(64), k.T)
attn_scores += matmul(q, position_bias.T)
attn_scores += bidirectional_attention_mask
x = x + dropout(out_proj(softmax(attn_scores) @ v))
x = LayerNorm(x)
x = x + Linear(3072 -> 768)(dropout(GELU(Linear(768 -> 3072)(x))))
x = final LayerNorm(x)
```

Text decoder prenet and postnet:

```text
decoder_input_ids [B, T_dec]
positions = sinusoidal positions from non-pad cumsum plus past length
x = Embedding(vocab_size, 768) * embed_scale + positions
x = dropout(x)
logits = Linear(768 -> vocab_size, bias=False), tied to decoder embedding when tie_word_embeddings
```

Speech decoder prenet and postnet:

```text
decoder_input_values [B, T_mel_steps, 80]
x = ReLU(Linear(80 -> 256)); consistent_dropout(always)
x = ReLU(Linear(256 -> 256)); consistent_dropout(always)
x = Linear(256 -> 768)
x = x + alpha * sinusoidal_pe[:T]
speaker = normalize([B, 512]) -> expand [B, T, 512]
x = ReLU(Linear(1280 -> 768)(cat(x, speaker)))
decoder output [B, T, 768]
before = Linear(768 -> 80 * reduction_factor).view(B, T * reduction_factor, 80)
stop_logits = Linear(768 -> reduction_factor).view(B, T * reduction_factor)
after = before + PostnetConvStack(before)
```

Decoder block, repeated 6 production layers:

```text
self = causal MHA over decoder states, with self KV cache
x = LayerNorm(x + dropout(self.out))
cross = MHA(query=x, key/value=encoder_hidden), with reusable cross KV cache
x = LayerNorm(x + dropout(cross.out))
x = x + FFN(x)
x = final LayerNorm(x)
```

HiFi-GAN vocoder:

```text
spectrogram [B, T_mel, 80] or [T_mel, 80]
if normalize_before: spectrogram = (spectrogram - mean[80]) / scale[80]
x = transpose -> [B, 80, T_mel]
x = Conv1d(80 -> 512, kernel=7, padding=3)
for rates 4,4,4,4:
  x = LeakyReLU(0.1)
  x = ConvTranspose1d(C -> C/2, kernel=8, stride=4, padding=2)
  x = average(3 residual blocks with kernels 3,7,11 and dilations 1,3,5)
x = LeakyReLU
x = Conv1d(32 -> 1, kernel=7, padding=3)
waveform = tanh(x).squeeze/collapse
```

## 6. Attention requirements

Encoder self-attention:

- Noncausal bidirectional MHA.
- Production shape: query/key/value projections `768 -> 768`, 12 heads, head dim 64.
- Relative position bias is query-dependent: `rel_pos_bias = matmul(q, position_bias.T)` with position bias shape `[T, T, 64]`, then reshaped to `[B*H, T, T]`.
- Padding mask is expanded to `[B, 1, T, T]` and added before softmax.

Decoder self-attention:

- Causal MHA with dynamic KV cache.
- Cached keys/values are stored after projection and reshape as `[B, H, T_cache, D]`.
- For production, per layer self-cache key and value are each `[B, 12, T_dec_cache, 64]`.
- Query scaling happens before score matmul.

Decoder cross-attention:

- Noncausal MHA from decoder query to encoder hidden states.
- Cross key/value cache stores projected encoder states as `[B, 12, T_enc, 64]` and marks each layer updated after first use.
- Cross cache can be precomputed after encoder completion for ASR/TTS/VC.

No sliding-window/local attention, ALiBi, RoPE, packed/varlen metadata, GQA, MQA, or native FlashAttention/SDPA dispatch is present in the inspected source.

## 7. Position encoding and custom math

Speech conv length reduction:

```python
def conv_out_length(length, kernels, strides):
    for kernel, stride in zip(kernels, strides):
        length = (length - kernel) // stride + 1
    return length
```

Encoder relative position bias:

```python
def speecht5_relative_positions(seq_len, max_length):
    pos = arange(seq_len)[:, None] - arange(seq_len)[None, :]
    pos = clamp(pos, -max_length, max_length - 1) + max_length
    return embedding(pos)  # [T, T, head_dim]
```

Encoder attention adds it as a content-dependent term:

```python
scores = bmm(q * head_dim**-0.5, k.transpose(-1, -2))
rel = matmul(q.reshape(B * H, T, D).transpose(0, 1), pos_bias.transpose(-2, -1))
scores = scores + rel.transpose(0, 1).view(B * H, T, T)
```

Text decoder positions use pad-aware cumulative positions:

```python
mask = input_ids.ne(pad_token_id).int()
pos = (cumsum(mask, dim=1) + past_key_values_length) * mask + pad_token_id
```

Speech decoder stop condition:

```python
prob = sigmoid(Linear(hidden -> reduction_factor)(last_decoder_output))
done = sum(prob, dim=-1) >= threshold
```

Precomputable: sinusoidal tables, mel filter bank, HiFi-GAN normalization buffers, relative-position index tables for static sequence lengths. Dynamic: speech conv output lengths, padding masks, autoregressive stop decisions, cache lengths.

## 8. Preprocessing and input packing

Speech input contract:

- Input waveform must be mono, float32, shaped `[T]` or batch `[B, T]`.
- Sampling rate must be 16 kHz; feature extractor warns when not supplied and raises on mismatch.
- `audio` path returns raw waveform `input_values` plus optional `attention_mask`; no STFT is run for speech encoder inputs.
- Attention masks are downsampled through the conv output length formula before encoder masking.

Mel feature contract:

- `audio_target` path computes log-mel spectrograms in CPU/data pipeline.
- Window: Hann, `win_length=64 ms`, `sample_size=1024` at 16 kHz.
- Hop: `hop_length=16 ms`, `sample_stride=256`.
- FFT length is `optimal_fft_length(1024)` from Transformers audio utilities.
- Mel bins: 80, Slaney norm/scale, `fmin=80`, `fmax=7600`, floor `1e-10`, `log10`, output transposed to `[T_mel, 80]`.
- TTS/VC decoder and HiFi-GAN consume `[B, T_mel, 80]`.

Text input contract:

- SentencePiece model file `spm_char.model`.
- Tokenizer appends EOS to single or pair input, returns zero token type ids, model ignores token type ids.
- Defaults: `bos_token_id=0`, `pad_token_id=1`, `eos_token_id=2`, `decoder_start_token_id=2`.
- Optional English number normalization is tokenizer-side CPU work.

TTS/VC generation packing:

- Generation starts with one all-zero mel frame `[B, 1, 80]`.
- Every iteration reruns the speech decoder prenet on the entire generated mel sequence, but only the last hidden state is passed to the decoder with KV cache.
- Each decode step predicts `reduction_factor` frames, appends all to accumulated output, but feeds back only the last frame.
- Completed batch entries are collected when stop probability threshold is met after `minlen`; generation hard-stops at `maxlen = int(encoder_len * maxlenratio / reduction_factor)`.

## 9. Graph rewrite / lowering opportunities

### Rewrite: Q/K/V projection grouping

Source pattern: separate `q_proj`, `k_proj`, `v_proj` linears with identical input for self-attention.

Replacement: one packed GEMM/Linear producing `[q, k, v]` row blocks, then split.

Preconditions:

- Self-attention only; cross-attention has different query versus key/value inputs.
- All projections have bias and identical input hidden size.
- Weight pack order must be explicit as all-Q, all-K, all-V, matching source module order.

Failure cases: output attentions requiring exact intermediate tensors should still be reconstructable; cross-attention needs either Q plus packed KV or no grouping.

Parity test: compare one attention layer scores/output with random hidden states and masks.

### Rewrite: encoder relative-bias attention fusion

Source pattern:

```text
scores = q @ k.T
scores += q @ relative_position_embedding.T
scores += mask
softmax(scores)
```

Replacement: fused attention pre-score hook or materialized additive score term before softmax.

Preconditions:

- Encoder self-attention only.
- Static or bounded `T` so relative index table can be cached.
- Head dim equals relative embedding dim.

Failure cases: do not route to vanilla FlashAttention unless it can accept this content-dependent relative score term.

### Rewrite: Conv1d feature extractor to library conv or im2col GEMM

Source pattern: NCL Conv1d stack with strides and no padding.

Replacement: cuDNN/MIOpen/oneDNN Conv1d first; optional im2col+GEMM for fixed production kernels.

Preconditions:

- Preserve NCL semantic layout.
- Dynamic audio length must produce exact floor output lengths.
- GroupNorm after first conv is channel-axis over NCL.

Failure cases: channel-last rewrite without adjusting GroupNorm/LayerNorm axes.

### Rewrite: postnet/vocoder conv regions as no-layout-translation islands

Source pattern: NLC mel tensors transpose to NCL, run Conv1d/BatchNorm/ConvTranspose1d stacks, then transpose/squeeze.

Replacement: keep NCL inside the island, optionally fuse activation+conv+norm locally.

Preconditions:

- All consumers inside island agree on channel-first.
- Boundary transposes are explicit and removable only if adjacent producer/consumer layout is also rewritten.

Failure cases: BatchNorm1d and ConvTranspose1d axis mistakes silently alter output.

### Rewrite: speech autoregressive loop staging

Source pattern: prenet is rerun on full mel prefix every step; decoder uses only last prenet hidden with cache.

Replacement: first integration can reproduce source loop. Later, cache prenet outputs or incrementalize the speech prenet position/dropout path only if parity with consistent dropout is solved.

Preconditions:

- Inference RNG/dropout behavior is defined.
- Speaker embedding projection stays equivalent across all prefix lengths.

Failure cases: assuming normal `GenerationMixin` text decode semantics for TTS/VC.

## 10. Kernel fusion candidates

Highest priority:

- LayerNorm + residual patterns in encoder/decoder blocks: every transformer layer uses two or three LayerNorms.
- Dense GEMM with bias for Q/K/V/O and FFN linears: production hidden/FFN sizes are standard `768/3072`.
- Attention score/softmax/value path for MHA; encoder needs special relative-bias support, decoder can use more standard cached MHA.
- Conv1d speech feature extractor and HiFi-GAN ConvTranspose1d/resblocks if waveform/vocoder parity is in scope.

Medium priority:

- FFN `Linear -> GELU -> Linear` fusion around activation/dropout-elided inference.
- Speech decoder postnet Conv1d + BatchNorm + tanh/dropout-elided inference.
- Speaker embedding normalize + expand + concat + projection.
- Mel output head `Linear(768 -> 160)` plus reshape and stop head `Linear(768 -> 2)`.

Lower priority:

- Processor log-mel extraction on GPU. Keep CPU/data pipeline first.
- Speech decoder prenet dropout-specific fusion; parity complexity is higher than likely performance value.
- HiFi-GAN weight-norm handling at runtime; checkpoints generally store normal weights after load, but loader parity should confirm.

## 11. Runtime staging plan

Stage 1: Parse configs and instantiate shape metadata for all four architectures. Reject unsupported `model_type="hifigan"` only if architecture/config class cannot disambiguate.

Stage 2: Implement tiny/base transformer block parity without prenets: encoder layer and decoder layer with masks, relative bias, and cache.

Stage 3: ASR path: waveform feature extractor conv stack, speech encoder, text decoder, LM head, and token generation. This is the closest to standard seq2seq generation.

Stage 4: TTS mel path without vocoder: text encoder, speech decoder prenet, custom `_generate_speech` loop, mel postnet, stop probabilities, and speaker embeddings.

Stage 5: VC path: reuse speech encoder and speech decoder stages; validate mask downsampling and speaker defaults.

Stage 6: HiFi-GAN vocoder: Conv1d/ConvTranspose1d/resblock path, normalization buffers, waveform shape handling.

Stage 7: Optimized lowering: grouped projections, attention kernels, conv libraries, postnet/vocoder fusion, and runtime loop scheduling.

Initially stubbable: training losses, guided attention, SpecAugment, vocoder for TTS/VC, output cross-attention returns, batched variable-length output collation.

## 12. Parity and validation plan

- Config tests: load official ASR/TTS/VC/HiFi-GAN configs plus tiny random config; verify derived `num_feat_extract_layers` and `inputs_to_logits_ratio`.
- Feature extractor tests: mono waveform padding, mask downsampling lengths, optional normalization, mel target extraction shape `[T_mel, 80]`.
- Single op tests: Conv1d floor length, GroupNorm first conv, LayerNorm channel axis, relative position index clipping, sinusoidal position ids with padding and past length.
- Encoder single-layer parity: random hidden states and masks; include relative position bias.
- Decoder single-layer parity: causal self-attention with and without cache; cross-attention first pass and subsequent cache reuse.
- ASR end-to-end logits parity: random/tiny checkpoint first, then official checkpoint for short waveform. Compare logits and generated token ids.
- TTS loop parity: fixed seed, one speaker embedding, no vocoder; compare spectrogram lengths, stop probabilities, and mel frames.
- VC loop parity: speech input plus speaker embedding; compare mel output before vocoder.
- HiFi-GAN parity: random short mel spectrogram `[B, T, 80]`, compare waveform shape and values.

Suggested tolerances:

- fp32: `rtol=1e-4`, `atol=1e-5` for transformer/conv blocks; log-mel/vocoder may need `atol=1e-4`.
- fp16/bf16: start with `rtol=5e-2`, `atol=5e-3` for full networks; tighten per fused kernel after numeric ordering is fixed.

## 13. Performance probes

- Processor throughput: waveform padding/normalization and log-mel extraction separately.
- Speech feature encoder throughput versus audio length and batch size.
- Encoder-only throughput for text and speech inputs.
- Decoder prefill/decode tokens/sec for ASR text generation.
- TTS/VC mel frames/sec, split into prenet rerun, decoder cached step, postnet, and stop handling.
- HiFi-GAN mel-to-waveform throughput by mel length and batch size.
- KV cache memory: production decoder has 6 layers, self K/V `[B, 12, T_dec, 64]`, cross K/V `[B, 12, T_enc, 64]` per layer.
- Attention backend comparison: eager BMM/softmax/BMM versus fused decoder MHA; encoder separately because of relative bias.
- Conv backend comparison: speech Conv1d stack and HiFi-GAN ConvTranspose1d/resblocks.

## 14. Skip/defer list

- Training losses: `SpeechT5SpectrogramLoss`, guided attention loss, CTC-like or CE training paths beyond logits.
- SpecAugment and LayerDrop.
- Gradient checkpointing, FSDP/DeepSpeed synchronization branches.
- Beam search and advanced text generation processors for ASR first pass; greedy parity is enough initially.
- Output cross-attention collection in TTS/VC generation.
- GPU log-mel extraction.
- Multi-GPU/tensor parallel.
- Quantization.
- Broad layout translation across speech conv/vocoder islands.

## 15. Final implementation checklist

- [ ] Parse `SpeechT5Config` and `SpeechT5HifiGanConfig`, including historical HiFi-GAN `model_type`.
- [ ] Load official ASR/TTS/VC/HiFi-GAN and tiny configs.
- [ ] Implement speech waveform feature extractor Conv1d stack with exact length/mask downsampling.
- [ ] Implement text and speech prenets, including sinusoidal positions and speaker embedding projection.
- [ ] Implement encoder MHA with query-dependent relative position bias.
- [ ] Implement decoder causal self-attention, cross-attention, and `EncoderDecoderCache` shape contract.
- [ ] Implement ASR LM head with tied decoder embedding/LM weight identity.
- [ ] Implement TTS/VC mel postnet and reduction-factor reshape/stop logits.
- [ ] Implement custom `_generate_speech` loop before optimizing it.
- [ ] Implement or compose HiFi-GAN Conv1d/ConvTranspose1d/resblock vocoder.
- [ ] Add no-layout-translation guards around NCL conv/postnet/vocoder regions.
- [ ] Add parity tests for single layer, encoder, decoder cache, ASR logits, TTS mel generation, VC mel generation, and HiFi-GAN waveform.
- [ ] Benchmark processor, encoder, decoder step, mel generation loop, and vocoder separately.
