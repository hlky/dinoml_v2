# CLVP Transformers Audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: susnato/clvp_dev
Config source: https://huggingface.co/susnato/clvp_dev/resolve/main/config.json
Source files inspected:
- X:/H/transformers/src/transformers/models/clvp/configuration_clvp.py
- X:/H/transformers/src/transformers/models/clvp/modeling_clvp.py
- X:/H/transformers/src/transformers/models/clvp/feature_extraction_clvp.py
- X:/H/transformers/src/transformers/models/clvp/tokenization_clvp.py
- X:/H/transformers/src/transformers/models/clvp/number_normalizer.py
- X:/H/transformers/src/transformers/models/clvp/processing_clvp.py
Any missing files or assumptions: only one native Transformers CLVP checkpoint was found. jbetker/tts-scores-clvp is an open legacy Tortoise CLVP repo with .pth/.json files but no Transformers config.json, so it is not treated as a compatible native checkpoint.
```

Source snapshots are under `agents/plans/transformers/clvp/_sources/`, including the inspected source files, `susnato/clvp_dev` config/preprocessor/tokenizer/generation JSON, and Hub API snapshots for `susnato/clvp_dev` and `jbetker/tts-scores-clvp`.

Primary source URLs:
- [modeling_clvp.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/clvp/modeling_clvp.py)
- [configuration_clvp.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/clvp/configuration_clvp.py)
- [feature_extraction_clvp.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/clvp/feature_extraction_clvp.py)
- [tokenization_clvp.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/clvp/tokenization_clvp.py)
- [number_normalizer.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/clvp/number_normalizer.py)
- [processing_clvp.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/clvp/processing_clvp.py)
- [susnato/clvp_dev](https://huggingface.co/susnato/clvp_dev)
- [jbetker/tts-scores-clvp](https://huggingface.co/jbetker/tts-scores-clvp) legacy non-native source

No 401/403/gated checkpoint gaps appeared in this pass.

## 2. High-level architecture

CLVP in Transformers is a composite Tortoise-style text/speech contrastive and conditional speech-token scoring model:

```text
text string -> CLVP tokenizer/number normalizer -> text token ids -> text encoder -> text projection -> L2 normalize
raw mono audio -> CPU log-mel feature extractor -> conditioning encoder -> causal speech-token decoder -> speech ids
speech ids -> speech encoder -> speech projection -> L2 normalize
normalized text/speech embeddings -> exp(logit_scale) * text @ speech.T -> logits_per_text/logits_per_speech
```

Stage decomposition:

- CPU/data pipeline: tokenizer with English number/abbreviation normalization; mono waveform padding/truncation; log-mel spectrogram extraction and mel-bin normalization.
- Independently cacheable branch: text encoder output/projection can be cached by tokenized text.
- Independently cacheable branch: speech encoder output/projection can be cached when `speech_ids` are already available.
- Conditional generation branch: conditioning encoder consumes `[B, 80, T]` log-mels plus text tokens, then a GPT-like causal decoder emits speech token logits/ids.
- Similarity head: projection, L2 normalization, `logit_scale.exp()`, and two matrix orientations.

First useful DinoML target should be contrastive scoring from `input_ids` plus already-known `speech_ids`. Full `input_features -> generated speech_ids -> contrastive logits` adds conditioning audio operators and autoregressive generation.

## 3. Important config dimensions

`susnato/clvp_dev` config-derived dimensions:

| Component | Field | Value |
|---|---:|---:|
| Top level | `projection_dim` | 768 |
| Top level | `logit_scale_init_value` | 2.6592 |
| Text encoder | `vocab_size` | 256 |
| Text encoder | `hidden_size` | 768 |
| Text encoder | `intermediate_size` | 1536 |
| Text encoder | `projection_dim` | 768 |
| Text encoder | `num_hidden_layers` | 20 |
| Text encoder | `num_attention_heads` | 12 |
| Text encoder | inferred `head_dim` | 64 |
| Text encoder | `use_rotary_embedding` | true |
| Text encoder | `use_attention_bias` | false |
| Text encoder | `summary_type` | mean |
| Speech encoder | `vocab_size` | 8192 |
| Speech encoder | `hidden_size` | 768 |
| Speech encoder | `intermediate_size` | 1536 |
| Speech encoder | `num_hidden_layers` | 20 |
| Speech encoder | `num_attention_heads` | 12 |
| Speech encoder | inferred `head_dim` | 64 |
| Speech encoder | `use_attention_bias` | false |
| Speech encoder | `summary_type` | mean |
| Decoder | `vocab_size` | 8194 |
| Decoder | `hidden_size` | 1024 |
| Decoder | `num_hidden_layers` | 30 |
| Decoder | `num_attention_heads` | 16 |
| Decoder | inferred `head_dim` | 64 |
| Decoder | `n_inner` effective default | 4096 if omitted |
| Decoder | `activation_function` | gelu_new |
| Decoder | `max_position_embeddings` | 608 |
| Decoder | `max_text_tokens` | 404 |
| Decoder | `feature_size` | 80 |
| Conditioning encoder | `num_mel_attn_blocks` | 6 |
| Generation config | `do_sample` | true |
| Generation config | `max_new_tokens` | 256 |

Processor/preprocessor config:

| Field | Value |
|---|---:|
| `sampling_rate` | 22050 |
| `default_audio_length` | 6 seconds |
| `feature_size` | 80 mel bins |
| `n_fft` | 1024 |
| `hop_length` | 256 |
| `chunk_length` | 30 seconds |
| default output shape observed in official test | `[B, 80, 517]` for 6 seconds |
| `return_attention_mask` in file | false |
| `mel_norms` | 80-element vector |

Representative checkpoint sweep:

| Source | Native CLVP? | Architecture/config facts | Notes |
|---|---|---|---|
| `susnato/clvp_dev` | yes | `ClvpModelForConditionalGeneration`, 20-layer text/speech encoders, 30-layer decoder | only native Transformers CLVP checkpoint found |
| source defaults | yes, random init | same text/speech defaults except speech encoder default `vocab_size=256` unless supplied | useful for unit tests, not pretrained parity |
| Transformers tiny tests | yes, random init | encoders use `hidden_size=128`, `projection_dim=16`, 2 layers; decoder uses 1-2 layers in composite tests | operator-shape smoke, not checkpoint parity |
| `jbetker/tts-scores-clvp` | no | legacy files: `clvp.pth`, `clvp_tok.json`, `mel_norms.pth`; no `config.json` | open mirror/original-style asset, route to converter/legacy audit |

## 3a. Family variation traps

- There is one native checkpoint, but source defaults differ from `susnato/clvp_dev`: default `ClvpEncoderConfig.vocab_size` is 256 for both text and speech, while the checkpoint speech encoder uses 8192.
- `ClvpConfig.projection_dim` exists at top level, but encoder projections use each encoder subconfig's `projection_dim`.
- Encoder attention applies partial rotary embedding to query, key, and value, not only q/k.
- Rotary dimension is `max(projection_dim // (num_heads * 2), 32)`, not simply `head_dim`.
- Encoders are bidirectional/noncausal and use RMSNorm plus gated MLP; decoder is causal and uses LayerNorm plus GPT-2 `Conv1D` MLP.
- Decoder `Conv1D` weights use GPT-2 storage semantics, effectively `[in_features, out_features]`, not PyTorch `nn.Linear` row-major weight layout.
- Text tokenizer behavior is model-coupled: `EnglishNormalizer` lowercases, ASCII-filters, expands numbers/currency/ordinals and abbreviations before byte-BPE.
- Conditioning encoder adds BOS/EOS around text inside modeling code, not tokenizer code.
- Audio model input layout is fixed as `[batch, feature_size, frames]`. `Conv1d` and `GroupNorm` are channel-first over mel channels; do not apply NHWC/channel-last layout translation here.
- `attention_mask` is reused across text encoder/conditioning paths but speech encoder attention over generated `speech_ids` normally does not receive a separate generated-speech mask in the composite forward.
- `patch_size` appears in `speech_config` but the inspected source does not read it.
- `decoder_config.n_ctx` appears in the checkpoint but the inspected source uses `max_position_embeddings`.

## 4. Operator coverage checklist

Tensor/layout ops:

- Embedding lookup for text tokens, speech tokens, decoder tokens, decoder positions, conditioning text positions.
- `arange`, `unsqueeze`, `cumsum`, scalar add/sub, broadcast add, padding of token sequences.
- `view`, `reshape`, `transpose`, `contiguous`, `chunk`, `cat`, `gather`, `repeat`.
- Argmax over decoder logits for composite `forward`.
- Mask construction: bidirectional additive mask and causal additive mask.

Neural primitives:

- Linear projections: encoder q/k/v/out `768 -> 768`; encoder gated MLP `768 -> 3072`, split to two `1536` tensors, GELU gate multiply, `1536 -> 768`; encoder projection `768 -> 768` no bias.
- RMSNorm over encoder hidden states with fp32 variance.
- LayerNorm for encoder final norm and decoder blocks.
- GPT-2 `Conv1D` decoder MLP: `1024 -> 4096 -> 1024` with `gelu_new`.
- Causal LM head: `1024 -> 8194` with bias.
- Conditioning audio: `Conv1d(80 -> 1024, kernel_size=1)`, GroupNorm over 1024 channels, six self-attention blocks.

Attention primitives:

- Bidirectional encoder MHA: 12 heads, head dim 64, scaled queries, optional partial rotary, additive mask, softmax over key axis, dropout disabled in inference, output projection.
- Decoder causal MHA: 16 heads, head dim 64, absolute position embeddings, dynamic KV cache through Transformers `Cache`.
- Conditioning mel self-attention: same `ClvpSelfAttention` on `[B, frames, 1024]`, no causal mask in source call.

Preprocessing-coupled ops:

- English text normalization and byte-level BPE with `[SPACE]` special handling.
- CPU log-mel spectrogram: Hann window, STFT/power spectrogram, mel filter bank, `log(clip(x, 1e-5))`, optional per-mel-bin division by `mel_norms`.

Contrastive head:

- Sequence summary pooling, projection, L2 norm, reciprocal/divide, `exp(logit_scale)`, `matmul(text_embeds, speech_embeds.T)`, transpose for `logits_per_speech`.

Generation/cache ops:

- Optional autoregressive speech-token generation with decoder KV cache.
- `fix_speech_decoder_output`: drop first token, replace EOS with code 83, fill after first stop, patch final codes.

## 5. Layer/block breakdown

Text/speech encoder, repeated `num_hidden_layers`:

```text
x = token_embedding(input_ids) or inputs_embeds                       # [B, S, 768]
mask = bidirectional_additive_mask(attention_mask)                    # [B, 1, S, S]
rot = rotary_cache(x) if enabled                                      # [1, S, rotary_dim]

for layer:
  residual = x
  h = RMSNorm(x)
  q = Linear(h) * (head_dim ** -0.5)                                  # [B, 12, S, 64]
  k = Linear(h)
  v = Linear(h)
  q,k,v = partial_rope(q,k,v, rot, position_ids)
  a = softmax(q @ k.transpose(-2, -1) + mask)
  x = residual + Linear(a @ v)
  residual = x
  h = RMSNorm(x)
  h, gate = Linear(h, 768 -> 3072).chunk(2, dim=-1)
  x = residual + Linear(gelu(gate) * h, 1536 -> 768)

x = LayerNorm(x)
pooled = mean(x, dim=1) for checkpoint
embeds = Linear(pooled, 768 -> 768, bias=False)
```

Conditioning encoder:

```text
text_ids, mask = add BOS and EOS in modeling code
text = text_token_embedding(text_ids) + text_position_embedding(mask.cumsum(-1)-1)
mel = Conv1d(input_features [B,80,T])                                  # [B,1024,T]
for 6 blocks:
  residual = mel.transpose(1,2)                                        # [B,T,1024]
  h = GroupNorm(mel).transpose(1,2)
  mel = SelfAttention(h)[0] + residual
  mel = mel.transpose(1,2)
mel = mel[:, :, 0].unsqueeze(1)                                        # [B,1,1024]
conditioning = concat([mel, text], dim=1)                              # [B,1+text_len+2,1024]
```

Decoder layer, repeated `num_hidden_layers`:

```text
x = token_embedding(input_ids) + position_embedding(position_ids)
mask = causal_additive_mask(attention_mask, past_key_values)
for layer:
  residual = x
  h = LayerNorm(x)
  h = causal_MHA(h, mask, past_key_values)
  x = residual + h
  residual = x
  h = LayerNorm(x)
  h = Conv1D(1024 -> 4096) -> gelu_new -> Conv1D(4096 -> 1024)
  x = residual + h
x = final LayerNorm(x)
logits = Linear(x, 1024 -> 8194, bias=True)
```

## 6. Attention requirements

Encoder attention:

- Noncausal self-attention.
- MHA only; no MQA/GQA.
- 12 heads, KV heads = 12, head dim = 64 for `susnato/clvp_dev`.
- Query is scaled before matmul.
- Additive mask must have shape `[B, 1, tgt_len, src_len]`.
- Partial rotary applies to q/k/v prefix dimensions.
- No KV cache.
- Flash/SDPA can replace the matmul-softmax-matmul path if q scaling, mask values, and value rotary are preserved.

Conditioning mel attention:

- Noncausal self-attention over time frames after `Conv1d`.
- 16 heads, head dim = 64 in decoder hidden width 1024.
- No explicit attention mask in source calls.
- Audio layout entering attention is `[B, T, C]` after transpose from `[B, C, T]`.

Decoder attention:

- Causal self-attention.
- 16 heads, KV heads = 16, head dim = 64.
- Uses absolute learned position embeddings, not RoPE.
- KV cache is Transformers `DynamicCache`; per layer key/value are stored after projection/reshape as `[B, heads, seq, head_dim]`.
- No cross-attention despite `ClvpDecoderConfig.add_cross_attention` existing and defaulting false.

## 7. Position encoding and custom math

Encoder rotary:

```python
rotary_dim = max(config.projection_dim // (config.num_attention_heads * 2), 32)
inv_freq = 1.0 / (10000 ** (arange(0, rotary_dim, 2).float() / rotary_dim))
freqs = einsum("i,j->ij", arange(seq_len), inv_freq)
rot = cat([freqs, freqs], dim=-1)  # [seq_len, rotary_dim]
```

Application:

```python
def clvp_partial_rope(q, k, v, rot, position_ids):
    cos = cos(rot)[position_ids].unsqueeze(1)
    sin = sin(rot)[position_ids].unsqueeze(1)
    def rotate_half(x):
        a, b = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return cat([-b, a], dim=-1)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin, v * cos + rotate_half(v) * sin
```

Decoder position encoding is learned absolute position embedding added before causal attention. Conditioning text also uses learned absolute text position embedding over `max_text_tokens`, with ids derived from `attention_mask.cumsum(-1) - 1`.

## 8. Preprocessing and input packing

Text:

- `ClvpTokenizer` is byte-level BPE with regex token splitting and optional leading-space behavior.
- `EnglishNormalizer` is part of tokenizer `_tokenize`: ASCII conversion, lowercasing, number/currency/ordinal expansion, abbreviation expansion, whitespace collapse, quote removal.
- Important special ids in `susnato/clvp_dev`: `[STOP]` id 0, `[UNK]` id 1, `[SPACE]` id 2, `<|endoftext|>` id 255.
- `tokenizer_config.model_max_length` is 402. `generate()` separately enforces `input_ids.shape[-1] <= max_text_tokens - 3`, i.e. 401 for checkpoint decoder config.

Audio:

- Raw audio must be mono. Batched arrays with rank greater than 2 are rejected.
- Sampling rate must match 22050 when supplied.
- Default waveform length for feature extraction is 6 seconds, padded/truncated to 132300 samples unless `max_length` is provided.
- Log-mel features are computed in the feature extractor using NumPy/audio utils, not by the model graph.
- Feature tensor contract into the model is `[batch, 80, frames]`, float32. Official integration test observes `[1, 80, 517]`.
- The preprocessor file has `return_attention_mask=false`, but `__call__` defaults `return_attention_mask=True`; callers should be explicit for parity-sensitive paths.

Input packing:

- Conditioning encoder inserts BOS/EOS around text tokens in modeling code.
- Generated speech ids are postprocessed by `fix_speech_decoder_output` before speech encoder scoring.
- For direct contrastive scoring with known `speech_ids`, the conditioning encoder and decoder can be skipped.

## 9. Graph rewrite / lowering opportunities

### Rewrite: Conv1d kernel-1 mel projection to per-frame Linear

Source pattern:

```text
Conv1d(input_features [B,80,T], out=1024, kernel_size=1)
```

Replacement:

```text
transpose [B,80,T] -> [B,T,80] -> Linear(80 -> 1024) -> transpose if GroupNorm path remains NCT
```

Preconditions:

- `kernel_size == 1`, `stride == 1`, `padding == 0`, `dilation == 1`, `groups == 1`.
- Preserve source `[B, C, T]` ABI at graph boundary.
- Weight transform: `linear.weight = conv.weight[:, :, 0]`, bias unchanged.
- Failure cases: any future grouped/strided conv or caller-provided NHWC/NLC audio tensor.

Parity test sketch: random `[2,80,517]` compare Conv1d output to Linear rewrite after transpose at fp32 tolerance.

### Rewrite: Encoder gated MLP to fused GEGLU-like block

Source pattern:

```text
proj = Linear(768 -> 3072)
h, gate = proj.chunk(2, dim=-1)
out = Linear(h * gelu(gate), 1536 -> 768)
```

Replacement: fused linear-split-GELU-multiply plus output GEMM.

Preconditions: split order is first half data, second half gate; activation is `hidden_act`; no dropout in inference.

### Rewrite: Similarity head as normalized GEMM

Source pattern:

```text
x = x / norm(x, dim=-1, keepdim=True)
y = y / norm(y, dim=-1, keepdim=True)
logits_per_text = x @ y.T * exp(logit_scale)
logits_per_speech = logits_per_text.T
```

Replacement: L2-normalize both embedding matrices and call one GEMM; expose both orientations by view/transpose or materialized copy.

Preconditions: embeddings are rank-2 `[batch, projection_dim]`; logit scale is scalar parameter.

### Layout guard: audio feature tensors

Do not translate `input_features` to NHWC/channel-last. The source graph expects NCL `[B, mel, frames]` for Conv1d and GroupNorm. A local NTC layout can be introduced only inside the guarded Conv1d-to-Linear rewrite and must be converted back or all downstream GroupNorm axes must be rewritten.

## 10. Kernel fusion candidates

Highest priority:

- RMSNorm + encoder attention/MLP residual sequence for text and speech encoders.
- Partial RoPE over q/k/v plus attention backend, preserving value rotary.
- Gated GELU MLP split/multiply fusion.
- Similarity head normalization + GEMM.

Medium priority:

- Conditioning `Conv1d(1x1)` + GroupNorm + time attention block, if full generation path is in scope.
- Decoder LayerNorm + causal attention + GPT-2 Conv1D MLP for speech-token generation.
- Token postprocessing kernels for `fix_speech_decoder_output` only if generation stays on GPU.

Lower priority:

- CPU feature extraction acceleration. It is preprocessing-bound and can initially remain outside DinoML runtime.
- Dropout paths and training losses.

## 11. Runtime staging plan

1. Parse `ClvpConfig`, nested text/speech/decoder configs, processor/tokenizer metadata.
2. Load encoder weights and run text encoder parity on `input_ids`.
3. Load speech encoder weights and run speech encoder parity on supplied `speech_ids`.
4. Implement projection, L2 normalization, logit scale, and bidirectional similarity matrices.
5. Add tokenizer/normalizer parity as CPU pipeline support or require pre-tokenized input first.
6. Add feature extractor parity as CPU preprocessing support, producing `[B,80,T]`.
7. Add conditioning encoder parity for `input_features + input_ids -> conditioning_embeds`.
8. Add decoder prefill/generation parity with KV cache and `fix_speech_decoder_output`.
9. Optimize attention, gated MLP, and similarity head.

Stubbable initially: feature extraction, tokenizer, conditioning encoder, decoder generation, training loss, `pad_to_max_mel_tokens`.

## 12. Parity and validation plan

- Unit test `EnglishNormalizer` for numbers, ordinals, currency, abbreviations, ASCII filtering, and quote removal.
- Tokenizer parity for known strings against `susnato/clvp_dev`.
- Feature extractor parity on the HF dummy LibriSpeech sample: expected shape `[1,80,517]`, compare the first mel row prefix with Transformers test tolerance `1e-4`.
- Random tensor parity for `ClvpRMSNorm`, partial q/k/v RoPE, gated MLP, sequence summary pooling, and similarity head.
- Single encoder layer parity for text encoder and speech encoder in fp32.
- Full text encoder parity on a short tokenized sentence.
- Full speech encoder parity on supplied speech token ids.
- Contrastive logits parity for direct `text_embeds/speech_embeds` and end-to-end `input_ids + speech_ids`.
- Conditioning encoder parity on fixed `input_features` and `input_ids`.
- Decoder single-step and cached generation parity only after generation path is admitted.

Recommended tolerances: fp32 `rtol=1e-4, atol=1e-4` for preprocessing/encoder slices; relaxed fp16 attention/GEMM tolerances only after fp32 parity is stable.

## 13. Performance probes

- Tokenizer/normalizer throughput by text length and batch size.
- Feature extractor throughput by audio seconds and batch size.
- Text encoder throughput by sequence length and batch size.
- Speech encoder throughput by speech-token length and batch size.
- Similarity matrix throughput for N text embeddings by M speech embeddings.
- Conditioning encoder throughput for `[B,80,517]` and longer frame counts.
- Decoder prefill/decode tokens/sec with and without KV cache.
- Attention backend comparison for encoder bidirectional and decoder causal paths.
- Layout rewrite probe: Conv1d kernel-1 versus Linear rewrite for mel projection.

## 14. Skip/defer list

- Training losses and gradient checkpointing.
- Contrastive loss for inference unless validation needs it.
- Full autoregressive generation for first contrastive scoring target.
- Beam search and advanced generation controllers.
- `pad_to_max_mel_tokens` compatibility mode.
- Legacy `jbetker/tts-scores-clvp` conversion path.
- Any NHWC/channel-last rewrite across audio feature tensors.
- Cross-attention; config has a flag but inspected decoder source does not instantiate cross-attention.

## 15. Final implementation checklist

- [ ] Parse nested `ClvpConfig` with text, speech, and decoder subconfigs.
- [ ] Load `susnato/clvp_dev` text encoder, speech encoder, decoder, and conditioning weights.
- [ ] Implement CLVP tokenizer/EnglishNormalizer parity or require pre-tokenized inputs for stage 1.
- [ ] Implement CPU CLVP feature extractor or require precomputed `input_features` for stage 1.
- [ ] Implement encoder RMSNorm.
- [ ] Implement encoder q/k/v/out MHA with q scaling and partial q/k/v RoPE.
- [ ] Implement encoder gated GELU MLP split order.
- [ ] Implement sequence summary pooling modes; require `mean` for checkpoint first.
- [ ] Implement encoder projection and L2-normalized contrastive head.
- [ ] Add direct `input_ids + speech_ids -> logits_per_text/logits_per_speech` parity.
- [ ] Add audio conditioning Conv1d/GroupNorm/self-attention path.
- [ ] Implement decoder GPT-2 `Conv1D` weight-layout loading.
- [ ] Implement decoder causal attention and KV cache.
- [ ] Implement `fix_speech_decoder_output`.
- [ ] Add guarded Conv1d-1x1-to-Linear rewrite.
- [ ] Add no-layout-translation guard for external audio `input_features`.
- [ ] Benchmark preprocessing, encoder-only, similarity GEMM, conditioning encoder, and decoder generation separately.
