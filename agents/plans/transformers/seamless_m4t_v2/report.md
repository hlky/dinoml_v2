# Transformers audit: seamless_m4t_v2

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: facebook/seamless-m4t-v2-large, with tiny/derivative configs for variation checks
Config source: local configuration file plus HF config/preprocessor/tokenizer snapshots under _sources/
Source files inspected:
  transformers/src/transformers/models/seamless_m4t_v2/configuration_seamless_m4t_v2.py
  transformers/src/transformers/models/seamless_m4t_v2/modeling_seamless_m4t_v2.py
  transformers/src/transformers/models/seamless_m4t_v2/convert_fairseq2_to_hf.py
  transformers/src/transformers/models/seamless_m4t/feature_extraction_seamless_m4t.py
  transformers/src/transformers/models/seamless_m4t/processing_seamless_m4t.py
  transformers/src/transformers/models/seamless_m4t/tokenization_seamless_m4t.py
Any missing files or assumptions:
  v2 has no separate processor/tokenizer files; it reuses SeamlessM4T feature extractor, processor, and tokenizer.
  Large generation_config.json files were accessible but not retained because language maps make them about 10 MB.
```

Primary runtime target for DinoML: staged multimodal inference, with text-to-text and speech-to-text text generation first, then text/speech-to-speech via text-to-unit and vocoder. Training losses, LayerDrop behavior, and gradient checkpointing are out of first-scope.

## 2. High-level architecture

SeamlessM4T v2 is a composite UnitY-style system:

```text
text tokens -> text encoder -> autoregressive text decoder -> text token generation
audio waveform -> CPU fbank features -> Conformer speech encoder -> autoregressive text decoder -> text token generation
generated text + decoder hidden states -> non-autoregressive text-to-unit model -> unit ids
unit ids + target language id + speaker id -> duration expansion + HiFi-GAN vocoder -> waveform
```

Stage decomposition:

- CPU/data pipeline: raw 16 kHz waveform, Kaldi-like fbank extraction, per-mel normalization, padding/truncation, stride-2 feature packing to `input_features [B, T_fbank/2, 160]`; BPE tokenization and language-token prefix/suffix insertion.
- Independently cacheable encoders: text encoder hidden states `[B, S_text, 1024]`; speech encoder hidden states `[B, S_audio', 1024]` after Conformer plus optional adapter stride.
- Autoregressive stage: text decoder uses self-attention KV cache plus encoder-decoder cross-attention cache.
- Speech synthesis stage: text decoder is replayed over generated text to produce t2u conditioning embeddings; the t2u path is non-autoregressive and duration-expanding; vocoder is convolutional and duration-expanding.

## 3. Important config dimensions

Source defaults and the official full checkpoint are effectively the same unless noted.

| Field | Official large value | Runtime significance |
|---|---:|---|
| `hidden_size` | 1024 | text encoder/decoder, t2u, vocoder conditioning width |
| `vocab_size` | 256102 | text embedding and LM head |
| `encoder_layers` / `decoder_layers` | 24 / 24 | text encoder-decoder depth |
| `encoder_attention_heads` / `decoder_attention_heads` | 16 / 16 | MHA, `head_dim=64` |
| `encoder_ffn_dim` / `decoder_ffn_dim` | 8192 / 8192 | ReLU FFN hidden width |
| `max_position_embeddings` | 4096 | sinusoidal positions for text encoder/decoder |
| `speech_encoder_layers` | 24 | Conformer depth |
| `speech_encoder_attention_heads` | 16 | speech MHA, `head_dim=64` |
| `speech_encoder_intermediate_size` | 4096 | Conformer FFN hidden width |
| `feature_projection_input_dim` | 160 | processor emits 80 mel bins packed by stride 2 |
| `add_adapter` / adapter stride | true / 8 | speech encoder adapter further subsamples sequence |
| `position_embeddings_type` | `relative_key` | Conformer attention adds clipped relative-key scores |
| `conv_depthwise_kernel_size` | 31 | causal depthwise Conv1d in Conformer |
| `t2u_vocab_size` / `char_vocab_size` | 10082 / 10943 | unit logits and char embedding table |
| `t2u_encoder_layers` / `t2u_decoder_layers` | 6 / 6 | text-to-unit encoder plus duration decoder |
| `unit_hifi_gan_vocab_size` | 10000 | vocoder unit embedding table |
| `unit_embed_dim` / `lang_embed_dim` / `spkr_embed_dim` | 1280 / 256 / 256 | vocoder input concat width 1792 |
| `upsample_rates` | `[5,4,4,2,2]` | waveform length multiplier 320 after unit-duration expansion |
| `sampling_rate` | 16000 | waveform and vocoder output ABI |
| `use_cache` | true | text decoder generation cache enabled |

Representative checkpoint sweep:

| Checkpoint | Scope | Dtype/config note | Operator-significant variation |
|---|---|---|---|
| `facebook/seamless-m4t-v2-large` | official full model | `torch_dtype=float32`, `SeamlessM4Tv2Model` | full text, speech, t2u, vocoder graph |
| `hf-internal-testing/tiny-random-SeamlessM4Tv2Model` | debug | `hidden_size=6`, 2-layer branches | useful for shape/loader tests only |
| `panoyo9829/seamless-m4t-v2-large-fp16` | derivative text head | `SeamlessM4Tv2ForTextToText`, `torch_dtype=float16` | same native graph dimensions, fp16 load/admission case |
| `jaman21/seamless-m4t-v2-t2tt` | derivative pruned config | omits speech/t2u fields in snapshot | should route to text-only target or use defaults carefully |
| `jaman21/seamless-m4t-v2-t2st` | derivative pruned config | keeps t2u fields, omits decoder/speech fields in snapshot | admission must confirm weights/architecture class before full speech path |
| `WueNLP/seamless-m4t-v2-large-speech-encoder` | custom code derivative | `model_type=seamlessm4t-v2-large-speech_encoder`, `auto_map` present | out of native-source scope; separate remote-code audit |

## 3a. Family variation traps

- The native source is multi-head attention only; no GQA/MQA fields exist. `hidden_size` must be divisible by attention heads.
- The speech path uses Conformer attention with relative-key embeddings and causal depthwise Conv1d, not the same BART-style attention as text.
- The speech adapter changes sequence length with Conv1d `kernel_size=stride=8`; attention masks must be recomputed, not reused.
- T2U is not autoregressive generation in the source path used after text generation. It uses character lengths, hard upsampling, duration prediction, then argmax or multinomial over unit logits.
- The vocoder is not a Transformer. It has dynamic `repeat_interleave`, padded batched duration expansion, ConvTranspose1d upsampling, residual dilated Conv1d blocks, and final `tanh`.
- Generation-language control is ABI-critical and lives in `generation_config`, not only tokenizer config: `text_decoder_lang_to_code_id`, `t2u_lang_code_to_id`, `vocoder_lang_code_to_id`, `id_to_text`, and `char_to_id`.
- The tokenizer uses source layout `[src_lang] tokens [eos]` and target layout `[eos, tgt_lang] tokens [eos]`; generated text is sliced as `sequences[:, 2:-1]` before t2u character conversion.
- Processor audio shape is already `[B, T, 160]`; do not introduce NCHW/NHWC assumptions. Conv1d regions use `[B, C, T]` internally after explicit transposes.
- Some derivative configs omit fields present in `SeamlessM4Tv2Config` defaults. DinoML should record effective defaults and reject weight/config mismatches rather than infer missing branch weights.

## 4. Operator coverage checklist

Tensor/layout ops:
- `view`, `reshape`, `transpose`, `permute`, `contiguous`, `cat`, `squeeze`, `unsqueeze`, `repeat`, `repeat_interleave`, `pad_sequence`, boolean masks, `masked_fill`, `where`, `clamp`, `round`, `expm1`, `arange`, reductions for sequence lengths.
- Dynamic shape and ragged-ish behavior for duration expansion in t2u and vocoder.

Neural primitives:
- Embedding with optional scale by `sqrt(hidden_size)`.
- Linear/GEMM with bias: text Q/K/V/O `1024 -> 1024`, text FFN `1024 -> 8192 -> 1024`, speech FFN `1024 -> 4096 -> 1024`, t2u LM head `1024 -> 10082`.
- LayerNorm over last dimension, dropout as no-op for inference, ReLU, Swish/SiLU, GLU over Conv1d channel split, leaky ReLU, tanh.
- Conv1d: pointwise `1024 -> 2048`, depthwise grouped `1024 groups, k=31`, adapter `1024 -> 2048, k=s=8`, variance predictor Conv1d with `padding="same"`.
- ConvTranspose1d for vocoder upsampling.

Attention primitives:
- Dense noncausal self-attention for text encoder.
- Dense causal self-attention with KV cache for text decoder.
- Dense encoder-decoder cross-attention with separately cacheable cross K/V.
- Conformer self-attention with clipped relative-key score bias.
- Bidirectional and causal mask construction with additive `finfo(dtype).min` style masks via Transformers mask utilities.

Position/custom math:
- Sinusoidal positional embeddings for text and t2u.
- Learned relative-distance embedding for Conformer attention, clipped to `[-64, 8]` by default.

Generation/cache ops:
- `EncoderDecoderCache(DynamicCache, DynamicCache)` for decoder self-attention and cross-attention.
- Cross-attention cache `is_updated[layer_idx]` means encoder K/V are computed once then reused.
- Generation controller must inject target-language decoder prefix and split `text_` versus `speech_` kwargs.

Preprocessing-coupled ops:
- CPU fbank extraction: 400-sample Povey window, hop 160, FFT 512, Kaldi mel filters, log mel, preemphasis 0.97, per-mel normalization.
- Stride-2 mel packing from `[B, frames, 80]` to `[B, frames//2, 160]`, attention mask selected at `indices % 2 == 1`.

Optional codec/vocoder ops:
- Required for speech output: t2u character packing, hard upsample, duration predictor, unit offset correction, language/speaker embeddings, duration repeat, HiFi-GAN.
- Deferred for text-only targets.

## 5. Layer/block breakdown

Text encoder layer, repeated `encoder_layers`:

```text
x = token_embedding(input_ids) * sqrt(1024) + sinusoidal_position
x = dropout(x)
for layer:
  residual = x
  x = self_attention(q=k=v=x, bidirectional_mask)
  x = LayerNorm(residual + dropout(x))
  residual = x
  x = Linear(1024 -> 8192) -> ReLU -> dropout -> Linear(8192 -> 1024)
  x = LayerNorm(residual + dropout(x))
x = final LayerNorm(x)
```

Text decoder layer, repeated `decoder_layers`:

```text
x = decoder_embedding + sinusoidal_position(cache_offset)
x = causal_self_attention(x, self_kv_cache)
x = LayerNorm(residual + x)
x = cross_attention(query=x, key/value=encoder_hidden_states, cross_kv_cache)
x = LayerNorm(residual + x)
x = FFN(1024 -> 8192 -> 1024, ReLU)
x = LayerNorm(residual + x)
logits = tied_or_projected LM head(x)
```

Speech Conformer layer, repeated `speech_encoder_layers`:

```text
x = x + 0.5 * FFN(LayerNorm(x), 1024 -> 4096 -> 1024, Swish)
x = x + MHA_relative_key(LayerNorm(x))
x = x + ConvModule(LayerNorm, pointwise Conv1d, GLU, left-pad depthwise Conv1d k=31, LayerNorm, Swish, pointwise Conv1d)
x = LayerNorm(x + 0.5 * FFN(LayerNorm(x)))
```

Speech encoder wrapper:

```text
input_features [B,T,160] -> LayerNorm(160) -> Linear(160 -> 1024)
-> 24 Conformer layers -> intermediate FFN ReLU -> inner LayerNorm
-> optional adapter layers: Conv1d k=s=8 + Conformer attention without relative positions + FFN
```

T2U path:

```text
text decoder hidden states -> t2u encoder(6 layers, no embeddings)
char_input_ids + char_count_per_id
char_hidden = hard_upsample(encoder_hidden, char_count_per_id)
char_hidden += char_embedding + char sinusoidal positions
dur = clamp(round(expm1(duration_predictor(char_hidden))), min=1)
unit_hidden = hard_upsample(char_hidden, dur)
unit_hidden += sinusoidal positions
6 t2u decoder layers with self-attention + Conv1d + FFN
unit_logits = Linear(1024 -> t2u_vocab_size)
```

Vocoder:

```text
unit ids -> Embedding(10000,1280)
speaker id -> Embedding(200,256)
lang id -> Embedding(36,256)
duration predictor over unit embeddings -> repeat_interleave over time
concat [lang, unit, speaker] -> Conv1d(1792 -> 512, k=7)
5 ConvTranspose1d upsamplers with rates [5,4,4,2,2]
each stage averages 3 residual dilated Conv1d blocks
Conv1d(final_channels -> 1, k=7) -> tanh -> waveform [B,T_wave]
```

## 6. Attention requirements

Text encoder:
- Noncausal dense self-attention, MHA, 16 heads, `head_dim=64`, additive bidirectional padding mask.
- No cache.

Text decoder:
- Causal dense self-attention plus encoder-decoder cross-attention.
- MHA, 16 heads, `head_dim=64`; no GQA/MQA.
- Query is scaled before matmul. Softmax is computed in fp32 then cast back to attention-score dtype.
- Cache stores keys/values as `[B, heads, S, head_dim]`; self-attention grows with decode length, cross-attention cache stores encoder K/V once per layer.
- Source uses `EncoderDecoderCache` wrapping two `DynamicCache` objects; cross-attention cache has an `is_updated` per-layer flag.

Speech Conformer:
- Noncausal dense self-attention over speech frames after feature projection.
- Relative-key embedding adds an extra score term from `einsum("bhld,lrd->bhlr", query, positional_embedding)`.
- The source default clips relative positions to left 64 and right 8. This asymmetry is operator-significant.

T2U:
- T2U encoder uses bidirectional dense self-attention over text-decoder hidden states.
- T2U decoder uses bidirectional dense self-attention after duration expansion; no autoregressive KV cache in the native post-text generation path.

FlashAttention/SDPA compatibility:
- Text attention can map to ordinary prefill/decode attention if fused kernels preserve fp32 softmax and mask addition order.
- Conformer relative-key attention needs a custom bias path or precomputed dense score bias; it is not plain SDPA unless the relative score term is materialized.
- T2U dynamic duration expansion creates variable lengths that should probably be bucketed before fused attention admission.

## 7. Position encoding and custom math

Sinusoidal positions are table-like and can be precomputed up to max positions, with runtime offset for cached decoder positions.

Conformer relative-key score math:

```python
def conformer_relative_key_scores(query, distance_embedding, left, right):
    # query: [B, H, Q, D]
    q = torch.arange(query.shape[2], device=query.device).view(-1, 1)
    k = torch.arange(query.shape[2], device=query.device).view(1, -1)
    distance = torch.clamp(k - q, -left, right)
    rel = distance_embedding(distance + left).to(query.dtype)  # [Q, K, D]
    return torch.einsum("bhld,lrd->bhlr", query, rel) / math.sqrt(query.shape[-1])
```

Duration prediction expansion:

```python
dur = torch.clamp(torch.round(torch.expm1(log_dur_pred)).long(), min=1)
dur = dur.masked_fill(~padding_mask.bool(), 0)
expanded = hard_upsample(hidden, dur)
```

Precomputable: sinusoidal tables, Conformer relative-distance embedding weights, mel filter bank/window in CPU preprocessing. Dynamic: masks, duration expansion, t2u char counts, vocoder waveform lengths.

## 8. Preprocessing and input packing

Audio processor contract:
- Input raw audio at 16 kHz; mono is expected, stereo uses the first channel.
- Fbank uses frame length 400, hop 160, FFT 512, no center padding, preemphasis 0.97, Kaldi mel scale, 80 bins, log mel.
- Optional per-mel zero-mean/unit-variance normalization is default.
- Padding defaults to longest; `pad_to_multiple_of=2` is the feature-extractor default.
- Features are reshaped from `[B, F, 80]` to `[B, F//2, 160]`; if `F` is odd, the tail is dropped.
- Attention mask is similarly reduced by selecting positions where `frame_index % 2 == 1`.

Text tokenizer/language control:
- Source text sequence: `[src_lang_code] tokens [eos]`.
- Target text sequence: `[eos, tgt_lang_code] tokens [eos]`.
- `generate(tgt_lang=...)` overrides supplied decoder input IDs with the target language id.
- End-to-end speech output requires generation config language maps. Missing maps should be a hard admission failure for speech generation.

T2U packing:
- After text generation, `sequences[:, 2:-1]` removes EOS/lang prefix and final EOS.
- Intermediate EOS tokens are replaced by pad id before subword-to-character expansion.
- `id_to_text` and `char_to_id` from generation config convert generated tokens into `char_input_ids` and `char_count_per_id`.

## 9. Graph rewrite / lowering opportunities

### Rewrite: packed QKV for text attention

Source pattern: three independent Linear layers `q_proj`, `k_proj`, `v_proj` with identical input and output width.

Replacement:

```text
Linear(x, concat([Wq, Wk, Wv]), concat([bq, bk, bv])) -> Split(Q,K,V)
```

Preconditions: same source tensor, same dtype/device, no weight sharing mutations, all three biases present or absent consistently. Split order is Q, K, V for the fused output even though module attributes are defined K/V/Q.

Failure cases: cross-attention may reuse cached K/V and only compute Q after cache warmup; decoder lowering should not recompute encoder K/V on cached decode steps.

Parity test: compare one encoder layer, decoder self-attention prefill, and decoder cross-attention first-token/next-token paths.

### Rewrite: Conv1d pointwise to GEMM

Source pattern: `Conv1d(Cin, Cout, kernel_size=1, stride=1, padding=0)` around Conformer pointwise and vocoder/projection-style regions.

Replacement:

```text
transpose [B,T,C] or [B,C,T] as needed -> Linear(Cin -> Cout) -> restore layout
```

Preconditions: kernel 1, groups 1, dilation 1. Respect source layout: speech Conformer uses `[B,C,T]` during Conv1d and `[B,T,C]` around LayerNorm.

Failure cases: depthwise Conv1d, adapter Conv1d `k=s=8`, ConvTranspose1d, and residual Hifi-GAN blocks are not pointwise.

### Rewrite: speech adapter Conv1d subsample to window GEMM

Source pattern: adapter projection `Conv1d(1024 -> 2048, kernel_size=8, stride=8)`.

Replacement:

```text
WindowFlatten non-overlap [B,T,1024] with window=8 -> GEMM(8192 -> 2048) -> reshape [B,T//8,2048]
```

Preconditions: exact `kernel_size == stride == 8`, padding 0, dilation 1, groups 1, input sequence divisible or source truncation/padding behavior reproduced.

Failure cases: dynamic masks must be subsampled with the source `_compute_sub_sample_lengths_from_attention_mask` rule; no implicit layout translation across Conformer boundaries.

### Rewrite: Conformer relative-key score materialization

Source pattern: attention scores plus `einsum(query, relative_position_embedding) / sqrt(D)`.

Replacement: custom fused attention bias callback or pre-matmul dense bias tensor `[1,H,Q,K]` generated from query and distance embeddings.

Preconditions: `position_embeddings_type == "relative_key"`, fixed left/right clip bounds, dense Q/K attention.

Failure cases: cannot use ordinary FlashAttention without preserving query-dependent relative score.

### Rewrite: inference dropout and LayerDrop removal

Dropout modules and LayerDrop random skips are training-only. For inference, remove dropout and never skip layers.

### Layout guidance

Protect audio feature tensors and Conv1d islands with a no-layout-translation guard unless a local pass rewrites every `[B,T,C] <-> [B,C,T]` transpose, Conv1d axis, LayerNorm axis, and downstream consumer together. There is no image/video NHWC work in this family.

## 10. Kernel fusion candidates

Highest priority:
- Text decoder attention with KV cache and cross-attention cache. This gates generation throughput.
- LayerNorm + residual patterns in encoder/decoder/Conformer blocks.
- FFN GEMMs `1024 -> 8192 -> 1024` and Conformer FFNs `1024 -> 4096 -> 1024`, with activation fusion.
- Conformer ConvModule: LayerNorm, pointwise Conv1d/GLU, left-pad depthwise Conv1d, LayerNorm, Swish, pointwise Conv1d.

Medium priority:
- Conformer relative-key attention score path.
- T2U duration predictor and hard upsample, especially bucketed/padded implementation for GPU.
- Vocoder ConvTranspose1d plus residual Conv1d blocks.
- Last-token-only logits for text generation if only next-token logits are needed.

Lower priority:
- Temperature sampling for t2u units.
- Full batched vocoder dynamic repeat optimization; first integration can be batch-1 or padded bucketed.
- Language-map/tokenizer-side helpers inside runtime; these can stay controller/data-pipeline work initially.

## 11. Runtime staging plan

Stage 1: parse native configs and load text encoder-decoder weights for `SeamlessM4Tv2ForTextToText`; implement token embedding, sinusoidal positions, dense MHA, FFN, LayerNorm, tied LM head.

Stage 2: text-to-text prefill/decode parity with `EncoderDecoderCache`, target-language decoder prefix, and greedy generation.

Stage 3: speech encoder parity from precomputed `input_features`; keep fbank extraction in CPU pipeline. Add Conformer relative-key attention, ConvModule, and adapter mask recomputation.

Stage 4: speech-to-text generation by composing speech encoder with text decoder and cache.

Stage 5: t2u model from generated sequences and text decoder hidden states. Initially greedy unit argmax only, with strict `generation_config` maps.

Stage 6: vocoder batch-1 or bucketed padded path; implement duration expansion, language/speaker embeddings, HiFi-GAN ConvTranspose1d/resblocks, waveform length reporting.

Stage 7: optimize fusions, cache reuse, dynamic batching, and speech synthesis batching.

Can be stubbed initially: training losses, beam search, t2u sampling, full batched vocoder repeat efficiency, remote-code speech-encoder derivatives.

## 12. Parity and validation plan

- Processor parity: raw waveform to `input_features` and `attention_mask`, including odd frame truncation and stride-2 mask selection.
- Single-op parity: Conformer relative-key attention scores, causal mask construction with cache offsets, `_compute_new_attention_mask`, duration predictor `expm1/round/clamp`, hard upsample.
- Block parity: one text encoder layer, one text decoder layer with and without cross-attention cache, one Conformer layer, one adapter layer, one t2u decoder layer, one Hifi-GAN residual block.
- Stage parity: text encoder output, speech encoder output from saved `input_features`, text prefill logits, decode next-token logits, t2u unit logits, vocoder waveform for short unit sequences.
- End-to-end parity: T2TT greedy token IDs, S2TT greedy token IDs, T2ST unit IDs and waveform lengths, S2ST output tuple.
- Suggested tolerances: fp32 `rtol=1e-4, atol=1e-4` for blocks excluding vocoder; fp16 `rtol=1e-2, atol=1e-2`; vocoder waveform should use looser signal-level checks plus length parity because ConvTranspose accumulates differences quickly.

## 13. Performance probes

- CPU preprocessing throughput: waveform seconds/sec and fbank frames/sec.
- Speech encoder throughput by input feature length before and after adapter subsampling.
- Text encoder prefill throughput by source length.
- Text decoder decode tokens/sec with self cache and cross cache enabled.
- Cross-attention cache memory and update cost by encoder length.
- T2U duration expansion: generated text length, char length, predicted unit length, padded waste.
- Vocoder throughput: unit length to waveform samples/sec, batch-1 versus padded batch.
- Attention backend comparison: dense eager, DinoML fused, and relative-key Conformer custom path.
- End-to-end S2TT, T2TT, T2ST, and S2ST latency split by processor/encoder/text decode/t2u/vocoder.

## 14. Skip/defer list

- Training losses, gradient checkpointing, LayerDrop stochastic behavior, dropout.
- Beam search and advanced generation processors for first parity; greedy text generation is enough to validate cache ABI.
- T2U multinomial sampling with temperature.
- Remote-code derivative `WueNLP/seamless-m4t-v2-large-speech-encoder`.
- General ragged GPU runtime; use bucketed/padded duration expansion first.
- Full vocoder batching efficiency and streaming vocoder.
- Quantized/8-bit derivative loading unless represented by DinoML constant/provider contracts.
- Multi-GPU/tensor parallel.

## 15. Final implementation checklist

- [ ] Parse `SeamlessM4Tv2Config` and record effective defaults for missing derivative fields.
- [ ] Load text embeddings, tied LM head, encoder, and decoder weights.
- [ ] Implement sinusoidal positional embedding with decoder cache offset.
- [ ] Implement text encoder dense bidirectional MHA and FFN blocks.
- [ ] Implement text decoder causal self-attention, cross-attention, and `EncoderDecoderCache` ABI.
- [ ] Add target-language generation prefix handling from generation config.
- [ ] Add T2TT greedy parity tests.
- [ ] Implement CPU/data-pipeline fbank or define an external preprocessing boundary.
- [ ] Implement speech feature projection, Conformer relative-key attention, ConvModule, and adapter subsampling.
- [ ] Add speech encoder and S2TT parity tests from saved processor outputs.
- [ ] Implement t2u character packing from `id_to_text` and `char_to_id`.
- [ ] Implement t2u hard upsample, duration predictor, t2u decoder, and unit LM head.
- [ ] Implement unit id cleanup and `vocoder_offset` handling.
- [ ] Implement vocoder duration expansion, language/speaker embeddings, HiFi-GAN ConvTranspose1d/resblocks, and waveform length reporting.
- [ ] Add T2ST/S2ST staged parity tests.
- [ ] Add rewrite tests for QKV packing, adapter Conv1d-to-window-GEMM, and pointwise Conv1d-to-GEMM.
- [ ] Benchmark processor, encoder, decode, t2u, and vocoder stages separately.
