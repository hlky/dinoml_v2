# Transformers MusicGen Melody Operator and Integration Report

## 1. Source basis

```text
Transformers commit/version:
  Local checkout transformers at b75feb2af64c3e29cbbc1bd859958c5432cc7ed4.
  Remote source basis: https://github.com/huggingface/transformers/tree/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4

Model id:
  Primary target: facebook/musicgen-melody.
  Representative configs: facebook/musicgen-melody,
  facebook/musicgen-melody-large, facebook/musicgen-stereo-melody,
  facebook/musicgen-stereo-melody-large, plus facebook/encodec_32khz.

Config source:
  Official Hugging Face raw files fetched under
  H:/dinoml_v2/agents/plans/transformers/musicgen_melody/.
  Fetched per MusicGen Melody checkpoint: config.json, generation_config.json,
  preprocessor_config.json, tokenizer_config.json.
  Fetched for EnCodec 32 kHz: config.json and preprocessor_config.json.

Source files inspected:
  transformers/src/transformers/models/musicgen_melody/modeling_musicgen_melody.py
  transformers/src/transformers/models/musicgen_melody/configuration_musicgen_melody.py
  transformers/src/transformers/models/musicgen_melody/feature_extraction_musicgen_melody.py
  transformers/src/transformers/models/musicgen_melody/processing_musicgen_melody.py
  transformers/src/transformers/models/musicgen_melody/convert_musicgen_melody_transformers.py
  transformers/src/transformers/models/encodec/modeling_encodec.py
  transformers/src/transformers/models/encodec/configuration_encodec.py
  transformers/src/transformers/models/t5/modeling_t5.py
  transformers/src/transformers/models/t5/configuration_t5.py

Any missing files or assumptions:
  No remote-code files were required for the inspected official checkpoints.
  No gated/401 Hugging Face configs were encountered in the representative
  sweep. This report targets inference-only
  MusicgenMelodyForConditionalGeneration: text/melody-conditioned audio-code
  generation plus EnCodec waveform decode. T5 text encoder internals and
  EnCodec codec internals should compose their own family audits for full
  coverage; this report owns the MusicGen Melody-specific conditioning,
  codebook/generation ABI, decoder/cache behavior, and processor coupling.
```

## 2. High-level architecture

MusicGen Melody is a composite text/audio-conditioned audio generator:

```text
text tokenizer + chroma feature extractor
  -> T5 text encoder and chroma projection
  -> concatenate chroma/text condition prefix with delayed audio-code embeddings
  -> causal decoder prefill/decode
  -> per-codebook audio-code logits/sampling
  -> delay-pattern unmasking
  -> EnCodec decode
  -> waveform
```

Stage decomposition:

- CPU/data pipeline: T5 tokenization and melody chroma extraction from waveform or Demucs stems.
- Text condition: T5 encoder returns `[B, S_text, 768]`, then `enc_to_dec_proj: Linear(768 -> H)` for all inspected Melody checkpoints because decoder `H` is 1536 or 2048.
- Melody/audio condition: processor emits one-hot chroma features `[B, T_chroma, 12]`; model applies `audio_enc_to_dec_proj: Linear(12 -> H)`, repeats or truncates to `chroma_length=235`, then prepends these states before text states.
- Decoder generation: audio-code ids enter as flattened `[B * num_codebooks, T]`, reshape to `[B, Cb, T]`, one embedding table per codebook is summed, sinusoidal positions are added, and a causal self-attention decoder predicts codebook logits.
- Cacheable boundaries: T5 outputs and projected chroma states can be precomputed; the first decoder step caches the full condition prefix as part of self-attention K/V; generated code ids can be decoded by EnCodec independently.

Important distinction from base MusicGen: this source does not construct decoder cross-attention layers. Text and melody conditions are concatenated as a causal prefix to the decoder token embeddings.

## 3. Important config dimensions

Worked example: `facebook/musicgen-melody`.

| Field | Value | Source |
|---|---:|---|
| top-level model_type | `musicgen_melody` | config.json |
| text encoder | T5, `t5-base` style | config.json |
| text hidden / layers / heads | 768 / 12 / 12 | config.json |
| text FFN / activation | 3072 / ReLU | config.json |
| tokenizer | T5Tokenizer, max length 512 | tokenizer_config.json |
| chroma bins / chroma length | 12 / 235 | config.json |
| melody feature sampling rate | 32000 Hz | preprocessor_config.json |
| melody STFT n_fft / hop | 16384 / 4096 | preprocessor_config.json |
| melody chunk length / samples | 30 s / 960000 | preprocessor_config.json |
| EnCodec model | `facebook/encodec_32khz` | config.json |
| EnCodec codebook size / dim | 2048 / 128 | config.json |
| EnCodec upsampling ratios / hop | `[8,5,4,4]` / 640 | config + source property |
| EnCodec frame rate | 50 Hz | inferred `ceil(32000 / 640)` |
| EnCodec quantizers | 4 | inferred `floor(2200 / (50 * 11))` |
| decoder hidden / layers / heads | 1536 / 48 / 24 | config.json |
| decoder head dim | 64 | inferred `hidden_size / heads` |
| decoder FFN / activation | 6144 / GELU | config.json |
| decoder vocab / embed rows | 2048 logits / 2049 embeddings | config + source |
| decoder codebooks | 4 mono, 8 stereo | config sweep |
| max decoder positions | 2048 | config.json |
| generation max length | 1500 delayed steps | generation_config.json |
| default guidance scale | 3.0 | generation_config.json |
| BOS/pad/decoder start token | 2048 | generation_config.json |
| cache | Dynamic self-attention cache | source |

Representative checkpoint sweep:

| Model id | Decoder H | Layers | Heads | FFN | Codebooks | Decoder channels | Generation max length |
|---|---:|---:|---:|---:|---:|---:|---:|
| facebook/musicgen-melody | 1536 | 48 | 24 | 6144 | 4 | 1 | 1500 |
| facebook/musicgen-melody-large | 2048 | 48 | 32 | 8192 | 4 | 1 | 1500 |
| facebook/musicgen-stereo-melody | 1536 | 48 | 24 | 6144 | 8 | 2 | 1500 |
| facebook/musicgen-stereo-melody-large | 2048 | 48 | 32 | 8192 | 8 | 2 | 1500 |

Config/default notes:

- `decoder.audio_channels` is 1 for mono and 2 for stereo; stereo doubles codebooks to 8.
- The nested EnCodec config remains mono (`audio_channels=1`) even for stereo Melody checkpoints; stereo waveform decode runs the mono codec separately on even and odd codebook streams.
- EnCodec `hop_length`, `frame_rate`, `codebook_nbits`, and `num_quantizers` are source properties, not serialized fields.
- Checkpoint `decoder.add_cross_attention=false` is consistent with this source: conditioning is prefix concatenation, not decoder cross-attention.

## 3a. Family variation traps

- Mono vs stereo is an ABI change: mono codebooks are `[0,1,2,3]`; stereo codebooks are interleaved `[L0,R0,L1,R1,L2,R2,L3,R3]`.
- Melody conditioning uses chroma features, not EnCodec audio prompt encoding. EnCodec is required to decode generated codes to waveform.
- The decoder public boundary is flattened `[B * Cb, T]`; internal codebook logic is `[B, Cb, T]`.
- Token id `2048` is both BOS/pad/start for delayed code streams. LM heads output only ids `0..2047`.
- Condition prefix length is `235 + S_text` when both melody and text are present. Decoder logits include prefix positions, so downstream loss/logit consumers must slice or interpret sequence positions carefully.
- CFG is generation-controller behavior: conditional and unconditional condition prefixes are batched together, then `ClassifierFreeGuidanceLogitsProcessor` combines logits.
- Beam search is explicitly rejected; source generation only admits greedy or sampling.
- Chroma features shorter than 235 are repeated along time; longer features are truncated. This repeat/truncate is model graph behavior, not just preprocessing.
- Layout translation must be guarded. Chroma/model tensors are `[B, T, C]`; EnCodec codec tensors are `[B, C, T]`; T5/text tensors are `[B, S, H]`.

## 4. Operator coverage checklist

Tensor/layout ops:

- Reshape/view `[B * Cb, T] <-> [B, Cb, T]`.
- Concatenate condition prefix and decoder embeddings along sequence axis.
- Concatenate attention masks for condition prefix and decoder tokens.
- Repeat chroma hidden states along time, slice/truncate to 235.
- Codebook interleave/deinterleave for stereo final decode.
- Triangular boolean masks for delayed codebook pattern.
- `where`, boolean filtering `output_ids != pad`, nonzero/min for first generation slot.
- Stack per-codebook logits and reshape `[B, Cb, T, 2048] -> [B * Cb, T, 2048]`.
- Index-select/gather for sinusoidal positions.

Neural network primitives:

- T5 encoder primitives: token embedding, T5 RMS-style layer norm, bidirectional self-attention with relative position bias, ReLU FFN. Compose T5 audit.
- `enc_to_dec_proj: Linear(768 -> H)` for text condition.
- `audio_enc_to_dec_proj: Linear(12 -> H)` for chroma condition.
- Decoder embeddings: `Cb` independent `Embedding(2049 -> H)` tables, summed.
- Decoder sinusoidal position add.
- Decoder block: LayerNorm, MHA self-attention, residual add, GELU MLP `Linear(H -> 4H) -> Linear(4H -> H)`, all decoder projections bias-free except LayerNorm affine parameters.
- LM heads: `Cb` independent `Linear(H -> 2048, bias=False)`.
- EnCodec decode: RVQ embedding lookup/sum, Conv1d/ConvTranspose1d, ELU, LSTM, residual blocks, weight norm folded for inference.

Attention/generation/cache ops:

- Causal MHA self-attention only in MusicGen Melody decoder.
- Dynamic KV cache per decoder layer, shape logically `[B, heads, T_cache, 64]` for keys and values.
- Prefix-cache semantics: first generation call includes projected chroma/text prefix plus decoder ids; later calls pass only new ids and set `encoder_hidden_states=None`.
- Delay-pattern mask build/apply and final pad filtering.
- Greedy/sample loop and CFG logits processor.

Preprocessing-coupled ops:

- Chroma extraction: optional resample to 32 kHz, STFT power spectrogram, chroma filter-bank einsum, infinity-norm normalize over chroma axis, argmax, scatter one-hot.
- Optional Demucs stem handling: select stem indices `[3,2]`, sum stems, average channels to mono, resample.
- Processor `batch_decode(audio=...)` strips waveform padding using attention mask.

## 5. Layer/block breakdown

Condition construction:

```text
input_ids [B, S_text] -> T5 encoder -> [B, S_text, 768]
  -> Linear(768 -> H) -> zero masked text positions

input_features [B, T_chroma, 12]
  -> Linear(12 -> H)
  -> repeat along T if T_chroma < 235
  -> slice [:, :235]

encoder_hidden_states = concat([audio_hidden_states, text_hidden_states], dim=1)
```

Decoder embedding:

```text
decoder_input_ids [B * Cb, T]
  -> reshape [B, Cb, T]
  -> sum_i Embedding_i(ids[:, i])  # [B, T, H]
  -> if condition prefix exists: concat([condition, token_embeds], dim=1)
  -> causal mask over combined sequence
  -> add sinusoidal positions with past offset
```

Decoder block, repeated N times:

```text
residual = x
x = LayerNorm(x)
q,k,v = Linear(H -> H, bias=False)
x = causal self_attention(q, k, v, cache)
x = residual + Linear(H -> H, bias=False)(x)

residual = x
x = LayerNorm(x)
x = GELU(Linear(H -> ffn_dim, bias=False)(x))
x = Linear(ffn_dim -> H, bias=False)(x)
x = residual + x
```

Output heads:

```text
hidden [B, T_total, H]
  -> stack Cb heads Linear(H -> 2048)
  -> logits [B * Cb, T_total, 2048]
```

Waveform decode:

```text
generated ids [B, Cb, T] -> add frame axis [1, B, Cb, T]
mono: EnCodec.decode([1, B, 4, T])
stereo: decode even codebooks and odd codebooks separately, concat channels
```

## 6. Attention requirements

MusicGen Melody decoder attention:

- Type: causal self-attention over a combined condition-prefix-plus-token sequence.
- Heads: ordinary MHA. No MQA/GQA field exists.
- Head dim: 64 for all inspected representative checkpoints.
- Projection widths: q/k/v/o all `H -> H`, bias-free.
- Masking: `create_causal_mask` receives concatenated condition and decoder masks. Condition states are therefore visible to generated tokens as prior prefix positions.
- Cache: one self-attention cache per layer. First decode step caches condition prefix K/V and token K/V; subsequent steps call `prepare_inputs_for_generation`, drop consumed prefix ids, set `encoder_hidden_states=None`, and append only new token K/V.
- Cached keys are plain projected keys after sinusoidal absolute positions are added to hidden states; no RoPE/ALiBi.
- Source supports Transformers attention interfaces: eager, SDPA, FlashAttention, and Flex through `ALL_ATTENTION_FUNCTIONS`. Fused backends must preserve `head_dim ** -0.5` scaling and additive mask semantics.

T5 encoder attention:

- Required only for text-conditioned generation.
- Bidirectional encoder self-attention with T5 relative position bias in the first layer, shared position bias through layers.
- No decode KV cache is needed for T5 in the first target.

No MusicGen Melody decoder cross-attention, sliding-window/local attention, packed varlen metadata, RoPE, ALiBi, or sparse attention is required.

## 7. Position encoding and custom math

Decoder sinusoidal embedding:

```python
def musicgen_melody_positions(seq_len, hidden, past_len):
    half = hidden // 2
    inv = exp(arange(half) * -(log(10000) / (half - 1)))
    pos = arange(seq_len) + past_len
    angles = pos[:, None] * inv[None, :]
    emb = concat([cos(angles), sin(angles)], dim=1)
    if hidden % 2:
        emb = concat([emb, zeros(seq_len, 1)], dim=1)
    return emb
```

Delay-pattern apply:

```python
def apply_delay_pattern_mask(input_ids, mask):
    mask = mask[..., : input_ids.shape[-1]]
    return where(mask == -1, input_ids, mask)
```

For mono, codebook `i` is shifted by `i`. For stereo, `channel_codebooks = num_codebooks // 2`, and the same shift is applied to left/right pairs with interleaved rows. The mask fills BOS/EOS delay regions with token `2048` and valid prediction slots with `-1`.

Chroma one-hot extraction:

```python
spec = Spectrogram(n_fft=16384, hop_length=4096, power=2, normalized=True)(wave)
raw = einsum("cf,...ft->...ct", chroma_filters, spec)
norm = normalize(raw, p=inf, dim=-2, eps=1e-6).transpose(1, 2)
idx = norm.argmax(-1, keepdim=True)
one_hot = zeros_like(norm).scatter_(-1, idx, 1)
```

EnCodec RVQ decode:

```python
quantized = 0
for quantizer, indices in enumerate(codes):  # codes [Q, B, T]
    quantized += embedding(indices, codebook[quantizer])  # [B, T, D]
```

## 8. Preprocessing and input packing

Text ABI:

- Tokenizer class: T5Tokenizer.
- Special tokens from tokenizer config: pad `<pad>` id 0, EOS `</s>` id 1, unk id 2, 100 extra ids.
- Model input tensors: `input_ids [B, S_text]`, `attention_mask [B, S_text]`.

Melody/audio feature ABI:

- Processor input can be raw mono/stereo waveform or Demucs output `[B, stems, channels, samples]`.
- Raw stereo waveform is averaged to mono for melody chroma extraction.
- Sampling rate should be passed; non-32 kHz input is resampled with torchaudio.
- Default max raw length is 30 seconds, `960000` samples at 32 kHz.
- Output `input_features [B, T_chroma, 12]`; with default 30 seconds and hop 4096, representative length is around 235 frames, matching `chroma_length`.
- Attention mask for audio features is optional and not consumed by the model graph for chroma conditioning.

Generation packing:

- `_prepare_decoder_input_ids_for_generation` creates `[B * Cb, 1]` filled with decoder start id 2048 unless the caller supplies decoder prompt ids.
- Generated `output_ids` are delay-mask-applied, pad-filtered, reshaped to `[B, Cb, T]`, then wrapped as `[1, B, Cb, T]` for EnCodec.
- Mono decode uses all 4 codebooks. Stereo decode uses `::2` codebooks for left and `1::2` for right, then concatenates resulting waveforms on channel axis.

## 9. Graph rewrite / lowering opportunities

### Rewrite: chroma repeat/truncate as bounded sequence packing

Source pattern:

```text
Linear(12 -> H)(input_features)
repeat along time until length >= 235
slice first 235 frames
```

Replacement:

```text
projected_chroma[b, t, h] = W[input_features[b, t % T_chroma]] + bias
```

Preconditions:

- `T_chroma > 0`.
- Target length is fixed `config.chroma_length`.
- Preserve source behavior for `T_chroma > 235` by truncating, not wrapping.

Failure cases:

- Zero-length chroma features cannot be repeated and should be rejected.

Parity test sketch:

- Test `T_chroma` values 1, 100, 235, and 300 against PyTorch repeat/slice.

### Rewrite: codebook embeddings -> summed gather

Source pattern:

```text
sum(Embedding_i(ids[:, i]) for i in codebooks)
```

Replacement:

```text
Cb independent gathers -> reduction sum over codebook axis
```

Preconditions:

- Input ids normalized to `[B, Cb, T]`.
- Embedding tables are distinct and shaped `[2049, H]`.

Failure cases:

- `decoder_inputs_embeds` bypasses ids and must skip this rewrite.

Parity test sketch:

- Random ids including 2048 for Cb=4 and Cb=8; compare source sum.

### Rewrite: prefix-conditioned decoder to cache-aware causal attention

Source pattern:

```text
concat([condition_prefix, token_embeds], dim=1)
causal self-attention with DynamicCache
```

Replacement:

```text
prefill condition K/V once as prefix cache, then decode token-by-token
```

Preconditions:

- Condition prefix is immutable for the request.
- Position ids for tokens include prefix length during initial prefill.
- Attention mask makes generated tokens attend to all prefix positions.

Failure cases:

- Changing text or chroma condition mid-generation invalidates prefix cache.

Parity test sketch:

- Compare source generate prefill plus two decode steps against explicit prefix-cache path, including cache length.

### Rewrite: per-codebook LM heads -> grouped GEMM

Source pattern:

```text
stack([Linear_i(hidden) for i in codebooks], dim=1)
```

Replacement:

```text
grouped GEMM over Cb independent `[2048, H]` weights
```

Preconditions:

- Bias-free heads.
- Same hidden states feed each head.
- Preserve output orientation before flattening.

Failure cases:

- Do not assume LM head weights are tied to input embeddings.

Parity test sketch:

- Compare logits for mono and stereo configs over prefix and decode positions.

### Rewrite: EnCodec RVQ decode -> fused gather-sum

Source pattern:

```text
embedding lookup per quantizer, sum quantized vectors
```

Replacement:

```text
fused multi-codebook gather-sum producing `[B, codebook_dim, T]`
```

Preconditions:

- Decode-only path with valid ids `0..2047`.
- Normalize axis order from `[1, B, Q, T]` to `[Q, B, T]`.

Failure cases:

- EnCodec encode requires residual nearest-neighbor quantization and is a separate workload.

Parity test sketch:

- Random valid code ids, compare quantized embeddings before the EnCodec decoder stack.

## 10. Kernel fusion candidates

Highest priority:

- Prefix-cache causal attention for the decoder: the 48-layer decoder dominates generation cost.
- Decoder LayerNorm + QKV/GEMM paths: repeated hot path with `H=1536/2048`.
- Grouped per-codebook LM heads: 4 or 8 independent projections over the same hidden state.
- Delay-pattern mask and final codebook filtering: required controller primitive for correct code layout.
- Chroma projection/repeat/truncate fusion: cheap but avoids dynamic repeat materialization and clarifies prefix shape.

Medium priority:

- Codebook embedding gather-sum.
- T5 encoder via existing T5 coverage; cacheable once per request.
- EnCodec RVQ gather-sum and decoder Conv1d/ConvTranspose1d stack for waveform latency.
- CFG batched execution path with efficient conditional/unconditional split.

Lower priority:

- Chroma STFT/filter-bank on GPU. It is processor-coupled and can start as CPU/data-pipeline work.
- EnCodec encode/quantize for audio-code prompts; Melody first target can accept text+chroma conditioning and generated decode.
- Training loss and codebook-wise cross entropy.

## 11. Runtime staging plan

Stage 1: config and ABI admission.

- Parse top-level, nested T5, nested EnCodec, and decoder configs.
- Admit only official mono/stereo Melody shapes: Cb=4 or 8, codebook size 2048, head dim 64, chroma bins 12.
- Accept precomputed `encoder_hidden_states` first to isolate decoder parity.

Stage 2: decoder prefill logits parity.

- Implement codebook embeddings, sinusoidal positions, prefix concatenation, causal mask, decoder blocks, and LM heads.
- Validate logits on short random fixtures with supplied condition prefix.

Stage 3: cached decode parity.

- Implement self-attention KV cache where condition prefix is cached on the first step.
- Add delay-pattern build/apply and final pad filtering.

Stage 4: text/chroma conditioning.

- Compose T5 encoder or accept precomputed T5 states.
- Implement `enc_to_dec_proj`, `audio_enc_to_dec_proj`, text mask zeroing, chroma repeat/truncate, and prefix concatenation order `[audio, text]`.

Stage 5: generation controller.

- Add greedy/sample loop, CFG batch duplication, default guidance scale behavior, and rejection of beam modes.

Stage 6: EnCodec decode.

- Implement RVQ gather-sum and EnCodec decoder stack.
- Add stereo even/odd codebook split and waveform channel concat.

Stage 7: processor and audio conditioning parity.

- Keep chroma extraction in CPU pipeline initially.
- Add processor-level tests for waveform-to-chroma shape and one-hot semantics.

## 12. Parity and validation plan

- Config parsing tests for all four MusicGen Melody checkpoints plus `facebook/encodec_32khz`.
- Chroma extractor parity on fixed sine/noise snippets: STFT length, one-hot chroma, resample behavior, Demucs stem path if targeted.
- Chroma repeat/truncate tests for `T_chroma` below, equal to, and above 235.
- Delay-pattern unit tests for mono Cb=4 and stereo Cb=8, including prompt length >1 and `max_length < 2 * channel_codebooks - 1`.
- Embedding gather-sum parity against source for ids including start/pad 2048.
- Single decoder block parity with random condition prefix and decoder ids.
- Full decoder prefill logits parity for small generated configs and representative H=1536/H=2048 checkpoints where feasible.
- Cached decode parity: first step with prefix, then several one-token steps; verify cache lengths include prefix.
- CFG parity: compare source guided logits for guidance scale 3.0.
- EnCodec RVQ decode parity from random valid code ids before and after waveform decoder.
- Stereo decode parity: verify even/odd codebook split and `[B, 2, samples]` output.

Recommended tolerances:

- fp32 unfused logits: `rtol=1e-4`, `atol=1e-4`.
- fp16/bf16 optimized decoder: start with `rtol=1e-2`, `atol=1e-2`.
- Waveform decode: combine sample-wise tolerance with aggregate checks because ConvTranspose/LSTM differences can accumulate.

## 13. Performance probes

- Processor throughput: chroma extraction and T5 tokenization separately.
- T5 encoder throughput by text length and batch size.
- Decoder prefill latency by prefix length `235 + S_text`, codebooks 4 vs 8, and H=1536 vs H=2048.
- Decode tokens/sec with prefix cache and generation length up to 1500.
- KV cache memory: `layers * 2 * B * heads * (prefix + generated) * 64`.
- Per-codebook LM head time and grouped-GEMM speedup.
- CFG overhead: guidance off vs default 3.0.
- EnCodec decode latency for mono/stereo generated code tensors.
- End-to-end split: preprocessing, text/chroma projection, prefill, decode loop, EnCodec waveform decode.

## 14. Skip/defer list

- Training, labels, and cross-entropy loss.
- Gradient checkpointing, LayerDrop, and dropout behavior.
- Beam search and advanced generation modes rejected by source.
- GPU chroma feature extraction; CPU processor path is acceptable first.
- EnCodec encode / audio-code prompt generation unless continuation prompts are a first milestone.
- General EnCodec chunked overlap-add; inspected 32 kHz config has `chunk_length_s=null`.
- Full T5 and full EnCodec operator ownership inside this report; compose separate audits.
- Quantization, speculative decoding, tensor parallel, continuous batching, and streaming audio postprocessing.

## 15. Final implementation checklist

- [ ] Parse MusicGen Melody composite config and nested configs.
- [ ] Admit mono Cb=4 and stereo Cb=8 codebook ABI.
- [ ] Load decoder codebook embeddings, decoder blocks, and per-codebook LM heads.
- [ ] Implement flattened `[B * Cb, T]` decoder input boundary.
- [ ] Implement chroma projection, repeat/truncate to 235, and `[audio, text]` prefix order.
- [ ] Compose or accept T5 encoder states plus `enc_to_dec_proj`.
- [ ] Implement MusicGen Melody sinusoidal positions with cache offset.
- [ ] Implement prefix-conditioned causal self-attention prefill.
- [ ] Implement self-attention KV cache including condition prefix ownership.
- [ ] Implement delay-pattern mask build/apply and final pad filtering.
- [ ] Implement CFG generation controller path.
- [ ] Implement grouped per-codebook LM head lowering.
- [ ] Implement EnCodec RVQ decode gather-sum.
- [ ] Implement EnCodec decoder waveform path and stereo even/odd split.
- [ ] Add parity tests for chroma packing, delay masks, prefill logits, cached decode, CFG, RVQ decode, and waveform decode.
- [ ] Benchmark processor, text encoder, prefill, decode, LM heads, EnCodec decode, and end-to-end generation.
