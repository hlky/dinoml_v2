# Transformers Audit: `speech_encoder_decoder`

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: composite family; representative ids listed below
Config source: local source plus Hugging Face config/preprocessor/tokenizer/generation JSON fetched 2026-05-13
Source files inspected:
- transformers/src/transformers/models/speech_encoder_decoder/configuration_speech_encoder_decoder.py
- transformers/src/transformers/models/speech_encoder_decoder/modeling_speech_encoder_decoder.py
- transformers/docs/source/en/model_doc/speech-encoder-decoder.md
- transformers/tests/models/speech_encoder_decoder/test_modeling_speech_encoder_decoder.py
- selected delegated owner source for ABI checks: wav2vec2 configuration/modeling, speech_to_text feature extractor
Any missing files or assumptions:
- There is no processing_speech_encoder_decoder.py in this checkout.
- The wrapper delegates neural bodies to AutoModel encoder and AutoModelForCausalLM decoder classes.
- Operator coverage below is wrapper-owned plus representative delegated-body requirements; exact neural ops must be owned by separate encoder/decoder family audits.
```

Representative configs saved under `_sources/`:

- `hf-internal-testing/tiny-random-speech-encoder-decoder`
- `facebook/wav2vec2-xls-r-300m-en-to-15`
- `facebook/wav2vec2-xls-r-1b-en-to-15`
- `facebook/wav2vec2-xls-r-2b-en-to-15`
- `facebook/s2t-wav2vec2-large-en-de`
- `patrickvonplaten/wav2vec2-2-bart-base`
- `KBLab/asr-voxrex-bart-base`

## 2. High-level architecture

`speech_encoder_decoder` is an audio encoder plus autoregressive text decoder wrapper. The wrapper owns composition, ABI, optional projection, encoder-mask adaptation, labels-to-decoder-input shifting, and Seq2Seq output packing. It does not own Wav2Vec2, HuBERT, BART, mBART, BERT LM head, or Speech2Text2 operators.

```text
raw waveform or precomputed audio features
  -> encoder-owned feature extraction/model body
  -> optional wrapper Linear encoder_hidden -> decoder_hidden
  -> decoder cross-attention prefill
  -> decoder autoregressive cache decode
  -> logits/sampling/tokenizer decode
```

Stage decomposition:

- CPU/data pipeline: audio decode/resampling, Wav2Vec2FeatureExtractor padding/normalization or feature extractor selected by encoder, decoder tokenizer/language prompt metadata.
- Audio encoder: independently cacheable per input audio; output is `encoder_last_hidden_state [B, Senc, Denc_out]`.
- Wrapper adaptation: optional `enc_to_dec_proj`; encoder attention mask downsampled by `encoder._get_feature_vector_attention_mask`.
- Decoder prefill/decode: delegated to decoder family, with self-attention KV cache and cross-attention over encoder hidden states.
- Generation controller: `decoder_start_token_id`, forced BOS/EOS, beam/search rules, suppressions if supplied by generation/tokenizer configs.

## 3. Important config dimensions

Wrapper-significant fields:

| Field | Meaning | Admission impact |
| --- | --- | --- |
| `encoder.model_type` | delegated audio encoder family | Must map to audited allowlisted encoder |
| `decoder.model_type` | delegated text decoder family | Must map to audited causal LM decoder with cross-attention |
| `encoder.hidden_size` | source config hidden width | Used by wrapper projection constructor and cross-attn hidden-size guard |
| `encoder.output_hidden_size` | actual Wav2Vec2 adapter output width when present | Used by wrapper projection decision |
| `decoder.hidden_size` / decoder d_model | decoder token width | Must match encoder output or projection output |
| `decoder.cross_attention_hidden_size` | decoder-owned cross-attn input width | If set, wrapper requires equality with `encoder.hidden_size` and skips wrapper projection |
| `tie_word_embeddings` | forced false by wrapper | Do not tie wrapper input/output embeddings; decoder may still own internal aliases |
| `decoder_start_token_id`, `forced_bos_token_id`, `forced_eos_token_id` | generation ABI | Controller metadata, not graph ops |
| `pad_token_id`, `attention_mask` | input and label padding | Required for mask downsampling and label shift |

Representative checkpoint sweep:

| Model | Encoder | Encoder dims | Adapter | Decoder | Decoder dims | Vocab | Generation/preprocessor |
| --- | --- | ---: | --- | --- | ---: | ---: | --- |
| `hf-internal-testing/tiny-random-speech-encoder-decoder` | Wav2Vec2 | 4 layers, H=16, heads=2 | no, output=16 | BERT LM | 5 layers, H=32, heads=4 | 1124 | 16 kHz, Wav2Vec2FeatureExtractor |
| `patrickvonplaten/wav2vec2-2-bart-base` | Wav2Vec2 | 12 layers, H=768, heads=12 | no, output=768 | BART | 6/6 layers, d_model=768, heads=12 | 50265 | max_length 200, beams 5 |
| `KBLab/asr-voxrex-bart-base` | Wav2Vec2 | 24 layers, H=1024, heads=16 | yes, output=1024 | BART | 6/6 layers, d_model=1024, heads=16 | 50185 | max_length 40 |
| `facebook/s2t-wav2vec2-large-en-de` | Wav2Vec2 | 24 layers, H=1024, heads=16 | not set in config snapshot | Speech2Text2 | encoder_layers=12, decoder_layers=7, d_model=1024, heads=4 | 10224 | max_length 200, beams 5 |
| `facebook/wav2vec2-xls-r-300m-en-to-15` | Wav2Vec2 | 24 layers, H=1024, heads=16 | yes, output=1024 | mBART | 12/12 layers, d_model=1024, heads=16 | 250054 | decoder_start/forced BOS language ids |
| `facebook/wav2vec2-xls-r-1b-en-to-15` | Wav2Vec2 | 48 layers, H=1280, heads=16 | yes, output=1024 | mBART | 12/12 layers, d_model=1024, heads=16 | 250054 | same XLS-R translation ABI |
| `facebook/wav2vec2-xls-r-2b-en-to-15` | Wav2Vec2 | 48 layers, H=1920, heads=16 | yes, output=1024 | mBART | 12/12 layers, d_model=1024, heads=16 | 250054 | same XLS-R translation ABI |

All fetched preprocessors use `Wav2Vec2FeatureExtractor`, `sampling_rate=16000`, `feature_size=1`, `padding_value=0.0`, and `return_attention_mask=True`. `do_normalize` varies by checkpoint and is CPU/data-pipeline behavior before the model graph.

## 3a. Family variation traps

- This is a wrapper family. Do not infer a fixed operator surface from `SpeechEncoderDecoderModel`; admit only exact encoder/decoder combinations whose delegated families have separate audits.
- `input_values` and `input_features` are mutually exclusive. The wrapper passes one positional `inputs` tensor into the encoder.
- Wrapper projection is optional. It is inserted when `encoder_output_dim != decoder.hidden_size` and decoder `cross_attention_hidden_size` is absent.
- Projection guard trap: source decides with `encoder.output_hidden_size` but constructs `nn.Linear(encoder.hidden_size, decoder.hidden_size)`. DinoML should reject adapter combinations where the actual encoder output width differs from the projection input width.
- If `decoder.cross_attention_hidden_size` is present, wrapper requires it to equal `encoder.hidden_size` and skips `enc_to_dec_proj`. For adapter encoders, this may not equal actual emitted width; require a delegated-family parity check before admitting.
- Encoder mask adaptation is encoder-owned through `_get_feature_vector_attention_mask`; Wav2Vec2-style downsampling depends on conv stride/kernel and optional adapters.
- Decoder `model_type` changes attention/cache/layout contracts: BERT LM head with cross-attention, BART/mBART decoder, and Speech2Text2 decoder are not interchangeable.
- mBART checkpoints carry language-control ids (`decoder_start_token_id`, `forced_bos_token_id`, `forced_eos_token_id`) that the generation controller must honor.
- The wrapper sets `tie_word_embeddings=False`; do not tie encoder/decoder embeddings at wrapper level.
- `_supports_flash_attn` and `_supports_sdpa` on the wrapper are only dispatch affordances. Each submodel must support the requested attention backend.

## 4. Operator coverage checklist

Wrapper-owned required ops:

- Tensor/layout ops: rank checks, `BaseModelOutput` tuple unpack/pack, optional `reshape` for loss only, `masked_fill` in label shift if training/loss is supported.
- Neural primitives: optional `Linear(Denc_out -> Ddec)` with bias. For XLS-R 1B/2B to mBART, no wrapper projection because Wav2Vec2 adapter output is 1024 and decoder d_model is 1024.
- Mask ops: input `attention_mask [B, Traw]` to encoder feature mask `[B, Senc]` by delegated encoder method.
- Generation/cache ops: pass-through `past_key_values` to decoder; pass-through `encoder_outputs` to skip encoder; return decoder `past_key_values`.
- Preprocessing-coupled ops: accept raw waveform `[B, T]` for Wav2Vec2-like encoders or feature sequences `[B, Tfeat, F]` for feature encoders.

Delegated-body operators, representative:

- Wav2Vec2 encoder: Conv1d feature extractor, group/layer norm, GELU, feature projection, convolutional positional embedding, noncausal self-attention, FFN, optional adapter Conv1d + GLU.
- BART/mBART/Speech2Text2 decoder: token embedding, learned/sinusoidal positions depending family, causal self-attention with KV cache, encoder-decoder cross-attention, LayerNorm, FFN/GELU or activation variant, LM head.
- BERT LM decoder variant: token/position/type embeddings as configured, self-attention with causal decoder mode, cross-attention, GELU FFN, LM prediction head.

Parameter aliasing:

- Wrapper-level encoder and decoder weights are independent.
- Wrapper forces `tie_word_embeddings=False`.
- Decoder-internal input embedding/LM-head tying remains decoder-family-owned and must preserve decoder checkpoint aliasing.

## 5. Layer/block breakdown

Wrapper forward:

```text
if encoder_outputs is absent:
  inputs = input_values or input_features or positional inputs
  encoder_outputs = encoder(inputs, attention_mask, output flags)
else:
  encoder_outputs = BaseModelOutput(*encoder_outputs) if tuple

encoder_hidden_states = encoder_outputs[0]             # [B, Senc, Denc_out]
if projection enabled:
  encoder_hidden_states = Linear(...)(encoder_hidden_states)  # [B, Senc, Ddec]

if attention_mask is present:
  encoder_attention_mask = encoder._get_feature_vector_attention_mask(Senc, attention_mask)
else:
  encoder_attention_mask = None

if labels and no decoder inputs:
  decoder_input_ids = shift_tokens_right(labels, pad_token_id, decoder_start_token_id)

decoder_outputs = decoder(
  input_ids/inputs_embeds,
  attention_mask=decoder_attention_mask,
  encoder_hidden_states=encoder_hidden_states,
  encoder_attention_mask=encoder_attention_mask,
  past_key_values=past_key_values,
  use_cache=use_cache
)
return Seq2SeqLMOutput(logits, past_key_values, decoder states, cross attentions, encoder states)
```

`shift_tokens_right` is training/loss-adjacent but also useful for teacher-forced parity: allocate zeros like labels, copy labels `[:, :-1]` to `[:, 1:]`, set first token to `decoder_start_token_id`, and replace `-100` with `pad_token_id`.

## 6. Attention requirements

Wrapper attention contract:

- Encoder self-attention: noncausal and delegated.
- Decoder self-attention: causal autoregressive and delegated.
- Cross-attention: decoder attends from target tokens `[B, Tdec, Ddec]` to encoder states `[B, Senc, Dcross]`.
- Cache: only decoder self-attention KV cache is wrapper-visible. Encoder outputs are independently cacheable but are not KV cache.
- Cross-attention cache, if any, is decoder-family-owned through the decoder `past_key_values` structure.
- Masking: `decoder_attention_mask` applies to target tokens; `encoder_attention_mask` is feature-length audio mask.
- Backend dispatch: wrapper advertises FlashAttention/SDPA, but admission must require both delegated bodies to support the selected backend or force eager fallback.

For representative BART/mBART decoders:

```text
self-attn cache per decoder layer: K,V shaped by decoder family, usually [B, heads, Tcache, head_dim]
cross-attn K,V may be cached inside decoder cache implementation
encoder_attention_mask is rectangular [B, Senc] expanded by decoder attention code
```

DinoML should not lower this as a single monolithic attention op. It should compose audited encoder attention, audited decoder causal attention/cache, and audited decoder cross-attention.

## 7. Position encoding and custom math

The wrapper has no custom positional math. Position encoding is delegated:

- Wav2Vec2-style encoders use convolutional positional embeddings after feature projection.
- BART/mBART decoders use learned positional embeddings and causal masks.
- Speech2Text2 uses its own decoder positional scheme.
- BERT LM head decoder uses BERT positional/token-type embeddings if admitted.

Wrapper custom helper:

```python
def shift_tokens_right(labels, pad_token_id, decoder_start_token_id):
    shifted = zeros_like(labels)
    shifted[:, 1:] = labels[:, :-1]
    shifted[:, 0] = decoder_start_token_id
    shifted[shifted == -100] = pad_token_id
    return shifted
```

Mask downsampling for Wav2Vec2-like encoders:

```python
def wav2vec2_feature_lengths(lengths, conv_kernel, conv_stride):
    for k, s in zip(conv_kernel, conv_stride):
        lengths = floor((lengths - k) / s) + 1
    return lengths
```

The exact mask construction after lengths is delegated to encoder source and must be parity-tested for each encoder family.

## 8. Preprocessing and input packing

There is no family-owned processor. The runtime ABI is selected by encoder/processor:

- Wav2Vec2FeatureExtractor checkpoints: input waveform is mono float audio, typically 16 kHz, padded with `0.0`, optional normalization from `preprocessor_config.json`, emitted as `input_values [B, Traw]` plus `attention_mask [B, Traw]`.
- Speech2Text-style feature encoders, if used through this wrapper, may consume `input_features [B, Tfeat, F]` rather than raw waveform. The wrapper accepts either but not both.
- For fetched Wav2Vec2FeatureExtractor configs: `sampling_rate=16000`, `feature_size=1`, `return_attention_mask=True`.
- Tokenizer/generation metadata is decoder-side. mBART checkpoints need language ids (`decoder_start_token_id`, forced BOS/EOS) packed by generation controller.

CPU/data-pipeline work:

- Audio decode, resample, mono conversion, waveform normalization, padding/truncation, and tokenization.
- Feature extraction stays outside first DinoML graph unless a delegated encoder audit explicitly admits it as graph work.

GPU/runtime work:

- Encoder body, optional projection, decoder prefill/decode, logits.
- Attention masks enter the graph as tensors/metadata; generation search remains controller-side.

## 9. Graph rewrite / lowering opportunities

### Rewrite: wrapper encoder-to-decoder projection

Source pattern:

```text
encoder_hidden_states [B, Senc, Denc] -> nn.Linear(Denc -> Ddec)
```

Replacement:

```text
Flatten(B*Senc, Denc) -> GEMM_RCR/Linear(+bias) -> Reshape(B, Senc, Ddec)
```

Preconditions:

- Projection module exists in wrapper state.
- Actual runtime encoder hidden width equals projection weight input width.
- Dense contiguous row-major hidden states or a supported accessor layout.
- Bias present as PyTorch `nn.Linear` default.

Failure cases:

- Adapter encoder emits `output_hidden_size` different from `encoder.hidden_size` while source-created projection expects `encoder.hidden_size`.
- Decoder sets `cross_attention_hidden_size`, in which case projection is absent and decoder cross-attention owns adaptation.

Parity test sketch: compare encoder hidden tensor through wrapper projection against PyTorch for `[B=2, Senc in {1, 17}, Denc, Ddec]`.

### Rewrite: encoder output cache

Source pattern:

```text
encoder_outputs supplied -> skip encoder call -> decoder cross-attends to supplied states
```

Replacement:

```text
AudioEncoder artifact -> persistent encoder_hidden/feature_mask -> Decoder artifact
```

Preconditions:

- Same encoder config, dtype, projection policy, and mask downsampling.
- Generation request reuses identical audio input.

Failure cases:

- Caller supplies tuple with unexpected hidden-state layout.
- Decoder family mutates expected cross-attn hidden width.

### Rewrite: feature mask length downsampling

Source pattern:

```text
attention_mask [B,Traw] -> cumsum/sum lengths -> conv length formula -> bool feature mask [B,Senc]
```

Replacement:

```text
LengthFromMask -> StaticConvLengthChain -> PrefixMask
```

Preconditions:

- Encoder family exposes Wav2Vec2-compatible `_get_feature_vector_attention_mask`.
- Conv kernels/strides are static and match config.
- Padding is right-padding prefix-valid mask.

Failure cases:

- Non-prefix masks or custom encoder attention-mask semantics.
- Optional adapters alter output lengths and are not included in the formula.

### Rewrite: last-token logits for decode

Source pattern:

```text
decoder(... one new token with cache ...) -> logits [B, 1, vocab]
```

Replacement:

```text
DecodeOneStep -> LMHead only for last position
```

Preconditions:

- Decoder family supports cache and emits only the new token hidden state.
- Generation controller does not request full-sequence logits.

## 10. Kernel fusion candidates

Highest priority:

- Decoder causal attention with KV cache for BART/mBART/BERT/Speech2Text2 delegated families.
- Decoder cross-attention over encoder states, including rectangular mask handling.
- Linear/GEMM + bias for optional `enc_to_dec_proj` and decoder LM head.
- Wav2Vec2 Conv1d feature extractor and feature projection for common admitted Wav2Vec2 encoders.

Medium priority:

- Wav2Vec2 convolutional positional embedding fusion around transpose/LayerNorm.
- LayerNorm + Linear regions in decoder blocks.
- FFN activation fusion: GELU/activation + second GEMM.
- Prefix-mask construction for Wav2Vec2 attention masks.

Lower priority:

- Training-only label shifting/loss.
- Beam-search controller optimizations.
- Processor CPU feature extraction acceleration unless DinoML chooses to own preprocessing.

## 11. Runtime staging plan

Stage 1: parse `SpeechEncoderDecoderConfig`, validate exact allowlisted `encoder.model_type` and `decoder.model_type`, and reject unsupported combinations before weight loading.

Stage 2: compose already-audited Wav2Vec2 encoder plus BART/mBART decoder without wrapper projection for common XLS-R and BART configs where encoder output width equals decoder width.

Stage 3: add wrapper `enc_to_dec_proj` with strict actual-width guards and parity tests.

Stage 4: split encoder precompute from decoder generation: cache encoder hidden states and feature masks independently from decoder KV cache.

Stage 5: enable decoder cache decode and cross-attention cache according to delegated decoder cache ABI.

Stage 6: add generation metadata parity for decoder start, forced BOS/EOS, padding, and tokenizer language control.

Stage 7: expand allowlist to Speech2Text2 and BERT LM decoders only after separate decoder audits cover their cross-attention/cache behavior.

Can stub initially: training loss, `shift_tokens_right`, beam search beyond greedy, output attentions/hidden states, `freeze_feature_encoder`, and direct wrapper `resize_token_embeddings`.

## 12. Parity and validation plan

- Config admission tests: accept known Wav2Vec2+BART/mBART no-projection configs; reject unknown encoder/decoder families; reject ambiguous adapter/projection width cases.
- Wrapper projection parity: random encoder hidden states through `enc_to_dec_proj` against PyTorch.
- Mask parity: raw `attention_mask [B,Traw]` through Wav2Vec2 mask downsampling against PyTorch for several waveform lengths and conv stride/kernel configs.
- Encoder-only parity: delegated Wav2Vec2 output and feature mask.
- Decoder prefill parity: feed saved encoder hidden states into delegated decoder and compare logits.
- Decode parity: one-token cached decode vs full-prefix decode for delegated decoder.
- End-to-end greedy generation parity for one tiny and one production checkpoint.
- Tolerances: fp32 `1e-4` absolute for logits; fp16/bf16 `1e-2` to `3e-2` depending delegated attention backend; token parity for deterministic greedy generation.

No DinoML tests or imports were run for this report per user scope.

## 13. Performance probes

- Processor throughput: audio decode/resample/normalization/padding samples/sec.
- Wav2Vec2 encoder throughput by raw waveform length and batch.
- Encoder output cache memory: `[B, Senc, Denc]` plus feature mask.
- Wrapper projection time and GEMM shape distribution.
- Decoder prefill time vs `Tdec`, `Senc`, vocab size.
- Decode tokens/sec with fixed encoder length and growing decoder cache.
- Cross-attention cost sweep over `Senc` from short utterances to long audio.
- Beam-size sweep for generation controller overhead.
- Attention backend comparison: eager vs SDPA/FlashAttention where both delegated bodies support it.
- End-to-end requests/hour split into CPU preprocessing, encoder, prefill, decode, tokenizer decode.

## 14. Skip/defer list

- Training loss and gradient checkpointing.
- `freeze_feature_encoder`.
- Direct `resize_token_embeddings` through the wrapper; source explicitly rejects it.
- Arbitrary encoder/decoder combinations outside an allowlist.
- Owning audio feature extraction inside DinoML GPU graph.
- Beam search and sampling beyond generation metadata needed for greedy parity.
- Output attentions/hidden states unless debugging parity requires them.
- Quantization and weight offload policies beyond delegated family support.

## 15. Final implementation checklist

- [ ] Parse wrapper config and nested encoder/decoder configs.
- [ ] Implement admission allowlist for audited encoder/decoder pairs.
- [ ] Load nested weights with stable `encoder.*`, `decoder.*`, and optional `enc_to_dec_proj.*` ownership.
- [ ] Preserve wrapper-level `tie_word_embeddings=False`.
- [ ] Implement `input_values`/`input_features` mutual-exclusion ABI.
- [ ] Implement or delegate Wav2Vec2 feature-mask downsampling.
- [ ] Implement optional `enc_to_dec_proj` with actual-width guard.
- [ ] Compose encoder artifact and decoder artifact with encoder-output cache boundary.
- [ ] Pass decoder `past_key_values` through according to delegated cache ABI.
- [ ] Honor generation metadata: decoder start, forced BOS/EOS, pad/eos ids, max length/beam metadata.
- [ ] Add no-projection Wav2Vec2+BART/mBART parity tests.
- [ ] Add projection parity tests for a safe synthetic config.
- [ ] Add rejection tests for unsupported decoder cross-attention width and ambiguous adapter width.
- [ ] Benchmark processor, encoder, projection, decoder prefill, and decode separately.

## Gated gaps for DinoML admission

- Need separate audits for each admitted encoder and decoder family; wrapper audit alone is insufficient.
- Need a precise cache manifest for each decoder family, including cross-attention cache ownership.
- Need a feature extractor ABI decision: CPU/data-pipeline owned first, with raw `input_values` as model input for Wav2Vec2-style encoders.
- Need strict projection/adaptation guards for `encoder.hidden_size`, `encoder.output_hidden_size`, `decoder.hidden_size`, and `decoder.cross_attention_hidden_size`.
- Need generation-controller metadata parity for multilingual mBART translation checkpoints.
- Need backend dispatch policy that only enables SDPA/FlashAttention when both delegated submodels support it.
