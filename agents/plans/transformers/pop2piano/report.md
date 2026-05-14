# Transformers audit: pop2piano

## 1. Source basis

Transformers commit/version: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`

Model id: [`sweetcocoa/pop2piano`](https://huggingface.co/sweetcocoa/pop2piano). Secondary public mirror checked for config variation: [`susnato/pop2piano_dev`](https://huggingface.co/susnato/pop2piano_dev).

Config source: HF `config.json`, `preprocessor_config.json`, `generation_config.json`, and `tokenizer_config.json` from both repos. Both public repos expose identical operator-significant configs. No 401/403 gated checkpoints were encountered.

Source files inspected:

- `X:/H/transformers/src/transformers/models/pop2piano/modeling_pop2piano.py`
- `X:/H/transformers/src/transformers/models/pop2piano/configuration_pop2piano.py`
- `X:/H/transformers/src/transformers/models/pop2piano/feature_extraction_pop2piano.py`
- `X:/H/transformers/src/transformers/models/pop2piano/tokenization_pop2piano.py`
- `X:/H/transformers/src/transformers/models/pop2piano/processing_pop2piano.py`
- `X:/H/transformers/tests/models/pop2piano/test_modeling_pop2piano.py`
- `X:/H/transformers/docs/source/en/model_doc/pop2piano.md`

Any missing files or assumptions: model source is native in-library PyTorch, copied mostly from T5 with Pop2Piano-specific audio conditioning and MIDI ABI. Feature extraction depends on `essentia`, `librosa`, `scipy`, and `torch`; MIDI postprocess depends on `pretty_midi`. The audit target is inference/generation from audio features to MIDI tokens and postprocessed MIDI notes/objects.

## 2. High-level architecture

Pop2Piano is an audio-to-music encoder-decoder model. The neural body is a T5-like encoder-decoder Transformer, but raw-audio preprocessing and generated-token postprocessing are strongly model-coupled.

Dataflow:

```text
raw mono audio -> CPU beat extraction/resample/log-mel chunks -> composer prefix embedding + mel feature sequence
  -> noncausal encoder -> autoregressive decoder with self/cross attention cache
  -> vocab logits / generation controller -> MIDI token decode -> notes / PrettyMIDI object
```

Stage decomposition:

- CPU/data pipeline: beat tracking with Essentia `RhythmExtractor2013`, beat interpolation with SciPy, optional Librosa resampling to 22050 Hz, STFT/mel/log transform, beatstep metadata packing, padding, and zero-row separators for batched examples.
- GPU/runtime stage 1: optional direct ABI accepting precomputed `input_features: [N_chunks, T_mel, 512]` plus `attention_mask`.
- GPU/runtime stage 2: composer embedding prefix is concatenated to each chunk sequence, then encoded independently as a batch of chunks.
- GPU/runtime stage 3: seq2seq generation with decoder self-attention KV cache and encoder-decoder cross-attention cache.
- CPU/postprocess: generated token rows are split back into original examples using feature-extractor separator metadata, then converted to note events and optional PrettyMIDI objects using extrapolated beatsteps.

Independently cacheable stages: precomputed audio features/beatsteps can be cached before the neural graph; encoder outputs for a fixed audio/composer prefix can be cached across decoder generation; cross-attention K/V are cached per decoder layer after first use.

## 3. Important config dimensions

Primary checkpoint dimensions from `sweetcocoa/pop2piano` `config.json`:

| Field | Value | Source / runtime meaning |
|---|---:|---|
| `vocab_size` | 2400 | token logits width |
| `composer_vocab_size` | 21 | composer embedding table size |
| `d_model` | 512 | hidden size / mel feature width |
| `d_kv` | 64 | per-head Q/K/V width |
| `num_heads` | 8 | MHA heads |
| `inner_dim` | 512 | `num_heads * d_kv`; equals `d_model` here but should not be inferred |
| `d_ff` | 2048 | FFN hidden width |
| `num_layers` | 6 | encoder layer count |
| `num_decoder_layers` | omitted | effective default is `num_layers` from config class |
| `relative_attention_num_buckets` | 32 | T5 relative-position buckets |
| `relative_attention_max_distance` | 128 | bucket saturation distance |
| `feed_forward_proj` | `gated-gelu` | chooses gated FFN module |
| `dense_act_fn` | omitted | effective default is `relu`; this means `gated-gelu` selects gated structure but activation lookup remains `relu` unless set |
| `dropout_rate` | 0.1 | disabled in eval |
| `layer_norm_epsilon` | 1e-6 | RMSNorm epsilon |
| `use_cache` | true | decoder generation cache enabled |
| `tie_word_embeddings` | false | checkpoint overrides source default `true`; no logits rescale by `d_model^-0.5` |
| `decoder_start_token_id` | 0 | generation starts with pad token |
| `eos_token_id` / `pad_token_id` | 1 / 0 | generation/tokenizer ABI |
| `dataset_target_length` | 256 | generation config also uses max length 256 |
| `dataset_input_length` | 1024 | historical/data-pipeline field; not read by modeling source |
| `dataset_sampling_rate` | 22050 | matches preprocessor |
| `n_fft` / `hop_length` / `n_mels` | 4096 / 1024 / 512 | historical names matching feature extractor defaults |

Feature extractor config:

| Field | Value |
|---|---:|
| `sampling_rate` | 22050 |
| `window_size` | 4096 |
| `hop_length` | 1024 |
| `min_frequency` | 10.0 |
| `feature_size` | 512 |
| `num_bars` | 2 |
| `padding_value` | 0 |

Generation config:

| Field | Value |
|---|---:|
| `max_length` | 256 |
| `decoder_start_token_id` | 0 |
| `eos_token_id` | 1 |
| `pad_token_id` | 0 |
| `return_dict_in_generate` | false |
| `composer_to_feature_token` | `composer1..composer21 -> 2052..2072` |

Representative checkpoint sweep:

| Repo | Type | Config variation | Notes |
|---|---|---|---|
| `sweetcocoa/pop2piano` | official/common | 512-dim, 6/6 layers, 8 heads, gated FFN, untied LM head | primary target |
| `susnato/pop2piano_dev` | public mirror/dev | identical operator-significant config | useful mirror only |
| `tests/models/pop2piano` synthetic config | debug | tiny dims such as `hidden_size=32`, `num_layers=2`, `d_ff=37`, decoder length 9 | source-level shape/cache tests, not a published checkpoint |

## 3a. Family variation traps

- `num_decoder_layers` can be omitted; config `__post_init__` sets it to `num_layers`.
- `hidden_size` aliases to `d_model`; `num_attention_heads` aliases to `num_heads`.
- `d_model == num_heads * d_kv` in public configs, but source explicitly computes `inner_dim = num_heads * d_kv`; DinoML should validate rather than infer.
- `feed_forward_proj="gated-gelu"` only sets `is_gated_act=True`; the activation function is `dense_act_fn`, whose source default is `relu`. Public config omits `dense_act_fn`.
- Source config default has `tie_word_embeddings=True`, but the public checkpoint sets `false`; this changes logits scaling in `forward`.
- `n_positions`, `output_past`, and `dataset_*` fields appear in checkpoint config but are not read by inspected modeling source for graph structure.
- Audio batch packing is unusual: one user audio can become multiple model examples/chunks, and batched user audios are separated by zero feature rows. Postprocess depends on those separators.
- The composer control is not a text token in `input_ids`; it is an embedding prefix derived from generation config IDs `2052..2072`, offset by the minimum value and looked up in a separate `composer_vocab_size` table.
- Layout translation should be disabled for the source semantic graph. Audio features are rank-3 `[chunk_batch, mel_seq, 512]`; attention projections and masks are sequence-major `[B, S, D]`. There is no image/channel-last region.

## 4. Operator coverage checklist

Tensor/layout ops:

- Embedding lookup for token IDs and composer IDs.
- `view`/reshape, `transpose(1, 2)`, `permute([2,0,1])`, `contiguous`, rank expansion with `unsqueeze`, sequence concatenation along dim 1.
- Attention mask expansion/inversion and broadcast add.
- Pad/zero-fill and separator row handling in CPU feature pipeline.
- Type casts to fp32 for RMSNorm/softmax and back to model dtype.
- `clamp`/`isinf` guarded fp16 clamp path exists in model blocks; inference may keep it for parity or guard it off for fp32/bf16.

Neural primitives:

- `Embedding(2400, 512)` shared by encoder/decoder input embeddings.
- Composer `Embedding(21, 512)`.
- Q/K/V/O linears per attention: `Linear(512 -> 512, bias=False)` x4.
- Gated FFN: `wi_0: Linear(512 -> 2048, bias=False)`, `wi_1: Linear(512 -> 2048, bias=False)`, activation, elementwise multiply, `wo: Linear(2048 -> 512, bias=False)`.
- Non-gated fallback FFN for config variants: `Linear(512 -> 2048) -> activation -> Linear(2048 -> 512)`, all biasless.
- RMSNorm/T5LayerNorm with learned scale only, no mean subtraction, fp32 variance.
- LM head `Linear(512 -> 2400, bias=False)`.

Attention primitives:

- Dense MHA self-attention for encoder and decoder.
- Dense MHA cross-attention in decoder.
- T5-style relative attention bias only on the first self-attention layer of each stack; position bias is then reused across layers in the stack.
- Causal decoder mask via `create_causal_mask`; encoder mask is additive `(1 - mask) * finfo.min`.
- Softmax in fp32 over score last dimension, cast back to score dtype.

Position/relative-bias ops:

- T5 bucketization with exact small buckets and logarithmic large buckets.
- Relative bias embedding table shape `[32, 8]`; output bias shape `[1, 8, Q, K]`.

Generation/cache ops:

- Encoder-decoder generation controller with `decoder_start_token_id=0`, `eos_token_id=1`, `max_length=256`.
- Decoder self-attention dynamic KV cache per layer.
- Encoder-decoder cross-attention cache per layer, with `is_updated[layer_idx]` preventing recomputation after first decode step.
- Beam/search variants are inherited from `GenerationMixin`; first DinoML target can implement greedy/sampling only if scoped.

Preprocessing-coupled ops:

- Mono waveform validation, optional resampling, Essentia beat tracking, SciPy interpolation.
- STFT power spectrogram with Hann window length 4096, hop 1024, mel filter bank 512 bins, HTK mel scale, log clamp at `1e-6`.
- Chunking by `num_bars * 4` beat steps, per-audio chunk padding to longest chunk waveform before mel transform.

Token/music postprocess ABI:

- Token IDs decode to token types `TOKEN_TIME`, `TOKEN_VELOCITY`, `TOKEN_NOTE`, `TOKEN_SPECIAL`.
- Generated rows are grouped per original audio by zero-separator rows in feature attention mask.
- Notes are sorted by onset/offset key and converted to PrettyMIDI with beatstep-derived seconds.

## 5. Layer/block breakdown

Audio feature path:

```text
raw mono waveform
  -> RhythmExtractor2013 -> beat_times
  -> interpolate beatsteps with steps_per_beat=2
  -> slice waveform from beatsteps[0] to beatsteps[-1]
  -> split into 2-bar chunks using extrapolated beatsteps
  -> per-audio pad chunks to common waveform length
  -> STFT/mel power spectrogram -> log(max(x, 1e-6))
  -> transpose to [num_chunks, mel_frames, 512]
```

Conditioning prefix:

```text
composer_value = generation_config.composer_to_feature_token[composer]
index_shifted = composer_value - min(composer_to_feature_token.values())
composer_embed = Embedding(21, 512)[index_shifted].unsqueeze(1)
encoder_inputs_embeds = cat([composer_embed, input_features], dim=1)
attention_mask = cat([attention_mask[:, 0:1], attention_mask], dim=1)
```

Encoder block, repeated 6 times:

```text
x_norm = RMSNorm(x)
q,k,v = Linear(512 -> 512, bias=False)(x_norm), split to [B, 8, S, 64]
scores = q @ k.T
scores += shared self position_bias + additive encoder mask
a = softmax(scores.float(), dim=-1).to(scores.dtype)
x = x + Linear(512 -> 512, bias=False)(a @ v)
y = RMSNorm(x)
ff = relu(wi_0(y)) * wi_1(y)        # public config effective activation
x = x + wo(ff)
```

Decoder block, repeated 6 times:

```text
x = decoder token embedding or shifted labels
x_norm = RMSNorm(x)
self_attn(q,k,v, causal mask, self KV cache, decoder relative position_bias)
x = residual add
x_norm = RMSNorm(x)
cross_attn(q from decoder, k/v from encoder_hidden_states, encoder mask, cross KV cache)
x = residual add
y = RMSNorm(x)
ff = relu(wi_0(y)) * wi_1(y)
x = residual add wo(ff)
```

Head:

```text
if tie_word_embeddings: sequence_output *= d_model ** -0.5
logits = lm_head(sequence_output)
```

For the public checkpoint `tie_word_embeddings=false`, so the scaling branch is not active.

## 6. Attention requirements

Encoder self-attention:

- Noncausal dense MHA.
- Q/K/V shape after projection: `[B, S_enc, 512] -> [B, 8, S_enc, 64]`.
- Mask input is `[B, S_enc]`; source expands to `[B, 1, 1, S_enc]` additive mask with `torch.finfo(dtype).min`.
- First encoder layer computes bidirectional T5 relative bias; all encoder layers reuse that position bias.

Decoder self-attention:

- Causal dense MHA with dynamic self-attention KV cache.
- Q/K/V projected width is 512, stored as `[B, 8, T, 64]`.
- Cached K/V are after linear projection and before score matmul; no RoPE is present.
- Relative bias is unidirectional because `is_decoder=True`; `past_seen_tokens` shifts the query positions during decode.

Decoder cross-attention:

- Dense MHA from decoder queries to encoder hidden states.
- No relative attention bias in cross-attention; source creates zeros position bias and adds encoder mask.
- With `EncoderDecoderCache`, cross-attention K/V are computed once per layer and then reused through `is_updated[layer_idx]`.

Backend compatibility:

- FlashAttention/SDPA could handle dense softmax attention if additive masks and relative bias are supplied as score bias. Need parity for fp32 softmax accumulation and score-bias order.
- No GQA/MQA, no sliding window, no packed varlen source path, no RoPE/ALiBi.
- Output attentions are optional/deferred for first DinoML target.

## 7. Position encoding and custom math

Pop2Piano uses T5-style relative attention buckets, not absolute position embeddings or RoPE.

```python
def relative_position_bucket(relative_position, bidirectional, num_buckets=32, max_distance=128):
    buckets = 0
    if bidirectional:
        num_buckets //= 2
        buckets += (relative_position > 0).long() * num_buckets
        relative_position = abs(relative_position)
    else:
        relative_position = -minimum(relative_position, 0)
    max_exact = num_buckets // 2
    is_small = relative_position < max_exact
    large = max_exact + (
        log(relative_position.float() / max_exact) / log(max_distance / max_exact)
        * (num_buckets - max_exact)
    ).long()
    large = minimum(large, num_buckets - 1)
    return buckets + where(is_small, relative_position, large)
```

Bias computation:

```text
context_position = arange(Q)[:, None] + past_seen_tokens
memory_position = arange(K)[None, :]
relative_position = memory_position - context_position
buckets -> Embedding(num_buckets, num_heads) -> [1, H, Q, K]
```

Precompute opportunities:

- Encoder bidirectional bias for fixed chunk sequence length can be precomputed per shape.
- Decoder prefill bias can be precomputed per target length.
- Decode-step bias depends on `past_seen_tokens` and current `K`; a small row-wise bucket kernel/table lookup is enough.

## 8. Preprocessing and input packing

Raw waveform contract:

- Single-channel audio only; non-rank-1 audio raises.
- Caller supplies sampling rate. For inference, `resample=True` is expected when input rate differs from 22050 Hz.
- Beat extraction is external-library CPU work. DinoML should not admit this into the first GPU graph.

Feature extraction math:

- Window: Hann of length `window_size=4096`.
- Hop: `1024`.
- Power spectrogram: `power=2.0`.
- Mel filters: 512 filters, min frequency 10 Hz, max frequency 11025 Hz, HTK mel scale, no normalization.
- Log clamp: `np.log(np.clip(mel_specs, a_min=1e-6, a_max=None))`.
- Model tensor: source transposes log-mel from `[chunks, mel_bins, frames]` to `[chunks, frames, mel_bins]`.

Chunking and padding:

- `steps_per_beat=2` by default in processor call.
- `num_steps = num_bars * 4`, so public config with `num_bars=2` slices 8 beat steps per chunk.
- `extrapolated_beatstep` extends by `(num_bars + 1) * 4 + 1`, used later for cutoff and MIDI timing.
- Within one audio, chunks are waveform-padded to the longest chunk before mel.
- For batched user audios, `input_features` from all audio examples are concatenated along chunk-batch dimension with an all-zero separator row after each user audio. `attention_mask[:, 0] == 0` identifies separators for postprocess.

Runtime ABI choices:

- First DinoML target should accept already-computed `input_features` and `attention_mask`; CPU feature extraction can be an integration wrapper.
- Postprocess requires `beatsteps`, `extrapolated_beatstep`, and their masks when batched. These are metadata tensors for the CPU ABI, not neural graph inputs.

## 9. Graph rewrite / lowering opportunities

### Rewrite: Composer Feature Token -> Prefix Embedding

Source pattern:

```text
generation_config.composer_to_feature_token[composer]
  -> subtract min token id
  -> composer embedding
  -> cat before input_features
```

Replacement pattern: host-side composer string validation emits a scalar composer index `[0..20]`; GPU graph does `Embedding(21,512)` and prefix concat.

Preconditions:

- Generation config has exactly `composer_vocab_size` entries.
- Values form the public contiguous range 2052..2072 or an explicitly validated offset/range.
- `input_features` last dim equals `d_model`.

Failure cases: unknown composer, non-contiguous or out-of-range generation mapping, mismatch with `composer_vocab_size`.

Parity test sketch: compare `get_mel_conditioner_outputs` for all 21 composers with and without masks.

### Rewrite: T5 Relative Bias as Shape-Keyed Constant/Small Kernel

Source pattern: per-forward bucketization with `arange`, `log`, `where`, embedding lookup, `permute`, and `unsqueeze`.

Replacement pattern: for fixed `(is_decoder, Q, K, past_seen_tokens)` emit cached bucket IDs or cached bias tensor; for decode use a tiny bucket lookup for the current row.

Preconditions:

- `relative_attention_num_buckets`, `relative_attention_max_distance`, and `num_heads` static.
- Position bias is shared across layers inside one encoder/decoder stack exactly as source does.

Failure cases: dynamic long sequence beyond cached plan without fallback, incorrect bidirectional flag, missing `past_seen_tokens` offset in decode.

Parity test sketch: compare bias tensors for encoder, decoder prefill, and decode step with nonzero past length.

### Rewrite: Biasless Linear Triples -> Packed QKV Projection

Source pattern: separate `q`, `k`, `v` linear layers, each `Linear(512 -> 512, bias=False)`.

Replacement pattern: concatenate weights as `[Wq; Wk; Wv]` for one GEMM producing `[B, S, 3, H, D]`, then split in Q/K/V order.

Preconditions:

- Same input tensor for Q/K/V self-attention.
- No projection bias.
- Weight layout transform respects PyTorch `nn.Linear` storage `[out_features, in_features]`.

Failure cases: cross-attention Q comes from decoder while K/V come from encoder, so only K/V can be packed together there; quantized/packed weight loading must preserve split metadata.

Parity test sketch: one attention layer projection outputs before matmul, exact fp32 comparison.

### Rewrite: Gated FFN Fusion

Source pattern:

```text
relu(wi_0(x)) * wi_1(x) -> wo
```

Replacement pattern: packed two-output GEMM for `wi_0/wi_1`, fused activation multiply, then output GEMM.

Preconditions:

- `config.is_gated_act=True`.
- Activation is the effective `dense_act_fn` from config, not inferred from `feed_forward_proj` suffix.
- No dropout in eval.

Failure cases: non-gated `feed_forward_proj`, custom `dense_act_fn`, training/dropout.

Parity test sketch: FFN block fp32/fp16 comparison for public config and a synthetic non-gated config.

### Rewrite: Audio Feature Extraction Boundary

Source pattern: external CPU beat tracking + STFT/mel/log + chunk packing.

Replacement pattern: first integration treats this as a host preprocessing ABI and compiles only `input_features -> logits`.

Preconditions:

- Feature extractor version/config recorded with artifact.
- `input_features` shape `[chunk_batch, T, 512]`; `attention_mask` shape `[chunk_batch, T]`, before composer prefix.

Failure cases: runtime tries to pass raw waveform directly to compiled graph, missing beatstep metadata for MIDI decode, batched decode without separator masks.

Parity test sketch: compare precomputed HF processor outputs to DinoML wrapper inputs and verify generated MIDI grouping.

## 10. Kernel fusion candidates

Highest priority:

- RMSNorm: every attention and FFN sublayer uses T5-style scale-only RMSNorm with fp32 variance.
- Biasless GEMM coverage: all projections/FFNs/LM head are biasless dense linears, good for existing GEMM paths.
- Dense attention with additive relative bias: prefill and decode need efficient score-bias-softmax-matmul while preserving fp32 softmax.
- Decoder KV cache and cross-attention cache: generation throughput depends on avoiding full decoder recomputation and avoiding repeated encoder K/V projection.

Medium priority:

- Packed self-attention QKV projection and packed cross-attention KV projection.
- Gated FFN fused activation multiply.
- Relative-bias precompute/cache for common source/target lengths.
- Last-token-only logits in decode, if generation controller only consumes the final row.

Lower priority:

- GPU STFT/mel frontend. Useful only after core model parity, because beat tracking remains external CPU work.
- PrettyMIDI postprocess acceleration. It is CPU object construction and not a kernel priority.
- Training loss and dropout paths.

## 11. Runtime staging plan

Stage 1: parse config/generation/preprocessor metadata and load weights for the public checkpoint. Reject remote-code or config variants not represented by native source.

Stage 2: compile a single encoder or decoder block with random tensors; validate RMSNorm, relative bias, attention masks, and gated FFN.

Stage 3: compile encoder-only path accepting precomputed `input_features` plus composer index/prefix and attention mask. Stub raw audio preprocessing.

Stage 4: compile full seq2seq prefill with teacher-forced `decoder_input_ids` and compare logits.

Stage 5: add generation decode with decoder self KV cache and encoder-decoder cross-attention cache. Start with greedy generation and fixed `max_length=256`.

Stage 6: integrate CPU processor wrapper and CPU MIDI postprocess ABI. Validate that chunk separator rows map generated token rows back to original audio examples.

Stage 7: add fusions: QKV packing, FFN fusion, relative-bias caching, optimized attention, last-token logits.

## 12. Parity and validation plan

- Feature extractor parity: raw mono waveforms at matching and non-matching sampling rates; compare `input_features`, `beatsteps`, masks, and separator rows against HF processor. Expect small floating differences only if replacing STFT/mel implementation.
- Composer prefix parity: all 21 composer labels, masked and unmasked batches; exact shape/mask behavior.
- Custom math parity: relative bucket IDs and bias tensors for encoder, decoder prefill, and decode with `past_seen_tokens > 0`.
- Single-block parity: fp32 encoder block and decoder block with fixed masks, tolerance around `1e-5`.
- Encoder parity: public checkpoint, precomputed random `input_features`, compare `encoder_last_hidden_state`.
- Teacher-forced seq2seq parity: fixed `decoder_input_ids`, compare logits. fp32 `rtol=1e-4`, fp16/bf16 wider tolerances around `1e-2` depending attention backend.
- Cache parity: prefill then one-token decode must match full-prefix decode for logits and cache lengths.
- End-to-end token parity: use HF processor on a short audio sample and greedy generation with composer1; compare generated token IDs before MIDI conversion.
- MIDI ABI parity: compare decoded notes/PrettyMIDI note start/end/pitch/velocity for single and batched examples with separator masks.

Do not require DinoML tests for this audit report; these are future integration tests.

## 13. Performance probes

- CPU preprocessing throughput: beat extraction, resampling, mel spectrogram, and chunk packing separately.
- Encoder throughput by `chunk_batch` and mel-frame length.
- Decoder prefill latency for target lengths up to 256.
- Decode tokens/sec with and without cross-attention K/V cache.
- Attention backend comparison: unfused matmul-softmax-matmul vs fused score-bias attention.
- Relative-bias generation overhead by `Q/K` length.
- Batch/user-audio packing sweep: number of chunks per audio, separator overhead, padded mel-frame waste.
- End-to-end requests/hour split into preprocessing, encoder, decode, and MIDI postprocess.
- KV cache memory: `2 * num_decoder_layers * B * num_heads * T * d_kv` for self cache plus cross cache sized by encoder length.

## 14. Skip/defer list

- Training, loss, dropout, and gradient checkpointing.
- Beam search and advanced `GenerationMixin` processors; greedy/sampling first is enough if explicitly scoped.
- Raw waveform GPU feature extraction; CPU feature pipeline first.
- PrettyMIDI object creation inside DinoML runtime; keep as host postprocess.
- Output attentions/hidden states for production path.
- Non-public or hypothetical config variants: non-gated FFN, tied embeddings, different `dense_act_fn`, asymmetric decoder layers. Support later with explicit tests.
- Quantized/packed weights; no source-coupled quantized format is present in this implementation.

## 15. Final implementation checklist

- [ ] Parse `Pop2PianoConfig`, generation config, tokenizer config, and preprocessor config.
- [ ] Load public checkpoint weights with tied/untied embedding contract preserved.
- [ ] Implement scale-only RMSNorm with fp32 variance.
- [ ] Implement T5 relative-position bucket/bias with encoder/decoder flags and decode `past_seen_tokens`.
- [ ] Implement encoder dense MHA with additive mask and shared position bias.
- [ ] Implement decoder self-attention KV cache.
- [ ] Implement decoder cross-attention and cross K/V cache reuse.
- [ ] Implement biasless GEMMs for Q/K/V/O, FFN, and LM head.
- [ ] Implement gated FFN using effective `dense_act_fn`.
- [ ] Implement composer prefix embedding and mask prefix concat.
- [ ] Define first runtime ABI for precomputed `input_features`, `attention_mask`, composer ID, and generated token IDs.
- [ ] Define host ABI for feature extractor metadata: `beatsteps`, `extrapolated_beatstep`, and masks.
- [ ] Implement MIDI token decode/postprocess wrapper or call out to HF tokenizer initially.
- [ ] Add parity tests for relative bias, one block, encoder, teacher-forced logits, decode cache, and MIDI decode.
- [ ] Add performance probes for preprocessing, encoder, prefill, decode, cache memory, and postprocess.
