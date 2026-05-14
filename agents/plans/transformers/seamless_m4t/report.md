# Transformers SeamlessM4T Operator and Integration Report

## 1. Source basis

```text
Transformers commit/version:
  Local checkout X:/H/transformers at b75feb2af64c3e29cbbc1bd859958c5432cc7ed4.
  Remote upstream source corresponds to https://github.com/huggingface/transformers.

Model id:
  Primary worked examples: facebook/hf-seamless-m4t-medium and facebook/hf-seamless-m4t-large.
  Native report scope is model_type="seamless_m4t" v1. The sibling
  seamless_m4t_v2 architecture is out of scope for this report.

Config source:
  https://huggingface.co/facebook/hf-seamless-m4t-medium/raw/main/config.json
  https://huggingface.co/facebook/hf-seamless-m4t-large/raw/main/config.json
  https://huggingface.co/facebook/hf-seamless-m4t-medium/raw/main/preprocessor_config.json
  https://huggingface.co/facebook/hf-seamless-m4t-large/raw/main/preprocessor_config.json
  https://huggingface.co/facebook/hf-seamless-m4t-medium/raw/main/generation_config.json
  https://huggingface.co/facebook/hf-seamless-m4t-large/raw/main/generation_config.json
  Local snapshots: H:/dinoml_v2/agents/plans/transformers/seamless_m4t/_sources/

Source files inspected:
  X:/H/transformers/src/transformers/models/seamless_m4t/modeling_seamless_m4t.py
  X:/H/transformers/src/transformers/models/seamless_m4t/configuration_seamless_m4t.py
  X:/H/transformers/src/transformers/models/seamless_m4t/feature_extraction_seamless_m4t.py
  X:/H/transformers/src/transformers/models/seamless_m4t/processing_seamless_m4t.py
  X:/H/transformers/src/transformers/models/seamless_m4t/tokenization_seamless_m4t.py
  X:/H/transformers/src/transformers/models/seamless_m4t/convert_fairseq2_to_hf.py
  X:/H/transformers/tests/models/seamless_m4t/test_modeling_seamless_m4t.py
  X:/H/transformers/tests/models/seamless_m4t/test_feature_extraction_seamless_m4t.py

Any missing files or assumptions:
  No trust_remote_code files are needed for the two official HF checkpoints.
  Only two official native v1 Transformers configs were found/inspected:
  facebook/hf-seamless-m4t-medium and facebook/hf-seamless-m4t-large. The
  third useful size point is the local source default, which is large-like for
  most structural fields. The official facebook/seamless-m4t-unity-small repo is
  a fairseq2/on-device export rather than a native `model_type="seamless_m4t"`
  Transformers config, so it is not used as an operator source here.
  facebook/seamless-m4t-v2-large was snapshotted only as an explicit
  out-of-scope contrast because it uses `model_type="seamless_m4t_v2"` and
  different native source. This report targets inference-first CUDA integration
  for text/speech translation paths, with speech synthesis staged after text
  generation. Training, LayerDrop, and checkpoint conversion are deferred.
```

## 2. High-level architecture

SeamlessM4T v1 is a multi-path translation system:

```text
text -> text encoder -> text decoder/generate -> text logits
audio -> fbank/stride pack -> Conformer speech encoder -> text decoder/generate -> text logits
text/audio -> text tokens -> decoder hidden states -> T2U encoder-decoder/generate -> unit tokens
unit tokens + lang/spkr ids -> duration expansion -> HiFi-GAN vocoder -> waveform
```

Stage decomposition:

- CPU/data pipeline: tokenizer language-token packing for text; Kaldi-style fbank extraction, per-mel normalization, padding/truncation, stride-2 feature packing, and attention-mask packing for audio.
- Independently cacheable encoders: text encoder and speech Conformer encoder can be validated separately. Speech encoder output length is further reduced by the adapter stride when `add_adapter=True`.
- Text decoder prefill/decode: BART-like causal decoder with self-cache and reusable cross-attention cache over text or speech encoder states.
- Speech synthesis: generation controller reruns/uses text decoder hidden states, feeds a separate text-to-unit seq2seq model, strips control tokens, offsets unit ids, then invokes the duration predictor and HiFi-GAN.
- Validatable slices: feature extractor, speech encoder, text encoder, text decoder prefill, cached decode, T2U prefill/decode, duration expansion, and vocoder waveform length can all be tested independently.

## 3. Important config dimensions

Worked example: `facebook/hf-seamless-m4t-large`.

| Field | Value | Source |
|---|---:|---|
| hidden_size | 1024 | config.json |
| text encoder layers | 24 | config.json |
| text decoder layers | 24 | config.json |
| text attention heads | 16 | config.json |
| head_dim | 64 | inferred `1024 / 16` |
| encoder_ffn_dim / decoder_ffn_dim | 8192 / 8192 | config.json |
| vocab_size | 256102 | config.json |
| max_position_embeddings | 1024 | config.json |
| scale_embedding | true | config.json/source |
| activation_function | relu | config.json |
| speech_encoder_layers | 24 | config.json |
| speech_encoder_attention_heads | 16 | config.json |
| speech_encoder_intermediate_size | 4096 | config.json |
| speech hidden act | swish | config.json |
| feature_projection_input_dim | 160 | config.json |
| speech position type | relative | config.json |
| max_source_positions | 4096 | config.json |
| conv positional kernel/groups | 128 / 16 | config.json |
| Conformer depthwise kernel | 31 | config.json |
| adapter layers/kernel/stride | 1 / 8 / 8 | config.json |
| t2u encoder/decoder layers | 6 / 6 | config.json |
| t2u vocab_size | 10082 | config.json |
| t2u max positions | 2048 | config.json |
| unit_hifi_gan_vocab_size | 10000 | config.json |
| unit/lang/spkr embed dims | 1280 / 256 / 256 | config.json |
| upsample rates/product | `[5,4,4,2,2]` / 160 | config.json |
| upsample initial channel | 512 | config.json |
| resblock kernels/dilations | `[3,7,11]` / three `[1,3,5]` groups | config.json |
| cache support | `EncoderDecoderCache(DynamicCache, DynamicCache)` | source |
| torch_dtype | float32 | config.json |

Representative checkpoint sweep:

| Checkpoint/source | H | text enc/dec | speech enc | T2U enc/dec | text FFN | vocab | max pos | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| source default `SeamlessM4TConfig()` | 1024 | 24 / 24 | 24 | 6 / 6 | 8192 | 256102 | 1024 | large-like source default; vocab differs from medium |
| facebook/hf-seamless-m4t-medium | 1024 | 12 / 12 | 12 | 4 / 4 | 4096 | 256206 | 4096 | smaller layer count, larger text vocab, longer text positions |
| facebook/hf-seamless-m4t-large | 1024 | 24 / 24 | 24 | 6 / 6 | 8192 | 256102 | 1024 | production large v1 |
| facebook/seamless-m4t-v2-large | 1024 | 24 / 24 | 24 | 6 / 6 | 8192 | 256102 | 4096 | out of scope; `model_type="seamless_m4t_v2"`, different source and UnitY2/T2U behavior |

Preprocessor sweep:

| Checkpoint | sampling_rate | num_mel_bins | stride | feature dim to model | attention mask |
|---|---:|---:|---:|---:|---|
| medium | 16000 | 80 | 2 | 160 | returned by default |
| large | 16000 | 80 | 2 | 160 | returned by default |

## 3a. Family variation traps

- The official native v1 family has medium and large topologies, not a broad size ladder. Do not fold `seamless_m4t_v2` into this report; v2 has separate source/config.
- The official `facebook/seamless-m4t-unity-small` checkpoint is an on-device fairseq2 export, not an in-library `SeamlessM4TConfig` checkpoint. Treat it as a separate audit if DinoML wants the small/mobile topology.
- `max_position_embeddings` is larger in medium (4096) than large (1024). Text source/target sequence limits are checkpoint-specific.
- Medium uses 12 text/speech layers and 4 T2U layers; large uses 24 text/speech layers and 6 T2U layers.
- Speech features are not `[B, mel, frames]` by the time they enter the model. The feature extractor reshapes fbank frames from `[B, frames, 80]` to `[B, frames//2, 160]`.
- Speech encoder attention masks are subsampled again after the adapter. Decoder cross masks must be recomputed from post-adapter lengths.
- Speech Conformer position embeddings are config-dependent: standard configs use relative position embeddings, but source also implements rotary and `None`.
- Text/T2U seq2seq use the same `SeamlessM4TEncoder`/`Decoder` classes after T2U config rewriting. T2U rewrites `t2u_*` fields into ordinary `vocab_size`, `encoder_layers`, `decoder_layers`, and token ids in a copied config.
- Text generation and T2U generation both use tied decoder embeddings/LM heads. Preserve aliasing for `shared.weight`, text encoder embeddings, text decoder embeddings, and LM head in text models; preserve T2U decoder embedding/LM head alias.
- Speech synthesis is not just another LM head: it uses arg/control-token postprocessing, duration prediction with `expm1 -> round -> clamp(min=1)`, value-dependent repeat interleave, padding, and ConvTranspose1d vocoder stages.
- `SeamlessM4TProcessor` rejects simultaneous `text` and `audio`; runtime entrypoints should similarly make modality selection explicit.
- Layout-sensitive code uses Conv1d tensors in `[B,C,T]` inside Conformer/vocoder, while transformer states are `[B,T,H]`. Initial lowering should preserve source axes and guard any local channel-last rewrite.

## 4. Operator coverage checklist

### Tensor/layout ops

- Embedding lookup, scaled embeddings by `sqrt(H)`.
- Position id creation from padding masks via `ne`, `cumsum`, masking, and gather/index-select.
- Reshape/view/transpose for attention `[B,T,H] <-> [B,heads,T,D]` and source eager BMM layout `[B*heads,T,D]`.
- Mask expansion to additive attention masks `[B,1,T,S]` with dtype min.
- Speech feature reshape `[B,F,80] -> [B,F/2,160]` and mask pick `indices % stride == 1`.
- Speech/vocoder Conv1d transposes `[B,T,C] <-> [B,C,T]`.
- `gather` for beam hidden-state selection and duration output lengths.
- `repeat_interleave`, `pad_sequence`, `cat`, `where`, `cumsum`, `clamp`, `round`, `expm1`, `tanh`.

### Neural network primitives

- Text/T2U Linear:
  - MHA Q/K/V/O `Linear(1024 -> 1024, bias=True)`.
  - FFN medium `Linear(1024 -> 4096) -> ReLU -> Linear(4096 -> 1024)`.
  - FFN large `Linear(1024 -> 8192) -> ReLU -> Linear(8192 -> 1024)`.
  - LM head `Linear(1024 -> vocab_size, bias=False)` and T2U `Linear(1024 -> 10082, bias=False)`.
- LayerNorm over last dim, `eps=1e-5`.
- Speech feature projection: `LayerNorm(160) -> Linear(160 -> 1024)`.
- Speech Conformer FFNs: `Linear(1024 -> 4096) -> Swish -> Linear(4096 -> 1024)`, half residual scaling in Conformer blocks.
- Conformer convolution module: LayerNorm, pointwise `Conv1d(1024 -> 2048,k=1,bias=False)`, GLU over channel dim, depthwise `Conv1d(1024 -> 1024,k=31,padding=same,groups=1024,bias=False)`, BatchNorm1d, Swish, pointwise `Conv1d(1024 -> 1024,k=1,bias=False)`.
- Speech adapter: two `Conv1d(1024 -> 2048,k=8,stride=8,padding=4)` plus GLU, then non-positional self-attention and ReLU FFN.
- Vocoder duration predictor: Conv1d/LayerNorm/ReLU/Dropout stack plus `Linear(1280 -> 1)`.
- HiFi-GAN: `Conv1d(1792 -> 512,k=7)`, five ConvTranspose1d upsamplers, residual Conv1d blocks, final `Conv1d(channels -> 1,k=7)`, tanh.

### Attention primitives

- Text encoder noncausal MHA, text decoder causal self-attention, text decoder cross-attention.
- Speech Conformer noncausal MHA with relative or rotary positional support.
- T2U encoder noncausal MHA and decoder causal/cross MHA.
- Eager source path uses explicit `torch.bmm`, softmax, dropout, and `torch.bmm`.

### Generation/cache ops

- `EncoderDecoderCache(DynamicCache, DynamicCache)` with self-cache append and cross-cache one-time update flag per layer.
- Decoder causal masks that account for `past_key_values`.
- Cross-attention K/V cache reuse after first generated token.
- Generation config language id lookups for text decoder, T2U decoder, and vocoder language embeddings.

### Preprocessing-coupled ops

- Kaldi-style fbank spectrogram with Povey window, preemphasis, mel filters, log floor, per-mel normalization.
- Tokenizer special-token processors:
  - source text: `[src_lang] tokens [eos]`.
  - target text: `[eos, tgt_lang] tokens [eos]`.
- Processor modality gate: text and audio are mutually exclusive.

## 5. Layer/block breakdown

Text encoder block, repeated `encoder_layers`:

```text
input ids -> scaled token embeddings + sinusoidal positions
residual = x
x = LayerNorm(x)
q,k,v = Linear(H -> H)(x), q *= D**-0.5
x = MHA(q,k,v, noncausal, additive padding mask)
x = residual + Dropout(Linear(H -> H)(x))
residual = x
x = LayerNorm(x)
x = Linear(H -> I) -> ReLU -> Dropout -> Linear(I -> H)
x = residual + Dropout(x)
```

Text decoder/T2U decoder block, repeated `decoder_layers`:

```text
input ids -> scaled token embeddings + sinusoidal positions(offset by cache len)
residual = x
x = LayerNorm(x)
q,k,v = Linear(H -> H)(x), self K/V update cache
x = causal MHA(q,k,v, self cache)
x = residual + output projection
residual = x
x = LayerNorm(x)
q = Linear(x); k,v = Linear(encoder states) or cached cross K/V
x = cross MHA(q,k,v, encoder mask)
x = residual + output projection
residual = x
x = LayerNorm(x)
x = Linear(H -> I) -> ReLU -> Linear(I -> H)
x = residual + x
```

Speech encoder:

```text
input_features: [B,S,160]
x = LayerNorm(160) -> Linear(160 -> 1024)
Conformer layer repeated N:
  x = x + 0.5 * FFN(LayerNorm(x), Swish)
  x = x + MHA(LayerNorm(x), relative/rotary positions)
  x = x + ConvModule(x)  # Conv1d/GLU/depthwise/BatchNorm/Swish
  x = LayerNorm(x + 0.5 * FFN(LayerNorm(x), Swish))
x = final LayerNorm(x)
x = x + 0.5 * ReLU-FFN(x)
if add_adapter:
  x = adapter Conv1d stride-8 subsample + MHA + ReLU-FFN
x = inner LayerNorm(x)
```

T2U and vocoder:

```text
t2u_input_embeds = text_decoder(sequences, encoder states).last_hidden_state
unit_ids = T2U seq2seq generate(inputs_embeds=t2u_input_embeds)
unit_ids = unit_ids after [eos, t2u_lang] prefix; eos -> pad; non-pad -= vocoder_offset
unit_emb = Embedding(unit_ids)                 # [B,U,1280]
log_dur = duration_predictor(unit_emb)         # [B,U]
dur = clamp(round(expm1(log_dur)), min=1)
expanded_units = repeat_interleave(unit_emb, dur)
inputs = concat(lang_embed, expanded_units, speaker_embed) over channels
waveform = HiFi-GAN Conv1d/ConvTranspose1d/resblocks/tanh
```

## 6. Attention requirements

| Site | Causal | Source states | Heads | KV heads | Head dim | Cache |
|---|---|---|---:|---:|---:|---|
| text encoder | no | text tokens | 16 | 16 | 64 | none |
| speech Conformer | no | speech tokens | 16 | 16 | 64 | none |
| speech adapter | no | subsampled speech tokens | 16 | 16 | 64 | none |
| text decoder self | yes | text decoder tokens | 16 | 16 | 64 | append K/V |
| text decoder cross | no | text/speech encoder states | 16 | 16 | 64 | reusable cross K/V |
| T2U encoder | no | decoder hidden embeddings | 16 | 16 | 64 | none |
| T2U decoder self/cross | yes/no | unit tokens / T2U encoder | 16 | 16 | 64 | append + reusable cross K/V |

All inspected configs use MHA, not GQA/MQA. Cache tensors are projected and stored as `[B, heads, seq, head_dim]`; the eager attention path reshapes them to `[B*heads, seq, head_dim]` for BMM. Cached keys/values are stored after linear projection and before the BMM reshape. Cross-attention cache sets `past_key_values.is_updated[layer_idx] = True` after the first update and then reuses projected encoder K/V.

Attention math order:

```text
q = q_proj(hidden_states) * head_dim**-0.5
k = k_proj(current_states)
v = v_proj(current_states)
scores = bmm(q, k^T)
scores += additive_mask if present
probs = softmax(scores, dim=-1)
probs = dropout(probs)  # inactive in eval
out = bmm(probs, v) -> out_proj
```

Speech Conformer relative attention differs: it adds Transformer-XL-style content and position scores, shifts the position term, then divides by `sqrt(head_dim)`.

FlashAttention/SDPA compatibility is straightforward for text/T2U attention if query scaling is folded or backend scaling is disabled. Speech Conformer relative attention needs a custom score-bias path or eager decomposition; rotary speech position mode can use normal RoPE-capable attention.

## 7. Position encoding and custom math

Text and T2U use sinusoidal embeddings with padding-aware positions:

```python
def seamless_positions(input_ids, padding_idx, past_len=0):
    mask = (input_ids != padding_idx).int()
    inc = (cumsum(mask, dim=1) + past_len) * mask
    return inc.long() + padding_idx
```

Speech relative position embeddings:

```python
def conformer_relative_positions(seq_len, hidden):
    pos = arange(seq_len)[:, None]
    div = exp(arange(0, hidden, 2) * -(log(10000.0) / hidden))
    pos_pos = interleave(sin(pos * div), cos(pos * div))
    pos_neg = interleave(sin(-pos * div), cos(-pos * div))
    return concat([flip(pos_pos, dim=0), pos_neg[1:]], dim=0)[None]
```

Speech relative attention shift:

```python
def relative_shift(scores_bd):
    zero = zeros(scores_bd.shape[:-1] + (1,))
    padded = concat([zero, scores_bd], dim=-1)
    padded = padded.view(scores_bd.shape[:2] + (scores_bd.shape[3] + 1, scores_bd.shape[2]))
    shifted = padded[:, :, 1:].view_as(scores_bd)
    return shifted[:, :, :, : shifted.size(-1) // 2 + 1]
```

Rotary mode, if enabled by config, applies RoPE to hidden states before Q/K projection, not to projected Q/K. Standard configs use `position_embeddings_type="relative"`.

Duration expansion custom math:

```python
dur = clamp(round(expm1(log_dur_pred)).long(), min=1)
expanded = repeat_interleave(unit_embeddings, dur, dim=time)
```

Relative position buffers and sinusoidal text position weights can be precomputed per max length/dtype/device, but text positions depend on padding masks and decode cache length. Duration expansion is value-dependent and produces variable waveform lengths.

## 8. Preprocessing and input packing

Audio CPU/data-pipeline contract:

- Raw speech is converted to float32 arrays; a 2D waveform uses the first channel.
- Sampling rate must match 16 kHz when supplied; no in-extractor resampling is performed.
- Waveform is squeezed and multiplied by `2**15` for Kaldi compliance.
- Spectrogram settings: Povey window length 400, frame_length 400, hop_length 160, fft_length 512, power 2.0, `center=False`, preemphasis 0.97, remove DC offset, Kaldi mel scale, 257 FFT frequency bins, 80 mel filters, min frequency 20, max frequency 8000, log mel floor `1.192092955078125e-07`.
- Feature extractor returns fbank features transposed to `[frames, mel]`, optionally normalized per mel bin as `(x - mean) / sqrt(var(ddof=1) + 1e-7)`.
- Padding defaults to true, `pad_to_multiple_of=2`, right padding, `padding_value=0.0`.
- Runtime model tensor is `input_features` shaped `[B, frames//2, 160]`; it is produced by reshaping adjacent pairs of 80-bin fbank frames.
- Audio attention mask is similarly reduced to `[B, frames//2]` by selecting positions where `indices % stride == 1`.

Text/tokenizer contract:

- Source text processor inserts a source language token before tokens and `eos` after tokens.
- Target text mode inserts `[eos, tgt_lang]` before target tokens and `eos` after tokens.
- Translation generation overrides decoder input ids with `text_decoder_lang_to_code_id[tgt_lang]`, one token per batch item.
- T2U generation starts with `[t2u_eos_token_id, t2u_lang_code_to_id[tgt_lang]]`.
- Speech synthesis also needs `vocoder_lang_code_to_id[tgt_lang]` and `spkr_id` for vocoder embeddings.

GPU/runtime split:

- First Dinoml integration should accept precomputed `input_ids`, `attention_mask`, `decoder_input_ids`, and/or `input_features` tensors. Feature extraction and tokenizer can remain CPU-side.
- End-to-end audio parity requires exact fbank and stride-packing behavior before the speech encoder.
- Speech output parity requires value-dependent unit duration expansion and variable-length waveform reporting; this should be staged as a separate runtime component from transformer decode.

## 9. Graph rewrite / lowering opportunities

### Rewrite: query-scale fold into Q projection

Source pattern:

```text
q = Linear(x, Wq, bq) * (head_dim ** -0.5)
scores = BMM(q, k^T)
```

Replacement:

```text
q = Linear(x, Wq * scale, bq * scale)
scores = BMM(q, k^T)
```

Preconditions:

- The scaled Q tensor has no observable consumer before attention.
- Backend attention does not apply another `1/sqrt(D)` scale.
- Applies to text/T2U attention and speech Conformer non-relative matmul branches.

Failure cases: debug hooks exposing raw Q, backend APIs with unavoidable scaling, or relative attention paths where fusion also has to preserve position terms.

Parity test sketch: one text decoder layer and one speech Conformer layer with fp32 random tensors before/after folding.

### Rewrite: text/T2U packed self-attention QKV

Source pattern:

```text
q = Linear(H -> H)(x)
k = Linear(H -> H)(x)
v = Linear(H -> H)(x)
```

Replacement:

```text
qkv = Linear(H -> 3H)(x)
split(qkv, [H,H,H])
```

Preconditions:

- Self-attention only; cross-attention K/V use encoder states.
- All three projections use bias in `SeamlessM4TAttention`.
- Query scale is either folded into Q rows or applied after split.

Weight transform:

```python
Wqkv = concat([Wq * scale, Wk, Wv], dim=0)
bqkv = concat([bq * scale, bk, bv], dim=0)
```

Failure cases: per-module weight loading/debug naming that requires separate parameters.

### Rewrite: cross-attention K/V precompute

Source pattern:

```text
for each decode token and layer:
  k = k_proj(encoder_hidden_states)
  v = v_proj(encoder_hidden_states)
```

Replacement: compute cross K/V once per layer after encoder or during prefill and keep `[B,heads,S,D]` cache.

Preconditions: encoder states and encoder mask are fixed for the request; beam reordering updates cache consistently.

Failure cases: speech generation code reruns encoder/text decoder for T2U staging; cache identity must be scoped to the exact stage.

### Rewrite: adapter Conv1d stride subsample to GEMM/im2col

Source pattern:

```text
transpose -> Conv1d(H -> 2H, kernel=8, stride=8, padding=4) -> GLU -> transpose
```

Replacement:

```text
WindowExtract1d -> MatMul(weight_flat.T) -> BiasAdd -> GLU
```

Preconditions:

- Dynamic shape guard uses PyTorch Conv1d output length formula.
- Padding, flatten order, and channel axis are source-faithful.
- Layout pass owns both transposes around the Conv1d.

Failure cases: partial NHWC/channel-last rewrite leaking into transformer `[B,T,H]` consumers.

### Rewrite: last-token LM projection

Source pattern: full `lm_head(decoder_hidden[:, :, :])`.

Replacement: project only `decoder_hidden[:, -1:, :]` in decode.

Preconditions: generation step only needs next-token logits and does not return full-sequence logits.

Failure cases: training/loss, score logging over all generated positions, and parity tests expecting full forward outputs.

### Rewrite: HiFi-GAN ConvTranspose1d provider path

Replacement: lower vocoder upsamplers to dedicated ConvTranspose1d/cuDNN-style kernels rather than generic scatter.

Preconditions: kernel/stride/padding values match config, output length formula is preserved, and residual Conv1d blocks remain in `[B,C,T]`.

Failure cases: value-dependent duration expansion creates unbounded time lengths; use bucketed max waveform lengths or a separate vocoder runtime path.

## 10. Kernel fusion candidates

Highest priority:

- Text decoder cached self-attention and cross-attention, reused for T2TT, S2TT, and the first stage of speech generation.
- T2U decoder cached self/cross-attention for speech output; max new unit tokens defaults to 1024.
- LayerNorm + QKV projection and FFN GEMMs in text/T2U blocks.
- Speech Conformer relative attention score path, including position projection and shift, if speech input is a first-class target.
- Last-token-only LM heads for both text vocab (~256K) and T2U vocab (10082).

Medium priority:

- Speech Conformer Conv1d/GLU/depthwise/BatchNorm/Swish block.
- Adapter stride-8 Conv1d + GLU subsampling and mask length recompute.
- Vocoder duration predictor Conv1d/LayerNorm stack.
- HiFi-GAN ConvTranspose1d + residual Conv1d blocks for end-to-end TTS/S2ST.

Lower priority:

- CPU/GPU fbank extraction; keep CPU baseline until transformer paths are stable.
- Beam selection hidden-state gather optimizations.
- Training-only loss and LayerDrop paths.

## 11. Runtime staging plan

Stage 1: config and tokenizer/processor metadata

- Parse `SeamlessM4TConfig`, preprocessor config, and generation language maps.
- Load shared text embeddings, text encoder/decoder, LM head, and preserve tied aliases.

Stage 2: text-to-text

- Implement text encoder, decoder prefill, cached decode, forced target-language decoder start, and text LM head.
- Stub tokenizer by accepting pretokenized `input_ids` and `attention_mask`.

Stage 3: speech-to-text

- Accept precomputed `input_features` `[B,S,160]` and `attention_mask` `[B,S]`.
- Implement speech feature projection, Conformer encoder, adapter subsampling, recomputed decoder cross mask, and text decoder reuse.

Stage 4: text/speech to units

- Implement T2U config rewrite, T2U encoder from text decoder hidden states, T2U cached decoder, T2U LM head, and unit-control-token postprocessing.

Stage 5: vocoder

- Implement duration predictor, duration expansion, language/speaker embeddings, HiFi-GAN, and output waveform length reporting.

Stage 6: end-to-end generation controller

- Add `format_speech_generation_kwargs`, target language validation, separate text/speech generation kwargs, `return_intermediate_token_ids`, and beam best-sequence selection.

Stage 7: optimized kernels and fusions

- Add attention backend dispatch, packed projections, cross K/V precompute, last-token logits, Conformer conv fusions, and vocoder provider kernels.

## 12. Parity and validation plan

- Config parity: instantiate medium and large configs, verify layer counts, vocab sizes, tied-weight names, T2U copied config fields, and vocoder dimensions.
- Feature extractor parity: compare fbank output, stride packing, and attention masks for mono and 2D stereo input. fp32 tolerance `atol=1e-5` for stored features if using identical CPU routines.
- Text encoder/decoder parity: one block then all blocks for medium-sized random inputs, with and without padding masks.
- Cached decode parity: prefill then one-token decode; check self-cache growth and cross-cache reuse flags.
- Speech encoder parity: feature projection, one Conformer layer, all speech layers, adapter output length, and recomputed mask.
- Relative attention parity: unit-test relative position table and shift logic against Transformers.
- T2U parity: hidden-state inputs into T2U encoder, T2U prefill/decode logits, unit id postprocessing.
- Vocoder parity: duration predictor values, rounded/clamped durations, repeated embeddings, waveform length formula, and waveform tensors.
- End-to-end smoke: T2TT, S2TT, T2ST, and S2ST greedy generation with fixed target language and speaker id.
- Recommended tolerances: fp32 hidden/logit parity `atol=1e-4, rtol=1e-4`; fp16/bf16 initial full-model tolerance `atol=2e-2, rtol=2e-2`; waveform tolerance likely looser and should be calibrated after conv/transposed-conv provider choice.

## 13. Performance probes

- CPU fbank throughput: audio seconds/sec, batch-size sweep, with/without per-mel normalization.
- Speech encoder throughput: `[B,S,160]` before and after adapter; sweep S and batch size.
- Text encoder throughput for medium and large sequence limits.
- Text decoder prefill throughput: prompt lengths 1, 16, 128, 512, 1024/4096 where supported.
- Text decode tokens/sec with cross-attention cache enabled.
- T2U decode unit tokens/sec up to `t2u_max_new_tokens=1024`.
- LM head cost: full-sequence versus last-token projection for 256K vocab.
- KV cache memory: text decoder and T2U decoder self/cross caches separately.
- Conformer relative attention backend comparison: eager BMM versus custom fused score path.
- Vocoder throughput: duration-expanded unit length sweep, ConvTranspose1d backend comparison, waveform samples/sec.
- End-to-end requests/hour split by T2TT, S2TT, T2ST, S2ST so vocoder cost does not hide transformer cost.

## 14. Skip/defer list

- Training losses, labels, gradient checkpointing, LayerDrop, dropout behavior.
- Checkpoint conversion from fairseq2.
- SeamlessM4T v2 source/config.
- Quantization and multi-GPU tensor parallelism.
- Beam search, sampling, constrained decoding, and speculative decoding beyond greedy first path.
- GPU fbank extraction until CPU preprocessing is measured as a bottleneck.
- Full batching of value-dependent vocoder duration expansion; first path can require batch size 1 or padded/bucketed expansion.
- Output attentions and hidden-state collection except where generation staging requires hidden states.

## 15. Final implementation checklist

- [ ] Parse `SeamlessM4TConfig`, preprocessor config, and generation config language maps.
- [ ] Load shared/tied text embeddings and LM head without breaking aliases.
- [ ] Implement text sinusoidal positions and padding-aware position ids.
- [ ] Implement text encoder/decoder MHA, masks, LayerNorm, ReLU FFN, and cache.
- [ ] Add target-language decoder start handling for text generation.
- [ ] Accept precomputed audio `input_features` `[B,S,160]` plus `[B,S]` mask.
- [ ] Implement speech feature projection and Conformer relative attention.
- [ ] Implement Conformer Conv1d/GLU/depthwise/BatchNorm/Swish block.
- [ ] Implement adapter stride-8 subsampling and post-adapter mask recompute.
- [ ] Add speech-to-text decoder cross-attention parity.
- [ ] Implement T2U config rewrite, model, tied LM head, and cached decode.
- [ ] Implement unit id postprocessing: prefix strip, eos-to-pad, `vocoder_offset` subtraction.
- [ ] Implement duration predictor and duration expansion.
- [ ] Implement HiFi-GAN Conv1d/ConvTranspose1d/resblock/tanh path and waveform length reporting.
- [ ] Add parity tests for feature extraction, text path, speech encoder, T2U, vocoder, and end-to-end generation.
- [ ] Benchmark preprocessing, encoders, prefill, decode, LM head, T2U, and vocoder separately.
