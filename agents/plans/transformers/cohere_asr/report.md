# Cohere ASR (`cohere_asr`) Transformers Audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: CohereLabs/cohere-transcribe-03-2026
Config source: official repo files returned 401 gated access; config facts below use source defaults plus open mirrors of the same base model.
Source files inspected:
- X:/H/transformers/src/transformers/models/cohere_asr/configuration_cohere_asr.py
- X:/H/transformers/src/transformers/models/cohere_asr/feature_extraction_cohere_asr.py
- X:/H/transformers/src/transformers/models/cohere_asr/processing_cohere_asr.py
- X:/H/transformers/src/transformers/models/cohere_asr/modeling_cohere_asr.py
- X:/H/transformers/src/transformers/models/cohere_asr/modular_cohere_asr.py
- X:/H/transformers/src/transformers/models/parakeet/configuration_parakeet.py
- X:/H/transformers/src/transformers/models/parakeet/modeling_parakeet.py
- X:/H/transformers/docs/source/en/model_doc/cohere_asr.md
Any missing files or assumptions: official raw config/preprocessor/processor/tokenizer/generation files for CohereLabs/cohere-transcribe-03-2026 require gated access. The Hub API reports `gated: auto`, public metadata, and siblings including `config.json`, `modeling_cohere_asr.py`, `processing_cohere_asr.py`, `tokenization_cohere_asr.py`, and `model.safetensors`.
```

Snapshots were saved under `agents/plans/transformers/cohere_asr/_sources/`, including exact Transformers source snapshots under `_sources/transformers_b75feb2/` and fetched Hub JSON/error records.

Gated primary links:

- [CohereLabs/cohere-transcribe-03-2026](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026)
- [official config.json](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026/resolve/main/config.json) returned HTTP 401 without access.
- [official preprocessor_config.json](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026/resolve/main/preprocessor_config.json) returned HTTP 401 without access.

Open representative configs inspected:

| Repo | Basis | Notes |
|---|---:|---|
| [onnx-community/cohere-transcribe-03-2026-ONNX](https://huggingface.co/onnx-community/cohere-transcribe-03-2026-ONNX) | config, preprocessor, processor, generation config | Open mirror/export tagged as quantized base of CohereLabs model. Best complete config basis. |
| [BarathwajAnandan/cohere-transcribe-fp8](https://huggingface.co/BarathwajAnandan/cohere-transcribe-fp8) | config, processor, tokenizer config | Same native `cohere_asr` dimensions plus FP8/export metadata. |
| [beshkenadze/cohere-transcribe-03-2026-mlx-8bit](https://huggingface.co/beshkenadze/cohere-transcribe-03-2026-mlx-8bit) | config, preprocessor, tokenizer config | MLX mirror with quantization metadata; config keeps historical/export fields but omits many native top-level dimensions. |
| [beshkenadze/cohere-transcribe-03-2026-mlx-4bit](https://huggingface.co/beshkenadze/cohere-transcribe-03-2026-mlx-4bit) | config, preprocessor, tokenizer config | Same operator shape as MLX 8-bit, different quantization. |
| [UsefulSensors/cohere_asr-tiny](https://huggingface.co/UsefulSensors/cohere_asr-tiny) | 401 | Example checkpoint referenced in source docstrings, unavailable without access in this environment. |

`modeling_cohere_asr.py` is generated from `modular_cohere_asr.py`; future Transformers source edits should be made in the modular file. The Cohere ASR encoder is not defined in the family itself: `CohereAsrModel` instantiates `AutoModel.from_config(config.encoder_config)`, which resolves to the Parakeet encoder.

## 2. High-level architecture

Primary DinoML target: automatic speech recognition with an audio encoder plus autoregressive text decoder, producing vocabulary logits for generation.

```text
raw mono waveform
  -> CPU/data-pipeline CohereAsrFeatureExtractor: chunk, pad, dither, preemphasis, STFT, mel, log, per-feature normalize
  -> input_features [batch_chunks, frames, 128] + attention_mask [batch_chunks, frames]
  -> Parakeet Fast Conformer encoder: 2D conv subsampling + 48 conformer blocks
  -> encoder hidden [batch_chunks, enc_time, 1280] + subsampled mask
  -> decoder encoder projection 1280 -> 1024
  -> token embedding + learned position embedding + 8 decoder layers with causal self-attn and encoder cross-attn
  -> Linear 1024 -> vocab_size logits
  -> generation + tokenizer decode + optional chunk reassembly
```

Stage decomposition:

| Stage | Runtime owner | Cacheable? | Notes |
|---|---|---:|---|
| Waveform chunking and log-mel extraction | CPU/data pipeline first | No, except saved features | Source can run STFT on `device`, but first DinoML target should treat this as preprocessing. |
| Parakeet encoder | GPU/runtime | Yes, encoder outputs can be reused for decoder generation | No autoregressive KV cache; full noncausal encoder. |
| Decoder prompt construction | CPU/tokenizer/processor | Yes | Processor creates `decoder_input_ids` from language/punctuation prompt. |
| Decoder prefill | GPU/runtime | Yes | Builds decoder self-attn KV and cross-attn KV cache. |
| Decoder token decode | GPU/runtime | Yes | Uses `EncoderDecoderCache`: self-attn cache grows, cross-attn cache is populated once per layer and reused. |
| Decode/reassembly | CPU/tokenizer/processor | No | `audio_chunk_index` is not consumed by model; used by processor decode. |

`ParakeetForCTC` exists in the nested source family, but Cohere ASR does not use the Parakeet CTC head. For this report, CTC is deferred except as context for shared encoder coverage.

## 3. Important config dimensions

Effective native dimensions from the ONNX mirror and matching FP8 mirror:

| Field | Cohere ASR decoder | Parakeet encoder |
|---|---:|---:|
| `model_type` | `cohere_asr` | `parakeet_encoder` |
| hidden size | 1024 | 1280 |
| layers | 8 | 48 |
| attention heads | 8 | 8 |
| KV heads | 8 | 8 |
| head dim | 128 | 160 inferred from `1280 / 8`; source config does not expose `head_dim` by default |
| intermediate size | 4096 | 5120 |
| activation | ReLU | SiLU |
| attention bias | true | true |
| conv bias | n/a | true |
| vocab size | 16384 | n/a |
| max position embeddings | 1024 learned decoder positions | 5000 encoder time positions |
| encoder mel bins | n/a | 128 |
| subsampling factor | n/a | 8 |
| subsampling conv channels | n/a | 256 |
| dropout fields | 0.0 in production config | 0.0 in production config |
| cache support | decoder `EncoderDecoderCache` | no KV cache |

Feature extractor / processor dimensions:

| Field | Value |
|---|---:|
| sampling rate | 16000 Hz |
| waveform channels | mono expected; multi-channel inputs are averaged to mono with warning |
| `feature_size` / mel bins | 128 |
| `n_fft` | 512 |
| `win_length` | 400 samples, Hann, `periodic=False` |
| `hop_length` | 160 samples |
| preemphasis | 0.97 |
| dither | deterministic, `1e-5`, seed per valid waveform length |
| log guard | `2**-24` before `torch.log` |
| normalization | per-feature mean/std over valid frames, padding zeroed |
| chunking | max 35 s, split long audio using a 5 s quiet-boundary search window |
| output | `input_features [batch_chunks, frames, 128]`, bool `attention_mask [batch_chunks, frames]` |

Representative checkpoint sweep:

| Repo | Access | Decoder dims | Encoder dims | Processor status | Operator-significant variation |
|---|---|---|---|---|---|
| CohereLabs/cohere-transcribe-03-2026 | gated raw files, public metadata | official raw config gated | official raw config gated | official raw files gated | Public API metadata says native Transformers `cohere_asr`, gated auto, Apache-2.0, ASR. |
| onnx-community/cohere-transcribe-03-2026-ONNX | open | 8x1024, 8 heads, vocab 16384 | 48x1280, 8 heads, 128 mel bins | complete | ONNX export variants include encoder/decoder ONNX files, q4/q4f16/quantized exports; native config unchanged. |
| BarathwajAnandan/cohere-transcribe-fp8 | open | same as ONNX mirror | same as ONNX mirror | processor config embedded; no standalone preprocessor | Adds FP8/export metadata, not read by native source. |
| beshkenadze/cohere-transcribe-03-2026-mlx-8bit | open | native dimensions largely omitted from top-level config | historical/export NeMo-style fields present | preprocessor config present | MLX quantization metadata is outside native Transformers runtime. |
| beshkenadze/cohere-transcribe-03-2026-mlx-4bit | open | same as MLX 8-bit | same as MLX 8-bit | preprocessor config present | Same graph shape; different external quantization. |

## 3a. Family variation traps

- `encoder_config` is a nested `parakeet_encoder`; DinoML must compose Parakeet Fast Conformer coverage, not treat Cohere ASR as a text-only seq2seq model.
- Decoder config allows `num_key_value_heads != num_attention_heads` and explicit `head_dim`; production configs use MHA (`8 == 8`) and `head_dim=128`.
- Encoder config sets `num_key_value_heads = num_attention_heads` in `ParakeetEncoderConfig.__post_init__`; no production GQA/MQA observed for encoder.
- Official primary raw files are gated. Mirrors may carry historical NeMo/export fields such as `encoder`, `transf_decoder`, `preprocessor`, `prompt_defaults`, `decoding`, and `quantization`. Native source reads only `CohereAsrConfig` fields plus `encoder_config`; route mirror-specific quantization/export fields to separate loading audits.
- `decoder_start_token_id` is `None` in native config mirrors but `generation_config.json` sets `decoder_start_token_id=13764`. The processor normally supplies a full language prompt as `decoder_input_ids`; first integration should require processor-provided decoder prompt or generation config.
- `CohereAsrForConditionalGeneration._tied_weights_keys` aliases `proj_out.weight` to `model.decoder.embed_tokens.weight`, but the module still declares `proj_out` with bias and `tie_word_embeddings=False`. Weight loading must preserve any actual checkpoint aliasing without assuming the bias is tied.
- Encoder attention has custom relative-position score bias (`matrix_bd`) added as the attention mask/bias to the backend. FlashAttention is disabled in Parakeet source because custom attention bias is not supported there; SDPA/flex/eager can be used if the bias is preserved.
- Layout-sensitive audio tensors are source-semantic `[B, T, F]` for features and `[B, C, T]` inside Conv1d modules. Initial lowering should use no layout translation for the feature/Conformer path; NHWC-like optimization would need guarded local regions only.
- Subsampling uses NCHW Conv2d on `[B, 1, T, F]`, masks time dimension after each Conv2d, then transposes/reshapes to `[B, T_sub, C * F_sub]`. Axis rewrites are unsafe unless the entire Conv2d stack, masking, transpose, flatten order, and linear consumer are rewritten together.
- Encoder Conformer Conv1d module uses `transpose(1,2)`, GLU on channel dim, depthwise Conv1d groups=`hidden_size`, BatchNorm1d, and mask fill derived from a `[B,1,T,T]` attention mask. Treat this region as axis-sensitive.
- Feature extraction includes stochastic-looking but deterministic dither seeded by waveform length; parity tests need dither enabled or explicitly set `dither=0`.

## 4. Operator coverage checklist

Tensor/layout ops:

- `unsqueeze`, `squeeze`, `transpose`, `permute`, `view`/`reshape`, `contiguous`
- `arange`, comparisons, boolean masks, `masked_fill`, mask expansion and transpose
- `cat`, `stack`, `expand`, `floor`, `floor_divide`, `sum`, `mean`, `sqrt`, `pow`
- `pad` for relative-position shift and waveform padding in preprocessing

Neural network primitives:

- Embedding(`vocab_size=16384`, `hidden=1024`) and decoder learned position embedding(`1024`, `1024`)
- Linear decoder: Q/O `1024 -> 1024`, K/V `1024 -> 1024`, MLP `1024 -> 4096 -> 1024`, encoder projection `1280 -> 1024`, LM head `1024 -> 16384`
- Linear encoder: Q/K/V/O `1280 -> 1280`, FFN `1280 -> 5120 -> 1280`, relative-k projection `1280 -> 1280`, subsampling linear `4096 -> 1280` for production `256 * (128 // 8)`
- LayerNorm on decoder and encoder hidden dimensions
- ReLU for decoder MLP and subsampling conv activations; SiLU for encoder FFN and Conformer conv activation
- Conv2d subsampling: first `1 -> 256`, kernel 3, stride 2, pad 1; then two depthwise `256 -> 256`, kernel 3, stride 2, pad 1, groups 256, each followed by pointwise `256 -> 256`, kernel 1, and ReLU
- Conv1d Conformer module per encoder layer: pointwise `1280 -> 2560`, GLU channel split, depthwise `1280 -> 1280`, kernel 9, groups 1280, pad 4, BatchNorm1d, SiLU, pointwise `1280 -> 1280`

Attention primitives:

- Decoder causal self-attention with cache and additive causal mask
- Decoder noncausal encoder cross-attention with reusable cross-attn K/V cache
- Encoder noncausal full self-attention with custom relative-position score bias and padding mask
- MHA production shape: 8 heads, decoder head dim 128, encoder head dim 160
- Softmax in fp32 then cast back to query dtype in eager path

Position/custom math:

- Decoder learned absolute position embedding
- Encoder relative sinusoidal positions over `2*T-1` positions, relative-k projection, global `bias_u`/`bias_v`, relative shift

Generation/cache ops:

- `EncoderDecoderCache(DynamicCache, DynamicCache)` with per-layer self-attn and cross-attn caches
- Cache reorder/reset support inherited from Transformers generation cache classes
- Processor prompt creation and `audio_chunk_index` passthrough ignored by `prepare_inputs_for_generation`

Preprocessing-coupled ops:

- Mono conversion, chunk splitting by low energy, padding
- Deterministic dither, preemphasis, `torch.stft`, magnitude squared, mel filter matmul, log, per-feature normalization
- Attention mask from frame lengths

## 5. Layer/block breakdown

Feature extractor:

```text
raw waveform [B or list, samples]
  -> optional average channels to mono
  -> energy chunking if duration > 30 s fast-path threshold
  -> pad to [B_chunks, samples]
  -> deterministic dither over valid samples
  -> preemphasis: x[t] = x[t] - 0.97 * x[t-1], padded samples zeroed
  -> STFT n_fft=512, win_length=400, hop=160, Hann periodic=False
  -> power magnitude [B, 257, frames]
  -> mel_filters[128,257] @ magnitudes
  -> log(mel + 2**-24)
  -> transpose to [B, frames, 128]
  -> normalize over valid frames and zero padding
```

Parakeet subsampling:

```text
input_features [B, T, 128]
  -> unsqueeze channel: [B, 1, T, 128]
  -> Conv2d k=3 s=2 p=1: [B, 256, ceil-ish(T/2), 64]
  -> ReLU
  -> depthwise Conv2d k=3 s=2 p=1 groups=256 -> pointwise Conv2d 1x1 -> ReLU
  -> depthwise Conv2d k=3 s=2 p=1 groups=256 -> pointwise Conv2d 1x1 -> ReLU
  -> transpose channel/time and flatten: [B, T_sub, 256 * 16]
  -> Linear(4096 -> 1280)
```

Parakeet encoder block, repeated 48 times:

```text
x = x + 0.5 * Linear(5120 -> 1280)(SiLU(Linear(1280 -> 5120)(LayerNorm(x))))
q,k,v = Linear(1280 -> 1280)(LayerNorm(x)), reshape to [B, 8, T, 160]
rel = Linear(1280 -> 1280)(relative_position_embeddings [B, 2*T-1, 1280])
matrix_bd = rel_shift((q + bias_v) @ rel^T)[..., :T] * (160 ** -0.5)
matrix_bd = masked_fill(~attention_mask, -inf)
attn = Attention(q + bias_u, k, v, additive_bias=matrix_bd, scale=160 ** -0.5)
x = x + Linear(1280 -> 1280)(attn)
conv = PointwiseConv1d(1280 -> 2560) -> GLU(dim=channel)
conv = mask padded time rows -> depthwise Conv1d(k=9, groups=1280) -> BatchNorm1d -> SiLU -> PointwiseConv1d(1280 -> 1280)
x = x + conv(LayerNorm(x))
x = x + 0.5 * Linear(5120 -> 1280)(SiLU(Linear(1280 -> 5120)(LayerNorm(x))))
x = LayerNorm(x)
```

Cohere decoder layer, repeated 8 times:

```text
x = token_embedding + learned_position_embedding
x = LayerNorm(x)
self_attn_in = LayerNorm(x)
q = Linear(1024 -> 1024), k/v = Linear(1024 -> 1024), reshape [B, 8, S, 128]
k/v = append/update self-attn cache when use_cache
x = x + causal_attention(q,k,v, mask)
cross_in = LayerNorm(x)
q = Linear(1024 -> 1024), encoder k/v = Linear(1024 -> 1024) over projected encoder states
encoder k/v = compute once and reuse cross-attn cache when use_cache
x = x + bidirectional_cross_attention(q,k,v, encoder_mask)
x = x + Linear(4096 -> 1024)(ReLU(Linear(1024 -> 4096)(LayerNorm(x))))
```

Head:

```text
decoder_last_hidden [B, S, 1024] -> Linear(1024 -> 16384, bias=True) -> logits
```

## 6. Attention requirements

Encoder attention:

- Noncausal self-attention only.
- Production MHA: 8 Q heads, 8 KV heads, head dim 160.
- Input/output hidden shape `[B, T_sub, 1280]`.
- Padding mask is built after subsampling as `[B, 1, T, T]` boolean.
- Custom relative-position bias `matrix_bd [B, heads, T, T]` is added to content scores through the attention interface. It is already scaled before attention; content scores are separately scaled inside eager attention.
- No KV cache. Encoder output can be cached as a whole for generation.
- Source advertises SDPA/flex support but disables FlashAttention because custom attention bias is not supported.

Decoder self-attention:

- Causal self-attention.
- Production MHA: 8 Q heads, 8 KV heads, head dim 128. Config supports GQA/MQA in principle.
- Cache tensor per layer before repeat expansion: key/value `[B, num_key_value_heads, seen_tokens, head_dim]`, production `[B, 8, S_seen, 128]`.
- No RoPE in generated Cohere ASR decoder despite comments inherited from modular code; positions are learned embeddings added before layers.
- Eager math: `matmul(q, k.T) * head_dim**-0.5`, add causal mask, `softmax(..., dtype=float32).to(query.dtype)`, dropout in training, matmul with V.

Decoder cross-attention:

- Noncausal attention from decoder queries to projected encoder states.
- Encoder hidden states are first projected `1280 -> 1024`.
- Cross K/V cache per layer stores projected encoder K/V `[B, 8, T_enc, 128]`. Source checks `past_key_values.is_updated[layer_idx]`; once populated, it reuses cross-attn K/V for subsequent decode tokens.
- Encoder attention mask is transformed by `create_bidirectional_mask` for cross-attention.

## 7. Position encoding and custom math

Decoder position encoding is a learned embedding table:

```python
past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
position_ids = torch.arange(seq_len, device=device) + past_seen_tokens
inputs_embeds = LayerNorm(token_embeds + pos_emb(position_ids))
```

Encoder relative positional encoding and shift:

```python
def parakeet_relative_positions(hidden_states):
    T = hidden_states.shape[1]
    position_ids = torch.arange(T - 1, -T, -1, device=hidden_states.device)
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, hidden_size, 2) / hidden_size))
    freqs = (inv_freq[None, :, None].float() @ position_ids[None, None, :].float()).transpose(1, 2)
    pos = torch.stack([freqs.sin(), freqs.cos()], dim=-1).reshape(hidden_states.shape[0], 2 * T - 1, hidden_size)
    return pos.to(hidden_states.dtype)

def rel_shift(scores):
    B, H, Q, P = scores.shape
    scores = pad(scores, pad=(1, 0))
    scores = scores.view(B, H, -1, Q)
    return scores[:, :, 1:].view(B, H, Q, P)
```

The inverse frequency buffer is static. The position tensor depends on current encoder sequence length after subsampling and batch size only for expansion. For production `T <= 5000`; source raises if exceeded.

## 8. Preprocessing and input packing

CPU/data-pipeline contract:

- Input audio must be sampled at 16 kHz. Missing `sampling_rate` logs a warning; mismatched sampling rate raises.
- Mono is expected. Batched tensors/lists with extra channel dimension are averaged over the last dimension with a warning.
- Audio longer than `max_audio_clip_s - overlap_chunk_second` (30 s with defaults) is split into chunks by searching for a low-energy boundary in the last 5 s of the chunk.
- Output batch dimension is chunk batch, not original sample batch. `audio_chunk_index` maps chunk outputs back to original samples and is used only by `processor.decode`.
- `input_features` are log-mel tensors `[B_chunks, frames, 128]`; `attention_mask` is `[B_chunks, frames]` with true/1 for valid frames.

GPU/runtime inputs for first integration:

- Accept precomputed `input_features` and `attention_mask` from the processor. Do not run STFT in DinoML initially.
- Accept `decoder_input_ids` from the processor. The prompt is language dependent and punctuation dependent:

```text
["▁", "<|startofcontext|>", "<|startoftranscript|>", "<|emo:undefined|>",
 "<|{language}|>", "<|{language}|>", "<|pnc| or <|nopnc|>",
 "<|noitn|>", "<|notimestamp|>", "<|nodiarize|>"]
```

- Supported languages in processor source: `ar`, `de`, `el`, `en`, `es`, `fr`, `it`, `ja`, `ko`, `nl`, `pl`, `pt`, `vi`, `zh`.
- Postprocessing uses tokenizer decode. When chunk reassembly is requested, Japanese and Chinese join chunks with empty separator; other languages join with a space.
- No timestamp, no diarization, no no-speech threshold, and no CTC decoder are implemented in the native `CohereAsrForConditionalGeneration` graph. Prompt tokens request `<|notimestamp|>` and `<|nodiarize|>`.

## 9. Graph rewrite / lowering opportunities

### Rewrite: decoder QKV sibling linears

Source pattern:

```text
q = Linear(x, Wq, bq)
k = Linear(x, Wk, bk)
v = Linear(x, Wv, bv)
```

Replacement:

```text
PackedLinear(x, concat_rows(Wq, Wk, Wv), concat(bq, bk, bv)) -> split [Q, K, V]
```

Preconditions: same input tensor, same dtype/device, biases all present or all representable, `num_key_value_heads` and `head_dim` determine split sizes exactly. Production split widths are `[1024, 1024, 1024]`. Failure cases: config GQA/MQA changes K/V widths; preserve split order Q,K,V.

### Rewrite: encoder subsampling Conv2d stack to layout-guarded conv/GEMM

Source pattern: NCHW Conv2d stack over `[B,1,T,F]`, with time masks after each Conv2d, then `transpose(1,2).reshape(B,T_sub, C*F_sub)` and Linear.

Replacement: only within a guarded region, use optimized Conv2d or im2col/GEMM kernels and fuse pointwise 1x1 where profitable.

Preconditions: kernel 3, stride 2, padding 1, dilation 1, static mel bins divisible through three stride-2 stages (`128 -> 64 -> 32 -> 16`), depthwise groups exactly 256 for the latter Conv2d layers, source flatten order preserved. Layout constraints: semantic source is NCHW; any NHWC/channel-last transform must also rewrite mask axis, transpose, flatten, and linear weight interpretation. Failure cases: dynamic feature size, non-power-of-two `subsampling_factor`, nonstandard `subsampling_conv_stride`, or consumers expecting source strides.

### Rewrite: Conv1d pointwise to Linear on `[B,T,C]`

Source pattern: transpose `[B,T,C] -> [B,C,T]`, Conv1d kernel 1, transpose back.

Replacement:

```text
Linear(C -> C_out) over last dimension
```

Preconditions: kernel size 1, stride 1, padding 0, groups 1, no intervening layout-sensitive op. Applies to Conformer pointwise Conv1d modules and subsampling pointwise Conv2d 1x1 only with separate 2D layout handling. Failure cases: depthwise Conv1d kernel 9 cannot be rewritten to Linear without local-window gather/im2col.

### Rewrite: Conformer FFN epilogue fusion

Source pattern:

```text
x + 0.5 * Linear2(SiLU(Linear1(LayerNorm(x))))
```

Replacement: fuse LayerNorm, first GEMM epilogue, SiLU, second GEMM epilogue, scalar residual add where backend supports it.

Preconditions: inference mode dropout=0, scalar factor exactly 0.5, no captured hidden-state output between pieces. Failure cases: training/dropout, output attentions/hidden states requiring intermediate tensors.

### Rewrite: last-token logits

Source pattern: full `Linear(1024 -> 16384)` over every decoder time step during decode.

Replacement: for token-by-token generation after prefill, run LM head only for final token `[B,1,1024]`.

Preconditions: generation only needs next-token logits; no caller requests full sequence logits. Failure cases: teacher-forced forward, scoring, or tests comparing all logits.

## 10. Kernel fusion candidates

Highest priority:

- Encoder custom relative-position attention: content attention plus additive `matrix_bd` bias dominates 48-layer encoder cost and blocks naive FlashAttention.
- Decoder cached self-attention and cross-attention: needed for practical generation; cross-attn K/V should be computed once.
- LayerNorm + Linear patterns in encoder/decoder: many pre-norm blocks.
- Conformer Conv1d module: pointwise GLU + depthwise Conv1d + BatchNorm + SiLU + pointwise Conv1d occurs 48 times.
- Subsampling Conv2d stack: front-end cost is fixed per audio chunk and shape-sensitive.

Medium priority:

- FFN SiLU/ReLU activation + GEMM epilogues, including 0.5 residual scaling in encoder FFNs.
- Packed QKV projections for decoder and encoder.
- Encoder projection `1280 -> 1024` fused with cross-attn K/V projection if encoder states are used only by decoder cross-attention.
- Last-token-only LM head for decode.

Lower priority:

- Processor-side STFT/mel GPU acceleration; useful later but not needed for graph parity if preprocessing stays outside DinoML.
- Chunk reassembly and tokenizer decode acceleration; CPU control logic is small.
- External quantization formats from MLX/FP8/ONNX mirrors; separate loading/provider effort.

## 11. Runtime staging plan

1. Parse `CohereAsrConfig` and nested `ParakeetEncoderConfig`; reject unsupported mirror-only configs that omit native dimensions unless defaults can be reconstructed exactly.
2. Load dense weights and preserve any embedding/LM-head aliasing plus LM-head bias.
3. Accept processor-produced `input_features`, `attention_mask`, and `decoder_input_ids`; stub tokenizer/audio preprocessing outside compiled runtime.
4. Implement Parakeet subsampling and one encoder block parity, including relative-position bias.
5. Run full encoder parity and validate returned subsampled `attention_mask`.
6. Implement decoder prefill without cache first, then `EncoderDecoderCache` self/cross cache ABI.
7. Add generation loop parity using processor prompt and generation config `decoder_start_token_id` fallback.
8. Add attention/GEMM/conv fusions after baseline parity.
9. Add optional processor GPU path or direct waveform frontend only after model graph is stable.

## 12. Parity and validation plan

- Feature extractor parity: compare processor output for short, padded, stereo-to-mono, and >35 s chunked audio. Use `dither=0` for deterministic numerical tests, then one dither-enabled golden test.
- Subsampling parity: random `[B,T,128]` with masks; compare hidden states and output lengths after each Conv2d layer and final linear.
- Relative position parity: test `T=1`, small `T`, and max-ish `T`; compare relative embeddings, `matrix_bd`, `_rel_shift`, and masked fill behavior.
- Single encoder block parity in fp32 with dropout disabled. Suggested tolerance: fp32 `1e-4` absolute/relative; fp16/bf16 looser around attention and BatchNorm.
- Full encoder parity on short audio features and variable-length batch masks.
- Decoder prefill parity: compare logits for prompt ids with encoder outputs, no cache.
- Decode parity: compare one-step and multi-step logits with `use_cache=True`; assert self cache grows and cross cache does not recompute after first update.
- End-to-end ASR smoke: processor -> generate -> decode for a short public audio sample once gated weights are available.

Do not use DinoML tests for this audit; this is a planning report only.

## 13. Performance probes

- Processor throughput: waveform seconds/sec, split short vs long chunked audio.
- STFT/mel throughput if considering GPU preprocessing.
- Encoder subsampling time by `T` and batch chunks.
- Encoder block time split: FFN, custom attention/bias, Conv1d module.
- Encoder full throughput across chunk length and batch-size sweeps.
- Decoder prefill latency for prompt length and encoder length.
- Decode tokens/sec with self/cross cache, batch-size sweep.
- KV cache memory: decoder self cache grows with text length; cross cache fixed at encoder length.
- LM head cost for full sequence vs last-token-only.
- Attention backend comparison: eager vs SDPA/flex for encoder custom bias and decoder cached attention.
- Layout experiment: source NCHW/NCL vs guarded channel-last conv regions, with parity checks on mask and flatten order.

## 14. Skip/defer list

- Training, losses, gradient checkpointing, LayerDrop/dropout behavior.
- Parakeet CTC head and CTC decoding; not used by Cohere ASR generation path.
- Beam search beyond greedy/beam-size-1 first parity, despite mirror config saying strategy `beam`.
- Timestamp, diarization, no-speech probability, and word-level probabilities; native prompt requests no timestamp/diarization and source does not implement Whisper-style processors.
- Remote-code files from the gated official repo until access is granted; native library source at pinned commit is the report basis.
- Mirror-specific FP8/MLX/ONNX quantization formats.
- Running feature extraction inside DinoML GPU runtime.
- Global NHWC/channel-last translation for audio tensors.

## 15. Final implementation checklist

- [ ] Parse `CohereAsrConfig` and nested `ParakeetEncoderConfig`.
- [ ] Admit processor-produced `input_features [B,T,128]`, `attention_mask [B,T]`, and `decoder_input_ids`.
- [ ] Implement Parakeet subsampling Conv2d stack with source NCHW semantics.
- [ ] Implement Parakeet relative positional encoding and `_rel_shift`.
- [ ] Implement encoder custom relative-position attention bias.
- [ ] Implement Conformer FFN, LayerNorm, Conv1d GLU/depthwise/BatchNorm/SiLU module.
- [ ] Implement decoder learned position embedding and LayerNorm embedding path.
- [ ] Implement decoder causal self-attention and encoder cross-attention.
- [ ] Define `EncoderDecoderCache` ABI for decoder self K/V and cross K/V.
- [ ] Implement LM head `Linear(1024 -> 16384)` with bias and alias-aware weight loading.
- [ ] Add no-layout-translation guards around feature, subsampling, and Conv1d regions.
- [ ] Add parity tests for feature extraction, subsampling, relative attention, one encoder block, full encoder, decoder prefill, and cached decode.
- [ ] Benchmark processor, encoder, prefill, decode, LM head, and cache memory separately.
