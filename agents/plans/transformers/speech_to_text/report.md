# Transformers audit: speech_to_text

## 1. Source basis

Transformers commit/version: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`.

Model id: representative checkpoints from `facebook/s2t-*`; primary first target should be `facebook/s2t-small-librispeech-asr`.

Config source: downloaded HF `config.json`, `preprocessor_config.json`, `tokenizer_config.json`, and `generation_config.json` snapshots under this folder for:

- `facebook/s2t-small-librispeech-asr`
- `facebook/s2t-medium-librispeech-asr`
- `facebook/s2t-large-librispeech-asr`
- `facebook/s2t-small-mustc-en-es-st`
- `facebook/s2t-medium-mustc-multilingual-st`
- `facebook/s2t-small-covost2-fr-en-st`

Source files inspected:

- `transformers/src/transformers/models/speech_to_text/modeling_speech_to_text.py`
- `transformers/src/transformers/models/speech_to_text/configuration_speech_to_text.py`
- `transformers/src/transformers/models/speech_to_text/feature_extraction_speech_to_text.py`
- `transformers/src/transformers/models/speech_to_text/processing_speech_to_text.py`
- `transformers/src/transformers/models/speech_to_text/tokenization_speech_to_text.py`
- `transformers/src/transformers/masking_utils.py`
- `transformers/src/transformers/cache_utils.py`
- `transformers/src/transformers/generation/utils.py`

Pinned source URLs:

- `https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/speech_to_text/modeling_speech_to_text.py`
- `https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/speech_to_text/configuration_speech_to_text.py`
- `https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/speech_to_text/feature_extraction_speech_to_text.py`
- `https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/speech_to_text/processing_speech_to_text.py`
- `https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/speech_to_text/tokenization_speech_to_text.py`

Representative config URLs:

- `https://huggingface.co/facebook/s2t-small-librispeech-asr/tree/main`
- `https://huggingface.co/facebook/s2t-medium-librispeech-asr/tree/main`
- `https://huggingface.co/facebook/s2t-large-librispeech-asr/tree/main`
- `https://huggingface.co/facebook/s2t-small-mustc-en-es-st/tree/main`
- `https://huggingface.co/facebook/s2t-medium-mustc-multilingual-st/tree/main`
- `https://huggingface.co/facebook/s2t-small-covost2-fr-en-st/tree/main`

Any missing files or assumptions: no gated/401 gaps for the representative metadata fetched. Weight files were not downloaded. Report scope is native in-library `Speech2TextForConditionalGeneration`; remote-code or `SpeechEncoderDecoderModel` compositions are out of scope.

## 2. High-level architecture

Speech2Text is an audio encoder + autoregressive text decoder:

```text
raw mono waveform -> CPU fbank + CMVN -> [B,T,80] input_features + mask
  -> stride-2 Conv1d+GLU subsampler -> sinusoidal source positions
  -> Transformer encoder
  -> Transformer decoder self-attn + encoder cross-attn
  -> tied/untied LM head -> logits -> generation controller/tokenizer decode
```

Stage decomposition:

- CPU/data pipeline: mono waveform validation, Kaldi-style log-mel fbank, padding/truncation, utterance CMVN, `attention_mask`.
- GPU/runtime encoder: `input_features [B,T,F]`, temporal Conv1d subsampler, encoder MHA/FFN stack.
- Cacheable encoder output: `encoder_last_hidden_state [B,T_enc,d_model]` plus downsampled encoder mask.
- Decoder prefill/decode: token embedding, sinusoidal target positions, causal self-attention KV cache, cross-attention cache over encoder states, LM projection.
- Generation controller: decoder start token, EOS/PAD handling, optional beam search, tokenizer language prefix for multilingual checkpoints.

Independently stageable pieces are fbank/CMVN parity, conv subsampler parity, one encoder layer, full encoder, one decoder layer with cross-attention, and decode with `EncoderDecoderCache`.

## 3. Important config dimensions

Effective source defaults from `Speech2TextConfig`: `d_model=256`, `encoder_layers=12`, `decoder_layers=6`, `encoder_attention_heads=4`, `decoder_attention_heads=4`, `encoder_ffn_dim=2048`, `decoder_ffn_dim=2048`, `activation_function="relu"`, `num_conv_layers=2`, `conv_kernel_sizes=(5,5)`, `conv_channels=1024`, `input_feat_per_channel=80`, `input_channels=1`, `max_source_positions=6000`, `max_target_positions=1024`, `use_cache=True`, `scale_embedding=True`, `tie_word_embeddings=True`.

| checkpoint | task/variant | d_model | enc/dec layers | heads | head_dim | FFN | vocab | conv | dropout family | preprocessor |
| --- | --- | ---: | --- | --- | ---: | --- | ---: | --- | --- | --- |
| `facebook/s2t-small-librispeech-asr` | ASR small | 256 | 12/6 | 4/4 | 64 | 2048/2048 | 10000 | 2x k5, 1024 mid | 0.1 | 16 kHz, 80 mel, CMVN |
| `facebook/s2t-medium-librispeech-asr` | ASR medium | 512 | 12/6 | 8/8 | 64 | 2048/2048 | 10000 | 2x k5, 1024 mid | 0.15 | 16 kHz, 80 mel, CMVN |
| `facebook/s2t-large-librispeech-asr` | ASR large | 1024 | 12/6 | 16/16 | 64 | 4096/4096 | 10000 | 2x k5, 1024 mid | 0.2 | 16 kHz, 80 mel, CMVN |
| `facebook/s2t-small-mustc-en-es-st` | speech translation | 256 | 12/6 | 4/4 | 64 | 2048/2048 | 10000 | 2x k5, 1024 mid | 0.1 | 16 kHz, 80 mel, CMVN |
| `facebook/s2t-medium-mustc-multilingual-st` | multilingual ST | 512 | 12/6 | 8/8 | 64 | 2048/2048 | 10000 | 2x k5, 1024 mid | 0.15 | 16 kHz, 80 mel, CMVN |
| `facebook/s2t-small-covost2-fr-en-st` | CoVoST ST | 256 | 12/6 | 4/4 | 64 | 2048/2048 | 10000 | 2x k5, 1024 mid | 0.1 | 48 kHz, 80 mel, CMVN |

Generation snapshots: `bos=0`, `pad=1`, `eos=2`, `decoder_start_token_id=2`, `max_length=200` for all fetched checkpoints. Older `config.json` files also carry `num_beams=5` and `early_stopping=true`; current generation behavior should be taken from `generation_config.json` where present.

## 3a. Family variation traps

- `head_dim` is `d_model / heads` and is 64 in the sampled official configs, but source rejects configs where `d_model` is not divisible by attention heads.
- No GQA/MQA: K/V heads equal query heads. Do not introduce repeat-kv logic.
- Attention projection modules are four independent biased `Linear(d_model -> d_model)` layers; no packed QKV storage.
- `tie_word_embeddings` defaults to true, and the LM head weight is tied to `model.decoder.embed_tokens.weight`. The multilingual MuST-C medium config explicitly sets `tie_word_embeddings=false`; preserve this aliasing difference.
- Source `_supports_flash_attn`, `_supports_sdpa`, and `_supports_flex_attn` are false. Even though attention calls `ALL_ATTENTION_FUNCTIONS`, initial DinoML parity should assume eager dense attention unless a later audit proves backend dispatch is enabled safely.
- Tokenizer language control is outside the neural graph. Multilingual MuST-C uses `lang_codes="mustc"`; tokenizer prepends a target-language token before text and appends EOS.
- CoVoST sample uses `sampling_rate=48000`, unlike the 16 kHz LibriSpeech/MuST-C samples. The feature extractor rejects mismatched explicit sampling rates.
- `classifier_dropout`, `gradient_checkpointing`, old `num_hidden_layers`, and some old generation fields appear in checkpoint configs but are not runtime neural graph operators in the inspected source.
- Source layout is `[B,T,F]` for audio features, then explicit transpose to `[B,F,T]` for Conv1d and back to `[B,T,d]`. Treat any channel-last/NHWC-style layout pass as a local conv-subsampler optimization only.

## 4. Operator coverage checklist

Tensor/layout ops:

- `transpose(1,2)`, `contiguous`, `view/reshape`, `index_select`, `arange`, `cumsum`, `flip`, equality/inequality masks, long/int casts.
- Mask construction: causal decoder mask and bidirectional encoder/cross masks with large negative additive values.
- Downsampled mask generation: length sum, repeated `(L - 1) // 2 + 1`, indexed scatter of a sentinel one, reverse cumulative sum.

Neural primitives:

- Conv1d temporal subsampler with stride 2, padding `k//2`, kernel usually 5.
- GLU over channel dimension: split output channels into halves, `a * sigmoid(b)`.
- Embedding lookup and optional `sqrt(d_model)` scaling.
- Sinusoidal positional embedding table, dynamic extension, `index_select`.
- LayerNorm, residual add, dropout disabled at inference.
- Linear/GEMM with bias for Q/K/V/O and FFN; LM head `Linear(d_model -> vocab, bias=False)`.
- ReLU FFN activation for sampled configs; source supports any `ACT2FN[activation_function]`.
- fp16 encoder clamp after each encoder layer: clamp to `[-finfo(fp16).max+1000, +finfo(fp16).max-1000]`.

Attention primitives:

- Encoder noncausal self-attention.
- Decoder causal self-attention with autoregressive KV cache.
- Decoder cross-attention over encoder outputs with cacheable cross K/V.
- Dense MHA only; no RoPE, ALiBi, relative bias, sliding window, packed varlen, GQA, or MQA.

Preprocessing-coupled ops:

- CPU fbank: waveform scaled by `2**15`; torchaudio `kaldi.fbank` when available.
- Numpy fallback: 400-sample frame, 160-hop, 512 FFT, Povey window, preemphasis 0.97, no centering, Kaldi mel scale, log mel floor `1.192092955078125e-07`, remove DC offset.
- Utterance-level CMVN over valid frames, then reset padded frames to `padding_value`.

Generation/cache ops:

- `EncoderDecoderCache(DynamicCache, DynamicCache)`.
- Cache reorder for beams via generic `Cache.reorder_cache`.
- Decoder start token and EOS/PAD control from generation config.

## 5. Layer/block breakdown

Conv subsampler:

```text
input_features: [B,T,input_feat_per_channel*input_channels]
x = transpose -> [B,F,T]
for i in num_conv_layers:
  y = Conv1d(in_ch, out_ch, kernel=k_i, stride=2, padding=k_i//2)
  x = GLU(y, dim=1)
x = transpose -> [B,T_enc,d_model]
T_enc = ceil(T / 2**num_conv_layers) for k=5,padding=2
```

For the common 2-layer configs: first conv is `80 -> 1024`, GLU to 512 channels; second conv is `512 -> 2*d_model`, GLU to `d_model`.

Encoder block, repeated `encoder_layers`:

```text
res = x
x = LayerNorm(x)
q,k,v = Linear(d -> d) with bias, reshape [B,T,H,64] -> [B,H,T,64]
x = dense noncausal self-attn(q,k,v, encoder_mask)
x = res + Linear(d -> d, bias)(x)
res = x
x = LayerNorm(x)
x = Linear(d -> encoder_ffn_dim, bias) -> ReLU -> Linear(encoder_ffn_dim -> d, bias)
x = res + x
if fp16: clamp finite range
```

Decoder block, repeated `decoder_layers`:

```text
res = x
x = LayerNorm(x)
x = causal self-attn(x, self_cache)
x = res + out_proj(x)
res = x
x = LayerNorm(x)
x = cross-attn(query=x, key/value=encoder_hidden_states, cross_cache)
x = res + out_proj(x)
res = x
x = LayerNorm(x)
x = Linear(d -> decoder_ffn_dim) -> ReLU -> Linear(decoder_ffn_dim -> d)
x = res + x
```

LM head:

```text
logits = Linear(d_model -> vocab_size, bias=False)(decoder_last_hidden_state)
```

## 6. Attention requirements

Encoder attention is bidirectional self-attention with shape:

```text
hidden: [B,S_enc,d]
q/k/v: [B,H,S_enc,head_dim]
scores: [B,H,S_enc,S_enc]
mask: broadcast additive mask, padding positions very negative
```

Decoder self-attention is causal MHA:

```text
hidden: [B,S_dec,d]
q/k/v: [B,H,S_dec,head_dim]
self cache per layer before attention backend: key/value [B,H,S_total,head_dim]
```

Decoder cross-attention is rectangular MHA:

```text
query: [B,H,S_dec,head_dim]
key/value source: encoder_hidden_states [B,S_enc,d]
cross cache per layer: key/value [B,H,S_enc,head_dim]
scores: [B,H,S_dec,S_enc]
```

Cache details:

- Cached self-attention K/V are stored after linear projection and head transpose; there is no position encoding applied to K/V beyond token position embedding already added to hidden states.
- Cross-attention K/V are computed from encoder hidden states, stored once per layer, and reused when `EncoderDecoderCache.is_updated[layer_idx]` is true.
- Beam search requires reordering both self and cross caches along batch/beam dimension.

FlashAttention/SDPA compatibility: source-level support flags are false. A future optimized lowering can still map these masks to dense attention kernels, but the admission guard should require parity for additive masks, cross-attention rectangular shapes, and cache update order.

## 7. Position encoding and custom math

No RoPE, ALiBi, or learned absolute table parameters. Source uses sinusoidal buffers with offset 2 and padding-aware position IDs.

```python
def speech2text_sinusoidal(num_embeddings, dim, padding_idx):
    half = dim // 2
    freq = exp(arange(half) * -(log(10000) / (half - 1)))
    emb = arange(num_embeddings)[:, None] * freq[None, :]
    emb = concat([sin(emb), cos(emb)], axis=1)
    if dim % 2 == 1:
        emb = concat([emb, zeros([num_embeddings, 1])], axis=1)
    emb[padding_idx] = 0
    return emb

def position_ids_from_input_ids(input_ids, padding_idx, past_len):
    mask = (input_ids != padding_idx).int()
    return ((cumsum(mask, dim=1) + past_len) * mask).long() + padding_idx
```

Encoder passes a synthetic padding mask into the same function: non-padding conv frames become increasing positions, padded frames stay at `padding_idx`.

## 8. Preprocessing and input packing

Raw waveform ABI:

- Input must be mono, one float sample per timestep. Batched numpy arrays may be rank 2; rank >2 is rejected.
- Explicit `sampling_rate` must match the feature extractor config.
- Feature extraction always wraps a single sample into a batch.

Feature tensor ABI:

- Model input is `input_features [B,T,80]`, `float32` from processor unless caller bypasses preprocessing.
- `attention_mask [B,T]`, `int32`/long-like, marks valid fbank frames before conv subsampling.
- Batched inference should pass `attention_mask`; source warns about subtle bugs otherwise.

CPU/data pipeline ownership:

- First DinoML target should accept precomputed `input_features` and `attention_mask`.
- End-to-end parity later can compose a CPU feature extractor. GPU fbank is optional and should be treated as a separate audio preprocessing provider.

Tokenizer/generation ABI:

- SentencePiece BPE with `vocab.json`.
- ASR tokenizers sampled use `do_lower_case=True`, no language codes.
- MuST-C multilingual tokenizer uses `lang_codes="mustc"` and prepends `<lang:xx>` as a prefix token before target text. This is tokenizer/controller ABI, not a model graph op.

## 9. Graph rewrite / lowering opportunities

### Rewrite: Conv1d GLU subsampler to temporal im2col GEMM

Source pattern: `transpose [B,T,F] -> [B,F,T]`, `Conv1d(stride=2,pad=k//2)`, `GLU(dim=1)`, repeated, then transpose back.

Replacement:

```text
PadTemporal -> WindowGather/Im2Col [B,T_out,F*k] -> GEMM(weight.T) + bias
  -> split channel -> sigmoid -> mul
```

Preconditions:

- `groups == 1`, dilation 1, stride 2, padding `kernel//2`.
- Input is dense contiguous after the source transpose or layout pass owns the conv region.
- GLU split dimension is the conv output channel dimension and output channels are even.

Shape equations:

- `T_out = floor((T + 2*pad - (k-1) - 1) / 2 + 1)`.
- For sampled `k=5,pad=2`, `T_out = ceil(T/2)`.

Failure cases: nonstandard kernel/padding/dilation/group, caller-provided strided features, or a layout pass that cannot rewrite both surrounding transposes.

Parity sketch: compare conv+GLU after one and two layers for random `[B,T,80]`, odd/even `T`, and valid padding masks.

### Rewrite: Separate Q/K/V projections to packed GEMM

Source pattern: independent `q_proj`, `k_proj`, `v_proj` with weights `[d,d]` and bias `[d]`.

Replacement: one `Linear(d -> 3d)` then split in Q,K,V order.

Preconditions:

- Self-attention only, same input hidden for all three projections.
- Bias presence matches all three modules.
- Weight packing order is `[q_proj; k_proj; v_proj]` in output rows for PyTorch linear convention.

Failure cases: cross-attention, where Q source is decoder hidden and K/V source is encoder hidden; packed K/V only is a separate rewrite.

### Rewrite: Cross-attention cached K/V projection hoist

Source pattern: every decode step calls cross-attention, but K/V are reused after first cache update.

Replacement: precompute per-layer encoder K/V immediately after encoder or at decoder prefill, store as cross cache.

Preconditions:

- Encoder hidden states are fixed for the request.
- Beam expansion/reorder semantics for cross cache are implemented.

Parity sketch: compare logits for first-token decode and second-token decode with and without hoisted cross K/V.

### Rewrite: last-token-only decode logits

Source pattern: LM head projects all decoder time positions.

Replacement: during incremental decode, project only `decoder_last_hidden_state[:, -1:, :]`.

Preconditions: generation controller only consumes next-token logits. Not valid for teacher-forced full-sequence scoring.

### Layout guidance

Keep semantic graph in source axes first. A local conv optimization may avoid physical `[B,T,F] <-> [B,F,T]` transposes by choosing a temporal-conv kernel that reads feature-last input. Protect attention/FFN blocks with no-layout-translation guards unless all reshape/view/head axes are rewritten together.

## 10. Kernel fusion candidates

Highest priority:

- Conv1d+GLU subsampler: early audio bottleneck and awkward transpose-heavy layout.
- LayerNorm + QKV projection for encoder/decoder self-attention.
- Dense MHA prefill and decode with KV cache, including rectangular cross-attention.
- FFN `Linear -> ReLU -> Linear` with residual add where epilogue support exists.
- Cross-attention K/V precompute/cache to avoid repeated encoder projections.

Medium priority:

- Sinusoidal position table/indexing folded into embedding/hidden add.
- Downsampled attention-mask construction as a small shape/mask kernel.
- Tied embedding/LM-head weight alias preservation through loading and lowering.
- Last-token-only LM head for decode.

Lower priority:

- CPU/GPU fbank provider. It is required for end-to-end product parity but separable from neural graph parity.
- Beam-search cache reorder optimization.
- fp16 clamp fusion in encoder residual epilogue.

## 11. Runtime staging plan

Stage 1: parse config, load weights, accept precomputed `input_features` and `attention_mask`.

Stage 2: implement Conv1d+GLU subsampler and downsampled mask parity.

Stage 3: run encoder-only parity for small ASR with eager dense attention.

Stage 4: run decoder teacher-forcing parity with encoder outputs, no cache.

Stage 5: implement `EncoderDecoderCache` ABI for self-attention and cross-attention, then validate incremental decode.

Stage 6: add generation-controller parity for `decoder_start_token_id=2`, EOS/PAD, `max_length=200`, and tokenizer language prefixes for multilingual ST.

Stage 7: enable fusions: conv rewrite, packed QKV, hoisted cross K/V, optimized attention, last-token logits.

Stage 8: optional end-to-end CPU fbank/CMVN integration and throughput tuning.

Initially stub: training loss, dropout, LayerDrop, gradient checkpointing, output attentions/hidden states, beam search, and CPU waveform preprocessing if the first target accepts precomputed features.

## 12. Parity and validation plan

- Feature extractor parity: compare fbank+CMVN against HF for 16 kHz and 48 kHz sampled configs, with padding/truncation and `pad_to_multiple_of`.
- Conv subsampler random tests: `[B,T,80]` with odd/even `T`, small/medium/large `d_model`, and mask length downsampling.
- Sinusoidal position tests: padding-aware input IDs, decoder incremental `past_key_values_length`, and encoder synthetic padding masks.
- Single encoder layer parity in fp32, then full encoder parity.
- Single decoder layer parity for self-attn only and cross-attn, then full model teacher-forced logits.
- Decode parity: first token with empty cache, second token with self/cross cache reuse, beam cache reorder smoke.
- End-to-end ASR/ST parity against HF generated token IDs for small official checkpoints.

Recommended tolerances: fp32 absolute/relative around `1e-4` for layer outputs; fp16/bf16 use looser `1e-2` style tolerances after attention/FFN, with logits/token parity as the final acceptance criterion.

## 13. Performance probes

- CPU preprocessing throughput: waveform seconds/sec for fbank+CMVN, split by sampling rate.
- Conv subsampler throughput over `B`, `T`, and `d_model`.
- Encoder-only throughput over `S_enc` after subsampling.
- Decoder prefill latency for target lengths 1, 16, 64, 200.
- Incremental decode tokens/sec with and without cross K/V hoist.
- KV cache memory by batch, beam, decoder layers, and target length.
- Attention backend comparison: eager dense vs fused prefill vs cached decode.
- LM head cost and last-token-only decode benefit.
- Batch/beam sweep for generation.

## 14. Skip/defer list

- Training loss and `shift_tokens_right` except for teacher-forced parity harnesses.
- Dropout, LayerDrop, gradient checkpointing.
- Output attentions and hidden-state recorder plumbing.
- Full ASR pipeline chunking/streaming; inspected source does not implement chunk splitting.
- Beam search initially, beyond cache reorder smoke.
- FlashAttention/SDPA routing until source support flags or local validation justify admission.
- Remote-code or `SpeechEncoderDecoderModel` variants such as wav2vec2 encoder + Speech2Text2 decoder.
- Quantization and multi-GPU/tensor parallelism.

## 15. Final implementation checklist

- [ ] Parse `Speech2TextConfig` and representative generation/tokenizer metadata.
- [ ] Preserve tied vs untied LM-head/embedding aliasing from config.
- [ ] Load Conv1d, LayerNorm, Linear, embedding, and LM-head weights.
- [ ] Accept `input_features [B,T,80]` plus `attention_mask [B,T]`.
- [ ] Implement Conv1d stride-2 padding + GLU subsampler.
- [ ] Implement downsampled feature attention mask.
- [ ] Implement sinusoidal positional embedding and padding-aware position IDs.
- [ ] Implement encoder MHA/FFN block with fp16 clamp guard.
- [ ] Implement decoder self-attention, cross-attention, and `EncoderDecoderCache` ABI.
- [ ] Implement causal and bidirectional additive masks.
- [ ] Implement generation start/EOS/PAD handling for greedy decode.
- [ ] Add multilingual tokenizer language-prefix admission metadata.
- [ ] Add conv+GLU rewrite parity tests.
- [ ] Add packed QKV and cross K/V hoist rewrites behind guards.
- [ ] Add single-layer, full-encoder, teacher-forced logits, and incremental decode parity tests.
- [ ] Benchmark preprocessing, encoder, prefill, decode, cache memory, and LM-head projection.
