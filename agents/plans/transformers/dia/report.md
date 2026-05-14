# Dia Transformers family audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: nari-labs/Dia-1.6B-0626 for native Transformers Dia; nari-labs/Dia-1.6B inspected as legacy official config
Config source: Hugging Face config/preprocessor/generation/tokenizer JSON plus Transformers source defaults
Source files inspected: configuration_dia.py, modeling_dia.py, modular_dia.py, generation_dia.py, processing_dia.py, feature_extraction_dia.py, tokenization_dia.py, generation/logits_process.py
Any missing files or assumptions: no gated/401/403 gaps found; no native tiny/debug Dia checkpoint found; Dia2 configs are out of scope for this native Dia source
```

Primary source links:

- Transformers Dia source at commit: [configuration](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/dia/configuration_dia.py), [modeling](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/dia/modeling_dia.py), [modular source](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/dia/modular_dia.py), [generation](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/dia/generation_dia.py), [processor](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/dia/processing_dia.py), [feature extractor](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/dia/feature_extraction_dia.py), [tokenizer](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/dia/tokenization_dia.py).
- Official HF configs: [nari-labs/Dia-1.6B-0626](https://huggingface.co/nari-labs/Dia-1.6B-0626/tree/main), [legacy nari-labs/Dia-1.6B](https://huggingface.co/nari-labs/Dia-1.6B/tree/main), codec [descript/dac_44khz](https://huggingface.co/descript/dac_44khz/tree/main).
- Local snapshots: `agents/plans/transformers/dia/_sources/`.

`modeling_dia.py` is generated from `modular_dia.py`; future source edits should target `modular_dia.py`, while runtime parity should inspect the generated `modeling_dia.py` actually imported by Transformers.

## 2. High-level architecture

Dia is a TTS encoder-decoder model:

```text
UTF-8 byte tokenizer -> noncausal text encoder -> causal multi-codebook audio decoder with cross-attention -> channel-flattened audio logits -> sampling/delay/EOS logic -> DAC codebook decode -> waveform
```

Stage decomposition:

- CPU/data pipeline: `DiaTokenizer` byte-tokenizes text; `DiaProcessor` builds delayed decoder prompts and masks; optional voice prompt audio is padded and encoded by a separate `DacModel`.
- Independently cacheable encoder: text encoder output `[B, S_text, 1024]` can be reused for multiple audio decodes.
- Decoder prefill/decode: decoder consumes audio code tensors `[B, T_audio, 9]`; generation flattens channels to `[B * 9, T]` around the HF sampling loop.
- Output stage: `logits_dense` maps decoder hidden `[B, T, 2048]` to `[B * 9, T, 1028]`; generation returns delayed codebooks `[B, T, 9]`.
- Codec stage: `DiaProcessor.batch_decode` reverses delay, trims prompt/EOS/PAD, transposes to DAC `[B, 9, T]`, and calls `descript/dac_44khz` `DacModel.decode`.

First useful DinoML target: native `DiaForConditionalGeneration` code-token generation through generated audio codes. DAC waveform synthesis should be a later composed model-family target.

## 3. Important config dimensions

Native checkpoint dimensions from `nari-labs/Dia-1.6B-0626/config.json`:

| Field | Encoder | Decoder self-attn | Decoder cross-attn |
|---|---:|---:|---:|
| hidden_size | 1024 | 2048 | query 2048, KV source 1024 |
| num_hidden_layers | 12 | 18 | 18 cross-attn modules |
| num_attention_heads | 16 | 16 | 16 |
| num_key_value_heads | 16 | 4 | 16 |
| head_dim | 128 | 128 | 128 |
| projection width | Q/K/V 2048 each? encoder uses 16 * 128 = 2048 over hidden 1024 | Q 2048, K/V 512 | Q/K/V 2048 |
| intermediate_size | 4096 | 8192 | n/a |
| vocab_size | text bytes 256 | audio vocab 1028 | n/a |
| max_position_embeddings | 1024 | 3072 | encoder positions only for KV source |
| activation | SiLU gated MLP | SiLU gated MLP | n/a |
| RoPE | theta 10000, head_dim 128 | theta 10000, head_dim 128 | none in cross-attn |
| cache support | no KV cache | self KV cache | cross KV cache over encoder states |

Important caveat: `hidden_size != num_heads * head_dim` for the encoder. The encoder attention projects `1024 -> 2048` for Q/K/V and `2048 -> 1024` for output. Do not infer projection width from hidden size.

Representative checkpoint/config sweep:

| Repo | Native `model_type=dia`? | Status | Operator-significant notes |
|---|---|---|---|
| [nari-labs/Dia-1.6B-0626](https://huggingface.co/nari-labs/Dia-1.6B-0626/blob/main/config.json) | yes | in scope | Native HF Dia schema; 9 channels; 1028 audio vocab; decoder GQA self-attn with 4 KV heads; float32 metadata. |
| [nari-labs/Dia-1.6B](https://huggingface.co/nari-labs/Dia-1.6B/blob/main/config.json) | no | legacy/out of scope for native source | Official older schema uses nested `data`/`model` names, `audio_*_value`, `n_embd`; processor config says `DacFeatureExtractor`. Do not treat as native without conversion. |
| [nari-labs/Dia2-1B](https://huggingface.co/nari-labs/Dia2-1B/blob/main/config.json) | no | out of scope | 34 channels, depformer, action vocab, different runtime schedule; current Dia source does not implement this architecture. |
| [nari-labs/Dia2-2B](https://huggingface.co/nari-labs/Dia2-2B/blob/main/config.json) | no | out of scope | Same Dia2 family shape, different layer/hidden sizes; requires separate audit. |
| [descript/dac_44khz](https://huggingface.co/descript/dac_44khz/blob/main/config.json) | codec, not Dia | later stage | DAC codebook contract: 9 codebooks, codebook size 1024, hop length 512, sampling rate 44100. |

## 3a. Family variation traps

- Encoder projection width is 2048 even though hidden size is 1024; output projection returns to 1024.
- Decoder self-attn is GQA: 16 query heads, 4 KV heads, repeat factor 4.
- Decoder cross-attn is MHA over encoder states: 16 Q heads, 16 KV heads, K/V input hidden size 1024.
- Attention scaling is source-set to `1`, not `1 / sqrt(head_dim)`.
- MLP is gated: one biasless `Linear(hidden -> 2 * intermediate)`, chunk order `(gate, up)`, then `up * silu(gate)`.
- Audio embedding is multi-channel with a single table of size `num_channels * vocab_size`; each channel adds an offset `channel * vocab_size` and the channel embeddings are summed.
- Generation is not ordinary seq2seq generation: channel flattening, delay masks, CFG duplication, EOS channel filtering, and delayed EOS forcing are required for parity.
- Audio token IDs are not text tokens: DAC codes use 0..1023; EOS=1024, PAD=1025, BOS=1026; vocab size 1028 leaves one extra ID not obviously used by current source.
- Processor output format matters: `return_tensors` must be `"pt"`; delayed `decoder_input_ids` are `[B, T, C]`, `decoder_attention_mask` is `[B, T]`.
- No NCHW/NHWC vision path exists. Guard against generic layout passes rewriting channel/time axes in audio tensors or head/time axes in attention tensors.

## 4. Operator coverage checklist

Tensor/layout ops:

- reshape/view/transpose/contiguous around `[B, T, C] <-> [B, C, T] <-> [B*C, T]`.
- chunk along last dim, sum over channel embedding dim, arange, clamp, stack, unbind, nonzero, indexed gather, where, masked_fill.
- strict axis guards: embedding channel axis is last in `[B,T,C]`; attention head layout is `[B,H,T,D]`; logits reshape is `[B,T,C,V] -> [B*C,T,V]`.

Neural primitives:

- Embedding(256 -> 1024) for text.
- Multi-channel Embedding(9 * 1028 -> 2048), then sum over 9 channels.
- Biasless Linear shapes: encoder Q/K/V `1024 -> 2048`, encoder O `2048 -> 1024`, decoder self Q `2048 -> 2048`, decoder self K/V `2048 -> 512`, decoder cross Q `2048 -> 2048`, decoder cross K/V `1024 -> 2048`, decoder O `2048 -> 2048`.
- Gated MLP: encoder `1024 -> 8192 -> 1024`; decoder `2048 -> 16384 -> 2048`.
- RMSNorm over last dim with fp32 variance.
- `logits_dense`: `2048 -> 9 * 1028 = 9252`, bias false.

Attention/generation/cache ops:

- Dense bidirectional self-attn in encoder.
- Causal decoder self-attn with KV cache.
- Noncausal decoder cross-attn with cacheable encoder K/V.
- RoPE on self-attn Q/K only; no RoPE in cross-attn.
- DynamicCache/EncoderDecoderCache update, cache length, cross-cache `is_updated` flags, and beam/cache reorder compatibility from HF generation.
- Logits processors: temperature, CFG, top-k mask, EOS channel filtering, EOS delay forcing, normal top-p/top-k sampling.

Preprocessing/codebook ops:

- UTF-8 byte tokenization with special tokens `[S1]` and `[S2]`.
- Audio prompt feature extraction and DAC encode/decode are processor-side, not Dia core graph.
- Delay/revert gather with per-channel offsets and BOS/PAD fill.
- DAC decode expects codebooks `[1, 9, T]` per sample.

## 5. Layer/block breakdown

Encoder, repeated 12 times:

```text
x: [B, S_text, 1024]
r = x
x = RMSNorm(x)
q,k,v = Linear(1024 -> 2048), reshape to [B, 16, S_text, 128]
q,k = RoPE(q,k)
x_attn = dense bidirectional attention(q,k,v, mask, scaling=1)
x = r + Linear(2048 -> 1024)(x_attn)
r = x
x = RMSNorm(x)
gate, up = chunk(Linear(1024 -> 8192)(x), 2)
x = r + Linear(4096 -> 1024)(up * silu(gate))
```

Decoder, repeated 18 times:

```text
x: [B, T_audio, 2048]
r = x
x = RMSNorm(x)
self q = Linear(2048 -> 2048) -> [B,16,T,128]
self k/v = Linear(2048 -> 512) -> [B,4,T,128]
self q,k = RoPE(q,k); update/reuse self KV cache
x = r + Linear(2048 -> 2048)(causal GQA attention)
r = x
x = RMSNorm(x)
cross q = Linear(2048 -> 2048) -> [B,16,T,128]
cross k/v = Linear(1024 -> 2048) from encoder -> [B,16,S_text,128]; cache after first compute
x = r + Linear(2048 -> 2048)(cross attention)
r = x
x = RMSNorm(x)
gate, up = chunk(Linear(2048 -> 16384)(x), 2)
x = r + Linear(8192 -> 2048)(up * silu(gate))
```

All listed projections are biasless in source.

## 6. Attention requirements

Encoder attention:

- Noncausal self-attention with bidirectional mask from `create_bidirectional_mask`.
- MHA with 16 Q/K/V heads, head dim 128, no cache.
- Source tensor layout after projection is `[B, H, S, D]`.

Decoder self-attention:

- Causal GQA self-attention, 16 query heads, 4 KV heads, head dim 128.
- Cache stores pre-repeat K/V after RoPE for keys: per layer K/V `[B, 4, T_cache, 128]`; attention repeats to `[B, 16, T_cache, 128]`.
- During decode with cache, only the final decoder time step is passed after the first iteration.
- Mask is created by `create_causal_mask`; if missing, decoder creates an all-ones mask length `past_len + seq_len`.

Decoder cross-attention:

- Noncausal MHA, 16 query heads, 16 KV heads, head dim 128.
- Cross cache stores projected encoder K/V `[B, 16, S_text, 128]` and uses `past_key_values.is_updated[layer_idx]` to avoid recomputation.
- Encoder outputs themselves are independently cacheable before decoder prefill.

Backend compatibility:

- Source advertises FlashAttention, SDPA, and flex attention support, with eager fallback.
- Fused attention parity must preserve scaling factor `1`, mask addition before softmax, fp32 softmax accumulation, and dropout only in training.

## 7. Position encoding and custom math

Dia uses default RoPE for encoder and decoder self-attention. `rope_theta=10000`, `head_dim=128`, and cos/sin are computed in fp32 then cast to the activation dtype.

Short parity snippets:

```python
def dia_rms_norm(x, weight, eps):
    y = x.float()
    y = y * torch.rsqrt(y.pow(2).mean(dim=-1, keepdim=True) + eps)
    return weight * y.to(x.dtype)
```

```python
def dia_rope(q, k, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    def rotate_half(x):
        a, b = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-b, a), dim=-1)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin
```

```python
def dia_swiglu(x, gate_up, down):
    gate, up = gate_up(x).chunk(2, dim=-1)
    return down(up * torch.nn.functional.silu(gate))
```

Cos/sin tables can be precomputed for static max lengths but dynamic RoPE plumbing should keep `position_ids` explicit because decode offsets depend on cache length.

## 8. Preprocessing and input packing

Text:

- `DiaTokenizer` encodes each UTF-8 byte as one token in vocab 0..255; tokenizer pad/unk token is `<pad>`.
- Processor defaults use right padding and max text length 1024 in tokenizer config.
- Encoder input tensors: `input_ids [B,S_text]`, `attention_mask [B,S_text]`.

Audio prompt/generation packing:

- `DiaFeatureExtractor` pads waveform batches to a multiple of `hop_length=512`, returns `input_values` and `padding_mask`, and enforces sampling rate. Native checkpoint preprocessor config uses mono, 44100 Hz, right padding.
- If audio prompt is supplied, processor calls DAC encode sample-by-sample, transposes returned codes to `[B,T_code,9]`, adds BOS/padding and optional EOS for training.
- If no audio prompt and generation is true, processor starts with `decoder_input_ids [B,1,9]` full of BOS and `decoder_attention_mask [B,1+max_delay]`.
- Default delay pattern is `[0,8,9,10,11,12,13,14,15]`; processor expands a prefill buffer to `[B, T+max_delay, 9]` and gathers per-channel shifted tokens.

Generation packing:

- `_prepare_decoder_input_ids_for_generation` trims right PAD positions, stores the full delayed mask as `decoder_delay_mask`, and returns channel-major IDs.
- `_main_generate_loop` flattens to `[B*9,T]` for HF sampling.
- `prepare_inputs_for_generation` reshapes back to `[B,T,9]`, reapplies delay mask with `torch.where`, and slices to the last time step when cache is active.
- CFG duplicates text inputs, masks, decoder inputs, decoder masks, and position IDs; logits processor combines conditioned and unconditioned logits.

Output decoding:

- `generate()` reshapes sequences from `[B*9,T]` to `[B,T,9]` and reapplies the original delay mask.
- `batch_decode()` reverts delay, determines start from prompt length or BOS count on channel 0, determines end from PAD count minus EOS, sends `[1,9,T]` codes to `DacModel.decode`, and returns waveform tensors.

CPU/data-pipeline versus GPU/runtime:

- Tokenization, feature extraction, DAC encode/decode, delay index construction, and final audio file writing are processor/data-pipeline work for first integration.
- Core GPU graph starts at text `input_ids/attention_mask` plus delayed audio `decoder_input_ids/decoder_attention_mask`.

## 9. Graph rewrite / lowering opportunities

### Rewrite: separate Q/K/V linears -> packed projection

Source pattern:

```text
q = Linear(x); k = Linear(x); v = Linear(x); reshape/transposes
```

Replacement: one GEMM producing packed QKV followed by split.

Preconditions:

- Same input tensor, same dtype/device, all biasless, no intervening consumers.
- Preserve exact split order Q,K,V and projection widths; encoder Q/K/V are each 2048, decoder self K/V are each 512.
- Cross-attn cannot pack Q with K/V because Q consumes decoder hidden and K/V consume encoder hidden.

Parity test: compare Q/K/V tensors before RoPE for random B/S/T with fp32 and bf16/fp16 tolerances.

### Rewrite: multi-channel embedding as gather-sum

Source pattern:

```text
tokens = audio_codes + channel_offsets
embeds = embedding(tokens).view(B,T,C,H).sum(dim=2)
```

Replacement: fused offset-gather plus channel reduction.

Preconditions:

- `audio_codes` rank 3 `[B,T,C]`; `C == config.num_channels`; token IDs in `[0,vocab_size)`.
- Channel offsets must be `arange(C) * vocab_size`; no layout pass may move the channel axis without rewriting offsets and sum axis.

Failure cases: flattened `[B*C,T]` generation IDs must be reshaped before this rewrite.

### Rewrite: last-token-only logits during decode

Source pattern: full `logits_dense` over `[B,T,H]`, then reshape to `[B*C,T,V]`.

Replacement: when `use_cache=True` and not first iteration, apply `logits_dense` only to `[B,1,H]`.

Preconditions:

- No caller requests full decoder logits for earlier steps.
- Generation path already slices decoder input to last step.

### Layout guard: no generic NHWC/channel-last translation

Dia has no image layout region. Protect:

- Audio code tensors `[B,T,C]`.
- DAC code tensors `[B,C,T]` at processor boundary.
- Attention tensors `[B,H,T,D]`.
- Logits `[B*C,T,V]`.

Any layout pass must explicitly rewrite `sum(dim=2)`, `transpose(1,2)`, `softmax(dim=-1)`, `chunk(dim=-1)`, `view(B,T,C,V)`, and channel/EOS indexing.

## 10. Kernel fusion candidates

Highest priority:

- RMSNorm over hidden 1024/2048 with fp32 accumulation.
- Biasless GEMMs for all projection and MLP layers.
- RoPE + Q/K reshape/transposes + attention prefill/decode.
- GQA FlashAttention decode for decoder self-attn, storing unrepeated K/V.
- Cross-attn K/V projection cache for encoder outputs.
- Gated MLP fusion: gate/up projection split, SiLU, multiply, down projection.

Medium priority:

- Multi-channel embedding offset gather + sum.
- `logits_dense` last-step only plus channel-aware reshape.
- Logits processor kernels for EOS filtering and top-k CFG if generation remains on GPU.
- Causal/bidirectional mask materialization elimination where attention backend accepts masks directly.

Lower priority:

- Processor delay/revert gather on GPU. Useful for large batches, but CPU-side preparation is acceptable initially.
- DAC codec lowering. It is required for waveform parity but separable from Dia code-token generation.

## 11. Runtime staging plan

1. Parse native `DiaConfig` and reject legacy Dia/Dia2 schemas for this target.
2. Load weights and validate one encoder layer plus one decoder layer with random inputs.
3. Implement encoder-only parity with byte-token inputs and bidirectional masks.
4. Implement decoder prefill with delayed `[B,T,9]` codes and cross-attn over cached encoder hidden states.
5. Implement autoregressive decode with `EncoderDecoderCache`: decoder self KV grows, cross KV cached once.
6. Add channel-flattened logits and HF-equivalent sampling control for greedy/sampling without CFG.
7. Add delay/EOS logits processors and CFG.
8. Compose with external DAC processor/model for waveform parity.
9. Add packed projection and attention fusions.

Stubbable initially: training loss, output labels, beam search, streamer, DAC decode, soundfile saving, voice prompt audio encode.

## 12. Parity and validation plan

- Random tensor tests for RMSNorm, RoPE, repeat_kv, multi-channel embedding, delay/revert indexing.
- Single encoder layer parity at fp32 against PyTorch source with fixed random weights.
- Single decoder layer parity for self-attn prefill, self-attn decode cache update, and cross-attn cache reuse.
- Full encoder parity on byte-token `input_ids` and right-padded masks.
- Full decoder prefill logits parity: compare `[B*9,T,1028]`.
- Decode parity: run 3-5 greedy tokens with cache and compare generated code IDs.
- End-to-end code-token parity using `DiaProcessor` output. Waveform parity deferred until DAC is composed.
- Suggested tolerances: fp32 `1e-4` absolute for layer outputs; fp16/bf16 `5e-2` for full-block outputs, tighter for isolated linear/norm ops.

No DinoML tests were run for this audit by user request.

## 13. Performance probes

- Processor throughput: tokenizer + delay mask construction for batch and sequence sweeps.
- Optional voice-prompt DAC encode throughput, separated from model time.
- Encoder-only latency/throughput over text length 128/512/1024.
- Decoder prefill over audio prompt length and max delay.
- Decode tokens/sec measured in audio frames, with batch and channel flattening overhead visible.
- KV cache memory: self cache `18 * 2 * B * 4 * T * 128 * dtype_size`; cross cache `18 * 2 * B * 16 * S_text * 128 * dtype_size`.
- Attention backend comparison: eager, SDPA, FlashAttention/flex where available.
- Last-token logits versus full-sequence logits.
- DAC decode throughput for generated code length to avoid hiding model wins behind codec time.

## 14. Skip/defer list

- Training loss and label generation.
- Gradient checkpointing.
- Beam search and `num_return_sequences > 1` (source rejects multiple return sequences).
- Speculative/assistant generation.
- Distributed generation and synced GPU control.
- Soundfile saving.
- Legacy `nari-labs/Dia-1.6B` non-native conversion path.
- Dia2 depformer/action-vocab architecture.
- DAC waveform synthesis in the first native Dia code-token milestone.

## 15. Final implementation checklist

- [ ] Parse native `DiaConfig` and validate `num_channels == len(delay_pattern)`.
- [ ] Reject legacy Dia and Dia2 configs for this native-source target.
- [ ] Load encoder, decoder, multi-channel embedding, and logits weights.
- [ ] Implement byte-token text input contract.
- [ ] Implement delayed audio code input contract `[B,T,9]`.
- [ ] Implement RMSNorm fp32-accumulation primitive.
- [ ] Implement Dia RoPE with explicit position IDs.
- [ ] Implement encoder noncausal attention with projection width guards.
- [ ] Implement decoder causal GQA self-attn and unrepeated KV cache.
- [ ] Implement decoder cross-attn and cross KV cache.
- [ ] Implement gated SiLU MLP.
- [ ] Implement channel-flattened logits reshape.
- [ ] Implement generation delay mask, EOS filtering, and EOS delay forcing.
- [ ] Add CFG logits processor parity.
- [ ] Add processor/code-token parity tests.
- [ ] Compose DAC decode as a later audio-output stage.
- [ ] Benchmark encoder, prefill, decode, cache memory, and codec separately.
