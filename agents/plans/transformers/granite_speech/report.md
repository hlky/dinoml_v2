# Transformers Audit: `granite_speech`

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: ibm-granite/granite-speech-3.3-2b, ibm-granite/granite-speech-3.3-8b; historical remote-code sample ibm-granite/granite-speech-3.2-8b
Config source: official Hugging Face config snapshots copied into _sources/
Source files inspected: configuration_granite_speech.py, modeling_granite_speech.py, feature_extraction_granite_speech.py, processing_granite_speech.py
Any missing files or assumptions: 3.3 processor_config.json returned 404; preprocessor_config.json is empty. Feature extraction defaults come from source. 3.2 config advertises remote-code auto_map and a non-native projector model_type, so it is not treated as native-source parity.
```

Primary sources:

- Local source snapshots: `_sources/configuration_granite_speech.py`, `_sources/modeling_granite_speech.py`, `_sources/feature_extraction_granite_speech.py`, `_sources/processing_granite_speech.py`.
- Official configs: [3.3-2b config](https://huggingface.co/ibm-granite/granite-speech-3.3-2b/resolve/main/config.json), [3.3-8b config](https://huggingface.co/ibm-granite/granite-speech-3.3-8b/resolve/main/config.json), [3.2-8b config](https://huggingface.co/ibm-granite/granite-speech-3.2-8b/resolve/main/config.json).
- Official adapter/config metadata: [3.3-2b adapter_config](https://huggingface.co/ibm-granite/granite-speech-3.3-2b/resolve/main/adapter_config.json), [3.3-2b tokenizer_config](https://huggingface.co/ibm-granite/granite-speech-3.3-2b/resolve/main/tokenizer_config.json), [3.3-2b model API](https://huggingface.co/api/models/ibm-granite/granite-speech-3.3-2b).
- Delegated in-library bodies sampled in the pinned checkout: `models/granite/modeling_granite.py` for the text decoder and `models/blip_2/modeling_blip_2.py` for the native 3.3 Q-Former.

Report target: inference for speech-conditioned autoregressive text generation. Training loss, CTC auxiliary behavior as a standalone ASR head, and remote-code 3.2 Q-Former are deferred.

## 2. High-level architecture

Granite Speech is a staged audio encoder plus projector plus causal LM:

```text
waveform preprocessing -> log-mel frame stacking -> Conformer audio encoder -> windowed Q-Former projector -> audio-token embedding stitch -> Granite causal LM prefill -> KV-cache decode -> logits/sampling
```

Stage decomposition:

| Stage | Owner | Runtime contract | Cacheability |
| --- | --- | --- | --- |
| Waveform to stacked log-mel | CPU/data pipeline first | `float` mono waveform at 16 kHz to `input_features [B, T_enc, 160]`; returns `input_features_mask [B, T_audio_tokens]` | Recompute or cache per audio sample outside decoder |
| Conformer encoder | Granite Speech source | `Linear(160 -> 1024)`, 16 or 10 Conformer blocks, optional mid-layer CTC-style softmax injection | Cache `last_hidden_state [B, T_enc, 1024]` per audio |
| Projector | Native 3.3 uses `blip_2_qformer` | Pad encoder time to windows of 15, run 3 learned queries per window over hidden states, linear project to LM hidden size | Cache projected audio embeddings `[B, ceil(T_enc/15)*3, H_text]` |
| Modality stitch | Granite Speech source | Replace `<|audio|>` token embeddings by flattened projected audio embeddings after count validation | Safe rewrite to indexed row copy with guards |
| Text decoder | Delegated Granite causal LM | GQA causal decoder with RoPE, RMSNorm, SwiGLU, KV cache, tied embeddings | Standard per-layer KV cache |

The first useful DinoML target is "projected-audio prefix plus Granite decode." The Conformer encoder/projector can be validated independently from LM prefill/decode by comparing projected audio embeddings.

## 3. Important config dimensions

Source defaults:

| Field | Default | Source |
| --- | ---: | --- |
| audio feature input dim | 160 | `GraniteSpeechEncoderConfig.input_dim` |
| encoder layers | 10 | source default |
| encoder hidden dim | 1024 | source default and 3.3 configs |
| encoder heads/head dim | 8 / 128 | source default and 3.3 configs |
| encoder feedforward mult | 4 | source default |
| encoder context size | 200 | source default and configs |
| encoder max relative position table | 512 | source default, present in 3.3 configs, omitted in 3.2 config and defaulted by source |
| encoder conv kernel/expansion | 15 / 2 | source default and configs |
| projector window/downsample | 15 / 5 | source default and 3.3 top-level config |
| projector queries per window | 3 | inferred from `window_size // downsample_rate` |
| audio token id/index | 49155 default, 49159 in 3.3 configs | source default and configs |

Representative checkpoint sweep:

| Model | Native source status | Encoder | Projector | Text decoder | Vocab/audio token | Dtype/cache |
| --- | --- | --- | --- | --- | --- | --- |
| `granite-speech-3.3-2b` | Native `granite_speech` | 16 layers, hidden 1024, output_dim 256 | `blip_2_qformer`, 2 layers, hidden 1024, 16 heads, MLP 4096 | Granite 2B: hidden 2048, 40 layers, 32 Q heads, 8 KV heads, MLP 8192, RoPE theta 1e7 | vocab 49160, audio token 49159 | bf16, `use_cache=true` |
| `granite-speech-3.3-8b` | Native `granite_speech` | Same as 3.3-2b | Same as 3.3-2b | Granite 8B: hidden 4096, 40 layers, 32 Q heads, 8 KV heads, MLP 12800, RoPE theta 1e7 | vocab 49160, audio token 49159 | bf16, `use_cache=true` |
| `granite-speech-3.2-8b` | Historical remote-code shape | 10 layers, hidden 1024, output_dim 42; omits `max_pos_emb`, source would default 512 | `granite_speech_qformer`, includes `llm_dim=4096`, not native 3.3 `blip_2_qformer` | Granite 8B: hidden 4096, 40 layers, 32 Q heads, 8 KV heads | vocab 49156, audio token 49155 | bf16, `auto_map` remote code |

Decoder dimensions are config-derived. Granite decoder behavior is delegated to `models/granite`; this report records the integration contract but should compose a separate Granite audit for full decoder coverage.

## 3a. Family variation traps

- 3.2 and 3.3 are not the same native-source surface. 3.2 advertises `auto_map` and `projector_config.model_type="granite_speech_qformer"`; the pinned native source defaults/3.3 configs use `blip_2_qformer`.
- The 3.3 checkpoints require PEFT LoRA for intended audio behavior. `generate()` enables adapters only when `input_features` is present and disables them for text-only generation. Adapter config targets `q_proj` and `v_proj` with rank 64.
- Audio token id changed: 49155 in 3.2 and source default, 49159 in 3.3.
- `input_features_mask` is not shaped like `input_features`; it is shaped over projector output tokens after windowing. Treat it as post-projector valid-token metadata.
- The modality stitch uses `masked_scatter`, but the processor creates a stricter placeholder expansion. DinoML should reject arbitrary boolean scatter and admit a guarded indexed row-copy form.
- Conformer attention is local/block-like over chunks of `context_size=200`, with Shaw relative-position bias as an additive SDPA mask. It is not decoder KV attention.
- Q-Former is kept in fp32 in BLIP-2 source and has self-attention plus cross-attention every layer for 3.3 configs.
- Text decoder uses GQA: `num_key_value_heads=8` and `num_attention_heads=32`. KV cache stores KV-head tensors before repeat expansion.
- Granite decoder has source-specific scalars: `embedding_multiplier`, `attention_multiplier`, `residual_multiplier`, and final `logits_scaling`.
- Source tensor layouts are mostly sequence-first `[B, T, C]`; only Conformer conv submodule temporarily permutes to `[B, C, T]` for Conv1d/BatchNorm1d. Keep no-layout-translation guards around encoder conv and attention axes unless a local pass rewrites every consumer.

## 4. Operator coverage checklist

Tensor/layout ops:

- `pad` on time axis to multiples of 2, 15, and 200.
- `reshape/view`, `transpose`, `permute`, `contiguous`, `chunk`, `expand`, `where`, clone.
- Boolean equality masks, `sum`, mask expansion, boolean indexing from `input_features_mask`, and bounded indexed write for audio embeddings.

Neural network primitives:

- Audio encoder: `Linear(160 -> 1024)`, repeated `LayerNorm`, `Linear(1024 -> 4096)`, SiLU, dropout disabled for inference, `Linear(4096 -> 1024)`, Conv1d pointwise, GLU, depthwise Conv1d groups=channels, BatchNorm1d inference, softmax, residual adds/scales.
- Mid encoder injection: at layer `num_layers // 2`, `Linear(1024 -> output_dim)`, Softmax over `output_dim`, `Linear(output_dim -> 1024)`, residual add.
- Projector: learned query parameter `[1, 3, 1024]`, BLIP-2 Q-Former self/cross attention, GELU MLP, LayerNorm, final `Linear(1024 -> H_text)`.
- Text decoder: token embedding, RMSNorm, Q/K/V/O projections, SwiGLU MLP, tied LM head, scalar multiplies/divide.

Attention primitives:

- Conformer local self-attention: Q projection and packed KV projection, shape `[B, blocks, heads, C=200, D=128]`; additive relative-position logits from `einsum`.
- Q-Former noncausal self-attention over query tokens and cross-attention from 3 query tokens to 15 encoder frames per window.
- Granite causal GQA self-attention with RoPE and KV cache.

Position/relative-bias ops:

- Conformer Shaw relative position embedding table indexed by precomputed `attention_dists [200, 200]`.
- Granite RoPE with `rope_theta=10000000.0` and optional future dynamic rope types from delegated Granite config.
- Q-Former uses absolute position support in config, but Granite Speech passes query embeddings only; no tokenizer text input to Q-Former in 3.3.

Generation/cache ops:

- Use delegated LM `prepare_inputs_for_generation`.
- On first generation iteration or when `use_cache=False`, pass `input_features`; cached decode omits audio inputs.
- Per-layer decoder KV cache from Granite LM. Independently cacheable audio embeddings are not KV cache.

Preprocessing-coupled ops:

- torchaudio MelSpectrogram, log10, clamp, dynamic padding, waveform collation.
- Placeholder expansion from one `<|audio|>` text token into `audio_embed_size` repeated tokens per audio sample.

Quantized/packed metadata:

- No source-coupled quantized weights in native Granite Speech. PEFT LoRA adapter is separate weight metadata and must be loaded/applied if `has_lora_adapter=true`.

## 5. Layer/block breakdown

Feature extraction:

```text
waveform [B, samples]
mel = MelSpectrogram(n_fft=512, win=400, hop=160, n_mels=80) -> [B, 80, frames]
logmel = log10(clamp(mel.T, min=1e-10))
logmel = max(logmel, global_max - 8) / 4 + 1
if frames odd: drop last frame
input_features = reshape adjacent pairs -> [B, frames/2, 160]
```

Conformer encoder:

```text
x = Linear(160 -> 1024)(input_features)
for layer i in 1..N:
  x = x + 0.5 * FFN(LayerNorm(x))
  x = x + ShawBlockAttention(LayerNorm(x), context_size=200)
  x = x + ConvModule(LayerNorm(x))
  x = x + 0.5 * FFN(LayerNorm(x))
  x = LayerNorm(x)
  if i == N // 2:
    ctc = Linear(1024 -> output_dim)(x)
    x = x + Linear(output_dim -> 1024)(Softmax(ctc, dim=-1))
```

Conformer attention:

```text
x_pad = right_pad_to_multiple(x, 200)
q = Linear(1024 -> 8*128, no bias)(x_pad)
k, v = chunk(Linear(1024 -> 2*8*128, no bias)(x_pad), 2)
q,k,v -> [B, num_blocks, 8, 200, 128]
rel = Embedding(2*max_pos_emb+1, 128)(attention_dists[200,200])
pos_attn = einsum("b m h c d, c r d -> b m h c r", q, rel) * rsqrt(128)
if tail block padded: mask invalid rows/cols in final block
out = scaled_dot_product_attention(q, k, v, attn_mask=pos_attn, scale=rsqrt(128))
out = Linear(8*128 -> 1024)(out)
```

Conformer conv module:

```text
y = LayerNorm(x)
y = Conv1d(1024 -> 4096, kernel=1)(permute y to [B,C,T])
y = GLU(dim=channel) -> [B, 2048, T]
y = symmetric_pad_time(y), depthwise Conv1d(2048 -> 2048, kernel=15, groups=2048, no bias)
y = SiLU(BatchNorm1d(2048)(y))
y = Conv1d(2048 -> 1024, kernel=1)(y)
y = permute back to [B,T,1024]
```

Projector:

```text
nblocks = ceil(T_encoder / 15)
x = right_pad_to(nblocks * 15)
x = view [B * nblocks, 15, 1024]
query = learned [1, 3, 1024]
qformer(query_embeds=query, encoder_hidden_states=x, encoder_attention_mask=None)
project = Linear(1024 -> H_text)
audio_embeds = view [B, nblocks * 3, H_text]
```

Modality stitch and decoder:

```text
input_ids with audio token positions
llm_input_ids = where(input_ids == audio_token_id, 0, input_ids)
inputs_embeds = token_embedding(llm_input_ids)
audio_embeds = audio_embeds[input_features_mask] if mask exists
validate num audio token slots == audio_embeds.numel()
inputs_embeds = masked_scatter(inputs_embeds, audio_embeds)
GraniteForCausalLM(inputs_embeds, attention_mask, position_ids, past_key_values, logits_to_keep)
```

## 6. Attention requirements

Conformer encoder attention:

- Noncausal self-attention inside independent 200-frame blocks after right padding.
- MHA: 8 heads, 128 dim, Q/K/V widths 1024.
- Additive attention mask is a Shaw relative-position score tensor `[B, blocks, heads, 200, 200]`, not a padding mask except for the final padded block.
- Source forces SDPA math backend for this call. A fused replacement must preserve additive relative-position logits and final-block invalid masking.
- No KV cache.

Q-Former attention:

- Native 3.3 projector uses BLIP-2 Q-Former with 2 layers, 16 heads, hidden 1024.
- Query length is fixed to 3 per 15-frame encoder window.
- Cross-attention source is the local 15-frame Conformer window. `encoder_attention_mask=None` in Granite Speech, so BLIP-2 creates an all-ones encoder mask.
- This is independently stageable and cacheable per audio sample, but not autoregressive decode.

Granite decoder attention:

- Causal self-attention, GQA with 32 query heads and 8 KV heads.
- Head dim is `hidden_size / num_attention_heads`: 64 for 2B, 128 for 8B.
- Q width is `num_attention_heads * head_dim`; K/V width is `num_key_value_heads * head_dim`.
- RoPE is applied to Q/K before cache update.
- Cache stores `[B, num_key_value_heads, cached_seq, head_dim]` K and V per layer, before `repeat_kv` expansion.
- Eager fallback applies mask addition, fp32 softmax, dropout disabled in inference, then matmul V. Optimized SDPA/FlashAttention must preserve Granite's `attention_multiplier` scaling.

## 7. Position encoding and custom math

Conformer relative attention:

```python
seq = arange(context_size)
relpos_dist = seq[:, None] - seq[None, :]
attention_dists = clamp(relpos_dist, -context_size, context_size) + max_pos_emb
rel = rel_pos_emb(attention_dists)          # [C, C, head_dim]
pos_attn = einsum("b m h c d, c r d -> b m h c r", q, rel) * scale
out = sdpa(q, k, v, attn_mask=pos_attn, scale=scale)
```

`attention_dists` is static for a config and can be materialized as a constant. The relative embedding lookup is weighted, so the embedding table remains a learned parameter.

Granite decoder custom math:

```python
inputs_embeds = token_embedding(ids) * embedding_multiplier
q, k = apply_rope(q, k, cos[position_ids], sin[position_ids])
attn_scores = q @ k.transpose(-1, -2) * attention_multiplier
x = residual + attention_out * residual_multiplier
x = residual + mlp_out * residual_multiplier
logits = lm_head(norm(x)) / logits_scaling
```

RoPE cos/sin depend on position ids and sequence length. Static tables can be cached by max position, but dynamic RoPE variants from delegated Granite config must be checked separately.

## 8. Preprocessing and input packing

Waveform contract:

- Source accepts floating torch tensors or numpy arrays. A rank-1 waveform becomes `[1, samples]`; a list of tensors is padded with zeros to batch max length.
- Source checks only floating dtype, not value range despite the error text saying 0 to 1.
- Model card examples assert mono 16 kHz, and source defaults use `sampling_rate=16000`.

Feature tensor contract:

- MelSpectrogram defaults: `n_fft=512`, `win_length=400`, `hop_length=160`, `n_mels=80`.
- Output features after adjacent-frame stacking: `[B, floor(mel_frames_even/2), 160]`.
- Audio embed size per raw sample:

```text
mel_length = raw_length // 160 + 1
encoder_length = mel_length // 2
nblocks = ceil(encoder_length / 15)
audio_embed_size = nblocks * (15 // 5) = nblocks * 3
```

Input packing:

- Processor replaces each text `<|audio|>` occurrence with `audio_embed_size` copies of the same token before tokenization.
- It does not validate text/audio count alignment in the processor. The model validates by comparing total placeholder embedding slots with flattened audio features.
- `input_features_mask [B, max_audio_embed_size]` masks valid projected audio tokens after projector downsampling.
- For first integration, DinoML can require one audio segment per request, processor-expanded contiguous audio token runs, and matching `sum(input_ids == audio_token_id) == input_features_mask.sum()`.

CPU/GPU split:

- Keep torchaudio MelSpectrogram and placeholder string expansion in CPU/data pipeline initially.
- GPU runtime starts at `input_features`, `input_ids`, `attention_mask`, and `input_features_mask`.

## 9. Graph rewrite / lowering opportunities

### Rewrite: audio `masked_scatter` -> guarded indexed row copy

Source pattern:

```text
special_audio_mask = (input_ids == audio_token_id).unsqueeze(-1).expand_as(inputs_embeds)
inputs_embeds = inputs_embeds.masked_scatter(special_audio_mask, audio_features)
```

Replacement:

```text
positions = nonzero(input_ids == audio_token_id) in row-major order
copy audio_features rows to inputs_embeds[positions, :]
```

Preconditions:

- `input_ids` is available, not only `inputs_embeds`.
- Placeholder positions are row-major and count equals `audio_features.shape[0]`.
- `audio_features.shape[-1] == text_hidden_size`.
- Optional first-stage stricter guard: audio placeholder positions are contiguous per sample and count equals `input_features_mask.sum(dim=1)`.

Failure cases:

- Caller supplies arbitrary `inputs_embeds` with no `input_ids`.
- Multiple audio segments with interleaved placeholders unless a segment descriptor is added.
- Mismatch between processor-expanded token count and projector output count.

Parity test sketch:

- Compare HF `get_merged_audio_embeddings` against indexed copy for one and two samples, ragged audio lengths, and an intentional count mismatch.

### Rewrite: Conformer 1x1 Conv1d -> Linear over time

Source pattern:

```text
permute [B,T,C] -> [B,C,T]
Conv1d(kernel=1) -> GLU or projection
permute back
```

Replacement:

```text
Linear(Cin -> Cout) on [B,T,Cin]
```

Preconditions:

- `kernel_size=1`, `stride=1`, `padding=0`, `dilation=1`, `groups=1`.
- Preserve Conv1d weight layout by reshaping `[Cout, Cin, 1] -> [Cout, Cin]`.
- Bias copied directly.

Failure cases:

- Depthwise Conv1d kernel 15 is not covered.
- Layout pass must keep BatchNorm1d/depthwise section in channel-first or rewrite all axes.

### Rewrite: Conformer depthwise Conv1d bounded provider

Source pattern:

```text
F.pad([B,C,T], (7,7)) -> depthwise Conv1d(groups=C, kernel=15, no bias)
```

Replacement:

```text
specialized depthwise temporal convolution on [B,T,C] or provider Conv1d path
```

Preconditions:

- Static kernel 15, groups equals channel count, stride 1, dilation 1.
- Padding follows source formula `(kernel//2, kernel//2)` for odd kernel.
- BatchNorm inference stats folded only if weights are loaded and eval mode is fixed.

Failure cases:

- Training BatchNorm semantics.
- Any layout translation that does not also rewrite GLU channel axis and BatchNorm channel axis.

### Rewrite: Projector window Q-Former batch folding

Source pattern:

```text
[B,T,1024] -> pad to 15 -> [B*nblocks,15,1024] -> QFormer with 3 queries -> [B,nblocks*3,H_text]
```

Replacement:

```text
Treat every 15-frame audio window as an independent mini-example for Q-Former.
```

Preconditions:

- `window_size=15`, `downsample_rate=5`, query count 3.
- No encoder attention mask or cross-window dependency.
- Right padding to full final window with zeros.

Failure cases:

- Remote-code 3.2 projector with different implementation.
- Configs where `window_size % downsample_rate != 0`.

### Rewrite: Conformer Shaw attention to local additive-bias attention

Source pattern:

```text
QK attention per 200-frame block plus additive relative logits from q and rel_pos_emb.
```

Replacement:

```text
local_attention(q,k,v, additive_bias=shaw_q_rel(q, table), block=200)
```

Preconditions:

- `context_size <= max_pos_emb`.
- Block size fixed for compile or bucketed.
- Final block padding mask is applied before softmax.

Failure cases:

- Treating `pos_attn` as a static bias is wrong because it depends on Q.
- Fusing into ordinary FlashAttention without custom bias would lose Shaw term.

## 10. Kernel fusion candidates

Highest priority:

- Granite decoder RMSNorm, RoPE plus GQA attention with KV cache, and SwiGLU MLP. These dominate decode throughput.
- Audio embedding indexed row copy. It avoids admitting general boolean scatter and makes multimodal prefill predictable.
- Conformer block FFN and conv module inference fusion: LayerNorm plus Linear plus SiLU, 1x1 Conv as Linear, depthwise Conv1d plus BatchNorm plus SiLU.

Medium priority:

- Conformer Shaw local attention fused kernel for block size 200, including Q-dependent relative logits.
- Q-Former tiny-query cross-attention. Query length is 3, key length 15, so a specialized small attention/GEMM path may outperform generic attention.
- Projector window packing and final `Linear(1024 -> H_text)` as a batched GEMM.

Lower priority:

- GPU MelSpectrogram/logmel pipeline. Useful only if CPU preprocessing becomes a bottleneck.
- Mid-layer CTC softmax injection. It is required for parity but probably not the main runtime bottleneck.
- Beam-search controller optimization. The source uses normal generation controller behavior; first parity can rely on existing controller semantics.

## 11. Runtime staging plan

1. Parse native 3.3 configs and reject remote-code 3.2 projector unless separately audited.
2. Load weights with explicit PEFT LoRA handling. If `has_lora_adapter=true`, require adapter availability or run a clearly labeled text-only/no-audio mode.
3. Implement feature tensor ABI starting from precomputed `input_features`; stub torchaudio preprocessing outside DinoML runtime.
4. Validate Conformer encoder only, including block padding, Shaw attention, depthwise conv, BatchNorm eval, and mid-layer softmax injection.
5. Validate projector only by feeding encoder outputs through windowed Q-Former and final linear.
6. Implement guarded audio-token stitch and run Granite LM prefill from `inputs_embeds`.
7. Compose full prefill logits parity for one prompt plus audio.
8. Add decoder KV-cache token-by-token generation through delegated Granite cache ABI.
9. Optimize decoder kernels, then Conformer and Q-Former bottlenecks.
10. Add optional processor parity and batching policies.

Initial stubs allowed: CPU torchaudio preprocessing, generation sampling/beam controller, standalone CTC head outputs, remote-code 3.2 projector.

## 12. Parity and validation plan

- Feature extraction parity: fixed waveform lengths around odd/even mel frame counts; compare `input_features`, `audio_embed_sizes`, and `input_features_mask`.
- Conformer unit parity: one layer with random weights in fp32; then full encoder for 3.3 config. Include final partial 200-frame block.
- Shaw attention parity: compare `pos_attn`, padded mask behavior, and SDPA output for `T < 200`, `T = 200`, `T = 201`.
- Conv module parity: test 1x1 Conv-to-Linear rewrite and depthwise Conv1d with BatchNorm eval.
- Projector parity: random encoder states with `T=1`, `15`, `16`, and long ragged batch. Validate `[B, ceil(T/15)*3, H_text]`.
- Stitch parity: compare masked scatter to indexed copy for contiguous and non-contiguous audio token placements; enforce rejection behavior.
- Prefill logits parity: 3.3-2b bf16 checkpoint, short audio prompt, `logits_to_keep=1`.
- Decode parity: one-token and multi-token cached generation; ensure `input_features` is passed only on first iteration with cache enabled.

Tolerances: fp32 custom op tests `rtol=1e-4, atol=1e-5`; bf16/fp16 full-model regions `rtol=1e-2, atol=1e-2` unless a delegated Granite audit gives stricter targets.

## 13. Performance probes

- CPU preprocessing throughput: waveform seconds/sec for MelSpectrogram plus logmel normalization.
- Encoder throughput by audio duration and batch size: 5 s, 30 s, 2 min; isolate Conformer attention and depthwise conv.
- Projector throughput by number of 15-frame windows; measure Q-Former query length 3 separately.
- Prefill throughput with audio prefix length sweep and text length sweep.
- Decode tokens/sec with and without PEFT LoRA merged/materialized.
- KV cache memory: 2B vs 8B, batch size, generated length.
- Audio embedding cache reuse: projected audio once plus multiple text prompts.
- Stitch overhead: masked scatter fallback vs indexed copy.
- Attention backend comparison: Granite decoder SDPA/FlashAttention/CUTLASS candidate, Conformer local attention custom kernel vs math SDPA.
- End-to-end ASR/AST requests/hour separating preprocessing, encoder/projector, prefill, and decode.

## 14. Skip/defer list

- Training loss and gradient checkpointing.
- Standalone `GraniteSpeechCTCEncoder` CTC logits as a public task head, beyond preserving its internal mid-layer injection for full model parity.
- Remote-code `granite_speech_qformer` from 3.2 until separately audited.
- Text-only Granite decoder full audit, except the integration contract recorded here.
- General boolean `masked_scatter`; only admit guarded audio-token row copy.
- GPU-native MelSpectrogram initially.
- Multi-audio arbitrary interleaving, unless processor emits explicit segment descriptors.
- Multi-GPU tensor parallel and continuous batching.
- Quantization or packed weights not present in native source.
- Beam search parity beyond normal generation-controller compatibility.

## 15. Final implementation checklist

- [ ] Parse `GraniteSpeechConfig`, `GraniteSpeechEncoderConfig`, native 3.3 `blip_2_qformer`, and delegated Granite config.
- [ ] Reject or route 3.2 remote-code `granite_speech_qformer`.
- [ ] Load base weights plus PEFT LoRA adapter when `has_lora_adapter=true`.
- [ ] Define runtime ABI for `input_features [B,T,160]`, `input_features_mask [B,T_audio]`, `input_ids`, and `attention_mask`.
- [ ] Implement Conformer LayerNorm, FFN, block Shaw attention, Conv1d/depthwise Conv1d, BatchNorm eval, residual scales, and mid-layer softmax injection.
- [ ] Implement projector window pad/view, Q-Former composition, and `Linear(1024 -> H_text)`.
- [ ] Implement guarded audio-token indexed row copy replacing `masked_scatter`.
- [ ] Compose Granite LM prefill through `inputs_embeds`.
- [ ] Implement delegated Granite KV-cache decode contract and first-iteration audio forwarding.
- [ ] Add feature extraction parity tests or explicitly keep preprocessing out of runtime.
- [ ] Add encoder-only, projector-only, stitch, prefill, and decode parity tests.
- [ ] Benchmark preprocessing, encoder, projector, prefill, decode, and stitch overhead separately.

