# Transformers Audit: `vibevoice_asr`

Primary target: `VibeVoiceAsrForConditionalGeneration` long-form ASR/diarization-style text generation on CUDA. This is a source/config audit only; no DinoML code or tests were run.

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: microsoft/VibeVoice-ASR-HF
Config source: official Hugging Face config, processor, tokenizer, and generation JSON; source defaults for missing preprocessor_config.json
Source files inspected: vibevoice_asr configuration/modeling/modular/processing, delegated vibevoice_acoustic_tokenizer configuration/modeling/feature_extraction, delegated qwen2 configuration/modeling
Any missing files or assumptions: preprocessor_config.json returned 404; older microsoft/VibeVoice-ASR and MLX/quantized mirrors use non-native or historical field names and are not treated as native-source parity.
```

Details and fetched-source notes are in `_sources/source_notes.md`.

Primary source links:

- [microsoft/VibeVoice-ASR-HF config.json](https://huggingface.co/microsoft/VibeVoice-ASR-HF/resolve/main/config.json)
- [processor_config.json](https://huggingface.co/microsoft/VibeVoice-ASR-HF/resolve/main/processor_config.json)
- [generation_config.json](https://huggingface.co/microsoft/VibeVoice-ASR-HF/resolve/main/generation_config.json)
- [tokenizer_config.json](https://huggingface.co/microsoft/VibeVoice-ASR-HF/resolve/main/tokenizer_config.json)

`modeling_vibevoice_asr.py` is generated from `modular_vibevoice_asr.py`. Runtime import uses the generated file; future Transformers edits should start from the modular source.

## 2. High-level architecture

VibeVoice ASR is not a CTC ASR model. It is an audio-conditioned causal LM:

```text
raw 24 kHz mono waveform
  -> VibeVoiceAcousticTokenizerFeatureExtractor normalization/padding
  -> acoustic ConvNeXt-1D encoder + semantic ConvNeXt-1D encoder
  -> stochastic acoustic latent sampling in source
  -> multimodal projector to Qwen2 hidden size
  -> replace audio placeholder token embeddings
  -> Qwen2 causal LM prefill
  -> KV-cache decode
  -> tokenizer decode and optional JSON-like speaker/timestamp parsing
```

Stage decomposition:

| Stage | Owner | Runtime contract | Cacheability |
| --- | --- | --- | --- |
| Waveform loading/normalization/padding | CPU/data pipeline first | mono float audio, 24 kHz, padded to multiples of 3200 samples | cache normalized waveform if useful |
| Acoustic/semantic tokenizers | GPU/runtime or precomputed feature boundary | two causal Conv1d encoder stacks over `[B, 1, samples]`, output latents `[B, ceil(samples/3200), 64 or 128]` | output audio embeddings can be cached per audio |
| Projector | `vibevoice_asr` | two `Linear -> RMSNorm -> Linear` branches to text hidden 3584, summed | cache `[T_audio, 3584]` |
| Embedding stitch | `vibevoice_asr` | `masked_scatter` into Qwen2 token embeddings where `input_ids == audio_token_id` | no cache; safe indexed-copy rewrite |
| Text decoder | delegated Qwen2 | causal GQA decoder with RoPE and KV cache | standard per-layer K/V cache |
| Postprocess | processor/tokenizer | raw text, parsed speaker dict, or transcription-only | CPU/controller work |

First useful DinoML target: precomputed/processor-produced `input_values`, `input_ids`, `attention_mask`, and `padding_mask` through audio encoders, projector, Qwen2 prefill, and cached text decode. The acoustic/semantic encoder outputs can be validated independently before composing the LM.

## 3. Important config dimensions

Official `microsoft/VibeVoice-ASR-HF` dimensions:

| Field | Value | Source |
| --- | ---: | --- |
| top-level dtype | `bfloat16` | config.json |
| audio token id | 151648 | config.json / processor token `<|box_start|>` |
| audio BOS/EOS ids | 151646 / 151647 | config.json |
| acoustic chunk size | 1440000 samples | config.json, 60 s at 24 kHz |
| audio sample rate | 24000 Hz | processor config / feature extractor source |
| tokenizer hop length | 3200 samples | product of downsampling ratios |
| acoustic latent hidden | 64 | config.json |
| semantic latent hidden | 128 | config.json |
| encoder channels | 1 | config.json |
| encoder `num_filters` | 32 | config.json |
| downsampling ratios | `[2,2,4,5,5,8]` | config.json |
| depths | `[3,3,3,3,3,3,8]` | config.json |
| Conv1d kernel | 7 | config.json |
| ConvNeXt FFN expansion | 4 | config/source |
| ConvNeXt activation | GELU | config/source |
| VAE noise std | 0.625 | config/source |
| text model type | Qwen2 | config.json |
| text hidden size | 3584 | config.json |
| text layers | 28 | config.json |
| text Q heads / KV heads | 28 / 4 | config.json |
| text head dim | 128 inferred | `3584 / 28` |
| text intermediate | 18944 | config.json |
| text vocab size | 152064 | config.json |
| text max positions | 131072 | config.json |
| RoPE theta/type | 1000000.0 / default | config.json |
| sliding attention | disabled, all layers `full_attention` | config.json |
| generation cache | `use_cache=true` | generation config |

Representative sweep:

| Config basis | Native source status | Audio encoders | Text decoder | Notable variation |
| --- | --- | --- | --- | --- |
| `microsoft/VibeVoice-ASR-HF` main | Native current target | acoustic 64 + semantic 128, same ConvNeXt topology, bf16 | Qwen2 28x3584, 28 Q heads, 4 KV heads, vocab 152064 | required target |
| `microsoft/VibeVoice-ASR-HF` refs/pr/5 | Same native shape in search result | same as main | same as main | no operator-significant delta observed from indexed snippet |
| source defaults | Native random config | acoustic defaults 64, semantic default overridden to 128 by `VibeVoiceAsrConfig`; Qwen2 defaults if omitted | Qwen2 default 32x4096, vocab 151936 | only useful for tiny/random tests, not checkpoint parity |
| `microsoft/VibeVoice-ASR` / MLX mirrors | Historical/export surface | keys use `acoustic_tokenizer_config` / `semantic_tokenizer_config`, not current native fields | often `decoder_config`; may include diffusion/quantization metadata | route to separate audit, do not admit as this source basis |

## 3a. Family variation traps

- This family is ASR by generation, not CTC. There is no CTC head or CTC blank/logit ABI in native `vibevoice_asr`.
- The neural body is composite: two `vibevoice_acoustic_tokenizer_encoder` AutoModels plus delegated Qwen2 causal LM. DinoML should compose separate Qwen2 coverage rather than re-auditing every decoder detail here.
- The processor expands each `<|box_start|>` audio placeholder into `ceil(valid_samples / 3200)` repeated audio tokens. The model then flattens valid audio features across the batch and uses `masked_scatter`.
- The model includes a `get_placeholder_mask` helper that validates feature/token count, but `forward` directly uses `masked_scatter` without calling that helper. DinoML should add its own count guard for the indexed-copy rewrite.
- Source adds random VAE noise to acoustic latents inside `get_audio_features` even under `torch.no_grad()`. Deterministic inference parity likely needs an admission decision: reproduce RNG exactly, expose `sample=False` only if source changes, or route to a deterministic preprocessing/cache boundary.
- Audio chunking through tokenizers uses causal Conv1d padding caches, not attention KV cache. Chunk size must be a multiple of hop length.
- Feature extractor only supports `return_tensors="pt"` and mono audio. It normalizes amplitude to target dB FS and clamps if peak exceeds 1.0.
- Qwen2 config uses GQA: 28 query heads and 4 KV heads, so cache tensors are smaller than query-head tensors before repeat expansion.
- Qwen2 q/k/v projections have bias; o projection and LM head are biasless. Qwen2 MLP is SwiGLU with biasless projections.
- The text config includes `use_mrope=false`; pinned Qwen2 source does not read `use_mrope`. Treat it as ignored for this source path.
- Historical configs may contain `rope_theta` or `rope_scaling` instead of current `rope_parameters`; import should normalize or reject explicitly.
- Source layout is audio NCL `[B,C,T]` inside Conv1d encoders and sequence `[B,T,H]` at latent/projector/LM boundaries. No NHWC-style layout translation is relevant; keep no-layout guards around Conv1d transposes and GLU/channel axes.

## 4. Operator coverage checklist

Tensor/layout ops:

- `unsqueeze`, `transpose`, `permute`, `view`/`reshape`, `contiguous`, `cat`, `split`, `expand_as`.
- `pad` causal left-padding for Conv1d, padding mask length compression, boolean equality masks, `sum`, `ceil`, `arange`, boolean indexing.
- `masked_scatter` source behavior, with DinoML replacement as guarded indexed row copy.
- Runtime shape arithmetic for `ceil(valid_samples / hop_length)` and chunk concatenation.

Audio preprocessing-coupled ops:

- Audio list normalization to mono rank-1 tensors.
- Sampling rate check at 24 kHz.
- RMS amplitude normalization: `x *= 10 ** (-25 / 20) / (rms + eps)`, then peak clamp if `max(abs(x)) > 1`.
- Right padding, padding mask creation, `pad_to_multiple_of=3200`, add channel dimension to `[B,1,T]`.
- Text prompt placeholder rewrite for `<|AUDIO_DURATION|>` and repeated `<|box_start|>` tokens.

Acoustic/semantic encoder primitives:

- Causal Conv1d stem: `Conv1d(1 -> 32, kernel=7, stride=1)`.
- Six downsampling Conv1d layers: channels `32->64->128->256->512->1024->2048`, kernels `2*ratio`, strides `[2,2,4,5,5,8]`.
- ConvNeXt-1D blocks: RMSNorm over channel dimension via transposes, depthwise causal Conv1d `groups=hidden_size`, kernel 7, GELU FFN `Linear(H -> 4H) -> GELU -> Linear(4H -> H)`, layer-scale gamma and residual add.
- Encoder head: causal Conv1d `2048 -> latent_hidden`, kernel 7, output permute to `[B,T_latent,H_latent]`.
- Conv padding cache state per causal Conv1d: `[B, in_channels, left_pad]`.

Projector primitives:

- Acoustic branch: `Linear(64 -> 3584) -> RMSNorm(3584) -> Linear(3584 -> 3584)`.
- Semantic branch: `Linear(128 -> 3584) -> RMSNorm(3584) -> Linear(3584 -> 3584)`.
- Elementwise add of branch outputs.

Text decoder primitives:

- Qwen2 token embedding and untied LM head `Linear(3584 -> 152064, bias=False)`.
- 28 decoder blocks: RMSNorm, Q/K/V linears with split widths `3584`, `512`, `512`, RoPE, causal GQA attention, O linear `3584 -> 3584`, RMSNorm, SwiGLU MLP `3584 -> 18944 -> 3584`.
- Final RMSNorm and last-token/full logits handling via Qwen2 `logits_to_keep`.

Attention/cache ops:

- Qwen2 causal self-attention, full attention layers only for official config.
- KV cache per layer stores K/V after RoPE and before KV repeat, shape `[B,4,S,128]`.
- Audio Conv1d padding cache for chunked encoding, separate from text KV cache.

Postprocess/tokenizer ops:

- Qwen2 tokenizer decode.
- Optional parse of generated JSON-like segment list with fields such as `Start`, `End`, `Speaker`, `Content`.
- `transcription_only` joins `Content` fields with spaces.

## 5. Layer/block breakdown

Audio feature extraction:

```text
audio list -> float32 torch tensors
require each waveform rank == 1
if normalize:
  rms = sqrt(mean(x**2))
  x = x * 10 ** (target_dB_FS / 20) / (rms + eps)
  if max(abs(x)) > 1: x = x / (max(abs(x)) + eps)
pad right to batch max and/or multiple of 3200
padding_mask marks original samples
input_values = input_values[:, None, :]  # [B,1,T]
```

One acoustic/semantic encoder:

```text
x [B,1,T]
x = causal Conv1d(1 -> 32, k=7, s=1)(x)
repeat depth[0]=3:
  x = ConvNeXt1dBlock(H=32)(x)
for stage i in 0..5:
  Hout = 32 * 2**(i+1)
  x = causal Conv1d(Hin -> Hout, k=2*ratio[i], stride=ratio[i])(x)
  repeat depth[i+1]:
    x = ConvNeXt1dBlock(H=Hout)(x)
x = causal Conv1d(2048 -> latent_hidden, k=7)(x)
latents = x.permute(0,2,1)  # [B,T/3200,latent_hidden]
```

ConvNeXt1d block:

```text
residual = x
y = RMSNorm(x.transpose(1,2)).transpose(1,2)
y = depthwise causal Conv1d(H -> H, k=7, groups=H)(y)
x = residual + y * gamma[:, None]

residual = x
y = RMSNorm(x.transpose(1,2))
y = Linear(4H -> H)(GELU(Linear(H -> 4H)(y))).transpose(1,2)
x = residual + y * ffn_gamma[:, None]
```

Audio feature assembly:

```text
for chunk in split(input_values, acoustic_tokenizer_chunk_size, dim=-1):
  acoustic_latents += acoustic_encoder(chunk, padding_cache=acoustic_cache, use_cache=True).latents
  semantic_latents += semantic_encoder(chunk, padding_cache=semantic_cache, use_cache=True).latents
acoustic_latents = cat(chunks, dim=1)
semantic_latents = cat(chunks, dim=1)
acoustic_latents += vae_std * randn([B])[:,None,None] * randn_like(acoustic_latents)
features = projector(acoustic_latents, semantic_latents)
if padding_mask:
  num_audio_tokens = ceil(padding_mask.sum(-1) / 3200)
  features = features[arange(max_tokens) < num_audio_tokens[:,None]]
```

Embedding stitch and decoder:

```text
inputs_embeds = qwen2.embed_tokens(input_ids)
audio_embeds = get_audio_features(...).pooler_output  # flattened valid rows if padding_mask is set
mask = (input_ids == audio_token_id).unsqueeze(-1)
inputs_embeds = inputs_embeds.masked_scatter(mask, audio_embeds)
return qwen2_lm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, past_key_values=past_key_values)
```

Qwen2 block:

```text
residual = x
x = RMSNorm(x)
q = Linear(3584 -> 3584, bias=True)(x).view(B,S,28,128).transpose(1,2)
k = Linear(3584 -> 512, bias=True)(x).view(B,S,4,128).transpose(1,2)
v = Linear(3584 -> 512, bias=True)(x).view(B,S,4,128).transpose(1,2)
q,k = RoPE(q,k,cos,sin)
k,v = cache.update(k,v,layer_idx) if cache
x = causal_attention(q,k,v,mask, repeat_kv=7)
x = residual + Linear(3584 -> 3584, bias=False)(x)
residual = x
x = RMSNorm(x)
x = residual + down_proj(silu(gate_proj(x)) * up_proj(x))
```

## 6. Attention requirements

Audio encoders:

- No attention. The acoustic and semantic branches are causal Conv1d/ConvNeXt stacks.
- Streaming/chunked audio state is a per-layer convolution padding cache, not KV cache.
- Each cache layer stores the previous `left_pad` samples/channels and returns them for concatenation before the current chunk.

Text decoder:

- Causal self-attention only; no cross-attention because audio is stitched into token embeddings before Qwen2.
- GQA: 28 query heads, 4 KV heads, 7 query groups per KV head, head dim 128.
- Query length is full fused audio/text prefill, then 1 or more text decode tokens.
- K/V cache tensors are `[B,4,S,128]` per layer before repeat expansion to 28 heads.
- Cached keys are stored after RoPE because Qwen2 updates cache after applying rotary embeddings.
- Masking uses Qwen2 causal mask from `attention_mask`; official config does not enable sliding-window attention.
- Eager fallback repeats KV, computes `q @ k.T * head_dim**-0.5`, adds mask, softmaxes in fp32, casts to query dtype, then multiplies by V.
- Source advertises SDPA/FlashAttention support through the Qwen2 attention interface. FlashAttention/GQA prefill and decode are valid targets if RoPE/cache/mask ordering is preserved.

Generation:

- `prepare_inputs_for_generation` forwards `input_values`, `padding_mask`, and `acoustic_tokenizer_chunk_size` only when `is_first_iteration=True`.
- Cached decode should not rerun audio encoders. With no cache, generation may include audio inputs again according to controller behavior.

## 7. Position encoding and custom math

Qwen2 RoPE:

```python
def qwen2_rope(q, k, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    def rotate_half(x):
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., :x.shape[-1] // 2]), dim=-1)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin
```

Default inverse frequencies:

```text
inv_freq = 1 / (rope_theta ** (arange(0, head_dim, 2) / head_dim))
freqs = inv_freq @ position_ids
cos/sin computed in fp32, scaled by attention_scaling, then cast to hidden dtype
```

Audio length math:

```text
hop_length = prod([2,2,4,5,5,8]) = 3200
processor_audio_tokens = ceil(valid_samples / 3200)
model_valid_audio_tokens = ceil(padding_mask.sum(-1) / 3200)
```

Conv cache update:

```python
shortfall = max(0, left_pad - current_T)
padding_states = cat([cache[:, :, -shortfall:], x], dim=-1) if shortfall > 0 else x[:, :, -left_pad:]
current_cache = cache.clone()
cache.copy_(padding_states)
padded_x = cat([current_cache, x], dim=-1)
```

The Conv cache has static-address mutation semantics in source when not compiling; a DinoML streaming ABI should make those cache tensors explicit session state.

## 8. Preprocessing and input packing

Waveform contract:

- Input audio must be 24 kHz. Mismatched `sampling_rate` raises; missing sampling rate logs a warning.
- Each audio example must be mono rank 1.
- Feature extractor returns `input_values [B,1,T_padded]` and `padding_mask [B,T_padded]`.
- Default processor pads to a multiple of 3200 samples.

Prompt/placeholder contract:

- Processor default text kwargs use left padding, `add_special_tokens=False`, `return_tensors="pt"`.
- `apply_transcription_request` builds a chat conversation with an audio item and optional prompt text, then uses the chat template.
- Processor replaces `<|AUDIO_DURATION|>` with seconds formatted to two decimals.
- Processor replaces each `<|box_start|>` placeholder with `ceil(valid_samples / 3200)` repeated placeholders before tokenization.
- Token IDs: `<|object_ref_start|>` 151646, `<|object_ref_end|>` 151647, `<|box_start|>` 151648 per official config/processor.

Runtime stitch:

- Source uses row-major boolean scatter over all samples, not per-sample segment descriptors.
- With `padding_mask`, `get_audio_features` flattens valid audio rows by boolean indexing. Therefore `audio_embeds` rank is `[sum_i ceil(valid_i/3200), 3584]`.
- First DinoML admission should require processor-expanded prompts and guard:
  - `sum(input_ids == audio_token_id) == audio_embeds.shape[0]`
  - per-sample placeholder counts match `ceil(padding_mask.sum(-1)/3200)` if batch metadata is available
  - `audio_embeds.shape[-1] == text_hidden_size`

Postprocess:

- `decode(return_format="raw")` delegates tokenizer decode.
- `return_format="parsed"` strips a leading `assistant`, expects a JSON array, and validates `Content` plus numeric `Start`/`End` fields.
- `return_format="transcription_only"` joins `Content` fields. This is CPU/controller work, not graph runtime.

## 9. Graph rewrite / lowering opportunities

### Rewrite: audio placeholder `masked_scatter` to indexed row copy

Source pattern:

```text
mask = (input_ids == audio_token_id).unsqueeze(-1)
inputs_embeds = inputs_embeds.masked_scatter(mask, audio_embeds)
```

Replacement:

```text
positions = nonzero(input_ids == audio_token_id) in row-major order
inputs_embeds[positions, :] = audio_embeds
```

Preconditions:

- `input_ids` is present.
- Placeholder count equals `audio_embeds.shape[0]`.
- Hidden dimension equals Qwen2 hidden size.
- For stricter first integration, placeholder runs are contiguous per sample and match the processor-computed audio token count.

Failure cases:

- Caller supplies arbitrary `inputs_embeds` without `input_ids`.
- Multiple/interleaved audio regions without segment metadata.
- Random source acoustic latent sampling makes `audio_embeds` nondeterministic unless RNG is controlled.

Parity sketch: compare PyTorch `masked_scatter` and indexed copy for one/batched samples, ragged valid audio lengths, and an intentional count mismatch.

### Rewrite: causal Conv1d no-cache path to explicit left-pad Conv1d

Source pattern:

```text
pad(x, (left_pad, 0)) -> Conv1d
```

Replacement: provider Conv1d with explicit causal-left padding or pre-pad plus Conv1d.

Preconditions:

- `left_pad = (kernel_size - 1) * dilation - (stride - 1)` is nonnegative.
- Preserve PyTorch Conv1d cross-correlation weight layout `[out_channels, in_channels/groups, kernel]`.
- For downsampling convs, output length must match PyTorch Conv1d with source padding.

Failure cases:

- Streaming cache path needs explicit state tensors and cannot be folded into a stateless pad.
- Dynamic chunk sizes not divisible by hop length should be rejected like source.

### Rewrite: ConvNeXt pointwise FFN to GEMM epilogues

Source pattern:

```text
RMSNorm([B,T,H]) -> Linear(H -> 4H) -> GELU -> Linear(4H -> H) -> scale -> residual
```

Replacement: RMSNorm plus two GEMMs, with GELU and residual scale fused where provider supports it.

Preconditions:

- Inference dropout-free path.
- Preserve transpose boundaries: FFN runs on `[B,T,H]`, mixer Conv1d runs on `[B,H,T]`.
- `ffn_expansion=4`, activation GELU.

Failure cases:

- Training/dropout or captured intermediate outputs.

### Rewrite: Qwen2 QKV sibling projections

Source pattern:

```text
q = Linear(3584 -> 3584)
k = Linear(3584 -> 512)
v = Linear(3584 -> 512)
```

Replacement:

```text
PackedLinear(3584 -> 4608) -> split [3584,512,512]
```

Preconditions:

- Same normalized input tensor and dtype.
- All three projections have bias.
- Split order is Q, K, V and K/V widths use `num_key_value_heads * head_dim`.

Failure cases:

- Any config with different projection bias behavior or explicit `head_dim` making `hidden_size != num_attention_heads * head_dim` should recompute split widths from config.

### Rewrite: last-token-only LM head

Source pattern: Qwen2 supports `logits_to_keep`; default may compute full logits.

Replacement: project only final token(s) for sampling/decode.

Preconditions:

- No loss/teacher-forcing path.
- Generation controller only needs next-token logits.
- Preserve full logits for validation modes that request them.

## 10. Kernel fusion candidates

Highest priority:

- Qwen2 GQA attention with RoPE and KV cache. This dominates decode and long fused prefill.
- Qwen2 SwiGLU MLP and RMSNorm. Large GEMMs at hidden 3584 and intermediate 18944 dominate decoder compute.
- Audio placeholder indexed copy. It avoids admitting a broad boolean scatter op.
- ConvNeXt encoder causal/depthwise Conv1d plus RMSNorm/FFN. Two encoders run over long audio and are independently optimizable.

Medium priority:

- Conv1d downsampling provider path for strides 2,2,4,5,5,8 with causal padding.
- Projector GEMMs `64/128 -> 3584 -> 3584`, including RMSNorm and branch sum.
- Conv padding-cache state ABI for chunked 60 s processing and future streaming.
- Last-token-only logits for generation.

Lower priority:

- GPU audio normalization/padding. CPU preprocessing is small relative to encoders/decoder.
- JSON postprocessing acceleration.
- Historical/remote-code VibeVoice-ASR training or diffusion-head configs.
- General arbitrary `masked_scatter`.

## 11. Runtime staging plan

1. Parse native `VibeVoiceAsrConfig` and nested acoustic/semantic encoder and Qwen2 configs; reject historical `acoustic_tokenizer_config`/`decoder_config` variants for this path.
2. Load dense bf16 weights, preserving untied Qwen2 embeddings/LM head and the two separate audio encoder branches.
3. Define runtime ABI for processor-produced `input_values [B,1,T]`, `padding_mask [B,T]`, `input_ids [B,S]`, and `attention_mask [B,S]`.
4. Validate one acoustic tokenizer encoder on fixed audio lengths, including causal padding and downsampling length.
5. Validate semantic encoder separately; same topology but latent width 128.
6. Add deterministic policy for source acoustic latent noise, then validate `get_audio_features` and projector output.
7. Implement guarded indexed audio embedding stitch and compare against source `masked_scatter`.
8. Compose Qwen2 prefill from `inputs_embeds`.
9. Add cached Qwen2 decode and generation first-iteration audio forwarding semantics.
10. Add optimized Qwen2 attention/MLP, Conv1d provider paths, and audio feature caching.

Initial stubs allowed: tokenizer/chat template, audio file I/O, JSON parsing, sampling controller, and GPU-native waveform preprocessing.

## 12. Parity and validation plan

- Feature extractor parity: mono validation, sampling-rate rejection, normalization/clamp, right padding to 3200, `padding_mask`.
- Conv1d causal padding parity: small lengths around `left_pad`, exact multiples/nonmultiples of stride, no-cache and cache paths.
- Encoder length parity: input lengths around 3200 boundaries and 60 s chunk boundaries; validate latent length `ceil(valid/3200)` as source observes after padding/chunking.
- Single ConvNeXt block parity in fp32, then bf16 tolerance.
- Full acoustic encoder and semantic encoder parity separately.
- Projector parity with synthetic acoustic `[B,T,64]` and semantic `[B,T,128]`.
- Noise policy parity: with controlled RNG, compare source stochastic acoustic latent sampling; otherwise explicitly test deterministic admission mode against a modified/external feature boundary.
- Stitch parity: source masked scatter versus indexed row copy, including batched ragged audio counts and mismatch rejection.
- Qwen2 prefill logits parity for fused text/audio embeddings.
- Cached decode parity: one-token and multi-token decode; assert audio encoders are not rerun after first cached iteration.
- End-to-end smoke once weights are available: processor `apply_transcription_request` -> generate -> `decode(return_format="parsed")`.

Suggested tolerances: fp32 component tests `rtol=1e-4, atol=1e-5`; bf16/fp16 full regions `rtol=1e-2, atol=1e-2`, with attention logits and stochastic audio handled separately.

## 13. Performance probes

- CPU preprocessing throughput: seconds of audio normalized/padded per second.
- Acoustic encoder throughput by duration: 5 s, 60 s, and multi-chunk long audio.
- Semantic encoder throughput by duration, separated from acoustic branch.
- Conv padding-cache overhead for chunked processing versus single-pass full audio.
- Projector throughput by audio token count.
- Audio embedding stitch overhead: `masked_scatter` baseline versus indexed copy.
- Qwen2 prefill throughput versus fused sequence length: text-only, 1 min audio, long audio.
- Qwen2 decode tokens/sec with populated cache.
- KV cache memory: 28 layers, 4 KV heads, head dim 128, batch/sequence sweep.
- Audio feature cache memory: `[sum_audio_tokens, 3584]` bf16.
- Attention backend comparison: eager/SDPA/FlashAttention for Qwen2 GQA.
- Conv1d backend comparison: direct Conv1d provider versus im2col/GEMM for stride/downsampling layers.
- End-to-end requests/hour split by preprocessing, audio encoders, prefill, decode, and postprocess.

## 14. Skip/defer list

- Training, labels, gradient checkpointing, and stochastic training objectives.
- CTC ABI and CTC decoding; native `vibevoice_asr` does not implement CTC.
- Historical `VibeVoiceForASRTraining`, diffusion heads, and older remote-code/export configs.
- General boolean scatter; admit only guarded audio-token indexed row copy.
- GPU-native audio normalization as a first target.
- Beam/speculative/advanced generation controllers; source generation config is greedy by default.
- Multi-GPU tensor parallel plans.
- Quantized MLX/bitsandbytes mirror loading; separate provider/loading audit.
- Arbitrary multi-audio interleaving without explicit segment metadata.

## 15. Final implementation checklist

- [ ] Parse native `VibeVoiceAsrConfig` plus nested acoustic, semantic, and Qwen2 configs.
- [ ] Reject historical/non-native config keys for this path or route to separate audit.
- [ ] Load dense bf16 weights for two audio encoders, projector, Qwen2 embedding, decoder, and LM head.
- [ ] Define ABI for `input_values`, `padding_mask`, `input_ids`, `attention_mask`, and optional `acoustic_tokenizer_chunk_size`.
- [ ] Implement VibeVoice acoustic tokenizer encoder causal Conv1d and ConvNeXt blocks.
- [ ] Implement Conv1d padding-cache state ABI for chunked audio.
- [ ] Implement projector `Linear/RMSNorm/Linear` acoustic and semantic branches.
- [ ] Decide and document deterministic handling of acoustic latent noise.
- [ ] Implement guarded indexed audio embedding stitch.
- [ ] Compose Qwen2 causal LM prefill from fused `inputs_embeds`.
- [ ] Implement Qwen2 GQA KV-cache decode and first-iteration audio forwarding.
- [ ] Add no-layout-translation guards around audio NCL Conv1d regions.
- [ ] Add parity tests for preprocessing, Conv1d cache, encoders, projector, stitch, prefill, and decode.
- [ ] Benchmark preprocessing, audio encoders, projector/stitch, prefill, decode, and cache memory separately.
