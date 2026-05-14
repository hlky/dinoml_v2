# AudioFlamingo3 Transformers Audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: nvidia/audio-flamingo-3-hf
Config source: HF config/processor/tokenizer/generation JSON plus source defaults
Primary runtime target: audio/text-to-text conditional generation, first target = audio encoder + projector + Qwen2 prefill/decode logits
```

Source files inspected:

- Local pinned source: `X:/H/transformers/src/transformers/models/audioflamingo3/configuration_audioflamingo3.py`
- Local pinned source: `X:/H/transformers/src/transformers/models/audioflamingo3/modular_audioflamingo3.py`
- Local pinned source: `X:/H/transformers/src/transformers/models/audioflamingo3/modeling_audioflamingo3.py`
- Local pinned source: `X:/H/transformers/src/transformers/models/audioflamingo3/processing_audioflamingo3.py`
- Neighbor/composed source: `qwen2/configuration_qwen2.py`, `qwen2/modeling_qwen2.py`, `whisper/feature_extraction_whisper.py`, and auto mappings.

External primary config URLs:

- [`nvidia/audio-flamingo-3-hf/config.json`](https://huggingface.co/nvidia/audio-flamingo-3-hf/blob/main/config.json)
- [`nvidia/audio-flamingo-3-hf/processor_config.json`](https://huggingface.co/nvidia/audio-flamingo-3-hf/raw/main/processor_config.json)
- [`nvidia/audio-flamingo-3-hf/tokenizer_config.json`](https://huggingface.co/nvidia/audio-flamingo-3-hf/raw/main/tokenizer_config.json)
- [`nvidia/audio-flamingo-3-hf/generation_config.json`](https://huggingface.co/nvidia/audio-flamingo-3-hf/raw/main/generation_config.json)
- [`nvidia/music-flamingo-hf/config.json`](https://huggingface.co/nvidia/music-flamingo-hf/blob/main/config.json)

Missing files or assumptions:

- `modeling_audioflamingo3.py` is generated from `modular_audioflamingo3.py`; the generated file is the executable source basis, but future upstream edits should be read in the modular file.
- `nvidia/audio-flamingo-3/raw/main/config.json` returned a Git LFS pointer rather than JSON in this environment.
- The prompt asks for 3-5 representative checkpoint configs. I found one main native checkpoint, one closely related music specialization, and historical/remote-code snapshots. They do not provide many structurally distinct native variants; this is a coverage gap, not evidence that variants do not exist.
- `think/config.json` in the HF repo is a legacy/remote-code Llava-style config, not native `model_type="audioflamingo3"` behavior.

## 2. High-level architecture

AudioFlamingo3 is an audio encoder + multimodal projector + causal language model. The audio encoder is Whisper-like, the projector is a two-layer MLP, and the decoder is composed via `AutoModelForCausalLM` from `text_config`, normally Qwen2.

```text
raw mono audio -> WhisperFeatureExtractor log-mel windows
  -> AudioFlamingo3Encoder conv/encoder/avg-pool/LayerNorm
  -> MLP projector to Qwen2 hidden size
  -> replace <sound> token embeddings in text sequence
  -> Qwen2 causal LM prefill/decode
  -> logits/sampling
```

Stage decomposition:

- CPU/data pipeline: audio load, 16 kHz mono validation, 30 s chunking, log-mel extraction, tokenization, chat-template application, `<sound>` expansion.
- Audio runtime branch: Conv1d front end, bidirectional Transformer encoder, avg pool, LayerNorm.
- Projector branch: Linear 1280->3584, GELU, Linear 3584->3584 for the main configs.
- Prefix construction: text token embedding plus audio-row replacement at placeholder positions.
- Prefill: Qwen2 causal LM over the mixed text/audio embedding sequence.
- Decode: language-model-only token loop when cache is enabled; audio tensors are forwarded only on first generation iteration or when cache is disabled.

Independently stageable/cacheable regions:

- Log-mel extraction can remain a CPU/data-pipeline contract at first.
- Audio encoder + projector outputs can be cached per audio window/prompt before LM prefill.
- Qwen2 decoder can be validated separately with pure text inputs.
- Embedding stitch can be validated as a bounded indexed row-copy contract before admitting any general boolean scatter.

## 3. Important config dimensions

Main native checkpoint dimensions from `nvidia/audio-flamingo-3-hf/config.json`:

| Field | Value | Source |
| --- | ---: | --- |
| audio hidden size | 1280 | config.json |
| audio layers | 32 | config.json |
| audio attention heads | 20 | config.json |
| audio head dim | 64 | inferred from 1280/20, source requires divisibility |
| audio FFN | 5120 | config.json |
| mel bins | 128 | config.json and processor_config |
| max source positions | 1500 | config.json |
| audio conv schedule | Conv1d stride 1 then Conv1d stride 2, AvgPool1d stride 2 | source |
| projector | 1280->3584 GELU 3584->3584, bias true | config/source |
| text model | Qwen2 causal LM | config/source |
| text hidden size | 3584 | config.json |
| text layers | 28 | config.json |
| text attention heads | 28 | config.json |
| text KV heads | 4 | config.json |
| text head dim | 128 | inferred from 3584/28 |
| text GQA repeat | 7 groups per KV head | inferred from 28/4 |
| text intermediate size | 18944 | config.json |
| vocab size | 151672 | config.json |
| max positions | 32768 | config.json |
| RoPE | default, theta 1000000.0 | config.json |
| text sliding window | disabled, `sliding_window=null`, all 28 `full_attention` | config.json |
| config `use_cache` | false | config.json; generation may still pass `use_cache` explicitly |
| audio token id | 151669 | config.json/processor |
| BOS/EOS/PAD | 151670 / 151645 / 151671 | config/generation |

Representative config sweep:

| Checkpoint/config | Native source scope? | Main differences |
| --- | --- | --- |
| `nvidia/audio-flamingo-3-hf` main | Yes | `dtype=float32`, text `model_max_length=32768`, Qwen2 28L/3584, audio 32L/1280, `use_cache=false`. |
| `nvidia/audio-flamingo-3-hf` older HF revisions | Yes, historical | Visible revisions show `dtype=bfloat16`, `model_max_length=8192`, and older audio `init_std`; strict current config declares `initializer_range`. |
| `nvidia/music-flamingo-hf` | Yes, same class | Same operator structure and dimensions, `dtype=bfloat16`, `model_max_length=8192`, legacy `rope_theta` alongside `rope_parameters`. |
| `audio-flamingo-3-hf/think/config.json` | No for native source | Remote-code/Llava-style config with vision/sound/speech towers, quant fields, `model_type=llava_llama`; route to separate audit/reject for this report. |
| `nvidia/audio-flamingo-3` original repo config | Not inspected as JSON | Raw URL returned Git LFS pointer; access/resolution would be needed before using it as a native config. |

## 3a. Family variation traps

- The native source only implements one audio modality branch despite historical configs mentioning vision, speech, sound, quantization, and Llava fields.
- The processor expands one `<sound>` string into many adjacent `<sound>` tokens before tokenization; the model later checks only total placeholder count, not original per-window grouping.
- `audio_features` is flattened across all valid post-pool rows by boolean indexing; first DinoML lowering should require processor-generated contiguous placeholder runs in the same order.
- Audio encoder positions are a learned/frozen table of length 1500 after Conv2 stride-2. The model assumes 3000 log-mel frames per 30 s chunk -> 1500 post-conv frames.
- Final audio token count per window is `((audio_feature_len - 1)//2 + 1 - 2)//2 + 1`; for a full 3000-frame chunk this is 750.
- The audio encoder uses NCL Conv1d and explicit permutes to BLC attention. A channel-last layout pass must guard this region carefully; source semantics are `[B, mel, time]` then `[B, time, hidden]`.
- Qwen2 has GQA: Q heads = 28, KV heads = 4, head_dim = 128. Do not allocate cache as 28 KV heads.
- Qwen2 `use_cache=false` is in checkpoint config, but source supports `DynamicCache`. Generation behavior depends on the generation controller and kwargs.
- Text `layer_types` are all `full_attention`; Qwen2 source can do sliding-window layers, but this checkpoint disables them. DinoML should reject/route `use_sliding_window=true` variants until separately audited.
- Audio attention pre-scales Q before attention and passes `scaling=1.0` to the backend for parity; fused attention must preserve this order.
- Qwen2 eager attention softmax upcasts to fp32 then casts back; audio eager attention does not explicitly request fp32 softmax dtype.
- `logits_to_keep` slices LM hidden states before the output projection. Last-token-only logits are semantically visible.
- Tokenizer config contains Qwen2-VL-style vision/video special tokens, but native AudioFlamingo3 code only consumes `<sound>` through `audio_token_id`.

## 4. Operator coverage checklist

Tensor/layout ops:

- Reshape/view, transpose/permute, contiguous materialization.
- Boolean/length mask construction from `sum`, `arange`, `<`, unsqueeze, expand.
- Boolean mask flatten gather for valid audio rows.
- Bounded placeholder replacement: row-copy from `[N_audio, H_text]` into `[B, S, H_text]` at `<sound>` positions. Source uses `masked_scatter`; DinoML should lower a stricter indexed copy with count/order guards.
- Left-padded text `attention_mask` handling and causal mask construction.

Audio preprocessing-coupled ops:

- CPU or data-pipeline STFT/log-mel: Hann window, `n_fft=400`, `hop_length=160`, 128 mel filters, log10 clamp, `(log_spec+4)/4`.
- Chunking: 16 kHz mono waveform, 480000 samples per 30 s window, max 20 windows from `max_audio_len=600`.
- Feature tensor: `input_features` `[num_windows_total, 128, 3000]`, `input_features_mask` `[num_windows_total, 3000]`.

Neural primitives:

- Conv1d 128->1280, kernel 3, padding 1, stride 1, bias true.
- Conv1d 1280->1280, kernel 3, padding 1, stride 2, bias true.
- GELU after each audio conv.
- Audio encoder LayerNorm and FFN Linear 1280->5120, GELU, Linear 5120->1280.
- AvgPool1d kernel 2 stride 2 over time after audio encoder.
- Final audio LayerNorm over hidden 1280.
- Projector Linear 1280->3584 with bias, GELU, Linear 3584->3584 with bias.
- Qwen2 RMSNorm over hidden 3584.
- Qwen2 attention Linear q 3584->3584 bias true, k/v 3584->512 bias true, o 3584->3584 bias false.
- Qwen2 MLP: gate/up 3584->18944 bias false, SiLU, multiply, down 18944->3584 bias false.
- LM head 3584->151672 bias false.

Attention primitives:

- Audio encoder self-attention: noncausal MHA, 20 heads, head dim 64, q/v/out bias true, k bias false, additive bidirectional padding mask.
- Qwen2 decoder self-attention: causal GQA, 28 Q heads, 4 KV heads, head dim 128, RoPE on Q/K, fp32 softmax in eager path, optional FlashAttention/SDPA backend.

Position/rotary/custom math:

- Audio learned positional embedding `[1500,1280]`, frozen, added after convs.
- Qwen2 default RoPE theta 1e6, cos/sin generated from `position_ids` and shared across layers.

Generation/cache ops:

- Qwen2 `DynamicCache` per layer stores post-RoPE keys and values shaped `[B, 4, T_cache, 128]`.
- Cache update happens after RoPE. Repeat to 28 heads is an attention-kernel view/compute concern, not cache storage.
- Audio tensors are needed for first generation iteration/prefill or when cache disabled, not for ordinary one-token decode.

Packed/varlen metadata ops:

- No `cu_seqlens` or packed varlen descriptors in native source. The processor emits fixed 30 s feature windows plus masks.
- Variable per-sample audio length is represented by `per_sample_windows`, feature masks, repeated placeholders, and a flattened valid-row tensor.

Quantized/packed weight metadata:

- Native HF source does not implement a custom quantized format. Historical `think/config.json` has quantization/training fields; reject for this native-source target unless a separate remote-code path is explicitly chosen.

## 5. Layer/block breakdown

Audio preprocessing and packing:

```text
for each user audio:
  split waveform into <=20 windows of <=480000 samples
  WhisperFeatureExtractor -> [windows, 128, 3000] log-mel
  input_features_mask -> [windows, 3000]
  expand one <sound> placeholder into sum(post_pool_len(window)) tokens
```

Audio encoder:

```text
x: [W, 128, 3000]
mask0: [W, 3000]
conv_len = (T - 1)//2 + 1
mask1 = arange(conv_len) < ((mask0.sum(-1) - 1)//2 + 1)
x = GELU(Conv1d(128 -> 1280, k=3, pad=1, stride=1)(x))
x = GELU(Conv1d(1280 -> 1280, k=3, pad=1, stride=2)(x))  # [W,1280,1500]
x = x.permute(0, 2, 1)                                   # [W,1500,1280]
x = x + embed_positions[0:1500]
for 32 layers:
  residual = x
  x = LayerNorm(x)
  x = AudioMHA(x, bidirectional_padding_mask)
  x = residual + x
  residual = x
  x = LayerNorm(x)
  x = Linear(1280 -> 5120)(x)
  x = GELU(x)
  x = Linear(5120 -> 1280)(x)
  x = residual + x
  if fp16: clamp to finite max - 1000
x = AvgPool1d(k=2,stride=2)(x.permute(0,2,1)).permute(0,2,1) # [W,750,1280]
x = LayerNorm(x)
```

Projector and flatten:

```text
audio = Linear(1280 -> 3584, bias)(x)
audio = GELU(audio)
audio = Linear(3584 -> 3584, bias)(audio)
post_lengths = ((input_features_mask.sum(-1) - 1)//2 + 1 - 2)//2 + 1
audio_rows = audio[arange(750) < post_lengths]  # [sum(post_lengths), 3584]
```

Mixed prompt and Qwen2:

```text
inputs_embeds = embed_tokens(input_ids)             # [B,S,3584]
mask = input_ids == 151669                          # [B,S]
assert mask.sum() == audio_rows.shape[0]
inputs_embeds[mask] = audio_rows in row-major mask order
Qwen2ForCausalLM(inputs_embeds, attention_mask, position_ids, past_key_values)
```

Qwen2 decoder block, repeated 28 times:

```text
residual = x
x = RMSNorm(x)
q = Linear(3584 -> 3584, bias)(x).view(B,S,28,128).transpose(1,2)
k = Linear(3584 -> 512, bias)(x).view(B,S,4,128).transpose(1,2)
v = Linear(3584 -> 512, bias)(x).view(B,S,4,128).transpose(1,2)
q,k = RoPE(q,k, position_ids)
k,v = cache.update(k,v) if cache
attn = causal_GQA(q,k,v, mask, repeat_kv=7)
x = residual + Linear(3584 -> 3584, no_bias)(attn)
residual = x
x = RMSNorm(x)
x = down_proj(SiLU(gate_proj(x)) * up_proj(x))
x = residual + x
```

## 6. Attention requirements

Audio attention:

- Noncausal self-attention only.
- MHA, 20 heads, head dim 64, q/k/v width 1280.
- Query is multiplied by `head_dim**-0.5` before attention; backend scaling is passed as `1.0`.
- Mask is a bidirectional additive padding mask produced from `input_features_mask` after conv downsampling.
- No KV cache for the audio encoder.
- FlashAttention/SDPA-compatible interfaces are wired, but eager fallback is ordinary matmul/softmax/matmul.

Qwen2 language attention:

- Causal self-attention.
- GQA: 28 query heads, 4 KV heads, 7 repeat groups, head dim 128.
- Q width 3584, K/V projection output width 512 each, attention output width 3584.
- RoPE is applied to Q/K before cache update; cached K is post-RoPE.
- Cache storage per layer: keys `[B,4,T,128]`, values `[B,4,T,128]`; repeat to 28 heads should happen inside attention.
- Masking uses `create_causal_mask`; sliding-window mask path exists in Qwen2 but is not active for this checkpoint because all `layer_types` are `full_attention`.
- Eager attention repeats KV, computes QK matmul, adds mask, softmax with `dtype=torch.float32`, casts to query dtype, then matmul with V.
- First integration can use dense causal attention for prefill and one-token decode. Optimized path should be GQA-aware FlashAttention/SDPA with cache.

Generation coupling:

- The composite model forwards `input_features` only when both `input_features` and `input_ids` are present.
- `prepare_inputs_for_generation` includes audio inputs only on first iteration or when `use_cache` is false.
- With cache disabled, repeated generation steps may recompute/stitch audio prefixes; for production DinoML should enable a clear prefill/decode split and cache policy.

## 7. Position encoding and custom math

Audio length equations:

```python
def audio_lengths(feature_mask_sum):
    conv_len = (feature_mask_sum - 1) // 2 + 1
    post_pool_len = (conv_len - 2) // 2 + 1
    return conv_len, post_pool_len
```

Qwen2 RoPE:

```python
def qwen2_default_rope(position_ids, head_dim=128, theta=1_000_000.0):
    inv = 1.0 / (theta ** (arange(0, head_dim, 2) / head_dim))
    freqs = outer(position_ids, inv)
    emb = concat(freqs, freqs, dim=-1)
    return cos(emb), sin(emb)

def apply_rope(q, k, cos, sin):
    cos = cos[:, None, :, :]
    sin = sin[:, None, :, :]
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin
```

Audio encoder attention scaling:

```python
q = q_proj(x) * (head_dim ** -0.5)
attn = attention_backend(q, k, v, mask, scaling=1.0)
```

Precomputable:

- Audio positional embedding is a weight table, not generated math.
- Qwen2 RoPE inverse frequencies are precomputable for theta/head_dim. Cos/sin depend on runtime `position_ids` and cache length.
- Feature extraction mel filters are precomputable for the processor config, but log-mel values depend on waveform and are best treated as preprocessing initially.

## 8. Preprocessing and input packing

Waveform contract:

- Mono audio only; Whisper feature extractor rejects batched numpy with rank > 2.
- Sampling rate must be 16000 when provided.
- Window size is `sampling_rate * chunk_length = 480000` samples.
- `max_audio_len=600` and `chunk_length=30` imply max 20 windows. Longer audio is truncated.
- Padding is `max_length`, right-padding in the feature extractor, returning attention mask.

Feature tensor contract:

- `input_features`: `[num_windows_total, 128, 3000]`, float32 from feature extractor by default.
- `input_features_mask`: `[num_windows_total, 3000]`, derived from waveform padding and hop length.
- Full 30 s window: 3000 feature frames -> 1500 post-conv frames -> 750 post-pool projected audio tokens.
- For shorter final windows, `input_features_mask.sum(-1)` drives post-pool valid row counts.

Text/audio packing:

- Chat template inserts a single textual `<sound>` token marker per audio item.
- Processor replaces each occurrence with `<sound>` repeated by the total projected audio token count for that sample.
- Tokenizer must map `<sound>` to `audio_token_id=151669`; if this id/token is missing, source count checks will fail later or the model will not stitch audio.
- Text tokenizer uses Qwen2 tokenizer config; model input padding side is left in processor common kwargs, while tokenizer config itself says right padding. The processor override is the model-coupled behavior for prepared inputs.

Scatter/indexed update:

- Source uses `masked_scatter` over expanded `[B,S,H]`.
- Processor-generated good case is stricter: all audio rows should be copied into `<sound>` slots in row-major token order, with exact count equality.
- DinoML should reject arbitrary boolean masks for first integration unless they are equivalent to processor-produced placeholder runs.

Generation controller:

- `generation_config.json` sets max new tokens 2048 and BOS/EOS/PAD ids.
- Transcription helper postprocess can strip fixed assistant prefixes; this is tokenizer/postprocessing, not neural graph.
- No timestamp logits processor or Whisper-style decoder language control is implemented in native AudioFlamingo3 source.

## 9. Graph rewrite / lowering opportunities

### Rewrite: audio Conv1d k=3 static front end

Source pattern:

```text
Conv1d(128->1280,k=3,pad=1,stride=1) -> GELU
Conv1d(1280->1280,k=3,pad=1,stride=2) -> GELU
```

Replacement pattern:

```text
specialized NCL conv1d kernels or im2col/GEMM for fixed channel/kernel/stride
```

Preconditions:

- Input layout exactly `[B,128,T]`.
- `kernel_size=3`, `padding=1`, dilation 1, groups 1.
- Conv2 stride exactly 2.
- Preserve zero-padding semantics and output length `(T-1)//2+1`.

Failure cases:

- Any channel-last rewrite that changes time/channel axes without rewriting masks and position table indexing.
- Dynamic `T` greater than processor max unless embedding positions and masks are bounded.

Parity test sketch:

- Random `[W,128,3000]` and shorter masked windows, compare conv outputs and post-conv mask lengths.

### Rewrite: AvgPool1d post-encoder -> local pairwise mean

Source pattern:

```text
x [B,T,1280] -> permute [B,1280,T] -> AvgPool1d(2,stride=2) -> permute [B,T/2,1280]
```

Replacement:

```text
y[:, i, :] = 0.5 * (x[:, 2*i, :] + x[:, 2*i+1, :])
```

Preconditions:

- Kernel 2, stride 2, no padding, `ceil_mode=False`, `count_include_pad=True`.
- `T` after conv should be 1500 for full windows; shorter valid lengths use mask-derived post lengths.

Failure cases:

- Odd `T` changes floor behavior; preserve PyTorch AvgPool1d output length.

### Rewrite: placeholder masked_scatter -> indexed row copy

Source pattern:

```text
special_audio_mask = input_ids == audio_token_id
inputs_embeds = inputs_embeds.masked_scatter(expand(mask), audio_rows)
```

Replacement:

```text
indices = nonzero(input_ids == audio_token_id) in row-major order
assert len(indices) == audio_rows.shape[0]
row_copy(inputs_embeds, indices, audio_rows)
```

Preconditions:

- `input_ids` available.
- `inputs_embeds` is generated from token embeddings before replacement.
- Mask count equals audio rows.
- Optional stricter guard: per sample, `<sound>` slots are contiguous groups matching processor expansion.

Failure cases:

- `inputs_embeds` supplied without `input_ids`; source then compares full embedding rows to the audio-token embedding. DinoML should reject this path initially.
- Arbitrary interleaved placeholder positions if the runtime chooses prefix-copy optimization.

### Rewrite: Qwen2 GQA attention

Source pattern:

```text
Q [B,28,S,128], K/V [B,4,T,128] -> repeat_kv by 7 -> attention
```

Replacement:

```text
GQA FlashAttention/SDPA consuming KV heads without materialized repeat
```

Preconditions:

- Head counts exactly divisible: 28 % 4 == 0.
- RoPE applied before cache update.
- Dense full causal attention for current configs.
- Preserve fp32 softmax behavior or accepted tolerance.

Failure cases:

- Sliding-window `layer_types` not all full attention.
- Cache stored after repeat rather than before repeat.

### Rewrite: last-token-only logits

Source pattern:

```text
logits = lm_head(hidden_states[:, slice(-logits_to_keep,None), :])
```

Replacement:

```text
for decode or `logits_to_keep=1`, compute output GEMM only for final token rows
```

Preconditions:

- No labels/loss path.
- Caller does not request full prefill logits.

## 10. Kernel fusion candidates

Highest priority:

- Qwen2 RMSNorm + residual scheduling: every decoder layer uses two RMSNorms over width 3584.
- Qwen2 GQA attention with RoPE and KV cache: this is the dominant prefill/decode cost and has nontrivial 28/4 head layout.
- Qwen2 SwiGLU MLP: gate/up GEMMs plus SiLU*multiply and down GEMM at 18944 intermediate width.
- Placeholder row-copy kernel: avoids admitting general `masked_scatter` while preserving multimodal packing.
- Audio Conv1d front end plus GELU: fixed shapes and small kernels, executed for every audio window.

Medium priority:

- Audio encoder LayerNorm + MHA + FFN kernels for 32 layers, sequence length up to 1500 and hidden 1280.
- Audio avg-pool pairwise mean + final LayerNorm.
- Projector MLP 1280->3584->3584, including GELU.
- Last-token-only LM head GEMM for decode against vocab 151672.

Lower priority:

- GPU log-mel extraction. Keep CPU/data pipeline first; GPU STFT can be a throughput optimization later.
- Layout translation around audio Conv1d. It may help provider kernels but requires strict axis/mask/position guards.
- Generation postprocessing helpers such as transcription prefix stripping.

## 11. Runtime staging plan

Stage 1: config and weights admission.

- Parse only native `model_type="audioflamingo3"` with `audio_config.model_type="audioflamingo3_encoder"` and `text_config.model_type="qwen2"`.
- Reject `think/config.json`/Llava remote-code fields for this target.
- Require current tested dimensions first: audio 1280/32/20, text 3584/28/4/28, projector GELU+bias.

Stage 2: processor-compatible tensor ABI.

- Accept precomputed `input_features`, `input_features_mask`, `input_ids`, `attention_mask`.
- Stub waveform log-mel extraction as external CPU preprocessing.
- Validate placeholder count and post-pool length equations.

Stage 3: audio encoder + projector parity.

- Run audio Conv1d/encoder/AvgPool/LayerNorm/projector and produce flattened audio rows.
- Validate per-window and batched/multi-window masks.

Stage 4: embedding stitch + Qwen2 prefill.

- Implement row-copy replacement, then run full mixed-sequence Qwen2 prefill.
- Initially compute full logits or `logits_to_keep=1` under a clear flag.

Stage 5: decode with KV cache.

- Enable Qwen2 DynamicCache-equivalent ABI storing 4 KV heads per layer.
- Keep audio/projector outputs cached and excluded from decode steps.

Stage 6: optimized attention and fusions.

- Replace dense attention with GQA FlashAttention/SDPA-style kernels.
- Add last-token LM head optimization and fused MLP/RMSNorm kernels.

Stage 7: longer-audio batching.

- Add multi-window scheduling and batching probes up to 20 windows per sample.
- Keep processor-derived mapping metadata explicit for reassembly/count validation.

## 12. Parity and validation plan

- Feature ABI parity: construct masks with lengths covering 1 frame, short final window, full 3000 frames, and 20 full windows; verify post-pool token counts.
- Audio Conv1d parity: compare source PyTorch and DinoML for random `[W,128,3000]`, fp32 tolerance `1e-4`.
- Audio encoder one-layer parity: include bidirectional mask and pre-scaled Q attention. Use fp32 first, then bf16/fp16 with relaxed tolerance.
- Audio encoder full-stack parity: compare last hidden state and flattened valid `pooler_output` for small `W` and mixed lengths.
- Projector parity: compare Linear/GELU/Linear and bias handling for 1280->3584->3584.
- Placeholder copy parity: random input ids with processor-like contiguous `<sound>` runs; verify equality with PyTorch `masked_scatter`.
- Reject tests: placeholder count mismatch, `inputs_embeds` without `input_ids`, noncontiguous/arbitrary mask if using stricter copy.
- Qwen2 block parity: one layer with GQA, RoPE theta 1e6, fp32 softmax.
- Prefill logits parity: text-only first, then mixed audio/text, with `logits_to_keep=1` and full logits.
- Decode parity: prefill then one-token decode; verify cache shape `[layers=28][K,V]=[B,4,T,128]` and token logits.
- End-to-end smoke: processor-prepared audio transcription prompt to generated first token(s), accepting generation sampling/controller differences.

Recommended tolerances:

- fp32 audio/projector/text: `rtol=1e-4`, `atol=1e-4` for most blocks; attention may need `2e-4`.
- bf16/fp16: start `rtol=2e-2`, `atol=2e-2`, tighten after matching softmax upcast and GEMM accumulation policy.

## 13. Performance probes

- CPU preprocessing throughput: waveform seconds/sec for Whisper log-mel extraction.
- Audio encoder throughput: windows/sec for `[W,128,3000]`, sweep W in 1, 2, 8, 20.
- Projector throughput and memory bandwidth for `[W,750,1280] -> [valid,3584]`.
- Placeholder copy bandwidth for long prompts with 750 to 15000 audio tokens.
- Qwen2 prefill tokens/sec for text-only versus audio-heavy prompts.
- Decode tokens/sec with cache enabled and `logits_to_keep=1`.
- KV cache memory: 28 layers * 2 * B * 4 * T * 128 * dtype_size.
- LM head cost: final token vs full sequence projection to vocab 151672.
- Attention backend comparison: eager dense, SDPA, FlashAttention-compatible GQA.
- Sequence-length sweep: prompt length with 0, 1, 5, 20 audio windows plus text.
- Batch sweep: batch size vs total windows, including uneven per-sample windows.

## 14. Skip/defer list

- Training, labels/loss, dropout, layerdrop, gradient checkpointing.
- `inputs_embeds`-only audio placeholder discovery by embedding equality.
- Native remote-code/Llava `think/config.json` path with vision/speech/sound towers.
- Voice-to-voice/TTS/chat talker modules; not implemented in native `AudioFlamingo3ForConditionalGeneration`.
- GPU STFT/log-mel extraction.
- Sliding-window Qwen2 variants.
- Tensor parallel/pipeline parallel plans.
- General boolean scatter; use bounded row-copy.
- Quantization fields from historical configs.
- Beam search and advanced generation processors beyond standard causal LM sampling.

## 15. Final implementation checklist

- [ ] Parse native AudioFlamingo3 config and reject remote-code/historical Llava configs.
- [ ] Load audio tower, projector, Qwen2 decoder, token embedding, and LM head weights with aliasing preserved for Qwen2 tied-key metadata.
- [ ] Define external preprocessing ABI for `input_features [W,128,3000]` and `input_features_mask [W,3000]`.
- [ ] Implement audio length equations and placeholder-count validation.
- [ ] Implement Conv1d/GELU audio front end.
- [ ] Implement audio noncausal MHA with pre-scaled Q and bidirectional padding mask.
- [ ] Implement audio FFN, LayerNorm, AvgPool1d, final LayerNorm.
- [ ] Implement projector Linear/GELU/Linear.
- [ ] Implement bounded `<sound>` indexed row copy replacing `masked_scatter`.
- [ ] Implement Qwen2 RMSNorm, RoPE theta 1e6, GQA attention, SwiGLU MLP.
- [ ] Implement Qwen2 KV cache with 4 KV heads per layer and post-RoPE keys.
- [ ] Implement `logits_to_keep` output projection optimization.
- [ ] Add parity tests for audio encoder/projector, stitch, Qwen2 block, prefill logits, and decode cache.
- [ ] Benchmark preprocessing, audio encoder, projector/stitch, prefill, decode, and LM head separately.
