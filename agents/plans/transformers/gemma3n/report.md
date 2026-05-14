# Gemma3n Transformers Audit

## 1. Source basis

Transformers commit/version: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4` in `X:/H/transformers`.

Model id: primary official targets are `google/gemma-3n-E2B-it` and `google/gemma-3n-E4B-it`. Both official repos are manually gated for raw files in unauthenticated access; Hugging Face model API metadata was readable and confirms `architectures=["Gemma3nForConditionalGeneration"]`, `model_type="gemma3n"`, `pipeline_tag="image-text-to-text"`, and processor/config files exist. Detailed checkpoint dimensions below use open mirrors labeled as such.

Config source:

- Source defaults: `src/transformers/models/gemma3n/configuration_gemma3n.py`.
- Official repo metadata: Hugging Face model API for `google/gemma-3n-E2B-it` at sha `5e092ebca197cdcd8d8b195040accf22693501bc`, last modified `2025-07-14T13:55:52Z`; `google/gemma-3n-E4B-it` at sha `c1221e9c62e34a43ab7ffacd1be0ea71f126ef10`, last modified `2025-07-14T13:56:17Z`.
- Open config mirrors: `unsloth/gemma-3n-E4B-it`, `mlx-community/gemma-3n-E2B-3bit`, `mlx-community/gemma-3n-E4B-it-bf16`, and `tiny-random/gemma-3n`.

Source files inspected:

- `src/transformers/models/gemma3n/configuration_gemma3n.py`
- `src/transformers/models/gemma3n/modeling_gemma3n.py`
- `src/transformers/models/gemma3n/modular_gemma3n.py`
- `src/transformers/models/gemma3n/processing_gemma3n.py`
- `src/transformers/models/gemma3n/feature_extraction_gemma3n.py`
- `src/transformers/models/gemma3n/convert_gemma3n_weights.py`
- `src/transformers/models/timm_wrapper/configuration_timm_wrapper.py`
- `src/transformers/models/timm_wrapper/modeling_timm_wrapper.py`
- Auto mappings for `gemma3n_vision -> TimmWrapperModel`.

Any missing files or assumptions:

- `modeling_gemma3n.py` and `configuration_gemma3n.py` are generated from `modular_gemma3n.py`; future source edits should target the modular file.
- Official Google raw `config.json`/processor files were gated. Open mirror values are treated as representative but not authoritative Google raw files.
- Vision backbone internals are delegated to timm `mobilenetv5_300m_enc`; this report records the consumed feature contract and marks full MobileNetV5 operator ownership as a nested-backbone follow-up.

## 2. High-level architecture

Primary runtime target: multimodal autoregressive generation with `Gemma3nForConditionalGeneration`. The text-only `Gemma3nForCausalLM` is also implemented and is useful as a staged bring-up target.

Architecture:

```text
text/audio/image preprocessing
  -> token ids with expanded image/audio placeholders
  -> optional Timm MobileNet vision tower -> soft visual tokens -> multimodal embedder
  -> optional USM-style audio encoder -> soft audio tokens -> multimodal embedder
  -> masked scatter into text embedding sequence
  -> Gemma3n text decoder prefill/decode
  -> tied LM head -> final tanh logit softcap -> logits/sampling
```

Stage decomposition:

- CPU/data pipeline: tokenizer/chat template, placeholder expansion, image resizing/rescale, audio waveform padding and log-mel extraction.
- Independently cacheable encoders/projectors: vision tower plus `Gemma3nMultimodalEmbedder`; audio encoder plus `Gemma3nMultimodalEmbedder`.
- Prefix construction: text token embedding, hard multimodal token embedding for placeholder-token ranges, soft image/audio feature `masked_scatter`, per-layer embedding construction.
- Prefill: mixed text/image/audio sequence through full/sliding text attention.
- Decode: text-only incremental decoding; `prepare_inputs_for_generation` drops multimodal tensors after first cached iteration.

## 3. Important config dimensions

Source defaults from `Gemma3nTextConfig`, `Gemma3nVisionConfig`, and `Gemma3nAudioConfig`:

| Field | Source default | Notes |
|---|---:|---|
| text hidden size | 2048 | also equals `num_attention_heads * head_dim` in sampled configs |
| text layers | 35 | E4B default |
| text intermediate size | 16384 per layer | E2B conversion script uses 8192 |
| attention heads / KV heads | 8 / 2 | GQA with 4 query groups per KV head |
| head dim | 256 | explicit config field; do not infer blindly |
| vocab size | 262400 | includes text plus multimodal token ranges |
| per-layer vocab / dim | 262144 / 256 | separate embedding table feeds each decoder layer |
| max positions | 32768 | shorter than many Gemma3 configs |
| layer pattern | 4 sliding then 1 full repeated | full layers when `(i + 1) % 5 == 0` |
| sliding window | 512 | local text attention |
| RoPE theta | local 10000, full 1000000 | separate `rope_parameters` buckets |
| final logit softcap | 30.0 | enabled by default |
| attention logit softcap | none for text | Gemma2 has text attention softcap; Gemma3n text does not pass one |
| AltUp inputs | 4 | hidden state stack dimension |
| KV shared layers | 15 | E4B default; E2B conversion uses 10 |
| LAUREL rank | 64 | low-rank residual augmentation |
| activation sparsity | first 10 layers 0.95, rest 0.0 | Gaussian cutoff before activation |
| vision tower | `mobilenetv5_300m_enc` via timm | `gemma3n_vision` maps to `TimmWrapperModel` |
| vision hidden / soft tokens | 2048 / 256 | consumed as `[B, 256, 2048]` before projection |
| audio hidden / mel bins | 1536 / 128 | USM-style encoder |
| audio soft tokens | 188 | processor always expands placeholders to 188 |

Representative checkpoint sweep:

| Config source | Scope | Text shape | KV sharing | Vision | Audio | Processor |
|---|---|---:|---:|---|---|---|
| `tiny-random/gemma-3n` open debug | parser/smoke | 4 layers, H=32, I=64, Q=1, KV=1, D=32 | 2 shared layers | MobileNetV5 test args, H=2048, 256 tokens | H=64, 2 conformer layers, 2 heads | image 768, audio 16 kHz, 188 tokens |
| `mlx-community/gemma-3n-E2B-3bit` open mirror | E2B-like quantized mirror | 30 layers, H=2048, I=8192, Q=8, KV=2, D=256 | 10 shared layers | MobileNetV5, H=2048, `do_pooling=True` in mirror | H=1536, 12 layers, 8 heads | not audited for MLX quant runtime |
| `unsloth/gemma-3n-E4B-it` open mirror | E4B-like bf16 mirror | 35 layers, H=2048, I=16384, Q=8, KV=2, D=256 | 15 shared layers | MobileNetV5, H=2048, `do_pooling=False` | H=1536, 12 layers, 8 heads | image 768, audio 16 kHz, 188 tokens |
| `mlx-community/gemma-3n-E4B-it-bf16` open mirror | E4B bf16 mirror | same as E4B | 15 shared layers | MobileNetV5, H=2048 | H=1536 | raw config contains many generic inherited fields |
| `google/gemma-3n-E2B-it` official API | gated official metadata | detailed raw config unavailable | unknown from API | config file exists | config file exists | processor/preprocessor files exist |
| `google/gemma-3n-E4B-it` official API | gated official metadata | detailed raw config unavailable | unknown from API | config file exists | config file exists | processor/preprocessor files exist |

## 3a. Family variation traps

- Gemma3n is not Gemma3 text plus a projector. It adds AltUp, LAUREL, per-layer embeddings, activation sparsity, and KV sharing.
- E2B and E4B differ structurally: E2B uses 30 layers, intermediate 8192, and 10 shared KV layers; E4B uses 35 layers, intermediate 16384, and 15 shared KV layers.
- Text layers after the sharing cutoff do not own `k_proj`, `v_proj`, `k_norm`, or `v_norm`; lowering must not expect those weights.
- KV sharing is by layer type: a shared sliding layer reuses the last non-shared sliding K/V; a shared full layer reuses the last non-shared full K/V.
- Sliding and full text attention have separate RoPE buckets. Missing `rope_parameters` is not missing behavior; config post-init fills local/global defaults.
- Text attention scaling is `1.0`, not `head_dim^-0.5` and not Gemma3 `query_pre_attn_scalar^-0.5`.
- Text attention does Q/K/V head RMSNorm before RoPE, with V norm having no learned scale.
- First 10 MLP layers can apply activation sparsity using mean/std/normal-icdf cutoff and ReLU before the configured activation.
- The multimodal token id ranges are semantically overloaded: text per-layer embeddings only accept `< 262144`; vision hard tokens are `[262144, 262272)`; audio hard tokens are `>=262272`.
- Processor expands a single image placeholder into 256 image soft tokens and a single audio placeholder into 188 audio soft tokens.
- Audio encoder may produce fewer than 188 soft tokens; model pads to 188 using the last audio embedding token before scatter.
- Vision is a nested timm MobileNetV5 backbone. DinoML should either compose a separate timm/MobileNet audit or initially treat it as an external encoder stage.
- Source tensor layout for vision is channels-first `pixel_values`; layout rewrites to NHWC must be guarded around timm feature consumers and the final reshape `[B,C,H,W] -> [B,256,C]`.
- Audio convolution source uses NCHW-style `Conv2d` over `[B,1,T,F]` and `Conv1d` over `[B,D,T]`; do not silently translate axes without rewriting pads, group norm axes, and flatten order.

## 4. Operator coverage checklist

Tensor/layout ops:

- Embedding lookup, including scaled embeddings.
- Reshape/view, `permute`, `transpose`, `contiguous`, stack over AltUp dimension, mean over AltUp dimension.
- `where`, `masked_scatter`, boolean masks, logical and, compare ranges, `gather`, slicing, concat, pad, unfold/framing.
- Last-token/indexed logits slicing through `logits_to_keep`.

Neural network primitives:

- Linear/GEMM without bias for almost all projections.
- RMSNorm with optional scale and fp32 accumulation.
- Gelu PyTorch tanh, SiLU, GLU, ReLU, tanh.
- MLP SwiGLU-like path: `down(act(gate(x)) * up(x))`.
- Gaussian activation sparsity: mean/std/normal-icdf cutoff plus ReLU.
- LAUREL: `Linear(H -> rank) -> Linear(rank -> H) -> RMSNorm -> residual`.
- AltUp router and coefficient matmuls over `[altup_num_inputs]`.
- Final LM head tied to text token embedding weight.
- Final logit softcap: `tanh(logits / 30) * 30`.

Attention primitives:

- Text causal GQA, Q=8, KV=2, D=256 in production configs.
- Hybrid full/sliding masks by layer type.
- RoPE before cache update.
- Q/K/V per-head RMSNorm before attention.
- KV sharing across trailing layers.
- Audio local chunk attention with relative positional embedding and softcapped logits.

Position/rotary/relative-bias ops:

- Text RoPE with separate local and global theta.
- Audio sinusoidal relative position embedding projected to per-head terms and shifted into `[B,N,U,W,C]`.

Generation/cache ops:

- Dynamic cache creation from config.
- Per-layer cache update only for non-shared text layers.
- Shared-KV state side channel for later shared layers.
- Multimodal encoder outputs cached outside the decoder KV cache after prefill.

Preprocessing-coupled ops:

- Image resize/rescale to channels-first `pixel_values`; sampled processors use size 768 and `do_normalize=False`.
- Audio 16 kHz waveform padding/truncation, preemphasis, Hann window, FFT, mel matmul, log clamp, optional per-bin normalization.

Scatter/indexed update ops for multimodal embedding stitch:

- Replace hard image/audio placeholder embeddings with soft projected features using expanded boolean masks and `masked_scatter`.
- Replace hard multimodal token ids in the base embedding stream using `torch.where`.

Nested backbone ops:

- Vision MobileNetV5/timm: conv, depthwise conv, possible MQA-like mobile blocks, normalization/activation. Treat as external until separately audited.
- Audio encoder: Conv2d, cumulative group norm, Conv1d depthwise causal, local attention, BMM.

## 5. Layer/block breakdown

Text decoder setup:

```text
input_ids -> scaled token embedding [B,T,H]
input_ids -> per-layer embedding [B,T,L,P]
inputs_embeds -> Linear(H -> L*P) -> reshape [B,T,L,P] -> RMSNorm
per_layer_inputs = (projected + token_per_layer) / sqrt(2)
AltUp initial stack:
  h0 = inputs_embeds
  h1..h3 = Linear(H -> H), magnitude-rescaled to h0
  hidden_states = stack([h0,h1,h2,h3])  # [4,B,T,H]
```

Decoder block, repeated `L` times:

```text
predictions = AltUp.predict(hidden_states)
active = predictions[altup_active_idx]
x_norm = RMSNorm(active)
laurel = x_norm + RMSNorm(Linear(Linear(x_norm)))
attn = Attention(x_norm, RoPE, mask, cache/shared_kv)
attn = RMSNorm(attn)
x = (active + attn + laurel) / sqrt(2)
ffn = RMSNorm(MLP(RMSNorm(x)))
x = x + ffn
corrected = AltUp.correct(predictions, x)
first = corrected[active_idx]
first = optional correct_output_scale * first
first = Linear(H -> P)(first)
first = activation(first) * per_layer_input_for_this_layer
first = RMSNorm(Linear(P -> H)(first))
corrected[1:] += first
hidden_states = corrected
```

Text attention:

```text
q = RMSNorm(Linear(H -> QH*D)(x).view(B,T,QH,D))
q = RoPE(q).transpose(1, 2)
if not shared:
  k = RMSNorm(Linear(H -> KVH*D)(x).view(B,T,KVH,D))
  k = RoPE(k).transpose(1, 2)
  v = RMSNorm_no_scale(Linear(H -> KVH*D)(x).view(B,T,KVH,D)).transpose(1, 2)
  cache.update(k, v, layer_idx)
else:
  k, v = shared_kv_states[kv_shared_layer_index]
attn = GQA(q, k, v, mask, scaling=1.0, sliding_window=maybe)
out = Linear(QH*D -> H)(attn)
```

Audio encoder:

```text
input_features [B,T,128]
-> unsqueeze [B,1,T,128]
-> Conv2d(1->128, k=3x3, stride=2x2, manual pad) + cumulative group norm + ReLU
-> Conv2d(128->32, k=3x3, stride=2x2, manual pad) + cumulative group norm + ReLU
-> flatten freq/channel -> Linear(32*32 -> 1536)
-> 12 conformer blocks:
   FFN residual with 0.5 scale
   local relative attention residual
   causal depthwise Conv1d light-conv residual
   FFN residual with 0.5 scale
   RMSNorm
-> optional time reduction by 4
-> zero masked positions
-> multimodal soft embedder to text H
```

Vision path:

```text
pixel_values [B,3,768,768] in sampled processors
-> TimmWrapperModel.forward_features()
-> last_hidden_state expected [B,2048,Hv,Wv] with Hv*Wv=256
-> reshape [B,2048,256] -> permute [B,256,2048]
-> multiply by sqrt(2048)
-> multimodal soft embedder to text H
```

## 6. Attention requirements

Text decoder:

- Causal self-attention only.
- GQA: production mirrors use Q heads 8, KV heads 2, head dim 256.
- Layer schedule: 4 sliding attention layers then 1 full attention layer repeated.
- Sliding window: 512.
- Masking: source builds a dict with `full_attention` from `create_causal_mask` and `sliding_attention` from `create_sliding_window_causal_mask`.
- Cache: `DynamicCache(config)` is created when `use_cache=True` and no cache is passed.
- Cached K is stored after Q/K RMSNorm and RoPE.
- Shared layers do not update cache and always consume `shared_kv_states`, because sliding cache may not retain full states.
- Eager fallback repeats K/V before matmul; production should avoid physical repeat for GQA.
- FlashAttention/SDPA compatibility requires native GQA, sliding-window masks, pre-rotated K cache, text scaling of 1.0, and no text attention softcap.

Text cache ABI:

- Non-shared layer pre-update K/V shape: `[B, KVH, T_step, D]`.
- Attention call after GQA repeat conceptually sees K/V `[B, QH, K_len, D]`.
- Full non-shared layers grow with sequence length.
- Sliding non-shared layers use config-aware sliding cache behavior.
- Shared layers read full-length K/V side-channel from the latest non-shared layer of the same layer type. DinoML needs a cache manifest that distinguishes physical cache owners from logical layer applications.

Audio attention:

- Non-generation encoder local self-attention, not KV-cache attention.
- Chunk size W=12; left context is 12 (`context_left=13` means `max_past_horizon=12`); right context 0; context size C=24.
- Q/K/V all use 8 heads, D=192 for H=1536.
- Per-dim query scale uses `head_dim^-0.5 / softplus(0) * softplus(per_dim_scale)`.
- Relative position term is added to QK content logits before softcap.
- Logit softcap is required: `tanh(logits / 50) * 50`.
- Mask combines input validity and local causal validity, then softmax is fp32.

## 7. Position encoding and custom math

Text RoPE:

```python
def gemma3n_rope(position_ids, head_dim, theta):
    inv = 1.0 / (theta ** (arange(0, head_dim, 2).float() / head_dim))
    freqs = inv[None, :, None] @ position_ids[:, None, :].float()
    emb = cat([freqs.transpose(1, 2), freqs.transpose(1, 2)], dim=-1)
    return cos(emb), sin(emb)

def apply_rope(x, cos, sin):
    cos = cos.unsqueeze(2)  # source uses [B,T,Hd] -> [B,T,1,Hd]
    sin = sin.unsqueeze(2)
    return x * cos + rotate_half(x) * sin
```

Local/sliding layers use theta 10000 by default; full layers use theta 1000000. Non-default `rope_parameters` can route through generic `ROPE_INIT_FUNCTIONS`, so first integration should admit only default/known linear variants.

Audio relative position:

```python
pos = arange(max_backward, -max_forward - 1, -1)
timing = cat([sin(pos * inv_timescales), cos(pos * inv_timescales)], dim=-1)
sin_emb = Linear(H -> num_heads * head_dim)(timing).view(F, N, D)
term_ac = q @ k.transpose(-1, -2)
term_bd = relative_shift(q @ sin_emb)
logits = term_ac + term_bd
logits = tanh(logits / softcap) * softcap
```

Precomputable: RoPE inverse frequencies and fixed cos/sin tables for static positions; audio timing tables and projected relative embeddings for fixed dtype/device if weights are loaded. Dynamic: `position_ids`, decode offsets, audio valid masks, and chunk/frame counts.

## 8. Preprocessing and input packing

Processor behavior:

- `Gemma3nProcessor` requires at least one of text, images, audio.
- Audio placeholder token is expanded to `boa + 188 * audio_token + eoa` with blank lines around it.
- Image placeholder token is expanded to `boi + 256 * image_token + eoi` with blank lines around it.
- `token_type_ids` are created manually: 0 text, 1 image placeholder, 3 audio placeholder. The current model forward accepts but does not use `token_type_ids`.
- Chat template uses `<audio_soft_token>` and `<image_soft_token>` placeholders before processor expansion.

Image contract from sampled open processors:

- `image_processor_type="SiglipImageProcessorFast"`, despite model vision config routing to timm MobileNet.
- Resize to 768x768, rescale by `1/255`, `do_normalize=False`, output `channels_first`.
- Runtime tensor: `pixel_values [num_images_or_batch, 3, 768, 768]`.
- Model expects timm `last_hidden_state` reshaped to `[B,2048,256]`, so feature map area must be 256.

Audio contract:

- Input waveform sampled at 16000 Hz, mono assumed by feature extractor shape construction.
- Default max length 480000 samples, matching 30 s; pad/truncate enabled; pad to multiple of 128.
- Frame length 512 samples, hop 160 samples, FFT length 1024 because `fft_overdrive=True`.
- Mel bins 128, min/max frequency 125/7600 Hz, preemphasis 0.97, mel floor `1e-5`.
- Output `input_features` are log-mel spectrograms `[B, frames, 128]`; `input_features_mask` marks valid frames.
- Feature extraction is NumPy CPU/data-pipeline code and intentionally differs from generic Transformers audio utilities.

Multimodal stitch:

- Before soft-feature scatter, hard multimodal token id ranges are embedded by `Gemma3nMultimodalEmbedder`.
- Image soft features and audio soft features are projected through the same embedder in `inputs_embeds` mode.
- Placeholder mask checks compare `inputs_embeds[mask].numel()` with feature tensor `numel()`.
- `masked_scatter` writes feature values in mask order; sequence placeholder count must exactly match `B * soft_tokens * H`.
- During cached decode, multimodal tensors are omitted unless `use_cache=False`.

## 9. Graph rewrite / lowering opportunities

### Rewrite: Text GQA without KV repeat

Source pattern: eager helper calls `repeat_kv` then dense QK/V matmuls.

Replacement: native grouped-query attention kernel consuming physical K/V `[B,KVH,T,D]`.

Preconditions:

- `num_attention_heads % num_key_value_heads == 0`.
- Scaling is 1.0 for Gemma3n text.
- Q/K already normalized and RoPE-applied.
- Mask semantics match full or sliding layer type.

Failure cases: requested attentions output with exact repeated layout, unsupported sliding mask, or shared-KV layer without side-channel manifest.

Parity test sketch: compare eager repeated-KV output to native GQA for full and sliding masks, with shared and non-shared layers.

### Rewrite: Per-layer embedding projection fuse

Source pattern:

```text
Linear(H -> L*P) -> reshape [B,T,L,P] -> RMSNorm(P)
plus optional token per-layer embedding -> scale by rsqrt(2)
```

Replacement: fused projection/reshape/norm/add for all layer slices or precomputed per-layer tensor.

Preconditions: fixed `num_hidden_layers`, `hidden_size_per_layer_input`, same dtype, no view alias needed outside decoder.

Failure cases: caller passes external `per_layer_inputs` with padded shape requiring slicing.

Parity test sketch: compare `project_per_layer_inputs` against fused path for input_ids and inputs_embeds-only cases.

### Rewrite: Last-token-only logits

Source pattern: `lm_head(hidden_states[:, slice_indices, :])`, then final softcap.

Replacement: suffix GEMM only, optionally only `T=1` in decode.

Preconditions: `logits_to_keep` is positive integer or known suffix; labels/loss not requested.

Failure cases: full prefill logits requested or tensor gather indices are dynamic.

Parity test sketch: compare full logits suffix and sliced logits before and after softcap.

### Rewrite: Audio SSCP Conv2d to layout-stable kernels

Source pattern: two manual-padded Conv2d blocks over `[B,1,T,F]` then flatten `[F,C]`.

Replacement: native Conv2d or im2col/GEMM with exact NCHW layout.

Preconditions: kernel 3x3, stride 2x2, padding `(F:1,1; T:0,2)`, groups 1, bias false.

Failure cases: altered config kernel/stride/pad, NHWC rewrite without flatten-order transform.

Parity test sketch: feed random mel tensors and compare conv/norm/ReLU outputs layer by layer.

### Rewrite: Multimodal scatter canonicalization

Source pattern: construct expanded boolean mask then `inputs_embeds.masked_scatter(mask, features)`.

Replacement: indexed copy into token positions from precomputed placeholder indices.

Preconditions: placeholder token count equals feature token count per batch; feature order is row-major `[B,S,H]`.

Failure cases: inputs supplied only as embeddings, duplicate embedding values matching placeholder embedding accidentally, dynamic ragged multimodal batching.

Parity test sketch: compare embedding stream after hard token replacement and soft feature scatter for text+image, text+audio, and both.

### Rewrite: Vision feature cache

Source pattern: `pixel_values -> vision_tower -> embed_vision -> masked_scatter`.

Replacement: precompute projected image soft tokens and feed them to prefix construction.

Preconditions: image bytes/preprocessing/config unchanged; same dtype/projection weights.

Failure cases: training, changing image order, or processor changing image size/token count.

Parity test sketch: cache projected image features and verify identical prefill logits.

## 10. Kernel fusion candidates

Highest priority:

- RMSNorm variants: used everywhere, including scale-less V norm and post-projection multimodal norms.
- Text GQA attention with RoPE-applied cache, sliding masks, and shared-KV ownership.
- AltUp coefficient matmuls and prediction/correction stack operations; this is unique overhead compared with Gemma2/Gemma3.
- MLP with activation sparsity plus gated multiply and down projection.
- Last-token-only LM head plus final tanh softcap.

Medium priority:

- Per-layer embedding projection/norm/add fusion.
- LAUREL low-rank residual fusion.
- Audio local attention kernel with relative shift and softcap.
- Audio causal depthwise Conv1d + GLU/light-conv block.
- Multimodal placeholder indexed scatter.

Lower priority:

- Full timm MobileNetV5 lowering, until selected as an owned DinoML vision target.
- Audio feature extraction on GPU; CPU pipeline is acceptable for first parity.
- Vision NHWC/channel-last rewrites; useful only with complete axis guards.

## 11. Runtime staging plan

Stage 1: parse configs and load text-only `Gemma3nForCausalLM`; implement scaled embeddings, RMSNorm, RoPE buckets, non-shared attention, MLP without activation sparsity optimization, and final softcap.

Stage 2: one-block text parity including AltUp, LAUREL, per-layer inputs, Q/K/V norms, and activation sparsity.

Stage 3: full text prefill parity with full/sliding masks, 4-sliding/1-full layer pattern, and `logits_to_keep`.

Stage 4: decode parity with config-aware cache plus shared-KV side-channel. Validate physical cache owner layers separately from logical layers.

Stage 5: multimodal prefix construction with external/stubbed projected image/audio features and exact placeholder scatter.

Stage 6: audio encoder parity for CPU-produced log-mel features, then audio projector/stitch into prefill.

Stage 7: vision path through external timm or separately audited MobileNetV5 lowering.

Stage 8: optimized kernels/fusions and batch scheduling.

Initially stub: timm vision tower internals, CPU audio feature extractor in runtime graph, training loss, attentions output, and MLX/quantized mirror-specific loaders.

## 12. Parity and validation plan

- Config parser tests for source defaults, E2B mirror, E4B mirror, and tiny-random.
- RMSNorm tests with scale and `with_scale=False`, fp32 accumulation, fp16/bf16 output.
- RoPE tests for `sliding_attention` theta 10000 and `full_attention` theta 1000000, including decode offset.
- Text attention tests for non-shared and shared layers; verify shared layers do not require K/V projection weights.
- Cache tests over sequence lengths greater than 512, separating full and sliding layers.
- AltUp unit tests for predict/correct shapes `[4,B,T,H]` and coefficient math.
- Activation sparsity tests for first sparse layer versus dense layer behavior.
- Per-layer embedding tests for input_ids, non-text multimodal token masking to zero, and external `inputs_embeds`.
- Audio feature extractor parity on a fixed waveform against NumPy source output.
- Audio encoder block parity: SSCP conv, local attention, light conv, full conformer block.
- Multimodal stitch parity for image-only, audio-only, and mixed prompts.
- Prefill logits parity against Transformers for tiny-random, then E4B mirror if weights are available.
- Decode token parity for a short greedy generation with `use_cache=True`.

Recommended tolerances: fp32 unit ops `rtol=1e-4, atol=1e-5`; fp16/bf16 full block/logit parity `rtol=2e-2, atol=2e-2` until optimized attention kernels have measured error envelopes.

## 13. Performance probes

- Text prefill throughput by sequence length, separating AltUp/per-layer overhead from attention.
- Decode tokens/sec by cache length and batch size, with full/sliding/shared layer split.
- KV memory probe: physical cache owner layers only versus naive per-layer cache.
- MLP sparse-layer probe: first 10 layers with Gaussian cutoff versus dense later layers.
- Per-layer embedding projection bandwidth and memory footprint.
- LM head full-vocab GEMM versus last-token-only GEMM.
- Audio preprocessing throughput: waveform to log-mel frames on CPU.
- Audio encoder throughput by mel frame count; separate SSCP conv, local attention, and conformer FFN.
- Vision encoder throughput for 768x768 through timm/MobileNet; separate projected feature cache reuse.
- End-to-end split: processor, encoders, scatter/prefix, text prefill, decode.
- Attention backend comparison: eager repeat-KV, native GQA full, native GQA sliding.

## 14. Skip/defer list

- Training, loss, gradient checkpointing, and optimizer behavior.
- Returning attention tensors for optimized paths.
- Beam-search cache reorder beyond generic cache support.
- Non-default/dynamic RoPE variants unless a target checkpoint uses them.
- Full timm MobileNetV5 ownership in the first text/audio integration.
- GPU audio feature extraction; keep NumPy/CPU pipeline first.
- MLX/3bit mirror quantization formats; treat them as mirror configs, not DinoML weight-loading requirements.
- Processor chat-template edge cases beyond required placeholder expansion.
- Multi-GPU tensor parallel plans.

## 15. Final implementation checklist

- [ ] Parse `Gemma3nConfig`, `Gemma3nTextConfig`, `Gemma3nAudioConfig`, and `Gemma3nVisionConfig`.
- [ ] Normalize missing `rope_parameters` into local/full RoPE buckets.
- [ ] Load tied token embedding / LM head without breaking alias identity.
- [ ] Load and apply per-layer embedding table.
- [ ] Implement Gemma3n RMSNorm with optional scale.
- [ ] Implement scaled token embeddings.
- [ ] Implement per-layer input projection/norm/add path.
- [ ] Implement AltUp predict/correct and output scale.
- [ ] Implement LAUREL residual block.
- [ ] Implement activation sparsity cutoff for configured layers.
- [ ] Implement text MLP gated projection.
- [ ] Implement Q/K/V head norms and RoPE-before-cache.
- [ ] Implement full/sliding GQA attention with scaling 1.0.
- [ ] Implement shared-KV layer manifest and cache ABI.
- [ ] Implement final norm, tied LM head, `logits_to_keep`, and final softcap.
- [ ] Implement multimodal hard token embedding ranges.
- [ ] Implement placeholder mask validation and indexed scatter.
- [ ] Implement or externalize MobileNetV5/Timm vision tower.
- [ ] Implement vision soft-token projector contract `[B,256,2048] -> [B,256,H]`.
- [ ] Implement audio feature extractor parity in CPU/data pipeline.
- [ ] Implement audio SSCP conv projection.
- [ ] Implement audio conformer FFN, local attention, light conv, and reduction.
- [ ] Add tiny-random end-to-end prefill/decode parity.
- [ ] Add E2B/E4B config parser parity from representative configs.
- [ ] Benchmark text prefill/decode, KV memory, audio encoder, vision encoder, and end-to-end multimodal generation.
