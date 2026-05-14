# pe_audio_video audit report

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: facebook/pe-av-large, with official sweep over pe-av-{small,base,large} and -16-frame variants
Config source: Hugging Face raw config/preprocessor/tokenizer JSON, fetched from official facebook repos on 2026-05-13
Source files inspected:
- X:/H/transformers/src/transformers/models/pe_audio_video/configuration_pe_audio_video.py
- X:/H/transformers/src/transformers/models/pe_audio_video/modeling_pe_audio_video.py
- X:/H/transformers/src/transformers/models/pe_audio_video/modular_pe_audio_video.py
- X:/H/transformers/src/transformers/models/pe_audio_video/processing_pe_audio_video.py
- X:/H/transformers/src/transformers/models/pe_audio/configuration_pe_audio.py
- X:/H/transformers/src/transformers/models/pe_audio/modeling_pe_audio.py
- X:/H/transformers/src/transformers/models/pe_audio/feature_extraction_pe_audio.py
- X:/H/transformers/src/transformers/models/pe_video/configuration_pe_video.py
- X:/H/transformers/src/transformers/models/pe_video/modeling_pe_video.py
- X:/H/transformers/src/transformers/models/pe_video/video_processing_pe_video.py
- X:/H/transformers/src/transformers/models/modernbert/configuration_modernbert.py
- X:/H/transformers/src/transformers/models/modernbert/modeling_modernbert.py
Any missing files or assumptions:
- modeling_pe_audio_video.py is generated from modular_pe_audio_video.py; the modular file is authoritative for upstream source edits.
- Video frame encoder is delegated to AutoModelForImageClassification over a timm_wrapper config, default architecture vit_pe_core_large_patch14_336. This report treats that neural body as externally owned and requires a separate timm/PE-core audit or allowlist.
- Official HF repos were accessible; no gated checkpoint was encountered. Snapshot files written beside this report: config_sweep_compact.json, facebook__pe-av-small-16-frame_snapshot.json, facebook__pe-av-large_snapshot.json.
```

Relevant HF links: [facebook/pe-av-small-16-frame](https://huggingface.co/facebook/pe-av-small-16-frame), [facebook/pe-av-small](https://huggingface.co/facebook/pe-av-small), [facebook/pe-av-base-16-frame](https://huggingface.co/facebook/pe-av-base-16-frame), [facebook/pe-av-base](https://huggingface.co/facebook/pe-av-base), [facebook/pe-av-large-16-frame](https://huggingface.co/facebook/pe-av-large-16-frame), [facebook/pe-av-large](https://huggingface.co/facebook/pe-av-large).

## 2. High-level architecture

PE-AV is a multimodal contrastive retrieval model, not an autoregressive generator. The first useful DinoML target should be embedding/logit parity for audio, video, audio-video, and optional text similarity.

```text
CPU audio/video/text preprocessing
  -> audio encoder, video encoder, text encoder
  -> optional audio-video fusion encoder
  -> contrastive projection heads
  -> pairwise similarity matrices [B, B] with learned scale and bias
```

Stage decomposition:

- CPU/data pipeline: audio file load/resample responsibility, mono waveform validation, reflect padding to hop multiple; video decode, frame sampling, resize/rescale/normalize, optional frame padding; tokenizer special-token layout for ModernBERT.
- Independently cacheable encoders: audio encoder output, video encoder output, text encoder `[CLS]` embedding, and audio-video fusion embedding can all be cached before pairwise similarity.
- Audio-video fusion: runs audio and video encoders, aligns video sequence length to audio sequence length by nearest-neighbor interpolation, concatenates features, projects to fusion hidden size, prepends class token, runs noncausal RoPE transformer blocks.
- Similarity heads: LayerNorm + bias-free Linear into text hidden size, then `A @ B.T`, multiply by learned scalar and add learned scalar bias.

## 3. Important config dimensions

Source-derived defaults from configuration classes:

| Component | Field | Default / observed |
| --- | --- | --- |
| Text | model_type | `modernbert` |
| Text | hidden/layers/heads/intermediate | 1024 / 22 / 16 / 2624 in PE-AV checkpoints |
| Text | attention pattern | ModernBERT full attention every 3 layers, sliding attention otherwise; local_attention 128 |
| Audio/video/fusion transformer | head_dim | 128 explicit; hidden size equals heads * 128 for official checkpoints |
| Audio/video/fusion transformer | attention | noncausal MHA by default; `num_key_value_heads == num_attention_heads`; attention bias false |
| Audio/video/fusion transformer | MLP | SwiGLU-style `silu(gate_proj(x)) * up_proj(x)` then down projection |
| Audio preprocessing | sample rate / feature size / hop | 48000 Hz mono, feature_size 1, hop_length 1920 |
| Audio DAC encoder | downsampling | ratios [2, 8, 10, 12], total hop 1920; codebook_dim 128; DAC hidden_size 1024 |
| Video preprocessing | image size | resize to 336 x 336, rescale + normalize |
| Video sampling | frame policy | `-16-frame` repos set `num_frames=16`; non-suffixed repos leave `num_frames=null` and allow variable length |
| Vision body | delegated config | `timm_wrapper`, architecture `vit_pe_core_large_patch14_336`, `do_pooling=true`, `global_pool=map` |

Representative checkpoint sweep, from `config.json` and processor configs:

| Model | Audio layers / hidden / heads | Video layers / hidden / heads | AV fusion layers / hidden / heads | Text | Video frames |
| --- | ---: | ---: | ---: | --- | --- |
| facebook/pe-av-small-16-frame | 12 / 768 / 6 | 4 / 768 / 6 | 6 / 768 / 6 | ModernBERT 22x1024, 16 heads | 16 |
| facebook/pe-av-small | 12 / 768 / 6 | 4 / 768 / 6 | 6 / 768 / 6 | same | variable |
| facebook/pe-av-base-16-frame | 16 / 1024 / 8 | 4 / 1024 / 8 | 6 / 1024 / 8 | same | 16 |
| facebook/pe-av-base | 16 / 1024 / 8 | 4 / 1024 / 8 | 6 / 1024 / 8 | same | variable |
| facebook/pe-av-large-16-frame | 28 / 1792 / 14 | 4 / 1792 / 14 | 6 / 1792 / 14 | same | 16 |
| facebook/pe-av-large | 28 / 1792 / 14 | 4 / 1792 / 14 | 6 / 1792 / 14 | same | variable |

## 3a. Family variation traps

- `-16-frame` changes processor behavior, not model class: frame indices are evenly spaced from first to last frame and no `padding_mask_videos` is emitted for equal 16-frame batches.
- Non-suffixed checkpoints support variable video length; processor pads to `[B, T_max, C, H, W]` and emits `padding_mask_videos` only when lengths differ.
- Video branch is a wrapper over timm PE-core; DinoML should allowlist exact delegated vision bodies or route to fallback until audited.
- Text branch is ModernBERT, not a decoder. It has mixed full/sliding bidirectional attention, fused QKV projection, packed q/k/v split order, and separate RoPE theta for global/sliding attention defaults.
- Audio branch includes a DAC-style Conv1d frontend under `torch.no_grad()` and `torch.backends.cudnn.flags(enabled=False)` in source. Treat this as model behavior for parity, even though inference gradients are irrelevant.
- MaskedGroupNorm uses masked mean/variance when padding masks exist; the class-token padding value is copied from `padding_mask[:, [0]]`, not synthesized as all-true.
- Audio-video fusion aligns video length to audio length with nearest-neighbor interpolation. When per-example valid lengths differ, source falls back to a Python loop and partial row assignment.
- Config class default `PeAudioVideoEncoderConfig.rope_parameters` omits `rope_type`, but official checkpoint configs include `{"rope_theta": 20000, "rope_type": "default"}`. DinoML should require normalized RoPE parameters or fill `rope_type=default` before construction.
- `attention_bias` is configurable but false in all official checkpoints. Biasful attention should be admitted only after checking weight names and tests.
- No autoregressive KV cache is required for the PE-AV runtime target; `past_key_values` arguments are inherited plumbing and not used to update caches in these encoder paths.
- Layout-sensitive regions: audio Conv1d expects `[B, C, T]` after processor/model transpose; video processor/model expects `[B, T, C, H, W]` before flattening to per-frame NCHW images; sequence transformers use `[B, S, H]`.

## 4. Operator coverage checklist

Tensor/layout ops:

- `view`, `reshape`, `transpose`, `contiguous`, `expand`, `cat`, `narrow`, `stack`, `flatten`, `chunk`, `unbind`, `new_zeros`, row slicing, per-example valid-length slicing.
- Video flatten/unflatten: `[B, T, C, H, W] -> [B*T, C, H, W] -> logits [B, T, C_v]`.
- Pairwise similarity orientation: modality embeddings are rows; logits are `left @ right.T` with shape `[B_left, B_right]`.

Neural primitives:

- Embedding + LayerNorm for ModernBERT token input.
- LayerNorm eps 1e-6 for contrastive heads and video fusion normalization; ModernBERT LayerNorm eps 1e-5, norm_bias false in configs.
- RMSNorm over hidden size and per-head q/k RMSNorm for PE audio/video/fusion blocks.
- Linear: PE attention q/k/v/o; PE MLP gate/up/down; output projection; data/projector heads; ModernBERT fused Wqkv and GLU MLP.
- Conv1d: audio DAC frontend/residual blocks, audio/video/fusion sequence ResNet block, video projection `Conv1d(video_hidden -> audio_hidden, kernel=1)`.
- SiLU, GELU, Snake1d `x + alpha^-1 * sin(alpha*x)^2`, dropout disabled in inference.
- `F.normalize` on video vision logits before projection.

Attention primitives:

- PE audio/video/fusion: bidirectional self-attention with RoPE, MHA in official configs, optional GQA via `num_key_value_heads`, q/k head RMSNorm before RoPE, softmax in fp32 then cast back.
- ModernBERT text: bidirectional full and sliding-window attention, fused Wqkv, RoPE, SDPA/Flash/Flex-compatible backend dispatch.

Preprocessing-coupled ops:

- Audio reflect pad to a multiple of 1920, padding mask remapped with `padding_mask[:, :: hop_length]`.
- Video frame sampling, resize/rescale/normalize, sequence padding, optional `padding_mask_videos`.

Wrapper/delegated ops:

- timm_wrapper PE-core image classifier body. Required ABI for PE-AV: consume per-frame `[B*T, C, 336, 336]`, produce `logits` width `vision_config.num_labels`, normalize over last dim, project to PE video hidden size.

## 5. Layer/block breakdown

Audio DAC frontend:

```text
input_values [B, 1, T]
Conv1d(1 -> 64, k=7, pad=3)
for strides [2,8,10,12]:
  residual units with Snake, Conv1d(k=7,dilation=1/3/9), Conv1d(k=1)
  Snake -> Conv1d(stride=s, k=2*s, channels double)
Snake -> Conv1d(d_model -> dac_hidden=1024, k=3)
Conv1d(1024 -> 128, k=1) bottleneck under no_grad
transpose to [B, S_audio, 128] -> Linear(128 -> H_audio)
```

PE encoder block, used by audio, video, and audio-video fusion:

```text
x [B, S, H]
res = x
x = RMSNorm(H)
q = Linear(H -> n_heads*128, bias=attention_bias)
k/v = Linear(H -> n_kv_heads*128, bias=attention_bias)
q,k = RMSNorm(128 per head) -> RoPE
x = residual + Linear(attention(q,k,v) -> H)
res = x
x = RMSNorm(H)
x = Linear(intermediate -> H)(silu(gate_proj(x)) * up_proj(x))
x = residual + x
```

Audio/video/fusion patch embedder:

```text
prepend learned class token [B,1,H]
optional padding_mask = cat(mask[:,[0]], mask)
transpose [B,S,H] -> [B,H,S]
two Conv1d blocks: MaskedGroupNorm(num_groups=1), SiLU, Conv1d(H -> H, k=3, padding=same)
residual add and transpose back
```

Audio-video embedder/fusion:

```text
audio_hidden [B,S_a,H_a], video_hidden [B,S_v,H_v]
video_hidden = Conv1d(H_v -> H_a, k=1) over sequence
video_hidden = nearest-neighbor interpolate to S_a, with mask-aware per-example path if needed
video_hidden = LayerNorm(H_a)
inputs = cat([audio_hidden, video_hidden], dim=-1)
Linear(H_a + H_v -> H_fusion) -> Linear(H_fusion -> H_fusion)
patch embedder -> 6 PE encoder blocks -> RMSNorm -> Linear(H -> H, bias=false)
pooler_output = class token, last_hidden_state excludes class token
```

Text branch:

```text
ModernBERT token embedding -> LayerNorm
22 encoder layers:
  layer 0 attention norm is Identity, later layers LayerNorm
  Wqkv Linear(1024 -> 3*1024), split order q,k,v
  full attention every 3 layers, sliding attention otherwise
  residual add
  LayerNorm -> Wi Linear(1024 -> 2*2624), chunk input/gate
  GELU(input) * gate -> Wo Linear(2624 -> 1024)
final LayerNorm
take hidden_states[-1][:,0] for PE-AV text embedding
```

## 6. Attention requirements

PE audio/video/fusion attention:

- Noncausal bidirectional self-attention over sequence tokens plus prepended class token.
- Official configs use MHA: heads 6/8/14, KV heads equal heads, head_dim 128. Source supports GQA/MQA through `repeat_kv`.
- Masking uses `create_bidirectional_mask` from padding masks when present.
- RoPE is applied to q/k after per-head RMSNorm. Cached keys are not relevant; this is encoder inference.
- FlashAttention/SDPA/Flex can be selected through Transformers attention interfaces. Eager math repeats KV first, computes `q @ k.T * head_dim^-0.5`, adds mask, softmax in fp32, dropout, then `attn @ v`.

ModernBERT text attention:

- Noncausal bidirectional encoder attention.
- Fused qkv weight layout is one Linear output reshaped to `[B,S,3,num_heads,head_dim]`, then `unbind(dim=-3)` in q,k,v order.
- Layer types alternate from config: layer 0 full, layers 1-2 sliding, layer 3 full, etc. `sliding_window = local_attention // 2 + 1` is passed to attention backend for sliding layers.
- No KV cache; text embeddings may be cached independently for retrieval.

## 7. Position encoding and custom math

PE RoPE:

```python
def pe_rope(q, k, cos, sin):
    dim = cos.size(-1)
    freqs = stack(cos[..., :dim//2], -sin[..., :dim//2], sin[..., :dim//2], cos[..., :dim//2])
    freqs = freqs.view(*cos.shape[:-1], dim // 2, 2, 2).unsqueeze(1)
    q = q.reshape(*q.shape[:-1], -1, 1, 2)
    k = k.reshape(*k.shape[:-1], -1, 1, 2)
    return (q * freqs).sum(-1).flatten(3), (k * freqs).sum(-1).flatten(3)
```

RoPE cos/sin are computed in float32 from `inv_freq @ position_ids`, duplicated along the head dimension, then cast to hidden dtype. Official PE configs use theta 20000 and head_dim 128. Cos/sin depend on sequence length but not on batch data, so they can be cached by `(seq_len, dtype, device, rope_parameters)`.

Snake1d audio activation:

```python
def snake(x, alpha):
    return x + (alpha + 1e-9).reciprocal() * sin(alpha * x).pow(2)
```

MaskedGroupNorm:

```python
mean = masked_mean(x.view(B, groups, group_size, S), mask, dim=(2,3), keepdim=True)
var = masked_var(..., unbiased=False)
y = (x - mean) * rsqrt(var + eps)
```

## 8. Preprocessing and input packing

Audio:

- Processor accepts file paths or arrays. File paths are loaded at 48 kHz; non-file input should pass `sampling_rate=48000`.
- Mono only for official feature_size 1. Stereo path explicitly raises.
- Arrays are cast to fp32, reflect-padded to a multiple of hop_length 1920, padded across batch, and returned as `input_values` shaped `[B, 1, T]` for tensors.
- `padding_mask` is produced when padding is enabled, then model downsamples it by `[:, :: 1920]`.
- No STFT/mel frontend; neural Conv1d DAC frontend consumes waveform directly.

Video:

- Processor owns video decode and sampling via BaseVideoProcessor. With `num_frames=16`, samples exact evenly spaced frame indices. With `num_frames=null`, it uses the base sampling path and may emit variable-length videos.
- Output tensor for model is `[B,T,C,H,W]`, with H=W=336, after resize, rescale, and normalize.
- For variable T, processor pads frames to `T_max` and emits `padding_mask_videos [B,T_max]` only if lengths differ.
- Model immediately flattens `[B,T,C,H,W]` to `[B*T,C,H,W]` for the delegated image classifier. Any NHWC/channel-last optimization must be local to this per-frame vision body and must preserve the outer `[B,T]` sequence contract.

Text:

- Tokenizer is AutoTokenizer from official repo. Neural graph consumes `input_ids` and optional `attention_mask`.
- PE-AV uses ModernBERT final hidden state at token index 0 as text embedding. Generation metadata is not applicable.

## 9. Graph rewrite / lowering opportunities

### Rewrite: Conv1d k=1 sequence projection -> Linear

Source pattern: `video_proj(video_hidden.transpose(1,2)).transpose(1,2)` with Conv1d `kernel_size=1`.

Replacement: `Linear(H_video -> H_audio)` over `[B,S,H_video]`.

Preconditions: stride 1, padding 0, dilation 1, groups 1, kernel 1, contiguous or stride-aware sequence tensor, bias preserved if present. Weight transform: `linear.weight = conv.weight[:, :, 0]`; `linear.bias = conv.bias`.

Failure cases: do not apply to Conv1d k=3 patch ResNet or audio DAC strided/dilated convolutions.

Parity test: compare random `[B,S,H_v]` fp32/fp16 output before/after rewrite for small/base/large hidden sizes.

### Rewrite: fixed-frame video flatten batch fusion

Source pattern: `[B,T,C,H,W].view(-1,C,H,W) -> vision_model -> logits.view(B,T,-1)`.

Replacement: keep a fused batch dimension `BT` through vision body and restore sequence axis only before PE video transformer.

Preconditions: no cross-frame ops inside vision body, known `[B,T]`, contiguous input after processor, fixed or padded `T`.

Failure cases: future temporal vision bodies or frame masks consumed inside the image model.

Parity test: compare logits reshape with B=2, T=16 and variable padded T.

### Rewrite: PE MLP to fused SwiGLU GEMM

Source pattern: `down_proj(silu(gate_proj(x)) * up_proj(x))`.

Replacement: packed gate/up GEMM, fused SiLU multiply, down GEMM.

Preconditions: both projections bias-free as in source, hidden_act `silu`, same intermediate size.

Failure cases: configurable activation changes or biasful variants.

Parity test: one-block PE encoder parity in fp32 and bf16.

### Rewrite: contrastive head + similarity

Source pattern: `LayerNorm -> Linear(bias=false) -> A @ B.T -> scale + bias`.

Replacement: cache normalized/projected modality embeddings; use GEMM_RCR for retrieval matrix.

Preconditions: projection weights stable, no return_loss requirement for first integration, same embedding dimension 1024.

Failure cases: training loss path or mixed embedding spaces from wrong head.

Parity test: compare all six logits matrices for B=3.

### Layout pass guard: audio/video frontends

Do not globally translate sequence axes. Audio DAC Conv1d and patch ResNet are channel-first `[B,C,T]`; video processor/model boundary is `[B,T,C,H,W]`; transformers are `[B,S,H]`. Any NHWC/NTHWC conversion must be scoped to the delegated vision image body or a fully controlled Conv1d lowering region with explicit transpose elimination.

## 10. Kernel fusion candidates

Highest priority:

- PE RMSNorm and per-head q/k RMSNorm: many calls across audio, video, and fusion; source upcasts variance math to fp32.
- PE attention q/k/v + q/k RMSNorm + RoPE + FlashAttention: dominant transformer work, especially large audio sequence lengths.
- Audio DAC Conv1d frontend: strided/dilated Conv1d plus Snake dominates raw waveform ingestion and is not GEMM-only.
- Contrastive projection + GEMM similarity: product-facing retrieval output, easy to isolate and benchmark.

Medium priority:

- PE SwiGLU MLP fused activation multiply.
- MaskedGroupNorm + SiLU + Conv1d k=3 patch block, including no-mask fast path.
- Nearest-neighbor video-to-audio alignment for common fixed-length/no-padding cases.
- ModernBERT local/full attention and GLU MLP, if text branch is in first target.

Lower priority:

- Training losses; not needed for inference retrieval.
- Dynamic RoPE variants; official configs use default RoPE.
- GQA/MQA PE attention; official checkpoints use MHA.

## 11. Runtime staging plan

Stage 1: Parse PE-AV configs and load weights with alias preservation for tied modules. Reject unsupported delegated vision bodies except allowlisted `vit_pe_core_large_patch14_336`.

Stage 2: Implement contrastive heads and similarity ABI over precomputed audio/video/text/audio-video embeddings. This gives useful retrieval matrix parity without owning encoders.

Stage 3: Audio encoder parity: waveform tensor -> DAC Conv1d frontend -> PE audio transformer -> pooled embedding. Include padding-mask path.

Stage 4: Video encoder parity with fallback/compose policy for timm_wrapper PE-core. First DinoML-owned part can start at per-frame logits/features -> PE video transformer.

Stage 5: Audio-video fusion parity: run/cache audio and video outputs, implement nearest alignment, fusion transformer, audio_video_head, audio-video logits.

Stage 6: Text branch composition: route ModernBERT through a separate audited ModernBERT implementation; use `[CLS]` output through PE-AV heads.

Stage 7: Optimize fusions: RMSNorm, RoPE attention, SwiGLU, Conv1d frontend, and cached embedding similarity.

## 12. Parity and validation plan

- Unit tests for Snake1d, PE RoPE, MaskedGroupNorm, Conv1d k=1 -> Linear rewrite, and nearest interpolation alignment.
- Single PE encoder block parity for small/base/large dimensions with and without padding masks.
- Audio frontend parity from random mono waveform lengths not divisible by 1920, checking reflect padding and downsampled masks.
- Video post-wrapper parity from synthetic per-frame logits/features into PE video transformer; separate full delegated vision parity once timm body is audited.
- Audio-video fusion parity for equal lengths and unequal masked lengths.
- Contrastive output parity for audio-video only and full audio-video-text path; verify logits orientation `[audio, text]`, `[video, text]`, `[audio, video]`, etc.
- Recommended tolerances: fp32 atol/rtol around 1e-4/1e-4 for isolated ops; bf16/fp16 around 3e-2 for end-to-end embeddings, with stricter checks on fp32 reference subgraphs.

## 13. Performance probes

- Processor throughput: audio load/pad and video decode/frame sampling separately.
- Audio DAC Conv1d frontend latency vs PE audio transformer latency.
- Per-frame vision wrapper throughput for `B*T` images, especially T=16 vs variable full-video T.
- PE video transformer throughput as a function of frame count.
- Audio-video fusion throughput as a function of audio length and video length.
- ModernBERT text branch throughput for query/document batch sizes and sequence lengths.
- Similarity GEMM throughput for square and rectangular retrieval batches.
- Memory probes for cached modality embeddings versus recomputing encoders.

## 14. Skip/defer list

- Training losses and `return_loss=True` are optional for first inference integration.
- End-to-end video decode can remain in the CPU/data pipeline.
- Full timm PE-core implementation should be a separate audit; PE-AV can initially compose or reject non-allowlisted bodies.
- ModernBERT internals can be composed from a separate ModernBERT audit unless text branch is selected as first target.
- GQA/MQA PE variants, attention biases, non-default RoPE, and dynamic rope updates can be rejected until a config requires them.
- Autoregressive generation, KV cache, beam search, speculative decoding, and logits sampling are not applicable.

## 15. Final implementation checklist

- [ ] Parse `PeAudioVideoConfig`, nested `PeAudioEncoderConfig`, `PeVideoEncoderConfig`, and ModernBERT text config.
- [ ] Preserve tied-weight aliases listed in `_tied_weights_keys`.
- [ ] Define admission policy for `timm_wrapper` `vit_pe_core_large_patch14_336`.
- [ ] Load contrastive head weights and scalar logit scale/bias parameters.
- [ ] Implement PE RMSNorm and per-head q/k RMSNorm.
- [ ] Implement PE RoPE exactly, including stack-freq rotation layout.
- [ ] Implement PE bidirectional attention with padding mask and optional backend dispatch.
- [ ] Implement PE SwiGLU MLP and output projection.
- [ ] Implement audio DAC Conv1d/Snake frontend and padding-mask downsampling.
- [ ] Implement MaskedGroupNorm sequence ResNet patch embedder.
- [ ] Implement video feature projection, normalization, and `[B,T]` flatten/restore ABI.
- [ ] Implement nearest-neighbor video-to-audio alignment, including masked per-example path or guarded rejection.
- [ ] Implement contrastive similarity matrices with correct orientation.
- [ ] Add parity tests for custom math and one PE encoder block.
- [ ] Add audio encoder, video post-wrapper, audio-video fusion, and full contrastive parity tests.
- [ ] Benchmark processor, encoder branches, fusion, text, and similarity GEMM separately.
