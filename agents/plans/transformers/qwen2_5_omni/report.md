# Qwen2.5-Omni Transformers Audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: Qwen/Qwen2.5-Omni-7B primary; Qwen/Qwen2.5-Omni-3B and quantized 7B variants for variation checks
Config source: Hugging Face raw config/preprocessor/tokenizer/index snapshots saved under _sources/
Primary runtime target: multimodal generation, text output first; audio output is a second-stage target
Dinoml assumptions: inference-only first, CUDA GPU target, faithful PyTorch axes before guarded layout rewrites, staged encoders/projectors/prefill/decode
```

Source files inspected:

- `transformers/src/transformers/models/qwen2_5_omni/modeling_qwen2_5_omni.py`
- `transformers/src/transformers/models/qwen2_5_omni/modular_qwen2_5_omni.py`
- `transformers/src/transformers/models/qwen2_5_omni/configuration_qwen2_5_omni.py`
- `transformers/src/transformers/models/qwen2_5_omni/processing_qwen2_5_omni.py`
- Comparison files: `qwen2_5_vl/modeling_qwen2_5_vl.py`, `qwen2_5_vl/processing_qwen2_5_vl.py`, `qwen2_audio/modeling_qwen2_audio.py`, `qwen2_audio/processing_qwen2_audio.py`

Snapshots saved:

- `_sources/Qwen2.5-Omni-3B_config.json`
- `_sources/Qwen2.5-Omni-7B_config.json`
- `_sources/Qwen2.5-Omni-7B-AWQ_config.json`
- `_sources/Qwen2.5-Omni-7B-GPTQ-Int4_config.json`
- `_sources/Qwen2.5-Omni-7B_preprocessor_config.json`
- `_sources/Qwen2.5-Omni-7B_tokenizer_config.json`
- `_sources/Qwen2.5-Omni-7B_generation_config.json`
- `_sources/Qwen2.5-Omni-7B_model.safetensors.index.json`
- `_sources/comparison_Qwen2.5-VL-7B-Instruct_config.json`
- `_sources/comparison_Qwen2-Audio-7B-Instruct_config.json`
- `_sources/Qwen2.5-Omni-7B_hf_api.json`

The generated `modeling_*.py` and `configuration_*.py` files say they are generated from `modular_qwen2_5_omni.py`; the modular file is authoritative for future Transformers edits, while this audit uses the generated modeling file as the exact runtime source basis.

`processor_config.json` is absent in the 7B repo; the processor contract comes from `preprocessor_config.json`, tokenizer config, and `processing_qwen2_5_omni.py`.

## 2. High-level architecture

Qwen2.5-Omni is a composite multimodal model:

```text
CPU/data preprocessing
  -> audio log-mel / image-video patch packing / text tokenization
  -> placeholder token expansion
  -> thinker encoders: audio encoder + vision encoder
  -> masked_scatter into text embeddings
  -> thinker causal decoder prefill/decode
  -> text logits/sampling
  -> optional talker causal decoder for speech-code tokens
  -> optional token2wav DiT ODE sampling
  -> optional BigVGAN waveform synthesis
```

Stage decomposition:

- CPU/data pipeline: Whisper feature extraction, Qwen2VL image/video processing, placeholder expansion, grid metadata, and optional audio/video chunk interleaving.
- Cacheable encoders/projectors: audio encoder output tokens, vision encoder output tokens, and text embedding stitch can be validated independently before thinker prefill.
- Thinker: Qwen-style causal LLM with image/audio/video embeddings already inserted into `inputs_embeds`; this is the first useful DinoML target.
- Talker: separate causal decoder conditioned on thinker token embeddings and hidden states; only needed for audio output.
- Token2wav: DiT diffusion sampler plus BigVGAN vocoder; source forces token2wav to fp32 and audio output is batch size 1.

## 3. Important config dimensions

Representative checkpoint dimensions from `config.json`:

| checkpoint | text hidden/layers | text heads/KV/head_dim | text MLP | vocab | vision | audio encoder | talker | token2wav | quant |
|---|---:|---:|---:|---:|---|---|---|---|---|
| Qwen2.5-Omni-3B | 2048 / 36 | 16 / 2 / 128 inferred | 11008 | 151936 | hidden 1280, depth 32, out 2048 | d_model 1280, 32 layers, 20 heads, 128 mel | hidden 896, 24 layers, 14 heads, 2 KV, head_dim 64 | DiT defaults mostly omitted, BigVGAN defaults | none |
| Qwen2.5-Omni-7B | 3584 / 28 | 28 / 4 / 128 inferred | 18944 | 152064 | hidden 1280, depth 32, out 3584 | d_model 1280, 32 layers, 20 heads, 128 mel | hidden 896, 24 layers, 12 heads, 4 KV, head_dim 128 | DiT defaults mostly omitted, BigVGAN defaults | none |
| Qwen2.5-Omni-7B-AWQ | 3584 / 28 | 28 / 4 / 128 inferred | 18944 | 152064 | same as 7B | same as 7B | same as 7B | DiT explicit 1024/22/16 | AWQ |
| Qwen2.5-Omni-7B-GPTQ-Int4 | 3584 / 28 | 28 / 4 / 128 inferred | 18944 | 152064 | same as 7B | same as 7B | same as 7B | DiT explicit 1024/22/16 | GPTQ |

Common config values:

- RoPE theta: `1000000.0` for thinker/talker text; multimodal RoPE section `[16, 24, 24]`.
- Max text positions: `32768`.
- `use_sliding_window=False` in inspected Omni checkpoints, so all thinker/talker layers are full causal attention despite `sliding_window=32768`.
- Vision patch: Conv3d kernel/stride `[temporal_patch_size=2, patch=14, patch=14]`, `spatial_merge_size=2`, window size 112, full attention vision blocks `[7, 15, 23, 31]`.
- Audio preprocessing: 16 kHz, 128 mel bins, `n_fft=400`, `hop_length=160`, max samples 4,800,000, chunk length 300 seconds.
- Token ids: audio `<|AUDIO|>` 151646, audio BOS/EOS 151647/151648, vision BOS/EOS 151652/151653, image `<|IMAGE|>` 151655, video `<|VIDEO|>` 151656.
- Source default `enable_audio_output=True`; model `from_pretrained` also requires `spk_dict.pt` for speakers.

## 3a. Family variation traps

- The 3B and 7B text dimensions differ, and the 3B talker uses `head_dim=64` while 7B talker uses `head_dim=128`.
- `hidden_size != num_attention_heads * head_dim` for the 7B talker: hidden size is 896, but 12 heads * 128 = 1536. The attention output width is `num_heads * head_dim` and `o_proj` maps that back to hidden size.
- GQA is required in thinker and talker: text 7B has 28 Q heads and 4 KV heads; talker 7B has 12 Q heads and 4 KV heads.
- Q/K/V projections have bias in thinker/talker attention; `o_proj` has no bias. MLP projections are bias-free in text/talker but vision MLP uses bias.
- The public 7B config lists architecture `Qwen2_5OmniModel`, while the in-library runtime class is `Qwen2_5OmniForConditionalGeneration`; treat the source class as the required runtime target.
- The plain 7B config omits many nested DiT fields that the config class supplies by default. DinoML should materialize effective defaults before shape planning.
- Quantized AWQ/GPTQ configs are loading/provider work, not different source graph definitions. DinoML should reject or route them until weight-format support exists.
- Placeholder counts must match encoded feature rows exactly; this is enforced through `masked_scatter` shape checks for image/video and audio length checks.
- `use_audio_in_video=True` changes text expansion order by interleaving video and audio placeholders per time chunk. It also changes M-RoPE position construction.
- Token2wav is forced to fp32 and falls back to SDPA if flash/eager attention is requested.
- Audio output is not batched: `generate` raises for batch size > 1 when returning audio.
- Layout-sensitive regions: audio Conv1d uses `[batch, mel, time]`; vision patch Conv3d receives flattened temporal/spatial patches then views as `[N, C, T, H, W]`; BigVGAN uses `[batch, channels, time]`. Do not apply global NHWC translation.

## 4. Operator coverage checklist

Tensor/layout ops:

- `view`, `reshape`, `transpose`, `permute`, `contiguous`, `flatten`, `split`, `cat`, `stack`, `repeat`, `repeat_interleave`, `expand`, `pad`, `argsort`, `unique_consecutive`, boolean masks, masked fill, masked scatter, advanced indexing.
- Packed sequence metadata: `cu_seqlens` for audio and vision varlen attention; window indices and reverse indices for vision.
- Dynamic length math: `(length - 1) // 2 + 1`, `(length - 2) // 2 + 1`, cumsum, chunk splitting.

Neural network primitives:

- Embedding, Linear, Conv1d, Conv3d, ConvTranspose1d, AvgPool1d, LayerNorm, RMSNorm, SiLU, GELU, ReLU, Tanh, Sigmoid, Softmax, Dropout disabled in inference, clamp.
- BigVGAN-specific 1D residual blocks, reflect/replicate padding, grouped ConvTranspose1d for sinc upsample, periodic `SnakeBeta`.
- ECAPA/TDNN speaker encoder for token2wav conditioning.

Attention primitives:

- Causal self-attention with GQA and RoPE for thinker/talker.
- Noncausal packed self-attention for audio encoder.
- Noncausal packed/windowed self-attention for vision encoder.
- Noncausal block-masked DiT attention; only the first head receives RoPE in source.

Position/rotary:

- Text/talker M-RoPE with 3 position streams and `mrope_section=[16,24,24]`.
- Vision 2D RoPE over height/width positions before vision attention.
- DiT 1D RoPE with source-specific interleaved `rotate_half_codec`.
- Sinusoidal absolute positions in the audio encoder and sinusoidal timestep embedding in DiT.

Generation/cache:

- `DynamicCache` per thinker/talker layer, storing K/V after RoPE and before KV repeat.
- `rope_deltas` persisted on the model object to continue multimodal positions during decode.
- Prepare-input rules drop image/video/audio tensors after the first cached iteration.
- Optional generation heads: thinker `lm_head` over text vocab; talker `codec_head` over 8448 codec tokens.

Preprocessing-coupled ops:

- WhisperFeatureExtractor-compatible log-mel pipeline is expected in CPU/data pipeline initially.
- Qwen2VL image/video processor packs `pixel_values`, `pixel_values_videos`, `image_grid_thw`, `video_grid_thw`, `video_second_per_grid`.
- Tokenizer special-token expansion determines exact multimodal sequence length.

Quantized/packed weight ops:

- Native source does not implement AWQ/GPTQ math directly. Quantized variants depend on Transformers quantization integration and should be treated as separate weight-loading/provider contracts.

## 5. Layer/block breakdown

Thinker audio encoder:

```text
input_features [mel, total_valid_time] after feature_attention_mask packing
split into chunks of n_window*2 = 200 frames
pad chunks -> Conv1d(128 -> 1280, k=3,p=1) + GELU
Conv1d(1280 -> 1280, k=3,stride=2,p=1) + GELU
+ sinusoidal position embedding
repeat 32:
  LayerNorm
  noncausal packed MHA, 20 heads, head_dim 64, q/v/out bias, k no bias
  residual
  LayerNorm
  Linear(1280 -> 5120) -> GELU -> Linear(5120 -> 1280)
  residual
split by after-CNN lengths
AvgPool1d(kernel=2,stride=2)
LayerNorm -> Linear(1280 -> text hidden)
```

Vision encoder:

```text
pixel_values packed patches -> view [-1, 3, 2, 14, 14]
Conv3d(3 -> 1280, kernel=stride=[2,14,14], bias=False)
compute 2D rotary positions and window_index
repeat 32:
  RMSNorm
  noncausal MHA, 16 heads, head_dim 80, q/k/v/proj bias
    blocks 7,15,23,31 use full image/video cu_seqlens
    other blocks use local window cu_seqlens
  residual
  RMSNorm
  SwiGLU MLP 1280 -> 3420 -> 1280 with bias
  residual
PatchMerger: RMSNorm, reshape 4 spatial patches, Linear(5120 -> 5120), GELU, Linear(5120 -> text hidden)
reverse window order
```

Thinker decoder block, repeated 28 for 7B and 36 for 3B:

```text
x = RMSNorm(x)
q = Linear(hidden -> num_heads*head_dim, bias=True)
k/v = Linear(hidden -> num_kv_heads*head_dim, bias=True)
q,k = multimodal RoPE(q,k, cos/sin, mrope_section)
k,v = KV cache update if enabled
attn = causal GQA attention with optional sliding_window from layer_type
x = residual + Linear(num_heads*head_dim -> hidden, bias=False)
x = RMSNorm(x)
x = residual + Linear(SiLU(gate) * up -> hidden)
```

Talker decoder:

```text
codec/input embeddings have width embedding_size, then thinker_to_talker_proj maps to talker hidden
same Qwen decoder block as thinker, but config can have hidden_size != num_heads*head_dim
codec_head maps hidden -> vocab_size 8448
```

Token2wav:

```text
speech codec tokens -> DiTCodecEmbedding, repeated by config.repeats
reference mel + speaker conditioning -> ECAPA TDNN speaker path
concat mel state, speaker embedding, reference conditioning, codec embedding -> Linear(... -> 1024)
DiT ODE sample, default 10 RK4 steps:
  timestep embedding
  repeat 22 DiT blocks with AdaLayerNormZero, noncausal block mask, first-head RoPE, GELU MLP
  final AdaLayerNormZero -> Linear(1024 -> 80 mel)
BigVGAN:
  process mel exp/log10/normalize
  Conv1d(80 -> 1536)
  six ConvTranspose1d upsample stages [5,3,2,2,2,2]
  AMP residual blocks with SnakeBeta activations
  Conv1d(... -> 1), clamp to [-1,1], squeeze, cpu()
```

## 6. Attention requirements

Thinker and talker:

- Causal self-attention, no cross-attention.
- GQA/MQA-style repeat: cached/stored K/V shape before repeat is `[batch, num_key_value_heads, seq, head_dim]`; attention expands to Q head count by repeat.
- Keys are cached after multimodal RoPE has been applied.
- Eager path computes `matmul(q, k.T) * head_dim**-0.5`, adds mask, softmax in fp32, casts back, then matmul V.
- Source can dispatch through SDPA/FlashAttention interfaces. For DinoML, a GQA FlashAttention prefill/decode path is high priority, but M-RoPE and cache update order must be preserved.
- Sliding window support exists in config/layer types, but inspected configs disable it; first integration can reject `use_sliding_window=True`.

Audio encoder:

- Noncausal packed self-attention over chunks. FA2 uses `cu_seq_lens_q/k` and `max_length_q/k`; non-FA2 builds an approximate 4D block mask with `torch.finfo(dtype).min` outside chunk blocks.

Vision encoder:

- Noncausal attention over packed image/video patches. Most blocks attend within local windows; full attention blocks use full per-frame/image/video cu-seqlens.
- Windowing happens by reindexing tokens before blocks and reversing after merger.

DiT:

- Noncausal attention with boolean block mask derived from `block_j - block_i`, layer-specific look-back/look-ahead, and source-specific first-head-only RoPE.

## 7. Position encoding and custom math

Multimodal RoPE uses three position streams:

```python
def apply_omni_mrope(q, k, cos, sin, sections):
    sections = sections * 2
    cos = cat([part[i % 3] for i, part in enumerate(cos.split(sections, dim=-1))], dim=-1)
    sin = cat([part[i % 3] for i, part in enumerate(sin.split(sections, dim=-1))], dim=-1)
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin
```

`get_rope_index` is runtime-shape-dependent. For pure text, temporal/height/width positions are identical. For image/video tokens, it uses grid metadata and `spatial_merge_size`; video temporal positions additionally use `video_second_per_grid * position_id_per_seconds`. With `use_audio_in_video=True`, audio spans are interleaved into video chunks and affect positions.

Vision RoPE uses height/width position ids produced from `grid_thw` after spatial merge and applies ordinary half-rotation in fp32 before casting back.

DiT RoPE is not the same rotate function: it reshapes pairs in the last dimension and only applies RoPE to `query[:, :1]` and `key[:, :1]`.

Precomputable: inverse frequencies and static rotary tables up to max grid/sequence. Dynamic: multimodal `position_ids`, `rope_deltas`, `cu_seqlens`, window indices, and video/audio chunk interleaving.

## 8. Preprocessing and input packing

Processor outputs:

- Text: `input_ids`, `attention_mask`.
- Images: `pixel_values`, `image_grid_thw`.
- Videos: `pixel_values_videos`, `video_grid_thw`, `video_second_per_grid`.
- Audio: `input_features`, `feature_attention_mask`.

Audio contract:

- 16 kHz waveform input, WhisperFeatureExtractor, 128 mel bins, `n_fft=400`, `hop_length=160`, right padding to max length.
- Processor computes `input_lengths = (feature_attention_mask.sum(-1) - 1) // 2 + 1` and placeholder count `(input_lengths - 2) // 2 + 1`.
- Runtime `get_audio_features` packs valid frames with boolean indexing and passes `feature_lens`/`aftercnn_lens` into the audio encoder.

Vision/video contract:

- Preprocessor uses Qwen2VL image processor settings: `patch_size=14`, `temporal_patch_size=2`, `merge_size=2`, min pixels 3136, max pixels 12845056.
- Placeholder count is `grid_thw.prod() // merge_size**2`.
- Video uses `video_second_per_grid = temporal_patch_size / fps` from processor kwargs.

Embedding stitch:

- The model first embeds all text tokens.
- Audio/image/video features are converted to text hidden dtype/device and inserted with `inputs_embeds.masked_scatter(mask, features)`.
- Image/video feature counts are checked against placeholder counts. Audio count is checked against `audio_output_lengths`.

Audio output controller:

- Full Omni `generate` first calls thinker generation. If audio is requested, it requests hidden states, builds talker text/code inputs, samples codec tokens with top-k/top-p/temperature defaults, then calls token2wav with speaker `cond` and `ref_mel` from `spk_dict.pt`.
- Audio output supports only batch size 1 in source.

## 9. Graph rewrite / lowering opportunities

### Rewrite: non-overlap Conv3d patch embed -> Linear

Source pattern:

```text
view [-1, 3, 2, 14, 14] -> Conv3d(3 -> 1280, kernel=stride=[2,14,14], bias=False) -> view [-1,1280]
```

Replacement:

```text
WindowFlatten(C,T,H,W) -> MatMul(weight_flat.T)
```

Preconditions: kernel equals stride, padding 0, dilation 1, groups 1, packed pixel values already match the source view order. Weight transform is `conv.weight.reshape(out_channels, in_channels*T*H*W)`. Failure cases: arbitrary raw NCHW/NCTHW tensors, changed patch sizes, or dynamic layout translation without verifying flatten order.

### Rewrite: Q/K/V separate linears -> packed projection

Preconditions: same input tensor, all projections enabled, exact bias handling preserved. Thinker/talker split order is all-Q, all-K, all-V as separate modules, not packed by head. Replacement can emit one GEMM with concatenated weight rows `[q; k; v]` and split outputs into Q/K/V widths.

Failure cases: quantized weight loaders with separate metadata, tensor-parallel sharding, or talker configs where output projection width differs from hidden size.

### Rewrite: SwiGLU MLP fusion

Source pattern:

```text
down_proj(silu(gate_proj(x)) * up_proj(x))
```

Preconditions: activation `silu`, same input, no dropout, bias flag known. Replacement: two-output GEMM or packed gate/up GEMM, fused SiLU multiply, down GEMM.

### Rewrite: placeholder masked_scatter -> indexed copy

Source pattern: boolean mask expanded to hidden width, `masked_scatter`.

Replacement: precompute placeholder row indices from `input_ids`, validate count, copy feature matrix into `inputs_embeds[index, :]`.

Preconditions: placeholder IDs known, feature rows are contiguous and already ordered like processor expansion. Failure cases: `inputs_embeds` passed without `input_ids`, or custom token embeddings equal to special token embeddings.

### Rewrite: local vision attention windows

Source pattern: compute `window_index`, reorder tokens, use packed `cu_window_seqlens`, then reverse.

Replacement: local-window attention kernel over packed windows. Preconditions: `window_size`, `patch_size`, `spatial_merge_size`, and grid divisibility/padding rules match source. Preserve full-attention blocks `[7,15,23,31]`.

### Layout guard

Do not globally convert to NHWC. Candidate local rewrites:

- Conv3d patch embed can lower to GEMM from the existing packed view.
- BigVGAN and audio Conv1d regions should stay NCT unless an entire local Conv1d chain is converted with all channel/time axis rewrites.
- Axis-sensitive ops needing guards: Conv1d dim=1 channel, `feature_attention_mask` boolean indexing, `AvgPool1d`, BigVGAN ConvTranspose1d, `softmax(dim=-1)` in attention, `mean(dim=2)` in speaker pooling.

## 10. Kernel fusion candidates

Highest priority:

- RMSNorm for thinker/talker/vision.
- GQA FlashAttention with KV cache and RoPE-applied cached keys.
- M-RoPE position gather/application fused near Q/K projection.
- Packed QKV projection for thinker/talker.
- SwiGLU MLP fusion.
- Placeholder indexed-copy stitch.

Medium priority:

- Vision patch Conv3d-to-GEMM and PatchMerger MLP.
- Vision local-window packed attention with `cu_seqlens`.
- Audio Conv1d + GELU and packed noncausal attention.
- Last-token-only logits for decode.
- Talker conditioning projection plus codec head for audio output.

Lower priority:

- Token2wav DiT block-mask attention and RK4 loop.
- BigVGAN ConvTranspose1d/AMPBlock fusion.
- ECAPA speaker encoder kernels.
- Quantized AWQ/GPTQ provider support.

## 11. Runtime staging plan

Stage 1: parse config, materialize effective nested defaults, load dense 7B/3B weights, and run tokenizer/preprocessor outside DinoML.

Stage 2: implement thinker text-only prefill/decode with M-RoPE degenerated to identical 3 streams, GQA KV cache, tied input/lm-head alias where present.

Stage 3: add image/video encoder parity and embedding stitch. Validate vision encoder output and full multimodal prefill logits.

Stage 4: add audio encoder parity and audio placeholder stitch. Keep Whisper feature extraction in CPU pipeline.

Stage 5: add optimized attention/fusions for thinker and encoders, including packed QKV and local vision windows.

Stage 6: add talker decoder for speech-code generation. Stub token2wav initially by returning codec tokens.

Stage 7: add token2wav fp32 path: DiT ODE sampling then BigVGAN waveform.

Stage 8: evaluate quantized variants as separate loading/provider work.

## 12. Parity and validation plan

- Unit tests for M-RoPE: pure text, image-only, video-only, audio-in-video interleaving, and decode continuation with `rope_deltas`.
- Unit tests for audio length math and placeholder expansion from `feature_attention_mask`.
- Vision encoder parity on small synthetic `grid_thw`, including window/full attention block switch and reverse index.
- Audio encoder parity on synthetic log-mel features with ragged chunk lengths.
- Single decoder block parity for thinker and talker, including GQA cache update.
- Full thinker text-only prefill logits, then one-token decode logits.
- Image+text, audio+text, video+text prefill logits after embedding stitch.
- Talker code generation parity with fixed sampling disabled or controlled seed.
- Token2wav parity with fixed initial noise or source-provided deterministic initial state; otherwise validate shape/range and approximate waveform statistics separately.
- Tolerances: fp32 `rtol=1e-4, atol=1e-4`; bf16/fp16 block parity `rtol=5e-2, atol=5e-2` for attention-heavy paths, tighter for isolated Linear/RMSNorm.

## 13. Performance probes

- Processor throughput: image/video resizing/patch packing and Whisper log-mel extraction.
- Vision encoder throughput by resolution/grid and local/full attention blocks.
- Audio encoder throughput by waveform seconds and chunk count.
- Thinker prefill tokens/sec for text-only, image+text, audio+text, video+text.
- Decode tokens/sec with KV cache, batch/sequence sweep, and last-token logits optimization.
- KV cache memory for thinker and talker separately.
- Placeholder stitch overhead by multimodal token count.
- Talker codec tokens/sec and sampling overhead.
- Token2wav DiT step time by codec length, `num_steps`, guidance scale, and block size.
- BigVGAN waveform synthesis throughput and CPU transfer cost caused by source `.cpu()` return.
- Quantized load/dequant/provider comparison for AWQ/GPTQ only after dense parity is stable.

## 14. Skip/defer list

- Training, loss paths, gradient checkpointing.
- Quantized AWQ/GPTQ variants.
- Audio output batching beyond batch size 1.
- Token2wav in first text-output integration.
- `use_sliding_window=True` text/talker configs, since inspected checkpoints disable it.
- Remote/custom code not present in pinned in-library source.
- Multi-GPU tensor parallel plans.
- End-to-end chat template fidelity beyond special-token/placeholder expansion.
- Full GPU preprocessing for Whisper features and image/video resizing.

## 15. Final implementation checklist

- [ ] Parse `Qwen2_5OmniConfig` and nested thinker/talker/token2wav defaults.
- [ ] Reject or route unsupported AWQ/GPTQ quantized configs.
- [ ] Load dense weights and preserve tied thinker embedding/lm-head alias when present.
- [ ] Implement Qwen2.5-Omni RMSNorm.
- [ ] Implement thinker/talker GQA attention with biased Q/K/V and bias-free O projection.
- [ ] Implement M-RoPE and `rope_deltas` continuation.
- [ ] Implement DynamicCache-compatible K/V ABI before repeat expansion.
- [ ] Implement SwiGLU MLP fusion path.
- [ ] Implement placeholder indexed-copy stitch for audio/image/video features.
- [ ] Implement vision Conv3d patch embedding or guarded Conv3d-to-GEMM rewrite.
- [ ] Implement vision packed local/full attention and PatchMerger.
- [ ] Implement audio Conv1d encoder, chunk packing, noncausal packed attention, and output length math.
- [ ] Add thinker text-only prefill/decode parity tests.
- [ ] Add image/audio/video prefill parity tests.
- [ ] Add talker codec decoder parity tests.
- [ ] Add token2wav fp32 DiT and BigVGAN parity tests after text path is stable.
- [ ] Benchmark processor, encoders, prefill, decode, talker, and token2wav separately.

