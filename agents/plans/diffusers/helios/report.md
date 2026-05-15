# Diffusers Helios Operator and Integration Report

Target slug: `helios`

## 1. Source basis

```text
Diffusers commit/version:
  diffusers at b3a515080752a3ba7ca92161e25530c7f280f629.

Model id(s):
  BestWishYsh/Helios-Base
  BestWishYsh/Helios-Mid
  BestWishYsh/Helios-Distilled
  YiYiXu/tiny-helios-pipe as open tiny/debug config evidence.

Config sources:
  H:/configs/BestWishYsh/Helios-Base/model_index.json
  H:/configs/BestWishYsh/Helios-Mid/model_index.json
  H:/configs/BestWishYsh/Helios-Distilled/model_index.json
  H:/configs/YiYiXu/tiny-helios-pipe/model_index.json
  Official raw Hugging Face reads, not saved because this worker owns only this report:
    */transformer/config.json
    */vae/config.json
    */scheduler/scheduler_config.json
    */text_encoder/config.json
    */tokenizer/tokenizer_config.json
    */tokenizer/special_tokens_map.json
    */guider/guider_config.json
    */modular_model_index.json
    */transformer/diffusion_pytorch_model.safetensors.index.json
    */text_encoder/model.safetensors.index.json
    Base/Mid transformer_init indexes and Distilled transformer_ode index.

Pipeline files inspected:
  diffusers/src/diffusers/pipelines/helios/pipeline_helios.py
  diffusers/src/diffusers/pipelines/helios/pipeline_helios_pyramid.py
  diffusers/src/diffusers/pipelines/helios/pipeline_output.py
  diffusers/docs/source/en/api/pipelines/helios.md
  diffusers/docs/source/en/using-diffusers/helios.md

Model files inspected:
  diffusers/src/diffusers/models/transformers/transformer_helios.py
  diffusers/src/diffusers/models/autoencoders/autoencoder_kl_wan.py

Scheduler/processors/helpers inspected:
  diffusers/src/diffusers/schedulers/scheduling_helios.py
  diffusers/src/diffusers/schedulers/scheduling_helios_dmd.py
  diffusers/src/diffusers/video_processor.py
  diffusers/src/diffusers/image_processor.py
  diffusers/src/diffusers/models/attention_dispatch.py
  diffusers/src/diffusers/loaders/lora_pipeline.py
  Tests:
    diffusers/tests/pipelines/helios/test_helios.py
    diffusers/tests/models/transformers/test_models_transformer_helios.py

External component configs inspected:
  UMT5EncoderModel / T5TokenizerFast configs from official Helios repos.

Any missing files or assumptions:
  Official BestWishYsh JSON configs are public and raw-readable; no gated
  production config blocker remained. Local cache only had model_index.json for
  Base/Mid/Distilled. hf-internal-testing tiny Helios transformer returned 401
  for unauthenticated raw reads, but public YiYiXu and official production
  configs cover the operator contract. This report focuses on the non-deprecated
  Helios Base/Pyramid family. Multi-GPU/context parallel, callbacks/interrupts,
  XLA/NPU/MPS/Flax/ONNX, safety, training, dropout, losses, and gradient
  checkpointing are out of scope.
```

Key source anchors:

| Anchor | File |
| --- | --- |
| `HeliosPipeline.__call__` | `src/diffusers/pipelines/helios/pipeline_helios.py:447` |
| `HeliosPyramidPipeline.__call__` | `src/diffusers/pipelines/helios/pipeline_helios_pyramid.py:510` |
| `HeliosTransformer3DModel.forward` | `src/diffusers/models/transformers/transformer_helios.py:658` |
| `HeliosTransformerBlock` | `src/diffusers/models/transformers/transformer_helios.py:374` |
| `HeliosAttnProcessor` | `src/diffusers/models/transformers/transformer_helios.py:99` |
| `HeliosScheduler` | `src/diffusers/schedulers/scheduling_helios.py:35` |
| `HeliosDMDScheduler` | `src/diffusers/schedulers/scheduling_helios_dmd.py:35` |
| `AutoencoderKLWan` | `src/diffusers/models/autoencoders/autoencoder_kl_wan.py:960` |
| `HeliosLoraLoaderMixin` | `src/diffusers/loaders/lora_pipeline.py:3444` |

## 2. Pipeline and component graph

Helios is an autoregressive video latent pipeline. The Base checkpoint uses
`HeliosPipeline` with one resolution per chunk and `HeliosScheduler`. Mid and
Distilled use `HeliosPyramidPipeline`, which denoises each chunk through
coarse-to-fine pyramid stages; Distilled swaps in `HeliosDMDScheduler`.

```text
prompt / negative prompt
  -> T5TokenizerFast + UMT5EncoderModel
  -> prompt embeddings [B,L,4096] and optional negative embeddings
  -> optional image/video preprocess + Wan VAE encode for I2V/V2V seeds
  -> latent history buffers [B,16,T_history,H/8,W/8]
  -> chunk loop over latent windows:
       prepare noisy chunk [B,16,9,H/8,W/8]
       HeliosTransformer3DModel(noisy chunk, history latents, timestep, text)
       optional separate unconditional transformer call + CFG/CFG-Zero*
       HeliosScheduler or HeliosDMDScheduler step
       append chunk to history
       AutoencoderKLWan decode recent chunk
  -> VideoProcessor postprocess_video
```

Required first-slice components for Base T2V:

- `T5TokenizerFast` and `UMT5EncoderModel`, or externally supplied prompt
  embeddings for the first denoiser slice.
- `HeliosTransformer3DModel` denoiser with short/mid/long history inputs.
- `HeliosScheduler` with Base single-stage dynamic-shift UniPC by default.
- `AutoencoderKLWan` decode boundary for generated latent chunks.
- `VideoProcessor` postprocess boundary.

Optional or variant components:

- Image input and video input are supported by the same `__call__` method. They
  add VAE encode, latent standardization, random noising, and history seeding.
- `HeliosPyramidPipeline` adds pyramid downsample/upsample, block-noise
  re-noising, multi-stage scheduler calls, CFG-Zero*, and DMD paths.
- `transformer` is marked optional in both pipelines, but it is required for
  actual generation.
- Modular pipeline metadata adds guider components:
  `ClassifierFreeGuidance` for Base, `ClassifierFreeZeroStarGuidance` for Mid,
  and `ClassifierFreeGuidance` scale 1.0 for Distilled.

Separate candidate reports:

| Candidate | Classes/files | Runtime delta |
| --- | --- | --- |
| `helios_pyramid_mid` | `HeliosPyramidPipeline`, `HeliosScheduler`, `ClassifierFreeZeroStarGuidance` | Coarse-to-fine pyramid stages, CFG-Zero* optimized scale, block-correlated noise, three stage scheduler ranges. |
| `helios_distilled_dmd` | `HeliosPyramidPipeline`, `HeliosDMDScheduler`, `transformer_ode` weights | DMD x0 path, per-stage start tensors, scheduler `add_noise` from predicted x0, guidance scale effectively ignored in classic CFG. |
| `helios_i2v_v2v` | Same pipeline classes, `prepare_image_latents`, `prepare_video_latents` | VAE encode, image/fake-image latent seeding, video chunk encode, frame-wise noise ranges, first-frame preservation. |
| `helios_lora_adapters` | `HeliosLoraLoaderMixin`, `PeftAdapterMixin` on transformer | Transformer-only LoRA load/fuse/unfuse/hotswap; also converts non-Diffusers/Musubi Wan LoRA keys. |
| `helios_wan_vae_codec` | `AutoencoderKLWan` | Shared Wan 3D causal VAE encode/decode, slicing/tiling/cache behavior, temporal compression. |
| `helios_modular_pipeline` | `src/diffusers/modular_pipelines/helios/*.py` | Blockized version of the same stages plus guider components; useful for integration but not a different operator family. |

No Helios folder classes were found for textual inversion, IP-Adapter,
ControlNet, T2I-Adapter, GLIGEN, inpaint, depth2img, or upscaling.

## 3. Important config dimensions

Representative checkpoint sweep:

| Repo | Pipeline | Scheduler | Stages | Guidance | Denoiser | VAE |
| --- | --- | --- | ---: | --- | --- | --- |
| `BestWishYsh/Helios-Base` | `HeliosPipeline` | `HeliosScheduler`, `scheduler_type=unipc`, `prediction_type=flow_prediction` | 1, `stage_range=[0,1]` | CFG scale 5.0 in guider config | 40 layers, 40 heads x 128, patch `(1,2,2)` | `AutoencoderKLWan`, z=16, temporal/spatial scale 4/8 |
| `BestWishYsh/Helios-Mid` | `HeliosPyramidPipeline`, `is_cfg_zero_star=true` | `HeliosScheduler`, UniPC | 3, thirds | CFG-Zero* scale 5.0, `zero_init_steps=2` in modular guider | same denoiser | same VAE |
| `BestWishYsh/Helios-Distilled` | `HeliosPyramidPipeline`, `is_distilled=true` | `HeliosDMDScheduler`, `scheduler_type=dmd` | 3, thirds | guider scale 1.0; classic CFG warning says ignored if >1 | same denoiser, plus `transformer_ode` side weights in repo | same VAE |
| `YiYiXu/tiny-helios-pipe` | `HeliosPipeline` | tiny `HeliosScheduler`, stages 1, no dynamic shift | 1 | debug | 2 layers, 2 heads x 12, text dim 32 | tiny Wan VAE base_dim 3 |

Transformer config facts:

| Field | Official value | Source default / note |
| --- | ---: | --- |
| `in_channels` / `out_channels` | 16 / 16 | VAE latent channel count, not packed channel width. |
| `patch_size` | `[1,2,2]` | Conv3d patchify over latent T/H/W; no temporal patch compression. |
| `num_attention_heads` x `attention_head_dim` | 40 x 128 | Inner dim 5120. |
| `num_layers` | 40 | All blocks have self-attn, cross-attn, FFN. |
| `ffn_dim` | 13824 | FeedForward uses GELU approximate. |
| `text_dim` | 4096 | UMT5 hidden size. |
| `freq_dim` | 256 | Timestep sinusoidal embedding input. |
| `rope_dim` | `[44,42,42]` | Temporal, height, width RoPE sections; sum 128 per head. |
| `cross_attn_norm` | true | Cross-attention input LayerNorm with affine. |
| `qk_norm` | `"rms_norm_across_heads"` | Q/K RMSNorm over all heads*dim. |
| `guidance_cross_attn` | true | Cross-attention applies only to original chunk tokens, not history tokens. |
| `zero_history_timestep` | true | History tokens get timestep 0 modulation. |
| `has_multi_term_memory_patch` | true | Enables short/mid/long history Conv3d paths. |
| `is_amplify_history` | false in official configs | Source can scale history self-attention keys when enabled. |

Text encoder:

| Component | Official config |
| --- | --- |
| `UMT5EncoderModel` | `d_model=4096`, `num_layers=24`, `num_heads=64`, `d_ff=10240`, vocab 256384, `torch_dtype=float32` in config. Docs load pipeline with `torch_dtype=torch.bfloat16`. |
| Tokenizer | T5/UMT5 tokenizer files, huge additional-special-token map. Pipeline cleans prompt text, pads/truncates to `max_sequence_length`, default 512 in `__call__`. |

VAE:

| Field | Official config or source default |
| --- | --- |
| Class | `AutoencoderKLWan` |
| `z_dim` | 16 |
| `base_dim` | 96 |
| `dim_mult` | `[1,2,4,4]` |
| `num_res_blocks` | 2 |
| `temperal_downsample` | `[false,true,true]` |
| `scale_factor_temporal` / spatial | Omitted in official config, source defaults 4 / 8. |
| Latent normalization | Official `latents_mean` and `latents_std`; pipeline uses `std_inv = 1 / latents_std` for standardized runtime latents. |

Weight metadata:

- Transformer safetensors index metadata reports `total_size=57249710336`
  bytes, 1101 weight-map entries, six shards for all three production repos.
- UMT5 text encoder safetensors index metadata reports `total_size=22723641344`
  bytes, 242 entries, five shards.
- VAE is a single `vae/diffusion_pytorch_model.safetensors` file.
- HF API metadata lists Apache-2.0 for the production repos.

Scheduler support:

- Base source default scheduler is `HeliosScheduler` with single-stage dynamic
  shift and UniPC step.
- Mid source default scheduler is `HeliosScheduler` with three stages and
  pyramid stage ranges.
- Distilled source default scheduler is `HeliosDMDScheduler`, linear time shift.
- Recommended first Dinoml scheduler slice: Base `HeliosScheduler` with
  `stages=1`, `scheduler_type="unipc"`, `prediction_type="flow_prediction"`,
  dynamic exponential shift, solver order 2. Add Euler as a simpler diagnostic
  only if parity is explicitly scoped away from the default checkpoint.

## 3a. Family variation traps

- This is a video/autoregressive model, not a Flux-style packed image DiT.
  Source latents are NCTHW `[B,16,T,H/8,W/8]`.
- `HeliosPipeline` and `HeliosPyramidPipeline` share components but have
  materially different denoising loops.
- Default `num_frames=132` and `num_latent_frames_per_chunk=9` imply
  `window_num_frames=(9-1)*4+1=33` decoded frames per chunk. The generated frame
  count is trimmed to `(frames-1)//4*4+1`.
- `history_sizes=[16,2,1]` become long/mid/short history slices. With
  `keep_first_frame=true`, one prefix latent is concatenated into the short
  history path.
- `guidance_cross_attn=true` means history tokens receive self-attention and
  FFN updates, but cross-attention is applied only to the original context
  chunk tokens.
- Official Base uses `stages=1`; Mid/Distilled use three pyramid stages. Do not
  infer pyramid block-noise behavior for Base.
- Distilled DMD scheduler predicts x0 from flow and re-noises from the stored
  noisy start tensor; it is not the same step math as UniPC or Euler.
- VAE config omits `scale_factor_temporal` and `scale_factor_spatial`, but
  `AutoencoderKLWan` source defaults make them 4 and 8. Artifact loaders must
  fill these effective defaults.
- The pipeline validates only height/width divisible by 16, while the VAE
  spatial scale and transformer patch imply practical H/8,W/8 evenness for
  patch size `(1,2,2)`. Treat odd latent H/W as a guarded failure until tested.
- The source includes context-parallel and XLA hooks; they are ignored for this
  CUDA single-device audit, but context-parallel comments in attention dispatch
  are a validation warning for future sequence-splitting providers.
- `is_amplify_history` is source-supported but disabled in official configs.

## 4. Runtime tensor contract

For the default Base call with `B=1`, `height=384`, `width=640`,
`num_latent_frames_per_chunk=9`, and `keep_first_frame=true`:

| Boundary | Tensor | Shape/layout | Notes |
| --- | --- | --- | --- |
| Tokenizer output | `input_ids`, `attention_mask` | `[B,512]` default | CPU/data path. Prompt cleaned with ftfy/html/whitespace helpers. |
| Prompt embeds | `prompt_embeds` | `[B,512,4096]` after pad/truncate | The actual valid length is taken from attention mask, then zero-padded back. |
| Negative embeds | `negative_prompt_embeds` | `[B,512,4096]` if CFG | Separate text encoder pass; no CFG batch concatenation. |
| Latent chunk noise | `latents` | `[B,16,9,48,80]`, NCTHW | Generated per chunk in fp32 then cast to transformer dtype. |
| History latents | `history_latents` | `[B,16,19,48,80]` initially | Long 16, mid 2, short 1, plus optional prefix for short path. |
| Short history input | `latents_history_short` | `[B,16,2,48,80]` when keep first frame | Prefix plus 1x history; Conv3d patch `(1,2,2)`. |
| Mid history input | `latents_history_mid` | `[B,16,2,48,80]` | Replicate pad to multiples `(2,4,4)`, Conv3d patch `(2,4,4)`. |
| Long history input | `latents_history_long` | `[B,16,16,48,80]` | Replicate pad to multiples `(4,8,8)`, Conv3d patch `(4,8,8)`. |
| Main patch tokens | after `patch_embedding` | `[B,9*24*40,5120]=[B,8640,5120]` | Original context length for output. |
| Total tokens | history + main | Approximately 10080 for default Base | Short 3840 + mid 480 + long 960 + main 8640 when all histories are active. |
| Text condition | `encoder_hidden_states` | `[B,512,4096] -> [B,512,5120]` | `PixArtAlphaTextProjection` with GELU-tanh. |
| Timestep | `timestep` | `[B]`, scheduler timestep dtype | Timestep embedding produces modulation for each token; history may use timestep zero. |
| Denoiser output | `noise_pred` | `[B,16,9,48,80]`, NCTHW | Unpatchified from only the original context tokens. |
| Scheduler output | `latents` | `[B,16,9,48,80]` | UniPC or DMD depending checkpoint. |
| Decode input | `current_latents` | `[B,16,<=9,48,80]` | Pipeline converts standardized latent to VAE latent: `latents / (1/std) + mean`, equivalent to `latents * std + mean`. |
| VAE decoded chunk | `current_video` | `[B,3,F,384,640]`, NCTHW | Wan VAE decodes frame by frame with causal conv cache. |
| Pipeline output | `frames` | np/pt/PIL video, batch-major | `VideoProcessor.postprocess_video`; latent output returns standardized history latents. |

Autoencoder encode boundaries:

- I2V image path preprocesses to NCHW image, unsqueezes to NCTHW with `T=1`,
  encodes with Wan VAE, then standardizes `(latent - mean) * (1/std)`.
- I2V also creates `fake_image_latents` by repeating the input image to the
  chunk frame count and taking the last latent frame.
- V2V path preprocesses video to NCTHW, encodes a first-frame latent plus full
  chunk latents, then standardizes the same way.

Cacheable stages:

- Prompt and negative prompt embeddings.
- Scheduler timesteps/sigmas for a fixed stage, step count, resolution, and
  `mu`.
- RoPE frequency tensors for fixed frame indices and latent resolution.
- Encoded image/video latents for repeated I2V/V2V calls.
- History prefix/index tensors for fixed `history_sizes` and
  `num_latent_frames_per_chunk`.

## 5. Operator coverage checklist

Tensor/layout ops:

- NCTHW latent allocation, split/cat over temporal dimension, chunk loops.
- Conv3d patchify and unpatchify reshape/permute:
  `Conv3d -> flatten(2).transpose(1,2)` and
  `reshape(B,Tp,Hp,Wp,pt,ph,pw,Cout) -> permute(0,7,1,4,2,5,3,6) -> flatten`.
- Replicate padding for 3D history patch inputs and RoPE tensors.
- `avg_pool3d` for center downsampling mid/long RoPE tensors.
- Pyramid path: bilinear downsample with scale factor multiply by 2, nearest
  upsample, block-correlated noise reshape/permute, Cholesky, matmul.
- CFG and CFG-Zero* arithmetic: separate positive/negative denoiser calls,
  vector dot/norm optimized scale, zero-init branch.
- Dynamic shift `mu` calculation from image sequence length.

Convolution/downsample/upsample ops:

- Main Conv3d patch embed: `Conv3d(16 -> 5120, kernel=(1,2,2), stride=(1,2,2))`.
- Short history Conv3d same kernel/stride.
- Mid history Conv3d: `Conv3d(16 -> 5120, kernel=(2,4,4), stride=(2,4,4))`.
- Long history Conv3d: `Conv3d(16 -> 5120, kernel=(4,8,8), stride=(4,8,8))`.
- Wan VAE causal Conv3d, Conv2d-like spatial resample helpers, nearest-exact
  upsample, stride temporal down/up paths, residual blocks.

GEMM/linear ops:

- UMT5 text encoder if compiled later: 24-layer encoder with 4096 hidden.
- Text projection: `PixArtAlphaTextProjection(4096 -> 5120)`.
- Timestep MLP: sinusoidal 256 -> `TimestepEmbedding` -> 5120 -> SiLU ->
  Linear to `6*5120`.
- Per block self-attention Q/K/V: 5120 -> 5120 with bias.
- Per block cross-attention Q/K/V: 5120 hidden and 5120 text projection,
  40 heads x 128.
- FeedForward approximate GELU with inner 13824.
- Output projection: 5120 -> `16*1*2*2 = 64`.

Attention primitives:

- Noncausal self-attention over history + current chunk tokens.
- Cross-attention from hidden tokens to UMT5 text embeddings; with
  `guidance_cross_attn=true`, only current chunk tokens attend to text.
- QK RMSNorm over all heads*head_dim before unflatten.
- 3-axis RoPE for self-attention only.
- No attention mask in the Helios denoiser calls inspected.

Normalization/adaptive conditioning:

- `FP32LayerNorm` for norm1/norm2/norm3 and output norm.
- RMSNorm for Q/K projections.
- AdaLN-style scale/shift/gate from `scale_shift_table + timestep_proj`.
- Output norm uses only the original context tokens and timestep embedding.

Scheduler and guidance arithmetic:

- `HeliosScheduler` UniPC flow-prediction conversion, multistep predictor and
  corrector, solver order 2.
- Euler path exists as `scheduler_type="euler"` but is not the sampled
  production default.
- `HeliosDMDScheduler`: x0 conversion and re-noising from `dmd_noisy_tensor`.
- I2V/V2V random sigma noise ranges for image and video latents.

VAE/postprocessing ops:

- Wan VAE `DiagonalGaussianDistribution` encode.
- Causal Conv3d cache state, quant/post-quant 1x1 Conv3d, clamp to `[-1,1]`.
- Video processor PIL/NumPy/Torch conversions and channel/time permutes.

## 6. Denoiser/model breakdown

`HeliosTransformer3DModel.forward`:

```text
hidden_states [B,16,T,H,W]
  -> patch_embedding Conv3d -> [B,5120,T,H/2,W/2]
  -> flatten/transpose -> main tokens [B,S_main,5120]
  -> RoPE from indices_hidden_states and post-patch grid
  -> optional short/mid/long history patch Conv3d paths
  -> concat history tokens before main tokens
  -> timestep embedding and text projection
  -> 40 HeliosTransformerBlock layers
  -> HeliosOutputNorm over only main tokens
  -> Linear 5120 -> 64
  -> unpatchify -> [B,16,T,H,W]
```

History paths:

- Short: same patch size `(1,2,2)` as main chunk.
- Mid: replicate-pad latent tensor to `(2,4,4)` multiples, Conv3d with kernel
  and stride `(2,4,4)`. RoPE is generated at short-grid resolution, padded
  `(2,2,2)`, then `avg_pool3d`.
- Long: replicate-pad to `(4,8,8)` multiples, Conv3d with `(4,8,8)`. RoPE is
  padded `(4,4,4)`, then `avg_pool3d`.

`HeliosTransformerBlock`:

```text
timestep_proj + scale_shift_table -> shift/scale/gate for self-attn and FFN
FP32LayerNorm -> adaptive scale/shift
self-attention with QK RMSNorm + RoPE over history+main tokens
gated residual add
if guidance_cross_attn:
  split history and main
  cross-attend main only to projected text
  concat history back
else:
  cross-attend all tokens
FP32LayerNorm -> adaptive scale/shift
FeedForward GELU approximate
gated residual add
```

Bias flags:

- Attention projections and output projections use bias.
- Norm1/norm3 are FP32 layer norms without affine; cross-attn norm2 is affine
  when enabled.
- RMSNorm Q/K has affine for normal Q/K and no affine for unused added K path.

## 7. Attention requirements

Primary implementation is local `HeliosAttnProcessor`, which calls
`dispatch_attention_fn` from `attention_dispatch.py`.

Required attention variants:

- Self-attention: Q/K/V from the concatenated history+main token stream, no
  mask, noncausal, RoPE applied to Q/K.
- Cross-attention: Q from hidden stream, K/V from projected UMT5 text stream,
  no RoPE, no mask in current pipeline calls.
- Heads/head dim: 40 x 128 for production, 2 x 12 for tiny config.
- QK norm: `torch.nn.RMSNorm(5120)` before unflatten to `[B,S,heads,head_dim]`.
- RoPE: `apply_rotary_emb_transposed` after unflatten, with cos/sin tensor
  produced by `HeliosRotaryPosEmbed` and flattened to match token order.
- History amplification: source can scale self-attention history keys when
  `is_amplify_history=true`, but official configs set false.
- Fused projections: `HeliosAttention.fuse_projections()` can fuse self QKV or
  cross KV for loading/runtime mutation. This is a supported optimization
  surface, not required for first parity.

Flash/provider feasibility:

- Eager/native dispatch defines parity.
- Base self-attention is a plausible flash-style provider target: noncausal,
  no mask, head dim 128, bf16/fp16 expected in docs, sequence around 10k tokens
  for default Base. QK RMSNorm and RoPE remain explicit pre-attention ops unless
  a fused provider admits them with guards.
- Cross-attention is also mask-free and noncausal, but has rectangular
  query/key lengths and no RoPE. It can use flash under the same dtype/head-dim
  constraints.
- Diffusers flash-attn 2/3 and Sage non-varlen paths reject non-None masks;
  this is fine for current Helios calls but must guard future masked text paths.
- Varlen flash/sage paths support normalized bool masks but add flatten/cumsum
  staging. Not first slice.
- Flex and xFormers can represent more masks; use only after parity tests
  confirm tensor layouts and dtype behavior.

## 8. Scheduler and denoising-loop contract

Base `HeliosPipeline`:

```text
image_seq_len = T_chunk * (H/8) * (W/8) / (pt*ph*pw)
sigmas = linspace(0.999, 0.0, num_inference_steps + 1)[:-1]
mu = calculate_shift(image_seq_len, base_image_seq_len, max_image_seq_len,
                     base_shift, max_shift)
scheduler.set_timesteps(num_inference_steps, sigmas=sigmas, mu=mu)
```

For default Base 384x640 and 9 latent frames, `image_seq_len=8640`, which is
above the default `max_image_seq_len=4096`; `calculate_shift` extrapolates
linearly rather than clamping. This should be preserved for parity.

`HeliosScheduler` set-up:

- `stages=1` uses supplied sigmas directly. With dynamic shifting, sigmas are
  shifted by exponential or linear time shift and timesteps become
  `sigmas[:-1] * num_train_timesteps`.
- `stages>1` uses per-stage precomputed timestep and sigma ranges, then applies
  dynamic shift.
- `scheduler_type="unipc"` routes `step()` through UniPC. `scheduler_type="euler"`
  exists but is not the official Base/Mid setting.
- UniPC conversion for `prediction_type="flow_prediction"` uses
  `x0_pred = sample - sigma * model_output` when predicting x0.

Pyramid `HeliosPyramidPipeline`:

- For each chunk, it first downsamples the latent chunk through
  `pyramid_num_stages - 1` bilinear halvings and multiplies by 2.
- Each stage calls `set_timesteps(pyramid_num_inference_steps_list[stage_idx],
  stage_idx, mu=mu)`.
- Stage >0 upsamples by nearest and adds block-correlated noise:
  `latents = alpha * latents + beta * noise`, where alpha/beta depend on
  `ori_start_sigmas[stage_idx]` and `gamma`.
- Mid's CFG-Zero* computes an optimized scale from flattened positive and
  negative predictions and zeroes early predictions when enabled.

Distilled DMD:

- `HeliosDMDScheduler.set_timesteps` adds one extra schedule point, drops the
  last model timestep, and uses linear time shift by default.
- `step()` converts flow prediction to x0 using the current sigma table. If not
  at the final sampling step, it re-noises predicted x0 with `dmd_noisy_tensor`
  at the next timestep; final step returns x0.

Host/runtime split:

- Keep chunk loops, pyramid stage loops, scheduler state, `history_latents`, and
  guidance orchestration host-visible initially.
- Compile a single denoiser step first, then add scheduler pointwise/multistep
  arithmetic as explicit runtime kernels after scheduler parity.

## 9. Position, timestep, and custom math

Custom math to reproduce:

- `calculate_shift`: linear interpolation/extrapolation of `mu` from image
  sequence length.
- `HeliosRotaryPosEmbed`: per-axis temporal/y/x frequencies using
  `theta=10000`, concatenated as cos(t), cos(y), cos(x), sin(t), sin(y), sin(x),
  then permuted to batch/token order.
- `apply_rotary_emb_transposed`: unflatten last dim into pairs and apply cos/sin
  with the transposed cos/sin layout.
- History timestep handling: if `zero_history_timestep=true`, history tokens
  receive a separate timestep-zero modulation while main chunk tokens receive
  the current timestep.
- Adaptive block modulation: `scale_shift_table + timestep_proj` splits into
  six tensors: self shift/scale/gate and FFN shift/scale/gate.
- CFG-Zero* `optimized_scale`: dot(positive, negative) divided by negative
  squared norm with `1e-8` stabilizer.
- Pyramid block noise: block covariance matrix
  `I*(1+gamma) - ones*gamma`, Cholesky factor, then per-block matmul and
  spatial reassembly.
- VAE latent standardization/unstandardization with config mean/std vectors.

Precomputable:

- RoPE for fixed indices/history sizes/resolution.
- UMT5 prompt embeddings and negative prompt embeddings.
- Stage sigmas/timesteps for fixed scheduler config and sequence length.
- Block-noise covariance Cholesky for fixed patch size and gamma.

Dynamic:

- Prompt valid lengths, chunk count from requested frames, image/video seed
  latents, random noise, guidance enablement, pyramid stage resolution, and DMD
  start tensors.

## 10. Preprocessing and input packing

Text:

- Prompt text is cleaned by ftfy/html unescape and whitespace normalization.
- Tokenizer pads/truncates to `max_sequence_length`, adds special tokens, and
  returns attention masks.
- Pipeline uses valid token length from the mask, slices each UMT5 hidden state
  to the valid length, then pads with zeros back to max sequence length.
- Embeddings are repeated for `num_videos_per_prompt`.
- Negative prompt follows the same path only when `guidance_scale > 1`.

Image/video:

- `VideoProcessor.preprocess` handles image input as NCHW, normalized per
  `VaeImageProcessor`.
- `VideoProcessor.preprocess_video` accepts lists/PIL/NumPy/Torch video forms
  and returns NCTHW `[B,C,T,H,W]`.
- I2V image is encoded as a one-frame video plus a repeated fake-video context.
- V2V video is chunked into minimum windows
  `(num_latent_frames_per_chunk-1)*scale_factor_temporal+1`.
- Inference path does not pack latents into 2D token matrices at pipeline level;
  patchification is inside `HeliosTransformer3DModel`.

Postprocess:

- Decode produces NCTHW video tensors, clamped by the VAE.
- `VideoProcessor.postprocess_video` iterates batch items, permutes to
  frame-major `[T,C,H,W]`, and delegates to image postprocess.

## 11. Graph rewrite / lowering opportunities

### Rewrite: Helios Conv3d patchify/unpatchify

Source pattern: NCTHW Conv3d patch embedding, flatten/transpose to tokens,
then linear projection and unpatchify reshape/permute.

Replacement: explicit `conv3d_patchify_tokenize` and
`linear_unpatchify_3d` primitives, or Conv3d + canonical view ops.

Preconditions: source layout NCTHW, patch size `(1,2,2)`, channel 16, H/W
divisible by 2 after VAE scale, no hidden layout translation between patchify
and unpatchify.

Weight transform: Conv3d weights stay OI(DHW) for NCTHW; a GEMM lowering would
flatten `[C,pt,ph,pw]` in the same order used by PyTorch Conv3d.

Failure cases: pyramid stages with changed spatial size, future non-1 temporal
patch size, NHWC/NDHWC pass that changes patch flatten order.

Parity test: random `[B,16,9,48,80]` through patchify/proj/unpatchify against
Diffusers with identity-style test weights.

### Rewrite: multi-term history patch streams

Source pattern: short/mid/long Conv3d patch paths plus replicate padding and
RoPE downsampling before token concat.

Replacement: explicit history-token builder with three independently guarded
Conv3d patch streams and RoPE streams.

Preconditions: `has_multi_term_memory_patch=true`, official history sizes,
known `keep_first_frame` behavior, source NCTHW axes.

Failure cases: `has_multi_term_memory_patch=false`, missing history inputs,
changed history sizes that make H1/W1 assumptions unsafe.

Parity test: synthetic histories with odd temporal/spatial lengths to verify
replicate pad and `avg_pool3d` RoPE downsampling.

### Rewrite: Helios attention region

Source pattern: adaptive norm -> QKV -> RMSNorm -> RoPE for self-attn ->
dispatch attention -> output projection -> gated residual; cross-attn operates
on main tokens when `guidance_cross_attn=true`.

Replacement: provider-backed self/cross attention nodes with pre/post norm and
gating fused when supported.

Preconditions: no attention mask, head dim 128, dtype fp16/bf16 or provider
supported dtype, text projection already materialized, history/main split sizes
artifact-visible.

Failure cases: masked backend, `is_amplify_history=true` without key-scale
support, provider cannot handle long sequences or rectangular cross-attn.

Parity test: one block with and without history tokens, `guidance_cross_attn`
true, fp32 then bf16.

### Rewrite: Base Helios scheduler slice

Source pattern: dynamic shift plus UniPC flow-prediction multistep solver.

Replacement: host-visible schedule tables plus compiled pointwise conversion
and UniPC predictor/corrector arithmetic.

Preconditions: `scheduler_type=unipc`, `solver_order=2`, `prediction_type=
flow_prediction`, `use_flow_sigmas=true`, no thresholding, no custom solver_p.

Failure cases: Euler diagnostic path, DMD path, staged scheduler path, changed
solver order or thresholding enabled.

Parity test: set_timesteps table parity and two-step UniPC parity for Base.

### Rewrite: guarded NDHWC codec islands

Source pattern: Wan VAE NCTHW causal Conv3d and residual blocks.

Replacement: optional NDHWC/channel-last VAE subgraph with transformed Conv3d
weights and explicit channel-axis rewrites.

Preconditions: VAE encode/decode island fully controlled, no tiling/slicing
active, cache state represented, all Group/RMS norms rewritten from channel=1
to channel=-1.

Failure cases: mixed source-layout tensors crossing into transformer, VAE
tiling/slicing, causal cache not modeled, postprocess expecting NCTHW.

Parity test: VAE decode random `[1,16,9,48,80]` under NCTHW vs NDHWC lowering.

## 12. Kernel fusion candidates

Highest priority:

- Conv3d patchify/unpatchify for main and history latents. Helios pays this
  cost every transformer step and every chunk.
- Self-attention provider with QK RMSNorm + RoPE staging for long video token
  sequences.
- Cross-attention provider for main-token to UMT5 text attention.
- AdaLN scale/shift/gate plus residual epilogues around attention and FFN.
- GELU approximate FeedForward fusion for 5120 -> 13824 -> 5120.
- Base scheduler UniPC flow-prediction arithmetic once denoiser parity is
  stable.

Medium priority:

- RoPE generation/cache kernels for temporal/history indices.
- CFG and CFG-Zero* reduction/pointwise kernels.
- Pyramid bilinear/nearest resize and block-correlated noise path.
- Wan VAE decode Conv3d/RMSNorm/SiLU/residual kernels.
- I2V/V2V VAE encode and random sigma noising.

Lower priority:

- DMD scheduler x0/re-noise path.
- LoRA hotswap/fuse/unfuse artifact mutation.
- UMT5 text encoder compilation; prompt embeddings can be external first.
- VAE tiling/slicing/cache-policy specialization.

NHWC/NDHWC layout notes:

- Transformer core is token-major `[B,S,C]`; NHWC does not apply there.
- Source VAE and latent staging are NCTHW. NDHWC can be explored only within
  fully controlled VAE/Conv3d islands and patchify helpers.
- Axis rewrites needed: Conv3d weights, normalization channel axis, temporal
  split/cat axes, `avg_pool3d`/padding order, postprocess permutes, and
  unpatchify flatten order.
- Protect scheduler broadcasting, text/token sequence ops, and history token
  ordering with no-layout-translation guards.

## 13. Runtime staging plan

Stage 1: Parse Base, Mid, Distilled, and tiny configs. Fill source defaults for
Wan VAE scale factors and scheduler omitted fields. Load tiny public weights
first, then production transformer metadata.

Stage 2: Admit a `helios_denoiser_step` artifact with external
`prompt_embeds`, `negative_prompt_embeds` optional, one latent chunk, history
latents, index tensors, and a timestep. Return `noise_pred`.

Stage 3: Implement Conv3d patchify/unpatchify and one
`HeliosTransformerBlock` parity with random tensors.

Stage 4: Compile full tiny `HeliosTransformer3DModel`, then production-shape
Base one-step denoiser parity at 384x640.

Stage 5: Add Base `HeliosScheduler` set_timesteps and UniPC one/two-step parity
with scheduler state kept host-visible.

Stage 6: Add chunk-loop orchestration in Python around compiled denoiser and
scheduler arithmetic. Decode each chunk through an external Wan VAE first.

Stage 7: Add Wan VAE decode artifact, then encode for I2V/V2V.

Stage 8: Add attention/norm/FFN fusions and guarded flash-style providers.

Stage 9: Separate Mid pyramid/CFG-Zero* and Distilled DMD implementation
slices after Base parity.

Initial stub recommendation: do not compile UMT5 or Wan VAE in the first
denoiser step. Accept prompt embeddings and latent histories as tensors.

## 14. Parity and validation plan

- Config parse/default-fill tests for Base, Mid, Distilled, and tiny.
- Prompt cleaning/tokenization/embedding shape parity with a short prompt.
- Latent standardization and unstandardization parity from VAE mean/std vectors.
- Conv3d patchify/unpatchify shape/order tests for `[B,16,9,48,80]`.
- RoPE frequency parity for main, short, mid, and long history indices.
- Single `HeliosAttnProcessor` self-attn and cross-attn parity.
- Single `HeliosTransformerBlock` parity with history tokens and
  `guidance_cross_attn=true`.
- Full tiny transformer parity.
- Base production-shape one-step denoiser parity with fixed embeddings and
  latents.
- `HeliosScheduler.set_timesteps` parity for Base dynamic exponential shift.
- UniPC step parity for first two steps, including lower-order warmup.
- CFG separate-call parity.
- Wan VAE decode parity on random latent chunks; encode parity for I2V/V2V.
- Mid pyramid stage resize/block-noise/CFG-Zero* parity.
- Distilled DMD x0 conversion and re-noise parity.
- Short deterministic loop smoke: one chunk, two denoising steps, output latent.
- Suggested tolerances: scheduler fp32 `rtol=1e-5, atol=1e-6`; transformer fp32
  `rtol=1e-4, atol=1e-5`; bf16/fp16 initially `rtol=2e-2, atol=2e-2`.

## 15. Performance probes

- Denoiser step by resolution and chunk length: 384x640/9 latent frames first,
  then longer/higher-resolution variants.
- Attention backend comparison for self-attn sequence length around 10k tokens
  and cross-attn against 512 text tokens.
- Time split per block: QKV/RMSNorm/RoPE, attention, output projection, FFN,
  adaptive residual.
- History-path overhead: no history, short only, short+mid+long.
- Scheduler overhead: UniPC host/state math vs fused pointwise pieces.
- CFG overhead: one denoiser call vs two calls.
- Pyramid overhead: downsample/upsample/block-noise cost per stage.
- Wan VAE encode/decode throughput by frames/chunk and resolution.
- VRAM/workspace for transformer activations at bf16 and fp16.
- LoRA fused/unfused latency once adapter support is admitted.

## 16. Scope boundary and separate candidates

Separate candidate reports, not ignored:

- `helios_pyramid_mid`: Mid three-stage pyramid with CFG-Zero* and block noise.
- `helios_distilled_dmd`: Distilled DMD scheduler and transformer_ode side
  weights.
- `helios_i2v_v2v`: image/video encode, noising, seeding, and first-frame
  contracts.
- `helios_lora_adapters`: transformer-only LoRA load/fuse/unfuse/hotswap and
  Wan LoRA key conversion.
- `helios_wan_vae_codec`: AutoencoderKLWan encode/decode, causal cache,
  tiling/slicing, temporal compression.
- `helios_modular_pipeline`: modular block graph and guider configs as a
  future orchestration surface.
- Advanced scheduler surfaces: Euler diagnostic path, thresholding, non-default
  solver order, `solver_p`, and amplify-first-chunk DMD schedule.

Unsupported/not found in non-deprecated Helios folder:

- Textual inversion/tokenizer embedding mutation.
- IP-Adapter.
- ControlNet.
- T2I-Adapter.
- GLIGEN.
- Inpaint.
- Depth2img.
- Upscaling.
- Separate img2img class. Image-to-video is supported inside the base call and
  should be treated as `helios_i2v_v2v`.

Genuinely out of scope for this audit:

- Multi-GPU/context parallel paths.
- Callback mutation and interactive interrupt.
- XLA, NPU, MPS, Flax, and ONNX branches.
- Safety checker and NSFW filtering.
- Training, losses, dropout, and gradient checkpointing.

## 17. Final implementation checklist

- [ ] Parse Helios Base/Mid/Distilled model indexes and component configs.
- [ ] Fill source defaults for `AutoencoderKLWan` scale factors and scheduler fields.
- [ ] Load tiny/public Helios transformer and VAE weights for smoke parity.
- [ ] Accept external UMT5 prompt embeddings and negative embeddings.
- [ ] Implement Helios Conv3d patchify/unpatchify.
- [ ] Implement short/mid/long history token builders and RoPE builders.
- [ ] Implement timestep/text embedding and AdaLN modulation.
- [ ] Implement `HeliosAttnProcessor` self-attn and cross-attn fallback parity.
- [ ] Implement `HeliosTransformerBlock` parity.
- [ ] Compile full tiny `HeliosTransformer3DModel`.
- [ ] Add production-shape Base denoiser-step parity.
- [ ] Implement Base `HeliosScheduler` dynamic-shift UniPC slice.
- [ ] Add CFG separate-call arithmetic parity.
- [ ] Add Wan VAE decode boundary and latent mean/std conversion.
- [ ] Add I2V/V2V VAE encode/noise/seeding tests.
- [ ] Add guarded attention provider lowering.
- [ ] Add pyramid Mid report/work for resize, block noise, CFG-Zero*.
- [ ] Add Distilled DMD report/work for x0/re-noise scheduler.
- [ ] Benchmark denoiser step, attention backend, scheduler overhead, and VAE decode.
