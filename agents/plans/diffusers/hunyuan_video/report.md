# Diffusers Hunyuan Video Operator and Integration Report

Candidate slug: `hunyuan_video`

## 1. Source basis

```text
Diffusers commit/version:
  Local checkout X:/H/diffusers at b3a515080752a3ba7ca92161e25530c7f280f629.

Model id(s):
  hunyuanvideo-community/HunyuanVideo
  hunyuanvideo-community/HunyuanVideo-I2V
  hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v
  hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_i2v
  Skywork/SkyReels-V1-Hunyuan-T2V and Skywork/SkyReels-V1-Hunyuan-I2V,
    inventoried as related variants from local model_index files/source only.

Config sources:
  H:/configs/hunyuanvideo-community/HunyuanVideo/model_index.json
  H:/configs/hunyuanvideo-community/HunyuanVideo-I2V/model_index.json
  H:/configs/hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v/model_index.json
  H:/configs/hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_i2v/model_index.json
  Official raw Hugging Face component configs for transformer, VAE, scheduler,
  text encoders, guider, and image encoder were inspected in-memory because the
  local cache held only model_index.json files for Hunyuan Video repos. They
  were not saved because this task's owned write path is limited to this report.

Pipeline files inspected:
  X:/H/diffusers/src/diffusers/pipelines/hunyuan_video/pipeline_hunyuan_video.py
  X:/H/diffusers/src/diffusers/pipelines/hunyuan_video/pipeline_hunyuan_video_image2video.py
  X:/H/diffusers/src/diffusers/pipelines/hunyuan_video/pipeline_hunyuan_skyreels_image2video.py
  X:/H/diffusers/src/diffusers/pipelines/hunyuan_video/pipeline_hunyuan_video_framepack.py
  X:/H/diffusers/src/diffusers/pipelines/hunyuan_video/pipeline_output.py
  X:/H/diffusers/src/diffusers/pipelines/hunyuan_video1_5/pipeline_hunyuan_video1_5.py
  X:/H/diffusers/src/diffusers/pipelines/hunyuan_video1_5/pipeline_hunyuan_video1_5_image2video.py
  X:/H/diffusers/src/diffusers/pipelines/hunyuan_video1_5/image_processor.py

Model files inspected:
  X:/H/diffusers/src/diffusers/models/transformers/transformer_hunyuan_video.py
  X:/H/diffusers/src/diffusers/models/transformers/transformer_hunyuan_video_framepack.py
  X:/H/diffusers/src/diffusers/models/transformers/transformer_hunyuan_video15.py
  X:/H/diffusers/src/diffusers/models/autoencoders/autoencoder_kl_hunyuan_video.py
  X:/H/diffusers/src/diffusers/models/autoencoders/autoencoder_kl_hunyuanvideo15.py
  X:/H/diffusers/src/diffusers/models/attention_dispatch.py
  X:/H/diffusers/src/diffusers/models/attention_processor.py
  X:/H/diffusers/src/diffusers/models/embeddings.py
  X:/H/diffusers/src/diffusers/models/normalization.py

Scheduler/processors/helpers inspected:
  X:/H/diffusers/src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py
  X:/H/diffusers/src/diffusers/video_processor.py
  X:/H/diffusers/src/diffusers/image_processor.py
  X:/H/diffusers/src/diffusers/loaders/lora_pipeline.py
  X:/H/diffusers/src/diffusers/guiders/classifier_free_guidance.py indirectly
    through HunyuanVideo 1.5 model_index/guider config.

External component configs inspected:
  Base HunyuanVideo: LlamaModel, LlamaTokenizerFast, CLIPTextModel,
  CLIPTokenizer.
  HunyuanVideo-I2V: LlavaForConditionalGeneration, CLIPTextModel,
  CLIPImageProcessor through pipeline source/model_index.
  HunyuanVideo 1.5: Qwen2_5_VLTextModel, Qwen2TokenizerFast, T5EncoderModel,
  ByT5Tokenizer, ClassifierFreeGuidance, and SiglipVisionModel for I2V.

Any missing files or assumptions:
  No official JSON config blocker was hit. Component configs were not retained
  locally due to the owned-write-path restriction. This report targets the
  original text-to-video `HunyuanVideoPipeline` as the first implementation
  slice; HunyuanVideo-I2V, SkyReels, Framepack, HunyuanVideo 1.5, LoRA, and VAE
  tiling/framewise policies are separate candidates. XLA/NPU/MPS/Flax/ONNX,
  safety/NSFW, training/loss/dropout/gradient checkpointing,
  multi-GPU/context parallel, callbacks, and interrupt behavior are out of
  scope.
```

## 2. Pipeline and component graph

Base Hunyuan Video is a latent text-to-video diffusion pipeline with two text
encoders, a 3D joint text/video transformer, FlowMatch Euler scheduling, and a
Hunyuan-specific causal 3D VAE.

```text
prompt
  -> Llama tokenizer/template/crop -> Llama hidden states [B,256,4096] + mask
  -> CLIP tokenizer/text encoder -> pooled prompt embeds [B,768]
  -> latent video noise [B,16,T_lat,H/8,W/8] NCTHW
  -> denoising loop:
       HunyuanVideoTransformer3DModel(latents, timestep, Llama tokens,
       Llama mask, CLIP pooled projection, embedded guidance)
       optional true CFG second negative denoiser call
       FlowMatchEulerDiscreteScheduler.step
  -> latents / VAE scaling_factor
  -> AutoencoderKLHunyuanVideo decode
  -> VideoProcessor postprocess
```

Required first-slice components:

| Component | Class/file | First-slice status |
| --- | --- | --- |
| Pipeline | `HunyuanVideoPipeline`, `pipeline_hunyuan_video.py` | First runtime contract. |
| Denoiser | `HunyuanVideoTransformer3DModel`, `transformer_hunyuan_video.py` | Required. |
| VAE | `AutoencoderKLHunyuanVideo`, `autoencoder_kl_hunyuan_video.py` | Decode required; encode for I2V later. |
| Scheduler | `FlowMatchEulerDiscreteScheduler` | Required with Hunyuan custom sigmas and static shift. |
| Text encoders | `LlamaModel`, `CLIPTextModel` | Accept external cached embeddings first. |
| Guidance | embedded guidance tensor plus optional true CFG | Embedded guidance is required by base config. |

Independently cacheable stages are Llama prompt embeddings and masks, CLIP pooled
projections, negative prompt embeddings, scheduler timesteps/sigmas, random
initial latents, and VAE latents/video decode.

Separate candidate reports:

| Candidate | Primary classes/files | Runtime delta |
| --- | --- | --- |
| `hunyuan_video_i2v` | `HunyuanVideoImageToVideoPipeline`, same transformer/VAE | Adds image preprocessing, VAE encode of reference image, either latent concat plus mask or token-replace first-frame conditioning, and Llava image/text prompt embedding. |
| `hunyuan_skyreels_i2v` | `HunyuanSkyreelsImageToVideoPipeline` | Similar Hunyuan I2V source with SkyReels-specific latent padding/condition handling. |
| `hunyuan_video_framepack` | `HunyuanVideoFramepackPipeline`, `HunyuanVideoFramepackTransformer3DModel` | Sliding/windowed generation with SigLIP image embeddings, history latents, index packing, and repeated scheduler windows. |
| `hunyuan_video_1_5` | `HunyuanVideo15Pipeline`, `HunyuanVideo15Transformer3DModel`, `AutoencoderKLHunyuanVideo15` | Different text encoders, 32-channel VAE, 65-channel transformer input, 54-layer transformer, classifier-free guider object. |
| `hunyuan_video_1_5_i2v` | `HunyuanVideo15ImageToVideoPipeline`, SigLIP image encoder | Adds SigLIP image tokens and first-frame latent/mask conditioning. |
| `hunyuan_video_lora_adapters` | `HunyuanVideoLoraLoaderMixin`, transformer `PeftAdapterMixin` | Runtime/load-time adapter mutation for transformer weights. |
| `hunyuan_video_vae_tiling_framewise` | `AutoencoderKLHunyuanVideo` toggles | Spatial tiling, temporal tiled encode/decode, batch slicing memory policies. |

No base Hunyuan Video ControlNet, T2I-Adapter, GLIGEN, depth2img, inpaint, or
upscaling pipeline class was found in the Hunyuan Video folders. Img2img-style
behavior is represented by I2V/framepack/SkyReels variants rather than a generic
img2img pipeline.

## 3. Important config dimensions

Representative checkpoint sweep:

| Repo | Pipeline | Transformer | VAE | Scheduler | Special |
| --- | --- | --- | --- | --- | --- |
| `HunyuanVideo` | `HunyuanVideoPipeline` | 16->16 channels, 20 dual + 40 single layers, 24 heads, head dim 128, inner 3072, patch `[1,2,2]` | 16 latent channels, spatial 8, temporal 4, scale 0.476986 | FlowMatch Euler, static shift 7.0 | Llama 4096 tokens + CLIP pooled 768, embedded guidance. |
| `HunyuanVideo-I2V` | `HunyuanVideoImageToVideoPipeline` | Same dimensions, `image_condition_type="token_replace"` in official config | Same VAE | FlowMatch Euler, static shift 17.0 | Llava image/text prompt encoder and first-frame token replacement. |
| `HunyuanVideo-1.5 480p T2V` | `HunyuanVideo15Pipeline` | 65->32 channels, 54 layers, 16 heads, head dim 128, inner 2048, patch `[1,1,1]` | 32 latent channels, spatial 16, temporal 4, scale 1.03682 | FlowMatch Euler, static shift 5.0 | Qwen2.5-VL text + ByT5, zero image embeds for T2V, CFG guider. |
| `HunyuanVideo-1.5 720p I2V` | `HunyuanVideo15ImageToVideoPipeline` | Same 1.5 width/depth, `task_type="i2v"`, target size 960 | Same 1.5 VAE | FlowMatch Euler, static shift 7.0 | SigLIP image tokens plus latent condition/mask. |

Base transformer dimensions:

| Field | Value | Source |
| --- | ---: | --- |
| `in_channels`, `out_channels` | 16 / 16 | transformer config |
| `num_attention_heads`, `attention_head_dim` | 24 / 128 | transformer config |
| inner dim | 3072 | inferred from heads * head dim |
| dual-stream layers | 20 | transformer config |
| single-stream layers | 40 | transformer config |
| token-refiner layers | 2 | transformer config |
| MLP ratio | 4.0 | transformer config |
| patch size | temporal 1, spatial 2x2 | transformer config |
| text embed dim | 4096 | transformer config / Llama config |
| pooled projection dim | 768 | transformer config / CLIP config |
| RoPE axes dim | `[16,56,56]`, theta 256 | transformer config |
| QK norm | RMS norm | transformer config |
| guidance embeds | true | transformer config |

Base VAE dimensions:

| Field | Value |
| --- | ---: |
| input/output channels | 3 / 3 |
| latent channels | 16 |
| block channels | `[128,256,512,512]` |
| layers per block | 2 |
| spatial compression | 8 |
| temporal compression | 4 |
| scaling factor | 0.476986 |
| mid-block attention | true |
| source default tiling/framewise | slicing false, tiling false, framewise encode/decode true |

Base text encoders:

| Component | Config facts |
| --- | --- |
| `LlamaModel` | hidden 4096, 32 layers, 32 attention heads, 8 KV heads, intermediate 14336, max positions 8192, fp16 metadata. |
| `CLIPTextModel` | hidden/projection 768, 12 layers, 12 heads, intermediate 3072, max positions 77, fp16 metadata. |

Recommended first Dinoml scheduler slice:

- `FlowMatchEulerDiscreteScheduler` with custom sigmas from the pipeline
  `linspace(1.0, 0.0, steps + 1)[:-1]`, static shift 7.0, terminal zero sigma,
  no stochastic sampling, and scalar timestep path.
- Defer I2V shift 17.0, HunyuanVideo 1.5 shifts 5.0/7.0, dynamic `mu`,
  stochastic sampling, and per-token timestep branches until base parity lands.

## 3a. Family variation traps

- Base Hunyuan Video source latent layout is NCTHW. Treat NDHWC as a guarded
  optimization only; semantic translation must keep channel dim 1 and temporal
  dim 2.
- The transformer patchifies internally with Conv3d kernel/stride `[1,2,2]`.
  There is no pipeline-level latent packing; tokenization is a model-internal
  patch embed.
- Latent temporal length is `(num_frames - 1) // 4 + 1`; default 129 frames
  gives 33 latent frames. Decode returns `(T_lat - 1) * 4 + 1`.
- Height/width are only checked as divisible by 16 in the pipeline, combining
  VAE spatial scale 8 and transformer spatial patch 2.
- Base true CFG is optional and separate from embedded guidance. Embedded
  guidance is a model-conditioning tensor equal to `guidance_scale * 1000`.
- The attention mask is real: the transformer constructs `[B,1,1,latent_seq +
  effective_text_seq]` from the Llama attention mask after token refinement.
- HunyuanVideo-I2V official config uses `image_condition_type="token_replace"`,
  not channel concat, even though source supports both. Token-replace denoises
  frames after the first latent frame and prepends the encoded image latent.
- HunyuanVideo 1.5 is not a minor config variant: it changes text encoders,
  latent channels, transformer input channels, spatial compression, attention
  width/depth, conditioning token composition, and guidance object semantics.
- Base VAE framewise decode is enabled by source default and triggers for
  representative latent lengths because `tile_sample_min_num_frames=16` and
  latent tile threshold is 4. First parity can disable or explicitly model this,
  but must state the memory-policy deviation.
- VAE mid-block attention flattens `[B,C,T,H,W]` to frame/spatial tokens with a
  causal temporal mask. Do not model the VAE as pure Conv3d.

## 4. Runtime tensor contract

For a base 720x1280, 129-frame run:

| Boundary | Tensor | Shape/layout | Notes |
| --- | --- | --- | --- |
| Llama prompt embeds | `prompt_embeds` | `[B,256,4096]` | Template adds 95-token prefix then crops; source default max sequence is 256 after crop. |
| Llama attention mask | `prompt_attention_mask` | `[B,256]` | Cast to transformer dtype by pipeline, then used as lengths/mask. |
| CLIP pooled prompt | `pooled_prompt_embeds` | `[B,768]` | CLIP `pooler_output`, duplicated per video. |
| negative embeds | optional | same as positive | Required only for true CFG. |
| noisy latents | `latents` | `[B,16,33,90,160]` NCTHW | Generated fp32 in base pipeline, then cast to transformer dtype per step. |
| guidance | `guidance` | `[B]` | `guidance_scale * 1000`, transformer dtype. |
| patch tokens | after Conv3d | `[B,33*45*80,3072]` | 118,800 latent tokens at 720p/129 frames. |
| text tokens | after token refiner | `[B,256,3072]` | Llama 4096 -> 3072 projection plus two refiner blocks. |
| attention mask | joint mask | `[B,1,1,119056]` | Latent tokens all valid; text padding masked by effective length. |
| denoiser output | `noise_pred` | `[B,16,33,90,160]` NCTHW | Unpatchified inside transformer. |
| scheduler state | sigmas/timesteps | `[steps+1]` sigmas, `[steps]` timesteps | FlowMatch step index and terminal zero sigma. |
| VAE decode input | scaled latents | `[B,16,33,90,160]` | Pipeline divides by 0.476986 before decode. |
| decoded video | output tensor | `[B,3,129,720,1280]` NCTHW | Postprocessed to PIL/NumPy/latent based on `output_type`. |

I2V token-replace boundary:

| Boundary | Tensor | Shape/layout | Notes |
| --- | --- | --- | --- |
| input image | preprocessed | `[B,3,H,W]` | `VideoProcessor.preprocess`, then unsqueezed to `[B,3,1,H,W]`. |
| image latent | VAE mode | `[B,16,1,H/8,W/8]` | Multiplied by VAE scale. |
| latent model input | token-replace | `[B,16,T_lat,H/8,W/8]` | `cat([image_latents, latents[:,:,1:]], dim=2)`. |
| scheduler sample | update target | `[B,16,T_lat-1,H/8,W/8]` | Scheduler steps `noise_pred[:,:,1:]`, then image latent is prepended. |

HunyuanVideo 1.5 T2V boundary:

- Base latents are `[B,32,T_lat,H/16,W/16]`.
- Model input is always `[B,65,T_lat,H/16,W/16]` from `cat([latents,
  cond_latents_concat, mask_concat], dim=1)`, where T2V condition tensors are
  zeros. I2V fills first-frame condition and a first-frame mask.
- Qwen embeddings are `[B,L,3584]`; ByT5 embeddings are `[B,L2,1472]`; image
  tokens are `[B,vision_tokens,1152]`, zeroed for T2V.

CPU/data-pipeline work includes prompt templating/tokenization, Llama/CLIP or
Qwen/ByT5/SigLIP execution when embeddings are not supplied, image resize/crop,
and output conversion. GPU/runtime work includes latent generation, denoiser
forward, optional true CFG/guider arithmetic, scheduler update, VAE encode for
I2V, and VAE decode.

## 5. Operator coverage checklist

### Tensor/layout ops

- NCTHW video latent tensors; view, reshape, flatten, transpose, permute.
- Conv3d patchify and unpatchify:
  `[B,C,T,H,W] -> [B,T*(H/2)*(W/2),inner] -> [B,C,T,H,W]`.
- Joint image/text sequence concat/split inside attention processors.
- Prompt embedding duplication and attention-mask length derivation.
- CFG separate-call arithmetic: `uncond + scale * (cond - uncond)`.
- I2V channel/time concat, first-frame slicing, and mask construction.
- VAE temporal/spatial tiled crop/concat/blend if memory policy is enabled.

### Convolution/downsample/upsample ops

- Transformer patch embed: `Conv3d(16 -> 3072, kernel=[1,2,2], stride=[1,2,2])`.
- VAE `HunyuanVideoCausalConv3d` with replicate causal temporal padding.
- VAE ResNet 3D blocks: GroupNorm -> SiLU -> causal Conv3d twice, optional
  shortcut Conv3d.
- VAE downsample causal Conv3d with stride patterns selected by compression.
- VAE upsample: first frame spatial-only nearest upsample, remaining frames
  nearest 3D upsample, then causal Conv3d.
- VAE 1x1 quant and post-quant Conv3d.

### GEMM/linear ops

- Llama and CLIP encoders if admitted later.
- Token refiner projection: Linear(4096 -> 3072).
- CLIP pooled projection through `PixArtAlphaTextProjection(768 -> 3072)`.
- Timestep and embedded-guidance MLPs.
- Attention Q/K/V and added text Q/K/V projections.
- FFN GELU-approximate MLPs and single-block parallel MLP path.
- Output projection: Linear(3072 -> `1*2*2*16=64`).

### Attention primitives

- Token-refiner self-attention over Llama tokens with a square padding mask.
- Dual-stream joint latent/text attention with added text Q/K/V projections.
- Single-stream joint attention over concatenated latent+text stream.
- VAE mid-block causal attention over flattened frame/spatial tokens.
- QK RMSNorm and 3D RoPE on latent token spans.

### Normalization and adaptive conditioning

- LayerNorm, FP32LayerNorm in helper classes, RMSNorm for Q/K projections.
- AdaLayerNormZero, AdaLayerNormZeroSingle, AdaLayerNormContinuous.
- HunyuanVideoAdaNorm gates in token refiner.
- VAE GroupNorm over channel dim.

### Scheduler and guidance arithmetic

- FlowMatch Euler static-shift table generation and `sample + dt * model_output`.
- Embedded guidance input tensor, not scheduler math.
- Optional true CFG as two transformer calls.
- HunyuanVideo 1.5 `ClassifierFreeGuidance` guider path as separate candidate.

### Video-specific ops

- Temporal compression/decompression by 4.
- Causal Conv3d temporal padding and temporal tile blending.
- VAE causal attention mask creation by frame index.
- Video postprocess from NCTHW to requested list/NumPy/PT output.

## 6. Denoiser/model breakdown

`HunyuanVideoTransformer3DModel.forward`:

```text
hidden_states [B,16,T,H,W]
-> 3D RoPE from latent frame/height/width and patch sizes
-> timestep + CLIP pooled + embedded guidance conditioning [B,3072]
-> Conv3d patch embed [1,2,2] -> latent tokens [B,T*H/2*W/2,3072]
-> Llama token refiner:
     masked mean pool -> timestep/text embed
     Linear(4096->3072)
     2 x masked self-attn + gated linear-SiLU FFN
-> build joint attention mask from latent tokens + valid text tokens
-> 20 dual-stream transformer blocks
-> 40 single-stream transformer blocks
-> AdaLayerNormContinuous + Linear(3072->64)
-> unpatchify to [B,16,T,H,W]
```

Dual-stream block:

```text
latent AdaLayerNormZero and text AdaLayerNormZero
latent QKV + text added QKV
QK RMSNorm
3D RoPE on latent query/key only
masked noncausal joint attention over latent+text
gated residual add to latent and text streams
LayerNorm + adaptive scale/shift
GELU-approximate FeedForward on both streams
gated residual add
```

Single-stream block:

```text
concat latent/text tokens
AdaLayerNormZeroSingle -> normalized stream and gate
split normalized stream into latent/text for joint attention
parallel GELU MLP from normalized concat stream
concat(attention output, MLP output) -> Linear -> gated residual
split latent/text streams
```

HunyuanVideo-I2V token-replace blocks reuse the same structure but apply
different adaptive modulation and gates to first-frame tokens versus the
remaining latent tokens. That is a separate first-frame-conditioning contract.

## 7. Attention requirements

Primary implementation is `HunyuanVideoAttnProcessor2_0` in
`transformer_hunyuan_video.py`, using `dispatch_attention_fn` from
`attention_dispatch.py`.

- Query/key/value tensors are shaped `[B,seq,heads,head_dim]` before dispatch.
- Base heads/head dim are 24 x 128.
- Token-refiner attention uses standard `Attention` with self-attention and a
  square bool mask derived from prompt masks.
- Dual-stream attention uses `added_kv_proj_dim=hidden_size`, so text receives
  added Q/K/V projections and output projection `to_add_out`.
- Single-stream attention uses `pre_only=True` and concatenates latent/text
  streams before and after the attention/MLP path.
- RoPE is applied to latent query/key tokens only. Text tokens are concatenated
  without RoPE.
- Attention masks are noncausal padding masks for transformer denoising; VAE
  mid-block attention uses a causal temporal mask.
- Source-supported fused QKV mutation may exist through generic attention
  mixins, but default Hunyuan Video parity is the unfused processor path.

Flash-style constraints:

- Base denoiser attention is a plausible Dinoml provider candidate only if the
  provider supports head_dim 128, long noncausal sequences, bool/additive masks,
  and joint latent/text spans.
- QK RMSNorm and RoPE are explicit pre-attention operations unless fused under
  exact guards.
- At 720x1280x129, latent sequence length is 118,800 before text tokens; first
  provider admission needs sequence/workspace guards and a fallback.
- I2V token-replace and Framepack add token-span semantics that should not be
  folded into the base attention lowering.
- Eager/native `dispatch_attention_fn` behavior defines parity.

## 8. Scheduler and denoising-loop contract

Base official config uses `FlowMatchEulerDiscreteScheduler`:

```text
num_train_timesteps = 1000
shift = 7.0
use_dynamic_shifting = false
base_shift = 0.5
max_shift = 1.15
base_image_seq_len = 256
max_image_seq_len = 4096
invert_sigmas = false
shift_terminal = null
use_karras/exponential/beta_sigmas = false
```

Base loop:

```text
sigmas = linspace(1.0, 0.0, steps + 1)[:-1]
timesteps = scheduler.set_timesteps(sigmas=sigmas)
guidance = guidance_scale * 1000
for t in timesteps:
  timestep = t.expand(B)
  noise_pred = transformer(latents, timestep, prompt_embeds, mask, pooled, guidance)
  if true CFG:
    neg = transformer(latents, timestep, negative_embeds, negative_mask, negative_pooled, guidance)
    noise_pred = neg + true_cfg_scale * (noise_pred - neg)
  latents = scheduler.step(noise_pred, t, latents)
```

FlowMatch step for the first slice is:

```text
sigma = sigmas[step_index]
sigma_next = sigmas[step_index + 1]
prev = sample + (sigma_next - sigma) * model_output
```

Keep scheduler iteration, step index, and sigma/timestep tables host-visible at
first. Compile CFG arithmetic and the pointwise scheduler update only after
fixed-output one-step parity is proven. I2V and 1.5 use the same scheduler class
but different shift values and conditioning-side state, so they should not be
claimed by the first base slice.

## 9. Position, timestep, and custom math

- 3D RoPE builds three axis grids from `[T//patch_t, H//patch, W//patch]`,
  applies `get_1d_rotary_pos_embed` per axis, and concatenates cos/sin pieces
  according to `rope_axes_dim=[16,56,56]`.
- Source comments note RoPE grid generation differs from original behavior by
  creating grids on the target device instead of CPU; use Diffusers source as
  parity for this audit.
- Timestep embeddings use `Timesteps(256, flip_sin_to_cos=True,
  downscale_freq_shift=0)` plus `TimestepEmbedding`.
- Embedded guidance uses the same sinusoidal projection path and is added to
  timestep and pooled text projections when `guidance_embeds=true`.
- Token refiner pools text embeddings with the prompt mask, then uses
  `CombinedTimestepTextProjEmbeddings` and gated residual attention/MLP.
- HunyuanVideo 1.5 optionally supports `use_meanflow` with `timestep_r`; sampled
  official configs set it false.

Precompute candidates: prompt embeddings/masks, CLIP pooled projections, RoPE
tables for fixed latent shape, scheduler sigmas/timesteps, and embedded
guidance projections for fixed guidance scale.

## 10. Preprocessing and input packing

Base text preprocessing:

```text
template(prompt)
LlamaTokenizer(max_length=256 + crop_start, padding=max_length)
LlamaModel(..., output_hidden_states=True).hidden_states[-3]
drop first crop_start tokens, default crop_start=95
duplicate for num_videos_per_prompt
CLIPTokenizer(max_length=77) -> CLIPTextModel.pooler_output [B,768]
```

Base latent preprocessing:

- Generate `[B,16,(frames-1)//4+1,H/8,W/8]` random latents in fp32.
- Cast latents to transformer dtype in the denoising loop.
- Decode uses `latents / vae.config.scaling_factor`.

I2V preprocessing:

- Preprocess input image to NCHW, unsqueeze to `[B,3,1,H,W]`, VAE encode, take
  posterior mode through `retrieve_latents(..., "argmax")`, multiply by VAE
  scale.
- `latent_concat` source branch builds `[latents, image_latents, mask]` along
  channels where model channels imply `(in_channels - 1) // 2` noisy channels.
- Official I2V `token_replace` branch uses the encoded image latent as the
  first latent frame and denoises only subsequent frames.
- Llava prompt encoding injects image embeddings into the Llama prompt sequence
  and then crops/reorders image/text token spans.

NHWC/NDHWC guarded notes:

- Preserve source NCTHW at all pipeline/VAE boundaries initially.
- NDHWC optimization is plausible inside VAE Conv3d islands and perhaps the
  Conv3d patch embed, but requires Conv3d weight transforms, GroupNorm channel
  axis rewrites, temporal padding/caches on dim 2, posterior splits on channel
  dim, and VAE blend axes.
- Transformer token core is layout-neutral after patchify; patchify/unpatchify,
  I2V concat/slice, VAE encode/decode, and scheduler sample layout should be
  guarded as no-layout-translation regions until rewritten explicitly.

## 11. Graph rewrite / lowering opportunities

### Rewrite: Conv3d patchify/unpatchify

Source pattern:

```text
Conv3d(kernel=stride=[1,2,2]) -> flatten(2).transpose(1,2)
Linear(3072 -> 64) -> reshape(B,T,H/2,W/2,C,1,2,2)
-> permute -> flatten back to NCTHW
```

Replacement: explicit video patch pack/unpack ops around a GEMM/Conv3d patch
projection.

Preconditions: source NCTHW layout, patch size `[1,2,2]`, H/W divisible by 2,
output projection has patch volume `1*2*2`, no I2V token-replace first-frame
special gating folded into base. Weight transform is Conv3d OIHW-like 5D weight
only under an NDHWC provider guard. Failure cases: HunyuanVideo 1.5 patch
`[1,1,1]`, Framepack reordered indices, and non-base image conditioning.

### Rewrite: Hunyuan joint attention canonicalization

Source pattern:

```text
latent QKV + text added-QKV -> QK RMSNorm -> RoPE on latent span
-> masked dispatch_attention_fn over concat(latent,text)
-> split latent/text -> output projections
```

Replacement: canonical joint-attention provider with explicit latent/text spans,
mask, RoPE, and QK norm pre-ops.

Preconditions: head dim 128 supported, mask format supported, noncausal mode,
sequence/workspace within provider limits, no token-replace special gating.
Failure cases: unsupported masks, huge 720p sequence with no workspace, VAE
causal attention, Framepack image prefix tokens.

### Rewrite: FlowMatch Euler static step

Source pattern:

```text
prev = sample + (sigma_next - sigma) * model_output
```

Replacement: fused pointwise scheduler update over NCTHW latents.

Preconditions: `stochastic_sampling=false`, scalar timestep path, static shift
sigmas already materialized, no per-token timesteps. Failure cases:
HunyuanVideo 1.5 future meanflow/per-token path, stochastic FlowMatch configs.

### Rewrite: VAE causal Conv3d island

Source pattern:

```text
GroupNorm -> SiLU -> HunyuanVideoCausalConv3d -> GroupNorm -> SiLU -> Conv3d
-> residual/shortcut
```

Replacement: Conv3d/norm/activation residual block with explicit causal
temporal pad.

Preconditions: NCTHW source layout, replicate pad mode, tiling/framewise policy
disabled or represented as host-visible tiles, group count divides channel
count. NDHWC requires axis/weight rewrites. Failure cases: VAE mid attention
blocks and temporal tiled blending.

## 12. Kernel fusion candidates

Highest priority:

- Large Linear/GEMM coverage for 3072-wide QKV, added QKV, FFN, token refiner,
  and output projection.
- QK RMSNorm + 3D RoPE + attention provider prelude, with mask and sequence
  guards.
- AdaLayerNormZero/Single/Continuous modulation, gates, and residual epilogues.
- Conv3d patchify/unpatchify for `[1,2,2]` spatial patches.
- FlowMatch Euler pointwise update and true CFG arithmetic.

Medium priority:

- VAE causal Conv3d + GroupNorm + SiLU + residual blocks.
- VAE mid-block causal attention and causal mask creation.
- VAE spatial/temporal tiled decode blend kernels after non-tiled parity.
- I2V token-replace first-frame packing and scheduler update.

Lower priority:

- HunyuanVideo 1.5 Qwen/ByT5/image-token reorder and 65-channel input pack.
- Framepack history/index packing and repeated window scheduler orchestration.
- LoRA fuse/unfuse/runtime adapter mutation.
- SkyReels-specific condition padding.

## 13. Runtime staging plan

Stage 1: Parse configs for `hunyuanvideo-community/HunyuanVideo`; load
transformer/VAE weights; accept external Llama prompt embeddings, masks, and
CLIP pooled projections.

Stage 2: Implement NCTHW latent contract, Conv3d patchify/unpatchify, 3D RoPE,
embedded guidance, and one token-refiner/transformer block parity at reduced
sequence length.

Stage 3: Full `HunyuanVideoTransformer3DModel` random-tensor parity on a small
latent grid, then representative shapes as memory allows.

Stage 4: Add optional true CFG as two denoiser calls plus explicit arithmetic.

Stage 5: Implement FlowMatch Euler static-shift scheduler slice with custom
sigmas and host-visible state; validate one denoising step.

Stage 6: Add `AutoencoderKLHunyuanVideo` decode for 16-channel latents. For
first parity either disable framewise/tiling explicitly or model the default
temporal tiled decode as visible host/runtime policy.

Stage 7: Run a short deterministic base T2V loop with scheduler in host control
and VAE decode.

Stage 8: Add HunyuanVideo-I2V token-replace VAE encode and first-frame denoise
contract.

Stage 9: Add HunyuanVideo 1.5 as a separate admission: 32-channel VAE,
65-channel model input, Qwen/ByT5/SigLIP conditioning, and guider object.

## 14. Parity and validation plan

- Config/default reconciliation for base and I2V JSONs, especially scheduler
  shift 7.0 vs 17.0 and source defaults.
- Patchify/unpatchify parity for small NCTHW tensors and a 720p latent shape.
- 3D RoPE parity for several `[T,H,W]` latent grids.
- Llama token-refiner parity with synthetic masks.
- One dual-stream and one single-stream block parity with fixed text/latent
  tokens and masks.
- Full transformer forward parity at reduced sequence length, then stress
  sequence-length admission tests.
- Embedded guidance projection parity for fixed `guidance_scale`.
- True CFG arithmetic parity with fixed positive/negative predictions.
- FlowMatch `set_timesteps(sigmas=...)` and one-step parity for shift 7.0.
- VAE decode parity for `[B,16,T,H,W]`, including mid-block attention and
  scaling factor.
- VAE encode posterior mode parity for I2V readiness.
- I2V token-replace pack/update parity.
- Suggested tolerances: scheduler fp32 `rtol=1e-5, atol=1e-6`; transformer/VAE
  fp32 `rtol=1e-4, atol=1e-5`; bf16/fp16 start at `rtol=2e-2, atol=2e-2`.

## 15. Performance probes

- One transformer step by latent grid: small synthetic, 320x512x61, and
  720x1280x129.
- Attention backend comparison at head_dim 128 and long latent sequence lengths.
- Block time split: token refiner, dual-stream attention, single-stream
  attention, FFN, patchify/unpatchify.
- CFG two-call overhead versus embedded-guidance-only path.
- FlowMatch scheduler overhead compared with denoiser time.
- VAE decode throughput and memory with default framewise decoding, non-tiled
  decode, and spatial tiling.
- VAE mid-block attention cost by latent frame/spatial size.
- I2V VAE encode and token-replace scheduler update overhead.
- Faithful NCTHW VAE path versus guarded NDHWC Conv3d island.

## 16. Scope boundary and separate candidates

Separate candidate reports, not ignored:

- `hunyuan_video_i2v`: Llava image/text prompt encoding, VAE encode, latent
  concat and official token-replace first-frame conditioning.
- `hunyuan_skyreels_i2v`: SkyReels Hunyuan condition padding and source-specific
  image-to-video loop.
- `hunyuan_video_framepack`: SigLIP image embeddings, history latents, index
  packing, clean-latent windows, and repeated scheduler windows.
- `hunyuan_video_1_5`: Qwen2.5-VL + ByT5 text encoders, 32-channel VAE,
  65-channel transformer input, 54-layer transformer, ClassifierFreeGuidance.
- `hunyuan_video_1_5_i2v`: SigLIP image encoder plus 1.5 first-frame latent and
  mask conditioning.
- `hunyuan_video_lora_adapters`: `HunyuanVideoLoraLoaderMixin` and transformer
  PEFT adapter state.
- `hunyuan_video_vae_tiling_framewise`: spatial/temporal tiling, framewise
  encode/decode, slicing, and blend policy.
- `flow_match_scheduler_advanced`: dynamic shifting, stochastic sampling,
  Karras/exponential/beta sigmas, and per-token timesteps.

Genuinely ignored/out of scope for this audit:

- Multi-GPU/context parallel paths.
- Callback mutation and interactive interrupt.
- XLA, NPU, MPS, Flax, and ONNX variants.
- Safety checker and NSFW filtering.
- Training, losses, dropout behavior, and gradient checkpointing.

## 17. Final implementation checklist

- [ ] Parse base HunyuanVideo model index and component configs.
- [ ] Load `HunyuanVideoTransformer3DModel` and `AutoencoderKLHunyuanVideo` weights.
- [ ] Accept external Llama prompt embeddings/masks and CLIP pooled projections.
- [ ] Implement NCTHW latent shape contract and random latent admission checks.
- [ ] Implement Conv3d `[1,2,2]` patchify/unpatchify parity.
- [ ] Implement Hunyuan 3D RoPE generation and application.
- [ ] Implement token refiner with masked pooling/self-attention.
- [ ] Implement dual-stream and single-stream transformer block parity.
- [ ] Implement QK RMSNorm + RoPE + masked joint attention fallback/provider.
- [ ] Implement embedded guidance projection and optional true CFG arithmetic.
- [ ] Implement FlowMatch Euler static shift 7.0 with custom sigmas.
- [ ] Implement VAE 16-channel decode with scale factor 0.476986.
- [ ] Add VAE encode posterior mode parity for I2V.
- [ ] Add one-step denoising parity and short-loop smoke.
- [ ] Add I2V token-replace first-frame conditioning as a separate stage.
- [ ] Add HunyuanVideo 1.5 only after base HunyuanVideo parity is stable.
- [ ] Add guarded NDHWC/Conv3d/VAE layout optimization tests after faithful NCTHW parity.
