# Diffusers QwenImage Operator and Integration Report

Candidate slug: `qwen_image`

## 1. Source basis

```text
Diffusers commit/version:
  Local checkout X:/H/diffusers at b3a515080752a3ba7ca92161e25530c7f280f629.

Model id(s):
  Qwen/Qwen-Image
  Qwen/Qwen-Image-2512
  Qwen/Qwen-Image-Edit
  Qwen/Qwen-Image-Edit-2509
  Qwen/Qwen-Image-Edit-2511
  Qwen/Qwen-Image-Layered
  yujiepan/qwen-image-tiny-random

Config sources:
  H:/configs/Qwen/Qwen-Image/model_index.json
  H:/configs/Qwen/Qwen-Image-2512/model_index.json
  H:/configs/Qwen/Qwen-Image-Edit/model_index.json
  H:/configs/Qwen/Qwen-Image-Edit-2509/model_index.json
  H:/configs/Qwen/Qwen-Image-Edit-2511/model_index.json
  H:/configs/Qwen/Qwen-Image-Layered/model_index.json
  H:/configs/yujiepan/qwen-image-tiny-random/model_index.json
  Official component configs were read from Hugging Face raw URLs without
  saving them, because the only owned write path for this task is this report.

Pipeline files inspected:
  X:/H/diffusers/src/diffusers/pipelines/qwenimage/pipeline_qwenimage.py
  X:/H/diffusers/src/diffusers/pipelines/qwenimage/pipeline_qwenimage_img2img.py
  X:/H/diffusers/src/diffusers/pipelines/qwenimage/pipeline_qwenimage_inpaint.py
  X:/H/diffusers/src/diffusers/pipelines/qwenimage/pipeline_qwenimage_edit.py
  X:/H/diffusers/src/diffusers/pipelines/qwenimage/pipeline_qwenimage_edit_plus.py
  X:/H/diffusers/src/diffusers/pipelines/qwenimage/pipeline_qwenimage_edit_inpaint.py
  X:/H/diffusers/src/diffusers/pipelines/qwenimage/pipeline_qwenimage_controlnet.py
  X:/H/diffusers/src/diffusers/pipelines/qwenimage/pipeline_qwenimage_controlnet_inpaint.py
  X:/H/diffusers/src/diffusers/pipelines/qwenimage/pipeline_qwenimage_layered.py
  X:/H/diffusers/src/diffusers/modular_pipelines/qwenimage/*.py

Model files inspected:
  X:/H/diffusers/src/diffusers/models/transformers/transformer_qwenimage.py
  X:/H/diffusers/src/diffusers/models/controlnets/controlnet_qwenimage.py
  X:/H/diffusers/src/diffusers/models/autoencoders/autoencoder_kl_qwenimage.py

Scheduler/processors/helpers inspected:
  X:/H/diffusers/src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py
  X:/H/diffusers/src/diffusers/models/attention_dispatch.py
  X:/H/diffusers/src/diffusers/models/attention_processor.py
  X:/H/diffusers/src/diffusers/models/attention.py
  X:/H/diffusers/src/diffusers/models/embeddings.py
  X:/H/diffusers/src/diffusers/models/normalization.py
  X:/H/diffusers/src/diffusers/image_processor.py
  X:/H/diffusers/src/diffusers/loaders/lora_pipeline.py

External component configs inspected:
  Qwen2_5_VLForConditionalGeneration / Qwen2Tokenizer configs for official
  and tiny random repos. Edit/layered processor configs use Qwen2VLProcessor.

Any missing files or assumptions:
  Base Qwen/Qwen-Image and Qwen/Qwen-Image-2512 have no processor config, as
  expected for text-to-image; raw URL returned 404. No official config path was
  gated or blocked. This report focuses on the base text-to-image pipeline and
  inventories img2img, inpaint, edit, layered, ControlNet, and LoRA as separate
  candidates. XLA/NPU/MPS/Flax/ONNX, safety/NSFW, training/loss/dropout/
  gradient checkpointing, multi-GPU/context parallel, callbacks, and interrupt
  paths are out of scope.
```

## 2. Pipeline and component graph

Base `QwenImagePipeline` wires `FlowMatchEulerDiscreteScheduler`,
`AutoencoderKLQwenImage`, `Qwen2_5_VLForConditionalGeneration`,
`Qwen2Tokenizer`, and `QwenImageTransformer2DModel`. The offload sequence is
`text_encoder->transformer->vae`.

```text
prompt
  -> Qwen prompt template + Qwen2Tokenizer
  -> Qwen2.5-VL text encoder hidden states [B,L,3584] + mask [B,L]
  -> latent noise [B,1,16,H/8,W/8] source NCTHW
  -> 2x2 latent packing to [B,(H/16)*(W/16),64]
  -> denoising loop:
       QwenImageTransformer2DModel(latent tokens, text tokens, mask,
                                   timestep, img_shapes, optional guidance)
       optional true CFG second denoiser call + norm-rescaled CFG
       FlowMatchEulerDiscreteScheduler.step
  -> unpack latent tokens to [B,16,1,H/8,W/8]
  -> unstandardize with VAE latents_mean/latents_std
  -> AutoencoderKLQwenImage decode
  -> select frame 0 and VaeImageProcessor postprocess
```

First-slice required components:

- Packed-token latent denoiser `QwenImageTransformer2DModel`.
- Externally supplied Qwen prompt embeddings and masks first.
- `FlowMatchEulerDiscreteScheduler` with dynamic shifting and terminal sigma.
- QwenImage latent pack/unpack and `img_shapes` construction.
- QwenImage VAE decode boundary, with 5D latent input and latent mean/std.

Separate candidate reports:

| Candidate | Classes/files | Runtime delta |
| --- | --- | --- |
| `qwen_image_img2img` | `QwenImageImg2ImgPipeline` | Adds image preprocessing, QwenImage VAE encode, strength timestep slicing, `scheduler.scale_noise`, and packed initial image latents. |
| `qwen_image_inpaint` | `QwenImageInpaintPipeline`, `QwenImageEditInpaintPipeline` | Adds mask preprocessing/interpolation, masked-image latent composition, and extra latent/control packing contracts. |
| `qwen_image_edit` | `QwenImageEditPipeline`, `QwenImageEditPlusPipeline` | Adds Qwen2VLProcessor image/text prompt encoding and image condition sequences; `Edit-2511` sets `zero_cond_t=true`. |
| `qwen_image_layered` | `QwenImageLayeredPipeline` | Uses `use_additional_t_cond=true`, `use_layer3d_rope=true`, layered image structure, and 4-channel VAE output images. |
| `qwen_image_controlnet` | `QwenImageControlNetPipeline`, `QwenImageControlNetModel`, `QwenImageMultiControlNetModel` | Adds VAE-encoded packed control condition tokens and per-block residual samples. |
| `qwen_image_controlnet_inpaint` | `QwenImageControlNetInpaintPipeline`, ControlNet with `extra_condition_channels` | Packs masked-image latents plus mask into the control condition branch. |
| `qwen_image_lora_adapters` | `QwenImageLoraLoaderMixin`, `PeftAdapterMixin` on transformer/controlnet | Transformer LoRA load/fuse/unfuse/runtime adapter state. |

No family-local IP-Adapter, T2I-Adapter, GLIGEN, depth2img, or upscaling
pipeline was found in the `qwenimage` folder. Upscale-like community LoRA names
exist in config cache, but that is not a Diffusers pipeline class surface.

## 3. Important config dimensions

Representative checkpoint sweep:

| Config | Pipeline | Layers | Heads x dim | Inner dim | Text dim | Latent/token | Scheduler | Variant flags |
| --- | --- | ---: | --- | ---: | ---: | --- | --- | --- |
| `Qwen/Qwen-Image` | `QwenImagePipeline` | 60 | 24 x 128 | 3072 | 3584 | VAE z=16, packed C=64 | FlowMatch, dynamic exponential, terminal 0.02 | base |
| `Qwen/Qwen-Image-2512` | `QwenImagePipeline` | 60 | 24 x 128 | 3072 | 3584 | same | same | updated repo, same operator shape |
| `Qwen/Qwen-Image-Edit` | `QwenImageEditPipeline` | 60 | 24 x 128 | 3072 | 3584 | same | same | Qwen2VLProcessor image input |
| `Qwen/Qwen-Image-Edit-2511` | `QwenImageEditPipeline` | 60 | 24 x 128 | 3072 | 3584 | same | same | `zero_cond_t=true` |
| `Qwen/Qwen-Image-Layered` | `QwenImageLayeredPipeline` | 60 | 24 x 128 | 3072 | 3584 | z=16, VAE `input_channels=4` | FlowMatch, dynamic linear, no terminal shift | `use_layer3d_rope`, `use_additional_t_cond` |
| `yujiepan/qwen-image-tiny-random` | `QwenImagePipeline` | 2 | 1 x 32 | 32 | 32 | z=16, packed C=64 | FlowMatch, dynamic exponential | debug-size parity target |

Transformer fields:

| Field | Official base value | Source default / note |
| --- | ---: | --- |
| `patch_size` | 2 | `proj_out` emits `patch_size * patch_size * out_channels = 64`. |
| `in_channels` | 64 | Packed 2x2 VAE latent tiles, not VAE channel count. |
| `out_channels` | 16 | Unpacked back to VAE latent channels. |
| `num_layers` | 60 | All dual-stream blocks; no Flux single-stream tail. |
| `num_attention_heads` | 24 | Inner dim = 3072. |
| `attention_head_dim` | 128 | RoPE axes sum = 128. |
| `joint_attention_dim` | 3584 | Qwen2.5-VL hidden size. |
| `axes_dims_rope` | `[16,56,56]` | Frame/height/width rotary sections. |
| `guidance_embeds` | false | Source supports flag, sampled official configs disable it. |

Text encoder config facts:

| Component | Official base dimensions |
| --- | --- |
| `Qwen2_5_VLForConditionalGeneration` | hidden 3584, 28 layers, 28 attention heads, 4 KV heads, intermediate 18944, vocab 152064, Qwen2.5-VL text/vision config present. |
| `Qwen2Tokenizer` | `model_max_length=131072`, pad token `<|endoftext|>`, EOS `<|im_end|>`, special vision/image/video tokens. |
| Prompt extraction | Pipeline template adds a system instruction and drops the first 34 tokens after masked hidden-state extraction. Default `max_sequence_length=512`; hard max 1024. |

QwenImage VAE:

| Field | Official base value |
| --- | --- |
| Class | `AutoencoderKLQwenImage` |
| `base_dim` | 96 |
| `z_dim` | 16 |
| `dim_mult` | `[1,2,4,4]` |
| `num_res_blocks` | 2 |
| `temperal_downsample` | `[false,true,true]`; source `vae_scale_factor = 2 ** len(...) = 8` |
| Latent stats | 16-element `latents_mean` and `latents_std`; decode uses `latents / (1/std) + mean`, equivalent to `latents * std + mean`. |

Scheduler support:

- Pipeline constructor is specifically `FlowMatchEulerDiscreteScheduler`.
- Sampled official configs set `use_dynamic_shifting=true`, `base_image_seq_len=256`,
  `max_image_seq_len=8192`, `base_shift=0.5`, `max_shift=0.9`,
  `shift=1.0`, and `stochastic_sampling=false`.
- Base/edit configs use `time_shift_type="exponential"` and
  `shift_terminal=0.02`; layered uses `time_shift_type="linear"` and
  `shift_terminal=false`.
- Recommended first Dinoml scheduler slice: FlowMatch Euler with custom sigmas,
  dynamic exponential shift, terminal stretch, and non-stochastic step. Add
  layered linear shift as a second scheduler variant.

## 3a. Family variation traps

- `in_channels=64` is packed-token width, not the VAE latent channel count.
  VAE latent maps are `[B,16,1,H/8,W/8]`; the transformer sees
  `[B,(H/16)*(W/16),64]`.
- Base QwenImage uses a 3D VAE with a singleton frame axis even for images.
  Treat VAE tensors as NCTHW until a guarded 2D decode specialization proves
  parity.
- Prompt embeddings are variable length after mask extraction, then padded to
  the batch maximum and optionally truncated to `max_sequence_length`.
- QwenImage true CFG is a second positive/negative denoiser call, not batch
  concatenation. It also rescales by `norm(noise_pred)/norm(comb_pred)` over
  the packed-channel dimension.
- `guidance_scale` is ignored for sampled base configs because
  `guidance_embeds=false`; `true_cfg_scale` is the active CFG mechanism.
- Text masks may be arbitrary, not only contiguous padding. The transformer
  builds a joint attention mask over `[text,image]`.
- Base and layered use different FlowMatch time-shift functions and terminal
  stretching.
- `Edit-2511` sets `zero_cond_t=true`, which doubles timestep conditioning and
  uses `modulate_index` to select modulation for image versus condition tokens.
- `Qwen-Image-Layered` switches to `QwenEmbedLayer3DRope`, adds an additional
  timestep condition embedding, and has a 4-channel VAE output boundary.
- Attention dispatch comments call out that joint attention in Qwen-Image can
  make local sequence lengths non-inferable from naive tensor splits. Ignore
  context parallel for this audit, but keep the sequence split trap in provider
  validation.

## 4. Runtime tensor contract

For a 1024x1024 base text-to-image request:

| Boundary | Tensor | Shape/layout | Notes |
| --- | --- | --- | --- |
| Prompt tokens | tokenizer output | `[B,L+34]` | CPU/data path; template plus user prompt. |
| Text hidden | raw Qwen output | `[B,L_full,3584]` | Last hidden state from Qwen2.5-VL. |
| Prompt embeds | `prompt_embeds` | `[B,L,3584]`, `L<=512` default | Extract mask-valid tokens, drop first 34, pad to max, truncate. |
| Prompt mask | `prompt_embeds_mask` | `[B,L]` or `None` | `None` when all valid. |
| Noise latent map | before pack | `[B,1,16,128,128]` NCTHW | Source random tensor. |
| Packed latents | transformer input | `[B,4096,64]` | 2x2 spatial pack of 16 channels. |
| `img_shapes` | metadata | `[[(1,64,64)]] * B` | Frame, packed latent H, packed latent W. |
| Timestep | model input | `[B]` | Pipeline passes `t / 1000`; embedding scales by 1000. |
| Joint attn mask | internal | `[B,1,1,L+4096]` in transformer; ControlNet path builds `[B,L+S]` before processor handling | Text mask plus all-true image mask. |
| Denoiser output | `noise_pred` | `[B,4096,64]` | Same packed-token shape. |
| Scheduler output | `latents` | `[B,4096,64]` | Flow Euler update. |
| Unpacked latents | decode input | `[B,16,1,128,128]` NCTHW | `_unpack_latents` restores frame axis. |
| VAE normalized latent | decode input | `[B,16,1,128,128]` | `latents * latents_std + latents_mean`. |
| VAE decoded | image/video tensor | `[B,3,1,1024,1024]` | Pipeline selects `[:, :, 0]`. |
| Postprocess | output image | `[B,3,1024,1024]` before conversion | `VaeImageProcessor` handles output type. |

CPU/data-pipeline work: tokenization, prompt templating, masked extraction,
padding, image/mask preprocessing for variants, PIL/NumPy conversion.
GPU/runtime work: text encoder if admitted, latent pack/unpack, denoiser,
scheduler arithmetic, CFG arithmetic, VAE encode/decode.

Cacheable stages: prompt embeddings/masks, VAE image latents for img2img/control
variants, `img_shapes`/RoPE frequencies for fixed resolution, scheduler sigma
tables per step count/resolution.

## 5. Operator coverage checklist

Tensor/layout ops:

- NCTHW latent allocation; NCTHW/NCHW image preprocessing boundaries.
- 2x2 latent pack:
  `view(B,C,H/2,2,W/2,2) -> permute(0,2,4,1,3,5) -> reshape(B,H/2*W/2,4C)`.
- 2x2 latent unpack:
  `view(B,H/2,W/2,C/4,2,2) -> permute(0,3,1,4,2,5) -> reshape(B,C/4,1,H,W)`.
- Masked select/split/pad in prompt embedding path if compiling text pipeline.
- Text/image joint mask concat and broadcast.
- CFG two-call arithmetic and norm ratio rescale.
- VAE tiling/slicing reshape/cat/blend paths as later codec work.

GEMM/linear ops:

- `img_in`: Linear(64 -> 3072).
- `txt_in`: Linear(3584 -> 3072), preceded by RMSNorm.
- Per-block image/text modulation MLP: SiLU + Linear(3072 -> 18432), split into
  scale/shift/gate for attention and MLP.
- Image Q/K/V and text added Q/K/V projections, all 3072-wide with bias.
- Attention output projections for image and text streams.
- GEGLU/GELU approximate `FeedForward(dim=3072)`.
- Final `AdaLayerNormContinuous` + Linear(3072 -> 64).

Attention primitives:

- 60 layers of noncausal joint text-image attention.
- QK RMSNorm on image and text query/key streams.
- Complex RoPE from `QwenEmbedRope` over frame/height/width and offset text
  positions.
- Additive/boolean text padding masks expanded over `[text,image]`.
- Eager/native attention dispatch is parity; flash-style lowering is guarded.

Normalization and adaptive conditioning:

- LayerNorm without affine before attention and MLP on image/text streams.
- RMSNorm on text encoder hidden states and attention Q/K.
- AdaLayerNormContinuous final norm.
- Scale/shift/gate modulation from timestep embedding.
- QwenImage VAE custom RMS normalization over channel-first 5D and 4D tensors.

Position/timestep/guidance embeddings:

- `Timesteps(256, flip_sin_to_cos=True, scale=1000)` plus `TimestepEmbedding`.
- Optional `guidance` path is source-supported but inactive in sampled configs.
- Optional `addition_t_cond` for layered variant.
- 3-axis RoPE with `axes_dims_rope=[16,56,56]`.

Scheduler and guidance arithmetic:

- FlowMatch `set_timesteps(sigmas, mu)`, dynamic shift, terminal stretch.
- Step: `prev = sample + (sigma_next - sigma) * model_output`.
- Img2img/control noising: `sigma * noise + (1 - sigma) * sample`.
- True CFG: `neg + scale * (pos - neg)`, then norm-rescale.

VAE/postprocessing ops:

- 3D causal Conv3d, 2D Conv2d in resample/attention subpaths, zero pad,
  nearest-exact upsample, stride downsample.
- SiLU, RMSNorm, residual adds, single-head SDPA in VAE attention block.
- Diagonal Gaussian encode distribution for img2img/control variants.
- Clamp decode to `[-1,1]`; postprocess.

## 6. Denoiser/model breakdown

`QwenImageTransformer2DModel.forward`:

```text
hidden_states [B,S_img,64] -> img_in -> [B,S_img,3072]
encoder_hidden_states [B,S_txt,3584] -> RMSNorm -> txt_in -> [B,S_txt,3072]
timestep (/1000 from pipeline, scaled back in embedding) -> temb [B,3072]
QwenEmbedRope(img_shapes, max_txt_seq_len) -> image/text RoPE
60 x QwenImageTransformerBlock
optional ControlNet residual add per interval
AdaLayerNormContinuous(hidden_states, temb) -> Linear -> [B,S_img,64]
```

`QwenImageTransformerBlock`:

```text
img_mod/text_mod: SiLU -> Linear(dim -> 6*dim)
image stream:
  LayerNorm -> adaptive scale/shift -> QKV
text stream:
  LayerNorm -> adaptive scale/shift -> added QKV
QK RMSNorm on both streams
RoPE on image and text Q/K
concat [text,image] Q/K/V -> dispatch_attention_fn -> split
image/text output projections
gated residual attention add
LayerNorm -> adaptive scale/shift -> GELU approximate FF -> gated residual add
fp16 clipping guard on both streams
```

Unlike Flux, QwenImage has no single-stream block tail in the base transformer.
Every block updates both text and image streams.

ControlNet model:

```text
hidden_states -> img_in
controlnet_cond -> zero Linear(in_channels + extra_condition_channels -> 3072)
add to image stream
run N copied QwenImageTransformerBlock layers
zero Linear per block -> scaled controlnet_block_samples
base transformer adds samples to hidden_states at block intervals
```

## 7. Attention requirements

Primary target path: `QwenDoubleStreamAttnProcessor2_0` in
`transformer_qwenimage.py`, which calls `dispatch_attention_fn` from
`attention_dispatch.py`.

Required behavior:

- Noncausal joint attention over concatenated `[text,image]` sequence.
- Q/K/V shape before dispatch is `[B,S,heads,head_dim]`.
- Official base: 24 heads, head dim 128.
- Separate image and text QKV projections; text uses `add_q_proj`,
  `add_k_proj`, `add_v_proj`, and `to_add_out`.
- QK RMSNorm before RoPE.
- Complex RoPE path uses `apply_rotary_emb_qwen(..., use_real=False)`.
- Attention mask, when present, masks only text padding while image tokens are
  all valid. Base no-padding prompts may pass `None`.

Flash-style/provider notes:

- `attention_dispatch.py` exposes native SDPA, flash, varlen, flex, sage,
  xFormers, and context-parallel wrappers. Eager/native dispatch defines first
  parity.
- Base QwenImage with no padding mask is a plausible flash-style target if the
  provider accepts head dim 128, dtype, noncausal attention, and sequence
  length around `L+4096` for 1024px.
- Masked prompts require a provider path with mask support or a fallback.
- RoPE and QK RMSNorm must remain explicit pre-attention ops unless a fused
  provider takes them under strict preconditions.
- Context-parallel comments specifically warn that Qwen-Image joint attention
  can make local sequence length inference wrong after concatenating text and
  image sequences. Single-device Dinoml can ignore the path but should retain a
  no-naive-split guard for future provider work.
- IP-Adapter attention processors in `attention_processor.py` are not active in
  the QwenImage family files inspected.

## 8. Scheduler and denoising-loop contract

Base pipeline schedule:

```text
sigmas = linspace(1.0, 1 / num_inference_steps, num_inference_steps)
image_seq_len = packed_latents.shape[1]
mu = calculate_shift(image_seq_len, base_image_seq_len, max_image_seq_len,
                     base_shift, max_shift)
scheduler.set_timesteps(num_inference_steps, sigmas=sigmas, mu=mu)
scheduler.set_begin_index(0)
```

For 1024px, `image_seq_len=4096`, so with official base scheduler config
`mu = 0.5 + (4096 - 256) * (0.9 - 0.5) / (8192 - 256)`, about 0.694.

`FlowMatchEulerDiscreteScheduler` details:

- `use_dynamic_shifting=true` requires `mu`.
- `time_shift_type="exponential"` for base/edit/2512/tiny:
  `exp(mu) / (exp(mu) + (1/t - 1)^sigma)`.
- `time_shift_type="linear"` for layered:
  `mu / (mu + (1/t - 1)^sigma)`.
- `shift_terminal=0.02` stretches the final shifted sigma for base/edit/2512;
  layered disables this.
- Non-stochastic step is `sample + dt * model_output`, where
  `dt = sigma_next - sigma`.
- Scheduler upcasts `sample` to fp32 for step math, then casts back to model
  output dtype when not using per-token timesteps.

Guidance loop:

- If `true_cfg_scale > 1` and a negative prompt/embedding exists, the pipeline
  performs a separate unconditional transformer call, not batch concatenation.
- Combined prediction:
  `comb = neg + true_cfg_scale * (pos - neg)`.
- QwenImage then rescales:
  `noise_pred = comb * (norm(pos, dim=-1) / norm(comb, dim=-1))`.
- Keep scheduler iteration, sigma table generation, optional true CFG second
  call, and callback/interrupt out of the compiled first artifact. Compile the
  denoiser step and pointwise scheduler/CFG arithmetic after parity.

## 9. Position, timestep, and custom math

Custom math Dinoml must reproduce:

- Qwen latent pack/unpack with a singleton frame axis.
- Prompt hidden-state extraction by attention mask, dropping the template prefix
  token span of 34.
- Dynamic scheduler shift and optional terminal stretch.
- Qwen RoPE:
  frame, height, and width complex frequencies are concatenated; with
  `scale_rope=true`, height/width use negative indices for the lower half and
  positive indices for the upper half.
- Text RoPE positions start at `max_vid_index`, where base uses the maximum of
  packed latent height/width after the scale-rope half-index logic.
- Timestep embedding uses `scale=1000`; pipeline passes `timestep / 1000`.
- Decode latent unstandardization uses VAE config vectors:
  `latents = latents * latents_std + latents_mean`.

Precomputable: RoPE frequencies for fixed `img_shapes` and text length, scheduler
sigmas/timesteps for fixed step count and resolution, prompt embeddings/masks,
VAE control/image latents for variant paths.

Dynamic: prompt length, image resolution, CFG enabled/disabled, timestep,
guidance distilled flag if a future checkpoint sets it, layered
`additional_t_cond`, and variant image/mask/control inputs.

## 10. Preprocessing and input packing

Text preprocessing:

- Base prompt template is a fixed system/user/assistant chat string asking for
  detailed image description.
- Tokenizer call uses `max_length=tokenizer_max_length + drop_idx`, padding,
  truncation, and returns tensors.
- Text encoder runs with `output_hidden_states=True`; the last hidden state is
  filtered by attention mask, prefix-dropped, padded, truncated, and repeated
  for `num_images_per_prompt`.
- Negative prompt embeddings follow the same path only when true CFG is active.

Latent preprocessing:

- Text-to-image starts from random `[B,1,16,H/8,W/8]` noise and packs to
  `[B,H/16*W/16,64]`.
- Img2img and some control paths encode image tensors through the QwenImage VAE,
  standardize latents as `(latent - mean) * std`, add noise with
  `scheduler.scale_noise`, then pack.
- ControlNet image path preprocesses control images, encodes through VAE when
  needed, standardizes, permutes to `[B,T,C,H,W]`, and packs.
- ControlNet inpaint packs masked-image latents plus mask, so control
  `extra_condition_channels` changes the control branch input width.
- Layered/edit variants use Qwen2VLProcessor image inputs and may have
  multi-layer `img_shapes` rather than one `(1,H,W)` tuple.

Image postprocessing:

- Base output unpacks latents, unstandardizes by VAE mean/std, decodes
  `[B,16,1,H/8,W/8]`, selects frame 0, and runs `VaeImageProcessor.postprocess`.

## 11. Graph rewrite / lowering opportunities

### Rewrite: Qwen latent pack/unpack

Source pattern: NCTHW latent map with `T=1`, spatial 2x2 tile packing into
tokens and reverse unpack before VAE.

Replacement: layout-aware `qwen_latent_pack2x2` and `qwen_latent_unpack2x2`
ops, or a canonical reshape/permute/reshape sequence.

Preconditions: `T=1`, source channel count 16, height/width divisible by 2,
token channel equals `4*C`, flatten order exactly row-major `(h2,w2,c,dh,dw)`.

Layout constraints: source semantic layout is NCTHW. NHWC/NDHWC translation may
only apply if pack/unpack flatten order is rewritten and consumers agree.

Failure cases: layered/edit multi-condition `img_shapes`, non-16 latent channel
experiments, direct user-provided packed latents with unknown provenance.

Parity test: random `[B,1,16,128,128]` pack/unpack round trip and exact match to
Diffusers pack outputs.

### Rewrite: Qwen dual-stream joint attention

Source pattern: image/text modulation -> separate QKV/add-QKV -> QK RMSNorm ->
RoPE -> concat text/image -> attention -> split -> projections.

Replacement: explicit `joint_attention(text,image)` graph region with provider
choice after QK norm and RoPE, or a fused projection/norm/RoPE/attention kernel.

Preconditions: no ControlNet residual inside the attention op, no adapter
processor mutation, known text and image sequence lengths, provider supports
mask state and head dim 128.

Failure cases: arbitrary text masks on a provider without mask support, future
IP/adapter branches, context-parallel sequence splitting, different RoPE mode.

Parity test: one `QwenDoubleStreamAttnProcessor2_0` call with and without a text
padding mask, fp32/bf16.

### Rewrite: FlowMatch Euler Qwen slice

Source pattern: scheduler table generation, dynamic shift, optional terminal
stretch, then `sample + dt * model_output`.

Replacement: host-visible schedule metadata plus fused pointwise step.

Preconditions: `stochastic_sampling=false`, scalar timestep path, no per-token
timesteps, explicit `sigma_idx` and `sigma_next`.

Failure cases: stochastic branch, per-token timesteps, Karras/exponential/beta
conversion modes beyond sampled configs.

Parity test: compare `set_timesteps` tables and one step for base exponential
and layered linear configs.

### Rewrite: VAE image-only decode specialization

Source pattern: QwenImage 3D VAE decode loops frame by frame with `T=1`, using
causal Conv3d caches and 2D resample subpaths.

Replacement: initially keep as a QwenImage VAE codec island; later specialize
the `T=1` path into Conv3d/Conv2d/RMSNorm/SDPA kernels with explicit cache
semantics.

Preconditions: base image decode only, `T=1`, tiling/slicing disabled, source
NCTHW axes preserved.

Failure cases: img2img encode path, video-like multi-frame calls, VAE tiling,
layered 4-channel output, cache state not modeled.

Parity test: decode fixed `[1,16,1,128,128]` latent against Diffusers.

## 12. Kernel fusion candidates

Highest priority:

- Qwen dual-stream attention provider: separate image/text QKV, QK RMSNorm,
  complex RoPE, concat attention, split projections. This is the 60-layer
  denoiser hot path.
- Adaptive LayerNorm/modulation/gated residual epilogues around attention and
  MLP.
- GELU approximate FeedForward fusion for 3072-wide MLPs.
- Latent pack/unpack kernels and shape guards.
- FlowMatch Euler step plus Qwen true CFG norm-rescale pointwise/reduction
  arithmetic.

Medium priority:

- RoPE frequency generation/cache for fixed image/text shapes.
- Text mask to joint attention mask construction.
- QwenImage VAE decode kernels: causal Conv3d, RMSNorm, nearest upsample,
  residual blocks, single-head spatial SDPA.
- Img2img/control VAE encode and `scale_noise` noising kernels.
- Layered `additional_t_cond` and `QwenEmbedLayer3DRope`.

Lower priority:

- LoRA load/fuse/unfuse and hotswap behavior.
- ControlNet residual branch and multi-control aggregation.
- VAE tiling/slicing overlap blending.
- Qwen2.5-VL text encoder compilation; prompt embeddings can be supplied
  externally first.

NHWC guarded notes:

- Transformer core is token-major `[B,S,C]`; NHWC is not relevant there.
- VAE and image processors are source NCTHW/NCHW. NHWC/NDHWC can be explored
  only inside fully controlled Conv/RMSNorm/resample islands.
- Required axis rewrites include RMSNorm channel axis `dim=1 -> dim=-1`,
  Conv3d/Conv2d weight transforms, interpolation dimensions, mask/interpolate
  axes in inpaint, and pack/unpack flatten order.
- Protect prompt/text/token sequence ops and scheduler broadcasting with a
  conceptual `no_layout_translation()` guard.

## 13. Runtime staging plan

Stage 1: Parse QwenImage model index and configs. Use
`yujiepan/qwen-image-tiny-random` for shape/parity smoke and official
`Qwen/Qwen-Image` for real dimensions. Accept external `prompt_embeds` and
`prompt_embeds_mask`.

Stage 2: Implement Qwen latent pack/unpack and `img_shapes` generation parity.

Stage 3: Implement one `QwenImageTransformerBlock` with random token tensors,
including QK RMSNorm, complex RoPE, text mask handling, modulation, gates, and
MLP.

Stage 4: Compile full tiny random `QwenImageTransformer2DModel`, then official
base denoiser step at `[1,4096,64]` with prompt embeds `[1,L,3584]`.

Stage 5: Add FlowMatch Euler dynamic exponential scheduler table and one-step
parity. Include terminal stretch.

Stage 6: Add true CFG two-call orchestration and norm-rescale arithmetic.

Stage 7: Add QwenImage VAE decode boundary or keep it as a separate codec
artifact with a clear call contract.

Stage 8: Optimize attention and norm/MLP fusions; then admit img2img,
ControlNet, edit, and layered variants as separate slices.

First Dinoml admission recommendation: admit `qwen_image_base_denoiser_step`
before the full pipeline. Inputs should be packed latents `[B,S,64]`, prompt
embeddings `[B,L,3584]`, optional prompt mask `[B,L]`, scalar/batch timestep,
and `img_shapes`. Output is packed latent derivative `[B,S,64]`. Keep Qwen2.5-VL
text encoding, true CFG orchestration, scheduler loop state, and VAE decode
outside the compiled artifact until block/full-denoiser parity is stable.

## 14. Parity and validation plan

- Config parse tests for official base, 2512, edit, edit-2511, layered, and tiny
  random model indexes plus raw component configs.
- Prompt embedding extraction parity for a fixed prompt: template tokens,
  mask-select, drop 34, pad/truncate, repeat.
- Pack/unpack parity for `[B,1,16,128,128] <-> [B,4096,64]`.
- Qwen RoPE frequency parity for `(1,64,64)` and text lengths 1, 512, 1024.
- Attention processor parity with no mask and with non-contiguous text masks.
- One `QwenImageTransformerBlock` random tensor parity.
- Full tiny random transformer parity.
- Official base one-step denoiser parity with fixed prompt embeddings and
  latents.
- FlowMatch scheduler table and one-step parity for base exponential terminal
  stretch and layered linear/no-terminal configs.
- True CFG arithmetic and norm-rescale parity.
- QwenImage VAE decode parity for `[1,16,1,128,128]`; add encode parity for
  img2img/control variants.
- End-to-end smoke only after text encoder, scheduler, denoiser, and VAE stages
  each have parity.
- Suggested tolerances: scheduler fp32 `rtol=1e-5, atol=1e-6`; transformer fp32
  `rtol=1e-4, atol=1e-5`; bf16/fp16 initially `rtol=2e-2, atol=2e-2`.

## 15. Performance probes

- Denoiser step by sequence length: 1024, 4096, 8192 packed image tokens plus
  text length sweep.
- Attention backend comparison for Qwen dual-stream attention: native SDPA,
  guarded flash-style provider, masked fallback.
- Per-block time split: QKV/add-QKV, QK RMSNorm/RoPE, attention, projections,
  MLP.
- True CFG overhead: one denoiser call versus two denoiser calls plus norm
  rescale.
- Scheduler/CFG overhead versus denoiser time.
- Pack/unpack overhead and memory traffic.
- VAE decode throughput for 1024px, with and without tiling later.
- Qwen2.5-VL text encoder throughput if/when admitted.
- VRAM/workspace by dtype and text length; 60 blocks at 4096+ text tokens are
  the main pressure point.

## 16. Scope boundary and separate candidates

Separate candidate reports, not ignored:

- `qwen_image_img2img`: VAE encode, strength slicing, `scale_noise`.
- `qwen_image_inpaint`: mask preprocessing, masked latents, mask packing.
- `qwen_image_edit`: Qwen2VLProcessor image/text conditioning and edit prompt
  structure.
- `qwen_image_edit_2511`: `zero_cond_t` modulation index behavior.
- `qwen_image_layered`: layer-3D RoPE, `additional_t_cond`, 4-channel VAE output.
- `qwen_image_controlnet`: transformer-side residual stream and packed control
  image latents.
- `qwen_image_controlnet_inpaint`: ControlNet plus mask/latent extra condition
  channels.
- `qwen_image_lora_adapters`: transformer LoRA load/fuse/unfuse/runtime adapter
  mutation.
- `qwen_image_vae_codec`: standalone QwenImage 3D VAE encode/decode,
  tiling/slicing/cache behavior.
- Rare FlowMatch options beyond sampled configs: stochastic sampling,
  per-token timesteps, Karras/exponential/beta sigma conversions.

Genuinely ignored/out of scope for this audit:

- Multi-GPU/context parallel paths.
- Callback mutation and interactive interrupt.
- XLA, NPU, MPS, Flax, and ONNX variants.
- Safety checker and NSFW filtering.
- Training, losses, dropout, and gradient checkpointing.
- Textual inversion, IP-Adapter, T2I-Adapter, GLIGEN, depth2img, and upscaling:
  no active QwenImage family implementation was found in the inspected
  Diffusers folder.

## 17. Final implementation checklist

- [ ] Parse QwenImage model index and component configs.
- [ ] Load `QwenImageTransformer2DModel` weights for tiny random and official base.
- [ ] Accept external Qwen prompt embeddings and prompt masks.
- [ ] Implement Qwen 2x2 latent pack/unpack.
- [ ] Implement `img_shapes` and Qwen RoPE frequency parity.
- [ ] Implement Qwen timestep embedding path.
- [ ] Implement `QwenDoubleStreamAttnProcessor2_0` fallback parity.
- [ ] Implement QK RMSNorm + complex RoPE + joint attention.
- [ ] Implement `QwenImageTransformerBlock` modulation/gates/MLP/residual path.
- [ ] Add full tiny random transformer parity.
- [ ] Add official base one-step denoiser parity.
- [ ] Implement FlowMatch Euler dynamic exponential terminal scheduler slice.
- [ ] Add true CFG two-call and norm-rescale parity.
- [ ] Add QwenImage VAE decode boundary or separate codec artifact.
- [ ] Benchmark attention and denoiser step at 1024px token length.
- [ ] Open separate reports for img2img, inpaint/edit/layered, ControlNet, LoRA, and QwenImage VAE codec.
