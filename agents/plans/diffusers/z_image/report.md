# Diffusers Z-Image Operator and Integration Report

Candidate slug: `z_image`

## 1. Source basis

```text
Diffusers commit/version:
  Local checkout diffusers at b3a515080752a3ba7ca92161e25530c7f280f629.

Model id(s):
  Tongyi-MAI/Z-Image
  Tongyi-MAI/Z-Image-Turbo
  yujiepan/z-image-tiny-random
  snake7gun/tiny-random-z-image-turbo
  tiny-random/z-image
  neuralvfx/Z-Image-SAM-ControlNet, variant/config reference only

Config sources:
  H:/configs/Tongyi-MAI/Z-Image/model_index.json
  H:/configs/Tongyi-MAI/Z-Image-Turbo/model_index.json
  H:/configs/yujiepan/z-image-tiny-random/model_index.json
  H:/configs/snake7gun/tiny-random-z-image-turbo/model_index.json
  H:/configs/tiny-random/z-image/model_index.json
  Official component configs were read from Hugging Face raw URLs without
  saving them, because the only owned write path for this task is this report.

Pipeline files inspected:
  diffusers/src/diffusers/pipelines/z_image/pipeline_z_image.py
  diffusers/src/diffusers/pipelines/z_image/pipeline_z_image_img2img.py
  diffusers/src/diffusers/pipelines/z_image/pipeline_z_image_inpaint.py
  diffusers/src/diffusers/pipelines/z_image/pipeline_z_image_omni.py
  diffusers/src/diffusers/pipelines/z_image/pipeline_z_image_controlnet.py
  diffusers/src/diffusers/pipelines/z_image/pipeline_z_image_controlnet_inpaint.py
  diffusers/src/diffusers/pipelines/z_image/pipeline_output.py

Model files inspected:
  diffusers/src/diffusers/models/transformers/transformer_z_image.py
  diffusers/src/diffusers/models/controlnets/controlnet_z_image.py
  diffusers/src/diffusers/models/autoencoders/autoencoder_kl.py
  diffusers/src/diffusers/models/autoencoders/vae.py

Scheduler/processors/helpers inspected:
  diffusers/src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py
  diffusers/src/diffusers/models/attention_processor.py
  diffusers/src/diffusers/models/attention_dispatch.py
  diffusers/src/diffusers/image_processor.py
  diffusers/src/diffusers/loaders/lora_pipeline.py
  diffusers/src/diffusers/loaders/lora_conversion_utils.py
  diffusers/src/diffusers/loaders/single_file_utils.py
  diffusers/src/diffusers/loaders/single_file_model.py

External component configs inspected:
  Qwen3Model / Qwen3ForCausalLM text encoder configs.
  Qwen2Tokenizer tokenizer configs.
  Siglip2VisionModel / Siglip2ImageProcessorFast source imports for Omni,
  but no official Omni component config was available in the local cache.

Any missing files or assumptions:
  Local config cache had model_index.json only for official and tiny Z-Image
  repos. Official component configs for base, Turbo, and tiny random were
  accessible over unauthenticated Hugging Face raw URLs. Alibaba ControlNet
  single-file repos returned 404 for config/model_index raw paths; the open
  neuralvfx/Z-Image-SAM-ControlNet config was used only to identify the
  ControlNet operator shape. This report focuses on base text-to-image and
  inventories img2img, inpaint, Omni, ControlNet, ControlNet-inpaint, LoRA,
  single-file conversion, and modular pipeline surfaces separately. XLA/NPU/
  MPS/Flax/ONNX, safety/NSFW, training/loss/dropout/gradient checkpointing,
  multi-GPU/context parallel, callbacks, and interrupt paths are out of scope.
```

## 2. Pipeline and component graph

Base `ZImagePipeline` wires `FlowMatchEulerDiscreteScheduler`,
`AutoencoderKL`, `Qwen3Model`/`Qwen3ForCausalLM` as a hidden-state provider,
`Qwen2Tokenizer`, and `ZImageTransformer2DModel`. The offload sequence is
`text_encoder->transformer->vae`.

```text
prompt
  -> Qwen chat template + Qwen2Tokenizer
  -> Qwen3 hidden states[-2], masked into per-sample variable-length lists
  -> latent noise [B,16,H/8,W/8] source NCHW at 1024px -> [B,16,128,128]
  -> transformer call receives list of [16,1,128,128] C-F-H-W tensors
  -> internal patchify: 2x2x1 patches, image tokens, caption tokens, RoPE ids
  -> denoising loop:
       ZImageTransformer2DModel(latent list, caption embeds list, timestep)
       optional batch CFG + optional norm clamp
       sign flip, FlowMatchEulerDiscreteScheduler.step
  -> VAE decode from latents / scaling_factor + shift_factor
  -> VaeImageProcessor postprocess
```

Required first-slice components:

- `ZImageTransformer2DModel` denoiser with externally supplied caption embeds.
- Internal patchify/unpatchify for C-F-H-W tensors and sequence padding to
  `SEQ_MULTI_OF=32`.
- Qwen3 caption projection from 2560 to transformer dim 3840 for official
  checkpoints.
- FlowMatch Euler with static shift, custom sigmas support, terminal zero sigma,
  and non-stochastic step.
- AutoencoderKL decode boundary with Flux-style latent scaling and shift.

Separate candidate reports:

| Candidate | Classes/files | Runtime delta |
| --- | --- | --- |
| `z_image_img2img` | `ZImageImg2ImgPipeline` | Adds image preprocessing, AutoencoderKL encode, strength timestep slicing, scheduler `scale_noise`, and initial image latents. |
| `z_image_inpaint` | `ZImageInpaintPipeline` | Adds mask preprocessing, masked-image VAE encode, latent mask blending each step, and callback-visible mask tensors. |
| `z_image_omni` | `ZImageOmniPipeline`, optional `siglip_embedder` in transformer | Adds condition images, SigLIP2 image embeddings, multi-image prompt chunking, per-token noisy/clean modulation, and Flux2 image processor. |
| `z_image_controlnet` | `ZImageControlNetPipeline`, `ZImageControlNetModel` | Adds VAE-encoded control image context, shared transformer modules via `from_transformer`, and per-layer residual samples. |
| `z_image_controlnet_inpaint` | `ZImageControlNetInpaintPipeline`, `ZImageControlNetModel` | Adds mask plus masked-image latents to the control context; rejects ControlNet configs whose `control_in_dim` equals transformer latent channels. |
| `z_image_lora_adapters` | `ZImageLoraLoaderMixin`, Z-Image LoRA conversion helpers | Transformer-only PEFT/LoRA load, hotswap, fuse/unfuse, and non-Diffusers key conversion. |
| `z_image_single_file` | `FromSingleFileMixin`, `convert_z_image_transformer_checkpoint_to_diffusers`, `convert_z_image_controlnet_checkpoint_to_diffusers` | Fused QKV checkpoint splitting and original checkpoint key remaps. |
| `z_image_modular_pipeline` | `modular_pipelines/z_image/*` | Modular blocks for text-to-image and image-to-image; useful for future explicit state staging but not the base source of parity. |

No Z-Image family-local IP-Adapter, T2I-Adapter, GLIGEN, depth2img, or upscaling
pipeline was found in the inspected non-deprecated `pipelines/z_image` folder.

## 3. Important config dimensions

Representative checkpoint sweep:

| Config | Pipeline | Layers | Refiner layers | Dim | Heads x dim | Latent | Text dim | Scheduler | Notes |
| --- | --- | ---: | ---: | ---: | --- | --- | ---: | --- | --- |
| `Tongyi-MAI/Z-Image` | `ZImagePipeline` | 30 | 2 noise + 2 context | 3840 | 30 x 128 | C=16, patch 2 | 2560 | FlowMatch shift 6.0 | Base, non-Turbo shift. |
| `Tongyi-MAI/Z-Image-Turbo` | `ZImagePipeline` | 30 | 2 + 2 | 3840 | 30 x 128 | C=16, patch 2 | 2560 | FlowMatch shift 3.0 | Main first production target. |
| `yujiepan/z-image-tiny-random` | `ZImagePipeline` | 2 | 2 + 2 | 64 | 2 x 32 | C=8, patch 2 | 8 | FlowMatch shift 3.0 | Tiny shape has `n_kv_heads=4` but source `Attention` path uses `heads`; validate config/model consistency before relying on GQA. |
| `snake7gun/tiny-random-z-image-turbo` | `ZImagePipeline` | 2 | 1 + 1 | 48 | 2 x 24 | C=16, patch 2 | 64 | FlowMatch shift 1.0 | Tiny Turbo smoke config. |
| `neuralvfx/Z-Image-SAM-ControlNet` | ControlNet config only | 30 | 2 + 2 | 3840 | 30 x 128 | control C=33 | 2560 | N/A | Control config, not base pipeline config. |

Transformer fields:

| Field | Official value | Source/default note |
| --- | ---: | --- |
| `all_patch_size` | `[2]` | Spatial 2x2 patching inside transformer. |
| `all_f_patch_size` | `[1]` | Singleton image frame axis is preserved as F=1. |
| `in_channels` / `out_channels` | 16 | Equal to AutoencoderKL latent channels. |
| `dim` | 3840 | Hidden width for all transformer streams. |
| `n_layers` | 30 | Main single-stream blocks after refiners. |
| `n_refiner_layers` | 2 | Separate noise and context refinement stacks. |
| `n_heads` | 30 | Head dim = 128. |
| `n_kv_heads` | 30 | Constructor records it, but current `Attention` creation does not expose a distinct KV head count. |
| `cap_feat_dim` | 2560 | Qwen3 hidden size projected through RMSNorm + Linear. |
| `axes_dims` | `[32,48,48]` | Sum equals head dim 128 for frame/text, height, width RoPE sections. |
| `axes_lens` | `[1536,512,512]` | Supports caption offsets plus spatial positions. |
| `rope_theta` | 256.0 | Complex RoPE frequencies. |
| `t_scale` | 1000.0 | Pipeline passes `(1000 - t) / 1000`, model multiplies by 1000. |

Text encoder config facts:

| Component | Official value |
| --- | --- |
| `Qwen3Model` / architecture metadata `Qwen3ForCausalLM` | hidden 2560, 36 layers, 32 attention heads, 8 KV heads, head dim 128, intermediate 9728, vocab 151936, max positions 40960. |
| `Qwen2Tokenizer` | `model_max_length=131072`, pad token `<|endoftext|>`, EOS `<|im_end|>`, Qwen image/video special tokens present. |
| Prompt extraction | Pipeline applies Qwen chat template with `enable_thinking=True`, tokenizes to `max_sequence_length=512` by default, runs text encoder with attention mask, then keeps `hidden_states[-2][mask]` as variable-length per-sample tensors. |

AutoencoderKL boundary:

| Field | Official value |
| --- | --- |
| Class | `AutoencoderKL` |
| Latent channels | 16 |
| Block channels | `[128,256,512,512]` |
| Scale factor | Pipeline derives `2 ** (len(block_out_channels)-1) = 8`; image processor uses `vae_scale_factor * 2 = 16`. |
| Scaling / shift | `scaling_factor=0.3611`, `shift_factor=0.1159`; decode uses `latents / scaling_factor + shift_factor`. |
| Quant convs | `use_quant_conv=false`, `use_post_quant_conv=false` |
| Mid attention | `mid_block_add_attention=true` |

Scheduler support:

- Pipeline constructor is specifically `FlowMatchEulerDiscreteScheduler`.
- Official base config uses static `shift=6.0`; Turbo uses static `shift=3.0`.
- Source pipeline still computes `mu` from image sequence length and passes it
  to `set_timesteps`, but sampled official configs have
  `use_dynamic_shifting=false`, so the scheduler ignores `mu` and applies static
  rational shifting.
- Recommended first Dinoml scheduler slice: FlowMatch Euler static shift with
  optional custom sigmas, terminal zero sigma, `stochastic_sampling=false`, and
  model-output sign flip handled at the pipeline boundary.

## 3a. Family variation traps

- Z-Image source latents are NCHW `[B,16,H/8,W/8]`; the transformer converts
  each batch item to C-F-H-W `[16,1,H/8,W/8]` before patchifying. This is not
  QwenImage-style pipeline-level packed tokens.
- The image processor requires generated image H/W to be divisible by
  `vae_scale_factor * 2`, so official 1024px uses latent maps `[B,16,128,128]`
  and patch tokens `(1*64*64)=4096`.
- Caption and image sequences are separately padded to multiples of 32 before
  being concatenated as `[image, caption]` in base mode. Padding tokens and
  boolean masks are part of parity.
- CFG is batch concatenation with positive samples first and negative samples
  second, then `pos + scale * (pos - neg)`. This sign and ordering differ from
  the more common `neg + scale * (pos - neg)` form.
- Pipeline negates the transformer output before the FlowMatch scheduler:
  `noise_pred = -noise_pred`.
- `cfg_truncation` can disable CFG after normalized time exceeds the configured
  cutoff. First slice can set it to the default 1.0 and still model the gate.
- `cfg_normalization` is an optional global vector-norm clamp over each output
  sample, not QwenImage's per-token norm rescale.
- The source declares `n_kv_heads`, but current attention construction uses
  Diffusers `Attention(heads=n_heads)` with separate Q/K/V projections and no
  obvious GQA path. Treat GQA as a config/schema trap, not an active first-slice
  requirement.
- Omni changes sequence order to `[caption, image, siglip]`, uses per-token
  noisy/clean AdaLN, and can include null condition image/siglip placeholders.
- ControlNet may use `control_in_dim=16` for plain control or larger widths such
  as 33 for SAM/inpaint-like control; this belongs in a separate control report.
- NHWC/channel-last is only a guarded optimization for VAE convs and possibly
  image preprocessing. Transformer token and patch flatten order must be
  protected by a `no_layout_translation()` guard unless the flatten/unflatten
  equations are rewritten explicitly.

## 4. Runtime tensor contract

For 1024x1024 official Turbo text-to-image with one image per prompt:

| Boundary | Tensor | Shape/layout | Notes |
| --- | --- | --- | --- |
| Tokenizer output | `input_ids`, `attention_mask` | `[B,512]` default max | CPU/data path plus Qwen chat template. |
| Text hidden | Qwen output | `[B,L,2560]` | Pipeline uses `hidden_states[-2]`. |
| Prompt embeds | list of tensors | each `[L_i,2560]` | Mask-selected, variable length; no batch padding until transformer. |
| Latent noise | `latents` | `[B,16,128,128]` NCHW | Allocated fp32, then cast to transformer dtype for denoiser. |
| Transformer input list | `x` | list of `[16,1,128,128]` C-F-H-W | Created by `latents.unsqueeze(2).unbind(0)`. |
| Patch tokens | internal image patches | `[4096,64]` per item before Linear | `64 = 1*2*2*16`. |
| Image token embed | `x` | `[B,S_img_padded,3840]` | Pad to multiple of 32 and replace pad rows with `x_pad_token`. |
| Caption token embed | `cap_feats` | `[B,S_cap_padded,3840]` | RMSNorm + Linear from 2560; context refiner is unmodulated. |
| Unified sequence | base mode | `[B,S_img+S_cap,3840]` | Order is `[image, caption]`; mask is `[B,S]` or `None`. |
| RoPE frequencies | `freqs_cis` | `[B,S,64]` complex | Sum of half dimensions = head_dim/2 complex pairs. |
| Timestep | pipeline `timestep` | `[B]` | `(1000 - scheduler_t) / 1000`; model embeds `t*1000`. |
| Denoiser output list | list tensors | each `[16,1,128,128]` | After final layer and unpatchify. |
| Scheduler input | `noise_pred` | `[B,16,128,128]` NCHW | Stack outputs, squeeze frame, negate. |
| Scheduler output | `latents` | `[B,16,128,128]` fp32 | `sample + dt * model_output`. |
| VAE decode input | normalized latent | `[B,16,128,128]` NCHW | `latents / 0.3611 + 0.1159`. |
| Decoded image | image tensor | `[B,3,1024,1024]` NCHW | `VaeImageProcessor.postprocess`. |

CPU/data-pipeline work: prompt string templating, tokenization, PIL/NumPy image
conversion, output conversion. GPU/runtime work: optional text encoder, latent
patchify/unpatchify, denoiser, CFG arithmetic, scheduler step, VAE encode/decode.

Cacheable stages: prompt embeds, negative prompt embeds, fixed-resolution RoPE
frequency tables and position IDs, scheduler timestep/sigma tables, VAE
condition latents for img2img/control/omni variants.

## 5. Operator coverage checklist

Tensor/layout ops:

- NCHW latent allocation, cast, repeat for CFG, stack/list conversion.
- C-F-H-W patchify:
  `view(C,F/pF,pF,H/pH,pH,W/pW,pW) -> permute(Ft,Ht,Wt,pF,pH,pW,C) -> reshape(tokens,pF*pH*pW*C)`.
- Unpatchify inverse:
  `view(Ft,Ht,Wt,pF,pH,pW,C) -> permute(C,Ft,pF,Ht,pH,Wt,pW) -> reshape(C,F,H,W)`.
- Variable-length list split/concat, `pad_sequence`, pad-mask construction,
  `torch.where(mask, pad_token, feats)`.
- Attention mask `[B,S] -> [B,1,1,S]`.
- CFG concat/chunk/list handling, sign flip, vector norm clamp.
- Img2img/inpaint/control variants add interpolation, mask blending, VAE encode,
  and control context concatenation.

GEMM/linear ops:

- `all_x_embedder["2-1"]`: Linear(64 -> 3840) official.
- `cap_embedder`: RMSNorm(2560) + Linear(2560 -> 3840).
- Timestep MLP: sinusoidal 256 -> Linear(256 -> 1024) -> SiLU -> Linear(1024 -> 256).
- AdaLN modulation in modulated blocks: Linear(256 -> 15360), split into
  scale/gate for attention and MLP.
- Attention Q/K/V and output projections: 3840-wide, bias-free in the block.
- SwiGLU-style feed-forward: `w1`, `w3` Linear(3840 -> 10240), gated SiLU,
  `w2` Linear(10240 -> 3840), all bias-free.
- Final layer: LayerNorm + SiLU/Linear modulation + Linear(3840 -> 64).

Attention primitives:

- Single-stream noncausal self-attention over unified image+caption sequence.
- Official shape: 30 heads x 128 head dim.
- QK RMSNorm enabled in sampled configs.
- Complex RoPE applied to Q/K before attention.
- Dispatch through `dispatch_attention_fn` with eager/native backend parity and
  optional optimized providers later.

Normalization and adaptive conditioning:

- RMSNorm before attention and FFN, plus RMSNorm on attention/FFN outputs.
- QK RMSNorm inside `Attention` when `qk_norm=true`.
- Final non-affine LayerNorm.
- Tanh gates and `1 + scale` AdaLN modulation from timestep embedding.

Position/timestep/guidance embeddings:

- Sinusoidal timestep embedding with max period 10000.
- Three-axis complex RoPE with position IDs for caption, frame, height, width.
- Optional Omni per-token noisy/clean timestep selection.

Scheduler/VAE/postprocessing ops:

- FlowMatch sigma table generation and `sample + dt * model_output`.
- `scale_noise` for img2img/inpaint initial image latents.
- AutoencoderKL Conv2d/GroupNorm/SiLU/resnet/downsample/upsample/mid-attention
  decode and encode for variants.
- Image postprocess clamp/denormalize/format conversion in `VaeImageProcessor`.

## 6. Denoiser/model breakdown

`ZImageTransformer2DModel.forward`:

```text
x list [C,F,H,W], cap_feats list [L_i,cap_dim], timestep [B]
  -> t_embedder(t * t_scale) -> adaln_input [B,256]
  -> patchify x to 2x2x1 latent patches
  -> pad image patches and caption features to multiple of 32
  -> Linear image patches to dim
  -> image noise_refiner blocks with timestep modulation
  -> RMSNorm+Linear caption projection
  -> caption context_refiner blocks without modulation
  -> concatenate [image, caption]
  -> 30 main ZImageTransformerBlock layers with modulation
  -> FinalLayer with timestep scale
  -> unpatchify to list [C,F,H,W]
```

`ZImageTransformerBlock`:

```text
adaln_input -> Linear -> scale_msa, gate_msa, scale_mlp, gate_mlp
RMSNorm(x) * scale_msa
  -> Q/K/V Linear
  -> QK RMSNorm
  -> complex RoPE
  -> dispatch_attention_fn(noncausal, optional mask)
  -> output Linear
  -> RMSNorm(attn_out), tanh gate, residual add
RMSNorm(x) * scale_mlp
  -> w1/w3 SiLU gate -> w2
  -> RMSNorm(ffn_out), tanh gate, residual add
```

Refiners use the same block class: noise refiners are modulated by timestep,
context refiners are unmodulated caption-only blocks. Main layers are modulated
and can add ControlNet residuals after each layer when provided.

## 7. Attention requirements

Primary implementation is the local `ZSingleStreamAttnProcessor` in
`transformer_z_image.py`, backed by Diffusers `Attention` and
`dispatch_attention_fn` from `attention_dispatch.py`.

Required behavior:

- Noncausal self-attention over unified token sequence.
- Q/K/V input shape after unflatten is `[B,S,heads,head_dim]`.
- Official base/Turbo uses 30 heads, head dim 128.
- QK RMSNorm is active when `qk_norm=true`.
- RoPE uses `torch.view_as_complex` over the last dimension and multiplies by
  precomputed complex frequencies.
- Boolean attention mask is valid-token style `[B,S]`, expanded to
  `[B,1,1,S]`; if all sequence lengths are equal, mask is `None`.

Flash-style/provider notes:

- Eager/native dispatch is the first parity target.
- Flash-style attention can be valid for base text-to-image only if it supports
  head dim 128, noncausal attention, the selected dtype, optional padding mask,
  and Q/K already modified by explicit QK norm + RoPE.
- Padding to multiples of 32 does not remove the need for masks when caption
  lengths differ.
- RoPE and QK RMSNorm should remain explicit ops until a fused provider admits
  exactly this pre-attention math.
- IP-Adapter added-K/V processors are not active in the inspected Z-Image
  family source.

## 8. Scheduler and denoising-loop contract

Base loop:

```text
image_seq_len = (latent_h // 2) * (latent_w // 2)
mu = calculate_shift(image_seq_len, base_image_seq_len, max_image_seq_len,
                     base_shift, max_shift)
scheduler.sigma_min = 0.0
scheduler.set_timesteps(num_inference_steps, sigmas=optional_sigmas, mu=mu)
scheduler.set_begin_index(0)
for t in timesteps:
  timestep = (1000 - t) / 1000
  optional CFG batch concat
  denoiser(list(latents.unsqueeze(2)), timestep, prompt_embeds)
  noise_pred = -stack(outputs).squeeze(2)
  latents = scheduler.step(noise_pred, t, latents)
```

`FlowMatchEulerDiscreteScheduler` details:

- Static official configs use `shift * sigma / (1 + (shift - 1) * sigma)`.
- If dynamic shifting is enabled by a future config, `mu` becomes required and
  `time_shift_type` may be exponential or linear.
- Scheduler appends terminal sigma zero when `invert_sigmas=false`.
- Non-stochastic step is `prev_sample = sample + (sigma_next - sigma) * model_output`.
- Step upcasts `sample` to fp32 and casts result back to model output dtype for
  scalar timesteps.
- Advanced scheduler branches such as stochastic sampling, Karras/exponential/
  beta sigma conversion, dynamic shifting, and per-token timesteps should be
  separate follow-up slices.

CFG behavior:

- `guidance_scale > 0` enables CFG in this pipeline, not `> 1`.
- Positive samples are first, negative samples second.
- Combined prediction uses `pos + scale * (pos - neg)`.
- Optional `cfg_normalization` clamps the vector norm of each sample to
  `norm(pos) * cfg_normalization`.
- Optional `cfg_truncation <= 1` disables CFG after the normalized timestep
  exceeds the threshold.

Keep schedule creation, iteration, CFG truncation gates, and callback mutation
host-visible first. Compile denoiser step, CFG arithmetic, sign flip, and
pointwise scheduler step after component parity is stable.

## 9. Position, timestep, and custom math

Custom math to reproduce:

- `calculate_shift`: linear interpolation between base and max shifts from
  image sequence length. Official static configs pass but ignore `mu`.
- Timestep inversion at the pipeline boundary:
  `model_t = (1000 - scheduler_t) / 1000`, then model embeds `model_t * 1000`.
- Patch position IDs:
  caption starts at `(1,0,0)` and image starts at `(cap_len + 1,0,0)` in base
  mode, so caption length changes image frame-axis RoPE offsets.
- Complex RoPE over `[frame/text, height, width]` axes.
- Padding to multiple of 32 with learned pad tokens and valid-token masks.
- Transformer output sign flip before scheduler step.
- VAE latent decode affine:
  `decode_latents = latents / scaling_factor + shift_factor`.

Precomputable: scheduler tables for fixed steps/sigmas, RoPE frequency tables,
position IDs for fixed prompt length and resolution, prompt embeddings, VAE
condition latents.

Dynamic: prompt length, image resolution, CFG enabled/disabled/truncated,
guidance scale, mask/condition images, Omni condition count, and custom sigmas.

## 10. Preprocessing and input packing

Text preprocessing:

- Base pipeline uses `tokenizer.apply_chat_template(..., add_generation_prompt=True, enable_thinking=True)`.
- Tokenizer pads/truncates to `max_sequence_length=512`.
- Text encoder receives attention mask and returns all hidden states; pipeline
  uses the penultimate hidden state.
- Prompt embeddings are stored as a Python list of valid-token tensors, not a
  padded batch tensor.
- Negative prompts follow the same path only when CFG is active.

Latent preprocessing:

- Text-to-image starts from Gaussian NCHW latents.
- Transformer receives per-batch tensors with a singleton frame axis.
- Internal patchify is model-coupled and must be represented in the denoiser
  graph, not as pipeline preprocessing.

Variant preprocessing:

- Img2img encodes preprocessed image through AutoencoderKL, standardizes
  `(latent - shift_factor) * scaling_factor`, then applies scheduler
  `scale_noise` at the strength-selected timestep.
- Inpaint resizes masks to latent dimensions, encodes masked image latents, and
  blends scheduler latents with original noised latents outside the denoiser.
- ControlNet encodes control images through VAE; if `control_in_dim` exceeds
  latent channels, it right-pads channel dimensions.
- ControlNet-inpaint concatenates control image latents, a resized mask, and
  masked-image latents for the control context.
- Omni prepares condition VAE latents and SigLIP2 image hidden states, then
  appends the noisy target image as the last image in the unified sequence.

## 11. Graph rewrite / lowering opportunities

### Rewrite: Z-Image patchify/unpatchify

Source pattern: C-F-H-W tensor list to 2x2x1 patches, linear projection, final
linear, and inverse unpatchify.

Replacement: explicit `z_image_patchify2d` / `z_image_unpatchify2d` ops or
canonical reshape/permute/reshape sequences.

Preconditions: source layout C-F-H-W, `F=1` for base image path,
`patch_size=2`, `f_patch_size=1`, H/W divisible by 2, token flatten order
`F,H,W,pF,pH,pW,C`.

Failure cases: Omni multi-image extraction offsets, non-2 patch configs,
NHWC layout rewrite without matching flatten-order transform.

Parity test: random `[16,1,128,128]` patchify/unpatchify round trip and final
projection shape `[4096,64]`.

### Rewrite: unified single-stream attention

Source pattern: separate image and caption refinement, concatenate tokens, QKV,
QK RMSNorm, complex RoPE, SDPA/dispatch attention, output projection.

Replacement: a `single_stream_rope_attention` graph region with provider
fallback; later fuse QKV + QK norm + RoPE + attention under strict guards.

Preconditions: no ControlNet residual inside attention op, known sequence order,
provider supports mask and head dim 128, RoPE already applied or fused exactly.

Failure cases: Omni per-token modulation, ControlNet residual scheduling,
future adapter processors, mismatched mask semantics.

Parity test: one `ZSingleStreamAttnProcessor` call with and without padding
mask, fp32 and bf16.

### Rewrite: FlowMatch static-shift Z-Image slice

Source pattern: static sigma shift, terminal zero sigma, sign-flipped model
output, Euler update.

Replacement: host-visible sigma table plus fused `sample + dt * (-model_out)`
pointwise step.

Preconditions: `use_dynamic_shifting=false`, `stochastic_sampling=false`,
scalar timestep, no per-token timesteps, explicit `sigma_idx`.

Failure cases: custom dynamic-shift config, stochastic branch, Karras/beta
sigma conversions, img2img begin-index mismatch.

### Rewrite: AutoencoderKL decode island

Source pattern: `latents / scaling_factor + shift_factor` then AutoencoderKL
decode with Conv2d/GroupNorm/SiLU/resnet/upsample/mid-attention.

Replacement: separate AutoencoderKL codec artifact first; later fuse Conv/Norm/
SiLU and upsample regions under NCHW or guarded NHWC.

Preconditions: `use_post_quant_conv=false`, `latent_channels=16`, tiling/slicing
disabled, source NCHW preserved at boundary.

Failure cases: variant encode path, VAE tiling/slicing, different quant conv
config, NHWC axis rewrites missing for GroupNorm.

## 12. Kernel fusion candidates

Highest priority:

- Single-stream attention hot path: QKV, QK RMSNorm, RoPE, attention, output
  projection over roughly `4096 + prompt_tokens` sequence length.
- AdaLN scale/gate/residual epilogues around attention and FFN.
- SwiGLU feed-forward fusion for 3840 -> 10240 -> 3840.
- Patchify/unpatchify kernels and pad/mask construction for fixed resolutions.
- FlowMatch step plus sign flip and CFG arithmetic.

Medium priority:

- Timestep embedding and AdaLN modulation MLP.
- RoPE frequency and position-ID caching.
- AutoencoderKL decode Conv2d/GroupNorm/SiLU/upblock/mid-attention kernels.
- Img2img/inpaint VAE encode and `scale_noise`.
- ControlNet residual add and control context projection.

Lower priority:

- LoRA load/fuse/hotswap and original checkpoint conversion.
- Omni SigLIP2 image encoder and per-token noisy/clean modulation.
- VAE tiling/slicing and output postprocess format conversions.
- Full Qwen3 text encoder compilation; prompt embeds can be supplied externally
  for first denoiser admission.

NHWC guarded notes:

- Transformer core is token-major and patch-flatten order sensitive; do not
  apply blanket NHWC translation.
- VAE Conv2d/GroupNorm/upsample regions are candidates for guarded NHWC only
  inside a fully controlled codec island.
- Required axis rewrites for NHWC include GroupNorm channel axis, Conv2d weight
  transforms, interpolate H/W axes, mask interpolation axes, and VAE latent
  scaling broadcasts.
- Protect token sequence, RoPE IDs, scheduler broadcasting, and patchify/
  unpatchify with a conceptual `no_layout_translation()` guard until an exact
  layout-aware rewrite exists.

## 13. Runtime staging plan

Stage 1: Parse model index and component configs for `Tongyi-MAI/Z-Image-Turbo`
and a tiny random repo. Accept external Qwen3 prompt embeddings as lists.

Stage 2: Implement patchify/unpatchify, sequence padding, pad tokens, attention
masks, and RoPE position IDs for a fixed 1024px latent.

Stage 3: Implement one `ZImageTransformerBlock`, including RMSNorm, QK norm,
complex RoPE attention, AdaLN gates, and SwiGLU FFN.

Stage 4: Compile full tiny random `ZImageTransformer2DModel`, then official
Turbo one-step denoiser with fixed prompt embeddings.

Stage 5: Add FlowMatch Euler static-shift scheduler table and one-step parity,
including transformer output sign flip.

Stage 6: Add CFG batch concatenation, CFG truncation gate, and optional
`cfg_normalization` clamp.

Stage 7: Add AutoencoderKL decode as a separate codec boundary or call into a
shared AutoencoderKL artifact.

Stage 8: Open separate slices for img2img, inpaint, ControlNet, Omni, LoRA, and
single-file conversion.

First Dinoml admission recommendation: admit `z_image_turbo_denoiser_step`.
Inputs should be NCHW latents `[B,16,H/8,W/8]`, caption embedding lists or a
padded-plus-length equivalent with hidden 2560, scalar/batch model timestep, and
optional attention mask metadata. Output should be NCHW model derivative before
pipeline sign flip, or explicitly name the sign-flipped scheduler input in the
artifact. Keep Qwen3 text encoding, VAE decode, CFG orchestration, and scheduler
loop state outside the compiled artifact until denoiser parity is proven.

## 14. Parity and validation plan

- Config parse tests for official base, Turbo, tiny random, and ControlNet
  reference configs.
- Prompt embed extraction parity for Qwen chat template, attention-mask
  selection, and hidden state `[-2]`.
- Patchify/unpatchify random tensor parity for `[16,1,128,128]`.
- Position ID and RoPE frequency parity for prompt lengths 1, 77, 512 and
  image token grid `(1,64,64)`.
- `ZSingleStreamAttnProcessor` parity with no mask and with variable caption
  lengths.
- One `ZImageTransformerBlock` parity, with and without modulation.
- Noise/context refiner block parity.
- Full tiny random transformer parity.
- Official Turbo one denoiser-step parity with fixed prompt embeddings and
  latents.
- FlowMatch scheduler table and one-step parity for shifts 3.0 and 6.0.
- CFG arithmetic, sign flip, truncation gate, and norm-clamp parity.
- AutoencoderKL decode parity for `[1,16,128,128]`; encode parity for img2img
  and control variants later.
- End-to-end smoke only after text encoder, denoiser, scheduler, and VAE stages
  each have local parity.
- Suggested tolerances: scheduler fp32 `rtol=1e-5, atol=1e-6`; transformer fp32
  `rtol=1e-4, atol=1e-5`; bf16/fp16 initially `rtol=2e-2, atol=2e-2`.

## 15. Performance probes

- Denoiser step latency by image token count: 1024, 4096, 8192, 16384 plus
  prompt length sweep.
- Attention backend comparison: native SDPA/eager dispatch versus guarded
  flash-style provider with and without masks.
- Per-block time split: QKV/QK norm/RoPE, attention, output projection, FFN,
  AdaLN/gate epilogues.
- Patchify/unpatchify and padding overhead.
- CFG overhead: one denoiser call versus doubled batch plus CFG pointwise math.
- FlowMatch scheduler and CFG overhead relative to denoiser.
- AutoencoderKL decode throughput for 1024px and higher resolutions.
- Qwen3 text encoder throughput only after denoiser-first admission.
- VRAM/workspace by dtype and prompt length; attention sequence length is the
  primary pressure point.

## 16. Scope boundary and separate candidates

Separate candidate reports, not ignored:

- `z_image_img2img`: VAE encode, strength timestep slicing, `scale_noise`.
- `z_image_inpaint`: mask preprocessing, masked-image latents, latent blending.
- `z_image_omni`: SigLIP2 conditioning, condition-image VAE latents, sequence
  order change, per-token noisy/clean modulation.
- `z_image_controlnet`: `ZImageControlNetModel`, control context patching, and
  per-layer residuals.
- `z_image_controlnet_inpaint`: control image plus mask plus masked-image
  latent context; `control_in_dim` variation.
- `z_image_lora_adapters`: transformer LoRA load/fuse/unfuse/hotswap and
  non-Diffusers LoRA conversion.
- `z_image_single_file`: fused QKV and original checkpoint key conversion.
- `z_image_modular_pipeline`: explicit modular pipeline blocks as a future
  state-modeling reference.
- Rare FlowMatch options beyond sampled configs: dynamic shift, stochastic
  sampling, per-token timesteps, Karras/exponential/beta sigma conversion.

Genuinely ignored/out of scope for this audit:

- Multi-GPU/context parallel paths.
- Callback mutation and interactive interrupt.
- XLA, NPU, MPS, Flax, and ONNX variants.
- Safety checker and NSFW filtering.
- Training, losses, dropout, and gradient checkpointing.
- IP-Adapter, T2I-Adapter, GLIGEN, depth2img, and upscaling because no active
  Z-Image family implementation was found in the inspected Diffusers folder.

## 17. Final implementation checklist

- [ ] Parse Z-Image model index and component configs.
- [ ] Load `ZImageTransformer2DModel` weights for tiny random and Turbo.
- [ ] Accept external Qwen3 prompt embedding lists or padded tensors plus lengths.
- [ ] Implement C-F-H-W patchify/unpatchify for patch 2, frame patch 1.
- [ ] Implement sequence padding, learned pad tokens, and valid-token masks.
- [ ] Implement Z-Image RoPE position IDs and complex RoPE application.
- [ ] Implement timestep embedding and AdaLN modulation.
- [ ] Implement `ZSingleStreamAttnProcessor` fallback parity.
- [ ] Implement QK RMSNorm + RoPE + noncausal attention.
- [ ] Implement `ZImageTransformerBlock` attention/FFN/gate/residual path.
- [ ] Add full tiny random transformer parity.
- [ ] Add official Turbo one-step denoiser parity.
- [ ] Implement FlowMatch Euler static-shift scheduler slice.
- [ ] Add output sign flip, CFG concat/arithmetic, truncation, and norm-clamp parity.
- [ ] Add AutoencoderKL decode boundary or separate codec artifact.
- [ ] Benchmark attention, denoiser step, and VAE decode at 1024px.
- [ ] Open separate reports for img2img, inpaint, Omni, ControlNet, LoRA, and single-file conversion.
