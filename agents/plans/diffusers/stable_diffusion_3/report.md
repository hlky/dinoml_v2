# Diffusers Stable Diffusion 3 Operator and Integration Report

Target slug: `stable_diffusion_3`

Status: prompt calibration report.

## Scope

This report covers the main SD3 text-to-image path in Diffusers, using `StableDiffusion3Pipeline` with `SD3Transformer2DModel`, `AutoencoderKL`, and `FlowMatchEulerDiscreteScheduler`. It treats img2img, inpaint, ControlNet, IP-Adapter, LoRA/runtime adapters, and SD3.5 variants as separate candidate reports with class/file inventory below.

Ignored per project scope: XLA/NPU/MPS branches, callback mutation/interrupt paths, multi-GPU/context-parallel paths, Flax/ONNX variants, safety/NSFW filtering, training, losses, dropout, and gradient checkpointing behavior.

## Source Files

- `X:/H/diffusers/src/diffusers/pipelines/stable_diffusion_3/pipeline_stable_diffusion_3.py`
- `X:/H/diffusers/src/diffusers/models/transformers/transformer_sd3.py`
- `X:/H/diffusers/src/diffusers/models/attention.py`
- `X:/H/diffusers/src/diffusers/models/attention_processor.py`
- Variant inventory:
  - `X:/H/diffusers/src/diffusers/pipelines/stable_diffusion_3/pipeline_stable_diffusion_3_img2img.py`
  - `X:/H/diffusers/src/diffusers/pipelines/stable_diffusion_3/pipeline_stable_diffusion_3_inpaint.py`
  - `X:/H/diffusers/src/diffusers/pipelines/controlnet_sd3/pipeline_stable_diffusion_3_controlnet.py`
  - `X:/H/diffusers/src/diffusers/pipelines/controlnet_sd3/pipeline_stable_diffusion_3_controlnet_inpainting.py`
  - `X:/H/diffusers/src/diffusers/models/controlnets/controlnet_sd3.py`

## Config Evidence

Primary worked examples were fetched with authenticated `hf` CLI and saved under `H:/configs/stabilityai/`:

- `stable-diffusion-3-medium-diffusers`
- `stable-diffusion-3.5-large`
- `stable-diffusion-3.5-medium`
- `stable-diffusion-3.5-large-turbo`

`model_index.json` declares:

- `StableDiffusion3Pipeline`
- `FlowMatchEulerDiscreteScheduler`
- `CLIPTextModelWithProjection`, `CLIPTextModelWithProjection`, `T5EncoderModel`
- `CLIPTokenizer`, `CLIPTokenizer`, `T5TokenizerFast`
- `SD3Transformer2DModel`
- `AutoencoderKL`

SD3 medium transformer config:

- `sample_size`: 128
- `patch_size`: 2
- `in_channels`: 16
- `out_channels`: 16
- `num_layers`: 24
- `num_attention_heads`: 24
- `attention_head_dim`: 64
- `joint_attention_dim`: 4096
- `caption_projection_dim`: 1536
- `pooled_projection_dim`: 2048
- `pos_embed_max_size`: 192

SD3.5 large and SD3.5 large turbo transformer config:

- `sample_size`: 128
- `patch_size`: 2
- `in_channels`: 16
- `out_channels`: 16
- `num_layers`: 38
- `num_attention_heads`: 38
- `attention_head_dim`: 64
- `joint_attention_dim`: 4096
- `caption_projection_dim`: 2432
- `pooled_projection_dim`: 2048
- `pos_embed_max_size`: 192
- `qk_norm`: `rms_norm`

SD3.5 medium transformer config:

- `sample_size`: 128
- `patch_size`: 2
- `in_channels`: 16
- `out_channels`: 16
- `num_layers`: 24
- `num_attention_heads`: 24
- `attention_head_dim`: 64
- `joint_attention_dim`: 4096
- `caption_projection_dim`: 1536
- `pooled_projection_dim`: 2048
- `pos_embed_max_size`: 384
- `qk_norm`: `rms_norm`
- `dual_attention_layers`: 0 through 12

SD3 medium VAE config:

- `latent_channels`: 16
- `sample_size`: 1024
- `scaling_factor`: 1.5305
- `shift_factor`: 0.0609
- `force_upcast`: true
- `use_quant_conv`: false
- `use_post_quant_conv`: false

SD3.5 large, medium, and large turbo use the same 16-channel VAE scaling contract as SD3 medium, with `scaling_factor=1.5305`, `shift_factor=0.0609`, `force_upcast=true`, and no quant/post-quant convs. Their VAE configs additionally record `mid_block_add_attention=true`.

Scheduler config:

- `FlowMatchEulerDiscreteScheduler`
- `num_train_timesteps`: 1000
- `shift`: 3.0

SD3.5 top-level pipeline class remains `StableDiffusion3Pipeline` with `FlowMatchEulerDiscreteScheduler`, three text encoders/tokenizers, `SD3Transformer2DModel`, and `AutoencoderKL`. Local `_amdgpu` cache folders still contain empty `{}` model indexes, so prefer the official fetched configs above.

## Pipeline Contract

Main inputs include:

- `prompt`, `prompt_2`, `prompt_3`
- negative prompt counterparts for CFG
- precomputed `prompt_embeds`, `negative_prompt_embeds`, `pooled_prompt_embeds`, `negative_pooled_prompt_embeds`
- `height`, `width`, `num_inference_steps`, `timesteps`, `sigmas`
- `guidance_scale`
- `clip_skip`
- `max_sequence_length`, default 256 for T5
- `skip_guidance_layers`, `skip_layer_guidance_scale`, `skip_layer_guidance_start`, `skip_layer_guidance_stop`
- IP-Adapter image/image embeds, as a separate candidate surface

The pipeline default image size is derived from `transformer.config.sample_size * vae_scale_factor`; SD3 medium is 128 latent pixels at VAE scale factor 8, so the native image size is 1024.

Output is either latent tensor or decoded image. The decode path applies:

```text
vae_input = (latents / vae.config.scaling_factor) + vae.config.shift_factor
image = vae.decode(vae_input)
```

## Text Conditioning

SD3 uses three tokenizer/encoder pairs:

- CLIP path 1: `CLIPTokenizer` + `CLIPTextModelWithProjection`
- CLIP path 2: `CLIPTokenizer` + `CLIPTextModelWithProjection`
- T5 path: `T5TokenizerFast` + `T5EncoderModel`

For CLIP, Diffusers takes the pooled projection from the encoder output and hidden states from either `hidden_states[-2]` or `hidden_states[-(clip_skip + 2)]`. The two CLIP sequence embeddings are concatenated on the feature axis.

For T5, Diffusers emits sequence embeddings at `joint_attention_dim` width. If `text_encoder_3` is absent, the pipeline creates zeros shaped:

```text
[batch * num_images_per_prompt, max_sequence_length, transformer.config.joint_attention_dim]
```

The CLIP sequence embedding is padded on the feature axis to match T5 width, then concatenated with T5 on the sequence axis:

```text
clip_prompt_embeds = cat([clip1_hidden, clip2_hidden], dim=-1)
clip_prompt_embeds = pad_to_width(clip_prompt_embeds, t5_width)
prompt_embeds = cat([clip_prompt_embeds, t5_prompt_embeds], dim=-2)
pooled_prompt_embeds = cat([clip1_pooled, clip2_pooled], dim=-1)
```

For CFG, negative embeddings are produced the same way and concatenated with positive embeddings before the denoising loop.

Optimization candidates:

- The CLIP concat, pad, and T5 sequence concat are shape-static for a given prompt batch and maximum sequence length.
- Dinoml should preserve this as explicit conditioning construction rather than folding it into the transformer contract.
- The optional T5 zero-fill behavior is a compatibility branch and must be represented in candidate reports when a variant omits `text_encoder_3`.

## Latent and Main Model Shapes

The denoiser input is NCHW latent image data:

```text
latents: [B, 16, H/8, W/8]
```

`SD3Transformer2DModel` patchifies internally using `PatchEmbed` with `patch_size=2`. For a 1024 x 1024 image, the latent map is 128 x 128 and the transformer token length is:

```text
(128 / 2) * (128 / 2) = 4096 image tokens
```

The model sequence width is:

```text
inner_dim = num_attention_heads * attention_head_dim
          = 24 * 64
          = 1536
```

At output, `proj_out` returns per-token `patch_size * patch_size * out_channels` values. Diffusers unpatchifies with:

```text
[B, H_tokens, W_tokens, patch, patch, C]
einsum("nhwpqc->nchpwq")
[B, C, H_tokens * patch, W_tokens * patch]
```

This is a source layout contract. NHWC/channel-last optimization may be useful around VAE/convolutional islands, but SD3 transformer entry/exit is NCHW plus explicit patchify/unpatchify. Any NHWC pass must preserve the patch axes and the exact unpatchify permutation.

## Denoising and Scheduler

The main scheduler is `FlowMatchEulerDiscreteScheduler`. SD3 is not a broad scheduler-swap family like SD1.x; first-slice integration should treat the FlowMatch scheduler contract as part of the family report.

If `scheduler.config.use_dynamic_shifting` is set and `mu` is not passed, the pipeline computes `mu` from image sequence length using `calculate_shift`, with config keys:

- `base_image_seq_len`
- `max_image_seq_len`
- `base_shift`
- `max_shift`

SD3 medium config only records `shift=3.0`, so the worked example is static-shift FlowMatch.

The denoising loop uses true classifier-free guidance:

```text
latent_model_input = cat([latents, latents], batch) when guidance_scale > 1
noise_pred = transformer(...)
noise_pred_uncond, noise_pred_text = chunk(noise_pred, 2)
noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
latents = scheduler.step(noise_pred, t, latents)
```

This differs from Flux schnell/dev’s embedded guidance tensor path. SD3’s guidance requires the extra unconditional/conditional batch and arithmetic.

Skip-layer guidance is another optional source-visible branch: when `skip_guidance_layers` is provided within the configured step window, the pipeline runs an additional transformer pass with `skip_layers` and adjusts:

```text
noise_pred += (noise_pred_text - noise_pred_skip_layers) * skip_layer_guidance_scale
```

The first Dinoml slice can leave skip-layer guidance as a separately guarded feature, but reports should document it because it changes denoiser call count and block execution.

## Transformer Internals

`SD3Transformer2DModel` components:

- `PatchEmbed` for NCHW latent image to token sequence plus 2D positional embedding
- `CombinedTimestepTextProjEmbeddings` for timestep plus pooled CLIP projections
- `context_embedder: Linear(joint_attention_dim -> caption_projection_dim)`
- `num_layers` `JointTransformerBlock`
- `AdaLayerNormContinuous` output norm
- `proj_out: Linear(inner_dim -> patch_size * patch_size * out_channels)`

`JointTransformerBlock` uses joint sample/context attention. The last block has `context_pre_only=True`. SD3.0 default has no dual attention layers. SD3.5 medium config sets `dual_attention_layers` for layers 0-12, which enables a second `Attention` module inside those blocks. SD3.5 large and large turbo configs do not list `dual_attention_layers`, but widen the model to 38 layers and 38 heads and use RMS QK norm.

Important operators:

- Conv/linear patch embedding
- 2D positional embedding add
- timestep/text projection MLPs
- context projection linear
- AdaLayerNormZero / AdaLayerNormContinuous modulation
- joint Q/K/V projections for image and text
- optional QK normalization
- concat along sequence dimension for joint attention
- scaled dot-product attention
- split image/text attention outputs
- residual adds
- feed-forward blocks
- final projection and unpatchify reshape/einsum

## Attention and Flash-style Candidates

Default SD3-like attention processor is `JointAttnProcessor2_0`, which uses PyTorch `F.scaled_dot_product_attention`. It projects image tokens with `to_q/to_k/to_v` and context tokens with `add_q_proj/add_k_proj/add_v_proj`, reshapes to:

```text
[B, heads, image_seq, head_dim]
[B, heads, text_seq, head_dim]
```

Then it concatenates image and text along sequence:

```text
query = cat([image_query, text_query], dim=2)
key   = cat([image_key, text_key], dim=2)
value = cat([image_value, text_value], dim=2)
```

After attention, the result is split back into image and text spans. When `context_pre_only` is false, the text span also goes through `to_add_out`.

Existing processors include:

- `JointAttnProcessor2_0`
- `FusedJointAttnProcessor2_0`
- `XFormersJointAttnProcessor`
- `SD3IPAdapterJointAttnProcessor2_0`

The model exposes experimental `fuse_qkv_projections()`, switching attention processors to `FusedJointAttnProcessor2_0`.

Flash-style Dinoml provider candidates:

- Base SD3 attention is mask-free and non-causal, so a flash-style provider may be valid when dtype/head_dim/backend constraints are satisfied.
- QK normalization must be kept before attention.
- The provider must support the concatenated joint image+text sequence and the later split semantics.
- IP-Adapter changes attention inputs and should be a separate candidate.
- Fused QKV is an existing Diffusers mutation path but should not be assumed active by default.

## Autoencoder Contract

Main text-to-image only decodes generated latents. The SD3 family still requires encode support for img2img, inpaint, and ControlNet inpainting variants.

For SD3 medium:

```text
decode_input = (latents / 1.5305) + 0.0609
```

Encode variants use the inverse:

```text
latents = (vae.encode(image_latents) - shift_factor) * scaling_factor
```

The VAE is a 16-channel latent VAE with no quant/post-quant convs. Treat AutoencoderKL as its own report because VAE channel count, scaling/shift, force-upcast, tiling/slicing, and NHWC convolution opportunities are independent of the transformer family.

## Variant and Extension Candidates

These are separate review candidates, not skipped surfaces:

| Candidate | Classes/files | Why separate |
| --- | --- | --- |
| `sd3_img2img` | `StableDiffusion3Img2ImgPipeline`, `pipeline_stable_diffusion_3_img2img.py` | Adds VAE encode, strength-based timestep slicing, and latent/image input normalization. |
| `sd3_inpaint` | `StableDiffusion3InpaintPipeline`, `pipeline_stable_diffusion_3_inpaint.py` | Adds mask preprocessing, masked image VAE encode, and transformer input channel modes. |
| `sd3_inpaint_33_channel` | Same inpaint pipeline plus transformer config with `in_channels=33` | Concatenates `[latent_model_input, mask, masked_image_latents]` as 16 + 1 + 16 channels. |
| `sd3_controlnet` | `StableDiffusion3ControlNetPipeline`, `SD3ControlNetModel`, `SD3MultiControlNetModel` | Adds control image preprocessing, controlnet forward pass, conditioning scales, and block residual injection. |
| `sd3_controlnet_inpaint` | `StableDiffusion3ControlNetInpaintingPipeline` | Combines ControlNet with mask/inpaint latent contracts. |
| `sd3_ip_adapter` | `SD3IPAdapterMixin`, `SD3IPAdapterJointAttnProcessor2_0` | Mutates attention inputs with image features and IP-specific K/V attention. |
| `sd3_lora_runtime_adapters` | `SD3LoraLoaderMixin`, PEFT adapter paths, `apply_lora_scale` | Runtime adapter mutation changes text encoders and transformer linear layers. |
| `sd3_skip_layer_guidance` | Base pipeline and transformer `skip_layers` | Adds extra denoiser call and block-level skipping in selected steps. |
| `sd3_5_main_models` | `StableDiffusion3Pipeline`, `SD3Transformer2DModel`; configs `stable-diffusion-3.5-large`, `stable-diffusion-3.5-medium`, `stable-diffusion-3.5-large-turbo` | Same pipeline family, but large/turbo change layer/head count, medium enables dual-attention layers, and all fetched 3.5 configs use RMS QK norm. |
| `sd3_5_controlnet` | Local cache names: `stable-diffusion-3.5-large-controlnet-{blur,canny,depth}` | Requires gated/full configs; likely separate control image and residual contracts. |

## Dinoml Integration Notes

First-slice target:

- SD3 medium text-to-image.
- `StableDiffusion3Pipeline`.
- `SD3Transformer2DModel`.
- `FlowMatchEulerDiscreteScheduler`.
- 16-channel SD3 `AutoencoderKL` decode.
- True CFG path, but skip-layer guidance off unless specifically enabled.
- No IP-Adapter/ControlNet/inpaint/img2img in first slice.

High-value op groups:

- Linear/GEMM-heavy transformer blocks.
- Joint attention projection + attention + output projection.
- AdaLayerNorm modulation and residual patterns.
- Patch embed/unpatchify reshape/einsum.
- Timestep/pooled text embedding MLPs.
- VAE convolutional decode path.

Fusion candidates:

- Linear + bias in Q/K/V projections and feed-forward projections.
- QK normalization + attention prelude as a guarded pattern.
- Attention softmax/value matmul via flash-style provider when sequence/head/dtype constraints pass.
- AdaLayerNorm modulation plus residual block epilogues where artifact-visible state remains clear.
- Patchify/unpatchify as explicit layout transforms, not hidden mutable compiler state.
- VAE conv/norm/activation islands, with NHWC guarded by provider/layout support.

Risks:

- Treating SD3 guidance like Flux embedded guidance would be wrong; SD3 uses true CFG by default.
- Treating SD3 patching like Flux latent packing would be wrong; SD3 patchifies inside the transformer with `PatchEmbed`.
- Inpaint can change transformer input channels from 16 to 33.
- Optional missing T5 encoder changes conditioning construction but not transformer `joint_attention_dim`.
- SD3.5 cannot be inferred from empty local AMDGPU model indexes; use the fetched official large/medium/turbo configs.

## Prompt Lessons

The prompt should explicitly require:

- Multi-encoder conditioning composition, including feature-axis concat, padding, sequence-axis concat, and pooled projection concat.
- Optional encoder absence behavior, such as SD3 zero-filled T5 embeddings.
- Patchify/unpatchify contracts as distinct from Flux-style latent packing.
- A family-specific guidance classification: embedded guidance tensor versus true CFG versus skip-layer/extra-pass guidance.
- Variant channel-contract inventory, especially inpaint transformers whose `in_channels` differ from the base model.
- Gated config attempts plus local cache limitations, without inferring full variant behavior from empty or partial configs.
- Same-family config comparison, because SD3.5 large/turbo and medium share the top-level pipeline class but differ in depth, width, QK normalization, and dual-attention layer use.
