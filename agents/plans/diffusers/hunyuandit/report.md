# Diffusers HunyuanDiT Base Operator and Integration Report

Target slug: `hunyuandit`

Runtime scope: non-deprecated base text-to-image `HunyuanDiTPipeline`, separate
from Hunyuan Image 2.1, HunyuanDiT ControlNet, and PAG variants. First Dinoml
slice should accept externally supplied Bert/T5 prompt embeddings and masks,
run the base `HunyuanDiT2DModel` denoiser on NCHW latents, keep DDPM scheduler
state host-visible, and decode with AutoencoderKL.

Ignored per task: XLA/NPU/MPS/Flax/ONNX, safety/NSFW behavior, training/losses,
dropout, gradient checkpointing, callbacks/interrupt mutation, and
multi-GPU/context-parallel paths.

## 1. Source basis

```text
Diffusers commit/version:
  X:/H/diffusers at b3a515080752a3ba7ca92161e25530c7f280f629.

Model id(s):
  Tencent-Hunyuan/HunyuanDiT-Diffusers
  Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers
  Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled
  Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers
  Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers-Distilled

Config sources:
  Local cache:
    H:/configs/Tencent-Hunyuan/HunyuanDiT-Diffusers/model_index.json
    H:/configs/Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled/model_index.json
    H:/configs/Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers/model_index.json
    H:/configs/Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers/transformer/config.json
    H:/configs/Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers/vae/config.json
    H:/configs/Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers/scheduler/scheduler_config.json
    H:/configs/Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers/text_encoder/config.json
    H:/configs/Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers/text_encoder_2/config.json
    H:/configs/Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers/tokenizer/tokenizer_config.json
    H:/configs/Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers/tokenizer_2/tokenizer_config.json
    H:/configs/Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers-Distilled/model_index.json
  Raw official HF URLs inspected, not saved because this worker owns only this report path:
    https://huggingface.co/Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers/raw/main/{model_index.json,transformer/config.json,scheduler/scheduler_config.json,vae/config.json}
    https://huggingface.co/Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers-Distilled/raw/main/{model_index.json,transformer/config.json,scheduler/scheduler_config.json}
    https://huggingface.co/Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled/raw/main/model_index.json

Pipeline files inspected:
  X:/H/diffusers/src/diffusers/pipelines/hunyuandit/pipeline_hunyuandit.py
  X:/H/diffusers/src/diffusers/pipelines/controlnet_hunyuandit/pipeline_hunyuandit_controlnet.py
  X:/H/diffusers/src/diffusers/pipelines/pag/pipeline_pag_hunyuandit.py

Model files inspected:
  X:/H/diffusers/src/diffusers/models/transformers/hunyuan_transformer_2d.py
  X:/H/diffusers/src/diffusers/models/controlnets/controlnet_hunyuan.py
  X:/H/diffusers/src/diffusers/models/autoencoders/autoencoder_kl.py

Scheduler/processors/helpers inspected:
  X:/H/diffusers/src/diffusers/schedulers/scheduling_ddpm.py
  X:/H/diffusers/src/diffusers/image_processor.py
  X:/H/diffusers/src/diffusers/models/attention.py
  X:/H/diffusers/src/diffusers/models/attention_processor.py
  X:/H/diffusers/src/diffusers/models/attention_dispatch.py
  X:/H/diffusers/src/diffusers/models/embeddings.py
  X:/H/diffusers/src/diffusers/models/normalization.py

External component configs inspected:
  BertModel / BertTokenizer for bilingual Hunyuan text encoder.
  T5EncoderModel / T5Tokenizer for mT5 prompt encoder.

Any missing files or assumptions:
  No gated blocker was hit for the inspected JSON files. v1.1 component configs
  that were absent locally were accessible through unauthenticated official raw
  URLs, so authenticated retry was not needed. The distilled repos were checked
  for model index and, for v1.2 distilled, transformer/scheduler config; no
  operator-structural delta from v1.2 base was visible in those inspected JSONs.
```

## 2. Pipeline and component graph

`HunyuanDiTPipeline` registers `vae`, `text_encoder`, `tokenizer`,
`text_encoder_2`, `tokenizer_2`, `transformer`, `scheduler`, and optional
safety components. The declared offload order is
`text_encoder->text_encoder_2->transformer->vae`. Optional components include
both text encoders/tokenizers and safety components, which enables prompt
embedding inputs without running tokenizers or encoders in the pipeline.

```text
prompt / external embeddings
  -> BertTokenizer + BertModel -> prompt embeds [B,77,1024] + mask
  -> T5Tokenizer + T5EncoderModel -> prompt embeds_2 [B,256,2048] + mask
  -> latent noise initialization [B,4,H/8,W/8]
  -> denoising loop:
       CFG batch concat
       DDPMScheduler.scale_model_input (identity)
       HunyuanDiT2DModel(latents, text, masks, timestep, size/style, RoPE)
       learned-sigma channel chunk
       CFG and optional guidance_rescale
       DDPMScheduler.step
  -> AutoencoderKL decode(latents / 0.13025)
  -> VaeImageProcessor postprocess
```

Required first-slice components are prompt embeddings as external cached inputs,
base `HunyuanDiT2DModel`, DDPM v-pred scheduler tables/step, CFG arithmetic,
optional guidance-rescale reduction, and VAE decode. Cacheable stages are Bert
and T5 prompt embeddings/masks, RoPE tables per binned resolution, scheduler
timesteps/alpha tables per step count, initial latent tensors, and VAE decode
inputs for postprocess experiments.

Separate candidate reports:

| Surface | Classes/files | Delta from base |
| --- | --- | --- |
| `controlnet_hunyuandit` | `HunyuanDiTControlNetPipeline`, `HunyuanDiT2DControlNetModel`, `HunyuanDiT2DMultiControlNetModel`, `controlnet_hunyuan.py` | VAE-encodes control images, runs 19 token residual slots, injects residuals into base transformer skip path. Already audited separately. |
| `hunyuandit_pag` | `HunyuanDiTPAGPipeline`, `PAGHunyuanAttnProcessor2_0`, `PAGCFGHunyuanAttnProcessor2_0` | Mutates selected attention processors and guidance call structure through PAG. |
| `hunyuandit_v1_1_vs_v1_2` | Same base pipeline/model files, different configs | v1.1 omits `use_style_cond_and_image_meta_size`, so current source default is true; v1.2 explicitly sets false. Scheduler `beta_end` also changes. |
| `hunyuandit_lora_adapters` | Generic Diffusers/PEFT attention and linear adapter surfaces; no HunyuanDiT-specific loader mixin found on base pipeline | Runtime weight mutation should be admitted separately if concrete artifacts require it. |
| `hunyuan_image` | `pipeline_hunyuanimage.py`, `transformer_hunyuanimage.py` | Newer Hunyuan Image 2.1 uses Qwen/ByT5, FlowMatch, 64-channel latents, and a different transformer/VAE. Already audited separately. |
| `hunyuan_video` / `hunyuan_video_1_5` | Hunyuan video pipeline folders and 3D transformer/VAE classes | Video tensor rank, temporal codecs, and HunyuanVideo LoRA loader are unrelated to this base image first slice. |

No base HunyuanDiT img2img, inpaint, depth2img, upscaling, IP-Adapter,
T2I-Adapter, or GLIGEN pipeline class was found under `pipelines/hunyuandit`.

## 3. Important config dimensions

Representative checkpoint sweep:

| Repo/config | Pipeline | Transformer | Scheduler | Operator-significant notes |
| --- | --- | --- | --- | --- |
| `HunyuanDiT-Diffusers` | `HunyuanDiTPipeline` | local cache has model index only | model index declares `DDPMScheduler` | Older root Diffusers export; use v1.1/v1.2 component configs for shape details. |
| `HunyuanDiT-v1.1-Diffusers` | `HunyuanDiTPipeline` | 40 layers, 16 heads, head dim 88, hidden 1408, patch 2, 4 latent channels, learned sigma | DDPM v-pred, `beta_end=0.03` | Omits `use_style_cond_and_image_meta_size`; effective source default is true. |
| `HunyuanDiT-v1.2-Diffusers` | `HunyuanDiTPipeline` | Same dimensions | DDPM v-pred, `beta_end=0.018` | Explicit `use_style_cond_and_image_meta_size=false`; local component configs available. |
| `HunyuanDiT-v1.2-Diffusers-Distilled` | `HunyuanDiTPipeline` | Same inspected transformer config as v1.2 | Same inspected scheduler config as v1.2 | Distillation does not change the inspected model graph dimensions. |

Core v1.2 dimensions:

| Field | Value | Source |
| --- | ---: | --- |
| VAE latent channels | 4 | VAE config |
| VAE scale factor | 8 | pipeline from four VAE block levels |
| VAE scaling factor | 0.13025 | VAE config |
| Default image size | 1024 x 1024 | `sample_size 128 * vae_scale_factor 8` |
| Default latent map | `[B,4,128,128]` | source/config inference |
| Transformer patch size | 2 | transformer config |
| Image tokens at default | 4096 | `(128/2) * (128/2)` |
| Hidden/inner dim | 1408 | transformer config, 16 * 88 |
| Transformer depth | 40 blocks | transformer config |
| Skip path count | 19 saved early blocks | source: layers `< num_layers // 2 - 1` |
| MLP hidden size | 6145 | `int(1408 * 4.3637)` |
| Bert prompt length/dim | 77 / 1024 | tokenizer/text config |
| T5 prompt length/dim | 256 / 2048 | pipeline/text config |
| T5 projection | 2048 -> 8192 -> 1024 | source `PixArtAlphaTextProjection` |
| Text sequence after concat | 333 tokens | source |
| Text padding behavior | learned replacement parameter `[333,1024]` | source |
| Output channels | 8 raw, first 4 used | `learn_sigma=true`, pipeline chunk |
| Guidance | true CFG batch concat, optional guidance rescale | pipeline source |

Scheduler config:

| Field | v1.1 | v1.2 | Notes |
| --- | ---: | ---: | --- |
| class | `DDPMScheduler` | `DDPMScheduler` | Pipeline type annotation is narrow. |
| prediction_type | `v_prediction` | `v_prediction` | Required for one-step parity. |
| beta_start | 0.00085 | 0.00085 | scaled-linear schedule. |
| beta_end | 0.03 | 0.018 | Version-visible numeric delta. |
| variance_type | `fixed_small` | `fixed_small` | Model-predicted variance is discarded by pipeline before scheduler step. |
| timestep_spacing | `leading` | `leading` | `step_ratio = 1000 // steps`, reversed, plus offset. |
| steps_offset | 1 | 1 | Included in timestep table. |
| clip_sample / thresholding | false / false | false / false | No x0 clamp in sampled configs. |

Recommended first Dinoml scheduler slice: DDPM v-pred with leading timesteps,
fixed-small variance, explicit stochastic noise/generator handling, and a
deterministic validation mode using fixed variance noise. The pipeline does not
expose custom timesteps or sigmas in `__call__`.

## 3a. Family variation traps

- Do not merge this with Hunyuan Image 2.1. Base HunyuanDiT uses Bert + T5,
  4-channel AutoencoderKL, DDPM v-pred, and patch-size-2 DiT. Hunyuan Image 2.1
  uses Qwen/ByT5, 64-channel HunyuanImage VAE, FlowMatch, and a different
  transformer.
- This is not ControlNet: base `HunyuanDiT2DModel.forward` has a residual input
  hook, but base pipeline never supplies `controlnet_block_samples`.
- `learn_sigma=true` doubles denoiser output channels to 8, but pipeline chunks
  on channel axis and discards the variance half before CFG and DDPM step.
- v1.1 configs omit `use_style_cond_and_image_meta_size`; current source
  defaults it to true. v1.2 explicitly disables this size/style branch.
- The transformer replaces padded text tokens with a learned embedding; base
  block attention does not pass an attention mask to SDPA.
- Source latents and VAE tensors are NCHW. Transformer core is token layout
  `[B,N,C]`. NHWC/channel-last is only a guarded optimization candidate for
  VAE and patch Conv2d islands.
- Resolution binning can silently map requested sizes to one of ten supported
  H/W pairs before latents and RoPE are created.
- `height` and `width` are floored to multiples of 16 before validation/binning;
  valid inputs must then be divisible by 8 for the VAE and by patch size 2 for
  tokenization.
- The copied `prepare_extra_step_kwargs` has DDIM-era `eta` plumbing, but the
  base scheduler is DDPM; DDPM uses `generator` for stochastic variance noise.
- `fuse_qkv_projections()` is source-supported but not default. Treat fused
  QKV/KV weights as a separate model mutation path.

## 4. Runtime tensor contract

For default 1024x1024 generation, batch `B`, one image per prompt:

| Boundary | Tensor | Shape/layout | Notes |
| --- | --- | --- | --- |
| Bert token ids | `input_ids` | `[B,77]` | CPU tokenizer, max length 77. |
| Bert embeds | `prompt_embeds` | `[B,77,1024]` | `BertModel(...)[0]`, repeated for `num_images_per_prompt`. |
| Bert mask | `prompt_attention_mask` | `[B,77]` | Later concatenated with T5 mask. |
| T5 token ids | `input_ids_2` | `[B,256]` | CPU tokenizer, pipeline max length 256. |
| T5 embeds | `prompt_embeds_2` | `[B,256,2048]` | Projected inside transformer to 1024. |
| T5 mask | `prompt_attention_mask_2` | `[B,256]` | Concatenated with Bert mask. |
| CFG text embeds | after concat | `[2B,77,1024]` and `[2B,256,2048]` | Negative first, positive second. |
| Add time ids | `image_meta_size` | `[2B,6]` under CFG | `[original_h, original_w, target_h, target_w, crop_y, crop_x]`; used only if model config enables style/size. |
| Style | `style` | `[2B]` under CFG | Single zero style id; used only if style branch active. |
| Latent noise | `latents` | `[B,4,H/8,W/8]`, NCHW | Multiplied by `scheduler.init_noise_sigma == 1.0`. |
| Denoiser input | `latent_model_input` | `[2B,4,H/8,W/8]` under CFG | `scale_model_input` is identity for DDPM. |
| RoPE | `image_rotary_emb` | `(cos,sin)` over `grid_h * grid_w`, head dim 88 | Default grid `[64,64]`, generated from crop region and latent token grid. |
| Patch tokens | internal | `[2B,grid_h*grid_w,1408]` | Conv2d(4 -> 1408, 2x2, stride 2), flatten/transpose. |
| Text tokens | internal | `[2B,333,1024]` | Bert concat projected T5; padded slots replaced by learned parameter. |
| Raw denoiser output | `noise_pred_raw` | `[2B,8,H/8,W/8]`, NCHW | Unpatchified transformer output. |
| Noise prediction | after channel chunk | `[2B,4,H/8,W/8]` | First half only. |
| Guided prediction | after CFG | `[B,4,H/8,W/8]` | Optional guidance_rescale applies std reduction over non-batch axes. |
| Scheduler state | timesteps/alphas | `[steps]`, alpha tables length 1000 | v-pred DDPM reverse step. |
| VAE decode input | `latents / 0.13025` | `[B,4,H/8,W/8]`, NCHW | No shift factor in sampled configs. |
| Decoded image | `image` | `[B,3,H,W]`, NCHW before postprocess | `VaeImageProcessor` converts/denormalizes. |

Autoencoder encode contract, needed by future variants even though base
text-to-image decodes only:

```text
image [B,3,H,W] -> AutoencoderKL.encode(image).latent_dist.sample()
latents = sample * 0.13025
```

CPU/data-pipeline work: prompt tokenization, text encoders if embeddings are not
precomputed, resolution binning, PIL/NumPy postprocess. GPU/runtime work:
denoiser, CFG/rescale, scheduler step, and VAE decode.

## 5. Operator coverage checklist

Tensor/layout ops:

- NCHW latent allocation, scalar multiply by `init_noise_sigma`.
- CFG `cat` and `chunk` on batch axis.
- Learned-sigma `chunk(2, dim=1)` on NCHW channel axis.
- Prompt embedding/mask `repeat`, `view`, `cat`, `unsqueeze(2).bool`.
- `torch.where` text padding replacement against `[333,1024]` parameter.
- Skip tensor stack/pop inside transformer blocks.
- Patchify `Conv2d -> flatten(2) -> transpose(1,2)`.
- Unpatchify `reshape(B,h,w,p,p,C) -> einsum("nhwpqc->nchpwq") -> reshape`.

Convolution/downsample/upsample ops:

- Patch embedding: `Conv2d(4 -> 1408, kernel=2, stride=2, bias=true)`.
- AutoencoderKL decode and future encode: Conv2d, quant/post-quant 1x1 convs,
  ResNet blocks, GroupNorm(32), SiLU, downsample/upsample blocks, mid-block
  attention by source default.

GEMM/linear ops:

- T5 projection `Linear(2048 -> 8192)`, FP32 SiLU, `Linear(8192 -> 1024)`.
- `HunyuanDiTAttentionPool`: Q/K/V/out projections over T5 sequence plus pooled
  mean token.
- Timestep embedding MLP and optional size/style/text extra projection.
- Per block: self-attention Q/K/V/out, cross-attention Q/K/V/out, feed-forward
  approximate GELU MLP, optional skip `Linear(2816 -> 1408)`.
- Output projection `Linear(1408 -> 2*2*8 = 32)`.

Attention primitives:

- Image-token self-attention, 16 heads x 88 dim, Q/K LayerNorm, RoPE on Q and K,
  no mask.
- Cross-attention from image tokens to 333 text tokens, Q/K LayerNorm, RoPE on
  query only, no attention mask after text padding replacement.
- T5 pooling attention in `HunyuanCombinedTimestepTextSizeStyleEmbedding`.
- Default processor is `HunyuanAttnProcessor2_0`; fused processor exists but is
  not default.

Normalization and adaptive conditioning:

- `AdaLayerNormShift`: LayerNorm plus timestep/text/size/style shift.
- `FP32LayerNorm` for cross-attention, feed-forward, and skip path.
- `AdaLayerNormContinuous` output norm with scale/shift from conditioning.
- Attention Q/K LayerNorm.
- VAE GroupNorm.

Scheduler and guidance arithmetic:

- DDPM leading timestep table, scaled-linear betas, alpha cumulative products.
- v-pred conversion `x0 = sqrt(alpha) * sample - sqrt(beta) * model_output`.
- Fixed-small variance noise branch for full stochastic parity.
- CFG `uncond + guidance_scale * (text - uncond)`.
- Optional guidance rescale: per-sample std ratio over non-batch axes.

## 6. Denoiser/model breakdown

Top-level `HunyuanDiT2DModel.forward`:

```text
hidden_states [B,4,Hl,Wl]
  -> PatchEmbed Conv2d(4,1408,2,stride=2) -> [B,N,1408]
  -> HunyuanCombinedTimestepTextSizeStyleEmbedding(timestep, T5, optional size/style) -> temb [B,1408]
  -> T5 projection 2048 -> 8192 -> 1024
  -> concat Bert/T5 tokens -> [B,333,1024]
  -> concat masks, unsqueeze, bool
  -> replace padded token embeddings with learned text_embedding_padding
  -> 40 HunyuanDiTBlock layers with early skip stack and later skip linear
  -> AdaLayerNormContinuous + Linear(1408 -> 32)
  -> unpatchify -> [B,8,Hl,Wl]
```

`HunyuanDiTBlock` active path:

```text
optional skip:
  concat current and popped skip [B,N,2816]
  FP32LayerNorm -> Linear(2816 -> 1408)
self-attention:
  AdaLayerNormShift(temb)
  Q/K/V projections, Q/K LayerNorm, image RoPE on Q and K
  noncausal SDPA, output projection
  residual add
cross-attention:
  FP32LayerNorm
  Q from image tokens, K/V from text tokens
  Q/K LayerNorm, RoPE on query only
  noncausal SDPA, output projection
  residual add
feed-forward:
  FP32LayerNorm
  Linear -> GELU(tanh approximate) -> Linear
  residual add
```

Conditioning embedding:

```text
timestep -> Timesteps(256, flip_sin_to_cos=True) -> TimestepEmbedding(1408)
T5 tokens -> attention pool -> [B,1024]
if style/size enabled:
  image_meta_size six scalars -> six Timesteps(256) chunks
  style id -> Embedding(1,1408)
  concat pooled + size + style -> PixArtAlphaTextProjection -> [B,1408]
else:
  pooled -> PixArtAlphaTextProjection -> [B,1408]
temb = timestep_emb + extra_embedding
```

## 7. Attention requirements

Primary implementation path is `HunyuanAttnProcessor2_0` in
`attention_processor.py`. It projects Q/K/V, reshapes to
`[B,heads,seq,head_dim]`, applies optional Q/K LayerNorm, applies RoPE when
provided, then calls `torch.nn.functional.scaled_dot_product_attention` with
`dropout_p=0` and `is_causal=false`.

Required base variants:

| Attention | Query | Key/value | Mask | RoPE | Notes |
| --- | --- | --- | --- | --- | --- |
| Self-attention | image patch tokens `[B,N,1408]` | same | none | Q and K | Default N=4096 at 1024 square. |
| Cross-attention | image patch tokens `[B,N,1408]` | text tokens `[B,333,1024]` projected to heads | none | Q only | Text padding is handled by learned replacement before attention. |
| T5 pooler | one pooled query over `[B,256,2048]` | T5 tokens plus mean token | none | none | Uses `F.multi_head_attention_forward`, 8 heads. |

Flash-style Dinoml feasibility:

- Self-attention is a dense noncausal flash candidate if head dim 88 and dtype
  are supported.
- Cross-attention is also dense/mask-free after padding replacement, but Q
  sequence length and KV sequence length differ and RoPE applies only to Q.
- Q/K LayerNorm and RoPE must be explicit pre-provider ops unless the provider
  advertises those epilogues/preludes.
- Fused QKV/KV projection is source-supported by `fuse_qkv_projections()` and
  `FusedHunyuanAttnProcessor2_0`, but first parity should keep explicit
  projections and treat fused weights as a guarded mutation path.
- PAG and ControlNet attention/residual branches change the processor or inputs
  and are separate candidates.

`attention_dispatch.py` is not the primary base HunyuanDiT path; it matters for
newer Hunyuan Image and other transformer families.

## 8. Scheduler and denoising-loop contract

Base loop:

```text
scheduler.set_timesteps(num_inference_steps, device)
latents = randn([B,4,H/8,W/8]) * scheduler.init_noise_sigma
for t in scheduler.timesteps:
  latent_model_input = cat([latents, latents]) when CFG
  latent_model_input = scheduler.scale_model_input(latent_model_input, t)  # identity
  t_expand = full([latent_model_input.batch], t, dtype=latent dtype)
  noise_pred = transformer(...)
  noise_pred = noise_pred.chunk(2, dim=1)[0]  # discard sigma half
  if CFG:
    noise_pred = uncond + guidance_scale * (text - uncond)
  if guidance_rescale:
    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text)
  latents = scheduler.step(noise_pred, t, latents, generator=generator)[0]
```

`DDPMScheduler.set_timesteps` with v1.2 config uses leading spacing:

```text
step_ratio = num_train_timesteps // num_inference_steps
timesteps = (arange(steps) * step_ratio).round()[::-1] + steps_offset
```

`DDPMScheduler.step` with `prediction_type="v_prediction"`:

```text
pred_original_sample = sqrt(alpha_prod_t) * sample - sqrt(beta_prod_t) * model_output
pred_prev = coeff_x0 * pred_original_sample + coeff_xt * sample + variance
```

Because configs use `variance_type="fixed_small"`, source adds random variance
noise for `t > 0` even though model-predicted variance was discarded. First
Dinoml parity should make the variance-noise input explicit. A deterministic
unit slice can pass fixed variance noise or validate the mean update separately,
but full image parity needs generator/noise-state ownership.

Source default scheduler is DDPM v-pred for all inspected base model indexes.
Unlike SD1/SDXL Karras-compatible pipelines, base HunyuanDiT `__call__` does
not expose custom timesteps or sigmas and the constructor annotation is
`DDPMScheduler`.

## 9. Position, timestep, and custom math

- `get_resize_crop_region_for_grid` maps the latent token grid into a
  512-derived base grid before RoPE generation.
- Default 1024 image: latent map `[128,128]`, patch grid `[64,64]`, base size
  `512 / 8 / 2 = 32`.
- `get_2d_rotary_pos_embed(head_dim=88, ..., output_type="pt")` asserts head dim
  divisible by 4 and returns real cos/sin tensors. Half the dimensions encode
  each spatial axis.
- Timestep embedding uses `Timesteps(256, flip_sin_to_cos=True,
  downscale_freq_shift=0)` plus `TimestepEmbedding`.
- v1.1/source-default size/style branch embeds six image metadata scalars with
  `Timesteps(256)` and a learned style id. v1.2 disables that branch and uses
  only pooled T5 condition beyond timestep.
- `rescale_noise_cfg` computes per-sample std over all non-batch axes, rescales
  guided prediction by `std_text / std_cfg`, then blends with original guided
  prediction by `guidance_rescale`.

Precomputable: RoPE per binned resolution, scheduler alpha/timestep tables per
step count/config, text embeddings/masks per prompt, pooled T5 projection per
prompt. Dynamic per step: timestep embedding, DDPM coefficients, stochastic
variance noise, CFG/guidance-rescale outputs.

## 10. Preprocessing and input packing

Bert path:

```text
BertTokenizer(max_length=77, padding=max_length, truncation=True)
BertModel(input_ids, attention_mask).last_hidden_state -> [B,77,1024]
attention_mask -> [B,77]
```

T5 path:

```text
T5Tokenizer(max_length=256, padding=max_length, truncation=True)
T5EncoderModel(input_ids, attention_mask).last_hidden_state -> [B,256,2048]
attention_mask -> [B,256]
```

For CFG, negative embeddings are generated with empty strings when no negative
prompt is supplied, then negative and positive embeddings are concatenated on
batch axis. Prompt embeddings are duplicated for `num_images_per_prompt` by
repeat/view.

Resolution path:

- `height` and `width` default to 1024 for v1.2.
- Source floors requested H/W to multiples of 16.
- With resolution binning, unsupported sizes map to one of:
  1024x1024, 1280x1280, 1024x768, 1152x864, 1280x960, 768x1024, 864x1152,
  960x1280, 1280x768, or 768x1280.
- The transformer patchifies internally; there is no Flux-style
  pipeline-level latent packing.

## 11. Graph rewrite / lowering opportunities

### Rewrite: Hunyuan patch embed as strided patch GEMM

Source pattern:

```text
Conv2d(4 -> 1408, kernel=2, stride=2) -> flatten(2) -> transpose(1,2)
```

Replacement: extract non-overlapping `2x2x4` patches and GEMM to 1408 channels,
or a fused patch-embedding primitive.

Preconditions: NCHW source layout, patch size 2, stride 2, no padding, even
latent H/W, bias preserved, weight flatten order matches PyTorch OIHW over
`C,pH,pW`. Failure cases: odd latent grid, NHWC activation without matching
weight/layout transform, or source config with changed patch size/channel count.

Parity test: random `[2,4,128,128]` and one non-square binned latent through
Diffusers `PatchEmbed`.

### Rewrite: unpatchify as planned token-to-map transform

Source pattern:

```text
Linear(1408 -> 32)
reshape [B,Ht,Wt,2,2,8]
einsum "nhwpqc->nchpwq"
reshape [B,8,Ht*2,Wt*2]
```

Replacement: explicit token-to-NCHW unpatchify primitive, optionally fused with
projection epilogue.

Preconditions: token count equals `Ht * Wt`, patch size 2, `out_channels=8`,
consumer channel chunk axis is source NCHW dim 1. Failure cases: NHWC rewrite
without changing chunk axis, learned-variance scheduler variant that observes
the second half, or mismatched dynamic grid.

### Rewrite: learned-sigma discard

Source pattern: materialize `[B,8,H,W]`, then keep first 4 channels before CFG
and scheduler.

Replacement: lower output projection for first 4 channels only, or slice in a
fused output staging kernel.

Preconditions: scheduler variance type remains `fixed_small` or another mode
that does not consume model-predicted variance; no caller requests raw denoiser
output. Failure cases: `variance_type="learned"`/`"learned_range"` or debug
parity requiring exact full transformer output.

### Rewrite: text padding replacement canonicalization

Source pattern:

```text
text = cat([bert, projected_t5], dim=1)
mask = cat([bert_mask, t5_mask], dim=-1).unsqueeze(2).bool()
text = torch.where(mask, text, text_embedding_padding)
```

Replacement: masked fill/select primitive with static padding table, run once
per prompt batch before all denoising steps.

Preconditions: fixed text lengths 77 and 256, padding table shape `[333,1024]`,
mask dtype/broadcast semantics preserved. Failure cases: dynamic prompt lengths
without fixed padding, tokenizer changes, or moving replacement after attention.

### Rewrite: DDPM v-pred mean update

Source pattern: alpha table loads, v-pred conversion, coefficient blend, plus
variance noise.

Replacement: fused pointwise update with per-step scalar coefficients and
explicit variance-noise tensor input.

Preconditions: config has `thresholding=false`, `clip_sample=false`,
`prediction_type=v_prediction`, variance mode represented. Failure cases:
learned variance, thresholding/clip enabled, or hidden generator state.

### Rewrite: guarded NHWC VAE/patch conv islands

Source pattern: VAE Conv2d/GroupNorm/SiLU blocks and patch Conv2d operate on
NCHW maps.

Replacement: local NHWC/channel-last conv islands with NCHW boundaries.

Preconditions: every op in the island has axis rewrites; Conv weights transform
OIHW to HWIO; GroupNorm channel axis changes from dim 1 to last dim; unpatchify
and scheduler/CFG boundaries remain NCHW or have explicit adapters. Failure
cases: crossing into transformer token ABI, VAE tiling/slicing policy, or
accidentally chunking learned-sigma on the wrong axis.

## 12. Kernel fusion candidates

Highest priority:

- Patch embed Conv2d + flatten/transpose and output projection + unpatchify.
- Self-attention QKV + Q/K LayerNorm + RoPE + SDPA + output projection.
- Cross-attention Q/K/V + Q/K LayerNorm + query RoPE + SDPA + output projection.
- AdaLayerNormShift / FP32LayerNorm plus residual epilogues in
  `HunyuanDiTBlock`.
- CFG + learned-sigma channel trim; add guidance-rescale reduction when needed.
- DDPM v-pred mean update with explicit variance-noise input.

Medium priority:

- Feed-forward approximate GELU MLP fusion.
- T5 projection and `HunyuanDiTAttentionPool` for prompt-cache integration.
- Skip concat + LayerNorm + Linear in the later transformer blocks.
- VAE decode Conv2d + GroupNorm + SiLU islands.
- Text padding replacement and mask concat as prompt-cache preprocessing.

Lower priority:

- Fused QKV/KV projection mutation via `FusedHunyuanAttnProcessor2_0`.
- v1.1 size/style conditioning specialization after v1.2 parity.
- Full stochastic DDPM image-loop parity beyond fixed-noise tests.
- Bert/T5 encoder execution inside Dinoml.
- PAG and ControlNet variants.

## 13. Runtime staging plan

Stage 1: Parse v1.2 base configs from `H:/configs`, load transformer/VAE
weights, and accept external Bert/T5 embeddings and masks.

Stage 2: Validate patch embed, RoPE generation, text projection/padding
replacement, and one `HunyuanDiTBlock` at fixed 1024 resolution.

Stage 3: Implement full `HunyuanDiT2DModel` v1.2 forward with no ControlNet
residuals and with learned-sigma channel trim.

Stage 4: Add DDPM v-pred scheduler table setup and one-step update with fixed
variance noise. Keep loop control in host runtime.

Stage 5: Add CFG and guidance-rescale parity.

Stage 6: Add AutoencoderKL decode boundary with scaling factor 0.13025. Keep
VAE encode documented for later variants.

Stage 7: Add v1.1 style/size conditioning admission and beta schedule
variation after v1.2 base is stable.

Stage 8: Add optimized attention, norm, MLP, patch, scheduler, and VAE kernels.
Keep ControlNet/PAG/Hunyuan Image/Hunyuan Video separate.

## 14. Parity and validation plan

- Config admission tests for v1.1, v1.2, and distilled model indexes; verify
  v1.1 omitted `use_style_cond_and_image_meta_size` resolves to source default
  true and v1.2 resolves to false.
- Prompt embedding contract tests for external `[B,77,1024]` and
  `[B,256,2048]` embeddings and masks.
- T5 projection and padding replacement parity with synthetic masks.
- Patch embed parity for 1024 square and one non-square supported bin.
- RoPE table parity for `[64,64]`, `[80,48]`, and `[48,80]` token grids.
- Conditioning embedding parity for v1.2 no-size/style and v1.1 size/style.
- Single `HunyuanDiTBlock` parity for self-attention, cross-attention, FFN, and
  skip variants.
- Full transformer forward parity with fixed embeddings, masks, RoPE, and
  latents.
- Learned-sigma channel chunk parity.
- CFG and guidance-rescale parity.
- DDPM `set_timesteps`, v-pred conversion, fixed-noise one-step update, and
  stochastic generator/noise tests.
- VAE decode parity for random `[B,4,H/8,W/8]` latents.
- Short deterministic denoising loop with externally supplied embeddings and
  fixed variance-noise tensors.

Suggested tolerances: fp32 scheduler/pointwise `rtol=1e-5, atol=1e-6`;
transformer/VAE fp16 or bf16 start at `rtol=2e-2, atol=2e-2`, then tighten
after provider-specific attention/norm choices are validated.

## 15. Performance probes

- Transformer forward by resolution bin: 1024x1024, 1280x1280, 1280x768, and
  768x1280.
- Attention backend comparison for image token lengths 4096, 6144, and 6400
  with head dim 88.
- Self-attention versus cross-attention time split.
- Patch embed/unpatchify bandwidth and NCHW versus guarded NHWC conv island.
- Skip-block concat/norm/linear cost in later layers.
- CFG enabled versus disabled; guidance_rescale reduction cost.
- DDPM scheduler update bandwidth and stochastic variance-noise generation or
  staging cost.
- VAE decode throughput and memory for 1024 and 1280 square.
- Prompt-side T5 projection/pooler cost when embeddings are cached versus
  recomputed.
- VRAM/workspace peak across batch, CFG, dtype, and resolution bins.

## 16. Scope boundary and separate candidates

Separate candidate reports, not ignored:

- `controlnet_hunyuandit`: base transformer residual injection plus one or
  multiple HunyuanDiT ControlNets.
- `hunyuandit_pag`: attention processor mutation and PAG guidance behavior.
- `hunyuandit_v1_1_vs_v1_2`: style/size conditioning default and scheduler beta
  differences if v1.1 support is required.
- `hunyuandit_lora_adapters`: PEFT/LoRA adapter state and fused/unfused weight
  mutation for concrete artifacts.
- `hunyuan_image`: newer image model family, separate architecture and configs.
- `hunyuan_video` / `hunyuan_video_1_5`: video transformer/codecs and
  HunyuanVideo LoRA loader.
- `ddpm_stochastic_full_parity`: generator/noise stream ownership and full
  stochastic reverse-loop reproducibility.
- `autoencoder_kl_hunyuandit`: detailed VAE encode/decode optimization if the
  shared AutoencoderKL report needs Hunyuan-specific scaling/upcast coverage.

Genuinely out of scope for this audit:

- Safety checker and NSFW filtering.
- Callback mutation and interactive interrupt.
- XLA, NPU, MPS, Flax, ONNX, and multi-GPU/context-parallel paths.
- Training, losses, dropout behavior, and gradient checkpointing.
- Hunyuan Image 2.1 and Hunyuan video runtime paths except as separate
  candidate inventory.

## 17. Final implementation checklist

- [ ] Parse v1.2 base HunyuanDiT model index and component configs.
- [ ] Record v1.1 omitted defaults, especially size/style conditioning.
- [ ] Load `HunyuanDiT2DModel`, AutoencoderKL, and DDPM scheduler metadata.
- [ ] Accept external Bert embeddings/masks and T5 embeddings/masks.
- [ ] Implement Hunyuan patch embed and unpatchify parity.
- [ ] Implement T5 projection and text padding replacement.
- [ ] Implement timestep, T5 pooler, and optional size/style conditioning.
- [ ] Implement `HunyuanDiTBlock` self-attention, cross-attention, FFN, and skip paths.
- [ ] Implement Q/K LayerNorm plus RoPE before SDPA.
- [ ] Implement full base transformer forward without ControlNet residuals.
- [ ] Implement learned-sigma channel discard.
- [ ] Implement CFG and guidance-rescale arithmetic.
- [ ] Implement DDPM v-pred leading timestep table and fixed-small variance step.
- [ ] Add fixed-variance-noise and stochastic scheduler parity tests.
- [ ] Add AutoencoderKL decode boundary with scaling factor 0.13025.
- [ ] Add one-step and short-loop parity with externally supplied embeddings.
- [ ] Add guarded attention, norm, MLP, patch, scheduler, and VAE fusion probes.
- [ ] Keep ControlNet, PAG, LoRA/adapters, Hunyuan Image, and Hunyuan Video as separate candidates.
