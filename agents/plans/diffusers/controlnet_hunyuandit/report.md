# Diffusers HunyuanDiT ControlNet Operator and Integration Report

Target slug: `controlnet_hunyuandit`

## 1. Source basis

```text
Diffusers commit/version:
  diffusers at b3a515080752a3ba7ca92161e25530c7f280f629.

Model id(s):
  Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers
  Tencent-Hunyuan/HunyuanDiT-v1.1-ControlNet-Diffusers-Canny
  Tencent-Hunyuan/HunyuanDiT-v1.1-ControlNet-Diffusers-Depth
  Tencent-Hunyuan/HunyuanDiT-v1.1-ControlNet-Diffusers-Pose
  Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers
  Tencent-Hunyuan/HunyuanDiT-v1.2-ControlNet-Diffusers-Canny
  Tencent-Hunyuan/HunyuanDiT-v1.2-ControlNet-Diffusers-Pose

Config sources:
  Local cache:
    H:/configs/Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers/model_index.json
    H:/configs/Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers/transformer/config.json
    H:/configs/Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers/vae/config.json
    H:/configs/Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers/scheduler/scheduler_config.json
    H:/configs/Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers/text_encoder/config.json
    H:/configs/Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers/text_encoder_2/config.json
  Network-inspected raw JSON, not saved because this task owns only this report path:
    https://huggingface.co/Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers/raw/main/{transformer,vae,scheduler,text_encoder,text_encoder_2}/config.json
    https://huggingface.co/Tencent-Hunyuan/HunyuanDiT-v1.1-ControlNet-Diffusers-{Canny,Depth,Pose}/raw/main/config.json
    https://huggingface.co/Tencent-Hunyuan/HunyuanDiT-v1.2-ControlNet-Diffusers-{Canny,Pose}/raw/main/config.json

Pipeline files inspected:
  diffusers/src/diffusers/pipelines/controlnet_hunyuandit/pipeline_hunyuandit_controlnet.py
  diffusers/src/diffusers/pipelines/hunyuandit/pipeline_hunyuandit.py

Model files inspected:
  diffusers/src/diffusers/models/controlnets/controlnet_hunyuan.py
  diffusers/src/diffusers/models/transformers/hunyuan_transformer_2d.py
  diffusers/src/diffusers/models/autoencoders/autoencoder_kl.py

Scheduler/processors/helpers inspected:
  diffusers/src/diffusers/schedulers/scheduling_ddpm.py
  diffusers/src/diffusers/image_processor.py
  diffusers/src/diffusers/models/attention.py
  diffusers/src/diffusers/models/attention_processor.py
  diffusers/src/diffusers/models/attention_dispatch.py
  diffusers/src/diffusers/models/embeddings.py
  diffusers/src/diffusers/models/normalization.py

External component configs inspected:
  BertModel / BertTokenizer configs for HunyuanDiT bilingual text encoder.
  T5EncoderModel / T5Tokenizer configs for the mT5 encoder.

Any missing files or assumptions:
  No local ControlNet component configs were present in H:/configs. Public raw config fetches succeeded; no gated
  config blocker was hit. This report ignores backend/training/safety/callback paths per task. Runtime target is
  inference-only CUDA, faithful NCHW/token layout first, with NHWC/channel-last only as a guarded optimization.
```

## 2. Pipeline and component graph

`HunyuanDiTControlNetPipeline` is the legacy HunyuanDiT image pipeline plus one
or more `HunyuanDiT2DControlNetModel` instances. The constructor registers
`vae`, `text_encoder`, `tokenizer`, `text_encoder_2`, `tokenizer_2`,
`transformer`, `scheduler`, optional safety components, and `controlnet`.
Lists or tuples of ControlNets are wrapped by
`HunyuanDiT2DMultiControlNetModel`. The offload sequence is
`text_encoder->text_encoder_2->transformer->vae`; ControlNet is registered but
not included in that string.

```text
prompt / external prompt embeddings
  -> Bert text encoder [B,77,1024] + mask
  -> T5 text encoder [B,256,2048] + mask
  -> control image preprocessing [B,3,H,W], VAE encode, scale 0.13025 -> control latents [B,4,H/8,W/8]
  -> latent initialization [B,4,H/8,W/8]
  -> denoising loop:
       DDPMScheduler.scale_model_input
       HunyuanDiT2DControlNetModel(noisy latents, control latents, text, timestep, size/style, RoPE)
       -> 19 token residuals
       HunyuanDiT2DModel(..., controlnet_block_samples=residuals)
       -> learned-sigma output [B,8,H/8,W/8], chunk to noise [B,4,H/8,W/8]
       CFG and optional guidance_rescale
       DDPMScheduler.step
  -> VAE decode(latents / 0.13025)
  -> VaeImageProcessor postprocess
```

Required first-slice components are externally supplied Bert/T5 embeddings and
masks, VAE encode for the control image, the ControlNet token branch, the base
HunyuanDiT transformer with residual injection, DDPMScheduler v-pred step, CFG,
and VAE decode. Cacheable stages include prompt embeddings/masks, encoded
control latents for a fixed control image and resolution, image RoPE tables,
style/time-size conditioning inputs, scheduler timesteps/alphas, initial
latents, and VAE decode inputs for repeated postprocess experiments.

Separate candidate reports:

| Candidate | Classes/files | Delta |
| --- | --- | --- |
| `hunyuandit_legacy` | `HunyuanDiTPipeline`, `HunyuanDiT2DModel` | Base text-to-image without ControlNet residuals; same Bert/T5, VAE, DDPMScheduler, learned-sigma output. |
| `controlnet_hunyuandit_multi` | `HunyuanDiT2DMultiControlNetModel` in `controlnet_hunyuan.py` | Runs multiple Hunyuan ControlNets and sums matching token residual slots. |
| `hunyuandit_v1_1_vs_v1_2` | Same pipeline/model files, different configs | v1.1 uses style and image-meta conditioning by default/source omission; v1.2 config sets `use_style_cond_and_image_meta_size=false`. |
| `hunyuandit_pag` | `pipelines/pag/pipeline_pag_hunyuandit.py`, PAG processors | Perturbed attention guidance for the legacy HunyuanDiT transformer. |
| `hunyuandit_lora_adapters` | Attention/linear adapter loader surfaces inherited through Diffusers mixins | Runtime weight mutation separate from the ControlNet tensor contract. |
| `hunyuan_image` / `hunyuan_image_control` | `pipeline_hunyuanimage.py`, `transformer_hunyuanimage.py`, `controlnet_hunyuan.py` only by name overlap | Newer Hunyuan Image 2.1 uses Qwen/ByT5, FlowMatch Euler, 64-channel latents, and a different transformer; do not merge with this report. |

No HunyuanDiT img2img, inpaint, depth2img, upscaling, IP-Adapter, T2I-Adapter,
or GLIGEN pipeline class was found under `controlnet_hunyuandit` or
`hunyuandit`.

## 3. Important config dimensions

Representative checkpoint sweep:

| Repo/config | Pipeline/model class | Transformer/ControlNet | Scheduler | Version-significant fields |
| --- | --- | --- | --- | --- |
| `HunyuanDiT-v1.1-Diffusers` | `HunyuanDiTPipeline`, `HunyuanDiT2DModel` | 40 layers, 16 heads, head dim 88, hidden 1408, patch 2, 4 latent channels, learned sigma | DDPMScheduler, `beta_end=0.03`, v-pred | `use_style_cond_and_image_meta_size` omitted, effective source default `true`. |
| `HunyuanDiT-v1.1-ControlNet-Diffusers-{Canny,Depth,Pose}` | `HunyuanDiT2DControlNetModel` | Same dims; `transformer_num_layers=40`, 19 ControlNet residual slots | Inherits base pipeline scheduler | `use_style_cond_and_image_meta_size` omitted, effective source default `true`. |
| `HunyuanDiT-v1.2-Diffusers` | `HunyuanDiTPipeline`, `HunyuanDiT2DModel` | Same dims | DDPMScheduler, `beta_end=0.018`, v-pred | `use_style_cond_and_image_meta_size=false`. |
| `HunyuanDiT-v1.2-ControlNet-Diffusers-{Canny,Pose}` | `HunyuanDiT2DControlNetModel` | Same dims plus `num_layers=40` and `transformer_num_layers=40` | Inherits base pipeline scheduler | `use_style_cond_and_image_meta_size=false`. |

Core dimensions:

| Field | Value | Source |
| --- | ---: | --- |
| VAE latent channels | 4 | VAE config |
| VAE scale factor | 8 | pipeline from 4 VAE block levels |
| VAE scaling factor | 0.13025 | VAE config |
| Default image size | 1024x1024 | pipeline default `sample_size 128 * vae_scale_factor 8` |
| Latent map at default | `[B,4,128,128]` | inferred from source/config |
| Transformer patch size | 2 | transformer/controlnet config |
| Image tokens at default | 4096 | `[128/2,128/2]` |
| Inner dim | 1408 | 16 heads * 88 head dim |
| Transformer depth | 40 blocks | config |
| ControlNet depth | `transformer_num_layers // 2 - 1 = 19` blocks | source/config |
| Control residual slots | 19 token tensors | source |
| Bert prompt dim/length | 1024 / 77 | text config and pipeline max length |
| T5 prompt dim/length | 2048 / 256 | text_encoder_2 config and pipeline max length |
| T5 projection | 2048 -> 8192 -> 1024 | source `PixArtAlphaTextProjection` |
| Text sequence after concat | 333 tokens | source |
| Learned sigma output | yes, transformer outputs 8 channels then pipeline keeps first 4 | config/source |
| RoPE head dim | 88 | `inner_dim // heads` |
| Guidance | classic CFG batch concat; optional `guidance_rescale` std ratio | pipeline source |

Scheduler support is narrow in the type annotation: `DDPMScheduler`. The
sampled configs use v-prediction, `scaled_linear` betas, `fixed_small`
variance, `steps_offset=1`, `timestep_spacing="leading"`, and
`clip_sample=false`. Recommended first Dinoml scheduler slice is deterministic
DDPM v-pred with variance disabled or stubbed to zero for parity experiments,
then add source stochastic variance if full image parity requires it.

## 3a. Family variation traps

- This is not Hunyuan Image 2.1: no Qwen/ByT5, no FlowMatch Euler, no
  64-channel latents, no HunyuanImage VAE.
- Control conditioning is not a ControlNet conv pyramid over RGB. The pipeline
  first VAE-encodes the control image, producing the same 4-channel latent map
  shape as noisy latents.
- ControlNet residuals are token tensors `[B,4096,1408]` at 1024 square, not
  NCHW down-block residual maps.
- The base transformer stores 19 early block outputs, then later blocks with
  `skip=True` pop these skip tensors. ControlNet residuals are added to those
  skip tensors before the skip linear.
- `learn_sigma=true` doubles transformer output channels. The pipeline chunks
  `[B,8,H/8,W/8]` on channel dim and discards the variance half before CFG and
  scheduler step.
- v1.1 ControlNet configs omit `use_style_cond_and_image_meta_size`; current
  source default is `true`. v1.2 explicitly sets it `false`.
- `HunyuanDiT2DControlNetModel.from_transformer` references
  `config.transformer_num_layers`, but base transformer configs use
  `num_layers`; this factory path looks fragile in current source. Loading
  published ControlNet configs directly avoids that issue.
- Source tensors are NCHW until patch embed; transformer core is token layout
  `[B,N,C]`. NHWC optimization is local to VAE/patch Conv2d islands only.
- Resolution binning silently maps unsupported sizes to a fixed supported set
  unless `use_resolution_binning=false`.
- CFG duplicates the control image before VAE encode because `guess_mode` is
  always passed as false in this pipeline.

## 4. Runtime tensor contract

For default 1024x1024, batch `B`, and CFG enabled:

| Boundary | Tensor | Shape/layout | Notes |
| --- | --- | --- | --- |
| Bert prompt embeds | `prompt_embeds` | `[2B,77,1024]` | negative then positive batch after CFG concat. |
| Bert mask | `prompt_attention_mask` | `[2B,77]` | concatenated with T5 mask and unsqueezed to `[2B,333,1]`. |
| T5 prompt embeds | `prompt_embeds_2` | `[2B,256,2048]` | projected to 1024 inside ControlNet and transformer. |
| T5 mask | `prompt_attention_mask_2` | `[2B,256]` | padding replaced by learned `text_embedding_padding`. |
| Control image | preprocessed image | `[2B,3,1024,1024]`, NCHW | `VaeImageProcessor` default normalization applies unless tensor is already supplied. |
| Control latent | `control_image` | `[2B,4,128,128]`, NCHW | VAE sampled encode times 0.13025. |
| Noisy latents | `latents` | `[B,4,128,128]`, NCHW | random normal times scheduler `init_noise_sigma`. |
| Denoiser input | `latent_model_input` | `[2B,4,128,128]`, NCHW | scheduler `scale_model_input` is identity for DDPM. |
| RoPE | `image_rotary_emb` | real cos/sin pair for grid `[64,64]`, dim 88 | built from crop region and grid. |
| Size/style | `add_time_ids`, `style` | `[2B,6]`, `[2B]` | used only when config/default enables style+size conditioning. |
| Patch tokens | `pos_embed(latents)` | `[2B,4096,1408]` | Conv2d(4 -> 1408, 2x2, stride 2), flatten/transpose. |
| Control residuals | `control_block_samples` | 19 x `[2B,4096,1408]` | zero Linear heads, scaled by conditioning scale. |
| Transformer raw output | `noise_pred_raw` | `[2B,8,128,128]`, NCHW | learned sigma channel half present. |
| Noise prediction | `noise_pred` | `[2B,4,128,128]` then `[B,4,128,128]` | channel chunk then CFG batch chunk. |
| Scheduler state | timesteps/alphas | `[steps]` timesteps; alpha tables length 1000 | v-pred DDPM step. |
| VAE decode input | final latents | `[B,4,128,128] / 0.13025` | decode to `[B,3,1024,1024]`. |

CPU/data-pipeline work: prompt tokenization, control image resize/binning,
PIL/NumPy conversion, optional safety/postprocess. GPU/runtime work: VAE encode
for control image, transformer/ControlNet denoising, CFG/rescale, DDPM step,
VAE decode. First Dinoml slice should accept precomputed text embeddings and
optionally precomputed control latents to isolate ControlNet/transformer parity.

## 5. Operator coverage checklist

Tensor/layout ops:

- NCHW image/latent tensors and `[B,N,C]` transformer tokens.
- `repeat_interleave`, CFG `cat`, `chunk`, list/tuple residual stack and pop.
- Patchify `Conv2d -> flatten(2) -> transpose(1,2)`.
- Unpatchify `reshape(B,h,w,p,p,C) -> einsum("nhwpqc->nchpwq") -> reshape`.
- Text concat, mask concat, `unsqueeze(2).bool`, `torch.where` against learned
  padding.
- Multi-ControlNet elementwise residual slot reduction.

Convolution/downsample/upsample ops:

- VAE encode/decode SD-style AutoencoderKL with 4 latent channels and scale
  factor 8.
- Transformer patch embed: `Conv2d(4 -> 1408, 2x2, stride=2, bias=True)`.
- ControlNet uses the same patch embed for `hidden_states` and
  `controlnet_cond`, plus zero Linear `input_block`; no RGB conv conditioner.

GEMM/linear ops:

- T5 projection `Linear(2048 -> 8192)`, FP32 SiLU, `Linear(8192 -> 1024)`.
- Timestep embedding MLP, optional size/style/text projection MLP.
- 40 base transformer blocks and 19 ControlNet blocks with Q/K/V/out linear
  projections, cross-attention K/V, feed-forward, skip linear in later blocks.
- Zero Linear heads for 19 ControlNet residual slots.
- Output projection `Linear(1408 -> 2*2*8=32)`.

Attention primitives:

- Image-token self-attention with 16 heads, head dim 88, Q/K LayerNorm, RoPE on
  Q and K.
- Text cross-attention from image tokens to 333 text tokens, Q/K LayerNorm, RoPE
  applied to query only in source because cross-attention keys are text.
- T5 attention pool in the conditioning embedding path.
- Primary implementation is `HunyuanAttnProcessor2_0` in
  `attention_processor.py`; fused QKV is available through
  `FusedHunyuanAttnProcessor2_0` but not required for parity.

Normalization and adaptive conditioning:

- `AdaLayerNormShift`: FP32 LayerNorm plus timestep-dependent shift.
- FP32 LayerNorm in cross-attention/feed-forward and skip paths.
- `AdaLayerNormContinuous` output norm.
- Bert/T5 external LayerNorms if text encoders enter scope.

Scheduler and guidance arithmetic:

- DDPM set_timesteps leading spacing, v-pred `x0 = sqrt(alpha)*sample -
  sqrt(beta)*model_output`, mean update, optional variance noise.
- CFG `uncond + guidance_scale * (text - uncond)`.
- Optional guidance rescale std reduction across non-batch axes.

## 6. Denoiser/model breakdown

ControlNet forward:

```text
hidden_states [B,4,Hl,Wl]
  -> PatchEmbed Conv2d(4,1408,2,stride=2) -> [B,N,1408]
controlnet_cond [B,4,Hl,Wl]
  -> same PatchEmbed -> zero Linear input_block -> add to hidden tokens
timestep + T5 pooled text + optional size/style -> temb [B,1408]
T5 tokens [B,256,2048] -> PixArtAlphaTextProjection -> [B,256,1024]
Bert/T5 concat -> [B,333,1024], masked padding replacement
19 HunyuanDiTBlock(skip=False) blocks
each block output -> zero Linear(1408 -> 1408) -> conditioning scale
return 19 token residuals
```

Base transformer forward:

```text
latents [B,4,Hl,Wl] -> PatchEmbed -> [B,N,1408]
same temb and text projection/mask replacement as ControlNet
for layers 0..40:
  layers 0..18 save skip tensors after block
  layers > 20 pop skip; if ControlNet active, skip += control residual
  skip blocks concatenate [hidden, skip] -> LayerNorm(2816) -> Linear(2816 -> 1408)
final AdaLayerNormContinuous + Linear(1408 -> patch_size^2*out_channels)
unpatchify -> [B,8,Hl,Wl]; pipeline keeps first 4 channels
```

`HunyuanDiTBlock`:

```text
optional skip concat/linear
AdaLayerNormShift(temb) -> self-attention(QK LayerNorm + RoPE) -> residual
FP32LayerNorm -> text cross-attention(QK LayerNorm, query RoPE only) -> residual
FP32LayerNorm -> FeedForward GELU approximate -> residual
```

## 7. Attention requirements

The required attention processor is `HunyuanAttnProcessor2_0`. It projects
Q/K/V, reshapes to `[B,heads,seq,head_dim]`, applies optional Q/K LayerNorm,
applies RoPE to query and, for self-attention only, key, then calls
`torch.nn.functional.scaled_dot_product_attention` with `dropout_p=0` and
`is_causal=false`.

Required variants:

- Self-attention over image patch tokens: default 4096 tokens, heads 16, head
  dim 88, no mask.
- Cross-attention from image tokens to concatenated Bert+T5 text tokens: query
  length 4096, key/value length 333, heads 16, head dim 88.
- T5 pooling attention inside `HunyuanCombinedTimestepTextSizeStyleEmbedding`.
- Fused QKV/KV projections are source-supported by `fuse_qkv_projections()` and
  `FusedHunyuanAttnProcessor2_0`; keep explicit projections first.

`attention_dispatch.py` is not the primary path for this target. A Dinoml
flash-style provider is valid under guards: dense noncausal attention, supported
head dim 88 and dtype, QK norm and RoPE performed before attention, no active
added-KV/IP-Adapter/PAG processor, and no unsupported attention mask. Native
SDPA behavior defines parity.

## 8. Scheduler and denoising-loop contract

`DDPMScheduler.set_timesteps(num_inference_steps, device)` uses leading spacing
for sampled configs:

```text
step_ratio = num_train_timesteps // num_inference_steps
timesteps = (arange(steps) * step_ratio).round()[::-1] + steps_offset
```

For v-prediction, source step computes:

```text
pred_original_sample = sqrt(alpha_prod_t) * sample - sqrt(beta_prod_t) * model_output
pred_prev = coeff_x0 * pred_original_sample + coeff_xt * sample + variance_noise
```

`scale_model_input` is identity for DDPM. The pipeline passes only the first
half of learned-sigma output to the scheduler, so `variance_type=fixed_small`
does not use model-predicted variance. `eta` is accepted by the pipeline helper
only for schedulers whose step signature uses it; DDPM ignores it.

Keep `set_timesteps`, timestep tables, stochastic generator state, and callback
mutation out of the first compiled graph. Candidate compiled pieces are CFG,
guidance rescale, and the deterministic part of the DDPM v-pred step.

## 9. Position, timestep, and custom math

- Image RoPE is generated with `get_2d_rotary_pos_embed` using head dim 88 and
  grid `(height/8/patch_size, width/8/patch_size)`. At 1024 square this is
  `[64,64]`.
- `get_resize_crop_region_for_grid` builds the crop coordinates for the RoPE
  grid against base size `512/8/patch_size = 32`.
- Timestep embedding uses `Timesteps(256, flip_sin_to_cos=True)` followed by
  `TimestepEmbedding`.
- `HunyuanCombinedTimestepTextSizeStyleEmbedding` adds timestep embedding to a
  projected extra condition. For v1.1/source default this extra condition
  concatenates pooled T5 projection, six size IDs projected to 1536 dims, and a
  style embedding. For v1.2 it uses only pooled T5 projection.
- Text padding is not an attention mask in the transformer blocks; source
  replaces padded text embeddings with a learned padding parameter before
  cross-attention.

Precompute RoPE and scheduler tables per resolution/step count. Text projections
and pooled extra conditioning can be cached per prompt/timestep policy, but
`temb` depends on the current timestep.

## 10. Preprocessing and input packing

Bert prompt path:

```text
BertTokenizer(max_length=77, padding=max_length, truncation=True)
BertModel(...).last_hidden_state -> [B,77,1024]
attention mask -> [B,77]
```

T5 prompt path:

```text
T5Tokenizer(max_length=256, padding=max_length, truncation=True)
T5EncoderModel(...).last_hidden_state -> [B,256,2048]
attention mask -> [B,256]
```

CFG duplicates prompt embeddings as `[negative, positive]`. Control image
preprocessing uses `VaeImageProcessor.preprocess` for non-tensor inputs, repeats
to effective batch, duplicates for CFG because `guess_mode=false`, casts to
pipeline dtype/device, VAE-encodes, samples from the latent distribution, and
multiplies by 0.13025. Tensor control images bypass generic preprocessing but
are still repeated/cast and VAE-encoded.

There is no pipeline-level latent packing. Patchify/unpatchify live inside the
ControlNet and base transformer. Multi-ControlNet expects `control_image` as a
list and creates a list of encoded control latents.

## 11. Graph rewrite / lowering opportunities

### Rewrite: Hunyuan patch embed as strided patch GEMM

Source pattern: `Conv2d(4 -> 1408, kernel=2, stride=2)` followed by
`flatten(2).transpose(1,2)`.

Replacement: extract non-overlapping 2x2x4 patches and GEMM to 1408 channels.

Preconditions: patch size 2, stride 2, no padding, NCHW source layout, even
latent height/width, bias preserved. Weight transform flattens OIHW to
`[16,1408]` with source patch order `C,pH,pW`. Failure cases: non-2 patch
configs or NHWC without matching activation and weight transforms.

Parity test: random `[2,4,128,128]` through Diffusers `PatchEmbed`.

### Rewrite: ControlNet residual contract

Source pattern: 19 early ControlNet block outputs each pass through zero
`Linear(1408 -> 1408)` and are multiplied by `conditioning_scale`, then added
to popped transformer skips.

Replacement: explicit residual-slot ABI with scale fused into the zero-linear
epilogue or into the skip add.

Preconditions: `transformer_num_layers=40`, residual slot count 19, matching
token shape and dtype, same CFG batch order as base transformer. Failure cases:
changed layer count, Multi-ControlNet heterogeneous shapes, or callback-mutated
conditioning scale.

### Rewrite: learned-sigma channel discard

Source pattern: transformer returns `[B,8,H,W]`; pipeline does
`noise_pred, _ = noise_pred.chunk(2, dim=1)`.

Replacement: lower output projection only for first 4 latent channels when
variance is unused, or materialize full output for exact state-dict parity.

Preconditions: scheduler variance is not learned/learned_range and no caller
observes latent output before chunk. Failure cases: future scheduler uses model
variance or output_type latent expects full raw output.

### Rewrite: DDPM deterministic v-pred step

Source pattern: alpha table loads, v-pred to x0, linear combination with sample,
optional variance.

Replacement: fused pointwise update with precomputed scalar coefficients per
timestep.

Preconditions: `variance_type=fixed_small`, deterministic validation disables
variance noise or uses supplied generator/noise, no threshold/clip. Failure
cases: stochastic image parity, learned variance, custom timesteps not admitted.

### Rewrite: guarded NHWC VAE/patch conv islands

Source pattern: VAE Conv2d/GroupNorm/SiLU and patch Conv2d run on NCHW.

Replacement: local NHWC conv islands with NCHW boundaries.

Preconditions: all consumers inside island rewritten; Conv weights OIHW->HWIO;
GroupNorm channel axis rewritten from dim 1 to last dim; unpatchify and token
flatten order preserved. Failure cases: transformer token ABI, VAE sampling,
or scheduler/CFG expecting NCHW maps outside the island.

## 12. Kernel fusion candidates

Highest priority:

- Patch embed Conv2d + flatten/transpose lowering for both noisy and control
  latents.
- QKV projection + QK LayerNorm + RoPE + SDPA + output projection for Hunyuan
  self-attention.
- Cross-attention projection and SDPA for image tokens attending to 333 text
  tokens.
- AdaLayerNormShift / FP32LayerNorm plus residual epilogues in
  `HunyuanDiTBlock`.
- ControlNet zero Linear + conditioning scale + skip add.

Medium priority:

- Feed-forward GELU-approximate MLP fusion with residual add.
- T5 projection and pooled conditioning MLPs for prompt-cache integration.
- CFG and guidance-rescale reductions.
- DDPM v-pred deterministic step.
- VAE encode/decode Conv2d + GroupNorm + SiLU blocks.

Lower priority:

- Multi-ControlNet residual accumulation specialization.
- Fused QKV weight transforms using `FusedHunyuanAttnProcessor2_0`.
- Stochastic DDPM variance branch.
- Bert/T5 encoder execution inside Dinoml.

## 13. Runtime staging plan

Stage 1: Parse v1.2 base configs from `H:/configs` and one public ControlNet
config. Load base transformer, ControlNet, VAE, and scheduler metadata. Accept
external Bert/T5 embeddings and masks.

Stage 2: Validate patch embed/unpatchify and one `HunyuanDiTBlock` with random
tokens, text, RoPE, and temb.

Stage 3: Implement `HunyuanDiT2DControlNetModel` forward for one fixed
resolution with externally supplied control latents.

Stage 4: Implement base `HunyuanDiT2DModel` residual injection and learned-sigma
channel chunk. Validate one denoiser step with fixed ControlNet residuals.

Stage 5: Add VAE control-image encode and VAE final decode boundaries.

Stage 6: Add DDPM v-pred scheduler with deterministic one-step parity, then
full short loop with scheduler in host control.

Stage 7: Add CFG and guidance-rescale parity.

Stage 8: Add v1.1 style+size conditioning variant and Multi-ControlNet residual
summation.

Stage 9: Optimize attention, norm, MLP, patch, and VAE kernels under guards.

## 14. Parity and validation plan

- Config admission tests for v1.1/v1.2 base and ControlNet JSONs, including
  omitted/default `use_style_cond_and_image_meta_size`.
- Patch embed parity for noisy and control latents at 1024 and one non-square
  supported resolution.
- RoPE table parity for `[64,64]`, `[80,48]`, and binned resolutions.
- Text concat/mask padding replacement parity with synthetic masks.
- `HunyuanCombinedTimestepTextSizeStyleEmbedding` parity for v1.1 and v1.2
  conditioning modes.
- Single `HunyuanDiTBlock` parity for self-attention, cross-attention, and skip
  variants.
- Full ControlNet forward parity: 19 residual slots, order, shape, dtype, and
  scale.
- Full base transformer forward parity with supplied ControlNet residuals.
- Learned-sigma chunk parity.
- DDPM `set_timesteps` and one v-pred step parity.
- VAE control encode and final decode parity.
- Short deterministic denoising loop with externally supplied embeddings and
  fixed control latents.
- Suggested tolerances: fp32 scheduler/pointwise `rtol=1e-5, atol=1e-6`;
  transformer/VAE fp16 or bf16 start at `rtol=2e-2, atol=2e-2`, then tighten per
  provider.

## 15. Performance probes

- ControlNet forward time by resolution, batch, and dtype.
- Base transformer forward with and without ControlNet residual injection.
- Attention backend comparison for self-attention sequence lengths 2304, 4096,
  and 6400.
- Cross-attention cost with 333 text tokens.
- Patch embed/unpatchify bandwidth and layout sensitivity.
- VAE encode cost for control image versus VAE decode cost for final image.
- Full denoising step split: ControlNet, transformer, CFG/rescale, scheduler.
- Multi-ControlNet scaling with two and three controls.
- VRAM/workspace peak for 1024 and 1280 square.
- NCHW faithful path versus guarded NHWC VAE/patch conv islands.

## 16. Scope boundary and separate candidates

Separate candidate reports, not ignored:

- `hunyuandit_legacy`: base HunyuanDiT text-to-image parity without ControlNet.
- `controlnet_hunyuandit_multi`: multiple ControlNets, list inputs, per-control
  scales, residual-slot summation.
- `hunyuandit_v1_1_vs_v1_2`: style/size conditioning default differences and
  scheduler beta differences.
- `hunyuandit_pag`: PAG attention processor mutation.
- `hunyuandit_lora_adapters`: LoRA/adapter weight mutation.
- `hunyuan_image`: newer Hunyuan Image 2.1 family, already separately audited;
  related by brand only for first-slice implementation.
- `scheduler_ddpm_stochastic`: full DDPM reverse variance/noise semantics beyond
  deterministic one-step validation.

Ignored/out of scope for this audit:

- Safety checker and NSFW filtering.
- Callback mutation and interactive interrupt.
- XLA, NPU, MPS, Flax, ONNX, and multi-GPU/context-parallel paths.
- Training, losses, dropout behavior, and gradient checkpointing.
- Backend-specific attention providers except as optimization candidates.

## 17. Final implementation checklist

- [ ] Parse v1.2 base and Hunyuan ControlNet component configs.
- [ ] Record v1.1 omitted defaults, especially style/size conditioning.
- [ ] Load base transformer, ControlNet, VAE, and scheduler weights.
- [ ] Accept external Bert/T5 embeddings and masks.
- [ ] Implement Hunyuan patch embed and unpatchify parity.
- [ ] Implement T5 projection, text concat, mask padding replacement.
- [ ] Implement timestep plus pooled text and optional size/style conditioning.
- [ ] Implement `HunyuanDiTBlock` self-attention, cross-attention, FFN, and skip paths.
- [ ] Implement `HunyuanDiT2DControlNetModel` 19-slot residual output.
- [ ] Implement ControlNet residual injection into base transformer skips.
- [ ] Implement learned-sigma output channel chunk.
- [ ] Implement DDPM v-pred scheduler table and deterministic step.
- [ ] Add VAE control encode and VAE final decode boundaries.
- [ ] Add CFG and guidance-rescale parity.
- [ ] Add one-step and short-loop HunyuanDiT ControlNet parity tests.
- [ ] Add Multi-ControlNet residual accumulation after single-ControlNet parity.
- [ ] Add guarded attention/norm/MLP/patch/VAE fusion probes.
