# Diffusers Hunyuan Image Operator and Integration Report

Target slug: `hunyuan_image`

## 1. Source basis

```text
Diffusers commit/version:
  X:/H/diffusers at b3a515080752a3ba7ca92161e25530c7f280f629.

Model id(s):
  hunyuanvideo-community/HunyuanImage-2.1-Diffusers
  hunyuanvideo-community/HunyuanImage-2.1-Refiner-Diffusers
  Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers, as older same-brand DiT inventory only.

Config sources:
  H:/configs/hunyuanvideo-community/HunyuanImage-2.1-Diffusers/model_index.json
  H:/configs/hunyuanvideo-community/HunyuanImage-2.1-Diffusers/transformer/config.json
  H:/configs/hunyuanvideo-community/HunyuanImage-2.1-Diffusers/vae/config.json
  H:/configs/hunyuanvideo-community/HunyuanImage-2.1-Diffusers/scheduler/scheduler_config.json
  H:/configs/hunyuanvideo-community/HunyuanImage-2.1-Diffusers/text_encoder/config.json
  H:/configs/hunyuanvideo-community/HunyuanImage-2.1-Diffusers/text_encoder_2/config.json
  H:/configs/hunyuanvideo-community/HunyuanImage-2.1-Diffusers/guider/guider_config.json
  H:/configs/hunyuanvideo-community/HunyuanImage-2.1-Diffusers/ocr_guider/guider_config.json
  H:/configs/hunyuanvideo-community/HunyuanImage-2.1-Refiner-Diffusers/*
  H:/configs/Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers/*

Pipeline files inspected:
  X:/H/diffusers/src/diffusers/pipelines/hunyuan_image/pipeline_hunyuanimage.py
  X:/H/diffusers/src/diffusers/pipelines/hunyuan_image/pipeline_hunyuanimage_refiner.py
  X:/H/diffusers/src/diffusers/pipelines/hunyuan_image/pipeline_output.py
  X:/H/diffusers/src/diffusers/pipelines/hunyuandit/pipeline_hunyuandit.py
  X:/H/diffusers/src/diffusers/pipelines/controlnet_hunyuandit/pipeline_hunyuandit_controlnet.py
  X:/H/diffusers/src/diffusers/pipelines/pag/pipeline_pag_hunyuandit.py

Model files inspected:
  X:/H/diffusers/src/diffusers/models/transformers/transformer_hunyuanimage.py
  X:/H/diffusers/src/diffusers/models/transformers/hunyuan_transformer_2d.py
  X:/H/diffusers/src/diffusers/models/autoencoders/autoencoder_kl_hunyuanimage.py
  X:/H/diffusers/src/diffusers/models/autoencoders/autoencoder_kl_hunyuanimage_refiner.py
  X:/H/diffusers/src/diffusers/models/controlnets/controlnet_hunyuan.py

Scheduler/processors/helpers inspected:
  X:/H/diffusers/src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py
  X:/H/diffusers/src/diffusers/guiders/adaptive_projected_guidance_mix.py
  X:/H/diffusers/src/diffusers/guiders/guider_utils.py
  X:/H/diffusers/src/diffusers/image_processor.py
  Shared attention/embedding/norm files through imports:
    attention.py, attention_processor.py, attention_dispatch.py, embeddings.py, normalization.py.

External component configs inspected:
  Qwen2_5_VLForConditionalGeneration / Qwen2Tokenizer
  T5EncoderModel / ByT5Tokenizer

Any missing files or assumptions:
  Local cache initially had only model_index.json for HunyuanImage 2.1; small official config files were fetched with
  `hf download`. No official config fetch blockers remained for inspected JSON files. This report focuses on the
  text-to-image base `HunyuanImagePipeline`; refiner, older HunyuanDiT, ControlNet, PAG, and Hunyuan Video are separate
  candidates. Ignored per task: XLA/NPU/MPS/Flax/ONNX, safety/NSFW, training/loss/dropout/gradient checkpointing,
  multi-GPU/context parallel, callbacks, and interrupt behavior.
```

## 2. Pipeline and component graph

`HunyuanImagePipeline` wires `Qwen2_5_VLForConditionalGeneration`,
`Qwen2Tokenizer`, `T5EncoderModel`, `ByT5Tokenizer`,
`HunyuanImageTransformer2DModel`, `AutoencoderKLHunyuanImage`,
`FlowMatchEulerDiscreteScheduler`, and optional `AdaptiveProjectedMixGuidance`
instances for normal and OCR/glyph prompts.

```text
prompt
  -> Qwen tokenizer/template/drop-prefix -> Qwen hidden states + mask
  -> quoted glyph extraction -> ByT5 embeddings + mask, or zero glyph tensors
  -> NCHW latent noise [B,64,H/32,W/32]
  -> denoising loop:
       HunyuanImageTransformer2DModel(latents, qwen text, byt5 text, masks, timestep, optional guidance)
       AdaptiveProjectedMixGuidance combines conditional/unconditional predictions when enabled
       FlowMatchEulerDiscreteScheduler.step
  -> AutoencoderKLHunyuanImage decode(latents / scaling_factor)
  -> VaeImageProcessor postprocess
```

Required first-slice components are the transformer denoiser, FlowMatch Euler,
prompt embeddings as external cached inputs, APG/CFG arithmetic or a disabled
guider mode, and VAE decode. Independently cacheable stages are Qwen prompt
embeds/masks, ByT5 glyph embeds/masks, scheduler timesteps/sigmas, random
initial latents, and VAE decoded images.

Separate candidate reports:

| Candidate | Classes/files | Delta |
| --- | --- | --- |
| `hunyuan_image_refiner` | `HunyuanImageRefinerPipeline`, `AutoencoderKLHunyuanImageRefiner`, same transformer class | Image-to-image refinement with VAE encode, 3D/temporal-looking refiner VAE, transformer `in_channels=128`, `out_channels=64`, `guidance_embeds=true`. |
| `hunyuandit_legacy` | `HunyuanDiTPipeline`, `HunyuanDiT2DModel` | Older image DiT family: BERT + T5, 4-channel AutoencoderKL, DDPMScheduler v-pred, learned sigma. |
| `hunyuandit_controlnet` | `HunyuanDiTControlNetPipeline`, `HunyuanDiT2DControlNetModel`, `HunyuanDiT2DMultiControlNetModel` | Control image preprocessing and transformer block residuals for legacy HunyuanDiT. |
| `hunyuandit_pag` | `HunyuanDiTPAGPipeline`, PAG Hunyuan attention processors | Perturbed attention guidance for legacy HunyuanDiT. |
| `hunyuan_image_lora_adapters` | `HunyuanImageTransformer2DModel` inherits `PeftAdapterMixin`; `apply_lora_scale` forward decorator | Runtime adapter mutation for transformer linear/attention weights; no pipeline-specific loader mixin was seen in `hunyuan_image`. |
| `hunyuan_video` / `hunyuan_video_1_5` | `pipelines/hunyuan_video*`, `HunyuanVideo*Transformer3DModel`, Hunyuan video VAEs | Separate video tensor rank, temporal patching, and video VAE family. |

No base `hunyuan_image` img2img, inpaint, depth2img, upscaling, IP-Adapter,
T2I-Adapter, or GLIGEN pipeline classes were found in the HunyuanImage folder.

## 3. Important config dimensions

Representative configs:

| Repo | Pipeline | Transformer | VAE | Scheduler |
| --- | --- | --- | --- | --- |
| `HunyuanImage-2.1-Diffusers` | `HunyuanImagePipeline` | `HunyuanImageTransformer2DModel`, 28 heads, head dim 128, 20 dual + 40 single layers, 2 refiner layers | `AutoencoderKLHunyuanImage`, 64 latent channels, compression 32 | `FlowMatchEulerDiscreteScheduler`, shift 5.0 |
| `HunyuanImage-2.1-Refiner-Diffusers` | `HunyuanImageRefinerPipeline` | Same class, 26 heads, head dim 128, `in_channels=128`, `out_channels=64`, `guidance_embeds=true`, 3-axis RoPE config | `AutoencoderKLHunyuanImageRefiner`, 32 latent channels, spatial compression 16, temporal compression 4 | FlowMatch Euler, shift 4.0 |
| `HunyuanDiT-v1.2-Diffusers` | `HunyuanDiTPipeline` | `HunyuanDiT2DModel`, 16 heads, head dim 88, 40 layers, patch 2, 4 latent channels, learned sigma | `AutoencoderKL`, 4 latent channels, scale 0.13025 | `DDPMScheduler`, v-pred |

Base HunyuanImage 2.1 dimensions:

| Field | Value | Source |
| --- | ---: | --- |
| VAE latent channels | 64 | VAE config |
| VAE spatial compression | 32 | VAE config / pipeline `vae_scale_factor` |
| VAE scaling factor | 0.75289 | VAE config |
| Default image size | 2048 x 2048 | pipeline default `64 * 32` |
| Default latent map | `[B,64,64,64]` at default size | inferred from source |
| Transformer patch size | `[1,1]` | transformer config |
| Transformer image tokens | 4096 at default size | inferred from `[64,64]` latent map and patch 1 |
| Inner dim | 3584 | 28 * 128 |
| Qwen text dim | 3584 | text encoder config / transformer `text_embed_dim` |
| ByT5 text dim | 1472 | text_encoder_2 config / transformer `text_embed_2_dim` |
| ByT5 projection | 1472 -> 2048 -> 2048 -> 3584 | source |
| Qwen max prompt length after template drop | 1000 | pipeline source |
| ByT5 glyph max length | 128 | pipeline source |
| QK norm | RMS norm | transformer config |
| RoPE axes dim | `[64,64]`, theta 256 | transformer config |
| Guidance embeds | false | transformer config |
| Guider | `AdaptiveProjectedMixGuidance`, enabled | guider configs |
| Dtype metadata | Qwen bfloat16, ByT5 float32 | external configs |

Scheduler support is FlowMatch-shaped for HunyuanImage 2.1. The pipeline passes
custom sigmas by default as `np.linspace(1.0, 0.0, steps + 1)[:-1]`, then lets
FlowMatch Euler apply `shift=5.0` and append a terminal zero sigma. Recommended
first Dinoml scheduler slice is FlowMatch Euler static shifting with custom
sigmas and non-stochastic step.

## 3a. Family variation traps

- Base HunyuanImage uses 64-channel VAE latents and compression 32, unlike SD3
  and Flux 16-channel VAEs.
- Transformer `in_channels=64` is the source latent channel count; it is not
  Flux-style packed 2x2 tokens.
- Patch size is `[1,1]`, so patchify is a Conv2d token projection over each
  latent cell, not a spatial downsampling patch.
- `height` and `width` must be divisible by `vae_scale_factor * 2`, i.e. 64 for
  the base pipeline, because the VAE compression is 32 and transformer/latent
  dimensions need even guarded handling.
- HunyuanImage base uses Qwen2.5-VL hidden states and optional ByT5 glyph
  embeddings; older HunyuanDiT uses BERT and T5 instead.
- Base transformer config has `guidance_embeds=false`; refiner has
  `guidance_embeds=true` and expects a distilled guidance tensor.
- Guidance is mediated by `AdaptiveProjectedMixGuidance`, not only simple
  concatenated CFG. It may run one or two denoiser calls depending on enabled
  range and APG/CFG state.
- ByT5 tokens are reordered with Qwen tokens inside the transformer:
  valid ByT5, valid Qwen, invalid ByT5, invalid Qwen.
- Attention masks are real in the main transformer path; flash-style lowering
  must support or guard the text padding mask.
- NHWC is only an optimization candidate around VAE/patch Conv2d regions. The
  semantic source contract is NCHW latent maps into and out of the transformer.
- Refiner VAE source uses Conv3d-like components and a temporal compression
  config even though it refines images; keep it separate.

## 4. Runtime tensor contract

For default 2048x2048 base generation and batch `B`:

| Boundary | Tensor | Shape/layout | Notes |
| --- | --- | --- | --- |
| Qwen embeds | `prompt_embeds` | `[B,1000,3584]` | Hidden state `-(skip+1)`, after dropping 34 template tokens. |
| Qwen mask | `prompt_embeds_mask` | `[B,1000]` | Bool inside transformer. |
| ByT5 glyph embeds | `prompt_embeds_2` | `[B,128,1472]` | Zero-filled when prompt has no quoted glyph text. |
| ByT5 mask | `prompt_embeds_mask_2` | `[B,128]` | Zero mask for no glyph text. |
| Latent noise | `latents` | `[B,64,H/32,W/32]`, NCHW | Default `[B,64,64,64]`. |
| RoPE | `(cos,sin)` | `[tokens, rope_dim/2]` pieces | Built from latent map size and `rope_axes_dim`. |
| Patch tokens | hidden stream | `[B,(H/32)*(W/32),3584]` | Conv2d kernel/stride `[1,1]`, flatten spatial. |
| Text tokens | encoder stream | `[B,<=1128,3584]` | Qwen token refiner plus optional ByT5 projection and reorder. |
| Attention mask | `attention_mask` | `[B,1,1,image_seq+text_seq]` | Image span padded true, text span from reordered mask. |
| Denoiser output | `noise_pred` | `[B,64,H/32,W/32]`, NCHW | Unpatchify back to latent map. |
| Scheduler state | sigmas/timesteps | `[steps+1]` sigmas, `[steps]` timesteps | Step index/begin index explicit. |
| VAE decode input | latent map | `[B,64,H/32,W/32]` | Pipeline uses `latents / 0.75289`; no shift factor. |
| VAE decode output | image tensor | `[B,3,H,W]`, NCHW | Postprocessed by `VaeImageProcessor`. |

CPU/data-pipeline work: prompt templating, tokenization, quote/glyph extraction,
Qwen/ByT5 execution if embeddings are not supplied, PIL/NumPy postprocess.
GPU/runtime work: denoiser forward, guider arithmetic, scheduler step, VAE
decode. First Dinoml slice should accept precomputed text embeddings and masks.

## 5. Operator coverage checklist

Tensor/layout ops:

- NCHW latent allocation and optional user latent cast.
- Conv2d patch projection followed by `flatten(2).transpose(1,2)`.
- Text mask cast to bool, per-row valid/invalid token indexing, concat, stack.
- Sequence concat/split for image/text streams.
- View/reshape/permute unpatchify for 2D and source-supported 3D shapes.
- Repeat/view prompt duplication for `num_images_per_prompt`.

Convolution/downsample/upsample ops:

- `HunyuanImagePatchEmbed`: Conv2d(64 -> 3584, 1x1, stride 1) for base.
- VAE Conv2d/GroupNorm/SiLU ResNet blocks with channels
  128/256/512/512/1024/1024.
- VAE custom downsample: Conv2d then 2x2 spatial-to-channel rearrange plus
  shortcut grouped mean.
- VAE custom upsample: Conv2d to `4*out_channels` then channel-to-2x2 spatial
  rearrange plus repeated shortcut.
- VAE mid self-attention via Conv2d Q/K/V and SDPA over `H*W`.

GEMM/linear ops:

- Text token refiner projection 3584 -> 3584.
- ByT5 projection 1472 -> 2048 -> 2048 -> 3584.
- Timestep/pooled text projection MLPs inside `CombinedTimestepTextProjEmbeddings`
  and `HunyuanImageCombinedTimeGuidanceEmbedding`.
- Dual-stream and single-stream Q/K/V/add-Q/add-K/add-V projections.
- Feed-forward GELU approximate and single-stream parallel MLP projections.
- Output projection `Linear(3584 -> 64)` for patch size 1.

Attention primitives:

- Masked text token self-attention in `HunyuanImageTokenRefiner`.
- Joint image/text attention in dual-stream blocks.
- Joint attention over concatenated image/text stream in single-stream blocks.
- QK RMSNorm and RoPE on image token spans.
- Base attention path dispatches through `dispatch_attention_fn`.

Normalization and adaptive conditioning:

- LayerNorm, RMSNorm through attention `qk_norm`, AdaLayerNormZero,
  AdaLayerNormZeroSingle, AdaLayerNormContinuous, custom `HunyuanImageAdaNorm`,
  VAE GroupNorm.

Scheduler and guidance arithmetic:

- FlowMatch Euler static shift, custom sigmas, step index, `sample + dt *
  model_output`.
- APG/CFG: conditional/unconditional denoiser calls, momentum buffer update,
  normalized guidance, optional guidance rescale.

## 6. Denoiser/model breakdown

Top-level `HunyuanImageTransformer2DModel.forward`:

```text
hidden_states [B,64,Hl,Wl]
  -> RoPE from source latent map
  -> timestep (+ optional timestep_r/guidance) embedding [B,3584]
  -> Conv2d patch embed -> image tokens [B,Hl*Wl,3584]
  -> Qwen token refiner with timestep/pooled text -> [B,1000,3584]
  -> optional ByT5 projection and valid/invalid token reorder -> [B,1128,3584]
  -> 20 HunyuanImageTransformerBlock dual-stream blocks
  -> 40 HunyuanImageSingleTransformerBlock blocks
  -> AdaLayerNormContinuous + Linear -> [B,image_seq,64]
  -> unpatchify -> [B,64,Hl,Wl]
```

Dual-stream block:

```text
image/text AdaLayerNormZero
image QKV + text added-QKV
QK RMSNorm
RoPE on image span only
concat image/text -> masked noncausal attention -> split
gated residual attention on image and text
LayerNorm + adaptive scale/shift
FeedForward GELU approximate
gated residual MLP on both streams
```

Single-stream block:

```text
concat image/text
AdaLayerNormZeroSingle -> normalized stream + gate
parallel attention input and GELU MLP input
attention with image/text split semantics
concat(attn, mlp) -> Linear -> gated residual
split image/text
```

Text refiner:

```text
masked mean pool text -> CombinedTimestepTextProjEmbeddings
Linear text_embed_dim -> inner_dim
2 IndividualTokenRefinerBlock layers:
  LayerNorm -> masked self-attention -> gated residual
  LayerNorm -> FeedForward(linear-silu) -> gated residual
```

## 7. Attention requirements

`HunyuanImageAttnProcessor` is the target attention processor. It calls
`dispatch_attention_fn` from `attention_dispatch.py`, not the older
`attention_processor.py` SDPA-only processors. Shapes after projection are
`[B,seq,heads,head_dim]`; base config is heads 28, head dim 128.

Required variants:

- Token-refiner self-attention over Qwen tokens with a square mask derived from
  text masks.
- Dual-stream joint image/text attention with added Q/K/V projections.
- Single-stream attention with image/text concatenation before AdaNorm and
  split after output.
- QK RMSNorm on image and added text query/key projections.
- RoPE applied to image query/key tokens, while text tokens are concatenated
  without RoPE.
- Noncausal attention with padding mask; no added K/V IP-Adapter branch in base.

Flash-style Dinoml provider is plausible only under strict guards: dtype/head
dim supported, noncausal mask format supported or mask-free proven for a
subcase, QK norm and RoPE executed before provider call, text/image split sizes
artifact-visible, and no context-parallel/backend-specific branch active. Eager
`dispatch_attention_fn`/native SDPA behavior is the parity path.

## 8. Scheduler and denoising-loop contract

The base pipeline creates sigmas with:

```text
sigmas = linspace(1.0, 0.0, num_inference_steps + 1)[:-1]
retrieve_timesteps(..., sigmas=sigmas)
```

`FlowMatchEulerDiscreteScheduler.set_timesteps` applies static shifting because
`use_dynamic_shifting=false` in sampled configs:

```text
sigma = shift * sigma / (1 + (shift - 1) * sigma)
```

It appends terminal zero sigma. The non-stochastic step is:

```text
dt = sigma_next - sigma
prev_sample = sample + dt * model_output
```

Source default scheduler for base HunyuanImage is FlowMatch Euler with shift
5.0. Refiner uses the same scheduler class with shift 4.0. Recommended first
Dinoml slice: static-shift FlowMatch Euler with custom sigmas, no per-token
timesteps, no stochastic sampling, scheduler loop/index state kept
host-visible.

Guidance is separate from scheduler math. If guider is disabled or not in range,
one conditional transformer call is enough. With APG/CFG enabled, the pipeline
runs conditional and unconditional denoiser calls separately and then combines
predictions through `AdaptiveProjectedMixGuidance`; APG also carries a momentum
buffer after its configured start step.

## 9. Position, timestep, and custom math

- `Timesteps(256, flip_sin_to_cos=True)` plus `TimestepEmbedding` creates
  timestep embeddings.
- Optional `use_meanflow` averages embeddings for `timestep` and `timestep_r`;
  base config omits/uses default false, but source supports it.
- Optional guidance embedding exists in source and is active in the refiner
  config, not base.
- `HunyuanImageRotaryPosEmbed` builds per-axis RoPE from latent map shape and
  `rope_axes_dim`; for base `[64,64]`, RoPE covers each latent cell.
- ByT5 glyph extraction is prompt-string dependent. Quoted text creates
  formatted strings such as `Text "..."`; otherwise ByT5 tensors are zeros.
- Text reordering by masks is data-dependent indexing and should initially stay
  in preprocessing/runtime glue rather than a fused transformer assumption.

## 10. Preprocessing and input packing

Qwen prompt construction:

```text
template(prompt)
Qwen2Tokenizer(max_length=1034, padding=max_length)
Qwen hidden_states[-3]
drop first 34 template tokens -> [B,1000,3584]
```

ByT5 glyph construction:

```text
extract quoted text from prompt
if none: zeros [1,128,1472], zero mask [1,128]
else: ByT5Tokenizer(max_length=128) -> T5EncoderModel -> [1,128,1472]
```

The transformer patchifies internally with Conv2d and unpatchifies internally
with reshape/permute. There is no Flux-style pipeline-level latent packing.
Image postprocessing uses `VaeImageProcessor` after VAE decode.

## 11. Graph rewrite / lowering opportunities

### Rewrite: Conv2d patch embed as 1x1 token GEMM

Source pattern: base `Conv2d(64 -> 3584, kernel=1, stride=1)` followed by
flatten/transpose.

Replacement: per-latent-cell GEMM to `[B,Hl*Wl,3584]`.

Preconditions: patch size exactly `[1,1]`, NCHW source layout known, no padding,
stride 1, weights transformed from OIHW to matrix `[64,3584]` with bias
preserved. Failure cases: refiner 3D patch config, non-1 patch configs, NHWC
layout without matching weight transform.

Parity test: random `[2,64,64,64]` latent map vs Diffusers patch embed.

### Rewrite: joint attention canonicalization

Source pattern: image QKV + text added-QKV, QK norm, image-only RoPE, concat,
masked attention, split, output projections.

Replacement: explicit provider node with plan-visible image/text sequence sizes
and mask contract.

Preconditions: no adapter branch, mask representable by provider, fixed
heads/head_dim, dtype supported, RoPE and QK norm kept outside provider.
Failure cases: unsupported padding mask, changed text reorder, context-parallel
branches, refiner 3D sequence assumptions.

### Rewrite: FlowMatch Euler step

Source pattern: `sample + (sigma_next - sigma) * model_output`.

Replacement: fused pointwise update.

Preconditions: `stochastic_sampling=false`, scalar timestep path, scheduler
state index explicit. Failure cases: per-token timesteps, stochastic branch, or
dynamic schedule mutation hidden from artifact.

### Rewrite: VAE spatial rearrange down/up samples

Source pattern: Conv2d plus 2x2 spatial-channel reshape/permute and shortcut
mean/repeat.

Replacement: layout-aware pixel-unshuffle/pixel-shuffle style primitive plus
conv and residual.

Preconditions: even spatial dimensions, NCHW flatten order preserved, channel
group sizes integral. NHWC optimization needs axis rewrites for GroupNorm
`dim=1`, shortcut mean over group axis, and Conv2d weight transforms. Protect
VAE tiling/slicing with a no-layout-translation guard initially.

## 12. Kernel fusion candidates

Highest priority:

- QKV/add-QKV projection + QK RMSNorm + image RoPE + masked attention provider.
- AdaLayerNormZero/Single/Continuous modulation, gates, and residual epilogues.
- GELU approximate/linear-SiLU feed-forward MLPs.
- FlowMatch Euler pointwise scheduler update.
- VAE decode Conv2d + GroupNorm + SiLU ResNet blocks.

Medium priority:

- 1x1 patch embed and output unpatchify projection.
- ByT5 projection MLP and Qwen token refiner blocks if text encoder/cache work
  enters scope.
- APG/CFG arithmetic, including momentum-buffer update and optional std-based
  guidance rescale.
- VAE custom upsample/downsample rearrange kernels.

Lower priority:

- Refiner 3D VAE kernels.
- Legacy HunyuanDiT DDPMScheduler and learned-sigma path.
- PAG and ControlNet attention/residual mutation.
- LoRA/PEFT runtime mutation.

## 13. Runtime staging plan

Stage 1: Parse base HunyuanImage configs and load transformer/VAE weights.
Accept external Qwen and ByT5 embeddings/masks.

Stage 2: Implement one transformer block parity with random image/text tokens
and masks.

Stage 3: Implement full base transformer forward for `[B,64,64,64]` latents,
with guider disabled first.

Stage 4: Add FlowMatch Euler static-shift scheduler and one denoising-step
parity.

Stage 5: Add VAE decode boundary for 64-channel HunyuanImage VAE.

Stage 6: Add `AdaptiveProjectedMixGuidance` CFG path, then APG momentum path.

Stage 7: Integrate prompt embedding cache contracts for Qwen/ByT5. Text encoder
execution can remain external until Dinoml admits Qwen2.5-VL and ByT5.

Stage 8: Optimize attention/norm/MLP and VAE conv/rearrange kernels.

Stage 9: Separate reports for refiner, legacy HunyuanDiT, ControlNet/PAG, and
Hunyuan Video.

## 14. Parity and validation plan

- Config parse tests for base and refiner JSON files.
- Patch embed parity for Conv2d -> token sequence.
- RoPE generation parity for `[64,64]` latent map.
- Text token reorder parity with synthetic Qwen/ByT5 masks.
- Single `HunyuanImageTransformerBlock` and
  `HunyuanImageSingleTransformerBlock` parity.
- Full transformer forward parity with fixed random embeddings and masks.
- FlowMatch Euler `set_timesteps` and one-step parity for shift 5.0.
- Guider parity for disabled, CFG-only early steps, and APG-enabled later
  steps with fixed predictions.
- VAE decode parity for random `[B,64,64,64]` latents.
- Short deterministic denoising loop with externally supplied embeddings.
- Suggested tolerances: fp32 scheduler/pointwise `rtol=1e-5, atol=1e-6`;
  transformer bf16/fp16 initially `rtol=2e-2, atol=2e-2`, then tighten per
  provider.

## 15. Performance probes

- One transformer forward by image size: 1024, 2048, and non-square multiples
  of 64.
- Attention backend comparison with and without text masks.
- Dual-stream vs single-stream block time.
- Token-refiner and ByT5 projection overhead when embeddings are not cached.
- APG/CFG overhead: one denoiser call vs two calls plus guidance math.
- Flow scheduler overhead and fused pointwise update bandwidth.
- VAE decode throughput and VRAM for 64-channel latents.
- VAE custom up/down rearrange kernels under NCHW vs guarded NHWC.
- Refiner separate probes for 128-channel transformer input and 3D VAE.

## 16. Scope boundary and separate candidates

Separate candidate reports, not ignored:

- `hunyuan_image_refiner`: image refinement, VAE encode/decode, guidance
  embeddings, 128-channel transformer input, refiner VAE.
- `hunyuandit_legacy`: older BERT/T5 + 4-channel AutoencoderKL + DDPMScheduler
  v-pred image pipeline.
- `hunyuandit_controlnet`: ControlNet residuals for legacy HunyuanDiT.
- `hunyuandit_pag`: PAG attention processor mutation for legacy HunyuanDiT.
- `hunyuan_image_lora_adapters`: PEFT adapter and LoRA scale application on the
  transformer.
- `hunyuan_video` and `hunyuan_video_1_5`: video transformer/codec families.
- `flow_match_scheduler_advanced`: dynamic shifting, stochastic sampling,
  Karras/exponential/beta sigmas, per-token timesteps.

Genuinely out of scope for this audit:

- XLA, NPU, MPS, Flax, and ONNX paths.
- Safety checker and NSFW filtering.
- Training, losses, dropout, and gradient checkpointing.
- Multi-GPU/context parallel.
- Callback mutation and interactive interrupt.
- Hunyuan 3D model repos in the config cache.

## 17. Final implementation checklist

- [ ] Parse base HunyuanImage model index and component configs.
- [ ] Load `HunyuanImageTransformer2DModel` weights.
- [ ] Accept external Qwen embeddings/masks and ByT5 glyph embeddings/masks.
- [ ] Implement Conv2d patch embed and unpatchify parity.
- [ ] Implement Qwen token refiner and ByT5 projection/reorder contract.
- [ ] Implement dual-stream `HunyuanImageTransformerBlock` parity.
- [ ] Implement single-stream `HunyuanImageSingleTransformerBlock` parity.
- [ ] Implement QK RMSNorm + image RoPE + masked joint attention.
- [ ] Implement FlowMatch Euler static shift 5.0 with custom sigmas.
- [ ] Implement VAE 64-channel decode boundary and scale factor 0.75289.
- [ ] Add disabled-guider one-step parity.
- [ ] Add CFG/APG `AdaptiveProjectedMixGuidance` parity.
- [ ] Add guarded attention provider lowering.
- [ ] Add VAE conv/rearrange fusion probes.
- [ ] Create separate reports for refiner, legacy HunyuanDiT, ControlNet/PAG,
  LoRA adapters, and Hunyuan Video.
