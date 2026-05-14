# Diffusers PixArt Alpha/Sigma Operator and Integration Report

Target slug: `pixart`

Runtime scope: PixArt Alpha/Sigma text-to-image base pipelines, with prompt
encoding and VAE treated as explicit adjacent stages. First Dinoml slice should
accept externally supplied T5 prompt embeddings, run the PixArt transformer
denoiser on NCHW latents, keep scheduler loop state host-visible, and decode
through AutoencoderKL.

Ignored per user scope: XLA/NPU/MPS, Flax/ONNX, safety/NSFW, training/loss/
dropout/gradient checkpointing, multi-GPU/context parallel, callbacks/interrupt.

## 1. Source basis

```text
Diffusers commit/version:
  X:/H/diffusers at b3a515080752a3ba7ca92161e25530c7f280f629.

Model id(s):
  PixArt-alpha/PixArt-XL-2-512x512
  PixArt-alpha/PixArt-XL-2-1024-MS
  PixArt-alpha/PixArt-Sigma-XL-2-512-MS
  PixArt-alpha/PixArt-Sigma-XL-2-1024-MS
  PixArt-alpha/PixArt-LCM-XL-2-1024-MS, variant scheduler reference

Config sources:
  Local cache model indexes:
    H:/configs/PixArt-alpha/PixArt-XL-2-512x512/model_index.json
    H:/configs/PixArt-alpha/PixArt-XL-2-1024-MS/model_index.json
    H:/configs/PixArt-alpha/PixArt-Sigma-XL-2-1024-MS/model_index.json
    H:/configs/PixArt-alpha/PixArt-Sigma-XL-2-512-MS/model_index.json
    H:/configs/PixArt-alpha/PixArt-LCM-XL-2-1024-MS/model_index.json
  Component configs fetched as raw official HF URLs for inspection only:
    transformer/config.json, scheduler/scheduler_config.json,
    vae/config.json, text_encoder/config.json for the Alpha 512,
    Alpha 1024, and Sigma 1024 repos.
    Sigma 512 transformer/config.json was fetched; scheduler_config.json
    returned 404.

Pipeline files inspected:
  X:/H/diffusers/src/diffusers/pipelines/pixart_alpha/pipeline_pixart_alpha.py
  X:/H/diffusers/src/diffusers/pipelines/pixart_alpha/pipeline_pixart_sigma.py

Model files inspected:
  X:/H/diffusers/src/diffusers/models/transformers/pixart_transformer_2d.py
  X:/H/diffusers/src/diffusers/models/attention.py
  X:/H/diffusers/src/diffusers/models/attention_processor.py
  X:/H/diffusers/src/diffusers/models/embeddings.py
  X:/H/diffusers/src/diffusers/models/normalization.py
  X:/H/diffusers/src/diffusers/models/autoencoders/autoencoder_kl.py

Scheduler/processors/helpers inspected:
  X:/H/diffusers/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py
  X:/H/diffusers/src/diffusers/schedulers/scheduling_ddim.py
  X:/H/diffusers/src/diffusers/schedulers/scheduling_ddpm.py
  X:/H/diffusers/src/diffusers/schedulers/scheduling_euler_discrete.py
  X:/H/diffusers/src/diffusers/image_processor.py

External component configs inspected:
  T5EncoderModel / T5Tokenizer configs from the official PixArt repos.

Any missing files or assumptions:
  Local cache had only top-level model indexes for the official PixArt repos.
  No component configs were written back to H:/configs because this task's owned
  write path is limited to this report. Sigma 512 scheduler_config.json was not
  available at the official raw URL; the first-slice scheduler recommendation
  relies on Alpha 512/1024 and Sigma 1024 DPMSolver configs.
```

## 2. Pipeline and component graph

PixArt Alpha and Sigma are latent text-to-image pipelines using a T5 encoder,
`PixArtTransformer2DModel`, a scheduler, and AutoencoderKL. Both pipelines
declare optional `tokenizer` and `text_encoder`, enabling cached prompt embeds.
The offload sequence is `text_encoder->transformer->vae`.

```text
prompt preprocessing and optional caption cleaning
  -> T5Tokenizer + T5EncoderModel
  -> prompt embeddings and attention masks, duplicated for CFG
  -> latent noise initialization [B,4,H/8,W/8]
  -> denoising loop:
       scheduler.scale_model_input
       PixArtTransformer2DModel NCHW latents + T5 context + mask + timestep
       true CFG arithmetic
       learned-sigma channel trim when out_channels == 2 * in_channels
       scheduler.step
  -> AutoencoderKL decode(latents / scaling_factor)
  -> optional resolution-bin resize/crop and image postprocess
```

Required first-slice components:

| Component | Class | File | Notes |
| --- | --- | --- | --- |
| Pipeline | `PixArtAlphaPipeline` | `pipeline_pixart_alpha.py` | Alpha 512/1024, micro-conditions enabled when `sample_size == 128`. |
| Pipeline | `PixArtSigmaPipeline` | `pipeline_pixart_sigma.py` | Sigma 512/1024/2048 bins; no added resolution/aspect conditions in current pipeline. |
| Text encoder | `T5EncoderModel` | external Transformers | First Dinoml slice can accept cached embeddings. |
| Tokenizer | `T5Tokenizer` | external Transformers | CPU/data-pipeline stage. |
| Denoiser | `PixArtTransformer2DModel` | `pixart_transformer_2d.py` | Patch-token DiT with self-attn, cross-attn, adaLN-single. |
| VAE | `AutoencoderKL` | `autoencoder_kl.py` | Decode required for text-to-image; encode needed for future img2img/inpaint-like variants. |
| Scheduler | `DPMSolverMultistepScheduler` | `scheduling_dpmsolver_multistep.py` | Official Alpha 512/1024 and Sigma 1024 defaults. |

Separate candidate reports:

| Surface | PixArt status | Candidate |
| --- | --- | --- |
| LoRA/runtime adapters | No PixArt-specific loader mixin on these pipeline classes; generic PEFT/LoRA machinery may still apply to module weights outside the base pipeline contract. | `pixart_lora_adapters` only if a concrete PixArt LoRA runtime path is selected. |
| Textual inversion | No PixArt pipeline textual inversion mixin; T5 tokenizer/embedding mutation is not a base feature. | Defer unless a T5 textual-inversion artifact is selected. |
| IP-Adapter | Not wired in PixArt Alpha/Sigma pipelines or `PixArtTransformer2DModel` default attention. | No base candidate; inspect only for third-party forks. |
| ControlNet | No PixArt ControlNet pipeline in the PixArt folder. Related DiT ControlNet code exists for Hunyuan/Sana/Flux/Qwen, not PixArt. | No PixArt base candidate. |
| T2I-Adapter | No family-local PixArt T2I-Adapter pipeline. | No base candidate. |
| GLIGEN | `BasicTransformerBlock` has a generic GLIGEN fuser branch via `cross_attention_kwargs["gligen"]`, but PixArt pipelines do not pass it. | `pixart_gligen_like_block_branch` only if an explicit pipeline/fork uses it. |
| img2img/inpaint/depth/upscale | No PixArt Alpha/Sigma variant files in the inspected folder. | Separate only for external/community variants. |
| LCM | `PixArt-LCM-XL-2-1024-MS` uses `LCMScheduler`. | `pixart_lcm` scheduler/step-distillation variant. |

## 3. Important config dimensions

Representative checkpoint sweep:

| Repo | Pipeline | sample | image default | patch | latent channels | out channels | layers | heads x dim | inner | text width -> cross | added cond | scheduler |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- | --- | --- |
| `PixArt-XL-2-512x512` | Alpha | 64 | 512 | 2 | 4 | 8 | 28 | 16 x 72 | 1152 | 4096 -> 1152 | false by source default | DPMSolver++ epsilon |
| `PixArt-XL-2-1024-MS` | Alpha | 128 | 1024 | 2 | 4 | 8 | 28 | 16 x 72 | 1152 | 4096 -> 1152 | true by source default | DPMSolver++ epsilon |
| `PixArt-Sigma-XL-2-512-MS` | Sigma | 64 | 512 | 2 | 4 | 8 | 28 | 16 x 72 | 1152 | 4096 -> 1152 | false inferred from sample != 128; scheduler missing | unknown official config missing |
| `PixArt-Sigma-XL-2-1024-MS` | Sigma | 128 | 1024 | 2 | 4 | 8 | 28 | 16 x 72 | 1152 | 4096 -> 1152 | false in config | DPMSolver++ epsilon |
| `PixArt-LCM-XL-2-1024-MS` | Alpha | 128 | 1024 | 2 | 4 | 8 | 28 | 16 x 72 | 1152 | 4096 -> 1152 | true by source default | LCMScheduler epsilon |

Common transformer dimensions from official component configs:

| Field | Value |
| --- | --- |
| `activation_fn` | `gelu-approximate` |
| `attention_bias` | true |
| `norm_type` | `ada_norm_single` |
| `norm_elementwise_affine` | false |
| `norm_eps` | `1e-6` |
| `num_embeds_ada_norm` | 1000 |
| `caption_channels` | 4096 |
| `cross_attention_dim` | 1152 |
| `patch_size` | 2 |
| `out_channels` | 8, learned sigma split active |

T5 text encoder config:

| Field | Value | Source |
| --- | ---: | --- |
| `d_model` | 4096 | text encoder config |
| `num_layers` | 24 | text encoder config |
| `num_heads` | 64 | text encoder config |
| `d_ff` | 10240 | text encoder config |
| `feed_forward_proj` | gated-gelu | text encoder config |
| `vocab_size` | 32128 | text encoder config |
| Alpha max prompt length | 120 | pipeline default |
| Sigma max prompt length | 300 | pipeline default |

VAE and scheduler dimensions:

| Repo | VAE latent | VAE sample | scale | force_upcast | scheduler keys |
| --- | ---: | ---: | ---: | --- | --- |
| Alpha 512 | 4 | 256 | 0.18215 | true | `solver_order=2`, `algorithm_type=dpmsolver++`, `prediction_type=epsilon`, `timestep_spacing=linspace` |
| Alpha 1024 | 4 | 256 | 0.18215 | true | same |
| Sigma 1024 | 4 | 512 | 0.13025 | false | same |
| LCM 1024 | likely 4 | not fetched | not fetched | not fetched | `LCMScheduler`, `original_inference_steps=50`, `timestep_spacing=leading` |

Recommended first Dinoml scheduler slice: `DPMSolverMultistepScheduler` with
epsilon prediction, `solver_order=2`, `algorithm_type=dpmsolver++`,
`solver_type=midpoint`, `lower_order_final=true`, and `timestep_spacing=linspace`.
LCM is a separate admission candidate because it changes timestep selection and
step arithmetic.

## 3a. Family variation traps

- PixArt uses NCHW latent maps and internal patchify/unpatchify. It does not use
  Flux-style packed 2x2 latent tokens at the pipeline boundary.
- Transformer `out_channels=8` with latent `in_channels=4` means the denoiser
  predicts learned sigma-like doubled channels; pipeline keeps `chunk(dim=1)[0]`
  before scheduler step.
- Alpha 1024 source default enables additional resolution/aspect-ratio
  conditioning when `sample_size == 128`. Sigma 1024 config explicitly sets
  `use_additional_conditions=false`, and the Sigma pipeline passes `None`.
- Alpha prompt length default is 120; Sigma default is 300. Prompt masks must
  follow the selected pipeline.
- VAE scaling differs: Alpha uses `0.18215`; Sigma 1024 uses `0.13025`.
- The model class name in repo configs is often `Transformer2DModel`, but
  Diffusers maps the pipeline component to `PixArtTransformer2DModel`.
- Resolution binning can change requested H/W before denoising and resize/crop
  the decoded image back afterward.
- `BasicTransformerBlock` contains generic GLIGEN/chunked-FF branches, but base
  PixArt pipelines do not activate them.
- NHWC is only a guarded optimization for local Conv2d/VAE/patchify regions.
  Source semantics are NCHW at the VAE and transformer boundaries, and BNC in
  transformer blocks.

## 4. Runtime tensor contract

For 1024 x 1024 Alpha, one image per prompt:

| Boundary | Tensor | Shape/layout | Notes |
| --- | --- | --- | --- |
| prompt ids | `input_ids` | `[B,120]` Alpha or `[B,300]` Sigma | CPU tokenizer output. |
| prompt mask | `prompt_attention_mask` | `[B,L]` | Repeated for `num_images_per_prompt`; concatenated for CFG. |
| prompt embeds | `prompt_embeds` | `[B,L,4096]` | T5 encoder output. |
| CFG embeds | after concat | `[2B,L,4096]` when guidance > 1 | Negative prompt first, positive second. |
| latent noise | `latents` | `[B,4,H/8,W/8]` NCHW | Multiplied by `scheduler.init_noise_sigma`. |
| Alpha 1024 conditions | `resolution`, `aspect_ratio` | `[2B,2]`, `[2B,1]` under CFG | Only when `transformer.sample_size == 128` and additional conditions are enabled. |
| model input | `latent_model_input` | `[2B,4,128,128]` NCHW | Scaled by scheduler. |
| timestep | `current_timestep` | `[2B]` | Expanded from scalar scheduler timestep. |
| patch tokens | internal | `[2B,4096,1152]` | Conv2d patch embed, flatten BCHW -> BNC. |
| caption projection | internal | `[2B,L,1152]` | Linear 4096 -> 1152, GELU(tanh), Linear 1152 -> 1152. |
| denoiser output | `noise_pred` | `[2B,8,128,128]` NCHW | Unpatchified. |
| post sigma split | `noise_pred` | `[B,4,128,128]` | CFG then `chunk(2, dim=1)[0]`. |
| scheduler output | `latents` | `[B,4,128,128]` | Host-loop state. |
| VAE decode input | `latents / scale` | `[B,4,128,128]` NCHW | No shift factor in inspected configs. |
| decoded image | `image` | `[B,3,1024,1024]` NCHW before postprocess | Optional resize/crop after binning. |

Patchify/unpatchify contract:

```text
Patchify:
  hidden_states [B,C,H,W]
  Conv2d(C -> 1152, kernel=2, stride=2)
  flatten(2).transpose(1,2): [B,1152,H/2,W/2] -> [B,(H/2)*(W/2),1152]
  add 2D sin-cos position embedding, interpolated when grid changes

Unpatchify:
  proj_out [B,N,patch*patch*out_channels]
  reshape [B,Ht,Wt,patch,patch,Cout]
  einsum "nhwpqc->nchpwq"
  reshape [B,Cout,Ht*patch,Wt*patch]
```

CPU/data-pipeline work: text cleanup, tokenization, truncation warnings,
resolution bin classification, PIL/NumPy postprocess. GPU/runtime work:
prompt/T5 embeddings if admitted, latent init, denoiser, CFG, scheduler step,
VAE decode, optional tensor resize/crop.

## 5. Operator coverage checklist

Tensor/layout ops:

- NCHW latent generation and scaling.
- CFG batch concat/chunk on batch axis.
- Prompt embedding/mask repeat, view, concat.
- Mask-to-bias conversion: `(1 - mask) * -10000`, `unsqueeze(1)`.
- Patchify Conv2d + flatten + transpose; unpatchify reshape/einsum/reshape.
- Optional decoded tensor resize/crop.

Convolution/downsample/upsample ops:

- Patch embedding: `Conv2d(4 -> 1152, 2x2, stride=2, bias=true)`.
- AutoencoderKL decode: Conv2d, ResNet, GroupNorm, SiLU, upsample blocks.
- AutoencoderKL encode for future variants, not required for base text-to-image.

GEMM/linear ops:

- Caption projection: Linear(4096 -> 1152), GELU(tanh), Linear(1152 -> 1152).
- AdaLN-single: timestep/size embedding MLP to `6 * 1152`.
- Per block: self-attn Q/K/V, self output projection, cross-attn Q/K/V, cross output projection.
- Feed-forward with approximate GELU.
- Final `proj_out`: Linear(1152 -> 32) for `patch_size=2`, `out_channels=8`.

Attention primitives:

- Self-attention over image patch tokens, no causal mask.
- Cross-attention from image tokens to T5 caption tokens with text mask bias.
- Default `AttnProcessor2_0` uses PyTorch scaled dot-product attention when available.
- Fused QKV/KV processor exists through `fuse_qkv_projections()`, not default.

Normalization and adaptive conditioning:

- LayerNorm with `elementwise_affine=false`, eps `1e-6`.
- AdaLN-single scale/shift/gates for self-attn and MLP.
- Final LayerNorm plus embedded-timestep scale/shift table.
- VAE GroupNorm.

Scheduler and guidance arithmetic:

- DPMSolver++ epsilon prediction conversion and multistep state.
- `scale_model_input`, `step`, lower-order warmup/final behavior.
- True CFG: `uncond + guidance_scale * (text - uncond)`.
- Learned sigma channel trim.

## 6. Denoiser/model breakdown

`PixArtTransformer2DModel.forward`:

```text
NCHW latents -> PatchEmbed Conv2d + 2D sin-cos position
timestep + optional resolution/aspect -> AdaLayerNormSingle embedding
T5 captions -> PixArtAlphaTextProjection 4096 -> 1152
28 BasicTransformerBlock(norm_type=ada_norm_single)
LayerNorm -> final scale/shift from embedded timestep -> Linear -> unpatchify
```

Each `BasicTransformerBlock` in the active PixArt config:

```text
self-attn branch:
  LayerNorm(no affine) -> adaptive scale/shift from timestep
  self Attention(hidden=1152, heads=16, dim=72, bias=true)
  gate_msa * attention_output + residual

cross-attn branch:
  no norm2 for PixArt ada_norm_single path
  cross Attention(query=1152, key/value context=1152, heads=16, dim=72)
  residual add

MLP branch:
  LayerNorm(no affine) -> adaptive scale/shift
  FeedForward(1152, gelu-approximate)
  gate_mlp * ff_output + residual
```

Config-controlled branches:

- `caption_projection` exists when `caption_channels=4096`, active in sampled configs.
- `use_additional_conditions` controls resolution/aspect embeddings in
  `AdaLayerNormSingle`.
- `attention_type="gated"` would enable GLIGEN-style fuser; sampled configs use
  `"default"`.
- `out_channels=None` would remove learned-sigma split; sampled configs use 8.

## 7. Attention requirements

Required base attention:

| Attention | Query | Key/value | Mask | Backend path |
| --- | --- | --- | --- | --- |
| self-attn | image patch tokens `[B,N,1152]` | same | none in base path | `Attention` + `AttnProcessor2_0` SDPA fallback/eager |
| cross-attn | image patch tokens `[B,N,1152]` | projected T5 tokens `[B,L,1152]` | encoder mask bias `[B,1,L]` prepared to SDPA mask | same |

Head geometry is 16 heads x 72 dim. There is no QK norm and no RoPE in base
PixArt. Position is absolute 2D sin-cos added after patch embed. Attention is
noncausal.

Flash-style constraints:

- Self-attention is a straightforward noncausal dense attention candidate.
- Cross-attention needs support for broadcast additive mask/bias derived from
  T5 attention masks. If a provider cannot handle that mask shape and dtype,
  fall back to SDPA/eager parity.
- Fused projections are source-supported: self-attn can fuse QKV; cross-attn
  can fuse K/V while Q remains separate. Treat this as a load/runtime mutation
  with weight-layout preconditions, not an assumed checkpoint format.
- `attention_processor.py` is the primary implementation path for PixArt; no
  target-specific `attention_dispatch.py` path is used.

## 8. Scheduler and denoising-loop contract

The base loop:

```text
timesteps = scheduler.set_timesteps(num_inference_steps or custom timesteps/sigmas)
latent_model_input = cat([latents, latents]) when CFG
latent_model_input = scheduler.scale_model_input(latent_model_input, t)
noise_pred = transformer(...)
noise_pred = uncond + scale * (text - uncond) when CFG
noise_pred = noise_pred.chunk(2, dim=1)[0] when out_channels == 2 * latent_channels
latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
```

Alpha has a special one-step branch returning element `[1]` from scheduler step
when `num_inference_steps == 1`; Sigma always uses element `[0]`. First Dinoml
parity should start with normal multi-step DPMSolver behavior and explicitly
guard Alpha one-step behavior if admitted.

DPMSolver source state to represent:

- alpha/sigma/lambda tables from beta schedule.
- `model_outputs` ring buffer of length `solver_order`.
- `lower_order_nums` warmup state and lower-order final decisions.
- prediction conversion for epsilon under `algorithm_type=dpmsolver++`.
- midpoint second-order update for sampled configs.

Host/runtime split: keep timestep setup, loop index, and multistep scheduler
state host-visible initially. Compile denoiser forward, CFG arithmetic,
learned-sigma trim, and one scheduler pointwise/multistep update once state
tensors are explicit.

## 9. Position, timestep, and custom math

Position:

- `PatchEmbed` creates 2D sin-cos position embeddings at construction.
- If runtime latent grid differs from configured grid, it recomputes/interpolates
  2D sin-cos embeddings using `base_size` and `interpolation_scale`.
- Sigma configs explicitly set `interpolation_scale=1` for 512 and `2` for
  1024. Alpha omits it, so source default is `max(sample_size // 64, 1)`.

Timestep/additional conditioning:

- `AdaLayerNormSingle` uses `PixArtAlphaCombinedTimestepSizeEmbeddings`.
- With additional conditions, resolution `[height,width]` and aspect ratio
  `[height / width]` join the timestep embedding before the `Linear(1152 -> 6912)`.
- Block-local and final `scale_shift_table` parameters are added to embedded
  timestep chunks before modulation.

Precomputable:

- T5 prompt embeddings/masks per prompt and negative prompt.
- 2D position embeddings for a fixed latent grid.
- Resolution/aspect tensors for fixed H/W.

Dynamic:

- Timestep embedding per denoising step.
- Scheduler tables for custom timesteps/sigmas.
- Position interpolation when resolution binning selects a non-native grid.

## 10. Preprocessing and input packing

Prompt path:

- Optional caption cleaning uses regex/html/ftfy/BeautifulSoup before tokenization.
- Tokenization pads to fixed max length, truncates, and emits attention masks.
- Prompt and negative prompt embeddings/masks are repeated for
  `num_images_per_prompt`, then concatenated on batch axis for CFG.

Resolution path:

- `height` and `width` default to `transformer.sample_size * vae_scale_factor`.
- Both pipelines require H/W divisible by 8.
- Resolution binning maps requested sizes to predefined bins:
  Alpha supports 256/512/1024 bins by `sample_size` 32/64/128.
  Sigma adds a 2048 bin for `sample_size == 256`.
- After VAE decode, tensor resize/crop returns the originally requested size.

There is no pipeline-level latent token packing. Patchify is inside the
transformer and must remain a model boundary operation for first parity.

## 11. Graph rewrite / lowering opportunities

### Rewrite: PatchEmbed as Conv2d-to-token canonical op

Source pattern:

```text
Conv2d(C -> D, kernel=P, stride=P) -> flatten(2) -> transpose(1,2) -> add pos
```

Replacement: a patch-embedding primitive or Conv2d followed by planned
NCHW-to-BNC layout transform.

Preconditions: NCHW input, H/W divisible by patch size, fixed `patch_size=2`,
position embedding generated for the exact output grid. NHWC lowering requires
Conv2d weight transform OIHW -> HWIO and exact flatten order preservation.

Failure cases: odd latent grid, dynamic interpolation not represented, or a
consumer expecting source NCHW after patch embed.

Parity test: random `[B,4,64,64]` and `[B,4,128,128]` patchify comparison,
including interpolated position path.

### Rewrite: Unpatchify as inverse layout transform

Source pattern:

```text
Linear(D -> P*P*C) -> reshape [B,Ht,Wt,P,P,C]
-> einsum("nhwpqc->nchpwq") -> reshape [B,C,Ht*P,Wt*P]
```

Replacement: a planned token-to-NCHW unpatchify primitive.

Preconditions: token count equals `Ht*Wt`, `P=2`, `C=out_channels`, and final
consumer accepts NCHW. NHWC variant must rewrite output layout contract and
the learned-sigma `dim=1` chunk.

Failure cases: changing flatten/einsum order or silently chunking the wrong
axis under NHWC.

### Rewrite: CFG plus learned-sigma trim

Source pattern:

```text
chunk batch -> uncond + scale * (text - uncond) -> chunk channels keep first
```

Replacement: fused pointwise kernel plus optional channel slice.

Preconditions: CFG batch order is negative then positive; `out_channels == 2 *
latent_channels`; source layout NCHW or axis rewrite for NHWC is explicit.

Failure cases: guidance disabled, model variant without doubled output
channels, or NHWC pass leaving `dim=1` unchanged.

### Rewrite: Attention projection fusion

Source pattern: separate Q/K/V linears for self-attn and Q plus K/V linears for
cross-attn, followed by SDPA and output projection.

Replacement: fused QKV for self-attn and fused KV for cross-attn.

Preconditions: no added-KV/IP branch, default processor or
`FusedAttnProcessor2_0`, compatible bias handling, no unsupported masks for the
chosen provider.

Failure cases: GLIGEN or custom attention kwargs, attention processor mutation,
or provider lacking cross-attention mask support.

## 12. Kernel fusion candidates

Highest priority:

- PatchEmbed Conv2d + flatten/transpose + position add.
- AdaLN-single LayerNorm + scale/shift + gated residual epilogues.
- Self/cross-attention GEMM + SDPA + output projection, with guarded fused QKV/KV.
- Feed-forward approximate GELU MLP.
- CFG arithmetic plus learned-sigma channel trim.

Medium priority:

- DPMSolver++ second-order scheduler update once state tensors are explicit.
- Timestep/resolution/aspect embedding MLP.
- Caption projection 4096 -> 1152 -> 1152.
- VAE decode Conv2d/GroupNorm/SiLU/up-block NHWC island.

Lower priority:

- Resolution-bin tensor resize/crop.
- LCM scheduler variant.
- GLIGEN branch inside shared `BasicTransformerBlock`.
- Generic LoRA/adapter loading for third-party PixArt weights.

## 13. Runtime staging plan

Stage 1: Parse PixArt Alpha 512 and 1024 configs, map config class
`Transformer2DModel` to `PixArtTransformer2DModel`, and load transformer weights.

Stage 2: Accept external `prompt_embeds [B,L,4096]` and masks. Implement
caption projection, mask bias conversion, and one transformer block parity.

Stage 3: Implement full `PixArtTransformer2DModel` for Alpha 512 with
`sample_size=64`, no additional conditions, and fixed NCHW latents.

Stage 4: Add Alpha 1024 additional resolution/aspect conditioning and position
interpolation/binning tests.

Stage 5: Add DPMSolver++ epsilon scheduler setup/step with host-visible loop
state, then one denoising-step parity including CFG and learned-sigma trim.

Stage 6: Add AutoencoderKL decode boundary with per-repo scaling factor.

Stage 7: Add Sigma 1024 differences: max sequence length 300, VAE scale
0.13025, `use_additional_conditions=false`, and force-upcast=false VAE.

Stage 8: Add fusions and guarded flash-style attention providers. Keep LCM,
community variants, and adapter mutation separate.

## 14. Parity and validation plan

- Random patchify/unpatchify parity for 512 and 1024 latent grids.
- `PixArtAlphaTextProjection` parity for `[B,L,4096]`.
- `AdaLayerNormSingle` embedding parity with and without resolution/aspect.
- One `BasicTransformerBlock` parity with self-attn, cross-attn, and mask bias.
- Full transformer forward parity for Alpha 512, then Alpha 1024, then Sigma 1024.
- CFG arithmetic and learned-sigma trim parity.
- DPMSolver++ `set_timesteps`, `scale_model_input`, and one/multistep update parity.
- VAE decode input scaling parity for Alpha and Sigma.
- Short deterministic denoising-loop smoke with external prompt embeddings.
- Suggested tolerances: fp32 `rtol=1e-4, atol=1e-5`; fp16/bf16 initially
  `rtol=2e-2, atol=2e-2`, tightened after attention/provider choices settle.

## 15. Performance probes

- T5 encoder throughput for sequence lengths 120 and 300.
- Transformer forward by latent grid: 64x64, 128x128, and Sigma 2048-style 256x256 if admitted.
- Self-attn vs cross-attn time split by prompt length.
- Attention backend comparison: SDPA/eager parity vs Dinoml flash-style provider.
- Patchify/unpatchify overhead.
- DPMSolver scheduler overhead across 20/30/50 steps.
- VAE decode throughput for Alpha and Sigma scaling configs.
- VRAM/workspace by batch, CFG enabled/disabled, and dtype.

## 16. Scope boundary and separate candidates

Separate candidate reports:

- `pixart_lcm`: `PixArt-LCM-XL-2-1024-MS`, `LCMScheduler`, short-step distilled scheduler contract.
- `pixart_sigma_2048`: Sigma pipeline has 2048 resolution bins for
  `sample_size=256`; requires representative component configs before admission.
- `pixart_lora_adapters`: generic PEFT/LoRA mutation if a concrete PixArt LoRA
  artifact must be supported.
- `pixart_gligen_like_block_branch`: shared `BasicTransformerBlock` gated fuser
  if a pipeline/fork passes GLIGEN kwargs.
- `pixart_img2img_inpaint_control`: only for external/community variants,
  because inspected official PixArt folder has no base variant pipeline files.
- `autoencoder_kl_pixart_sigma`: VAE scaling/upcast differences, best folded
  into the existing AutoencoderKL family unless PixArt Sigma decode parity diverges.

Genuinely out of scope for this audit:

- XLA/NPU/MPS branches.
- Flax/ONNX exports.
- Safety checker and NSFW filtering.
- Training, losses, dropout, and gradient checkpointing.
- Multi-GPU/context parallel.
- Callback mutation and interactive interrupt paths.

## 17. Final implementation checklist

- [ ] Parse PixArt model indexes and component configs.
- [ ] Map `Transformer2DModel` config entries to `PixArtTransformer2DModel`.
- [ ] Load transformer weights for Alpha 512 first.
- [ ] Accept external T5 prompt embeddings and masks.
- [ ] Implement mask-to-bias conversion for cross-attention.
- [ ] Implement PatchEmbed Conv2d + 2D sin-cos position path.
- [ ] Implement `PixArtAlphaTextProjection`.
- [ ] Implement AdaLN-single timestep and optional size/aspect conditioning.
- [ ] Implement PixArt `BasicTransformerBlock` self-attn, cross-attn, and MLP.
- [ ] Implement final norm/projection/unpatchify.
- [ ] Implement CFG arithmetic and learned-sigma channel trim.
- [ ] Implement DPMSolver++ epsilon `solver_order=2` first slice.
- [ ] Add AutoencoderKL decode boundary with Alpha/Sigma scaling factors.
- [ ] Add Alpha 1024 and Sigma 1024 parity tests.
- [ ] Add guarded attention and patch layout fusions.
- [ ] Keep LCM, adapters, GLIGEN, and community img2img/inpaint/control variants separate.
