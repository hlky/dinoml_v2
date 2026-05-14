# Diffusers SD1 GLIGEN Audit Report

## 1. Source basis

```text
Diffusers commit/version:
  Local checkout X:/H/diffusers at b3a515080752a3ba7ca92161e25530c7f280f629.
  Remote upstream available as https://github.com/huggingface/diffusers.git.

Model id(s):
  Primary text-box examples: masterful/gligen-1-4-generation-text-box,
  gligen/diffusers-generation-text-box.
  Inpaint text-box examples: masterful/gligen-1-4-inpainting-text-box,
  gligen/diffusers-inpainting-text-box.
  Text-image source path exists for anhnct/Gligen_Text_Image and
  anhnct/Gligen_Inpainting_Text_Image examples, but no local component configs
  were present under H:/configs for those ids.

Config sources:
  Local cache:
    H:/configs/gligen/diffusers-generation-text-box/model_index.json
    H:/configs/gligen/diffusers-inpainting-text-box/model_index.json
    H:/configs/gligen/gligen-generation-text-box/model_index.json (empty {})
    H:/configs/masterful/gligen-1-4-generation-text-box/model_index.json
    H:/configs/masterful/gligen-1-4-inpainting-text-box/model_index.json
    H:/configs/shanquanming/gligen-1-4-generation-text-box/model_index.json
  Hub HTML/raw-inspected configs without writing outside this report path:
    gligen/diffusers-generation-text-box unet/config.json,
    scheduler/scheduler_config.json, text_encoder/config.json.
    gligen/diffusers-inpainting-text-box unet/config.json,
    scheduler/scheduler_config.json, vae/config.json, text_encoder/config.json.
    masterful/gligen-1-4-generation-text-box unet/config.json.

Pipeline files inspected:
  X:/H/diffusers/src/diffusers/pipelines/deprecated/stable_diffusion_gligen/pipeline_stable_diffusion_gligen.py
  X:/H/diffusers/src/diffusers/pipelines/deprecated/stable_diffusion_gligen/pipeline_stable_diffusion_gligen_text_image.py
  X:/H/diffusers/src/diffusers/pipelines/deprecated/stable_diffusion_gligen/__init__.py

Model files inspected:
  X:/H/diffusers/src/diffusers/models/unets/unet_2d_condition.py
  X:/H/diffusers/src/diffusers/models/unets/unet_2d_blocks.py
  X:/H/diffusers/src/diffusers/models/attention.py
  X:/H/diffusers/src/diffusers/models/attention_processor.py
  X:/H/diffusers/src/diffusers/models/embeddings.py
  X:/H/diffusers/src/diffusers/models/autoencoders/autoencoder_kl.py

Scheduler/processors/helpers inspected:
  X:/H/diffusers/src/diffusers/image_processor.py
  Scheduler behavior inherited from SD1.5 report: PNDM sampled default,
  compatible KarrasDiffusionSchedulers surface.

External component configs inspected:
  CLIPTextModel config from GLIGEN repos: openai/clip-vit-large-patch14,
  hidden_size=768, max_position_embeddings=77.
  For text-image pipeline source, CLIPVisionModelWithProjection and
  CLIPProcessor are required by constructor, but local configs were unavailable.

Any missing files or assumptions:
  This audit treats base SD1.4/SD1.5 UNet/VAE/scheduler operators as already
  covered by agents/plans/diffusers/stable_diffusion_1_5/report.md and focuses
  on GLIGEN deltas. Safety checker, XLA/NPU/MPS/Flax/ONNX, callbacks,
  distributed paths, training/loss/dropout/gradient-checkpointing are ignored.
  No gated official config blocked access; the configs inspected were public.
```

## 2. Pipeline and component graph

GLIGEN is a deprecated Stable Diffusion 1.x variant that injects grounded object
tokens into the UNet transformer blocks through gated self-attention. The base
text-box pipeline wires `AutoencoderKL`, `CLIPTextModel`, `CLIPTokenizer`,
`UNet2DConditionModel`, and `KarrasDiffusionSchedulers`; safety components are
optional and ignored here. The text-image pipeline adds `CLIPProcessor`,
`CLIPVisionModelWithProjection`, and `CLIPImageProjection`.

```text
prompt + grounding inputs
  -> CLIP tokenizer/text encoder for prompt embeddings
  -> CLIP text pooled features for each grounded phrase
  -> optional CLIP image encoder + image projection for each grounded image
  -> GLIGEN position_net: boxes + phrase/image features + masks -> object tokens
  -> latent initialization or optional VAE inpaint encode/mask concat
  -> denoising loop:
       scheduler scale_model_input
       -> UNet2DConditionModel with cross_attention_kwargs["gligen"]
       -> gated self-attention fuser inside BasicTransformerBlock
       -> CFG arithmetic + scheduler.step
  -> VAE decode/postprocess
```

Required text-box GLIGEN inputs are `gligen_phrases: list[str]` and
`gligen_boxes: list[list[float]]`, with normalized `[xmin, ymin, xmax, ymax]`
coordinates. Optional inpaint input `gligen_inpaint_image` changes the UNet
input from 4 latent channels to 9 channels for inpaint checkpoints. The
text-image pipeline also accepts `gligen_images`, `input_phrases_mask`,
`input_images_mask`, and `gligen_normalize_constant`.

Separate candidate reports: base SD1 text-to-image, SD1 img2img/inpaint/depth
and upscaling, LoRA/textual inversion/runtime adapter mutation, ControlNet,
T2I-Adapter, and IP-Adapter remain separate candidates. This report is itself
the GLIGEN candidate.

## 3. Important config dimensions

Representative config sweep:

| Repo/config | Pipeline class in model_index | UNet grounding flag | UNet in channels | cross dim | VAE scale | Scheduler |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `gligen/diffusers-generation-text-box` | `StableDiffusionPipeline` | legacy `use_gated_attention: true` | 4 | 768 | 0.18215 inferred or VAE config when present | PNDM, `skip_prk_steps=true`, `steps_offset=1`, `clip_sample=false` |
| `gligen/diffusers-inpainting-text-box` | `StableDiffusionPipeline` | legacy `use_gated_attention: true` | 9 | 768 | 0.18215 in VAE config | same PNDM config |
| `masterful/gligen-1-4-generation-text-box` | `StableDiffusionPipeline` | current `attention_type: "gated"` | 4 | 768 | SD1 VAE default | PNDM in model index |
| `masterful/gligen-1-4-inpainting-text-box` | `StableDiffusionPipeline` | cache had model_index only; Hub history says attention_type was set to gated | expected 9 | 768 | SD1 VAE default | PNDM in model index |

Shared GLIGEN text-box UNet dimensions match SD1: block channels
`320, 640, 1280, 1280`, `layers_per_block=2`, cross-attention dim `768`,
attention head dim `8`, sample size `64`, out channels `4`, norm groups `32`.
The operator-significant differences are `attention_type="gated"` and inpaint
`in_channels=9`.

Config trap: current `UNet2DConditionModel` creates `position_net` only when
`attention_type` is `"gated"` or `"gated-text-image"`. Some public GLIGEN
diffusers configs still contain old `use_gated_attention: true`; Dinoml should
normalize that legacy key to `attention_type="gated"` for text-only GLIGEN or
reject the artifact with a clear diagnostic.

## 3a. Family variation traps

- Text-box GLIGEN uses `feature_type="text-only"` and one object token per
  padded box slot.
- Text-image GLIGEN uses `attention_type="gated-text-image"` and emits text
  object tokens plus image object tokens, doubling the object-token sequence
  from `max_objs` to `2 * max_objs`.
- `max_objs` is hard-coded to 30 in both pipelines; extra boxes/phrases/images
  are truncated with a warning.
- CFG masks differ by pipeline. Text-box duplicates object tensors for CFG and
  zeros masks in the unconditional half. Text-image computes grounded and
  all-zero-grounding kwargs separately, runs the UNet twice per denoising step,
  and combines grounded text prediction with ungrounded unconditional prediction.
- Inpaint mode concatenates `[masked_latent, mask]` to latent model input, so
  the UNet must be a 9-channel GLIGEN inpaint checkpoint. The source contains a
  guard that replaces `latents` with random 4-channel latents when channel count
  is not 4 before the concat.
- Gated fusers are mutable module state. `enable_fuser(True)` is called before
  the loop and text-box disables fusers after
  `int(gligen_scheduled_sampling_beta * len(timesteps))`; text-image computes
  the same integer but current source does not use it to disable fusers.

## 4. Runtime tensor contract

Text-box tensors for a 512x512, batch `B`, `N=30`, CFG-enabled request:

| Boundary | Tensor | Shape | Layout/dtype notes |
| --- | --- | --- | --- |
| prompt embeds | `prompt_embeds` | `[2B, 77, 768]` | CLIP hidden states, dtype follows text encoder/UNet |
| boxes | `boxes` | `[2B, 30, 4]` | normalized xyxy; unconditional masks zeroed |
| phrase pooled embeddings | `positive_embeddings` | `[2B, 30, 768]` | CLIP pooler output padded with zeros |
| object masks | `masks` | `[2B, 30]` | 1 for valid object, 0 for padding; CFG uncond half set to 0 |
| object tokens | `objs` | `[2B, 30, 768]` | output of `GLIGENTextBoundingboxProjection` |
| latent state | `latents` | `[B, 4, 64, 64]` | source NCHW |
| inpaint model input | `latent_model_input` | `[2B, 9, 64, 64]` | `[latents, masked_latent, mask]` |
| UNet noise | `noise_pred` | `[2B, 4, 64, 64]` | chunked on batch for CFG |

Text-image adds `phrases_embeddings`, `image_embeddings`, `phrases_masks`, and
`image_masks`, each padded to 30. `position_net` returns `[B, 60, 768]` because
it concatenates text-projected object tokens and image-projected object tokens.

The grounding preprocessing is partly CPU/data-pipeline work (PIL crop/resize,
tokenization, CLIP processor) and partly GPU/runtime work (CLIP encoder calls,
projection MLP, mask application, VAE encode for inpaint). Object tokens are
constant across denoising steps for fixed grounding inputs and can be cached
until fuser scheduling disables them.

## 5. Operator coverage checklist

Additional GLIGEN operators beyond base SD1:

- Tensor/layout ops: padded object tensor construction, `unsqueeze`, `expand` or
  `repeat`, `clone`, `cat`, CFG batch duplication, mask broadcast over object
  feature dimension, all-zero grounding construction.
- Box Fourier embedding: `arange`, power base `100 ** (i/d)`, multiply by box
  coordinates, `sin`, `cos`, `stack`, `permute`, `reshape` to
  `[B, N, fourier_freqs * 2 * 4]`. With default `fourier_freqs=8`, position
  dim is 64.
- Grounding projection GEMMs: text-only MLP
  `Linear(768+64 -> 512) -> SiLU -> Linear(512 -> 512) -> SiLU -> Linear(512 -> 768)`.
  Text-image has separate text and image copies of that MLP.
- Learnable null tensors: `null_position_feature[64]`, text-only
  `null_positive_feature[768]`, text-image `null_text_feature[768]` and
  `null_image_feature[768]`.
- Gated self-attention dense: `Linear(context_dim=768 -> query_dim)`,
  self-attention over concatenated `[visual_tokens, object_tokens]`, tanh scalar
  gates `alpha_attn` and `alpha_dense`, GEGLU feed-forward.
- Inpaint-only: VAE encode, scheduler `add_noise`, inpaint mask creation,
  elementwise blend `noisy_inpaint_latent * mask + latents * (1-mask)`, and
  channel concat to 9-channel UNet input.

Base SD1 Conv2d, GroupNorm, SiLU, cross-attention, GEGLU, scheduler, and VAE
decode operators are inherited from the SD1.5 report.

## 6. Denoiser/model breakdown

`UNet2DConditionModel.__init__` passes `attention_type` into all
`BasicTransformerBlock` instances. For gated GLIGEN, each block constructs a
`GatedSelfAttentionDense`. The UNet also creates `position_net`:

```text
attention_type="gated" -> GLIGENTextBoundingboxProjection(feature_type="text-only")
attention_type="gated-text-image" -> GLIGENTextBoundingboxProjection(feature_type="text-image")
```

Forward path delta:

```text
sample -> time embedding -> conv_in
cross_attention_kwargs["gligen"] input tensors
  -> position_net(boxes, masks, phrase/image embeddings)
  -> cross_attention_kwargs["gligen"] = {"objs": object_tokens}
down/mid/up transformer blocks:
  self-attn residual
  -> fuser(hidden_states, object_tokens)
  -> cross-attn to prompt embeddings
  -> FFN residual
```

`GatedSelfAttentionDense`:

```text
objs = Linear(context_dim -> query_dim)(objs)
tokens = cat([visual_tokens, objs], dim=1)
visual_update = Attention(LayerNorm(tokens))[:, :n_visual, :]
x = x + tanh(alpha_attn) * visual_update
x = x + tanh(alpha_dense) * FeedForward(LayerNorm(x))
```

The scalar gates are learned parameters initialized to zero in source, so the
new fuser path can initially behave like a no-op before training/fine-tuning.

## 7. Attention requirements

GLIGEN keeps the normal SD1 cross-attention to CLIP prompt tokens, then adds
gated self-attention over visual tokens plus object tokens. It does not use KV
cache, causal masks, RoPE, QK norm, or varlen packing in the inspected source.
The parity path is Diffusers `Attention` through standard attention processors
in `attention_processor.py`; fused projections are source-supported for the
base `Attention` module but are not required for GLIGEN admission.

For an SD1 block with `query_dim=320`, `attention_head_dim=8`, the fuser
attention operates over `H*W + 30` tokens for text-box or `H*W + 60` for
text-image, with hidden width 320 after projecting object tokens to query dim.
At larger blocks the width follows the block channel width. A Dinoml
flash-style provider is valid only under the same mask-free, dropout-zero,
dense-token preconditions as base SD attention; the extra object tokens simply
extend key/value length in self-attention.

## 8. Scheduler and denoising-loop contract

The sampled GLIGEN repos use PNDM with `scaled_linear` betas, 1000 train
timesteps, `skip_prk_steps=true`, `steps_offset=1`, and `clip_sample=false`.
The scheduler family surface remains SD-compatible (`KarrasDiffusionSchedulers`
typing), but first parity should use the checkpoint scheduler.

Text-box loop delta:

```text
num_grounding_steps = int(beta * len(timesteps))
enable_fuser(True)
for i,t in timesteps:
  if i == num_grounding_steps: enable_fuser(False)
  optional inpaint latent blend and 9-channel concat
  one UNet call with grounded kwargs
  normal CFG chunk/arithmetic
  scheduler.step
```

Text-image loop delta:

```text
one UNet call with grounded text/image kwargs
one UNet call with zero grounding kwargs
CFG uses uncond from ungrounded call and text from grounded call
```

The fuser enable/disable state should be explicit runtime state in Dinoml, not
hidden Python mutation.

## 9. Position, timestep, and custom math

Box Fourier embedding is the custom math GLIGEN adds:

```python
emb = 100 ** (torch.arange(fourier_freqs) / fourier_freqs)
emb = emb[None, None, None] * boxes.unsqueeze(-1)
emb = torch.stack((emb.sin(), emb.cos()), dim=-1)
emb = emb.permute(0, 1, 3, 4, 2).reshape(batch, num_boxes, fourier_freqs * 2 * 4)
```

With `fourier_freqs=8`, each box becomes 64 features. Position embeddings and
object projection outputs depend only on boxes and phrase/image features, not on
timestep or latent resolution, so they can be precomputed per request.

## 10. Preprocessing and input packing

Text-box preprocessing tokenizes the main prompt to max CLIP length as in SD1,
then separately tokenizes `gligen_phrases` with padding and uses CLIP
`pooler_output` as the grounded phrase feature. Padded object slots use zeros
until `position_net` replaces masked entries with learned null features.

Text-image preprocessing uses `CLIPProcessor` for each PIL image,
`CLIPVisionModelWithProjection.image_embeds`, then `CLIPImageProjection` to the
phrase embedding space. Source normalizes projected image features by their norm
and multiplies by `gligen_normalize_constant` (default 28.7).

Inpaint preprocessing center-crops non-square input to the VAE sample size,
resizes with Lanczos, runs `VaeImageProcessor.preprocess` to NCHW `[-1,1]`, VAE
encodes, scales by `vae.config.scaling_factor`, constructs a latent-space mask
from the normalized boxes, and concatenates masked latent plus mask to the UNet
input.

## 11. Graph rewrite / lowering opportunities

### Rewrite: legacy GLIGEN config normalization

Source pattern: config has `use_gated_attention: true` but current source reads
`attention_type`.

Replacement: normalize to `attention_type="gated"` during artifact admission for
text-box checkpoints; require `"gated-text-image"` only for text-image variants.

Preconditions: repo is identified as GLIGEN and UNet weights contain fuser and
position-net parameters compatible with gated attention.

Failure cases: vanilla SD checkpoints with accidental legacy key; text-image
weights admitted as text-only.

Parity test: instantiate current Diffusers with normalized config and verify
`unet.position_net` exists and `BasicTransformerBlock.fuser` modules are present.

### Rewrite: precompute GLIGEN object tokens

Source pattern: every UNet forward calls `position_net(**gligen_args)` through
`UNet2DConditionModel.forward`.

Replacement: compute object tokens once per request and pass `{"objs": ...}` to
compiled denoiser blocks, or represent `position_net` as a tiny pre-loop graph.

Preconditions: boxes, masks, phrase/image embeddings, and dtype are constant for
the denoising loop; fuser schedule only toggles use of the tokens, not their
values.

Failure cases: runtime allows dynamic grounding mutation between timesteps.

Parity test: compare full UNet output with source kwargs vs precomputed `objs`
for one timestep.

### Rewrite: fuser attention as extended self-attention

Source pattern: `cat([visual, projected_objs], dim=1) -> self_attention ->
slice first n_visual`.

Replacement: dense noncausal attention over a token matrix of length
`H*W + N_obj`, then slice visual output.

Preconditions: no masks, dropout zero, object projection already applied,
attention processor is default/SDPA-compatible.

Failure cases: custom attention processors or text-image sequence length not
reflected in shape guards.

Parity test: compare one `GatedSelfAttentionDense` for `N_obj=30` and `60`.

## 12. Kernel fusion candidates

Highest priority:

- Base SD1 UNet/VAE fusions from the SD1.5 report, because GLIGEN still spends
  most time in the SD UNet.
- Gated fuser attention over visual plus object tokens, reusing the same
  attention provider as base SD attention with different sequence length.
- GLIGEN object-token projection MLP and mask/null replacement as a pre-loop
  fusion when compiling the full request graph.

Medium priority:

- Inpaint mask blend plus scheduler `add_noise` and channel concat.
- Text-image two-UNet-call CFG scheduling/memory planning.

Lower priority:

- CLIP vision/image projection path for text-image GLIGEN; useful only after
  text-box GLIGEN and base SD operators are admitted.

## 13. Runtime staging plan

Stage 1: Do not admit as an early Dinoml target. Finish base SD1 text-to-image,
AutoencoderKL decode/encode, and one SD scheduler slice first.

Stage 2: Add GLIGEN config admission only: identify legacy
`use_gated_attention`, normalize or reject, and inventory fuser weights without
running full GLIGEN.

Stage 3: Implement `GLIGENTextBoundingboxProjection` parity for text-only
features, including Fourier box embedding, null features, masks, and MLP.

Stage 4: Implement `GatedSelfAttentionDense` parity inside one
`BasicTransformerBlock`, with object tokens supplied externally.

Stage 5: One text-box GLIGEN UNet step parity with precomputed prompt embeddings
and phrase embeddings. Keep scheduler loop in host code.

Stage 6: Add text-box full denoising loop with explicit fuser schedule.

Stage 7: Add inpaint 9-channel GLIGEN path only if SD inpaint/VAE encode is
already stable.

Stage 8: Add text-image GLIGEN with CLIP vision/image projection as a later,
separate extension within this candidate.

First Dinoml staging/admission recommendation: lower priority. GLIGEN is
deprecated in Diffusers, has low reuse relative to base SD/ControlNet/IP-Adapter
coverage, mutates fuser module state during the loop, and depends on legacy
config normalization. It is a good compatibility target after SD1 operators are
solid, not a first-slice admission target.

## 14. Parity and validation plan

- Config admission tests for `attention_type="gated"` and legacy
  `use_gated_attention: true`.
- Fourier box embedding parity for fixed boxes and fp32/fp16.
- `GLIGENTextBoundingboxProjection` text-only parity with valid and padded masks.
- Text-image projection parity with phrase/image masks when configs are
  available.
- `GatedSelfAttentionDense` parity for random visual tokens and object tokens at
  channel widths 320, 640, and 1280.
- One `BasicTransformerBlock` parity with and without `gligen` kwargs.
- One full GLIGEN UNet forward parity at 512 latent shape for text-box.
- Text-box scheduled-sampling parity: fuser enabled for the expected number of
  steps and disabled afterward.
- Inpaint path parity for mask construction, VAE encode scaling, latent blend,
  and 9-channel concat.
- Suggested tolerances follow base SD1: fp32 around `rtol=1e-4, atol=1e-5`;
  fp16/bf16 initially `rtol=2e-2, atol=2e-2`.

## 15. Performance probes

- Base SD1 UNet step with and without GLIGEN fuser enabled.
- Fuser attention cost by latent resolution and object count 0, 1, 10, 30, 60.
- Precomputed object tokens vs computing `position_net` inside every UNet call.
- Text-box one-call CFG vs text-image two-call CFG memory and latency.
- Inpaint overhead: VAE encode, mask construction, scheduler add_noise, blend,
  and 9-channel UNet input.
- Scheduler/fuser-toggle host overhead for short and long denoising loops.

## 16. Scope boundary and separate candidates

Related separate candidate reports:

- Base `stable_diffusion_1_5`: core SD1 prompt embeddings, UNet, scheduler,
  VAE decode.
- `sd1_img2img_inpaint_depth_upscale`: non-GLIGEN inpaint/img2img/depth/upscale
  contracts; GLIGEN inpaint should wait on this.
- `sd1_lora_textual_inversion_adapters`: prompt/token/weight mutation surfaces
  inherited by GLIGEN pipelines.
- `controlnet_sd`, `sd1_t2i_adapter`, `sd1_ip_adapter`: different conditioning
  injection mechanisms with broader practical priority than deprecated GLIGEN.
- `autoencoder_kl`: VAE encode/decode optimization needed for GLIGEN inpaint.

Ignored/out of scope for this audit:

- Safety checker and NSFW filtering.
- XLA/NPU/MPS/Flax/ONNX branches.
- Multi-GPU/context parallel paths.
- Callback mutation and interactive interrupt.
- Training, losses, dropout, and gradient checkpointing.

## 17. Final implementation checklist

- [ ] Add GLIGEN artifact detection and config normalization/rejection.
- [ ] Parse `attention_type` and legacy `use_gated_attention` safely.
- [ ] Load/verify `position_net` and fuser weights in gated UNet checkpoints.
- [ ] Implement box Fourier embedding.
- [ ] Implement text-only `GLIGENTextBoundingboxProjection`.
- [ ] Implement mask/null feature replacement.
- [ ] Implement `GatedSelfAttentionDense` parity.
- [ ] Add one gated `BasicTransformerBlock` parity test.
- [ ] Add one GLIGEN text-box UNet step parity with external prompt/phrase embeddings.
- [ ] Represent fuser enable/disable schedule as explicit runtime state.
- [ ] Add PNDM loop parity only after base SD1 scheduler support lands.
- [ ] Add inpaint 9-channel path after SD inpaint/VAE encode support lands.
- [ ] Add text-image GLIGEN after CLIP vision/image projection admission.
- [ ] Benchmark GLIGEN fuser overhead and object-token precompute.
