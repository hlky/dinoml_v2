# Flux variants/control runtime audit

Target slug: `flux_variants_control`

Status: focused delta report over the base `flux` audit. This report covers
Flux img2img, inpaint, fill, direct control, Flux ControlNet, IP-Adapter,
Kontext, and Prior Redux surfaces. It does not repeat the base Flux transformer
block inventory except where a variant changes input/output contracts.

## 1. Source basis

Diffusers commit/version: local checkout `diffusers` at
`b3a515080752a3ba7ca92161e25530c7f280f629`.

Read first / contrast reports:

- `agents/plans/diffusers/flux/report.md`
- `agents/plans/diffusers/stable_diffusion_3/report.md`
- `agents/plans/diffusers/sd1_ip_adapter/report.md`
- `agents/plans/diffusers/controlnet_sd/report.md`

Pipeline files inspected:

- `src/diffusers/pipelines/flux/pipeline_flux_img2img.py`
- `src/diffusers/pipelines/flux/pipeline_flux_inpaint.py`
- `src/diffusers/pipelines/flux/pipeline_flux_fill.py`
- `src/diffusers/pipelines/flux/pipeline_flux_control.py`
- `src/diffusers/pipelines/flux/pipeline_flux_control_img2img.py`
- `src/diffusers/pipelines/flux/pipeline_flux_control_inpaint.py`
- `src/diffusers/pipelines/flux/pipeline_flux_controlnet.py`
- `src/diffusers/pipelines/flux/pipeline_flux_controlnet_image_to_image.py`
- `src/diffusers/pipelines/flux/pipeline_flux_controlnet_inpainting.py`
- `src/diffusers/pipelines/flux/pipeline_flux_kontext.py`
- `src/diffusers/pipelines/flux/pipeline_flux_kontext_inpaint.py`
- `src/diffusers/pipelines/flux/pipeline_flux_prior_redux.py`
- `src/diffusers/pipelines/flux/modeling_flux.py`

Model/loader/helper files inspected:

- `src/diffusers/models/transformers/transformer_flux.py`
- `src/diffusers/models/controlnets/controlnet_flux.py`
- `src/diffusers/loaders/transformer_flux.py`
- `src/diffusers/loaders/ip_adapter.py`
- `src/diffusers/models/attention_dispatch.py`
- `src/diffusers/models/attention_processor.py`
- `src/diffusers/image_processor.py`
- `src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py`

Config sources:

| Repo/config | Source | Useful fields |
| --- | --- | --- |
| `H:/configs/black-forest-labs/FLUX.1-dev/*` | local cache | base dev transformer, VAE, scheduler, text encoder configs |
| `H:/configs/black-forest-labs/FLUX.1-schnell/*` | local cache | base schnell comparison |
| `black-forest-labs/FLUX.1-Fill-dev` | authenticated HF metadata, not saved | `FluxFillPipeline`, transformer `in_channels=384`, `out_channels=64`, guidance embeds true |
| `black-forest-labs/FLUX.1-Canny-dev` | authenticated HF metadata, not saved | `FluxControlPipeline`, transformer `in_channels=128`, `out_channels=64` |
| `black-forest-labs/FLUX.1-Depth-dev` | authenticated HF metadata, not saved | same direct-control channel contract as Canny |
| `black-forest-labs/FLUX.1-Kontext-dev` | local cache plus authenticated HF metadata | `FluxKontextPipeline`, transformer `in_channels=64`, `out_channels=null`, guidance embeds true |
| `black-forest-labs/FLUX.1-Redux-dev` | authenticated HF metadata, not saved | `FluxPriorReduxPipeline`, SigLIP vision hidden 1152, `ReduxImageEncoder` to T5 width 4096 |
| `H:/configs/XLabs-AI/flux-controlnet-*` | local cache | only `{}` placeholders; no useful component configs |
| `H:/configs/alimama-creative/FLUX.1-dev-Controlnet-Inpainting-*` | local cache | only `{}` placeholders; no useful component configs |

Blockers and limits:

- Official Black Forest Labs Fill/Canny/Depth/Redux/Kontext configs are gated
  to unauthenticated HTTP. Authenticated local `hf` credentials for user `hlky`
  succeeded for metadata/config reads. Per task write limits, fetched configs
  were not persisted under `H:/configs`.
- XLabs and Alimama local caches inspected here contain empty `model_index.json`
  placeholders only. Source behavior was audited from Diffusers classes rather
  than those checkpoint configs.
- Ignored per task: XLA/NPU/MPS/Flax/ONNX, safety/NSFW, training/loss/dropout/
  gradient checkpointing, multi-GPU/context parallel, callbacks/interrupt.

## 2. Variant graph summary

All selected variants keep the same Flux backbone ideas from the base report:
T5 token embeddings `[B,L,4096]`, CLIP pooled embeddings `[B,768]`, packed VAE
latent tokens, 3-axis RoPE ids, FlowMatch Euler denoising, and 16-channel VAE
scale/shift decode. The variant differences are in how the denoiser input
tokens are formed, whether an extra side model produces residuals, and whether
attention processors receive image adapter tokens.

```text
base prompt embeddings
  -> variant-specific image/control/mask/reference preprocessing
  -> VAE encode or direct control packing where needed
  -> packed latent tokens and ids
  -> denoising loop:
       direct-control/fill: widened hidden_states tokens
       Flux ControlNet: side FluxControlNetModel residuals -> base transformer
       IP-Adapter: extra image K/V attention branches
       Kontext: optional reference image tokens appended to latent sequence
       true CFG variants: optional second negative transformer call
  -> FlowMatch Euler step on generated latent tokens only
  -> unpack generated latent tokens -> VAE decode
```

Class/file map:

| Surface | Primary classes/files | Delta from base Flux |
| --- | --- | --- |
| img2img | `FluxImg2ImgPipeline` | VAE encodes init image, slices timesteps by `strength`, uses `scheduler.scale_noise`, then denoises packed tokens. |
| inpaint | `FluxInpaintPipeline` | VAE encodes init and masked image, packs mask and masked-image tokens, blends latents with mask after scheduler steps. Supports true CFG/IP-Adapter. |
| fill | `FluxFillPipeline` | Concatenates generated latent tokens with packed masked-image plus packed high-resolution mask tokens along feature dim before transformer. Requires widened transformer config. |
| direct control | `FluxControlPipeline` | VAE encodes control image to packed control tokens and concatenates `[latents, control_image]` along feature dim. |
| ControlNet | `FluxControlNetPipeline`, `FluxControlNetModel`, `FluxMultiControlNetModel` | Side Flux transformer produces per-block and per-single-block residual samples injected into base transformer blocks. |
| IP-Adapter | `FluxIPAdapterMixin`, `FluxIPAdapterAttnProcessor`, `MultiIPAdapterImageProjection` | Replaces dual-stream attention processors with extra image K/V attention branches and per-adapter scale. |
| Kontext | `FluxKontextPipeline`, `FluxKontextInpaintPipeline` | Optional reference image is encoded/packed as extra image tokens appended to hidden-state sequence; ids mark reference image tokens with first id channel set to 1. |
| Prior Redux | `FluxPriorReduxPipeline`, `ReduxImageEncoder` | Separate preconditioning pipeline: SigLIP image tokens -> MLP -> T5-width image embeddings appended to prompt embeddings. No denoising itself. |

## 3. Config dimensions that change operators

| Variant/config | Pipeline class | Transformer input contract | Guidance/scheduler |
| --- | --- | --- | --- |
| Base `FLUX.1-dev` | `FluxPipeline` | `in_channels=64`, packed 16-channel latents | embedded guidance true, dynamic FlowMatch shift |
| Base `FLUX.1-schnell` | `FluxPipeline` | `in_channels=64` | no guidance embeds, static shift |
| `FLUX.1-Canny-dev` / `Depth-dev` | `FluxControlPipeline` | `in_channels=128`, `out_channels=64`: generated latent tokens 64 + control tokens 64 | embedded guidance true, dynamic shift |
| `FLUX.1-Fill-dev` | `FluxFillPipeline` | `in_channels=384`, `out_channels=64`: generated tokens 64 + masked-image 64 + packed mask 256 | embedded guidance true, dynamic shift |
| `FLUX.1-Kontext-dev` | `FluxKontextPipeline` | `in_channels=64`, reference image increases sequence length, not feature width | embedded guidance true, dynamic shift |
| `FLUX.1-Redux-dev` | `FluxPriorReduxPipeline` | no denoiser; emits prompt embeddings | SigLIP vision + Redux MLP conditioning |

Flux ControlNet source defaults in `FluxControlNetModel`:

| Field | Default/source behavior |
| --- | --- |
| `in_channels` | 64 by default; side model input follows packed latent/control-token width |
| layers | `num_layers=19`, `num_single_layers=38`, but `from_transformer()` can create smaller 4/10 side models |
| hidden | 24 heads, head dim 128, inner dim 3072 |
| residual heads | zero-initialized Linear(3072 -> 3072) for every dual block and single block |
| hint path | optional `ControlNetConditioningEmbedding` when `conditioning_embedding_channels` is set; otherwise control condition is already token-like |
| union path | optional `num_mode` adds a mode embedding token before text embeddings |

Prior Redux config:

| Component | Config evidence |
| --- | --- |
| `SiglipVisionModel` | hidden 1152, 27 layers, 16 heads, patch 14, image size 384, bf16 metadata |
| `ReduxImageEncoder` | `redux_dim=1152`, `txt_in_features=4096`; Linear(1152 -> 12288) -> SiLU -> Linear(12288 -> 4096) |
| output | image embeddings are concatenated to T5 prompt embeddings along sequence dim, then weighted/summed across references |

## 4. Runtime tensor contracts

Base packed latent reminder for 1024x1024:

```text
VAE latent map:      [B,16,128,128]  NCHW
packed latent tokens [B,4096,64]     2x2 tiles, channel width 16*4
latent image ids:    [4096,3]
transformer output:  [B,4096,64]
```

Img2img:

- Input image is preprocessed to NCHW `[B,3,H,W]` in image processor range, or
  accepted as latent map `[B,16,H/8,W/8]`.
- `_encode_vae_image` returns `(vae.encode(image) - shift_factor) *
  scaling_factor`.
- `strength` selects `t_start`; only `timesteps[t_start * scheduler.order:]`
  are used.
- Initial noisy map is `scheduler.scale_noise(image_latents, latent_timestep,
  noise)`, then packed to `[B,S,64]`.
- Scheduler updates only packed generated latents; decode uses standard unpack.

Inpaint:

- Uses img2img latent/noise/image-latent preparation and also returns packed
  `noise` and packed `image_latents`.
- Mask is resized to latent map size `[B,1,H/8,W/8]`, repeated across 16
  latent channels, then packed to `[B,S,64]`.
- Masked image is VAE encoded to `[B,16,H/8,W/8]`, scaled/shifted, then packed
  to `[B,S,64]`.
- The inpaint transformer input stays feature width 64 for generated tokens in
  source branch shown; mask/image are used for denoising/blend contracts rather
  than the Fill widened input. After step, source blends with
  `(1 - mask) * image_latents + mask * latents` in packed-token space.
- True CFG, if enabled, runs a second transformer call with negative prompt and
  optional negative IP image embeds.

Fill:

- Mask starts at image resolution, not latent resolution. It is reshaped into
  64 channels per latent cell from 8x8 VAE-scale pixels, then packed again by
  the Flux 2x2 packer.
- For 1024x1024, packed mask width is `64 * 4 = 256`.
- Packed masked image width is 64. Transformer input is:

```text
cat([latents, masked_image_latents, mask], dim=-1)
[B,S,64 + 64 + 256] = [B,S,384]
```

- The official Fill transformer config confirms `in_channels=384` and
  `out_channels=64`; scheduler state remains only the generated latent tokens
  `[B,S,64]`.

Direct Flux control:

- Control image is preprocessed to NCHW and, if 4D, VAE encoded to 16-channel
  latents using the same scale/shift as base Flux.
- Control latent map is packed to `[B,S,64]`.
- Generated latents are prepared as `[B,S,64]`.
- Denoiser input is `torch.cat([latents, control_image], dim=2)`, giving
  `[B,S,128]` for official Canny/Depth configs. The transformer still outputs
  `[B,S,64]`; scheduler updates only the generated latent tokens.

Flux ControlNet:

- Pipeline prepares generated latents `[B,S,64]` and a control condition.
- If `FluxControlNetModel.input_hint_block is None`, the control image is VAE
  encoded and packed to `[B,S,64]`; this is the InstantX-style token condition.
- If `input_hint_block` exists, the control image stays NCHW and is embedded by
  `ControlNetConditioningEmbedding`, reshaped/packed into tokens inside the
  side model; this is the XLabs-style hint path.
- Side model returns:

```text
controlnet_block_samples:        list length up to num dual blocks, each [B,S,3072]
controlnet_single_block_samples: list length up to num single blocks, each [B,S,3072]
```

- Base `FluxTransformer2DModel.forward` adds these residuals to hidden states
  after corresponding dual/single blocks. If `controlnet_blocks_repeat=True`,
  residuals repeat modulo length for XLabs-style shorter residual lists;
  otherwise they are spaced by `ceil(num_blocks / len(samples))`.
- `control_guidance_start/end` create host-side per-step keep scalars. Multi
  ControlNet sums matching residual tensors.

IP-Adapter:

- `prepare_ip_adapter_image_embeds` accepts either images or list entries of
  precomputed embeds; list length must equal number of loaded IP adapters.
- Loader installs `MultiIPAdapterImageProjection` and swaps non-single Flux
  attention processors to `FluxIPAdapterAttnProcessor`. Single transformer
  blocks keep their existing processor class.
- Standard Flux IP projection infers 4 image text tokens from `proj.weight`,
  or 16 tokens when rows are 65536. Cross-attention dim is rows/token count,
  typically 4096 for Flux.
- In attention, base joint text+image attention is computed first. Then for
  each adapter:

```text
ip_key/value = Linear(ip_hidden_states) -> [B,T_ip,heads,head_dim]
ip_output = attention(query_from_image_stream, ip_key, ip_value)
ip_attn_output += scale_j * ip_output
```

- Processor returns `(hidden_states, encoder_hidden_states, ip_attn_output)`;
  block code adds the IP output into the image residual path. IP masks are in
  the processor signature but not implemented in the Flux processor body audited
  here.

Kontext:

- Optional reference image is resized toward a preferred Kontext resolution,
  VAE encoded with `sample_mode="argmax"`, scaled/shifted, and packed to
  `[B,S_ref,64]`.
- Reference image ids are generated like latent ids but first id channel is set
  to 1; generated latent ids keep first channel 0.
- Denoiser hidden sequence is augmented by concatenating generated latent tokens
  and reference image tokens along sequence, not channel. The model output span
  corresponding to generated latents is the portion that feeds the scheduler.
- `max_area` may auto-adjust requested generation height/width to fit Flux
  model area constraints before packing.

Prior Redux:

- Image preprocessing uses `SiglipImageProcessor` to 384-size RGB tensors.
- `SiglipVisionModel.last_hidden_state` is repeated by images per prompt, then
  `ReduxImageEncoder` maps each token from width 1152 to 4096.
- If text encoders are absent, it creates zero T5 prompt embeds
  `[B,512,4096]` and zero CLIP pooled embeds `[B,768]`.
- It concatenates text and image embeddings along sequence dim, scales prompt
  and pooled embeddings by per-reference weights, then sums across reference
  batch to output one conditioning pair.

## 5. Guidance and scheduler deltas

- Direct control, Fill, Kontext, and Flux ControlNet all keep
  `FlowMatchEulerDiscreteScheduler` and the Flux dynamic shift calculation from
  packed generated latent sequence length.
- Fill/Canny/Depth/Kontext official configs use guidance embeddings. The
  denoiser receives a `guidance` tensor expanded to `[B]`, and model code
  multiplies it by 1000 before embedding.
- Some variant pipelines add true CFG separately from embedded guidance:
  `true_cfg_scale > 1` with negative prompt/embeds runs a second negative
  transformer pass and computes `neg + scale * (pos - neg)`.
- Flux ControlNet has two guidance points: the side ControlNet can use its own
  `config.guidance_embeds`, and the base transformer separately checks
  `transformer.config.guidance_embeds`.
- Control windows are host loop state: `controlnet_keep[i]` gates each
  ControlNet conditioning scale by start/end percentages.
- The first Dinoml scheduler slice can reuse the base Flux FlowMatch Euler
  implementation; the new work is keeping variant latents/control residuals
  separate so scheduler math applies only to generated latent tokens.

## 6. Operator and fusion candidates

Highest priority:

- Packed latent encode/decode variants: faithful 2x2 pack/unpack plus Fill mask
  reshape/pack and Kontext reference-token ids.
- Direct-control widened input projection: Linear(128 -> 3072) and Fill
  Linear(384 -> 3072), both producing output width 64.
- Flux ControlNet residual ABI: side transformer block outputs, zero Linear
  residual heads, per-step scale gates, residual injection into base blocks.
- IP-Adapter added K/V branch for Flux dual-stream attention: image projection,
  per-adapter K/V Linear, shared image query attention, scaled add.
- True CFG two-call arithmetic as explicit graph/host loop contract.

Medium priority:

- ControlNet hint path: NCHW `ControlNetConditioningEmbedding` conv stack,
  reshape to packed tokens, then token Linear.
- Multi-ControlNet residual accumulation and union mode token embedding.
- Kontext reference-token concatenation/splitting and RoPE id handling.
- Prior Redux SigLIP token projection MLP if Dinoml wants to own the
  preconditioning pipeline; otherwise accept Redux prompt embeddings.

Lower priority:

- IP-Adapter multiple adapters/token counts and processor scale mutation beyond
  one adapter.
- Fill high-resolution mask pack fusion with mask preprocessing.
- Full VAE encode/decode optimization for every image/init/mask path.
- CLIP/T5/SigLIP encoder compilation; useful later, but external embeddings are
  enough for first variant admission.

## 7. Graph rewrite / lowering opportunities

### Rewrite: variant packed inputs as explicit token feature schemas

Source pattern:

```text
base:    hidden_states = latents                         width 64
control: hidden_states = cat(latents, control_latents)    width 128
fill:    hidden_states = cat(latents, masked, mask_pack)  width 384
```

Replacement: represent the transformer input feature schema in the manifest and
lower the same Flux block stack after the first `x_embedder` Linear.

Preconditions: `transformer.config.in_channels` and `out_channels` match the
schema; scheduler update consumes only the generated `[B,S,64]` output.

Failure cases: treating `in_channels` as VAE latent channels, applying scheduler
to concatenated control/mask features, or using base 64-wide weights for Fill/
Control variants.

### Rewrite: Flux ControlNet residual injection

Source pattern:

```text
side_controlnet(latents, control_cond, prompt, t)
  -> block_samples, single_block_samples
base_transformer(..., controlnet_*_samples)
  -> hidden_states += sample at scheduled block indices
```

Replacement: explicit side-model subgraph plus residual tuple artifact ABI.

Preconditions: residual sample widths equal base inner dim 3072; residual list
length and repeat/interval policy are artifact-visible; conditioning scale is a
runtime scalar or per-step host input.

Failure cases: hidden Python list length, mixed XLabs/InstantX hint contracts
without a selected schema, or union mode without mode-token support.

### Rewrite: Flux IP-Adapter branch

Source pattern: base joint attention plus independent image K/V attention using
the image stream query.

Replacement: compound attention op or base attention plus one added K/V branch:

```text
out_base = joint_attn(text+image)
out_ip = attn(Q_image, K_ip, V_ip)
image_residual += scale * out_ip
```

Preconditions: no IP masks for first slice, fixed adapter count, known IP token
count, dropout 0, noncausal dense attention, scale represented explicitly.

Failure cases: assuming concatenating IP tokens into the base joint attention is
equivalent; it is not, because the IP branch uses only the image query and an
independent scaled output path.

### Rewrite: Prior Redux as prompt-embedding producer

Source pattern: SigLIP hidden tokens -> Redux MLP -> concat with T5 prompt
embeds -> per-reference scale -> sum.

Replacement: compile only `ReduxImageEncoder` and conditioning arithmetic, or
treat the full Prior Redux output as external cached prompt embeddings.

Preconditions: prompt embedding width 4096, pooled width 768, reference batch and
scale list fixed or runtime-visible.

Failure cases: folding Redux into the Flux denoiser, because it is a separate
pipeline that produces conditioning tensors.

## 8. Staging and admission recommendation

Recommended first Dinoml slice:

1. `flux_img2img_preencoded`: accept external prompt embeds, pooled embeds,
   packed/noisy latents or NCHW VAE latents, fixed FlowMatch timestep table, and
   validate `scheduler.scale_noise` plus strength-based timestep slicing. This
   proves VAE encode/packed-init/scheduler semantics without widening the
   transformer.
2. `flux_control_direct_canny_depth`: admit direct-control Canny/Depth with
   prepacked control tokens `[B,S,64]` first. Compile a transformer whose
   `x_embedder` is `Linear(128 -> 3072)` and output is `[B,S,64]`. This is the
   cleanest control admission because it avoids side residual list ABI.
3. `flux_fill_masked_tokens`: add the Fill `in_channels=384` schema with
   caller-supplied packed masked image and packed mask. Implement the mask pack
   kernel after transformer parity is stable.
4. `flux_controlnet_residuals`: admit `FluxControlNetModel` as a side model with
   explicit residual tuple outputs and base transformer residual inputs. Start
   with one ControlNet, no union, no MultiControlNet, prepacked control condition.
5. `flux_ip_adapter_single`: one standard adapter, precomputed projected image
   tokens, no masks, scale as explicit runtime scalar. Then add image projection.
6. `flux_kontext_reference_tokens`: append one reference image token span with
   explicit ids; keep reference VAE encode external first.
7. `flux_prior_redux`: separate preconditioning report/slice; initially accept
   Redux output embeddings as external inputs.

First admission label: `frontend-only` for schema parsing and manifest-visible
variant contracts; `bounded-cuda` once direct-control widened transformer input
and one denoising step validate against Diffusers.

Why direct control before ControlNet/IP-Adapter:

- It exercises the most important packed-latent variant trap, `in_channels`
  meaning packed feature width rather than VAE channels.
- It reuses the base Flux transformer with only `x_embedder` width changed.
- It keeps residual list ABI, side-model scheduling, and attention-processor
  mutation out of the first control parity target.

## 9. Parity and validation plan

- Pack/unpack parity for `[B,16,H,W] <-> [B,S,64]` at 1024x1024 and a non-square
  size divisible by 16.
- Img2img: VAE encode scale/shift, `scheduler.scale_noise`, strength timestep
  slicing, and one denoising step with fixed noise.
- Direct control: prepacked generated/control tokens, `Linear(128 -> 3072)`
  transformer forward parity, and scheduler update only on generated tokens.
- Fill: mask reshape from image mask to packed `[B,S,256]`, masked-image pack
  `[B,S,64]`, concatenated `[B,S,384]` transformer input, output `[B,S,64]`.
- Flux ControlNet: one side dual block and one single block residual head
  parity, then full residual tuple count/order/shape, then base transformer
  injection parity with fixed residuals.
- Control windows: start/end gates produce expected per-step scales.
- IP-Adapter: one Flux dual attention layer with random projected image tokens;
  scale 0 must match base Flux attention output.
- Kontext: reference image latent ids have first channel set to 1, generated ids
  0; appended sequence length and output split are correct.
- Prior Redux: Redux MLP parity on SigLIP hidden states, concat/scale/sum output
  prompt embeddings.
- Suggested tolerances: fp32 `rtol=1e-4, atol=1e-5`; fp16/bf16
  `rtol=2e-2, atol=2e-2` until attention/norm kernels are tuned.

## 10. Performance probes

- Base Flux vs direct-control widened `x_embedder` cost; transformer body should
  dominate after the first projection.
- Fill mask pack and widened input projection overhead.
- One Flux ControlNet side forward by residual count: 4/10 small side model vs
  full 19/38 side model.
- Base transformer with residual injection cost and memory.
- IP-Adapter added K/V attention overhead by IP token count 4 vs 16.
- Kontext sequence-length sweep: generated tokens plus reference tokens.
- VAE encode overhead for img2img/inpaint/control/fill versus accepting
  preencoded latents.
- Scheduler/true-CFG overhead: one transformer call vs positive+negative calls.
- VRAM/workspace with residual tuple materialization versus streaming residual
  injection.

## 11. Scope boundary and separate candidates

Separate candidates, not ignored:

- `flux_img2img`: VAE encode plus strength/timestep/noise contract.
- `flux_inpaint`: mask/latent blend plus true CFG and optional IP-Adapter.
- `flux_fill`: widened 384-channel packed token transformer and mask pack.
- `flux_control_direct`: Canny/Depth direct-control widened 128-channel input.
- `flux_controlnet`: side transformer residual model and control guidance
  windows.
- `flux_controlnet_multi_union`: multiple controls, union mode embedding, list
  residual accumulation.
- `flux_ip_adapter`: attention-processor mutation and image K/V branches.
- `flux_kontext`: reference image token sequence and preferred resize behavior.
- `flux_kontext_inpaint`: combines Kontext reference tokens with mask/inpaint
  contracts.
- `flux_prior_redux`: SigLIP/Redux prompt-conditioning producer.
- `flux_lora_runtime_adapters`: text/transformer adapter mutation and scale
  handling.

Out of scope for this audit:

- Re-auditing base Flux transformer blocks, text encoders, VAE internals, and
  FlowMatch scheduler beyond variant deltas.
- XLA/NPU/MPS/Flax/ONNX branches.
- Safety checker/NSFW.
- Training, losses, dropout, gradient checkpointing.
- Multi-GPU/context-parallel paths.
- Callback mutation and interactive interrupt.

## 12. Final implementation checklist

- [ ] Parse Flux variant model indexes and transformer `in_channels/out_channels`.
- [ ] Represent packed token schemas: base 64, direct-control 128, Fill 384.
- [ ] Keep scheduler state tied to generated latent tokens only.
- [ ] Add img2img strength/timestep and `scale_noise` parity.
- [ ] Add direct-control one-step parity with prepacked control tokens.
- [ ] Add Fill masked-image and mask packing parity.
- [ ] Add Flux ControlNet residual tuple ABI and base transformer injection.
- [ ] Add control guidance start/end scale gates.
- [ ] Add one Flux IP-Adapter branch with explicit scale.
- [ ] Add Kontext reference token/id concatenation and output split.
- [ ] Treat Prior Redux output as cached prompt embeddings first.
- [ ] Add parity tests before broadening to MultiControlNet, union, masks, or
      runtime adapter mutation.
