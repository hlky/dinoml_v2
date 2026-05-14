# Diffusers Sana Operator and Integration Report

Target slug: `sana`

Status: focused Sana-family audit report.

## 1. Source basis

```text
Diffusers commit/version:
  Local checkout X:/H/diffusers at b3a515080752a3ba7ca92161e25530c7f280f629.

Model id(s):
  Efficient-Large-Model/Sana_1600M_1024px_diffusers
  Efficient-Large-Model/Sana_1600M_4Kpx_BF16_diffusers
  Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers
  Efficient-Large-Model/SANA1.5_4.8B_1024px_diffusers
  Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers
  Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers
  katuni4ka/tiny-random-sana
  katuni4ka/tiny-random-sana-sprint

Config sources:
  H:/configs/Efficient-Large-Model/*/model_index.json
  H:/configs/Efficient-Large-Model/*/transformer/config.json
  H:/configs/Efficient-Large-Model/*/scheduler/scheduler_config.json
  H:/configs/Efficient-Large-Model/*/vae/config.json
  H:/configs/Efficient-Large-Model/*/text_encoder/config.json
  H:/configs/Efficient-Large-Model/*/tokenizer/tokenizer_config.json
  H:/configs/katuni4ka/tiny-random-sana*/...
  Local model indexes existed first; component configs were fetched with
  `huggingface_hub.hf_hub_download` and saved under H:/configs.

Pipeline files inspected:
  X:/H/diffusers/src/diffusers/pipelines/sana/pipeline_sana.py
  X:/H/diffusers/src/diffusers/pipelines/sana/pipeline_sana_sprint.py
  X:/H/diffusers/src/diffusers/pipelines/sana/pipeline_sana_sprint_img2img.py
  X:/H/diffusers/src/diffusers/pipelines/sana/pipeline_sana_controlnet.py
  X:/H/diffusers/src/diffusers/pipelines/pag/pipeline_pag_sana.py

Model files inspected:
  X:/H/diffusers/src/diffusers/models/transformers/sana_transformer.py
  X:/H/diffusers/src/diffusers/models/controlnets/controlnet_sana.py
  X:/H/diffusers/src/diffusers/models/autoencoders/autoencoder_dc.py

Scheduler/processors/helpers inspected:
  X:/H/diffusers/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py
  X:/H/diffusers/src/diffusers/schedulers/scheduling_scm.py
  X:/H/diffusers/src/diffusers/models/attention_processor.py
  X:/H/diffusers/src/diffusers/models/attention_dispatch.py
  X:/H/diffusers/src/diffusers/models/embeddings.py
  X:/H/diffusers/src/diffusers/models/normalization.py
  X:/H/diffusers/src/diffusers/image_processor.py

External component configs inspected:
  Gemma2Model text encoder configs and GemmaTokenizer/GemmaTokenizerFast
  tokenizer configs from the repos above.

Any missing files or assumptions:
  This report covers the base text-to-image Sana and Sana Sprint image
  pipelines. Sana ControlNet, PAG, Sprint img2img, Sana video, LoRA/runtime
  adapter mutation, quantized Nunchaku variants, and DC-AE as a standalone codec
  deserve separate reports. Multi-GPU/context parallel, callback mutation,
  interactive interrupt, XLA/NPU/MPS/Flax/ONNX, safety/NSFW, and training/loss/
  dropout/gradient checkpointing paths were intentionally ignored.
```

## 2. Pipeline and component graph

Base Sana is a latent image pipeline with a Gemma2 text encoder, an
`AutoencoderDC` codec, a `SanaTransformer2DModel` denoiser, and
`DPMSolverMultistepScheduler`. Sana Sprint shares the same denoiser class and
codec boundary but uses `SanaSprintPipeline`, embedded guidance, and
`SCMScheduler`.

```text
prompt preprocessing
  -> GemmaTokenizerFast/GemmaTokenizer + Gemma2Model
  -> prompt embeddings [B,L,2304] + attention mask [B,L]
  -> latent noise [B,C,H/32,W/32] in source NCHW
  -> denoising loop:
       SanaTransformer2DModel(latents, text embeds, text mask, timestep,
                              optional embedded guidance/control residuals)
       CFG arithmetic for base Sana or embedded-guidance SCM math for Sprint
       scheduler.step
  -> AutoencoderDC decode(latents / scaling_factor)
  -> optional resolution-bin resize/crop
  -> PixArtImageProcessor postprocess
```

Required first-slice components for base Sana:

- `SanaTransformer2DModel` with NCHW latent input and internal patch embed.
- Externally supplied prompt embeddings and masks first; Gemma2 can be a later
  compiled component.
- `DPMSolverMultistepScheduler` restricted to the sampled `flow_prediction`,
  `dpmsolver++`, solver-order-2 config.
- `AutoencoderDC` decode boundary with 32 latent channels and `scaling_factor =
  0.41407`.
- CFG batch duplication and `uncond + scale * (text - uncond)` arithmetic.

Separate candidate reports:

| Candidate | Classes/files | Runtime delta |
| --- | --- | --- |
| `sana_sprint` | `SanaSprintPipeline`, `SCMScheduler`, same `SanaTransformer2DModel` | Embedded guidance tensor, trigflow/SCM timestep math, default two-step schedule, no true CFG batch. |
| `sana_sprint_img2img` | `SanaSprintImg2ImgPipeline` | Adds image preprocessing, VAE encode, strength/start timestep behavior, SCM latent/noise mixing. |
| `sana_controlnet` | `SanaControlNetPipeline`, `SanaControlNetModel` | Adds control image condition, control transformer residuals, control scale injection into base transformer blocks. |
| `sana_pag` | `SanaPAGPipeline`, `PAGCFGSanaLinearAttnProcessor2_0`, `PAGIdentitySanaLinearAttnProcessor2_0` | Mutates self-attention processors and batch composition for perturbed attention guidance. |
| `sana_lora_adapters` | `SanaLoraLoaderMixin`, `PeftAdapterMixin` on transformer/text encoder | Runtime/load-time adapter mutation for transformer and Gemma2 layers. |
| `sana_video` | `pipelines/sana_video/*`, `transformer_sana_video.py` | Distinct video transformer/codec/temporal contract. |
| `sana_dcae_codec` | `AutoencoderDC`, `dc-ae-f32c32-sana-*` configs | Shared codec island with pixel shuffle/unshuffle, EfficientViT blocks, tiling/slicing. |
| `nunchaku_sana_quant` | `nunchaku-ai/nunchaku-sana` configs/loaders | Quantized runtime and weight-loading surface; separate from base operator parity. |

## 3. Important config dimensions

Representative checkpoint sweep:

| Config | Pipeline | Scheduler | Layers | Heads x dim | Cross heads x dim | Inner dim | Latent/sample | QK norm | Guidance |
| --- | --- | --- | ---: | --- | --- | ---: | --- | --- | --- |
| `Sana_1600M_1024px` | `SanaPipeline` | DPM++ order 2, flow prediction | 20 | 70 x 32 | 20 x 112 | 2240 | C=32, sample=32 | omitted -> None | true CFG |
| `Sana_1600M_4Kpx_BF16` | `SanaPipeline` | DPM++ order 2, flow prediction | 20 | 70 x 32 | 20 x 112 | 2240 | C=32, sample=128 | omitted -> None | true CFG |
| `SANA1.5_1.6B_1024px` | `SanaPipeline` | DPM++ order 2, flow prediction | 20 | 70 x 32 | 20 x 112 | 2240 | C=32, sample=32 | `rms_norm_across_heads` | true CFG |
| `SANA1.5_4.8B_1024px` | `SanaPipeline` | DPM++ order 2, flow prediction | 60 | 70 x 32 | 20 x 112 | 2240 | C=32, sample=32 | `rms_norm_across_heads` | true CFG |
| `Sana_Sprint_0.6B_1024px` | `SanaSprintPipeline` | SCM trigflow | 28 | 36 x 32 | 16 x 72 | 1152 | C=32, sample=32 | `rms_norm_across_heads` | embedded |
| `Sana_Sprint_1.6B_1024px` | `SanaSprintPipeline` | SCM trigflow | 20 | 70 x 32 | 20 x 112 | 2240 | C=32, sample=32 | `rms_norm_across_heads` | embedded |
| `tiny-random-sana` | `SanaPipeline` | DPM++ order 2 | 2 | 2 x 4 | 2 x 4 | 8 | C=4, sample=16 | omitted -> None | true CFG |

Shared major dimensions from configs:

| Component | Source-derived contract |
| --- | --- |
| Text encoder | `Gemma2Model`, hidden size 2304, 26 layers, 8 attention heads, 4 KV heads, head dim 256, max positions 8192, vocab 256000. Tiny random uses hidden size 8. |
| Tokenizer | `GemmaTokenizerFast` for official repos; `GemmaTokenizer` in tiny random repos. Pipeline forces `padding_side="right"`. |
| Prompt sequence | Pipeline default `max_sequence_length=300`; after optional complex human instruction it selects token 0 plus the last `max_sequence_length - 1` tokens. |
| Denoiser input/output | NCHW latent map `[B, in_channels, H/32, W/32]`; patch size 1 in sampled configs, so token count equals latent H*W. |
| VAE | `AutoencoderDC`, latent channels 32 official / 4 tiny, spatial compression `2 ** (len(encoder_block_out_channels)-1)` = 32 official, scaling factor 0.41407. |
| Base scheduler | `DPMSolverMultistepScheduler`, `prediction_type="flow_prediction"`, `algorithm_type="dpmsolver++"`, `solver_order=2`, `num_train_timesteps=1000`. |
| Sprint scheduler | `SCMScheduler`, `prediction_type="trigflow"`, `sigma_data=0.5`, default timesteps `[1.57080, 1.3, 0]` for two-step mode. |

Recommended first Dinoml scheduler slice: base Sana DPM++ order-2 with
`flow_prediction` because it matches the non-distilled base family and reuses
existing DPM-style scheduler concepts. Sprint SCM should be second because its
trigonometric pre/post-conditioning and embedded guidance are materially
different despite sharing the transformer class.

## 3a. Family variation traps

- Sana transformer input is an NCHW latent map, not Flux-style packed latent
  tokens. Patchify/unpatchify happens inside `SanaTransformer2DModel`.
- Official Sana uses 32 latent channels and VAE scale factor 32. Tiny random
  repos use 4 channels and a much smaller VAE only for tests.
- Base Sana uses true CFG by batch concatenation. Sana Sprint uses embedded
  guidance and does not duplicate unconditional/conditional batches in the same
  way.
- Base 1.0 configs omit `qk_norm`, so source default `None` is effective.
  Sana 1.5 and Sprint configs set `rms_norm_across_heads`.
- `sample_size=32` means 1024px native image with the 32x AutoencoderDC; 4K
  config uses `sample_size=128` and `interpolation_scale=2.0`.
- Self-attention is linear attention (`SanaLinearAttnProcessor2_0`), not SDPA or
  standard flash attention. Cross-attention is SDPA-style with a text attention
  mask.
- The feed-forward block temporarily unflattens tokens to `[B,C,H,W]`, applies
  `GLUMBConv`, then flattens back. Layout passes must preserve this exact H/W
  mapping.
- `AutoencoderDC` uses pixel shuffle/unshuffle and multiscale linear attention
  inside the codec. Treat it as a separate codec island before pulling it into a
  denoiser compile.
- Resolution binning changes requested height/width before denoising, then
  resizes/crops decoded output back to the original size.
- ControlNet is present for Sana and injects per-block residuals; it should not
  inflate the base first-slice op list.

## 4. Runtime tensor contract

For 1024x1024 official base Sana, one image per prompt:

| Boundary | Tensor | Shape/layout | Notes |
| --- | --- | --- | --- |
| Prompt embeddings | `prompt_embeds` | `[B,300,2304]` | From Gemma2 output after selection/duplication. |
| Prompt mask | `prompt_attention_mask` | `[B,300]` | Converted by transformer to additive bias `[B,1,300]`. |
| CFG prompt embeddings | concat negative/positive | `[2B,300,2304]` | Base Sana only when `guidance_scale > 1`. |
| Latent noise | `latents` | `[B,32,32,32]` NCHW | `randn_tensor`, fp32 before transformer dtype cast. |
| Transformer input | `latent_model_input` | `[2B,32,32,32]` for CFG | Timestep expanded to batch and multiplied by `timestep_scale`. |
| Patch tokens | internal | `[B,H_lat*W_lat,inner_dim]` | Patch size 1 for sampled configs; 1024px -> 1024 tokens. |
| Cross text | internal | `[B,300,inner_dim]` | `PixArtAlphaTextProjection(2304 -> inner_dim)` then RMSNorm. |
| Denoiser output | `noise_pred` | `[2B,32,32,32]` | If `out_channels // 2 == latent_channels`, learned-sigma half is dropped. Not active for sampled official configs because out=32. |
| Scheduler state | timesteps/sigmas/state | scheduler-dependent | DPM++ stores multistep model outputs and step index. |
| VAE decode input | `latents / 0.41407` | `[B,32,32,32]` NCHW | No shift factor in Sana pipeline. |
| Decoded image | `image` | `[B,3,1024,1024]` NCHW | `PixArtImageProcessor` postprocess to PIL/NumPy. |

For 4K config, the denoiser latent map is `[B,32,128,128]` and the transformer
sequence is 16384 image tokens before cross-attention. For Sprint, latents are
first multiplied by `scheduler.config.sigma_data`; each step uses
`latents / sigma_data`, a trigflow timestep transform, embedded guidance, and
returns both next latent and denoised sample.

CPU/data-pipeline work: tokenization, optional caption cleaning, complex human
instruction prefixing, resolution bin choice, PIL/NumPy conversion. GPU/runtime
work: denoiser, scheduler arithmetic, CFG arithmetic, VAE decode, and optional
resize/crop if output tensor parity is needed.

## 5. Operator coverage checklist

Tensor/layout ops:

- NCHW latent allocation and shape validation.
- Patch embed/unpatchify: Conv2d/patch projection via `PatchEmbed`, reshape,
  permute, flatten.
- Token `[B,S,C]` to latent map `[B,C,H,W]` and back inside each feed-forward
  block.
- CFG concat/chunk and elementwise guidance arithmetic.
- Text mask conversion from keep/discard mask to additive attention bias.
- Pixel shuffle/unshuffle, repeat-interleave, channel-group mean in
  `AutoencoderDC`.

GEMM/linear ops:

- Patch projection `in_channels -> inner_dim`.
- Timestep embedding MLPs: `Timesteps` + `TimestepEmbedding` or
  `SanaCombinedTimestepGuidanceEmbeddings`.
- Caption projection `PixArtAlphaTextProjection(2304 -> inner_dim)`.
- Self-attention Q/K/V/out projections.
- Cross-attention Q/K/V/out projections.
- Final `proj_out(inner_dim -> patch_size * patch_size * out_channels)`.
- ControlNet zero-initialized linear residual projections in separate candidate.

Attention primitives:

- Self linear attention over image tokens: ReLU Q/K, fp32 `value @ key` then
  `(scores @ query) / denom`, output cast back to original dtype.
- Cross-attention from image tokens to text tokens: SDPA with noncausal mask.
- Optional QK RMSNorm when `qk_norm` is configured.
- `SanaMultiscaleLinearAttention` in AutoencoderDC EfficientViT blocks.

Normalization and adaptive conditioning:

- LayerNorm without affine for image token modulation.
- RMSNorm for caption projection and some q/k or channel-last conv outputs.
- `AdaLayerNormSingle` or guidance-aware time embedding producing six chunks
  of scale/shift/gate conditioning.
- `SanaModulatedNorm` before final projection.

VAE/postprocessing ops:

- Conv2d, depthwise Conv2d, 1x1 Conv2d, SiLU/ReLU/ReLU6.
- Pixel shuffle/unshuffle down/up blocks.
- Bilinear resize and center crop in `PixArtImageProcessor`.
- Tiling/slicing paths are source-supported but not first-slice.

## 6. Denoiser/model breakdown

`SanaTransformer2DModel.forward`:

```text
hidden_states [B,C,H,W]
  -> PatchEmbed -> image tokens [B,H*W,inner_dim]
  -> timestep embedding, optional guidance embedding -> [B,6*inner_dim]
  -> caption_projection + RMSNorm(text) -> [B,L,inner_dim]
  -> N SanaTransformerBlock layers
  -> SanaModulatedNorm + final Linear
  -> reshape/permute unpatchify -> [B,out_channels,H,W]
```

`SanaTransformerBlock`:

```text
scale_shift_table + timestep -> shift/scale/gate for MSA and MLP
LayerNorm -> adaptive scale/shift -> self linear attention -> gated residual
LayerNorm -> cross-attention to text with text mask -> residual
LayerNorm -> adaptive scale/shift
  -> unflatten tokens to H,W -> GLUMBConv -> flatten tokens
  -> gated residual
```

`GLUMBConv`:

```text
Conv1x1(dim -> 2*hidden) -> SiLU
Depthwise Conv3x3(groups=2*hidden)
chunk channels -> value * SiLU(gate)
Conv1x1(hidden -> dim, bias=false)
optional channel RMSNorm
optional residual
```

ControlNet variant: `SanaControlNetModel` patch-embeds both latent input and
`controlnet_cond`, adds a zero-initialized input projection on the control
tokens, runs a shorter stack of `SanaTransformerBlock`s, then emits
zero-initialized linear residual samples multiplied by `conditioning_scale`.

## 7. Attention requirements

Base Sana uses two attention families:

- Image self-attention in `SanaTransformerBlock.attn1` uses
  `SanaLinearAttnProcessor2_0`, not `attention_dispatch.py`.
- Text cross-attention in `attn2` uses local `SanaAttnProcessor2_0`, which
  calls `torch.nn.functional.scaled_dot_product_attention`.

Self linear-attention contract:

- Query/key/value are projected from `[B,S,inner_dim]`.
- Optional QK norm applies before reshaping when configured.
- Reshape order creates `[B,heads,head_dim,S]`-like operands for linear
  attention.
- ReLU is applied to query and key.
- Query/key/value are upcast to fp32.
- Value is padded with a denominator row; output divides by denominator plus
  `1e-15`.
- fp16 output is clipped to `[-65504, 65504]`.

Cross-attention contract:

- Query is image sequence; key/value are text sequence after caption projection.
- Attention mask is text mask converted to additive bias.
- Noncausal SDPA is the eager parity path.
- For official 1.6B configs: image self heads 70 x 32; cross heads 20 x 112.
  Sprint 0.6B uses 36 x 32 self and 16 x 72 cross.

Flash-style constraints:

- Standard flash attention is not a drop-in for Sana self-attention because the
  required operation is linear attention, not softmax attention.
- Cross-attention could use a flash-style provider only when additive mask
  support, dtype, head dim, and sequence length constraints are satisfied.
  Mask-free assumptions are unsafe because prompt attention masks are active.
- `attention_dispatch.py` is useful background for flash/native/xFormers
  constraints, but the base Sana source path does not route self-attention
  through it.

## 8. Scheduler and denoising-loop contract

Base Sana:

- Calls `retrieve_timesteps` on `DPMSolverMultistepScheduler`; custom timesteps
  or sigmas are source-supported when the scheduler supports them.
- Denoiser timestep is expanded to batch and multiplied by
  `transformer.config.timestep_scale`.
- True CFG is implemented by concatenating latents and prompt embeddings along
  batch, then:

```text
noise_pred = uncond + guidance_scale * (text - uncond)
latents = scheduler.step(noise_pred, t, latents)
```

- Recommended first Dinoml slice: host-visible DPM++ order-2 flow-prediction
  loop with compiled denoiser step and compiled pointwise CFG arithmetic.

Sana Sprint:

- `SCMScheduler.set_timesteps` requires either custom timesteps or
  `max_timesteps`; default pipeline uses `num_inference_steps=2`,
  `max_timesteps=1.57080`, `intermediate_timesteps=1.3`.
- Latents are multiplied by `sigma_data` before the loop.
- Pipeline computes `scm_timestep = sin(t) / (cos(t) + sin(t))`.
- Model input is scaled by
  `sqrt(scm_timestep^2 + (1 - scm_timestep)^2)`.
- Model output is transformed by additional trigflow arithmetic, then
  `SCMScheduler.step` computes `pred_x0 = cos(s) * sample - sin(s) *
  model_output` and stochastic previous sample for multistep schedules.

Keep scheduler iteration and state host-visible first. Compile the pure tensor
arithmetic once the scalar timestep/state contract is explicit.

## 9. Position, timestep, and custom math

- `PatchEmbed` handles patch projection and optional sin-cos positional
  embedding when `interpolation_scale` is configured. The 4K checkpoint sets
  `interpolation_scale=2.0`; 1024px official configs omit it.
- Base time embedding uses `AdaLayerNormSingle(inner_dim)`, which emits both
  `timestep` modulation chunks and an embedded timestep for final norm.
- Sprint guidance uses `SanaCombinedTimestepGuidanceEmbeddings`: timestep and
  guidance each pass through sinusoidal `Timesteps(256)` and `TimestepEmbedding`,
  are summed, then projected to `6 * inner_dim`.
- Pipeline multiplies base timesteps by `timestep_scale`; sampled official base
  configs either omit it or set source default 1.0.
- Caption preprocessing, complex human instruction prefixing, and select-index
  prompt slicing are CPU/data-pipeline work and should not be hidden inside the
  denoiser graph.

## 10. Preprocessing and input packing

Text preprocessing:

- Optional `clean_caption` uses BeautifulSoup/ftfy when installed; otherwise it
  falls back with warnings.
- Pipeline lowercases/strips prompts when not cleaning.
- `complex_human_instruction` prepends an instruction string in default calls.
- Tokenizer uses max-length padding and truncation, then selects token index 0
  plus the last `max_sequence_length - 1` tokens.
- Prompt embeddings and masks are repeated for `num_images_per_prompt`.
- Base Sana separately prepares negative prompt embeddings for CFG; Sprint does
  not use negative embeddings in the base pipeline.

Image/latent preprocessing:

- Base text-to-image starts from random latent noise.
- Resolution binning maps requested H/W to fixed aspect-ratio bins depending on
  `transformer.config.sample_size`: 16->512, 32->1024, 64->2048, 128->4096.
- After decode, `resize_and_crop_tensor` bilinearly resizes NCHW decoded images
  and center crops to original requested H/W.

No pipeline-level latent packing exists for Sana. Patchify/unpatchify is a
model-internal NCHW contract.

## 11. Graph rewrite / lowering opportunities

### Rewrite: patch size 1 PatchEmbed to Conv2d + flatten

Source pattern: `PatchEmbed` on NCHW latent map with `patch_size=1`.

Replacement: Conv2d/linear projection to `inner_dim` followed by NCHW-to-token
flatten.

Preconditions: patch size 1, source NCHW contiguous or explicitly described
strides, positional embedding behavior represented, output sequence order
matches `height * width` row-major flattening.

Failure cases: non-1 patch size, 4K interpolation positional embedding not
modeled, NHWC layout pass changing flatten order without a matching rewrite.

Parity test: random latent map through `PatchEmbed` versus rewritten conv +
flatten for each sampled transformer config.

### Rewrite: Sana self linear attention provider

Source pattern: Q/K/V projections -> optional QK norm -> ReLU(Q/K) -> fp32
linear attention with denominator padding -> output projection.

Replacement: dedicated provider-backed `sana_linear_attention` op or fused
kernel family.

Preconditions: noncausal self path, no PAG processor mutation, known heads/head
dim, no attention mask, ReLU feature map, denominator epsilon `1e-15`, fp16
clip preserved.

Failure cases: PAG attention processors, changed feature map, cross-attention
path, q/k norm layout mismatch.

Parity test: processor-level random tensors for qk_norm None and
`rms_norm_across_heads`, fp32/bf16/fp16.

### Rewrite: GLUMBConv as fused conv-gated depthwise block

Source pattern: 1x1 conv -> SiLU -> depthwise 3x3 -> chunk -> gated SiLU ->
1x1 conv -> optional RMSNorm/residual.

Replacement: explicit Conv2d/depthwise Conv2d plus fused activation/gate
epilogue where profitable.

Preconditions: dense NCHW or guarded NHWC island, groups equal channels,
chunk axis is channel axis, residual shape identical.

Failure cases: layout translation does not rewrite channel axis, nonstandard
norm type, dynamic channel group mismatch.

### Rewrite: AutoencoderDC pixel shuffle/unshuffle island

Source pattern: Conv2d + pixel shuffle/unshuffle + shortcut repeat/group mean.

Replacement: layout-aware codec kernels or graph rewrite into explicit
reshape/permute/copy operations.

Preconditions: NCHW semantic axes preserved; scale factor 2; channel
divisibility proven.

Failure cases: arbitrary tensor strides, NHWC translation without axis rewrite,
tiling path active.

## 12. Kernel fusion candidates

Highest priority:

- `SanaLinearAttnProcessor2_0` provider: this is the family-defining operation
  and not covered by normal flash attention.
- LayerNorm/adaptive scale-shift/gate plus residual epilogues in
  `SanaTransformerBlock`.
- GLUMBConv fused pointwise/depthwise/gate sequence.
- DPM++/CFG pointwise arithmetic for the base loop.
- PatchEmbed/unpatchify layout kernels for NCHW latent maps.

Medium priority:

- Cross-attention SDPA/flash-style provider with additive prompt mask support.
- QK RMSNorm before attention for Sana 1.5 and Sprint.
- AutoencoderDC decode conv/pixel-shuffle/EfficientViT blocks.
- Sprint embedded guidance and SCM trigflow arithmetic.

Lower priority:

- PAG attention processor mutation.
- ControlNet residual branch.
- VAE tiling/slicing and overlap blending.
- Gemma2 text encoder compilation; prompt embeddings can be supplied
  externally first.

## 13. Runtime staging plan

Stage 1: Parse Sana base component configs and load weights. Accept external
Gemma prompt embeddings and attention masks.

Stage 2: Implement transformer-only random tensor parity for the tiny random
Sana config, including mask conversion and patch/unpatchify.

Stage 3: Add the `SanaLinearAttnProcessor2_0` provider or decomposed fallback,
then validate one `SanaTransformerBlock`.

Stage 4: Compile full `SanaTransformer2DModel` for
`Sana_1600M_1024px_diffusers` with fixed latent shape `[B,32,32,32]`.

Stage 5: Add base CFG arithmetic and one DPM++ flow-prediction denoising step
with scheduler state in Python.

Stage 6: Add AutoencoderDC decode boundary or call out to a separate codec
artifact.

Stage 7: Add 4K shape admission only after memory and linear-attention
performance are measured.

Stage 8: Add Sana 1.5 QK RMSNorm and 4.8B depth; then separate Sprint SCM and
ControlNet reports.

First Dinoml admission recommendation: admit `sana_base_denoiser_step` as a
bounded transformer-denoiser slice, not a full pipeline. Inputs should be
`latents`, `prompt_embeds`, `prompt_attention_mask`, and scalar/batch timestep;
outputs should be predicted latent residual/noise in source NCHW. Keep Gemma2,
AutoencoderDC, DPM++ loop state, and CFG orchestration outside the compiled
artifact until the denoiser step has parity.

## 14. Parity and validation plan

- Config parse tests for all sampled official and tiny repos.
- Prompt embedding/mask shape tests using cached external tensors.
- PatchEmbed + unpatchify random tensor parity for patch size 1.
- `SanaLinearAttnProcessor2_0` parity for fp32, bf16, fp16, qk_norm None, and
  qk_norm `rms_norm_across_heads`.
- `SanaAttnProcessor2_0` cross-attention parity with additive prompt masks.
- `GLUMBConv` and one full `SanaTransformerBlock` random tensor parity.
- Full tiny random `SanaTransformer2DModel` parity.
- Full official 1.6B one-step denoiser parity at `[1,32,32,32]`.
- CFG arithmetic parity.
- DPM++ scheduler one-step and short-loop parity for flow prediction.
- AutoencoderDC decode parity for `[1,32,32,32]` if included in the first
  product slice.
- Suggested tolerances: fp32 `rtol=1e-4, atol=1e-5`; bf16/fp16 initially
  `rtol=2e-2, atol=2e-2`, tightened after provider kernels stabilize.

## 15. Performance probes

- Self linear-attention provider by latent token length: 1024 for 1024px, 4096
  for 2048px, 16384 for 4K.
- Cross-attention time by prompt length 300 and batch/CFG factor.
- GLUMBConv time split: 1x1 GEMM-like conv, depthwise conv, activation/gate,
  output projection.
- Full denoiser step latency by config: tiny, 0.6B Sprint, 1.6B base, 4.8B.
- DPM++ loop overhead versus denoiser time.
- AutoencoderDC decode throughput and memory with and without tiling.
- NHWC/channel-last probe for GLUMBConv and AutoencoderDC only under guarded
  local layout regions.
- VRAM/workspace at 1024px and 4K; 4K attention length is the main pressure
  point.

## 16. Scope boundary and separate candidates

Separate candidate reports, not ignored:

- `sana_sprint`: embedded guidance plus SCM scheduler/trigflow math.
- `sana_sprint_img2img`: VAE encode and image-strength/noising contract.
- `sana_controlnet`: control latent branch and per-block residual injection.
- `sana_pag`: perturbed-attention processor mutation for Sana linear attention.
- `sana_lora_adapters`: transformer/text-encoder LoRA load/fuse/unfuse/runtime
  adapter state.
- `sana_dcae_codec`: AutoencoderDC encode/decode, tiling, pixel shuffle, and
  multiscale linear attention as a reusable codec report.
- `sana_video`: separate video pipeline/model family.
- `nunchaku_sana_quant`: quantized Sana runtime/weight-loading path.
- Rare or swapped schedulers beyond the sampled DPM++ and SCM configs.

Genuinely ignored/out of scope for this audit:

- Multi-GPU/context parallel paths.
- Callback mutation and interactive interrupt.
- XLA, NPU, MPS, Flax, and ONNX variants.
- Safety checker and NSFW filtering.
- Training, losses, dropout, and gradient checkpointing.

## 17. Final implementation checklist

- [ ] Parse Sana model index and component configs from `H:/configs`.
- [ ] Load `SanaTransformer2DModel` weights for tiny random and 1.6B base.
- [ ] Accept external Gemma prompt embeddings and attention masks.
- [ ] Implement mask-to-bias conversion for text cross-attention.
- [ ] Implement patch size 1 PatchEmbed/unpatchify parity.
- [ ] Implement or lower `SanaLinearAttnProcessor2_0`.
- [ ] Implement QK RMSNorm variants needed by Sana 1.5/Sprint.
- [ ] Implement `SanaTransformerBlock` adaptive norm/gate/residual path.
- [ ] Implement GLUMBConv decomposition/fusion.
- [ ] Add full tiny random transformer parity.
- [ ] Add one official base denoiser-step parity.
- [ ] Add base CFG arithmetic.
- [ ] Add DPM++ flow-prediction scheduler step parity.
- [ ] Add AutoencoderDC decode boundary or separate codec artifact.
- [ ] Benchmark 1024px and 4K token lengths.
- [ ] Open separate reports for Sprint, ControlNet, PAG, DC-AE, and Sana video.
