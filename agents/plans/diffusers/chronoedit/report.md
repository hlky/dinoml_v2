# Diffusers ChronoEdit Operator and Integration Report

Candidate slug: `chronoedit`

## 1. Source basis

```text
Diffusers commit/version:
  Local checkout diffusers at b3a515080752a3ba7ca92161e25530c7f280f629.

Model id(s):
  nvidia/ChronoEdit-14B-Diffusers
  nvidia/ChronoEdit-14B-Diffusers-Upscaler-Lora
  nvidia/ChronoEdit-14B-Diffusers-Paint-Brush-Lora
  Local/mirror stubs:
    H:/configs/kayte0342/chronoedit/model_index.json
    H:/configs/vantagewithai/ChronoEdit-GGUF/model_index.json

Config sources:
  Local cache was checked first. It only contained a Wan I2V mirror model_index
  for `kayte0342/chronoedit` and an empty GGUF model_index stub.
  Official raw reads for `nvidia/ChronoEdit-14B-Diffusers` returned 401
  unauthenticated; authenticated `huggingface_hub` retry returned 404/no-access
  in this environment. The public Hugging Face commit
  `34ab6b28471e9f77ce026ab7778d0beac5193596` was inspected for component
  config contents. No configs were saved because the owned write path for this
  task is only this report.

Pipeline files inspected:
  diffusers/src/diffusers/pipelines/chronoedit/pipeline_chronoedit.py
  diffusers/src/diffusers/pipelines/chronoedit/pipeline_output.py

Model files inspected:
  diffusers/src/diffusers/models/transformers/transformer_chronoedit.py
  diffusers/src/diffusers/models/transformers/transformer_wan.py
  diffusers/src/diffusers/models/autoencoders/autoencoder_kl_wan.py
  diffusers/src/diffusers/models/attention_dispatch.py
  diffusers/src/diffusers/models/attention.py
  diffusers/src/diffusers/models/attention_processor.py
  diffusers/src/diffusers/models/embeddings.py
  diffusers/src/diffusers/models/normalization.py

Scheduler/processors/helpers inspected:
  diffusers/src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py
  diffusers/src/diffusers/schedulers/scheduling_unipc_multistep.py
    (used because the official commit/model_index and Diffusers docs still show
    UniPCMultistepScheduler examples despite current ChronoEditPipeline typing
    FlowMatchEulerDiscreteScheduler)
  diffusers/src/diffusers/video_processor.py
  diffusers/src/diffusers/image_processor.py
  diffusers/src/diffusers/loaders/lora_pipeline.py

External component configs inspected:
  UMT5EncoderModel / T5TokenizerFast config fields from public commit diff.
  CLIPVisionModelWithProjection and CLIPImageProcessor config fields from public
  commit diff. Current pipeline imports CLIPVisionModel and uses hidden states,
  not projection output.

Missing files or assumptions:
  Official component configs could not be fetched through raw or authenticated
  HF APIs in this environment. Dimensions below are source defaults plus public
  commit-diff config facts. The current source file is newer than the 0.33.1
  model_index in the public commit: current code registers
  `ChronoEditPipeline`, `ChronoEditTransformer3DModel`, `CLIPVisionModel`, and
  `FlowMatchEulerDiscreteScheduler`, while the commit model_index names
  `WanImageToVideoPipeline`, `WanTransformer3DModel`,
  `CLIPVisionModelWithProjection`, and `UniPCMultistepScheduler`.
```

## 2. Pipeline and component graph

ChronoEdit is an image-conditioned video/editing pipeline built from Wan-style
latent video components. The first image is VAE-encoded into a latent condition;
the transformer denoises a latent video map concatenated with a condition map;
CLIP image tokens and UMT5 text tokens are projected into the transformer
cross-attention context. Optional temporal reasoning keeps a multi-frame latent
trajectory for early steps, then drops to the first and last latent frames.

```text
input image + edit prompt
  -> CLIPImageProcessor + CLIPVisionModel hidden states
  -> T5TokenizerFast/AutoTokenizer + UMT5EncoderModel prompt embeddings
  -> VideoProcessor preprocess image to NCHW
  -> AutoencoderKLWan encode [B,3,T,H,W] condition video
  -> latent noise [B,16,T_lat,H/8,W/8] + condition [B,20,T_lat,H/8,W/8]
  -> denoising loop:
       concat latent + condition along channel -> [B,36,T_lat,H/8,W/8]
       ChronoEditTransformer3DModel patchifies to video tokens
       positive denoiser call; optional negative denoiser call for CFG
       scheduler.step over latent sample only
       optional temporal-reasoning latent/history prune
  -> unstandardize latents with Wan VAE stats
  -> AutoencoderKLWan decode
  -> VideoProcessor postprocess
```

Required first-slice components:

| Component | Class/file | Notes |
| --- | --- | --- |
| Pipeline | `ChronoEditPipeline`, `pipeline_chronoedit.py` | Current source contract. |
| Denoiser | `ChronoEditTransformer3DModel`, `transformer_chronoedit.py` | Wan-derived 3D DiT with 36-channel input and 16-channel output. |
| VAE | `AutoencoderKLWan`, `autoencoder_kl_wan.py` | Encode required for image condition; decode required for output. |
| Text encoder | `UMT5EncoderModel` + `AutoTokenizer`/`T5TokenizerFast` | Accept external prompt embeds first. |
| Image encoder | `CLIPVisionModel` in current source | Produces hidden states used as added K/V cross-attention context. |
| Image/video processors | `CLIPImageProcessor`, `VideoProcessor` | 224px CLIP image path plus NCHW/NCTHW VAE path. |
| Scheduler | Current source type `FlowMatchEulerDiscreteScheduler`; public config/examples `UniPCMultistepScheduler` | Treat scheduler parity as a blocker until artifact config is resolved. |

Separate candidate reports:

| Candidate | Classes/files | Runtime delta |
| --- | --- | --- |
| `chronoedit_lora_adapters` | `WanLoraLoaderMixin`, transformer `PeftAdapterMixin`; NVIDIA upscaler/paint-brush/distill LoRA repos | LoRA load/fuse/unfuse/runtime adapter mutation, including multiple LoRA stacking shown in docs. |
| `chronoedit_temporal_reasoning` | `ChronoEditPipeline.__call__` | Mid-loop frame prune and scheduler history mutation; should follow base 5-frame edit parity. |
| `chronoedit_scheduler_compat` | `FlowMatchEulerDiscreteScheduler`, `UniPCMultistepScheduler` | Source/config mismatch needs a focused parity decision. |
| `wan_i2v_shared_codec` | `AutoencoderKLWan`, `WanTransformer3DModel` | Shared Wan 2.1 I2V base behavior and VAE optimization island. |

No family-local ControlNet, T2I-Adapter, GLIGEN, IP-Adapter, inpaint,
depth2img, or upscaling pipeline class exists under `pipelines/chronoedit`.
Upscaler/Paint-Brush surfaces are LoRA artifacts, not separate pipeline classes.

## 3. Important config dimensions

Representative config sweep:

| Config source | Pipeline/model class facts | Denoiser dims | Scheduler | Variant trap |
| --- | --- | --- | --- | --- |
| Public NVIDIA commit `34ab6b2` | `WanImageToVideoPipeline`, `WanTransformer3DModel`, `AutoencoderKLWan`, `CLIPVisionModelWithProjection`, `UMT5EncoderModel` | 40 layers, 40 heads, head dim 128, ffn 13824, `in_channels=36`, `out_channels=16`, patch `(1,2,2)`, `text_dim=4096`, `image_dim=1280`, `added_kv_proj_dim=5120` | `UniPCMultistepScheduler`, flow prediction, flow shift 5.0, order 2 | Public config lags current source class names. |
| Current Diffusers source | `ChronoEditPipeline`, `ChronoEditTransformer3DModel`, `CLIPVisionModel`, `FlowMatchEulerDiscreteScheduler` typing | Same constructor defaults plus `rope_temporal_skip_len=8`; class-specific RoPE handles 2-frame edit path | `FlowMatchEulerDiscreteScheduler` annotation and source import | Scheduler mismatch with official config/docs. |
| `H:/configs/kayte0342/chronoedit` | Mirror model_index: Wan I2V classes, `CLIPVisionModelWithProjection`, `UniPCMultistepScheduler` | No component configs cached | UniPC model_index only | Useful as mirror evidence, not a ChronoEdit component sweep. |
| `H:/configs/vantagewithai/ChronoEdit-GGUF` | Empty `{}` model_index | Unknown | Unknown | Not useful for operator dimensions. |

Denoiser config facts:

| Field | Value |
| --- | --- |
| `patch_size` | `(1,2,2)` |
| `in_channels` / `out_channels` | 36 / 16 |
| `num_attention_heads` / `attention_head_dim` | 40 / 128 |
| inner dim | 5120 |
| `num_layers` | 40 |
| `ffn_dim` | 13824 |
| `text_dim` / `image_dim` | 4096 / 1280 |
| `added_kv_proj_dim` | 5120 |
| `qk_norm` | `rms_norm_across_heads` |
| `cross_attn_norm` | true |
| `rope_max_seq_len` | 1024 |
| `rope_temporal_skip_len` | source default 8; omitted from public 0.33.1 config, so current source default applies if loaded into current class. |

VAE config facts:

| Field | Value |
| --- | --- |
| Class | `AutoencoderKLWan` |
| `base_dim`, `z_dim` | 96, 16 |
| `dim_mult` | `[1,2,4,4]` |
| `num_res_blocks` | 2 |
| `temperal_downsample` | `[false,true,true]` |
| `scale_factor_temporal`, `scale_factor_spatial` | Source defaults 4 / 8; public config omits these, current source supplies defaults. |
| latent stats | 16-element Wan `latents_mean` and `latents_std`; pipeline standardizes with `(latent - mean) * (1/std)` and decodes with `latent / (1/std) + mean`. |

External encoder facts:

| Component | Config facts |
| --- | --- |
| UMT5 encoder | `d_model=4096`, `d_ff=10240`, `d_kv=64`, gated GELU, dropout 0.1, architecture `UMT5EncoderModel`. |
| Tokenizer | Pipeline imports `AutoTokenizer`; public model_index says `T5TokenizerFast`; source prompt path pads/truncates to `max_sequence_length=512`. |
| CLIP vision | hidden 1280, 32 layers, 16 heads, patch 14, image size 224, projection 1024 in public config; source uses hidden states `[-2]`, so projected pooled output is not first-slice runtime input. |
| CLIP processor | resize/center-crop/normalize to 224 with CLIP mean/std. |

Recommended first Dinoml scheduler slice:

- For current source parity, start with `FlowMatchEulerDiscreteScheduler` static
  non-stochastic step only if the loaded artifact has a FlowMatch config.
- For official public checkpoint parity, a separate `UniPCMultistepScheduler`
  flow-prediction slice with `flow_shift=5.0`, `solver_order=2`,
  `solver_type="bh2"`, `predict_x0=true`, `use_flow_sigmas=true`, and
  `final_sigmas_type="zero"` is required.
- Do not claim end-to-end parity until the actual loaded scheduler config is
  resolved.

## 3a. Family variation traps

- Current source and public config disagree on pipeline, transformer, CLIP, and
  scheduler class names. The model code is Wan-derived, but current source has
  ChronoEdit-specific input width and RoPE behavior.
- Transformer input is NCTHW latent maps, not pre-packed tokens. Patchify is an
  internal Conv3d with kernel/stride `(1,2,2)`.
- `in_channels=36` is source-consistent but easy to misread: noisy latents are
  16 channels; the encoded condition is 16 channels; the first-frame mask is
  reshaped from frame space into 4 latent mask channels because the VAE temporal
  scale is 4. Thus `16 + 16 + 4 = 36`.
- The transformer cross-attention processor hardcodes the text context length as
  512 and treats any prefix tokens in `encoder_hidden_states` as image context.
  Prompt max length changes are unsafe unless this split is updated.
- CLIP image tokens are not IP-Adapter; they are added K/V projections inside
  the Wan cross-attention path.
- Temporal reasoning prunes latents to first/last frames mid-loop and mutates
  scheduler history tensors. This is a separate runtime state trap.
- `num_frames` is forced to 5 when temporal reasoning is disabled. When enabled,
  `(num_frames - 1)` must be divisible by VAE temporal scale 4.
- Source latents and VAE tensors are NCTHW. NDHWC is only a guarded VAE/Conv3d
  optimization candidate.

## 4. Runtime tensor contract

For default non-temporal-reasoning 480x832, 5-frame edit:

| Boundary | Tensor | Shape/layout | Notes |
| --- | --- | --- | --- |
| CLIP image input | processor pixels | `[B,3,224,224]` NCHW | CPU/GPU preprocessing; CLIP normalization. |
| CLIP image embeds | `image_embeds` | `[B,257,1280]` inferred from ViT-H/14 hidden states | Source returns hidden state `[-2]`; repeated with `repeat(batch_size,1,1)`. |
| Prompt embeds | `prompt_embeds` | `[B,512,4096]` | Mask-valid tokens copied, padded to max length. |
| Negative embeds | `negative_prompt_embeds` | `[B,512,4096]` when CFG | Empty string default if not provided. |
| Input image | VAE/video preprocess | `[B,3,H,W]` then `[B,3,1,H,W]` | `VideoProcessor.preprocess` returns NCHW for image path. |
| Condition video | VAE encode input | `[B,3,5,H,W]` NCTHW | First frame image, remaining frames zeros. |
| Noisy latents | denoised sample | `[B,16,2,H/8,W/8]` for 5 frames | `T_lat=(5-1)//4+1=2`, e.g. `[B,16,2,60,104]`. |
| Condition | `condition` | `[B,20,2,H/8,W/8]` | Four reshaped mask channels plus 16 encoded condition latent channels. |
| Transformer input | `latent_model_input` | `[B,36,2,60,104]` | 16 noisy latent channels plus 20 condition channels. |
| Patch tokens | internal | `[B,2*30*52,5120] = [B,3120,5120]` | Conv3d `(1,2,2)` then flatten/transpose. |
| Cross context | `encoder_hidden_states` | `[B,769,5120]` after image/text projection concat | Image prefix 257, text suffix 512; split is hardcoded. |
| Denoiser output | `noise_pred` | `[B,16,2,60,104]` | Same as latent sample. |
| Scheduler sample | `latents` | `[B,16,2,60,104]` | Scheduler updates noisy latents only, not condition channels. |
| Decode input | unstandardized latents | `[B,16,2,60,104]` | Wan VAE decode. |
| Decoded video | output | `[B,3,5,H,W]` | `VideoProcessor.postprocess_video` returns `[B,T,H,W,C]` for np. |

CPU/data-pipeline work: text cleanup/tokenization, CLIP image preprocessing,
PIL/NumPy conversion, output formatting. GPU/runtime work: UMT5/CLIP if
admitted, VAE encode/decode, transformer, CFG arithmetic, scheduler update, and
temporal reasoning prune.

Cacheable: prompt embeddings, negative embeddings, CLIP image embeddings, VAE
condition latents, RoPE tables for fixed latent shape, scheduler tables, and
possibly the condition mask.

## 5. Operator coverage checklist

Tensor/layout ops:

- NCHW image preprocess, NCTHW video/latent tensors.
- `unsqueeze`, concat, repeat, repeat_interleave, view, transpose, flatten,
  permute, chunk, stack, mask fill.
- Temporal reasoning slice `[:, :, [0,-1]]` and scheduler history slicing.
- Per-channel latent standardization and unstandardization over NCTHW.
- Patchify/unpatchify inside transformer:
  `Conv3d(C -> 5120, kernel=stride=(1,2,2))`, flatten spatial/temporal tokens,
  then inverse reshape/permute/flatten after `Linear(5120 -> 16*1*2*2)`.

Convolution/downsample/upsample ops:

- Transformer patch Conv3d, source NCTHW.
- Wan VAE causal Conv3d, Conv2d spatial resamplers, zero pad, nearest-exact
  upsample, AvgDown3D/DupUp3D reshape paths.

GEMM/linear ops:

- UMT5 encoder if admitted later.
- CLIP ViT image encoder if admitted later.
- Timestep embedding MLP, PixArt text projection 4096 -> 5120.
- CLIP image projection via `WanImageEmbedding`: LayerNorm, FeedForward
  1280 -> 5120, LayerNorm.
- 40 blocks of Q/K/V, added K/V, output projections, and GELU-approximate FFN.

Attention primitives:

- Video-token self-attention with QK RMSNorm and 3D RoPE.
- Text cross-attention plus added image K/V branch, implemented as two
  attention calls whose outputs are summed.
- VAE single-head spatial SDPA in `WanAttentionBlock`.

Normalization/adaptive conditioning:

- FP32LayerNorm before attention/FFN and final output norm.
- RMSNorm over flattened Q/K projection width.
- Wan VAE RMS norm over channel axis.
- Ada scale/shift/gate from timestep for attention and FFN residuals.

Scheduler and guidance arithmetic:

- CFG as two separate transformer calls: `uncond + scale * (cond - uncond)`.
- FlowMatch Euler or UniPC flow update depending loaded scheduler.
- No guidance-rescale in current ChronoEdit source.

## 6. Denoiser/model breakdown

`ChronoEditTransformer3DModel.forward`:

```text
hidden_states [B,C,T,H,W]
-> ChronoEditRotaryPosEmbed(hidden_states)
-> Conv3d patch_embedding C -> 5120, kernel/stride (1,2,2)
-> flatten to [B,S,5120]
-> Timesteps + TimestepEmbedding + Linear -> timestep_proj [B,6,5120]
-> PixArtAlphaTextProjection text [B,512,4096] -> [B,512,5120]
-> optional WanImageEmbedding image [B,257,1280] -> [B,257,5120]
-> concat image/text context [B,769,5120]
-> 40 x WanTransformerBlock
-> final FP32LayerNorm with timestep shift/scale
-> Linear 5120 -> 64
-> unpatchify to [B,16,T,H,W]
```

`WanTransformerBlock`:

```text
scale_shift_table + timestep_proj -> six vectors
norm1 fp32 -> scale/shift -> self-attention(QKV, QK RMSNorm, RoPE) -> gated residual
norm2 fp32 -> cross-attention(Q from latents, K/V from text, added K/V image branch) -> residual
norm3 fp32 -> scale/shift -> GELU-approx FeedForward -> gated residual
```

VAE encode/decode:

- Encoder chunks time as first frame then groups of 4 frames, using cached
  causal Conv3d features.
- Decode processes latent frames one at a time, reusing cached Conv3d features;
  first chunk has special temporal upsample trimming.
- Tiling/slicing exist but should be disabled for first parity.

## 7. Attention requirements

Primary transformer attention implementation is `WanAttnProcessor` in
`transformer_chronoedit.py`, calling `dispatch_attention_fn`.

Required variants:

- Self-attention over latent video patch tokens, noncausal, no mask.
- Cross-attention from latent tokens to text tokens, no mask in current source.
- Added-K/V image branch when `added_kv_proj_dim` is present. Processor splits
  context as `image_context_length = encoder_hidden_states.shape[1] - 512`,
  then text context is the last 512 tokens.
- Q/K RMSNorm before head unflatten for self and text cross attention; added
  image K has its own RMSNorm.
- 3D RoPE applies to self-attention Q/K only. ChronoEdit RoPE has a special
  two-frame path selecting temporal positions `[0,-1]` from a skip length of 8.
- Fused QKV/KV projections are source-supported through `fuse_projections`, but
  not required for first parity.

Flash-style/provider notes:

- Self-attention is a plausible flash-style target: head dim 128, noncausal,
  no mask, sequence length around 3120 for 480x832x5.
- Cross-attention needs separate validation because it has an added image K/V
  branch implemented as an extra attention call plus sum.
- RoPE and QK RMSNorm are explicit pre-attention operations unless fused under
  exact guards.
- The hardcoded 512 text split is a provider/layout guard; variable prompt
  length cannot silently change the context split.

## 8. Scheduler and denoising-loop contract

Current source imports and types `FlowMatchEulerDiscreteScheduler` and calls:

```text
scheduler.set_timesteps(num_inference_steps, device=device)
for t in timesteps:
  latent_model_input = cat([latents, condition], dim=1)
  noise_pred = transformer(... prompt_embeds, image_embeds)
  if CFG:
    noise_uncond = transformer(... negative_prompt_embeds, image_embeds)
    noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)
  latents = scheduler.step(noise_pred, t, latents)
```

However, public NVIDIA model_index/config and docs show UniPC flow scheduler
usage. UniPC config facts from the public commit: `prediction_type =
flow_prediction`, `use_flow_sigmas=true`, `flow_shift=5.0`, `solver_order=2`,
`solver_type=bh2`, `predict_x0=true`, `lower_order_final=true`,
`final_sigmas_type=zero`, `timestep_spacing=linspace`.

First implementation should keep scheduler table generation and loop index state
host-visible. Compile denoiser, CFG arithmetic, and one scheduler step only
after actual loaded scheduler class is known. Temporal reasoning history pruning
should remain host control until base loop parity is stable.

## 9. Position, timestep, and custom math

- `ChronoEditRotaryPosEmbed` splits head dim 128 into temporal/height/width
  rotary sections. Height and width get `2 * (head_dim // 6)` each; temporal
  receives the remainder.
- For two latent frames, temporal RoPE uses the first and last positions from
  `temporal_skip_len=8`, not consecutive positions.
- Timestep embedding uses `Timesteps(freq_dim=256, flip_sin_to_cos=True)` then
  `TimestepEmbedding` and SiLU+Linear to six modulation vectors.
- Final output norm uses a separate timestep-derived shift/scale pair.
- Wan VAE standardization:
  encode condition uses `(latent - mean) * (1/std)`;
  decode uses `latents / (1/std) + mean`.

Precompute candidates: prompt embeddings, CLIP image embeddings, VAE condition
latents/mask, RoPE tables for fixed latent shape and temporal mode, scheduler
tables.

Dynamic: image resolution, temporal-reasoning flag, number of latent frames,
prompt length if source is fixed, guidance scale, and scheduler class/config.

## 10. Preprocessing and input packing

Text:

- `prompt_clean` applies ftfy/html/whitespace cleanup when ftfy is available.
- Tokenizer pads/truncates to `max_sequence_length=512` by default.
- Source computes valid token lengths from attention mask, slices hidden states
  to valid length, then pads back to 512. It does not pass an attention mask to
  transformer cross-attention.
- Negative prompt defaults to empty string for CFG.

Image/video:

- CLIP path resizes/crops/norms to 224 and returns hidden state `[-2]`.
- VAE path preprocesses the input image to NCHW at requested H/W, then constructs
  a video with image at frame 0 and zero frames after it.
- VAE encode returns posterior mode (`sample_mode="argmax"`) for the condition.
- Condition mask is one for the first frame and zero after it, expanded to
  latent temporal grouping and concatenated with encoded condition latents.
- Non-temporal mode overrides `num_frames` to 5.
- Decode path optionally decodes reasoning frames and edit frames separately
  when temporal reasoning remains with more than two latent frames.

## 11. Graph rewrite / lowering opportunities

### Rewrite: ChronoEdit latent/condition admission

Source pattern: build `[latents, mask, VAE(condition)]` by channel concat, then
Conv3d patchify.

Replacement: explicit condition-pack op with a shape assertion before
transformer admission.

Preconditions: VAE `z_dim=16`, latent sample channels 16, VAE temporal scale 4
so the mask reshapes to four channels, and transformer `in_channels=36`.
Failure cases: different VAE temporal scale, direct user-supplied condition with
unknown mask packing, or future ChronoEdit variants that alter condition
channels.

### Rewrite: Conv3d patchify/unpatchify

Source pattern: `Conv3d(C -> 5120, kernel=stride=(1,2,2))`, flatten to tokens,
then `Linear(5120 -> 64)` and inverse reshape/permute.

Replacement: canonical video patch embed/unpatch op or Conv3d/Linear lowering.

Preconditions: source NCTHW, H/W divisible by 2 after VAE scale, `patch_t=1`.
Layout constraints: NDHWC requires Conv3d weight transform and inverse
unpatchify flatten-order rewrite. Failure cases: future patch sizes or
condition-channel mismatch.

### Rewrite: Wan attention with added image K/V

Source pattern: QKV -> QK RMSNorm -> RoPE for self attention; cross attention
splits image/text context, computes text attention and image added-K/V attention
separately, sums outputs.

Replacement: explicit attention region with optional fused projection/RMSNorm/
RoPE prelude and added-K/V branch.

Preconditions: text suffix length exactly 512, image prefix length known,
provider supports head_dim 128 and noncausal attention. Failure cases: prompt
length changes, missing image branch support, masked text attention.

### Rewrite: Scheduler step

Source pattern: FlowMatch Euler or UniPC flow step over NCTHW latents.

Replacement: scheduler-specific pointwise/multistep kernel with explicit
artifact-visible state.

Preconditions: actual checkpoint scheduler is known. Failure cases: current
source/config mismatch, temporal reasoning mid-loop history slicing.

## 12. Kernel fusion candidates

Highest priority:

- Shape-guarded ChronoEdit condition concat and patch Conv3d admission.
- 5120-wide QKV/add-KV projections, QK RMSNorm, RoPE, and attention dispatch.
- Ada scale/shift/gate plus residual epilogues for 40 Wan blocks.
- GELU-approx FeedForward fusion at hidden 5120 / inner 13824.
- Wan VAE encode/decode Conv3d + RMSNorm + SiLU residual blocks.

Medium priority:

- CLIP image embedding projection `LayerNorm -> FeedForward -> LayerNorm`.
- CFG two-call arithmetic over NCTHW latents.
- Scheduler step kernels for FlowMatch Euler and/or UniPC flow after config
  resolution.
- RoPE table generation/cache including two-frame temporal skip.
- VAE condition encode caching.

Lower priority:

- Temporal reasoning mid-loop prune and scheduler history mutation.
- LoRA multi-adapter load/fuse/unfuse.
- VAE tiling/slicing and cache-aware chunk scheduling.
- CLIP/UMT5 full encoder compilation; embeddings can be supplied externally
  first.

## 13. Runtime staging plan

Stage 1: Parse current source and public config facts. Add a hard admission check
that the pipeline condition packing constructs the transformer's configured
36-channel input.

Stage 2: Build a base denoiser-step artifact with externally supplied prompt
embeds `[B,512,4096]`, image embeds `[B,257,1280]`, NCTHW latents, condition
tensor, and scalar timestep.

Stage 3: Validate patch Conv3d, RoPE, one `WanTransformerBlock`, and full
`ChronoEditTransformer3DModel` on synthetic tensors with the actual loaded
channel count.

Stage 4: Add VAE encode of the first-frame condition and VAE decode of output
latents, tiling disabled.

Stage 5: Resolve scheduler class. Implement FlowMatch Euler for current source
smoke and UniPC flow for public checkpoint parity if the loaded checkpoint uses
the public config.

Stage 6: Add CFG two-call orchestration.

Stage 7: Run short non-temporal 5-frame edit loop with scheduler in host
control.

Stage 8: Add temporal reasoning mode: full latent trajectory for early steps,
then first/last latent prune and scheduler history mutation.

Stage 9: Add LoRA/distill/paint-brush/upscaler adapter handling as separate
admissions.

First Dinoml staging recommendation: `chronoedit_base_denoiser_step`, with
external text and image embeddings and prebuilt condition tensor. Do not start
with end-to-end pipeline parity because scheduler and input-channel contracts
need artifact confirmation.

## 14. Parity and validation plan

- Config probe: attempt authenticated component config read; compare actual
  loaded transformer `in_channels` to pipeline concat output.
- Prompt embedding parity for tokenization/valid-length padding to 512.
- CLIP image embed extraction parity for hidden state `[-2]`.
- VAE condition encode parity and latent standardization.
- Condition mask construction parity for 5 frames and a temporal-reasoning
  frame count such as 81.
- Patch Conv3d/unpatchify parity for `[B,C,2,60,104]`.
- RoPE parity for two-frame path and multi-frame temporal-reasoning path.
- One attention processor parity with added image K/V.
- One `WanTransformerBlock` parity.
- Full ChronoEdit transformer forward parity on reduced spatial size.
- CFG arithmetic parity.
- Scheduler table/step parity for FlowMatch Euler and UniPC flow, selected by
  actual artifact config.
- VAE decode parity for `[B,16,T,H/8,W/8]`.
- Temporal reasoning prune parity, including scheduler history slices.
- Suggested tolerances: scheduler fp32 `rtol=1e-5, atol=1e-6`; transformer fp32
  `rtol=1e-4, atol=1e-5`; bf16/fp16 initially `rtol=2e-2, atol=2e-2`.

## 15. Performance probes

- Denoiser step by latent resolution: 480x832x5, 720x1280x5, 960x960x5.
- Temporal reasoning sweep by frame count and prune step.
- Attention backend comparison for head dim 128 with and without added image
  K/V branch.
- Per-block time split: patch Conv3d, QKV/add-KV, attention, FFN, gates.
- VAE condition encode throughput and reuse benefit across prompts.
- VAE decode throughput for 2 latent frames and long reasoning trajectories.
- CFG overhead: two transformer calls versus guidance scale disabled.
- Scheduler overhead for FlowMatch Euler versus UniPC flow.
- VRAM and temporary usage for 40-layer 5120-wide transformer.

## 16. Scope boundary and separate candidates

Separate candidate reports, not ignored:

- `chronoedit_lora_adapters`: NVIDIA distill/upscaler/paint-brush LoRAs,
  multiple adapter loading, fuse/unfuse, and LoRA scale.
- `chronoedit_temporal_reasoning`: mid-loop latent prune and scheduler history
  mutation.
- `chronoedit_scheduler_compat`: current FlowMatch source versus public UniPC
  checkpoint config.
- `chronoedit_wan_vae_codec`: Wan VAE encode/decode tiling, slicing, temporal
  cache, and NDHWC optimization.
- `chronoedit_clip_umt5_encoders`: full text/image encoder compilation instead
  of external embeddings.

Genuinely ignored/out of scope for this audit:

- Multi-GPU/context parallel paths.
- Callback mutation and interactive interrupt.
- XLA, NPU, MPS, Flax, and ONNX variants.
- Safety checker and NSFW filtering.
- Training, losses, dropout, and gradient checkpointing.
- ControlNet, T2I-Adapter, GLIGEN, IP-Adapter, inpaint, depth2img, and pipeline
  upscaling: no ChronoEdit folder implementation was found.

## 17. Final implementation checklist

- [ ] Parse ChronoEdit public/local configs and record source/config mismatch.
- [ ] Resolve actual loaded scheduler class for `nvidia/ChronoEdit-14B-Diffusers`.
- [ ] Validate transformer `in_channels=36` against pipeline condition concat shape.
- [ ] Accept external UMT5 prompt and negative prompt embeddings.
- [ ] Accept external CLIP image hidden states.
- [ ] Implement ChronoEdit condition mask and VAE condition latent contract.
- [ ] Implement NCTHW Conv3d patchify and unpatchify parity.
- [ ] Implement ChronoEdit 3D RoPE, including two-frame temporal skip.
- [ ] Implement Wan attention with QK RMSNorm, RoPE, text cross-attention, and added image K/V branch.
- [ ] Implement one ChronoEdit/Wan transformer block parity.
- [ ] Implement full ChronoEdit transformer denoiser-step parity.
- [ ] Implement CFG two-call arithmetic.
- [ ] Implement selected scheduler step: FlowMatch Euler or UniPC flow.
- [ ] Implement AutoencoderKLWan encode/decode boundary with tiling disabled.
- [ ] Add 5-frame non-temporal edit loop parity.
- [ ] Add temporal reasoning prune and scheduler history parity as a separate stage.
- [ ] Benchmark denoiser step, attention backend, VAE encode/decode, and CFG overhead.
