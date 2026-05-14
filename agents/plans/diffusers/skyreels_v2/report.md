# Diffusers SkyReels V2 Operator and Integration Report

Candidate slug: `skyreels_v2`

## 1. Source basis

```text
Diffusers commit/version:
  Local checkout X:/H/diffusers at b3a515080752a3ba7ca92161e25530c7f280f629.

Model id(s):
  Skywork/SkyReels-V2-T2V-14B-540P-Diffusers
  Skywork/SkyReels-V2-T2V-14B-720P-Diffusers
  Skywork/SkyReels-V2-I2V-1.3B-540P-Diffusers
  Skywork/SkyReels-V2-I2V-14B-720P-Diffusers
  Skywork/SkyReels-V2-DF-1.3B-540P-Diffusers
  Skywork/SkyReels-V2-DF-14B-720P-Diffusers

Config sources:
  Local cache checked first:
    H:/configs/Skywork/SkyReels-V2-DF-1.3B-540P-Diffusers/model_index.json
    H:/configs/Skywork/SkyReels-V2-DF-14B-720P-Diffusers/model_index.json
    H:/configs/Skywork/SkyReels-V2-I2V-1.3B-540P-Diffusers/model_index.json
  Official Hub API/raw component configs inspected in-memory, not saved because
  this worker owns only this report path:
    transformer/config.json, vae/config.json, scheduler/scheduler_config.json,
    text_encoder/config.json, tokenizer/tokenizer_config.json, plus I2V
    image_encoder/config.json and image_processor/preprocessor_config.json.
  Official Hub API model SHAs checked:
    T2V 540P: 8479c72a1a6ffe1821215aa413e3f6072c8ee10c
    T2V 720P: b07940972bf0efad02694755022931e469430e96
    I2V 1.3B 540P: fd853c3d47f7746c8ca1f5226c323e55e53c7bf2
    I2V 14B 720P: 375e297a4952bacfdb01af224f495514738381c5
    DF 1.3B 540P: 958acd63685c7e632e4b194549f2a703e34bd98b
    DF 14B 720P: 3d2ebd783060183743ef1d0ff884049aca4fe4f0

Pipeline files inspected:
  X:/H/diffusers/src/diffusers/pipelines/skyreels_v2/pipeline_skyreels_v2.py
  X:/H/diffusers/src/diffusers/pipelines/skyreels_v2/pipeline_skyreels_v2_i2v.py
  X:/H/diffusers/src/diffusers/pipelines/skyreels_v2/pipeline_skyreels_v2_diffusion_forcing.py
  X:/H/diffusers/src/diffusers/pipelines/skyreels_v2/pipeline_skyreels_v2_diffusion_forcing_i2v.py
  X:/H/diffusers/src/diffusers/pipelines/skyreels_v2/pipeline_skyreels_v2_diffusion_forcing_v2v.py
  X:/H/diffusers/src/diffusers/pipelines/skyreels_v2/pipeline_output.py
  X:/H/diffusers/src/diffusers/pipelines/skyreels_v2/__init__.py

Model files inspected:
  X:/H/diffusers/src/diffusers/models/transformers/transformer_skyreels_v2.py
  X:/H/diffusers/src/diffusers/models/autoencoders/autoencoder_kl_wan.py
  X:/H/diffusers/src/diffusers/models/attention_dispatch.py
  X:/H/diffusers/src/diffusers/models/embeddings.py
  X:/H/diffusers/src/diffusers/models/normalization.py

Scheduler/processors/helpers inspected:
  X:/H/diffusers/src/diffusers/schedulers/scheduling_unipc_multistep.py
  X:/H/diffusers/src/diffusers/video_processor.py
  X:/H/diffusers/src/diffusers/image_processor.py
  X:/H/diffusers/src/diffusers/loaders/lora_pipeline.py

External component configs inspected:
  UMT5EncoderModel and T5TokenizerFast configs for text conditioning.
  CLIPVisionModelWithProjection and CLIPImageProcessor configs for I2V only.

Any missing files or assumptions:
  No gated blocker was hit. Official raw/API reads were public. 404s for
  image_encoder/image_processor under T2V and DF T2V repos are expected absent
  components, not access failures. The report targets the non-deprecated
  SkyReels V2 family, with plain T2V as the recommended first slice and I2V,
  diffusion forcing, V2V, LoRA, and VAE memory policies as separate candidates.
  Multi-GPU/context parallel, callbacks/interrupt mutation, XLA/NPU/MPS/Flax/
  ONNX, safety/NSFW, and training/loss/dropout/gradient-checkpointing paths are
  out of scope.
```

## 2. Pipeline and component graph

SkyReels V2 is a Wan-adjacent latent video diffusion family. The common stack is
UMT5 text conditioning, `SkyReelsV2Transformer3DModel`, `UniPCMultistepScheduler`
configured for flow prediction, and `AutoencoderKLWan` decode.

Plain T2V:

```text
prompt / negative prompt
  -> T5TokenizerFast + UMT5EncoderModel -> prompt embeds [B,512,4096]
  -> latent noise [B,16,T_lat,H/8,W/8] NCTHW
  -> denoising loop:
       SkyReelsV2Transformer3DModel(latents, scalar timestep, text embeds)
       true CFG as separate cond/uncond transformer calls
       UniPCMultistepScheduler.step
  -> latents * std + mean
  -> AutoencoderKLWan decode
  -> VideoProcessor postprocess
```

I2V adds CLIP image tokens and VAE latent conditioning:

```text
input image and optional last_image
  -> CLIPImageProcessor + CLIPVisionModelWithProjection hidden_states[-2]
  -> VideoProcessor preprocess image(s) to NCHW
  -> build sparse condition video
  -> AutoencoderKLWan encode(mode)
  -> normalize condition latents and build 4-channel temporal mask
  -> concat noisy latents + mask + condition latents -> 36-channel transformer input
  -> same UniPC denoising loop with image added-K/V attention branch
```

Diffusion-forcing T2V/I2V/V2V changes the loop shape rather than the base
denoiser weights:

```text
latents [B,16,T_lat,H/8,W/8]
  -> generate per-frame timestep matrix and update masks
  -> for each matrix row:
       slice a valid latent-frame interval
       transformer(..., timestep=[B,F_interval], enable_diffusion_forcing=True)
       optional true CFG second call
       one UniPC scheduler copy per latent frame updates only masked frames
  -> accumulate long-video overlap windows when requested
  -> Wan VAE decode
```

Required first-slice components:

| Component | Class/file | First-slice status |
| --- | --- | --- |
| Pipeline | `SkyReelsV2Pipeline`, `pipeline_skyreels_v2.py` | Recommended first runtime contract. |
| Denoiser | `SkyReelsV2Transformer3DModel`, `transformer_skyreels_v2.py` | Required. |
| VAE | `AutoencoderKLWan`, `autoencoder_kl_wan.py` | Decode required; encode for I2V/V2V later. |
| Scheduler | `UniPCMultistepScheduler`, `scheduling_unipc_multistep.py` | Required with SkyReels flow config. |
| Text encoder | `UMT5EncoderModel` | Accept external cached prompt embeddings first. |
| Image encoder | `CLIPVisionModelWithProjection` | I2V-only candidate. |

Independently cacheable stages are prompt embeddings, negative prompt
embeddings, CLIP image hidden states for I2V, scheduler timesteps/sigmas, random
initial latents, VAE image/video condition latents, and final VAE decode.

Separate candidate reports:

| Candidate | Primary classes/files | Runtime delta |
| --- | --- | --- |
| `skyreels_v2_i2v` | `SkyReelsV2ImageToVideoPipeline`, `pipeline_skyreels_v2_i2v.py` | Adds CLIP image tokens, VAE encode of first/last image conditions, 4-channel mask, and 36-channel transformer input. |
| `skyreels_v2_diffusion_forcing` | `SkyReelsV2DiffusionForcingPipeline`, `pipeline_skyreels_v2_diffusion_forcing.py` | Adds per-frame timestep matrices, per-frame scheduler state, optional causal-block attention, long-video overlap, and optional FPS conditioning. |
| `skyreels_v2_df_i2v` | `SkyReelsV2DiffusionForcingImageToVideoPipeline` | Uses VAE latent prefix conditioning instead of CLIP image added-K/V; supports first/last image handling and long-video overlap. |
| `skyreels_v2_df_v2v` | `SkyReelsV2DiffusionForcingVideoToVideoPipeline` | Encodes an input video overlap as prefix latents, then appends generated video to source video on output. |
| `skyreels_v2_lora_adapters` | `SkyReelsV2LoraLoaderMixin`, `lora_pipeline.py` | Transformer LoRA load/fuse/unfuse, plus T2V-LoRA zero expansion for I2V added image K/V layers. |
| `skyreels_v2_wan_vae_policy` | `AutoencoderKLWan` toggles | VAE slicing, tiling, causal cache, and encode/decode memory-policy report. |

No SkyReels V2 ControlNet, T2I-Adapter, GLIGEN, depth2img, inpaint, or
upscaling pipeline class was found in the non-deprecated `skyreels_v2` folder.
SkyReels V1 Hunyuan pipelines and SkyReels V3 repos are separate families, not
part of this selected target.

## 3. Important config dimensions

Representative checkpoint sweep:

| Repo | Pipeline | Transformer input/output | Layers / heads / head dim | FFN | Scheduler | Special |
| --- | --- | --- | --- | ---: | --- | --- |
| `SkyReels-V2-T2V-14B-540P-Diffusers` | `SkyReelsV2Pipeline` | 16 -> 16 | 40 / 40 / 128 | 13824 | UniPC flow, shift 1 | Plain T2V 14B, 540P target. |
| `SkyReels-V2-T2V-14B-720P-Diffusers` | `SkyReelsV2Pipeline` | 16 -> 16 | 40 / 40 / 128 | 13824 | UniPC flow, shift 1 | Plain T2V 14B, 720P target. |
| `SkyReels-V2-I2V-1.3B-540P-Diffusers` | `SkyReelsV2ImageToVideoPipeline` | 36 -> 16 | 30 / 12 / 128 | 8960 | UniPC flow, shift 1 | CLIP image tokens, `added_kv_proj_dim=1536`. |
| `SkyReels-V2-I2V-14B-720P-Diffusers` | `SkyReelsV2ImageToVideoPipeline` | 36 -> 16 | 40 / 40 / 128 | 13824 | UniPC flow, shift 1 | CLIP image tokens, `added_kv_proj_dim=5120`. |
| `SkyReels-V2-DF-1.3B-540P-Diffusers` | `SkyReelsV2DiffusionForcingPipeline` | 16 -> 16 | 30 / 12 / 128 | 8960 | UniPC flow, shift 1 | `inject_sample_info=true`; FPS embedding path active. |
| `SkyReels-V2-DF-14B-720P-Diffusers` | `SkyReelsV2DiffusionForcingPipeline` | 16 -> 16 | 40 / 40 / 128 | 13824 | UniPC flow, shift 1 | Diffusion forcing, no FPS sample-info injection. |

Common transformer fields:

| Field | Value | Runtime effect |
| --- | --- | --- |
| `patch_size` | `[1,2,2]` | Conv3d patchify over H/W only; temporal patch is 1. |
| `text_dim` | 4096 | UMT5 hidden width. |
| `freq_dim` | 256 | Sinusoidal timestep embedding width. |
| `qk_norm` | `rms_norm_across_heads` | Query/key RMSNorm before attention. |
| `cross_attn_norm` | true | FP32LayerNorm before cross-attention. |
| `rope_max_seq_len` | 1024 | Per-axis 3D RoPE table limit. |
| `num_frame_per_block` | 1 in sampled configs | No causal self-attention mask unless runtime overrides `causal_block_size`. |
| `image_dim` | null for T2V/DF T2V, 1280 for I2V | Enables CLIP image projection path for I2V. |
| `added_kv_proj_dim` | null for T2V/DF T2V, inner dim for I2V | Enables added image K/V attention branch. |

Wan VAE config used by sampled SkyReels V2 repos:

| Field | Value |
| --- | --- |
| `z_dim` | 16 |
| `base_dim` | 96 |
| `dim_mult` | `[1,2,4,4]` |
| `num_res_blocks` | 2 |
| `temperal_downsample` | `[False, True, True]` |
| spatial / temporal compression | 8 / 4 from pipeline factors |
| latent stats | 16-element `latents_mean` and `latents_std` |
| `attn_scales` | `[]` in sampled configs |

Text encoder config facts:

| Component | Fields |
| --- | --- |
| `UMT5EncoderModel` | `d_model=4096`, `num_layers=24`, `num_heads=64`, `d_ff=10240`, gated GELU, vocab 256384, config metadata `torch_dtype=float32`. |
| `T5TokenizerFast` | T5/UMT5 tokenizer family. Pipeline `__call__` defaults `max_sequence_length=512`; helper default is 226 but the call path passes 512. |

I2V image encoder config facts:

| Component | Fields |
| --- | --- |
| `CLIPVisionModelWithProjection` | image 224, patch 14, 32 layers, hidden 1280, 16 heads, intermediate 5120, projection 1024, config metadata `torch_dtype=float32`. |
| `CLIPImageProcessor` | resize shortest edge 224, center crop 224, RGB, rescale by 1/255, CLIP mean/std normalization. |

Scheduler config facts:

| Field | SkyReels V2 value |
| --- | --- |
| class | `UniPCMultistepScheduler` |
| `prediction_type` | `flow_prediction` |
| `use_flow_sigmas` | true |
| `flow_shift` | 1.0 |
| `solver_order` | 2 |
| `solver_type` | `bh2` |
| `predict_x0` | true |
| `final_sigmas_type` | `zero` |
| `timestep_spacing` | `linspace` |
| `thresholding` | false |

Recommended first Dinoml scheduler slice:

- Implement the official `UniPCMultistepScheduler` flow-prediction config above,
  with host-visible multistep state.
- Do not substitute FlowMatch Euler just because SkyReels is a video
  transformer family; the official sampled SkyReels V2 configs are UniPC flow.
- Diffusion forcing adds one scheduler copy per latent frame. That is a
  separate scheduler-orchestration candidate, not the first base T2V slice.

## 3a. Family variation traps

- Plain T2V and diffusion forcing share `SkyReelsV2Transformer3DModel`, but
  diffusion forcing passes per-frame timestep tensors and changes adaptive norm
  shapes throughout the block stack.
- I2V is not just an image prompt wrapper: transformer input channels change
  from 16 to 36, and cross-attention gains a CLIP image added-K/V branch.
- DF I2V is different from plain I2V: it does not register a CLIP image encoder
  in source; it uses VAE-encoded image latents as prefix/condition frames.
- `num_frames` is rounded to `k * vae_scale_factor_temporal + 1`, so sampled
  common defaults 97 and 121 become latent lengths 25 and 31.
- Height and width are checked only as multiples of 16, combining VAE spatial
  scale 8 and transformer spatial patch size 2.
- Source latent/video layout is NCTHW. Treat NDHWC as a guarded optimization
  only, with explicit rewrites for channel concat, mask channels, latent
  mean/std, Conv3d weights, VAE cache axes, and scheduler broadcasts.
- The transformer token sequence length gets large quickly: 544x960x97 gives
  `25 * 34 * 60 = 51,000` latent tokens after patching; 720x1280x121 gives
  `31 * 45 * 80 = 111,600` latent tokens.
- `num_frame_per_block=1` in sampled configs, but runtime `causal_block_size`
  can call `_set_ar_attention` and create a block-causal self-attention mask.
  This must be a guarded variant.
- The attention processor hard-codes text context length 512 when splitting
  image context from text context in I2V.
- DF 1.3B has `inject_sample_info=true`; the pipeline maps FPS 16 to class 0
  and all other FPS values to class 1, then adds an FPS projection into the
  timestep modulation path.
- The V2V source effectively needs `overlap_history` for its long-video loop:
  the default is `None`, but the code computes latent overlap from it
  unconditionally after validation.
- VAE tiling/slicing and causal feature caches are memory/runtime policies. Do
  not fold them into the base denoiser contract.

## 4. Runtime tensor contract

Plain T2V at source defaults 544x960, 97 frames:

| Boundary | Tensor | Shape/layout | Notes |
| --- | --- | --- | --- |
| prompt embeds | `prompt_embeds` | `[B,512,4096]` | UMT5 last hidden state, trimmed by mask then zero-padded back to max length. |
| negative embeds | `negative_prompt_embeds` | `[B,512,4096]` | Used only for true CFG. |
| latent noise | `latents` | `[B,16,25,68,120]` NCTHW | `T_lat=(97-1)/4+1`, H/8, W/8. |
| transformer tokens | after Conv3d patch | `[B,25*34*60,inner]` | `inner=heads*head_dim`; 5120 for 14B, 1536 for 1.3B. |
| timestep | `timestep` | `[B]` | Scalar per batch in plain T2V/I2V. |
| denoiser output | `noise_pred` | `[B,16,25,68,120]` | Unpatchified inside transformer. |
| scheduler state | `sigmas`, history | CPU sigma table plus UniPC model-output history | `scale_model_input` is identity. |
| VAE decode input | denormalized latents | `[B,16,25,68,120]` | Pipeline restores `latents / (1/std) + mean`. |
| decoded video | output tensor | `[B,3,97,544,960]` NCTHW | Postprocessed to `np`, PIL/list, PT, or latent based on output type. |

I2V conditioning:

| Boundary | Tensor | Shape/layout | Notes |
| --- | --- | --- | --- |
| input image | preprocessed | `[B,3,H,W]` | `VideoProcessor.preprocess` to float32 NCHW for VAE condition path. |
| CLIP image embeds | hidden state | `[B,257,1280]` for one image | `hidden_states[-2]`; optional first+last image list can double image entries before repeat. |
| condition video | first/last frame sparse video | `[B,3,T,H,W]` | First image at frame 0; optional last image at final frame; middle frames zeros. |
| latent condition | normalized VAE mode | `[B,16,T_lat,H/8,W/8]` | `(vae.encode(...).mode - mean) * (1/std)`. |
| temporal mask | mask channels | `[B,4,T_lat,H/8,W/8]` | Built from source-frame mask and temporal factor 4. |
| transformer input | concatenated | `[B,36,T_lat,H/8,W/8]` | 16 noisy + 4 mask + 16 condition. |

Diffusion-forcing loop tensors:

| Boundary | Tensor | Shape/layout | Notes |
| --- | --- | --- | --- |
| `step_matrix` | timesteps per latent frame | `[num_iterations,T_lat]` | Generated from UniPC timesteps plus boundary values 999 and 0. |
| `step_update_mask` | bool update mask | `[num_iterations,T_lat]` | Only masked frames are written back. |
| valid interval | frame slice | list of `(start,end)` | Limits each transformer call to active latent-frame span. |
| timestep input | per-frame timestep | `[B,F_interval]` | Triggers diffusion-forcing adaptive norm path. |
| per-frame schedulers | copied UniPC objects | `T_lat` scheduler states | Each latent frame maintains independent solver history. |

CPU/data-pipeline work includes prompt cleaning/tokenization, UMT5 execution
when embeddings are not supplied, CLIP image preprocessing/encoding for I2V,
image/video preprocessing for condition paths, and output conversion. GPU/runtime
work includes latent noise, transformer denoising, CFG arithmetic, scheduler
updates, VAE encode/decode, condition packing, and latent normalization.

## 5. Operator coverage checklist

### Tensor/layout ops

- NCTHW video latents and video samples; view, reshape, flatten, transpose,
  permute, clone, cat, chunk/split, repeat, repeat_interleave.
- Conv3d patchify/unpatchify:
  `[B,C,T,H,W] -> [B,T*(H/2)*(W/2),inner] -> [B,C,T,H,W]`.
- I2V channel concatenation: noisy latents + 4 mask channels + condition latents.
- DF interval slicing and masked in-place frame updates.
- Per-channel latent mean/std broadcasts over `[B,C,T,H,W]`.
- CLIP image token repeat and image/text context concatenation.

### Convolution/downsample/upsample ops

- Transformer patch embedding:
  `Conv3d(in_channels -> inner_dim, kernel=[1,2,2], stride=[1,2,2])`.
- Wan VAE causal `Conv3d`, 1x1 quant/post-quant convs, residual blocks,
  temporal/spatial downsample and upsample helpers.
- VAE causal feature-cache state for encode/decode.

### GEMM/linear ops

- UMT5 and CLIPVision if admitted into Dinoml later.
- Timestep MLP: sinusoidal embed -> `TimestepEmbedding` -> SiLU -> linear to
  six modulation vectors.
- PixArt-style text projection: `Linear(4096 -> inner_dim)` inside
  `PixArtAlphaTextProjection`.
- Optional CLIP image projection: FP32LayerNorm -> FeedForward(1280 -> inner)
  -> FP32LayerNorm.
- Attention Q/K/V, added image K/V, cross-attention K/V, output projections.
- FeedForward GELU-approximate MLP with `ffn_dim=8960` or `13824`.
- Output projection: `Linear(inner_dim -> 16 * 1 * 2 * 2)`.

### Attention primitives

- Latent-token self-attention with QK RMSNorm and 3D RoPE.
- Text cross-attention over UMT5 tokens.
- I2V added image K/V attention branch, added to text cross-attention output.
- Optional block-causal self-attention mask when `causal_block_size > 1`.

### Normalization and adaptive conditioning

- `FP32LayerNorm` with affine and non-affine variants.
- `torch.nn.RMSNorm` for Q/K and added image K.
- Adaptive scale/shift/gate from timestep projection and per-block
  `scale_shift_table`.
- VAE channel GroupNorm/RMSNorm-like paths inherited from Wan VAE.

### Position/timestep/guidance embeddings

- 3D RoPE split across temporal, height, and width axes.
- `SkyReelsV2Timesteps` using sin/cos embedding with `flip_sin_to_cos=True`.
- Scalar timestep path for T2V/I2V; per-frame timestep path for diffusion forcing.
- Optional FPS class embedding for DF 1.3B `inject_sample_info=true`.
- True CFG: separate cond/uncond calls and pointwise combination.

### Scheduler and guidance arithmetic

- UniPC flow-prediction conversion: `x0_pred = sample - sigma * model_output`.
- UniP/UniC multistep predictor/corrector history and lower-order warmup.
- DF per-frame scheduler copies and update masks.
- Optional `addnoise_condition` blend for prefix clean latents:
  `(1-noise_factor) * latent + noise_factor * randn_like`.

### VAE/postprocessing ops

- AutoencoderKLWan encode/decode, posterior mode for condition encode.
- Latent stats normalization and denormalization.
- Causal Conv3d cache reset and per-frame decode.
- `VideoProcessor.postprocess_video`.

### Video-specific ops

- Temporal compression/decompression by 4.
- Long-video overlap accumulation over latent frames.
- Block-level diffusion-forcing schedule generation and optional causal-block
  mask.

## 6. Denoiser/model breakdown

`SkyReelsV2Transformer3DModel.forward`:

```text
hidden_states [B,C,T,H,W]
-> 3D RoPE from source latent shape
-> Conv3d patch_embedding [1,2,2]
-> flatten/transpose to tokens [B,T*(H/2)*(W/2),inner]
-> optional block-causal self-attention mask when num_frame_per_block > 1
-> timestep/text/image embedding module
-> optional concat(image tokens, text tokens)
-> N x SkyReelsV2TransformerBlock
-> adaptive output LayerNorm
-> Linear to patch volume
-> reshape/permute/flatten back to [B,out_channels,T,H,W]
```

`SkyReelsV2TransformerBlock`:

```text
scale_shift_table + timestep projection -> six modulation tensors
FP32LayerNorm -> scale/shift -> self-attention(QKV, QK RMSNorm, RoPE, optional mask)
gated residual add
FP32LayerNorm/Identity -> text cross-attention plus optional image added-K/V branch
residual add
FP32LayerNorm -> scale/shift -> GELU-approximate FeedForward
gated residual add
```

Width examples:

- 1.3B configs: `inner_dim=12*128=1536`, 30 blocks, FFN 8960.
- 14B configs: `inner_dim=40*128=5120`, 40 blocks, FFN 13824.

Diffusion forcing changes only the timestep/adaptive norm shape:

- Scalar path: `timestep=[B]`, modulation tensors broadcast over all tokens.
- DF path: `timestep=[B,T_interval]`, timestep embeddings are repeated over
  patched H/W tokens and become token-position-dependent modulation tensors.

## 7. Attention requirements

Primary implementation is `SkyReelsV2AttnProcessor` in
`transformer_skyreels_v2.py`, calling `dispatch_attention_fn` from
`attention_dispatch.py`.

Required variants:

| Variant | Shape/behavior |
| --- | --- |
| Self-attention | Query/key/value from latent tokens, `[B,seq,heads,head_dim]`, head dim 128, QK RMSNorm, 3D RoPE. |
| Text cross-attention | Query from latent tokens; key/value from projected UMT5 tokens. No text mask is passed by pipeline after zero padding. |
| I2V image added-K/V | Processor splits `encoder_hidden_states` into image prefix and hard-coded 512 text tokens, runs separate attention to image K/V, and adds image attention output. |
| Causal-block self-attention | If `num_frame_per_block > 1`, model builds a block-causal bool mask over latent tokens. |

Fallback parity is the eager/native `dispatch_attention_fn` path. Source supports
projection fusion through `fuse_projections`, but fused projections are a runtime
mutation/optimization, not required for first parity.

Flash-style feasibility:

- Plain self-attention is a plausible Dinoml flash provider candidate under
  head_dim 128, dtype, sequence-length, and workspace guards.
- The very long video sequences are the main admission risk: 720P 121-frame
  latents produce more than 100k tokens after spatial patching.
- QK RMSNorm and RoPE are pre-attention operations and should remain explicit
  unless fused under exact provider guards.
- Cross-attention to 512 text tokens is a separate provider shape.
- I2V image added-K/V is a separate attention branch; do not silently merge it
  with text cross-attention because the output is an explicit sum of two
  attentions.
- Causal-block masks from diffusion forcing need a mask-capable flash path or a
  fallback.

## 8. Scheduler and denoising-loop contract

All sampled official SkyReels V2 configs use `UniPCMultistepScheduler`:

```text
prediction_type = flow_prediction
use_flow_sigmas = true
flow_shift = 1.0
solver_order = 2
solver_type = bh2
predict_x0 = true
timestep_spacing = linspace
final_sigmas_type = zero
thresholding = false
```

Plain T2V/I2V loop:

```text
scheduler.set_timesteps(num_inference_steps, device)
for t in timesteps:
  latent_model_input = latents or cat([latents, condition], dim=1)
  timestep = t.expand(B)
  noise_pred = transformer(latent_model_input, timestep, prompt_embeds, optional image_embeds)
  if CFG:
    noise_uncond = transformer(latent_model_input, timestep, negative_prompt_embeds, optional image_embeds)
    noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)
  latents = scheduler.step(noise_pred, t, latents)
```

Diffusion-forcing loop:

```text
step_matrix, step_update_mask, valid_interval = generate_timestep_matrix(...)
sample_schedulers = [deepcopy(scheduler) for each latent frame]
for row t in step_matrix:
  latent_interval = latents[:, :, start:end]
  timestep = t.expand(B, -1)[:, start:end]
  noise_pred = transformer(latent_interval, timestep, enable_diffusion_forcing=True)
  if CFG: run negative call and combine
  for each frame idx in interval:
    if step_update_mask[row, idx]:
      latents[:, :, idx] = sample_schedulers[idx].step(noise_pred[:, :, local_idx], t[idx], latents[:, :, idx])
```

Keep `set_timesteps`, multistep UniPC history, DF timestep-matrix generation,
per-frame scheduler ownership, and long-video windowing as host-visible runtime
state initially. Compile/fuse CFG and scheduler pointwise update only after
one-step parity is established.

## 9. Position, timestep, and custom math

- 3D RoPE divides the head dimension into temporal, height, and width pieces:
  `h_dim=w_dim=2*(head_dim//6)`, `t_dim=head_dim-h_dim-w_dim`. For head dim 128,
  this yields temporal 44 and height/width 42 each.
- RoPE tables are generated per axis up to `rope_max_seq_len=1024` and expanded
  over the post-patch `[T,H/2,W/2]` grid.
- `SkyReelsV2Timesteps` calls `get_1d_sincos_pos_embed_from_grid` and restores
  the original timestep tensor rank, which is required for DF `[B,F]` timesteps.
- `SkyReelsV2TimeTextImageEmbedding` returns the base time embedding, six-way
  timestep projection, projected text tokens, and optional projected image
  tokens.
- DF 1.3B `inject_sample_info=true` adds a learned FPS class embedding after
  mapping `fps==16` to 0 and other FPS values to 1.
- Precompute candidates: prompt embeddings, negative embeddings, CLIP image
  hidden states, RoPE tables for fixed latent grids, scheduler timesteps/sigmas,
  and DF timestep matrices for fixed step/base-frame settings.

## 10. Preprocessing and input packing

Text:

- Prompt strings are cleaned with `prompt_clean`, tokenized to a fixed max
  length, encoded by UMT5, trimmed to attention-mask lengths, and zero-padded
  back to `max_sequence_length`.
- Pipeline `__call__` defaults `max_sequence_length=512`; negative prompt
  defaults to empty string under CFG.
- Prompt embeddings are duplicated by repeating on the sequence dimension then
  reshaping to `[B*num_videos_per_prompt,L,D]`.

Plain T2V:

- Random source latents are NCTHW `[B,16,T_lat,H/8,W/8]`.
- The denoiser internally patchifies to tokens; there is no pipeline-level
  latent packing.
- Decode restores Wan VAE mean/std before calling `vae.decode`.

I2V:

- CLIP image path is CPU/data-pipeline plus optional GPU model execution:
  resize/crop/normalize to 224, CLIP vision hidden state `[-2]`.
- VAE condition path preprocesses source image(s) to NCHW, builds a sparse
  video, encodes with Wan VAE posterior mode, normalizes with Wan stats, and
  concatenates mask+condition channels.
- The plain I2V transformer receives image embeddings as `encoder_hidden_states_image`.

DF I2V/V2V:

- DF I2V source does not use CLIP image embeddings; image or last-image VAE
  latents are inserted as prefix/end condition frames.
- DF V2V preprocesses the input video, encodes the overlap history, uses that
  latent prefix during generation, then concatenates original and generated
  videos on output.

Layout guard notes:

- Preserve NCTHW at all pipeline/VAE/scheduler boundaries first.
- Candidate NDHWC islands are Conv3d-heavy VAE regions and possibly the
  transformer patch Conv3d, but only after rewriting Conv3d weights, channel
  concat axes, GroupNorm/RMSNorm axes, temporal cache axes, and latent stat
  broadcasts.
- Mark DF interval slicing/update and I2V condition packing as
  no-layout-translation regions until dedicated parity tests prove equivalence.

## 11. Graph rewrite / lowering opportunities

### Rewrite: Conv3d patchify/unpatchify

Source pattern:

```text
Conv3d(kernel=stride=[1,2,2]) -> flatten(2).transpose(1,2)
Linear(inner -> out_channels*4) -> reshape(B,T,H/2,W/2,1,2,2,C)
-> permute -> flatten back to NCTHW
```

Replacement: explicit video patch projection and inverse patch unpack.

Preconditions: source NCTHW layout, `T % 1 == 0`, `H/W` divisible by 2 after
VAE scaling, patch size exactly `[1,2,2]`, no active layout translation inside
the token sequence. Weight transform is required for any NDHWC Conv3d provider.
Failure cases: unsupported nonstandard patch sizes, DF interval slices whose
token order must exactly match source, and I2V 36-channel input variants without
matching weights.

### Rewrite: SkyReels attention prelude

Source pattern:

```text
Linear Q/K/V -> RMSNorm(Q,K) -> unflatten heads -> RoPE(Q,K) -> attention
```

Replacement: canonical attention provider with explicit QK norm and RoPE pre-ops.

Preconditions: noncausal or mask-supported mode, head dim 128, supported dtype,
sequence length under provider limits, no image added-K/V unless lowered as a
second attention/add branch. Failure cases: causal-block mask unsupported,
I2V image branch incorrectly merged with text branch, RoPE table mismatch.

### Rewrite: I2V condition pack

Source pattern:

```text
latent_condition = (vae.encode(condition).mode - mean) * (1/std)
mask_lat_size -> 4 temporal mask channels
latent_model_input = cat([latents, mask_lat_size, latent_condition], dim=1)
```

Replacement: plan-visible condition-pack kernel over NCTHW tensors.

Preconditions: 16-channel Wan VAE, temporal scale 4, source channel axis 1,
first/last frame mask semantics preserved. Failure cases: DF I2V prefix
conditioning, last-image shape mismatch, or future non-16-channel VAE.

### Rewrite: UniPC flow scheduler step

Source pattern:

```text
x0_pred = sample - sigma * model_output
UniP/UniC multistep predictor/corrector update
```

Replacement: explicit scheduler state tensors plus pointwise kernels around a
host-controlled multistep loop.

Preconditions: official SkyReels scheduler fields, no dynamic shift, no custom
sigmas, flow prediction, thresholding disabled. Failure cases: DF per-frame
scheduler copies, alternate UniPC solver types, dynamic shift, stochastic
variants.

### Rewrite: DF timestep matrix

Source pattern:

```text
generate_timestep_matrix -> valid intervals -> per-frame masked scheduler steps
```

Replacement: artifact-visible loop schedule metadata and frame-update masks.

Preconditions: fixed `num_inference_steps`, `base_num_frames`,
`causal_block_size`, `ar_step`, and overlap settings. Failure cases: runtime
changes to overlap/base frames, prefix truncation by causal block alignment, or
source V2V default `overlap_history=None` ambiguity.

## 12. Kernel fusion candidates

Highest priority:

- Large GEMMs for 14B Q/K/V, cross-attention, FFN, text projection, and output
  projection.
- QK RMSNorm + RoPE + attention provider prelude for latent self-attention.
- Adaptive norm scale/shift/gate plus residual epilogues around attention and
  FFN.
- Conv3d patchify/unpatchify and faithful token-order transforms.
- CFG arithmetic and UniPC flow scheduler pointwise pieces.

Medium priority:

- I2V condition-pack kernel: VAE latent normalization, temporal mask build, and
  channel concat.
- Wan VAE causal Conv3d + norm + activation residual decode blocks.
- DF timestep/update-mask orchestration and per-frame scheduler-state kernels.
- Prefix `addnoise_condition` blend for long-video/DF condition latents.

Lower priority:

- CLIP image added-K/V branch fusion.
- VAE tiling/slicing and overlap blend kernels.
- LoRA fuse/unfuse/runtime adapter state.
- Causal-block masked attention optimization for asynchronous DF.

## 13. Runtime staging plan

Stage 1: Parse configs for `SkyReels-V2-T2V-14B-540P-Diffusers` or 720P; load
transformer/VAE weights; accept external UMT5 prompt and negative embeddings.

Stage 2: Implement NCTHW latent contract, Conv3d `[1,2,2]` patchify/unpatchify,
3D RoPE, scalar timestep embedding, and one transformer block parity.

Stage 3: Full `SkyReelsV2Transformer3DModel` random-tensor parity on reduced
latent grids, then representative grids as memory allows.

Stage 4: Add true CFG as two explicit denoiser calls and one fixed-timestep
noise prediction parity.

Stage 5: Implement official UniPC flow-prediction scheduler slice with
host-visible multistep state; validate one step and a short loop.

Stage 6: Add `AutoencoderKLWan` 16-channel decode with tiling/slicing disabled
and explicit latent mean/std denormalization.

Stage 7: Add plain I2V as a separate stage: CLIP image hidden states, VAE encode
mode, mask/condition concat, added image K/V attention branch.

Stage 8: Add diffusion forcing T2V: per-frame timestep embeddings,
`generate_timestep_matrix`, update masks, per-frame scheduler copies, and
optional FPS embedding for DF 1.3B.

Stage 9: Add DF I2V and DF V2V prefix/overlap contracts, then long-video
window accumulation.

Stage 10: Add LoRA and VAE memory-policy variants only after base parity is
stable.

## 14. Parity and validation plan

- Config/default reconciliation for all six sampled repos, including absent
  image components in T2V and DF T2V.
- Patchify/unpatchify parity for `[B,16,25,68,120]` and `[B,16,31,90,160]`.
- 3D RoPE table/index parity for multiple `[T,H,W]` latent grids.
- One `SkyReelsV2TransformerBlock` parity at 1.3B and 14B widths.
- Full transformer forward parity on a small synthetic grid.
- Scalar timestep versus DF per-frame timestep embedding parity.
- Optional FPS embedding parity for DF 1.3B.
- CFG arithmetic parity with fixed positive/negative predictions.
- UniPC `set_timesteps` and one-step flow-prediction parity for shift 1.0.
- Wan VAE decode parity with 16-element mean/std.
- Wan VAE encode posterior mode parity for I2V and V2V conditions.
- I2V condition-mask packing parity and 36-channel transformer input parity.
- CLIP image added-K/V attention branch parity with synthetic image tokens.
- DF timestep-matrix/update-mask parity for synchronous `ar_step=0` and
  asynchronous `ar_step=5`.
- V2V overlap-history parity, including source prefix truncation when latent
  overlap is not divisible by `causal_block_size`.
- Suggested tolerances: scheduler fp32 `rtol=1e-5, atol=1e-6`; transformer/VAE
  fp32 `rtol=1e-4, atol=1e-5`; fp16/bf16 start at `rtol=2e-2, atol=2e-2`.

## 15. Performance probes

- One denoiser step by latent grid: small synthetic, 544x960x97, and
  720x1280x121.
- 1.3B versus 14B block time split: patch Conv3d, self-attention, cross-attn,
  FFN, output projection.
- Attention backend comparison at head dim 128 and long token counts.
- CFG two-call overhead versus single denoiser call.
- UniPC scheduler overhead compared with denoiser time.
- VAE decode throughput and memory for `[B,16,25,68,120]` and
  `[B,16,31,90,160]`.
- I2V VAE encode, CLIP image encode, condition-pack, and added-K/V overhead.
- DF timestep-matrix and per-frame scheduler overhead by `T_lat`,
  `causal_block_size`, and `ar_step`.
- Long-video overlap accumulation and prefix noise timing.
- Faithful NCTHW path versus guarded NDHWC Conv3d/VAE islands.

## 16. Scope boundary and separate candidates

Separate candidate reports, not ignored:

- `skyreels_v2_i2v`: CLIP image encoder, VAE image condition encode, temporal
  mask channels, 36-channel transformer input, and image added-K/V attention.
- `skyreels_v2_diffusion_forcing`: per-frame timesteps, timestep matrices,
  per-frame scheduler copies, optional causal-block masks, long-video overlap,
  and FPS embedding.
- `skyreels_v2_df_i2v`: VAE image prefix/end conditioning under diffusion
  forcing; no CLIP image encoder in the source class.
- `skyreels_v2_df_v2v`: input-video VAE encode, overlap-history prefix, and
  original+generated output concatenation.
- `skyreels_v2_lora_adapters`: transformer PEFT adapter state, non-Diffusers
  Wan/Musubi key conversion, T2V-to-I2V zero expansion, fuse/unfuse.
- `skyreels_v2_wan_vae_policy`: VAE slicing, tiling, causal feature-cache
  residency, and encode/decode chunk policy.
- `scheduler_unipc_diffusion_forcing`: per-latent-frame UniPC state management
  and update-mask loop integration.
- `skyreels_v1_hunyuan` and `skyreels_v3`: related SkyReels names but separate
  pipeline/model families from this non-deprecated SkyReels V2 target.

Not present as SkyReels V2 source surfaces:

- Textual inversion loader.
- IP-Adapter loader. The I2V image K/V path is built into the transformer, not
  the generic IP-Adapter loader surface.
- ControlNet, T2I-Adapter, GLIGEN.
- Inpaint, depth2img, and upscaling pipelines.

Genuinely ignored/out of scope for this audit:

- Multi-GPU/context parallel paths.
- Callback mutation and interactive interrupt.
- XLA, NPU, MPS, Flax, and ONNX variants.
- Safety checker and NSFW filtering.
- Training, losses, dropout behavior, and gradient checkpointing.

## 17. Final implementation checklist

- [ ] Parse SkyReels V2 model indexes and component configs.
- [ ] Load `SkyReelsV2Transformer3DModel` and `AutoencoderKLWan` weights.
- [ ] Accept external UMT5 prompt and negative prompt embeddings.
- [ ] Implement NCTHW latent shape admission and frame rounding checks.
- [ ] Implement Conv3d `[1,2,2]` patchify/unpatchify parity.
- [ ] Implement SkyReels 3D RoPE generation and application.
- [ ] Implement scalar timestep/text embedding path.
- [ ] Implement one `SkyReelsV2TransformerBlock` with QK RMSNorm, attention, cross-attention, FFN, and gates.
- [ ] Implement full transformer forward parity for T2V.
- [ ] Implement true CFG two-call arithmetic.
- [ ] Implement UniPC flow-prediction scheduler shift 1.0.
- [ ] Implement Wan VAE 16-channel decode with mean/std denormalization.
- [ ] Add short T2V loop parity with scheduler in host control.
- [ ] Add I2V VAE encode, mask/condition concat, CLIP image tokens, and added-K/V attention.
- [ ] Add diffusion-forcing per-frame timestep/adaptive-norm path.
- [ ] Add DF timestep-matrix/update-mask and per-frame scheduler state.
- [ ] Add DF I2V/V2V overlap-prefix variants.
- [ ] Add LoRA adapter-state admission separately from base inference.
- [ ] Add guarded NDHWC/Conv3d/VAE layout optimization only after faithful NCTHW parity.
