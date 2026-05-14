# Diffusers Cosmos Video/World Operator and Integration Report

## 1. Source basis

```text
Diffusers commit/version:
  Local checkout X:/H/diffusers at b3a515080752a3ba7ca92161e25530c7f280f629.

Model id(s):
  nvidia/Cosmos-1.0-Diffusion-7B-Text2World
  nvidia/Cosmos-1.0-Diffusion-14B-Text2World
  nvidia/Cosmos-1.0-Diffusion-7B-Video2World
  nvidia/Cosmos-1.0-Diffusion-14B-Video2World
  nvidia/Cosmos-Predict2-2B-Text2Image
  nvidia/Cosmos-Predict2-14B-Text2Image
  nvidia/Cosmos-Predict2-2B-Video2World
  nvidia/Cosmos-Predict2-14B-Video2World
  nvidia/Cosmos-Predict2.5-2B, revisions diffusers/base/pre-trained and diffusers/base/post-trained
  nvidia/Cosmos-Predict2.5-14B, revisions diffusers/base/pre-trained and diffusers/base/post-trained
  nvidia/Cosmos-Transfer2.5-2B, source inspected but component configs blocked by gated access

Config sources:
  Local cache checked first:
    H:/configs/nvidia/Cosmos-Predict2.5-14B/model_index.json
    H:/configs/city96/Cosmos-Predict2-14B-Text2Image-gguf/model_index.json
    H:/configs/carlosabadia/cosmos/model_index.json
  The local official Cosmos Predict2.5 file was an empty JSON object, the city96
  GGUF mirror file was also empty, and carlosabadia/cosmos was an unrelated old
  StableDiffusionPipeline model_index. Official Hugging Face component configs
  were therefore fetched and inspected through `huggingface_hub` / authenticated
  `hf` into the HF cache only; they were not copied under H:/configs because this
  task's owned workspace write path is only this report.

Pipeline files inspected:
  X:/H/diffusers/src/diffusers/pipelines/cosmos/pipeline_cosmos_text2world.py
  X:/H/diffusers/src/diffusers/pipelines/cosmos/pipeline_cosmos_video2world.py
  X:/H/diffusers/src/diffusers/pipelines/cosmos/pipeline_cosmos2_text2image.py
  X:/H/diffusers/src/diffusers/pipelines/cosmos/pipeline_cosmos2_video2world.py
  X:/H/diffusers/src/diffusers/pipelines/cosmos/pipeline_cosmos2_5_predict.py
  X:/H/diffusers/src/diffusers/pipelines/cosmos/pipeline_cosmos2_5_transfer.py
  X:/H/diffusers/src/diffusers/pipelines/cosmos/pipeline_output.py

Model files inspected:
  X:/H/diffusers/src/diffusers/models/transformers/transformer_cosmos.py
  X:/H/diffusers/src/diffusers/models/autoencoders/autoencoder_kl_cosmos.py
  X:/H/diffusers/src/diffusers/models/autoencoders/autoencoder_kl_wan.py
  X:/H/diffusers/src/diffusers/models/controlnets/controlnet_cosmos.py
  X:/H/diffusers/src/diffusers/models/attention_processor.py
  X:/H/diffusers/src/diffusers/models/attention_dispatch.py
  X:/H/diffusers/src/diffusers/models/embeddings.py
  X:/H/diffusers/src/diffusers/models/normalization.py

Scheduler/processors/helpers inspected:
  X:/H/diffusers/src/diffusers/schedulers/scheduling_edm_euler.py
  X:/H/diffusers/src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py
  X:/H/diffusers/src/diffusers/schedulers/scheduling_unipc_multistep.py
  X:/H/diffusers/src/diffusers/video_processor.py
  X:/H/diffusers/src/diffusers/image_processor.py
  X:/H/diffusers/src/diffusers/loaders/single_file_utils.py
  X:/H/diffusers/src/diffusers/loaders/single_file_model.py

External component configs inspected:
  T5EncoderModel/T5TokenizerFast configs for Cosmos 1.0 and Predict2.
  Qwen2_5_VLForConditionalGeneration/AutoTokenizer configs for Predict2.5.

Any missing files or assumptions:
  Transfer2.5 source was inspected, including CosmosControlNetModel, but
  authenticated `hf download nvidia/Cosmos-Transfer2.5-2B model_index.json
  --revision diffusers/general` returned a gated 403. Transfer2.5 config facts
  below are therefore source-derived unless explicitly labeled unavailable.
  Multi-GPU/context parallel, callbacks/interrupt mutation, XLA/NPU/MPS/Flax/ONNX,
  safety/guardrail execution, and training/loss/dropout/gradient checkpointing
  are out of scope.
```

## 2. Pipeline and component graph

Cosmos in Diffusers is three related but materially different runtime families:

- Cosmos 1.0 Text2World / Video2World: T5-11B text embeddings, `CosmosTransformer3DModel`, `AutoencoderKLCosmos`, and `EDMEulerScheduler`.
- Cosmos Predict2 Text2Image / Video2World: T5-11B text embeddings, the same Cosmos transformer class, Wan VAE (`AutoencoderKLWan`), and `FlowMatchEulerDiscreteScheduler`.
- Cosmos Predict2.5 base: Qwen2.5-VL hidden-state features, the same Cosmos transformer class, Wan VAE, and `UniPCMultistepScheduler`. The base pipeline supports text-to-world, image-to-world, and video-to-world through a single conditioning path.

Recommended first Dinoml target: `Cosmos2_5_PredictBasePipeline` with externally supplied Qwen prompt embeddings, no Transfer controlnet, and Wan VAE decode. This is the active video/world shape, uses the modern scheduler, and avoids the heavier Cosmos 1.0 codec.

```text
prompt / negative prompt
  -> Qwen2.5-VL chat-template tokenization
  -> Qwen hidden states from layers 1..N, per-layer normalized and concatenated
  -> optional image/video preprocessing and Wan VAE encode
  -> latent noise or conditioned latents [B,16,T,H/8,W/8]
  -> denoising loop:
       condition mask + per-frame/token timestep tensor
       CosmosTransformer3DModel(latents, timestep, prompt embeds, padding mask)
       optional second negative denoiser call for CFG
       UniPCMultistepScheduler.step
  -> Wan latent denormalization with 16-channel mean/std
  -> AutoencoderKLWan decode
  -> VideoProcessor postprocess
```

Required first-slice components:

| Component | Class/file | First-slice role |
| --- | --- | --- |
| Pipeline | `Cosmos2_5_PredictBasePipeline`, `pipeline_cosmos2_5_predict.py` | Runtime contract for T2W/I2W/V2W. |
| Denoiser | `CosmosTransformer3DModel`, `transformer_cosmos.py` | Required; NCTHW latent maps patchified to tokens internally. |
| VAE | `AutoencoderKLWan`, `autoencoder_kl_wan.py` | Decode required; encode required for image/video conditioning. |
| Scheduler | `UniPCMultistepScheduler` | Required first scheduler for Predict2.5 configs. |
| Text encoder | `Qwen2_5_VLForConditionalGeneration` | Accept external prompt/negative embeddings first. |

Separate candidate reports:

| Candidate | Primary classes/files | Runtime delta |
| --- | --- | --- |
| `cosmos_1_text2world` | `CosmosTextToWorldPipeline`, `AutoencoderKLCosmos`, `EDMEulerScheduler` | Older text-to-video/world path with Cosmos-specific codec and EDM denoising. |
| `cosmos_1_video2world` | `CosmosVideoToWorldPipeline` | Adds video/image conditioning, VAE encode, condition mask, augment sigma, and x0/eps EDM loop. |
| `cosmos_predict2_text2image` | `Cosmos2TextToImagePipeline` | Single-frame path, T5 encoder, Wan VAE, FlowMatch Euler preconditioning. |
| `cosmos_predict2_video2world` | `Cosmos2VideoToWorldPipeline` | Adds input image/video conditioning and FlowMatch Euler skip/out preconditioning. |
| `cosmos_transfer2_5` | `Cosmos2_5_TransferPipeline`, `CosmosControlNetModel` | Control video latents, ControlNet residual injection every N blocks, autoregressive chunking. Configs gated. |
| `cosmos_autoencoder_kl` | `AutoencoderKLCosmos` | Cosmos 1.0 codec: Haar/rearrange patching, causal Conv3d, spatial/temporal attention. |
| `cosmos_single_file_conversion` | `single_file_utils.py`, `single_file_model.py` | Checkpoint key mapping and GGUF/single-file model identification. |

## 3. Important config dimensions

Representative checkpoint sweep:

| Repo/revision | Pipeline | Denoiser width/depth | Denoiser channels | Text features | VAE | Scheduler | Special |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Cosmos 1.0 7B T2W | `CosmosPipeline` in config, source has `CosmosTextToWorldPipeline` | 28 layers, 32 heads, head 128, hidden 4096 | 16 -> 16 | T5 d_model 1024 | `AutoencoderKLCosmos`, latent 16, patch 4, temporal/spatial compression 8 | EDMEuler epsilon, `sigma_data=0.5`, `sigma_max=80` | `extra_pos_embed_type=learnable`, rope `[2,1,1]`. |
| Cosmos 1.0 14B T2W | `CosmosTextToWorldPipeline` | 36 layers, 40 heads, head 128, hidden 5120 | 16 -> 16 | T5 d_model 1024 | Same Cosmos VAE | EDMEuler epsilon | rope `[2,2,2]`. |
| Cosmos 1.0 14B V2W | config unexpectedly reports `CosmosTextToWorldPipeline` | 36 layers, hidden 5120 | 17 -> 16 | T5 d_model 1024 | Same Cosmos VAE | EDMEuler epsilon | Extra input channel for condition mask. |
| Predict2 2B T2I | `Cosmos2TextToImagePipeline` | 28 layers, 16 heads, head 128, hidden 2048 | 16 -> 16 | T5 d_model 1024 | `AutoencoderKLWan`, z=16 | FlowMatch Euler, `sigma_data=1`, `sigma_max=80` | one latent frame, rope `[1,4,4]`. |
| Predict2 14B T2I | `Cosmos2TextToImagePipeline` | 36 layers, 40 heads, head 128, hidden 5120 | 16 -> 16 | T5 d_model 1024 | Wan z=16 | FlowMatch Euler | one latent frame, rope `[1,4,4]`. |
| Predict2 2B V2W | `Cosmos2VideoToWorldPipeline` | 28 layers, hidden 2048 | 17 -> 16 | T5 d_model 1024 | Wan z=16 | FlowMatch Euler | condition mask channel, rope `[1,3,3]`. |
| Predict2 14B V2W | `Cosmos2VideoToWorldPipeline` | 36 layers, hidden 5120 | 17 -> 16 | T5 d_model 1024 | Wan z=16 | FlowMatch Euler | condition mask channel, rope `[0.8333,2,2]`. |
| Predict2.5 2B base | `Cosmos2_5_PredictBasePipeline` | 28 layers, 16 heads, head 128, hidden 2048 | 17 -> 16 | Qwen2.5-VL hidden concat 100352 -> projected 1024 | Wan z=16 | UniPC flow, sigma max 200/min 0.01 | T2W/I2W/V2W in one pipeline. |
| Predict2.5 14B base | `Cosmos2_5_PredictBasePipeline` | 36 layers, 40 heads, head 128, hidden 5120 | 17 -> 16 | Qwen2.5-VL hidden concat 100352 -> projected 1024 | Wan z=16 | UniPC flow, sigma max 200/min 0.01 | 14B is open; 2B repo is gated but configs accessible. |

Transformer fields:

| Field | Cosmos 1.0 | Predict2 | Predict2.5 |
| --- | --- | --- | --- |
| `patch_size` | `[1,2,2]` | `[1,2,2]` | `[1,2,2]` |
| `max_size` | `[128,240,240]` | `[128,240,240]` | `[128,240,240]` |
| `concat_padding_mask` | true | true | true |
| `extra_pos_embed_type` | `learnable` | null | null |
| `qk_norm` | source forces RMSNorm | source forces RMSNorm | source forces RMSNorm |
| `use_crossattn_projection` | false | false | true |
| `crossattn_proj_in_channels` | n/a | n/a | 100352 |
| `encoder_hidden_states_channels` | n/a | n/a | 1024 |

VAE fields:

| VAE | Key fields | Boundary formula |
| --- | --- | --- |
| `AutoencoderKLCosmos` | `latent_channels=16`, `patch_size=4`, `spatial_compression_ratio=8`, `temporal_compression_ratio=8`, `num_layers=2`, per-latent-frame mean/std arrays | Cosmos 1.0 decodes `latents * std / sigma_data + mean` or `latents / sigma_data`. |
| `AutoencoderKLWan` | `z_dim=16`, `base_dim=96`, `dim_mult=[1,2,4,4]`, `temperal_downsample=[False,True,True]`, 16-channel mean/std | Predict2 decodes `latents * std / sigma_data + mean`; Predict2.5 decodes `latents * std + mean`. |

Recommended first Dinoml scheduler slice:

- For the selected Predict2.5 first path, implement `UniPCMultistepScheduler` with official sampled config: `prediction_type="flow_prediction"`, `solver_order=2`, `solver_type="bh2"`, `use_flow_sigmas=true`, `flow_shift=1.0`, `sigma_max=200.0`, `sigma_min=0.01`.
- Keep Predict2 FlowMatch Euler and Cosmos 1.0 EDMEuler as separate scheduler candidates. They use different preconditioning and do not share the same loop math.

## 3a. Family variation traps

- Do not treat Cosmos as one pipeline. Cosmos 1.0, Predict2, and Predict2.5 differ in text encoder, VAE class, scheduler, prompt embedding width, and denoising preconditioning.
- Source latent layout is NCTHW at transformer and VAE boundaries. NDHWC is only a guarded optimization.
- `CosmosTransformer3DModel` always concatenates a resized spatial padding mask when `concat_padding_mask=true`; Predict2/Predict2.5 Video2World also pass `condition_mask`, which adds another channel before padding-mask concatenation.
- Predict2.5 Qwen prompt embedding is not the final hidden state. Source normalizes every hidden state from layers 1..N over the feature dimension and concatenates them to width 100352 before a transformer-side projection to 1024.
- Predict2.5 timestep can be `[B]` for pure T2I/T2W-like paths or `[B,1,T,1,1]` when condition masks assign low timestep values to conditioned frames.
- Cosmos 1.0 EDMEuler loop calls `scheduler.step` once to compute predicted original sample, decrements `_step_index`, applies CFG in x0 space, then calls `scheduler.step` again with `pred_original_sample`. Treat this as a source-specific loop contract, not generic scheduler behavior.
- Predict2 FlowMatch pipelines construct custom sigmas with `torch.linspace(0, 1, num_steps)`, replace final sigma with `sigma_min` for `"sigma_min"`, and use explicit `c_in`, `c_skip`, `c_out` preconditioning around the transformer.
- Predict2.5 base pipeline uses UniPC directly on velocity/noise prediction; condition pixels/latents are hard-overridden with `gt_velocity` and condition masks.
- Cosmos 1.0 `AutoencoderKLCosmos` has a very different codec from Wan: Haar/rearrange 3D patching, causal Conv3d, GroupNorm over `B*T`, and spatial/temporal attention.
- Transfer2.5 is not a minor option. It adds `CosmosControlNetModel`, control latents, residual injection into transformer blocks, image-context placeholder support, and autoregressive chunking.

## 4. Runtime tensor contract

Predict2.5 base, default 704x1280 and 93 frames:

| Boundary | Tensor | Shape/layout | Notes |
| --- | --- | --- | --- |
| Qwen input ids | token ids | `[B,512]` | Chat template with system/user messages. |
| prompt embeds | `prompt_embeds` | `[B,512,100352]` | Concatenated normalized Qwen hidden states; transformer projects to 1024. |
| negative embeds | `negative_prompt_embeds` | `[B,512,100352]` | Empty/default negative prompt path for CFG. |
| latents | denoiser sample | `[B,16,24,88,160]` NCTHW | `T=(93-1)//4+1=24`, H/8, W/8. |
| condition latents | `cond_latents` | `[B,16,24,88,160]` | zeros for text-only; Wan VAE encode for image/video. |
| condition mask | `cond_mask` | `[B,1,24,88,160]` | Concatenated to transformer input as a channel. |
| condition indicator | `cond_indicator` | `[B,1,24,1,1]` | Builds token/frame timestep tensor. |
| timestep | pure/conditioned | `[B]` or `[B,1,T,1,1]` | Transformer expands 5D timesteps over patch grid when needed. |
| padding mask | spatial | `[1,1,H,W]` | Resized to latent H/W inside transformer and repeated over T. |
| patch tokens | transformer internal | `[B,T*(H/2)*(W/2), hidden]` | With `[1,2,2]`, default token count is `24*44*80=84480`. |
| transformer output | velocity/noise | `[B,16,24,88,160]` | Same NCTHW latent shape. |
| VAE decode input | denormalized | `[B,16,24,88,160]` | Predict2.5 uses `latents * std + mean`. |
| decoded video | sample | `[B,3,T_out,H,W]` | `_match_num_frames` repeats/trims frames when needed. |

Cosmos 1.0 704x1280, 121 frames:

- Latent shape is `[B,16,16,88,160]` because temporal compression is 8.
- Transformer token count is `16*44*80=56320`.
- Decode uses `AutoencoderKLCosmos`, not Wan VAE.

Predict2 Text2Image 768x1360:

- `num_frames=1`, latent shape is `[B,16,1,96,170]`.
- Transformer still uses 3D patch code with temporal patch size 1.
- Output image is `video[:, frame 0]` after Wan decode.

CPU/data-pipeline work includes tokenization/chat templating, Qwen/T5 text encoder execution when prompt embeddings are not supplied, PIL/video preprocessing, guardrail checks, and output conversion. GPU/runtime work includes latent RNG, VAE encode/decode, condition mask packing, transformer denoise, CFG, scheduler arithmetic, and latent normalization.

## 5. Operator coverage checklist

### Tensor/layout ops

- NCTHW video latents and Wan/Cosmos VAE tensors.
- Reshape/permute/flatten patchify: `[B,C,T,H,W] -> [B,T,H/2,W/2,C*1*2*2] -> Linear`.
- Unpatchify with Cosmos-specific output order: `proj_out -> unflatten(p_h,p_w,p_t,C) -> unflatten(T,H,W) -> permute(0,7,1,6,2,4,3,5) -> flatten`.
- Cat along channel dim for `condition_mask` and padding mask.
- Resize nearest for spatial padding mask from pixel H/W to latent H/W.
- CFG separate denoiser calls, chunk/add arithmetic.
- `torch.where`/mask-style blends for conditioned latents and ground-truth velocity.
- Per-channel latent mean/std broadcast over `[B,C,T,H,W]`.

### Convolution/downsample/upsample ops

- Wan VAE causal Conv3d, ResNet, downsample/upsample, attention block if enabled.
- Cosmos VAE causal Conv3d with repeated first-frame temporal padding.
- Cosmos VAE `CosmosConvProjection3d`: spatial Conv3d `(1,3,3)` then temporal Conv3d `(3,1,1)`.
- Cosmos VAE `CosmosDownsample3d`/`CosmosUpsample3d`: Conv3d branches plus avg-pool or interpolate-like repeat paths.
- Haar/rearrange patch/unpatch operations for `AutoencoderKLCosmos`.

### GEMM/linear ops

- Transformer patch embed Linear: `in_channels * 1 * 2 * 2 -> hidden`.
- Qwen hidden concat projection for Predict2.5: `100352 -> 1024 -> GELU`.
- Time embedding: sinusoidal Timesteps -> Linear -> SiLU -> Linear(3*hidden) plus RMSNorm.
- Self-attention Q/K/V and output projections, no bias by default.
- Cross-attention K/V from text embeddings; optional image-context Q/K/V in Transfer2.5.
- FeedForward GELU MLP with `mlp_ratio=4`.
- Final projection: `hidden -> out_channels * 1 * 2 * 2`.

### Attention primitives

- Noncausal latent-token self-attention with RMSNorm on Q/K and 3D RoPE.
- Cross-attention from latent tokens to text tokens; optional additive mask.
- Predict2.5/Transfer image-context branch in `CosmosAttnProcessor2_5`, separate image attention output added to text attention output.
- Cosmos VAE spatial attention per frame and temporal attention per spatial location.
- GQA-like key/value repeat in processor: key/value heads are repeated to match query heads when needed.

### Normalization and adaptive conditioning

- `RMSNorm` for timestep projection and Q/K norms.
- `LayerNorm(eps=1e-6, affine=False)` in AdaLayerNorm variants.
- `CosmosAdaLayerNormZero`: shift, scale, gate from timestep projection; gated residual for attention and FFN.
- `CosmosAdaLayerNorm`: final shift/scale before output projection.
- Wan VAE norms and Cosmos VAE `CosmosCausalGroupNorm`.

### Scheduler and guidance arithmetic

- Predict2.5 UniPC flow scheduler state/history.
- Predict2 FlowMatch custom sigma and preconditioning (`c_in`, `c_skip`, `c_out`).
- Cosmos 1.0 EDMEuler x0/eps two-step pattern.
- True CFG by separate positive and negative denoiser calls.
- Hard condition replacement: `gt_velocity + pred * (1 - cond_mask)` and conditioned x0/latents.

### Video/world-specific ops

- Temporal latent compression `(frames - 1)//4 + 1` for Wan VAE; `(frames - 1)//8 + 1` for Cosmos VAE.
- Image/video conditioning through VAE encode and last-frame padding.
- Predict2.5 `_match_num_frames` repeat/truncate after VAE decode.
- Transfer2.5 autoregressive chunk windowing and overlap removal.

## 6. Denoiser/model breakdown

`CosmosTransformer3DModel.forward`:

```text
hidden_states [B,C,T,H,W]
-> optional cat(condition_mask, dim=1)
-> resize padding_mask to latent H/W and cat over channel dim
-> 3D RoPE cos/sin from [T,H,W], patch size, fps
-> optional learnable positional embed for Cosmos 1.0
-> CosmosPatchEmbed: reshape/permute + Linear
-> flatten to tokens [B,T*H/2*W/2,hidden]
-> timestep embedding:
     [B] path or [B,1,T,1,1] path expanded over patch tokens
-> optional cross-attention projection for Predict2.5 Qwen concat features
-> N x CosmosTransformerBlock
-> CosmosAdaLayerNorm + Linear projection
-> Cosmos-specific unpatchify back to [B,out_channels,T,H,W]
```

`CosmosTransformerBlock`:

```text
optional before_proj for controlnet block
optional add extra learnable position embedding
AdaLayerNormZero -> self-attention(QK RMSNorm + RoPE) -> gated residual
AdaLayerNormZero -> text cross-attention, or text + image-context attention -> gated residual
AdaLayerNormZero -> GELU FeedForward -> gated residual
optional controlnet residual add / after_proj output for ControlNet
```

Shape examples:

- Predict2.5 2B: hidden 2048, 16 heads, head dim 128, 28 blocks.
- Predict2.5 14B and Cosmos/Predict2 14B: hidden 5120, 40 heads, head dim 128, 36 blocks.
- Default 704x1280x93 Predict2.5 token length is 84,480; 704x1280x121 Cosmos 1.0 token length is 56,320 because its VAE temporal compression is 8.

## 7. Attention requirements

Primary implementation is in `transformer_cosmos.py`; processors call `dispatch_attention_fn` from `attention_dispatch.py`.

- Self-attention: noncausal, mask-free in the base path, over latent patch tokens.
- Cross-attention: latent queries attend to text context. T5 paths zero padded embeddings but do not pass a text attention mask; the processor supports a mask.
- Q/K normalization: `RMSNorm` after head split for both self and cross attention.
- RoPE: 3D cos/sin generated from temporal, height, width token coordinates. Temporal RoPE is fps-scaled for video; images pass `fps=None`.
- GQA support: key/value heads are repeated to match query head count when ratios differ.
- Predict2.5 image-context branch: available in `CosmosAttnProcessor2_5`; Transfer source creates zero image context if `img_context_dim_in` is set.
- Fused projections are not a default requirement in source. Parity fallback is the eager/native `dispatch_attention_fn` path.

Flash-style constraints:

- Base self-attention is a candidate only with strong sequence-length/workspace guards; 84k tokens is far beyond typical image DiT sizes.
- QK RMSNorm and RoPE must be explicit pre-attention ops or fused under exact guards.
- Cross-attention is a separate provider shape; Predict2.5 text context is 512 tokens by 1024 after projection.
- Image-context attention must remain a separate attention/add branch; it cannot be silently folded into text context unless masks and projections are represented.
- Cosmos VAE spatial/temporal attention is a different attention family over codec feature maps and should not share transformer-token assumptions.

## 8. Scheduler and denoising-loop contract

Predict2.5 base pipeline:

```text
scheduler.set_timesteps(num_inference_steps)
for t in timesteps:
  sigma_t = scheduler.sigmas[i]
  in_latents = cond_mask * cond_latent + (1 - cond_mask) * latents
  in_timestep = cond_indicator * conditional_frame_timestep + (1 - cond_indicator) * sigma_t
  pred = transformer(in_latents, in_timestep, prompt_embeds, condition_mask)
  pred = gt_velocity + pred * (1 - cond_mask)
  if CFG:
    pred_neg = transformer(... negative_prompt_embeds ...)
    pred_neg = gt_velocity + pred_neg * (1 - cond_mask)
    pred = pred + guidance_scale * (pred - pred_neg)
  latents = UniPC.step(pred, t, latents)
```

Predict2 FlowMatch loop:

```text
sigmas = linspace(0,1,num_steps)
current_t = sigma / (sigma + 1)
c_in = c_skip = 1 - current_t
c_out = -current_t
model_input = latents * c_in or conditioned blend
model_x0 = c_skip * latents + c_out * transformer(model_input)
velocity = (latents - model_x0) / sigma
latents = FlowMatchEuler.step(velocity, t, latents)
```

Cosmos 1.0 EDMEuler loop:

```text
model_input = scheduler.scale_model_input(latents, t)
eps_or_denoised = transformer(model_input, timestep=t, prompt)
x0 = scheduler.step(model_output, t, sample, return_dict=False)[1]
scheduler._step_index -= 1
if CFG: x0 = x0_cond + scale * (x0_cond - x0_uncond)
latents = scheduler.step(x0, t, latents, pred_original_sample=x0)[0]
```

First Dinoml slice should keep scheduler table generation and history as host-visible runtime state. Compile one denoiser call, condition/CFG arithmetic, and one UniPC flow-prediction step only after the one-step tensor contract is proven.

## 9. Position, timestep, and custom math

- `CosmosRotaryPosEmbed` splits head dim into temporal, height, and width parts: height and width each get `hidden_size // 6 * 2`, and temporal gets the remainder.
- RoPE scale changes by family and checkpoint. Predict2.5 base uses `[1,3,3]`; Predict2 T2I uses `[1,4,4]`; Predict2 14B V2W uses `[0.8333,2,2]`; Cosmos 1.0 uses learnable extra pos embeddings plus RoPE.
- Temporal RoPE uses `seq / fps * base_fps` for video and plain sequence for images.
- `CosmosLearnablePositionalEmbed` normalizes the summed learned T/H/W position vector in fp32.
- Timestep embedding uses Diffusers `Timesteps(hidden, flip_sin_to_cos=True, downscale_freq_shift=0)`, then a two-layer SiLU MLP that emits `3*hidden` modulation features.
- Predict2.5 prompt embeddings normalize each Qwen hidden-state layer independently:
  `normalized = (hidden - mean(hidden,lastdim)) / (std(hidden,lastdim) + 1e-8)`, then concatenate layers 1..N.
- Predict2.5 condition timesteps are real sigma values for unconditioned frames and a small fixed `conditional_frame_timestep` for conditioned frames.

Precompute candidates: prompt/negative embeddings, cross-attention-projected text context for fixed prompts, RoPE tables for fixed shape/fps, padding masks, condition masks, and scheduler tables.

## 10. Preprocessing and input packing

Text:

- Cosmos 1.0/Predict2 tokenize with T5 tokenizer, max length 512 by pipeline default, zeroing padded embeddings after T5.
- Predict2.5 applies Qwen chat template with a fixed system message, asks for hidden states, normalizes and concatenates all non-embedding hidden layers, and duplicates for `num_videos_per_prompt`.
- Negative prompt defaults differ: older pipelines use a large default negative prompt; Transfer2.5 source uses empty string if none is provided.

Image/video:

- Source processors yield NCHW images and NCTHW videos.
- Predict2.5 T2W creates zero video input and zero condition latents/masks.
- Predict2.5 I2W turns one image into a video padded with repeated/zero frames, then VAE encodes.
- Predict2.5 V2W can use one or two latent conditional frames; pixel frames extracted are `4*(num_latent_conditional_frames-1)+1`.
- Predict2 V2W pads missing video frames by repeating the last input frame; Cosmos 1.0 V2W pads with zeros in `prepare_latents` for older source.
- Transfer2.5 preprocesses controls as video, VAE-encodes controls, normalizes with Wan stats, then feeds `CosmosControlNetModel`.

Layout guards:

- Preserve NCTHW at VAE and transformer boundaries initially.
- Treat transformer patchify/unpatchify as a no-layout-translation region until exact axis tests cover the unusual output permutation.
- NDHWC Conv3d islands are plausible only inside VAE blocks with explicit channel-axis norm rewrites, Conv3d weight transforms, temporal cache/padding mapping, and mean/std broadcast rewrites.

## 11. Graph rewrite / lowering opportunities

### Rewrite: Cosmos transformer patchify/unpatchify

Source pattern:

```text
NCTHW -> reshape(B,C,T/pt,pt,H/ph,ph,W/pw,pw)
-> permute(B,T,H,W,C,pt,ph,pw) -> Linear
Linear -> unflatten(p_h,p_w,p_t,C) -> unflatten(T,H,W)
-> permute(0,7,1,6,2,4,3,5) -> flatten
```

Replacement: explicit Cosmos video-token patchify/unpatchify op pair.

Preconditions: NCTHW source layout, patch `[1,2,2]`, H/W divisible by 2, exact Cosmos output permutation. Failure cases: assuming normal inverse patch order; changing temporal patch size without tests; NDHWC rewrite without weight/order transform.

### Rewrite: Qwen hidden concat projection

Source pattern:

```text
for hidden_states[1:]:
  normalize over feature dim
concat over feature dim -> Linear(100352 -> 1024) -> GELU
```

Replacement: prompt-embedding cache stage plus transformer-visible projection.

Preconditions: Qwen2.5-VL config with 28 hidden layers of width 3584, max sequence 512, no vision tokens in prompt path. Failure cases: different Qwen depth/width, visual prompt content, or tokenizer template changes.

### Rewrite: condition-mask pack

Source pattern:

```text
in_latents = cond_mask * cond_latent + (1 - cond_mask) * latents
hidden_states = cat([in_latents, cond_mask, resized_padding_mask], dim=1)
```

Replacement: fused condition-pack kernel producing transformer input channels.

Preconditions: Wan z=16, condition mask `[B,1,T,H,W]`, padding mask source `[1,1,Hpix,Wpix]`, nearest resize. Failure cases: Transfer controls or image-context branch requiring extra tensors.

### Rewrite: Predict2.5 UniPC condition/CFG step

Source pattern:

```text
pred = gt_velocity + pred * (1 - cond_mask)
pred = pred + guidance * (pred - pred_neg)
latents = scheduler.step(pred, t, latents)
```

Replacement: explicit condition+CFG pointwise kernel feeding host-owned UniPC state.

Preconditions: official Predict2.5 UniPC config, deterministic separate negative call, no callback mutation. Failure cases: alternate scheduler, stochastic branch, controlnet residuals.

### Rewrite: Cosmos VAE patching

Source pattern:

```text
Haar/rearrange 3D patch embed -> causal Conv3d codec -> unpatcher
```

Replacement: codec-level patch/unpatch operators separate from transformer patchify.

Preconditions: `AutoencoderKLCosmos` selected, patch method known, patch size 4. Failure cases: using Wan VAE path; confusing codec patching with transformer token patching.

## 12. Kernel fusion candidates

Highest priority:

- GEMM/Linear coverage for hidden 2048/5120 transformer QKV, cross-attention, FFN, final projection, and Predict2.5 cross-attention projection.
- QK RMSNorm + 3D RoPE + attention provider prelude with severe sequence/workspace guards.
- AdaLayerNormZero scale/shift/gate plus residual epilogues for self-attn, cross-attn, and FFN.
- Cosmos patchify/unpatchify kernels and condition/padding-mask channel packing.
- Wan VAE decode/encode for Predict2.5, matching the existing Wan report's first codec staging.

Medium priority:

- UniPC flow scheduler pointwise/history update for Predict2.5.
- Predict2 FlowMatch Euler preconditioning and velocity conversion.
- Cosmos 1.0 EDMEuler x0/eps two-step loop helpers.
- Cosmos VAE causal Conv3d + GroupNorm + SiLU + residual blocks.
- Prompt embedding cache/projection for Qwen hidden-state concat.

Lower priority:

- Transfer2.5 ControlNet residual injection and autoregressive chunking.
- Cosmos VAE Haar wavelet patch/unpatch fused kernels.
- Image-context attention branch fusion.
- Guardrail/safety processing, which is out of Dinoml first runtime scope.
- Single-file/GGUF conversion key remapping.

## 13. Runtime staging plan

Stage 1: Parse Predict2.5 14B configs from revision `diffusers/base/post-trained`; accept external Qwen prompt and negative prompt embeddings shaped `[B,512,100352]`.

Stage 2: Implement `CosmosTransformer3DModel` patchify/unpatchify, timestep embedding, 3D RoPE, Qwen cross-attention projection, and one block parity on reduced latent grids.

Stage 3: Full Predict2.5 transformer forward parity for random tensors at a small grid, then default 704x1280x93 shape if memory allows.

Stage 4: Add T2W path with zero condition latents/masks and true CFG as two explicit denoiser calls.

Stage 5: Implement official Predict2.5 UniPC flow-prediction scheduler slice with host-visible history.

Stage 6: Add Wan VAE decode for z=16 using Predict2.5 `latents * std + mean`, tiling disabled.

Stage 7: Add I2W/V2W VAE encode, condition mask, condition timestep, and ground-truth velocity replacement.

Stage 8: Add Predict2 FlowMatch Text2Image/Video2World as a separate path once base Cosmos transformer parity is stable.

Stage 9: Add Cosmos 1.0 only after `AutoencoderKLCosmos` gets an individual codec admission.

Stage 10: Add Transfer2.5 ControlNet after gated configs are available or a confirmed open config mirror is approved.

## 14. Parity and validation plan

- Config/default reconciliation for Predict2.5 2B/14B, Predict2 2B/14B, and Cosmos 1.0 7B/14B.
- Qwen prompt embedding parity: chat template, hidden-state layer normalization, concatenation width 100352.
- Cosmos patchify/unpatchify parity for `[B,17,24,88,160]`.
- RoPE parity for Predict2.5 `fps=None` and `fps=16/30` paths.
- One `CosmosTransformerBlock` parity for hidden 2048 and 5120.
- Full `CosmosTransformer3DModel` random tensor parity with and without 5D timestep.
- Condition-mask pack and gt-velocity replacement parity.
- CFG parity for positive/negative denoiser outputs.
- UniPC `set_timesteps` and one `step` parity for Predict2.5 config.
- Wan VAE decode parity for `[B,16,24,88,160]`; encode posterior sample/mode parity for I2W/V2W.
- Predict2 FlowMatch preconditioning parity as a separate test group.
- Cosmos 1.0 EDMEuler x0/eps loop parity as a separate test group.
- Suggested tolerances: fp32 scheduler and pointwise `rtol=1e-5, atol=1e-6`; transformer fp32 `rtol=1e-4, atol=1e-5`; bf16/fp16 start at `rtol=2e-2, atol=2e-2`.

## 15. Performance probes

- One Predict2.5 transformer step by latent grid: tiny synthetic, 704x1280x93, and higher frame counts.
- Attention backend comparison by token count and head count for hidden 2048 vs 5120.
- Qwen prompt embedding/projection time separated from denoiser time.
- CFG two-call cost and memory.
- Conditioned V2W overhead: VAE encode, condition-mask pack, gt-velocity replacement.
- UniPC scheduler overhead compared with transformer step.
- Wan VAE decode throughput at z=16.
- Cosmos 1.0 VAE decode throughput separately, especially Haar patch/unpatch and causal Conv3d.
- Transfer2.5 ControlNet overhead after config access is available.
- Faithful NCTHW path versus guarded NDHWC VAE Conv3d island.

## 16. Scope boundary and separate candidates

Separate candidate reports, not ignored:

- `cosmos_predict2_5_qwen_prompt`: Qwen2.5-VL prompt embedding extraction, hidden concat width 100352, and cache/projection strategy.
- `cosmos_predict2_5_i2w_v2w`: image/video conditioning, VAE encode, condition timesteps, gt velocity.
- `cosmos_transfer2_5_controlnet`: gated configs, control video latents, `CosmosControlNetModel`, residual injection, autoregressive chunks.
- `cosmos_predict2_flowmatch`: T5 prompt path and FlowMatch Euler c-in/c-out preconditioning.
- `cosmos_1_edm`: EDMEuler two-step x0/eps loop and Cosmos VAE boundary.
- `cosmos_autoencoder_kl`: `AutoencoderKLCosmos` codec, Haar/rearrange patching, spatial/temporal attention.
- `cosmos_single_file_gguf`: single-file checkpoint detection/conversion and GGUF mirror compatibility.

Ignored/out of scope for this audit:

- Multi-GPU/context parallel paths.
- Callback mutation and interactive interrupt.
- XLA, NPU, MPS, Flax, and ONNX branches.
- Safety checker / Cosmos guardrail runtime behavior.
- Training, losses, dropout, and gradient checkpointing.

## 17. Final implementation checklist

- [ ] Parse Predict2.5 14B component configs from `diffusers/base/post-trained`.
- [ ] Accept external Qwen prompt/negative embeddings `[B,512,100352]`.
- [ ] Implement Qwen cross-attention projection `100352 -> 1024 -> GELU`.
- [ ] Implement NCTHW Cosmos transformer patchify/unpatchify parity.
- [ ] Implement 3D RoPE with fps scaling and checkpoint-specific rope scale.
- [ ] Implement Cosmos timestep embedding, AdaLayerNormZero, gated residuals, and final AdaLayerNorm.
- [ ] Implement self-attention and text cross-attention with QK RMSNorm and RoPE.
- [ ] Implement full `CosmosTransformer3DModel` forward parity for Predict2.5 2B/14B shapes.
- [ ] Implement condition-mask and padding-mask channel packing.
- [ ] Implement CFG and ground-truth velocity replacement.
- [ ] Implement Predict2.5 UniPC flow-prediction scheduler slice.
- [ ] Implement Wan VAE z=16 decode with Predict2.5 mean/std boundary.
- [ ] Add Wan VAE encode and condition timestep path for I2W/V2W.
- [ ] Add short deterministic T2W loop parity with scheduler in host control.
- [ ] Add Predict2 FlowMatch and Cosmos 1.0 EDMEuler as separate scheduler/runtime candidates.
- [ ] Add `AutoencoderKLCosmos` codec report before Cosmos 1.0 full admission.
- [ ] Revisit Transfer2.5 after gated config access is granted.
