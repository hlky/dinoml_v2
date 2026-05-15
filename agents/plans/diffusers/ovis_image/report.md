# Diffusers Ovis Image Operator and Integration Report

Candidate slug: `ovis_image`

## 1. Source basis

```text
Diffusers commit/version:
  Local checkout diffusers at b3a515080752a3ba7ca92161e25530c7f280f629.

Model id(s):
  AIDC-AI/Ovis-Image-7B

Config sources:
  H:/configs had no cached Ovis Image files.
  Official Hugging Face raw/API reads succeeded for AIDC-AI/Ovis-Image-7B:
    model_index.json
    transformer/config.json
    vae/config.json
    scheduler/scheduler_config.json
    text_encoder/config.json
    tokenizer/tokenizer_config.json
    tokenizer/special_tokens_map.json
    tokenizer/vocab.json
    text_encoder/model.safetensors.index.json
    transformer/diffusion_pytorch_model.safetensors.index.json
  Hub API metadata checked:
    AIDC-AI/Ovis-Image-7B @ ac8fb1056c6df0b22901ddcabc965336eb9bdc41
  The selected official repo is public and not gated. No authenticated retry was
  needed. Configs were not saved under H:/configs because this worker owns only
  this report path.

Pipeline files inspected:
  diffusers/src/diffusers/pipelines/ovis_image/pipeline_ovis_image.py
    calculate_shift: line 56
    retrieve_timesteps: line 70
    OvisImagePipeline: line 129
    __init__: line 156
    encode prompt helpers: lines 182, 201, 240
    latent id/pack/unpack helpers: lines 320, 334, 342, 357
    __call__: line 414
  diffusers/src/diffusers/pipelines/ovis_image/pipeline_output.py

Model files inspected:
  diffusers/src/diffusers/models/transformers/transformer_ovis_image.py
    OvisImageAttnProcessor: line 68
    OvisImageAttention: line 135
    OvisImageSingleTransformerBlock: line 215
    OvisImageTransformerBlock: line 272
    OvisImagePosEmbed: line 356
    OvisImageTransformer2DModel: line 386
    forward: line 478
  diffusers/src/diffusers/models/autoencoders/autoencoder_kl.py
  diffusers/src/diffusers/models/autoencoders/vae.py
  diffusers/src/diffusers/models/resnet.py
  diffusers/src/diffusers/models/unets/unet_2d_blocks.py

Scheduler/processors/helpers inspected:
  diffusers/src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py
  diffusers/src/diffusers/models/attention_dispatch.py
  diffusers/src/diffusers/models/attention.py
  diffusers/src/diffusers/models/embeddings.py
  diffusers/src/diffusers/models/normalization.py
  diffusers/src/diffusers/image_processor.py
  diffusers/src/diffusers/loaders/peft.py
  diffusers/src/diffusers/loaders/lora_pipeline.py

External component configs/source inspected:
  Official AIDC-AI/Ovis-Image-7B Qwen3 text encoder and Qwen2 tokenizer configs.
  Installed Transformers 4.57.3 source:
    transformers/models/qwen3/modeling_qwen3.py
    transformers/models/qwen2/tokenization_qwen2_fast.py

Related reports read:
  agents/plans/diffusers/qwen_image/report.md
  agents/plans/diffusers/llada2/report.md
  agents/plans/diffusers/scheduler_matrix/report.md
  agents/plans/diffusers/omnigen/report.md was not present.

Any missing files or assumptions:
  No tiny/debug Ovis Image Diffusers config was found in H:/configs or public
  search results. Unauthenticated raw reads for guessed private/internal repos
  such as hf-internal-testing/tiny-ovis-image-pipe returned 401 and were not
  used for operator claims. This report focuses on the non-deprecated base
  text-to-image pipeline. XLA/NPU/MPS/Flax/ONNX, callbacks/interrupt mutation,
  safety/NSFW, training/loss/dropout/gradient checkpointing, and multi-GPU/
  context parallel are out of scope.
```

## 2. Pipeline and component graph

`OvisImagePipeline` wires `FlowMatchEulerDiscreteScheduler`, `AutoencoderKL`,
external `Qwen3Model`, external `Qwen2TokenizerFast`, and
`OvisImageTransformer2DModel`. The offload sequence is
`text_encoder->transformer->vae`.

```text
prompt
  -> Ovis system prompt + Qwen chat template + Qwen2 tokenizer
  -> Qwen3Model hidden states [B,284,2048]
  -> mask zeroing and prefix drop to prompt embeds [B,256,2048]
  -> latent noise map [B,16,H/8,W/8] source NCHW
  -> 2x2 latent packing to [B,(H/16)*(W/16),64]
  -> denoising loop:
       OvisImageTransformer2DModel(latent tokens, text tokens,
                                   timestep, text ids, image ids)
       optional true CFG second denoiser call
       FlowMatchEulerDiscreteScheduler.step
  -> unpack latent tokens to [B,16,H/8,W/8]
  -> unscale/unshift for AutoencoderKL
  -> AutoencoderKL decode
  -> VaeImageProcessor postprocess
```

First-slice required components:

| Component | Class/file | Runtime role |
| --- | --- | --- |
| Pipeline | `OvisImagePipeline`, `pipeline_ovis_image.py` | Prompt encoding, latent pack/unpack, dynamic FlowMatch setup, true CFG orchestration, VAE boundary. |
| Denoiser | `OvisImageTransformer2DModel`, `transformer_ovis_image.py` | Packed-token MMDiT with 6 dual-stream blocks and 27 single-stream blocks. |
| Scheduler | `FlowMatchEulerDiscreteScheduler` | Dynamic-shift FlowMatch Euler update over packed latent tokens. |
| VAE | `AutoencoderKL` | 2D NCHW latent decode from 16 channels to RGB. Encode is not used by base text-to-image but is a codec candidate. |
| Text encoder | `Qwen3Model` | External prompt embedding producer. First Dinoml slice can accept prompt embeds. |
| Tokenizer | `Qwen2TokenizerFast` | CPU/data-pipeline tokenization and chat templating. |
| Processor | `VaeImageProcessor` | Image resize/postprocess. Pipeline constructs it with `vae_scale_factor * 2`. |

Cacheable stages: prompt embeddings, text ids for fixed prompt length, latent
image ids for fixed resolution, RoPE cos/sin for fixed `(text_len, image_grid)`,
FlowMatch timesteps/sigmas for fixed resolution and step count, and VAE decode
as a separate artifact.

Separate candidate reports:

| Candidate | Classes/files | Runtime delta |
| --- | --- | --- |
| `ovis_image_lora_peft` | `OvisImageTransformer2DModel` inherits `PeftAdapterMixin`; generic loaders in `loaders/peft.py` and transformer LoRA pipeline mixins in `loaders/lora_pipeline.py` | Runtime adapter load/fuse/unfuse state over transformer linear layers. The base pipeline class itself does not inherit a family-local LoRA loader mixin. |
| `ovis_image_from_original_single_file` | `OvisImageTransformer2DModel` inherits `FromOriginalModelMixin`; repo also exposes single-file-ish `ovis_image.safetensors` and `ae.safetensors` | Weight conversion and source-name mapping candidate, not a denoiser op requirement. |
| `ovis_image_autoencoderkl_codec` | `AutoencoderKL`, `autoencoder_kl.py`, `vae.py` | Standalone 16-channel AutoencoderKL encode/decode with Ovis scaling and shift factors. |
| `ovis_image_text_encoder_qwen3` | external Transformers `Qwen3Model` | Qwen3 prompt encoder compilation, including GQA self-attention, QK norm, RoPE, RMSNorm, and SwiGLU MLP. |
| `ovis_image_scheduler_flowmatch_advanced` | `scheduling_flow_match_euler_discrete.py` | Karras/exponential/beta sigma conversions, stochastic sampling, custom timesteps, and per-token timestep branch beyond the sampled base contract. |

No family-local IP-Adapter, ControlNet, T2I-Adapter, GLIGEN, img2img,
inpaint, depth2img, or upscaling pipeline was found in the non-deprecated
`ovis_image` folder.

## 3. Important config dimensions

Representative checkpoint sweep:

| Config | Pipeline | Transformer depth | Heads x dim | Inner dim | Text dim | Latent/token | Scheduler | Weight metadata |
| --- | --- | ---: | --- | ---: | ---: | --- | --- | --- |
| `AIDC-AI/Ovis-Image-7B` | `OvisImagePipeline` | 6 dual + 27 single | 24 x 128 | 3072 | 2048 | VAE z=16, packed C=64 | FlowMatch Euler, dynamic exponential shift | Transformer shards total 14,740,899,456 bytes; text encoder 1,720,574,976 params / 6,882,299,904 bytes |

Transformer config facts:

| Field | Official value | Source default / note |
| --- | ---: | --- |
| `patch_size` | 1 | `proj_out` emits `patch_size * patch_size * out_channels`; packing is pipeline-level 2x2, not model-internal patchify. |
| `in_channels` | 64 | Packed 2x2 latent tiles: `4 * latent_channels`. |
| `out_channels` | `null` | Source resolves to `in_channels`, so output width is 64. |
| `num_layers` | 6 | Dual-stream image/text transformer blocks. |
| `num_single_layers` | 27 | Single-stream blocks over concatenated text+image, then split back. |
| `num_attention_heads` | 24 | Inner dim = 3072. |
| `attention_head_dim` | 128 | Head dim equals sum of RoPE axes. |
| `joint_attention_dim` | 2048 | Qwen3 hidden size. |
| `axes_dims_rope` | `[16,56,56]` | Three-axis ids: text ids and image ids both have 3 coordinates. |

VAE config facts:

| Field | Official value |
| --- | --- |
| Class | `AutoencoderKL` |
| `block_out_channels` | `[128,256,512,512]` |
| `down_block_types` / `up_block_types` | four `DownEncoderBlock2D` / four `UpDecoderBlock2D` |
| `latent_channels` | 16 |
| `sample_size` | 1024 |
| `layers_per_block` | 2 |
| `norm_num_groups` | 32 |
| `act_fn` | `silu` |
| `scaling_factor` | 0.3611 |
| `shift_factor` | 0.1159 |
| `force_upcast` | true |
| `use_quant_conv` / `use_post_quant_conv` | false / false |
| Effective `vae_scale_factor` | `2 ** (len(block_out_channels)-1) = 8` |

Scheduler config facts:

| Field | Official value | Effective default when omitted |
| --- | ---: | --- |
| Class | `FlowMatchEulerDiscreteScheduler` | pipeline constructor type is specific, not Karras enum. |
| `num_train_timesteps` | 1000 | source default 1000 |
| `shift` | 3.0 | source default 1.0, not used directly when dynamic shifting is on |
| `use_dynamic_shifting` | true | source default false |
| `base_image_seq_len` | 256 | source default 256 |
| `max_image_seq_len` | 4096 | source default 4096 |
| `base_shift` | 0.5 | source default 0.5 |
| `max_shift` | 1.15 | source default 1.15 |
| `time_shift_type` | omitted | effective source default `exponential` |
| `shift_terminal` | omitted | effective source default `None` |
| `invert_sigmas` | omitted | false |
| `use_karras_sigmas` / `use_exponential_sigmas` / `use_beta_sigmas` | omitted | all false |
| `stochastic_sampling` | omitted | false |

External Qwen3 text encoder config:

| Field | Official value |
| --- | --- |
| Class | `Qwen3Model` |
| `hidden_size` | 2048 |
| `num_hidden_layers` | 28 |
| `num_attention_heads` / `num_key_value_heads` | 16 / 8 |
| `head_dim` | 128 |
| `intermediate_size` | 6144 |
| `vocab_size` | 151936 |
| `max_position_embeddings` | 40960 |
| `rope_theta` | 1000000 |
| `hidden_act` | `silu` |
| `rms_norm_eps` | 1e-6 |
| `attention_bias` | false |
| `tie_word_embeddings` | true |
| `use_sliding_window` | false |

Tokenizer config facts:

| Field | Official value / note |
| --- | --- |
| Class | model index says `Qwen2TokenizerFast`; tokenizer config says `Qwen2Tokenizer`. Diffusers imports `Qwen2TokenizerFast`. |
| `model_max_length` | 131072 |
| `add_bos_token` | false |
| `pad_token` | `<|endoftext|>` |
| `eos_token` | `<|im_end|>` |
| Chat template | tokenizer config omits one, but repo has `tokenizer/chat_template.jinja`; pipeline calls `apply_chat_template(..., enable_thinking=False)`. |

Recommended first Dinoml scheduler slice: FlowMatch Euler with custom pipeline
sigmas, dynamic exponential shift from image sequence length, terminal sigma 0,
and non-stochastic `sample + dt * model_output`. Defer Karras/exponential/beta
conversion flags, stochastic branch, and per-token timesteps.

## 3a. Family variation traps

- `transformer.config.in_channels=64` is the packed-token channel width, not
  the VAE latent channel count. VAE latents are source NCHW `[B,16,H/8,W/8]`;
  the denoiser sees `[B,(H/16)*(W/16),64]`.
- Ovis packing is pipeline-level 2x2 latent packing. The transformer
  `patch_size=1` does not patchify spatial maps internally.
- `out_channels=null` in config resolves to 64 through the model constructor.
- Prompt embeddings are fixed to 256 tokens after a source-defined 28-token
  prefix drop. Pipeline `max_sequence_length` is validated but not otherwise
  used in `encode_prompt`; the hard source limits are `tokenizer_max_length=284`
  and `user_prompt_begin_id=28`.
- The prompt attention mask is not passed to the transformer. The pipeline
  zeroes padded token hidden states before slicing, then attends over the full
  256-token text sequence.
- True CFG is implemented as a separate unconditional transformer call, not
  batch concatenation. There is no guidance-rescale norm correction in Ovis.
- Ovis uses Qwen3 text embeddings with width 2048, unlike QwenImage's
  Qwen2.5-VL width 3584.
- Ovis VAE is normal 2D `AutoencoderKL`, unlike QwenImage's 3D singleton-frame
  VAE. Do not carry QwenImage NCTHW assumptions into this target.
- The base pipeline has no ControlNet/IP-Adapter/inpaint side branch in source;
  do not inflate first-slice scope with inactive generic Diffusers branches.
- Source attention can use fused projections when `AttentionModuleMixin` mutates
  `OvisImageAttention`, but official config does not require fused projection
  modules at load time.
- `joint_attention_kwargs` are accepted by the pipeline but the target
  processor only recognizes parameters in `OvisImageAttnProcessor.__call__`;
  unknown keys are warned and ignored.
- AutoencoderKL source supports slicing/tiling and attention processor changes.
  Official base decode path does not enable slicing or tiling by config.

## 4. Runtime tensor contract

For the default 1024x1024 text-to-image request:

| Boundary | Tensor | Shape/layout | Notes |
| --- | --- | --- | --- |
| Prompt message | `messages` | Python list of chat strings | `system_prompt + prompt`, then Qwen chat template. |
| Token IDs | `input_ids` | `[B,284] int64` | `tokenizer_max_length = 256 + 28`, no extra special tokens. |
| Attention mask | `attention_mask` | `[B,284] int64/bool-like` | Used only to zero text hidden states. |
| Text hidden | `outputs.last_hidden_state` | `[B,284,2048]` | Qwen3 output. |
| Prompt embeds | `prompt_embeds` | `[B*num_images,256,2048]` | Multiply by mask, drop first 28 tokens, repeat per image. |
| Text ids | `text_ids` | `[256,3]` | First coordinate 0; coordinates 1 and 2 are token indices. |
| Noise latent map | before pack | `[B*num_images,16,128,128]` NCHW | Random normal at `prompt_embeds.dtype`. |
| Packed latents | denoiser input | `[B*num_images,4096,64]` | 2x2 spatial pack. |
| Image ids | `latent_image_ids` | `[4096,3]` | First coordinate 0; height/width grid coordinates. |
| Timestep | model input | `[B*num_images]` | Pipeline passes `t / 1000`; transformer multiplies by 1000. |
| Denoiser output | `noise_pred` | `[B*num_images,4096,64]` | Same packed-token shape. |
| Scheduler output | `latents` | `[B*num_images,4096,64]` | Flow Euler update in packed-token space. |
| Unpacked latents | VAE input before scale | `[B*num_images,16,128,128]` NCHW | Reverse 2x2 packing. |
| VAE decode input | `latents / 0.3611 + 0.1159` | `[B*num_images,16,128,128]` NCHW | Source uses scaling then shift. |
| VAE decoded image | `image` | `[B*num_images,3,1024,1024]` NCHW | Then postprocessed to PIL/NumPy/torch per `output_type`. |

CPU/data-pipeline work: prompt string construction, tokenizer, chat template,
PIL/NumPy output conversion. GPU/runtime work: Qwen3 text encoder if admitted,
latent id generation, pack/unpack, denoiser, CFG arithmetic, scheduler step,
VAE decode.

## 5. Operator coverage checklist

Tensor/layout ops:

- NCHW latent allocation and VAE boundaries.
- Ovis 2x2 latent pack:
  `view(B,C,H/2,2,W/2,2) -> permute(0,2,4,1,3,5) -> reshape(B,H/2*W/2,4C)`.
- Ovis 2x2 latent unpack:
  `view(B,H/2,W/2,C/4,2,2) -> permute(0,3,1,4,2,5) -> reshape(B,C/4,H,W)`.
- `text_ids` and `img_ids` construction, concat over sequence dimension.
- Prompt mask multiply and prefix slice if compiling prompt encoding.
- True CFG pointwise arithmetic over packed tokens.
- VAE decode scaling and shift: `latents / scaling_factor + shift_factor`.

GEMM/linear ops:

- Transformer `x_embedder`: Linear(64 -> 3072).
- Transformer `context_embedder`: RMSNorm(2048) + Linear(2048 -> 3072).
- Timestep embedding: sinusoidal 256 -> Linear/activation/Linear to 3072.
- Dual-stream block adaptive norm projections from `AdaLayerNormZero` for image
  and text streams.
- Dual-stream attention image Q/K/V and text added Q/K/V, each 3072-wide with
  bias, plus image/text output projections.
- Dual-stream SwiGLU feed-forward for image and text streams.
- Single-stream block `AdaLayerNormZeroSingle`, Q/K/V self-attention, MLP gate
  projection `Linear(3072 -> 2 * 4*3072)`, and `proj_out`.
- Final `AdaLayerNormContinuous` and Linear(3072 -> 64).
- External Qwen3 text encoder: embeddings, QKV, output projections, SwiGLU MLP,
  tied LM embeddings not used by `Qwen3Model` pipeline output.

Attention primitives:

- Dual-stream joint text-image noncausal attention for 6 blocks.
- Single-stream noncausal attention over concatenated `[text,image]` tokens for
  27 blocks.
- Q/K RMSNorm in every Ovis attention path.
- 3-axis RoPE from text and image ids, using Diffusers `apply_rotary_emb` with
  real cos/sin tensors and `sequence_dim=1`.
- No attention mask in the base denoiser call.

Normalization and adaptive conditioning:

- RMSNorm on prompt embeddings before context projection.
- AdaLayerNormZero, AdaLayerNormZeroSingle, AdaLayerNormContinuous.
- LayerNorm without affine before dual-stream MLPs.
- GroupNorm in VAE encoder/decoder ResNet blocks.
- Qwen3 RMSNorm in text encoder.

Position/timestep/guidance embeddings:

- `Timesteps(256, flip_sin_to_cos=True, downscale_freq_shift=0)`.
- `TimestepEmbedding(256 -> 3072)`.
- Ovis 3-axis RoPE with `axes_dims_rope=[16,56,56]`.
- `guidance_scale` is true CFG only; no embedded guidance tensor.

Scheduler and guidance arithmetic:

- Pipeline sigmas default: `linspace(1.0, 1/steps, steps)` unless caller passes
  `sigmas` or scheduler config uses `use_flow_sigmas`.
- Dynamic shift `mu = image_seq_len * m + b`.
- FlowMatch Euler `prev = sample + (sigma_next - sigma) * model_output`.
- CFG: `neg + guidance_scale * (cond - neg)`.

VAE/postprocessing ops:

- AutoencoderKL decode: Conv2d, ResNetBlock2D, GroupNorm, SiLU,
  upsampling blocks, optional mid-block attention depending source defaults,
  final Conv2d.
- `force_upcast=true` is a precision policy for VAE admission.
- VaeImageProcessor postprocess: denormalize/clamp/format conversion.

## 6. Denoiser/model breakdown

`OvisImageTransformer2DModel.forward`:

```text
hidden_states [B,S_img,64] -> x_embedder -> [B,S_img,3072]
encoder_hidden_states [B,256,2048] -> RMSNorm -> context_embedder -> [B,256,3072]
timestep (/1000 from pipeline) -> *1000 -> Timesteps -> TimestepEmbedding -> temb [B,3072]
ids = concat(txt_ids [256,3], img_ids [S_img,3])
OvisImagePosEmbed(ids) -> cos/sin RoPE sections [S_txt+S_img,128]
6 x OvisImageTransformerBlock
27 x OvisImageSingleTransformerBlock
AdaLayerNormContinuous(hidden_states, temb) -> proj_out -> [B,S_img,64]
```

Dual-stream `OvisImageTransformerBlock`:

```text
image stream:
  AdaLayerNormZero(hidden, temb) -> norm hidden + gate/shift/scale
text stream:
  AdaLayerNormZero(context, temb) -> norm context + gate/shift/scale
OvisImageAttention:
  separate image QKV and text added QKV
  unflatten heads, RMSNorm Q/K for both streams
  concat text then image along sequence
  apply shared RoPE
  dispatch_attention_fn
  split text/image outputs
  output projections
gated residual attention adds
LayerNorm -> adaptive scale/shift -> SwiGLU FeedForward -> gated residual adds
```

Single-stream `OvisImageSingleTransformerBlock`:

```text
concat [text,image]
AdaLayerNormZeroSingle -> norm states + gate
Linear -> split MLP value/gate -> SiLU(gate) * value
self-attention over concatenated tokens with QK RMSNorm and RoPE
concat attention output and MLP hidden over channel axis
gate * Linear(dim + mlp_hidden -> dim)
residual add
split [text,image]
```

VAE decode:

```text
packed latents -> unpack to [B,16,H/8,W/8]
latents / scaling_factor + shift_factor
AutoencoderKL.decode:
  optional post_quant_conv is disabled by config
  Decoder conv_in -> mid block -> up blocks
  GroupNorm -> SiLU -> conv_out
VaeImageProcessor.postprocess
```

## 7. Attention requirements

Primary implementation: `OvisImageAttnProcessor` in
`transformer_ovis_image.py`, which calls `dispatch_attention_fn` from
`attention_dispatch.py`.

Required behavior:

- Q/K/V tensors are projected in `[B,S,C]`, then unflattened to
  `[B,S,heads,head_dim]`.
- Official base uses 24 heads and head dim 128.
- Dual-stream blocks have image Q/K/V plus text added Q/K/V. Single-stream
  blocks self-attend over already concatenated text+image tokens.
- Q/K RMSNorm is applied before RoPE.
- RoPE is applied to the concatenated sequence after text/image concat in
  dual-stream attention and directly to the concatenated sequence in
  single-stream attention.
- No base attention mask is passed from the pipeline. Prompt padding becomes
  zero hidden states, not an attention mask.
- Dropout is configured but inference uses no training dropout.

Backend/flash feasibility:

- Diffusers attention dispatch supports native SDPA, native flash, flash-attn,
  flash varlen, flex, sage, and xFormers backends. Base Ovis does not force a
  backend; native/eager dispatch is the parity definition.
- Flash-style Dinoml provider is plausible for first inference when
  `attention_mask is None`, dtype is fp16/bf16/fp32 as supported, head dim 128
  is supported, and QK RMSNorm/RoPE are pre-applied or fused explicitly.
- Flash-attn 2 paths in Diffusers reject `attn_mask`; Ovis base avoids this.
  A future prompt-mask-aware Ovis variant would need a varlen or fallback path.
- Fused projections are source-supported through `AttentionModuleMixin`, but
  dual-stream added-QKV fusion must keep the exact text/image split and output
  projection semantics.
- Layout candidate: keep graph boundaries token-major `[B,S,C]`; provider
  internals may use head-major or sequence-major layouts. NHWC/NCHW layout
  translation is only relevant to VAE/image stages, not transformer attention.

## 8. Scheduler and denoising-loop contract

Pipeline setup:

```text
sigmas = linspace(1.0, 1 / num_inference_steps, num_inference_steps)
if scheduler.config.use_flow_sigmas:
  sigmas = None
image_seq_len = latents.shape[1]
mu = calculate_shift(image_seq_len, base_image_seq_len, max_image_seq_len,
                     base_shift, max_shift)
retrieve_timesteps(scheduler, num_inference_steps, device, sigmas=sigmas, mu=mu)
scheduler.set_begin_index(0)
```

For 1024x1024, `image_seq_len=4096`; with official config
`base_image_seq_len=256`, `max_image_seq_len=4096`, `base_shift=0.5`,
`max_shift=1.15`, `mu=1.15`.

Loop body:

```text
timestep = t.expand(batch).to(latents.dtype)
cond = transformer(latents, timestep / 1000, prompt_embeds, text_ids, image_ids)
if guidance_scale > 1:
  neg = transformer(latents, timestep / 1000, negative_prompt_embeds, negative_text_ids, image_ids)
  noise_pred = neg + guidance_scale * (cond - neg)
latents = scheduler.step(noise_pred, t, latents)
```

`FlowMatchEulerDiscreteScheduler` source behavior for first slice:

- `use_dynamic_shifting=true` requires `mu`.
- Omitted `time_shift_type` means exponential dynamic shift.
- Omitted `shift_terminal` means no terminal stretch beyond appending final
  zero sigma.
- `set_timesteps` appends a terminal zero sigma when `invert_sigmas=false`.
- `step` upcasts sample to fp32, computes `dt = sigma_next - sigma`, then
  returns `sample + dt * model_output`, cast back to model output dtype.

Host-control first: schedule construction, timestep iteration, true CFG second
call, and callback/interrupt handling should remain host-visible. Candidate
compiled kernels are CFG arithmetic and one FlowMatch Euler pointwise step.

Source default scheduler and recommended first Dinoml scheduler slice are the
same family: FlowMatch Euler dynamic exponential shift, custom sigmas, no
stochastic branch. Advanced FlowMatch sigma conversion modes are separate.

## 9. Position, timestep, and custom math

Custom math to preserve:

- `calculate_shift`: linear interpolation from image sequence length to `mu`.
- Text ids: `zeros(seq,3)` with both coordinate 1 and 2 set to token index.
- Image ids: `zeros(h,w,3)` with coordinate 1 as row and coordinate 2 as
  column, flattened row-major.
- Ovis 3-axis RoPE: for each axis, `get_1d_rotary_pos_embed(axis_dim, ids)`,
  concatenate cos and sin sections. Source uses float64 freqs except on MPS/NPU.
- Timestep path: pipeline passes `t/1000`; transformer multiplies by 1000
  before sinusoidal projection. Dinoml should keep one canonical representation
  and test equivalence.
- VAE decode boundary: `latents = latents / scaling_factor + shift_factor`.

Precomputable for fixed resolution and prompt length: image ids, text ids,
RoPE cos/sin, scheduler timesteps/sigmas, negative prompt embeddings.

Dynamic: prompt content, batch size, image resolution, custom sigmas,
guidance-scale branch, and `output_type`.

## 10. Preprocessing and input packing

Text preprocessing:

- Pipeline prepends a fixed Ovis system prompt asking for image-description
  detail.
- It calls tokenizer `apply_chat_template(..., tokenize=False,
  add_generation_prompt=True, enable_thinking=False)`.
- Tokenizer call uses `padding="max_length"`, `truncation=True`,
  `max_length=284`, `return_tensors="pt"`, and `add_special_tokens=False`.
- Qwen3 hidden states are multiplied by the attention mask, then the first 28
  tokens are dropped to produce 256 prompt tokens.
- Prompt embeddings are repeated by `num_images_per_prompt` through
  repeat/view.
- Negative prompt embeddings follow the same path only when
  `guidance_scale > 1`.

Latent preprocessing:

- Base text-to-image starts from random NCHW latents
  `[B*num_images,16,H/8,W/8]`.
- Height and width are internally rounded down to multiples of
  `vae_scale_factor * 2 = 16` for packing.
- Packed latent shape is `[B*num_images,H/16*W/16,64]`.
- User-supplied `latents` are assumed already packed; the pipeline returns them
  as-is after dtype/device conversion and only regenerates `latent_image_ids`.
  This is a validation trap for direct latent callers.

Image postprocessing:

- Unpack packed latents, apply VAE scaling/shift, decode NCHW through
  AutoencoderKL, then `VaeImageProcessor.postprocess`.

## 11. Graph rewrite / lowering opportunities

### Rewrite: Ovis latent pack/unpack

Source pattern: NCHW latent map with channel count 16, spatial 2x2 packing into
tokens and reverse unpack before VAE.

Replacement pattern: `ovis_latent_pack2x2` and `ovis_latent_unpack2x2` ops, or
a canonical reshape/permute/reshape sequence.

Preconditions: source layout NCHW, latent height/width divisible by 2, channel
count 16 for the official VAE, packed channel count 64, row-major flatten order
exactly `(h2,w2,c,dh,dw)`.

Layout constraints: NHWC translation may only apply if the pack/unpack flatten
order and VAE/transformer boundaries are explicitly rewritten.

Failure cases: caller supplies already-packed latents with unknown provenance,
non-16-channel VAE variants, odd latent spatial dimensions.

Parity test sketch: random `[B,16,128,128]` pack/unpack round trip and exact
comparison to Diffusers helper outputs.

### Rewrite: Ovis joint attention region

Source pattern: adaptive norm -> separate Q/K/V projections -> QK RMSNorm ->
RoPE -> dispatch attention -> projection -> gated residual.

Replacement pattern: explicit `joint_attention(text,image)` region with
provider selection after QK norm/RoPE; optionally fuse projections and prelude.

Preconditions: no attention mask, no adapter mutation, known text/image sequence
lengths, head dim 128 provider support, RoPE ids fixed for the call.

Failure cases: future masks, PEFT modules not folded into weights, fused
projection state mismatch, changed processor class.

Parity test sketch: one dual block and one single block with random tensors,
with native dispatch as reference.

### Rewrite: FlowMatch Euler Ovis slice

Source pattern: dynamic-shift sigmas plus `sample + dt * model_output`.

Replacement pattern: host-visible schedule metadata plus fused pointwise step.

Preconditions: `stochastic_sampling=false`, scalar timestep path,
`invert_sigmas=false`, no Karras/exponential/beta conversions, explicit sigma
index.

Failure cases: custom scheduler config enabling stochastic sampling or
conversion modes, per-token timestep branch.

Parity test sketch: compare `set_timesteps` tables and one-step updates for
1024px and a smaller resolution.

### Rewrite: VAE decode conv island

Source pattern: AutoencoderKL decode NCHW Conv2d/ResNet/GroupNorm/SiLU/upsample.

Replacement pattern: keep a codec artifact first; later lower fully controlled
Conv2d islands to NHWC with weight transforms and axis rewrites.

Preconditions: slicing/tiling disabled, official channel schedule, no
post-quant conv, no latent embeds.

Failure cases: tiling/slicing enabled, force-upcast mismatch, attention
processor mutation inside mid block.

Parity test sketch: decode fixed `[1,16,128,128]` latent against Diffusers in
fp32 and bf16/fp16 policy modes.

## 12. Kernel fusion candidates

Highest priority:

- Ovis attention prelude and provider: Q/K/V projections, QK RMSNorm, RoPE,
  noncausal attention, output projection, and gated residual in both dual and
  single blocks.
- AdaLayerNormZero/AdaLayerNormZeroSingle modulation and gated residual
  epilogues.
- SwiGLU feed-forward fusion for 3072-wide hidden states.
- Ovis latent pack/unpack kernels.
- FlowMatch Euler step and true CFG pointwise arithmetic.

Medium priority:

- RoPE cos/sin cache and apply kernel for `[256+4096,128]` at 1024px.
- Text encoder Qwen3 attention and MLP kernels if prompt encoding is compiled.
- VAE decode Conv2d + GroupNorm + SiLU islands, upsample blocks, and mid-block
  attention if active.
- Projection fusion and PEFT-folded weight materialization.

Lower priority:

- Autoencoder encode path, since base text-to-image only decodes.
- VAE tiling/slicing overlap logic.
- Advanced FlowMatch sigma conversion and stochastic branches.
- Single-file/from-original conversion and adapter mutation workflows.

Layout notes:

- Transformer core is token-major `[B,S,C]`; do not apply NHWC reasoning there.
- VAE and processor boundaries are source NCHW. NHWC can be explored only as a
  guarded local optimization for Conv/GroupNorm/resample islands.
- Axis rewrites needed for NHWC VAE islands include GroupNorm channel axis,
  Conv2d weight layout, upsample spatial axes, and postprocess expectations.
- Protect latent pack/unpack and scheduler broadcasting with a conceptual
  `no_layout_translation()` guard until a dedicated rewrite proves parity.

## 13. Runtime staging plan

1. Stage 1: parse official `AIDC-AI/Ovis-Image-7B` model index and component
   configs. Accept external `prompt_embeds`, `text_ids`, and `latent_image_ids`
   first.
2. Stage 2: implement Ovis latent id generation plus 2x2 pack/unpack parity.
3. Stage 3: implement one `OvisImageTransformerBlock` and one
   `OvisImageSingleTransformerBlock` with random token tensors, QK RMSNorm,
   RoPE, modulation, gates, attention, and SwiGLU.
4. Stage 4: compile the full Ovis transformer denoiser step with external
   prompt embeddings at official dimensions.
5. Stage 5: add FlowMatch Euler dynamic exponential scheduler table and one
   step parity over packed latents.
6. Stage 6: add true CFG two-call orchestration and pointwise arithmetic.
7. Stage 7: add AutoencoderKL decode as a separate codec artifact or explicit
   pipeline stage.
8. Stage 8: admit Qwen3 text encoder or keep prompt embedding cache as the
   public first integration boundary.
9. Stage 9: optimize attention/norm/MLP/VAE fusions, then evaluate PEFT/LoRA
   and single-file conversion candidates.

First Dinoml admission recommendation: `ovis_image_denoiser_step` with inputs
`latents [B,S,64]`, `prompt_embeds [B,256,2048]`, `timestep [B]`,
`text_ids [256,3]`, and `image_ids [S,3]`; output `noise_pred [B,S,64]`. Keep
tokenization, text encoder, scheduler loop, true CFG orchestration, and VAE
decode outside until block and full-denoiser parity are stable.

## 14. Parity and validation plan

- Config parse test for model index, transformer, VAE, scheduler, text encoder,
  tokenizer, and safetensors index metadata.
- Prompt preprocessing parity for fixed prompts: chat string, token length,
  attention-mask zeroing, drop 28, repeat per image.
- Latent pack/unpack parity for `[B,16,128,128] <-> [B,4096,64]`.
- Text/image id and RoPE frequency parity for 512px, 768px, and 1024px.
- Timestep embedding parity proving `pipeline t/1000` plus transformer `*1000`.
- Attention processor parity for dual-stream and single-stream blocks.
- One dual block and one single block random tensor parity.
- Full transformer one-step parity with external prompt embeddings.
- FlowMatch scheduler table and one-step parity for official dynamic shift.
- True CFG parity: two denoiser outputs and `neg + scale * (cond - neg)`.
- AutoencoderKL decode parity for `[1,16,128,128]`.
- Short deterministic denoising-loop parity with scheduler in Python.
- End-to-end image smoke after text encoder, denoiser, scheduler, and VAE each
  have isolated parity.
- Suggested tolerances: scheduler fp32 `rtol=1e-5, atol=1e-6`; transformer fp32
  `rtol=1e-4, atol=1e-5`; bf16/fp16 initially `rtol=2e-2, atol=2e-2`; image
  decode tolerances should account for VAE upcast policy.

## 15. Performance probes

- Denoiser step by resolution: 512, 768, 1024, and custom packed sequence
  lengths.
- Attention backend comparison: native SDPA/eager parity, guarded flash-style
  provider, fused-projection variants.
- Per-block split: adaptive norm, QKV, QK RMSNorm/RoPE, attention, MLP.
- Dual-stream 6-block cost versus single-stream 27-block cost.
- True CFG overhead: one denoiser call versus two calls.
- Scheduler/CFG overhead versus denoiser time.
- Pack/unpack memory bandwidth.
- VAE decode throughput at 1024px with force-upcast policy.
- Qwen3 text encoder throughput and prompt-cache hit/miss behavior.
- VRAM and temporary/workspace usage for bf16 transformer shards, text encoder,
  VAE, and optional offload sequence.

## 16. Scope boundary and separate candidates

Separate candidate reports related to this family:

- `ovis_image_lora_peft`: PEFT adapter load/set/enable/disable/fuse/unfuse on
  `OvisImageTransformer2DModel`.
- `ovis_image_from_original_single_file`: original/single-file conversion and
  weight-name mapping for repo-level `ovis_image.safetensors`/`ae.safetensors`.
- `ovis_image_autoencoderkl_codec`: Ovis 16-channel AutoencoderKL encode/decode
  with scaling factor 0.3611 and shift factor 0.1159.
- `ovis_image_text_encoder_qwen3`: external Qwen3 prompt encoder compilation.
- `ovis_image_scheduler_flowmatch_advanced`: custom timesteps, sigma conversion
  modes, stochastic sampling, and per-token timesteps.

Genuinely ignored/out of scope for this audit:

- Multi-GPU/context parallel paths.
- Callback mutation and interactive interrupt.
- XLA, NPU, MPS, Flax, and ONNX variants.
- Safety checker and NSFW filtering.
- Training, losses, dropout, and gradient checkpointing.
- IP-Adapter, ControlNet, T2I-Adapter, GLIGEN, img2img, inpaint, depth2img, and
  upscaling: no active Ovis Image family implementation was found in the
  inspected Diffusers folder.

## 17. Final implementation checklist

- [ ] Parse `AIDC-AI/Ovis-Image-7B` model index and component configs.
- [ ] Load `OvisImageTransformer2DModel` weights and reconcile `out_channels=null`.
- [ ] Accept external prompt embeddings and ids for first denoiser parity.
- [ ] Implement Ovis 2x2 latent pack/unpack.
- [ ] Implement text/image id generation and Ovis 3-axis RoPE parity.
- [ ] Implement Ovis timestep embedding path.
- [ ] Implement Ovis QK RMSNorm + RoPE + attention fallback parity.
- [ ] Implement dual-stream `OvisImageTransformerBlock`.
- [ ] Implement single-stream `OvisImageSingleTransformerBlock`.
- [ ] Add full transformer one-step parity at official dimensions.
- [ ] Implement FlowMatch Euler dynamic exponential Ovis scheduler slice.
- [ ] Add true CFG two-call arithmetic parity.
- [ ] Add AutoencoderKL decode boundary with Ovis scale/shift.
- [ ] Add optional Qwen3 text encoder prompt-cache integration.
- [ ] Benchmark attention, single-stream tail, pack/unpack, scheduler/CFG, and VAE decode.
- [ ] Open separate candidates for PEFT/LoRA, single-file conversion, Qwen3 text encoder, and Ovis AutoencoderKL codec.
