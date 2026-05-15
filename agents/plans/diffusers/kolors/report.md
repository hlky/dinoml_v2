# Diffusers Kolors Operator and Integration Report

Target slug: `kolors`

## 1. Source basis

```text
Diffusers commit/version:
  Local checkout diffusers at b3a515080752a3ba7ca92161e25530c7f280f629.

Model id(s):
  Kwai-Kolors/Kolors-diffusers
  hf-internal-testing/tiny-kolors-pipe
  Kwai-Kolors/Kolors-Inpainting
  Kwai-Kolors/Kolors-IP-Adapter-Plus
  Kwai-Kolors/Kolors-IP-Adapter-FaceID-Plus
  Kwai-Kolors/Kolors-ControlNet-Canny
  Kwai-Kolors/Kolors-ControlNet-Depth
  Kwai-Kolors/Kolors-ControlNet-Pose

Config sources:
  H:/configs/Kwai-Kolors/Kolors-diffusers/model_index.json
  H:/configs/Kwai-Kolors/Kolors/model_index.json
  H:/configs/Kwai-Kolors/Kolors-Inpainting/model_index.json
  H:/configs/Kwai-Kolors/Kolors-ControlNet-*/model_index.json
  H:/configs/Kwai-Kolors/Kolors-IP-Adapter-*/model_index.json
  Official component configs were missing from H:/configs for the primary repos,
  so authenticated `hf download` was used. Downloaded configs landed in the
  Hugging Face cache, for example:
    C:/Users/user/.cache/huggingface/hub/models--Kwai-Kolors--Kolors-diffusers/snapshots/7e091c75199e910a26cd1b51ed52c28de5db3711/
    C:/Users/user/.cache/huggingface/hub/models--hf-internal-testing--tiny-kolors-pipe/snapshots/5b8eaba51c19a982b5fb4a475a36a06bf4a4cacf/
    C:/Users/user/.cache/huggingface/hub/models--Kwai-Kolors--Kolors-Inpainting/snapshots/3bb98b424c9aa02d11164bf78904a3e7fba17c09/

Pipeline files inspected:
  diffusers/src/diffusers/pipelines/kolors/pipeline_kolors.py
  diffusers/src/diffusers/pipelines/kolors/pipeline_kolors_img2img.py
  diffusers/src/diffusers/pipelines/kolors/pipeline_output.py
  diffusers/src/diffusers/pipelines/pag/pipeline_pag_kolors.py

Model files inspected:
  diffusers/src/diffusers/pipelines/kolors/text_encoder.py
  diffusers/src/diffusers/pipelines/kolors/tokenizer.py
  diffusers/src/diffusers/models/unets/unet_2d_condition.py
  diffusers/src/diffusers/models/unets/unet_2d_blocks.py
  diffusers/src/diffusers/models/resnet.py
  diffusers/src/diffusers/models/attention.py
  diffusers/src/diffusers/models/attention_processor.py
  diffusers/src/diffusers/models/embeddings.py
  diffusers/src/diffusers/models/autoencoders/autoencoder_kl.py

Scheduler/processors/helpers inspected:
  diffusers/src/diffusers/schedulers/scheduling_euler_discrete.py
  diffusers/src/diffusers/image_processor.py
  diffusers/src/diffusers/loaders/lora_pipeline.py
  diffusers/src/diffusers/loaders/ip_adapter.py
  diffusers/src/diffusers/loaders/unet.py
  diffusers/src/diffusers/models/controlnets/controlnet.py
  diffusers/tests/pipelines/kolors/test_kolors.py
  diffusers/tests/pipelines/kolors/test_kolors_img2img.py
  Prior reports: stable_diffusion_xl, sdxl_controlnet_adapters,
  scheduler_matrix, stable_diffusion_1_5.

External component configs inspected:
  Kolors local ChatGLMModel config under text_encoder/config.json.
  Kolors local ChatGLMTokenizer tokenizer_config.json.
  CLIP image encoder configs for Kolors IP-Adapter Plus and FaceID Plus.

Any missing files or assumptions:
  Base text-to-image and Kolors img2img are non-deprecated Diffusers pipelines.
  There is no Kolors-specific non-deprecated inpaint, ControlNet, or T2I-Adapter
  pipeline class in this checkout; related official repos were inventoried as
  separate candidate surfaces. `Kwai-Kolors/Kolors-IP-Adapter-Plus/config.json`
  is `{}` and `ip_adapter/config.json` 404s, so IP-Adapter operator details are
  source/load-path plus image-encoder-config backed, not adapter-weight-header
  backed. Safety, callbacks, XLA/NPU/MPS/Flax/ONNX, training, and multi-GPU
  paths are out of scope.
```

## 2. Pipeline and component graph

Kolors is an SDXL-adjacent latent text-to-image pipeline that replaces SDXL's
dual CLIP text stack with one ChatGLM text encoder and a ChatGLM tokenizer. The
UNet still uses SDXL-style `addition_embed_type="text_time"` micro-conditioning,
but it receives ChatGLM token embeddings before projecting them from 4096 to
2048 inside `UNet2DConditionModel`.

```text
prompt strings or cached prompt embeddings
  -> ChatGLMTokenizer: input_ids, attention_mask, position_ids
  -> ChatGLMModel hidden states [S,B,4096]
  -> prompt embeddings [B,S,4096] and pooled last-token hidden [B,4096]
  -> CFG concat of negative/positive prompt and pooled embeddings
  -> added time IDs: original size, crop top-left, target size
  -> initialize latent noise [B,4,H/8,W/8]
  -> denoising loop:
       scheduler.scale_model_input
       -> UNet2DConditionModel(latents, t, prompt_embeds,
                               added_cond_kwargs={text_embeds,time_ids})
       -> CFG arithmetic
       -> scheduler.step
  -> AutoencoderKL decode(latents / scaling_factor)
  -> VaeImageProcessor postprocess
```

Required base components:

| Component | Class | File |
| --- | --- | --- |
| pipeline | `KolorsPipeline` | `pipelines/kolors/pipeline_kolors.py` |
| text encoder | `ChatGLMModel` | `pipelines/kolors/text_encoder.py` |
| tokenizer | `ChatGLMTokenizer` | `pipelines/kolors/tokenizer.py` |
| denoiser | `UNet2DConditionModel` | `models/unets/unet_2d_condition.py` |
| scheduler | `EulerDiscreteScheduler` in official config, typed as `KarrasDiffusionSchedulers` | `schedulers/scheduling_euler_discrete.py` |
| codec | `AutoencoderKL` | `models/autoencoders/autoencoder_kl.py` |
| output | `KolorsPipelineOutput` | `pipelines/kolors/pipeline_output.py` |

Optional base-pipeline components: `image_encoder` and `feature_extractor` for
IP-Adapter. Declared offload order is
`text_encoder->image_encoder->unet->vae` for text-to-image and
`text_encoder->image_encoder-unet->vae` for img2img.

Cacheable stages: tokenization, prompt embeddings, negative embeddings, pooled
embeddings, size/crop time ID tensors, IP image embeddings, scheduler
timesteps/sigmas for a fixed schedule, caller-supplied initial latents, and VAE
latents for img2img.

Separate candidate reports:

| Candidate | Support status and anchors | Variant delta |
| --- | --- | --- |
| `kolors_lora_textual_inversion_adapters` | `KolorsPipeline` inherits `StableDiffusionLoraLoaderMixin`; no `TextualInversionLoaderMixin` is inherited in the current Kolors class. Loader anchors: `loaders/lora_pipeline.py`, `loaders/peft.py`. | LoRA/PEFT mutates UNet and possibly text-encoder attention/linear weights. Textual inversion is not a base Kolors mixin surface unless a future pipeline adds it. |
| `kolors_ip_adapter` | `IPAdapterMixin`, `loaders/ip_adapter.py`, `IPAdapterAttnProcessor*`, official `Kwai-Kolors/Kolors-IP-Adapter-Plus` image encoder config. | Adds CLIP image encoder or precomputed image embeds, image projection layers, and added K/V image attention branches. Kolors has a special restore path for `text_encoder_hid_proj` because its base UNet already owns `encoder_hid_proj="text_proj"`. |
| `kolors_controlnet` | No Kolors ControlNet pipeline class in checkout. Official Canny/Depth/Pose configs are ControlNet-shaped with `_class_name` values `ControlNetModel_JQ` or `Kolors_ControlNetModel`; source anchor is generic `ControlNetModel`. | Adds condition-image preprocessing and down/mid residuals into the Kolors/SDXL-shaped UNet; config loading may need class-name normalization before current Diffusers can load those repos directly. |
| `kolors_t2i_adapter` | No Kolors T2I-Adapter pipeline or official T2I adapter config found in this audit. Generic anchors: `pipelines/t2i_adapter`, `models/adapter.py`. | Treat as unsupported for current Kolors first slice unless a downstream repo supplies a compatible adapter and pipeline glue. |
| `kolors_pag` | `KolorsPAGPipeline` in `pipelines/pag/pipeline_pag_kolors.py`. | Perturbed attention guidance changes UNet attention processors and guidance batching. Separate runtime-adapter candidate. |
| `kolors_img2img` | `KolorsImg2ImgPipeline` in `pipelines/kolors/pipeline_kolors_img2img.py`. | Adds VAE encode, strength and denoising-start timestep slicing, image noising, and optional latent input. |
| `kolors_inpaint` | No Kolors inpaint pipeline class in checkout. `Kwai-Kolors/Kolors-Inpainting` has a 9-channel UNet config but model index still names `StableDiffusionXLPipeline`. | Requires an explicit inpaint pipeline admission before relying on it: mask/masked-image latents and 9-channel UNet input are real config deltas, but source wiring is absent here. |
| `kolors_depth2img` | No non-deprecated Kolors depth pipeline found. | Unsupported for this family in current Diffusers. |
| `kolors_upscale` | No Kolors upscaler pipeline found. | Unsupported for this family in current Diffusers. |
| `kolors_gligen` | No Kolors GLIGEN pipeline found. | Unsupported; SD1 GLIGEN remains deprecated and separate. |

## 3. Important config dimensions

Representative checkpoint sweep:

| Repo | Pipeline class | Text hidden | Prompt seq used by pipeline | UNet in/out | UNet channels | Cross dim | Encoder hid dim/type | Add embed input | Scheduler | VAE |
| --- | --- | ---: | ---: | --- | --- | ---: | --- | ---: | --- | --- |
| `Kwai-Kolors/Kolors-diffusers` | `KolorsPipeline` | 4096 | max 256 | 4/4 | 320,640,1280 | 2048 | 4096 / `text_proj` | 5632 | EulerDiscrete, epsilon, 1100 train steps | sample 1024, scale 0.13025 |
| `hf-internal-testing/tiny-kolors-pipe` | `KolorsPipeline` | 8 | max 256 source cap, tiny ChatGLM seq 8192 | 4/4 | 2,4 | 8 | null | 56 | EulerDiscrete, epsilon, 1000 train steps | sample 128, scale 0.18215 |
| `Kwai-Kolors/Kolors-Inpainting` | model index says `StableDiffusionXLPipeline` | 4096 | source cap 256 | 9/4 | 320,640,1280 | 2048 | 4096 / `text_proj` | 5632 | EulerDiscrete, same as base | same as base |

Base UNet dimensions:

| Field | Value | Source |
| --- | --- | --- |
| `sample_size` | 128 latent, normally 1024 pixel output | UNet config |
| `in_channels`, `out_channels` | 4, 4 | UNet config |
| `down_block_types` | `DownBlock2D`, `CrossAttnDownBlock2D`, `CrossAttnDownBlock2D` | UNet config |
| `up_block_types` | `CrossAttnUpBlock2D`, `CrossAttnUpBlock2D`, `UpBlock2D` | UNet config |
| `layers_per_block` | 2 | UNet config |
| `transformer_layers_per_block` | 1, 2, 10 | UNet config |
| `attention_head_dim` | 5, 10, 20, interpreted by SDXL convention as heads for 64-wide head dim | UNet config and block construction convention |
| `cross_attention_dim` | 2048 | UNet config |
| `encoder_hid_dim_type` | `text_proj` | UNet config |
| `encoder_hid_proj` | Linear 4096 -> 2048 | UNet source plus config |
| `addition_embed_type` | `text_time` | UNet config |
| `addition_time_embed_dim` | 256 | UNet config |
| `projection_class_embeddings_input_dim` | 5632 = 4096 pooled + 6*256 time IDs | UNet config and pipeline `_get_add_time_ids` |
| `use_linear_projection` | true | UNet config |

ChatGLM text encoder dimensions:

| Field | Base value | Tiny value | Source |
| --- | ---: | ---: | --- |
| layers | 28 | 2 | text encoder config |
| hidden size | 4096 | 8 | text encoder config |
| FFN hidden size | 13696 | 16 | text encoder config |
| attention heads | 32 | 4 | text encoder config |
| KV channels/head dim | 128 | 2 | text encoder config |
| multi-query groups | 2 | 2 | text encoder config |
| QKV projection output | 4096 + 2*2*128 = 4608 | 8 + 2*2*2 = 16 | source formula |
| tokenizer vocab | 65024 padded vocab | 65024 | config |
| source sequence length | 32768 | 8192 | config |
| pipeline max sequence length | 256 hard cap | 256 hard cap | pipeline `check_inputs` |
| dtype metadata | `torch_dtype="float16"` | `float32` | text encoder config |
| text encoder shard metadata | 12,487,168,064 bytes, 200 tensors | not inspected | safetensors index metadata |

VAE and scheduler dimensions:

| Component | Base value | Notes |
| --- | --- | --- |
| VAE scale factor | 8 | computed from 4 VAE channel blocks |
| VAE latent channels | 4 | AutoencoderKL |
| VAE block channels | 128,256,512,512 | config |
| VAE `force_upcast` | omitted in base config, effective source default `true` | source default |
| VAE quant/post-quant conv | omitted in base config, effective `true`/`true` | source default |
| scheduler | `EulerDiscreteScheduler` | model index and scheduler config |
| train timesteps | 1100 | Kolors base scheduler differs from SDXL base 1000 |
| beta range | 0.00085 to 0.014, scaled linear | scheduler config |
| prediction type | epsilon | scheduler config |
| timestep spacing | leading, `steps_offset=1` | scheduler config |
| compatible scheduler set | Karras enum via pipeline typing | source annotation |
| recommended first Dinoml slice | EulerDiscrete epsilon with 1100-step table | exact source default |

## 3a. Family variation traps

- Kolors is not a dual-CLIP SDXL pipeline. It has one ChatGLM tokenizer and one
  ChatGLM text encoder with `[S,B,4096]` internal layout.
- The prompt-token conditioning width is 4096 until the UNet's
  `encoder_hid_proj` maps it to 2048. A Dinoml stage that accepts prompt embeds
  must decide whether the boundary is pre-projection `[B,S,4096]` or
  post-projection `[B,S,2048]`.
- `projection_class_embeddings_input_dim=5632` is Kolors-specific:
  `4096 pooled + 6*256 size/crop embeddings`, not SDXL base's 2816.
- `force_zeros_for_empty_prompt=false` for `Kolors-diffusers`; older
  `Kwai-Kolors/Kolors` and `Kolors-Inpainting` cached model indexes say true.
- Official scheduler uses 1100 training timesteps and beta_end 0.014; do not
  borrow SDXL's 1000-step Euler table.
- Source tensors for UNet, VAE, control images, and img2img are NCHW. NHWC is a
  guarded optimization only. GroupNorm, channel concat, VAE latent stats,
  attention flatten/restore, and scheduler broadcasting are axis-sensitive.
- ChatGLM source uses sequence-major hidden states `[S,B,H]`, causal attention,
  MQA expansion, RoPE cache indexing from tokenizer `position_ids`, and RMSNorm.
  This is a separate text-encoder operator slice from CLIP.
- The text encoder's `use_cache` defaults true, but the pipeline calls full
  prompt encoding and reads hidden states; KV cache outputs are not reused
  across diffusion steps.
- Kolors base inherits IP-Adapter mixins. Loading IP-Adapter replaces or wraps
  UNet encoder/image projection state, and unload has Kolors-specific restore
  logic for `encoder_hid_proj`.
- `Kolors-Inpainting` has a real 9-channel UNet config but no Kolors inpaint
  pipeline class in the current non-deprecated source folder.

## 4. Runtime tensor contract

Base text-to-image at 1024x1024 with CFG and one image per prompt:

| Boundary | Tensor | Source layout | Shape |
| --- | --- | --- | --- |
| tokenizer output | `input_ids` | `[B,S]`, left padded | `[B,256]` by pipeline default cap |
| tokenizer mask | `attention_mask` | `[B,S]` | `[B,256]`; 0 for left padding |
| tokenizer positions | `position_ids` | `[B,S]` | `[B,256]`; left padding positions are 0 |
| ChatGLM embedding | hidden | `[S,B,4096]` | after embedding transpose |
| ChatGLM hidden states | per-layer tuple | `[S,B,4096]` | source returns all hidden states when requested |
| prompt embeds | penultimate hidden | `[B,S,4096]` | `output.hidden_states[-2].permute(1,0,2).clone()` |
| pooled embeds | final hidden last sequence row | `[B,4096]` | `output.hidden_states[-1][-1,:,:]` |
| CFG prompt embeds | concat negative/positive | `[2B,S,4096]` | before UNet projection |
| UNet encoder projection | text proj | `[2B,S,2048]` | internal `Linear(4096 -> 2048)` |
| add time IDs | size/crop | `[2B,6]` after repeat | each scalar projected to 256 |
| add text embeds | pooled | `[2B,4096]` | concatenated with 1536 time features |
| latent state | denoising latents | NCHW | `[B,4,128,128]` |
| UNet input | CFG duplicated latents | NCHW | `[2B,4,128,128]` |
| UNet output | predicted noise | NCHW | `[2B,4,128,128]`, then chunk on batch |
| scheduler output | updated latents | NCHW | `[B,4,128,128]` |
| VAE decode input | unscaled latents | NCHW | `[B,4,128,128] / 0.13025` |
| decoded image | VAE sample | NCHW | `[B,3,1024,1024]` |
| postprocess | output images | NHWC for numpy/PIL | `[B,1024,1024,3]` for numpy |

Img2img adds:

| Boundary | Tensor | Shape/layout | Notes |
| --- | --- | --- | --- |
| preprocessed image | image | NCHW `[B,3,H,W]`, normalized by `VaeImageProcessor` | CPU/data path then GPU tensor |
| encoded latents | `vae.encode(image)` sample/mode | NCHW `[B,4,H/8,W/8]` | scaled by `vae.config.scaling_factor`, or latent mean/std formula if present |
| noised latents | `scheduler.add_noise(init_latents, noise, timestep)` | NCHW | controlled by `strength` or `denoising_start` |
| supplied latent image | image with 4 channels | NCHW `[B,4,H/8,W/8]` | bypasses VAE encode |

CPU/data-pipeline work: sentencepiece tokenization, left padding, position ID
construction, PIL/numpy image normalization, and output conversion. GPU/runtime
work: ChatGLM forward if text encoder is compiled, UNet denoising, scheduler
pointwise math, VAE encode/decode, and adapter/control side models.

## 5. Operator coverage checklist

Tensor/layout ops:

- NCHW image/latent maps, with guarded NHWC conv islands only after parity.
- ChatGLM `[B,S,H] -> [S,B,H]` transpose, prompt hidden `[S,B,H] -> [B,S,H]`
  permute/clone, left-padding mask and position tensors.
- CFG batch concat/chunk for latents, prompt embeddings, pooled embeddings, and
  added time IDs.
- `repeat`, `view`, `reshape`, `cat`, `split/chunk`, `expand`, `contiguous`,
  and scalar broadcasts over latent maps.
- Img2img VAE latent scaling and optional latent mean/std application over
  channel axis.

Convolution/downsample/upsample ops:

- Base UNet `Conv2d(4 -> 320, 3x3, padding=1)` and output
  `Conv2d(320 -> 4, 3x3, padding=1)`.
- Kolors inpaint candidate `Conv2d(9 -> 320, 3x3, padding=1)`.
- UNet `ResnetBlock2D` 3x3 conv pairs, 1x1 shortcuts when channels change,
  `Downsample2D`, `Upsample2D`.
- VAE encoder/decoder Conv2d, ResNet, down/up blocks, quant and post-quant
  convs.
- ControlNet candidate conditioning embedding Conv2d stack
  3 -> 16 -> 32 -> 96 -> 256 -> 320.

GEMM/linear ops:

- ChatGLM token embedding lookup and output vocabulary projection.
- ChatGLM per layer: QKV Linear 4096 -> 4608 with Q bias for all QKV; output
  Linear 4096 -> 4096 without bias; MLP Linear 4096 -> 27392 and
  13696 -> 4096 without bias.
- UNet `encoder_hid_proj` Linear 4096 -> 2048.
- UNet timestep embedding, `text_time` add embedding MLP
  5632 -> time-embedding dim -> time-embedding dim.
- UNet attention Q/K/V and output projections; `use_linear_projection=true`.

Attention primitives:

- ChatGLM causal self-attention with RoPE, MQA groups 2 expanded to 32 heads,
  SDPA fast path when PyTorch 2 is available.
- UNet noncausal spatial self-attention and cross-attention to prompt tokens.
- IP-Adapter candidate branch-wise image K/V attention.
- PAG candidate attention-processor mutations.

Normalization and adaptive conditioning:

- ChatGLM RMSNorm over hidden dimension, fp32 variance and dtype cast-back.
- UNet/VAE GroupNorm over channel axis and LayerNorm over token dimension.
- SiLU, SwigLU, GELU/GEGLU in UNet feed-forward blocks.
- SDXL-style time embedding add into ResNet blocks and `text_time` addition
  into UNet time embedding.

Scheduler and guidance arithmetic:

- EulerDiscrete `set_timesteps`, `scale_model_input`, epsilon prediction
  update, sigma table on CPU, step index.
- CFG: `uncond + guidance_scale * (text - uncond)`.
- Optional embedded guidance scale path if `time_cond_proj_dim` is set; inactive
  in inspected base configs.
- Img2img `add_noise` and timestep slicing by strength.

## 6. Denoiser/model breakdown

Base Kolors UNet active path:

```text
sample [B,4,H,W]
-> Timesteps + TimestepEmbedding
-> add_embedding(concat(pooled_glm [B,4096],
                        add_time_proj(6 scalars) [B,1536]))
-> encoder_hid_proj(prompt_embeds 4096 -> 2048)
-> Conv2d(4 -> 320)
-> DownBlock2D(320)
-> CrossAttnDownBlock2D(640), transformer_layers_per_block=2
-> CrossAttnDownBlock2D(1280), transformer_layers_per_block=10
-> UNetMidBlock2DCrossAttn(1280)
-> CrossAttnUpBlock2D(1280)
-> CrossAttnUpBlock2D(640)
-> UpBlock2D(320)
-> GroupNorm -> SiLU -> Conv2d(320 -> 4)
```

`ResnetBlock2D` is the SD/SDXL additive time-conditioning path because
`resnet_time_scale_shift="default"`:

```text
GroupNorm -> SiLU -> Conv2d
time embedding -> SiLU -> Linear -> add as channel bias
GroupNorm -> SiLU -> Conv2d -> residual add
```

`BasicTransformerBlock` in the UNet follows the latent diffusion pattern:

```text
NCHW map -> flatten spatial tokens [B,H*W,C]
LayerNorm -> self-attention -> residual
LayerNorm -> cross-attention(prompt tokens width 2048 after projection) -> residual
LayerNorm -> feed-forward GEGLU/GELU -> residual
restore NCHW map
```

ChatGLM text encoder active path:

```text
input_ids [B,S]
-> Embedding [B,S,4096] -> transpose/contiguous [S,B,4096]
-> 28 x GLMBlock:
     RMSNorm
     Linear QKV 4096 -> 4608
     split Q [S,B,32,128], K/V [S,B,2,128]
     RoPE on Q and K
     expand K/V groups 2 -> 32 heads
     causal SDPA or eager baddbmm/softmax/bmm
     Linear output 4096 -> 4096
     residual
     RMSNorm
     Linear 4096 -> 27392
     SwigLU chunk to 13696
     Linear 13696 -> 4096
     residual
-> final RMSNorm
```

## 7. Attention requirements

Kolors has two attention classes in the base slice:

| Attention | Source | Shape | Masking | Backend path |
| --- | --- | --- | --- | --- |
| ChatGLM causal self-attention | `CoreAttention` and `SelfAttention` in `text_encoder.py` | Q `[S,B,32,128]`, K/V expanded from MQA groups 2 to 32 | Causal if no explicit mask and square sequence; padding mask otherwise | PyTorch 2 SDPA after permute to `[B,H,S,D]`; eager baddbmm/softmax/bmm fallback |
| UNet self/cross attention | `Attention` and processors in shared model files | latent query tokens by resolution, prompt K/V length up to 256, width 2048 after projection | no causal mask for UNet base | `AttnProcessor2_0` SDPA default or eager `AttnProcessor` fallback |

Flash feasibility:

- ChatGLM text attention could use a flash-style causal provider under fixed
  dtype/head-dim/dropout-0 guards, but MQA expansion and RoPE must be explicit.
  A better optimized path would avoid materializing expanded K/V where the
  provider supports GQA/MQA; the source parity path expands to dense heads.
- UNet attention is dense, noncausal, and dropout-free in inference. A
  flash-style provider is valid for base self/cross attention when there are no
  added K/V branches, unsupported masks, live LoRA adapter mutations, or custom
  processors.
- IP-Adapter attention should remain separate branch-wise attention. Concating
  text and image tokens is not generally equivalent because per-adapter scales
  and optional masks are applied after each image branch.
- Current Diffusers semantic source is `attention_processor.py` for UNet and
  `text_encoder.py` for ChatGLM. `attention_dispatch.py` is not the primary
  implementation path for Kolors base.

## 8. Scheduler and denoising-loop contract

Official base scheduler:

```text
EulerDiscreteScheduler
num_train_timesteps=1100
beta_start=0.00085
beta_end=0.014
beta_schedule=scaled_linear
prediction_type=epsilon
timestep_spacing=leading
steps_offset=1
use_karras_sigmas=false
final_sigmas_type=zero by source default when omitted
```

Loop-side graph:

```text
timesteps = scheduler.set_timesteps(num_inference_steps, custom timesteps/sigmas optional)
latents = randn([B,4,H/8,W/8]) * scheduler.init_noise_sigma
for t in timesteps:
  latent_model_input = cat([latents, latents]) if CFG else latents
  latent_model_input = scheduler.scale_model_input(latent_model_input, t)
  noise_pred = unet(latent_model_input, t, prompt_embeds,
                    timestep_cond=None,
                    added_cond_kwargs={text_embeds,time_ids})
  if CFG:
    noise_pred = noise_uncond + guidance_scale * (noise_text - noise_uncond)
  latents = scheduler.step(noise_pred, t, latents, eta/generator if accepted)
```

The pipeline helper accepts custom `timesteps` or `sigmas` only if the selected
scheduler's `set_timesteps` signature supports them. `denoising_end` slices the
schedule by a cutoff computed from `scheduler.config.num_train_timesteps`.

Recommended first Dinoml scheduler slice: exact EulerDiscrete epsilon with the
Kolors 1100-step scaled-linear table. Broader Karras-compatible swaps should be
separate scheduler candidates, reusing the scheduler matrix.

Host-control first: `set_timesteps`, custom schedule validation,
`denoising_start/end`, scheduler step index, and progress/callback state.
Compile candidates: `scale_model_input`, CFG, and one Euler step as explicit
pointwise kernels over NCHW latents.

## 9. Position, timestep, and custom math

ChatGLM positional math:

```text
rotary_pos_emb = RotaryEmbedding(kv_channels // 2)(seq_length)
rotary_pos_emb = rotary_pos_emb[position_ids]
rotary_pos_emb = rotary_pos_emb.transpose(0, 1).contiguous()
apply_rotary_pos_emb(x):
  reshape last rotary dim into pairs
  [x0*cos - x1*sin, x1*cos + x0*sin]
  flatten and concat passthrough tail
```

ChatGLM masks are lower-triangular causal masks plus left-padding masks when
padding is present. The tokenizer left-pads `input_ids`, `attention_mask`, and
`position_ids`; padded `position_ids` are 0.

UNet timestep and size conditioning:

```text
time_emb = TimestepEmbedding(Timesteps(t))
add_time_ids = [original_h, original_w, crop_top, crop_left, target_h, target_w]
time_ids_proj = Timesteps(addition_time_embed_dim=256)(flatten(add_time_ids))
add_embeds = TimestepEmbedding(concat(pooled_glm[4096], time_ids_proj[1536]))
emb = time_emb + add_embeds
```

Embedded guidance-scale conditioning is source-supported through
`get_guidance_scale_embedding` if `unet.config.time_cond_proj_dim` is not null.
It is inactive for base, tiny, and inpaint configs inspected.

## 10. Preprocessing and input packing

Prompt preprocessing:

- `ChatGLMTokenizer` wraps SentencePiece, adds `[gMASK]` and `sop` prefix
  tokens, and emits `input_ids`, `attention_mask`, and `position_ids`.
- Pipeline tokenization uses `padding="max_length"`, `max_length=256`,
  truncation, and a hard validation error above 256.
- The prompt encoder requests `output_hidden_states=True`. It uses the
  penultimate hidden state for token embeddings and the final hidden state's
  last sequence position as pooled embedding.
- Negative prompts are encoded through the same path unless
  `force_zeros_for_empty_prompt` and missing negative prompt allow zero tensors.

Image preprocessing:

- Text-to-image only samples latents or accepts caller-supplied latents.
- Img2img calls `VaeImageProcessor.preprocess(image)`, VAE-encodes images unless
  the image tensor already has 4 channels, scales latents, and applies
  scheduler `add_noise` unless `denoising_start` supplies a partially denoised
  start.
- VAE decode divides by `scaling_factor=0.13025` for base Kolors.
- Output postprocess denormalizes NCHW decoded tensors to PIL or NHWC numpy.

IP-Adapter preprocessing:

- `ip_adapter_image` uses `CLIPImageProcessor` and
  `CLIPVisionModelWithProjection`; Kolors IP-Adapter Plus uses an
  OpenAI CLIP 336 vision config.
- Precomputed `ip_adapter_image_embeds` is a list of 3D or 4D tensors. Under
  CFG the pipeline expects negative and positive image embeds concatenated in
  the same order as prompt embeddings.

## 11. Graph rewrite / lowering opportunities

### Rewrite: ChatGLM prompt encoder as optional compiled stage

Source pattern: sentencepiece/tokenizer on CPU, then ChatGLM embedding,
RMSNorm, MQA/RoPE causal attention, SwigLU MLP, final RMSNorm, hidden-state
extraction.

Replacement pattern: keep tokenization host-side; compile ChatGLM forward or
accept cached `prompt_embeds` and `pooled_prompt_embeds`.

Preconditions: fixed max prompt length <= 256, no prefix tuning, dropout 0,
no generation/KV-cache reuse, output hidden states available. Shape equations:
base QKV width `4096 + 2*2*128 = 4608`; MLP intermediate chunk
`2*13696 -> 13696`.

Failure cases: dynamic tokenizer behavior, prefix tuning `pre_seq_len`, using
KV cache for incremental generation, or relying on full 32768 sequence length.
Parity test: tiny ChatGLM prompt forward with fixed token IDs and masks,
compare penultimate and pooled outputs.

### Rewrite: UNet text projection hoist

Source pattern: `UNet2DConditionModel` projects encoder hidden states from
4096 to 2048 every UNet step.

Replacement pattern: optionally precompute projected prompt embeddings once per
request and mark the denoiser boundary as `[B,S,2048]`.

Preconditions: `encoder_hid_dim_type="text_proj"`, no IP-Adapter load that
moves/restores `encoder_hid_proj`, no live LoRA changing the text projection,
fixed prompt embeddings across timesteps.

Failure cases: runtime adapter mutation, IP-Adapter `ip_image_proj` state,
cross-attention kwargs that expect source width, or compiling a shared UNet
artifact that still owns the projection weights internally.
Parity test: one UNet step with internal projection vs hoisted projection.

### Rewrite: guarded NCHW conv island to NHWC

Source pattern: NCHW Conv2d, GroupNorm, SiLU, residual add, downsample/upsample
inside UNet/VAE/ControlNet conv regions.

Replacement pattern: NCHW boundary -> NHWC island -> NCHW boundary with OIHW
weights transformed to HWIO.

Preconditions: all channel-axis ops rewrite dim 1 to last dim; concat/chunk,
GroupNorm, VAE latent stats, and attention flatten/restore are either inside
layout-aware lowering or outside the island.

Failure cases: inpaint 9-channel concat, control residual ABI, attention token
reshape, scheduler broadcasting, or image processor contracts not rewritten.
Parity test: ResnetBlock2D and VAE decode blocks at 128/64/32 latent sizes.

### Rewrite: Euler step and CFG fusion

Source pattern: `scale_model_input`, UNet output chunk, CFG arithmetic,
`scheduler.step`.

Replacement pattern: scalar table lookup plus fused pointwise kernels over
latent maps.

Preconditions: EulerDiscrete epsilon, known sigma index, no ancestral/stochastic
branch, no guidance rescale in Kolors base, source and output layouts matched.

Failure cases: scheduler swap, custom sigmas needing different table
generation, v-pred/sample prediction, or img2img begin-index mismatch.
Parity test: one-step scheduler parity with fixed random latents and noise.

## 12. Kernel fusion candidates

Highest priority:

- UNet Conv2d + GroupNorm + SiLU ResNet blocks at Kolors/SDXL widths.
- ChatGLM RMSNorm + QKV GEMM + RoPE + causal attention + output GEMM, especially
  if Dinoml chooses to compile the text encoder instead of requiring cached
  prompt embeddings.
- ChatGLM SwigLU MLP: Linear 4096 -> 27392, chunk, SiLU*gate, Linear
  13696 -> 4096.
- UNet `encoder_hid_proj` hoist or fused projection with K/V cross-attention.
- CFG plus EulerDiscrete scheduler pointwise kernels.
- VAE decode conv/resnet/up blocks.

Medium priority:

- SDXL-style `text_time` add embedding MLP for 5632-wide input.
- UNet attention Q/K/V projection plus dense noncausal attention.
- Prompt/pooled embedding duplication and CFG concat staging.
- Guarded NHWC conv islands for UNet, VAE, and ControlNet candidates.
- Img2img VAE encode and `add_noise` fusion.

Lower priority:

- IP-Adapter Plus image projection and branch-wise image attention.
- ControlNet residual side branch and multi-control residual sums.
- PAG attention perturbation processors.
- Rare Karras scheduler swaps beyond EulerDiscrete.

## 13. Runtime staging plan

Stage 1: Parse Kolors model index and component configs. Admit base
`KolorsPipeline` with externally supplied `prompt_embeds`,
`pooled_prompt_embeds`, negative equivalents, latents, and Euler scheduler
config.

Stage 2: Load UNet and VAE weights. Implement Kolors-specific `text_time`
conditioning with 4096 pooled width and 5632 add-embedding input.

Stage 3: One UNet block parity: ResnetBlock2D and BasicTransformerBlock with
pre-projected or internal-projected ChatGLM embeddings.

Stage 4: Full UNet forward at tiny Kolors dimensions, then production
320/640/1280 widths.

Stage 5: One denoising step with EulerDiscrete 1100-step scheduler, CFG, and
fixed prompt embeddings.

Stage 6: Full Python-controlled denoising loop and VAE decode.

Stage 7: Compile or integrate ChatGLM prompt encoder. Keep tokenizer on host;
validate RMSNorm, MQA/RoPE causal attention, and SwigLU MLP.

Stage 8: Add img2img: VAE encode, strength slicing, scheduler `add_noise`, and
latent image input.

Stage 9: Add optimized attention and NHWC conv islands.

Stage 10: Separate candidate reports for IP-Adapter, ControlNet, inpaint, PAG,
and any downstream T2I/upscale/depth surfaces if real Kolors pipelines appear.

## 14. Parity and validation plan

- Config parse parity for base, tiny, and inpaint UNet/VAE/scheduler/text
  configs.
- Tokenizer parity for left padding, `[gMASK]`/`sop`, attention mask, and
  position IDs at max sequence length 256.
- ChatGLM RMSNorm, RoPE, MQA expansion, causal attention, and SwigLU unit tests.
- ChatGLM full tiny prompt encoder parity for penultimate hidden and pooled
  hidden.
- UNet `encoder_hid_proj` parity and optional projection-hoist parity.
- `text_time` added embedding parity for 4096 pooled plus six time IDs.
- ResnetBlock2D parity at 320/640/1280 channels.
- BasicTransformerBlock parity with cross-attention dim 2048 and prompt length
  256.
- Full tiny Kolors pipeline smoke using Diffusers test config.
- EulerDiscrete `set_timesteps`, `scale_model_input`, and `step` parity for
  the 1100-step schedule.
- One denoising step parity with fixed latents, prompt embeddings, and timestep.
- VAE decode parity for `[B,4,128,128] -> [B,3,1024,1024]`.
- Img2img VAE encode, strength slicing, and `add_noise` parity.
- Suggested tolerances: fp32 text/scheduler `rtol=1e-4, atol=1e-5`; fp16/bf16
  denoiser and VAE initially `rtol=2e-2, atol=2e-2`, then tighten per kernel.

## 15. Performance probes

- ChatGLM prompt encoder latency for sequence lengths 64, 128, and 256.
- Prompt projection hoist vs internal per-step projection cost.
- One Kolors UNet step by resolution, batch, dtype, and CFG mode.
- UNet conv/resnet vs attention time split at 1024 output.
- Euler scheduler and CFG overhead as separate kernels.
- VAE encode/decode throughput and `force_upcast` overhead.
- Img2img strength sweep: VAE encode + shortened denoising loop.
- Attention backend comparison: eager, PyTorch SDPA, Dinoml flash-style for
  ChatGLM causal and UNet noncausal attention.
- NCHW faithful path vs guarded NHWC conv islands.
- VRAM and temporary usage with and without ChatGLM compiled in the same graph.
- IP-Adapter Plus memory and attention overhead, because docs warn it needs
  more than 24GB VRAM.

## 16. Scope boundary and separate candidates

Separate candidate reports, not ignored:

- `kolors_chatglm_text_encoder`: tokenizer stays host-side; compile ChatGLM
  RMSNorm/MQA/RoPE/SwigLU prompt encoder and prompt-cache boundary.
- `kolors_lora_runtime_adapters`: LoRA/PEFT mutation for Kolors UNet and
  ChatGLM, with projection-hoist invalidation rules.
- `kolors_ip_adapter`: CLIP 336 image encoder, image projection, Kolors
  `encoder_hid_proj` restore path, branch-wise image K/V attention, masks and
  per-adapter scales.
- `kolors_controlnet`: official Canny/Depth/Pose configs are available, but
  class-name normalization and Kolors pipeline glue need separate source/load
  admission.
- `kolors_img2img`: real non-deprecated source pipeline with VAE encode and
  timestep slicing.
- `kolors_inpaint`: official 9-channel UNet config exists, but current source
  lacks a Kolors inpaint pipeline class; admit only after source wiring is
  identified or implemented.
- `kolors_pag`: non-deprecated PAG pipeline uses attention-processor mutation
  and altered guidance batching.
- Rare scheduler swaps: the pipeline type permits Karras-compatible schedulers,
  but first parity should be exact EulerDiscrete.

Unsupported or not found for current Kolors audit:

- Textual inversion mixin on base Kolors pipeline.
- T2I-Adapter, GLIGEN, depth2img, and upscaling Kolors-specific pipelines.

Ignored/out of scope:

- Callback mutation and interactive interrupt.
- Safety checker and NSFW filtering.
- Multi-GPU/context-parallel paths.
- XLA, NPU, MPS, Flax, and ONNX branches.
- Training, losses, dropout, and gradient checkpointing.

## 17. Final implementation checklist

- [ ] Parse `KolorsPipeline` model index and component configs.
- [ ] Preserve Kolors scheduler config: EulerDiscrete, 1100 train steps, beta_end 0.014.
- [ ] Load UNet, VAE, and optionally ChatGLM text encoder weights.
- [ ] Accept external `[B,S,4096]` prompt embeds and `[B,4096]` pooled embeds.
- [ ] Implement or explicitly hoist UNet `encoder_hid_proj` 4096 -> 2048.
- [ ] Implement Kolors `text_time` conditioning with 5632 input.
- [ ] Implement Conv2d/GroupNorm/SiLU/ResnetBlock2D parity for 320/640/1280.
- [ ] Implement UNet BasicTransformerBlock self/cross attention at cross dim 2048.
- [ ] Implement CFG concat/chunk/arithmetic.
- [ ] Implement EulerDiscrete scale/step parity for Kolors config.
- [ ] Add one-step denoising parity with cached embeddings.
- [ ] Add AutoencoderKL decode with scaling factor 0.13025.
- [ ] Add optional ChatGLM tokenizer/text encoder parity: RMSNorm, RoPE, MQA, causal attention, SwigLU.
- [ ] Add img2img VAE encode, strength slicing, and scheduler `add_noise`.
- [ ] Add guarded NHWC conv-island rewrite with axis-rewrite tests.
- [ ] Add attention provider probes for ChatGLM causal and UNet noncausal attention.
- [ ] Create separate Kolors reports or work items for IP-Adapter, ControlNet, inpaint, PAG, and LoRA/adapters.
