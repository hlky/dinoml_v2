# Diffusers OmniGen Operator and Integration Report

Candidate slug: `omnigen`

## 1. Source basis

```text
Diffusers commit/version:
  Local checkout X:/H/diffusers at b3a515080752a3ba7ca92161e25530c7f280f629.

Model id(s):
  Shitao/OmniGen-v1-diffusers
  BAAI/OmniGen-v1
  Local/cache-only related entries:
    H:/configs/OmniGen2/OmniGen2
    H:/configs/Azily/Macro-OmniGen2
    H:/configs/DFloat11/OmniGen2-transformer-DF11
    H:/configs/silveroxides/OmniGen-V1
    H:/configs/stablediffusionapi/omnigenxl-nsfw-sfw

Config sources:
  H:/configs/Shitao/OmniGen-v1-diffusers/model_index.json
  Official raw/API reads for Shitao/OmniGen-v1-diffusers:
    model_index.json
    transformer/config.json
    scheduler/scheduler_config.json
    vae/config.json
    tokenizer/tokenizer_config.json
    tokenizer/special_tokens_map.json
  Hugging Face API metadata for Shitao/OmniGen-v1-diffusers:
    repo sha 016e2f61d12a98303f6bbdf122687694d7984268
    transformer/diffusion_pytorch_model.safetensors, 7,750,667,816 bytes
    vae/diffusion_pytorch_model.safetensors, 334,643,268 bytes
  Official raw/API reads for BAAI/OmniGen-v1:
    config.json, tokenizer_config.json, special_tokens_map.json, vae/config.json
    repo sha 0c2b202e4382e1cb7952522a89d24b9cfe7f2923
    model.safetensors, 15,501,299,112 bytes
  Local cache had model_index-only or empty JSON for several related OmniGen/OmniGen2 repos.

Pipeline files inspected:
  X:/H/diffusers/src/diffusers/pipelines/omnigen/pipeline_omnigen.py
  X:/H/diffusers/src/diffusers/pipelines/omnigen/processor_omnigen.py
  X:/H/diffusers/src/diffusers/pipelines/omnigen/__init__.py

Model files inspected:
  X:/H/diffusers/src/diffusers/models/transformers/transformer_omnigen.py
  X:/H/diffusers/src/diffusers/models/autoencoders/autoencoder_kl.py
  X:/H/diffusers/src/diffusers/models/autoencoders/vae.py

Scheduler/processors/helpers inspected:
  X:/H/diffusers/src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py
  X:/H/diffusers/src/diffusers/models/attention_processor.py
  X:/H/diffusers/src/diffusers/models/embeddings.py
  X:/H/diffusers/src/diffusers/models/normalization.py
  X:/H/diffusers/src/diffusers/image_processor.py
  X:/H/diffusers/tests/pipelines/omnigen/test_pipeline_omnigen.py
  X:/H/diffusers/tests/models/transformers/test_models_transformer_omnigen.py
  X:/H/diffusers/docs/source/en/api/pipelines/omnigen.md
  X:/H/diffusers/docs/source/en/using-diffusers/omnigen.md

External component configs inspected:
  LlamaTokenizerFast tokenizer metadata from Shitao/OmniGen-v1-diffusers.
  Original BAAI/OmniGen-v1 config is Phi-3-like remote/original format, not a
  Diffusers model_index pipeline contract.

Any missing files or assumptions:
  Shitao/OmniGen-v1-diffusers official component configs were public; no gated
  retry was needed. Raw tokenizer/added_tokens.json and safetensors index paths
  returned 404 because the repo exposes tokenizer.json and single safetensors
  files instead. silveroxides/OmniGen-V1 returned repository-not-found through
  authenticated API and raw 401/404-style failures; it is not used as source
  evidence. OmniGen2 cache entries reference custom `OmniGen2Pipeline` /
  `OmniGen2Transformer2DModel` classes not present in this checkout's
  non-deprecated Diffusers `omnigen` folder, so they are separate external
  candidates, not this target. XLA/NPU/MPS/Flax/ONNX, callbacks/interrupt,
  safety, training/loss/dropout/gradient checkpointing, and multi-GPU/context
  parallel are out of scope except where shared CPU/CUDA source comments define
  parity traps.
```

## 2. Pipeline and component graph

`OmniGenPipeline` wires `OmniGenTransformer2DModel`,
`FlowMatchEulerDiscreteScheduler`, `AutoencoderKL`, and `LlamaTokenizerFast`.
The offload sequence is `transformer->vae`. There is no external CLIP/T5/Qwen
text encoder in the Diffusers v1 pipeline; token IDs and input-image latent
tokens are consumed directly by the denoising transformer.

```text
prompt + optional input_images
  -> OmniGenMultiModalProcessor:
       Llama token IDs, CFG/negative/image-CFG rows, 2D causal+image attention mask,
       position IDs, preprocessed input image tensors
  -> optional VAE encode of input images for edit/control-like prompts
  -> latent noise [B,4,H/8,W/8] source NCHW
  -> denoising loop:
       concat latent batch for CFG or text+image CFG
       OmniGenTransformer2DModel(text/image condition tokens + time token + output image patch tokens)
       CFG arithmetic or text/image CFG arithmetic
       FlowMatchEulerDiscreteScheduler.step
  -> AutoencoderKL decode(latents / scaling_factor)
  -> VaeImageProcessor postprocess
```

First-slice required components:

| Component | Class/file | Required contract |
| --- | --- | --- |
| Pipeline | `OmniGenPipeline`, `pipeline_omnigen.py` | Prompt/image preprocessing, CFG batch construction, FlowMatch loop, VAE boundary. |
| Processor | `OmniGenMultiModalProcessor`, `OmniGenCollator`, `processor_omnigen.py` | Llama tokenization, image placeholders, input-image crop/normalize, causal/image masks, position IDs. |
| Denoiser | `OmniGenTransformer2DModel`, `transformer_omnigen.py` | Patch-conv image token transformer with text embedding, input-image latent replacement, time token, RoPE, RMSNorm blocks. |
| Scheduler | `FlowMatchEulerDiscreteScheduler` | Custom sigma list plus `invert_sigmas=true`, non-stochastic Euler update. |
| VAE | `AutoencoderKL` | Decode for all image outputs; encode for any prompt using `input_images`. |
| Tokenizer | `LlamaTokenizerFast` | CPU/data path initially; special tokens and pad ID affect sequence masks. |

Separate candidate reports:

| Surface | Classes/files | Runtime delta |
| --- | --- | --- |
| `omnigen_multimodal_edit` | Same `OmniGenPipeline` plus `processor_omnigen.py` | Not a separate pipeline class, but image edit, visual reasoning, subject-preserving, and controllable generation all activate VAE encode, image placeholder token replacement, bidirectional input-image attention spans, triple CFG rows, and `img_guidance_scale`. |
| `omnigen_processor_runtime` | `OmniGenMultiModalProcessor`, `OmniGenCollator` | Token/image preprocessing, variable text lengths, left padding, dense `[B,S,S]` masks, and input-image crop/resize/normalize. |
| `omnigen_vae_codec` | `AutoencoderKL` | Standard SD-style VAE with unusual `scaling_factor=0.13025`; encode and decode are both required by the unified prompt surface. |
| `omnigen2_external` | Local cache model indexes for `OmniGen2/OmniGen2` and `Azily/Macro-OmniGen2` | Configs reference custom classes not present in this checkout's non-deprecated `omnigen` source; needs a separate source/config audit if added to Diffusers. |
| `z_image_omni` | `pipelines/z_image/pipeline_z_image_omni.py` | Separate Z-Image folder target, not OmniGen v1. |
| `omnigen_original_baai_conversion` | BAAI original `config.json`, single `model.safetensors` | Original Phi3-style checkpoint conversion/loading, separate from Diffusers component graph. |

No family-local LoRA, textual inversion, runtime adapter mutation, IP-Adapter,
ControlNet, T2I-Adapter, GLIGEN, inpaint, depth2img, or upscaling classes were
found in the non-deprecated `pipelines/omnigen` folder. Img2img/edit-like and
control-like workflows are prompt-driven through `input_images`, not distinct
pipeline classes.

## 3. Important config dimensions

Representative checkpoint/config sweep:

| Config | Pipeline/source status | Transformer | VAE | Scheduler | Notes |
| --- | --- | --- | --- | --- | --- |
| `Shitao/OmniGen-v1-diffusers` | Official Diffusers pipeline | 32 layers, hidden 3072, 32 heads, 32 KV heads, patch 2, in/out latent C=4 | AutoencoderKL z=4, scale 0.13025, sample 1024 | FlowMatch, custom sigmas, `invert_sigmas=true`, no dynamic shift | Primary source-backed target. |
| `BAAI/OmniGen-v1` | Original non-Diffusers format | Phi3-like config: 32 layers, hidden 3072, 32 heads/KV, bf16 metadata | VAE config present, older defaults omit force_upcast/quant flags | No Diffusers scheduler config | Useful conversion reference only. |
| `H:/configs/OmniGen2/OmniGen2` | Cache model_index only | Custom `OmniGen2Transformer2DModel` absent from checkout | AutoencoderKL slot | FlowMatch slot | Separate external/custom candidate. |
| `H:/configs/Azily/Macro-OmniGen2` | Cache model_index only | Same custom OmniGen2 names | AutoencoderKL slot | FlowMatch slot | Separate external/custom candidate. |
| `H:/configs/DFloat11/OmniGen2-transformer-DF11` | Empty local JSON | No usable config | N/A | N/A | Not source evidence. |

Transformer config facts for `Shitao/OmniGen-v1-diffusers`:

| Field | Value / effective default |
| --- | --- |
| `in_channels` / output channels | 4 latent channels. |
| `patch_size` | 2, implemented by Conv2d stride 2, not pipeline-level latent packing. |
| Output token count at 1024 | Latents `[B,4,128,128]` -> patch tokens `[B,4096,3072]`. |
| `hidden_size` | 3072. |
| `num_layers` | 32 `OmniGenBlock` layers. |
| `num_attention_heads` / `num_key_value_heads` | 32 / 32, head dim 96. |
| `intermediate_size` | 8192, SwiGLU-style feed-forward. |
| `vocab_size` / `pad_token_id` | 32064 / 32000. |
| RoPE | SU-scaled RoPE, `max_position_embeddings=131072`, `original_max_position_embeddings=4096`, `rope_base=10000`, `short_factor` and `long_factor` arrays length 48. |
| Absolute image pos embed | 2D sin-cos table with `pos_embed_max_size=192`, cropped by latent patch H/W. At VAE latent 128x128 and patch 2, crop is 64x64. |
| Timestep path | `Timesteps(256, flip_sin_to_cos=True, downscale_freq_shift=0)` plus two `TimestepEmbedding` MLPs: one time token and one AdaLayerNorm condition. |

VAE and tokenizer:

| Component | Config facts |
| --- | --- |
| `AutoencoderKL` | `latent_channels=4`, `block_out_channels=[128,256,512,512]`, `layers_per_block=2`, GroupNorm 32, mid attention enabled, quant/post-quant conv enabled, `force_upcast=true`, `sample_size=1024`, `scaling_factor=0.13025`, no shift/mean/std vectors. |
| `LlamaTokenizerFast` | `add_bos_token=true`, `add_eos_token=false`, vocab 32064, pad/eos token `<|endoftext|>` id 32000, special role tokens include `<|assistant|>`, `<|system|>`, `<|user|>`, `<|end|>`, and placeholder tokens. |

Scheduler:

| Field | Value / source |
| --- | --- |
| Class | `FlowMatchEulerDiscreteScheduler`. |
| Pipeline schedule input | `sigmas = linspace(1,0,num_steps+1)[:num_steps]`; `timesteps` argument can override only if scheduler supports it. |
| Config | `num_train_timesteps=1`, `shift=1.0`, `invert_sigmas=true`, `use_dynamic_shifting=false`, `base_shift=0.5`, `max_shift=1.15`, `base_image_seq_len=256`, `max_image_seq_len=4096`. |
| Effective timesteps | With custom sigmas and `invert_sigmas=true`, scheduler flips sigmas to `1 - sigmas`, uses `timesteps = sigmas * num_train_timesteps`, and appends terminal sigma `1.0`. |
| Step | Non-stochastic source default: `prev = sample + (sigma_next - sigma) * model_output`, fp32 step math, cast back to model output dtype. |

Recommended first Dinoml scheduler slice: FlowMatch Euler with custom sigmas,
`shift=1.0`, `invert_sigmas=true`, scalar timestep path, and non-stochastic
step. Dynamic shifting, stochastic sampling, per-token timesteps, and Karras/
exponential/beta sigma conversions are not active for the primary config.

## 3a. Family variation traps

- OmniGen v1 uses a single transformer as both text/image conditioning consumer
  and denoiser. Do not look for a separate text encoder output tensor.
- `input_images=None` still creates positive and negative token rows, then the
  pipeline duplicates latent input twice and applies standard CFG. With input
  images, it creates three rows: conditional, unconditional negative, and
  image-only CFG.
- The dense attention mask is `[B,S,S]`, not a simple padding mask. It combines
  causal text/time behavior with bidirectional attention inside output image
  token spans and input-image placeholder spans.
- Source latents are NCHW `[B,4,H/8,W/8]`; the transformer patchifies
  internally with Conv2d stride 2. This differs from Flux/Qwen/LongCat-style
  pipeline-level reshape/permute packing.
- `height` and `width` should be divisible by `vae_scale_factor * 2 = 16` for
  patch/VAE compatibility. Source currently warns but does not resize generated
  latent dimensions in `prepare_latents`; first Dinoml admission should guard
  divisibility.
- Input images are cropped/resized to multiples of 16 and then VAE-encoded.
  Their latent patches replace zero placeholder token IDs inside the condition
  token sequence.
- `use_input_image_size_as_output` changes output H/W after preprocessing to the
  first input image tensor's actual cropped size.
- The official Diffusers scheduler uses `invert_sigmas=true`, unlike most
  FlowMatch image reports. Validate the flipped sigma table before reusing a
  Flux/SD3 FlowMatch slice.
- `pos_embed_max_size=192` bounds latent patch grid size, so default source
  admits up to latent patch 192x192, corresponding to image 3072x3072 with VAE
  scale 8 and patch 2, subject to memory.
- Original BAAI config is Phi3-style and not directly the Diffusers
  `OmniGenTransformer2DModel` component config, even though dimensions align.
- The local OmniGen2 cache is not an OmniGen v1 variant in this checkout.

## 4. Runtime tensor contract

For 1024x1024 text-to-image, one image per prompt:

| Boundary | Tensor | Shape/layout | Notes |
| --- | --- | --- | --- |
| Prompt text | Python strings | `[B]` | Processor prepends `<|user|>...` instruction and appends assistant diffusion markers. |
| Token IDs | `input_ids` | `[2B,L_txt] int64` | Positive and negative rows for text-to-image. Left padded with id 32000. |
| Attention mask | `attention_mask` | `[2B,S,S]`, `S=L_txt+4096+1` | Dense 0/1 mask from collator; converted to additive mask `[B,1,S,S]` in transformer. |
| Position IDs | `position_ids` | `[2B,S] int64` | Left padding positions are 0; non-padding positions range through text+image+time token length. |
| Initial latents | `latents` | `[B,4,128,128]` NCHW fp32 | Sampled by `randn_tensor`; caller-provided latents accepted as-is after dtype/device cast. |
| Latent model input | `latent_model_input` | `[2B,4,128,128]` text-only CFG | `[3B,4,H/8,W/8]` when input images activate image CFG. |
| Time input | `timestep` | `[2B]` or `[3B]` | Scheduler timesteps from inverted sigma table, dtype follows scheduler. |
| Output patch tokens | internal | `[2B,4096,3072]` before final projection | Conv2d patch embed of output latent map, plus cropped 2D sin-cos position. |
| Full transformer sequence | internal | `[2B,L_txt+1+4096,3072]` | Condition tokens, one time token, then output image tokens. |
| Denoiser output | `noise_pred` | `[2B,4,128,128]` | Split into cond/uncond for CFG; same shape as latents. |
| Scheduler output | `latents` | `[B,4,128,128]` | FlowMatch Euler update. |
| VAE decode input | `latents / 0.13025` | `[B,4,128,128]` NCHW | Cast to VAE dtype before decode. |
| VAE decoded image | image tensor | `[B,3,1024,1024]` NCHW | Postprocessed to PIL/NumPy unless `output_type="latent"`. |

With `input_images`:

| Boundary | Tensor | Shape/layout | Notes |
| --- | --- | --- | --- |
| Preprocessed image | `input_pixel_values` list | each `[1,3,H_i,W_i]` NCHW in `[-1,1]` | Cropped/resized to max size and multiples of 16. |
| Input image latents | `input_img_latents` list | each `[1,4,H_i/8,W_i/8]` | VAE encode sample multiplied by 0.13025. |
| Input image patch tokens | internal | each `[1,(H_i/16)*(W_i/16),3072]` | `input_image_proj` Conv2d stride 2 plus cropped pos embed. |
| Placeholder spans | `input_image_sizes` | dict `batch -> [[start,end], ...]` | These token spans are overwritten by image patch embeddings. |
| CFG rows | condition, negative, image-only | `[3B,...]` | Output split as `cond`, `uncond`, `img_cond`. |
| Guidance arithmetic | `noise_pred` | `[B,4,H/8,W/8]` | `uncond + img_guidance_scale*(img_cond-uncond) + guidance_scale*(cond-img_cond)`. |

CPU/data-pipeline work: PIL loading, crop/resize/normalization, tokenization,
prompt placeholder parsing, left padding, dense attention/position mask
construction, final PIL/NumPy conversion. GPU/runtime work: optional input VAE
encode, output latent denoiser, CFG arithmetic, scheduler step, VAE decode.

Cacheable stages: token IDs/masks/position IDs for fixed prompts and output
size, input-image VAE latents, cropped 2D image position embeddings for fixed
latent H/W, SU-RoPE cos/sin for fixed sequence length, and scheduler sigma/
timestep tables per step count.

## 5. Operator coverage checklist

Tensor/layout ops:

- NCHW latent allocation and random normal.
- Conv2d patchify and unpatchify-like final reshape:
  `Linear -> reshape(B,H/2,W/2,2,2,C) -> permute(0,5,1,3,2,4) -> flatten`.
- Text/image token concat, left padding, placeholder token replacement, list of
  variable input image latents.
- Dense attention mask creation and additive-mask conversion:
  `(1 - mask) * finfo(dtype).min`, then `unsqueeze(1)`.
- CFG and image-CFG batch concat/split.
- FlowMatch Euler pointwise step.

Convolution/downsample/upsample ops:

- Transformer patch embedding: two Conv2d layers, `4 -> 3072`, kernel/stride
  2, one for output latents and one for input-image latents.
- AutoencoderKL encode/decode Conv2d/ResNet/GroupNorm/SiLU/downsample/upsample
  and mid attention.

GEMM/linear ops:

- Token embedding `[32064,3072]`.
- Time MLPs: `256 -> 3072 -> 3072` twice (`time_token` and `t_embedder`).
- Per block Q/K/V projections: hidden 3072, 32 heads, head dim 96, no bias.
- Attention output projection: 3072 -> 3072, no bias.
- Feed-forward: bias-free `gate_up_proj` 3072 -> 16384, chunk, SiLU gate,
  elementwise multiply, `down_proj` 8192 -> 3072.
- Final AdaLayerNorm modulation linear 3072 -> 6144 and `proj_out`
  3072 -> 16 with bias.

Attention primitives:

- 32 layers of non-dropout SDPA with additive dense mask.
- Full MHA, not GQA for official config (`num_key_value_heads=32`).
- Q/K RoPE over head dim 96 using `apply_rotary_emb(..., use_real_unbind_dim=-2)`.
- RMSNorm before attention and before MLP.

Normalization and adaptive conditioning:

- RMSNorm over hidden/head dimension with fp32 variance.
- AdaLayerNorm final norm with OmniGen/CogVideoX chunk order `shift, scale`.
- SiLU activation for time embeddings and MLP gate.

Position/timestep/custom math:

- 2D sin-cos image position embedding precomputed to max 192x192 then center
  cropped by latent patch grid.
- SU-scaled RoPE with short/long factors selected by max position ID relative
  to `original_max_position_embeddings=4096`.
- Timestep embedding from FlowMatch timesteps with no pipeline-side `/1000`
  scaling.

Scheduler/VAE/postprocessing:

- FlowMatch custom sigma table with inversion.
- VAE latent scale divide before decode; encode scale multiply for input images.
- Image postprocess through `VaeImageProcessor`.

## 6. Denoiser/model breakdown

`OmniGenTransformer2DModel.forward`:

```text
hidden_states [B,4,H,W] NCHW latent map
  -> output_image_proj Conv2d(k=2,s=2) + cropped 2D sin-cos pos
  -> output image tokens [B,(H/2)*(W/2),3072]
timestep [B]
  -> Timesteps(256) -> time_token [B,1,3072]
  -> TimestepEmbedding -> temb [B,3072]
input_ids + input_img_latents + input_image_sizes
  -> token embeddings [B,L,3072]
  -> input-image VAE latents through input_image_proj Conv2d(k=2,s=2)
  -> replace placeholder spans with image patch embeddings
concat condition tokens, time token, output image tokens
  -> dense mask conversion and SU-RoPE
  -> 32 x OmniGenBlock
  -> RMSNorm
  -> keep final output-image token span
  -> AdaLayerNorm(temb) -> Linear(3072 -> 16)
  -> unpatchify to [B,4,H,W]
```

`OmniGenBlock`:

```text
RMSNorm
  -> Q/K/V projections from same normalized hidden states
  -> reshape [B,heads,S,96]
  -> RoPE on Q/K
  -> scaled_dot_product_attention(Q,K,V, additive dense mask)
  -> output Linear
  -> residual add
RMSNorm
  -> gate_up Linear, chunk gate/up
  -> SiLU(gate) * up
  -> down Linear
  -> residual add
```

Active config branches:

- Official config uses full MHA (`kv_heads=heads=32`), no QKV/output bias, and
  no dropout path in the OmniGen processor.
- Input-image replacement is active only when prompt placeholders and
  `input_images` are supplied.
- Gradient checkpointing and training branches are inactive for inference.

## 7. Attention requirements

Primary implementation: `OmniGenAttnProcessor2_0` in
`transformer_omnigen.py`, using `torch.nn.functional.scaled_dot_product_attention`
directly. `attention_dispatch.py` is not the active source path for OmniGen v1.

Required behavior:

- Self-attention over the concatenated sequence `[condition tokens, time token,
  output image tokens]`.
- Official base: 32 heads, 32 KV heads, head dim 96, hidden size 3072.
- Additive dense mask with shape `[B,1,S,S]`. Text/time tokens are causal;
  output image tokens attend bidirectionally to the full sequence; input-image
  placeholder spans are made bidirectional inside each image span.
- RoPE on Q/K after projection and before attention, using real-unbind dim -2.
- No cross-attention module, added-KV branch, IP-Adapter branch, or joint
  text/image split processor in this target.

Flash/provider feasibility:

- Text-to-image sequence length at 1024 is roughly `L_txt + 1 + 4096`; head dim
  96 is flash-friendly in principle, but the dense mixed causal/bidirectional
  mask is the main blocker for naive flash.
- A Dinoml flash-style provider could be valid for a stricter mask pattern if
  it supports dense additive masks or a custom block mask. Otherwise native SDPA
  or an eager masked attention provider defines first parity.
- Input-image prompts increase `L_txt` by VAE-patch placeholder spans and
  require exact mask preservation. Do not collapse to a padding-only mask.
- Q/K RoPE and RMSNorm should remain explicit pre-attention ops until a fused
  provider admits the exact SU-RoPE, head dim, dtype, and mask pattern.

## 8. Scheduler and denoising-loop contract

Pipeline schedule setup:

```text
sigmas = np.linspace(1, 0, num_inference_steps + 1)[:num_inference_steps]
retrieve_timesteps(scheduler, num_inference_steps, device, timesteps, sigmas=sigmas)
```

For the official scheduler config:

- `shift=1.0` leaves supplied sigmas unchanged before inversion.
- `use_dynamic_shifting=false`, so no `mu` is used.
- `invert_sigmas=true` converts the table to `1 - sigmas`, recomputes
  `timesteps = sigmas * num_train_timesteps`, and appends terminal sigma `1.0`.
- With 50 steps, the denoising timesteps ascend from 0 toward 0.98 because of
  inversion, even though the pipeline supplies descending raw sigmas.

Per-step loop:

```text
latent_model_input = cat([latents] * 2) for text-to-image
latent_model_input = cat([latents] * 3) for input-image prompts
timestep = t.expand(latent_model_input.shape[0])
noise_pred = transformer(latent_model_input, timestep, token/mask/image state)
if text-only:
  cond, uncond = split(noise_pred, 2)
  noise_pred = uncond + guidance_scale * (cond - uncond)
else:
  cond, uncond, img_cond = split(noise_pred, 3)
  noise_pred = uncond + img_guidance_scale * (img_cond - uncond)
                      + guidance_scale * (cond - img_cond)
latents = scheduler.step(noise_pred, t, latents)
```

Keep loop iteration, callback mutation, and progress state as host control
first. Compile the denoiser step and pointwise CFG/scheduler update after
one-step parity is stable.

## 9. Position, timestep, and custom math

Custom math to reproduce:

- Prompt wrapper:
  `<|user|>\nGenerate an image according to the following instructions\n...<|end|>\n<|assistant|>\n<|diffusion|>`.
- Placeholder parsing for `<|image_i|>` inside `<img>...</img>` wrappers; image
  IDs must start at 1 and be continuous.
- Input image token count is `(H_i/16)*(W_i/16)` because VAE scale 8 and
  transformer patch size 2 combine.
- Output image token count is `(height/16)*(width/16)`.
- Dense attention mask: causal among text/time tokens, all-ones rows for output
  image tokens, and bidirectional sub-blocks for input-image placeholder spans.
- 2D sin-cos pos embed crop from a 192x192 table with center crop.
- SU-RoPE recomputes inverse frequencies using short or long factor arrays
  based on maximum position ID, then applies the scale
  `sqrt(1 + log(max/original) / log(original))` only when max/original > 1.
- Final unpatchify order:
  `reshape(B,hp,wp,p,p,C) -> permute(0,5,1,3,2,4) -> flatten`.
- VAE scale: encode input images as `sample * 0.13025`; decode generated
  latents as `latents / 0.13025`.

Precomputable: image crop positional embeddings for fixed H/W, scheduler
tables, prompt token/mask/position IDs for fixed prompt and output size, input
image latents for repeated edit/control prompts. Dynamic: prompt length, number
and sizes of input images, CFG mode, output size, timestep, and max input image
size.

## 10. Preprocessing and input packing

Text-to-image:

- Processor wraps the prompt with fixed role/instruction text.
- Tokenizer is Llama-style and adds BOS by default.
- Processor always builds negative prompt rows using a fixed long negative
  prompt.
- Collator left-pads token IDs to the batch maximum and builds dense
  sequence masks for each CFG row.

Input-image prompts:

- Each input PIL/path image is RGB-converted, repeatedly box-downsampled if
  extremely large, bicubic-scaled to `max_input_image_size`, upscaled if the
  shorter side is below 16, center-cropped by modulo remainder to multiples of
  16, converted to tensor, and normalized by mean/std 0.5.
- Text chunks around `<|image_i|>` placeholders are tokenized; placeholder
  spans are filled with zero IDs whose length equals image latent patch count.
- VAE encode uses the posterior sample, not mode, then multiplies by
  `vae.config.scaling_factor`.
- `use_input_image_size_as_output=True` uses the first preprocessed input
  image H/W as the generated output H/W.

Postprocessing:

- `output_type="latent"` returns latent maps `[B,4,H/8,W/8]`, not packed
  tokens.
- Otherwise VAE decode returns NCHW image tensor and `VaeImageProcessor`
  performs output conversion.

## 11. Graph rewrite / lowering opportunities

### Rewrite: OmniGen patchify/unpatchify

Source pattern: Conv2d patch embeddings with kernel/stride 2 for output latents
and input-image latents, plus final Linear-to-pixel unpatchify.

Replacement: explicit `patch_embed_conv2d` and `unpatchify2x2_nchw` operators,
or canonical Conv2d/GEMM plus reshape/permute/flatten.

Preconditions: NCHW latent tensors, `patch_size=2`, latent H/W divisible by 2,
`in_channels=4`, final projection width `patch_size^2 * channels = 16`.

Layout constraints: source is NCHW. NHWC is safe only inside a fully controlled
patch/VAE island with Conv2d weight transform and exact unpatchify order rewrite.

Failure cases: non-divisible height/width, future patch sizes, user-supplied
latents with mismatched shape.

Parity test: random latent patchify/unpatchify tensor shape tests and full
transformer output shape parity for 16, 512, and 1024 image sizes.

### Rewrite: dense OmniGen attention mask

Source pattern: processor builds dense 0/1 `[B,S,S]` mask, transformer converts
to additive min-dtype mask and calls SDPA.

Replacement: represent mask as structured block mask metadata when possible:
left padding, causal text/time prefix, full image rows, bidirectional input
image placeholder spans.

Preconditions: mask comes from `OmniGenCollator`, known text length, known
output image token count, known placeholder spans.

Failure cases: arbitrary user-supplied masks, changed processor semantics,
provider lacking mixed causal/bidirectional support.

Parity test: compare dense masks and attention outputs for text-only, one input
image, and two input image prompts.

### Rewrite: OmniGen CFG/image-CFG

Source pattern: duplicate latent batch 2x or 3x, run transformer once, split,
and combine.

Replacement: host-visible CFG mode plus fused pointwise combine kernel; later
consider separate transformer calls only if memory pressure beats batching.

Preconditions: fixed CFG row order from collator, identical latent input rows,
matching transformer output shapes.

Failure cases: custom callbacks mutating latents, nonstandard collator row
order, future negative/image CFG variants.

Parity test: one random `noise_pred` batch for both text-only and image-CFG
formulas.

### Rewrite: FlowMatch inverted sigma slice

Source pattern: custom `linspace(1,0)` sigmas passed to scheduler with
`invert_sigmas=true`, then pointwise Euler update.

Replacement: precomputed scheduler table plus fused
`sample + (sigma_next - sigma) * model_output`.

Preconditions: `shift=1`, non-stochastic, no dynamic shifting, no per-token
timesteps.

Failure cases: custom timesteps, stochastic sampling, future scheduler configs
with Karras/beta/exponential conversion.

Parity test: table parity for 1, 2, 50 steps and one-step fp32 update parity.

## 12. Kernel fusion candidates

Highest priority:

- Masked transformer attention provider for OmniGen's dense mixed causal/image
  mask, head dim 96, and long sequences.
- RMSNorm + QKV + RoPE + SDPA prelude fusion where mask/provider support is
  explicit.
- SwiGLU feed-forward fusion: bias-free gate/up projection, SiLU, multiply,
  down projection.
- Final AdaLayerNorm + projection + unpatchify epilogue.
- CFG/image-CFG and FlowMatch Euler step pointwise fusion.

Medium priority:

- Patch Conv2d + pos-embed add for output and input-image latents.
- Structured mask generation from token lengths and image spans.
- SU-RoPE cos/sin cache keyed by position length and short/long factor path.
- VAE encode/decode Conv2d/GroupNorm/SiLU/mid-attention islands.

Lower priority:

- Full tokenizer/preprocessor compilation; keep CPU/data path first.
- VAE tiling/slicing policy, exposed only through deprecated pipeline wrappers
  that forward to VAE methods.
- Original BAAI checkpoint conversion and OmniGen2 custom surfaces.

Layout candidates:

- Transformer core should stay token-major `[B,S,C]`.
- VAE and patch Conv2d source tensors are NCHW. NHWC can be explored only for
  local Conv/GroupNorm/resample islands with channel-axis rewrites and weight
  transforms.
- Protect dense mask, position IDs, sequence concat, and unpatchify flatten
  order with a conceptual `no_layout_translation()` guard until layout-aware
  rewrites are validated.

## 13. Runtime staging plan

Stage 1: Parse `Shitao/OmniGen-v1-diffusers` model index and component configs.
Use the existing Diffusers dummy test dimensions for smoke and official config
for real shape planning.

Stage 2: Reproduce `OmniGenMultiModalProcessor` outputs for text-only prompts:
token IDs, dense masks, position IDs, CFG row order.

Stage 3: Implement one denoiser-step artifact with external precomputed
`input_ids`, `attention_mask`, and `position_ids`; no input images first.

Stage 4: Add one `OmniGenBlock` parity with RMSNorm, RoPE, dense masked SDPA,
and SwiGLU MLP.

Stage 5: Compile full dummy and then official `OmniGenTransformer2DModel` for a
small output size and 1024 shape planning.

Stage 6: Add FlowMatch inverted-sigma table and one-step scheduler parity.

Stage 7: Add text-only CFG combine and short denoising-loop parity with VAE
decode.

Stage 8: Admit multimodal input images: VAE encode, placeholder replacement,
input-image bidirectional mask spans, and image-CFG triple combine.

Stage 9: Optimize attention/norm/MLP and VAE/patch Conv2d islands; open
separate OmniGen2/original-conversion work only after v1 is stable.

First Dinoml admission recommendation: `omnigen_v1_denoiser_step_external_tokens`,
with inputs `latents [B,4,H/8,W/8]`, `timestep [B_cfg]`, `input_ids [B_cfg,L]`,
`attention_mask [B_cfg,S,S]`, `position_ids [B_cfg,S]`, and initially an empty
input-image latent list. Output is latent derivative `[B_cfg,4,H/8,W/8]`. Keep
processor, scheduler loop, CFG combine, input-image encode, and VAE decode
outside the first compiled artifact until one-block/full-denoiser parity is
boring.

## 14. Parity and validation plan

- Config parse tests for Shitao Diffusers component configs and BAAI original
  conversion-reference config.
- Processor parity for text-only prompt: wrapped text, token IDs, left padding,
  attention mask, position IDs, CFG row order.
- Processor parity for one and two input images: crop/resize dimensions,
  placeholder spans, dense mask sub-blocks, and `use_input_image_size_as_output`.
- VAE encode parity for input images: posterior sample with fixed generator if
  possible, scaling by 0.13025.
- Patch embedding and cropped pos-embed parity for `[B,4,2h,2w]` latent maps.
- SU-RoPE parity below and above `original_max_position_embeddings=4096`.
- Attention processor parity with dense masks, fp32 first, then bf16.
- One `OmniGenBlock` parity, then full dummy transformer parity against
  `tests/models/transformers/test_models_transformer_omnigen.py`.
- Official-config denoiser step parity at a reduced resolution, then 1024 shape
  smoke if memory allows.
- Scheduler table parity for `invert_sigmas=true` and one-step update parity.
- CFG/image-CFG arithmetic parity.
- VAE decode parity for `[1,4,128,128]` and end-to-end short-loop smoke.
- Suggested tolerances: scheduler fp32 `rtol=1e-5, atol=1e-6`; transformer fp32
  `rtol=1e-4, atol=1e-5`; bf16 initially `rtol=2e-2, atol=2e-2`.

## 15. Performance probes

- Denoiser step by output image size: 512, 768, 1024, and input-image prompt
  lengths.
- Attention backend comparison: native SDPA/eager dense mask versus structured
  mask provider/flash-style candidate.
- Memory impact of dense `[B,S,S]` masks at 1024 and with one/two input images.
- Per-block split: QKV/RoPE/attention, FFN, RMSNorm, and residual overhead.
- CFG mode comparison: 2-row text CFG versus 3-row image CFG.
- Input-image VAE encode and patch-token insertion overhead by
  `max_input_image_size` 256/512/1024, matching docs' memory-sensitivity note.
- VAE decode throughput for 1024 and non-square multiples of 16.
- Scheduler/CFG pointwise overhead versus denoiser time.
- VRAM and temporary/workspace usage for bf16 and fp32 fallback paths.

## 16. Scope boundary and separate candidates

Separate candidate reports, not ignored:

- `omnigen_multimodal_edit`: prompt-driven image edit, controllable generation,
  subject/object preservation, and visual reasoning through `input_images`.
- `omnigen_processor_runtime`: CPU/data processor, dense mask builder, and
  tokenizer/special-token contract.
- `omnigen_vae_codec`: AutoencoderKL encode/decode with scale 0.13025 and
  tiling/slicing policy.
- `omnigen_original_baai_conversion`: original Phi3-style single-file model
  and conversion/key mapping to Diffusers components.
- `omnigen2_external`: cached OmniGen2 model indexes with custom classes absent
  from this checkout.
- `z_image_omni`: separate Z-Image pipeline file containing "omni" in its name.
- `scheduler_flowmatch_inverted`: reusable FlowMatch Euler `invert_sigmas=true`
  slice, distinct from SD3/Flux static/dynamic-shift cases.

Genuinely ignored/out of scope for this audit:

- Multi-GPU/context parallel.
- Callback mutation and interactive interrupt.
- XLA, NPU, MPS, Flax, and ONNX branches.
- Safety checker and NSFW filtering.
- Training, losses, dropout, gradient checkpointing.
- LoRA, textual inversion, runtime adapters, IP-Adapter, ControlNet,
  T2I-Adapter, GLIGEN, inpaint, depth2img, and upscaling: no active
  family-local implementation was found in the non-deprecated `omnigen` folder.

Blockers / validation notes:

- Official Shitao v1 Diffusers configs are accessible and not gated.
- `silveroxides/OmniGen-V1` is unavailable through API/raw reads and should not
  be used as evidence.
- Local OmniGen2 cache has only model indexes and points to classes missing from
  the inspected checkout; do not infer OmniGen2 operator contracts from v1.
- Source warns but does not force output H/W divisibility by 16 before latent
  allocation; Dinoml admission should reject or normalize non-divisible shapes
  explicitly rather than silently changing semantics.

## 17. Final implementation checklist

- [ ] Parse Shitao OmniGen v1 Diffusers component configs.
- [ ] Accept external `input_ids`, dense `attention_mask`, and `position_ids`.
- [ ] Implement text-only processor parity for wrapped prompts and CFG rows.
- [ ] Implement OmniGen 2D sin-cos pos-embed crop.
- [ ] Implement Conv2d patch embedding and final NCHW unpatchify.
- [ ] Implement SU-scaled RoPE parity.
- [ ] Implement RMSNorm and final AdaLayerNorm OmniGen chunk order.
- [ ] Implement `OmniGenAttnProcessor2_0` dense masked SDPA fallback parity.
- [ ] Implement `OmniGenBlock` SwiGLU/residual path.
- [ ] Add full dummy transformer parity, then official-config denoiser-step parity.
- [ ] Implement FlowMatch Euler custom-sigma `invert_sigmas=true` scheduler slice.
- [ ] Add text-only CFG and image-CFG arithmetic kernels.
- [ ] Add AutoencoderKL decode boundary and input-image encode boundary.
- [ ] Add multimodal placeholder/image-token replacement parity.
- [ ] Benchmark dense-mask attention, denoiser step, VAE encode/decode, and CFG/scheduler overhead.
- [ ] Open separate follow-ups for multimodal edit/control-like prompts, processor runtime, VAE codec, original conversion, and OmniGen2 external configs.
