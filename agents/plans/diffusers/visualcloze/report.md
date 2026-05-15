# Diffusers VisualCloze Operator and Integration Report

## 1. Source basis

```text
Diffusers commit/version:
  Local checkout diffusers at b3a515080752a3ba7ca92161e25530c7f280f629.

Model id(s):
  VisualCloze/VisualClozePipeline-384
  VisualCloze/VisualClozePipeline-512
  Both model indexes name black-forest-labs/FLUX.1-Fill-dev as the source base.

Config sources:
  H:/configs/VisualCloze/VisualClozePipeline-384/model_index.json
  H:/configs/VisualCloze/VisualClozePipeline-384/transformer/config.json
  H:/configs/VisualCloze/VisualClozePipeline-384/scheduler/scheduler_config.json
  H:/configs/VisualCloze/VisualClozePipeline-384/vae/config.json
  H:/configs/VisualCloze/VisualClozePipeline-384/text_encoder/config.json
  H:/configs/VisualCloze/VisualClozePipeline-384/text_encoder_2/config.json
  H:/configs/VisualCloze/VisualClozePipeline-384/tokenizer/tokenizer_config.json
  H:/configs/VisualCloze/VisualClozePipeline-384/tokenizer_2/tokenizer_config.json
  H:/configs/VisualCloze/VisualClozePipeline-384/transformer/diffusion_pytorch_model.safetensors.index.json
  H:/configs/VisualCloze/VisualClozePipeline-384/text_encoder_2/model.safetensors.index.json
  H:/configs/VisualCloze/VisualClozePipeline-512/... same component config set
  H:/configs/VisualCloze/VisualCloze/model_index.json exists but is an empty placeholder.
  The 384/512 JSON configs and index metadata were fetched with huggingface_hub.

Pipeline files inspected:
  diffusers/src/diffusers/pipelines/visualcloze/pipeline_visualcloze_combined.py
  diffusers/src/diffusers/pipelines/visualcloze/pipeline_visualcloze_generation.py
  diffusers/src/diffusers/pipelines/visualcloze/visualcloze_utils.py
  diffusers/src/diffusers/pipelines/flux/pipeline_flux_fill.py

Model files inspected:
  diffusers/src/diffusers/models/transformers/transformer_flux.py
  diffusers/src/diffusers/models/attention_dispatch.py
  diffusers/src/diffusers/models/attention.py
  diffusers/src/diffusers/models/embeddings.py
  diffusers/src/diffusers/models/normalization.py
  diffusers/src/diffusers/models/autoencoders/autoencoder_kl.py

Scheduler/processors/helpers inspected:
  diffusers/src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py
  diffusers/src/diffusers/image_processor.py
  diffusers/docs/source/en/api/pipelines/visualcloze.md
  diffusers/tests/pipelines/visualcloze/test_pipeline_visualcloze_generation.py
  diffusers/tests/pipelines/visualcloze/test_pipeline_visualcloze_combined.py

External component configs inspected:
  CLIPTextModel / CLIPTokenizer configs bundled in the VisualCloze repos.
  T5EncoderModel / T5TokenizerFast configs bundled in the VisualCloze repos.

Any missing files or assumptions:
  The report covers non-deprecated VisualClozePipeline and VisualClozeGenerationPipeline CPU/CUDA inference behavior.
  It treats FluxFillPipeline upsampling as an inherited FLUX-fill stage and points to the Flux report for broader FLUX variants.
  Safety, training, callbacks, XLA/NPU/MPS/Flax/ONNX, and multi-GPU/context parallel paths are out of scope.
```

## 2. Pipeline and component graph

`VisualClozePipeline` is a two-stage wrapper. `__init__` registers one shared component set, then constructs `VisualClozeGenerationPipeline` and `FluxFillPipeline` from the same VAE, text encoders, tokenizer pair, transformer, and scheduler. The combined `__call__` first runs the custom VisualCloze generation stage. If `upsampling_strength == 0`, it returns those cropped stage-1 results. Otherwise it turns each generated target into a full-white inpaint mask and runs FLUX Fill SDEdit-style upsampling.

```text
visual in-context image grid + task/content prompts
  -> VisualClozeProcessor resize/crop/normalize + target masks + layout prompt
  -> CLIPTokenizer/CLIPTextModel pooled prompt embeds [B,768]
  -> T5TokenizerFast/T5EncoderModel token embeds [B,L,4096]
  -> VAE encode every row after horizontal concatenation
  -> pack 2x2 latent tiles and target masks
  -> denoising loop: FluxTransformer2DModel + embedded guidance + FlowMatchEulerDiscreteScheduler
  -> unpack target row latents
  -> AutoencoderKL decode + crop target image positions
  -> optional FluxFillPipeline upsampling/inpainting over each target image
```

Required components for the selected runtime scope:

| Component | Class | Source anchor | Required first slice |
|---|---|---|---|
| `scheduler` | `FlowMatchEulerDiscreteScheduler` | `pipeline_visualcloze_generation.py:159`, `scheduling_flow_match_euler_discrete.py:48` | yes |
| `vae` | `AutoencoderKL` | `pipeline_visualcloze_generation.py:160` | encode and decode for generation; encode/decode again for upsampling |
| `text_encoder` | `CLIPTextModel` | `pipeline_visualcloze_generation.py:161` | pooled prompt conditioning |
| `tokenizer` | `CLIPTokenizer` | `pipeline_visualcloze_generation.py:162` | CPU text preprocessing |
| `text_encoder_2` | `T5EncoderModel` | `pipeline_visualcloze_generation.py:163` | token prompt conditioning |
| `tokenizer_2` | `T5TokenizerFast` | `pipeline_visualcloze_generation.py:164` | CPU text preprocessing |
| `transformer` | `FluxTransformer2DModel` | `pipeline_visualcloze_generation.py:165`, `transformer_flux.py:525` | denoiser |
| `image_processor` | `VisualClozeProcessor` | `visualcloze_utils.py:22` | CPU/PIL/grid preparation plus postprocess |

No optional components are declared by VisualCloze. The wrapper and generation pipeline inherit `FluxLoraLoaderMixin`, `TextualInversionLoaderMixin`, and `FromSingleFileMixin`.

Separate candidate reports:

| Surface | VisualCloze status | Class/file anchors | Candidate slug/order |
|---|---|---|---|
| LoRA | supported through inherited FLUX LoRA loader on transformer/text encoders | `FluxLoraLoaderMixin` in both VisualCloze pipeline classes | `visualcloze_lora_textual_inversion`, after base |
| Textual inversion | supported for both tokenizers through inherited mixin and `maybe_convert_prompt` | `pipeline_visualcloze_generation.py:207`, `:254` | same as LoRA |
| Runtime adapter mutation | PEFT LoRA scale can arrive through `joint_attention_kwargs["scale"]` | `pipeline_visualcloze_generation.py:327` | same as LoRA |
| IP-Adapter | not exposed by VisualCloze pipelines; `FluxTransformer2DModel` has an inactive IP branch when `joint_attention_kwargs` carries image embeds and the model has an encoder projection | `transformer_flux.py:709`, `transformer_flux.py:142` | `flux_ip_adapter`, not VisualCloze base |
| ControlNet | not a VisualCloze pipeline component; transformer accepts optional residual samples from other FLUX control pipelines | `transformer_flux.py:648`, `:734`, `:766` | `flux_variants_control` |
| T2I-Adapter | not present in VisualCloze source | none in target files | no VisualCloze candidate unless a future pipeline adds it |
| GLIGEN | not present in VisualCloze/FLUX Fill source | none in target files | no base candidate |
| img2img | stage-2 upsampling is an image-to-image SDEdit/inpaint style `FluxFillPipeline` call | `pipeline_visualcloze_combined.py:343`, `pipeline_flux_fill.py:754` | included only as upsampling boundary; broader FLUX img2img separate |
| inpaint | stage-2 uses full-white masks to repaint each generated target | `pipeline_visualcloze_combined.py:323`, `:343` | included as upsampling boundary |
| depth2img | task examples can use depth images as ordinary grid cells; no depth-specific encoder | docs examples | no new model ops beyond image grid |
| upscaling | built into combined pipeline through `FluxFillPipeline` | `pipeline_visualcloze_combined.py:160`, `:343` | `visualcloze_upsampling_flux_fill`, after stage-1 |

## 3. Important config dimensions

Checkpoint sweep:

| Model id | Pipeline class | Diffusers config version | Generation resize policy | Transformer | Scheduler | VAE |
|---|---|---:|---|---|---|---|
| `VisualCloze/VisualClozePipeline-384` | `VisualClozePipeline` | `0.33.0.dev0` | default constructor arg `resolution=384`; not serialized, caller must pass it | FLUX Fill-style VisualCloze transformer | FlowMatch Euler, dynamic shift | FLUX AutoencoderKL |
| `VisualCloze/VisualClozePipeline-512` | `VisualClozePipeline` | `0.34.0.dev0` | caller should pass `resolution=512`; topology unchanged | same as 384 | same as 384 | same as 384 |

Transformer config facts:

| Field | 384 | 512 | Source |
|---|---:|---:|---|
| `in_channels` | 384 | 384 | transformer config |
| `out_channels` | 64 | 64 | transformer config |
| `patch_size` | 1 | 1 | transformer config |
| hidden size | 3072 | 3072 | inferred from `24 * 128` in config/source |
| heads / head dim | 24 / 128 | 24 / 128 | transformer config |
| dual-stream blocks | 19 | 19 | transformer config |
| single-stream blocks | 38 | 38 | transformer config |
| text joint dim | 4096 | 4096 | transformer config |
| pooled projection dim | 768 | 768 | transformer config |
| RoPE axes dims | `[16, 56, 56]` | `[16, 56, 56]` | transformer config |
| guidance embeddings | true | true | transformer config |
| transformer index total size | 23,804,782,720 bytes, 1160 tensors, 3 shards | same | safetensors index metadata |

VAE config facts:

| Field | Value | Source |
|---|---:|---|
| class | `AutoencoderKL` | VAE config |
| `latent_channels` | 16 | VAE config |
| `block_out_channels` | `[128, 256, 512, 512]` | VAE config |
| VAE scale factor | 8 | inferred from 4 block stages in pipeline source |
| `sample_size` | 1024 | VAE config |
| `scaling_factor` / `shift_factor` | `0.3611` / `0.1159` | VAE config |
| `force_upcast` | true | VAE config |
| quant/post-quant conv | false / false | VAE config |

Text encoder facts:

| Component | Shape facts | Source |
|---|---|---|
| CLIP | 12 layers, hidden 768, 12 heads, intermediate 3072, max positions 77, pooled projection 768, bf16 weights | `text_encoder/config.json` |
| T5 encoder | 24 layers, `d_model=4096`, `d_ff=10240`, 64 heads, gated GELU feed-forward, bf16 weights, max tokenization length capped by pipeline at 512 | `text_encoder_2/config.json`, tokenizer config, pipeline check |
| T5 index | 9,524,621,312 bytes, 219 tensors, 2 shards | safetensors index metadata |

Scheduler facts:

| Field | Value | Source |
|---|---:|---|
| class | `FlowMatchEulerDiscreteScheduler` | scheduler config |
| `num_train_timesteps` | 1000 | scheduler config |
| `shift` | 3.0 | scheduler config |
| dynamic shifting | true | scheduler config |
| `base_image_seq_len` / `max_image_seq_len` | 256 / 4096 | scheduler config |
| `base_shift` / `max_shift` | 0.5 / 1.15 | scheduler config |
| `time_shift_type` | exponential | scheduler config |
| Karras/exponential/beta/invert sigmas | all false | scheduler config |
| recommended first Dinoml scheduler slice | flow-match Euler with custom sigmas, dynamic exponential shift, deterministic `prev = sample + dt * model_output` | source plus config |

Guidance modes:

| Mode | Required? | Contract |
|---|---|---|
| Embedded FLUX guidance | yes | if `transformer.config.guidance_embeds`, make a `[B]` float32 tensor filled with `guidance_scale`; transformer embeds it with timestep Fourier features |
| True classifier-free guidance | no | VisualCloze does not create unconditional prompt embeddings or duplicate the denoiser batch for CFG arithmetic |
| Guidance rescale / skip-layer guidance | no | not present in VisualCloze call path |

## 3a. Family variation traps

- The public docs describe broad universal image tasks, but Diffusers implements them as ordinary visual grid conditioning. Depth, edge, mask, pose, restoration, relighting, and try-on examples are image cells plus text, not separate encoders or control branches.
- The model is not a UNet. It is a FLUX MMDiT transformer over packed latent tokens.
- The generation transformer input channel count is 384 because each token concatenates noisy latents `[64]`, masked-image latents `[64]`, and packed mask features `[256]`.
- `resolution` is a constructor argument and is not serialized by the tests; loading with the wrong value changes preprocessing and memory use.
- Stage 1 resizes each image cell to approximately `resolution^2` area and a multiple of 16 before row-wise concatenation. Stage 2 upsampling uses requested `height`/`width` rounded down by FLUX Fill preprocessing to multiples of `vae_scale_factor * 2 = 16`.
- The stage-1 dynamic scheduler shift `mu` is computed from `processor_output["image_size"][0]`; heterogeneous batch layouts may not get per-sample shift parity.
- Source tensors are NCHW image/latent maps until FLUX packs 2x2 latent tiles into `[B,N,C]` tokens. A layout pass must guard VAE/image processor boundaries and rewrite mask/pack axes exactly.
- The attention sequence contains text tokens first and image tokens second for RoPE. `txt_ids` are zeros, `img_ids` carry row index and patch h/w.
- The target regions are cropped after VAE decode from the last query row only. Multi-target query rows produce a list of images per sample.
- The non-deprecated processor is `FluxAttnProcessor` in `transformer_flux.py`; deprecated wrappers in `attention_processor.py` forward to it and should not define parity.
- Combined pipeline upsampling reuses the same scheduler object as generation, but `set_timesteps` is called separately by each stage, resetting scheduler state.

## 4. Runtime tensor contract

CPU/data-pipeline inputs:

| Input | Contract |
|---|---|
| `task_prompt` | str or list[str], required |
| `content_prompt` | str, list[str], or None per sample |
| `image` | for single sample, `List[List[PIL/tensor/ndarray/None]]`; rows are in-context examples plus final query row; every non-query row must have no `None`; query row must have at least one `None` target |
| `max_sequence_length` | <= 512 |
| `upsampling_strength` | combined pipeline only; 0 returns stage-1 output, >0 runs FluxFill upsampling |

Prompt boundary:

```text
layout_prompt = "A grid layout with R rows and C columns, ..."
prompt = layout_prompt + task_prompt + "The last image of the last row depicts: {content_prompt}"
CLIPTokenizer -> CLIPTextModel.pooler_output -> pooled_prompt_embeds [B*num_images,768]
T5TokenizerFast -> T5EncoderModel[0] -> prompt_embeds [B*num_images,L,4096], L <= 512
text_ids = zeros [L,3]
```

Stage-1 image and latent boundary:

```text
PIL/np/torch images
  -> resize/crop each cell to area about resolution^2, dimensions multiple of 16
  -> tensor [1,3,H_i,W_i], normalized to [-1,1]
  -> row concat on width: [1,3,H_row,sum W_row]
  -> AutoencoderKL.encode sample/mode: [1,16,H_row/8,sum W_row/8]
  -> scale: (latent - shift_factor) * scaling_factor
  -> pack 2x2: [1,(H_row/16)*(sum W_row/16),64]
  -> concat rows on sequence dim
```

Mask packing:

```text
mask cells [1,1,H,W], 1 only for target positions in the query row
  -> row concat [1,1,H_row,sum W_row]
  -> reshape into 8x8 latent cells: [1,64,H_row/8,sum W_row/8]
  -> pack 2x2: [1,(H_row/16)*(sum W_row/16),256]
masked_image_latents = cat(masked_latents[64], mask[256], dim=-1) -> [B,N,320]
latent_model_input = cat(latents[64], masked_image_latents[320], dim=-1) -> [B,N,384]
```

Denoiser step:

| Tensor | Shape | Layout | Notes |
|---|---|---|---|
| `hidden_states` | `[B,N_img,384]` | token-major | concatenated noisy latents, masked latents, mask |
| `timestep` | `[B]` | scalar batch | pipeline passes `t / 1000`; model multiplies by 1000 internally |
| `guidance` | `[B]` or None | scalar batch | present for official configs |
| `pooled_projections` | `[B,768]` | dense | CLIP pooled |
| `encoder_hidden_states` | `[B,L,4096]` | dense sequence | T5 embeddings |
| `txt_ids` | `[L,3]` | positions | all zeros |
| `img_ids` | `[N_img,3]` | positions | row index, patch h, patch w |
| output | `[B,N_img,64]` | token-major | velocity/noise prediction for packed latent tokens |

Scheduler state:

```text
sigmas [num_steps + 1] float32 on device, terminal zero
timesteps [num_steps] float32 on device
step_index mutable host scheduler state
scale_noise(image_latents, first_timestep, noise) = sigma * noise + (1 - sigma) * image_latents
step(noise_pred, t, latents) = latents + (sigma_next - sigma) * noise_pred, upcast to fp32 then cast back
```

Decode boundary:

```text
latents [B,N_total,64]
  -> split last row segment by image sizes
  -> unpack 2x2 tiles: [1,16,H_query/8,sum target/condition W/8]
  -> unscale: latent / scaling_factor + shift_factor
  -> AutoencoderKL.decode: [1,3,H_query,sum W]
  -> VaeImageProcessor.postprocess
  -> crop target x ranges from the query row only
```

Stage-2 upsampling boundary:

```text
stage-1 target PIL images
  -> full-white RGB mask of same size
  -> FluxFillPipeline(prompt=content_prompt, image=target, mask_image=white, height/width, strength)
  -> same FLUX Fill latent pack/mask denoising path
  -> VAE decode/postprocess to final image type
```

Precomputable and reusable tensors: tokenized text, CLIP pooled embeds, T5 prompt embeds, text ids, layout prompt strings for fixed grid shapes, VAE-encoded non-target in-context rows, target masks, latent image ids for fixed row/cell sizes, and scheduler timesteps for fixed `num_inference_steps`, `sigmas`, `mu`, and `strength`.

## 5. Operator coverage checklist

Tensor/layout ops:

- PIL/NumPy/Torch image conversion, resize/crop, normalization, postprocess.
- NCHW concat along width for grid rows and sequence concat for row tokens.
- `view`/`permute`/`reshape` 2x2 latent packing and unpacking.
- Mask reshape from pixel space to 8x8 latent cell features, then 2x2 pack.
- Token concat/split: text/image attention sequence concat, output split, query-row crop split.
- Repeat/expand for `num_images_per_prompt`, batch duplication, scalar timestep expansion.
- Dynamic shape arithmetic for `H/8`, `W/8`, `H/16`, `W/16`, and row width sums.

GEMM/linear ops:

- Transformer input projections: `Linear(384 -> 3072)` and `Linear(4096 -> 3072)`.
- Time/guidance/text conditioning MLPs: sinusoidal 256 -> 3072 -> 3072 plus pooled projection 768 -> 3072.
- Dual blocks: Q/K/V for image and added Q/K/V for text, all `3072 -> 3072`, plus output projections and feed-forward MLPs.
- Single blocks: Q/K/V `3072 -> 3072`, MLP `3072 -> 12288`, concatenated projection `(3072 + 12288) -> 3072`.
- Final adaptive norm projection and `proj_out: 3072 -> 64`.
- CLIP/T5 encoders if included in Dinoml scope; otherwise prompt embeddings can be external.

Attention primitives:

- Joint text+image attention with QK RMSNorm and RoPE over `[L_text + N_img]`.
- Single-stream self-attention over concatenated text/image sequence.
- Native parity backend is `dispatch_attention_fn` with PyTorch SDPA shape `[B,S,heads,head_dim]` after internal permutes.
- No attention mask in base VisualCloze denoiser call.

Normalization and adaptive conditioning:

- `LayerNorm(eps=1e-6, affine=False)`.
- `RMSNorm(head_dim=128, eps=1e-6)` on Q/K.
- `AdaLayerNormZero`, `AdaLayerNormZeroSingle`, and `AdaLayerNormContinuous`: SiLU + Linear generate shift/scale/gates.
- fp16 clamp to `[-65504,65504]` after residuals in transformer blocks.

Position/timestep/guidance embeddings:

- Timestep sinusoidal embedding with 256 channels, flip sin/cos.
- Separate guidance embedding for official configs.
- CLIP pooled projection through PixArt-style text projection.
- 3-axis RoPE from `txt_ids` and `img_ids`, axes dims `[16,56,56]`.

Scheduler and guidance arithmetic:

- `calculate_shift`: linear interpolation from image token count to `mu`.
- `np.linspace(1.0, 1/steps, steps)` default sigma list.
- FlowMatch Euler dynamic time shift, terminal zero sigma, `scale_noise`, and deterministic Euler step.
- Embedded guidance only; no CFG batch split arithmetic required first.

VAE/postprocessing ops:

- AutoencoderKL encode/decode: Conv2d, ResNet blocks, GroupNorm, SiLU, mid attention, up/downsample.
- Latent scale/shift before/after VAE.
- Stage-1 target crop after decode; stage-2 full image postprocess.

Variant/control ops:

- FLUX Fill upsampling adds full-white mask preprocessing and masked-image VAE encode.
- LoRA/textual inversion mutate weights/token prompts outside the base static denoiser graph.
- ControlNet/IP-Adapter branches are inactive for base VisualCloze.

## 6. Denoiser/model breakdown

`FluxTransformer2DModel.forward`:

```text
hidden_states [B,N,384] -> x_embedder -> [B,N,3072]
timestep/guidance/pooled CLIP -> CombinedTimestepGuidanceTextProjEmbeddings -> temb [B,3072]
T5 prompt_embeds [B,L,4096] -> context_embedder -> [B,L,3072]
cat(txt_ids, img_ids) -> FluxPosEmbed -> RoPE cos/sin
19 dual-stream FluxTransformerBlock
38 single-stream FluxSingleTransformerBlock
AdaLayerNormContinuous(hidden, temb)
proj_out -> [B,N,64]
```

Dual-stream block:

```text
image stream:
  AdaLayerNormZero(hidden, temb) -> norm_hidden, gate_msa, shift_mlp, scale_mlp, gate_mlp
text stream:
  AdaLayerNormZero(context, temb) -> norm_context, c_* gates/shifts/scales
joint attention:
  image QKV + text added QKV -> RMSNorm Q/K -> concat text then image -> RoPE -> SDPA
  split output back to text and image -> projection
image residual:
  hidden += gate_msa * attn_out
  LayerNorm -> scale/shift -> FeedForward(GELU approx) -> hidden += gate_mlp * ff
text residual:
  context += c_gate_msa * context_attn
  LayerNorm -> scale/shift -> FeedForward(GELU approx) -> context += c_gate_mlp * context_ff
```

Single-stream block:

```text
cat(context, hidden) on sequence
AdaLayerNormZeroSingle -> norm + gate
parallel-ish branches:
  MLP: Linear 3072 -> 12288 -> GELU(tanh)
  attention: QKV -> RMSNorm Q/K -> RoPE -> SDPA
cat(attn_out, mlp_out) -> Linear 15360 -> 3072
residual += gate * projection
split context/image back by original text length
```

Bias flags are true for the FLUX attention and MLP projections in this implementation. Dropout modules exist but are inactive for inference.

## 7. Attention requirements

Required base attention:

| Attention | Sequence | Heads | Head dim | Mask | Norm/pos | Source path |
|---|---|---:|---:|---|---|---|
| dual-stream joint attention | text + image tokens | 24 | 128 | none | QK RMSNorm, 3-axis RoPE | `FluxTransformerBlock` -> `FluxAttnProcessor` |
| single-stream self attention | text + image tokens after concat | 24 | 128 | none | QK RMSNorm, 3-axis RoPE | `FluxSingleTransformerBlock` -> `FluxAttnProcessor` |

The authoritative non-deprecated processor is `FluxAttnProcessor` in `transformer_flux.py`. `FluxAttnProcessor2_0`, `FusedFluxAttnProcessor2_0`, and IP-Adapter `*_2_0` names in `attention_processor.py` are deprecated compatibility wrappers.

Processor/backend dispatch:

```text
Linear Q/K/V
unflatten last dim -> [B,S,heads,head_dim]
RMSNorm query/key
optional added text Q/K/V concat
apply_rotary_emb(sequence_dim=1)
dispatch_attention_fn(..., backend=self._attention_backend)
flatten heads -> Linear output
```

`dispatch_attention_fn` uses the active Diffusers attention backend when the processor backend is `None`; by default this is the native PyTorch SDPA backend unless `DIFFUSERS_ATTN_BACKEND` or `attention_backend(...)` changes it. Diffusers also registers flash, flash-varlen, flash3, AITER, Sage, Flex, xFormers, and native cuDNN/flash/efficient/math backends with package and dtype/shape checks.

Dinoml flash feasibility:

- Valid candidate: dense, no-mask, non-causal attention with `B`, `S`, `H=24`, `D=128`, bf16/fp16 is a strong flash-style target.
- Required preconditions: Q/K/V must be contiguous or supported-strided after QK RMSNorm and RoPE; text+image concatenation order must be preserved; output split sizes must match `[L_text, N_img]`; no IP-Adapter or ControlNet residual branch active; no attention mask.
- Varlen is not required for first parity because VisualCloze pads T5 to `max_sequence_length`. It may become useful if Dinoml later avoids padded text tokens.
- Added-KV in the dual block is not external K/V after attention; it is projected text tokens concatenated into the same self-attention problem. A fused provider can treat it as one dense sequence after projection, but QKV projection fusion must account for separate image and text projection weights.

## 8. Scheduler and denoising-loop contract

Stage 1:

1. Compute `image_seq_len = sum((H/8/2) * (W/8/2))` for all cells in the first sample.
2. Compute `mu = image_seq_len * ((max_shift - base_shift)/(max_seq_len - base_seq_len)) + base_shift - m * base_seq_len`.
3. Default `sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)`.
4. Call `scheduler.set_timesteps(sigmas=sigmas, device=device, mu=mu)`.
5. Call `get_timesteps(num_inference_steps, strength=1.0)`, setting scheduler begin index to zero.
6. Encode image latents, draw Gaussian noise, and call `scheduler.scale_noise(image_latents, first_timestep, noise)`.
7. For each timestep, run transformer and `scheduler.step(noise_pred, t, latents)`.

Stage 2 upsampling uses the same `FlowMatchEulerDiscreteScheduler` class but `FluxFillPipeline.get_timesteps(num_inference_steps, upsampling_strength)` skips early timesteps when `strength < 1`.

Host vs compiled split:

- Host first: `set_timesteps`, dynamic `mu`, `get_timesteps`, mutable `step_index`, random noise generation, progress/callback handling.
- Compile candidates: `scale_noise`, per-step Euler update, timestep/guidance tensor fill, latent/model-input concat, and possibly the full denoiser step.
- First Dinoml scheduler slice should implement deterministic FlowMatch Euler with dynamic shift and custom sigmas, not broader Karras/exponential/beta/stochastic branches.

## 9. Position, timestep, and custom math

Custom math to preserve:

```python
mu = image_seq_len * ((max_shift - base_shift) / (max_seq_len - base_seq_len)) \
     + base_shift - ((max_shift - base_shift) / (max_seq_len - base_seq_len)) * base_seq_len
```

```python
scale_noise(sample, sigma, noise) = sigma * noise + (1 - sigma) * sample
step(sample, model_output) = sample + (sigma_next - sigma) * model_output
```

Timestep and guidance embeddings use standard Diffusers sinusoidal timestep embedding with 256 channels, then two-layer MLPs. Official configs use `CombinedTimestepGuidanceTextProjEmbeddings`, so `guidance_scale` affects the model through an embedding, not through CFG arithmetic.

RoPE uses three ID axes. Text IDs are all zeros; image IDs are row index, patch row, patch column. The row index starts at 1 for each concatenated grid row in VisualCloze generation. For fixed grid shapes, `img_ids` and RoPE tables can be precomputed.

The source uses float64 frequency computation for RoPE except MPS/NPU guards. Dinoml CUDA parity can use a fixed high-precision or fp32 table policy, but should test against Diffusers for large grids because token count affects RoPE values.

## 10. Preprocessing and input packing

VisualCloze-specific preprocessing:

- Validate rectangular row lengths and require target `None` only in the last row.
- Resize each image cell by preserving aspect ratio with target area `resolution^2`, then floor dimensions to multiples of `2 * vae_scale_factor = 16`.
- Fill target `None` cells with black RGB blanks of the row size.
- Generate binary target masks: 1 for `None` target cells in query row, 0 otherwise.
- Build a layout prompt from grid row/column count and concatenate it with task/content text.
- Convert images to NCHW tensors and normalize before VAE encode.

Prompt composition is CPU-side first. For a denoiser-only Dinoml slice, accept `prompt_embeds`, `pooled_prompt_embeds`, `text_ids`, packed `latents`, `masked_image_latents`, and `img_ids` as external inputs.

Packing and unpatching are model-coupled GPU/runtime candidates:

```text
pack_latents:
  [B,C,H,W] -> view [B,C,H/2,2,W/2,2]
  -> permute [B,H/2,W/2,C,2,2]
  -> reshape [B,(H/2)*(W/2),C*4]

unpack_latents:
  [B,N,C*4] -> view [B,H/2,W/2,C,2,2]
  -> permute [B,C,H/2,2,W/2,2]
  -> reshape [B,C,H,W]
```

Stage-2 upsampling preprocessing is inherited from `FluxFillPipeline`: preprocess image to requested output size, preprocess full-white mask to grayscale/binary, encode masked image, pack mask into latent tokens, concatenate `[latents, masked_image_latents, mask]`, and denoise.

## 11. Graph rewrite / lowering opportunities

1. VisualCloze mask pack canonicalization

- Source pattern: pixel mask `[B,1,H,W] -> view into 8x8 cells -> permute -> reshape `[B,64,H/8,W/8]` -> 2x2 pack -> `[B,(H/16)*(W/16),256]`.
- Replacement: one layout-aware mask-packing kernel from NCHW mask to token features.
- Preconditions: `H` and `W` multiples of 16; VAE scale factor 8; binary or scalar mask values; NCHW contiguous source.
- Shape equations: `N = (H/16)*(W/16)`, output channels 256.
- Failure cases: nonstandard VAE scale factor, non-contiguous strided masks, channel-last source without explicit axis rewrite.
- Test sketch: random binary masks at several H/W and compare packed tensor exactly.

2. Latent 2x2 pack/unpack as view-permute lowering

- Source pattern: `view -> permute -> reshape` for latents before/after transformer.
- Replacement: layout transform kernel or metadata view when consumer can read packed layout.
- Preconditions: NCHW contiguous, even latent H/W, channel count known.
- Weight transform: none.
- Failure cases: odd latent dimensions, existing non-contiguous tensors, downstream VAE requiring NCHW map.
- Test sketch: round-trip pack/unpack for fp32/bf16 and compare exact/bitwise for simple arange tensors.

3. Joint attention projection fusion

- Source pattern: separate image Q/K/V and text added Q/K/V linear projections, QK RMSNorm, concat text+image, RoPE, SDPA, output split/projections.
- Replacement: grouped GEMM or two-stream QKV projection fusion feeding a dense attention provider.
- Preconditions: same hidden size/head shape, no IP-Adapter, no attention mask, fixed text/image split.
- Weight transform: stack Q/K/V weights separately for image and text paths; cannot blindly combine image and text inputs because weights differ.
- Failure cases: fused projection flags, LoRA active at runtime, IP-Adapter external K/V, ControlNet residual variants.
- Test sketch: one dual block with random inputs, compare projections, attention output, and split outputs.

4. AdaLN + residual fusion

- Source pattern: SiLU(temb) -> Linear -> chunk shift/scale/gates -> LayerNorm -> affine -> residual gated add.
- Replacement: fused adaptive norm and gated residual kernel.
- Preconditions: affine-free LayerNorm eps 1e-6, hidden size 3072, gate broadcast over sequence.
- Failure cases: different norm type or affine flags in other families.
- Test sketch: random block inputs at fp32 and bf16 with tolerance against PyTorch.

5. Stage-1 crop after decode

- Source pattern: decode full query row image, then PIL/NumPy crop target x ranges.
- Replacement: optional decode/crop scheduling or postprocess crop kernel.
- Preconditions: target ranges aligned to decoded pixel coordinates; all targets from last row; output type not `latent`.
- Failure cases: VAE tiled decode, PIL-only color conversions, multiple target cells with differing heights.
- Test sketch: decode synthetic row tensor and compare crop rectangles.

Layout candidates:

- Candidate NHWC/channel-last region: VAE conv/resnet blocks and image processor staging after NCHW semantic import.
- No-layout-translation guard: FLUX latent pack/unpack, mask 8x8 reshape, GroupNorm channel axis, VAE scale/shift boundary, row-width concat, and post-decode crop coordinate math.
- Axis rewrites needed for channel-last: concat `dim=3` row width remains width axis after NCHW->NHWC rewrite, GroupNorm channel axis changes, mask `view` order must be replaced rather than axis-renamed, and pack/unpack kernels need separate NHWC definitions.

## 12. Kernel fusion candidates

Highest priority:

- FLUX dense attention provider for no-mask joint/self attention with QK RMSNorm and RoPE. VisualCloze exercises long text+multi-image sequences and 24x128 heads.
- GEMM/linear coverage for 3072-wide FLUX blocks, including bf16 weights and activations.
- AdaLayerNormZero/AdaLayerNormContinuous fused norm/gating kernels. Every dual and single block uses these.
- Latent/mask packing kernels. The 384-channel input assembly is a VisualCloze/FLUX Fill first-slice blocker.
- FlowMatch Euler `scale_noise` and `step` kernels to avoid Python tensor overhead inside denoising loops.

Medium priority:

- QKV projection fusion with separate image/text projection groups.
- Feed-forward GELU approximate fusion for `3072 -> 12288 -> 3072`.
- VAE decode conv + GroupNorm + SiLU fusion for final image production.
- Stage-2 FLUX Fill mask/latent preparation and SDEdit strength scheduling.

Lower priority:

- CLIP/T5 encoder compilation. Useful for end-to-end but prompt embeddings can be cached or external initially.
- Dynamic grid preprocessing on GPU. PIL resize/crop likely remains CPU first.
- Crop/postprocess fusion after decode.
- LoRA runtime merge and textual inversion token mutation.

## 13. Runtime staging plan

1. Parse VisualCloze component configs for 384 and 512 and record constructor-only `resolution`.
2. Implement a denoiser-only slice with externally supplied `hidden_states [B,N,384]`, T5 embeds, CLIP pooled embeds, ids, timestep, and guidance.
3. Add latent/mask pack/unpack parity kernels and FlowMatch Euler scheduler arithmetic.
4. Add VAE encode/decode boundary support using the existing AutoencoderKL plan; initially allow CPU/PyTorch preprocessing to provide packed latents.
5. Run one VisualCloze generation denoising step at fixed grid sizes with Diffusers prompt embeddings and VAE latents.
6. Run the full stage-1 denoising loop with scheduler in Python and Dinoml transformer step.
7. Add stage-1 decode/crop parity.
8. Add stage-2 FluxFill upsampling as a separate inherited FLUX Fill integration, reusing the same denoiser kernels but different preprocessing and `strength`.
9. Optimize attention/norm/GEMM, then add LoRA/textual inversion mutation as separate candidates.

Stubs acceptable initially: tokenizers, CLIP/T5 encoders, PIL resize/crop, VAE encode/decode, and optional upsampling. The first useful Dinoml target is one transformer denoiser step plus scheduler arithmetic.

## 14. Parity and validation plan

- Config parse test: load both official configs and assert transformer/VAE/scheduler dimensions match the table above.
- Pack/unpack tests: arange tensors for latent maps and masks across 1x2, 2x2, and 3-column visual grids.
- Prompt embedding parity: Diffusers-generated `prompt_embeds`, `pooled_prompt_embeds`, and `text_ids` for fixed prompts; verify shapes and duplicate behavior.
- VAE encode/decode boundary: compare scaled latents and decoded output for one row concat image.
- Single transformer block parity: dual block and single block with random bf16/fp32 tensors, including QK RMSNorm and RoPE.
- One denoiser step parity: fixed packed latents, masks, prompt embeddings, timestep, guidance, and scheduler state.
- Scheduler parity: `calculate_shift`, `set_timesteps`, `scale_noise`, and `step` against Diffusers for `num_inference_steps` 2, 30, 50 and strengths 1.0/0.4.
- Stage-1 short loop: deterministic 2-step generation using the tiny VisualCloze test model first, then official config with sampled random weights if real weights are too large.
- Combined smoke: `upsampling_strength=0` returns nested target list; `upsampling_strength=0.4` returns upsampled targets with dimensions rounded to multiples of 16.

Suggested tolerances: fp32 scheduler/packing exact or `1e-6`; transformer fp32 `1e-4`; bf16/fp16 block and step `1e-2` to `3e-2` depending on attention backend; end-to-end image smoke should use perceptual or loose pixel tolerances because VAE/attention dtype changes accumulate.

## 15. Performance probes

- Text encoder time: CLIP pooled and T5 token embeddings separately, especially T5 XXL memory pressure.
- Stage-1 preprocessing time by grid rows/columns and `resolution=384/512`.
- VAE encode time for row-concat in-context grids.
- One transformer step by image token count: 2x2 cells, 3-column subject-driven examples, and multi-target query rows.
- Attention backend comparison: native SDPA vs Dinoml flash-style provider at `L_text=512` and varying `N_img`.
- Norm/GEMM/attention split inside 19 dual + 38 single FLUX blocks.
- Scheduler/guidance overhead with Python host loop vs fused tensor kernels.
- VAE decode and crop time for query row only.
- Stage-2 FluxFill upsampling by output size and `strength` step count.
- VRAM: prompt encoders, transformer weights (23.8 GB index metadata), T5 weights (9.5 GB index metadata), VAE activations, packed mask/input temporaries.

## 16. Scope boundary and separate candidates

Separate candidate reports:

- `visualcloze_upsampling_flux_fill`: stage-2 `FluxFillPipeline` SDEdit upsampling, full-white inpaint masks, strength-dependent timestep skip.
- `visualcloze_lora_textual_inversion`: inherited FLUX LoRA, PEFT scale mutation, and textual inversion prompt conversion.
- `flux_ip_adapter`: inactive in base VisualCloze but supported by `FluxTransformer2DModel`/attention processors when pipeline components supply image embeddings.
- `flux_variants_control`: ControlNet residual samples accepted by the transformer but not wired by VisualCloze.
- `autoencoder_kl_flux`: FLUX AutoencoderKL encode/decode optimization.
- `clip_t5_prompt_encoders`: end-to-end prompt encoder compilation and caching.
- `scheduler_flow_match_euler`: shared FlowMatch Euler scheduler implementation across FLUX-like pipelines.

Ignored/out of scope for this audit:

- Multi-GPU/context parallel plans and distributed attention.
- Callback mutation and interactive interrupt behavior.
- XLA, NPU, MPS, Flax, and ONNX branches.
- Safety checker and NSFW filtering.
- Training, losses, dropout, gradient checkpointing.
- Gated weight downloads beyond JSON/index metadata. No blockers occurred fetching official VisualCloze JSON configs.

## 17. Final implementation checklist

- [ ] Parse `VisualClozePipeline-384` and `VisualClozePipeline-512` configs, including constructor `resolution`.
- [ ] Load or accept external CLIP pooled and T5 token embeddings.
- [ ] Implement VisualCloze grid image/mask metadata and shape validation.
- [ ] Implement latent 2x2 pack/unpack and mask pack parity.
- [ ] Implement FLUX timestep/guidance/text conditioning embeddings.
- [ ] Implement `FluxTransformer2DModel` denoiser forward for `in_channels=384`, `out_channels=64`.
- [ ] Add QK RMSNorm + RoPE + no-mask dense attention provider path.
- [ ] Implement FlowMatch Euler dynamic-shift scheduler tables, `scale_noise`, and `step`.
- [ ] Integrate AutoencoderKL encode/decode or accept VAE latents as a staged boundary.
- [ ] Add stage-1 decode and target crop parity.
- [ ] Add optional FluxFill upsampling stage as a separate follow-up.
- [ ] Add parity tests for pack/mask, one transformer block, one denoiser step, scheduler, and short loop.
- [ ] Benchmark one denoiser step by grid token count, attention backend, VAE decode, and full stage-1 loop.
