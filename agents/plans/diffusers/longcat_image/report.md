# Diffusers LongCat Image Operator and Integration Report

Candidate slug: `longcat_image`

## 1. Source basis

```text
Diffusers commit/version:
  Local checkout diffusers at b3a515080752a3ba7ca92161e25530c7f280f629.

Model id(s):
  meituan-longcat/LongCat-Image
  meituan-longcat/LongCat-Image-Dev
  meituan-longcat/LongCat-Image-Edit
  meituan-longcat/LongCat-Image-Edit-Turbo

Config sources:
  H:/configs/meituan-longcat/LongCat-Image/model_index.json
  H:/configs/meituan-longcat/LongCat-Image-Dev/model_index.json
  H:/configs/meituan-longcat/LongCat-Image-Edit/model_index.json
  H:/configs/meituan-longcat/LongCat-Image-Edit-Turbo/model_index.json
  Official component configs were read through Hugging Face Hub authenticated/local-cache tools:
    config.json
    scheduler/scheduler_config.json
    transformer/config.json
    vae/config.json
    text_encoder/config.json
    tokenizer/tokenizer_config.json
    text_processor/preprocessor_config.json
    text_processor/tokenizer_config.json
  They were not copied into H:/configs by this worker because this task owns only this report path.

Pipeline files inspected:
  diffusers/src/diffusers/pipelines/longcat_image/pipeline_longcat_image.py
  diffusers/src/diffusers/pipelines/longcat_image/pipeline_longcat_image_edit.py
  diffusers/src/diffusers/pipelines/longcat_image/pipeline_output.py
  diffusers/src/diffusers/pipelines/longcat_image/system_messages.py

Model files inspected:
  diffusers/src/diffusers/models/transformers/transformer_longcat_image.py
  diffusers/src/diffusers/models/autoencoders/autoencoder_kl.py
  diffusers/src/diffusers/models/autoencoders/vae.py

Scheduler/processors/helpers inspected:
  diffusers/src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py
  diffusers/src/diffusers/models/attention_dispatch.py
  diffusers/src/diffusers/models/attention.py
  diffusers/src/diffusers/models/normalization.py
  diffusers/src/diffusers/image_processor.py
  diffusers/src/diffusers/loaders/lora_pipeline.py

External component configs inspected:
  Qwen2_5_VLForConditionalGeneration / Qwen2Tokenizer / Qwen2_5_VLProcessor configs from official LongCat repos.

Any missing files or assumptions:
  No official LongCat Image config read was gated or blocked. Local H:/configs contained only model_index.json files,
  so component facts below are from official Hub component configs and source defaults. This report focuses on the
  base text-to-image pipeline and inventories edit, prompt-rewrite, LoRA/adapter, single-file, and absent control-like
  surfaces separately. Multi-GPU/context parallel, callback mutation and interrupt, XLA/NPU/MPS/Flax/ONNX, safety,
  training/loss/dropout/gradient checkpointing are out of scope.
```

## 2. Pipeline and component graph

`LongCatImagePipeline` is a Qwen2.5-VL-conditioned, FlowMatch Euler latent image transformer pipeline. It uses
Qwen2.5-VL twice in source: optionally as a prompt rewriter via `generate`, then as a text encoder whose final hidden
state conditions the denoiser. The denoiser is `LongCatImageTransformer2DModel`, a Flux-like dual-stream plus
single-stream transformer over 2x2 packed VAE latent tokens.

```text
prompt
  -> optional Qwen2.5-VL prompt rewrite generation
  -> LongCat prompt tokenizer path with quote-sensitive token splitting
  -> Qwen2.5-VL final hidden states [B,512,3584]
  -> latent noise [B,16,H/8,W/8] source NCHW
  -> 2x2 pack to transformer tokens [B,(H/16)*(W/16),64]
  -> denoising loop:
       LongCatImageTransformer2DModel(latent tokens, prompt embeds,
                                      timestep/1000, text ids, image ids)
       optional true CFG second denoiser call
       optional base-pipeline CFG renorm
       FlowMatchEulerDiscreteScheduler.step
  -> unpack latent tokens to [B,16,H/8,W/8]
  -> AutoencoderKL decode((latents / 0.3611) + 0.1159)
  -> VaeImageProcessor postprocess
```

First-slice required components:

| Component | Class/file | First-slice role |
| --- | --- | --- |
| Pipeline | `LongCatImagePipeline`, `pipeline_longcat_image.py` | Prompt embedding boundary, latent packing, CFG/renorm, scheduler loop, VAE decode. |
| Denoiser | `LongCatImageTransformer2DModel`, `transformer_longcat_image.py` | Packed-token transformer step. |
| Scheduler | `FlowMatchEulerDiscreteScheduler` | Dynamic-shift custom-sigma FlowMatch Euler. |
| VAE | `AutoencoderKL` | Decode required for text-to-image; encode required by edit. |
| Text encoder | `Qwen2_5_VLForConditionalGeneration` | External first, then optional compiled/cacheable text stage. |
| Tokenizer/processor | `Qwen2Tokenizer`, `Qwen2VLProcessor` | CPU/data path for tokenization, prompt rewrite, and edit image prompt inputs. |

Separate candidate reports:

| Surface | Classes/files | Runtime delta |
| --- | --- | --- |
| `longcat_image_edit` | `LongCatImageEditPipeline`, `pipeline_longcat_image_edit.py` | Adds image resize/preprocess, Qwen2VL visual tokens, VAE encode of the source image, packed source-image latent tokens concatenated with target noise tokens, and modality-2 image ids. |
| `longcat_image_edit_turbo` | Same edit pipeline, `LongCat-Image-Edit-Turbo` config | Same graph as edit, but scheduler config sets `shift=1.0` and `base_shift=max_shift=1.15`; intended as a distilled/fast variant. |
| `longcat_prompt_rewrite` | `LongCatImagePipeline.rewire_prompt`, `system_messages.py` | Qwen2.5-VL causal generation stage before diffusion; cacheable or disableable with `enable_prompt_rewrite=False`. |
| `longcat_lora_adapters` | `PeftAdapterMixin` on transformer and VAE; generic `lora_pipeline.py` | Component-level PEFT/adapter mutation exists, but no LongCat-specific pipeline LoRA loader mixin was found. Treat as separate artifact-state work. |
| `longcat_single_file_original` | `FromSingleFileMixin` on pipelines, `FromOriginalModelMixin` on transformer | Original/single-file checkpoint conversion and key mapping, separate from base ops. |
| `longcat_vae_codec` | `AutoencoderKL` config with z=16, scale/shift | VAE encode/decode operator island shared by base/edit. |

No LongCat Image folder implementation was found for textual inversion, IP-Adapter, ControlNet, T2I-Adapter, GLIGEN,
inpaint, depth2img, or upscaling. Generic Diffusers infrastructure for some of these exists elsewhere, but it is not an
active LongCat Image first-slice surface.

## 3. Important config dimensions

Representative checkpoint sweep:

| Config | Pipeline | Transformer blocks | Heads x dim | Text dim | Latent/token | Scheduler | Notes |
| --- | --- | ---: | --- | ---: | --- | --- | --- |
| `LongCat-Image` | `LongCatImagePipeline` | 10 dual + 20 single | 24 x 128 | 3584 | VAE z=16, packed C=64 | FlowMatch dynamic exponential; shift 3.0 config present | Base text-to-image. |
| `LongCat-Image-Dev` | `LongCatImagePipeline` | 10 + 20 | 24 x 128 | 3584 | same | same as base | Same operator/config shape as base. |
| `LongCat-Image-Edit` | `LongCatImageEditPipeline` | 10 + 20 | 24 x 128 | 3584 | target tokens plus source-image tokens | same as base | Image-conditioned edit, Qwen2VL image prompt, VAE encode. |
| `LongCat-Image-Edit-Turbo` | `LongCatImageEditPipeline` | 10 + 20 | 24 x 128 | 3584 | same as edit | dynamic, `base_shift=max_shift=1.15`, `shift=1.0` | Same model graph, different scheduler staging. |

Transformer config facts:

| Field | Official configs | Source default / note |
| --- | ---: | --- |
| `patch_size` | 1 | The pipeline already packs 2x2 latents; model projection is Linear(64 -> inner). |
| `in_channels` / `out_channels` | 64 / 64 | Packed token width, not VAE latent channels. |
| `num_layers` | 10 | Source default is 19; use config, not source default. |
| `num_single_layers` | 20 | Source default is 38. |
| `num_attention_heads` | 24 | Inner dim = 3072. |
| `attention_head_dim` | 128 | Head dim also equals RoPE dim sum. |
| `joint_attention_dim` | 3584 | Qwen2.5-VL hidden size. |
| `pooled_projection_dim` | 3584 | Stored but not used by current `forward`. |
| `axes_dims_rope` | omitted | Effective source default `[16,56,56]`. |

Text and processor config facts:

| Component | Dimensions / behavior |
| --- | --- |
| `Qwen2_5_VLForConditionalGeneration` | hidden 3584, 28 layers, 28 attention heads, 4 KV heads, intermediate 18944, vocab 152064, max positions 128000. |
| Qwen vision subconfig | depth 32, hidden 1280, 16 heads, patch size 14, out hidden 3584. Used by edit and prompt rewrite processor paths. |
| `Qwen2Tokenizer` | `model_max_length=131072`, pad `<|endoftext|>`, EOS `<|im_end|>`. Pipeline truncates diffusion prompt payload to 512 user tokens. |
| `Qwen2_5_VLProcessor` | `patch_size=14`, `merge_size=2`, `min_pixels=3136`, `max_pixels=12845056`. Edit uses processor image tokens. |

VAE and scheduler:

| Component | Key fields |
| --- | --- |
| `AutoencoderKL` | `latent_channels=16`, sample size 1024, block channels `[128,256,512,512]`, 2 layers/block, GroupNorm 32, mid attention enabled, `force_upcast=true`, no quant/post-quant conv, `scaling_factor=0.3611`, `shift_factor=0.1159`. |
| Base/dev/edit scheduler | `FlowMatchEulerDiscreteScheduler`, `use_dynamic_shifting=true`, `base_shift=0.5`, `max_shift=1.15`, `base_image_seq_len=256`, `max_image_seq_len=4096`, `num_train_timesteps=1000`, source default `time_shift_type="exponential"`, non-stochastic. |
| Edit-Turbo scheduler | Same class, `use_dynamic_shifting=true`, `base_shift=max_shift=1.15`, `shift=1.0`; source default exponential, non-stochastic. |

Weight metadata from Hub file metadata: each official repo has a 12.54 GB transformer safetensors file, a 0.168 GB VAE
safetensors file, and five Qwen text-encoder safetensors shards totaling about 16.58 GB. Example code loads with
`torch_dtype=torch.bfloat16`; dtype facts are from examples/weight size inference, not explicit component config fields.

Recommended first Dinoml scheduler slice: FlowMatch Euler with custom sigma list, dynamic exponential shift `mu`, terminal
sigma append, non-stochastic `prev = sample + (sigma_next - sigma) * model_output`. Add Edit-Turbo as a config-parity
case because its constant shift settings are a useful schedule table regression.

## 3a. Family variation traps

- `in_channels=64` is the packed 2x2 latent token width. The VAE latent map is `[B,16,H/8,W/8]`.
- The checkpoint transformer depth is 10 dual + 20 single blocks; source defaults 19 + 38 are inactive for official
  configs.
- `axes_dims_rope` is omitted by official transformer configs; effective value comes from source default `[16,56,56]`.
- Base prompt embeddings are fixed to 512 user-token slots after prefix/suffix removal, but edit prompt embeddings include
  Qwen2VL image tokens plus user prompt tokens between the kept `vision_start` span and suffix.
- Prompt rewrite is enabled by default in base text-to-image and invokes Qwen2.5-VL generation before text embedding.
  Disable or externalize it for first diffusion parity.
- LongCat true CFG uses two separate transformer calls (`cond` and `uncond` cache contexts), not batch concatenation.
- Base text-to-image applies optional CFG renorm by token-channel norm; edit pipeline does not apply that renorm.
- Edit concatenates source-image latent tokens to the target noisy latent sequence and then slices the transformer output
  back to the target token count. Do not model it as ControlNet residual injection.
- The edit pipeline has an apparent latent-tensor shortcut bug: `__call__` checks `image.size(1) == self.latent_channels`,
  but this pipeline class does not define `self.latent_channels`. PIL/image inputs take the normal path; direct latent
  edit inputs need validation before admission.
- Scheduler class supports broader FlowMatch options, but sampled LongCat configs require dynamic exponential shifting and
  non-stochastic scalar-timestep steps only.

## 4. Runtime tensor contract

For 1024x1024 base text-to-image, one image per prompt:

| Boundary | Tensor | Shape/layout | Notes |
| --- | --- | --- | --- |
| User prompt tokens | tokenizer ids | up to 512 user tokens | Quote-sensitive splitting tokenizes quoted text character-by-character. |
| Full text encoder input | ids/mask | `[B, prefix + 512 + suffix]` | Prefix/suffix are LongCat template strings. |
| Prompt embeds | `prompt_embeds` | `[B,512,3584]` | Final Qwen hidden state, prefix/suffix removed. |
| Text ids | `text_ids` | `[512,3]` | Modality 0, positions `[0..511]` on axes 1 and 2. |
| Initial latent map | random | `[B,16,128,128]` NCHW | For general H/W: `[B,16,H/8,W/8]`, rounded to pack-compatible size. |
| Packed latents | transformer input | `[B,4096,64]` | 2x2 spatial pack: `(128/2)*(128/2)` tokens. |
| Image ids | `latent_image_ids` | `[4096,3]` | Modality 1, starts at `(512,512)`, grid `[64,64]`. |
| Timestep | model input | `[B]` | Pipeline passes scheduler `t / 1000`; transformer multiplies by 1000. |
| Denoiser output | `noise_pred` | `[B,4096,64]` | Same packed-token shape. |
| CFG renorm | base only | norm over packed channel dim `-1` | `cond_norm / (noise_norm + 1e-8)` clamped to `[cfg_renorm_min,1]`. |
| Scheduler output | `latents` | `[B,4096,64]` | Flow Euler update. |
| Unpacked latents | decode input map | `[B,16,128,128]` NCHW | Reverse 2x2 unpack. |
| VAE decode input | shifted/unscaled | `[B,16,128,128]` | `(latents / scaling_factor) + shift_factor`. |
| Decoded image | VAE output | `[B,3,1024,1024]` NCHW | Postprocessed to PIL/NumPy/etc. |
| `output_type="latent"` | packed latents | `[B,4096,64]` | Returned before unpack/decode. |

For the edit pipeline, target resolution is derived from the input image aspect ratio with target area `1024*1024` and
rounded up to multiples of 16. The input image is resized/preprocessed for VAE encode, and a half-resolution copy is
fed to Qwen2VL prompt encoding. Source image latents are encoded as
`(vae.encode(image).latent_dist.mode() - shift_factor) * scaling_factor`, packed to `[B,S,64]`, assigned modality 2 ids,
and concatenated to target latent tokens for the transformer. Only the first target `S` output tokens are scheduler-updated.

CPU/data-pipeline work: tokenization, prompt rewrite generation, Qwen2VL image preprocessing, image resize/PIL conversion,
and optional text encoder execution until admitted. GPU/runtime work: packed denoiser, scheduler/CFG arithmetic, latent
pack/unpack, VAE encode for edit, VAE decode for all non-latent outputs.

Cacheable stages: prompt rewrite text, prompt embeddings/text ids, edit image embeddings/visual prompt tokens,
VAE-encoded source image latents, image ids/RoPE tables for a fixed resolution, and scheduler sigma/timestep tables.

## 5. Operator coverage checklist

Tensor/layout ops:

- NCHW latent allocation and random normal.
- 2x2 latent pack:
  `view(B,C,H/2,2,W/2,2) -> permute(0,2,4,1,3,5) -> reshape(B,H/2*W/2,4C)`.
- 2x2 latent unpack:
  `view(B,H/2,W/2,C/4,2,2) -> permute(0,3,1,4,2,5) -> reshape(B,C/4,H,W)`.
- Text/image position id construction and concat for RoPE.
- Edit token concat/slice: target latents plus source-image latents, then output slice `[:, :image_seq_len]`.
- CFG separate-call arithmetic and base CFG renorm norm/reduction.
- VAE/image processor resize, preprocess, postprocess as CPU or codec-boundary work first.

GEMM/linear ops:

- `x_embedder`: Linear(64 -> 3072).
- `context_embedder`: Linear(3584 -> 3072).
- Timestep embedding MLP: sinusoidal 256 -> 3072.
- Dual-block AdaLN modulation MLPs: two `SiLU + Linear(3072 -> 6*3072)` paths.
- Single-block AdaLN modulation: `SiLU + Linear(3072 -> 3*3072)`.
- Dual attention image Q/K/V and text added Q/K/V projections, all 3072-wide with bias.
- Attention output projections for image and text streams.
- FeedForward GELU-approximate MLPs in image and text branches.
- Single-block `proj_mlp` Linear(3072 -> 12288), GELU tanh, then Linear(15360 -> 3072).
- Final `AdaLayerNormContinuous` + Linear(3072 -> 64).

Attention primitives:

- 10 dual-stream joint text-image attention blocks with Q/K RMSNorm and RoPE.
- 20 single-stream blocks over concatenated text+image tokens with Q/K RMSNorm and RoPE.
- Noncausal attention, no mask in base/edit denoiser calls.
- Source processor dispatches through `dispatch_attention_fn`; native/eager is parity, flash-style is guarded.

Normalization and adaptive conditioning:

- LayerNorm without affine, RMSNorm over head dim, AdaLayerNormZero, AdaLayerNormZeroSingle, AdaLayerNormContinuous.
- Gated residual attention and MLP paths.
- fp16 clipping guards in blocks.

Scheduler and guidance arithmetic:

- FlowMatch custom sigma table and dynamic shift.
- Per-step update `sample + dt * model_output`.
- True CFG two-call arithmetic.
- Base CFG renorm reduction over token channel dimension.

VAE/postprocessing ops:

- AutoencoderKL Conv2d/ResNet/GroupNorm/SiLU/downsample/upsample/mid-attention decode.
- Edit VAE encode with DiagonalGaussian mode, then LongCat scale/shift standardization.
- Decode scale/shift and image postprocess.

## 6. Denoiser/model breakdown

`LongCatImageTransformer2DModel.forward`:

```text
hidden_states [B,S_img,64] -> x_embedder -> [B,S_img,3072]
encoder_hidden_states [B,S_txt,3584] -> context_embedder -> [B,S_txt,3072]
timestep (/1000 from pipeline, *1000 inside model) -> time_embed [B,3072]
ids = cat(txt_ids, img_ids) -> LongCatImagePosEmbed cos/sin
10 x LongCatImageTransformerBlock
20 x LongCatImageSingleTransformerBlock
AdaLayerNormContinuous -> Linear(3072 -> 64)
```

Dual-stream block:

```text
image: AdaLayerNormZero(hidden, temb) -> QKV
text:  AdaLayerNormZero(context, temb) -> added QKV
QK RMSNorm on both streams
concat text+image Q/K/V -> RoPE -> dispatch_attention_fn
split text/image outputs -> output projections
gated residual attention on both streams
LayerNorm -> adaptive scale/shift -> GELU FF -> gated residual on both streams
```

Single-stream block:

```text
cat(text, image)
AdaLayerNormZeroSingle -> normed states + gate
attention over concatenated stream, pre_only=True
parallel GELU MLP branch
cat(attn_output, mlp_output) -> Linear -> gate -> residual
split back to text/image
```

The `guidance` argument exists in pipeline/model signatures but current `LongCatImageTransformer2DModel.forward` does not
consume it. Embedded guidance is therefore not an active LongCat Image config surface; true CFG is the active guidance
mechanism.

## 7. Attention requirements

Primary implementation: `LongCatImageAttnProcessor` in `transformer_longcat_image.py`, which calls
`dispatch_attention_fn` from `attention_dispatch.py`.

Required behavior:

- Noncausal attention with Q/K/V shaped `[B,S,heads,head_dim]`.
- Official head shape: 24 heads, head dim 128.
- Q/K RMSNorm before RoPE.
- RoPE from three-axis ids: modality id, row/text position, column/text position, effective axes `[16,56,56]`.
- Dual blocks use separate image QKV plus text added-QKV projections, concatenate text first then image, and split after
  attention.
- Single blocks concatenate text and image before projection and split after the residual.
- Base/edit source does not pass an attention mask to the denoiser.
- Attention projection fusion is source-supported through `AttentionModuleMixin.fuse_projections()`, producing `to_qkv`
  and `to_added_qkv` when enabled.

Flash/provider feasibility:

- Base text-to-image at 1024 has attention length about `512 + 4096 = 4608`, head dim 128, no mask, noncausal. A
  Dinoml flash-style provider is plausible under dtype/sequence-length/workspace guards.
- Edit at 1024 can concatenate target and source image tokens, so image-side sequence is about 8192 plus text/vision
  prompt tokens; it is a separate stress case.
- QK RMSNorm and RoPE must remain explicit before the attention provider unless fused under exact preconditions.
- Mask-capable fallback should remain available for future prompt/variant paths even though base LongCat denoiser calls
  are mask-free.
- `attention_dispatch.py` supports native SDPA, flash-attn 2/3/hub, flash varlen, flex, sage, and xFormers backends, but
  many flash/sage paths reject masks. Use native/eager for parity and admit flash as a provider selection.

## 8. Scheduler and denoising-loop contract

Both base and edit pipelines synthesize default sigmas before calling the scheduler:

```text
sigmas = linspace(1.0, 1.0 / num_inference_steps, num_inference_steps)
image_seq_len = packed_target_latents.shape[1]
mu = calculate_shift(image_seq_len, base_image_seq_len, max_image_seq_len,
                     base_shift, max_shift)
scheduler.set_timesteps(num_inference_steps, sigmas=sigmas, mu=mu)
```

For 1024x1024 base, `image_seq_len=4096`, so the base/dev/edit `mu` is 1.15. The FlowMatch scheduler then applies
dynamic exponential time shift by source default. It appends a terminal sigma and stores sigmas on CPU.

Per-step loop:

```text
timestep = t.expand(batch).to(latents.dtype)
noise_pred_text = transformer(latents_or_edit_concat, timestep / 1000, prompt_embeds, ids)
if guidance_scale > 1:
  noise_pred_uncond = transformer(... negative embeddings ...)
  noise_pred = uncond + guidance_scale * (text - uncond)
  if base enable_cfg_renorm:
    noise_pred *= clamp(norm(text) / (norm(noise_pred) + 1e-8), cfg_renorm_min, 1)
latents = scheduler.step(noise_pred, t, latents)
```

Scheduler step math for the active non-stochastic scalar-timestep path:

```text
dt = sigma_next - sigma
prev_sample = sample + dt * model_output
```

Host/runtime split: keep schedule table generation, loop iteration, CFG branch orchestration, prompt rewrite, and edit
image preprocessing host-visible first. Compile one denoiser step plus pointwise FlowMatch update and CFG/renorm kernels
after tensor parity.

## 9. Position, timestep, and custom math

Custom math to reproduce:

- `split_quotation`: quoted prompt spans are tokenized one character at a time; unquoted spans are tokenized normally.
- LongCat prompt templates around the user prompt; base strips prefix/suffix and preserves the 512 padded user-token
  hidden states.
- Prompt rewrite language detection and Qwen2.5-VL generation. This is a separate generative text stage.
- 2x2 latent pack/unpack with source NCHW flatten order.
- Position ids: text modality 0, target image modality 1, edit source image modality 2. Image starts use the prompt
  length as both row and column offset.
- RoPE computed in float64 except MPS/NPU branches, then returned as cos/sin tensors.
- Timestep embedding: pipeline passes `t/1000`, model multiplies by 1000, then applies `Timesteps(256,
  flip_sin_to_cos=True, downscale_freq_shift=0)` and `TimestepEmbedding`.
- Edit size selection: `calculate_dimensions(1024*1024, input_width/input_height)` rounds width and height up to
  multiples of 16.
- VAE LongCat scaling: encode standardizes `(latent - shift_factor) * scaling_factor`; decode inverts with
  `(latent / scaling_factor) + shift_factor`.

Precomputable: prompt embeddings, image ids/text ids, RoPE frequencies for fixed prompt length and resolution, scheduler
tables, edit image latents. Dynamic: prompt length in edit, source image aspect ratio, CFG on/off, timestep, and prompt
rewrite output.

## 10. Preprocessing and input packing

Base text-to-image:

- Optional prompt rewrite builds language-specific system/user text and runs Qwen2.5-VL `.generate(max_new_tokens=512)`.
- `_encode_prompt` splits quotes, truncates user tokens to 512, pads to exactly 512, wraps with a fixed captioning
  prefix/suffix, and runs Qwen2.5-VL with `output_hidden_states=True`.
- It slices final hidden states from `prefix_len` to `-suffix_len`, yielding `[B,512,3584]`.
- Prompt embeds are repeated for `num_images_per_prompt`.
- Latents are sampled as NCHW `[B,16,H/8,W/8]`, then packed to `[B,S,64]`.

Edit:

- Input image is resized to a 1024-square-equivalent area preserving aspect ratio; another half-resolution image feeds
  Qwen2VL visual prompt encoding.
- Qwen2VL processor produces `pixel_values` and `image_grid_thw`; the image token placeholder is expanded to
  `prod(image_grid_thw) // merge_size**2` tokens.
- The final hidden state slice starts at the `vision_start` token and removes only the suffix, so visual tokens are part
  of `prompt_embeds`.
- The source image tensor is VAE-encoded, mode-selected, standardized, packed, and concatenated with the target noisy
  latent tokens.

Postprocessing:

- `output_type="latent"` returns packed latents.
- Otherwise the pipeline unpacks to NCHW VAE latents, applies scale/shift, casts to VAE dtype if needed, decodes through
  AutoencoderKL, and uses `VaeImageProcessor.postprocess`.

## 11. Graph rewrite / lowering opportunities

### Rewrite: LongCat latent pack/unpack

Source pattern: NCHW VAE latent map with even latent H/W, 2x2 spatial tile pack to `[B,S,64]`, and reverse before VAE.

Replacement: `longcat_pack2x2_nchw` and `longcat_unpack2x2_nchw`, or canonical reshape/permute/reshape nodes.

Preconditions: rank-4 NCHW, C=16 for official VAE, H/W divisible by 2 after VAE scale, flatten order exactly
`h2,w2,c,dh,dw`.

Failure cases: NHWC translation without matching flatten rewrite, user-supplied packed latents with unknown provenance,
future non-16-channel VAE configs.

Parity test: random `[B,16,128,128] <-> [B,4096,64]` round trip and exact comparison with Diffusers pack outputs.

### Rewrite: LongCat joint attention island

Source pattern: AdaLN -> QKV/add-QKV -> QK RMSNorm -> text/image concat -> RoPE -> noncausal attention -> split ->
projection -> gated residual.

Replacement: explicit joint-attention primitive with provider selection after QK RMSNorm/RoPE, optionally fused
projection input when `fuse_projections()` is applied.

Preconditions: known text and image spans, no adapter/IP branch, no attention mask, head dim 128, dtype/provider support.

Failure cases: edit long sequences exceeding provider limits, future masks, adapter processor mutation, mismatched RoPE
axes.

Parity test: one dual block and one single block with fixed random tensors; compare native dispatch and provider output.

### Rewrite: FlowMatch Euler LongCat slice

Source pattern: custom sigma list plus dynamic exponential shift, then pointwise Euler update.

Replacement: host-visible scheduler table plus fused pointwise update kernel.

Preconditions: `stochastic_sampling=false`, scalar timestep path, no per-token timesteps, explicit sigma index/state.

Failure cases: scheduler config enabling stochastic, Karras/exponential/beta conversions, or per-token timesteps.

Parity test: table comparison for base/dev/edit and Edit-Turbo configs, then one-step latent update.

### Rewrite: edit source-image conditioning pack

Source pattern: VAE encode -> LongCat standardize -> pack -> concat target/source tokens -> transformer -> slice target
tokens.

Replacement: explicit edit-conditioning stage with source-image token cache and target-token output slice.

Preconditions: one source image, same packed grid as target, source modality id 2, target modality id 1, no masks.

Failure cases: direct latent shortcut until `self.latent_channels` issue is resolved, multiple images/batch broadcasting
edge cases, aspect-ratio rounding mismatch.

Parity test: fixed PIL image through resize/preprocess/encode/pack/id construction and one transformer call.

## 12. Kernel fusion candidates

Highest priority:

- QKV/add-QKV projection + QK RMSNorm + RoPE + attention provider for head-dim-128 LongCat blocks.
- AdaLayerNorm modulation and gated residual epilogues around attention and MLP.
- GELU approximate FFN fusion, especially single-block `proj_mlp` + attention parallel branch + `proj_out`.
- Latent pack/unpack kernels with NCHW/NHWC guards.
- FlowMatch Euler step plus CFG and base CFG-renorm reductions.

Medium priority:

- Prompt/text id and image id generation as cached runtime metadata for fixed shapes.
- VAE decode Conv2d/GroupNorm/SiLU/resnet/mid-attention island, shared with Flux-like AutoencoderKL z=16 reports.
- Edit VAE encode and source-image pack/cache.
- Qwen2VL image processor/token expansion parity tests for edit.

Lower priority:

- Prompt rewrite generation; useful product feature but not first denoiser runtime.
- Component PEFT/LoRA hotswap/fuse/unfuse.
- Single-file original checkpoint conversion.
- VAE tiling/slicing overlap blend.

NHWC/layout candidates:

- Transformer core is token-major `[B,S,C]`; NHWC is not relevant inside the denoiser.
- VAE encode/decode are source NCHW Conv2d/GroupNorm regions. NHWC can be explored only inside controlled VAE conv islands
  with Conv2d weight transforms and GroupNorm channel-axis rewrite `dim=1 -> dim=-1`.
- Latent pack/unpack flatten order is axis-sensitive; protect it with a no-layout-translation guard until a layout-aware
  rewrite exists.
- Scheduler, CFG norm over token channels, prompt/text tokens, and RoPE ids should be guarded from image-layout
  translation.

## 13. Runtime staging plan

Stage 1: Parse LongCat model indexes and component configs. Use official component configs from Hub/cache and external
prompt embeddings first.

Stage 2: Implement base latent pack/unpack and id generation parity for multiple resolutions, including 1024x1024 and
the example 768x1344 shape.

Stage 3: Implement one `LongCatImageTransformerBlock` and one `LongCatImageSingleTransformerBlock` with random tensors,
including AdaLN gates, QK RMSNorm, RoPE, and native attention fallback.

Stage 4: Compile full `LongCatImageTransformer2DModel` for official 10+20 block shape with external prompt embeddings.

Stage 5: Add FlowMatch Euler dynamic exponential scheduler table and one-step parity.

Stage 6: Add true CFG two-call orchestration and base CFG renorm arithmetic.

Stage 7: Add VAE decode boundary with LongCat scale/shift. Keep VAE encode for edit as separate codec work if needed.

Stage 8: Admit `longcat_image_edit`: Qwen2VL image prompt path as external embeddings first, then VAE encode/source-token
cache and target/source token concat.

Stage 9: Optimize attention/norm/MLP and VAE conv islands; then handle prompt rewrite, LoRA/adapter state, and single-file
conversion separately.

First Dinoml admission recommendation: `longcat_image_base_denoiser_step_external_text`, with inputs
`latents [B,S,64]`, `prompt_embeds [B,512,3584]`, `timestep [B]`, `txt_ids [512,3]`, and `img_ids [S,3]`. Output is packed
latent derivative `[B,S,64]`. Keep prompt rewrite, Qwen2.5-VL text encoding, scheduler loop, and VAE decode outside the
compiled artifact until block/full-denoiser parity is stable.

## 14. Parity and validation plan

- Config parse tests for base, dev, edit, and edit-turbo model indexes plus official component configs.
- Prompt rewrite disabled/enabled boundary tests; first denoiser tests should use supplied prompt embeddings.
- Quote-sensitive tokenization and prefix/suffix slicing tests for base prompt embeddings.
- Edit prompt embedding shape tests with Qwen2VL image token placeholder expansion.
- Latent pack/unpack parity for `[B,16,128,128]`, non-square example shapes, and edit target/source token concat.
- Text/image id and RoPE parity for base and edit modality ids.
- Attention processor parity with no mask for dual and single blocks.
- Single block parity and then full transformer parity with random external embeddings.
- FlowMatch scheduler table and one-step parity for base dynamic shift and Edit-Turbo constant-shift settings.
- True CFG arithmetic and base CFG renorm parity.
- VAE decode parity for `[1,16,128,128]`; VAE encode/mode/standardize parity for edit.
- End-to-end smoke only after denoiser, scheduler, CFG, and VAE boundaries each have parity.
- Suggested tolerances: scheduler fp32 `rtol=1e-5, atol=1e-6`; transformer fp32 `rtol=1e-4, atol=1e-5`; bf16/fp16
  initially `rtol=2e-2, atol=2e-2`.

## 15. Performance probes

- One denoiser step by packed image sequence length: 1024, 4032, 4096, and edit's target+source token count.
- Attention backend comparison: native SDPA, Dinoml fallback, guarded flash-style provider.
- Per-block split: QKV/QK norm/RoPE/attention/projection versus MLP/AdaLN.
- Base CFG overhead: one call versus two transformer calls plus CFG renorm.
- Scheduler/CFG pointwise overhead versus denoiser time.
- Pack/unpack memory traffic and latency.
- VAE decode throughput for 1024 and non-square resolutions; edit VAE encode throughput.
- Prompt rewrite and Qwen2.5-VL text encoder throughput if admitted.
- VRAM/workspace by dtype, prompt length, and edit token count.

## 16. Scope boundary and separate candidates

Separate candidate reports, not ignored:

- `longcat_image_edit`: image-conditioned edit pipeline, VAE encode, visual prompt tokens, source latent concat.
- `longcat_image_edit_turbo`: same edit graph with distinct scheduler table config.
- `longcat_prompt_rewrite`: Qwen2.5-VL generation stage and system prompt behavior.
- `longcat_lora_adapters`: component PEFT/adapter mutation, fuse/unfuse, artifact-state policy.
- `longcat_single_file_original`: single-file/original checkpoint conversion.
- `longcat_vae_codec`: AutoencoderKL z=16 decode/encode, tiling/slicing, scale/shift contract.
- `scheduler_flowmatch_longcat`: dynamic exponential custom-sigma FlowMatch plus Edit-Turbo constant-shift regression.

Genuinely ignored/out of scope for this audit:

- Multi-GPU/context parallel paths.
- Callback mutation and interactive interrupt.
- XLA, NPU, MPS, Flax, and ONNX branches.
- Safety checker and NSFW filtering.
- Training, losses, dropout, and gradient checkpointing.
- Textual inversion, IP-Adapter, ControlNet, T2I-Adapter, GLIGEN, generic img2img, inpaint, depth2img, and upscaling:
  no active LongCat Image implementation was found in the inspected Diffusers folder.

Blockers / validation notes:

- Direct latent input to `LongCatImageEditPipeline.__call__` should be source-validated before admission because it
  references `self.latent_channels`, which is not defined by the pipeline class in this checkout.
- Local `H:/configs` only has LongCat model indexes. Component configs were readable from official Hub paths, but future
  agents may want to cache those JSON files under `H:/configs/meituan-longcat/*` once they own that write.

## 17. Final implementation checklist

- [ ] Parse LongCat Image model indexes and component configs.
- [ ] Accept external Qwen prompt embeddings for base denoiser parity.
- [ ] Implement LongCat 2x2 NCHW latent pack/unpack.
- [ ] Implement text/image modality ids and LongCat 3-axis RoPE parity.
- [ ] Implement timestep embedding path.
- [ ] Implement `LongCatImageAttnProcessor` native fallback parity.
- [ ] Implement QK RMSNorm + RoPE + joint attention.
- [ ] Implement dual-stream `LongCatImageTransformerBlock` AdaLN/gates/MLP/residual path.
- [ ] Implement single-stream `LongCatImageSingleTransformerBlock` concat/split path.
- [ ] Add full official transformer denoiser-step parity.
- [ ] Implement FlowMatch Euler dynamic exponential custom-sigma scheduler slice.
- [ ] Add true CFG two-call and base CFG-renorm parity.
- [ ] Add AutoencoderKL LongCat scale/shift decode boundary.
- [ ] Add edit-source VAE encode/pack/cache as a separate slice.
- [ ] Benchmark attention, denoiser step, scheduler/CFG, pack/unpack, and VAE decode.
- [ ] Open separate follow-ups for edit, prompt rewrite, LoRA/adapters, single-file conversion, and VAE codec work.
