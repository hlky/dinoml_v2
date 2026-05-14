# Diffusers HiDream Image Operator and Integration Report

Candidate slug: `hidream_image`

## 1. Source basis

```text
Diffusers commit/version:
  Local checkout X:/H/diffusers at b3a515080752a3ba7ca92161e25530c7f280f629.

Model id(s):
  HiDream-ai/HiDream-I1-Full
  HiDream-ai/HiDream-I1-Dev
  HiDream-ai/HiDream-I1-Fast
  HiDream-ai/HiDream-E1-Full
  HiDream-ai/HiDream-E1-1
  azaneko/HiDream-I1-{Full,Dev,Fast}-nf4, shuttleai/HiDream-I1-Full-FP8,
  deAPI-ai and Runware mirrors were checked as local model-index references only.

Config sources:
  H:/configs/HiDream-ai/HiDream-I1-Full/model_index.json
  H:/configs/HiDream-ai/HiDream-I1-Dev/model_index.json
  H:/configs/HiDream-ai/HiDream-I1-Fast/model_index.json
  H:/configs/HiDream-ai/HiDream-E1-Full/model_index.json
  H:/configs/HiDream-ai/HiDream-E1-1/model_index.json
  Official raw component configs were read with authenticated Hugging Face
  requests but not saved, because this worker owns only this report path.

Pipeline files inspected:
  X:/H/diffusers/src/diffusers/pipelines/hidream_image/pipeline_hidream_image.py
  X:/H/diffusers/src/diffusers/pipelines/hidream_image/__init__.py
  X:/H/diffusers/src/diffusers/pipelines/hidream_image/pipeline_output.py

Model files inspected:
  X:/H/diffusers/src/diffusers/models/transformers/transformer_hidream_image.py
  X:/H/diffusers/src/diffusers/models/autoencoders/autoencoder_kl.py

Scheduler/processors/helpers inspected:
  X:/H/diffusers/src/diffusers/schedulers/scheduling_flow_match_lcm.py
  X:/H/diffusers/src/diffusers/schedulers/scheduling_unipc_multistep.py
  X:/H/diffusers/src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py
  X:/H/diffusers/src/diffusers/image_processor.py
  X:/H/diffusers/src/diffusers/loaders/lora_pipeline.py
  X:/H/diffusers/src/diffusers/loaders/single_file_model.py
  X:/H/diffusers/src/diffusers/loaders/single_file_utils.py

External component configs inspected:
  Official HiDream CLIP-L, OpenCLIP-bigG, T5-XXL configs/tokenizer configs.
  `meta-llama/Meta-Llama-3.1-8B-Instruct` / `meta-llama/Llama-3.1-8B-Instruct`
  metadata was reachable but config/tokenizer raw reads returned 403 even with
  authenticated Hugging Face token. Llama dimensions are therefore taken from
  source usage plus open mirror `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit`.

Any missing files or assumptions:
  Official HiDream repos declare `text_encoder_4` and `tokenizer_4` in
  model_index.json but do not contain `text_encoder_4/config.json` or
  `tokenizer_4/tokenizer_config.json`; README instructs users to load gated
  Meta Llama 3.1 externally. Official E1 model indexes name
  `HiDreamImageEditingPipeline`, but this checkout exports only
  `HiDreamImagePipeline`; E1 is source-blocked for this non-deprecated Diffusers
  target and should be a separate candidate after the editing pipeline lands.
  This report focuses on base text-to-image. XLA/NPU/MPS/Flax/ONNX, safety,
  training/loss/dropout/gradient checkpointing, callbacks/interrupt, and
  multi-GPU/context-parallel paths are out of scope.
```

## 2. Pipeline and component graph

`HiDreamImagePipeline` wires `AutoencoderKL`, two `CLIPTextModelWithProjection`
encoders, `T5EncoderModel`, external `LlamaForCausalLM`, four tokenizers,
`HiDreamImageTransformer2DModel`, and either `UniPCMultistepScheduler` or
`FlowMatchLCMScheduler` depending on the checkpoint. The offload order is
`text_encoder->text_encoder_2->text_encoder_3->text_encoder_4->transformer->vae`.

```text
prompt / prompt_2 / prompt_3 / prompt_4
  -> CLIP-L pooled [B,768] + OpenCLIP-bigG pooled [B,1280]
  -> T5-XXL tokens [B,L,4096]
  -> Llama 3.1 hidden states stack [32,B,L,4096]
  -> latent noise [B,16,H/8,W/8] source NCHW
  -> transformer-internal 2x2 patchify [B,S,64] plus ids/masks
  -> denoising loop:
       HiDreamImageTransformer2DModel(latents, timestep, T5, Llama stack,
                                      pooled CLIP concat)
       negate transformer output, optional CFG chunk/arithmetic
       scheduler.step
  -> unpatchified latent map [B,16,H/8,W/8]
  -> AutoencoderKL decode((latents / 0.3611) + 0.1159)
  -> VaeImageProcessor postprocess
```

First-slice required components are the HiDream transformer denoiser, internal
patchify/unpatchify, external prompt embedding inputs, chosen scheduler step,
CFG batching/chunking, and VAE decode boundary. Prompt encoders are large and
cacheable; treat them as external inputs first.

Separate candidate reports:

| Surface | Classes/files | Status and runtime delta |
| --- | --- | --- |
| LoRA/runtime adapters | `HiDreamImageLoraLoaderMixin` in `loaders/lora_pipeline.py`; `PeftAdapterMixin` on transformer | Supported for transformer only. Adds load/fuse/unfuse/hotswap state and non-Diffusers key conversion. |
| Textual inversion | No HiDream textual inversion mixin on pipeline | Not supported by this folder. |
| IP-Adapter | No HiDream IP mixin or attention processor branch | Not supported in base source. |
| ControlNet | No HiDream ControlNet model/pipeline in checkout | Not supported in base source. |
| T2I-Adapter | No HiDream T2I pipeline | Not supported. |
| GLIGEN | No HiDream GLIGEN pipeline | Not supported. |
| img2img/inpaint/depth/upscale | No non-deprecated HiDream variants in checkout | Not supported in this Diffusers folder. |
| E1 editing | Official model index names `HiDreamImageEditingPipeline`, README references `pipeline_hidream_image_editing.py` | Separate candidate, blocked in this checkout because the class/file is absent. |
| Single-file/Comfy conversion | `single_file_model.py`, `single_file_utils.py` | Separate loader candidate for original checkpoint key mapping. |
| Quantized mirrors | nf4/fp8 model indexes in local cache | Separate quantization/encoded-weight candidate; not base operator semantics. |

## 3. Important config dimensions

Representative checkpoint sweep:

| Config | Pipeline class | Scheduler | Transformer shape | Max latent tokens | VAE | Notes |
| --- | --- | --- | --- | ---: | --- | --- |
| `HiDream-I1-Full` | `HiDreamImagePipeline` | UniPC flow, order 2, `flow_shift=3` | 16 double + 32 single blocks, 20 x 128 heads, MoE 4 routed / top-2 | 4096 | AutoencoderKL z=16, scale/shift | Full quality source-default. |
| `HiDream-I1-Dev` | `HiDreamImagePipeline` | FlowMatchLCM, `shift=6` | same transformer | 4096 | same | Distilled LCM-style stochastic step. |
| `HiDream-I1-Fast` | `HiDreamImagePipeline` | FlowMatchLCM, `shift=3` | same transformer | 4096 | same | Faster distilled variant. |
| `HiDream-E1-Full` / `E1-1` | `HiDreamImageEditingPipeline` in model index | UniPC flow, order 2 | same width/depth, `max_resolution=[96,192]` | 4608 | same | Source pipeline absent in checkout; separate blocked candidate. |

Transformer fields from official configs:

| Field | Value |
| --- | --- |
| `patch_size` | 2 |
| `in_channels` / `out_channels` | 16 / 16 VAE latent channels |
| token width before embed | `16 * 2 * 2 = 64` |
| `num_layers` / `num_single_layers` | 16 / 32 |
| heads x head dim | 20 x 128, inner dim 2560 |
| `caption_channels` | `[4096,4096]` for Llama/T5 projections |
| `text_emb_dim` | 2048 from concatenated CLIP pooled projections |
| `axes_dims_rope` | `[64,32,32]` over 3 id axes |
| `llama_layers` | 48 entries: layers 0-31, then layer 31 repeated 16 times |
| MoE | 4 routed experts, 2 activated experts, plus shared SwiGLU expert |

Text encoder configs:

| Component | Dimensions |
| --- | --- |
| CLIP-L | hidden/projection 768, 12 layers, 12 heads, max positions 248 in config; tokenizer config max length 77. |
| OpenCLIP-bigG | hidden/projection 1280, 32 layers, 20 heads, config max positions 218; tokenizer config max length 77. |
| T5-XXL encoder | `d_model=4096`, `d_ff=10240`, 24 layers, 64 heads, gated GELU, tokenizer max length 512. |
| Llama 3.1 8B external | hidden 4096, 32 layers, 32 heads, 8 KV heads, intermediate 14336, vocab 128256, max positions 131072 from open mirror; official Meta config is gated 403. |

VAE and scheduler:

| Component | Key fields |
| --- | --- |
| AutoencoderKL | `latent_channels=16`, block channels `[128,256,512,512]`, 2 layers per block, mid attention enabled, `force_upcast=true`, `scaling_factor=0.3611`, `shift_factor=0.1159`, no quant/post-quant conv. |
| UniPC | `use_flow_sigmas=true`, `prediction_type=flow_prediction`, `solver_order=2`, `solver_type=bh2`, `predict_x0=true`, `final_sigmas_type=zero`, `timestep_spacing=linspace`. |
| FlowMatchLCM | `shift=6` for Dev, `shift=3` for Fast, `use_dynamic_shifting=false`, exponential time shift config present but inactive, no scale factors. |

Recommended first Dinoml scheduler slice: UniPC flow-prediction for
`HiDream-I1-Full` if matching source-default quality, or FlowMatchLCM for
distilled Dev/Fast if a smaller first loop is acceptable. Do not call the family
complete until both scheduler families are represented.

## 3a. Family variation traps

- HiDream uses VAE latent maps `[B,16,H/8,W/8]`; unlike Flux/QwenImage, packing
  is model-internal, not a pipeline-level latent token contract.
- The pipeline rescales requested image size to preserve a 1024x1024-equivalent
  pixel budget, then rounds to `vae_scale_factor * 2 = 16`. Non-square inputs
  are padded inside the transformer to `max_seq`, with `hidden_states_masks`.
- `calculate_shift(self.transformer.max_seq)` is passed to FlowMatchLCM
  schedulers, but official Dev/Fast configs set `use_dynamic_shifting=false`, so
  `mu` is currently ignored. UniPC path bypasses `retrieve_timesteps` and does
  not pass `mu`.
- The pipeline constructor annotation/import names `FlowMatchEulerDiscreteScheduler`,
  while official I1 Dev/Fast model indexes use `FlowMatchLCMScheduler` and Full
  uses `UniPCMultistepScheduler`. Do scheduler admission from loaded config, not
  from the annotation alone.
- CFG is classic batch concatenation: latents, pooled CLIP, T5 embeddings, and
  Llama hidden-state stacks are concatenated on their respective batch axes.
- Transformer output is negated before guidance and scheduler step.
- Llama hidden states are a stack of all hidden layers except embedding output.
  The transformer indexes 48 layer slots even though the model has 32 layers,
  repeating final layer 31 for late blocks.
- MoE inference uses top-k, argsort, bincount to CPU NumPy, per-expert loops,
  and `scatter_reduce_`; this is a major graph/lowering trap.
- E1 configs are official but not source-admissible in this checkout because
  `HiDreamImageEditingPipeline` is absent.
- Official model repos do not include Llama weights/configs despite declaring
  `text_encoder_4`; artifact loading must bind an external gated component.

## 4. Runtime tensor contract

For the default 1024x1024 text-to-image case:

| Boundary | Tensor | Shape/layout | Notes |
| --- | --- | --- | --- |
| CLIP pooled 1 | `pooled_prompt_embeds_1` | `[B,768]` | From first CLIP text encoder output tuple index 0. |
| CLIP pooled 2 | `pooled_prompt_embeds_2` | `[B,1280]` | Concatenated with first CLIP to `[B,2048]`. |
| T5 tokens | `prompt_embeds_t5` | `[B,L,4096]`, default `L=128` | Max is min(requested, tokenizer model max). |
| Llama stack | `prompt_embeds_llama3` | `[32,B,L,4096]` before duplication; `[32,B*num_images,L,4096]` after | Produced by stacking `outputs.hidden_states[1:]`. |
| CFG embeddings | concat | pooled/T5 batch dim 0, Llama dim 1 | Negative then positive. |
| Initial latents | `latents` | `[B,16,128,128]` NCHW | Random normal or user-provided; dtype from pooled embeds. |
| Patch tokens | internal | `[B,4096,64]` for square 1024 | `view/permute/reshape` 2x2 patches. |
| Non-square patch tokens | internal | `[B,max_seq,64]` | Zero padded with mask `[B,max_seq]`; E1 max is 4608. |
| Image ids | `img_ids` | `[B,S,3]` | Axis 0 all zero, axes 1/2 are patch grid row/col. |
| Text ids | `txt_ids` | `[B,S_txt,3]`, zeros | Concatenated after image ids for RoPE. |
| Timestep | model input | `[B]` or `[2B]` under CFG | Expanded from scheduler timestep. |
| Denoiser output | `noise_pred` | `[B,16,128,128]` | Unpatchified by transformer. Pipeline negates it. |
| Scheduler state | timesteps/sigmas/history | scheduler-dependent | UniPC has multistep history; FlowMatchLCM samples fresh noise each step. |
| VAE decode input | latents | `[B,16,128,128]` | `(latents / scaling_factor) + shift_factor`. |
| Decoded image | VAE output | `[B,3,1024,1024]` NCHW | Postprocessed to PIL/NumPy. |

CPU/data-pipeline work: tokenization, text encoders if not cached, size
normalization, random seeding, PIL conversion. GPU/runtime work: denoiser,
CFG arithmetic, scheduler step, VAE decode, and eventually prompt encoders.

Cacheable stages: pooled CLIP embeddings, T5 embeddings, Llama hidden-state
stack, image/text ids and RoPE frequencies for fixed resolution/sequence
length, scheduler tables/history metadata per scheduler/step count.

Autoencoder encode is not used by the base text-to-image call, but img2img/edit
variants would require `AutoencoderKL.encode(image).latent_dist` with the same
scale/shift convention inverted around encode/decode.

## 5. Operator coverage checklist

Tensor/layout ops:

- NCHW latent allocation and user-latent shape validation.
- Size rescale/rounding to 16-pixel multiples.
- Patchify:
  square `reshape(B,C,pH,2,pW,2) -> permute(0,2,4,3,5,1) -> reshape(B,S,64)`;
  non-square pads a `[B,C,max_seq,4]` buffer before token reshape.
- Unpatchify:
  token `[B,S,64] -> reshape(B,pH,pW,2,2,16) -> permute(0,5,1,3,2,4) -> [B,16,H,W]`.
- Concats/chunks for CFG and text/image sequences.
- Top-k, argsort, bincount, gather/indexing, per-expert scatter-reduce for MoE.
- VAE NCHW decode and image postprocess.

GEMM/linear ops:

- Patch embed Linear(64 -> 2560).
- Timestep MLP 256 -> 2560 and pooled CLIP MLP 2048 -> 2560.
- 49 caption projections: 48 Llama slots Linear(4096 -> 2560), final T5
  Linear(4096 -> 2560), all bias-free.
- Per-block AdaLN modulation: double blocks Linear(2560 -> 30720), single
  blocks Linear(2560 -> 15360).
- Attention Q/K/V and text Q/K/V projections, width 2560.
- SwiGLU FFNs; image branch may be MoE with 4 routed experts + shared expert.
- Final AdaLN + Linear(2560 -> 64).

Attention primitives:

- 16 double-stream noncausal joint attention blocks over image tokens plus
  selected Llama/T5 text tokens.
- 32 single-stream blocks over concatenated image + persistent text tokens +
  one selected Llama layer.
- Q/K RMSNorm before RoPE.
- SDPA fallback via `torch.nn.functional.scaled_dot_product_attention`.
- Optional key masking for padded non-square image tokens by multiplying image
  keys by mask, not by an SDPA attention mask.

Normalization and conditioning:

- LayerNorm without affine, RMSNorm, AdaLN scale/shift/gates, SiLU.
- MoE gating softmax/top-k and weighted expert reduction.
- AutoencoderKL GroupNorm/SiLU/Conv2d/resnet/mid-attention blocks.

Scheduler/guidance arithmetic:

- `noise_pred = -transformer_output`.
- CFG: `uncond + guidance_scale * (text - uncond)`.
- UniPC flow-prediction multistep or FlowMatchLCM stochastic consistency step.

## 6. Denoiser/model breakdown

`HiDreamImageTransformer2DModel.forward`:

```text
latents [B,16,H,W] -> patchify -> [B,S,64], mask/img_ids/img_sizes
patch Linear -> image tokens [B,S,2560]
timestep embedding + pooled CLIP embedding -> temb [B,2560]
Llama hidden-state stack -> select configured layers -> Linear 4096->2560
T5 hidden states -> Linear 4096->2560
RoPE ids = cat(image ids, zero text ids)
16 double-stream blocks
concat image with persistent text tokens
32 single-stream blocks, each temporarily appending one Llama layer
slice image tokens -> final AdaLN + Linear -> [B,S,64] -> unpatchify
```

Double-stream block:

```text
temb -> 12-way modulation for image/text attention and MLP
image/text LayerNorm -> scale/shift
image QKV + text QKV -> Q/K RMSNorm -> concat -> RoPE -> SDPA -> split
gated residual attention on both streams
image LayerNorm -> scale/shift -> MoE or SwiGLU -> gated residual
text LayerNorm -> scale/shift -> SwiGLU -> gated residual
```

Single-stream block:

```text
temb -> 6-way modulation
LayerNorm -> scale/shift
single-stream QKV -> Q/K RMSNorm -> RoPE -> SDPA -> output projection
gated residual attention
LayerNorm -> scale/shift -> MoE/SwiGLU -> gated residual
```

MoE image FFN:

```text
gate = softmax(linear(tokens, gate_weight))
topk_idx, topk_weight = topk(gate, k=2, sorted=False)
for each expert: run SwiGLU on selected tokens
scatter_reduce sum weighted expert outputs
add shared SwiGLU expert output
```

## 7. Attention requirements

Primary implementation is `HiDreamAttnProcessor` local to
`transformer_hidream_image.py`, not the shared `attention_dispatch.py` registry.
It uses PyTorch SDPA directly.

Required behavior:

- Noncausal attention with Q/K/V shaped `[B,S,20,128]`.
- Double-stream attention concatenates image then text sequence and splits
  after attention.
- Single-stream attention runs on an already concatenated token stream.
- Q and K RMSNorm operate on the flattened inner dim before reshaping to heads.
- RoPE is applied after image/text concat. If head dim equals RoPE dim, all head
  channels rotate; otherwise the tensor is split and only the first half rotates.
- Non-square image padding masks multiply image keys before concat; no additive
  attention mask is supplied to SDPA.
- No IP-Adapter added-KV branch, no cross-attention mask, no causal mask.

Flash feasibility:

- Base square 1024 path is a plausible flash-style provider target: no mask,
  head dim 128, noncausal, sequence lengths around image 4096 plus text.
- Non-square padded path is not equivalent to a standard mask because source
  only zeroes image keys. Preserve eager parity first or add an exact mask/key
  zeroing precondition.
- QK RMSNorm and RoPE must remain explicit pre-attention ops unless a provider
  accepts them as fused inputs with strict parity tests.
- MoE and text-layer appending make block-level sequence lengths dynamic; do
  not infer a single fixed joint attention shape from one block.

## 8. Scheduler and denoising-loop contract

Pipeline loop:

```text
height/width normalized
prompt embeddings encoded/duplicated; CFG concatenates negative + positive
latents sampled [B,16,H/8,W/8]
if UniPC: scheduler.set_timesteps(num_inference_steps)
else: retrieve_timesteps(..., sigmas=sigmas, mu=calculate_shift(transformer.max_seq))
for t in timesteps:
  latent_model_input = cat([latents]*2) when CFG else latents
  timestep = t.expand(batch)
  noise_pred = -transformer(...)
  if CFG: noise_pred = uncond + scale * (text - uncond)
  latents = scheduler.step(noise_pred, t, latents)
decode or return latent
```

`UniPCMultistepScheduler` official Full/E1 config uses flow sigmas and
`prediction_type=flow_prediction`, so the scheduler converts
`x0 = sample - sigma * model_output` and applies second-order UniPC `bh2`
history updates with lower-order warmup/final behavior.

`FlowMatchLCMScheduler` Dev/Fast config uses static shifted sigmas and a
stochastic update:

```text
x0_pred = sample - sigma * model_output
noise = randn_like(x0_pred)
prev = (1 - sigma_next) * x0_pred + sigma_next * noise
```

Keep schedule generation, multistep history, random noise source, and loop
iteration host-visible first. Candidate compiled kernels are CFG arithmetic,
model-output negation, and the pointwise parts of scheduler updates.

## 9. Position, timestep, and custom math

Custom math to reproduce:

- `calculate_shift(max_seq)` from the pipeline passes a resolution shift value
  into non-UniPC schedulers; currently inert for sampled Dev/Fast configs.
- Timestep embedding uses `Timesteps(256, flip_sin_to_cos=True,
  downscale_freq_shift=0)` followed by a `TimestepEmbedding` MLP.
- CLIP pooled embeddings are concatenated to 2048 and run through another
  `TimestepEmbedding`-style MLP, then added to timestep embedding.
- 3-axis RoPE builds `[cos,-sin,sin,cos]` blocks with float64 on CUDA/CPU in
  source and concatenates axes `[64,32,32]`.
- Llama hidden-state routing: configured `llama_layers` list supplies one Llama
  layer per double/single block plus repeated final layer slots.
- MoE routing must match `topk(sorted=False)`, optional top-k normalization
  disabled, per-expert weighted scatter-sum, and shared expert addition.
- VAE decode uses `(latents / 0.3611) + 0.1159`; encode variants should invert
  that convention.

Precompute for fixed request: RoPE ids/frequencies, text embeddings,
caption-projection outputs if text is cached after projection, scheduler tables.
Dynamic: prompt length, Llama stack, CFG on/off, non-square size/mask, timestep,
MoE routing, and FlowMatchLCM random noise.

## 10. Preprocessing and input packing

Prompt preprocessing:

- `prompt_2`, `prompt_3`, `prompt_4` default to `prompt`; negative variants
  default to `negative_prompt`.
- CLIP helper truncates to `min(max_sequence_length, 218)` even though tokenizer
  configs advertise 77. It uses output tuple index 0 as pooled text output.
- T5 helper truncates to `min(max_sequence_length, tokenizer_3.model_max_length)`.
- Llama helper requests `output_hidden_states=True` and `output_attentions=True`,
  stacks hidden states `[1:]`, and does not cast to dtype explicitly in helper.
- `num_images_per_prompt` duplication differs by tensor rank: Llama duplicates
  on stack batch dimension then reshapes to `[layers,B*num_images,L,D]`.

Latent preprocessing:

- Pipeline chooses default `height=width=128*vae_scale_factor=1024`.
- It rescales arbitrary requested dimensions to keep the same area budget and
  rounds both axes down to multiples of 16.
- `prepare_latents` allocates `[B,16,H/8,W/8]`; the transformer owns patchify.
- Non-square latent maps are padded to `self.max_seq` inside the transformer and
  carry a mask through attention.

Postprocess:

- If `output_type="latent"`, return scheduler latents before VAE scale/shift.
- Otherwise decode through AutoencoderKL and `VaeImageProcessor.postprocess`.

## 11. Graph rewrite / lowering opportunities

### Rewrite: HiDream patchify/unpatchify

Source pattern: model-internal 2x2 NCHW patchify/unpatchify around latent maps.

Replacement: canonical `patchify2x2_nchw` / `unpatchify2x2_nchw` or explicit
reshape/permute graph.

Preconditions: rank-4 NCHW, channels 16, H/W divisible by 2, max-token padding
only for non-square path, flatten order exactly as source.

Failure cases: NHWC translation without matching flatten rewrite, non-square
mask elision, future editing pipeline with additional conditioning latents.

Parity test: round trip square and non-square latent maps; compare masks and ids.

### Rewrite: HiDream attention provider island

Source pattern: Q/K/V projection -> RMSNorm -> optional key mask -> concat ->
RoPE -> SDPA -> split/projection.

Replacement: explicit attention subgraph with native fallback and guarded flash
provider for square/no-mask case.

Preconditions: head dim 128, no adapter branches, no non-square key mask unless
provider reproduces key-zero behavior, static split sizes.

Failure cases: padded tokens, future IP/Control branches, provider applying
standard additive masks instead of source key multiplication.

Parity test: one double-stream block and one single-stream block with square
and padded non-square inputs.

### Rewrite: MoE expert routing

Source pattern: top-k softmax gate, per-expert token bucketing, expert SwiGLU,
weighted scatter-reduce, shared expert add.

Replacement: initially a reference MoE runtime region; later grouped GEMM or
block-sparse expert dispatch.

Preconditions: inference mode, top_k=2, four experts, fixed hidden size 2560,
token routing visible.

Failure cases: training aux loss, top-k tie instability, CPU NumPy bincount in
source, dynamic shapes without a routing workspace.

Parity test: random tokens with forced gate weights covering empty and nonempty
experts.

### Rewrite: FlowMatchLCM/UniPC scheduler kernels

Source pattern: pointwise flow conversion and update over latent maps.

Replacement: host scheduler tables + compiled per-step update kernels.

Preconditions: explicit scheduler family, sigma index, random noise tensor for
LCM, UniPC history buffers visible.

Failure cases: hidden scheduler mutation, stochastic noise not supplied,
custom sigmas not validated by family.

## 12. Kernel fusion candidates

Highest priority:

- HiDream patchify/unpatchify kernels and non-square mask/id generation.
- QKV + RMSNorm + RoPE + attention provider path for square base inference.
- AdaLN modulation + gated residual epilogues around attention and MLP.
- MoE routing plus expert SwiGLU dispatch; this is the distinctive hot path.
- CFG arithmetic plus model-output negation.

Medium priority:

- Caption projection caching/fusion for 48 Llama slots plus T5.
- Timestep + pooled CLIP embedding MLP.
- UniPC flow and FlowMatchLCM scheduler pointwise updates.
- AutoencoderKL 16-channel decode conv/resnet/attention island.

Lower priority:

- LoRA hotswap/fuse/unfuse and non-Diffusers LoRA conversion.
- Single-file original checkpoint conversion.
- Text encoder compilation; embeddings can be supplied externally first.
- E1 editing path after source lands.

NHWC/layout candidates:

- Source latent/VAE tensors are NCHW. NHWC may be profitable inside VAE conv
  islands only with Conv2d weight transforms and GroupNorm axis rewrite
  `dim=1 -> dim=-1`.
- Transformer core is token-major `[B,S,C]`; NHWC does not apply.
- Patchify/unpatchify flatten order is axis-sensitive and should be protected by
  a no-layout-translation guard until a layout-aware rewrite exists.

## 13. Runtime staging plan

Stage 1: Parse official HiDream configs and support external Llama component
binding. Use cached prompt embeddings for first denoiser work.

Stage 2: Implement patchify/unpatchify, image ids, non-square masks, and RoPE
frequency parity.

Stage 3: Implement one double-stream block with explicit T5/Llama projected
inputs, RMSNorm/RoPE/SDPA fallback, AdaLN gates, and image/text FFNs.

Stage 4: Implement one single-stream block and MoE inference parity.

Stage 5: Compile full `HiDreamImageTransformer2DModel` for official I1 shape at
small batch with external text embeddings.

Stage 6: Add source-default `HiDream-I1-Full` UniPC flow scheduler one-step and
short-loop parity.

Stage 7: Add FlowMatchLCM Dev/Fast stochastic scheduler with explicit noise
inputs and deterministic generator parity.

Stage 8: Add VAE decode boundary, then optional AutoencoderKL decode artifact.

Stage 9: Add LoRA/single-file/quantized variants as separate admission slices.

First admission recommendation: `hidream_i1_denoiser_step`, with inputs
`latents [B,16,H,W]`, `timestep [B]`, T5 tokens `[B,L,4096]`, Llama stack
`[32,B,L,4096]`, pooled CLIP `[B,2048]`, and optional precomputed patch ids/mask.
Output is latent derivative `[B,16,H,W]` before pipeline negation.

## 14. Parity and validation plan

- Config parse tests for Full, Dev, Fast, and E1 blocked-source metadata.
- External Llama binding test: fail clearly when gated component is absent.
- CLIP/T5/Llama prompt duplication parity with CFG and `num_images_per_prompt`.
- Patchify/unpatchify parity for square `[1,16,128,128]` and non-square cases.
- RoPE parity for `[64,32,32]` axes and max_seq 4096/4608.
- Attention processor parity for double and single blocks.
- MoE routing parity with controlled expert selection and empty experts.
- Full transformer parity with random embeddings and fixed latents/timestep.
- `noise_pred = -transformer_output` and CFG chunk arithmetic parity.
- UniPC flow scheduler table/step/history parity for Full.
- FlowMatchLCM step parity with supplied deterministic noise for Dev/Fast.
- AutoencoderKL decode parity for `[1,16,128,128]`; encode parity reserved for
  future edit/img2img candidates.
- Suggested tolerances: scheduler fp32 `rtol=1e-5, atol=1e-6`; transformer fp32
  `rtol=1e-4, atol=1e-5`; fp16/bf16 initially `rtol=2e-2, atol=2e-2`.

## 15. Performance probes

- One denoiser step at 512, 768, 1024, and non-square E1-like token counts.
- Block time split: attention vs MoE FFN vs caption projections.
- MoE routing overhead and grouped-expert GEMM utilization.
- Attention backend comparison: eager SDPA, Dinoml native, guarded flash.
- Scheduler overhead: UniPC history update vs FlowMatchLCM stochastic step.
- CFG overhead: one batch vs two-batch latents and text embeddings.
- VAE decode throughput for 1024 with 16 latent channels.
- Text encoder throughput and cache memory: CLIP-L, OpenCLIP-bigG, T5-XXL,
  gated Llama 8B hidden-state stack.
- VRAM/workspace by dtype and prompt length; transformer weights alone are
  about 34.2 GB by safetensors index metadata.

## 16. Scope boundary and separate candidates

Separate candidate reports, not ignored:

- `hidream_image_lora_adapters`: transformer LoRA loading, hotswap, fuse/unfuse,
  DoRA filtering, and non-Diffusers HiDream LoRA conversion.
- `hidream_image_single_file`: original checkpoint conversion and default
  component binding from `single_file_utils.py`.
- `hidream_image_quantized`: nf4/fp8 mirror repos and encoded-weight/runtime
  loading policy.
- `hidream_e1_editing`: official editing model indexes and README examples, but
  blocked until `HiDreamImageEditingPipeline` source is available in Diffusers.
- `hidream_scheduler_matrix`: UniPC flow and FlowMatchLCM coverage, including
  stochastic noise and optional scale-factor upscaling branch.
- `hidream_vae_decode`: AutoencoderKL 16-channel decode island shared with
  Flux-like models.

Genuinely ignored/out of scope for this audit:

- Multi-GPU/context parallel paths.
- Callback mutation and interactive interrupt.
- XLA, NPU, MPS, Flax, and ONNX variants.
- Safety checker and NSFW filtering.
- Training, losses, dropout, and gradient checkpointing.
- Textual inversion, IP-Adapter, ControlNet, T2I-Adapter, GLIGEN, img2img,
  inpaint, depth2img, and upscaling: no active non-deprecated HiDream source
  implementation was found in this checkout.

## 17. Final implementation checklist

- [ ] Parse HiDream model indexes and component configs.
- [ ] Add explicit external Llama 3.1 component binding/error reporting.
- [ ] Accept external CLIP pooled, T5 token, and Llama hidden-stack embeddings.
- [ ] Implement HiDream NCHW 2x2 patchify/unpatchify.
- [ ] Implement non-square max-seq padding masks and image/text ids.
- [ ] Implement HiDream 3-axis RoPE.
- [ ] Implement timestep + pooled CLIP embedding path.
- [ ] Implement caption projection and Llama layer routing.
- [ ] Implement `HiDreamAttnProcessor` fallback parity.
- [ ] Implement AdaLN modulation/gated residuals.
- [ ] Implement SwiGLU and MoE top-k expert routing.
- [ ] Add double-stream and single-stream block parity.
- [ ] Add full transformer denoiser-step parity.
- [ ] Implement `noise_pred = -output` and CFG arithmetic.
- [ ] Implement UniPC flow scheduler slice for I1-Full.
- [ ] Implement FlowMatchLCM scheduler slice for I1-Dev/Fast.
- [ ] Add AutoencoderKL scale/shift decode boundary.
- [ ] Benchmark attention, MoE, scheduler, and VAE decode.
- [ ] Open separate reports for LoRA, single-file, quantized, E1 editing, and
  HiDream scheduler variants.
