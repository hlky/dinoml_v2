# Diffusers Shap-E Pipeline Audit

Candidate slug: `shap_e`

## 1. Source basis

```text
Diffusers commit/version:
  Local checkout X:/H/diffusers at b3a515080752a3ba7ca92161e25530c7f280f629.

Model id(s):
  openai/shap-e
  openai/shap-e-img2img

Config sources:
  Local cache had model indexes only:
    H:/configs/openai/shap-e/model_index.json
    H:/configs/openai/shap-e-img2img/model_index.json
  Official raw Hugging Face configs inspected transiently, not saved because this
  worker owns only this report path:
    openai/shap-e: model_index.json, prior/config.json,
      scheduler/scheduler_config.json, text_encoder/config.json,
      tokenizer/tokenizer_config.json, renderer/config.json,
      shap_e_renderer/config.json
    openai/shap-e-img2img: model_index.json, prior/config.json,
      scheduler/scheduler_config.json, image_encoder/config.json,
      image_processor/preprocessor_config.json, renderer/config.json,
      shap_e_renderer/config.json
  Safetensors header metadata inspected by HTTP range reads:
    openai/shap-e prior/diffusion_pytorch_model.fp16.safetensors
    openai/shap-e text_encoder/model.fp16.safetensors
    openai/shap-e renderer/diffusion_pytorch_model.fp16.safetensors
    openai/shap-e-img2img prior/diffusion_pytorch_model.fp16.safetensors
    openai/shap-e-img2img image_encoder/model.fp16.safetensors
    openai/shap-e-img2img renderer/diffusion_pytorch_model.fp16.safetensors

Pipeline files inspected:
  X:/H/diffusers/src/diffusers/pipelines/shap_e/pipeline_shap_e.py
  X:/H/diffusers/src/diffusers/pipelines/shap_e/pipeline_shap_e_img2img.py
  X:/H/diffusers/src/diffusers/pipelines/shap_e/__init__.py

Model files inspected:
  X:/H/diffusers/src/diffusers/models/transformers/prior_transformer.py
  X:/H/diffusers/src/diffusers/models/attention.py
  X:/H/diffusers/src/diffusers/models/attention_processor.py
  X:/H/diffusers/src/diffusers/models/activations.py
  X:/H/diffusers/src/diffusers/models/embeddings.py

Scheduler/processors/helpers inspected:
  X:/H/diffusers/src/diffusers/schedulers/scheduling_heun_discrete.py
  X:/H/diffusers/src/diffusers/pipelines/shap_e/renderer.py
  X:/H/diffusers/src/diffusers/pipelines/shap_e/camera.py
  X:/H/diffusers/src/diffusers/utils/pil_utils.py

External component configs inspected:
  CLIPTextModelWithProjection text config from openai/shap-e/text_encoder/config.json.
  CLIPTokenizer config from openai/shap-e/tokenizer/tokenizer_config.json.
  CLIPVisionModel config from openai/shap-e-img2img/image_encoder/config.json.
  CLIPImageProcessor config from openai/shap-e-img2img/image_processor/preprocessor_config.json.

Any missing files or assumptions:
  No gated configs blocked the audit. The `shap_e_renderer` folder exposes a
  PyTorch `.bin` but no fp16 safetensors file; the legacy `renderer` folder has
  fp16 safetensors. The model index declares both `renderer` and
  `shap_e_renderer`; current pipeline constructors register only
  `shap_e_renderer`. Mesh decoder lookup-table buffers are initialized in source
  as zero tensors, and the inspected `renderer/*.fp16.safetensors` headers did
  not contain `mesh_decoder.cases` or `mesh_decoder.masks`; mesh parity should
  verify the `.bin` path before relying on fp16-safetensors renderer artifacts.
```

## 2. Pipeline and component graph

Shap-E is not an image-latent VAE pipeline. The diffusion model denoises a
sequence of 3D latent tokens, then a renderer converts those tokens into either
20 rendered NeRF views or a triangle mesh.

```text
text-to-3D:
  prompt strings
  -> CLIPTokenizer(max_length=77, pipeline forces pad_token_id=0)
  -> CLIPTextModelWithProjection.text_embeds [B,768]
  -> normalize and scale prompt embeddings, optional CFG zero embedding
  -> Heun scheduler over prior latents [B,1024,1024]
  -> PriorTransformer self-attention denoiser
  -> renderer parameter projection and NeRF/STF MLP
  -> [B,20,H,W,3] rendered views, latent tokens, or MeshDecoderOutput

image-to-3D:
  PIL/np/torch image
  -> CLIPImageProcessor resize/center-crop/normalize to [B,3,224,224]
  -> CLIPVisionModel.last_hidden_state[:,1:,:] [B,256,1024]
  -> optional CFG zero image patch embeddings
  -> same Heun/PriorTransformer latent denoising
  -> same renderer outputs
```

Required components:

| Pipeline | Source class/file | Required components | Cacheable boundaries |
| --- | --- | --- | --- |
| Text-to-3D | `ShapEPipeline`, `pipeline_shap_e.py` | `PriorTransformer`, `CLIPTextModelWithProjection`, `CLIPTokenizer`, `HeunDiscreteScheduler`, `ShapERenderer` | token ids, prompt embeddings, scheduler tables, initial latents, final 3D latent tokens |
| Image-to-3D | `ShapEImg2ImgPipeline`, `pipeline_shap_e_img2img.py` | `PriorTransformer`, `CLIPVisionModel`, `CLIPImageProcessor`, `HeunDiscreteScheduler`, `ShapERenderer` | processed image pixels, image patch embeddings, scheduler tables, initial latents, final 3D latent tokens |

Pipeline metadata:

| Field | Text pipeline | Image pipeline |
| --- | --- | --- |
| `model_cpu_offload_seq` | `text_encoder->prior` | `image_encoder->prior` |
| `_exclude_from_cpu_offload` | `["shap_e_renderer"]` | `["shap_e_renderer"]` |
| Output types | `np`, `pil`, `latent`, `mesh` | `np`, `pil`, `latent`, `mesh` |
| Default `guidance_scale` | source call default `4.0`; doc example uses `15.0` | source call default `4.0`; doc example uses `3.0` |

Separate candidate reports and extension surfaces:

| Surface | Shap-E support | Candidate note |
| --- | --- | --- |
| img2img | Present as `ShapEImg2ImgPipeline`; image encoder conditioning replaces text conditioning | Include in this report as the second first-class Shap-E target |
| LoRA/runtime adapters | `PriorTransformer` inherits `UNet2DConditionLoadersMixin` and `PeftAdapterMixin`, and attention processors can be mutated | Separate `shap_e_prior_adapters` candidate if adapter loading is admitted; not part of first parity |
| textual inversion | No Shap-E textual inversion pipeline mixin; text path uses plain CLIP tokenizer | Not supported as a family surface in inspected source |
| IP-Adapter | No Shap-E IP-Adapter loader or added-K/V image branch | Not supported |
| ControlNet | No Shap-E ControlNet pipeline/model branch | Not supported |
| T2I-Adapter | No adapter feature-pyramid branch | Not supported |
| GLIGEN | No grounded attention branch in active pipeline; GLIGEN code in `BasicTransformerBlock` is inactive | Not supported for Shap-E |
| inpaint/depth2img/upscaling | No Shap-E pipeline variants in non-deprecated folder | Not supported |
| mesh renderer | Present via `ShapERenderer.decode_to_mesh` and `MeshDecoder` | Separate `shap_e_renderer_mesh` candidate if marching cubes and dynamic mesh outputs become runtime targets |

## 3. Important config dimensions

Representative checkpoint sweep:

| Repo | Pipeline | Conditioning | Prior hidden | Heads x head dim | Layers | Conditioning proj | Prior output | Scheduler | Renderer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `openai/shap-e` | `ShapEPipeline` | CLIP text `text_embeds` [B,768] | 1024 | 16 x 64 | 24 | `embedding_proj_dim=768`, no proj norm | [B,1024,2048], split to [B,1024,1024] sample plus discarded variance | Heun, Karras, sample prediction | `d_latent=1024`, NeRF/STF MLP |
| `openai/shap-e-img2img` | `ShapEImg2ImgPipeline` | CLIP ViT patch tokens [B,256,1024] | 1024 | 8 x 128 | 24 | `embedding_proj_dim=1024`, LayerNorm before projection | [B,1024,2048], split to [B,1024,1024] sample plus discarded variance | Heun, Karras, sample prediction | same renderer config |

Prior config details:

| Field | Text-to-3D | Image-to-3D | Source/default note |
| --- | --- | --- | --- |
| `embedding_dim` | 1024 | 1024 | latent token width |
| `num_embeddings` | 1024 | 1024 | latent token count |
| `clip_embed_dim` | 2048 | 2048 | prior output width before pipeline split |
| `num_layers` | 24 | 24 | `BasicTransformerBlock` stack |
| `time_embed_dim` | 4096 | 4096 | timestep MLP hidden/output path |
| `time_embed_act_fn` | `gelu` | `gelu` | differs from source default `silu` |
| `added_emb_type` | null | null | disables source default `prd` token |
| `additional_embeddings` | 0 | 0 | source default is 4 |
| `encoder_hid_proj_type` | null | null | no separate encoder hidden states in pipeline |
| `norm_in_type` | `layer` | `layer` | source default is null |
| `embedding_proj_norm_type` | null | `layer` | image path normalizes CLIP patch tokens |

External CLIP configs:

| Component | Class | Key dims |
| --- | --- | --- |
| Text encoder | `CLIPTextModelWithProjection` | hidden 768, projection 768, 12 layers, 12 heads, max position 77, vocab 49408 |
| Tokenizer | `CLIPTokenizer` | `model_max_length=77`; tokenizer config pad token is end-of-text, but Shap-E source sets `pad_token_id=0` before tokenization |
| Image encoder | `CLIPVisionModel` | 224 image, patch size 14, 16 x 16 = 256 patch tokens plus CLS, hidden 1024, 24 layers, 16 heads |
| Image processor | `CLIPImageProcessor` | resize 224, center crop 224, normalize with CLIP mean/std, input/output pixel layout [B,3,224,224] |

Scheduler config:

| Field | Value | Effective default note |
| --- | --- | --- |
| Class | `HeunDiscreteScheduler` | pipeline constructor type is fixed to Heun |
| `num_train_timesteps` | 1024 | source default 1000 |
| `beta_schedule` | `exp` | source default `linear` |
| `prediction_type` | `sample` | source supports `epsilon`, `v_prediction`, `sample` |
| `use_karras_sigmas` | true | source default false |
| `clip_sample` | true | source default false |
| `clip_sample_range` | 1.0 | source default 1.0 |
| omitted fields | `beta_start=0.00085`, `beta_end=0.012`, `timestep_spacing="linspace"`, `steps_offset=0`, no exponential/beta sigmas | source defaults apply |

Renderer config:

| Field | Value | Runtime implication |
| --- | --- | --- |
| `param_shapes` | `(256,93)`, `(256,256)`, `(256,256)`, `(256,256)` | latent token ranges project dynamic weights for MLP layers 0-3 |
| `d_latent` | 1024 | every projection consumes latent width 1024 |
| `d_hidden` | 256 | NeRF/STF MLP hidden width |
| `n_hidden_layers` | 6 | MLP has 7 Linear layers including final output |
| `n_output` | 12 | channels map to SDF, coarse/fine densities, STF RGB, coarse/fine NeRF RGB |
| `insert_direction_at` | 4 | layer 4 receives extra 51 encoded direction channels |
| `background` | white `[255,255,255]` | `VoidNeRFModel` background normalized by 255 |

Weight precision metadata:

| Artifact | Header fact |
| --- | --- |
| Prior fp16 safetensors | text prior: 401 tensors, F16 only; image prior: 403 tensors, F16 only; both include `clip_mean` and `clip_std` [1,2048], although pipeline does not call `post_process_latents` |
| Text encoder fp16 safetensors | 198 tensors, F16 plus I64 `position_ids` |
| Image encoder fp16 safetensors | 392 tensors, F16 plus I64 `position_ids` |
| Renderer fp16 safetensors | 31 tensors, F16 only in legacy `renderer` folder; no mesh LUT buffers observed in header |

Recommended first Dinoml scheduler slice: implement HeunDiscrete with Karras
sigmas and `prediction_type="sample"` for exact Shap-E parity. Unlike Stable
Diffusion, this family does not present a broad scheduler-compatible pipeline
surface in the source constructor.

## 3a. Family variation traps

- This is a 3D latent-token diffusion pipeline, not a NCHW image-latent UNet or
  VAE pipeline. First-slice runtime tensors are mostly [B, sequence, channel].
- Text and image variants share the renderer and scheduler but change the
  prior attention geometry: text sequence length is 1026, image sequence length
  is 1281.
- Prior config disables the source default PRD token (`added_emb_type=null`);
  do not carry unCLIP PRD behavior into Shap-E parity.
- The prior emits 2048 channels per latent token; pipeline splits the last
  dimension at 1024 and discards the second half as variance.
- `PriorTransformer.post_process_latents()` is present but inactive in both
  Shap-E pipelines.
- Text pipeline mutates `tokenizer.pad_token_id = 0` at prompt-encode time.
- Renderer mutates `self.mlp` layer weights with generated parameters for each
  latent. This is hidden mutable model state in source and should become an
  explicit dynamic-weight renderer boundary for Dinoml.
- Renderer image output is NHWC per frame: `[20, frame_size, frame_size, 3]`
  per sample. This should not be confused with CLIP image input NCHW.
- `decode_to_image` batches rays with `n_batches = rays.shape[1] // 4096`;
  non-divisible `20 * frame_size * frame_size` values would drop remainder rays
  in source.
- `StratifiedRaySampler.sample` calls `torch.manual_seed(0)` internally and
  `ImportanceRaySampler.sample` calls `torch.rand` without a pipeline generator;
  renderer determinism is source-coupled Python randomness, not scheduler
  generator state.
- Mesh output has dynamic vertex/face counts and source uses marching-cubes
  indexing, `torch.unique`, boolean masks, and gather/scatter. It is not a
  fixed-shape tensor output.

## 4. Runtime tensor contract

Pipeline inputs:

| Boundary | Tensor contract |
| --- | --- |
| Text prompt | string/list on CPU; tokenizer pads/truncates to 77 ids |
| Text embedding | `text_encoder(...).text_embeds` [B,768], repeated by `num_images_per_prompt`, L2-normalized over dim -1, multiplied by `sqrt(768)`. CFG prepends a zero tensor to produce [2B,768] |
| Image input | PIL/list/tensor; processor produces [B,3,224,224] NCHW float pixels for CLIP |
| Image embedding | `CLIPVisionModel.last_hidden_state[:,1:,:]` [B,256,1024], repeated by `num_images_per_prompt`. CFG prepends zeros to [2B,256,1024] |
| Initial prior latents | Gaussian [B,1024*1024], dtype from embeddings, scaled by `scheduler.init_noise_sigma`, then reshaped to [B,1024,1024] |

Denoising step:

```text
latents [B,1024,1024]
  -> CFG concat [2B,1024,1024] when guidance_scale > 1
  -> scheduler.scale_model_input(sample,t) = sample / sqrt(sigma^2 + 1)
  -> PriorTransformer(hidden_states, timestep=t, proj_embedding=condition)
  -> predicted_image_embedding [batch_or_2batch,1024,2048]
  -> split last dim into sample [*,1024,1024] and variance [*,1024,1024]
  -> optional CFG: uncond + guidance_scale * (cond - uncond)
  -> Heun scheduler.step(...).prev_sample [B,1024,1024]
```

Prior internals:

| Variant | Input concat before transformer |
| --- | --- |
| Text | projected text token [B,1,1024] + time token [B,1,1024] + latent tokens [B,1024,1024] = [B,1026,1024] |
| Image | projected image patch tokens [B,256,1024] + time token [B,1,1024] + latent tokens [B,1024,1024] = [B,1281,1024] |

Renderer image decode:

```text
latents [1,1024,1024]
  -> ShapEParamsProjModel:
       token ranges [0:256], [256:512], [512:768], [768:1024]
       -> generated MLP weights [256,93], [256,256], [256,256], [256,256]
  -> copy generated weights into MLP layers 0-3
  -> create_pan_cameras(frame_size): 20 cameras, rays [1,20*H*W,2,3]
  -> for each ray batch of 4096:
       coarse stratified 64 samples
       MLP(position encodings) -> density/channels
       integrate via cumsum/exp/sum
       fine importance 128 samples plus coarse samples
       integrate and add void background
  -> frames [20,H,W,3], then pipeline stacks to [B,20,H,W,3]
```

Renderer mesh decode:

```text
latents [1,1024,1024]
  -> same generated MLP weights
  -> query regular grid 128^3 positions in batches of 4096
  -> MLP(rendering_mode="stf") signed_distance [1,128^3,1]
  -> pad to [1,130,130,130] with negative border
  -> MeshDecoder marching cubes
  -> query STF RGB at dynamic mesh vertices
  -> MeshDecoderOutput(verts [Nv,3], faces [Nf,3], vertex_channels {"R","G","B"})
```

CPU/data-pipeline work: tokenization, PIL image processing, PIL conversion,
camera construction, mesh object assembly, and Python progress/offload hooks.
GPU/runtime candidates: prior denoising, scheduler arithmetic, renderer MLP
projection, ray MLP evaluation, volume integration, and possibly mesh field
evaluation. Marching cubes is dynamic-shape and should remain host/Python until
Dinoml has a dynamic mesh output contract.

## 5. Operator coverage checklist

Tensor/layout ops:

- `reshape/view`, `unsqueeze`, `squeeze`, `repeat_interleave`, `cat`, `split`,
  `chunk`, `pad`, `stack`, `broadcast_to`, `gather`, `sort`, `searchsorted`,
  boolean masks, `unique`, dynamic indexing, `where`.
- CLIP image input uses NCHW [B,3,224,224]; renderer output uses NHWC
  [B,20,H,W,3]. Prior and renderer MLP use sequence-last feature layouts.

GEMM/linear ops:

- Prior input projections: 1024->1024 latent, text 768->1024 or image 1024->1024
  condition, timestep MLP 1024->4096->1024.
- Prior block per layer: self-attention Q/K/V 1024->1024 with bias, output
  1024->1024, FFN 1024->4096 GELU then 4096->1024.
- Renderer parameter projection: four `ChannelsProj` einsums equivalent to
  batched per-vector GEMMs plus LayerNorm, with shapes `(256,93,1024)` and
  three `(256,256,1024)` weight banks.
- Renderer MLP: layer 0 `93->256`, layers 1-3 `256->256`, layer 4 `307->256`
  because direction encoding adds 51 channels, layer 5 `256->256`, output
  `256->12`.

Attention primitives:

- Prior self-attention only, no active cross-attention.
- Text variant: sequence 1026, heads 16, head dim 64.
- Image variant: sequence 1281, heads 8, head dim 128.
- No QK norm, no RoPE, no causal mask in active pipeline calls.

Normalization and adaptive conditioning:

- LayerNorm on prior input (`norm_in`), each transformer block norm1/norm3,
  prior `norm_out`, optional image condition projection norm, renderer
  `ChannelsProj.norm`.
- No AdaLayerNorm, RMSNorm, GroupNorm, or adaptive residual gates active in the
  Shap-E prior config.

Position/timestep/custom math:

- `Timesteps(inner_dim=1024, flip_sin_to_cos=True, downscale_freq_shift=0)`
  plus `TimestepEmbedding` GELU MLP.
- Learned positional embedding [1,1024,1024], source pads zeros in front when
  additional condition/time tokens make the runtime sequence longer.
- NeRF positional encodings: position 93 dims, direction 51 dims.

Scheduler/guidance arithmetic:

- Heun sigma table construction, Karras conversion, model input scaling,
  sample-prediction branch, clipping to [-1,1], first/second-order derivative
  state, CFG arithmetic.

Renderer/postprocessing:

- Ray-box intersection, safe divide, linspace, uniform random sampling,
  importance sampling through CDF/searchsorted, cumulative density integration,
  exp/transmittance, sigmoid/tanh/relu/SILU, sRGB-to-linear conversion.
- Marching cubes requires bit operations, LUT gather, dynamic compaction,
  interpolation, and dynamic face/vertex outputs.

## 6. Denoiser/model breakdown

`PriorTransformer` forward path:

1. Normalize scalar/0D timestep to [batch] and compute sinusoidal timestep
   features [B,1024].
2. Apply `TimestepEmbedding`: Linear 1024->4096, GELU, Linear 4096->1024.
3. Optionally LayerNorm condition embeddings. Active only for img2img.
4. Project conditioning tokens and latent tokens to hidden width 1024.
5. Concatenate condition tokens, one time token, and latent tokens.
6. Pad/add learned positional embeddings. With Shap-E configs, source pads two
   zero rows for text and 257 zero rows for image before the learned 1024 latent
   positions.
7. Apply `norm_in` LayerNorm.
8. Run 24 `BasicTransformerBlock`s.
9. Apply `norm_out`, slice off condition/time tokens, project 1024->2048.

Active `BasicTransformerBlock` config:

```text
norm1 LayerNorm
-> self Attention(query/key/value from same sequence, bias=True)
-> residual
-> norm3 LayerNorm
-> FeedForward: Linear 1024->4096, GELU, Dropout(0), Linear 4096->1024
-> residual
```

Inactive branches in the shared block: cross-attention, double self-attention,
AdaLayerNorm variants, GLIGEN fuser, positional sinusoidal embedding inside the
block, feed-forward chunking unless mutated externally, dropout effects during
inference, and gated attention.

`ShapERenderer` breakdown:

- `ShapEParamsProjModel` slices 1024 latent tokens into four 256-token groups.
  Each group applies a learned per-vector projection to generate one MLP weight
  tensor and a LayerNorm over generated output channels.
- Source copies those generated tensors into `self.mlp.state_dict()` for layer
  weights `nerstf.mlp.0.weight` through `nerstf.mlp.3.weight`.
- `MLPNeRSTFModel` evaluates NeRF/STF outputs over position and optional
  direction encodings. It returns density `relu`, signed distance `tanh`, and
  channels `sigmoid`.
- `decode_to_image` performs two-pass volume rendering, coarse then fine.
- `decode_to_mesh` evaluates an SDF grid, runs marching cubes, then evaluates
  per-vertex texture colors.

## 7. Attention requirements

Required attention is standard full self-attention over dense token sequences.

| Variant | Shape | Processor path | Masking |
| --- | --- | --- | --- |
| Text Shap-E prior | [B or 2B,1026,1024], 16 heads x 64 | `Attention` defaults to `AttnProcessor2_0` when PyTorch SDPA is available and `scale_qk=True`, otherwise `AttnProcessor` eager bmm/softmax/bmm | no mask in active pipeline |
| Img2Img Shap-E prior | [B or 2B,1281,1024], 8 heads x 128 | same | no mask in active pipeline |

`attention_processor.py` is the primary implementation path. The native
optimized path uses `torch.nn.functional.scaled_dot_product_attention` with
query/key/value shaped [B,heads,seq,head_dim], dropout 0, `is_causal=False`.
The eager parity path uses Q/K/V projections, `get_attention_scores`,
softmax over the key dimension, `torch.bmm`, and output projection.

No current Shap-E requirement for joint attention, cross-attention, added K/V,
IP-Adapter branches, QK norm, RoPE, GQA, varlen packing, or causal masks. A
Dinoml flash-style provider is feasible for the prior under strict guards:
dense fixed-length sequence, no mask, no dropout, head dims 64 or 128, and dtype
fp16/fp32/bf16 as admitted by the provider. Fused QKV is an optimization only;
Diffusers source uses separate `to_q`, `to_k`, and `to_v` Linear layers.

## 8. Scheduler and denoising-loop contract

Source scheduler: `HeunDiscreteScheduler` with Karras sigmas, exponential beta
schedule, 1024 train timesteps, `prediction_type="sample"`, and clipping.

`set_timesteps(num_inference_steps, device)`:

- Builds descending linspace timesteps by default.
- Converts alpha-cumprod schedule to sigmas.
- Applies Karras conversion when `use_karras_sigmas=true`.
- Appends terminal sigma 0.
- Expands sigmas for Heun first/second-order alternation:
  `sigmas = [s0, s1, s1, s2, s2, ..., 0]`.
- Expands timesteps similarly:
  `timesteps = [t0, t1, t1, t2, t2, ...]`.
- Keeps `self.sigmas` on CPU after construction; scale/step index into that
  table and rely on PyTorch scalar/device behavior.

Loop-side graph work:

```text
for t in scheduler.timesteps:
  latent_model_input = cat([latents, latents]) if CFG else latents
  scaled = latent_model_input / sqrt(sigma_i^2 + 1)
  model_output = prior(scaled, t, condition)
  model_output = split(model_output, 1024, dim=2)[0]
  model_output = CFG(model_output) when guidance_scale > 1
  latents = heun_step(model_output, t, latents).prev_sample
```

`step` with `prediction_type="sample"` treats the model output as predicted
original sample. In first-order mode it stores derivative, `dt`, and sample; in
second-order mode it averages the previous derivative with the new derivative,
uses the stored sample, then clears the state. `clip_sample=true` clamps the
predicted original sample to [-1,1].

Keep `set_timesteps`, duplicate timestep indexing, and Heun state ownership as
host-visible runtime state initially. Compile `scale_model_input`, CFG, output
split, sample clipping, derivative update, and one `step` after the scheduler
schema is explicit.

## 9. Position, timestep, and custom math

Prior timestep embedding is source-standard Diffusers `Timesteps` plus
`TimestepEmbedding`, but with hidden width 1024 and GELU activation.

Prior position handling is learned, not RoPE or sin-cos. Config stores only
1024 learned latent positions; source pads zeros for condition/time prefix
tokens when the runtime sequence is longer than 1024.

Renderer NeRF position encoding:

```text
posenc_nerf(x, max_deg):
  scales = 2 ** arange(max_deg)
  xb = flatten(x * scales)
  emb = sin(cat([xb, xb + pi / 2]))
  return cat([x, emb])
```

For 3D positions, max degree 15 yields 93 channels. For directions, max degree
8 yields 51 channels. Direction features are still concatenated as zeros when
`direction=None`.

Custom renderer math needing parity:

- AABB ray intersection with epsilon-biased divide and clamp.
- Stratified ray sampling with deterministic `torch.manual_seed(0)` in source.
- Importance sampling via CDF, `searchsorted`, gather, random interpolation,
  and sort.
- Volume integration using cumulative density, exponential transmittance, and
  weighted channel sum.
- Marching-cubes bitmask/LUT/gather/unique/interpolate path.

Precomputable: tokenizer outputs, CLIP embeddings, scheduler timesteps/sigmas,
camera rays for a fixed `frame_size`, volume grid query points for a fixed
`grid_size`, NeRF encoding frequency scales. Dynamic: timestep embeddings,
latents, generated renderer weights, ray sample locations, importance samples,
and dynamic mesh vertices/faces.

## 10. Preprocessing and input packing

Text path:

- Pipeline accepts string or list.
- It forces `tokenizer.pad_token_id = 0`, tokenizes to `max_length=77`, and
  separately tokenizes `padding="longest"` only to warn about truncation.
- Uses `text_encoder(...).text_embeds`, not per-token hidden states.
- Repeats embeddings by `num_images_per_prompt`.
- Normalizes by L2 norm over hidden dim, then scales by `sqrt(768)`.
- CFG uses an all-zero negative embedding, not a negative prompt string.

Image path:

- Accepts PIL, tensor, or lists of PIL/tensor. Tensor lists are stacked or
  concatenated before processing.
- Non-tensor images go through `CLIPImageProcessor` to [B,3,224,224].
- Uses `CLIPVisionModel(...).last_hidden_state`, drops CLS, and keeps 256 patch
  tokens.
- Repeats embeddings by `num_images_per_prompt`.
- CFG uses all-zero patch embeddings.

Latent packing:

- Initial noise is allocated as flat [B,1048576], scaled by init sigma, then
  reshaped to [B,1024,1024].
- No VAE encode/decode and no NCHW latent map packing.

Output postprocess:

- `output_type="latent"` returns final latent tokens directly.
- `output_type="np"` returns stacked renderer arrays after CPU transfer.
- `output_type="pil"` calls `numpy_to_pil` as host-side utility.
- `output_type="mesh"` returns `MeshDecoderOutput` objects with dynamic tensor
  fields; this is not a dense batch tensor.

## 11. Graph rewrite / lowering opportunities

1. Name: prior QKV packing.
   Source pattern: per attention layer, separate `to_q`, `to_k`, `to_v` Linear
   on the same normalized hidden states.
   Replacement: one packed GEMM producing QKV, then split heads.
   Preconditions: self-attention only, same input tensor, all three bias flags
   true, no adapter/LoRA mutation active, static hidden width 1024, no QK norm.
   Shape equations: [B,S,1024] x [1024,3*1024] -> [B,S,3072], S in {1026,1281}
   for first configs.
   Weight transform: concatenate weights and biases in Q,K,V order matching
   the runtime split.
   Layout constraints: token-major [B,S,C]; no NHWC translation involved.
   Failure cases: active PEFT/attention processor replacement, cross-attention
   or added-K/V processors, changed bias policy.
   Parity test: one attention block with random hidden states, compare packed
   QKV against three-linears source before attention.

2. Name: prior FFN fusion.
   Source pattern: LayerNorm -> Linear 1024->4096 -> GELU -> Linear 4096->1024
   -> residual.
   Replacement: fused layernorm plus GEMM/GELU/GEMM scheduling, optionally fuse
   final residual add.
   Preconditions: activation_fn `gelu`, dropout disabled in eval, no feed-forward
   chunking, hidden dim 1024.
   Failure cases: chunked feed-forward, training/dropout, adapter-modified FFN.
   Parity test: one `BasicTransformerBlock` MLP branch at fp32/fp16 tolerances.

3. Name: renderer `ChannelsProj` as grouped GEMM.
   Source pattern: `einsum("bvd,vcd->bvc")` with per-vector weights, LayerNorm,
   and per-vector bias.
   Replacement: grouped/batched GEMM over 256 vectors per projected MLP layer.
   Preconditions: vector count and channel shapes match config; latent layout
   [B,V,1024]; LayerNorm axis is channel dim -1.
   Shape equations: for layer 0, 256 independent [B,1024] x [1024,93] products;
   layers 1-3 use [1024,256].
   Weight transform: store projection weight as [V,D,C] or [V,C,D] according to
   provider layout; preserve source bias add after LayerNorm, because source
   applies norm before adding reshaped projection bias.
   Failure cases: changed `param_shapes`, dynamic vector counts, generated
   renderer weights not modeled as explicit tensors.
   Parity test: `ShapEParamsProjModel` output dict against source.

4. Name: explicit dynamic renderer weights.
   Source pattern: generated tensors are copied into `self.mlp.state_dict()`.
   Replacement: pass generated weights as explicit inputs to functional MLP
   layers 0-3.
   Preconditions: single-sample decode or batched dynamic-weight kernel with
   per-sample weight selection; biases for layers 0-3 remain static source MLP
   biases.
   Failure cases: multiple samples rendered concurrently with mutable module
   state, graph capture expecting static weights.
   Parity test: decode one ray batch with source state-copy path versus
   functional dynamic-weight path.

5. Name: ray-camera precompute.
   Source pattern: `create_pan_cameras(size)` constructs deterministic 20-view
   rays every decode.
   Replacement: cache rays by `(frame_size,dtype,device)`.
   Preconditions: same source camera constants, frame size divisible by ray
   batch expectations or remainder handling matches source.
   Failure cases: user-supplied camera support added later.
   Parity test: compare generated rays bitwise/fp tolerance.

6. Name: renderer no-layout-translation guard.
   Source pattern: ray/sample tensors use feature-last layouts [B,R,S,C].
   Replacement: keep feature-last for MLP and reductions; do not apply image
   NHWC/NCHW layout passes inside renderer.
   Preconditions: renderer tensors are not convolutions and reductions happen
   over sample dim -2 or feature dim -1.
   Failure cases: generic vision layout pass rewrites `dim=-1` LayerNorm or
   `dim=-2` cumsum/sum incorrectly.
   Parity test: renderer unit tests with layout pass enabled must leave these
   axes unchanged.

## 12. Kernel fusion candidates

Highest priority:

- Prior LayerNorm + QKV projection + attention + output projection for sequence
  lengths 1026 and 1281. This is the main denoising cost.
- Prior FFN GELU block 1024->4096->1024, 24 layers.
- Heun scheduler and CFG arithmetic over [B,1024,1024] tensors, including
  output split/discard and `clip_sample`.
- Renderer functional dynamic-weight MLP for large ray/sample batches.

Medium priority:

- `ChannelsProj` grouped GEMM plus LayerNorm for generating renderer weights.
- NeRF integration fusion: density activation, cumulative sum, exp,
  transmittance, weighted RGB sum.
- Camera ray and volume query precompute/caching.
- CLIP image patch embedding/vision encoder only if Dinoml chooses to own CLIP
  external components rather than treat embeddings as inputs.

Lower priority:

- PIL/np postprocessing and sRGB conversion.
- Marching cubes dynamic mesh extraction.
- `searchsorted`/importance sampling optimization; useful only after renderer
  is admitted as a compiled target.
- Adapter/PEFT mutation support on the prior.

## 13. Runtime staging plan

Stage 1: parse configs and load weights for `openai/shap-e` prior plus Heun
scheduler. Treat CLIP text embeddings and final renderer as external/stubbed.

Stage 2: one PriorTransformer block parity with externally supplied prompt
embedding, latent tokens [B,1024,1024], and fixed timestep. Use eager attention
or native SDPA equivalent for parity.

Stage 3: full prior denoising step parity, including prior output split,
sample-prediction Heun step, clipping, and CFG.

Stage 4: full denoising loop with Heun scheduler in Python host control. Return
`output_type="latent"` only.

Stage 5: add text prompt embedding cache integration. Keep CLIP text encoder as
external at first; later compile CLIP as a transformer-family target if useful.

Stage 6: add `openai/shap-e-img2img` conditioning with external CLIP image
embeddings first, then optional CLIP vision/image processor ownership.

Stage 7: renderer image decode as a separate compiled island: parameter
projection, explicit dynamic MLP weights, ray MLP, and volume integration.

Stage 8: mesh decode only after Dinoml has a dynamic mesh/scatter/indexing
contract. Until then keep `output_type="mesh"` in Python.

## 14. Parity and validation plan

- Config/default reconciliation: instantiate source components from inspected
  configs and assert effective PriorTransformer and Heun fields match report.
- Attention unit parity: one attention layer for text S=1026 and image S=1281,
  fp32 first, then fp16 with looser tolerance.
- Prior block parity: one `BasicTransformerBlock` with no mask and eval mode.
- Prior full forward parity: fixed random latents, timestep, and text/image
  conditioning; verify [B,1024,2048] output and split first half.
- Scheduler parity: `set_timesteps(25/64)`, sigma/timestep duplication,
  `scale_model_input`, first-order step, second-order step, and clipping for
  `prediction_type="sample"`.
- CFG parity: text/image zero negative embeddings, concat/chunk, guidance
  scales 1.0 and >1.0.
- Short loop parity: 2 or 4 denoising steps with fixed latents/generator and
  external embeddings, compare final latent tokens.
- Renderer projection parity: `ShapEParamsProjModel` output dict and generated
  MLP weight values.
- Renderer ray parity: `create_pan_cameras(64)` rays and one ray batch coarse
  render in fp32.
- Image decode smoke: one latent to [20,64,64,3] numeric output. Recommended
  initial tolerance fp32 `rtol=1e-4, atol=1e-5`; fp16 renderer may need
  `rtol=5e-2, atol=5e-3` because ray integration compounds errors.
- Mesh smoke: source-only first, verify `.bin`/safetensors LUT availability,
  then compare dynamic counts and sampled vertex/channel values if admitted.

## 15. Performance probes

- CLIP text encoder throughput by batch and prompt length 77.
- CLIP vision encoder throughput for [B,3,224,224].
- Prior one-step latency for S=1026 text and S=1281 image, with and without CFG.
- Prior attention vs FFN time split over 24 layers.
- Heun scheduler overhead for [B,1024,1024] latents and 25/64 inference steps.
- Full latent-only denoising loop by step count and batch size.
- Renderer parameter projection latency.
- Renderer image decode by `frame_size` 64/128/256 and ray batch size.
- Renderer MLP vs integration time split for coarse/fine passes.
- Mesh SDF grid evaluation for grid 64/128 and marching-cubes host overhead.
- VRAM/workspace for prior CFG batch doubling and renderer ray batches.
- Attention backend comparison: eager bmm/softmax/bmm, PyTorch SDPA, Dinoml
  flash-style provider under no-mask fixed-sequence guards.

## 16. Scope boundary and separate candidates

Separate candidate reports related to Shap-E:

- `shap_e_prior_adapters`: model-level PEFT/attention processor mutation on
  `PriorTransformer`; source class has adapter mixins, but base pipelines do
  not require adapter state.
- `shap_e_renderer_image`: compiled NeRF image renderer with dynamic generated
  MLP weights and volume integration.
- `shap_e_renderer_mesh`: STF/SDF grid evaluation plus marching cubes and
  dynamic mesh outputs.
- `clip_text_encoder` and `clip_vision_encoder`: external Transformers
  components if Dinoml wants to own preprocessing/encoder stages instead of
  accepting embeddings.
- `heun_scheduler`: reusable scheduler candidate if other Diffusers families
  use Heun beyond Shap-E.

Genuinely ignored/out of scope for this audit:

- XLA branch and `xm.mark_step`.
- NPU/MPS-specific activation/backend branches.
- Flax and ONNX.
- Training, losses, dropout behavior, and gradient checkpointing.
- Callback mutation, interactive interrupt, safety checking, and multi-GPU or
  context-parallel execution.
- Deprecated unCLIP pipeline behavior except for copied helper provenance.
- Point-cloud output: no non-deprecated Shap-E point-cloud output contract was
  found in the inspected Diffusers source.

## 17. Final implementation checklist

- [ ] Parse Shap-E model indexes, including `renderer`/`shap_e_renderer` alias handling.
- [ ] Parse PriorTransformer configs for text and img2img variants.
- [ ] Parse Heun scheduler config and materialize Karras sample-prediction tables.
- [ ] Load prior weights and reconcile fp16/fp32 artifact choices.
- [ ] Accept external text/image conditioning embeddings for first parity.
- [ ] Implement PriorTransformer LayerNorm, Linear, GELU FFN, self-attention, and timestep embedding path.
- [ ] Implement prior output split and discard variance half.
- [ ] Implement CFG concat/chunk arithmetic with zero negative embeddings.
- [ ] Implement Heun `scale_model_input` and first/second-order `step` for `prediction_type="sample"`.
- [ ] Add one-block prior parity tests for S=1026 and S=1281.
- [ ] Add one-step and short-loop latent parity tests.
- [ ] Add text prompt preprocessing parity or explicitly require precomputed CLIP text embeddings.
- [ ] Add image preprocessing/CLIP vision parity or explicitly require precomputed image patch embeddings.
- [ ] Model renderer generated MLP weights as explicit tensors instead of hidden state mutation.
- [ ] Add renderer parameter projection parity.
- [ ] Add renderer image decode smoke for [20,64,64,3].
- [ ] Fence mesh output behind dynamic mesh/marching-cubes validation.
- [ ] Benchmark prior attention/FFN, scheduler overhead, and renderer ray MLP separately.
