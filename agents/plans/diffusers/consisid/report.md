# Diffusers ConsisID Operator and Integration Report

Candidate slug: `consisid`

## 1. Source basis

```text
Diffusers commit/version:
  Local checkout diffusers at b3a515080752a3ba7ca92161e25530c7f280f629.

Model id(s):
  BestWishYsh/ConsisID-preview

Config sources:
  H:/configs/BestWishYsh/ConsisID-preview/model_index.json
  Official Hugging Face component configs inspected through huggingface_hub:
    scheduler/scheduler_config.json
    text_encoder/config.json
    tokenizer/tokenizer_config.json
    transformer/config.json
    vae/config.json
  Hugging Face snapshot cache paths inspected:
    C:/Users/user/.cache/huggingface/hub/models--BestWishYsh--ConsisID-preview/snapshots/950bc3f0902db44799e223a12ad972f9c52b341d/*

Pipeline files inspected:
  diffusers/src/diffusers/pipelines/consisid/pipeline_consisid.py
  diffusers/src/diffusers/pipelines/consisid/consisid_utils.py
  diffusers/src/diffusers/pipelines/consisid/pipeline_output.py

Model files inspected:
  diffusers/src/diffusers/models/transformers/consisid_transformer_3d.py
  diffusers/src/diffusers/models/autoencoders/autoencoder_kl_cogvideox.py
  diffusers/src/diffusers/models/attention.py
  diffusers/src/diffusers/models/attention_processor.py
  diffusers/src/diffusers/models/embeddings.py
  diffusers/src/diffusers/models/normalization.py

Scheduler/processors/helpers inspected:
  diffusers/src/diffusers/schedulers/scheduling_dpm_cogvideox.py
  diffusers/src/diffusers/schedulers/scheduling_ddim_cogvideox.py
  diffusers/src/diffusers/video_processor.py
  diffusers/src/diffusers/image_processor.py
  diffusers/src/diffusers/loaders/lora_pipeline.py

External component configs inspected:
  T5EncoderModel / T5Tokenizer configs from the official ConsisID repo.
  Face helper assets listed in the official repo: EVA02 CLIP weights, antelopev2
  ONNX models, facexlib detector/parser weights. These have no Diffusers
  component config files.

Any missing files or assumptions:
  The local H:/configs cache had only model_index.json; component configs were
  accessible from the official Hugging Face repo without a gated-auth blocker.
  Only one official ConsisID Diffusers checkpoint/config was found. Face
  detection/alignment helper models are treated as preprocessing/conditioning
  candidates, not first-slice Dinoml runtime graph work. Multi-GPU/context
  parallel, callbacks/interrupt mutation, safety filtering, training/loss,
  dropout, gradient checkpointing, XLA/NPU/MPS/Flax/ONNX, and unrelated backend
  paths are out of scope.
```

## 2. Pipeline and component graph

ConsisID is a CogVideoX-style image-to-video pipeline with an identity-aware
transformer. The pipeline always starts from a conditioning image, encodes that
image through the CogVideoX VAE, concatenates the image-condition latents with
noisy video latents, and denoises with T5 text conditioning plus optional face
identity tensors.

```text
prompt / negative prompt
  -> T5Tokenizer + T5EncoderModel -> prompt embeds [B,226,4096]
face image preprocessing, if supplied through helper path
  -> insightface/facexlib/EVA02-CLIP helpers -> id_cond [B,1280],
     id_vit_hidden list[5] of [B,577,1024], optional keypoint image
conditioning image
  -> VideoProcessor preprocess -> VAE encode -> image latents [B,13,16,60,90]
random latent video noise [B,13,16,60,90]
  -> concat noisy latents + image latents on channel axis -> [B,13,32,60,90]
  -> denoising loop:
       ConsisIDTransformer3DModel + CFG + CogVideoXDPMScheduler
  -> VAE decode [B,16,13,60,90] -> video [B,3,49,480,720]
  -> VideoProcessor postprocess
```

Required first-slice components:

| Component | Class/file | First-slice role |
| --- | --- | --- |
| Pipeline | `ConsisIDPipeline`, `pipeline_consisid.py` | Runtime contract for image, prompt, identity tensors, CFG, scheduler loop, VAE encode/decode. |
| Denoiser | `ConsisIDTransformer3DModel`, `consisid_transformer_3d.py` | Required. CogVideoX joint text/video transformer with identity cross-attention enabled by config. |
| VAE | `AutoencoderKLCogVideoX`, `autoencoder_kl_cogvideox.py` | Encode required for image condition; decode required for output video. |
| Scheduler | `CogVideoXDPMScheduler`, `scheduling_dpm_cogvideox.py` | Official default; v-prediction DPM-Solver++ style update with old predicted sample state. |
| Text encoder | `T5EncoderModel`, `T5Tokenizer` | Accept external prompt embeds first; full T5 can be separate. |
| Face helpers | `consisid_utils.py` plus external insightface/facexlib/EVA CLIP | Preprocessing/conditioning surface; not a Diffusers model component. |

Separate candidate reports:

| Candidate | Primary classes/files | Runtime delta |
| --- | --- | --- |
| `consisid_face_preprocess` | `consisid_utils.py`, external insightface/facexlib/EVA02-CLIP | Face detection, alignment, parsing, identity embedding, ViT hidden states, and keypoint drawing before the denoiser. |
| `consisid_lora_adapters` | `CogVideoXLoraLoaderMixin`, `PeftAdapterMixin` | Runtime/load-time adapter mutation for transformer/text encoder weights. |
| `consisid_kps` | `ConsisIDPipeline.prepare_latents`, transformer `is_kps` branch | Adds keypoint image draw/preprocess/VAE encode and changes image-condition packing. Official preview config has `is_kps=false`. |
| `cogvideox_vae` | `AutoencoderKLCogVideoX` | Shared temporal VAE codec report, including tiling/slicing and temporal caches. |
| `cogvideox_schedulers` | `CogVideoXDPMScheduler`, `CogVideoXDDIMScheduler` | Video-specific scheduler matrix beyond the preview default. |

No ControlNet, T2I-Adapter, GLIGEN, depth2img, inpaint, or upscaling pipeline is
present in the `pipelines/consisid` folder.

## 3. Important config dimensions

Representative checkpoint sweep:

| Repo | Pipeline | Transformer | VAE | Scheduler | Notes |
| --- | --- | --- | --- | --- | --- |
| `BestWishYsh/ConsisID-preview` | `ConsisIDPipeline` | 42 layers, 48 heads, head dim 64, inner 3072, `in_channels=32`, `out_channels=16` | CogVideoX VAE, latent 16, spatial scale 8, temporal scale 4, scaling 0.7 | `CogVideoXDPMScheduler`, v-pred, trailing, zero-SNR | Only official Diffusers config found; identity branch enabled with `is_train_face=true`. |

Transformer config facts:

| Field | Value | Runtime effect |
| --- | ---: | --- |
| `sample_frames`, latent frames | 49 source frames -> 13 latent frames | `(F - 1) // 4 + 1`. |
| `sample_height`, `sample_width` | 60 x 90 latent grid | Decodes to 480 x 720 through VAE scale 8. |
| `patch_size` | 2 | 2D Conv2d patching per latent frame; token grid 30 x 45. |
| `max_text_seq_length` | 226 | T5 prompt embed length expected by patch embedding. |
| `num_attention_heads`, `attention_head_dim` | 48, 64 | Inner dim 3072; joint attention head dim 64. |
| `num_layers` | 42 | 42 ConsisID blocks. |
| `text_embed_dim`, `time_embed_dim` | 4096, 512 | T5 projection and timestep MLP sizes. |
| `use_rotary_positional_embeddings` | true | Pipeline prepares 3D RoPE for video tokens. |
| `use_learned_positional_embeddings` | true | Patch embed also applies learned/registered positional embeddings and rejects non-default learned-position resolutions. |
| `is_train_face` | true | Enables Local Facial Extractor and Perceiver cross-attention branch. |
| `is_kps` | false | Official first path does not encode a keypoint condition image. |
| `cross_attn_interval` | 2 | Identity cross-attention after every second transformer block, 21 times. |
| LFE dims | id 1280, ViT 1024, depth 10, 32 queries, output 2048 | Face branch emits `[B,32,2048]`. |

VAE config facts:

| Field | Value |
| --- | --- |
| Class | `AutoencoderKLCogVideoX` |
| Channels | RGB in/out 3, latent 16 |
| Blocks | `[128,256,256,512]`, 4 down/up blocks, 3 layers per block |
| Compression | spatial 8, temporal 4 |
| Scaling | `scaling_factor=0.7`, no shift, no latent mean/std |
| Quant convs | `use_quant_conv=false`, `use_post_quant_conv=false` |
| Runtime policy | `force_upcast=true`; tiling/slicing disabled by default |

Scheduler config facts:

| Field | Value |
| --- | --- |
| Class | `CogVideoXDPMScheduler` |
| `prediction_type` | `v_prediction` |
| Betas | scaled linear, `beta_start=0.00085`, `beta_end=0.012`, 1000 train timesteps |
| Timestep spacing | trailing |
| Zero-SNR | `rescale_betas_zero_snr=true`, `snr_shift_scale=1.0` |
| Clip | `clip_sample=false` |
| State | carries `old_pred_original_sample` between steps |

Text encoder config facts:

| Field | Value |
| --- | --- |
| Class | `T5EncoderModel`, `_name_or_path=google/t5-v1_1-xxl` |
| Width/depth | `d_model=4096`, 24 layers, 64 heads, `d_ff=10240` |
| Activation | gated GELU / `gelu_new` |
| Dtype metadata | `torch_dtype=bfloat16` |
| Tokenizer | T5 tokenizer, vocab 32128 plus extra IDs; pipeline max length 226 |

Weight metadata from safetensors index:

| Component | Metadata |
| --- | --- |
| Text encoder | 2 shards, total size 9,524,621,312 bytes, 219 mapped tensors |
| Transformer | 2 shards, total size 12,434,205,824 bytes, 1344 mapped tensors |
| VAE | single safetensors file; no index file in repo |

Recommended first Dinoml scheduler slice: start with `CogVideoXDPMScheduler`
using the official v-prediction trailing config and stateful
`old_pred_original_sample`. `CogVideoXDDIMScheduler` is source-compatible but is
not the preview checkpoint default.

## 3a. Family variation traps

- Pipeline latents are source `[B,F,C,H,W]`, while the CogVideoX VAE consumes
  `[B,C,F,H,W]`. Do not silently convert these two contracts.
- Transformer input channels are 32 only because noisy 16-channel latents are
  concatenated with 16-channel image-condition latents on source dim 2.
- The denoiser output is 16 channels even though the input is 32 channels.
- Learned positional embeddings and RoPE are both active in the preview config.
  Learned positions make non-default latent H/W unsafe without fallback or
  exact source behavior.
- Official preview has identity branch enabled. First parity cannot omit
  `id_cond` and `id_vit_hidden` unless the report explicitly stubs identity
  conditioning and validates a non-identity custom config.
- Keypoint conditioning is inactive in the official config; do not count the
  extra keypoint VAE encode path as first-slice runtime work.
- The face helper pipeline depends on external packages and ONNX/PyTorch assets
  that are not normal Diffusers components.
- CogVideoX VAE temporal caches, slicing, and tiling are memory-policy paths.
  Keep them disabled for first parity.
- NDHWC/channel-last is a guarded optimization only. Axis-sensitive source ops
  include VAE Conv3d channel dim 1, pipeline channel concat dim 2, patch Conv2d
  over reshaped B*F NCHW maps, LayerNorm last-dim tokens, and scheduler scalar
  broadcasting over `[B,F,C,H,W]`.

## 4. Runtime tensor contract

Default 480 x 720, 49-frame run:

| Boundary | Tensor | Shape/layout | Notes |
| --- | --- | --- | --- |
| Prompt embeds | `prompt_embeds` | `[B,226,4096]` | T5 hidden states, duplicated per prompt. |
| CFG prompt embeds | `prompt_embeds` after concat | `[2B,226,4096]` | Negative then positive when guidance > 1. |
| Input image | preprocessed | `[B,3,480,720]` NCHW | From `VideoProcessor.preprocess`. |
| VAE image input | image unsqueezed | `[B,3,1,480,720]` NCTHW | One frame encoded. |
| Image latents | after VAE encode and permute | `[B,1,16,60,90]` then padded to `[B,13,16,60,90]` | Multiplied by scaling factor 0.7; remaining frames zero. |
| Noisy latents | `latents` | `[B,13,16,60,90]` BFCHW | Gaussian noise times scheduler `init_noise_sigma`. |
| Transformer input | concat | `[B,13,32,60,90]` BFCHW | Source concat dim 2. CFG doubles batch. |
| Patch tokens | after Conv2d patch | `[B,17550,3072]` | 13 * 30 * 45 video tokens. |
| Joint tokens | text + video | `[B,17776,3072]` | 226 text tokens plus video tokens. |
| Identity condition | `id_cond` | `[B,1280]` | 512 insightface embedding plus 768 EVA CLIP embedding. |
| Identity ViT hidden | `id_vit_hidden` | list[5] of `[B,577,1024]` | Consumed by Local Facial Extractor. |
| Face tokens | `valid_face_emb` | `[B,32,2048]` | Used by Perceiver cross-attention every 2 blocks. |
| Noise prediction | `noise_pred` | `[B,13,16,60,90]` | v-pred model output for scheduler. |
| Scheduler state | old x0 prediction | `[B,13,16,60,90]` or `None` | Returned and fed into next DPM step. |
| VAE decode input | permuted latents | `[B,16,13,60,90]` NCTHW | Divided by scaling factor 0.7. |
| Decoded video | `frames` | `[B,3,49,480,720]` NCTHW | Postprocessed to PIL/NumPy/list output. |

CPU/data-pipeline work: text tokenization, optional T5 execution, face
detection/alignment/parsing/EVA CLIP, image load/resize, keypoint drawing, and
output conversion. GPU/runtime work: VAE encode/decode, latent packing,
transformer denoise, identity LFE/cross-attention if admitted, CFG, and
scheduler updates.

Cacheable tensors: prompt/negative embeddings, face `id_cond`,
`id_vit_hidden`, keypoint image if used, image-condition latents, RoPE tables,
learned positional embeddings, scheduler timesteps/alpha tables.

## 5. Operator coverage checklist

### Tensor/layout ops

- Source BFCHW latents in pipeline and NCTHW tensors at VAE boundaries.
- NCHW image preprocessing, unsqueeze frame dim, permute `[B,C,F,H,W] <->
  [B,F,C,H,W]`.
- Concatenate/chunk for CFG and image-condition channel packing.
- Reshape/flatten/transpose for Conv2d patchify and unpatchify.
- Zero padding of image latents over latent frames.
- Optional keypoint draw/preprocess/VAE encode for `is_kps=true`.

### Convolution/downsample/upsample ops

- Patch embedding: `Conv2d(32 -> 3072, kernel=2x2, stride=2)` over `B*F`
  NCHW latent maps.
- CogVideoX VAE temporal Conv3d/resnet/down/up blocks over NCTHW, GroupNorm,
  SiLU, temporal caches, and spatial up/downsampling.

### GEMM/linear ops

- T5 external encoder if included later.
- Patch text projection `Linear(4096 -> 3072)`.
- Timestep MLP: sinusoidal `Timesteps(3072)` then `TimestepEmbedding(3072 ->
  512)`.
- Joint attention Q/K/V and output projections for 42 blocks.
- FFN GEGLU/GELU-approximate layers over 3072-wide text+video tokens.
- Local Facial Extractor Linear/LayerNorm/LeakyReLU mappings and Perceiver Q/KV.
- Final `Linear(3072 -> 2*2*16=64)` unpatchify projection.

### Attention primitives

- Joint text+video self-attention using `CogVideoXAttnProcessor2_0` and PyTorch
  SDPA fallback semantics.
- Q/K LayerNorm in attention heads when `qk_norm=true`.
- 3D RoPE applied only to video-token Q/K slices after text tokens.
- Local Facial Extractor Perceiver attention and Perceiver cross-attention
  implemented manually with matmul + softmax, not through `AttentionProcessor`.

### Normalization and adaptive conditioning

- `CogVideoXLayerNormZero`: LayerNorm plus timestep-conditioned scale/shift/gate
  for text and video streams.
- `AdaLayerNorm` final output modulation.
- LayerNorm in LFE mappings, Perceiver attention, final transformer norm.
- VAE GroupNorm/RMS-like temporal block norms from CogVideoX VAE source.

### Scheduler and guidance arithmetic

- CFG batch concat/chunk and `uncond + scale * (text - uncond)`.
- Optional dynamic CFG cosine schedule.
- CogVideoX DPM v-pred conversion and old-pred state update.
- Stochastic noise term in scheduler step.

### VAE/postprocessing ops

- VAE encode mode/sample retrieval through `retrieve_latents`.
- VAE latent scaling by 0.7 on encode and reciprocal on decode.
- Video postprocess from NCTHW tensor to requested output type.

## 6. Denoiser/model breakdown

`ConsisIDTransformer3DModel.forward`:

```text
hidden_states [B,F,32,H,W]
id_cond/id_vit_hidden -> optional LocalFacialExtractor -> [B,32,2048]
timestep -> Timesteps(3072) -> TimestepEmbedding -> emb [B,512]
CogVideoXPatchEmbed(text [B,226,4096], video [B,F,32,H,W])
  text Linear -> [B,226,3072]
  video Conv2d patch per frame -> [B,F*(H/2)*(W/2),3072]
  add learned/sincos positional embedding if enabled
42 x ConsisIDBlock
  optional PerceiverCrossAttention after every 2 blocks
final concat text+video -> LayerNorm -> take video tokens
AdaLayerNorm(temb) -> Linear(3072 -> 64)
unpatchify -> [B,F,16,H,W]
```

`ConsisIDBlock`:

```text
CogVideoXLayerNormZero(hidden, text, temb)
-> joint Attention over text+video tokens with QK LayerNorm and optional 3D RoPE
-> gated residual add to hidden and text streams
CogVideoXLayerNormZero
-> concat normalized text+video tokens
-> FeedForward activation_fn="gelu-approximate"
-> split text/video outputs
-> gated residual add
```

Identity path:

```text
id_cond [B,1280] -> id_embedding_mapping -> 5 identity tokens [B,5,1024]
learned 32 latent queries + identity tokens
for each of 5 ViT scales:
  map ViT hidden [B,577,1024]
  context = identity tokens + mapped ViT tokens
  two PerceiverAttention + FF residual layers
take 32 query latents -> matmul proj_out [1024,2048]
every 2 transformer blocks:
  PerceiverCrossAttention(face [B,32,2048], video [B,17550,3072])
  add local_face_scale * output
```

## 7. Attention requirements

Primary Diffusers attention implementation for the main transformer is
`CogVideoXAttnProcessor2_0` in `attention_processor.py`, called through the
shared `Attention` module from `attention.py`. It uses
`torch.nn.functional.scaled_dot_product_attention` as the parity path.

| Attention | Shape | Requirements |
| --- | --- | --- |
| Joint text/video attention | Q/K/V `[B,48,17776,64]` at default shape | Noncausal SDPA, no mask in first path, QK LayerNorm, text+video concatenation, output split. |
| 3D RoPE | video token slice `[B,48,17550,64]` | Cos/sin from temporal 13 and grid 30 x 45; text tokens are not rotated. |
| LFE Perceiver attention | latent queries 37 tokens, context 5+577 tokens per scale | Manual Q/K/V Linear, matmul softmax in fp32, output Linear. |
| Identity Perceiver cross-attention | Q video `[B,17550,3072]`, K/V face `[B,32,2048]` | Manual attention, heads 16, head dim 128, output add to video tokens. |

Fused QKV is source-supported for CogVideoX through `FusedCogVideoXAttnProcessor2_0`, but the preview source default
constructs `CogVideoXAttnProcessor2_0`; fused projection mutation is an
optimization, not required for parity.

Flash-style Dinoml provider candidates are valid for the main joint attention
when head dim 64, noncausal mode, no mask, dtype, and sequence length are
supported. QK LayerNorm and RoPE must be explicit pre-ops unless fused under
guards. The manual Perceiver attention paths need separate provider admission or
GEMM/softmax/GEMM lowering; they do not follow `attention_dispatch.py`.

## 8. Scheduler and denoising-loop contract

Official default: `CogVideoXDPMScheduler`.

```text
scheduler.set_timesteps(num_inference_steps, device)
old_pred_original_sample = None
for each t:
  latent_model_input = cat([latents]*2) if CFG else latents
  latent_model_input = scheduler.scale_model_input(latent_model_input, t)
  latent_image_input = cat([image_latents]*2) if CFG else image_latents
  transformer_input = cat([latent_model_input, latent_image_input], dim=2)
  timestep = t.expand(batch)
  noise_pred = transformer(transformer_input, prompt_embeds, timestep, identity)
  if dynamic CFG: update guidance scale with cosine schedule
  if CFG: noise_pred = uncond + guidance * (text - uncond)
  latents, old_pred_original_sample =
    scheduler.step(noise_pred, old_pred_original_sample, t, previous_t, latents)
```

Scheduler step semantics:

- `scale_model_input` is identity for `CogVideoXDPMScheduler`.
- `prediction_type=v_prediction` converts model output to predicted original
  sample with alpha/beta cumulative products.
- First step returns `prev_sample` and stores `pred_original_sample`.
- Later steps use old and current predicted original samples in the DPM-Solver++
  style second-order correction.
- The source samples fresh noise in each step through `randn_tensor`.

Keep timestep table generation, old-pred state, generator/noise handling, and
loop dispatch host-visible first. Fuse CFG and scheduler pointwise arithmetic
only after one-step scheduler parity is proven.

## 9. Position, timestep, and custom math

- Timestep embeddings use `Timesteps(inner_dim=3072, flip_sin_to_cos=true,
  freq_shift=0)` followed by `TimestepEmbedding(3072 -> 512)` with SiLU.
- `CogVideoXLayerNormZero` produces six modulation tensors:
  hidden shift/scale/gate and text shift/scale/gate.
- `AdaLayerNorm` final block uses timestep embedding to scale/shift video
  tokens before the final projection.
- `get_3d_rotary_pos_embed` splits head dim 64 into temporal/spatial frequency
  parts and builds cos/sin tables over latent temporal size and patch grid.
- `CogVideoXPatchEmbed` positional embeddings use `sample_frames=49` before
  temporal compression. At default shape the precomputed learned position
  buffer aligns with 226 text plus 17,550 video tokens.
- Dynamic CFG scale is source math:
  `1 + guidance_scale * (1 - cos(pi * ((steps - t) / steps) ** 5)) / 2`.

Precompute candidates: learned/sincos positional embeddings for the default
grid, RoPE cos/sin tables, timestep embedding tables for fixed scheduler
timesteps, image-condition latents, prompt embeddings, and face identity tokens.

## 10. Preprocessing and input packing

Text:

- Tokenize with T5 tokenizer to max length 226.
- T5 encoder output `[B,226,4096]`.
- Negative prompt defaults to empty string under CFG.
- CFG concatenates negative and positive prompt embeddings on batch dim.

Image/video:

- `VideoProcessor.preprocess` returns NCHW image tensor at requested H/W.
- Image is unsqueezed to one-frame NCTHW for VAE encode.
- Encoded image latents are permuted to BFCHW and multiplied by 0.7.
- The remaining latent frames are zeros; official config does not add keypoint
  latents.
- Noisy video latents and image-condition latents are concatenated on channel
  axis to form 32-channel transformer input.
- Decode permutes BFCHW latents to NCTHW, divides by 0.7, VAE decodes, then
  postprocesses video.

Face helpers:

- `process_face_embeddings_infer` loads/normalizes an RGB image, resizes long
  edge to 1024, detects face and keypoints, aligns/crops face to 512, parses
  face/background, feeds a 336 image into EVA02-CLIP visual, normalizes CLIP
  embedding, and concatenates insightface 512-d identity embedding with EVA
  768-d embedding.
- These helper models are best treated as a separate CPU/GPU preprocessing
  candidate before being admitted into Dinoml runtime.

Layout guards:

- Preserve BFCHW pipeline latent contract and NCTHW VAE contract initially.
- Mark patchify/unpatchify and channel concat as no-layout-translation regions
  until exact axis rewrites are implemented.
- Candidate NDHWC VAE islands must rewrite Conv3d weights, GroupNorm axes,
  temporal cache slicing, and latent scale broadcasts.

## 11. Graph rewrite / lowering opportunities

### Rewrite: ConsisID condition pack

Source pattern:

```text
image -> VAE.encode -> mode/sample -> permute NCTHW to BFCHW
image_latents = 0.7 * image_latents
image_latents = cat([first_frame_latents, zeros], dim=1)
transformer_input = cat([noisy_latents, image_latents], dim=2)
```

Replacement: explicit image-condition encode/pack op.

Preconditions: official `is_kps=false`, latent channels 16, temporal compression
4, BFCHW pipeline layout, H/W divisible by 8. Failure cases: keypoint branch,
alternate VAE scaling/shift, non-default temporal compression, or layout pass
that changes channel axis.

### Rewrite: Conv2d patchify/unpatchify

Source pattern:

```text
[B,F,C,H,W] -> reshape [B*F,C,H,W] -> Conv2d(k=2,s=2)
-> view/flatten to [B,F*(H/2)*(W/2),D]
Linear(D -> 64) -> reshape/permute/flatten -> [B,F,16,H,W]
```

Replacement: explicit video-frame patchify/unpatchify primitive or lowered
Conv2d/GEMM plus inverse token map.

Preconditions: source BFCHW, patch size 2, H/W divisible by 2, output channels
16. Failure cases: `patch_size_t` paths from shared CogVideoX code, NDHWC
without exact weight/layout transform, or non-default patch size.

### Rewrite: CogVideoX joint attention

Source pattern:

```text
cat(text, video) -> Q/K/V Linear -> head reshape -> QK LayerNorm
-> apply RoPE to video-token Q/K only -> SDPA -> output Linear -> split
```

Replacement: canonical joint-attention provider with explicit text/video
segments and RoPE slice.

Preconditions: no mask, noncausal, head dim 64, sequence length within provider
limits, QK LayerNorm parity. Failure cases: backend lacks long sequence support,
fused QKV processor mutation, or text/video segment metadata missing.

### Rewrite: LFE Perceiver attention

Source pattern:

```text
LayerNorm -> Linear Q/KV -> reshape heads -> matmul -> softmax(fp32) -> matmul -> Linear
```

Replacement: small attention primitive over identity/ViT tokens and face tokens.

Preconditions: fixed LFE dims from official config, no masks, no dropout.
Failure cases: missing external face helper tensors, altered number of ViT
scales, or dynamic token count.

### Rewrite: CogVideoX DPM v-pred step

Source pattern:

```text
pred_x0 = sqrt(alpha_t) * sample - sqrt(beta_t) * model_output
prev = mult0 * sample - mult1 * pred_x0 + mult_noise * noise
second-order correction uses old_pred_original_sample after first step
```

Replacement: explicit scheduler-state update with alpha tables and old-pred
buffer.

Preconditions: official scheduler config, trailing timesteps, v-prediction,
same noise generation policy. Failure cases: DDIM scheduler swap, custom
timesteps not supported by this scheduler, eta/variance noise variants.

## 12. Kernel fusion candidates

Highest priority:

- Main transformer GEMMs: 42 layers of 3072-wide QKV/out projections and FFN.
- Joint QKV + QK LayerNorm + RoPE + attention provider prelude for
  `[B,48,17776,64]`.
- Adaptive LayerNormZero scale/shift/gate plus residual epilogues.
- Conv2d patchify/unpatchify for 32-channel BFCHW video latents.
- CFG arithmetic plus v-pred scheduler pointwise kernels over `[B,13,16,60,90]`.

Medium priority:

- Local Facial Extractor mappings, Perceiver attention, and identity
  cross-attention over 17,550 video tokens x 32 face tokens.
- CogVideoX VAE encode/decode Conv3d + GroupNorm + SiLU blocks.
- Image-condition latent pack and zero-padding kernel.
- RoPE/position table generation and cache.

Lower priority:

- Fused CogVideoX QKV processor mutation.
- VAE tiling/slicing blend kernels and temporal cache specialization.
- Full face preprocessing helpers inside Dinoml.
- Keypoint conditioning path and LoRA adapter mutation.

## 13. Runtime staging plan

Stage 1: Parse `BestWishYsh/ConsisID-preview` configs and load transformer/VAE
weights; accept external T5 prompt embeddings and external face tensors.

Stage 2: Implement VAE encode of the conditioning image and image-latent pack
for `is_kps=false`.

Stage 3: Implement transformer patchify/unpatchify, learned positional
embedding, 3D RoPE, timestep embedding, and one `ConsisIDBlock` parity.

Stage 4: Add Local Facial Extractor and Perceiver cross-attention branch, or
explicitly create a temporary non-identity stub target for denoiser bring-up
only. Do not claim ConsisID parity without the identity branch.

Stage 5: Full `ConsisIDTransformer3DModel` forward parity on small synthetic
grids and default config shapes where memory allows.

Stage 6: Add CFG and dynamic-CFG arithmetic.

Stage 7: Implement official `CogVideoXDPMScheduler` v-pred slice with
host-visible old-pred state.

Stage 8: Add CogVideoX VAE decode and short deterministic I2V loop smoke.

Stage 9: Add T5 prompt encoding integration.

Stage 10: Split out face preprocessing, keypoint conditioning, LoRA, and
alternate CogVideoX schedulers as separate admissions.

## 14. Parity and validation plan

- Config/default reconciliation for pipeline, transformer, VAE, scheduler, and
  T5 configs.
- VAE image encode parity: `[B,3,1,480,720] -> [B,16,1,60,90]`.
- Image-condition pack parity including BFCHW permutation, 0.7 scale, zeros,
  and concat with noisy latents.
- Patchify/unpatchify parity for `[B,13,32,60,90] -> tokens -> [B,13,16,60,90]`.
- 3D RoPE cos/sin table parity for 13 x 30 x 45 grid.
- One `ConsisIDBlock` parity with text and video tokens.
- Local Facial Extractor parity for fixed random `id_cond` and list[5]
  `id_vit_hidden` tensors.
- Perceiver cross-attention parity at one injection point.
- Full transformer forward parity on reduced sequence, then default shape if
  memory allows.
- CFG and dynamic CFG schedule parity.
- `CogVideoXDPMScheduler.set_timesteps` and one/two-step parity with fixed
  noise.
- VAE decode parity `[B,16,13,60,90] -> [B,3,49,480,720]`.
- Short deterministic pipeline loop with external prompt/face tensors.
- Suggested tolerances: fp32 scheduler/pointwise `rtol=1e-5, atol=1e-6`;
  transformer/VAE fp32 `rtol=1e-4, atol=1e-5`; bf16/fp16 start at
  `rtol=2e-2, atol=2e-2`.

## 15. Performance probes

- One denoiser step at default 49-frame 480 x 720 and smaller synthetic grids.
- Main attention backend comparison for sequence 17,776, heads 48, head dim 64.
- Transformer block time split: QKV/attention/FFN/adaptive norm.
- LFE and identity cross-attention cost with and without precomputed face tokens.
- VAE encode/decode throughput for one conditioning frame and 13 latent frames.
- Scheduler/CFG overhead by step count.
- Prompt encoder throughput and cache benefit.
- VRAM/workspace usage with transformer, VAE, and T5 resident separately.
- Faithful BFCHW/NCTHW path versus guarded channel-last VAE/patchify islands.

## 16. Scope boundary and separate candidates

Separate candidate reports, not ignored:

- `consisid_face_preprocess`: insightface/facexlib/EVA02-CLIP helper graph,
  ONNX/PyTorch assets, face parsing, alignment, embedding concat, and keypoints.
- `consisid_lora_adapters`: CogVideoX LoRA loader and transformer/text encoder
  adapter mutation.
- `consisid_kps`: keypoint drawing, VAE encode of keypoint condition, and
  changed image-latent packing for configs with `is_kps=true`.
- `cogvideox_vae`: temporal VAE tiling/slicing/caches and codec optimization.
- `cogvideox_scheduler_ddim_dpm`: DDIM and DPM scheduler variants beyond the
  official ConsisID preview config.
- `t5_v1_1_xxl_encoder`: full T5 text encoder admission rather than external
  prompt embeddings.

Ignored/out of scope for this audit:

- Multi-GPU/context parallel paths.
- Callback mutation and interactive interrupt behavior.
- Safety checker/NSFW filtering.
- XLA, NPU, MPS, Flax, and ONNX pipeline variants.
- Training, losses, dropout behavior, and gradient checkpointing.
- Unrelated backend/training/safety/callback paths outside the selected source.

## 17. Final implementation checklist

- [ ] Parse ConsisID model index and component configs.
- [ ] Load `ConsisIDTransformer3DModel` and `AutoencoderKLCogVideoX` weights.
- [ ] Accept external T5 prompt/negative embeddings.
- [ ] Accept external `id_cond` and `id_vit_hidden` tensors.
- [ ] Implement CogVideoX VAE image encode for conditioning.
- [ ] Implement image-condition latent pack and BFCHW channel concat.
- [ ] Implement Conv2d patchify/unpatchify for `[B,F,C,H,W]`.
- [ ] Implement learned positional embedding plus 3D RoPE table application.
- [ ] Implement `CogVideoXLayerNormZero`, `AdaLayerNorm`, and gated residuals.
- [ ] Implement CogVideoX joint attention with QK LayerNorm and video-token RoPE.
- [ ] Implement Local Facial Extractor and Perceiver cross-attention branch.
- [ ] Implement full transformer forward parity.
- [ ] Implement CFG and optional dynamic CFG arithmetic.
- [ ] Implement official `CogVideoXDPMScheduler` v-pred DPM slice.
- [ ] Implement CogVideoX VAE decode with scaling factor 0.7.
- [ ] Add one-step and short-loop parity tests.
- [ ] Add performance probes for attention, FFN, LFE, VAE, scheduler, and memory.
- [ ] Add guarded channel-last layout optimizations only after faithful BFCHW/NCTHW parity.
