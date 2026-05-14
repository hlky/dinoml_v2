# Kandinsky Family Diffusers Audit

Candidate slug: `kandinsky_family`

## 1. Source basis

```text
Diffusers commit/version:
  X:/H/diffusers at b3a515080752a3ba7ca92161e25530c7f280f629.

Model ids / configs inspected:
  Local model_index cache:
    H:/configs/kandinsky-community/kandinsky-2-1/model_index.json
    H:/configs/kandinsky-community/kandinsky-2-1-inpaint/model_index.json
    H:/configs/kandinsky-community/kandinsky-2-1-prior/model_index.json
    H:/configs/kandinsky-community/kandinsky-2-2-decoder/model_index.json
    H:/configs/kandinsky-community/kandinsky-2-2-decoder-inpaint/model_index.json
    H:/configs/kandinsky-community/kandinsky-2-2-prior/model_index.json
    H:/configs/kandinsky-community/kandinsky-2-2-controlnet-depth/model_index.json
    H:/configs/kandinsky-community/kandinsky-3/model_index.json
    H:/configs/kandinskylab/Kandinsky-5.0-T2I-Lite-sft-Diffusers/model_index.json
    H:/configs/kandinskylab/Kandinsky-5.0-I2I-Lite-sft-Diffusers/model_index.json
    H:/configs/kandinskylab/Kandinsky-5.0-T2V-Lite-sft-5s-Diffusers/model_index.json
    H:/configs/kandinskylab/Kandinsky-5.0-I2V-Pro-sft-5s-Diffusers/model_index.json
    H:/configs/kandinskylab/KVAE-2D-1.0/model_index.json
    H:/configs/kandinskylab/KVAE-3D-1.0/model_index.json
  Component configs inspected by official HF raw/API reads, not saved:
    kandinsky-community/kandinsky-2-1
    kandinsky-community/kandinsky-2-1-prior
    kandinsky-community/kandinsky-2-2-decoder
    kandinsky-community/kandinsky-2-2-decoder-inpaint
    kandinsky-community/kandinsky-2-2-prior
    kandinsky-community/kandinsky-3
    kandinskylab/Kandinsky-5.0-T2I-Lite-sft-Diffusers
    kandinskylab/Kandinsky-5.0-I2I-Lite-sft-Diffusers
    kandinskylab/Kandinsky-5.0-T2V-Lite-sft-5s-Diffusers
    kandinskylab/Kandinsky-5.0-I2V-Pro-sft-5s-Diffusers
    kandinskylab/KVAE-2D-1.0
    kandinskylab/KVAE-3D-1.0

Pipeline files inspected:
  pipelines/kandinsky/pipeline_kandinsky_prior.py
  pipelines/kandinsky/pipeline_kandinsky.py
  pipelines/kandinsky/pipeline_kandinsky_img2img.py
  pipelines/kandinsky/pipeline_kandinsky_inpaint.py
  pipelines/kandinsky/pipeline_kandinsky_combined.py
  pipelines/kandinsky/text_encoder.py
  pipelines/kandinsky2_2/pipeline_kandinsky2_2_prior.py
  pipelines/kandinsky2_2/pipeline_kandinsky2_2_prior_emb2emb.py
  pipelines/kandinsky2_2/pipeline_kandinsky2_2.py
  pipelines/kandinsky2_2/pipeline_kandinsky2_2_img2img.py
  pipelines/kandinsky2_2/pipeline_kandinsky2_2_inpainting.py
  pipelines/kandinsky2_2/pipeline_kandinsky2_2_controlnet.py
  pipelines/kandinsky2_2/pipeline_kandinsky2_2_controlnet_img2img.py
  pipelines/kandinsky2_2/pipeline_kandinsky2_2_combined.py
  pipelines/kandinsky3/pipeline_kandinsky3.py
  pipelines/kandinsky3/pipeline_kandinsky3_img2img.py
  pipelines/kandinsky5/pipeline_kandinsky_t2i.py
  pipelines/kandinsky5/pipeline_kandinsky_i2i.py
  pipelines/kandinsky5/pipeline_kandinsky.py
  pipelines/kandinsky5/pipeline_kandinsky_i2v.py
  pipelines/kandinsky5/pipeline_output.py

Model / helper files inspected:
  models/transformers/prior_transformer.py
  models/unets/unet_2d_condition.py
  models/unets/unet_2d_blocks.py
  models/unets/unet_kandinsky3.py
  models/transformers/transformer_kandinsky.py
  models/autoencoders/vq_model.py
  models/autoencoders/autoencoder_kl.py
  models/autoencoders/autoencoder_kl_hunyuan_video.py
  models/autoencoders/autoencoder_kl_kvae.py
  models/attention.py
  models/attention_processor.py
  models/attention_dispatch.py
  models/embeddings.py
  models/normalization.py
  models/resnet.py
  models/downsampling.py
  models/upsampling.py
  image_processor.py
  video_processor.py
  schedulers/scheduling_ddim.py
  schedulers/scheduling_ddpm.py
  schedulers/scheduling_unclip.py
  schedulers/scheduling_flow_match_euler_discrete.py

Missing files or assumptions:
  Local H:/configs mostly had model_index.json only for Kandinsky repos.
  Component configs were readable through official Hugging Face raw/API endpoints.
  No gated official config blocked this audit. Kandinsky 4 cache entries are
  out of scope because this report targets non-deprecated Kandinsky 1/2.2/3/5
  Diffusers folders requested by the task.
```

## 2. Pipeline and component graph

Kandinsky is not one pipeline shape. It has three useful generations:

```text
Kandinsky 2.1 / 2.2:
  prompt/image preprocessing
    -> CLIP prior text/image encoders
    -> PriorTransformer denoises CLIP image embedding with UnCLIPScheduler
    -> decoder UNet2DConditionModel denoises 4-channel latents with image_embeds
    -> VQModel/MOVQ decode -> image postprocess

Kandinsky 3:
  T5 tokenizer/text encoder
    -> Kandinsky3UNet denoises 4-channel latent map with text hidden states
    -> VQModel/MOVQ decode -> image postprocess

Kandinsky 5:
  Qwen2.5-VL prompt encoder + CLIP pooled text encoder
    -> NHWDC-style latent noise / optional encoded image or video condition
    -> Kandinsky5Transformer3DModel denoising loop with FlowMatch Euler
    -> AutoencoderKL image decode or AutoencoderKLHunyuanVideo video decode
    -> image/video postprocess
```

Required first-slice components for Kandinsky 2.2 decoder parity are `UNet2DConditionModel`, `DDPMScheduler`, `VQModel`, and externally supplied `image_embeds` / `negative_image_embeds`. The prior is an independently cacheable stage: prompt CLIP hidden states, text projection, negative prompt embeddings, image encoder zero embedding, prior latents, and final image embedding can all be cached before decoder inference.

Separate candidate reports:

| Surface | Classes / files | Why separate |
| --- | --- | --- |
| Kandinsky 2.x prior | `KandinskyPriorPipeline`, `KandinskyV22PriorPipeline`, `PriorTransformer` | Denoises CLIP image embeddings, not image latents; scheduler is `UnCLIPScheduler`. |
| Kandinsky 2.x img2img | `KandinskyImg2ImgPipeline`, `KandinskyV22Img2ImgPipeline` | Adds VQ encode, strength-based timestep slicing, and scheduler `add_noise`. |
| Kandinsky 2.x inpaint | `KandinskyInpaintPipeline`, `KandinskyV22InpaintPipeline` | UNet input changes to 9 channels for 2.2 inpaint; masks and masked image latents enter the denoiser. |
| Kandinsky 2.2 ControlNet depth | `KandinskyV22ControlnetPipeline`, `KandinskyV22ControlnetImg2ImgPipeline` | Adds hint/depth conditioning branch and control residual contracts. |
| Kandinsky 3 img2img | `Kandinsky3Img2ImgPipeline` | VQ encode + strength slicing for the custom Kandinsky3UNet. |
| Kandinsky 5 image variants | `Kandinsky5T2IPipeline`, `Kandinsky5I2IPipeline` | Same 3D transformer, but I2I concatenates encoded image latents plus mask channels. |
| Kandinsky 5 video variants | `Kandinsky5T2VPipeline`, `Kandinsky5I2VPipeline` | Adds temporal latent frames, sparse temporal attention masks, Hunyuan-style video VAE, and first-frame conditioning. |
| Kandinsky 5 LoRA | `KandinskyLoraLoaderMixin`, Kandinsky5 pipelines | Runtime adapter mutation on transformer/text encoders deserves a loader report. |

No IP-Adapter, GLIGEN, or T2I-Adapter classes are local to these non-deprecated Kandinsky folders. ControlNet exists only for Kandinsky 2.2 in this family.

## 3. Important config dimensions

| Repo | Pipeline | Denoiser | Latent contract | Scheduler |
| --- | --- | --- | --- | --- |
| `kandinsky-community/kandinsky-2-1` | `KandinskyPipeline` | `UNet2DConditionModel`, `in_channels=4`, `out_channels=8`, `sample_size=64`, block channels `[384,768,1152,1536]`, `layers_per_block=3`, `cross_attention_dim=768`, `encoder_hid_dim=1024`, `encoder_hid_dim_type=text_image_proj`, `addition_embed_type=text_image` | MOVQ `VQModel`, latent channels 4, scale 0.18215, source NCHW latents | `DDIMScheduler`, linear beta, epsilon |
| `kandinsky-community/kandinsky-2-1-prior` | `KandinskyPriorPipeline` | `PriorTransformer`, `embedding_dim=768`, `num_layers=20`, `num_attention_heads=32`, `head_dim=64`, `num_embeddings=77` | CLIP image embedding vector `[B,768]` | `UnCLIPScheduler`, `prediction_type=sample`, `variance_type=fixed_small_log` |
| `kandinsky-community/kandinsky-2-2-decoder` | `KandinskyV22Pipeline` | `UNet2DConditionModel`, `in_channels=4`, `out_channels=8`, same block widths, `encoder_hid_dim=1280`, `encoder_hid_dim_type=image_proj`, `addition_embed_type=image` | MOVQ `VQModel`, latent channels 4, scale 0.18215 | `DDPMScheduler`, linear beta, epsilon |
| `kandinsky-community/kandinsky-2-2-decoder-inpaint` | `KandinskyV22InpaintPipeline` | Same UNet widths but `in_channels=9`, `out_channels=8` | Latents + mask + masked image latents | `DDPMScheduler`, linear beta, epsilon |
| `kandinsky-community/kandinsky-2-2-prior` | `KandinskyV22PriorPipeline` | `PriorTransformer`, `embedding_dim=1280`, `num_layers=20`, `num_attention_heads=32`, `head_dim=64`, `num_embeddings=77` | CLIP image embedding vector `[B,1280]` | `UnCLIPScheduler`, `prediction_type=sample`, `variance_type=fixed_small_log` |
| `kandinsky-community/kandinsky-3` | `Kandinsky3Pipeline` | `Kandinsky3UNet`, `in_channels=4`, block channels `[384,768,1536,3072]`, `layers_per_block=3`, `cross_attention_dim=4096` | MOVQ `VQModel`, latent channels 4, scale 0.18215, wider codec channels `[256,512,512,1024]` | `DDPMScheduler`, squared-cosine beta, epsilon, fixed-small variance |
| `kandinskylab/Kandinsky-5.0-T2I-Lite-sft-Diffusers` | `Kandinsky5T2IPipeline` | `Kandinsky5Transformer3DModel`, source defaults `in_visual_dim=4`, `model_dim=2048`, `ff_dim=5120`, text blocks 2, visual blocks 32, patch `(1,2,2)`, heads inferred as `2048 / sum(16,24,24)=32` | Image VAE `AutoencoderKL`, latent channels 16, scale 0.3611, shift 0.1159; pipeline transformer latent channels come from config `in_visual_dim` | `FlowMatchEulerDiscreteScheduler`, shift 5.0 |
| `kandinskylab/Kandinsky-5.0-T2V-Lite-sft-5s-Diffusers` | `Kandinsky5T2VPipeline` | Same transformer class and patch config | Video VAE `AutoencoderKLHunyuanVideo`, latent channels 16, spatial compression 8, temporal compression 4, scale 0.476986 | `FlowMatchEulerDiscreteScheduler`, shift 5.0 |

Text/image encoders:

| Family | Tokenizer / processor | Encoder output used by pipeline |
| --- | --- | --- |
| Kandinsky 2.1 decoder | `XLMRobertaTokenizerFast`, local `MultilingualCLIP` | Decoder text hidden states plus prior image embedding in `text_image_proj` path. |
| Kandinsky 2.1 prior | `CLIPTokenizer`; `CLIPTextModelWithProjection`; `CLIPVisionModelWithProjection`; `CLIPImageProcessor` | `text_embeds`, `last_hidden_state`, text mask, image embeddings. |
| Kandinsky 2.2 prior | CLIP text/vision configs are larger: text hidden/projection 1280, vision hidden 1664, projection 1280 | Same prior contract, wider embedding. |
| Kandinsky 3 | `T5Tokenizer`, `T5EncoderModel`, source slices/truncates to `max_sequence_length=128` by default | `prompt_embeds [B,seq,4096]` and bool attention mask. |
| Kandinsky 5 | `Qwen2VLProcessor`, `Qwen2_5_VLForConditionalGeneration`; `CLIPTokenizer`, `CLIPTextModel` | Qwen hidden states `[B,S,3584]`, `cu_seqlens`, and CLIP pooled `[B,768]`. |

## 3a. Family variation traps

- Kandinsky 2.1 and 2.2 share a UNet shape but differ in conditioning: 2.1 uses `addition_embed_type=text_image` with 1024-dimensional multilingual CLIP projection; 2.2 decoder uses image-only conditioning with 1280-dimensional CLIP image embeddings.
- Decoder UNets output 8 channels while latent state is 4 channels. Scheduler variance handling may preserve or split the second 4-channel half depending on scheduler `variance_type`.
- Kandinsky 2.x prior latents are embedding vectors `[B,embedding_dim]`, not spatial latent maps. Do not admit it as a normal UNet scheduler loop.
- Kandinsky 3 uses a custom `Kandinsky3UNet`, not `UNet2DConditionModel`; it has conditional GroupNorm, conv-transpose upsampling, custom attention pooling, and 1x1/3x3 bottleneck ResNet blocks.
- Kandinsky 5 source transformer latents are `[B,T,H,W,C]` before patching, unlike the NCHW/NCDHW VAEs. The report's first implementation should preserve this layout at the transformer boundary or insert explicit, guarded transposes.
- Kandinsky 5 component configs inspected show `transformer/config.json` only exposing `_class_name` and `patch_size`; many dimensions come from source defaults. Treat them as effective source defaults unless a checkpoint weight shape probe later proves otherwise.
- Kandinsky 5 image VAE config has 16 latent channels, but source transformer default `in_visual_dim=4`. This mismatch means first admission must verify actual transformer config/weights before assuming a 16-channel image latent denoiser. The pipeline uses `self.transformer.config.in_visual_dim` for latent channels and slices `latents[..., :num_channels_latents]` before VAE decode.
- Kandinsky 5 T2V/I2V uses `AutoencoderKLHunyuanVideo`, not KVAE configs directly; `KVAE-2D/3D` official cache entries are codec-adjacent and should be a separate codec audit if selected.
- Layout-sensitive axes: Kandinsky 2/3 latents and VQ/MOVQ use NCHW with channel dim 1; Kandinsky 5 transformer uses channel-last NHWDC; VAE/video codec uses NCHW or NCTHW. Any layout pass must rewrite `cat(dim=1)` vs `cat(dim=-1)`, GroupNorm channel axes, `permute(0,2,3,1)` postprocess, and patchify/unpatchify view order.

## 4. Runtime tensor contract

Kandinsky 2.2 decoder first-slice tensors:

```text
image_embeds: [B, 1280], dtype = UNet dtype, positive prior output
negative_image_embeds: [B, 1280], same dtype
CFG conditioning: cat([negative_image_embeds, image_embeds], dim=0) after repeat_interleave
latents: [B * num_images_per_prompt, 4, H/8, W/8], NCHW, multiplied by scheduler.init_noise_sigma
UNet input under CFG: [2B, 4, H/8, W/8]
UNet output: [2B, 8, H/8, W/8] if CFG; split into noise and variance halves
scheduler step input: [B, 8, H/8, W/8] when learned variance is active, otherwise [B, 4, H/8, W/8]
MOVQ decode input: [B, 4, H/8, W/8], `force_not_quantize=True`
image output: [B, 3, H, W] then clamp/rescale and optional NHWC CPU output
```

Kandinsky 2.x prior tensors:

```text
token ids / masks: [B, 77]
CLIP text_embeds: [B, 768 or 1280]
CLIP last_hidden_state: [B, 77, 768 or 1280]
prior latent sample: [B, 768 or 1280]
PriorTransformer sequence: text tokens + projected text embedding + timestep embedding + image embedding (+ optional PRD token)
prior output: predicted_image_embedding [B, 768 or 1280]
post_process_latents -> image_embeds / negative_image_embeds
```

Kandinsky 3 tensors:

```text
T5 prompt_embeds: [B, S<=128, 4096], attention_mask [B,S]
latents: [B,4,H/8,W/8], NCHW
UNet timestep embedding: broadcast [B], sinusoidal -> MLP
Kandinsky3UNet output: [B,4,H/8,W/8]
MOVQ decode: force_not_quantize=True
```

Kandinsky 5 image tensors:

```text
Qwen prompt_embeds_qwen: [B,S<=max_sequence_length,3584]
prompt_cu_seqlens: [B+1], int32 cumulative lengths
CLIP prompt_embeds_clip: [B,768]
latents before transformer: [B,1,H/8,W/8,Cv], channel-last; Cv = transformer.config.in_visual_dim
patchify: view [B,T/1,1,H/16,2,W/16,2,Cv] -> permute -> Linear((1*2*2*Cv) -> 2048)
visual tokens inside transformer: [B,T,H/16,W/16,2048] then optional flatten
transformer output: [B,1,H/8,W/8,Cv]
VAE decode bridge: reshape to [B,Cv,H/8,W/8], divide by vae.config.scaling_factor, AutoencoderKL.decode
```

Kandinsky 5 I2I adds `image_latents` from `vae.encode(image).latent_dist.sample`, scales by VAE factor, unsqueezes a temporal dimension, permutes to `[B,1,H/8,W/8,C]`, and concatenates `[noise, image_latents, ones_mask]` on the last channel. I2V similarly replaces or protects the first latent frame and concatenates a visual conditioning tensor plus mask.

CPU/data-pipeline work includes tokenization, prompt cleaning, Qwen chat-template construction, image/video resize, PIL/NumPy postprocess, and scheduler table setup. GPU/runtime work includes text encoder calls if included, denoiser/codec execution, CFG arithmetic, scheduler step arithmetic, VQ/Autoencoder scaling, masks, and layout transforms. Prompt embeddings, prior image embeddings, zero image embeddings, scheduler tables, RoPE tables/positions, and VAE image latents are cacheable.

## 5. Operator coverage checklist

Tensor/layout ops:

- NCHW latent allocation, concat/chunk/split, repeat/repeat_interleave, reshape/view, permute, flatten/unflatten, pad, clamp.
- Kandinsky 5 channel-last `[B,T,H,W,C]` patchify/unpatchify with exact view/permute order.
- Inpaint/I2I masks: nearest/bilinear preprocessing, mask latent broadcast, `cat(dim=1)` for 2.x inpaint, `cat(dim=-1)` for 5.x visual conditioning.

Convolution/downsample/upsample:

- Kandinsky 2.x `UNet2DConditionModel`: ResNet down/up blocks, `SimpleCrossAttn*` blocks, 3x3 convs, downsample/upsample.
- VQ/MOVQ: `Encoder`/`Decoder` conv stacks, `DownEncoderBlock2D`, `AttnDownEncoderBlock2D`, `AttnUpDecoderBlock2D`, `post_quant_conv`.
- Kandinsky 3: 1x1 and 3x3 Conv2d bottleneck ResNet blocks, Conv2d stride-2 downsample, ConvTranspose2d stride-2 upsample.
- Kandinsky 5 image VAE: AutoencoderKL Conv2d/ResNet/upsample/downsample. Video VAE: HunyuanVideo 3D causal conv/down/up blocks.

GEMM/linear:

- PriorTransformer projections, transformer FFN, time embedding MLP, CLIP projection.
- Kandinsky 3 conditional GroupNorm MLP and attention Q/K/V/out projections.
- Kandinsky 5 Qwen/CLIP external encoders, patch Linear, time/text Linear+LayerNorm, Q/K/V/out Linear, GELU FFN, modulation Linear, output Linear.

Attention:

- PriorTransformer causal/sequence self-attention with additive mask repeated per head.
- UNet `SimpleCrossAttn` and VQ mid/block attention through shared `Attention`.
- Kandinsky3 2D self/cross attention over `H*W` tokens, plus attention pooling over text.
- Kandinsky5 RMSNorm Q/K, RoPE, self attention on text, self attention on visual tokens, cross attention from visual to text, dispatch through `dispatch_attention_fn`.

Normalization/adaptive conditioning:

- GroupNorm, LayerNorm, RMSNorm, conditional GroupNorm, Ada-style modulation with shift/scale/gate.
- VQ spatial norm branch if config uses it; sampled Kandinsky VQ configs use standard VQModel path but source supports spatial norm.

Scheduler/guidance:

- DDIM/DDPM epsilon, UnCLIP sample prediction for prior, FlowMatch Euler velocity update.
- CFG concatenated batch for 2.x/3, separate positive/negative transformer calls for Kandinsky 5.
- Learned-variance channel split for 2.x decoder when scheduler config requires it.

## 6. Denoiser/model breakdown

Kandinsky 2.2 decoder UNet:

```text
sample [B,4,H,W]
  -> time embedding
  -> added_cond_kwargs["image_embeds"] projected by UNet addition embedding
  -> down blocks:
       ResnetDownsampleBlock2D then SimpleCrossAttnDownBlock2D x3
  -> mid block attention/resnet
  -> up blocks:
       SimpleCrossAttnUpBlock2D x3 then ResnetUpsampleBlock2D
  -> output conv [B,8,H,W]
```

2.1 is structurally similar but the conditioning projection combines text and image embeddings. In decoder-only staging, provide the image embedding externally and avoid admitting the prior in the same artifact.

PriorTransformer:

```text
hidden image embedding [B,D]
  -> timestep sinusoidal projection + TimestepEmbedding
  -> optional LayerNorm/projection of CLIP text/image projection
  -> concat text hidden states, projected text/image embedding, time token, current image embedding
  -> positional embedding + causal/additive mask
  -> BasicTransformerBlock stack, depth 20
  -> norm_out and proj_to_clip_embeddings
  -> post_process_latents
```

Kandinsky3UNet:

```text
latents [B,4,H,W]
  -> conv_in to 192 channels
  -> T5 hidden projection and time embedding
  -> down blocks with conditional GroupNorm -> SiLU -> Conv2d bottlenecks,
     self-attention and cross-attention where configured
  -> up blocks concatenate skip states on channel axis
  -> conditional GroupNorm -> SiLU -> Conv2d out [B,4,H,W]
```

Kandinsky5Transformer3DModel:

```text
hidden_states [B,T,H,W,Cv]
  -> visual patch Linear over patch (1,2,2)
  -> Qwen text Linear+LayerNorm, CLIP pooled Linear added to time embedding
  -> text encoder blocks: AdaLayerNorm-like modulation, self attention + RoPE, GELU FFN
  -> visual decoder blocks: visual self attention + 3D RoPE, visual/text cross attention, GELU FFN
  -> output modulation + Linear
  -> unpatchify to [B,T,H,W,Cv]
```

The Kandinsky5 attention processor normalizes Q/K in fp32 via RMSNorm, applies RoPE in fp32, casts RoPE output to bfloat16 in source, and calls the attention dispatch registry. That cast is a parity hazard if running fp16/fp32 variants.

## 7. Attention requirements

| Target | Attention path | Requirements |
| --- | --- | --- |
| PriorTransformer | `BasicTransformerBlock` from `attention.py` | Sequence self-attention, additive causal/text mask, D=768 or 1280, 32 heads x 64, no spatial layout. |
| Kandinsky 2.x decoder | `UNet2DConditionModel` and `SimpleCrossAttn*` | 2D latent self/cross attention, image/text addition embedding, attention mask support through shared processor. |
| VQ/MOVQ | autoencoder attention blocks | Spatial attention in high-level blocks; NCHW to sequence and back inside shared attention. |
| Kandinsky3 | `Kandinsky3AttentionBlock` | Flatten NCHW to `[B,H*W,C]`, self/cross attention with optional text mask, then restore NCHW; plus attention pooling. |
| Kandinsky5 | `Kandinsky5AttnProcessor` | Linear QKV, RMSNorm Q/K, 1D or 3D RoPE, SDPA/dispatch backend, optional sparse STA mask, cross attention from visual to text. |

`attention_processor.py` is enough for the classic UNet/VQ paths. `attention_dispatch.py` is the important backend boundary for Kandinsky 5. Dinoml flash-style attention is valid only under guarded preconditions: dense masks or supported block masks, matching head dim 64, Q/K RMSNorm folded or fused, RoPE applied in the same dtype semantics, and no unsupported sparse/fractal path. Eager/shared attention defines parity for 2.x/3; Kandinsky5 parity is the source `dispatch_attention_fn` behavior with its Q/K normalization and RoPE.

## 8. Scheduler and denoising-loop contract

Kandinsky 2.1 decoder uses DDIM epsilon in sampled config. Kandinsky 2.2 decoder uses DDPMScheduler epsilon. In both decoder loops:

```text
scheduler.set_timesteps(num_inference_steps)
latents = randn(shape) * init_noise_sigma
for t in timesteps:
  latent_model_input = cat([latents]*2) if CFG else latents
  noise_pred = unet(latent_model_input, t, image/text conditioning)
  if CFG:
    split noise/variance halves, CFG only noise half, keep text variance half
  if scheduler variance is not learned:
    drop variance half
  latents = scheduler.step(noise_pred, t, latents).prev_sample
```

The prior loop uses `UnCLIPScheduler`, `prediction_type=sample`, vector latents, and an explicit `prev_timestep` argument. It needs separate scheduler admission from image-latent DDPM/DDIM.

Kandinsky 3 sampled config is DDPMScheduler epsilon with squared-cosine beta and fixed-small variance. The loop does normal concatenated CFG, but the formula is `(guidance_scale + 1) * text - guidance_scale * uncond`, equivalent to `text + guidance_scale * (text - uncond)`.

Kandinsky 5 uses FlowMatch Euler with shift 5.0 and true separate positive/negative transformer calls for CFG:

```text
pred_velocity = transformer(latents, positive text)
uncond_pred_velocity = transformer(latents, negative text)
pred_velocity = uncond + guidance_scale * (pred_velocity - uncond)
latents = scheduler.step(pred_velocity, t, latents)
```

For I2I/T2V/I2V variants only the denoised latent channel slice is updated (`latents[..., :Cv]` or `latents[:, 1:, ..., :Cv]`), preserving visual conditioning/mask channels. First Dinoml scheduler slices: DDPMScheduler epsilon for Kandinsky 2.2 decoder and Kandinsky 3; UnCLIP scheduler later for prior; FlowMatch Euler static shift later for Kandinsky 5.

## 9. Position, timestep, and custom math

- PriorTransformer uses sinusoidal timestep projection, MLP time embedding, learned positional embeddings, optional positional padding, and a causal attention mask.
- Kandinsky3 uses sinusoidal time projection plus TimestepEmbedding. Its custom `Kandinsky3ConditionalGroupNorm` computes `GroupNorm(x) * (scale + 1) + shift`, where scale/shift come from `SiLU(time_embed) -> Linear(2*C)`.
- Kandinsky3 attention flattens spatial maps via `reshape(B,C,H*W).permute(0,2,1)` and must restore exactly; this region is a no-layout-translation guard unless the whole attention block is rewritten.
- Kandinsky5 time embedding computes cos/sin over `torch.outer(time, freqs)`, then Linear -> SiLU -> Linear. CLIP pooled text embedding is projected to the time embedding dimension and added.
- Kandinsky5 1D/3D RoPE constructs 2x2 rotation matrices from cos/sin. Visual RoPE concatenates temporal, height, and width frequency bands with `axes_dims=(16,24,24)`.
- Kandinsky5 sparse temporal attention helper builds masks for video variants. First image slice can set `sparse_params=None`; video slices need separate mask parity.

Precompute candidates: prior positional embeddings, scheduler scalar tables, zero image embedding, Qwen/CLIP prompt embeddings, `text_rope_pos`, `visual_rope_pos`, and RoPE frequency buffers. Dynamic dependencies: resolution, frame count, prompt sequence length, CFG scale, and scheduler timestep.

## 10. Preprocessing and input packing

Kandinsky 2.x prior tokenizes to CLIP max length 77, uses bool attention masks, repeats per image, and concatenates unconditional/conditional batches for CFG. `get_zero_embed` runs the CLIP vision encoder on an all-zero `[1,3,224,224]` image to form the negative image embedding when no negative prior prompt is provided.

Kandinsky 2.x decoder accepts prior embeddings directly. Img2img encodes the source image through MOVQ/VQ, adds noise at a strength-derived timestep, then denoises. Inpaint prepares mask/masked image tensors and feeds a 9-channel UNet for 2.2 inpaint.

Kandinsky 3 tokenizes T5 prompts, supports precomputed prompt embeddings and masks, concatenates negative/positive embeddings under CFG, and decodes MOVQ latents exactly like 2.x.

Kandinsky 5 builds Qwen prompts through a template and slices hidden states after `prompt_template_encode_start_idx` (`55` for T2I/I2I, `129` for video). It also obtains CLIP pooled embeddings with max length 77. Qwen embeddings are repeated by reshaping sequence batches and `cu_seqlens` are reconstructed from original prompt lengths.

Kandinsky 5 T2I initializes `[B,1,H/8,W/8,Cv]` random latents. I2I encodes an input image with VAE, scales it, permutes to channel-last, and concatenates noise/image/mask. T2V/I2V uses `num_latent_frames=(num_frames-1)//temporal_scale+1`, validates `num_frames % temporal_scale == 1`, and bridges channel-last transformer latents back to NCTHW video latents before VAE decode.

## 11. Graph rewrite / lowering opportunities

**VQ decode scale and postprocess**

- Source pattern: `movq.decode(latents, force_not_quantize=True).sample`, then `image * 0.5 + 0.5`, clamp, CPU `permute`.
- Replacement: keep decode as codec stage; fuse post-decode affine+clamp on GPU when output remains tensor.
- Preconditions: output type tensor/np/pil choice known; VQ config uses direct latent decode; no quantization codebook lookup.
- Failure cases: `force_not_quantize=False`, spatial norm configs with extra `zq`, tiled decode.
- Test: random latent decode parity plus postprocess parity.

**Kandinsky 2.x variance split**

- Source pattern: UNet outputs 8 channels; split noise/variance; CFG only noise; scheduler may require both.
- Replacement: graph-level split node with scheduler-family guard.
- Preconditions: latent channels 4, output channels 8, DDPMScheduler learned variance status known.
- Failure cases: scheduler config without learned variance, inpaint 9-channel input but still 4 latent channels.
- Test: one denoising step with learned and fixed variance configs.

**Kandinsky3 spatial attention flatten**

- Source pattern: NCHW -> `[B,H*W,C]` -> attention -> NCHW.
- Replacement: canonical 2D attention lowering or fused NCHW attention bridge.
- Preconditions: contiguous NCHW, static H/W for first slice, no layout translation across block.
- Failure cases: dynamic H/W without shape guards, NHWC translation that changes flatten order.
- Test: single `Kandinsky3AttentionBlock` parity.

**Kandinsky5 patchify/unpatchify**

- Source pattern: channel-last `view -> permute -> flatten -> Linear`, inverse `Linear -> view -> permute -> flatten`.
- Replacement: patch embedding provider or GEMM with explicit activation layout.
- Preconditions: `T,H,W` divisible by `(1,2,2)`, channel-last contiguous, patch size exactly config.
- Failure cases: visual conditioning changes channel count, non-contiguous latents, patch size variants.
- Test: patchify+unpatchify identity-style weight test and full transformer input/output shape test.

**Kandinsky5 QKV+RMSNorm+RoPE attention**

- Source pattern: Linear Q/K/V -> reshape heads -> RMSNorm Q/K -> RoPE -> dispatch attention -> Linear.
- Replacement: fused attention prep plus provider-backed attention.
- Preconditions: head dim 64, dense or supported sparse mask, RoPE dtype semantics preserved.
- Failure cases: sparse STA mask unsupported, source bfloat16 cast not matched, variable sequence/fractal flatten path.
- Test: block-level parity for text self-attn, visual self-attn, visual/text cross-attn.

## 12. Kernel fusion candidates

Highest priority:

- Conv2d + GroupNorm + SiLU and ResNet blocks for VQ/MOVQ and classic UNet paths. Kandinsky 2.x and 3 spend most non-attention time here.
- Attention QKV/SDPA/out projection for UNet, PriorTransformer, and Kandinsky3 spatial attention.
- Kandinsky 2.x decoder CFG + variance split + scheduler step, because it is bandwidth-bound and loop-side.
- Kandinsky5 patchify/unpatchify and QKV + RMSNorm + RoPE + attention, because transformer tokens dominate runtime.

Medium priority:

- Conditional GroupNorm / AdaLayerNorm modulation with residual gates for Kandinsky3 and 5.
- VQ/MOVQ decode Conv2d/attention/upsample fusion as an independently useful codec island.
- FlowMatch Euler step and true CFG arithmetic for Kandinsky5.
- HunyuanVideo VAE Conv3d/temporal decode blocks for Kandinsky5 video.

Lower priority:

- PriorTransformer full compile; useful but cacheable and smaller than decoder loops.
- Sparse temporal attention mask construction and fractal flatten/unflatten for Kandinsky5 video.
- Runtime LoRA merge/unmerge and loader mutation.

## 13. Runtime staging plan

1. Parse configs and load weights for `kandinsky-community/kandinsky-2-2-decoder`, with prompt/image prior embeddings supplied externally.
2. Implement one Kandinsky 2.2 decoder denoising step: UNet2DConditionModel, DDPMScheduler epsilon, CFG, learned/fixed variance split, and MOVQ decode as optional separate stage.
3. Validate MOVQ decode separately with `force_not_quantize=True` and sampled VQ config.
4. Add Kandinsky 2.2 prior as a separate embedding-vector pipeline: CLIP text/image encoders external first, then `PriorTransformer` and `UnCLIPScheduler`.
5. Add Kandinsky 3 custom UNet after 2.x decoder primitives are stable; its custom conditional GroupNorm and attention blocks need focused tests.
6. Add Kandinsky 5 T2I only after FlowMatch Euler and transformer patch/attention primitives are admitted; keep Qwen and CLIP encoders external at first.
7. Add Kandinsky 5 I2I/T2V/I2V as separate variant slices for VAE encode, visual conditioning channels, temporal latents, and video VAE decode.

First Dinoml staging/admission recommendation: start with **Kandinsky 2.2 decoder-only**. It reuses the existing Stable Diffusion-class UNet/VQ/scheduler primitive families, has clear external `image_embeds` inputs, and avoids the prior and Kandinsky5 text/3D transformer complexity. Admit it as a bounded runtime target: static NCHW image latents, `float16`/`bfloat16` CUDA first, fixed representative resolution, DDPMScheduler epsilon, and VQ decode either separate or stubbed.

## 14. Parity and validation plan

- Config parsing tests for all sampled repos: assert component class names, latent channel counts, scheduler class/prediction type, and effective source defaults for Kandinsky5 transformer fields omitted from config.
- MOVQ/VQ random latent decode parity with `force_not_quantize=True`; separate encode/decode roundtrip only for img2img.
- Kandinsky 2.2 decoder one-step parity with fixed `image_embeds`, `negative_image_embeds`, timestep, latents, and scheduler state.
- Kandinsky 2.2 prior vector denoising parity for one `UnCLIPScheduler` step and full short loop with fixed CLIP embeddings.
- Kandinsky3 block parity: conditional GroupNorm, ResNetBlock, AttentionBlock, and one full UNet forward at small latent size.
- Kandinsky5 patchify/unpatchify parity, RoPE parity, attention block parity, and one transformer forward with synthetic Qwen/CLIP embeddings.
- Scheduler parity: DDPM epsilon fixed variance, DDIM epsilon, UnCLIP sample prediction, FlowMatch Euler static shift.
- Short deterministic denoising-loop smoke for decoder-only, then end-to-end pipeline smoke with precomputed prior embeddings.

Suggested tolerances: fp32 scheduler and layout ops `rtol=1e-5, atol=1e-6`; fp16/bf16 denoiser blocks `rtol=2e-2, atol=2e-2` initially; codec image decode compare in latent/image tensors before PIL conversion.

## 15. Performance probes

- PriorTransformer embedding-vector denoising time vs decoder UNet time.
- Kandinsky 2.2 one UNet step by resolution and batch/CFG state.
- MOVQ decode throughput for 512, 768, and 1024 output sizes.
- Kandinsky3 custom UNet time split: conv/resnet vs attention.
- Kandinsky5 transformer token sweep by image resolution: `[T,H/16,W/16]` visual tokens and Qwen text length.
- Kandinsky5 Qwen/CLIP prompt encoding time vs cached prompt-embedding path.
- FlowMatch scheduler/CFG overhead for separate positive/negative transformer calls.
- Video VAE decode throughput by frame count and temporal compression.
- VRAM for Kandinsky5 attention, especially dense vs sparse STA masks.

## 16. Scope boundary and separate candidates

Separate candidate reports:

- `kandinsky_2x_prior`: `PriorTransformer`, CLIP text/vision encoders, `UnCLIPScheduler`, interpolation.
- `kandinsky_2x_img2img_inpaint`: MOVQ encode, strength slicing, mask contracts, 9-channel inpaint UNet.
- `kandinsky_2_2_controlnet_depth`: depth/control conditioning pipeline.
- `kandinsky3_custom_unet`: custom UNet blocks and T5 conditioning.
- `kandinsky5_t2i_transformer`: Qwen/CLIP conditioning, channel-last transformer, FlowMatch.
- `kandinsky5_video`: T2V/I2V temporal latents, sparse temporal attention, HunyuanVideo VAE.
- `kandinsky5_lora`: `KandinskyLoraLoaderMixin` runtime adapter state.
- `kandinsky_codecs`: MOVQ/VQModel, AutoencoderKL image, AutoencoderKLHunyuanVideo, and KVAE 2D/3D codec comparison.

Ignored/out of scope for this audit: multi-GPU/context parallel, callbacks/interrupt mutation, XLA/NPU/MPS/Flax/ONNX, safety/NSFW, training/loss/dropout/gradient checkpointing, and Kandinsky 4 cache entries.

## 17. Final implementation checklist

- [ ] Parse Kandinsky 2.2 decoder component configs and expose image embedding inputs.
- [ ] Load UNet2DConditionModel and VQModel weights with explicit NCHW latent contracts.
- [ ] Admit DDPMScheduler epsilon fixed/learned variance step for decoder loop.
- [ ] Implement CFG + variance split graph pattern.
- [ ] Validate one decoder denoising step against Diffusers.
- [ ] Validate MOVQ decode with `force_not_quantize=True`.
- [ ] Add short decoder-only loop parity with precomputed image embeddings.
- [ ] Add PriorTransformer/UnCLIP as a separate second-stage candidate.
- [ ] Add Kandinsky3 conditional GroupNorm and custom attention block parity tests.
- [ ] Add Kandinsky5 patchify/RoPE/attention prototypes after FlowMatch support is stable.
