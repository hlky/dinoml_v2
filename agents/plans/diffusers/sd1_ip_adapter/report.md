# SD1 IP-Adapter runtime audit

## 1. Source basis

Diffusers commit/version: local checkout `X:/H/diffusers` at `b3a515080752a3ba7ca92161e25530c7f280f629`.

Model id(s):

- Base family reference: Stable Diffusion 1.x / 1.5 from `../stable_diffusion_1_5/report.md`.
- IP-Adapter weights/configs: `h94/IP-Adapter`.
- Related candidate inventory only: `h94/IP-Adapter-FaceID`, `InstantX/SD3.5-Large-IP-Adapter`, `InstantX/FLUX.1-dev-IP-Adapter`.

Config sources:

| Repo/path | Source | Notes |
| --- | --- | --- |
| `H:/configs/h94/IP-Adapter/model_index.json` | local cache | Empty `{}` placeholder, not useful as component config. |
| `h94/IP-Adapter/models/image_encoder/config.json` | official config inspected via `huggingface_hub` after local placeholder check | SD1 CLIP vision config: hidden 1280, projection 1024, 32 layers, 16 heads, image 224, patch 14, fp16 metadata. Not persisted under `H:/configs` because this task's owned write path is report-only. |
| `h94/IP-Adapter/sdxl_models/image_encoder/config.json` | official config inspected via `huggingface_hub` for candidate inventory | SDXL ViT-H/G style config: hidden 1664, projection 1280, 48 layers, 16 heads, image 224, patch 14. Not persisted under `H:/configs` because this task's owned write path is report-only. |
| `h94/IP-Adapter` safetensors headers | official repo via `huggingface_hub` | Inspected headers for `models/ip-adapter_sd15.safetensors`, `ip-adapter-plus_sd15.safetensors`, `ip-adapter_sd15_vit-G.safetensors`, `ip-adapter-full-face_sd15.safetensors`, `ip-adapter-plus-face_sd15.safetensors`. |
| `H:/configs/InstantX/SD3.5-Large-IP-Adapter/model_index.json`, `H:/configs/InstantX/FLUX.1-dev-IP-Adapter/model_index.json` | local cache | Empty `{}` placeholders; listed only for separate candidate inventory. |

Pipeline files inspected:

- `src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py`
- Existing local report: `agents/plans/diffusers/stable_diffusion_1_5/report.md`
- Existing style reference only: `agents/plans/diffusers/controlnet_sd/report.md`

Model/helper files inspected:

- `src/diffusers/loaders/ip_adapter.py`
- `src/diffusers/loaders/unet.py`
- `src/diffusers/models/unets/unet_2d_condition.py`
- `src/diffusers/models/attention_processor.py`
- `src/diffusers/models/embeddings.py`
- `src/diffusers/image_processor.py`
- Candidate-only paths: `src/diffusers/loaders/transformer_sd3.py`, `src/diffusers/loaders/transformer_flux.py`, `src/diffusers/models/transformers/transformer_sd3.py`, `src/diffusers/models/transformers/transformer_flux.py`

Missing files or assumptions:

- No gated official config blocked this audit. The official `h94/IP-Adapter` repo does not provide a meaningful `model_index.json`; image encoder configs were inspected from official repo subfolders after checking local cache.
- This report does not audit base SD1 UNet/VAE/scheduler again; it inherits that runtime surface from `stable_diffusion_1_5/report.md`.
- Ignored per task: XLA/NPU/MPS/Flax/ONNX, safety/NSFW, training/loss/dropout/gradient checkpointing, multi-GPU/context parallel, callbacks/interrupt.

## 2. Runtime surface summary

SD1 IP-Adapter is not a separate denoiser. It mutates a loaded SD1 UNet so cross-attention processors receive extra image-derived K/V branches.

```text
prompt + optional negative prompt
  -> CLIP text encoder -> prompt_embeds [B,77,768]
ip_adapter_image or precomputed ip_adapter_image_embeds
  -> CLIP vision encoder or caller-supplied embeds
  -> MultiIPAdapterImageProjection
  -> per-adapter image tokens
latents + timestep + prompt_embeds + image tokens
  -> SD1 UNet with IPAdapterAttnProcessor on cross-attn layers
  -> scheduler/CFG/VAE decode from base SD1 report
```

The loader path is:

1. `IPAdapterMixin.load_ip_adapter(...)` reads state dicts with top-level `image_proj` and `ip_adapter` groups.
2. If needed, it registers a `CLIPVisionModelWithProjection` image encoder and `CLIPImageProcessor`.
3. `UNet2DConditionLoadersMixin._load_ip_adapter_weights(...)` replaces UNet cross-attention processors and installs `MultiIPAdapterImageProjection`.
4. The UNet config is mutated to `encoder_hid_dim_type="ip_image_proj"`.
5. At pipeline call time, `added_cond_kwargs={"image_embeds": image_embeds}` enters `UNet2DConditionModel.forward`.
6. `UNet2DConditionModel.process_encoder_hidden_states(...)` projects image embeds and rewrites `encoder_hidden_states` into `(text_encoder_hidden_states, projected_image_embeds)`.
7. Each IP-aware cross-attention processor computes normal text cross-attention plus one or more independent image K/V attentions and adds scaled image-attention output before the attention output projection.

Required SD1 components remain the base SD1 components: tokenizer, CLIP text encoder, UNet, VAE, scheduler. IP-Adapter adds an optional CLIP vision encoder plus feature extractor when callers use `ip_adapter_image`; callers can bypass this with precomputed `ip_adapter_image_embeds`.

## 3. Classes and files

| Surface | Class/function | File | SD1 role |
| --- | --- | --- | --- |
| Pipeline mixin | `IPAdapterMixin` | `loaders/ip_adapter.py` | Loads adapter weights, image encoder, feature extractor; sets scales; unloads mutation. |
| UNet loader | `_convert_ip_adapter_attn_to_diffusers` | `loaders/unet.py` | Creates IP-aware processors for cross-attention layers and loads `to_k_ip`/`to_v_ip`. |
| UNet loader | `_convert_ip_adapter_image_proj_to_diffusers` | `loaders/unet.py` | Infers adapter projection type from weight keys and constructs projection module. |
| UNet loader | `_load_ip_adapter_weights` | `loaders/unet.py` | Calls conversion, installs processors, installs `MultiIPAdapterImageProjection`, mutates `encoder_hid_dim_type`. |
| UNet forward | `process_encoder_hidden_states` | `models/unets/unet_2d_condition.py` | Requires `added_cond_kwargs["image_embeds"]`, projects image tokens, passes tuple to attention. |
| Projection | `ImageProjection` | `models/embeddings.py` | Original IP-Adapter pooled CLIP image embedding -> 4 image tokens. |
| Projection | `IPAdapterPlusImageProjection` | `models/embeddings.py` | Perceiver/resampler projection from CLIP hidden states -> `num_queries` image tokens. |
| Projection | `IPAdapterFullImageProjection` | `models/embeddings.py` | Feed-forward projection preserving full CLIP token sequence. |
| Projection | `IPAdapterFaceIDImageProjection`, `IPAdapterFaceIDPlusImageProjection` | `models/embeddings.py` | FaceID variants; include ID embedding and optional LoRA surface. |
| Projection wrapper | `MultiIPAdapterImageProjection` | `models/embeddings.py` | Accepts list of adapter image embeds and returns list of projected tensors. |
| Attention processor | `IPAdapterAttnProcessor` | `models/attention_processor.py` | Eager bmm path. |
| Attention processor | `IPAdapterAttnProcessor2_0` | `models/attention_processor.py` | PyTorch SDPA path; likely default on current PyTorch. |
| Attention processor | `IPAdapterXFormersAttnProcessor` | `models/attention_processor.py` | xFormers variant; candidate only for Dinoml, not first parity. |
| Mask helper | `IPAdapterMaskProcessor.downsample` | `image_processor.py` | Resizes spatial masks to latent query-token grid and broadcasts over value dim. |

Separate candidate reports:

- `sdxl_ip_adapter`: same UNet mutation pattern but SDXL uses dual text conditioning and wider UNet/cross-attention dimensions; official `h94/IP-Adapter/sdxl_models` includes separate ViT-H/G image encoder config and SDXL weights.
- `sd3_ip_adapter`: transformer/joint-attention path. Uses `SD3IPAdapterMixin`, `IPAdapterTimeImageProjection`, and `SD3IPAdapterJointAttnProcessor2_0`, with timestep-conditioned image projection and joint attention.
- `flux_ip_adapter`: transformer path. Uses `FluxIPAdapterMixin`, `FluxIPAdapterAttnProcessor`, `transformer_flux.py`, and dispatch attention; scales are per transformer block.
- `sd1_faceid_ip_adapter`: FaceID adds ID embedding projections and may auto-load FaceID LoRA weights through PEFT; keep separate from first SD1 image-prompt slice.

## 4. Config and weight dimensions

Image encoder configs:

| Target | Config | Hidden | Projection | Layers | Heads | Image/patch | Notes |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| SD1 `h94/IP-Adapter/models/image_encoder` | CLIP vision with projection | 1280 | 1024 | 32 | 16 | 224 / 14 | Used by standard SD1 weights with pooled `image_embeds` dim 1024 and hidden-state dim 1280. |
| SDXL candidate `h94/IP-Adapter/sdxl_models/image_encoder` | CLIP vision with projection | 1664 | 1280 | 48 | 16 | 224 / 14 | Candidate only; not first SD1 slice. |

Representative official SD1 weight headers:

| Weight | Projection signal | Image embed input | Projected tokens | Cross-attn dim | IP K/V layer shapes |
| --- | --- | ---: | ---: | ---: | --- |
| `models/ip-adapter_sd15.safetensors` | `image_proj.proj.weight (3072,1024)` | pooled 1024 | 4 | 768 | 16 cross-attn processors, `to_k_ip/to_v_ip` widths 320/640/1280 by UNet block. |
| `models/ip-adapter_sd15_vit-G.safetensors` | `image_proj.proj.weight (3072,1280)` | pooled 1280 | 4 | 768 | Same UNet K/V shapes as SD1; only image encoder/proj input differs. |
| `models/ip-adapter-plus_sd15.safetensors` | `image_proj.latents (1,16,768)` | CLIP hidden sequence 1280 -> hidden 768 | 16 | 768 | Same UNet K/V shapes. |
| `models/ip-adapter-full-face_sd15.safetensors` | FF + norm, output 768 | CLIP full token sequence hidden 1280 | 257 | 768 | Same UNet K/V shapes; 256 patch tokens + CLS. |
| `models/ip-adapter-plus-face_sd15.safetensors` | resampler latents `(1,16,768)` | CLIP hidden sequence | 16 | 768 | Same UNet K/V shapes. |

For a standard 512x512 SD1 run, base UNet latent query lengths are:

| Attention resolution | Query tokens | Hidden size | Heads/head dim from SD1 base report |
| --- | ---: | ---: | --- |
| 64x64 | 4096 | 320 | 8 heads, head dim 40 |
| 32x32 | 1024 | 640 | 8 heads, head dim 80 |
| 16x16 | 256 | 1280 | 8 heads, head dim 160 |
| 8x8 mid/up | 64 | 1280 | 8 heads, head dim 160 |

The inspected `ip_adapter` weight keys are numbered `1,3,5,...,31`, corresponding only to cross-attention processors. Self-attention processors are left as their original processor class.

## 5. Runtime tensor contract

Pipeline inputs:

| Input | Accepted form | Runtime consequence |
| --- | --- | --- |
| `ip_adapter_image` | image or list of images, one entry per loaded adapter | Pipeline preprocesses with `CLIPImageProcessor`, encodes with CLIP vision, and creates positive plus zero/hidden-state negative embeds for CFG. |
| `ip_adapter_image_embeds` | list of tensors; each tensor can already contain CFG-concatenated negative/positive halves | Bypasses image encoder; pipeline chunks first dim for CFG and repeats for `num_images_per_prompt`. |
| `cross_attention_kwargs["ip_adapter_masks"]` | list of `[1, num_images, H, W]` tensors or legacy tensor convertible to list | Masks per image prompt and per adapter; downsampled to query grid inside attention processor. |
| `set_ip_adapter_scale(scale)` | float, list, or block dictionary/list | Mutates processor `scale`; dicts match attention processor names such as `down...` or `up...`. |

Image encoding and projection:

- Standard IP-Adapter uses `image_encoder(image).image_embeds`, shape `[B,1024]` for the SD1 official config. Negative embeds are zeros with same shape.
- Plus/full variants request `output_hidden_states=True` and take `hidden_states[-2]`, repeated over images. For the SD1 official CLIP vision config this is `[B,257,1280]` before projection.
- `prepare_ip_adapter_image_embeds` wraps each adapter tensor with an adapter axis: image encoder outputs become `[1, B, D]` for pooled forms or `[1, B, 257, D]` for hidden-state forms, then repeat/CFG makes each list element `[2B, 1, ...]` under CFG.
- `MultiIPAdapterImageProjection.forward` expects a list length equal to number of loaded adapters. Each tensor shape is `[batch, num_images, embed_dim]` or `[batch, num_images, sequence, embed_dim]`; it flattens `[batch*num_images, ...]`, projects, then restores `[batch, num_images, num_tokens, 768]`.

UNet and attention:

- `added_cond_kwargs["image_embeds"]` is a list of projected-or-projectable image embeds, not a single concatenated tensor.
- `process_encoder_hidden_states` transforms `encoder_hidden_states` from text tensor `[B or 2B, 77, 768]` into `(text_hidden_states, image_embeds_list)`.
- `IPAdapterAttnProcessor*` receives that tuple. Normal text cross-attention still uses text K/V and any text attention mask. IP-Adapter image attention uses the same query but separate `to_k_ip/to_v_ip` projections and no text mask.
- Without masks, each adapter contributes:

```text
Q = to_q(latent_tokens)                         [B,H,Q,Dh]
K_text,V_text = to_k/to_v(prompt_tokens)        [B,H,77,Dh]
base = attention(Q,K_text,V_text, text_mask)    [B,Q,C]
K_ip,V_ip = to_k_ip/to_v_ip(image_tokens)       [B,H,T_ip,Dh]
ip = attention(Q,K_ip,V_ip, no_mask)            [B,Q,C]
hidden = base + scale * ip
out = to_out(hidden)
```

- With masks and multiple reference images, the processor loops over image index `i`, computes attention to that image's tokens independently, downsamples mask to `[B,Q,C]`, and adds `scale[i] * ip_i * mask_i`.
- Latent map layout remains NCHW. Attention processors flatten 4D hidden states by `view(B,C,H*W).transpose(1,2)` and restore to NCHW after output projection.

## 6. Attention requirements

First parity target:

- Cross-attention only. Self-attention (`attn1.processor`) remains base SD1 `AttnProcessor`/`AttnProcessor2_0`.
- SD1 text cross-attention plus added image K/V attention in every cross-attention layer.
- No KV cache. No causal mask. Dropout is zero in inference.
- Text mask, when present, applies only to text K/V path. IP path is unmasked unless spatial IP masks are supplied; IP spatial masks multiply the IP attention output after attention, not the attention logits.
- Multiple adapters are additive: one K/V projection pair per adapter per cross-attn layer, with one scale entry per adapter.
- Multiple images for one adapter with masks are additive over image index; without masks, tensor shape can carry `num_images` into the projected token dimension as `[B,num_images,T,C]`, and processor uses the list/tensor shape directly.

Primary source path:

- `IPAdapterAttnProcessor2_0` is the expected parity path on PyTorch 2 because it uses `F.scaled_dot_product_attention` for base text attention and for each IP attention branch.
- `IPAdapterAttnProcessor` eager bmm path defines equivalent fallback semantics.
- `IPAdapterXFormersAttnProcessor` is source-supported but should not be the first Dinoml semantic target.

Flash-style constraints for Dinoml:

- A normal dense flash/MHA provider can cover the base text attention branch and each IP branch separately under strict guards: dense Q/K/V, noncausal, no unsupported mask form, dropout 0, supported dtype/head dim, and fixed batch/head layout.
- A single fused attention over concatenated `[text_tokens, image_tokens]` is not semantically equivalent when text masks are present or when IP scales/masks differ; it would need output decomposition or logit/value-side weighting that preserves `base + scale * ip`.
- The high-value fusion is not a vanilla larger K/V concat. It is a specialized "shared Q, two K/V banks, scaled residual sum" attention op:

```text
base = SDPA(Q, K_text, V_text, text_mask)
ip_j = SDPA(Q, K_ip_j, V_ip_j, None)
out_preproj = base + sum_j(scale_j * ip_j)
```

- Masked multi-image IP attention is a stricter second slice because masks are spatially downsampled to query tokens and applied post-attention. It blocks simple flash fusion unless represented as a separate output-mask multiply/add stage.

## 7. Operator and fusion candidates

Highest priority:

- **IP image projection, standard pooled form**: `Linear(1024 or 1280 -> 4*768) -> reshape -> LayerNorm(768)`. This is the smallest useful admission slice because it supports `ip-adapter_sd15` and `ip-adapter_sd15_vit-G`.
- **Added image K/V projection per cross-attn layer**: `Linear(768 -> hidden_size, bias=False)` for K and V at hidden sizes 320/640/1280.
- **Shared-Q dual attention lowering**: reuse base SD1 cross-attention lowering for text branch, add separate IP K/V attention and scaled add before `to_out`.
- **Scale mutation as artifact-visible runtime parameter**: model `scale` as explicit per-adapter/per-layer scalar data, not hidden Python object mutation.

Medium priority:

- **Plus resampler projection**: `LayerNorm`, self/cross attention over learned latents and CLIP hidden states, GELU feed-forward, final `Linear + LayerNorm`. Useful for common Plus weights but larger than standard adapter.
- **Spatial IP masks**: `IPAdapterMaskProcessor.downsample` equivalent, mask broadcast to `[B,Q,C]`, and fused `ip_output * mask * scale + hidden`.
- **Multi-adapter accumulation**: compile a fixed number of loaded adapters as explicit branches, with per-adapter scale and token counts in manifest.

Lower priority:

- **Full-face 257-token projection**: feed-forward projection over full CLIP token sequence. Token length is larger but still simple after standard path exists.
- **FaceID LoRA coupling**: loader may auto-load extra LoRA weights. Keep separate because it mutates UNet weights beyond IP K/V processors.
- **xFormers processor parity**: not needed for semantic first slice.

Ops needed beyond base SD1:

- List/tuple handling or static lowering equivalent for `(encoder_hidden_states, ip_hidden_states)`.
- Batched `repeat_interleave`, `cat`, `chunk`, reshape/view for CFG image embeds.
- CLIP vision encoder if `ip_adapter_image` is supported inside Dinoml; otherwise accept precomputed image embeds first.
- Linear, LayerNorm, GELU, FeedForward, noncausal attention for projection modules.
- Post-attention scalar/list scale multiply and add.
- Bicubic interpolate for masks if mask preprocessing enters runtime; can initially be CPU/data-pipeline.

## 8. Graph rewrite and lowering opportunities

### Rewrite: SD1 cross-attention -> text attention plus IP attention branches

Source pattern:

```text
encoder_hidden_states = (text_tokens, ip_tokens_list)
base = attention(Q, K_text, V_text, text_mask)
for adapter in adapters:
  ip = attention(Q, K_ip_adapter, V_ip_adapter, None)
  base += scale_adapter * ip
out = to_out(base)
```

Replacement: lower as an explicit compound attention node or as base attention node plus one IP branch per adapter, followed by a fused scale/add before output projection.

Preconditions:

- Processor is `IPAdapterAttnProcessor2_0` or eager equivalent.
- `cross_attention_dim=768`, hidden sizes match SD1 block channels, dropout 0, noncausal.
- No spatial IP masks for first slice.
- Adapter count and token counts are compile-time visible.

Failure cases:

- Masked IP images, per-image scale lists, FaceID LoRA mutation, SDXL/SD3/Flux processor classes, xFormers-only behavior.

Parity test sketch:

- One SD1 cross-attention layer at hidden size 320/640/1280 with random text tokens and random projected IP tokens. Compare PyTorch processor output before/after `to_out` at fp32 and fp16.

### Rewrite: standard image projection canonicalization

Source pattern:

```text
image_embeds [B,1024] -> Linear(1024,3072) -> reshape [B,4,768] -> LayerNorm
```

Replacement: standard linear + reshape + layernorm lowering, with weight names converted from `proj.*` to `image_embeds.*`.

Preconditions: state dict has `image_proj.proj.weight`; output rows divisible by 4; cross-attn dim matches UNet `cross_attention_dim`.

Failure cases: Plus/full/FaceID projections have different keys and token contracts.

### Rewrite: IP scale as runtime manifest data

Source pattern: `set_ip_adapter_scale` mutates Python processor objects, with optional block-name dictionaries.

Replacement: compile per-layer scale slots into manifest/runtime state and lower scales as scalar inputs or mutable constants.

Preconditions: selected scale config expanded to all cross-attention processors before compile, or runtime API can update named scale slots.

Failure cases: dictionary matching over module names must be resolved before codegen; lists of per-image scales require mask/multi-image path.

## 9. Staging and admission recommendation

Recommended first Dinoml staging:

1. **Admission target**: `sd1_ip_adapter_precomputed_embeds_standard`.
   - Base SD1 UNet/VAE/scheduler remain from the SD1 report.
   - Accept `prompt_embeds`, `negative_prompt_embeds`, latents, timestep, and precomputed `ip_adapter_image_embeds`.
   - Support one standard SD1 IP-Adapter with projected token count 4 and cross-attention dim 768.
   - No image encoder inside compiled graph; no masks; no FaceID LoRA; no Plus resampler.
2. Lower `ImageProjection` or require caller-provided already-projected `[B or 2B, 1, 4, 768]` tokens for an even smaller smoke.
3. Replace only cross-attention processors with explicit added K/V branches. Self-attention and base text cross-attention remain as in SD1.
4. Expose scale as explicit runtime scalar per cross-attention layer or one broadcast scale.
5. Validate one UNet attention block, then one denoising step, then full SD1 loop with scheduler in host control flow.

Suggested completion label: `frontend-only` until the processor mutation and projection are represented in manifests; then `bounded-cuda` once the single-adapter/no-mask cross-attention branch lowers and validates.

Why not start with Plus/FaceID/masks:

- Plus adds a small Perceiver-style transformer and CLIP hidden-state dependency; useful but not needed to prove added K/V attention.
- Masks change the IP branch from pure attention add to attention plus spatial query mask multiply.
- FaceID may bring PEFT LoRA mutation, which crosses into separate adapter-weight mutation scope.

## 10. Parity and validation plan

- Loader conversion unit check: given official `ip-adapter_sd15.safetensors` headers/state dict, verify 16 cross-attention processors are IP-aware, self-attention processors are unchanged, token count is 4, and K/V shapes match block hidden sizes.
- Projection parity: `ImageProjection` for pooled `[B,1024]` and ViT-G `[B,1280]` inputs against Diffusers.
- Processor parity without masks: compare `IPAdapterAttnProcessor2_0` for one layer at query lengths 4096, 1024, 256, and 64 using text length 77 and IP token length 4.
- CFG contract parity: precomputed `ip_adapter_image_embeds` list where tensor first dim contains negative/positive halves; verify chunk/repeat/cat batch order matches prompt CFG order `[negative, positive]`.
- Scale parity: scalar scale 0, 0.5, 1.0; scale 0 must match base SD1 cross-attention exactly after processor replacement.
- One UNet block parity: SD1 cross-attn down/up/mid block with image tokens and no masks.
- Full denoising-step parity: scheduler in Python, compiled UNet step with supplied text/image embeds.
- Later tests: masked per-image scale, Plus resampler, full-face 257-token projection, FaceID LoRA mutation.

## 11. Scope boundary

In scope for this report:

- SD1 IP-Adapter runtime surface, loader mutations, image/embed contracts, projection layers, added K/V attention processors, scales, masks, tensor contracts, and Dinoml staging.

Out of scope:

- Re-auditing base SD1, ControlNet, T2I-Adapter, LoRA/textual inversion broadly, img2img/inpaint/upscale.
- Deep SDXL/SD3/Flux IP-Adapter audits; they are separate candidates above.
- Safety checker/NSFW, training/loss/dropout/gradient checkpointing, callbacks/interrupt, XLA/NPU/MPS/Flax/ONNX, and multi-GPU/context parallel paths.
