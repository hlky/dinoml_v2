# Diffusers PAG Operator and Integration Report

Target slug: `pag`

## 1. Source basis

```text
Diffusers commit/version:
  Local checkout X:/H/diffusers at b3a515080752a3ba7ca92161e25530c7f280f629.

Model id(s):
  PAG has no separate checkpoint format in the inspected source. It is a
  pipeline/runtime wrapper over existing checkpoint families.
  Representative inherited configs checked:
    stable-diffusion-v1-5/stable-diffusion-v1-5
    stabilityai/stable-diffusion-xl-base-1.0
    stabilityai/stable-diffusion-3-medium-diffusers
    Kwai-Kolors/Kolors-diffusers
    Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers
    PixArt-alpha/PixArt-Sigma-XL-2-1024-MS
    Efficient-Large-Model/Sana_1600M_1024px_diffusers
    ByteDance/AnimateDiff-Lightning, source-only/model-index-only here

Config sources:
  H:/configs/stable-diffusion-v1-5/stable-diffusion-v1-5/
  H:/configs/stabilityai/stable-diffusion-xl-base-1.0/
  H:/configs/stabilityai/stable-diffusion-3-medium-diffusers/
  H:/configs/Kwai-Kolors/Kolors-diffusers/model_index.json
  H:/configs/Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers/
  H:/configs/PixArt-alpha/PixArt-Sigma-XL-2-1024-MS/model_index.json
  H:/configs/Efficient-Large-Model/Sana_1600M_1024px_diffusers/
  H:/configs/ByteDance/AnimateDiff-Lightning/model_index.json
  Existing family reports supplied the missing component-config details where
  local PAG-specific configs do not exist.

Pipeline files inspected:
  X:/H/diffusers/src/diffusers/pipelines/pag/__init__.py
  X:/H/diffusers/src/diffusers/pipelines/pag/pag_utils.py
  X:/H/diffusers/src/diffusers/pipelines/pag/pipeline_pag_sd.py
  X:/H/diffusers/src/diffusers/pipelines/pag/pipeline_pag_sd_img2img.py
  X:/H/diffusers/src/diffusers/pipelines/pag/pipeline_pag_sd_inpaint.py
  X:/H/diffusers/src/diffusers/pipelines/pag/pipeline_pag_controlnet_sd.py
  X:/H/diffusers/src/diffusers/pipelines/pag/pipeline_pag_controlnet_sd_inpaint.py
  X:/H/diffusers/src/diffusers/pipelines/pag/pipeline_pag_sd_xl.py
  X:/H/diffusers/src/diffusers/pipelines/pag/pipeline_pag_sd_xl_img2img.py
  X:/H/diffusers/src/diffusers/pipelines/pag/pipeline_pag_sd_xl_inpaint.py
  X:/H/diffusers/src/diffusers/pipelines/pag/pipeline_pag_controlnet_sd_xl.py
  X:/H/diffusers/src/diffusers/pipelines/pag/pipeline_pag_controlnet_sd_xl_img2img.py
  X:/H/diffusers/src/diffusers/pipelines/pag/pipeline_pag_sd_3.py
  X:/H/diffusers/src/diffusers/pipelines/pag/pipeline_pag_sd_3_img2img.py
  X:/H/diffusers/src/diffusers/pipelines/pag/pipeline_pag_hunyuandit.py
  X:/H/diffusers/src/diffusers/pipelines/pag/pipeline_pag_kolors.py
  X:/H/diffusers/src/diffusers/pipelines/pag/pipeline_pag_pixart_sigma.py
  X:/H/diffusers/src/diffusers/pipelines/pag/pipeline_pag_sana.py
  X:/H/diffusers/src/diffusers/pipelines/pag/pipeline_pag_sd_animatediff.py
  X:/H/diffusers/src/diffusers/pipelines/auto_pipeline.py

Model files inspected:
  X:/H/diffusers/src/diffusers/models/attention_processor.py
  X:/H/diffusers/src/diffusers/models/attention.py
  Source context from prior reports:
    UNet2DConditionModel and blocks for SD/SDXL/Kolors/AnimateDiff.
    SD3Transformer2DModel for SD3.
    HunyuanDiT2DModel, PixArtTransformer2DModel, SanaTransformer2DModel.
    AutoencoderKL and VAE shared modules.

Scheduler/processors/helpers inspected:
  X:/H/diffusers/src/diffusers/guiders/perturbed_attention_guidance.py
  X:/H/diffusers/src/diffusers/hooks/layer_skip.py
  X:/H/diffusers/src/diffusers/image_processor.py
  X:/H/diffusers/src/diffusers/video_processor.py
  X:/H/diffusers/tests/pipelines/pag/
  agents/plans/diffusers/stable_diffusion_1_5/report.md
  agents/plans/diffusers/stable_diffusion_xl/report.md
  agents/plans/diffusers/stable_diffusion_3/report.md
  agents/plans/diffusers/kolors/report.md
  agents/plans/diffusers/pixart/report.md
  agents/plans/diffusers/sana/report.md
  agents/plans/diffusers/scheduler_matrix/report.md

External component configs inspected:
  CLIP, CLIP bigG, T5, ChatGLM, PixArt/Sana T5 configs as inherited from their
  base-family config caches and reports.

Any missing files or assumptions:
  No `model_index.json` under H:/configs declared a PAG pipeline class; `PAG`
  is selected by Diffusers source through `enable_pag=True`, `from_pipe`, or by
  directly instantiating PAG pipeline classes. No new authenticated config retry
  was needed for this PAG audit because the audited PAG behavior is source
  defined and inherits existing checkpoint configs. Kolors component configs
  beyond `model_index.json` are recorded in the Kolors report as authenticated
  Hugging Face cache reads. Safety, callbacks, training, XLA/NPU/MPS/Flax/ONNX,
  and multi-GPU/context-parallel paths are out of scope.
```

## 2. Pipeline and component graph

PAG is a guidance overlay. It does not add trainable weights or a new scheduler.
It changes the denoising call by:

1. Expanding conditioning tensors from the normal 1x or CFG 2x batch to a PAG
   2x or CFG+PAG 3x batch.
2. Replacing selected self-attention processors in the UNet or transformer.
3. Computing a normal text/conditional prediction and a perturbed prediction in
   one denoiser call.
4. Applying extra guidance arithmetic before the inherited scheduler step.

Dataflow:

```text
prompt/image/video preprocessing
  -> inherited text/image encoders or cached embeddings
  -> inherited latent initialization or VAE encode
  -> PAG conditioning batch expansion
  -> denoising loop:
       selected self-attention processors mutated to PAG processors
       denoiser returns uncond/text/perturbed predictions
       CFG and PAG guidance arithmetic
       inherited scheduler.step
  -> inherited VAE decode/postprocess
```

Class/file anchors:

| Variant | Class anchor | Inherited model contract |
| --- | --- | --- |
| SD text-to-image | `pipeline_pag_sd.py:157`, `__call__`: `:747` | SD 1.x `UNet2DConditionModel`, CLIP text, AutoencoderKL, Karras-compatible scheduler. |
| SD img2img | `pipeline_pag_sd_img2img.py:152`, `__call__`: `:784` | SD 1.x plus VAE encode and strength/timestep slicing. |
| SD inpaint | `pipeline_pag_sd_inpaint.py:184`, `__call__`: `:912` | SD inpaint masks/masked latents, 4-channel or 9-channel UNet inherited from checkpoint. |
| SD ControlNet | `pipeline_pag_controlnet_sd.py:168`, `__call__`: `:865` | SD 1.x plus ControlNet down/mid residuals and condition image. |
| SD ControlNet inpaint | `pipeline_pag_controlnet_sd_inpaint.py:134`, `__call__`: `:974` | SD ControlNet plus mask/masked-image latents. |
| SDXL text-to-image | `pipeline_pag_sd_xl.py:172`, `__call__`: `:836` | SDXL dual CLIP, pooled/text-time conditioning, UNet, AutoencoderKL. |
| SDXL img2img | `pipeline_pag_sd_xl_img2img.py:191`, `__call__`: `:989` | SDXL plus VAE encode, strength, denoising range. |
| SDXL inpaint | `pipeline_pag_sd_xl_inpaint.py:204`, `__call__`: `:1080` | SDXL inpaint masks/masked latents and optional 9-channel UNet. |
| SDXL ControlNet | `pipeline_pag_controlnet_sd_xl.py:185`, `__call__`: `:1003` | SDXL plus ControlNet residuals. |
| SDXL ControlNet img2img | `pipeline_pag_controlnet_sd_xl_img2img.py:164`, `__call__`: `:1080` | SDXL ControlNet plus VAE encode and strength. |
| SD3 text-to-image | `pipeline_pag_sd_3.py:136`, `__call__`: `:686` | SD3 `SD3Transformer2DModel`, triple encoders, FlowMatch Euler, 16-channel VAE. |
| SD3 img2img | `pipeline_pag_sd_3_img2img.py:152`, `__call__`: `:737` | SD3 plus VAE encode and strength. |
| HunyuanDiT | `pipeline_pag_hunyuandit.py:152`, `__call__`: `:581` | HunyuanDiT transformer with dual text encoders, style/time/image RoPE metadata. |
| Kolors | `pipeline_pag_kolors.py:128`, `__call__`: `:667` | Kolors ChatGLM text stack plus SDXL-like UNet text-time conditioning. |
| PixArt Sigma | `pipeline_pag_pixart_sigma.py:144`, `__call__`: `:576` | PixArt transformer with T5 embeddings and patch/unpatch latent contract. |
| Sana | `pipeline_pag_sana.py:148`, `__call__`: `:650` | Sana transformer with linear attention processors and Sana VAE/DC-AE contract. |
| AnimateDiff | `pipeline_pag_sd_animatediff.py:89`, `__call__`: `:577` | SD/AnimateDiff UNet motion modules, video latents, CLIP text, AutoencoderKL decode per frames. |

AutoPipeline enables PAG by class substitution in `auto_pipeline.py`; it maps
base text-to-image/image-to-image/inpainting classes to matching PAG classes
when `enable_pag=True`.

Separate candidate reports:

| Candidate | Anchors | Why separate |
| --- | --- | --- |
| `pag_modular_guider` | `guiders/perturbed_attention_guidance.py:36`, `hooks/layer_skip.py:41` | New modular guider API perturbs attention through hooks and layer-skip config rather than PAG pipeline processors. It is source-related but not the same lowering surface as `PAGMixin`. |
| `sd_pag_controlnet` and `sdxl_pag_controlnet` | PAG ControlNet pipeline files plus `models/controlnets/controlnet.py` | Adds side ControlNet forward pass, residual aggregation, and condition-image preprocessing. |
| `sd_pag_inpaint` and `sdxl_pag_inpaint` | PAG inpaint files | Adds mask tensors, masked-image VAE encode, and 9-channel UNet variants. |
| `pag_ip_adapter` | PAG SD/SDXL/Kolors pipelines inherit `IPAdapterMixin`; processors in `attention_processor.py` | Adds image encoder/projection and added K/V attention branches before PAG batch expansion. |
| `animate_diff_pag` | `pipeline_pag_sd_animatediff.py` | Video latents and motion modules make this more than a 2D PAG branch. |
| `sana_pag_linear_attention` | `PAGCFGSanaLinearAttnProcessor2_0`, `PAGIdentitySanaLinearAttnProcessor2_0` | Uses Sana linear attention math, not SDPA identity-only self-attention. |

## 3. Important config dimensions

PAG-specific runtime dimensions:

| Field | Value/source | Operator impact |
| --- | --- | --- |
| `pag_applied_layers` | Constructor default is usually `"mid"` for UNet families; `"blocks.1"` for SD3/Hunyuan/PixArt; `"transformer_blocks.0"` for Sana; `"mid_block.*attn1"` for AnimateDiff. | Regex/selects self-attention modules to mutate. |
| `pag_scale` | `__call__` default `3.0` for all inspected PAG pipelines. | Extra guidance coefficient. `0.0` disables PAG. |
| `pag_adaptive_scale` | `__call__` default `0.0`. | If positive, per-step scale is `max(pag_scale - pag_adaptive_scale * (1000 - t), 0)`. |
| CFG active test | Inherited per family: SD/SDXL/Kolors use guidance scale > 1; PixArt/Sana local bool; SD3 has true CFG; ControlNet examples can set guidance 0 with PAG only. | Determines 2-way vs 3-way batch and processor choice. |
| PAG batch multiplier | 2 without CFG, 3 with CFG. | Directly increases denoiser input, prompt embedding, masks, pooled embeds, ControlNet/IP-Adapter embeds, and scheduler model-input batch. |
| Attention processor family | `PAGCFGIdentitySelfAttnProcessor2_0`/`PAGIdentitySelfAttnProcessor2_0` by default; SD3 uses joint processors; Hunyuan uses Hunyuan processors; Sana uses linear attention processors. | Defines perturbed attention math and flash/provider feasibility. |

Representative inherited checkpoint dimensions:

| Family | Latent contract | Denoiser/main model | Text/condition dims | Scheduler default checked |
| --- | --- | --- | --- | --- |
| SD 1.5 | `[B,4,H/8,W/8]` NCHW | UNet channels 320/640/1280, cross dim 768 | CLIP `[B,77,768]` | PNDM epsilon in primary config; broad Karras-compatible family. |
| SDXL base | `[B,4,H/8,W/8]` NCHW | UNet 320/640/1280, cross dim 2048 | dual CLIP concat `[B,77,2048]`, pooled `[B,1280]`, time IDs | EulerDiscrete epsilon; broad Karras-compatible family. |
| SD3 medium | `[B,16,H/8,W/8]` NCHW | 24-layer joint transformer, patch size 2, 24 heads x 64 | CLIP+T5 sequence to joint dim 4096, pooled 2048 | FlowMatchEulerDiscrete, shift 3.0. |
| Kolors | `[B,4,H/8,W/8]` NCHW | SDXL-like UNet, cross dim 2048, text projection 4096->2048 | ChatGLM prompt `[B,S,4096]`, pooled `[B,4096]`, time-ID input 5632 | EulerDiscrete epsilon, 1100 train steps. |
| HunyuanDiT | `[B,C,H/8,W/8]` then transformer patch grid | HunyuanDiT transformer with image RoPE | CLIP-like 77 tokens and T5-like 256 tokens plus style/time IDs | Scheduler config in cache; source uses scheduler `set_timesteps` and `scale_model_input`. |
| PixArt Sigma | transformer latent map, patch model, optional learned sigma output | PixArt transformer | T5 prompt embeds and attention mask | Uses compatible discrete schedulers via `retrieve_timesteps`; learned sigma output is split off when configured. |
| Sana | transformer latent map from Sana VAE/DC-AE path | Sana transformer with linear attention | T5 prompt embeds and attention mask | Source uses scheduler `retrieve_timesteps` and `step`; first slice should inherit Sana report scheduler choice. |
| AnimateDiff | video latents `[B,C,F,H/8,W/8]` source path | UNet + motion modules | CLIP prompt repeated per frame | Inherits AnimateDiff/SD scheduler surface; video loop repeats prompt embeddings by frame count. |

## 3a. Family variation traps

- PAG is not a model architecture. It inherits every base-family trap:
  SD/SDXL/Kolors use UNet NCHW latent maps, SD3/PixArt/Sana/Hunyuan use
  transformer patch/token models, and AnimateDiff adds frame dimensions.
- `pag_applied_layers` is regex-based. Bad regexes can match extra layers, and
  partial numeric names are guarded only by the mixin's simple fake-integral
  check.
- PAG only mutates self-attention modules where `not module.is_cross_attention`.
  Cross-attention and added K/V branches are not directly perturbed by the
  default mixin selection.
- CFG+PAG changes batch order from normal `[uncond, text]` to
  `[uncond, text, perturbed-text]`. PAG-only uses `[text, perturbed-text]`.
- Prompt masks, pooled embeddings, SDXL/Kolors `add_time_ids`, ControlNet
  conditioning images, IP-Adapter image embeds, and inpaint masks must be
  expanded with the same PAG batch policy. Missing one branch silently breaks
  batch alignment.
- SD3 joint attention PAG uses a full mask for the perturbed path that makes
  image-token self-attention identity while still allowing non-image/context
  interactions. This is different from the SD/SDXL identity-self-attention
  processor that uses V projection directly for the perturbed path.
- Hunyuan PAG keeps QK norm and image RoPE on the original path, but its
  perturbed path uses only the V projection and output projection.
- Sana PAG uses linear-attention math with ReLU Q/K, fp32 matmul, padded value,
  and division by a normalization row; it is not an SDPA flash target.
- The modular `PerturbedAttentionGuidance` guider is source-related but uses
  hooks to skip attention scores and warns that model-agnostic joint latent
  conditioning is not handled. Do not conflate it with the PAG pipeline classes.

## 4. Runtime tensor contract

Common PAG contract:

| Boundary | No CFG + PAG | CFG + PAG |
| --- | --- | --- |
| Conditioning tensors | `cat([cond, cond])` | `cat([uncond, cond, cond])` |
| Latent model input | `cat([latents] * 2)` | `cat([latents] * 3)` |
| Denoiser prediction chunks | `text, perturb` | `uncond, text, perturb` |
| Guidance output | `text + pag * (text - perturb)` | `uncond + cfg * (text - uncond) + pag * (text - perturb)` |

`_prepare_perturbed_attention_guidance(cond, uncond, cfg)` in
`pag_utils.py:132` first duplicates `cond` twice, then prepends `uncond` if CFG
is active. `_apply_perturbed_attention_guidance(...)` in `pag_utils.py:100`
chunks the denoiser prediction and applies the formulas above.

Inherited tensor examples:

| Variant group | PAG-expanded tensors | Source layout |
| --- | --- | --- |
| SD/SDXL/Kolors UNet | latent maps, prompt embeds, optional negative embeds, optional IP image embeds, SDXL/Kolors pooled and time IDs | NCHW latent maps; prompt `[B,S,C]`. |
| SD/SDXL inpaint | mask `[B,1,H/8,W/8]`, masked-image latents `[B,4,H/8,W/8]`, prompt/time tensors | NCHW, channel concat must preserve source order. |
| ControlNet PAG | control images/embeds and ControlNet prompt batches, then UNet residuals | NCHW condition image and latent maps. |
| SD3/PixArt/Sana/Hunyuan transformers | latent maps or patch-token internal inputs, prompt embeds, prompt attention masks, pooled/style/time metadata where present | Pipeline entry is mostly NCHW latent map; transformers patchify internally. |
| AnimateDiff | latent video tensor plus prompt embeds repeated by frame count | Source video latent rank includes frame dimension; prompt batch is repeated per frame. |

CPU/data-pipeline work remains inherited: tokenization, text encoders,
processor resize/normalize, image/video postprocess, and optional image encoder
for IP-Adapter. GPU/runtime work added by PAG is batch expansion, selected
attention processor swap, extra denoiser compute, guidance arithmetic, and
restoring processors after the loop.

Precomputable/reusable:

- Base positive/negative prompt embeddings before PAG expansion.
- SDXL/Kolors pooled embeddings and add-time IDs before PAG expansion.
- Control/image/IP embeddings before PAG expansion.
- Scheduler timesteps/sigmas from the inherited scheduler.

## 5. Operator coverage checklist

PAG-specific ops:

- Regex/name-based selection of self-attention modules, represented as
  artifact-visible processor/mutation state rather than hidden global state.
- Tensor `cat` along batch for every denoiser conditioning input.
- Tensor `chunk(2)` or `chunk(3)` along batch for denoiser predictions.
- PAG adaptive scale pointwise scalar math:
  `max(pag_scale - pag_adaptive_scale * (1000 - timestep), 0)`.
- Guidance arithmetic:
  `cfg_base + pag * (text - perturb)`.
- Denoiser input batch expansion using the ratio between conditioning batch and
  latent batch.
- Processor restore after run, or equivalent immutable execution-plan
  separation in Dinoml.

Inherited operators still required by first parity:

| Group | Required families |
| --- | --- |
| Tensor/layout ops | reshape/view/permute/concat/chunk/repeat/repeat_interleave, NCHW/NCDHW latent maps, transformer patchify/unpatchify, attention mask expansion. |
| Conv/down/up ops | SD/SDXL/Kolors UNet and AutoencoderKL Conv2d/GroupNorm/SiLU, AnimateDiff spatial UNet, VAE decode/encode. |
| GEMM/linear ops | UNet Q/K/V/out and FFN projections, SD3/Hunyuan/PixArt/Sana transformer projections, text-time/timestep MLPs, ChatGLM/T5/CLIP when text encoders are compiled. |
| Attention primitives | identity-self PAG, joint SDPA PAG, Hunyuan RoPE/QK-norm PAG, Sana linear-attention PAG, inherited cross-attention and IP-Adapter added K/V when active. |
| Scheduler/guidance | inherited scheduler `scale_model_input` and `step`; CFG, optional guidance rescale; PAG adds one extra guidance vector. |
| VAE/postprocess | inherited scale/shift, decode, encode for img2img/inpaint, video frame decode for AnimateDiff. |

## 6. Denoiser/model breakdown

### UNet PAG families: SD, SDXL, Kolors, ControlNet, inpaint, AnimateDiff

The selected self-attention modules receive either
`PAGCFGIdentitySelfAttnProcessor2_0` or `PAGIdentitySelfAttnProcessor2_0`.
The original chunk follows normal self-attention:

```text
optional spatial_norm/group_norm
Q = to_q(hidden)
K = to_k(hidden)
V = to_v(hidden)
SDPA(Q,K,V)
to_out
optional residual/rescale
```

The perturbed chunk does not compute Q/K attention scores. It applies
`to_v(hidden_ptb)` and then the same output projection. This is equivalent to
identity attention over the self-attention sequence after V projection for the
selected layer. The processor preserves the normal residual/rescale behavior.

For CFG+PAG, the processor chunks `[uncond, text, perturb]`, concatenates
`[uncond, text]` for the original path, processes `perturb` through identity
attention, and returns `[original_uncond, original_text, perturbed_text]`.

### SD3 joint transformer PAG

`StableDiffusion3PAGPipeline` installs
`PAGCFGJointAttnProcessor2_0`/`PAGJointAttnProcessor2_0`. The original path
projects image and context tokens separately, concatenates them on sequence,
runs SDPA, splits image/context outputs, and applies output projections. The
perturbed path creates a full attention mask where the image-token block is
`-inf` except the diagonal is zero, so image-token self-attention becomes
identity while joint/context portions remain represented by the concatenated
sequence. This preserves SD3's joint image/text attention structure more
faithfully than the UNet identity-V shortcut.

### HunyuanDiT PAG

Hunyuan installs `PAGCFGHunyuanAttnProcessor2_0` or
`PAGHunyuanAttnProcessor2_0`. The original path includes Q/K normalization and
`apply_rotary_emb` for image rotary embeddings. The perturbed path uses the V
projection and output projection directly. The pipeline also expands both text
encoder branches, text masks, style tensor, and size/time metadata for PAG.

### PixArt Sigma PAG

PixArt uses the default identity self-attention processors from `PAGMixin` for
selected transformer self-attention layers. It expands T5 prompt embeddings and
attention masks. If `transformer.config.out_channels // 2 == latent_channels`,
the learned sigma half of the model output is split after PAG/CFG guidance.

### Sana PAG

Sana passes specialized processors:
`PAGCFGSanaLinearAttnProcessor2_0` and
`PAGIdentitySanaLinearAttnProcessor2_0`. The original path computes Sana linear
attention using ReLU Q/K, fp32 matmul, value padding, normalization by the last
row, output projection, and optional fp16 clipping. The perturbed path uses
only `to_v` and output projection.

### AnimateDiff PAG

AnimateDiff uses SD-style identity processors over selected self-attention
modules, but the pipeline repeats prompt embeddings by `num_frames` and expands
latents with a video-aware batch ratio. First Dinoml support should treat this
as a separate video candidate because rank/layout and motion modules differ
from 2D SD.

## 7. Attention requirements

Processor anchors:

| Processor | Anchor | Required by |
| --- | --- | --- |
| `PAGIdentitySelfAttnProcessor2_0` | `attention_processor.py:5043` | UNet/PixArt PAG without CFG. |
| `PAGCFGIdentitySelfAttnProcessor2_0` | `attention_processor.py:5142` | UNet/PixArt PAG with CFG. |
| `PAGJointAttnProcessor2_0` | `attention_processor.py:1508` | SD3 PAG without CFG. |
| `PAGCFGJointAttnProcessor2_0` | `attention_processor.py:1664` | SD3 PAG with CFG. |
| `PAGHunyuanAttnProcessor2_0` | `attention_processor.py:3325` | HunyuanDiT PAG without CFG. |
| `PAGCFGHunyuanAttnProcessor2_0` | `attention_processor.py:3448` | HunyuanDiT PAG with CFG. |
| `PAGCFGSanaLinearAttnProcessor2_0` | `attention_processor.py:5393` | Sana PAG with CFG. |
| `PAGIdentitySanaLinearAttnProcessor2_0` | `attention_processor.py:5448` | Sana PAG without CFG. |

Backend and flash feasibility:

- Base parity is the explicit processor code in `attention_processor.py`.
- UNet identity PAG does not need a flash kernel for the perturbed branch; it
  needs V projection plus output projection. The original branch can use the
  same SDPA/flash candidate as base self-attention if masks/dropout/dtype/head
  dimensions are supported.
- SD3 joint PAG's perturbed branch uses an additive mask with a diagonal image
  identity block. A standard flash provider must support the exact block mask,
  or Dinoml should lower it as a specialized masked attention or an explicit
  identity-image plus context-attention decomposition. Treat generic flash as
  guarded, not automatic.
- Hunyuan PAG original path can use a flash-style provider only if QK norm and
  RoPE are applied before the provider and mask/dtype/head constraints pass.
  Its perturbed path is V projection only.
- Sana PAG is not a normal softmax attention path. It needs linear-attention
  provider work or GEMM/fused elementwise decomposition; flash SDPA is not
  semantically applicable.
- IP-Adapter added K/V processors, ControlNet residual paths, LoRA live
  adapter mutation, and xFormers/custom processors must be separately admitted
  before combining with PAG lowering.

## 8. Scheduler and denoising-loop contract

PAG does not change scheduler class or scheduler tables. It changes the model
output passed to `scheduler.step`.

Inherited first scheduler slices:

| PAG family | Source/default scheduler evidence | Recommended first Dinoml slice |
| --- | --- | --- |
| SD PAG | SD 1.5 primary config PNDM epsilon; Karras-compatible pipeline surface | PNDM epsilon for default parity, plus DDIM/Euler as scheduler-matrix follow-ups. |
| SDXL PAG | SDXL base EulerDiscrete epsilon; Karras-compatible surface | EulerDiscrete epsilon. |
| SD3 PAG | FlowMatchEulerDiscrete, shift 3.0 for SD3 medium | FlowMatch Euler static shift. |
| Kolors PAG | EulerDiscrete epsilon with 1100 training steps | Exact Kolors EulerDiscrete table. |
| HunyuanDiT PAG | Hunyuan config cache and source use `scheduler.set_timesteps`, `scale_model_input`, `step` | Inherit HunyuanDiT report; do not substitute SD scheduler assumptions. |
| PixArt PAG | PixArt source uses `retrieve_timesteps`, scheduler `scale_model_input`, and `step` | Inherit PixArt DPM/LCM-compatible first scheduler choice. |
| Sana PAG | Sana source uses `retrieve_timesteps`, scheduler `step` | Inherit Sana report, including SCM/Flow/DPM variations by checkpoint. |
| AnimateDiff PAG | Inherits SD/AnimateDiff scheduler set | Treat video loop and scheduler separately from 2D SD. |

Loop-side graph:

```text
if PAG:
  expand prompt/condition tensors to 2x or 3x
  install PAG processors on selected self-attention layers
for t in timesteps:
  latent_model_input = repeat(latents, condition_batch / latent_batch)
  latent_model_input = scheduler.scale_model_input(latent_model_input, t)
  pred = denoiser(latent_model_input, t, expanded_conditions)
  pred = cfg_pag_formula(pred, guidance_scale, pag_scale_for_t)
  latents = scheduler.step(pred, t, latents)
restore original attention processors
```

Host control first: processor mutation/restoration, loop iteration, scheduler
state, and selected layer resolution. Compiled candidates: batch expansion,
PAG/CFG arithmetic, and per-family scheduler pointwise step once explicit.

## 9. Position, timestep, and custom math

PAG custom math:

```text
pag_scale_for_t = pag_scale
if pag_adaptive_scale > 0:
  pag_scale_for_t = max(pag_scale - pag_adaptive_scale * (1000 - t), 0)

if CFG:
  pred = pred_uncond
       + guidance_scale * (pred_text - pred_uncond)
       + pag_scale_for_t * (pred_text - pred_perturb)
else:
  pred = pred_text + pag_scale_for_t * (pred_text - pred_perturb)
```

Timestep caveat: `_get_pag_scale(t)` assumes a 1000-timestep-style scalar. For
FlowMatch or schedulers using float sigma/timestep values, Dinoml should match
Diffusers exactly rather than reinterpret adaptive scaling as a normalized step
fraction.

Inherited position/time math remains required:

- SD/SDXL/Kolors sinusoidal timestep embeddings and SDXL/Kolors size/crop time
  IDs.
- SD3 patch positional embeddings and pooled text/timestep embeddings.
- Hunyuan image RoPE and size/style metadata.
- PixArt/Sana timestep and prompt conditioning.
- AnimateDiff frame/video timing from the base AnimateDiff report.

## 10. Preprocessing and input packing

PAG adds no tokenizer or image processor. It duplicates already-prepared
runtime tensors.

Required packing rules:

- Apply PAG expansion after prompt/image/control embeddings are prepared and
  before the denoising loop.
- For SDXL/Kolors, expand `prompt_embeds`, pooled text embeds, and add-time IDs
  together.
- For SD3, expand sequence prompt embeddings and pooled projections together.
- For PixArt/Sana, expand prompt attention masks together with prompt embeds.
- For ControlNet, expand condition images or image embeds consistently with the
  UNet/control batch. Multi-ControlNet residual summation is inherited.
- For inpaint, expand `mask` and `masked_image_latents` exactly like prompt
  tensors when PAG is active.
- For AnimateDiff, run PAG expansion before frame repeat where source does so,
  then repeat prompt embeddings by `num_frames`.

Output packing is unchanged: scheduler returns the normal latent batch, VAE
decode/postprocess uses inherited output type, and PAG-only intermediate
predictions are not public outputs.

## 11. Graph rewrite / lowering opportunities

### Rewrite: PAG batch expansion as an explicit guidance node

Source pattern:

```text
cond2 = cat([cond, cond])
cond3 = cat([uncond, cond, cond])
latent_model_input = cat([latents] * batch_ratio)
```

Replacement: an explicit `pag_prepare_conditions` runtime node that emits
expanded condition views/copies plus a `prediction_layout` enum
`text_perturb` or `uncond_text_perturb`.

Preconditions: every denoiser-side condition tensor has a known batch axis and
the same semantic order. Failure cases: nested lists of IP image embeds,
multi-control lists, or frame-expanded video embeddings without explicit batch
metadata. Parity test: compare all expanded tensors for SDXL, SD3, inpaint,
ControlNet, and AnimateDiff tiny cases.

### Rewrite: identity PAG self-attention

Source pattern: selected UNet/PixArt self-attention perturbed branch computes
`to_v(hidden_ptb) -> to_out`.

Replacement: skip Q/K/SDPA for the perturbed branch; lower only V GEMM and out
GEMM, while preserving group/spatial norm, residual, and rescale factor.

Preconditions: selected processor is one of the identity self-attention PAG
processors and no custom processor changes semantics. Failure cases: SD3 joint
processor, Sana linear processor, Hunyuan RoPE processor, xFormers-only
processor, added K/V attention branch.

### Rewrite: SD3 joint PAG masked attention

Source pattern: full joint attention for original path; perturbed path uses a
mask whose image-token self block is identity.

Replacement options:

1. Specialized masked attention provider with block-diagonal identity mask.
2. Decompose the image self block to identity while preserving image-to-text,
   text-to-image, and text/context attention if algebraically verified.

Preconditions: fixed image token count and context token count; no IP-Adapter
processor; supported dtype/head dimension. Failure cases: arbitrary attention
masks, context-only blocks, or changing joint attention topology.

### Rewrite: PAG guidance arithmetic fusion

Source pattern: chunk denoiser output and compute CFG+PAG vector.

Replacement: one fused elementwise kernel over latent output:

```text
out = uncond + cfg * (text - uncond) + pag * (text - perturb)
```

Preconditions: prediction chunks have equal shape and dtype; scheduler
prediction type is not changing chunk layout. Failure cases: PixArt learned
sigma split order must remain after guidance as source does, and guidance
rescale needs the `noise_pred_text` tensor preserved.

### Rewrite: immutable PAG execution plan

Source pattern: mutate `module.processor`, run the loop, then restore original
processors.

Replacement: compile separate attention variants and choose them through an
execution-plan layer selection table, avoiding hidden mutable model state.

Preconditions: `pag_applied_layers` is resolved at compile/load time or a
bounded runtime layer set is admitted. Failure cases: arbitrary user regex at
run time over module names, live LoRA/IP/custom processor mutation.

## 12. Kernel fusion candidates

Highest priority:

- PAG/CFG fused arithmetic and optional guidance-rescale reduction reuse.
- Identity-PAG perturbed branch as V-projection plus output projection, avoiding
  unnecessary Q/K/attention work.
- Existing base-family attention providers for original branches: UNet
  self-attention, SD3 joint attention, Hunyuan RoPE/QK-norm attention.
- Processor/layer selection as artifact-visible metadata so PAG does not rely
  on mutable Python module processors.

Medium priority:

- Batch expansion views/copies for prompt embeddings, masks, pooled embeddings,
  time IDs, IP image embeds, masks, and video prompt repeats.
- SD3 joint-PAG block-mask provider or verified decomposition.
- Sana linear-attention PAG provider: ReLU Q/K, fp32 value-key accumulation,
  normalization division, output projection.
- Scheduler step plus PAG/CFG arithmetic fusion per inherited scheduler family.

Lower priority:

- Combining PAG with IP-Adapter added K/V branches in one attention provider.
- ControlNet+PAG residual scheduling and multi-control aggregation fusion.
- AnimateDiff video PAG fusion before 2D image PAG parity is stable.
- Modular guider hook support, because it is a separate API surface from
  current PAG pipelines.

## 13. Runtime staging plan

Stage 1: Admit PAG as a guidance overlay, not a checkpoint family. Parse the
base pipeline class and selected `pag_applied_layers`, and require inherited
family configs to be already supported.

Stage 2: Implement PAG batch expansion and output guidance arithmetic for one
SDXL tiny or SD tiny UNet with cached prompt embeddings. Keep attention
processor selection explicit and static.

Stage 3: Implement identity self-attention PAG lowering for UNet
self-attention. Validate `pag_scale=0` equals base pipeline and `pag_scale>0`
diverges from base like Diffusers tests assert.

Stage 4: Add SDXL text-time and guidance-rescale preservation. First full
one-step parity: SDXL PAG with EulerDiscrete and supplied embeddings.

Stage 5: Add SD inpaint and ControlNet batch-expansion surfaces, still using
inherited denoiser/control support.

Stage 6: Add SD3 joint transformer PAG, including CFG+PAG batch layout and
joint attention perturbed mask behavior.

Stage 7: Add Kolors/PixArt/Hunyuan/Sana after their base reports are admitted,
because PAG is small compared with their text/model-specific contracts.

Stage 8: Add AnimateDiff PAG as a video variant after video latent and motion
module support exists.

Stage 9: Evaluate modular `PerturbedAttentionGuidance` as a separate guider API
with hook/layer-skip metadata.

## 14. Parity and validation plan

- Unit test `_prepare_perturbed_attention_guidance` for 2x and 3x tensors over
  prompt embeds, masks, pooled embeds, and latent masks.
- Unit test `_apply_perturbed_attention_guidance` formulas for CFG and non-CFG,
  including adaptive scaling at representative integer and float timesteps.
- Processor selection tests: regex matching, fake integral guard
  (`blocks.1` must not match `blocks.10`), no-match error, and restore state.
- Identity processor parity against Diffusers for one selected UNet
  self-attention layer with and without CFG.
- SDXL one-step parity with fixed latents, prompt embeds, pooled embeds, time
  IDs, Euler timestep, `pag_scale=3.0`, and guidance rescale 0/nonzero.
- SD inpaint parity for mask and masked-image latent expansion.
- ControlNet PAG parity for condition image batch expansion and residual
  injection.
- SD3 joint processor parity for one transformer block, checking original path,
  perturbed mask path, and final guidance arithmetic.
- Sana processor parity for linear attention PAG math in fp32/fp16.
- `pag_scale=0.0` regression: output should match the corresponding base
  pipeline within base tolerance, and no PAG processors should be installed.
- End-to-end tiny PAG smoke for each test file already present under
  `X:/H/diffusers/tests/pipelines/pag/`.
- Suggested tolerances: guidance arithmetic fp32 `rtol=1e-5, atol=1e-6`;
  fp16/bf16 denoiser parity initially `rtol=2e-2, atol=2e-2`, tightened after
  attention/provider-specific validation.

## 15. Performance probes

- Denoiser step latency base vs PAG-only 2x vs CFG+PAG 3x for SDXL and SD3.
- Selected layer count sweep: `"mid"`, one block, multiple blocks, broad regex.
- Original attention provider vs identity-PAG optimized perturbed branch.
- SD3 joint-PAG masked attention: SDPA mask path vs specialized provider.
- Memory and temporary use from expanded prompt/latent/control/IP batches.
- PAG arithmetic and guidance-rescale overhead as separate kernels.
- Scheduler step overhead after guidance fusion.
- ControlNet+PAG and inpaint+PAG incremental cost over their base variants.
- AnimateDiff frame-count sweep for prompt repeat and video latent expansion.
- Sana linear attention PAG performance vs base Sana attention.

## 16. Scope boundary and separate candidates

Separate candidate reports, not ignored:

- `pag_modular_guider`: `PerturbedAttentionGuidance`, `LayerSkipConfig`, and
  hook-based attention-score skipping. It can cover future modular pipelines but
  has different state and joint-conditioning limits.
- `sd_pag_img2img_inpaint_controlnet`: VAE encode, masks, masked-image latents,
  ControlNet residuals, and strength/timestep slicing.
- `sdxl_pag_img2img_inpaint_controlnet`: SDXL dual text/time conditioning plus
  the same variant surfaces.
- `sd3_pag_img2img`: VAE encode and FlowMatch strength handling over 16-channel
  latents.
- `animate_diff_pag`: video latents and motion modules.
- `kolors_pag_chatglm`: only after Kolors ChatGLM prompt boundary and UNet text
  projection are admitted.
- `pixart_pag`, `sana_pag`, `hunyuan_pag`: transformer-specific PAG variants
  with T5/masks/RoPE/linear attention details.
- `pag_ip_adapter`: added K/V image branches and image embedding expansion.
- `pag_lora_runtime_adapters`: layer selection and projection-hoist invalidation
  when LoRA/PEFT mutates active attention modules.
- Rare scheduler combinations: inherited Karras/DPM/UniPC/Flow variants remain
  scheduler-matrix candidates, not PAG-specific operators.

Ignored/out of scope for this audit:

- Safety checker and NSFW filtering.
- Callback mutation and interactive interrupt.
- Training, losses, dropout, and gradient checkpointing.
- XLA, NPU, MPS, Flax, and ONNX branches.
- Multi-GPU/context parallel paths.

## 17. Final implementation checklist

- [ ] Add a PAG guidance overlay schema with `pag_scale`, `pag_adaptive_scale`,
      resolved `pag_applied_layers`, and processor family.
- [ ] Represent selected attention layers in manifest/execution-plan metadata.
- [ ] Implement 2x/3x PAG batch expansion for prompt embeds, masks, pooled
      embeds, time IDs, masks, ControlNet inputs, and IP image embeds.
- [ ] Implement PAG/CFG fused guidance arithmetic.
- [ ] Preserve `noise_pred_text` for guidance-rescale paths.
- [ ] Implement identity self-attention PAG processor parity for UNet/PixArt.
- [ ] Implement processor restore or immutable selected-processor dispatch.
- [ ] Add SDXL one-step PAG parity with EulerDiscrete and supplied embeddings.
- [ ] Add SD inpaint and ControlNet PAG expansion parity.
- [ ] Add SD3 joint PAG processor parity, including perturbed attention mask.
- [ ] Add Hunyuan QK-norm/RoPE PAG parity after Hunyuan base support.
- [ ] Add Sana linear-attention PAG parity after Sana base support.
- [ ] Add AnimateDiff PAG only after video latent/motion module support.
- [ ] Add `pag_scale=0.0` base-equivalence regressions.
- [ ] Benchmark denoiser batch multiplier, selected layer count, attention
      provider choices, and guidance arithmetic overhead.
