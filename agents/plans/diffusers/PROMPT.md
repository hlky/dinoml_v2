# Diffusers pipeline and model assessment prompt

Diffusers source is available in `diffusers`.
Local Hugging Face config cache is available in `H:/configs`. This cache often
contains `model_index.json` files and may also contain component configs saved
by prior audits.

This is not a runtime code edit task. Do not implement Dinoml operators, edit
Dinoml runtime code, commit changes, or run Dinoml tests unless the user
explicitly changes scope. Reading source files, fetching Hugging Face configs,
and writing Markdown reports is expected.

Diffusers does not follow the same shape as `transformers`: a target pipeline is
usually composed from pipeline code, one or more model modules, scheduler code,
processors, loader mixins, and external text/image/audio encoders. Do not assume
the full inference graph lives in one modeling file.

Output should be written in this workspace for `dinoml_v2` under
`agents/plans/diffusers/{{TARGET_SLUG}}/report.md`. Cross-target prompt review
notes may be written under `agents/plans/diffusers/report_review.md`.

## Prompt

You are helping develop `dinoml_v2`, a high-performance inference/runtime stack.
Your task is to inspect a Hugging Face Diffusers pipeline or model family and
write a standardized report that helps Dinoml agents plan operator coverage,
graph rewrites, kernel fusion candidates, scheduler/runtime staging, and staged
integration work.

### Target

- Pipeline or model family: `{{TARGET_SLUG}}`
- Hugging Face model id(s), if known: `{{HF_MODEL_IDS}}`
- Diffusers source file(s): `{{DIFFUSERS_SOURCE_URLS}}`
- Commit or version to inspect: `{{DIFFUSERS_COMMIT_OR_VERSION}}`
- Primary inference task: `{{TASK}}`
- Runtime scope: `{{RUNTIME_SCOPE}}`
- Dinoml assumptions: `{{DINOML_ASSUMPTIONS}}`

Examples for `RUNTIME_SCOPE`:

```text
text-to-image base pipeline, with related variant/adaptor candidates inventoried separately
latent denoiser only, externally supplied prompt embeddings
VAE decode only
video transformer denoising loop, text encoder and VAE treated as separate stages
```

Examples for `DINOML_ASSUMPTIONS`:

```text
Inference-only first.
CUDA GPU target.
NHWC/channel-last preferred for vision tensors and latent feature maps when safe.
Treat schedulers and CFG arithmetic as explicit runtime graph/state, not hidden Python control flow.
Prefer graph rewrite rules that canonicalize common patterns into Conv/GEMM/attention/norm primitives.
```

Treat NCHW/NCDHW -> NHWC/NDHWC/channel-last as a guarded layout/fusion
optimization, not as the default semantic translation. Initial graph
translation should remain faithful to Diffusers/PyTorch axes unless a region is
local and fully controlled. Reports should identify candidate regions where
layout translation is safe, regions that need a no-layout-translation guard, and
axis-sensitive attrs that a layout pass would have to rewrite, such as
`dim=1 -> dim=-1`, concat/reduction/pooling/group-norm axes, view/reshape
assumptions, scheduler tensor broadcasting, VAE scale/shift application, and
downstream component layout contracts.

### What to inspect

Inspect the official Diffusers implementation and representative checkpoint
configs. Prefer exact source at a pinned commit. Use primary sources only when
possible:

- `src/diffusers/pipelines/{{family}}/pipeline_*.py`
- `src/diffusers/models/**/{{model_file}}.py`
- shared model files referenced by the target, such as `attention.py`,
  `attention_processor.py`, `attention_dispatch.py`, `embeddings.py`,
  `normalization.py`, `resnet.py`, `downsampling.py`, and `upsampling.py`
- scheduler files such as `scheduling_*.py`
- image/video/audio processors and pipeline helpers, such as
  `image_processor.py`, `video_processor.py`, `free_*_utils.py`, or
  family-local helper files
- loader/mixin files enough to inventory related candidate reports when LoRA,
  textual inversion, runtime adapters, IP-Adapter, ControlNet, T2I-Adapter, or
  single-file conversion changes runtime tensors or weights; inspect them
  deeply when that extension is the selected target
- Hugging Face repo files: `model_index.json`, component `config.json` files,
  scheduler config, tokenizer/processor configs, and safetensors index metadata
- external `transformers` component configs when the diffusers pipeline depends
  on CLIP, T5, SigLIP, Llama-like text encoders, or vision encoders

Do not rely only on README/model-card claims. Confirm operations from code.

For each target, trace the component wiring from the pipeline constructor and
`__call__` method before reading model internals. Record every component name,
class, optional component, and offload sequence declared by the pipeline.

Inspect at least 2-4 representative checkpoint/component configs when available:

- a small/debug checkpoint, if present
- a common production checkpoint
- a variant that changes operator structure, such as UNet vs DiT/transformer,
  epsilon vs v-prediction vs flow prediction, text encoder count, latent channel
  count, patch size, attention processor, ControlNet/IP-Adapter branch, image or
  video conditioning, or scheduler family

Check `H:/configs/<namespace>/<repo>/` for cached `model_index.json` and
component configs before fetching from the network. Use official repos when
accessible. If official repos look gated or unavailable through unauthenticated
HTTP, retry with the authenticated `hf` CLI or `huggingface_hub` Python API
before marking the config unavailable. If useful, save fetched configs under
`H:/configs/<namespace>/<repo>/` for future audits and record that local path in
the report. If official repos remain unavailable, use an open mirror and label
that fact clearly. Separate facts that come from component `config.json` files
from facts that come from scheduler configs, pipeline defaults, safetensors/index
metadata, or source defaults.

If source files are generated, copied, or shared through "Copied from" comments,
inspect the local target file plus the shared source when practical and state
which file is authoritative for future source edits.

If any component config omits fields that the current Diffusers class supplies
by default, list the omitted fields and effective defaults per component. Include
source defaults from model, scheduler, processor, and VAE classes, not just the
top-level pipeline config.

Record constructor-time compatibility repairs or config mutations in pipeline
`__init__`, such as scheduler `steps_offset`/`clip_sample` rewrites, because
artifact loading parity may depend on them.

Ignore multi-GPU/context-parallel paths, callback mutation and interactive
interrupt paths, safety checker and NSFW filtering, training/loss/dropout/
gradient-checkpointing paths, and XLA, NPU, MPS, Flax, and ONNX-specific code
paths unless the selected target explicitly asks for them. Mention only when
such a branch changes shared CPU/CUDA inference source structure or masks
CPU/CUDA behavior.

For broad Diffusers classes, separate branches active under the selected
component config from branches merely available in source. Do not count inactive
ControlNet/IP-Adapter/adapter/class/addition-embedding paths as required
first-slice ops unless the target pipeline enables them.

For extension and variant surfaces, create an explicit "separate candidate
reports" inventory rather than hiding them in a skip/defer list. At minimum,
document whether the family supports each item, list relevant class names and
source files, summarize how the pipeline variant differs from the base pipeline,
and propose a candidate slug/order when applicable:

- LoRA, textual inversion, and runtime adapter mutation.
- IP-Adapter.
- ControlNet.
- T2I-Adapter.
- GLIGEN.
- img2img.
- inpaint.
- depth2img.
- upscaling.

These are separate review candidates unless the selected target is one of them.
Their operators may be out of the base first-slice implementation, but their
classes, files, and variant deltas should still be documented.

### Report requirements

Write a Markdown report with the following sections.

## 1. Source basis

List exact files, URLs, commits, and checkpoint/component configs inspected.

Include:

```text
Diffusers commit/version:
Model id(s):
Config sources:
Pipeline files inspected:
Model files inspected:
Scheduler/processors/helpers inspected:
External component configs inspected:
Any missing files or assumptions:
```

## 2. Pipeline and component graph

Describe the top-level pipeline stages and component boundaries.

Include a simple dataflow diagram in text:

```text
prompt/image/video/audio preprocessing
  -> text/image/audio encoders
  -> latent initialization or VAE encode
  -> denoising loop: denoiser + guidance + scheduler
  -> VAE decode/postprocess
```

List required and optional components for the stated runtime scope.
Call out independently cacheable stages such as prompt embeddings, image
embeddings, VAE latents, video conditioning, and scheduler timesteps.

Also include a "separate candidate reports" list for supported extension and
variant surfaces. Include class names and files for LoRA/textual inversion/
runtime adapter mutation, IP-Adapter, ControlNet, T2I-Adapter, GLIGEN, img2img,
inpaint, depth2img, and upscaling when present in the family.

## 3. Important config dimensions

Extract dimensions from configs and present them as tables.

Include relevant fields such as:

```text
latent channels, VAE scale factor, sample size, patch size
UNet block channels / layers / attention dims
DiT hidden size / heads / head dim / depth / MLP ratio
cross-attention dims and text encoder count
pooled prompt embed dims and time/text conditioning dims
image/video frame shape, tubelet/patch sizes, temporal compression
scheduler prediction_type, timestep/sigma schedule, solver order
supported scheduler family/set, source default scheduler for sampled checkpoints,
and recommended first Dinoml scheduler slice
guidance mode, CFG batching strategy, guidance embeddings, skip-layer guidance
dtype and weight precision metadata
```

Also include a representative checkpoint sweep table. Make operator-significant
variation visible rather than only reporting the smallest example.

If a pipeline can load multiple scheduler families, do not infer the whole
pipeline contract from one checkpoint's default scheduler. State the supported
set, then identify which schedulers are required for first parity. If the source
default scheduler and the recommended first Dinoml scheduler slice differ,
explain the parity and staging tradeoff. Some families are broad, such as Stable
Diffusion 1.x supporting DDIM/Euler/DPM-style scheduler swaps through compatible
scheduler APIs, while other families may only support flow-style or one
model-specific scheduler.

If a family has more than one guidance mechanism, report them separately. For
example, embedded guidance is a model-conditioning tensor, while true
classifier-free guidance usually means an additional positive/negative denoiser
call and explicit CFG arithmetic.

## 3a. Family variation traps

List config-dependent behavior that invalidates naive assumptions, such as:

```text
UNet vs transformer denoiser
latent channels 4 vs 8/16/32
NCHW/NCDHW source latent maps versus NHWC/NDHWC/channel-last layout-pass candidates
axis-sensitive ops that need no-layout-translation guards
source permute/transpose/contiguous patterns that can be eliminated, sunk, or folded
single CLIP encoder vs dual CLIP vs T5/LLM text encoder
multi-encoder conditioning composition: feature concat, zero-fill, padding,
sequence concat, pooled projection concat, and optional missing encoder behavior
cross-attention vs joint text-image attention
epsilon/v_prediction/sample/flow_prediction target
fixed timestep embeddings vs guidance/text/size conditioning
2D image vs 3D/video/audio tensors
ControlNet/IP-Adapter/T2I-Adapter side inputs
LoRA/textual inversion/runtime adapter mutation surfaces
img2img/inpaint/depth/upscale variant pipeline contracts
attention processors that fuse QKV or add external K/V branches
scheduler custom timestep/sigma support
VAE tiling/slicing or temporal decode paths
constructor-time config repair or compatibility mutation
inactive branches in shared classes that should not inflate first-slice scope
folder-level variants that share a directory but have distinct pipeline/model
classes
same-family checkpoints that share a pipeline class but change depth, width,
QK norm, dual-attention layers, latent channels, or scheduler/guidance defaults
```

## 4. Runtime tensor contract

Document exact runtime tensors at each boundary:

- pipeline inputs after preprocessing
- prompt/text/image/audio embeddings and masks
- latent tensor shape, layout, dtype, scaling factor, and noise initialization
- both source latent-map shapes and packed transformer-token shapes when a
  pipeline packs latents, including exact view/permute/reshape order
- patchify/unpatchify contracts when the model patchifies internally, including
  exact patch axes, reshape/einsum/permute order, and how this differs from
  pipeline-level latent packing
- source layout and any candidate optimized layout at every vision/video/audio
  component boundary
- denoiser inputs and outputs for one denoising step
- scheduler state tensors and scalar tables
- autoencoder encode and decode inputs and outputs, even when the primary
  text-to-image path only decodes
- postprocessing tensors

Separate CPU/data-pipeline work from GPU/runtime work. Identify tensors that can
be precomputed and reused across denoising steps or requests.

## 5. Operator coverage checklist

List required runtime operators grouped by category.

Use categories like:

```text
Tensor/layout ops
Convolution/downsample/upsample ops
GEMM/linear ops
Attention primitives
Normalization and adaptive conditioning
Position/timestep/guidance embeddings
Scheduler and guidance arithmetic
VAE/postprocessing ops
Control/adapter/indexed-update ops
Video/audio-specific ops
```

Be explicit. Prefer shape-aware examples such as `Conv2d(4 -> 320, 3x3,
padding=1)` or `JointAttention(hidden=3072, heads=24, context=4096)` when known.
For vision/video operators, state source tensor layout and any candidate
optimized layout explicitly, especially around VAE encode/decode, UNet
ResNet/downsample/upsample blocks, patchify/unpatchify, convolutions, pooling,
normalization, and image/video processors. If an optimized layout would change
axis numbers, list those required axis rewrites instead of silently changing the
semantic graph.

## 6. Denoiser/model breakdown

For each major denoiser or model block, describe the forward path.

Examples:

```text
UNet down block:
  ResnetBlock2D: GroupNorm -> SiLU -> Conv2d -> time embedding add/scale-shift -> GroupNorm -> SiLU -> Conv2d -> residual
  BasicTransformerBlock: norm -> self/cross attention -> residual -> feed-forward -> residual
  downsample

DiT/joint transformer block:
  adaptive norm from timestep/text guidance
  image/text QKV projection
  optional QK norm and RoPE
  joint attention
  gated residual attention and MLP
```

Include shapes, layouts, bias flags, adaptive norm/gating behavior, attention
processor variants, and any branch controlled by config.

## 7. Attention requirements

Describe every attention variant required.

Include:

```text
self-attention, cross-attention, joint attention, added-KV attention, IP-Adapter attention
2D latent sequence attention, patch attention, temporal/video attention
head count / head dim / cross-attention dim
masking style and packed/varlen requirements
QK norm / RMSNorm / LayerNorm on query/key
RoPE, 2D/3D positional embeddings, absolute learned positions
processor/backend dispatch path
fallback eager path and optimized path
```

State whether `attention_processor.py` or `attention_dispatch.py` is the primary
implementation for the target and whether fused projections are source-supported
or required.

Do not assume FlashAttention support is either available or unavailable.
Diffusers has target- and backend-specific flash, native SDPA, xFormers, flex,
sage, and varlen paths, and some families only use a subset today. For each
target, record the exact processor/backend dispatch path, whether existing
flash-style kernels support the required masks, joint attention, added K/V, QK
norm, RoPE, GQA/varlen, and dtype, and what eager/native fallback defines
parity. If Diffusers does not currently use flash for a target, still identify
whether a Dinoml flash-style provider could be valid under stricter
preconditions.

## 8. Scheduler and denoising-loop contract

Document scheduler setup, step math, and loop-side graph work.

Include:

```text
set_timesteps inputs and output tables
timesteps/sigmas device and dtype behavior
scale_model_input or scale_noise behavior
prediction_type conversion
stateful step index / order / lower-order warmup
CFG batching or separate positive/negative denoiser calls
guidance_rescale, true CFG, embedded guidance, skip-layer guidance
callbacks or interrupt behavior if relevant
```

Mark which parts should remain host control flow initially and which parts are
candidates for compiled runtime kernels.

State the source default scheduler for the checkpoint and the recommended first
Dinoml scheduler slice. If they differ, explain the parity and staging tradeoff.

## 9. Position, timestep, and custom math

Document sinusoidal timestep embeddings, Fourier embeddings, size/crop
conditioning, 2D/3D sin-cos embeddings, RoPE, AdaLayerNorm variants, and any
model-specific custom math. Include short snippets only for custom functions
Dinoml may need to reproduce.

Mention what can be precomputed and what depends on dynamic image size, frame
count, prompt length, guidance scale, or timestep.

Note source comments or backend guards that mention layout, dtype, or backend
limitations. Treat them as evidence for validation priorities, not automatic
Dinoml limitations.

## 10. Preprocessing and input packing

Document model-coupled preprocessing that affects runtime graph shape:

```text
tokenization and prompt embed duplication
negative prompt handling
multi-encoder prompt composition, including padding and sequence concatenation
image/video resize/crop/normalize and VAE scaling
mask/image latent concatenation for inpaint/img2img
patchify/unpatchify and latent image/video IDs
ControlNet/T2I-Adapter/IP-Adapter conditioning
audio feature extraction or vocoder paths
image postprocess
```

Separate CPU/data-pipeline work from GPU/runtime work.

## 11. Graph rewrite / lowering opportunities

Identify patterns that can be canonicalized into simpler primitives.

For each rewrite, include:

```text
name
source pattern
replacement pattern
exact preconditions
shape equations
weight transform
layout constraints
failure cases
parity test sketch
```

Be strict. Do not propose unsafe rewrites without guards. For convolution or
patch rewrites, include layout-aware preconditions, activation flatten order,
weight flatten/permutation, bias handling, dynamic guards, and fallback behavior.
For layout rewrites, call out source `permute`/`transpose`/`contiguous` patterns
that can be eliminated by a guarded layout/fusion pass. Include required axis
rewrites, weight transforms, consumer layout constraints, and failure cases where
the source layout must be preserved for parity. If useful, identify regions that
should be protected by a conceptual `no_layout_translation()` guard.

## 12. Kernel fusion candidates

Rank likely fusion/kernel work by priority.

Use categories:

```text
Highest priority
Medium priority
Lower priority
```

For each candidate, explain why it matters and which target families exercise
it. Examples:

```text
GroupNorm/LayerNorm/RMSNorm and adaptive norm
Conv2d + GroupNorm + SiLU in UNet ResNet blocks
QKV or joint QKV projection + QK norm + attention
AdaLayerNorm gating + residual
GEGLU/GELU feed-forward activation
patchify/unpatchify
CFG arithmetic and guidance rescale
scheduler step arithmetic
VAE decode conv/upsample/resnet blocks
video temporal attention and 3D VAE blocks
```

## 13. Runtime staging plan

Propose a staged Dinoml integration path.

Example:

```text
Stage 1: parse component configs and load weights for one small pipeline
Stage 2: compile/validate VAE decode or one denoiser block
Stage 3: one denoising step parity with externally supplied prompt embeddings
Stage 4: full denoising loop with scheduler in Python
Stage 5: add text encoder or prompt embed cache integration
Stage 6: add optimized attention/norm/conv fusions
Stage 7: add img2img/inpaint/control/video variants
```

Include what can be stubbed initially.
For pipelines that use an autoencoder, include whether autoencoder encode,
decode, or both are required by the family report, and prefer a separate
autoencoder-family report for codec-specific optimization candidates.

## 14. Parity and validation plan

Give concrete parity tests.

Include:

```text
random tensor tests for custom ops
single block parity
one denoiser step parity at fixed timestep/sigma
scheduler step parity
VAE encode/decode parity
prompt embedding and CFG parity
short deterministic denoising-loop parity
end-to-end image/video/audio output smoke
recommended tolerances for fp32/fp16/bf16
```

## 15. Performance probes

List benchmark probes that separate bottlenecks:

```text
text encoder throughput
VAE encode/decode throughput
one denoiser step by resolution/batch
full denoising loop by step count
scheduler/guidance overhead
attention backend comparison
UNet conv/resnet vs attention time split
DiT patch length sweep
video frame-count sweep
VRAM and temporary/workspace usage
offload/load timing
```

If benchmark observations or prior measurements are included, label provenance
and separate them from source-derived facts.

## 16. Scope boundary and separate candidates

Split this section into two lists.

First, list separate candidate reports that are related to the family but are
not part of the selected target's first implementation slice. These are not
"ignored"; include class/file anchors and the reason they deserve a separate
review. Candidate examples:

```text
LoRA merge/application at runtime
textual inversion token/embedding mutation
runtime adapter mutation
IP-Adapter
ControlNet
T2I-Adapter
GLIGEN
img2img
inpaint
depth2img
upscaling
rare schedulers
```

Second, list genuinely ignored/out-of-scope surfaces for the current audit, such
as:

```text
multi-GPU/context parallel paths
callback mutation and interactive interrupt
XLA, NPU, MPS, Flax, and ONNX pipeline variants
safety checker and NSFW filtering
training, losses, dropout, and gradient checkpointing
```

Only defer or ignore features if they are not required for the stated task.

## 17. Final implementation checklist

End with a compact checklist Dinoml agents can copy into issues:

```markdown
- [ ] Parse component configs
- [ ] Load weights
- [ ] Implement operator X
- [ ] Implement scheduler step Y
- [ ] Add rewrite Z
- [ ] Add parity test A
- [ ] Benchmark B
```

### Style constraints

- Be concrete and shape-aware.
- Prefer exact source-derived claims over guesses.
- If making an inference, label it as an inference.
- Label whether dtype, parameter count, license, task, model size, and scheduler
  behavior come from component configs, scheduler configs, safetensors/index
  metadata, Hugging Face repo metadata, or source defaults.
- Distinguish required functionality from optimization opportunities.
- Include snippets for custom math, but keep them short.
- Do not paste large source blocks.
- Do not overfit to PyTorch naming if a more general runtime concept is clearer.
- Always state which files complete the picture when modeling code is split
  across pipeline, shared model, scheduler, processor, and loader files.
