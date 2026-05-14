# Diffusers Report Review and Prompt Refinement Notes

## Reports reviewed

- `agents/plans/diffusers/stable_diffusion_1_5/report.md`
- `agents/plans/diffusers/autoencoder_kl/report.md`
- `agents/plans/diffusers/flux/report.md`
- `agents/plans/diffusers/stable_diffusion_3/report.md`
- `agents/plans/diffusers/stable_diffusion_2_x/report.md`
- `agents/plans/diffusers/stable_diffusion_xl/report.md`
- `agents/plans/diffusers/sd1_img2img_inpaint_depth_upscale/report.md`
- `agents/plans/diffusers/sd1_lora_textual_inversion_adapters/report.md`
- `agents/plans/diffusers/controlnet_sd/report.md`
- `agents/plans/diffusers/sd1_ip_adapter/report.md`
- `agents/plans/diffusers/sd1_t2i_adapter/report.md`
- `agents/plans/diffusers/sd1_gligen/report.md`
- `agents/plans/diffusers/wan/report.md`
- `agents/plans/diffusers/cogvideox/report.md`
- `agents/plans/diffusers/video_autoencoders/report.md`
- `agents/plans/diffusers/autoencoder_tiny/report.md`
- `agents/plans/diffusers/sdxl_controlnet_adapters/report.md`
- `agents/plans/diffusers/pixart/report.md`
- `agents/plans/diffusers/sana/report.md`
- `agents/plans/diffusers/scheduler_matrix/report.md`
- `agents/plans/diffusers/flux_variants_control/report.md`
- `agents/plans/diffusers/qwen_image/report.md`
- `agents/plans/diffusers/hunyuan_image/report.md`
- `agents/plans/diffusers/ltx_video/report.md`
- `agents/plans/diffusers/stable_audio/report.md`
- `agents/plans/diffusers/auraflow_lumina/report.md`
- `agents/plans/diffusers/hunyuan_video/report.md`
- `agents/plans/diffusers/mochi_video/report.md`
- `agents/plans/diffusers/audioldm_musicldm/report.md`
- `agents/plans/diffusers/z_image/report.md`
- `agents/plans/diffusers/stable_cascade/report.md`
- `agents/plans/diffusers/deepfloyd_if/report.md`
- `agents/plans/diffusers/kandinsky_family/report.md`
- `agents/plans/diffusers/ace_step/report.md`
- `agents/plans/diffusers/allegro/report.md`
- `agents/plans/diffusers/animatediff_stable_video_diffusion/report.md`
- `agents/plans/diffusers/bria_bria_fibo/report.md`
- `agents/plans/diffusers/chroma/report.md`
- `agents/plans/diffusers/chronoedit/report.md`
- `agents/plans/diffusers/cogview3_4/report.md`
- `agents/plans/diffusers/consisid/report.md`
- `agents/plans/diffusers/consistency_ddim_ddpm_latent_diffusion/report.md`
- `agents/plans/diffusers/controlnet_sd3/report.md`
- `agents/plans/diffusers/controlnet_hunyuandit/report.md`
- `agents/plans/diffusers/cosmos/report.md`
- `agents/plans/diffusers/dit/report.md`
- `agents/plans/diffusers/easyanimate/report.md`
- `agents/plans/diffusers/ernie_glm_image/report.md`
- `agents/plans/diffusers/flux2/report.md`
- `agents/plans/diffusers/helios/report.md`
- `agents/plans/diffusers/hunyuandit/report.md`
- `agents/plans/diffusers/kolors/report.md`
- `agents/plans/diffusers/latent_consistency_models/report.md`
- `agents/plans/diffusers/hidream_image/report.md`
- `agents/plans/diffusers/latte/report.md`
- `agents/plans/diffusers/ledits_pp/report.md`
- `agents/plans/diffusers/llada2/report.md`
- `agents/plans/diffusers/longcat_audio_dit/report.md`
- `agents/plans/diffusers/longcat_image/report.md`
- `agents/plans/diffusers/ltx2/report.md`
- `agents/plans/diffusers/lucy/report.md`
- `agents/plans/diffusers/marigold/report.md`
- `agents/plans/diffusers/nucleusmoe_image/report.md`
- `agents/plans/diffusers/omnigen/report.md`
- `agents/plans/diffusers/ovis_image/report.md`
- `agents/plans/diffusers/pag/report.md`
- `agents/plans/diffusers/prx/report.md`
- `agents/plans/diffusers/sana_video/report.md`
- `agents/plans/diffusers/shap_e/report.md`
- `agents/plans/diffusers/skyreels_v2/report.md`
- `agents/plans/diffusers/visualcloze/report.md`

## Review findings

### Distributed audit batch 1

- SDXL fetched official base/refiner/turbo component configs successfully and
  did not leave any config blocker.
- SD2 and SD2 depth are a special case: the original StabilityAI repos/config
  paths were removed. Per project steering, use `https://huggingface.co/sd2-community`
  as the SD2.x config source for this audit rather than treating it as a
  temporary mirror or access blocker.
- SD1 variants confirmed that img2img/inpaint/depth/upscale are not just input
  wrappers: they add VAE encode, strength/timestep slicing, mask/depth/low-res
  conditioning, and sometimes UNet input-channel changes.
- SD1 adapter mutation confirmed LoRA/textual inversion should be modeled as
  artifact/tokenizer/weight-state mutation first, with unfused runtime adapters
  requiring an explicit adapter-state schema before admission.
- Stable Diffusion ControlNet confirmed residual side-branch execution and
  multi-control residual reduction should be separate candidate/runtime stages,
  not hidden inside the base SD report.

### Distributed audit batch 2

- SD1 IP-Adapter has no useful `model_index.json` shape in `h94/IP-Adapter`;
  the report instead anchors on official image encoder configs and safetensors
  headers. Treat the image encoder/projection modules and added K/V attention
  processors as runtime graph additions, not base SD attention.
- SD1 T2I-Adapter local TencentARC model indexes were empty, but public raw
  adapter configs were accessible. Adapter feature pyramids and MultiAdapter
  scale/sum behavior are separate residual-input stages from ControlNet.
- GLIGEN is deprecated and lower priority, but it has real graph deltas:
  grounded box/text/image projection, legacy `use_gated_attention` config
  normalization, `attention_type` admission, fuser weights, null feature
  replacement, and gated self-attention in UNet transformer blocks. Text-image
  GLIGEN was source-audited but not config-swept.
- Wan established the first video-family audit shape: video latent layouts,
  temporal packing, Wan transformer blocks, Wan VAE, and UniPC/FlowMatch-style
  scheduler concerns. Wan2.2 I2V official `model_index.json` declares null
  `image_encoder` and `image_processor`; treat those as absent component slots,
  not gated-config failures.

### Distributed audit batch 3

- CogVideoX confirmed the second video-family report shape: local cache often
  has only `model_index.json`, but official raw transformer/VAE/scheduler/text
  configs were accessible. Temporal patching, video latent layout, gated
  residual modulation, and scheduler state need separate validation from Wan.
- Video autoencoders should be tracked as their own codec lane. Wan, CogVideoX,
  LTX/LTX2, and Mochi vary in temporal compression, causal/3D conv structure,
  tiling/chunking policy, scaling, and latent layout. This should not be folded
  into one generic AutoencoderKL assumption.
- AutoencoderTiny is a compact inference codec candidate with accessible TAESD,
  TAESDXL, TAESD3, and TAEF1 configs. It is lower risk than full AutoencoderKL
  but has different scaling and block structure, so it deserves its own
  admission path.
- SDXL ControlNet/T2I/IP adapter variants mirror SD1 concepts but inherit SDXL
  dual text/pooled/time-id conditioning. Local config entries were placeholders;
  official raw config/header data was accessible and no gated path blocked.

### Distributed audit batch 4

- PixArt fills the DiT bridge between UNet SD and SD3/Flux: T5 embeddings,
  patch-token transformer, AdaLN-single style conditioning, VAE boundary, and
  DPM/LCM scheduler variants. `PixArt-Sigma-XL-2-512-MS` scheduler config
  returned 404, so scheduler behavior for that exact checkpoint should not be
  asserted beyond family evidence.
- Sana introduces efficient-transformer details and `AutoencoderDC`, plus SCM
  scheduler variants. Official Sana component configs were fetched and saved
  under `H:/configs`, which makes it a better config-backed target than several
  placeholder-only adapter repos.
- The scheduler matrix is now a shared reference for DDIM/DDPM/PNDM/Euler,
  DPM-Solver, UniPC, FlowMatch, CogVideoX, and related prediction/state
  contracts. Future family reports should link to it instead of restating every
  scheduler detail.
- Flux variant/control configs from Black Forest Labs require authenticated
  reads for HTTP access; authenticated metadata reads succeeded. Local variant
  caches are often `{}` placeholders, so config-backed claims should cite the
  authenticated official metadata path or source-only status explicitly.

### Distributed audit batch 5

- QwenImage introduces a Qwen2 prompt-template and masked-token extraction path,
  packed 2x2 latent tokens, FlowMatch scheduling, QwenImage VAE, and control
  variants. Base text-to-image repos return 404 for
  `processor/preprocessor_config.json`, which is expected because that component
  is absent in the base pipeline.
- HunyuanImage 2.1 is config-backed now: base/refiner and older HunyuanDiT JSON
  configs were saved under `H:/configs`. It has Qwen/ByT5-style conditioning,
  token reordering, patch-size-1 image tokens, and FlowMatch scheduler staging.
- LTX Video adds another packed-token video family with per-token timestep and
  condition-token variants. Its component configs were reachable without gated
  retry across 0.9.x, spatial upscaler, and LTX-2 scope-separation examples.
- Stable Audio is the first audio target. The official
  `stabilityai/stable-audio-open-1.0` repo was reported gated by the worker and
  mirror configs were used. The main thread has authenticated `hf` as `hlky`, so
  a follow-up should retry official JSON fetches before treating mirror details
  as canonical.

### Distributed audit batch 6

- AuraFlow/Lumina rounds out several image DiT variants. Lumina Next raw resolve
  URLs returned Git-LFS pointer text or 404 for some component configs, but
  local/authenticated HF cache access worked. Future config fetchers should
  handle LFS pointer responses explicitly instead of treating them as JSON.
- Hunyuan Video has many related source surfaces: base, I2V, SkyReels,
  Framepack, and HunyuanVideo 1.5. Local cache had `model_index.json` only, but
  official raw component configs for representative base/I2V/1.5 repos were
  accessible.
- Mochi video was cleanly accessible and not gated. Its local cache only had
  `model_index.json`; official component configs and safetensors metadata filled
  in transformer/VAE/scheduler/text defaults.
- AudioLDM/MusicLDM had no gated blockers. Some 404s are expected absent
  components, such as AudioLDM v1 and MusicLDM lacking AudioLDM2-only
  `text_encoder_2`, GPT2, or projection components.

### Distributed audit batch 7

- The audit scope is now expanded from curated high-value targets to all
  non-deprecated Diffusers pipeline folders. `candidates.md` has an exhaustive
  backlog section derived from `src/diffusers/pipelines`, excluding
  `deprecated`.
- Z-Image base/turbo configs were accessible from official raw URLs, but
  Alibaba Z-Image ControlNet single-file repos returned 404 for raw
  config/model-index paths. The report uses `neuralvfx/Z-Image-SAM-ControlNet`
  only as a variant/config reference.
- Stable Cascade completed without gated blockers and adds a multi-stage
  prior/decoder Paella VQGAN stack distinct from SD-style AutoencoderKL.
- DeepFloyd IF completed without blocking gated configs. Missing component paths
  are expected stage absences: Stage I lacks image noising scheduler/VAE, Stage
  II lacks VAE/low-res scheduler, and Stage III uses the x4 upscaler's
  `low_res_scheduler`.
- Kandinsky family completed across non-deprecated Kandinsky 1/2.2/3/5 folders
  plus KVAE-adjacent codec configs. Local cache mostly had model indexes only,
  but official raw/API reads filled in component details.

### Distributed audit batch 8

- ACE-Step 1.5 current source does not have an official Diffusers
  `model_index.json` or scheduler config in the repo. The report therefore
  anchors the current source contract and separates the older
  `ACE-Step-v1-3.5B` stack.
- Allegro base is config-backed. `Allegro-TI2V` references
  `AllegroTransformerTI2V3DModel`, which is not present in this checkout, so
  TI2V is fenced as a blocked variant rather than inferred.
- AnimateDiff/SVD completed with one caveat: `ByteDance/AnimateDiff-Lightning`
  root `config.json` returned 404 and local cache only had an empty
  `model_index.json`. It is inventoried as a separate LCM/Lightning candidate.
- Bria/FIBO completed with access caveats: official `briaai/BRIA-3.2` returned
  404 even with a token, so an open mirror was used and labeled; `briaai/Fibo-Edit`
  metadata listed components but raw config reads returned 403, so edit is
  source/test-audited and blocked for official config sweep.

### Distributed audit batch 9

- Chroma completed cleanly and fetched official component configs under
  `H:/configs/lodestones/*`; it is another Flux-like FlowMatch transformer
  family with Chroma-specific config/defaults.
- ChronoEdit has two unresolved audit caveats: official
  `nvidia/ChronoEdit-14B-Diffusers` config access returned raw 401 and
  authenticated no-access/404, and current source names
  `FlowMatchEulerDiscreteScheduler` while public config/docs name UniPC/Wan
  classes. Scheduler compatibility is a first validation blocker.
- CogView3/4 base configs were accessible, but CogView4-Control configs were
  gated/unavailable without accepted access. The control variant is source-only
  for now.
- ConsisID completed cleanly; local cache had only `model_index.json`, but
  official component configs were accessible and cover the identity-conditioned
  CogVideoX-adjacent video path.

### Distributed audit batch 10

- Classic consistency/DDIM/DDPM/latent-diffusion pipelines completed without
  gated blockers. These reports are useful as small operator and scheduler
  baselines, distinct from Stable Diffusion family complexity.
- SD3 ControlNet completed with empty cached ControlNet model indexes and one
  explicit 404: `calcuis/sd3.5-large-controlnet/config.json`; no SD3.5-large/8B
  ControlNet config contract is inferred.
- HunyuanDiT ControlNet completed cleanly using public raw Canny/Depth/Pose
  ControlNet configs; local cache lacked those component configs.
- Cosmos completed with a real gated blocker:
  `nvidia/Cosmos-Transfer2.5-2B` remains gated even with authenticated `hf`
  as `hlky`. Transfer2.5 config details are source-derived/blocked until access
  is granted.

### Distributed audit batch 11

- DiT completed cleanly and acts as a compact class-conditional patch-token
  baseline using DDIM and VAE decode, distinct from text-conditioned PixArt/SD3.
- EasyAnimate completed cleanly; official V5.1 component configs were reachable
  and temporary fetched copies were removed after inspection.
- ERNIE/GLM Image completed cleanly with official configs accessible and only
  expected absent components.
- Flux2 completed with a real access caveat: official Black Forest Labs 9B/KV
  configs returned 403 even as authenticated `hlky`. Mirror configs are used
  only as labeled shape evidence.

### Distributed audit batch 12

- Helios completed with public production configs for Base/Mid/Distilled and
  covers both the base and pyramid pipeline surfaces. The only access caveat is
  an internal tiny Helios raw-config 401, which does not block production
  operator contracts.
- Base HunyuanDiT completed cleanly as a separate family from Hunyuan Image 2.1
  and HunyuanDiT ControlNet. Missing local v1.1 component configs were reachable
  via official raw URLs.
- Kolors completed with authenticated config reads. Base/img2img are source
  wired; inpaint has a real 9-channel UNet config but no Kolors-specific
  inpaint pipeline class in this checkout, and Kolors ControlNet/IP-Adapter use
  separate nonstandard candidate surfaces.
- Latent Consistency Models completed without gated blockers. Some
  `latent-consistency/*` entries are intentionally LoRA-only or UNet-only
  rather than full pipeline repos, so admission should distinguish folder-local
  LCM pipelines from SD/SDXL pipelines paired with `LCMScheduler`.

### Distributed audit batch 13

- HiDream Image completed with real access/availability caveats: official repos
  omit `text_encoder_4`/`tokenizer_4` and require external gated Meta Llama 3.1
  components, which returned 403 even with the available token. E1 configs also
  reference `HiDreamImageEditingPipeline`, but that class/file is absent in this
  checkout.
- Latte completed cleanly against `maxin-cn/Latte-1`. The older
  `maxin-cn/Latte` cache/raw paths are not a current full Diffusers pipeline
  contract, and the temporal decoder is a separate candidate because the active
  model index wires framewise `AutoencoderKL`.
- LEDITS++ completed as an inversion/edit overlay over SD1/SDXL components, not
  a separate checkpoint family. The SDXL IP-Adapter branch is present but
  source-marked incomplete/TODO, so it remains fenced as a separate candidate.
- LLaDA2 completed with no gated blocker but an unusual integration shape:
  official repos intentionally lack `model_index.json`, scheduler config, and
  generation config because the pipeline wires external Transformers
  model/tokenizer objects and uses block-refinement scheduling.

### Distributed audit batch 14

- LongCat Audio DiT completed with configs saved under `H:/configs`. The public
  Diffusers repo lacks `scheduler/scheduler_config.json`, so admission must
  materialize the source constructor fallback
  `FlowMatchEulerDiscreteScheduler(shift=1.0, invert_sigmas=True)`.
- LongCat Image completed cleanly with Hub component configs. The edit pipeline
  has a source validation note around a direct latent shortcut referencing
  undefined `self.latent_channels`.
- LTX2 completed with official LTX-2 configs, but official LTX-2.3/fp8/nvfp4
  repos expose only root safetensors-style files and no Diffusers component
  configs/model index. LTX-2.3 Diffusers facts are therefore mirror-derived and
  labeled; one 2.3-dev mirror has connector/transformer caption-channel
  mismatch.
- Lucy completed cleanly as a Wan-adjacent edit pipeline; Decart repos are
  public and component configs are accessible.

### Distributed audit batch 15

- Marigold completed with official `prs-eth` configs; missing component JSONs
  were fetched with `hf download`. `marigold-depth-hr-v1-1` advertises
  `MarigoldDepthHRPipeline`, but that class is absent from this checkout and is
  fenced rather than inferred.
- NucleusMoE Image completed cleanly. Only one official full Diffusers
  checkpoint/config set was found, and the report captures source/test defaults
  as the only variation.
- OmniGen completed against public Shitao v1 Diffusers configs. OmniGen2 cached
  entries point to custom classes absent from this checkout, and source only
  warns on non-divisible H/W, so those need explicit guards before admission.
- Ovis Image completed against public official configs. Internal tiny Ovis raw
  reads returned 401 but were not needed for the production contract.

### Distributed audit batch 16

- PAG completed as source/runtime wrapper behavior over existing family
  configs, not as its own checkpoint family. No PAG-specific gated retry was
  needed.
- PRX completed against public Photoroom component configs. Top-level and
  transformer safetensors index files are absent because transformer/VAE weights
  are single safetensors files; text-encoder index metadata is present.
- Sana Video completed with public configs. The LongLive variant references
  `SanaVideoCausalTransformer3DModel`, which is absent from this checkout and
  is fenced as a source/config mismatch.
- Shap-E completed without gated blockers. Current pipelines register
  `shap_e_renderer`; model indexes also contain legacy `renderer`. The
  `shap_e_renderer` folder has `.bin` weights but no fp16 safetensors, and mesh
  decoder LUT buffers need `.bin` verification before claiming fp16 renderer
  mesh parity.
- SkyReels v2 completed cleanly across T2V, I2V, and DF variants; expected 404s
  for image components on non-I2V repos are absent component slots.
- VisualCloze completed cleanly and saved official 384/512 configs under
  `H:/configs/VisualCloze`.

### Stable Diffusion 1.5 report

- The prompt shape worked: starting from the pipeline exposed important runtime
  graph work that would be missed by reading only `UNet2DConditionModel`,
  including CFG batch concatenation/chunking, scheduler state, latent scaling
  before VAE decode, optional guidance rescale, and callback mutation surfaces.
- The report confirmed the NHWC guidance needs to be careful. SD source uses
  NCHW for latents and images, but candidate Dinoml optimization should be
  guarded NHWC conv islands. GroupNorm, concat/chunk, attention flatten/reshape,
  scheduler broadcasting, and all "reduce over non-batch dims" operations need
  explicit axis treatment.
- Component configs often omit fields that source classes default. SD 1.5 VAE
  config omits `scaling_factor`, and scheduler `prediction_type` appears
  null/omitted while PNDM defaults to epsilon. The prompt should explicitly ask
  reports to reconcile each component config against its component class
  defaults, not just model config defaults in the abstract.
- Pipeline constructors can mutate stale component configs for compatibility.
  SD pipeline repairs `steps_offset` and `clip_sample` for old scheduler
  configs. The prompt should ask reports to record constructor-time config
  compatibility repairs because they affect artifact loading parity.
- Diffusers model files expose many optional branches that are inactive for a
  target pipeline. The prompt should require an "active vs available branches"
  note so reports do not inflate first-slice op scope with ControlNet,
  IP-Adapter, GLIGEN, class embeddings, or addition embeddings when the selected
  component config disables them.
- Scheduler choice is a strategic integration decision. Exact SD 1.5 default is
  PNDM, but DDIM/Euler may be a simpler first compiled scheduler. The prompt
  should ask reports to name both "source default scheduler" and "recommended
  first Dinoml scheduler slice" when they differ.
- Upstream source comments can contain layout/backend caveats; for this target,
  upsample paths mention PyTorch `upsample_nearest_nhwc` limitations. Future
  reports should capture such caveats as evidence, without treating PyTorch
  limitations as Dinoml limitations.
- The first report over-centered one checkpoint's PNDM scheduler. Stable
  Diffusion 1.x pipelines accept broad compatible scheduler swaps, while other
  families such as Flux are flow-scheduler shaped. Future reports should list
  supported scheduler families and avoid extrapolating from a single default.
- The report treated AutoencoderKL mostly as decode-only. Family reports should
  still record autoencoder encode because img2img/inpaint/control workflows need
  it, while detailed codec optimization belongs in separate autoencoder reports.
- The report initially treated ControlNet, IP-Adapter, LoRA/textual inversion,
  GLIGEN, and img2img/inpaint/depth/upscale variants like skip/defer items.
  That is too weak for prompt calibration. They should be separate candidate
  reports with class/file anchors and a short variant-delta summary in the base
  family report.
- XLA/NPU/MPS/Flax/ONNX-specific branches are not useful for the current CUDA
  Dinoml audit unless they change shared CPU/CUDA source structure. The same
  applies to multi-GPU/context-parallel paths, callback mutation/interactive
  interrupt, safety checker/NSFW filtering, and training/loss/dropout/gradient
  checkpointing paths.
- FlashAttention support in Diffusers is target-specific and constraint-driven.
  Reports should neither assume it is usable nor assume it is impossible just
  because the current source does not select it. They should check the exact
  `attention_processor.py` or `attention_dispatch.py` path, use the eager/native
  path for parity, and then state whether a Dinoml flash-style provider could be
  valid under stricter preconditions.

## Prompt refinements to apply before scaling

1. Require per-component default reconciliation.

   Suggested prompt addition:

   ```text
   If any component config omits fields that the current Diffusers class supplies
   by default, list the omitted fields and effective defaults per component.
   Include source defaults from model, scheduler, processor, and VAE classes.
   ```

2. Require pipeline constructor compatibility notes.

   Suggested prompt addition:

   ```text
   Record constructor-time compatibility repairs or config mutations in pipeline
   `__init__`, such as scheduler `steps_offset`/`clip_sample` rewrites, because
   artifact loading parity may depend on them.
   ```

3. Add active-vs-available branch discipline.

   Suggested prompt addition:

   ```text
   For broad Diffusers classes, separate branches active under the selected
   component config from branches merely available in source. Do not count
   inactive ControlNet/IP-Adapter/adapter/class/addition-embedding paths as
   required first-slice ops unless the target pipeline enables them.
   ```

4. Ask for source-default scheduler and first-slice scheduler separately.

   Suggested prompt addition:

   ```text
   State the source default scheduler for the checkpoint and the recommended
   first Dinoml scheduler slice. If they differ, explain the parity and staging
   tradeoff.
   ```

5. Capture upstream backend/layout caveats.

   Suggested prompt addition:

   ```text
   Note source comments or backend guards that mention layout, dtype, or backend
   limitations. Treat them as evidence for validation priorities, not automatic
   Dinoml limitations.
   ```

6. Ignore non-target backend branches.

   Suggested prompt addition:

   ```text
   Ignore XLA, NPU, MPS, Flax, and ONNX-specific code paths unless the selected
   target explicitly asks for them. Mention only when such a branch changes
   shared source structure or masks CPU/CUDA behavior.
   ```

7. Require scheduler family support, not only default scheduler.

   Suggested prompt addition:

   ```text
   If a pipeline can load multiple scheduler families, list the supported set
   and identify which schedulers are required for first parity. Do not infer the
   whole pipeline contract from a single checkpoint default.
   ```

8. Require extension and variant surfaces as separate review candidates.

   Suggested prompt addition:

   ```text
   For LoRA/textual inversion/runtime adapter mutation, IP-Adapter, ControlNet,
   T2I-Adapter, GLIGEN, img2img, inpaint, depth2img, and upscaling, state
   whether the target family supports them, list relevant classes/files, and add
   them as separate candidate reports rather than hiding them in skip/defer.
   ```

9. Require autoencoder encode/decode accounting.

   Suggested prompt addition:

   ```text
   For pipelines that use an autoencoder, record encode and decode tensor
   contracts. Prefer separate autoencoder-family reports for codec-specific
   optimization candidates.
   ```

10. Require attention backend support checks.

   Suggested prompt addition:

   ```text
   Do not assume FlashAttention support is either available or unavailable.
   Record the exact processor/backend dispatch path, flash/native/xFormers/flex
   support, required masks, joint attention, added K/V, QK norm, RoPE, varlen,
   and dtype constraints. If Diffusers does not currently use flash for a
   target, still identify whether a Dinoml flash-style provider could be valid
   under stricter preconditions.
   ```

11. Split separate candidates from ignored/out-of-scope paths.

   Suggested prompt addition:

   ```text
   In the final scope section, split related candidate reports from genuinely
   ignored audit scope. Ignore multi-GPU/context parallel paths, callback
   mutation and interactive interrupt, XLA/NPU/MPS/Flax/ONNX pipeline variants,
   safety checker/NSFW filtering, and training/loss/dropout/gradient
   checkpointing unless explicitly selected.
   ```

12. Codec reports need explicit latent boundary formulas.

   Suggested prompt addition:

   ```text
   For autoencoders/codecs, record both internal model tensors and the pipeline
   boundary transforms around them: scaling_factor, shift_factor, latents_mean,
   latents_std, force_upcast policy, encode sample-vs-mode choice, and any
   pipeline-level packing/unpacking such as Flux 2x2 latent packing.
   ```

13. Autoencoder variants should be candidate reports, not one blob.

   Suggested prompt addition:

   ```text
   For an AutoencoderKL report, list AutoencoderTiny, AsymmetricAutoencoderKL,
   ConsistencyDecoderVAE, video autoencoders, audio autoencoders, and tiling/
   slicing runtime policy as separate candidate reports when present.
   ```

14. Use local config cache and retry gated Hugging Face configs with authenticated tools.

   Suggested prompt addition:

   ```text
   Check `H:/configs/<namespace>/<repo>/` for cached `model_index.json` and
   component configs before fetching. If unauthenticated HTTP returns 401/403
   for official Hugging Face component configs, retry with authenticated
   `hf download` or `huggingface_hub` before marking the config unavailable.
   Save useful fetched configs under `H:/configs` and record which access path
   succeeded.
   ```

15. Distinguish embedded guidance from true CFG.

   Suggested prompt addition:

   ```text
   For pipelines with both model guidance embeddings and classifier-free
   guidance, report them separately. Embedded guidance is a model conditioning
   tensor; true CFG usually implies an additional positive/negative denoiser
   call and explicit CFG arithmetic.
   ```

16. Treat latent packing as a pipeline contract.

   Suggested prompt addition:

   ```text
   For packed-latent transformer families such as Flux, document both the VAE
   latent map shape and the packed transformer token shape, including exact
   view/permute/reshape order and how this interacts with NHWC/NCHW layout
   optimization.
   ```

17. Require multi-encoder conditioning composition.

   Suggested prompt addition:

   ```text
   For pipelines with multiple text encoders, document feature-axis concat,
   sequence-axis concat, padding, pooled projection concat, optional missing
   encoder behavior, and whether zero-filled embeddings preserve the transformer
   contract.
   ```

18. Separate patchify/unpatchify from latent packing.

   Suggested prompt addition:

   ```text
   Distinguish model-internal patchify/unpatchify contracts from pipeline-level
   latent packing. Record exact patch axes, reshape/einsum/permute order, and
   no-layout-translation requirements.
   ```

19. Compare same-family configs before generalizing.

   Suggested prompt addition:

   ```text
   Inspect multiple configs inside a family when available. Check whether
   checkpoints sharing a pipeline class differ in depth, width, QK norm,
   dual-attention layers, latent channel count, scheduler settings, or guidance
   defaults.
   ```
