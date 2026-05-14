# Diffusers Scheduler Matrix Report

Candidate slug: `scheduler_matrix`

## 1. Source basis

```text
Diffusers commit/version:
  Local checkout X:/H/diffusers at b3a515080752a3ba7ca92161e25530c7f280f629.

Model / pipeline reports used as staging context:
  agents/plans/diffusers/stable_diffusion_1_5/report.md
  agents/plans/diffusers/stable_diffusion_2_x/report.md
  agents/plans/diffusers/stable_diffusion_xl/report.md
  agents/plans/diffusers/stable_diffusion_3/report.md
  agents/plans/diffusers/flux/report.md
  agents/plans/diffusers/wan/report.md
  agents/plans/diffusers/cogvideox/report.md

Scheduler source files inspected:
  X:/H/diffusers/src/diffusers/schedulers/scheduling_ddim.py
  X:/H/diffusers/src/diffusers/schedulers/scheduling_ddpm.py
  X:/H/diffusers/src/diffusers/schedulers/scheduling_pndm.py
  X:/H/diffusers/src/diffusers/schedulers/scheduling_euler_discrete.py
  X:/H/diffusers/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py
  X:/H/diffusers/src/diffusers/schedulers/scheduling_unipc_multistep.py
  X:/H/diffusers/src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py
  X:/H/diffusers/src/diffusers/schedulers/scheduling_ddim_cogvideox.py
  X:/H/diffusers/src/diffusers/schedulers/scheduling_dpm_cogvideox.py
  X:/H/diffusers/src/diffusers/schedulers/scheduling_utils.py

Pipeline call sites inspected:
  Stable Diffusion / SDXL / SD3 / Flux / Wan / CogVideoX pipeline folders,
  focused on retrieve_timesteps, set_timesteps, custom timesteps/sigmas,
  scale_model_input, add_noise, init_noise_sigma, guidance, and scheduler.step.

Config sources inspected:
  H:/configs scheduler_config.json files for SD2, SDXL, SDXL Turbo, SD3/3.5,
  FLUX.1-dev/schnell, x4 upscaler main and low_res schedulers, plus cached
  model_index scheduler metadata where component scheduler configs were absent.

Ignored by scope:
  XLA, NPU, MPS, Flax, ONNX, training, callbacks/interrupt mutation, and
  scheduler classes not exercised by the selected first pipeline reports except
  where they remain compatible separate candidates.
```

## 2. Scheduler family matrix

| Family | Source class | Required by reports | Core schedule input | Prediction types | State/history | First Dinoml status |
| --- | --- | --- | --- | --- | --- | --- |
| DDIM alpha-product | `DDIMScheduler` | SD1/SD2 swaps, SD2 v-pred configs, SD x4 upscaler | `num_inference_steps`; no custom timesteps in inspected source | `epsilon`, `sample`, `v_prediction` | Stateless per step after timesteps; optional eta noise | First alpha-product slice for SD2 v-pred and x4 upscaler |
| DDPM alpha-product noising | `DDPMScheduler` | SD x4 low-res noising; training-like add-noise utility | `num_inference_steps` or descending `custom_timesteps` | `epsilon`, `sample`, `v_prediction`; optional learned variance output split | Stateless for add_noise; stochastic reverse step if used | Add-noise-only first, reverse step later |
| PNDM PRK/PLMS | `PNDMScheduler` | SD1 default/common, SD2 base/depth/inpaint sampled defaults | `num_inference_steps` | `epsilon`, `v_prediction` | `counter`, `ets`, `cur_model_output`, PRK/PLMS timetable | Separate first SD classic follow-up after DDIM |
| Euler discrete Karras-compatible | `EulerDiscreteScheduler` | SDXL base/refiner defaults; SD1/SD2/SDXL swaps | `num_inference_steps`, custom `timesteps`, or custom `sigmas` | `epsilon`, `sample`/`original_sample`, `v_prediction` | `step_index`, `begin_index`; optional stochastic churn branch | First SDXL parity slice |
| DPM-Solver multistep | `DPMSolverMultistepScheduler` | SD1/SD2/SDXL compatible swap, fast sampler surface | `num_inference_steps` or `custom_timesteps`; optional Karras/exponential/beta/flow sigma modes | `epsilon`, `sample`, `v_prediction`, `flow_prediction` depending algorithm | `step_index`, `begin_index`, `model_outputs`, `lower_order_nums` | Separate candidate; multistep state and algorithm variants are broad |
| UniPC multistep | `UniPCMultistepScheduler` | Wan official configs; SD compatible swap | `num_inference_steps`; optional custom `sigmas` only when `use_flow_sigmas` | `epsilon`, `sample`, `v_prediction`, `flow_prediction` | `step_index`, `begin_index`, `model_outputs`, `timestep_list`, `last_sample`, `lower_order_nums`, corrector state | Wan first scheduler slice, then SD-compatible candidate |
| FlowMatch Euler | `FlowMatchEulerDiscreteScheduler` | SD3/3.5, Flux, Flux variants; source annotation in Wan | `num_inference_steps`, custom `sigmas`, custom `timesteps`, optional `mu` | Flow derivative by contract; no model-output conversion branch in step | `step_index`, `begin_index`; optional per-token timesteps and stochastic sampling | First SD3/Flux flow slice |
| CogVideoX DDIM | `CogVideoXDDIMScheduler` | CogVideoX default sampled configs | `num_inference_steps` or custom timesteps/sigmas through copied retrieve helper | `epsilon`, `sample`, `v_prediction` | Stateless per step | CogVideoX first video scheduler slice |
| CogVideoX DPM | `CogVideoXDPMScheduler` | CogVideoX optional/video2video examples | Custom `timestep_back` plus scheduler timesteps | `epsilon`, `sample`, `v_prediction` | Carries caller-visible `old_pred_original_sample`; stochastic noise | Separate CogVideoX scheduler candidate |

The Karras-compatible enum in `scheduling_utils.py` is broader than the focused
source list: DDIM, DDPM, PNDM, LMS, Euler, Heun, Euler Ancestral, DPM-Solver
single/multistep/SDE, KDPM2, DEIS, UniPC, and EDMEuler. Stable Diffusion and
SDXL pipeline constructors use `KarrasDiffusionSchedulers`, so do not overfit
classic SD1 to PNDM. The first Dinoml matrix should still stage a small subset
with clear fallbacks rather than treating the entire enum as implemented.

## 3. Pipeline family needs

| Pipeline family | Source scheduler contract | Sampled/default configs | First parity scheduler | Follow-up scheduler surface |
| --- | --- | --- | --- | --- |
| Stable Diffusion 1.x | `KarrasDiffusionSchedulers`; copied `retrieve_timesteps` supports custom timesteps/sigmas when the scheduler accepts them | Common default PNDM epsilon, but pipelines allow DDIM/Euler/DPM swaps | PNDM epsilon for default parity or DDIM epsilon for simpler stateless slice; do both before claiming broad SD1 | Euler/Euler Ancestral, DPM-Solver multistep, LMS/Heun/KDPM/DEIS |
| Stable Diffusion 2.x | Same Karras-compatible pipeline surface | SD2 768/2.1 configs commonly DDIM v-pred; base/depth/inpaint PNDM epsilon | DDIM v-pred, then PNDM epsilon/v-pred | Euler and DPM-Solver swaps; x4 upscaler low-res DDPM add_noise |
| Stable Diffusion XL | `KarrasDiffusionSchedulers`; base/refiner code uses `scale_model_input` then `step` | Base/refiner EulerDiscrete epsilon leading; Turbo Euler Ancestral epsilon trailing | EulerDiscrete epsilon | Euler Ancestral for Turbo, DDIM/PNDM/DPM-compatible swaps |
| Stable Diffusion 3 / 3.5 | Constructor type is `FlowMatchEulerDiscreteScheduler`; `retrieve_timesteps` accepts sigmas | FlowMatch Euler shift 3.0 | FlowMatch Euler static shift | Custom sigmas/timesteps, skip-layer guidance loop integration |
| Flux | Constructor type is `FlowMatchEulerDiscreteScheduler`; pipeline may synthesize sigmas and `mu` for dynamic shift | Schnell shift 1.0 static; dev shift 3.0 dynamic | FlowMatch Euler with custom sigmas, static shifting first | Dynamic `mu`, true CFG second call, per-token/stochastic flow options as separate |
| Wan | Pipeline annotations mention FlowMatch Euler, but docs/configs and reports show official UniPC flow configs | UniPC flow-prediction, `use_flow_sigmas=true`, flow shift 3 or 5, solver order 2 | UniPC flow-prediction `bh2` with flow sigmas | Dual-transformer boundary dispatch, `expand_timesteps`, FlowMatch source-annotation compatibility |
| CogVideoX | Explicit `CogVideoXDDIMScheduler | CogVideoXDPMScheduler` in base/I2V/video2video | DDIM v-pred trailing with SNR shift; DPM optional | CogVideoX DDIM v-pred | CogVideoX DPM with `old_pred_original_sample`, stochastic noise, video2video `add_noise` |

## 4. Common loop contract

Most image and video pipelines share this host-visible shape:

```text
scheduler.set_timesteps(...)
latents = init_noise * scheduler.init_noise_sigma or add_noise(init_latents, noise, timestep)
for t in scheduler.timesteps:
  model_input = CFG concat or separate positive/negative call
  model_input = scheduler.scale_model_input(model_input, t)
  model_output = denoiser(model_input, t, conditioning)
  model_output = CFG / guidance_rescale / dynamic guidance arithmetic
  latents = scheduler.step(model_output, t, latents, ...)
```

Keep `set_timesteps`, timestep slicing by `strength` or denoising ranges,
custom schedule validation, and multistep history ownership in host-visible
runtime state first. Compile/fuse `scale_model_input`, CFG arithmetic, noising,
and one scheduler `step` only after the scheduler family and prediction type are
explicit in the artifact.

## 5. Prediction type conversions

| Conversion target | Families | Formula shape to preserve |
| --- | --- | --- |
| Epsilon to original sample | DDIM/DDPM/DPM/UniPC | `x0 = (sample - sqrt(beta) * eps) / sqrt(alpha)` or sigma-form `(sample - sigma_t * eps) / alpha_t` |
| Sample/original_sample | DDIM/DDPM/Euler/DPM/UniPC | Model output is already `x0` / denoised sample; Euler keeps legacy `original_sample` alias |
| V-prediction to original sample | DDIM/DDPM/CogVideoX DDIM/DPM | `x0 = sqrt(alpha) * sample - sqrt(beta) * v` |
| V-prediction to epsilon | DDIM/PNDM/DPM/UniPC | `eps = sqrt(alpha) * v + sqrt(beta) * sample` or sigma-form `alpha_t * v + sigma_t * sample` |
| Flow prediction | FlowMatch Euler, UniPC flow, DPM-Solver flow | FlowMatch Euler treats model output as derivative and updates `sample + dt * model_output`; UniPC/DPM flow conversion uses `x0 = sample - sigma * model_output` when solving in data-prediction form |
| Predicted variance split | DDPM/DPM-Solver epsilon branches | If model output has double channels and variance type is learned/learned_range, mean and variance parts split before conversion |
| Thresholding/clip sample | DDIM/DDPM/DPM/UniPC | Optional dynamic threshold or clamp mutates predicted original sample before final update; first slices can disable except where checkpoint config requires parity |

The first runtime schema should name both the scheduler family and the
model-output semantic target. `prediction_type` alone is insufficient for
FlowMatch because FlowMatch Euler has no alpha-product conversion branch, while
UniPC/DPM flow modes convert the same string through multistep solver math.

## 6. State and history requirements

| State | Schedulers | Artifact-visible fields |
| --- | --- | --- |
| Timesteps and sigmas tables | All inspected schedulers | Device/dtype behavior; CPU-resident sigma tables in Euler/DPM/UniPC source; optional terminal sigma |
| Step index / begin index | Euler, FlowMatch, DPM-Solver, UniPC | `_step_index`, `_begin_index`, duplicate timestep lookup rule, `set_begin_index` for img2img/add_noise paths |
| Model-output history | PNDM, DPM-Solver, UniPC | PNDM `ets`; DPM/UniPC ring buffer sized by `solver_order` |
| Lower-order warmup/final fallback | DPM-Solver, UniPC | `lower_order_nums`, `lower_order_final`, `euler_at_final`, solver order chosen per step |
| Corrector state | UniPC | `last_sample`, `this_order`, optional disabled corrector indices |
| PRK/PLMS state | PNDM | `prk_timesteps`, `plms_timesteps`, `counter`, `cur_model_output`, skip PRK behavior |
| Old prediction sample | CogVideoX DPM | Caller-visible `old_pred_original_sample` carried between steps |
| Random noise source | DDIM eta, DDPM, Euler churn, DPM-SDE, Flow stochastic, CogVideoX DPM | Generator/variance noise inputs, `s_noise`, eta, stochastic flags |
| Per-token timesteps | FlowMatch Euler, Wan TI2V loop | `per_token_timesteps` and token-shaped sigma lookup; separate from scalar timestep |

## 7. Timestep and sigma inputs

The copied `retrieve_timesteps` helper used by SD, SD3, Flux, CogVideoX, and
some variants checks scheduler signatures before passing `timesteps` or
`sigmas`. This means Dinoml should model schedule admission, not accept every
field for every scheduler.

| Scheduler | `num_inference_steps` | custom timesteps | custom sigmas | Extra schedule knobs |
| --- | --- | --- | --- | --- |
| DDIM | Yes | Not in focused source | No | `steps_offset`, `timestep_spacing` in config |
| DDPM | Yes | `custom_timesteps` descending | No | variance type; add_noise lookup |
| PNDM | Yes | No | No | `skip_prk_steps`, `set_alpha_to_one`, `steps_offset` |
| EulerDiscrete | Yes | Yes unless Karras/exponential/beta/continuous-v guards reject | Yes | Karras/exponential/beta conversion, final sigma type, `timestep_spacing` |
| DPM-Solver multistep | Yes | `custom_timesteps` | Not in focused source interface | Karras/exponential/beta/flow modes, algorithm type, solver order |
| UniPC | Yes | No | Yes only with `use_flow_sigmas=True` | `mu` for dynamic shift, flow shift, Karras/exponential/beta modes |
| FlowMatch Euler | Yes | Yes | Yes | `mu`, shift terminal, invert sigmas, Karras/exponential/beta, stochastic, per-token |
| CogVideoX DDIM/DPM | Yes | Pipeline helper may pass custom timesteps/sigmas if accepted | Pipeline helper checks signature | SNR shift scale and video-specific DPM back timestep |

## 8. First parity slices

1. **DDIM v-pred alpha-product slice** for SD2 and SD x4 main scheduler.
   Keep `eta=0`, no variance noise, no thresholding unless a selected config
   needs it. Validate `set_timesteps`, `add_noise`, and one `step`.

2. **EulerDiscrete epsilon slice** for SDXL base/refiner.
   Cover `scale_model_input = sample / sqrt(sigma^2 + 1)`,
   `pred_original_sample = sample - sigma * eps`, and Euler derivative update.
   Defer churn, Karras/exponential/beta conversions, and ancestral variants.

3. **FlowMatch Euler static-shift slice** for SD3 and Flux schnell.
   Cover default or custom sigmas, static `shift`, and
   `prev = sample + (sigma_next - sigma) * model_output`.

4. **PNDM epsilon/v-pred slice** for SD1 default and SD2 base/depth/inpaint.
   Make PRK/PLMS history explicit. Start with `skip_prk_steps=true` if matching
   a selected config, then add full PRK behavior.

5. **UniPC flow-prediction slice** for Wan.
   Fix official Wan settings first: `use_flow_sigmas=true`, `flow_shift=3.0`,
   `solver_order=2`, `solver_type="bh2"`, `predict_x0=true`,
   `final_sigmas_type="zero"`, `timestep_spacing="linspace"`.

6. **CogVideoX DDIM v-pred slice** for CogVideoX.
   Cover trailing timesteps, SNR shift config, CFG arithmetic outside the
   scheduler, and the simplified DDIM previous-sample formula.

## 9. Ops and fusion candidates

Highest priority:

- Scheduler table generation and validation as host/runtime metadata:
  timesteps, sigmas, alphas, alpha cumulative products, and per-step scalar
  coefficients should be artifact-visible and cacheable per config and step
  count.
- Fused pointwise scheduler step for DDIM v-pred, Euler epsilon, FlowMatch
  Euler, and CogVideoX DDIM. These are bandwidth-bound elementwise maps over
  latents and should compose with CFG output tensors.
- CFG arithmetic and guidance rescale fusion:
  `uncond + scale * (text - uncond)` plus optional std reductions over all
  non-batch axes.
- `scale_model_input` fusion into denoiser input staging for Euler/DPM-family
  schedulers where the scale is a scalar per step.

Medium priority:

- `add_noise` kernels for img2img/inpaint/video2video and x4 low-res DDPM:
  `sqrt_alpha * original + sqrt_beta * noise` or sigma-form
  `alpha_t * original + sigma_t * noise`, broadcasting over image/video ranks.
- Multistep state update kernels for PNDM, DPM-Solver, and UniPC history
  buffers once single-step parity is stable.
- Schedule conversion helpers for Karras/exponential/beta sigmas and
  FlowMatch dynamic shift, implemented as host table generation first.

Lower priority:

- Stochastic branches: DDIM eta, DDPM reverse variance, Euler churn,
  DPM-SDE, FlowMatch stochastic sampling, and CogVideoX DPM noise injection.
- Dynamic thresholding/clip-sample provider fusion. Keep as explicit pointwise
  and reduction ops until checkpoint coverage proves the need.
- Ancestral, LMS, Heun, KDPM, DEIS, EDMEuler, LCM/TCD, and inverse schedulers.

## 10. Validation plan

Config/table tests:

- Parse representative scheduler configs from `H:/configs` for SD2, SDXL,
  SDXL Turbo, SD3/3.5, Flux schnell/dev, SD x4 upscaler, Wan, and CogVideoX.
- For each first slice, compare `set_timesteps` outputs, terminal sigmas,
  dtype/device placement, and `init_noise_sigma` against Diffusers.
- Add admission tests proving invalid custom timetable/sigma combinations fail
  with scheduler-specific reasons.

One-step numeric parity:

- DDIM epsilon/sample/v-pred on fixed NCHW latents with `eta=0`.
- DDPMScheduler `add_noise` for image and video-rank tensors; reverse DDPM only
  after variance branches are admitted.
- PNDM epsilon and v-pred with controlled `ets`/counter progression.
- EulerDiscrete epsilon with and without `scale_model_input` pre-call covered.
- DPM-Solver and UniPC conversion unit tests before full multistep loop tests.
- FlowMatch Euler static and dynamic shifting, custom sigmas, and per-token
  timestep branch as separate cases.
- CogVideoX DDIM v-pred and CogVideoX DPM old-pred state handoff.

Loop parity:

- One denoising step per family with compiled denoiser output stubbed or fixed.
- Short deterministic loops with scheduler in host control and latents compared
  to Diffusers after each step.
- Img2img/inpaint/video2video strength slicing and `set_begin_index` behavior
  before claiming variant compatibility.

Suggested tolerances: fp32 `rtol=1e-5, atol=1e-6` for pure scheduler arithmetic;
fp16/bf16 initially `rtol=2e-2, atol=2e-2`, then tighten per fused kernel.

## 11. Separate scheduler candidates

Keep these as separate candidate reports or work items:

- `scheduler_euler_ancestral_turbo`: required for SDXL Turbo and stochastic
  ancestral update semantics; do not fold into EulerDiscrete first slice.
- `scheduler_dpmsolver_matrix`: DPM-Solver multistep, singlestep, SDE, Karras,
  exponential/beta/flow modes, algorithm type differences, solver order, and
  variance noise.
- `scheduler_unipc_sd_compat`: UniPC outside Wan, including non-flow SD
  predictions, solver-p/corrector options, and Karras-compatible swaps.
- `scheduler_karras_remaining`: LMS, Heun, KDPM2, DEIS, EDMEuler, and related
  enum members not deeply inspected here.
- `scheduler_flowmatch_advanced`: FlowMatch Karras/exponential/beta sigmas,
  dynamic shifting, shift terminal, invert sigmas, stochastic sampling, and
  per-token timesteps.
- `scheduler_cogvideox_dpm`: CogVideoX DPM with previous original-sample state,
  stochastic noise, `timestep_back`, and video2video interaction.
- `scheduler_lcm_tcd_distilled`: LCM/TCD/consistency-style schedulers for
  turbo/distilled models not covered by the focused source list.
- `scheduler_inverse_editing`: DDIM inverse and editing/inversion schedulers.

## 12. Scope boundary

This report is a staging matrix for Dinoml scheduler/runtime work. It does not
implement operators, mutate Dinoml runtime code, or update the ranked project
queue. The correct first implementation shape is explicit scheduler state in
manifests/profile reports/execution plans, not hidden Python object mutation.
Public support should be admitted per scheduler family, prediction conversion,
schedule input mode, and state/history contract.
