# GEMM and CUTLASS Port Plan

The v2 GEMM path must not reintroduce a naive `matmul`. GEMM/BMM lands through
library-backed external kernel families with candidate generation, profiling,
and a reusable support-library cache.

## First Families

Start with the two base dense layouts from v1:

- `gemm_rcr`: A row-major, B column-major, C row-major.
- `gemm_rrr`: A row-major, B row-major, C row-major.

The v2 scaffold records these as CUTLASS-backed external kernel families in
`dinoml.kernels.external`, and `dinoml.backends.cutlass` now generates a cached
`libdinoml_cutlass_gemm.so` containing real `float32`, `float16`, and
`bfloat16` CUTLASS launchers and profiler entrypoints for both families. Each
family declares:

- provider: `cutlass`
- required libraries: `cutlass`, `cublaslt`
- generated launcher symbol
- profiler symbol
- layout attrs
- epilogue family

cuBLASLt is not the product path for fused GEMM, but it is useful as a
correctness and performance reference while CUTLASS candidate generation is
being wired.

## v1 Concepts To Preserve

- Candidate enumeration is separate from model wrapper code.
- Profilers are generated and compiled once per unique op/config family.
- Profile cache keys include target arch, dtype/layout, op version, compiler
  version, and candidate config.
- Epilogues are part of the GEMM family signature, not post-hoc elementwise
  graph nodes when a fused library epilogue exists.
- Generated model code calls a named launcher with pointers, strides, problem
  sizes, and selected candidate id.

## Implemented v2 Integration Points

Current support-library artifacts already emit:

```text
kernel_manifest.json
kernel_codegen_plan.json
```

The CUTLASS slice adds:

- `discover_cuda_libraries()` for CUDA/CUB/cuBLASLt/cuDNN/CUTLASS availability.
- `build_external_kernel_plan()` for CUDA external kernel family metadata.
- `ensure_cutlass_gemm_support_lib()` to generate and compile
  `libdinoml_cutlass_gemm.so` once per CUDA arch/cache key.
- Exported launcher/profiler symbols use long dtype names and CUTLASS
  candidate ids, for example
  `dinoml_cutlass_gemm_rrr_float16_tensorop_sm80_16816_256x128x32_s3_w4x2x1_f32_align8` and
  `dinoml_profile_cutlass_gemm_rrr_float16_tensorop_sm80_16816_256x128x32_s3_w4x2x1_f16_align8`.

The runtime GEMM port now wires model lowering into that support library:

1. `gemm_rcr`/`gemm_rrr`, bias epilogue ops
   `gemm_rcr_bias`/`gemm_rrr_bias`, ReLU epilogue ops
   `gemm_rcr_bias_relu`/`gemm_rrr_bias_relu`, v1-style activation epilogue ops
   `*_bias_{gelu,fast_gelu,sigmoid,tanh,swish,hardswish}`, and the first
   residual epilogues `gemm_{rcr,rrr}_bias_{add,add_add,mul,mul_add}` plus
   `gemm_rcr_bias_{add_relu,add_add_relu,mul_tanh,sigmoid_mul,sigmoid_mul_tanh}` are
   explicit frontend ops for `float32`, `float16`, and `bfloat16`, not a generic
   `matmul`; they preserve dynamic `M/N` shape metadata while requiring rank-2
   matrix tensors and compatible max-shape `K`.
2. The kernel manifest records `cutlass_gemm` as an external support library
   with real launcher/profiler symbols.
3. Generated CUDA model wrappers link `libdinoml_cutlass_gemm.so` and call the
   cached launcher with runtime `M/N/K`, so smaller runtime `M/N` values use the
   same max-shape artifact.
4. CPU has reference execution only; compiled CPU GEMM still rejects until a
   real CPU library path exists.

`dinoml profile <artifact>` now executes exported profiler symbols for every
manifest CUTLASS candidate, writes `debug/profile_report.json`, writes the first
profile-selected `debug/execution_plan.json`, and caches results under the
support-library cache. The execution plan chooses the lowest elapsed-time
candidate per profiled node/shape and emits a static overlay only when all
profiled shapes for an op/dtype/candidate-set agree on the same winner. When
GEMM input `shape_spec` contains explicit `Dim.buckets` and no runtime override
is supplied, profiling now expands those buckets into concrete workload cases
and carries `shape.case_id`, dynamic dim values, and dim sources into the report
and execution plan. Profiling also honors optional dense layout element
alignment metadata on GEMM A/B tensors: when both operands are annotated,
candidate workloads whose CUTLASS `align` exceeds the smaller A/B alignment are
pruned before timing. v2 also mirrors v1's shape-derived A/B alignment rule:
`gemm_rrr` caps candidate alignment by `gcd(K, N)`, `gemm_rcr` caps it by `K`,
manifest defaults use static dimensions or dynamic `Dim.divisible_by`, and
profiling workloads use each concrete bucket, override, or max-shape case.
Kernel manifests preserve this all-runtime `cutlass_alignment_cap`, and
execution-plan overlays are rejected in strict mode when they try to install a
profiled candidate whose CUTLASS alignment exceeds that cap.
Generated CUDA modules also check the selected candidate's A/B pointer
byte-alignment requirement before calling the CUTLASS launcher, while the common
runtime support path applies ABI byte offsets to logical tensor pointers,
validates offset-adjusted byte capacity, and still requires contiguous row-major
strides when stride metadata is supplied. Profile results, cache keys,
execution-plan selections, and static overlays preserve `split_k` plus
`workspace_nbytes` as launch/result metadata. Base and bias/activation
`device::Gemm` candidates now advertise
v1-style split-K search metadata, the profiler expands split-K values using the
v1 `K // max(M, N)` heuristic, and generated CUDA uses companion split-K
launcher/profiler symbols plus a session-owned CUTLASS workspace when an
execution plan selects `split_k > 1`. Residual/broadcast epilogue families still
profile and launch with `split_k=1` until their CUTLASS broadcast path has the
same workspace ABI coverage. The report/cache key records a
best-effort CUDA hardware/toolchain fingerprint, support-library source/binary
hashes, support-build provenance, and the candidate set/config keys. CUTLASS
support manifests also record compile flags, NVCC version output, dependency header
hashes, and a provenance key that participates in support-cache reuse. The
support cache writes a
`dinoml.support_source_manifest` at `src/source_manifest.json` beside the rendered
CUTLASS source; that manifest maps source files to the used candidate set keys,
candidate config keys, launcher/profiler symbols, and support build units so
future generated candidates can be inspected without embedding generated source
in model artifacts. The support source is currently rendered from a checked-in
static source file and pruned to only the launcher/profiler symbols required by
the manifest candidate plan. The first epilogue slice uses a structured GEMM descriptor split:
`dinoml.kernels.families.gemm` owns layout/shape/epilogue contracts and
`dinoml.kernels.providers.cutlass.gemm` owns CUTLASS symbol/candidate metadata.
GEMM candidate sets now mirror the v1 SM80 TensorOp 16816 tile list for
`float16`/`bfloat16`, including alignment variants and `float16` versus
`float32` accumulation choices where v1 generated both. Default `float32`
candidates now total 221 variants: 57 regular TF32 TensorOp, 57
`multiply_add_fast_f16` TensorOp, 57 `multiply_add_fast_bf16` TensorOp, 39
3xTF32 `multiply_add_fast_f32` TensorOp, and 11 exact f32 SIMT fallback
candidates. All TensorOp float32 families are optional; `no_tf32=True` filters
them out and leaves only the 11 exact f32 SIMT candidates. Each candidate
records tile, stage count, warp count, alignment, math mode, optional status,
math operator, and accumulator dtype so profiling can distinguish real kernel
variants. Generated support source is pruned from the macro-backed checked-in
source to only the launcher/profiler symbols in the used candidate plan, and
activation/residual epilogue exports instantiate CUTLASS thread epilogue
functors directly with the selected candidate accumulator type. Candidate policy
aliases carry and apply the selected alignment and math operator, including
`OpMultiplyAddFastF16`, `OpMultiplyAddFastBF16`, and `OpMultiplyAddFastF32` for
the optional fast TensorOp float32 candidates. Target policy now participates in
the per-artifact manifest: `Target(use_fp16_acc=True)` selects only
fp16-accumulation fp16 launchers/profilers and changes the support/profile cache
keys. `Target(no_tf32=True)` selects the SIMT f32 fallback launchers/profilers
and changes those keys too. The logical GEMM shape contract now accepts
`A[..., K]` with rank-2 `B`, preserves `C[..., N]`, and flattens the leading
`A` dimensions into the CUTLASS `m` argument. The first folded residual coverage
is wired for RCR epilogues
`gemm_rcr_bias_{add,mul,add_add,mul_add,add_add_relu,mul_tanh,sigmoid_mul,sigmoid_mul_tanh}`,
including frontend shape metadata, CPU reference execution, CUDA lowering checks,
support-library runtime smoke coverage, and profiler workload shapes.

`dml.compile(..., execution_plan=...)` and `dinoml compile --execution-plan`
now consume the static overlay from a profile-selected execution plan before
writing `kernel_manifest.json`, `kernel_codegen_plan.json`, or generated CUDA
source. `dml.compile(..., profile=True)` and `dinoml compile --profile` provide
the first opt-in closed loop: build the candidate artifact, profile it, load the
generated execution plan, and rebuild with the plan applied. The bootstrap timing
report is preserved as `debug/bootstrap_profile_report.json` on the final
artifact. For profiled dynamic shapes whose buckets choose different candidates
or split-K values, the manifest now carries guarded per-node dispatch selections:
generated CUDA checks profiled `M/N/K` cases, calls the selected candidate
symbol, sizes a shared CUTLASS workspace for split-K dispatches, and falls back
to the safe manifest default when no guard matches. Next steps should prioritize
richer tensor-accessor offset/layout alignment filtering and extending split-K
coverage to residual/broadcast epilogues once the CUTLASS broadcast path is proven. Broader
broadcast/folded-M arithmetic epilogues, `elup1`, v1 `dual_gemm`/dual-output
GEMM families, beyond-v1 CUTLASS epilogues where CUTLASS gives useful fused
functionality, BMM and grouped GEMM parity should wait behind that profiling
loop so v2 does not accumulate more declared surface area without v1-grade
selection behavior.

## Dependency Discovery

`dinoml.backends.cuda_libraries.discover_cuda_libraries()` records available
CUDA-side dependencies:

- CUDA toolkit
- CUB
- cuBLASLt
- cuDNN
- CUTLASS

CUTLASS is header-only. v2 looks at `DINOML_CUTLASS_ROOT`, `CUTLASS_ROOT`,
`/workspace/dinoml_v2/third_party/cutlass`, `/workspace/dinoml_v2/3rdparty/cutlass`,
and finally the v1 checkout at `/workspace/dinoml/3rdparty/cutlass`. The v2 repo
now carries CUTLASS as a submodule under `third_party/cutlass`; initialize it
with:

```sh
git submodule update --init --recursive
```

Support builds enable NVCC `--split-compile=8` when the local compiler supports
it. Set `DINOML_NVCC_SPLIT_COMPILE=1` to disable that flag, or set it to another
positive integer to tune the number of device optimization workers.

## Remaining Near-Term Non-Goals

- No naive C++/CUDA matmul.
- No full epilogue visitor port beyond the bias, activation, and first rank-2
  residual fused CUTLASS slices.
- No grouped GEMM.
- No convolution implicit-GEMM yet.
- No public `dml.ops.matmul` until CUDA and CPU/reference behavior are both
  represented in the IR/runtime contract.
