# GEMM and CUTLASS Port Plan

The v2 GEMM path must not reintroduce a naive `matmul`. GEMM/BMM lands through
library-backed external kernel families with candidate generation, profiling,
and a reusable support-library cache.

## First Families

Start with the two base dense layouts from v1:

- `gemm_rcr`: A row-major, B column-major, C row-major.
- `gemm_rrr`: A row-major, B row-major, C row-major.

The v2 scaffold records these as CUTLASS-backed external kernel families in
`dinoml.kernels.external`, and the CUDA backend now builds cached op/dtype
modules containing real `float32`, `float16`, and `bfloat16` CUTLASS launchers
and profiler entrypoints for both families. Each family declares:

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
- The repo CMake `dinoml_cutlass_gemm` aggregate target builds op/dtype GEMM
  static archives such as `libdinoml_cutlass_gemm_rcr_bias_float32.a` once per
  CUDA architecture cache, generating chunked op/dtype instantiation sources
  from `kernels/cuda/src/cutlass_gemm.cu` at CMake build time instead of
  carrying checked-in unit sources or rendering/pruning a per-artifact GEMM
  support source. The matching `dinoml_cutlass_bmm` targets do the same for BMM
  op/dtype archives such as `libdinoml_cutlass_bmm_rrr_float32.a`. Generated
  artifacts request only the archives referenced by their kernel manifest and
  link them into the generated `module.so`; they do not distribute CUTLASS
  GEMM/BMM support `.so` files. The targets are gated by explicit
  `DINOML_ENABLE_CUTLASS_GEMM` and `DINOML_ENABLE_CUTLASS_BMM` CMake options.
- Exported launcher/profiler symbols use long dtype names and CUTLASS
  candidate ids, for example
  `dinoml_cutlass_gemm_rrr_float16_tensorop_sm80_16816_256x128x32_s3_w4x2x1_f32_align8` and
  `dinoml_profile_cutlass_gemm_rrr_float16_tensorop_sm80_16816_256x128x32_s3_w4x2x1_f16_align8`.

The runtime GEMM port now wires model lowering into that support library:

1. `gemm_rcr`/`gemm_rrr`, bias epilogue ops
   `gemm_rcr_bias`/`gemm_rrr_bias`, ReLU epilogue ops
   `gemm_rcr_bias_relu`/`gemm_rrr_bias_relu`, v1-style activation epilogue ops
   `*_bias_{gelu,fast_gelu,sigmoid,tanh,swish,hardswish,elup1}`, and the first
   residual epilogues
   `gemm_{rcr,rrr}_bias_{add,add_add,mul,mul_add,add_relu,add_add_relu,mul_tanh,sigmoid_mul,sigmoid_mul_tanh}` are
   explicit frontend ops for `float32`, `float16`, and `bfloat16`, not a generic
   `matmul`; they preserve dynamic `M/N` shape metadata while requiring rank-2
   matrix tensors and compatible max-shape `K`.
2. The kernel manifest records `cutlass_gemm` as an external support library
   with real launcher/profiler symbols.
3. Generated CUDA model wrappers link the selected
   `libdinoml_cutlass_<op>_<dtype>.so` modules and call the cached launcher with
   runtime `M/N/K`, so smaller runtime `M/N` values use the same max-shape
   artifact.
4. CPU now has a bounded naive generated path for `gemm_rcr`,
   `gemm_rcr_bias`, `gemm_rcr_bias_fast_gelu`, `bmm_rcr`, and `bmm_rrr`:
   compiled CPU artifacts flatten `A[..., K]` into runtime `M` for GEMM, keep
   rank-3 `B x M x N` semantics for BMM, and run straightforward row-major
   loops for `float32`, `float16`, and `bfloat16`, including rank-1 or
   `[1, N]` bias for the admitted GEMM bias epilogues, the Transformers/v1
   tanh-based `fast_gelu` epilogue for `gemm_rcr_bias_fast_gelu`, the distinct
   CLIP QuickGELU `x * sigmoid(1.702 * x)` epilogue for
   `gemm_rcr_bias_quick_gelu`, zero-stride batch broadcast for both admitted
   BMM layouts, and row-major-logical `B[B, K, N]` handling for `bmm_rrr`.
   This is an explicit bridge for CLIP CPU artifacts, not a library-backed
   provider path. Other compiled CPU GEMM/BMM families still reject until a
   better CPU library/runtime path lands.

GGUF runtime dequantization before GEMM now has a bounded CUDA runtime slice for
`gemm_rrr`, `gemm_rcr`, `gemm_rrr_bias`, and `gemm_rcr_bias` with a GGUF RHS
constant declared as `materialization="dequantize_on_gpu_before_launch"` and
`residency="manual_runtime_load"`. The CUTLASS manifest records a
`gguf_runtime_dequant` plan with status `lowered_runtime_dequant_scratch`,
qtype, encoded size, logical dense shape, scratch size, and the dense CUTLASS
handoff. The manifest also exposes the shared per-session
`gguf_runtime_dequant_scratch` resource as a max-sized CUDA-device allocation
derived from all lowered runtime-dequant GEMM plans in the artifact. Generated
CUDA modules store the RHS constant as encoded bytes, allocate that
session-owned dense dequant scratch buffer, and require a reproducible direct
link against DinoML's CMake-built `libgguf_cuda_native` static archive from the
vendored libgguf CUDA sources. Generated lowering calls
`libgguf_cuda_dequantize_rows_on_stream(...)` directly; the old runtime-set
function-pointer fallback is removed.
This is intentionally not a general offload scheduler and does not cover
non-bias GEMM epilogues, `bfloat16`, or direct in-kernel quantized RHS
execution yet.

The first model-level CUDA workflow is intentionally small:
`examples/cuda_linear.py` builds a single explicit `gemm_rrr_bias` linear layer
with dense weight/bias constants and a bucketed dynamic batch dimension. Its
runtime test compiles for CUDA with `no_tf32=True`, verifies the artifact's
CUTLASS GEMM manifest/candidate metadata, overrides the constants from CUDA
torch tensors, and runs a smaller runtime batch through the real CUDA module.
This keeps the visible workflow distinct from CPU examples without forcing the
full default float32 candidate build or profile-assisted compile loop.
The same path now has a cheap profile-assisted compile regression test: it uses
the real `cuda_linear` spec and no-TF32 CUTLASS manifest, stubs only backend
build/profiler timing, emits a compact static execution plan for the validation
batch shape, and verifies the final manifest plus codegen plan consume the
profile-selected candidate.

Base BMM layout contracts have their first CUTLASS runtime slice. The public
frontend and CPU reference cover
`bmm_{ccc,ccr,crc,crr,rcc,rcr,rrc,rrr}` and matching `_add` variants with the
same v1 layout semantics: A and B `c` layouts transpose the last two logical
dimensions, and C `c` layouts return `[B, N, M]` output. `_add` accepts an
output-shaped addend or v1-style trailing-bias addend after leading `1`s are
squeezed. Compiled CPU artifacts now also have bounded naive `bmm_rcr` and
`bmm_rrr` bridges for CLIP attention: rank-3 row-major `A[B, M, K]`,
column-major-logical `B[B, N, K]` for `bmm_rcr`, row-major-logical
`B[B, K, N]` for `bmm_rrr`, row-major `C[B, M, N]`, and v1-style batch
broadcast through zero batch strides. This now covers the CLIP attention
context matmul as a compiled CPU artifact. With the matching
`gemm_rcr_bias_fast_gelu` CPU bridge, deeper CLIP text CPU artifacts now run.
With the matching bounded naive `conv2d_bias` CPU bridge for static groups=1
NCHW/OIHW `float32`/`float16`, bounded CLIP vision and full two-tower CPU
artifacts now also run against local Transformers. The base `bmm_*` ops now
register a separate `cutlass_bmm`
external library with a real batched GEMM ABI carrying batch count,
per-operand batch strides, leading dimensions, C-layout-aware output handling,
v1-style batch broadcast through zero batch strides, candidate metadata, and
alignment fallbacks. `_add` BMM variants now have a CUTLASS add epilogue ABI
for full-output `d0` tensors and v1-style trailing-bias `d0` tensors after
leading `1`s are squeezed. The launcher passes `d0` as CUTLASS source C, uses
zero source-C stride/leading-dimension for trailing bias, writes the result to
the output tensor, and profiles/selects candidates with `d0` in the epilogue
alignment context. BMM profiling workloads now feed `dinoml profile`
reports/cache through native op/dtype profilers such as
`dinoml_cutlass_bmm_profiler_bmm_rrr_float32`, which allocate each BMM problem
once, filter the candidate table by operand/epilogue alignment, run all
remaining candidates, and return candidate timing rows to Python bindings.
Reports/cache preserve batch count, batch strides, leading dimensions,
epilogue inputs, and batch-aware execution-plan shape keys.
Confident static BMM profile selections are consumed during compile by
selecting the profiled BMM candidate in the kernel manifest. Conflicting BMM
profile selections now generate guarded runtime dispatch on profiled batch/M/N/K
shapes with pointer-alignment guards and default-launch fallback.

`dinoml profile <artifact>` now executes exported profiler symbols for the
primary profile candidates, writes `debug/profile_report.json`, writes the first
profile-selected `debug/execution_plan.json`, and caches results under the
support-library cache. Profiling can repeat each workload sample and records
median/mean/min/max/stddev timing statistics; the execution plan chooses the
lowest median-time candidate per profiled node/shape, but only emits consumable
selections when the winner clears repeat-count, absolute/relative margin, and
confidence-interval thresholds over the runner-up. Low-confidence winners are
recorded as non-consumable audit metadata, and execution-plan application refuses
explicit low-confidence static or guarded payloads before they can alter the
kernel manifest. Execution-plan application also refuses stale CUTLASS
launcher/profiler symbols, malformed guarded positive-integer shape metadata,
missing or stale guarded `node_id` values, and duplicate static or guarded
selection entries for the same op/dtype/candidate-set key before those payloads
can attach generated dispatch. Static overlays are emitted
only when all profiled shapes for an op/dtype/candidate-set agree on the same
confident winner. When
GEMM input `shape_spec` contains explicit `Dim.buckets` and no runtime override
is supplied, profiling now expands those buckets into concrete workload cases
and carries `shape.case_id`, dynamic dim values, and dim sources into the report
and execution plan. Profiling also builds a CUTLASS alignment context for each
workload. The context combines optional dense layout element alignment on either
GEMM A/B operand, known tensor or layout storage offsets, current output and
epilogue-input alignment metadata, and the profiled shape-derived cap. Candidate
workloads whose CUTLASS A/B or epilogue alignment exceeds that context are
pruned before timing, then profiling keeps only the highest remaining A/B
alignment variant for each otherwise identical CUTLASS policy. Lower-alignment
manifest candidates stay available as generated runtime fallbacks instead of
being timed as duplicate policy candidates. v2 also mirrors v1's shape-derived A/B alignment rule:
`gemm_rrr` caps candidate alignment by `gcd(K, N)`, `gemm_rcr` caps it by `K`,
manifest defaults use static dimensions or dynamic `Dim.divisible_by`, and
profiling workloads use each concrete bucket, override, or max-shape case.
Kernel manifests preserve this all-runtime `cutlass_alignment` context plus the
legacy `cutlass_alignment_cap`, and execution-plan overlays are rejected in
strict mode when they try to install a profiled candidate whose CUTLASS
alignment exceeds that cap. Generated CUDA modules branch on selected-candidate
A/B pointer byte alignment and fall back through lower-alignment CUTLASS
candidates before failing, while the common runtime support path applies ABI
byte offsets to logical tensor pointers, validates offset-adjusted byte
capacity, and still requires contiguous row-major strides when stride metadata
is supplied. Profile results, cache keys,
execution-plan selections, and static overlays preserve `split_k` plus
`workspace_nbytes` as launch/result metadata. Base, bias/activation, and
additive residual `device::Gemm` candidates now advertise
v1-style split-K search metadata, the profiler expands split-K values using the
v1 `K // max(M, N)` heuristic, and generated CUDA uses companion split-K
launcher/profiler symbols plus a session-owned CUTLASS workspace when an
execution plan selects `split_k > 1`. Non-additive residual and broader
broadcast epilogue families still profile and launch with `split_k=1`; their
fused epilogues need epilogue-specific partition behavior before serial split-K
can avoid reapplying residual operands or final activations incorrectly. The
report/cache key records a
best-effort CUDA hardware/toolchain fingerprint, support-library source/binary
hashes, support-build provenance, and the candidate set/config keys. CUTLASS
GEMM and BMM now use repo CMake op/dtype archive targets and compact
`cutlass_gemm_manifest.json` / `cutlass_bmm_manifest.json` records for the
CMake-built support cache; generated CUDA artifact builds no longer use
per-artifact rendered/pruned GEMM/BMM support `.so` libraries. The GEMM CMake
targets generate chunked op/dtype instantiation sources from `cutlass_gemm.cu`
using the same candidate metadata renderer that feeds manifests and profilers,
with provider-shared helper definitions kept in `cutlass_common.cuh` and chunk
boundaries kept in the build tree for practical compile parallelism. The first
epilogue slice uses a structured GEMM descriptor split:
`dinoml.kernels.families.gemm` owns layout/shape/epilogue contracts and
`dinoml.kernels.providers.cutlass.gemm` owns CUTLASS symbol/candidate metadata.
GEMM candidate generation starts from the v1 SM80 TensorOp 16816 tile list for
`float16`/`bfloat16`, including alignment variants and `float16` versus
`float32` accumulation choices where v1 generated both, then filters candidates
through CUTLASS SM80 tensor-op thread-map divisibility rules for the op's A/B
layouts before they reach manifests or support-source generation. This keeps
RRR reduced-precision manifests from advertising the N=96/160/224 tile shapes
that CUTLASS rejects at compile time with `ShapeInAccesses must be divisible by
WarpThreadArrangement`; default RRR fp16/bf16 manifests now carry 111 buildable
candidates per accumulator policy, while RCR keeps the full 138 candidates.
Default `float32` candidates still total 221 variants: 57 regular TF32 TensorOp, 57
`multiply_add_fast_f16` TensorOp, 57 `multiply_add_fast_bf16` TensorOp, 39
3xTF32 `multiply_add_fast_f32` TensorOp, and 11 exact f32 SIMT fallback
candidates. All TensorOp float32 families are optional; `no_tf32=True` filters
them out and leaves only the 11 exact f32 SIMT candidates. Residual broadcast
GEMMs use a local CUTLASS selector so TensorOp policies use
`DefaultGemmWithBroadcast` while SIMT policies use the SIMT broadcast epilogue
path. Each candidate records tile, stage count, warp count, alignment, math
mode, optional status, math operator, and accumulator dtype so profiling can
distinguish real kernel variants. Generated support source is pruned from the
macro-backed checked-in
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
is wired for RCR and RRR epilogues
`gemm_{rcr,rrr}_bias_{add,add_relu,mul,add_add,mul_add,add_add_relu,mul_tanh,sigmoid_mul,sigmoid_mul_tanh}`,
including frontend shape metadata, CPU reference execution, CUDA lowering checks,
support-library runtime smoke coverage, and profiler workload shapes.

`dml.compile(..., execution_plan=...)` and `dinoml compile --execution-plan`
now consume the static overlay from a profile-selected execution plan before
writing `kernel_manifest.json`, `kernel_codegen_plan.json`, or generated CUDA
source. Keyed execution plans are now checked against their payload before
compile applies provider selections, and artifacts record the applied execution
plan summary in `compile_config.json` and top-level `manifest.json`.
`dml.compile(..., profile=True)` and `dinoml compile --profile` provide
the first opt-in closed loop: build the candidate artifact, profile it, load the
generated execution plan, and rebuild with the plan applied. The bootstrap timing
report is preserved as `debug/bootstrap_profile_report.json` on the final
artifact, and the loop reuses the same lowered IR/constants for both builds while
refreshing generated CUDA sources for the selected plan. For profiled dynamic shapes whose buckets choose different candidates
or split-K values, the manifest now carries guarded per-node dispatch selections:
generated CUDA checks profiled `M/N/K` cases, calls the selected candidate
symbol, sizes a shared CUTLASS workspace for split-K dispatches, and falls back
to the safe manifest default when no guard matches. The v1 `ELUp1` activation is now
available as a bias-activation GEMM epilogue, and additive residual epilogues now
have partition-aware serial split-K launch/profiling coverage. Non-additive
residual/broadcast split-K remains a targeted follow-up. The permuted
`gemm_rcr_permute_elup1` form remains part of the later layout-fused family.
Broader broadcast arithmetic epilogues, v1 `dual_gemm`/dual-output
GEMM families, beyond-v1 CUTLASS epilogues where CUTLASS gives useful fused
functionality, grouped GEMM parity, and broader non-trailing BMM broadcast
forms should wait behind that profiling loop so v2 does not accumulate more
declared surface area without v1-grade selection behavior.

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

ROCm Composable Kernel source is also carried as a plain submodule under
`third_party/composable_kernel`, pinned to AMD's `rocm-7.2.3` release tag from
the `ROCm/composable_kernel` mirror. AMD's canonical development home is now
the ROCm monorepo path `ROCm/rocm-libraries/projects/composablekernel`, but the
mirror keeps CK at repository root and avoids requiring sparse-checkout state in
fresh clones or CI. This is source provenance for future CK provider work only;
it does not make any ROCm GEMM/BMM/Conv runtime path available yet.

Support builds enable NVCC `--split-compile=<effective-cpu-count>` when the
local compiler supports it. The default mirrors v1-style cgroup quota handling
and prefers physical Linux cores when CPU topology is available. Set
`DINOML_NVCC_SPLIT_COMPILE=1` to disable that flag, or set it to another
positive integer to tune the number of device optimization workers.
