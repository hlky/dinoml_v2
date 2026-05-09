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
   `gemm_rcr_bias_relu`/`gemm_rrr_bias_relu`, and v1-style activation epilogue
   ops `*_bias_{gelu,fast_gelu,sigmoid,tanh,swish,hardswish}` are explicit
   frontend ops for `float32`, `float16`, and `bfloat16`, not a generic
   `matmul`; they preserve dynamic `M/N` shape metadata while requiring rank-2
   matrix tensors and compatible max-shape `K`.
2. The kernel manifest records `cutlass_gemm` as an external support library
   with real launcher/profiler symbols.
3. Generated CUDA model wrappers link `libdinoml_cutlass_gemm.so` and call the
   cached launcher with runtime `M/N/K`, so smaller runtime `M/N` values use the
   same max-shape artifact.
4. CPU has reference execution only; compiled CPU GEMM still rejects until a
   real CPU library path exists.

`dinoml profile <artifact>` now executes the exported profiler symbol for the
manifest-selected CUTLASS candidate, writes
`debug/profile_report.json`, and caches results under the support-library cache.
The report/cache key records a best-effort CUDA hardware/toolchain fingerprint,
support-library source/binary hashes, support-build provenance, and the
candidate set/config keys. CUTLASS support manifests also record compile flags,
NVCC version output, dependency header hashes, and a provenance key that
participates in support-cache reuse. The support cache writes a
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
`float32` accumulation choices where v1 generated both. `float32` candidates use
the v1 SM80 TensorOp 1688 TF32 tile list, are marked optional rather than exact
f32 parity, and avoid being mixed with 16816 tile shapes. Each candidate records
tile, stage count, warp count, alignment, math mode, optional status, and
accumulator dtype so profiling can distinguish real kernel variants. Generated
support source is pruned from the macro-backed checked-in source to only the
launcher/profiler symbols in the used candidate plan, and activation epilogue
exports instantiate CUTLASS thread epilogue functors directly with the selected
candidate accumulator type.

Next steps are target-level TF32/accumulation policy gates, broader
broadcast/arithmetic epilogues, broader v1 candidate enumeration, BMM and
grouped GEMM parity, and then public `matmul` layout selection.

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

## Remaining Near-Term Non-Goals

- No naive C++/CUDA matmul.
- No full epilogue visitor port beyond the bias and ReLU fused CUTLASS slices.
- No grouped GEMM.
- No convolution implicit-GEMM yet.
- No public `dml.ops.matmul` until CUDA and CPU/reference behavior are both
  represented in the IR/runtime contract.
