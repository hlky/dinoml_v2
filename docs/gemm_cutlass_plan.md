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
- Exported launcher symbols:
  - `dinoml_cutlass_gemm_rrr_{f32,f16,bf16}`
  - `dinoml_cutlass_gemm_rcr_{f32,f16,bf16}`
- Exported profiler symbols:
  - `dinoml_profile_cutlass_gemm_rrr_{f32,f16,bf16}`
  - `dinoml_profile_cutlass_gemm_rcr_{f32,f16,bf16}`

The first runtime GEMM port now wires model lowering into that support library:

1. `gemm_rcr`/`gemm_rrr` are explicit frontend ops for `float32`, `float16`,
   and `bfloat16`, not a generic `matmul`; they preserve dynamic `M/N` shape
   metadata while requiring rank-2 tensors and compatible max-shape `K`.
2. The kernel manifest records `cutlass_gemm` as an external support library
   with real launcher/profiler symbols.
3. Generated CUDA model wrappers link `libdinoml_cutlass_gemm.so` and call the
   cached launcher with runtime `M/N/K`, so smaller runtime `M/N` values use the
   same max-shape artifact.
4. CPU has reference execution only; compiled CPU GEMM still rejects until a
   real CPU library path exists.

`dinoml profile <artifact>` now executes the exported profiler symbol for the
manifest-selected `cutlass_default` candidate, writes
`debug/profile_report.json`, and caches results under the support-library cache.
The report/cache key records a best-effort CUDA hardware/toolchain fingerprint,
support-library source/binary hashes, support-build provenance, and the
candidate config key. CUTLASS support manifests also record compile flags, NVCC
version output, dependency header hashes, and a provenance key that participates
in support-cache reuse. Next steps are candidate enumeration beyond the single
default CUTLASS instance, bias/activation epilogues, optional
accumulation-policy variants, and then public `matmul` layout selection.

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

## Non-Goals For The First GEMM Patch

- No naive C++/CUDA matmul.
- No full epilogue visitor port.
- No grouped GEMM.
- No convolution implicit-GEMM yet.
- No public `dml.ops.matmul` until CUDA and CPU/reference behavior are both
  represented in the IR/runtime contract.
