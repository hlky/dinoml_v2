# DinoML v2

DinoML v2 is a greenfield vertical slice of a portable ahead-of-time ML
compiler/runtime for production inference. The current MVP compiles native
Python frontend graphs into versioned `.dinoml/` artifacts, links them against
shared CPU/CUDA runtime and kernel libraries, and runs through a small C ABI
without requiring PyTorch at runtime.

## Quick Start

```sh
pip install -e .
python -m dinoml.cli compile examples/fused_elementwise.py --target cpu --out build/fused_elementwise_cpu.dinoml
python -m dinoml.cli inspect build/fused_elementwise_cpu.dinoml
python -m dinoml.cli validate build/fused_elementwise_cpu.dinoml --against examples/fused_elementwise.py
```

For CUDA smoke coverage, add `--target cuda --arch sm_86` and choose a CUDA
artifact path.

The first milestone intentionally keeps the executable surface small:
registered elementwise graphs are fused into generated kernels, with CPU
and CUDA `float32`/`float16`/`bfloat16` fused-elementwise execution.
The IR records symbolic `Dim` shape constraints, and generated CPU/CUDA modules
materialize runtime shape buffers for dynamic validation and generic
broadcasting.

## Build Layout

The generated artifact is intentionally split into reusable and model-specific
pieces:

```text
model.dinoml/
  manifest.json
  metadata.json
  graph.dinoir.json
  compile_config.json
  constants.bin
  module.so
  lib/
    libdinoml_runtime.so
    libdinoml_cuda_runtime.so
    libdinoml_cuda_kernels.so
    libdinoml_cutlass_gemm.so
  debug/
    generated_src/
    pass_dumps/
```

`libdinoml_runtime.so`, `libdinoml_cuda_runtime.so`, and
`libdinoml_cuda_kernels.so` are built with CMake and cached per CUDA architecture
and required-kernel manifest under `~/.cache/dinoml_v2/support/`. CPU artifacts
use `libdinoml_runtime.so` plus `libdinoml_cpu_kernels.so`.
CUDA artifacts that need CUTLASS GEMM also carry `libdinoml_cutlass_gemm.so`;
the cached support build writes `cutlass_gemm_manifest.json` with compile flags,
NVCC version, dependency header hashes, source/library hashes, and a provenance
key used for cache reuse and profiling fingerprints. The support cache also
writes a `dinoml.support_source_manifest` at `src/source_manifest.json`, which
maps the reviewable support source to candidate set keys, candidate config keys,
launcher/profiler symbols, and support build units for later generated CUTLASS
candidates.

Generated model code is a small Jinja2 wrapper that links against those
libraries. It loads runtime metadata from `metadata.json` and contains launch
order, memory bindings, constant bindings, runtime shape-buffer updates, and
model-specific generated fused-elementwise kernels. Reusable kernels and shared
scalar math helpers live outside the model wrapper. The previous naive matmul
placeholder was removed; current `gemm_rrr`/`gemm_rcr` and
`gemm_rrr_bias`/`gemm_rcr_bias` plus
`gemm_rrr_bias_relu`/`gemm_rcr_bias_relu`, v1-style bias activation epilogues,
including `ELUp1`, first rank-2 residual epilogues, and v1 RCR compound residual activation
epilogues call the cached CUTLASS support library.
Broader GEMM/BMM coverage should extend that library-backed path.

Constants are loaded from `constants.bin` when a module is opened and can also be
updated at runtime with `RuntimeModule.set_constant_numpy(...)`.

CPU kernels do not require OpenMP. CMake uses it when available on supported
platforms, and it can be disabled with `-DDINOML_ENABLE_OPENMP=OFF`.

## Development

```sh
git submodule update --init --recursive
pip install -e ".[dev]"
python -m pytest -q
python tools/benchmark_fused_elementwise.py --suite quick --targets cpu,cuda
python tools/benchmark_softmax.py --suite quick --targets cpu,cuda
python tools/benchmark_reductions.py --suite quick --targets cpu,cuda
python -m dinoml.cli profile build/model.dinoml --iterations 20 --repeats 3
python -m dinoml.cli compile model.py --target cuda --profile --profile-repeats 3 --out build/model-profiled.dinoml
```

Generated artifacts, support-library build products, benchmark output, and local
profile data are ignored by git. Use `tmp/` for scratch generated modules when
reviewing codegen.

See [docs/architecture.md](docs/architecture.md) for the op/backend/kernel split,
[agents/plans/op_porting_checklist.md](agents/plans/op_porting_checklist.md) for the porting map,
and [agents/plans/v1_gap_audit.md](agents/plans/v1_gap_audit.md) for foundations still missing
from v1.
