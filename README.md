# DinoML v2

DinoML v2 is a greenfield vertical slice of a portable ahead-of-time ML
compiler/runtime for production inference. The current MVP compiles native
Python frontend graphs into versioned `.dinoml/` artifacts, links them against
shared CPU/CUDA runtime and kernel libraries, and runs through a small C ABI
without requiring PyTorch at runtime.

## Quick Start

```sh
pip install -e .
python -m dinoml.cli compile examples/fused_elementwise.py --target cuda --arch sm_86 --out build/fused_elementwise.dinoml
python -m dinoml.cli inspect build/fused_elementwise.dinoml
python -m dinoml.cli validate build/fused_elementwise.dinoml --against examples/fused_elementwise.py
```

The first milestone intentionally keeps the executable surface small:
registered elementwise graphs are fused into generated kernels, with CPU
`float32` execution and CUDA `float32`/`float16`/`bfloat16` execution.
The IR records symbolic `Dim` shape constraints, and generated CPU/CUDA modules
materialize runtime shape buffers for dynamic validation and generic
broadcasting.

## Build Layout

The generated artifact is intentionally split into reusable and model-specific
pieces:

```text
model.dinoml/
  manifest.json
  graph.dinoir.json
  compile_config.json
  constants.bin
  module.so
  lib/
    libdinoml_runtime.so
    libdinoml_cuda_runtime.so
    libdinoml_cuda_kernels.so
  debug/
    generated_src/
    pass_dumps/
```

`libdinoml_runtime.so`, `libdinoml_cuda_runtime.so`, and
`libdinoml_cuda_kernels.so` are built with CMake and cached per CUDA architecture
and required-kernel manifest under `~/.cache/dinoml_v2/support/`. CPU artifacts
use `libdinoml_runtime.so` plus `libdinoml_cpu_kernels.so`.

Generated model code is a small Jinja2 wrapper that links against those
libraries. It contains metadata, launch order, memory bindings, constant
bindings, runtime shape-buffer updates, and model-specific generated
fused-elementwise kernels. Reusable kernels and shared scalar math helpers live
outside the model wrapper. The previous naive matmul placeholder was removed;
GEMM/BMM should land through the real library-backed op port.

Constants are loaded from `constants.bin` when a module is opened and can also be
updated at runtime with `RuntimeModule.set_constant_numpy(...)`.

CPU kernels do not require OpenMP. CMake uses it when available on supported
platforms, and it can be disabled with `-DDINOML_ENABLE_OPENMP=OFF`.

## Development

```sh
pip install -e ".[dev]"
python -m pytest -q
```

Generated artifacts, support-library build products, benchmark output, and local
profile data are ignored by git. Use `tmp/` for scratch generated modules when
reviewing codegen.

See [docs/architecture.md](docs/architecture.md) for the op/backend/kernel split,
[docs/op_porting_checklist.md](docs/op_porting_checklist.md) for the porting map,
and [docs/v1_gap_audit.md](docs/v1_gap_audit.md) for foundations still missing
from v1.
