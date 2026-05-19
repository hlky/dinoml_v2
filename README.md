# DinoML v2

DinoML v2 is a greenfield vertical slice of a portable ahead-of-time ML
compiler/runtime for production inference. The current MVP compiles native
Python frontend graphs into versioned `.dinoml/` artifacts, links them against
shared CPU/CUDA runtime and kernel libraries, and runs through a small C ABI
without requiring PyTorch at runtime.

## Quick Start

```sh
pip install -e ".[validate]"
python -m dinoml.cli compile examples/fused_elementwise.py --out build/fused_elementwise_cpu.dinoml
python -m dinoml.cli inspect build/fused_elementwise_cpu.dinoml
python -m dinoml.cli validate build/fused_elementwise_cpu.dinoml --against examples/fused_elementwise.py
```

The `validate` extra installs the PyTorch dependency used by example reference
checks. `compile` defaults to the CPU target; pass `--target cuda --arch sm_86`
when building a CUDA artifact. `--target rocm` is registered as a distinct
toolchain scaffold with default arch `gfx1201`, but no ROCm op lowering or
generated HIP artifact build is admitted yet. The model argument is a Python
file that defines `build_spec()`. After compiling the artifact, the same CPU
module can be loaded and run directly through the Python runtime API:

```sh
python - <<'PY'
import numpy as np
import runpy

from dinoml.runtime import load

example = runpy.run_path("examples/fused_elementwise.py")
inputs = example["build_validation_inputs"]()
constants = example["build_constants"]()
x = inputs["x"]
expected = np.maximum(
    x * constants["scale"] + constants["bias"] - (1.0 / (1.0 + np.exp(-x))),
    0.0,
) * 0.5

module = load("build/fused_elementwise_cpu.dinoml")
session = module.create_session()
try:
    actual = session.run_numpy(inputs)["y"]
    np.testing.assert_allclose(actual, expected.astype(np.float32), atol=1e-6, rtol=1e-6)
finally:
    session.close()
    module.close()

print("runtime ok")
PY
```

For a small image-style CPU workflow using existing pad and pooling primitives:

```sh
python -m dinoml.cli compile examples/image_pooling.py --target cpu --out build/image_pooling_cpu.dinoml
python -m dinoml.cli validate build/image_pooling_cpu.dinoml --against examples/image_pooling.py
```

For a small CPU selection workflow using existing `topk` and `batch_gather`
helpers:

```sh
python -m dinoml.cli compile examples/candidate_selection.py --target cpu --out build/candidate_selection_cpu.dinoml
python -m dinoml.cli validate build/candidate_selection_cpu.dinoml --against examples/candidate_selection.py
```

For a compact layout workflow using existing sub-pixel rearrangement helpers:

```sh
python -m dinoml.cli compile examples/subpixel_upsample.py --target cpu --out build/subpixel_upsample_cpu.dinoml
python -m dinoml.cli validate build/subpixel_upsample_cpu.dinoml --against examples/subpixel_upsample.py
```

Additional compact CPU workflows live under `examples/`, including
`coordinate_ramp.py` for creation helpers feeding fused elementwise math,
`clip_model_workflow.py` for a bounded synthetic two-tower CLIP artifact proof,
and `clip_checkpoint_workflow.py` for a cached local Hugging Face CLIP
checkpoint compile/load/run parity workflow.

For a compact CUDA linear workflow using existing explicit GEMM+bias ops,
runtime-settable constants, a bucketed dynamic batch dimension, and a visible
CUTLASS provider manifest:

```sh
python -m dinoml.cli compile examples/cuda_linear.py --target cuda --arch sm_86 --no-tf32 --out build/cuda_linear.dinoml
python -m dinoml.cli validate build/cuda_linear.dinoml --against examples/cuda_linear.py --atol 1e-2 --rtol 1e-2
```

For CUDA smoke coverage, add `--target cuda --arch sm_86` and choose a CUDA
artifact path.
`python -m dinoml.cli validate` explicitly loads artifact constants for the
validation run, so artifacts compiled with
`--constant-load-policy deferred` can still use the same correctness check.

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
  debug/
    generated_src/
    pass_dumps/
```

`libdinoml_runtime.so`, `libdinoml_cuda_runtime.so`, and
`libdinoml_cuda_kernels.so` are built with CMake and cached per CUDA architecture
and required-kernel manifest under `~/.cache/dinoml_v2/support/`. CPU artifacts
use `libdinoml_runtime.so` plus `libdinoml_cpu_kernels.so`.
The ROCm scaffold can also build `dinoml_runtime`, `dinoml_rocm_runtime`, and
`dinoml_rocm_kernels` support libraries under a `support/rocm-gfx1201/...`
cache directory, using the platform library suffix (`.dll` on Windows). Those
ROCm libraries are not copied into model artifacts until a generated HIP module
path and op lowering are admitted.
CUDA artifacts that need CUTLASS GEMM link the required op/dtype static archives
into the generated `module.so`; they do not carry CUTLASS GEMM support `.so`
files. The CMake aggregate target `dinoml_cutlass_gemm` builds every archive for
release/prebuilt bundles, while artifact builds request only the archive targets
referenced by the kernel manifest. These archives are cached once per CUDA
architecture under `cutlass-gemm/cmake-full` instead of being rendered and
pruned per artifact. The target generates chunked op/dtype instantiation
sources from `kernels/cuda/src/cutlass_gemm.cu` under the CMake build tree,
matching the BMM/Conv support-library flow while preserving GEMM compile
parallelism and keeping shared CUTLASS helpers in
`kernels/cuda/src/cutlass_common.cuh`. Its
`cutlass_gemm_manifest.json` records the CMake targets, source hash, and archive
hashes used by profiling fingerprints. The CMake target is controlled by
`-DDINOML_ENABLE_CUTLASS_GEMM=ON/OFF`. CUDA
artifacts with `cutlass_conv` carry
`libdinoml_cutlass_conv.so` as a real bounded runtime support library:
generated modules can pack NCHW/OIHW inputs into NHWC/OHWI temporaries and run
the admitted static groups=1 rank-4 `conv2d_bias` family through selected
CUTLASS launchers, while broader Conv surface remains intentionally out of
scope.

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
updated at runtime with `RuntimeModule.set_constant_numpy(...)`. GGUF-backed
encoded constants can be loaded explicitly with
`RuntimeModule.load_encoded_constants(...)`; CPU and the default CUDA GGUF path
still dequantize at load time into the dense constant ABI. The newer
`materialization="dequantize_on_gpu_before_launch"` path is narrower: today it
only supports CUDA `gemm_rrr`, `gemm_rcr`, `gemm_rrr_bias`, and
`gemm_rcr_bias` with a GGUF RHS constant, `residency="manual_runtime_load"`,
and `float32`/`float16` output. Unsupported uses such as non-bias GEMM
epilogues or elementwise consumers are rejected
instead of being treated as runtime-loadable encoded constants.

CPU kernels do not require OpenMP. CMake uses it when available on supported
platforms, and it can be disabled with `-DDINOML_ENABLE_OPENMP=OFF`.

## Development

```sh
git submodule update --init --recursive
pip install -e ".[dev]"
python -m pytest -q
python -m pytest -q tests/ir
python -m pytest -q tests/cpu
python -m pytest -q tests/cuda
python tools/benchmark_fused_elementwise.py --suite quick --targets cpu,cuda
python tools/benchmark_softmax.py --suite quick --targets cpu,cuda
python tools/benchmark_reductions.py --suite quick --targets cpu,cuda
python tools/benchmark_clip_checkpoint.py --target cpu --warmup 1 --iters 3
python -m dinoml.cli profile build/model.dinoml --iterations 20 --repeats 3
python -m dinoml.cli compile model.py --target cuda --profile --profile-repeats 3 --out build/model-profiled.dinoml
```

The repo currently carries third-party source submodules for CUTLASS, libgguf,
and ROCm Composable Kernel. CK is vendored from AMD's
`ROCm/composable_kernel` mirror and pinned to the `rocm-7.2.3` release tag;
the canonical upstream development home is the ROCm monorepo under
`ROCm/rocm-libraries/projects/composablekernel`. The checked-out CK source is
provider groundwork only and does not by itself broaden the supported ROCm
runtime surface.

On Windows ROCm development, the pip/venv SDK layout is resolved through
`python -m rocm_sdk` rather than assuming a system install. The support-library
smoke can run from any Python environment with `pytest` installed while using
the ROCm venv explicitly for SDK resolution:

```powershell
$env:PYTHONPATH = (Get-Location).Path + "\src"
$env:DINOML_ROCM_PYTHON_EXECUTABLE = (Resolve-Path ".venv/rocm/Scripts/python.exe").Path
$env:DINOML_RUN_ROCM_SUPPORT_BUILD_SMOKE = "1"
python -m pytest -q tests/backends/test_rocm_scaffold.py
```

Tests are organized by intent: `tests/ir/` covers frontend/IR/reference NumPy
behavior, `tests/cpu/` compiles and runs CPU artifacts, and `tests/cuda/`
compiles and runs CUDA artifacts when the CUDA toolchain is available.

Generated artifacts, support-library build products, benchmark output, and local
profile data are ignored by git. Use `tmp/` for scratch generated modules when
reviewing codegen.

See [docs/architecture.md](docs/architecture.md) for the op/backend/kernel split,
[agents/plans/op_porting_checklist.md](agents/plans/op_porting_checklist.md) for the porting map,
and [agents/plans/v1_gap_audit.md](agents/plans/v1_gap_audit.md) for foundations still missing
from v1.
