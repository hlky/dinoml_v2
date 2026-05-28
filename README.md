# DinoML v2

DinoML v2 is an experimental ML compiler/runtime for turning Python-defined model graphs into standalone native artifacts for CPU, CUDA, and ROCm targets.

The project is intentionally small enough to understand end to end, but it touches the hard parts of production inference systems: graph tracing, IR validation, shape handling, code generation, native runtime ABI design, CUDA/HIP backend integration, kernel selection, benchmarking, and quantized model weights.

## What This Shows

This repository is a working systems project, not a wrapper around one library. It demonstrates:

- A Python frontend for tracing model-like modules into a portable DinoML IR.
- A native artifact format with manifests, graph hashes, constants, runtime metadata, and ABI versioning.
- CPU, CUDA, and ROCm lowering paths with generated C++/CUDA/HIP module code.
- A C++ runtime and Python `ctypes` loader for executing compiled artifacts from NumPy inputs.
- CUTLASS-backed CUDA candidate generation for GEMM, BMM, and convolution paths.
- Composable Kernel-backed ROCm candidate generation and profiling scaffolding.
- Profiling-driven execution plans that benchmark candidate kernels and rebuild artifacts with selected implementations.
- Dynamic shape support through symbolic dimensions, runtime validation, and shape-buffer lowering.
- GGUF constant integration and prototype fused GGUF quantized GEMM work for ROCm.
- Test and benchmark surfaces for compiler contracts, runtime behavior, kernel lowering, and op-level performance.

In short: DinoML v2 is a proof of ability across product-shaped Python APIs, compiler/runtime architecture, and low-level GPU inference engineering.

## Status

| Area | Status |
| --- | --- |
| Python tracing frontend | active prototype |
| CPU backend | working test target |
| CUDA backend | active prototype with CUTLASS candidate support |
| ROCm backend | active prototype with Composable Kernel scaffolding |
| Runtime loader | working native module/session API |
| Profiling/execution plans | active prototype |
| GGUF quantized kernels | research/prototype |
| Public API stability | not stable |

This is not a polished user-facing framework yet. It is a compiler/runtime lab for building and testing inference-system ideas quickly.

## Why It Exists

Most ML deployment stacks force a choice between high-level ergonomics and low-level control. DinoML is an experiment in keeping both:

- Define the model in Python.
- Trace it into a compact, inspectable IR.
- Lower it into a standalone artifact.
- Generate target-specific native code.
- Select kernels using measured performance, not only static heuristics.
- Load and run the artifact without depending on the original Python model.

The work is aimed at the awkward but valuable layer where model behavior, runtime performance, portability, and deployment reliability meet.

## Quick Start

Clone with submodules:

```powershell
git clone --recurse-submodules https://github.com/hlky/dinoml_v2.git
cd dinoml_v2
```

Install in editable mode:

```powershell
python -m pip install -e ".[dev,validate]"
```

Compile a small CPU artifact:

```powershell
python -m dinoml.cli compile examples/cuda_linear.py --target cpu --out build/cuda_linear_cpu.dinoml
```

Inspect it:

```powershell
python -m dinoml.cli inspect build/cuda_linear_cpu.dinoml
```

Validate against the Python reference:

```powershell
python -m dinoml.cli validate build/cuda_linear_cpu.dinoml --against examples/cuda_linear.py
```

Benchmark the runtime session:

```powershell
python -m dinoml.cli benchmark build/cuda_linear_cpu.dinoml --against examples/cuda_linear.py --warmup 5 --iterations 20
```

## CUDA And ROCm Targets

DinoML can also emit CUDA and ROCm artifacts when the local toolchain is available.

CUDA example:

```powershell
python -m dinoml.cli compile examples/cuda_linear.py --target cuda --arch sm_86 --out build/cuda_linear_cuda.dinoml
```

ROCm example:

```powershell
python -m dinoml.cli compile examples/cuda_linear.py --target rocm --arch gfx1201 --out build/cuda_linear_rocm.dinoml
```

Profiling-enabled compile:

```powershell
python -m dinoml.cli compile examples/cuda_linear.py --target cuda --profile --out build/cuda_linear_profiled.dinoml
```

The profiling path emits a bootstrap artifact, benchmarks supported candidate kernels, writes an execution plan, then rebuilds the final artifact using the selected candidates.

## Frontend Example

```python
import numpy as np
import dinoml as dml


class Linear(dml.Module):
    def __init__(self):
        self.weight = dml.Parameter([8, 6], dtype="float32", name="weight")
        self.bias = dml.Parameter([6], dtype="float32", name="bias")

    def forward(self, x):
        y = dml.ops.gemm_rrr_bias(x, self.weight, self.bias)
        return dml.ops.output(y, "y")


batch = dml.Dim("batch", min=1, max=4, typical=3, buckets=(1, 3, 4))
constants = {
    "weight": np.random.randn(8, 6).astype("float32"),
    "bias": np.random.randn(6).astype("float32"),
}

spec = dml.trace(
    Linear(),
    inputs={"x": dml.TensorSpec([batch, 8], "float32")},
    constants=constants,
    name="linear",
)

artifact = dml.compile(spec, dml.Target("cpu"), "build/linear_cpu.dinoml")
```

Runtime execution:

```python
import numpy as np
from dinoml import runtime

module = runtime.load("build/linear_cpu.dinoml")
session = module.create_session()

outputs = session.run_numpy({
    "x": np.random.randn(3, 8).astype("float32"),
})

print(outputs["y"])
```

## Architecture

```text
Python module
  -> trace frontend
  -> DinoML IR
  -> validation and graph passes
  -> target lowering
  -> kernel manifest
  -> generated C++ / CUDA / HIP module
  -> native artifact directory
  -> Python runtime loader
  -> native session execution
```

Key pieces:

- `src/dinoml/frontend.py`: module, tensor, parameter, and tracing API.
- `src/dinoml/ir.py`: IR schema, dtype handling, graph hashing, artifact metadata.
- `src/dinoml/passes/`: validation and transformation passes.
- `src/dinoml/lowering/`: CPU, CUDA, ROCm, and op-specific lowering.
- `src/dinoml/backends/`: target registry and CMake-backed artifact builds.
- `src/dinoml/kernels/`: kernel manifests, codegen plans, profiling, and provider integration.
- `runtime/`: C++ runtime ABI and CUDA/ROCm runtime support.
- `kernels/`: native CPU, CUDA, and ROCm kernel libraries.
- `tools/`: CUTLASS/CK codegen and profiling helpers.
- `examples/`: small model workflows and CLIP-oriented compiler exercises.
- `tests/`: compiler, runtime, backend, profiling, and benchmark contracts.

## Artifact Layout

A compiled `.dinoml` artifact is a directory containing the native module and the metadata needed to load it:

```text
artifact.dinoml/
  manifest.json
  graph.dinoir.json
  metadata.json
  compile_config.json
  kernel_manifest.json
  kernel_codegen_plan.json
  constants.bin
  module.so / module.dll
  lib/
    libdinoml_runtime.*
    libdinoml_*_kernels.*
```

The manifest records the artifact schema, runtime ABI version, target, graph hash, constant loading policy, and support libraries. The runtime checks ABI compatibility before creating sessions.

## Kernel And Profiling Work

DinoML v2 does not treat every op as a single hard-coded implementation. For GPU targets, it can build candidate sets and choose implementations from benchmark data.

Current areas include:

- CUTLASS GEMM/BMM/Conv candidate generation for CUDA.
- ROCm Composable Kernel GEMM/BMM/Conv candidate scaffolding.
- Shape override support for profiling dynamic workloads.
- Execution-plan JSON emitted from profiler runs.
- Op-level benchmark suites for DinoML and PyTorch comparison.
- Native runtime benchmarking using target-appropriate timers.

Representative CLI commands:

```powershell
python -m dinoml.cli benchmark-ops cpu --only add --only reduce_sum
python -m dinoml.cli benchmark-ops cuda --only gemm_rrr_bias --jobs 4
python -m dinoml.cli benchmark-torch-ops --device cuda --only gemm_rrr_bias
```

## GGUF And Quantization Work

The repo includes integration points for GGUF constants and prototype ROCm kernels for fused quantized GEMM paths.

Implemented or scaffolded pieces include:

- `GGUFConstant` descriptors for binding GGUF tensors into model specs.
- Constant materialization policies and runtime metadata.
- Encoded constant manifests for future runtime loading paths.
- ROCm GGUF Q8/Q4/K-family decode and fused GEMM prototype code.
- Benchmark tooling for CK GGUF quantized GEMM experiments.

This area is research-grade, but it is the direction of the project: keep quantized model weights close to their deployed representation and avoid unnecessary full dense materialization where possible.

## Examples

- `examples/cuda_linear.py`: minimal traced linear layer with dynamic batch.
- `examples/fused_elementwise.py`: elementwise fusion path.
- `examples/image_pooling.py`: pooling and image-like tensor operations.
- `examples/subpixel_upsample.py`: layout/shape-sensitive image op.
- `examples/clip_text_workflow.py`: bounded CLIP text tower workflow.
- `examples/clip_model_workflow.py`: bounded CLIP text+vision model workflow.
- `examples/clip_checkpoint_workflow.py`: checkpoint-oriented CLIP workflow.
- `examples/candidate_selection.py`: profiling and candidate-selection workflow.

## Tests

Run the Python test suite:

```powershell
python -m pytest
```

Useful focused checks:

```powershell
python -m pytest tests/ir tests/cpu
python -m pytest tests/test_runtime_benchmark.py
python -m pytest tests/test_ops_benchmark_suite.py
python -m pytest tests/backends/test_ck_gguf_q8_gemm_kernel.py
```

GPU tests require the matching CUDA or ROCm toolchain and hardware.

## Dependencies

Core Python dependencies:

- Python 3.10+
- NumPy
- Jinja2
- pybind11

Development and validation:

- pytest
- ninja
- torch, for validation/reference paths

Native/GPU toolchains:

- CMake
- C++ compiler
- CUDA toolkit for CUDA targets
- ROCm/HIP SDK for ROCm targets
- CUTLASS submodule for CUDA provider work
- Composable Kernel submodule for ROCm provider work
- `third_party/libgguf` for GGUF integration

## Notes For Reviewers

If you are scanning this as a hiring signal, the interesting parts are not just the happy-path examples. The project is meant to show the ability to move across layers:

- Python API ergonomics and model authoring.
- Compiler IR and validation.
- CMake/native artifact builds.
- C++ runtime ABI design.
- CUDA/HIP kernel integration.
- Vendor-library candidate generation.
- Runtime and op benchmarking.
- Quantized model-weight representation.
- Testable contracts around all of the above.

That layer-crossing is the point.
