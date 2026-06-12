# DinoML v2

DinoML v2 is an experimental ML compiler/runtime for turning Python-defined model graphs into standalone native artifacts for CPU, CUDA, and ROCm targets.

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

## Agent Workflow

Most DinoML v2 work was developed through a human-directed agent workflow rather
than one-off prompt sessions. The workflow uses repo-local steering docs,
project memory, provider contracts, op admission rules, validation gates, and
ranked work queues to keep autonomous implementation bounded and reviewable.

A curated public snapshot of those process docs lives at:

https://github.com/hlky/dinoml_v2_agents

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

This path uses the artifact's native `dino_session_benchmark` entry point rather
than timing repeated Python calls around `run_numpy(...)`.

## CUDA And ROCm Targets

DinoML can also emit CUDA and ROCm artifacts when the local toolchain is available.

For this repository's remote CUDA verification workflow, the current preferred
container image is `hlky/dinoml:ubuntu-nodeps`. It is intended for Runpod/Codex
remote validation and already contains:

- a live DinoML v2 git checkout at `/opt/src/dinoml_v2`
- the Python environment at `/opt/venvs/dinoml`
- CUDA 12.9 tooling
- `transformers` and `diffusers` source checkouts

When using the repo-local Runpod helper with that image, prefer reusing the
prebaked repo path:

```powershell
python .codex/skills/runpod-codex-remote/scripts/create_runpod_codex_remote.py `
  --gpu-id "NVIDIA RTX 2000 Ada Generation" `
  --repo https://github.com/hlky/dinoml_v2.git `
  --name dinoml-ubuntu-nodeps-rtx2000ada `
  --image hlky/dinoml:ubuntu-nodeps `
  --existing-project-path /opt/src/dinoml_v2 `
  --volume-gb 20 `
  --ports 22/tcp `
  --auto-connect
```

For disposable verification pods, delete the pod when done. `runpodctl pod stop` only stops compute; volume storage remains billable until `runpodctl pod delete`.

The current smoke-validation baseline for that image is:

- `torch` CUDA tensor/matmul
- tiny `transformers` GPT-2 forward on CUDA
- tiny `diffusers` UNet/scheduler forward on CUDA
- DinoML v2 trace/compile/load/run smoke on CUDA

See [.codex/skills/runpod-codex-remote/SKILL.md](.codex/skills/runpod-codex-remote/SKILL.md)
for the repository-local remote workflow details.

CUDA example:

```powershell
python -m dinoml.cli compile examples/cuda_linear.py --target cuda --arch sm_86 --out build/cuda_linear_cuda.dinoml
```

ROCm example:

```powershell
python -m dinoml.cli compile examples/cuda_linear.py --target rocm --arch gfx1201 --out build/cuda_linear_rocm.dinoml
```

See [docs/model_pipeline_benchmarking.md](docs/model_pipeline_benchmarking.md)
for the current benchmarking surfaces and reporting rules.

Profiling-enabled compile:

```powershell
python -m dinoml.cli compile examples/cuda_linear.py --target cuda --profile --out build/cuda_linear_profiled.dinoml
```

The profiling path materializes the profiling inputs and metadata, benchmarks supported candidate kernels, reuses a target-scoped provider-problem cache when safe, writes an execution plan, then rebuilds the final artifact using the selected candidates.

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
session.close()
module.close()
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

Additional project docs:

- `docs/project_invariants.md`: durable architecture, provider, constant, and op rules.
- `docs/provider_contract.md`: provider maturity and required pieces.
- `docs/op_admission.md`: checklist for new or expanded public ops.
- `docs/model_pipeline_benchmarking.md`: benchmarking surfaces and reporting rules.
- `docs/cuda_remote_verification.md`: preferred Runpod image and CUDA remote validation flow.

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
python -m dinoml.cli profile build/cuda_linear_cuda.dinoml --iterations 20 --repeats 3
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
- `examples/coordinate_ramp.py`: simple coordinate-generation workflow.
- `examples/fused_elementwise.py`: elementwise fusion path.
- `examples/image_pooling.py`: pooling and image-like tensor operations.
- `examples/subpixel_upsample.py`: layout/shape-sensitive image op.
- `examples/candidate_selection.py`: profiling and candidate-selection workflow.

Model-oriented pipeline and parity tools currently live under `tools/`, for
example:

- `tools/benchmark_glm_ocr_static_cache_pipeline.py`
- `tools/benchmark_qwen2_5_vl_static_cache_pipeline.py`
- `tools/check_glm_ocr_decode_parity.py`

## Tests

Run the Python test suite:

```powershell
python -m pytest
```

Useful focused checks:

```powershell
python -m pytest tests/ir tests/cpu
python -m pytest tests/runtime/test_runtime_benchmark.py
python -m pytest tests/benchmarks/test_ops_benchmark_suite.py
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

## License

Original DinoML v2 code and documentation are licensed under the Apache License,
Version 2.0. See [LICENSE](LICENSE). Third-party projects under `third_party/`
retain their upstream licenses.
