# Benchmarking Surfaces

DinoML has multiple benchmarking surfaces. They are useful for different questions and should not be reported as interchangeable.

## 1. Native Artifact Benchmark

Use the native session benchmark path when the goal is whole-artifact runtime throughput.

- Python entry points: `Session.benchmark_numpy(...)` and `Session.benchmark_device_pointers(...)`
- Native entry point: `dino_session_benchmark`
- Runtime metadata describes the timing source and, on GPU targets, whether graph replay was required

This is the preferred artifact-level timing path because the timing happens inside the compiled module rather than around repeated Python calls.

## 2. Op-Level Benchmark Suite

Use the existing benchmark suites when the goal is per-op throughput or provider-case comparisons.

- CLI entry point: `python -m dinoml.cli benchmark-ops ...`
- Implementation path: `src/dinoml/benchmarks/ops.py`

The op suite compiles focused artifacts and then uses `session.benchmark_numpy(...)`. Do not replace that with wall-clock timing around `session.run_numpy(...)` unless the task is explicitly about host orchestration overhead instead of artifact throughput.

## 3. Pipeline-Orchestration Timing

Pipeline scripts such as the GLM-OCR and Qwen2.5-VL static-cache tools are allowed to time around repeated `run_numpy(...)` or `run_device_pointers(...)` calls when the benchmark question is about higher-level orchestration:

- prefill plus decode sequencing
- cache update flow
- module residency and unload/reload policy
- device-pointer setup and reuse
- whole-pipeline output parity against an external reference

These scripts are primarily contract and parity harnesses with timing attached. Report them as pipeline or orchestration timing, not as equivalent to the native artifact benchmark path.

## 4. Provider Profiling

Provider profiling and execution-plan selection answer a different question again: candidate selection for a provider-backed kernel family.

- Use provider profiling when comparing candidate kernels or validating execution-plan behavior.
- Do not present provider profiling output as end-to-end model throughput.

## Reporting Rules

When reporting benchmark results, always state:

- the timing surface
- the execution path, for example `benchmark_numpy`, `benchmark_device_pointers`, `run_numpy`, or `run_device_pointers`
- whether the result is artifact throughput, op throughput, pipeline timing, or provider profiling
- any important runtime conditions, such as external stream usage or graph replay requirements

## Validation Split

For model and pipeline workflows, keep these claims separate:

- artifact contract validation
- artifact execution success
- parity against an external reference such as Transformers
- throughput or latency claims

Passing one does not imply the others.
