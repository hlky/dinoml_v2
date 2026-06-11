# Agent Instructions

You are working in the DinoML v2 repository.

DinoML v2 is an experimental ML compiler/runtime for turning Python-defined
model graphs into standalone native artifacts for CPU, CUDA, and ROCm targets.
This repo is a compiler/runtime lab, not a polished end-user framework.

## Core Rules

- Keep changes bounded and reviewable.
- Prefer correctness, inspectability, and reproducibility over polish.
- Use v1 as a behavioral reference, not a structural template.
- Prefer explicit artifact-visible state over hidden mutable compiler state.
- Do not expand public surface area casually. Finish and validate existing
  systems before adding more.
- Tests and executable harnesses outrank stale prose docs when they disagree.

## Read First

Before making broad changes, read the relevant current codepaths and the
project docs that apply to the task:

- `README.md`
- `docs/project_invariants.md`
- `docs/provider_contract.md`
- `docs/op_admission.md`
- `docs/model_pipeline_benchmarking.md`

Then read the local source and tests nearest the requested behavior.

## Working Style

- Use `rg` / `rg --files` first for search.
- Scope searches to first-party code by default. Exclude `third_party/`,
  `build/`, `.venv/`, `artifacts/`, `_tmp/`, `.tmp/`, and
  `.pytest_artifacts/` unless the task is specifically about them.
- Prefer repository-local patterns, helper APIs, naming, and organization over
  new abstractions.
- Keep edits close to the behavior being changed.
- Do not refactor unrelated areas unless it is required to complete the task
  safely.
- Do not edit vendored code under `third_party/` unless the task explicitly
  targets it or the root cause clearly lives there.

## Validation

- Validation should match the risk of the change.
- Start with the narrowest relevant check.
- Prefer stable first-party checks such as `tests/ir`, `tests/cpu`, targeted
  scaffold or manifest tests, and runtime benchmark tests before gated GPU
  contract runs when that still exercises the change.
- Separate artifact contract validation, artifact execution, parity against
  external references, and throughput claims.
- For performance work, name the timing surface explicitly.
- For op-level and whole-artifact throughput, prefer
  `benchmark_numpy`, `benchmark_device_pointers`, `dino_session_benchmark`, or
  the existing benchmark suites over timing around `run_numpy`.
- Host timing around `run_numpy` or `run_device_pointers` is acceptable for
  pipeline-orchestration benchmarks, but label it as pipeline timing rather
  than native artifact benchmark timing.
- Distinguish compiler/runtime regressions from missing toolchains, drivers, or
  backend environment issues.

## Human-In-The-Loop Boundaries

- Stay anchored to the current user request.
- If two consecutive investigation passes in the same narrow area do not
  produce a failing test, a concrete contract gap, or a user-directed reason to
  continue, stop and re-anchor with the user.
- If the user names a specific local or private repo, artifact, branch, file,
  or source, use local or authenticated access first and ask before
  substituting or skipping it.

## Git And Edits

- Use `apply_patch` for manual file edits.
- Assume the worktree may be dirty. Never revert user changes unless explicitly
  asked.
- Prefer non-interactive git commands.
- Do not create branches or commits unless the user asks.
- Never use destructive git commands such as `git reset --hard` or
  `git checkout --` unless explicitly requested.
