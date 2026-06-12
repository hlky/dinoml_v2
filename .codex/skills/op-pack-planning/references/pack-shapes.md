# Pack Shapes

All new packs created through this workflow should live under:

- `.work/op/`

Choose one of the following pack shapes.

## Shape A: compact pack

Use for:

- `thin-frontend-rewrite`
- small `new-ir-op` tasks
- tasks with limited backend implications

Recommended files:

- `00_manifest.md`
- `00_overall_task.md`
- `01_scope.md`
- `02_known_pitfalls.md`
- `03_tests_and_acceptance.md`
- `04_verification_commands.md`
- `goal_prompt.md`

## Shape B: full execution pack

Use for:

- `real-backend-lowering-work`
- provider-stack tasks
- tasks with significant ROCm and CUDA verification requirements
- tasks where failure-mode guardrails need to be explicit

Recommended files:

- `00_manifest.md`
- `00_overall_task.md`
- `01_invariants.md`
- `02_known_pitfalls.md`
- `03_progress_reporting.md`
- `04_task_public_api_and_reference.md`
- backend-specific task files as needed
- lowering, manifest, or codegen task file as needed
- `08_task_tests_and_acceptance.md`
- `09_verification_commands.md`
- `goal_prompt.md`

## Shape selection rules

Use Shape B when any of the following is true:

- the task extends a provider-backed architecture
- runtime parity, not just lowering, is part of completion
- ROCm and CUDA support claims matter to task completion
- backend work can be partially implemented in misleading ways
- the task has known recurring failure modes that deserve pack-local guardrails

Use Shape A only when the simpler pack is unlikely to let the task drift or be completed dishonestly.

## Naming guidance

Prefer pack folder names that are stable and specific:

- single op: `masked_fill`
- op family: `conv_transpose1d`
- grouped work only when tightly coupled in API, lowering, runtime shape, and verification

Do not group unrelated ops just because they appeared together in the same usage report.

## Required pack content

Every pack must state:

- covered ops
- explicit non-scope
- required touchpoints
- provider expectations when applicable
- parity oracle
- required verification surface
- what does and does not count as completion

Do not rely on implied scope.

## Provider rules

For conv, GEMM, and BMM tasks that fit the existing provider architecture:

- ROCm should use the existing CK-based provider path
- CUDA should use the existing CUTLASS-based provider path

Do not invent a separate ad hoc backend path when the task belongs in an existing provider-backed family.

If the current provider architecture cannot support the scoped task honestly, record that as a blocker in the pack. Do not silently substitute a different implementation style and still present the task as completed.

## Remote CUDA verification rules

When a pack requires CUDA runtime verification:

- use the repo-local skill `H:\dinoml_v2\.codex\skills\runpod-codex-remote\SKILL.md`
- prefer the validated image `hlky/dinoml:ubuntu-nodeps`
- prefer reusing the prebaked repo path with `--existing-project-path /opt/src/dinoml_v2`
- recommended maximum hourly budget is `$0.50/hour`
- prefer these GPUs within budget when available:
  - `A40`
  - `3090`
  - `A4000`
  - `A5000`

The pack should say explicitly when remote CUDA verification is required for completion.
