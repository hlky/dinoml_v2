# Agent Instructions

This repository is developed with autonomous agents. Treat the files under `agents/` as project memory and steering state.

## Required loop

At the beginning of each work loop:

1. Review the current repository state and recent commits.
2. Read the steering docs under `agents/`.
3. Read active plans under `agents/plans/`.
4. Pick one bounded, high-value task.
5. Use subagents for investigation, implementation, and review when useful.
6. Validate the work with targeted tests or explain why validation could not be run.
7. Update relevant docs/checklists.
8. Commit completed work with a clear summary.
9. Update Codex Progress.
10. Continue the loop unless blocked.

## Required reading

Read these before planning work:

- `agents/AGENT_OPERATING_LOOP.md`
- `agents/INVARIANTS.md`
- `agents/CURRENT_FOCUS.md`
- `agents/BLOCKED_OR_DEFERRED.md`
- `agents/PROVIDER_CONTRACT.md`
- `agents/OP_ADMISSION.md`
- `agents/NEXT_CANDIDATE_WORK.md`
- `agents/plans/gemm_cutlass_plan.md`
- `agents/plans/op_porting_checklist.md`
- `agents/plans/v1_gap_audit.md`

## Core rules

- Docs are project memory. If behavior changes, update the relevant doc or checklist in the same commit.
- Tests are truth. Do not mark work complete unless validation supports it.
- Use v1 as a behavioral reference, not as a design template.
- Preserve v2 architecture: prefer explicit artifact-visible state over hidden mutable compiler state.
- Provider decisions should flow through manifests, profile reports, execution plans, and generated lowering.
- Do not expand public surface area without satisfying the op/provider admission guidance.
- Prefer stabilizing recent work before adding more features.
