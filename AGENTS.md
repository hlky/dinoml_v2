# Agent Instructions

This repository is developed with autonomous agents. Treat the files under `agents/` as project memory and steering state.

## Steering hierarchy

Use the agent docs in this order:

```text
AGENTS.md                  boot protocol
CURRENT_FOCUS.md           optional human override / active direction
NEXT_CANDIDATE_WORK.md     default ranked queue when no override exists
BLOCKED_OR_DEFERRED.md     hard fences / design-sensitive traps
plans/*.md                 long-form project memory
```

`agents/CURRENT_FOCUS.md` may be empty. If it is empty, do not treat that as stale documentation; fall back to `agents/NEXT_CANDIDATE_WORK.md` and the active plans.

## Required loop

At the beginning of each work loop:

1. Review the current repository state and recent commits.
2. Read the steering docs under `agents/`.
3. Check whether the current user request names external or private context;
   use authenticated/local access first, and ask before substituting or
   skipping it.
4. Read active plans under `agents/plans/`.
5. Pick one bounded, high-value task.
6. Use subagents for investigation, implementation, and review when useful.
7. Validate the work with targeted tests or explain why validation could not be run.
8. Update relevant docs/checklists, including `agents/NEXT_CANDIDATE_WORK.md` when the ranked queue changes.
9. Commit completed work with a clear summary.
10. Update Codex Progress.
11. Continue the loop unless blocked.

## Monitored autonomy and clarification gates

Autonomy means continuing through known work, not silently changing direction
when a user-specified source, repo, dependency, artifact, or design input cannot
be accessed or interpreted.

Use `request_user_input` before continuing when:

- A user names a specific private repo, branch, file, tool, package, config, or
  artifact and authenticated/local access fails or is ambiguous.
- A simple access/auth/path problem would otherwise cause the agent to skip the
  requested source, use an unrelated public substitute, or design a workaround.
- The next step would materially change the requested direction, support claim,
  backend/provider choice, or acceptance bar.
- The docs and the user's current instruction point in different directions.

Before asking, try the appropriate authenticated/local access path when obvious,
such as `gh` or the GitHub connector for private GitHub repositories. If that
access fails, ask for direction instead of inferring from public web 404s, search
misses, missing local files, or unauthenticated tooling errors.

## Required reading

Read these before planning work:

- `agents/OPERATING_LOOP.md`
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

## Documentation friction rule

Docs are project memory, not the product. Minor documentation friction should not become the main task unless it blocks implementation, validation, or user-requested work.

When documentation friction is found:

1. Fix it opportunistically if the change is small and directly related to the current task.
2. Otherwise record it briefly in `agents/NEXT_CANDIDATE_WORK.md` only if it is likely to confuse future work.
3. Continue with the selected bounded implementation, test, or review task.
4. Do not spend a work loop only cleaning stale links, wording, empty optional sections, or minor reference drift unless explicitly requested.
