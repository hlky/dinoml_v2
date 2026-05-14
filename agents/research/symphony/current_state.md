# Current Agent Process State

## What Is Already Strong

DinoML v2 already has many of the ingredients needed for issue-driven agentic
development:

- `AGENTS.md` defines a steering hierarchy and required autonomous loop.
- `agents/CURRENT_FOCUS.md` gives a human-directed active lane.
- `agents/NEXT_CANDIDATE_WORK.md` provides a ranked queue when no stronger
  override exists.
- `agents/INVARIANTS.md`, `agents/PROVIDER_CONTRACT.md`, and
  `agents/OP_ADMISSION.md` turn vague implementation work into checkable rules.
- `agents/PM_PROMPT.md` has a supervisor-agent concept with worker loops,
  external developer review, stakeholder pressure, anti-drift checks, and model
  allocation policy.
- `agents/plans/transformers/`, `agents/plans/diffusers/`, and
  `agents/plans/auxiliary/` contain a substantial research base that can seed
  many future issues.

This is already more mature than a flat TODO list. It is a project memory system
with norms, gates, and a current direction.

## Main Gaps Versus Symphony

The current process is still prompt-driven rather than issue-driven.

Missing pieces:

- no repo-owned `WORKFLOW.md` with tracker, workspace, hook, and agent config
- no issue-state model such as `Todo -> In Progress -> Human Review -> Rework`
- no per-issue isolated workspace or git worktree policy
- no persistent issue workpad that records plan, acceptance criteria,
  validation, notes, and confusions
- no structured retry, stall detection, or reconciliation loop
- no standard promotion path from research report to executable issue
- no observability layer for multiple concurrent runs
- no normalized issue taxonomy separating exploration, implementation, review,
  validation, and design-first work

The largest current process pressure point is `agents/NEXT_CANDIDATE_WORK.md`.
It is valuable, but it is carrying both loop history and active scheduling. As
the research corpus grows, that file should stop being the main backlog and
become a high-level steering pointer into the actual issue system.

## Research Corpus Readiness

The model and UI research can be converted into issue material, but not by
creating one active issue per report immediately.

Recommended interpretation:

- Reports under `agents/plans/transformers/*/report.md` are source-grounded
  research artifacts. They should usually produce parent issues or roadmap
  issues, not immediate implementation work by themselves.
- Reports under `agents/plans/diffusers/*/report.md` often span pipelines,
  schedulers, loaders, components, and runtime state. They should be split into
  component or runtime-stage subissues.
- `agents/plans/auxiliary/` and `agents/research/uis/` are best used as
  product-relevance and integration-surface inputs.
- `agents/plans/*/report_review.md` and trackers are useful coordinator memory
  and should inform issue creation priorities.

## Readiness Assessment

DinoML v2 is ready to start an issue-based backlog now, with a small pilot.

Good first issue lanes:

1. Current top `NEXT_CANDIDATE_WORK.md` GGUF/native load-path parity item.
2. One CUTLASS/provider maturity review.
3. One runtime/container lifecycle review.
4. One Transformers report-clustering exploration.
5. One Diffusers first-implementation-lane exploration.

Avoid immediately flooding the tracker with hundreds of model-family issues.
Instead, use exploratory agents to cluster reports into issue trees.

