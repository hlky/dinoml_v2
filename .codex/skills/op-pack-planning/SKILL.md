---
name: op-pack-planning
description: Use when selecting the next DinoML v2 op implementation candidates from usage or missing-op audit data and turning them into new scoped `.work/op/` implementation packs. This skill filters out already-covered ops, rewrite-only cases, and broader representation traps, then drafts bounded DinoML-specific packs with explicit scope, acceptance criteria, and verification requirements.
---

# Op Pack Planning

Use this skill when the task is to:

- rank next op candidates from usage reports or missing-op audits
- decide whether a candidate is already covered, rewrite-only, a new IR op, or real backend work
- convert selected candidates into new DinoML `.work/op/` packs
- draft bounded implementation packs that are hard to misread or partially complete

Do not use this skill for ordinary implementation of an already-scoped op pack. Use it to choose and package the work first.

## Read first

Always read:

- `references/classification-rubric.md`
- `references/pack-shapes.md`
- `references/common-failure-modes.md`
- `references/source-data.md`

Read `references/pack-template.md` when you are drafting a new `.work/op/` pack.

Then read only the smallest local inputs needed for the current request:

- the usage or missing-op report the user named
- the bundled audit snapshot when the user wants to start from the current Transformers or Diffusers audit baseline
- the current v2 op surface under `src/dinoml/ops`
- nearby lowering or test files only when needed to determine whether an op is already covered or would need real backend work

## Required workflow

Follow this order.

### 1) Build the candidate table

For each candidate op, record:

- usage or commonality signal from the provided audit data
- whether v2 already supports it directly
- whether it is already covered by a thin rewrite or composition
- whether it needs a new frontend mapping only
- whether it needs a new IR op
- whether it needs real lowering, runtime, or backend work
- whether it implies broader representation work
- whether it has dynamic-output risk
- whether it has complex-dtype risk
- recommended pack shape

Do not treat surface spelling differences as missing support until checked against the current v2 surface.

### 2) Classify each candidate

Use exactly one of these labels:

- `already-covered`
- `thin-frontend-rewrite`
- `new-ir-op`
- `real-backend-lowering-work`
- `broader-representation-work`
- `defer`

Apply the rubric from `references/classification-rubric.md`.

Do not silently upgrade a rewrite-only candidate into a full backend task.
Do not silently downgrade a real backend task into a frontend-only task.

### 3) Choose pack format

Use the pack-shape rules from `references/pack-shapes.md`.

All new packs created by this skill should live under:

- `.work/op/`

Do not add new work to `.work/ops/`.

### 4) Scope the task explicitly

For every selected task, state:

- covered ops
- explicit non-scope
- required v2 touchpoint stack
- allowed implementation style
- forbidden shortcuts
- required parity oracle
- required verification backends
- completion boundary

If decomposition is acceptable completion, say so directly.
If decomposition is not acceptable completion, say so directly.

### 5) Encode failure-mode guardrails

Use `references/common-failure-modes.md` and write pack-local guidance that blocks the failure modes relevant to the chosen task.

## Output requirements

When the user asks for selection only, return:

- the selected ops
- classification for each
- why each was kept
- why obvious exclusions were dropped

When the user asks for pack creation, produce:

- the chosen pack shape
- the file list to create under `.work/op/`
- draft content aligned to the pack template

## Verification planning rules

Plan verification honestly from the start.

- CPU or reference verification is required but not sufficient when the task claims GPU support.
- Local ROCm verification is required unless the scoped pack explicitly removes it.
- CUDA verification must be planned through the repository workflow, not hand-waved.
- When remote CUDA verification is required, use the repo-local `runpod-codex-remote` skill and follow the budget and GPU guidance in `references/pack-shapes.md`.
- Lowering or render coverage does not count as runtime parity.
- If a task is backend work, the pack must say so clearly and must not let the implementation stop at API or scaffold coverage.

## Style

Be concrete and conservative.

- Prefer bounded tasks over roadmap prose.
- Prefer explicit non-scope over implied future work.
- Prefer “not selected because already covered by X” over vague dismissal.
- Prefer “defer because this implies broader representation work” over pretending the op is routine.

If uncertain, tighten scope rather than broadening it.
