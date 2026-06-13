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

Unless there is a real, explicit reason to narrow scope, plan DinoML op implementation packs as three-backend work:

- CPU
- CUDA
- ROCm

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
- whether it needs a new frontend convenience or translation surface only
- whether it needs a new IR op
- whether it needs real lowering, runtime, or backend work
- whether completion requires a real provider or kernel implementation
- whether it implies broader representation work
- whether it has dynamic-output risk
- whether it has complex-dtype risk
- recommended pack shape

Do not treat surface spelling differences as missing support until checked against the current v2 surface.
For Torch-spelling candidates, also check whether the model author can already call the existing DinoML op directly with no extra translation rule. If yes, prefer `already-covered` over creating a frontend-only pack.

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
Do not leave the pack ambiguous about kernel expectations.
For any candidate that is not `already-covered` or `thin-frontend-rewrite`, prefer a real kernel/provider implementation.

### 3) Choose pack format

Use the pack-shape rules from `references/pack-shapes.md`.

All new packs created by this skill should live under:

- `.work/op/`

Do not add new work to `.work/ops/`.

### 4) Scope the task explicitly

For every selected task, state:

- covered ops
- explicit non-scope
- shape contract: static-only, static-rank dynamic, runtime-shape-dependent, or deferred
- target public API surface
- official API reference URL when the target op comes from `torch`, `torch.Tensor`, `torch.nn`, or `torch.nn.functional`
- required v2 touchpoint stack
- kernel expectation: frontend rewrite only for thin rewrites, otherwise real provider/kernel work required
- allowed implementation style
- forbidden shortcuts
- required parity oracle
- required verification backends
- completion boundary

For ordinary DinoML op packs, required verification backends should be CPU, CUDA, and ROCm. If any backend is intentionally excluded, say so explicitly in non-scope and explain why. Do not leave backend scope to implication.

If decomposition is acceptable completion, say so directly.
If decomposition is not acceptable completion, say so directly.
If the task requires a real backend/provider/kernel implementation for honest completion, say that directly.
For any pack that is not `already-covered` or `thin-frontend-rewrite`, composition through existing ops does not count as completion.
For any non-rewrite DinoML op pack, partial backend completion does not count as completion. Say directly that CPU-only, CPU+ROCm-only, or CPU+CUDA-only completion is insufficient unless the pack explicitly narrows scope.

For thin frontend or rewrite-only packs, be stricter:

- name the concrete user-visible API file or module to edit when known, especially under `src/dinoml/nn/` or another named local surface
- name the exact existing DinoML op(s) or helper(s) that admitted calls must route to or rewrite into
- say why the pack is worth having instead of calling the existing DinoML op directly from model integrations
- say directly that no parallel backend, kernel, or duplicate runtime path should be added
- avoid role words such as `frontend`, `export`, `binding`, `Torch-facing`, `integration`, or `pipeline` unless they are immediately anchored to a concrete local file or symbol
- prefer `Add <api> in <file>` over `Add a Torch-facing path`
- prefer `Route admitted calls to <existing op>` over `Wire the frontend mapping`
- if the exact file is not yet known, say that the pack must identify the concrete local API file before implementation rather than inventing subsystem wording

Use a frontend-only pack only when at least one of these is true:

- the upstream Torch spelling is common enough that supporting it removes repeated translation work in model integrations
- DinoML's canonical op name or calling convention differs materially from the common Torch spelling, such as creation helpers routed to `full` or `matmul` routed to the appropriate GEMM or BMM op family
- the convenience surface belongs in an existing named local API module such as `src/dinoml/nn/functional.py` or `src/dinoml/nn/__init__.py`

Do not create a frontend-only pack when the honest answer is just "call the existing DinoML op directly". For example, if `torch.where` adds no real convenience beyond `ops.where`, classify it as `already-covered` rather than drafting a new thin pack.

For `new-ir-op` versus `real-backend-lowering-work`, be stricter about kernel truth:

- use `new-ir-op` only when the main planning distinction is that a new explicit IR op is needed in addition to the required real backend/kernel work
- use `real-backend-lowering-work` when the primary difficulty is provider, lowering, runtime, or kernel implementation
- do not describe a task as having an "honest lowering path" if that wording would make composed execution sound acceptable as the end state
- do not use a broad composition of existing ops as the planned end state for a newly selected op pack

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
- the exact official PyTorch API reference link for each scoped torch-family op when applicable

## Verification planning rules

Plan verification honestly from the start.

- CPU or reference verification is required but not sufficient when the task claims GPU support.
- Local ROCm verification is required unless the scoped pack explicitly removes it.
- CUDA verification must be planned through the repository remote-CUDA workflow, not hand-waved and not replaced by local inference.
- When remote CUDA verification is required, use the repo-local `runpod-codex-remote` skill and follow the budget and GPU guidance in `references/pack-shapes.md`.
- Lowering or render coverage does not count as runtime parity.
- If a task is backend work, the pack must say so clearly and must not let the implementation stop at API or scaffold coverage.
- A ROCm-only local machine does not justify scoping CUDA out. It changes the execution venue for CUDA verification, not the required backend surface.
- Do not draft packs with wording that lets "unvalidated backend claim" be misread as permission to avoid implementing CUDA. The pack should say that CUDA remains required and must be verified through the remote workflow.

## Style

Be concrete and conservative.

- Prefer bounded tasks over roadmap prose.
- Prefer explicit non-scope over implied future work.
- Prefer "not selected because already covered by X" over vague dismissal.
- Prefer "defer because this implies broader representation work" over pretending the op is routine.
- For thin frontend packs, prefer file-and-op wording such as `Add \`interpolate\` in \`src/.../functional.py\`; route admitted calls to \`upsampling2d\`` over abstract wording such as `Add a Torch-facing interpolate path`.

If uncertain, tighten scope rather than broadening it.
