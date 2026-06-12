---
name: op-pack-implementation
description: Use when implementing an existing DinoML `.work/op/` pack. This skill enforces pack-first execution, required architecture adherence, honest backend-support claims, bounded reading, and explicit completion checks so implementation work does not stop early at frontend parity, lowering-only coverage, or partially verified backend support.
---

# Op Pack Implementation

Use this skill when the task is to execute an existing DinoML `.work/op/` pack.

This skill is for implementation, not candidate selection or pack drafting. If the task is to choose the next ops or create a new pack, use `op-pack-planning` instead.

## Read first

Read the named pack files first.

At minimum, identify and read:

- the pack overview or overall task file
- the scope or invariants file
- the pitfalls file
- the tests and acceptance file
- the verification commands file
- the goal prompt if present

If the pack includes backend-specific task files, read the ones that apply to the claimed implementation surface.

Do not begin by rereading broad repo docs when the pack already gives the local contract.

When the work includes adding or revising tests, also read and follow the repo-local skill:

- `H:\dinoml_v2\.codex\skills\op-test-implementation\SKILL.md`

## Required execution workflow

Follow this order.

### 1) Extract the task contract from the pack

Before editing, write down internally:

- covered ops
- explicit non-scope
- shape contract: static-only, static-rank dynamic, runtime-shape-dependent, or deferred
- target public API surface
- official API reference URL when the target op comes from `torch`, `torch.Tensor`, `torch.nn`, or `torch.nn.functional`
- required v2 touchpoint stack
- required architecture or provider path
- parity oracle
- required verification backends
- what counts as completion
- what does not count as completion

If any of these are unclear, resolve that before broad implementation work.
If a torch-family op pack is missing the official PyTorch API reference URL, treat that as a pack contract gap and resolve it before broad implementation work.

### 2) Read only the necessary local codepaths

Start from the touchpoints named by the pack.

Read the minimum needed to identify:

- the current contract
- the edit site
- the validation path

Do not broaden the read set just to feel safer once the target module and validation path are clear.

### 3) Implement within the required architecture

If the pack names a specific stack, subsystem, or provider architecture, implement the task there.

Examples:

- provider-backed conv work should stay in the conv provider architecture
- provider-backed GEMM or BMM work should stay in the existing provider-backed family
- template-backed lowering should stay template-backed unless the task explicitly says otherwise

Do not treat “not already present” as permission to switch implementation style.

If the required architecture cannot support the task within scope, record the blocker explicitly and keep the task incomplete.

### 4) Keep backend support claims honest

Do not present a backend as supported unless the implemented path is actually wired and validated at the level required by the pack.

These do not count as backend support by themselves:

- frontend acceptance
- reference parity only
- lowering or render success
- scaffold or manifest generation
- code inspection

If a backend is intentionally unsupported during development, reject it explicitly and test that rejection when appropriate.

### 5) Validate in the order the pack expects

Prefer the narrowest discriminating check first.

Keep these surfaces separate:

- semantic or compiler correctness
- reference parity
- lowering or render checks
- runtime parity
- performance measurement

Do not let one surface stand in for another.

When implementing the test files themselves, use the repo-local `op-test-implementation` skill to choose file layout, shared parity helpers, and backend-specific test boundaries.

### 6) Check completion against the pack, not against partial progress

Before calling the task complete, verify that the pack’s completion boundary is satisfied.

Do not silently narrow scope because the hard part is inconvenient.
Do not treat partial backend work as complete unless the pack explicitly allows that boundary.

## Backend execution rules

### ROCm

For work that claims ROCm support:

- probe and use local ROCm when required by the pack
- distinguish backend availability from op-specific parity
- do not describe ROCm as unavailable unless the direct probes fail

### CUDA

For work that claims CUDA support:

- use the repo-local `runpod-codex-remote` skill when remote CUDA verification is required
- prefer the validated image `hlky/dinoml:ubuntu-nodeps`
- prefer reusing `/opt/src/dinoml_v2` with `--existing-project-path /opt/src/dinoml_v2`
- stay within the usual verification budget guidance when the pack uses the standard repo constraints

Do not mark CUDA support complete when remote CUDA validation required by the pack is still pending.

### CK and CUTLASS

For conv, GEMM, and BMM tasks that fit the existing provider architecture:

- ROCm should use the existing CK-based provider path
- CUDA should use the existing CUTLASS-based provider path

Do not replace an existing provider-backed family with an ad hoc backend path unless the pack explicitly allows that.

## Common failure modes to block

Watch for these and stop them explicitly:

- frontend-only completion
- lowering-only support claims
- architecture switching
- silent scope shrink
- rewrite as dodge for real backend work
- broad representation work disguised as a routine op
- static-shape-only completion for a task that requires runtime shape support
- CUDA pending treated as done
- v1 structure imported directly into v2
- validation that does not exercise the changed path

## Reporting style

When reporting progress, keep it specific:

- what is known
- what is unknown
- what the next discriminating step is

When reporting blockers, distinguish:

- semantic bug
- missing implementation
- environment or toolchain issue
- pending required verification

## Style

- Stay bounded to the pack.
- Prefer small auditable diffs.
- Prefer explicit failures over implicit fallback.
- Do not opportunistically refactor unrelated areas.
- Do not claim certainty that the current evidence has not earned.
