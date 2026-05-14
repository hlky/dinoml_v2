# Issue Backlog Model

## Tracker Shape

Use the issue tracker as the control plane. The repository docs remain project
memory, but executable work should live as issues with clear state, ownership,
acceptance criteria, and validation.

Suggested top-level projects or epics:

- Agent process and orchestration
- Runtime and container contract
- GGUF and encoded constants
- CUTLASS and provider maturity
- Core op coverage
- Transformers model-family support
- Diffusers pipeline support
- Auxiliary and UI model support
- Developer onboarding and examples

## Issue Types

### Epic

Large product or platform objective. Should not be directly picked up by a
coding agent except to create or maintain subissues.

Examples:

- `Epic: GGUF-backed CUDA weight runtime lifecycle`
- `Epic: Stable Diffusion 1.5 first executable path`
- `Epic: Transformers encoder baseline support`

### Exploration

Read-only or mostly read-only work that turns fuzzy areas into executable
issues. Exploration agents should not implement unless the issue explicitly
allows it.

Examples:

- `Explore CUTLASS provider maturity gaps against provider contract`
- `Cluster Transformers reports by missing primitive families`
- `Review Diffusers reports and propose first three implementation epics`

### Design-First

Work that should produce a design note or decision proposal before code.

Examples:

- `Design explicit offload policy beyond manual runtime load`
- `Design adapter-state schema for LoRA/textual inversion runtime mutation`

### Implementation

Bounded code change with targeted tests.

Examples:

- `Fix native GGUF manual-runtime-load parity`
- `Add CUDA GGUF linear load-run-unload-reload regression`
- `Add provider cache stale-key rejection regression`

### Validation

Adds or hardens a proof without necessarily changing product behavior.

Examples:

- `Add CUDA integration coverage for mixed dense plus manual GGUF constants`
- `Add generated-code regression for profile-selected CUTLASS candidate use`

### Review

Skeptical pass over a recent commit, issue lane, or plan.

Examples:

- `Review latest GGUF runtime-dequant commits for hidden state leaks`
- `Review op checklist claims against current tests`

## Suggested States

Minimum useful state machine:

- `Backlog`: known but not ready for execution
- `Todo`: ready for an agent to claim
- `In Progress`: an agent is actively working
- `Human Review`: PR or report is ready for human review
- `Rework`: human or reviewer requested changes
- `Merging`: approved and ready to land
- `Done`: terminal
- `Blocked`: true external or design blocker

For a simpler pilot, use only:

- `Backlog`
- `Todo`
- `In Progress`
- `Review`
- `Done`
- `Blocked`

## Labels

Recommended labels:

- `type:exploration`
- `type:implementation`
- `type:validation`
- `type:review`
- `type:design`
- `area:gguf`
- `area:cutlass`
- `area:runtime`
- `area:ops`
- `area:transformers`
- `area:diffusers`
- `area:auxiliary`
- `risk:architecture`
- `risk:lifecycle`
- `risk:cuda`
- `needs:gpu`
- `needs:human-design`
- `ready-for-agent`
- `blocked`

## Exploration Issue Output Contract

Every exploration issue should produce the same shape so that a project manager
agent can convert findings into implementation issues.

```md
## Findings

- Finding 1
- Finding 2

## Top Executable Issues

1. Title:
   Scope:
   Acceptance criteria:
   Validation:
   Dependencies:
   Likely touched systems:

2. Title:
   Scope:
   Acceptance criteria:
   Validation:
   Dependencies:
   Likely touched systems:

## Design-First Or Blocked Items

- Item:
  Why blocked:
  What would unblock:
  Suggested home:

## Not Worth Ticketing Yet

- Item:
  Reason:

## Source Material Read

- File or URL
- File or URL
```

## Model Report Promotion Pattern

Do not create one active implementation issue per model report. Use a staged
promotion path:

1. Report exists under `agents/plans/...`.
2. Exploration agent clusters related reports by shared missing primitives.
3. PM creates one parent issue per executable lane.
4. Parent issue gets subissues for:
   - parser/config loading
   - operator coverage
   - provider/runtime support
   - parity tests
   - end-to-end example or integration regression
5. Implementation agents claim subissues only when acceptance criteria and
   validation are explicit.

Example for Transformers:

```text
Epic: BERT-like encoder baseline
  Exploration: Cluster BERT/RoBERTa/ALBERT/ELECTRA shared requirements
  Implementation: Add embedding + position/token-type parse path
  Implementation: Add one encoder block parity path
  Validation: Add tiny random BERT full-encoder parity test
  Follow-up: Add task heads only after base encoder contract is stable
```

Example for Diffusers:

```text
Epic: SD1.5 first executable path
  Exploration: Split SD1.5 report into component/runtime stages
  Implementation: AutoencoderKL decode-only path
  Implementation: UNet residual block primitive coverage
  Implementation: Scheduler state and CFG arithmetic plan
  Validation: Tiny latent decode or denoiser-stage parity test
```

