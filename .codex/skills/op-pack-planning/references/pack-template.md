# Pack Generation Template

Use this template when creating a new `.work/op/<pack-name>/` directory.

Choose the compact or full shape based on `pack-shapes.md`.

## Suggested directory

`.work/op/<pack-name>/`

## Minimal compact-pack skeleton

### `00_manifest.md`

- task name
- covered ops
- classification
- priority
- commonality
- ease

### `00_overall_task.md`

Include:

- short why
- exact completion target
- explicit non-scope
- required touchpoints
- allowed implementation style
- forbidden shortcuts

### `01_scope.md`

Include:

- public API expectations
- for torch-family targets, the exact official PyTorch API reference URL for the scoped op
- whether decomposition is acceptable
- whether new IR is required
- backend truth expected at completion

### `02_known_pitfalls.md`

Include only relevant pitfalls from `common-failure-modes.md`.

### `03_tests_and_acceptance.md`

List:

- parity oracle
- required tests
- what counts as acceptance
- what does not count as acceptance

### `04_verification_commands.md`

List narrow first commands, then broader required verification.

### `goal_prompt.md`

Short execution-oriented prompt that:

- names the pack
- tells the agent to read the pack-local files first
- states the completion boundary clearly

## Full-pack skeleton

### `00_manifest.md`

- task name
- covered ops
- classification
- priority
- commonality
- ease
- pack shape

### `00_overall_task.md`

Include:

- why this work matters
- covered ops
- explicit non-scope
- required architecture
- required provider path when applicable
- completion contract

### `01_invariants.md`

Include:

- architecture invariants
- backend honesty requirements
- provider or lowering constraints
- any task-specific invariants

### `02_known_pitfalls.md`

Select relevant failures from `common-failure-modes.md` and write them as task-local warnings.

### `03_progress_reporting.md`

Require updates to distinguish:

- known
- unknown
- next discriminating step

and to separate:

- semantic correctness
- backend availability
- backend runtime parity

### `04_task_public_api_and_reference.md`

Include:

- user-facing or frontend surface
- reference oracle source
- for torch-family targets, the exact official PyTorch API reference URL for the scoped op
- v1 reading budget if relevant

### Backend-specific files

Create only what the task needs, for example:

- `05_task_rocm_backend.md`
- `06_task_cuda_backend.md`
- `07_task_manifest_lowering_and_codegen.md`

Each should say:

- what must be implemented
- what does not satisfy the task
- what backend evidence is required
- which provider path is required when the task belongs to an existing provider-backed family

### `08_task_tests_and_acceptance.md`

Include:

- narrow tests first
- broader parity tests
- required backend runtime checks
- explicit acceptance checklist

### `09_verification_commands.md`

List concrete commands in the intended order.

When CUDA is required, include:

- the remote verification expectation
- the requirement to use the repo-local `runpod-codex-remote` skill
- the budget ceiling
- the preferred GPU list
- the validated image and prebaked repo-path preference

### `goal_prompt.md`

Keep it short and explicit:

- read pack-local files first
- keep scope to covered ops only
- use named architecture
- do not claim completion without required verification

## Template writing rules

- Write direct instructions, not general advice.
- Prefer explicit non-scope over long prose.
- If an implementation shortcut is forbidden, say so plainly.
- If a backend is required for completion, say so plainly.
- If a backend is out of scope, say so plainly.
