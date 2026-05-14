# Agent Runner Options

## Practical Recommendation

Use Codex Desktop for supervision and manual inspection, but use the Codex CLI
for programmable execution.

Recommended progression:

1. Pilot with `codex exec` and one git worktree per issue.
2. Add a thin local scheduler that reads issues and launches `codex exec`.
3. Move to `codex app-server` when retry, streaming, status, and long-running
   sessions matter.
4. Consider Codex Cloud only after the local workflow is stable and the remote
   environment can reproduce the needed dependencies.

## Option 1: Codex Desktop

Best for:

- human-supervised exploration
- reviewing plans and diffs
- debugging complex agent failures
- manually launching a small number of sessions

Limitations:

- not the best programmable launch surface
- not a natural fit for issue polling, retries, and per-issue workspaces
- harder to observe and coordinate many concurrent runs

Desktop should remain the cockpit, not the orchestrator.

## Option 2: `codex exec`

Best for the first pilot.

Local help shows this command runs Codex non-interactively and supports:

- `--cd <DIR>`
- `--sandbox <MODE>`
- `--ask-for-approval <POLICY>`
- `--json`
- `--output-last-message <FILE>`
- `--model <MODEL>`
- `--output-schema <FILE>`

Example:

```powershell
codex exec `
  --cd H:\dinoml_v2_worktrees\DINO-123 `
  --sandbox workspace-write `
  --ask-for-approval never `
  --json `
  --output-last-message H:\dinoml_v2_runs\DINO-123\final.md `
  "Work issue DINO-123. Read AGENTS.md and follow the repository steering docs."
```

Recommended pilot architecture:

```text
issue tracker or local queue
  -> small scheduler script
  -> create/update git worktree
  -> render issue prompt from WORKFLOW template
  -> run codex exec
  -> capture JSONL log and final message
  -> update issue/workpad manually or through tracker API
```

Pros:

- simple
- works with the installed CLI
- easy to reason about on Windows
- enough for exploratory agents and bounded implementation runs

Cons:

- less streaming/control than app-server
- scheduler must parse process output and manage retries itself
- continuation semantics are coarser

## Option 3: `codex app-server`

Best for a real Symphony-style orchestrator.

Local help shows support for:

- `codex app-server --listen stdio://`
- `codex app-server --listen ws://IP:PORT`
- `codex app-server generate-json-schema --out <DIR>`
- `codex app-server generate-ts --out <DIR>`

This matches the architecture described by Symphony: an orchestrator creates or
reuses a workspace, renders a workflow prompt from issue data, launches a Codex
app-server session, streams updates, tracks run state, and retries or cancels as
needed.

Use app-server when DinoML needs:

- multiple concurrent agents
- event-level observability
- stall detection
- continuation turns
- issue-state reconciliation
- durable run metadata
- a status surface

Pros:

- closest to Symphony
- designed for orchestration
- supports generated protocol schemas/bindings

Cons:

- more code to build and operate
- requires an orchestrator implementation
- still experimental in the inspected CLI

## Option 4: Codex Cloud

Local help shows `codex cloud exec` can submit tasks with:

- `--env <ENV_ID>`
- `--branch <BRANCH>`
- `--attempts <N>`

Example:

```powershell
codex cloud exec --env <ENV_ID> --branch main --attempts 1 "Work issue DINO-123."
```

Cloud is useful when:

- the environment is reproducible remotely
- the task is not dependent on local-only paths or hardware
- best-of-N attempts are useful

For DinoML's current state, local execution is probably safer because the repo
references local paths, CUDA/runtime details, and adjacent checkouts.

## Workspace Strategy

Use one git worktree per issue.

Suggested layout:

```text
H:\dinoml_v2
H:\dinoml_v2_worktrees\DINO-123
H:\dinoml_v2_worktrees\DINO-124
H:\dinoml_v2_runs\DINO-123
H:\dinoml_v2_runs\DINO-124
```

Each run directory should capture:

- rendered prompt
- JSONL event log
- final assistant message
- validation output summary
- branch name
- commit SHAs
- issue state transitions

## Safety Defaults

For read-only exploration:

```powershell
codex exec --sandbox read-only --ask-for-approval never ...
```

For bounded implementation in an isolated worktree:

```powershell
codex exec --sandbox workspace-write --ask-for-approval never ...
```

Avoid broad `danger-full-access` for unattended runs unless the workspace is
externally isolated and the task truly needs it.

