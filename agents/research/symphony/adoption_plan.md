# Adoption Plan

## Phase 0: Keep Current Process Working

Do not replace the current agent steering loop immediately.

Keep:

- `AGENTS.md`
- `agents/CURRENT_FOCUS.md`
- `agents/NEXT_CANDIDATE_WORK.md`
- `agents/INVARIANTS.md`
- `agents/PROVIDER_CONTRACT.md`
- `agents/OP_ADMISSION.md`
- active plans and checklists

Treat the issue backlog as an additional execution layer until it proves it can
carry the work.

## Phase 1: Issue Backlog Pilot

Create a small number of issues manually.

Recommended first batch:

1. Implementation issue for the top GGUF native load-path parity item.
2. Exploration issue for CUTLASS/provider maturity.
3. Exploration issue for runtime/container lifecycle risks.
4. Exploration issue for Transformers report clustering.
5. Exploration issue for Diffusers first implementation lanes.

Each issue should have:

- type label
- area label
- clear acceptance criteria
- validation requirements
- links to relevant `agents/` docs
- explicit "do not expand scope" guidance

Success criteria:

- agents can start from an issue and know what to do
- final output is reviewable without reconstructing the whole thread
- work produces either validated code or high-quality follow-up issues
- `NEXT_CANDIDATE_WORK.md` becomes less central to scheduling

## Phase 2: Local `codex exec` Runner

Build or script a small local runner.

Responsibilities:

- read issues from the chosen tracker or a local queue
- create one git worktree per issue
- render a prompt from issue fields plus a workflow template
- run `codex exec`
- capture JSONL logs and final messages
- update a workpad location
- stop after one issue or one bounded batch

This phase can be intentionally small. It does not need full Symphony behavior.

Minimum useful command shape:

```powershell
codex exec `
  --cd <issue-worktree> `
  --sandbox workspace-write `
  --ask-for-approval never `
  --json `
  --output-last-message <run-dir>\final.md `
  <rendered-prompt>
```

Success criteria:

- two or three exploratory agents can run independently without file conflicts
- one implementation issue can produce a clean branch and validation summary
- workpad/result artifacts are enough for review

## Phase 3: Workflow Contract

Introduce a repo-owned workflow file.

This could be:

- `WORKFLOW.md`
- `WORKFLOW.symphony.md`
- `agents/WORKFLOW.md`

The file should combine:

- issue-state policy
- workspace policy
- agent prompt body
- validation gates
- PR/review handoff expectations
- blocked-work behavior
- follow-up issue creation rules

Do not duplicate all of `AGENTS.md`. The workflow should reference the steering
docs and add orchestration-specific behavior.

Success criteria:

- a new agent can work from an issue plus workflow without reading PM-specific
  private instructions
- exploration and implementation issues have different allowed behaviors
- validation and handoff bars are explicit

## Phase 4: App-Server Orchestrator

Move from process-level `codex exec` to `codex app-server` only when needed.

Add:

- issue polling
- active/terminal state reconciliation
- retry queue
- stall detection
- per-issue run metadata
- streamed status surface
- controlled continuation turns

This is the point where the implementation starts looking like Symphony.

Success criteria:

- the orchestrator can run unattended for hours
- stalled or crashed runs retry cleanly
- terminal or blocked issue states stop active runs
- multiple agents can run without workspace collisions
- every run has an inspectable event trail

## Phase 5: Research Promotion Automation

Once the tracker workflow is stable, add agents whose only job is backlog
curation.

Examples:

- GGUF research-to-issue agent
- CUTLASS provider maturity reviewer
- core op checklist curator
- Transformers model-family clusterer
- Diffusers component-stage planner
- auxiliary/UI product-value reviewer

These agents should create or propose issues, not implement code, unless
explicitly assigned implementation work.

Success criteria:

- research does not accumulate without a path to execution
- the tracker shows dependencies between model support, ops, providers, and
  runtime infrastructure
- humans can steer by moving issues between states rather than rewriting prompts

## Phase 6: Harden And Scale

Only after the pilot succeeds:

- add dashboards/status pages
- add GPU runner pools
- add resource-aware scheduling
- add cloud execution for tasks that do not need local-only dependencies
- add automated PR feedback sweeps
- add merge/landing automation

The risk to avoid is building a large orchestrator before the backlog contract is
clear. The first useful win is disciplined issue execution, not infrastructure
polish.

