# DinoML Workflow Contract Sketch

This is a sketch for a future repo-owned workflow file. It is not active
configuration.

## Front Matter Shape

```yaml
---
tracker:
  kind: linear # or github, local
  project_slug: dinoml-v2
  active_states:
    - Todo
    - In Progress
    - Rework
  review_states:
    - Human Review
  terminal_states:
    - Done
    - Closed
    - Cancelled
    - Duplicate

workspace:
  root: H:/dinoml_v2_worktrees
  run_logs: H:/dinoml_v2_runs
  strategy: git_worktree

agent:
  max_concurrent_agents: 3
  max_turns: 5
  default_model: gpt-5.4
  hard_task_model: gpt-5.5
  exploration_model: gpt-5.4-mini

codex:
  command: codex exec
  sandbox: workspace-write
  approval_policy: never

policy:
  require_validation_for_code_changes: true
  require_workpad: true
  allow_exploration_without_code_changes: true
---
```

## Prompt Body Sketch

```md
You are working on DinoML v2 issue {{ issue.identifier }}.

Title: {{ issue.title }}
State: {{ issue.state }}
Labels: {{ issue.labels }}
URL: {{ issue.url }}

Issue body:
{{ issue.description }}

## Repository Steering

Read and follow:

- AGENTS.md
- agents/OPERATING_LOOP.md
- agents/INVARIANTS.md
- agents/CURRENT_FOCUS.md
- agents/BLOCKED_OR_DEFERRED.md
- agents/PROVIDER_CONTRACT.md
- agents/OP_ADMISSION.md
- agents/NEXT_CANDIDATE_WORK.md

For implementation work, also read the relevant active plans under
agents/plans/.

## Work Type Routing

If the issue has label type:exploration:

- Treat the task as read-only unless the issue explicitly allows doc edits.
- Do not modify source code.
- Produce findings and proposed executable issues using the exploration output
  contract.
- Do not commit unless explicitly requested.

If the issue has label type:implementation:

- Pick the bounded task described in the issue.
- Implement only that scope.
- Run targeted validation.
- Update relevant docs/checklists when behavior changes.
- Commit only if validation supports the change.

If the issue has label type:review:

- Use code-review posture.
- Lead with findings ordered by severity.
- Do not make unrelated changes.

## Workpad

Maintain one persistent workpad for the issue.

The workpad must contain:

- Plan
- Acceptance Criteria
- Validation
- Notes
- Confusions

Keep it current as reality changes.

## Completion Bar

Before moving to review:

- acceptance criteria are complete
- validation has run or an explicit reason is recorded
- docs/checklists are updated if behavior changed
- no unrelated changes are included
- final summary is concise and points to validation evidence
```

## Workpad Template

````md
## Codex Workpad

```text
<host>:<workspace>@<head-sha>
```

### Plan

- [ ] 1. Understand issue and relevant steering docs
- [ ] 2. Identify exact scope and validation
- [ ] 3. Execute bounded work
- [ ] 4. Validate
- [ ] 5. Summarize handoff

### Acceptance Criteria

- [ ] Criterion 1
- [ ] Criterion 2

### Validation

- [ ] targeted test or read-only review evidence

### Notes

- 

### Confusions

- 
````

## First Pilot Issues

1. `Fix native GGUF manual-runtime-load parity`
   - Type: implementation
   - Area: GGUF/runtime
   - Source: `agents/NEXT_CANDIDATE_WORK.md`

2. `Explore CUTLASS provider maturity gaps`
   - Type: exploration
   - Area: CUTLASS/provider
   - Source: `agents/PROVIDER_CONTRACT.md`,
     `agents/plans/gemm_cutlass_plan.md`

3. `Explore runtime/container lifecycle risks`
   - Type: exploration
   - Area: runtime/container
   - Source: `agents/plans/v1_gap_audit.md`

4. `Cluster Transformers reports into first implementation epics`
   - Type: exploration
   - Area: Transformers
   - Source: `agents/plans/transformers/`

5. `Cluster Diffusers reports into first implementation epics`
   - Type: exploration
   - Area: Diffusers
   - Source: `agents/plans/diffusers/`
