# Agent Operating Loop

This repository uses docs as project memory and tests as truth.

## Main Loop

1. Read current state:
   - `README.md`
   - `docs/architecture.md`
   - `agents/plans/v1_gap_audit.md`
   - `agents/plans/op_porting_checklist.md`
   - `agents/PROVIDER_CONTRACT.md`
   - `agents/OP_ADMISSION.md`
   - `agents/INVARIANTS.md`
   - `agents/CURRENT_FOCUS.md`
   - `agents/BLOCKED_OR_DEFERRED.md`

2. Inspect recent commits and current tests.

3. Check whether the current user request names external or private context.
   Use authenticated/local access first, and ask before substituting or skipping
   that source.

4. Pick one bounded slice of work.

5. Use subagents to investigate, implement, and review.

6. Validate with targeted tests.

7. Update docs/checklists.

8. Commit.

9. Update Codex Progress.

10. Repeat.

## Work Selection Rules

Prefer:
- stabilizing recent work
- closing known v1 gaps
- improving provider maturity
- adding tests for newly landed surface
- small complete slices

Avoid:
- broad unvalidated op expansion
- new architecture exceptions
- provider-specific shortcuts that bypass manifests or execution plans
- stale documentation
- turning fixable access/auth/path ambiguity into a broad workaround or skipped
  user-requested source
