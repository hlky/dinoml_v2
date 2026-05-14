# Symphony-Oriented Agent Process Research

This directory captures a read-only assessment of moving DinoML v2 from
prompt-driven autonomous loops toward an issue-driven agent backlog inspired by
OpenAI Symphony.

## Short Conclusion

DinoML v2 is in a good state to begin converting agent work into an issue-based
backlog. The repository already has strong steering docs, invariants, provider
and op admission rules, and a large body of model research. The missing piece is
not more research; it is an executable control plane that turns research and
plans into issue-scoped runs with isolated workspaces, validation gates, and
clear handoff states.

The recommended first path is:

1. Keep using Codex Desktop for supervision and manual inspection.
2. Pilot issue-driven execution with `codex exec` and one git worktree per issue.
3. Use the Symphony `SPEC.md` model as the target architecture.
4. Promote to `codex app-server` only after the issue workflow proves useful.
5. Treat the Symphony reference implementation as calibration/prototype code,
   not as the long-term DinoML orchestrator unless its assumptions match the
   project environment.

## Files

- [current_state.md](current_state.md): assessment of the current `agents/`
  docs and research corpus.
- [issue_backlog_model.md](issue_backlog_model.md): how to turn plans, model
  reports, and exploratory work into epics, issues, and subissues.
- [runner_options.md](runner_options.md): practical ways to launch agents from
  the current Codex ecosystem.
- [adoption_plan.md](adoption_plan.md): staged migration plan from the current
  loop to a Symphony-style workflow.
- [workflow_contract_sketch.md](workflow_contract_sketch.md): a DinoML-oriented
  `WORKFLOW.md` contract sketch.

## Source Basis

- OpenAI Symphony repository: <https://github.com/openai/symphony>
- Symphony service specification:
  <https://github.com/openai/symphony/blob/main/SPEC.md>
- Symphony reference workflow:
  <https://github.com/openai/symphony/blob/main/elixir/WORKFLOW.md>
- Symphony reference implementation README:
  <https://github.com/openai/symphony/blob/main/elixir/README.md>
- OpenAI Symphony announcement:
  <https://openai.com/index/open-source-codex-orchestration-symphony/>
- Local Codex CLI inspection from `codex --help`, `codex exec --help`,
  `codex app-server --help`, and `codex cloud exec --help`.
- Current DinoML steering docs under `agents/` and `AGENTS.md`.

