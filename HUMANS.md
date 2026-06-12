# Human Instructions

This document is intended for humans only. Do not follow this guidance if you are an agent.

## Base/Developer Instructions

`.codex/base.md` and `.codex/developer.md` are repo-local custom Base and Developer instructions.

The intent is to:
- improve workflow adherence
- reduce failure modes
- decrease cortisol spikes

Using `developer_instructions_file` is not currently supported in upstream Codex. To help prioritize it upstream, add a reaction to [openai/codex#12926](https://github.com/openai/codex/issues/12926). You can use [this branch](https://github.com/hlky/codex/tree/feat/developer-instructions-file) in a custom build, or inline the contents of `developer.md` into `developer_instructions` in `config.toml`.

It is also recommended to disable the default built-in instruction sections, especially `include_collaboration_mode_instructions`, because they appear to contribute to assumptions and behavioral drift.

```toml
include_permissions_instructions = false
include_apps_instructions = false
include_collaboration_mode_instructions = false
```

See [openai/codex#27587](https://github.com/openai/codex/issues/27587) for further analysis.

## Local Environments

This is another custom feature. To help prioritize upstream adoption, add a reaction to [openai/codex#27336](https://github.com/openai/codex/issues/27336).

The current iteration is present on [`hlky/codex#hlky`](https://github.com/hlky/codex/tree/hlky), which also includes the `developer_instructions_file` fix and other local patches, including work related to [openai/codex#26351](https://github.com/openai/codex/issues/26351) and [openai/codex#27592](https://github.com/openai/codex/issues/27592).

`.codex/config.toml` includes `local_environments.msvc` and `local_environments.rocm`, which are specific to my local machine.

Using this feature from Codex Desktop currently requires a modified build that is not published at the moment. It is usable through the app-server protocol, and partially usable in the CLI/TUI via `default_local_environment`.

The intent of this feature is to reduce environment-related failure modes, especially toolset and runtime discovery. In practice it helps by making the active environment explicit and by ensuring the expected toolchain is already on `PATH` without prompt hacks.

If you do not need venvs, custom `PATH`, or custom environment variables, you probably do not need this feature.

## CUDA / RunPod

I do not have local CUDA, so I use `.codex/skills/runpod-codex-remote`. See its `README.md` for setup.

That skill is referenced by the repo workflows and prompts. If you have local CUDA, you may want to adjust those workflows, prompts, and skill references accordingly.
