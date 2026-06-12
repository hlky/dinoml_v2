# Runpod Codex Remote

Create a Runpod GPU pod, prepare it for Codex Desktop SSH remote connections, install useful development CLIs, clone a project into `/workspace`, and seed Codex Desktop state where the current app supports it.

The skill is centered on `scripts/create_runpod_codex_remote.py`. Use the natural-language prompts below through Codex, or run the helper directly when you want explicit flags.

For DinoML v2 CUDA remote verification, the current preferred image is
`hlky/dinoml:ubuntu-nodeps`. That image already contains:

- `/opt/src/dinoml_v2`
- `/opt/venvs/dinoml`
- CUDA 12.9 tooling
- `transformers` and `diffusers` source trees

So the preferred DinoML flow is to reuse the prebaked repo path with
`--existing-project-path /opt/src/dinoml_v2`.
For that flow, fresh pods should normally be treated as disposable and deleted after verification unless the user explicitly wants to keep them.

For DinoML v2 task execution, an explicit request or goal prompt that says CUDA verification is required counts as approval to create a temporary verification pod within the task's stated budget and GPU guidance. Do not stop for a second approval unless the requested resources exceed that guidance or the user explicitly requires confirmation.

## What It Does

- Creates a Runpod GPU pod for a requested GPU, such as RTX 3090, RTX 4090, H100, or another `runpodctl gpu list` GPU ID.
- Configures the pod with SSH, an exposed `22/tcp` port, and a `/workspace` volume.
- Adds a concrete SSH alias to `~/.ssh/config`, because Codex Desktop discovers SSH remotes from OpenSSH config.
- Ensures Codex Desktop has `[features] remote_connections = true` in `~/.codex/config.toml`.
- Installs Node.js/npm, Codex CLI, Runpod CLI, Hugging Face CLI, GitHub CLI, AWS CLI.
- Clones the requested GitHub repository to `/workspace/<repo>`.
- Optionally copies local Codex auth, GitHub CLI auth, and Git identity to the pod when explicitly requested.
- Best-effort seeds Codex Desktop remote project state using discovered SSH host IDs like `remote-ssh-discovered:<ssh-alias>`.

Remote connections are an alpha Codex Desktop feature. The supported discovery path is still SSH config plus Settings > Connections. Direct state seeding is a convenience and may need app refresh or restart.

For routine DinoML CUDA verification, treat the created pod as disposable unless the user says to keep it.

## Requirements

- `runpodctl` installed locally and authenticated.
- A Runpod SSH key configured locally, usually `~/.runpod/ssh/RunPod-Key-Go`.
- Codex Desktop remote connections enabled:

```toml
[features]
remote_connections = true
```

- Local `ssh` available on PATH.
- For auth transfer flags, local `codex login status` and/or `gh auth status` should already work.

## Direct Usage

Create a fresh pod and clone a project:

```bash
python .codex/skills/runpod-codex-remote/scripts/create_runpod_codex_remote.py \
  --gpu-id "NVIDIA GeForce RTX 3090" \
  --repo https://github.com/hlky/libgguf.git \
  --name libgguf-3090 \
  --volume-gb 20 \
  --ports 22/tcp
```

Create a fresh DinoML CUDA verification pod from the validated image:

```bash
python .codex/skills/runpod-codex-remote/scripts/create_runpod_codex_remote.py \
  --gpu-id "NVIDIA RTX 2000 Ada Generation" \
  --repo https://github.com/hlky/dinoml_v2.git \
  --name dinoml-ubuntu-nodeps-rtx2000ada \
  --image hlky/dinoml:ubuntu-nodeps \
  --existing-project-path /opt/src/dinoml_v2 \
  --volume-gb 20 \
  --ports 22/tcp \
  --auto-connect
```

Use an existing pod without reinstalling tools:

```bash
python .codex/skills/runpod-codex-remote/scripts/create_runpod_codex_remote.py \
  --gpu-id "NVIDIA GeForce RTX 3090" \
  --repo https://github.com/hlky/libgguf.git \
  --name libgguf-3090 \
  --pod-id POD_ID \
  --skip-bootstrap
```

Pre-authenticate Codex and GitHub CLI on the remote:

```bash
python .codex/skills/runpod-codex-remote/scripts/create_runpod_codex_remote.py \
  --gpu-id "NVIDIA GeForce RTX 3090" \
  --repo https://github.com/hlky/libgguf.git \
  --name libgguf-3090 \
  --copy-codex-auth \
  --copy-gh-auth \
  --copy-git-identity
```

The auth flags copy usable credentials to the pod. Use them only for trusted pods. `--copy-git-identity` copies local `git config --global user.name` and `git config --global user.email`; by default it sets the remote global Git identity.

Set repo-local Git identity instead:

```bash
python .codex/skills/runpod-codex-remote/scripts/create_runpod_codex_remote.py \
  --gpu-id "NVIDIA GeForce RTX 3090" \
  --repo https://github.com/hlky/libgguf.git \
  --name libgguf-3090 \
  --pod-id POD_ID \
  --skip-bootstrap \
  --copy-git-identity \
  --git-identity-scope repo
```

## Example Prompts

```text
[$runpod-codex-remote](.codex\skills\runpod-codex-remote\SKILL.md) Create a Runpod pod with an RTX 3090 and clone hlky/libgguf to it
```

```text
[$runpod-codex-remote](.codex\skills\runpod-codex-remote\SKILL.md) Create a 4090 Runpod Codex remote named libgguf-4090, clone https://github.com/hlky/libgguf.git, and use a 50GB workspace volume
```

```text
[$runpod-codex-remote](.codex\skills\runpod-codex-remote\SKILL.md) Create a Runpod H100 pod for hlky/dinoml, pre-authenticate Codex on the remote, and copy my current GitHub CLI auth
```

```text
[$runpod-codex-remote](.codex\skills\runpod-codex-remote\SKILL.md) Use the existing Runpod pod r5qypko262ma4i, register it as libgguf-3090-fullflow in SSH config, seed the Codex remote project for /workspace/libgguf, and skip bootstrap
```

```text
[$runpod-codex-remote](.codex\skills\runpod-codex-remote\SKILL.md) Fix the Runpod Codex remote skill based on the latest Codex SSH remote connections behavior, then test it on an existing pod
```

```text
[$runpod-codex-remote](.codex\skills\runpod-codex-remote\SKILL.md) Stop the Runpod pod named libgguf-3090-fullflow after verifying the repo and Codex login still work
```

```text
[$runpod-codex-remote](.codex\skills\runpod-codex-remote\SKILL.md) Create a Runpod pod from the official PyTorch 2.4 template with an RTX 4090, clone owner/repo, enable Codex remote connections, and add the SSH alias to ~/.ssh/config
```

## Verification

After a run, verify the same SSH alias Codex Desktop will use:

```bash
ssh libgguf-3090 "bash -lc 'codex --version && codex login status && test -d /workspace/libgguf/.git'"
```

Check GPU visibility:

```bash
ssh libgguf-3090 "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader"
```

For the DinoML verification image, a minimal end-to-end smoke baseline is:

- `torch` import plus CUDA tensor/matmul
- tiny `transformers` GPT-2 forward on CUDA
- tiny `diffusers` UNet/scheduler forward on CUDA
- DinoML v2 trace/compile/load/run smoke on CUDA

Check GitHub CLI auth if copied:

```bash
ssh libgguf-3090 "gh auth status"
```

Check Git identity if copied:

```bash
ssh libgguf-3090 "git config --global user.name && git config --global user.email"
ssh libgguf-3090 "git -C /workspace/libgguf config user.name && git -C /workspace/libgguf config user.email"
```

## Codex Desktop Notes

Codex Desktop discovers concrete SSH aliases from `~/.ssh/config`. The helper writes aliases like:

```text
Host libgguf-3090
  HostName 203.0.113.10
  User root
  Port 40036
  IdentityFile ~/.runpod/ssh/RunPod-Key-Go
  StrictHostKeyChecking accept-new
```

The current app uses discovered host IDs for remote projects:

```text
remote-ssh-discovered:<ssh-alias>
```

If the host or project does not appear immediately, refresh or restart Codex Desktop. If the app prompts for Codex login on the remote, either complete the login flow or rerun with `--copy-codex-auth` for a trusted pod.

## Cleanup

For disposable CUDA verification pods created by this workflow, prefer delete when verification is complete. `stop` ends compute billing but leaves volume storage billable.

Use stop only when the pod is intentionally being preserved:

```bash
runpodctl pod stop POD_ID
```

Use delete directly when you know the pod should be terminated:

```bash
runpodctl pod delete POD_ID
```

If this workflow created a fresh disposable pod in the current task and it is clearly failed or no longer needed, delete it as part of cleanup rather than leaving billable junk behind. Do not delete a pre-existing pod, or a pod that may contain user work the task intends to keep, unless the user explicitly asks for deletion or clearly treats it as disposable.
