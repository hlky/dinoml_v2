---
name: runpod-codex-remote
description: Create and prepare Runpod GPU pods as Codex remote SSH connections. Use when the user asks to create a Runpod pod for a specified GPU, provision it for Codex remote work, install Codex CLI plus companion CLIs such as runpodctl, hf, gh, and aws, register the pod in the Codex app remote connections, and clone a specified Git or GitHub project onto the pod.
---

# Runpod Codex Remote

## Overview

Provision a Runpod pod, make it usable from Codex as a remote SSH environment, install development CLIs inside the pod, and clone the requested project into `/workspace`.

Use `scripts/create_runpod_codex_remote.py` for the main flow whenever possible. It keeps SSH config registration, Codex feature-flag setup, SSH parsing, remote bootstrap, clone, and best-effort app state seeding consistent.

For DinoML v2 CUDA remote verification, prefer the validated repo image:

- `hlky/dinoml:ubuntu-nodeps`

That image already contains `/opt/src/dinoml_v2` and `/opt/venvs/dinoml`, so the preferred flow is to reuse the prebaked repo path with `--existing-project-path /opt/src/dinoml_v2` instead of recloning.

## Inputs

Collect or infer these values before running:

- GPU ID or name for `runpodctl pod create --gpu-id`, for example `NVIDIA GeForce RTX 4090`.
- Project repository URL or `owner/repo` GitHub shorthand.
- Pod name or Codex remote display name. Default to a slug based on the repo and GPU.
- Pod image or template. Default to the Runpod PyTorch image unless the user specifies a template or image.
- SSH identity path. Default to `~/.runpod/ssh/RunPod-Key-Go`, matching `runpodctl ssh add-key`.
- Workspace volume size. Default to `20` GB mounted at `/workspace`; use a larger value when the repo or outputs need it.
- Exposed ports. Default to `22/tcp` in addition to `--ssh` so SSH details are available consistently.

Before creating a billable pod, make sure the user has clearly asked to create one. If the requested GPU is ambiguous, run `runpodctl gpu list` and choose the closest exact GPU ID/name.

### Approval Semantics

For DinoML v2 work, an explicit task request or goal prompt that requires remote CUDA verification counts as clear user approval to create a temporary low-cost verification pod.

In that case:

- do not interrupt the task to request a second approval for ordinary verification pod creation
- stay within the task's stated GPU preferences and budget guidance when present
- prefer the cheapest suitable stocked GPU that satisfies the verification need
- treat the pod as disposable verification infrastructure unless the user says to keep it

Ask for separate approval only when one of these is true:

- the user explicitly says not to create remote resources automatically
- the required GPU, image, or storage would materially exceed the stated budget guidance
- the task requires a long-lived pod or non-routine paid resources beyond ordinary verification

## GPU Price Selection

Use Runpod's GraphQL API for current Secure Cloud pod prices. `runpodctl gpu list` is useful for GPU IDs and broad availability, but it does not expose hourly prices.

Query the same minimal GPU type data used by the Runpod UI:

```graphql
query MinimalGpuTypes {
  gpuTypes {
    lowestPrice(input: {gpuCount: 1, secureCloud: true}) {
      stockStatus
      __typename
    }
    id
    secureCloud
    communityCloud
    displayName
    memoryInGb
    manufacturer
    securePrice
    __typename
  }
}
```

Endpoint:

```text
https://api.runpod.io/graphql?operation=MinimalGpuTypes
```

On Windows/PowerShell, group the cheapest stocked 1x Secure Cloud GPU by VRAM tier:

```powershell
$body = @{
  operationName = 'MinimalGpuTypes'
  variables = @{}
  query = 'query MinimalGpuTypes {
    gpuTypes {
      lowestPrice(input: {gpuCount: 1, secureCloud: true}) { stockStatus __typename }
      id secureCloud communityCloud displayName memoryInGb manufacturer securePrice __typename
    }
  }'
} | ConvertTo-Json -Depth 10

$gpuTypes = (Invoke-RestMethod `
  -Uri 'https://api.runpod.io/graphql?operation=MinimalGpuTypes' `
  -Method Post `
  -ContentType 'application/json' `
  -Body $body).data.gpuTypes

$gpuTypes |
  Where-Object {
    $_.secureCloud -eq $true -and
    $_.securePrice -gt 0 -and
    -not [string]::IsNullOrWhiteSpace($_.lowestPrice.stockStatus)
  } |
  Group-Object memoryInGb |
  ForEach-Object { $_.Group | Sort-Object securePrice | Select-Object -First 1 } |
  Sort-Object memoryInGb |
  Select-Object `
    @{n='VRAM';e={$_.memoryInGb}},
    displayName,
    id,
    securePrice,
    @{n='stock';e={$_.lowestPrice.stockStatus}},
    communityCloud |
  Format-Table -AutoSize
```

Interpretation:

- `securePrice` is the current Secure Cloud hourly price for the GPU type.
- `lowestPrice.stockStatus` indicates whether the queried 1x Secure Cloud configuration currently has stock (`Low`, `Medium`, `High`, or null).
- Prefer stocked rows for immediate pod creation. Null stock can still be useful for price awareness but should not be treated as currently rentable without confirming in the Runpod console or datacenter availability.
- Use the returned `id` value directly as `runpodctl pod create --gpu-id`.

## Workflow

1. Ensure local Runpod CLI access:

```bash
runpodctl doctor
runpodctl gpu list
```

If `runpodctl` is missing, install it from the official installer:

```bash
curl -sSL https://cli.runpod.net | bash
```

2. Run the helper from this skill:

```bash
python .codex/skills/runpod-codex-remote/scripts/create_runpod_codex_remote.py \
  --gpu-id "NVIDIA GeForce RTX 4090" \
  --repo https://github.com/owner/project.git \
  --name project-4090 \
  --volume-gb 20 \
  --ports 22/tcp
```

If you are not running from the repository root, use the absolute script path under the current repo's `.codex\skills\runpod-codex-remote\scripts\...`.

For the DinoML v2 CUDA verification image, use:

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

3. If the helper cannot parse the pod ID or SSH info automatically, use the equivalent manual sequence:

```bash
runpodctl pod create --image runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404 --gpu-id "GPU NAME" --name "REMOTE NAME" --ssh --ports 22/tcp --volume-in-gb 20 --volume-mount-path /workspace
runpodctl pod list --all --name "REMOTE NAME"
runpodctl ssh info POD_ID
ssh -i ~/.runpod/ssh/RunPod-Key-Go -p PORT root@HOST "bash -lc 'echo ok'"
```

4. Confirm the helper added a concrete SSH alias to `~/.ssh/config`, then test that alias:

```bash
ssh project-4090 "bash -lc 'codex --version && test -d /workspace/project/.git'"
```

5. In Codex Desktop, use Settings > Connections to enable the discovered SSH host, then choose the remote project folder. If the host or seeded project does not appear immediately, restart or refresh the app; the desktop app may cache `~/.ssh/config`, `~/.codex/config.toml`, and `~/.codex/.codex-global-state.json`.

## Remote Bootstrap

Install these inside the pod:

- Node.js 20+ and npm before installing Codex. On Ubuntu images, use NodeSource if the image does not already provide Node 20+:

```bash
if ! command -v node >/dev/null 2>&1 || ! node -e 'process.exit(Number(process.versions.node.split(".")[0]) >= 20 ? 0 : 1)' >/dev/null 2>&1; then
  curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
  apt-get install -y nodejs
fi
node --version
npm --version
```

- `codex` via `npm install -g @openai/codex`.
- `runpodctl` via `curl -sSL https://cli.runpod.net | bash`.
- `hf` via `curl -LsSf https://hf.co/cli/install.sh | bash`.
- `gh` from the GitHub CLI apt repository on Debian/Ubuntu.
- `aws` from AWS CLI v2 installer.

Do not embed API keys or auth tokens in the skill. If a project requires GitHub, Hugging Face, AWS, OpenAI, or Runpod authentication inside the pod, prompt for the appropriate interactive login or environment-secret approach after the pod is reachable.

### Codex Auth

Codex Desktop may prompt the user to log in the first time an SSH host is enabled. That is expected when the remote user's `~/.codex/auth.json` is missing or invalid. After login, verify remotely:

```bash
ssh project-4090 "codex login status"
```

If the user explicitly asks to pre-authenticate the pod, the helper can copy the local Codex auth file:

```bash
python .codex/skills/runpod-codex-remote/scripts/create_runpod_codex_remote.py \
  --gpu-id "GPU NAME" \
  --repo https://github.com/owner/project.git \
  --name project-4090 \
  --pod-id POD_ID \
  --skip-bootstrap \
  --copy-codex-auth
```

Only do this when the user explicitly asks, because it copies a usable Codex auth token to the pod. Never paste `auth.json` contents into chat or command output.

### GitHub CLI Auth Transfer

If the user asks to copy the current local GitHub CLI auth into the pod, prefer token transfer over copying config files. Local `gh` auth may live in the OS keyring, so `hosts.yml` may not contain the token.

Verify local auth:

```bash
gh auth status
gh auth token
```

Then pipe the token into the remote GitHub CLI without printing it:

```bash
gh auth token | ssh -i KEY -p PORT root@HOST "mkdir -p /root/.config/gh && gh auth login --hostname github.com --with-token"
ssh -i KEY -p PORT root@HOST "gh auth status"
```

The helper supports this as an explicit opt-in:

```bash
python .codex/skills/runpod-codex-remote/scripts/create_runpod_codex_remote.py \
  --gpu-id "GPU NAME" \
  --repo https://github.com/owner/project.git \
  --name project-4090 \
  --pod-id POD_ID \
  --skip-bootstrap \
  --copy-gh-auth
```

Only do this when the user explicitly asks, because it copies a usable GitHub token to the pod. Never paste the token into chat or command output.

### Git Identity

If the user asks to copy local Git identity, read local global identity and set it on the pod:

```bash
git config --global user.name
git config --global user.email
```

The helper supports this as an explicit opt-in:

```bash
python .codex/skills/runpod-codex-remote/scripts/create_runpod_codex_remote.py \
  --gpu-id "GPU NAME" \
  --repo https://github.com/owner/project.git \
  --name project-4090 \
  --pod-id POD_ID \
  --skip-bootstrap \
  --copy-git-identity
```

By default this sets remote global Git config. For repo-local config, add:

```bash
--git-identity-scope repo
```

Verify:

```bash
ssh project-4090 "git config --global user.name && git config --global user.email"
ssh project-4090 "git -C /workspace/project config user.name && git -C /workspace/project config user.email"
```

## Project Clone

Clone into `/workspace/<repo-name>` by default:

```bash
git clone REPO_URL /workspace/REPO_NAME
```

For GitHub shorthand like `owner/repo`, normalize to `https://github.com/owner/repo.git` unless the user requests SSH cloning. Prefer HTTPS for public repos because it works before GitHub SSH keys are configured inside the pod.

After cloning, verify:

```bash
ssh -i KEY -p PORT root@HOST "bash -lc 'codex --version && git -C /workspace/REPO_NAME status --short'"
```

If the pod image already contains the repository, reuse that path instead of cloning again. The helper supports this with `--existing-project-path` and will seed Codex against that path:

```bash
python .codex/skills/runpod-codex-remote/scripts/create_runpod_codex_remote.py \
  --gpu-id "GPU NAME" \
  --repo https://github.com/owner/project.git \
  --name project-prebaked \
  --image owner/project:image \
  --existing-project-path /opt/src/project \
  --auto-connect
```

In this mode the helper still bootstraps Codex/CLI tools, but it does not run `git clone`.

For DinoML v2 this is the preferred CUDA remote verification mode when the pod image is `hlky/dinoml:ubuntu-nodeps`.

## Codex SSH Config

Codex SSH remote connections are an alpha feature. Ensure `~/.codex/config.toml` contains:

```toml
[features]
remote_connections = true
```

Codex discovers concrete SSH aliases from `~/.ssh/config`; pattern-only hosts are ignored. Register the Runpod SSH endpoint as a named alias matching the pod or project name:

```text
Host project-4090
  HostName HOST
  User root
  Port PORT
  IdentityFile ~/.runpod/ssh/RunPod-Key-Go
  StrictHostKeyChecking accept-new
```

Prefer using `ssh project-4090` for verification after this point, because that is the same route the Codex app will use.

## Codex Remote State

Best-effort remote connection and remote project seeding currently lives in `~/.codex/.codex-global-state.json` under:

- top-level `remote-connection-auto-connect-by-host-id`
- top-level `remote-projects`
- top-level `project-order`
- top-level `active-remote-project-id`
- top-level `selected-remote-host-id`
- `electron-persisted-atom-state.agent-mode-by-host-id`
- optionally top-level `codex-managed-remote-connections` for legacy managed entries

Treat direct state updates as app-version-sensitive convenience, and verify after creation because Desktop may rewrite state from memory while running. New remote projects seeded externally are most reliable when Codex Desktop is not running or after a restart; a running app can preserve the discovered connection while later removing a project entry it did not create itself. The helper retries and warns when Desktop is running. Current Codex Desktop remote projects created through the UI use discovered SSH host IDs shaped like:

```text
remote-ssh-discovered:<ssh-alias>
```

To pre-enable a discovered SSH alias, set:

```json
{
  "remote-ssh-discovered:project-4090": true
}
```

inside top-level `remote-connection-auto-connect-by-host-id`.

Also set the same discovered host ID to `full-access` inside `electron-persisted-atom-state.agent-mode-by-host-id`, and set top-level `selected-remote-host-id` to the discovered host ID.

Create a remote project entry for the cloned repository, place its generated project ID at the front of `project-order`, and set `active-remote-project-id` to that project ID:

```json
{
  "id": "generated-uuid",
  "hostId": "remote-ssh-discovered:project-4090",
  "remotePath": "/workspace/project",
  "label": "project"
}
```

After seeding, reread `~/.codex/.codex-global-state.json` and confirm:

- `remote-projects` contains the new `{ hostId, remotePath, label }`.
- `project-order[0]` and `active-remote-project-id` are the new project ID.
- `remote-connection-auto-connect-by-host-id[hostId]` is `true` when `--auto-connect` was used.
- `electron-persisted-atom-state.agent-mode-by-host-id[hostId]` is `full-access`.

Only seed legacy managed remote connection entries when needed for older app builds. Those entries use host IDs like `remote-ssh-codex-managed:<display-name>` and look like:

```json
{
  "hostId": "remote-ssh-codex-managed:project-4090",
  "displayName": "project-4090",
  "source": "codex-managed",
  "autoConnect": false,
  "sshAlias": "project-4090",
  "sshHost": "root@HOST",
  "sshPort": 12345,
  "identity": "~/.runpod/ssh/RunPod-Key-Go"
}
```

## Cleanup

Prefer stopping a pod to end compute billing while preserving state:

```bash
runpodctl pod stop POD_ID
```

If this workflow created a fresh disposable pod in the current task and it turns out to be unusable, failed, or clearly no longer needed, delete it as part of cleanup rather than leaving billable junk behind:

```bash
runpodctl pod delete POD_ID
```

Do not delete a pre-existing pod, or a pod that may contain user work the task intends to keep, unless the user explicitly asks for deletion or clearly treats the pod as disposable.
