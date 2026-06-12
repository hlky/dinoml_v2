#!/usr/bin/env python3
"""Create a Runpod pod, add an SSH config alias for Codex, bootstrap tools, and clone or reuse a repo."""

from __future__ import annotations

import argparse
import getpass
import json
import os
import posixpath
import re
import shlex
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any


DEFAULT_IMAGE = "runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404"
DEFAULT_IDENTITY = "~/.runpod/ssh/RunPod-Key-Go"
DEFAULT_VOLUME_GB = 20
DEFAULT_PORTS = "22/tcp"


def run(cmd: list[str], *, check: bool = True, capture: bool = True) -> subprocess.CompletedProcess[str]:
    print("+ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, text=True, capture_output=capture, check=check)


def normalize_repo(repo: str) -> tuple[str, str]:
    if re.fullmatch(r"[\w.-]+/[\w.-]+", repo):
        repo = f"https://github.com/{repo}.git"
    name = repo.rstrip("/").removesuffix(".git").split("/")[-1]
    if not name:
        raise SystemExit(f"Could not derive repository name from {repo!r}")
    return repo, re.sub(r"[^A-Za-z0-9._-]+", "-", name)


def slug(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-")
    return value or "runpod-codex"


def parse_pod_id(text: str) -> str | None:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = None

    if isinstance(data, dict):
        value = data.get("id")
        if isinstance(value, str) and value:
            return value
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                value = item.get("id")
                if isinstance(value, str) and value:
                    return value

    patterns = [
        r"(?im)^\s*pod[-_ ]?id[:=\s]+([A-Za-z0-9_-]{6,})\s*$",
        r"(?im)^\s*id[:=\s]+([A-Za-z0-9_-]{6,})\s*$",
        r"\b([A-Za-z0-9]{8,})\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def parse_ssh_info(text: str) -> tuple[str, int, str | None]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = None

    if isinstance(data, dict):
        host = data.get("ip") or data.get("host")
        port = data.get("port")
        key = data.get("ssh_key", {}).get("path") if isinstance(data.get("ssh_key"), dict) else None
        if isinstance(host, str) and host and port:
            return f"root@{host}", int(port), key if isinstance(key, str) and key else None

        command = data.get("ssh_command")
        if isinstance(command, str):
            command_match = re.search(
                r"ssh(?:\s+-i\s+(?P<key>\S+))?\s+(?P<userhost>[\w.-]+@[\w.-]+)(?:\s+-p\s+(?P<port>\d+))?",
                command,
            )
            if command_match and command_match.group("port"):
                return (
                    command_match.group("userhost"),
                    int(command_match.group("port")),
                    command_match.group("key"),
                )

    ssh_match = re.search(
        r"ssh(?:\s+-i\s+(?P<key>\S+))?(?:\s+-p\s+(?P<port>\d+))?\s+(?P<userhost>[\w.-]+@[\w.-]+)",
        text,
    )
    if ssh_match and ssh_match.group("port"):
        return ssh_match.group("userhost"), int(ssh_match.group("port")), ssh_match.group("key")

    user = re.search(r"\bUser[:=\s]+(\w+)", text, re.IGNORECASE)
    host = re.search(r"\bHost(?:name)?[:=\s]+([\w.-]+)", text, re.IGNORECASE)
    port = re.search(r"\bPort[:=\s]+(\d+)", text, re.IGNORECASE)
    if host and port:
        userhost = f"{user.group(1) if user else 'root'}@{host.group(1)}"
        return userhost, int(port.group(1)), None

    raise SystemExit("Could not parse SSH host/port from runpodctl ssh info output.")


def create_pod(args: argparse.Namespace) -> str:
    cmd = ["runpodctl", "pod", "create", "--name", args.name, "--gpu-id", args.gpu_id, "--ssh"]
    if args.template_id:
        cmd += ["--template-id", args.template_id]
    else:
        cmd += ["--image", args.image]
    if args.gpu_count:
        cmd += ["--gpu-count", str(args.gpu_count)]
    if args.container_disk_gb:
        cmd += ["--container-disk-in-gb", str(args.container_disk_gb)]
    if args.volume_gb:
        cmd += ["--volume-in-gb", str(args.volume_gb)]
    if args.volume_mount_path:
        cmd += ["--volume-mount-path", args.volume_mount_path]
    if args.ports:
        cmd += ["--ports", args.ports]

    result = run(cmd, check=True)
    output = (result.stdout or "") + "\n" + (result.stderr or "")
    pod_id = parse_pod_id(output)
    if pod_id:
        return pod_id

    time.sleep(args.wait_seconds)
    listing = run(["runpodctl", "pod", "list", "--all", "--name", args.name], check=True)
    output = (listing.stdout or "") + "\n" + (listing.stderr or "")
    pod_id = parse_pod_id(output)
    if not pod_id:
        raise SystemExit("Pod was created, but the pod ID could not be parsed. Run `runpodctl pod list --all`.")
    return pod_id


def ensure_runpod_config(args: argparse.Namespace) -> None:
    if args.api_key:
        run(["runpodctl", "config", "--apiKey", args.api_key], check=True, capture=False)
        return

    configured = run(["runpodctl", "user"], check=False)
    if configured.returncode == 0:
        return

    if not args.prompt_api_key:
        print("runpodctl is not configured. Re-run with --prompt-api-key or --api-key, or run `runpodctl config --apiKey ...`.", file=sys.stderr)
        raise SystemExit(2)

    api_key = getpass.getpass("Runpod API key: ").strip()
    if not api_key:
        raise SystemExit("No Runpod API key provided.")
    run(["runpodctl", "config", "--apiKey", api_key], check=True, capture=False)


def ensure_runpod_ssh_key(identity: str) -> None:
    key_path = Path(os.path.expanduser(identity))
    if key_path.exists():
        return

    print(f"Runpod SSH key not found at {identity}; running `runpodctl ssh add-key`.", flush=True)
    run(["runpodctl", "ssh", "add-key"], check=True, capture=False)

    if not key_path.exists():
        print(
            f"Warning: expected SSH key still not found at {identity}. "
            "Continuing because `runpodctl ssh info` may report a different identity.",
            file=sys.stderr,
        )


def ssh_info(pod_id: str, *, attempts: int = 24, wait_seconds: int = 15) -> tuple[str, int, str | None]:
    last_output = ""
    for attempt in range(1, attempts + 1):
        result = run(["runpodctl", "ssh", "info", pod_id], check=False)
        last_output = (result.stdout or "") + "\n" + (result.stderr or "")
        try:
            return parse_ssh_info(last_output)
        except SystemExit:
            if attempt == attempts:
                break
            print(f"SSH info not ready yet; retrying in {wait_seconds}s ({attempt}/{attempts}).", flush=True)
            time.sleep(wait_seconds)

    raise SystemExit(f"Could not parse SSH host/port from runpodctl ssh info output:\n{last_output}")


def remote_shell(userhost: str, port: int, identity: str, command: str, *, check: bool = True) -> None:
    ssh_cmd = [
        "ssh",
        "-i",
        os.path.expanduser(identity),
        "-p",
        str(port),
        "-o",
        "StrictHostKeyChecking=accept-new",
        userhost,
        "bash -s",
    ]
    print("+ " + " ".join(ssh_cmd), flush=True)
    script = command.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")
    subprocess.run(ssh_cmd, input=script, check=check)


def copy_codex_auth(userhost: str, port: int, identity: str) -> None:
    auth_path = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex")) / "auth.json"
    if not auth_path.exists():
        raise SystemExit(f"Local Codex auth file not found: {auth_path}")

    remote_shell(userhost, port, identity, "mkdir -p /root/.codex && chmod 700 /root/.codex")
    scp_cmd = [
        "scp",
        "-i",
        os.path.expanduser(identity),
        "-P",
        str(port),
        "-o",
        "StrictHostKeyChecking=accept-new",
        str(auth_path),
        f"{userhost}:/root/.codex/auth.json",
    ]
    run(scp_cmd, check=True, capture=False)
    remote_shell(userhost, port, identity, "chmod 600 /root/.codex/auth.json && codex login status")


def copy_github_auth(userhost: str, port: int, identity: str) -> None:
    token_result = run(["gh", "auth", "token"], check=True)
    token = (token_result.stdout or "").strip()
    if not token:
        raise SystemExit("Local GitHub CLI did not return a token. Run `gh auth status` locally.")

    ssh_cmd = [
        "ssh",
        "-i",
        os.path.expanduser(identity),
        "-p",
        str(port),
        "-o",
        "StrictHostKeyChecking=accept-new",
        userhost,
        "bash -lc 'mkdir -p /root/.config/gh && gh auth login --hostname github.com --with-token && gh auth status'",
    ]
    print("+ " + " ".join(ssh_cmd), flush=True)
    subprocess.run(ssh_cmd, text=True, input=token + "\n", check=True)


def copy_git_identity(userhost: str, port: int, identity: str, *, clone_path: str | None = None, scope: str = "global") -> None:
    name_result = run(["git", "config", "--global", "user.name"], check=False)
    email_result = run(["git", "config", "--global", "user.email"], check=False)
    name = (name_result.stdout or "").strip()
    email = (email_result.stdout or "").strip()
    if not name or not email:
        raise SystemExit("Local git identity is incomplete. Set `git config --global user.name` and `git config --global user.email` first.")

    git_config = "git config --global"
    if scope == "repo":
        if not clone_path:
            raise SystemExit("--git-identity-scope repo requires a clone path.")
        git_config = f"git -C {shlex.quote(clone_path)} config"

    command = f"""
set -euo pipefail
{git_config} user.name {shlex.quote(name)}
{git_config} user.email {shlex.quote(email)}
{git_config} --get user.name
{git_config} --get user.email; gh auth setup-git
""".strip()
    remote_shell(userhost, port, identity, command)


def bootstrap_command(repo_url: str, project_path: str, *, skip_clone: bool) -> str:
    clone_or_verify = (
        f"""
if [ ! -d {json.dumps(project_path)} ]; then
  echo "Expected existing project path missing: {project_path}" >&2
  exit 1
fi
"""
        if skip_clone
        else f"""
if [ ! -d {json.dumps(project_path)}/.git ]; then
  rm -rf {json.dumps(project_path)}
  git clone {json.dumps(repo_url)} {json.dumps(project_path)}
fi
"""
    )
    return f"""
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y ca-certificates curl git gnupg lsb-release unzip wget openssh-client build-essential python3 python3-pip python3-venv
if ! command -v node >/dev/null 2>&1 || ! node -e 'process.exit(Number(process.versions.node.split(".")[0]) >= 20 ? 0 : 1)' >/dev/null 2>&1; then
  curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
  apt-get install -y nodejs
fi
npm install -g @openai/codex
curl -sSL https://cli.runpod.net | bash
curl -LsSf https://hf.co/cli/install.sh | bash || true
if ! command -v gh >/dev/null 2>&1; then
  mkdir -p -m 755 /etc/apt/keyrings
  wget -nv -O /tmp/githubcli-archive-keyring.gpg https://cli.github.com/packages/githubcli-archive-keyring.gpg
  cat /tmp/githubcli-archive-keyring.gpg > /etc/apt/keyrings/githubcli-archive-keyring.gpg
  chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" > /etc/apt/sources.list.d/github-cli.list
  apt-get update && apt-get install -y gh
fi
if ! command -v aws >/dev/null 2>&1; then
  curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscliv2.zip
  unzip -q /tmp/awscliv2.zip -d /tmp
  /tmp/aws/install --update
fi
mkdir -p /workspace
{clone_or_verify.strip()}
codex --version
git -C {json.dumps(project_path)} status --short || true
""".strip()


def codex_state_path() -> Path:
    base = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex"))
    return base / ".codex-global-state.json"


def codex_config_path() -> Path:
    base = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex"))
    return base / "config.toml"


def ssh_config_path() -> Path:
    return Path.home() / ".ssh" / "config"


def ensure_codex_remote_connections_feature() -> None:
    path = codex_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    if re.search(r"(?ms)^\[features\]\s*.*?^\s*remote_connections\s*=\s*true\s*$", text):
        return

    if re.search(r"(?m)^\[features\]\s*$", text):
        text = re.sub(r"(?m)^\[features\]\s*$", "[features]\nremote_connections = true", text, count=1)
    else:
        text = text.rstrip() + "\n\n[features]\nremote_connections = true\n"
    path.write_text(text, encoding="utf-8")
    print(f"Enabled Codex remote_connections feature in {path}")


def split_userhost(userhost: str) -> tuple[str, str]:
    if "@" not in userhost:
        return "root", userhost
    user, host = userhost.split("@", 1)
    return user or "root", host


def register_ssh_config_alias(alias: str, userhost: str, port: int, identity: str) -> None:
    user, host = split_userhost(userhost)
    path = ssh_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    block = (
        f"Host {alias}\n"
        f"  HostName {host}\n"
        f"  User {user}\n"
        f"  Port {port}\n"
        f"  IdentityFile {identity}\n"
        f"  StrictHostKeyChecking accept-new\n"
    )

    pattern = re.compile(rf"(?ms)^Host\s+{re.escape(alias)}\s*$.*?(?=^Host\s+|\Z)")
    if pattern.search(text):
        text = pattern.sub(lambda _match: block, text).rstrip() + "\n"
    else:
        text = text.rstrip() + ("\n\n" if text.strip() else "") + block
    path.write_text(text, encoding="utf-8")
    print(f"Registered SSH config alias {alias!r} in {path}")


def register_remote(name: str, userhost: str, port: int, identity: str, *, auto_connect: bool) -> str:
    path = codex_state_path()
    if not path.exists():
        raise SystemExit(f"Codex global state file not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    atoms: dict[str, Any] = data.setdefault("electron-persisted-atom-state", {})
    remotes: list[dict[str, Any]] = data.setdefault("codex-managed-remote-connections", [])
    host_id = f"remote-ssh-codex-managed:{name}"
    entry = {
        "hostId": host_id,
        "displayName": name,
        "source": "codex-managed",
        "autoConnect": False,
        "sshAlias": name,
        "sshHost": userhost,
        "sshPort": port,
        "identity": identity,
    }

    for index, existing in enumerate(remotes):
        if existing.get("hostId") == host_id or existing.get("displayName") == name:
            remotes[index] = {**existing, **entry}
            break
    else:
        remotes.append(entry)

    data.setdefault("remote-connection-auto-connect-by-host-id", {})[host_id] = bool(auto_connect)
    atoms.setdefault("agent-mode-by-host-id", {})[host_id] = "full-access"

    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)
    print(f"Registered Codex remote {name!r} in {path}")
    return host_id


def discovered_host_id(alias: str) -> str:
    return f"remote-ssh-discovered:{alias}"


def codex_desktop_process_running() -> bool:
    if sys.platform != "win32":
        return False

    result = subprocess.run(
        ["tasklist", "/FI", "IMAGENAME eq Codex.exe", "/NH"],
        text=True,
        capture_output=True,
        check=False,
    )
    return "Codex.exe" in (result.stdout or "")


def write_codex_state(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def upsert_remote_project(data: dict[str, Any], host_id: str, remote_path: str, label: str) -> str:
    normalized_path = posixpath.normpath(remote_path)
    projects: list[dict[str, Any]] = data.setdefault("remote-projects", [])

    project = None
    for existing in projects:
        if existing.get("hostId") == host_id and posixpath.normpath(str(existing.get("remotePath", ""))) == normalized_path:
            project = existing
            break

    if project is None:
        project = {
            "id": str(uuid.uuid4()),
            "hostId": host_id,
            "remotePath": normalized_path,
            "label": label,
        }
        projects.insert(0, project)
    else:
        project["remotePath"] = normalized_path
        project["label"] = label or project.get("label") or posixpath.basename(normalized_path)

    order: list[str] = data.setdefault("project-order", [])
    project_id = project["id"]
    data["project-order"] = [project_id, *[item for item in order if item != project_id]]
    data["active-remote-project-id"] = project_id
    data["selected-remote-host-id"] = host_id
    return str(project_id)


def register_remote_project(host_id: str, remote_path: str, label: str) -> None:
    path = codex_state_path()
    if not path.exists():
        raise SystemExit(f"Codex global state file not found: {path}")

    normalized_path = posixpath.normpath(remote_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    upsert_remote_project(data, host_id, normalized_path, label)
    write_codex_state(path, data)
    print(f"Registered Codex remote project {label!r} at {normalized_path}")


def seed_discovered_remote_alias(alias: str, remote_path: str, label: str, *, auto_connect: bool) -> None:
    path = codex_state_path()
    if not path.exists():
        raise SystemExit(f"Codex global state file not found: {path}")

    host_id = discovered_host_id(alias)
    normalized_path = posixpath.normpath(remote_path)
    project_id = None
    desktop_running = codex_desktop_process_running()

    for attempt in range(1, 6):
        data = json.loads(path.read_text(encoding="utf-8"))
        data.setdefault("remote-connection-auto-connect-by-host-id", {})[host_id] = bool(auto_connect)
        atoms: dict[str, Any] = data.setdefault("electron-persisted-atom-state", {})
        atoms.setdefault("agent-mode-by-host-id", {})[host_id] = "full-access"
        project_id = upsert_remote_project(data, host_id, normalized_path, label)
        write_codex_state(path, data)

        time.sleep(2.0)
        check = json.loads(path.read_text(encoding="utf-8"))
        projects = check.get("remote-projects", [])
        project_ok = any(
            isinstance(project, dict)
            and project.get("id") == project_id
            and project.get("hostId") == host_id
            and posixpath.normpath(str(project.get("remotePath", ""))) == normalized_path
            for project in projects
        )
        if (
            check.get("remote-connection-auto-connect-by-host-id", {}).get(host_id) == bool(auto_connect)
            and check.get("electron-persisted-atom-state", {}).get("agent-mode-by-host-id", {}).get(host_id) == "full-access"
            and check.get("project-order", [None])[0] == project_id
            and check.get("active-remote-project-id") == project_id
            and project_ok
        ):
            break

        if attempt == 5:
            print(
                "Warning: Codex Desktop rewrote part of the seeded remote state. "
                "The SSH alias is registered, but the app may still need Settings > Connections > Refresh.",
                file=sys.stderr,
            )

    print(f"Registered Codex remote project {label!r} at {normalized_path}")
    print(f"Seeded discovered Codex SSH host {alias!r} as {host_id!r}")
    if desktop_running:
        print(
            "Note: Codex Desktop is currently running. It may rewrite remote-project state from memory; "
            "if the project does not appear automatically, restart Codex Desktop or add the project once through Home > New remote project.",
            file=sys.stderr,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-id", required=True, help="Runpod GPU ID/name, as accepted by runpodctl.")
    parser.add_argument("--repo", required=True, help="Git URL or GitHub owner/repo shorthand to clone.")
    parser.add_argument("--name", help="Pod and Codex remote display name.")
    parser.add_argument("--image", default=DEFAULT_IMAGE, help="Runpod pod image to use when --template-id is omitted.")
    parser.add_argument("--template-id", help="Runpod template ID. Overrides --image.")
    parser.add_argument("--gpu-count", type=int)
    parser.add_argument("--container-disk-gb", type=int)
    parser.add_argument("--volume-gb", type=int, default=DEFAULT_VOLUME_GB)
    parser.add_argument("--volume-mount-path")
    parser.add_argument("--ports", default=DEFAULT_PORTS, help="Comma-separated exposed ports. Default: 22/tcp.")
    parser.add_argument("--identity", help=f"SSH private key path stored in Codex remote config. Default: {DEFAULT_IDENTITY}.")
    parser.add_argument("--api-key", help="Runpod API key. Prefer --prompt-api-key to avoid storing secrets in shell history.")
    parser.add_argument("--prompt-api-key", action="store_true", help="Prompt for a Runpod API key if runpodctl is not configured.")
    parser.add_argument("--clone-path", help="Destination path on pod when cloning. Default: /workspace/<repo-name>.")
    parser.add_argument("--existing-project-path", help="Use an already-present project path on the pod instead of cloning.")
    parser.add_argument("--pod-id", help="Use an existing pod instead of creating one.")
    parser.add_argument("--wait-seconds", type=int, default=15)
    parser.add_argument("--skip-bootstrap", action="store_true")
    parser.add_argument("--skip-register", action="store_true")
    parser.add_argument("--skip-remote-project", action="store_true")
    parser.add_argument("--skip-managed-state", action="store_true", help="Do not seed legacy codex-managed remote connection state.")
    parser.add_argument("--skip-ssh-config", action="store_true")
    parser.add_argument("--skip-feature-flag", action="store_true")
    parser.add_argument("--copy-codex-auth", action="store_true", help="Copy local Codex auth.json to /root/.codex/auth.json on the pod.")
    parser.add_argument("--copy-gh-auth", action="store_true", help="Copy local GitHub CLI auth token into gh on the pod.")
    parser.add_argument("--copy-git-identity", action="store_true", help="Copy local global git user.name and user.email to the pod.")
    parser.add_argument("--git-identity-scope", choices=["global", "repo"], default="global", help="Where to set copied git identity on the pod. Default: global.")
    parser.add_argument("--project-label", help="Codex remote project label. Default: repository directory name.")
    parser.add_argument("--auto-connect", action="store_true")
    args = parser.parse_args()

    if not shutil.which("runpodctl"):
        raise SystemExit("runpodctl is not on PATH. Install it first: curl -sSL https://cli.runpod.net | bash")
    if not shutil.which("ssh"):
        raise SystemExit("ssh is not on PATH.")

    repo_url, repo_dir = normalize_repo(args.repo)
    args.name = slug(args.name or f"{repo_dir}-{args.gpu_id}")
    project_path = args.existing_project_path or args.clone_path or f"/workspace/{repo_dir}"
    skip_clone = bool(args.existing_project_path)
    configured_identity = args.identity or DEFAULT_IDENTITY

    ensure_runpod_config(args)
    ensure_runpod_ssh_key(configured_identity)

    pod_id = args.pod_id or create_pod(args)
    print(f"Pod ID: {pod_id}")
    userhost, port, parsed_identity = ssh_info(pod_id)
    identity = args.identity or parsed_identity or configured_identity
    print(f"SSH: {userhost} port {port} identity {identity}")

    if not args.skip_feature_flag:
        ensure_codex_remote_connections_feature()
    if not args.skip_ssh_config:
        register_ssh_config_alias(args.name, userhost, port, identity)

    if not args.skip_bootstrap:
        remote_shell(userhost, port, identity, bootstrap_command(repo_url, project_path, skip_clone=skip_clone))
    if args.copy_codex_auth:
        copy_codex_auth(userhost, port, identity)
    if args.copy_gh_auth:
        copy_github_auth(userhost, port, identity)
    if args.copy_git_identity:
        copy_git_identity(userhost, port, identity, clone_path=project_path, scope=args.git_identity_scope)

    if not args.skip_register:
        if not args.skip_managed_state:
            register_remote(args.name, userhost, port, identity, auto_connect=args.auto_connect)
        if not args.skip_remote_project:
            seed_discovered_remote_alias(args.name, project_path, args.project_label or repo_dir, auto_connect=args.auto_connect)

    print(f"Ready: {args.name} -> {userhost}:{port}, project at {project_path}")
    print(f"Cleanup hint: if this was a disposable verification pod, run `runpodctl pod delete {pod_id}` when done; `stop` leaves volume storage billable.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
