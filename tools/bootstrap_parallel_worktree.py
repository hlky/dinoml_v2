from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_LOCAL_ENVIRONMENT = "rocm"


class BootstrapPlan:
    def __init__(
        self,
        *,
        task_id: str,
        worktree_path: Path,
        branch: str,
        cache_dir: Path,
        venv_path: Path,
        config_path: Path,
    ) -> None:
        self.task_id = task_id
        self.worktree_path = worktree_path
        self.branch = branch
        self.cache_dir = cache_dir
        self.venv_path = venv_path
        self.config_path = config_path


def bootstrap_parallel_worktree(
    *,
    task_id: str,
    branch: str | None,
    base_cache_dir: str | Path,
    venv_path: str | Path,
    worktree_root: str | Path | None = None,
    create_venv: bool = False,
    install_editable: bool = False,
    python_exe: str | None = None,
) -> dict[str, Any]:
    plan = build_bootstrap_plan(
        task_id=task_id,
        branch=branch,
        base_cache_dir=base_cache_dir,
        venv_path=venv_path,
        worktree_root=worktree_root,
    )
    _git_worktree_add(plan.worktree_path, plan.branch)
    _copy_base_cache(Path(base_cache_dir), plan.cache_dir)
    if create_venv:
        _create_venv(plan.venv_path, python_exe=python_exe)
    if install_editable:
        _install_editable(plan.worktree_path, plan.venv_path)
    write_worktree_codex_config(plan.config_path, venv_path=plan.venv_path, cache_dir=plan.cache_dir)
    return bootstrap_summary(plan)


def build_bootstrap_plan(
    *,
    task_id: str,
    branch: str | None,
    base_cache_dir: str | Path,
    venv_path: str | Path,
    worktree_root: str | Path | None = None,
) -> BootstrapPlan:
    normalized_task_id = str(task_id).strip()
    if not normalized_task_id:
        raise ValueError("task_id must be non-empty")
    root = default_worktree_root() if worktree_root is None else Path(worktree_root).resolve()
    worktree_path = root / normalized_task_id
    branch_name = default_branch_name(normalized_task_id) if branch is None else str(branch).strip()
    if not branch_name:
        raise ValueError("branch must be non-empty")
    cache_dir = worktree_path / ".cache" / "dinoml_v2"
    resolved_venv = Path(venv_path)
    if not resolved_venv.is_absolute():
        resolved_venv = (worktree_path / resolved_venv).resolve()
    else:
        resolved_venv = resolved_venv.resolve()
    return BootstrapPlan(
        task_id=normalized_task_id,
        worktree_path=worktree_path.resolve(),
        branch=branch_name,
        cache_dir=cache_dir.resolve(),
        venv_path=resolved_venv,
        config_path=(worktree_path / ".codex" / "config.toml").resolve(),
    )


def default_branch_name(task_id: str) -> str:
    slug = "".join(ch.lower() if ch.isalnum() else "-" for ch in task_id).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return f"codex/ops/{slug or 'worktree'}"


def default_worktree_root() -> Path:
    return REPO_ROOT.parent / f"{REPO_ROOT.name}_worktrees"


def write_worktree_codex_config(config_path: Path, *, venv_path: Path, cache_dir: Path) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(build_codex_config_text(venv_path=venv_path, cache_dir=cache_dir), encoding="utf-8")


def build_codex_config_text(*, venv_path: Path, cache_dir: Path) -> str:
    lines = [
        'model_instructions_file = "base.md"',
        'developer_instructions_file = "developer.md"',
        "",
        "include_permissions_instructions = false",
        "include_apps_instructions = false",
        "include_collaboration_mode_instructions = false",
        "",
        f'default_local_environment = "{DEFAULT_LOCAL_ENVIRONMENT}"',
        "",
        "[local_environments.msvc]",
        'description = "MSVC x64 developer shell"',
        "",
        "[local_environments.msvc.script]",
        f"script = {toml_string(r'C:\\Program Files\\Microsoft Visual Studio\\2022\\Professional\\VC\\Auxiliary\\Build\\vcvars64.bat')}",
        'shell = "cmd"',
        "args = []",
        "",
        "[local_environments.rocm]",
        'description = "ROCm venv"',
        "",
        "[local_environments.rocm.shell_environment_policy]",
        'inherit = "all"',
        f"set = {{ VIRTUAL_ENV = {toml_string(str(venv_path))}, DINOML_CACHE_DIR = {toml_string(str(cache_dir))} }}",
        f"path_prepend = [{toml_string(str(_venv_bin_dir(venv_path, os.name)))}]",
        "",
        "[skills]",
        "include_instructions = true",
        "",
    ]
    return "\n".join(lines)


def bootstrap_summary(plan: BootstrapPlan) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "parallel_worktree_bootstrap",
        "task_id": plan.task_id,
        "branch": plan.branch,
        "worktree_path": str(plan.worktree_path),
        "cache_dir": str(plan.cache_dir),
        "venv_path": str(plan.venv_path),
        "config_path": str(plan.config_path),
        "spawn_workdir": str(plan.worktree_path),
    }


def _git_worktree_add(worktree_path: Path, branch: str) -> None:
    worktree_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["git", "worktree", "add", str(worktree_path), "-b", branch, "HEAD"]
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(
            "git worktree add failed\n"
            f"Command: {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )


def _copy_base_cache(base_cache_dir: Path, cache_dir: Path) -> None:
    source = base_cache_dir.resolve()
    if not source.exists():
        raise FileNotFoundError(f"Base cache directory does not exist: {source}")
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, cache_dir)


def _create_venv(venv_path: Path, *, python_exe: str | None) -> None:
    if venv_path.exists():
        shutil.rmtree(venv_path)
    interpreter = python_exe or sys.executable
    cmd = [interpreter, "-m", "venv", str(venv_path)]
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(
            "venv creation failed\n"
            f"Command: {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )


def _install_editable(worktree_path: Path, venv_path: Path) -> None:
    python_bin = _venv_bin_dir(venv_path) / ("python.exe" if os.name == "nt" else "python")
    cmd = [str(python_bin), "-m", "pip", "install", "-e", "."]
    proc = subprocess.run(cmd, cwd=str(worktree_path), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(
            "editable install failed\n"
            f"Command: {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a DinoML worktree, copy a seeded base cache into it, and write worktree-local Codex config."
    )
    parser.add_argument("task_id", help="Worktree task id / folder name, e.g. T20_one_hot")
    parser.add_argument("--branch", help="Git branch to create. Defaults to codex/ops/<task-id>.")
    parser.add_argument("--base-cache-dir", required=True, help="Seed cache root to copy, e.g. H:/dinoml_v2_rocm_cache")
    parser.add_argument("--venv-path", required=True, help="Venv path to write into the worktree config.")
    parser.add_argument("--worktree-root", help="Root directory for created worktrees.")
    parser.add_argument("--create-venv", action="store_true", help="Create the venv before writing config.")
    parser.add_argument("--install-editable", action="store_true", help="Run pip install -e . in the created venv.")
    parser.add_argument("--python-exe", help="Python interpreter to use with --create-venv. Defaults to the current interpreter.")
    parser.add_argument("--out", help="Write the JSON summary to this file.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = bootstrap_parallel_worktree(
        task_id=args.task_id,
        branch=args.branch,
        base_cache_dir=args.base_cache_dir,
        venv_path=args.venv_path,
        worktree_root=args.worktree_root,
        create_venv=args.create_venv,
        install_editable=args.install_editable,
        python_exe=args.python_exe,
    )
    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def toml_string(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _venv_bin_dir(venv: Path, os_name: str | None = None) -> Path:
    current_os = os.name if os_name is None else os_name
    if current_os == "nt":
        return venv / "Scripts"
    scripts = venv / "Scripts"
    if hasattr(scripts, "exists") and scripts.exists():
        return scripts
    return venv / "bin"


if __name__ == "__main__":
    raise SystemExit(main())
