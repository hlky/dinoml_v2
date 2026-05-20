#!/usr/bin/env python
"""Generate a Codex shell_environment_policy for a project venv."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def toml_string(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def split_path(path_value: str) -> list[str]:
    return [part for part in path_value.split(os.pathsep) if part and "${PATH}" not in part]


def is_base_python_path(path: str) -> bool:
    normalized = os.path.normcase(os.path.normpath(os.path.expandvars(path)))
    parts = normalized.split(os.sep)
    if len(parts) < 3:
        return False
    if parts[-1] == "scripts":
        parts = parts[:-1]
    return len(parts) >= 3 and parts[-1].startswith("python3") and parts[-2] == "python" and parts[-3] == "programs"


def current_windows_path() -> list[str]:
    if os.name != "nt":
        return split_path(os.environ.get("PATH", ""))

    paths: list[str] = []
    try:
        import winreg

        keys = [
            (winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"),
            (winreg.HKEY_CURRENT_USER, "Environment"),
        ]
        for hive, subkey in keys:
            try:
                with winreg.OpenKey(hive, subkey) as key:
                    value, _ = winreg.QueryValueEx(key, "Path")
                    paths.extend(split_path(value))
            except OSError:
                pass
    except ImportError:
        pass

    if not paths:
        paths.extend(split_path(os.environ.get("PATH", "")))
    return paths


def dedupe_path(paths: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for path in paths:
        key = os.path.normcase(os.path.normpath(os.path.expandvars(path)))
        if key in seen:
            continue
        seen.add(key)
        result.append(path)
    return result


def filter_base_python_paths(paths: list[str]) -> list[str]:
    return [path for path in paths if not is_base_python_path(path)]


def resolve_venv(path: str) -> Path:
    return Path(path).expanduser().resolve()


def venv_bin_dir(venv: Path) -> Path:
    scripts = venv / "Scripts"
    if scripts.exists() or os.name == "nt":
        return scripts
    return venv / "bin"


def generate_policy(venv: Path) -> str:
    path_entries = dedupe_path([str(venv_bin_dir(venv)), *filter_base_python_paths(current_windows_path())])
    path_value = os.pathsep.join(path_entries)
    return "\n".join(
        [
            "[shell_environment_policy]",
            'inherit = "all"',
            f"set = {{ VIRTUAL_ENV = {toml_string(str(venv))}, PATH = {toml_string(path_value)} }}",
            "",
        ]
    )


def write_project_config(policy: str, config_path: Path) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(policy, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a Codex .codex/config.toml shell_environment_policy for a venv."
    )
    parser.add_argument("venv", help="Virtual environment path. Relative paths are resolved from the current directory.")
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write the generated policy to .codex/config.toml instead of printing it.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(".codex") / "config.toml",
        help="Config path to use with --write. Defaults to .codex/config.toml.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    venv = resolve_venv(args.venv)
    policy = generate_policy(venv)

    if args.write:
        write_project_config(policy, args.config)
        print(f"Wrote {args.config}")
    else:
        sys.stdout.write(policy)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
