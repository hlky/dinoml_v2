#!/usr/bin/env python
"""Generate a Codex shell_environment_policy for a project venv."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def toml_string(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def toml_list(values: list[str]) -> str:
    return "[" + ", ".join(toml_string(value) for value in values) + "]"


def path_separator(os_name: str | None = None) -> str:
    current_os = os.name if os_name is None else os_name
    return ";" if current_os == "nt" else ":"


def split_path(path_value: str, os_name: str | None = None) -> list[str]:
    placeholders = ("${PATH}", "$PATH", "%PATH%")
    separator = path_separator(os_name)
    return [part for part in path_value.split(separator) if part and not any(token in part for token in placeholders)]


def is_base_python_path(path: str) -> bool:
    normalized = os.path.normcase(os.path.normpath(os.path.expandvars(path)))
    parts = normalized.split(os.sep)
    if len(parts) < 3:
        return False
    if parts[-1] == "scripts":
        parts = parts[:-1]
    return len(parts) >= 3 and parts[-1].startswith("python3") and parts[-2] == "python" and parts[-3] == "programs"


def current_windows_path() -> list[str]:
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
                    paths.extend(split_path(value, "nt"))
            except OSError:
                pass
    except ImportError:
        pass

    if not paths:
        paths.extend(split_path(os.environ.get("PATH", ""), "nt"))
    return paths


def current_system_path(os_name: str | None = None) -> list[str]:
    current_os = os.name if os_name is None else os_name
    if current_os == "nt":
        return current_windows_path()
    return split_path(os.environ.get("PATH", ""), current_os)


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


def venv_bin_dir(venv: Path, os_name: str | None = None) -> Path:
    current_os = os.name if os_name is None else os_name
    if current_os == "nt":
        return venv / "Scripts"
    scripts = venv / "Scripts"
    if hasattr(scripts, "exists") and scripts.exists():
        return scripts
    return venv / "bin"


def path_policy_key(os_name: str | None = None) -> str:
    current_os = os.name if os_name is None else os_name
    return "Path" if current_os == "nt" else "PATH"


def path_policy_excludes(os_name: str | None = None) -> list[str]:
    current_os = os.name if os_name is None else os_name
    return ["PATH"] if current_os == "nt" else []


def generate_policy(venv: Path, os_name: str | None = None) -> str:
    current_os = os.name if os_name is None else os_name
    path_key = path_policy_key(current_os)
    excludes = path_policy_excludes(current_os)
    path_entries = dedupe_path([str(venv_bin_dir(venv, current_os)), *filter_base_python_paths(current_system_path(current_os))])
    path_value = path_separator(current_os).join(path_entries)
    lines = ["[shell_environment_policy]", 'inherit = "all"']
    if excludes:
        lines.append(f"exclude = {toml_list(excludes)}")
    lines.extend(
        [
            "set = {",
            f"  VIRTUAL_ENV = {toml_string(str(venv))},",
            f"  {path_key} = {toml_string(path_value)}",
            "}",
            "",
        ]
    )
    return "\n".join(lines)


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
