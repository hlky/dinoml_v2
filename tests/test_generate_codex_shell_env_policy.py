from __future__ import annotations

import importlib.util
from pathlib import Path, PurePosixPath


def load_module():
    module_path = Path(__file__).resolve().parents[1] / "tools" / "generate_codex_shell_env_policy.py"
    spec = importlib.util.spec_from_file_location("generate_codex_shell_env_policy", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generate_policy_uses_windows_path_key_and_excludes_uppercase_path(monkeypatch):
    module = load_module()
    monkeypatch.setattr(module, "current_windows_path", lambda: [r"C:\Base\Bin", r"C:\Python\Scripts"])

    policy = module.generate_policy(Path(r"C:\repo\.venv\rocm"), os_name="nt")

    assert 'exclude = ["PATH"]' in policy
    assert 'Path = "C:\\\\repo\\\\.venv\\\\rocm\\\\Scripts;C:\\\\Base\\\\Bin;C:\\\\Python\\\\Scripts"' in policy
    assert "PATH =" not in policy


def test_generate_policy_uses_linux_path_key_without_exclude(monkeypatch):
    module = load_module()
    monkeypatch.setenv("PATH", "/usr/local/bin:/usr/bin:/bin")

    policy = module.generate_policy(PurePosixPath("/repo/.venv/rocm"), os_name="posix")

    assert "exclude =" not in policy
    assert 'PATH = "/repo/.venv/rocm/bin:/usr/local/bin:/usr/bin:/bin"' in policy
    assert "Path =" not in policy
