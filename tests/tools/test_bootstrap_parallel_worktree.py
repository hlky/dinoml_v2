from __future__ import annotations

import importlib.util
from pathlib import Path


def load_module():
    module_path = Path(__file__).resolve().parents[2] / "tools" / "bootstrap_parallel_worktree.py"
    spec = importlib.util.spec_from_file_location("bootstrap_parallel_worktree", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_codex_config_text_includes_cache_and_escaped_windows_paths():
    module = load_module()
    config = module.build_codex_config_text(
        venv_path=Path(r"H:\dinoml_v2_worktrees\T20\.venv\rocm"),
        cache_dir=Path(r"H:\dinoml_v2_worktrees\T20\.cache\dinoml_v2"),
    )

    assert 'default_local_environment = "rocm"' in config
    assert 'model_instructions_file = "base.md"' in config
    assert 'developer_instructions_file = "developer.md"' in config
    assert "include_permissions_instructions = false" in config
    assert "include_apps_instructions = false" in config
    assert "include_collaboration_mode_instructions = false" in config
    assert '[local_environments.msvc]' in config
    assert 'description = "MSVC x64 developer shell"' in config
    assert "[local_environments.msvc.script]" in config
    assert "vcvars64.bat" in config
    assert 'shell = "cmd"' in config
    assert "args = []" in config
    assert "[local_environments.rocm]" in config
    assert 'description = "ROCm venv"' in config
    assert "[local_environments.rocm.shell_environment_policy]" in config
    assert 'inherit = "all"' in config
    assert 'VIRTUAL_ENV = "H:\\\\dinoml_v2_worktrees\\\\T20\\\\.venv\\\\rocm"' in config
    assert 'DINOML_CACHE_DIR = "H:\\\\dinoml_v2_worktrees\\\\T20\\\\.cache\\\\dinoml_v2"' in config
    assert 'path_prepend = ["H:\\\\dinoml_v2_worktrees\\\\T20\\\\.venv\\\\rocm\\\\Scripts"]' in config
    assert "Path =" not in config
    assert "[skills]" in config
    assert "include_instructions = true" in config


def test_copy_base_cache_replaces_existing_tree(tmp_path):
    module = load_module()
    source = tmp_path / "base"
    target = tmp_path / "target"
    (source / "support" / "rocm-gfx1201").mkdir(parents=True)
    (source / "support" / "rocm-gfx1201" / "seed.txt").write_text("seed", encoding="utf-8")
    target.mkdir()
    (target / "old.txt").write_text("old", encoding="utf-8")

    module._copy_base_cache(source, target)

    assert (target / "support" / "rocm-gfx1201" / "seed.txt").read_text(encoding="utf-8") == "seed"
    assert not (target / "old.txt").exists()


def test_build_bootstrap_plan_defaults_branch_and_cache_dir(tmp_path, monkeypatch):
    module = load_module()
    monkeypatch.setattr(module, "default_worktree_root", lambda: tmp_path / "worktrees")

    plan = module.build_bootstrap_plan(
        task_id="T20_one_hot",
        branch=None,
        base_cache_dir=tmp_path / "base_cache",
        venv_path=Path(".venv") / "rocm",
    )

    assert plan.branch == "codex/ops/t20-one-hot"
    assert plan.worktree_path == (tmp_path / "worktrees" / "T20_one_hot").resolve()
    assert plan.cache_dir == (tmp_path / "worktrees" / "T20_one_hot" / ".cache" / "dinoml_v2").resolve()
    assert plan.venv_path == (tmp_path / "worktrees" / "T20_one_hot" / ".venv" / "rocm").resolve()
