from __future__ import annotations

from dinoml.backends import cpu as cpu_backend


def test_cpu_cmake_generator_prefers_visual_studio_on_windows(monkeypatch):
    monkeypatch.setattr(cpu_backend.os, "name", "nt")
    monkeypatch.delenv("CMAKE_GENERATOR", raising=False)
    monkeypatch.delenv("DINOML_CMAKE_GENERATOR", raising=False)
    monkeypatch.delenv("DINOML_CMAKE_GENERATOR_PLATFORM", raising=False)

    cmd = cpu_backend._with_default_cmake_generator(["cmake", "-S", ".", "-B", "build"])

    assert cmd[-4:] == ["-G", "Visual Studio 17 2022", "-A", "x64"]


def test_cpu_cmake_generator_honors_explicit_environment_generator(monkeypatch):
    monkeypatch.setattr(cpu_backend.os, "name", "nt")
    monkeypatch.setenv("DINOML_CMAKE_GENERATOR", "Ninja")
    monkeypatch.delenv("DINOML_CMAKE_GENERATOR_PLATFORM", raising=False)

    cmd = cpu_backend._with_default_cmake_generator(["cmake", "-S", ".", "-B", "build"])

    assert cmd[-2:] == ["-G", "Ninja"]


def test_cpu_platform_library_names_follow_windows_loader_convention(monkeypatch):
    monkeypatch.setattr(cpu_backend.os, "name", "nt")

    assert cpu_backend._shared_library_name("dinoml_runtime") == "dinoml_runtime.dll"
    assert cpu_backend._generated_module_name() == "module.dll"


def test_cpu_build_dir_is_cleared_when_cached_generator_changes(tmp_path, monkeypatch):
    monkeypatch.setattr(cpu_backend.os, "name", "nt")
    monkeypatch.delenv("CMAKE_GENERATOR", raising=False)
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    (build_dir / "CMakeCache.txt").write_text("CMAKE_GENERATOR:INTERNAL=Ninja\n", encoding="utf-8")
    stale_file = build_dir / "stale.txt"
    stale_file.write_text("old", encoding="utf-8")

    cpu_backend._prepare_cmake_build_dir(build_dir)

    assert not stale_file.exists()


def test_cpu_build_dir_preserves_matching_environment_generator(tmp_path, monkeypatch):
    monkeypatch.setenv("CMAKE_GENERATOR", "Ninja")
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    (build_dir / "CMakeCache.txt").write_text("CMAKE_GENERATOR:INTERNAL=Ninja\n", encoding="utf-8")
    kept_file = build_dir / "kept.txt"
    kept_file.write_text("current", encoding="utf-8")

    cpu_backend._prepare_cmake_build_dir(build_dir)

    assert kept_file.exists()
