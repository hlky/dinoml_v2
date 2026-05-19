from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

import dinoml as dml
from dinoml.backends import get_backend_spec, registered_backend_names
from dinoml.backends import rocm as rocm_backend
from dinoml.backends.rocm import ensure_rocm_support_libs
from dinoml.kernels.codegen import create_codegen_plan
from tests.cases import elementwise_case


def test_rocm_target_is_registered_as_distinct_backend():
    assert "rocm" in registered_backend_names()
    target = dml.Target("rocm")
    assert target.to_json() == {
        "name": "rocm",
        "arch": "gfx1201",
        "no_tf32": False,
        "use_fp16_acc": False,
    }

    spec = get_backend_spec("rocm")
    assert spec.default_arch == "gfx1201"
    assert spec.resolve_build_function().__name__ == "build_rocm_module"
    assert spec.cmake.support_build_targets == (
        "dinoml_runtime",
        "dinoml_rocm_runtime",
        "dinoml_rocm_kernels",
    )
    suffix = ".dll" if os.name == "nt" else ".dylib" if sys.platform == "darwin" else ".so"
    assert spec.support_libraries["rocm_runtime_library"].endswith(f"dinoml_rocm_runtime{suffix}")
    assert spec.support_libraries["kernel_library"].endswith(f"dinoml_rocm_kernels{suffix}")


def test_rocm_codegen_plan_uses_arch_specific_support_cache(tmp_path):
    manifest = {
        "target": {"name": "rocm", "arch": "gfx1201"},
        "cache_key": "abcdef0123456789",
        "required_kernels": [],
    }

    plan = create_codegen_plan(manifest, tmp_path)

    assert plan.target == {"name": "rocm", "arch": "gfx1201"}
    assert plan.support_cache_dir == tmp_path / "support" / "rocm-gfx1201" / "abcdef0123456789"


def test_rocm_runtime_paths_resolve_from_active_rocm_sdk_command(tmp_path, monkeypatch):
    sdk_root = tmp_path / "sdk"
    sdk_bin = sdk_root / "bin"
    llvm_bin = sdk_root / "lib" / "llvm" / "bin"
    sdk_bin.mkdir(parents=True)
    llvm_bin.mkdir(parents=True)

    monkeypatch.setattr(
        rocm_backend.shutil,
        "which",
        lambda name: "rocm-sdk.exe" if name == "rocm-sdk" else None,
    )

    def fake_run(cmd, *, text, stdout, stderr):
        del text, stdout, stderr
        assert cmd[:2] == ["rocm-sdk.exe", "path"]
        if cmd[-1] == "--root":
            value = sdk_root
        elif cmd[-1] == "--bin":
            value = sdk_bin
        else:
            raise AssertionError(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout=str(value))

    monkeypatch.setattr(rocm_backend.subprocess, "run", fake_run)

    assert rocm_backend._rocm_runtime_paths() == [str(sdk_bin), str(llvm_bin)]


def test_rocm_runtime_paths_can_use_active_python_rocm_sdk_module(tmp_path, monkeypatch):
    sdk_root = tmp_path / "sdk"
    sdk_bin = sdk_root / "bin"
    llvm_bin = sdk_root / "lib" / "llvm" / "bin"
    sdk_bin.mkdir(parents=True)
    llvm_bin.mkdir(parents=True)

    def fake_which(name):
        if name == "python":
            return "python.exe"
        return None

    monkeypatch.setattr(rocm_backend.shutil, "which", fake_which)

    def fake_run(cmd, *, text, stdout, stderr):
        del text, stdout, stderr
        if cmd[:2] == ["python.exe", "-c"]:
            return subprocess.CompletedProcess(cmd, 0)
        assert cmd[:4] == ["python.exe", "-m", "rocm_sdk", "path"]
        if cmd[-1] == "--root":
            value = sdk_root
        elif cmd[-1] == "--bin":
            value = sdk_bin
        else:
            raise AssertionError(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout=str(value))

    monkeypatch.setattr(rocm_backend.subprocess, "run", fake_run)

    assert rocm_backend._rocm_runtime_paths() == [str(sdk_bin), str(llvm_bin)]


def test_rocm_runtime_paths_allow_regular_hip_sdk_without_rocm_sdk(monkeypatch):
    monkeypatch.setattr(rocm_backend.shutil, "which", lambda name: None)

    assert rocm_backend._rocm_runtime_paths() == []


def test_rocm_support_configure_requires_ninja(monkeypatch):
    monkeypatch.setattr(rocm_backend.shutil, "which", lambda name: None)

    with pytest.raises(RuntimeError, match="require Ninja"):
        rocm_backend._with_default_cmake_generator(["cmake", "-S", ".", "-B", "build"])


def test_visual_studio_environment_filters_vcvars_payload(monkeypatch):
    vcvars = Path("C:/VS/VC/Auxiliary/Build/vcvars64.bat")
    monkeypatch.setattr(rocm_backend, "_find_vcvars64", lambda: vcvars)

    def fake_run(cmd, *, text, stdout, stderr):
        del text, stdout, stderr
        assert cmd == ["cmd", "/s", "/c", f'"{vcvars}" >nul && set']
        output = "\n".join(
            [
                r"INCLUDE=C:\VS\include",
                r"LIB=C:\VS\lib",
                r"LIBPATH=C:\VS\libpath",
                r"PATH=C:\VS\bin;C:\Windows\System32",
                r"VCToolsVersion=14.43.34808",
                "UNRELATED_SECRET=do-not-copy",
            ]
        )
        return subprocess.CompletedProcess(cmd, 0, stdout=output)

    monkeypatch.setattr(rocm_backend.subprocess, "run", fake_run)

    env = rocm_backend._visual_studio_environment()

    assert env == {
        "INCLUDE": r"C:\VS\include",
        "LIB": r"C:\VS\lib",
        "LIBPATH": r"C:\VS\libpath",
        "PATH": r"C:\VS\bin;C:\Windows\System32",
        "VCToolsVersion": "14.43.34808",
    }


def test_rocm_compile_fails_before_claiming_any_op_support(tmp_path):
    case = elementwise_case()

    with pytest.raises(NotImplementedError, match="rocm backend does not support op fused_elementwise"):
        dml.compile(case.build_spec(), dml.Target("rocm"), tmp_path / "elementwise_rocm.dinoml")


@pytest.mark.skipif(
    os.environ.get("DINOML_RUN_ROCM_SUPPORT_BUILD_SMOKE") != "1",
    reason="set DINOML_RUN_ROCM_SUPPORT_BUILD_SMOKE=1 with rocm-sdk on PATH",
)
def test_rocm_support_libraries_build_with_real_toolchain(tmp_path, monkeypatch):
    monkeypatch.setenv("DINOML_CACHE_DIR", str(tmp_path / "cache"))

    libs = ensure_rocm_support_libs("gfx1201")

    assert libs.runtime_lib.exists()
    assert libs.rocm_runtime_lib.exists()
    assert libs.kernels_lib.exists()
    assert (Path(libs.rocm_runtime_lib).parent / "support_manifest.json").exists()
