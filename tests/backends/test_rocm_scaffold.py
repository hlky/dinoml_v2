from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

import dinoml as dml
from dinoml.backends import get_backend_spec, registered_backend_names
from dinoml.backends import rocm as rocm_backend
from dinoml.backends.rocm import ensure_rocm_support_libs
from dinoml.ir import read_json
from dinoml.kernels.codegen import create_codegen_plan
from dinoml.ops.elementwise import FusedElementwise
from dinoml.runtime import load
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
    monkeypatch.setattr(rocm_backend.sys, "executable", "current-python-without-rocm-sdk.exe")

    def fake_run(cmd, *, text, stdout, stderr):
        del text, stdout, stderr
        if cmd[:2] == ["current-python-without-rocm-sdk.exe", "-c"]:
            return subprocess.CompletedProcess(cmd, 1)
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


def test_rocm_runtime_paths_prefer_current_python_rocm_sdk_module(tmp_path, monkeypatch):
    sdk_root = tmp_path / "sdk"
    sdk_bin = sdk_root / "bin"
    llvm_bin = sdk_root / "lib" / "llvm" / "bin"
    sdk_bin.mkdir(parents=True)
    llvm_bin.mkdir(parents=True)

    monkeypatch.setattr(rocm_backend.shutil, "which", lambda name: None)
    monkeypatch.setattr(rocm_backend.sys, "executable", "active-python.exe")

    def fake_run(cmd, *, text, stdout, stderr):
        del text, stdout, stderr
        if cmd[:2] == ["active-python.exe", "-c"]:
            return subprocess.CompletedProcess(cmd, 0)
        assert cmd[:4] == ["active-python.exe", "-m", "rocm_sdk", "path"]
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
    monkeypatch.setattr(rocm_backend, "_python_has_rocm_sdk", lambda python: False)

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


def test_rocm_fused_elementwise_is_publicly_admitted():
    binding = FusedElementwise.backend_kernels["rocm"]

    assert binding.library == "model"
    assert binding.source_template == "fused_elementwise_gpu"


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


@pytest.mark.skipif(
    os.environ.get("DINOML_RUN_ROCM_MODULE_COMPILE_SMOKE") != "1",
    reason="set DINOML_RUN_ROCM_MODULE_COMPILE_SMOKE=1 with rocm-sdk on the active Python/PATH",
)
def test_rocm_fused_elementwise_module_compiles_with_real_toolchain(tmp_path, monkeypatch):
    if not _rocm_module_compile_toolchain_available():
        pytest.skip("ROCm module compile toolchain not found from active Python/PATH")
    monkeypatch.setenv("DINOML_CACHE_DIR", str(tmp_path / "cache"))
    case = elementwise_case()

    artifact = dml.compile(case.build_spec(), dml.Target("rocm"), tmp_path / "elementwise_rocm.dinoml")

    assert (artifact.path / "module.so").exists()
    if os.name == "nt":
        assert not list(artifact.path.glob("amdhip*.dll"))
        assert not list((artifact.path / "lib").glob("amdhip*.dll"))
    manifest = read_json(artifact.path / "manifest.json")
    assert manifest["target"]["name"] == "rocm"
    assert manifest["files"]["rocm_runtime_library"].endswith("dinoml_rocm_runtime.dll" if os.name == "nt" else "libdinoml_rocm_runtime.so")
    source_manifest = read_json(artifact.path / "debug" / "generated_src" / "source_manifest.json")
    assert source_manifest["target"] == "rocm"
    assert {source["op"] for source in source_manifest["sources"]} == {"fused_elementwise"}
    assert all(str(source["emitted_source_path"]).endswith(".hip") for source in source_manifest["sources"])

    module = load(artifact.path, load_constants=False)
    try:
        assert module.target_name == "rocm"
    finally:
        module.close()


@pytest.mark.skipif(
    os.environ.get("DINOML_RUN_ROCM_HEADER_COMPILE_SMOKE") != "1",
    reason="set DINOML_RUN_ROCM_HEADER_COMPILE_SMOKE=1 with hipcc or .venv/rocm available",
)
def test_rocm_common_headers_compile_with_real_hipcc(tmp_path):
    hipcc = _find_hipcc()
    if hipcc is None:
        pytest.skip("hipcc not found on PATH, HIPCC, or .venv/rocm/Scripts")
    repo_root = Path(__file__).resolve().parents[2]
    source = tmp_path / "dinoml_rocm_header_smoke.hip"
    obj = tmp_path / ("dinoml_rocm_header_smoke.obj" if os.name == "nt" else "dinoml_rocm_header_smoke.o")
    source.write_text(
        r"""
#include <dinoml/device.h>
#include <dinoml/runtime_rocm.h>
#include <dinoml/math.h>
#include <dinoml/tensor_accessor.h>

extern "C" __global__ void dinoml_rocm_header_smoke_kernel(float* y, half* h, dinoml::bfloat16* b) {
  y[0] = dinoml::math::to_float(h[0]) + dinoml::math::to_float(b[0]);
  h[0] = dinoml::math::from_float<half>(y[0]);
  b[0] = dinoml::math::from_float<dinoml::bfloat16>(y[0]);
  dinoml::access::TensorAccessor accessor;
  y[1] = static_cast<float>(accessor.index(0)) + LDG(y);
}

extern "C" int dinoml_rocm_header_smoke_launch(float* y, half* h, dinoml::bfloat16* b, dinoml::DeviceStream stream) {
  dinoml_rocm_header_smoke_kernel<<<1, 1, 0, stream>>>(y, h, b);
  DINO_ROCM_CHECK(hipGetLastError());
  return 0;
}
""",
        encoding="utf-8",
    )
    proc = subprocess.run(
        [
            str(hipcc),
            "--offload-arch=" + os.environ.get("DINOML_ROCM_TEST_ARCH", "gfx1201"),
            "-std=c++17",
            "-I",
            str(repo_root / "runtime" / "include"),
            "-I",
            str(repo_root / "kernels" / "common" / "include"),
            "-c",
            str(source),
            "-o",
            str(obj),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    _wait_for_path(obj, timeout_s=10.0)
    assert proc.returncode == 0 and obj.exists(), (
        f"hipcc failed to compile ROCm header smoke\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )


def _find_hipcc() -> Path | None:
    configured = os.environ.get("HIPCC")
    if configured:
        path = Path(configured)
        if path.exists():
            return path
    found = shutil.which("hipcc")
    if found is not None:
        return Path(found)
    repo_root = Path(__file__).resolve().parents[2]
    suffix = "hipcc.exe" if os.name == "nt" else "hipcc"
    local = repo_root / ".venv" / "rocm" / "Scripts" / suffix
    if local.exists():
        return local
    return None


def _rocm_module_compile_toolchain_available() -> bool:
    if rocm_backend._rocm_sdk_command() is not None:
        return True
    if shutil.which("hipconfig") is not None:
        return True
    return bool(os.environ.get("ROCM_PATH") or os.environ.get("HIP_PATH"))


def _wait_for_path(path: Path, *, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if path.exists():
            return
        time.sleep(0.1)
