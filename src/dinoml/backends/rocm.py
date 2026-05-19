from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from dinoml.backends.registry import _shared_library_name
from dinoml.ir import write_json
from dinoml.kernels.manifest import build_support_manifest
from dinoml.lowering.gpu import render_template
from dinoml.lowering.rocm import render_rocm_module
from dinoml.lowering.ops import collect_generated_sources


_CMAKE_ENV: dict[str, str] | None = None

_VISUAL_STUDIO_ENV_KEYS = frozenset(
    {
        "INCLUDE",
        "LIB",
        "LIBPATH",
        "PATH",
        "VCINSTALLDIR",
        "VCToolsInstallDir",
        "VCToolsVersion",
        "VSINSTALLDIR",
        "WindowsLibPath",
        "WindowsSdkBinPath",
        "WindowsSdkDir",
        "WindowsSDKLibVersion",
        "WindowsSDKVersion",
    }
)


@dataclass(frozen=True)
class RocmSupportLibs:
    runtime_lib: Path
    rocm_runtime_lib: Path
    kernels_lib: Path
    runtime_include: Path
    common_include: Path
    kernels_include: Path


def build_rocm_module(
    ir: Mapping[str, Any],
    *,
    target: Any,
    artifact_dir: Path,
    generated_src_dir: Path,
    kernel_manifest: Mapping[str, Any],
) -> Mapping[str, str] | None:
    support_libs = ensure_rocm_support_libs(target.arch, kernel_manifest=kernel_manifest)
    artifact_lib_dir = artifact_dir / "lib"
    artifact_lib_dir.mkdir(parents=True, exist_ok=True)
    runtime_lib = artifact_lib_dir / support_libs.runtime_lib.name
    rocm_runtime_lib = artifact_lib_dir / support_libs.rocm_runtime_lib.name
    kernels_lib = artifact_lib_dir / support_libs.kernels_lib.name
    shutil.copy2(support_libs.runtime_lib, runtime_lib)
    shutil.copy2(support_libs.rocm_runtime_lib, rocm_runtime_lib)
    shutil.copy2(support_libs.kernels_lib, kernels_lib)

    generated_src_dir.mkdir(parents=True, exist_ok=True)
    tensor_map = {tensor["name"]: tensor for tensor in ir["tensors"]}
    generated_sources = collect_generated_sources(
        "rocm",
        ir["nodes"],
        tensor_map,
        generated_src_dir=generated_src_dir,
    )
    (generated_src_dir / "module.hip").write_text(
        render_rocm_module(ir, generated_kernels=generated_sources["kernels"], kernel_manifest=kernel_manifest),
        encoding="utf-8",
    )
    repo_root = _repo_root()
    (generated_src_dir / "CMakeLists.txt").write_text(
        render_template(
            "rocm_module_cmake.txt.j2",
            {
                "rocm_sdk_cmake": _cmake_path(repo_root / "cmake" / "DinoMLROCmSdk.cmake"),
                "runtime_lib": _cmake_path(runtime_lib),
                "rocm_runtime_lib": _cmake_path(rocm_runtime_lib),
                "kernels_lib": _cmake_path(kernels_lib),
                "runtime_implib": _cmake_path(_import_library_path(support_libs.runtime_lib)),
                "rocm_runtime_implib": _cmake_path(_import_library_path(support_libs.rocm_runtime_lib)),
                "kernels_implib": _cmake_path(_import_library_path(support_libs.kernels_lib)),
                "runtime_include": _cmake_path(support_libs.runtime_include),
                "common_include": _cmake_path(support_libs.common_include),
                "kernels_include": _cmake_path(support_libs.kernels_include),
            },
        ),
        encoding="utf-8",
    )

    build_dir = generated_src_dir / "build"
    _run_cmake(
        [
            "cmake",
            "-S",
            str(generated_src_dir),
            "-B",
            str(build_dir),
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_HIP_ARCHITECTURES={_cmake_arch(target.arch)}",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={artifact_dir}",
            f"-DCMAKE_RUNTIME_OUTPUT_DIRECTORY={artifact_dir}",
        ],
        cwd=artifact_dir,
    )
    _run_cmake(["cmake", "--build", str(build_dir), "--target", "module", "--parallel"], cwd=artifact_dir)
    module_lib = artifact_dir / "module.so"
    if not module_lib.exists():
        raise RuntimeError(f"Expected ROCm generated module at {module_lib}, but it was not produced")
    return None


def ensure_rocm_support_libs(arch: str, *, kernel_manifest: Mapping[str, Any] | None = None) -> RocmSupportLibs:
    repo_root = _repo_root()
    cache_root = Path(os.environ.get("DINOML_CACHE_DIR", Path.home() / ".cache" / "dinoml_v2"))
    manifest_key = "full" if kernel_manifest is None else str(kernel_manifest.get("support_cache_key", kernel_manifest["cache_key"]))[:16]
    support_root = cache_root / "support" / f"rocm-{_cmake_arch(arch)}" / manifest_key
    build_dir = support_root / "build"
    lib_dir = support_root / "lib"
    runtime_lib = lib_dir / _shared_library_name("dinoml_runtime")
    rocm_runtime_lib = lib_dir / _shared_library_name("dinoml_rocm_runtime")
    kernels_lib = lib_dir / _shared_library_name("dinoml_rocm_kernels")
    lib_dir.mkdir(parents=True, exist_ok=True)
    _prepare_cmake_build_dir(build_dir)
    configure_cmd = [
        "cmake",
        "-S",
        str(repo_root),
        "-B",
        str(build_dir),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DDINOML_ENABLE_CUDA=OFF",
        "-DDINOML_ENABLE_ROCM=ON",
        f"-DCMAKE_HIP_ARCHITECTURES={_cmake_arch(arch)}",
        f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={lib_dir}",
        f"-DCMAKE_RUNTIME_OUTPUT_DIRECTORY={lib_dir}",
        f"-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY={lib_dir}",
    ]
    _run_cmake(configure_cmd, cwd=repo_root)
    _run_cmake(
        [
            "cmake",
            "--build",
            str(build_dir),
            "--target",
            "dinoml_runtime",
            "dinoml_rocm_runtime",
            "dinoml_rocm_kernels",
            "--parallel",
        ],
        cwd=repo_root,
    )
    missing = [str(path) for path in (runtime_lib, rocm_runtime_lib, kernels_lib) if not path.exists()]
    if missing:
        raise RuntimeError(f"Expected ROCm support libraries under {lib_dir}, but these were not produced: {missing}")
    support_target = (
        dict(kernel_manifest.get("target", {"name": "rocm", "arch": _cmake_arch(arch)}))
        if kernel_manifest is not None
        else {"name": "rocm", "arch": _cmake_arch(arch)}
    )
    write_json(
        lib_dir / "support_manifest.json",
        build_support_manifest(
            target=support_target,
            libraries={
                "runtime": runtime_lib.name,
                "rocm_runtime": rocm_runtime_lib.name,
                "kernels": kernels_lib.name,
            },
            required_kernel_cache_key=None if kernel_manifest is None else str(kernel_manifest.get("support_cache_key", kernel_manifest["cache_key"])),
        ),
    )
    return RocmSupportLibs(
        runtime_lib=runtime_lib,
        rocm_runtime_lib=rocm_runtime_lib,
        kernels_lib=kernels_lib,
        runtime_include=repo_root / "runtime" / "include",
        common_include=repo_root / "kernels" / "common" / "include",
        kernels_include=repo_root / "kernels" / "rocm" / "include",
    )


def _import_library_path(shared_library: Path) -> Path | None:
    if os.name != "nt":
        return None
    candidate = shared_library.with_suffix(".lib")
    return candidate if candidate.exists() else None


def _cmake_path(path: Path | None) -> str:
    if path is None:
        return ""
    return path.resolve().as_posix()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _cmake_arch(arch: str) -> str:
    if not arch:
        raise ValueError("Expected ROCm arch like 'gfx1201'")
    if not str(arch).startswith("gfx"):
        raise ValueError(f"Expected ROCm arch like 'gfx1201', got {arch!r}")
    return str(arch)


def _run_cmake(cmd: list[str], *, cwd: Path) -> None:
    cmd = _with_default_cmake_generator(cmd)
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=_cmake_env(),
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "CMake command failed\n"
            f"Command: {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )


def _with_default_cmake_generator(cmd: list[str]) -> list[str]:
    if len(cmd) < 2 or Path(cmd[0]).name != "cmake":
        return cmd
    if "--build" in cmd or "-G" in cmd:
        return cmd
    if "-S" not in cmd:
        return cmd
    if shutil.which("ninja") is None:
        raise RuntimeError(
            "ROCm support builds require Ninja on PATH. Activate the ROCm development "
            "environment or install the DinoML dev dependencies into that environment."
        )
    return [*cmd, "-G", "Ninja"]


def _prepare_cmake_build_dir(build_dir: Path) -> None:
    cache_path = build_dir / "CMakeCache.txt"
    if not cache_path.exists():
        return
    generator = _cmake_cache_value(cache_path, "CMAKE_GENERATOR")
    if generator and generator != "Ninja":
        shutil.rmtree(build_dir)


def _cmake_cache_value(cache_path: Path, key: str) -> str | None:
    for line in cache_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith(f"{key}:"):
            _, value = line.split("=", 1)
            return value
    return None


def _cmake_env() -> dict[str, str]:
    global _CMAKE_ENV
    if _CMAKE_ENV is not None:
        return dict(_CMAKE_ENV)
    env = os.environ.copy()
    if os.name == "nt":
        env.update(_visual_studio_environment())
        _prepend_paths(env, _rocm_runtime_paths())
    _CMAKE_ENV = env
    return dict(env)


def _rocm_runtime_paths() -> list[str]:
    paths = []
    root = _run_rocm_sdk_path("--root")
    bin_dir = _run_rocm_sdk_path("--bin")
    if bin_dir:
        paths.append(bin_dir)
    if root:
        paths.append(str(Path(root) / "lib" / "llvm" / "bin"))
    return paths


def _run_rocm_sdk_path(arg: str) -> str | None:
    rocm_sdk = _rocm_sdk_command()
    if rocm_sdk is None:
        return None
    proc = subprocess.run(
        [*rocm_sdk, "path", arg],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    if proc.returncode != 0:
        return None
    value = proc.stdout.strip()
    return value or None


def _rocm_sdk_command() -> list[str] | None:
    rocm_sdk = shutil.which("rocm-sdk") or shutil.which("rocm_sdk")
    if rocm_sdk is not None:
        return [rocm_sdk]
    seen = set()
    for python in (sys.executable, shutil.which("python"), shutil.which("python3")):
        if not python:
            continue
        normalized = os.path.normcase(os.path.abspath(python))
        if normalized in seen:
            continue
        seen.add(normalized)
        if _python_has_rocm_sdk(python):
            return [python, "-m", "rocm_sdk"]
    return None


def _python_has_rocm_sdk(python: str) -> bool:
    proc = subprocess.run(
        [
            python,
            "-c",
            "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('rocm_sdk') else 1)",
        ],
        text=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return proc.returncode == 0


def _prepend_paths(env: dict[str, str], paths: list[str]) -> None:
    existing = env.get("PATH", "")
    separator = os.pathsep
    seen = set()
    ordered = []
    for path in paths:
        if not path:
            continue
        normalized = os.path.normcase(os.path.abspath(path))
        if normalized in seen or not os.path.exists(path):
            continue
        seen.add(normalized)
        ordered.append(path)
    if ordered:
        env["PATH"] = separator.join([*ordered, existing]) if existing else separator.join(ordered)


def _visual_studio_environment() -> dict[str, str]:
    vcvars = _find_vcvars64()
    if vcvars is None:
        return {}
    proc = subprocess.run(
        ["cmd", "/s", "/c", f'"{vcvars}" >nul && set'],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    if proc.returncode != 0:
        return {}
    values: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key in _VISUAL_STUDIO_ENV_KEYS:
            values[key] = value
    return values


def _find_vcvars64() -> Path | None:
    program_files_x86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
    vswhere = Path(program_files_x86) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    if vswhere.exists():
        proc = subprocess.run(
            [
                str(vswhere),
                "-latest",
                "-products",
                "*",
                "-requires",
                "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                "-property",
                "installationPath",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        install = proc.stdout.strip()
        if proc.returncode == 0 and install:
            candidate = Path(install) / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
            if candidate.exists():
                return candidate
    visual_studio_root = Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Microsoft Visual Studio"
    if not visual_studio_root.exists():
        return None
    return next(visual_studio_root.rglob("vcvars64.bat"), None)
