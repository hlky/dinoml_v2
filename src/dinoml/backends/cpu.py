from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from dinoml.ir import write_json
from dinoml.kernels.manifest import build_support_manifest
from dinoml.lowering.cpu import render_cpu_module, render_template
from dinoml.lowering.ops import collect_generated_sources


@dataclass(frozen=True)
class CpuSupportLibs:
    runtime_lib: Path
    kernels_lib: Path
    runtime_implib: Path | None
    kernels_implib: Path | None
    runtime_include: Path
    common_include: Path
    kernels_include: Path


def build_cpu_module(
    ir: Mapping,
    *,
    target,
    artifact_dir: Path,
    generated_src_dir: Path,
    kernel_manifest: Mapping[str, object],
) -> Mapping[str, str]:
    support_libs = ensure_cpu_support_libs(kernel_manifest=kernel_manifest)
    artifact_lib_dir = artifact_dir / "lib"
    artifact_lib_dir.mkdir(parents=True, exist_ok=True)
    runtime_lib = artifact_lib_dir / support_libs.runtime_lib.name
    kernels_lib = artifact_lib_dir / support_libs.kernels_lib.name
    shutil.copy2(support_libs.runtime_lib, runtime_lib)
    shutil.copy2(support_libs.kernels_lib, kernels_lib)

    generated_src_dir.mkdir(parents=True, exist_ok=True)
    tensor_map = {tensor["name"]: tensor for tensor in ir["tensors"]}
    generated_sources = collect_generated_sources(
        "cpu",
        ir["nodes"],
        tensor_map,
        generated_src_dir=generated_src_dir,
    )
    (generated_src_dir / "module.cpp").write_text(
        render_cpu_module(ir, generated_kernels=generated_sources["kernels"]),
        encoding="utf-8",
    )
    (generated_src_dir / "CMakeLists.txt").write_text(
        render_template(
            "cpu_module_cmake.txt.j2",
            {
                "runtime_lib": _cmake_path(runtime_lib),
                "kernels_lib": _cmake_path(kernels_lib),
                "runtime_implib": _cmake_path(support_libs.runtime_implib) if support_libs.runtime_implib is not None else None,
                "kernels_implib": _cmake_path(support_libs.kernels_implib) if support_libs.kernels_implib is not None else None,
                "link_kernels_lib": os.name != "nt" or support_libs.kernels_implib is not None,
                "runtime_include": _cmake_path(support_libs.runtime_include),
                "common_include": _cmake_path(support_libs.common_include),
                "kernels_include": _cmake_path(support_libs.kernels_include),
            },
        ),
        encoding="utf-8",
    )
    build_dir = generated_src_dir / "build"
    _prepare_cmake_build_dir(build_dir)
    _run_cmake(
        [
            "cmake",
            "-S",
            str(generated_src_dir),
            "-B",
            str(build_dir),
            "-DCMAKE_BUILD_TYPE=Release",
            *_cmake_output_dir_args("CMAKE_LIBRARY_OUTPUT_DIRECTORY", artifact_dir),
            *_cmake_output_dir_args("CMAKE_RUNTIME_OUTPUT_DIRECTORY", artifact_dir),
            *([f"-DDINOML_ENABLE_OPENMP={os.environ['DINOML_ENABLE_OPENMP']}"] if "DINOML_ENABLE_OPENMP" in os.environ else []),
        ],
        cwd=artifact_dir,
    )
    _run_cmake(["cmake", "--build", str(build_dir), "--config", "Release", "--target", "module", "--parallel"], cwd=artifact_dir)
    return {
        "module": _generated_module_name(),
        "runtime_library": f"lib/{support_libs.runtime_lib.name}",
        "kernel_library": f"lib/{support_libs.kernels_lib.name}",
    }


def ensure_cpu_support_libs(*, kernel_manifest: Mapping[str, object] | None = None) -> CpuSupportLibs:
    repo_root = Path(__file__).resolve().parents[3]
    cache_root = Path(os.environ.get("DINOML_CACHE_DIR", Path.home() / ".cache" / "dinoml_v2"))
    manifest_key = "full" if kernel_manifest is None else str(kernel_manifest.get("support_cache_key", kernel_manifest["cache_key"]))[:16]
    support_root = cache_root / "support" / "cpu" / manifest_key
    build_dir = support_root / "build"
    lib_dir = support_root / "lib"
    runtime_lib = lib_dir / _shared_library_name("dinoml_runtime")
    kernels_lib = lib_dir / _shared_library_name("dinoml_cpu_kernels")
    runtime_implib = lib_dir / "dinoml_runtime.lib" if os.name == "nt" else None
    kernels_implib = lib_dir / "dinoml_cpu_kernels.lib" if os.name == "nt" else None
    lib_dir.mkdir(parents=True, exist_ok=True)
    configure_cmd = [
        "cmake",
        "-S",
        str(repo_root),
        "-B",
        str(build_dir),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DDINOML_ENABLE_CUDA=OFF",
        *_cmake_output_dir_args("CMAKE_LIBRARY_OUTPUT_DIRECTORY", lib_dir),
        *_cmake_output_dir_args("CMAKE_RUNTIME_OUTPUT_DIRECTORY", lib_dir),
        *_cmake_output_dir_args("CMAKE_ARCHIVE_OUTPUT_DIRECTORY", lib_dir),
    ]
    if "DINOML_ENABLE_OPENMP" in os.environ:
        configure_cmd.append(f"-DDINOML_ENABLE_OPENMP={os.environ['DINOML_ENABLE_OPENMP']}")
    _prepare_cmake_build_dir(build_dir)
    _run_cmake(configure_cmd, cwd=repo_root)
    _run_cmake(
        [
            "cmake",
            "--build",
            str(build_dir),
            "--config",
            "Release",
            "--target",
            "dinoml_runtime",
            "dinoml_cpu_kernels",
            "--parallel",
        ],
        cwd=repo_root,
    )
    if not runtime_lib.exists() or not kernels_lib.exists():
        raise RuntimeError(f"Expected CPU support libraries under {lib_dir}, but they were not produced")
    if runtime_implib is not None and not runtime_implib.exists():
        raise RuntimeError(f"Expected CPU import libraries under {lib_dir}, but they were not produced")
    if kernels_implib is not None and not kernels_implib.exists():
        kernels_implib = None
    write_json(
        lib_dir / "support_manifest.json",
        build_support_manifest(
            target={"name": "cpu", "arch": "native"},
            libraries={"runtime": runtime_lib.name, "kernels": kernels_lib.name},
            required_kernel_cache_key=None if kernel_manifest is None else str(kernel_manifest.get("support_cache_key", kernel_manifest["cache_key"])),
        ),
    )
    return CpuSupportLibs(
        runtime_lib=runtime_lib,
        kernels_lib=kernels_lib,
        runtime_implib=runtime_implib,
        kernels_implib=kernels_implib,
        runtime_include=repo_root / "runtime" / "include",
        common_include=repo_root / "kernels" / "common" / "include",
        kernels_include=repo_root / "kernels" / "cpu" / "include",
    )


def _run_cmake(cmd: list[str], *, cwd: Path) -> None:
    cmd = _with_default_cmake_generator(cmd)
    proc = subprocess.run(cmd, cwd=str(cwd), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
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
    generator = _selected_cmake_generator()
    if generator is not None:
        name, platform = generator
        out = [*cmd, "-G", name]
        if platform:
            out.extend(["-A", platform])
        return out
    if shutil.which("ninja") is None:
        return cmd
    return [*cmd, "-G", "Ninja"]


def _selected_cmake_generator() -> tuple[str, str | None] | None:
    if os.environ.get("CMAKE_GENERATOR"):
        return None
    if "DINOML_CMAKE_GENERATOR" in os.environ:
        generator = os.environ["DINOML_CMAKE_GENERATOR"]
        platform = os.environ.get("DINOML_CMAKE_GENERATOR_PLATFORM")
        return generator, platform
    if os.name == "nt":
        return "Visual Studio 17 2022", os.environ.get("DINOML_CMAKE_GENERATOR_PLATFORM", "x64")
    return None


def _prepare_cmake_build_dir(build_dir: Path) -> None:
    cache_path = build_dir / "CMakeCache.txt"
    if not cache_path.exists():
        return
    if os.environ.get("CMAKE_GENERATOR"):
        expected_generator = os.environ["CMAKE_GENERATOR"]
    else:
        expected = _selected_cmake_generator()
        expected_generator = expected[0] if expected is not None else ("Ninja" if shutil.which("ninja") is not None else None)
    if expected_generator is None:
        return
    generator = _cmake_cache_value(cache_path, "CMAKE_GENERATOR")
    if generator and generator != expected_generator:
        shutil.rmtree(build_dir)


def _cmake_cache_value(cache_path: Path, key: str) -> str | None:
    for line in cache_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith(f"{key}:"):
            _, value = line.split("=", 1)
            return value
    return None


def _cmake_output_dir_args(name: str, path: Path) -> list[str]:
    args = [f"-D{name}={_cmake_path(path)}"]
    if os.name == "nt":
        args.append(f"-D{name}_RELEASE={_cmake_path(path)}")
    return args


def _cmake_path(path: Path) -> str:
    return Path(path).resolve().as_posix()


def _shared_library_name(stem: str) -> str:
    if os.name == "nt":
        return f"{stem}.dll"
    return f"lib{stem}.so"


def _generated_module_name() -> str:
    if os.name == "nt":
        return "module.dll"
    return "module.so"
