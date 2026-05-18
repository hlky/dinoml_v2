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
) -> None:
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
                "runtime_lib": str(runtime_lib),
                "kernels_lib": str(kernels_lib),
                "runtime_include": str(support_libs.runtime_include),
                "common_include": str(support_libs.common_include),
                "kernels_include": str(support_libs.kernels_include),
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
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={artifact_dir}",
            *([f"-DDINOML_ENABLE_OPENMP={os.environ['DINOML_ENABLE_OPENMP']}"] if "DINOML_ENABLE_OPENMP" in os.environ else []),
        ],
        cwd=artifact_dir,
    )
    _run_cmake(["cmake", "--build", str(build_dir), "--target", "module", "--parallel"], cwd=artifact_dir)


def ensure_cpu_support_libs(*, kernel_manifest: Mapping[str, object] | None = None) -> CpuSupportLibs:
    repo_root = Path(__file__).resolve().parents[3]
    cache_root = Path(os.environ.get("DINOML_CACHE_DIR", Path.home() / ".cache" / "dinoml_v2"))
    manifest_key = "full" if kernel_manifest is None else str(kernel_manifest.get("support_cache_key", kernel_manifest["cache_key"]))[:16]
    support_root = cache_root / "support" / "cpu" / manifest_key
    build_dir = support_root / "build"
    lib_dir = support_root / "lib"
    runtime_lib = lib_dir / "libdinoml_runtime.so"
    kernels_lib = lib_dir / "libdinoml_cpu_kernels.so"
    lib_dir.mkdir(parents=True, exist_ok=True)
    configure_cmd = [
        "cmake",
        "-S",
        str(repo_root),
        "-B",
        str(build_dir),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DDINOML_ENABLE_CUDA=OFF",
        f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={lib_dir}",
    ]
    if "DINOML_ENABLE_OPENMP" in os.environ:
        configure_cmd.append(f"-DDINOML_ENABLE_OPENMP={os.environ['DINOML_ENABLE_OPENMP']}")
    _run_cmake(configure_cmd, cwd=repo_root)
    _run_cmake(
        [
            "cmake",
            "--build",
            str(build_dir),
            "--target",
            "dinoml_runtime",
            "dinoml_cpu_kernels",
            "--parallel",
        ],
        cwd=repo_root,
    )
    if not runtime_lib.exists() or not kernels_lib.exists():
        raise RuntimeError(f"Expected CPU support libraries under {lib_dir}, but they were not produced")
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
    if "-S" not in cmd or shutil.which("ninja") is None:
        return cmd
    return [*cmd, "-G", "Ninja"]
