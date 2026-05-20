from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from dinoml.backends.cuda_libraries import require_cuda_library
from dinoml.ir import read_json, write_json
from dinoml.kernels.manifest import build_support_manifest
from dinoml.kernels.providers.cutlass.bmm import cutlass_bmm_cmake_target, cutlass_bmm_static_library_name, cutlass_bmm_used_candidate_plan
from dinoml.kernels.providers.cutlass.conv import cutlass_conv_cmake_target, cutlass_conv_static_library_name
from dinoml.kernels.providers.cutlass.gemm import cutlass_gemm_cmake_target, cutlass_gemm_static_library_name
from dinoml.libgguf_cuda import (
    file_sha256,
    libgguf_provenance_key,
    libgguf_source_provenance,
)
from dinoml.lowering.gpu import render_template
from dinoml.lowering.cuda import render_cuda_module
from dinoml.lowering.ops import collect_generated_sources


CUTLASS_GEMM_CMAKE_CHUNK_SIZE = 16


@dataclass(frozen=True)
class SupportLibs:
    runtime_lib: Path
    cuda_runtime_lib: Path
    kernels_lib: Path
    cutlass_gemm_archives: tuple[Path, ...]
    cutlass_bmm_archives: tuple[Path, ...]
    cutlass_conv_archives: tuple[Path, ...]
    gguf_cuda_native_lib: Path | None
    gguf_cuda_native_manifest: Path | None
    runtime_include: Path
    common_include: Path
    kernels_include: Path


def build_cuda_module(
    ir: Mapping[str, Any],
    *,
    target: Any,
    artifact_dir: Path,
    generated_src_dir: Path,
    kernel_manifest: Mapping[str, Any],
) -> Mapping[str, str] | None:
    support_libs = ensure_cuda_support_libs(target.arch, kernel_manifest=kernel_manifest)
    artifact_lib_dir = artifact_dir / "lib"
    artifact_lib_dir.mkdir(parents=True, exist_ok=True)
    runtime_lib = artifact_lib_dir / support_libs.runtime_lib.name
    cuda_runtime_lib = artifact_lib_dir / support_libs.cuda_runtime_lib.name
    kernels_lib = artifact_lib_dir / support_libs.kernels_lib.name
    gguf_cuda_native_lib = (
        None if support_libs.gguf_cuda_native_lib is None else artifact_lib_dir / support_libs.gguf_cuda_native_lib.name
    )
    gguf_cuda_native_manifest = (
        None
        if support_libs.gguf_cuda_native_manifest is None
        else artifact_lib_dir / support_libs.gguf_cuda_native_manifest.name
    )
    shutil.copy2(support_libs.runtime_lib, runtime_lib)
    shutil.copy2(support_libs.cuda_runtime_lib, cuda_runtime_lib)
    shutil.copy2(support_libs.kernels_lib, kernels_lib)
    if support_libs.gguf_cuda_native_lib is not None and gguf_cuda_native_lib is not None:
        shutil.copy2(support_libs.gguf_cuda_native_lib, gguf_cuda_native_lib)
    if support_libs.gguf_cuda_native_manifest is not None and gguf_cuda_native_manifest is not None:
        shutil.copy2(support_libs.gguf_cuda_native_manifest, gguf_cuda_native_manifest)

    generated_src_dir.mkdir(parents=True, exist_ok=True)
    tensor_map = {tensor["name"]: tensor for tensor in ir["tensors"]}
    generated_sources = collect_generated_sources(
        "cuda",
        ir["nodes"],
        tensor_map,
        generated_src_dir=generated_src_dir,
    )
    module_kernel_manifest = dict(kernel_manifest)
    if gguf_cuda_native_lib is not None:
        module_kernel_manifest["gguf_cuda_native_library"] = f"lib/{gguf_cuda_native_lib.name}"
    (generated_src_dir / "module.cu").write_text(
        render_cuda_module(ir, generated_kernels=generated_sources["kernels"], kernel_manifest=module_kernel_manifest),
        encoding="utf-8",
    )
    (generated_src_dir / "CMakeLists.txt").write_text(
        render_template(
            "gpu_module_cmake.txt.j2",
            {
                "runtime_lib": str(runtime_lib),
                "cuda_runtime_lib": str(cuda_runtime_lib),
                "kernels_lib": str(kernels_lib),
                "cutlass_gemm_archives": [str(path) for path in support_libs.cutlass_gemm_archives],
                "cutlass_bmm_archives": [str(path) for path in support_libs.cutlass_bmm_archives],
                "cutlass_conv_archives": [str(path) for path in support_libs.cutlass_conv_archives],
                "gguf_cuda_native_lib": "" if gguf_cuda_native_lib is None else str(gguf_cuda_native_lib),
                "gguf_cuda_native_lib_kind": "" if gguf_cuda_native_lib is None else _cmake_library_kind(gguf_cuda_native_lib),
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
            f"-DCMAKE_CUDA_ARCHITECTURES={_cmake_arch(target.arch)}",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={artifact_dir}",
            *(
                [f"-DDINOML_ENABLE_CUDA_FAST_MATH={os.environ['DINOML_ENABLE_CUDA_FAST_MATH']}"]
                if "DINOML_ENABLE_CUDA_FAST_MATH" in os.environ
                else []
            ),
        ],
        cwd=artifact_dir,
    )
    _run_cmake(["cmake", "--build", str(build_dir), "--target", "module", "--parallel"], cwd=artifact_dir)
    if gguf_cuda_native_lib is None:
        return None
    files = {"gguf_cuda_native_library": f"lib/{gguf_cuda_native_lib.name}"}
    if gguf_cuda_native_manifest is not None:
        files["gguf_cuda_native_manifest"] = f"lib/{gguf_cuda_native_manifest.name}"
    return files


def ensure_cuda_support_libs(arch: str, *, kernel_manifest: Mapping[str, Any] | None = None) -> SupportLibs:
    repo_root = _repo_root()
    cache_root = Path(os.environ.get("DINOML_CACHE_DIR", Path.home() / ".cache" / "dinoml_v2"))
    manifest_key = "full" if kernel_manifest is None else kernel_manifest.get("support_cache_key", kernel_manifest["cache_key"])[:16]
    support_root = cache_root / "support" / f"cuda-{_cmake_arch(arch)}" / manifest_key
    build_dir = support_root / "build"
    lib_dir = support_root / "lib"
    runtime_lib = lib_dir / "libdinoml_runtime.so"
    cuda_runtime_lib = lib_dir / "libdinoml_cuda_runtime.so"
    kernels_lib = lib_dir / "libdinoml_cuda_kernels.so"
    cutlass_gemm_archives: tuple[Path, ...] = ()
    cutlass_bmm_archives: tuple[Path, ...] = ()
    cutlass_conv_archives: tuple[Path, ...] = ()
    gguf_cuda_native_lib = None
    gguf_cuda_native_manifest = None

    lib_dir.mkdir(parents=True, exist_ok=True)
    _run_cmake(
        [
            "cmake",
            "-S",
            str(repo_root),
            "-B",
            str(build_dir),
            "-DCMAKE_BUILD_TYPE=Release",
            "-DDINOML_ENABLE_CUDA=ON",
            "-DDINOML_ENABLE_CUTLASS_GEMM=OFF",
            "-DDINOML_ENABLE_CUTLASS_BMM=OFF",
            "-DDINOML_ENABLE_CUTLASS_CONV=OFF",
            "-DDINOML_ENABLE_LIBGGUF_CUDA=OFF",
            f"-DCMAKE_CUDA_ARCHITECTURES={_cmake_arch(arch)}",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={lib_dir}",
            f"-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY={lib_dir}",
        ],
        cwd=repo_root,
    )
    _run_cmake(
        [
            "cmake",
            "--build",
            str(build_dir),
            "--target",
            "dinoml_runtime",
            "dinoml_cuda_runtime",
            "dinoml_cuda_kernels",
            "--parallel",
        ],
        cwd=repo_root,
    )
    if not runtime_lib.exists() or not cuda_runtime_lib.exists() or not kernels_lib.exists():
        raise RuntimeError(f"Expected support libraries under {lib_dir}, but they were not produced")
    if _requires_kernel_library(kernel_manifest, "cutlass_gemm"):
        cutlass_gemm_archives = _ensure_cmake_cutlass_gemm_archives(arch, kernel_manifest)
    if _requires_kernel_library(kernel_manifest, "cutlass_bmm"):
        cutlass_bmm_archives = _ensure_cmake_cutlass_bmm_archives(arch, kernel_manifest)
    if _requires_kernel_library(kernel_manifest, "cutlass_conv"):
        cutlass_conv_archives = _ensure_cmake_cutlass_conv_archives(arch, kernel_manifest)
    if _requires_gguf_cuda_native_library(kernel_manifest):
        gguf_cuda_native_lib, gguf_cuda_native_manifest = _ensure_cmake_libgguf_cuda_native_archive(
            arch,
            cache_root=cache_root,
            cache_key=kernel_manifest.get("support_cache_key", kernel_manifest["cache_key"])[:16],
            repo_root=repo_root,
        )
    libraries = {
        "runtime": runtime_lib.name,
        "cuda_runtime": cuda_runtime_lib.name,
        "kernels": kernels_lib.name,
    }
    if cutlass_gemm_archives:
        libraries["cutlass_gemm_static"] = [archive.name for archive in cutlass_gemm_archives]
    if cutlass_bmm_archives:
        libraries["cutlass_bmm_static"] = [archive.name for archive in cutlass_bmm_archives]
    if cutlass_conv_archives:
        libraries["cutlass_conv_static"] = [archive.name for archive in cutlass_conv_archives]
    if gguf_cuda_native_lib is not None:
        libraries["gguf_cuda_native_static"] = gguf_cuda_native_lib.name
    default_target = {"name": "cuda", "arch": f"sm_{_cmake_arch(arch)}"}
    support_target = dict(kernel_manifest.get("target", default_target)) if kernel_manifest is not None else default_target
    write_json(
        lib_dir / "support_manifest.json",
        build_support_manifest(
            target=support_target,
            libraries=libraries,
            required_kernel_cache_key=None if kernel_manifest is None else kernel_manifest.get("support_cache_key", kernel_manifest["cache_key"]),
        ),
    )
    return SupportLibs(
        runtime_lib=runtime_lib,
        cuda_runtime_lib=cuda_runtime_lib,
        kernels_lib=kernels_lib,
        cutlass_gemm_archives=cutlass_gemm_archives,
        cutlass_bmm_archives=cutlass_bmm_archives,
        cutlass_conv_archives=cutlass_conv_archives,
        gguf_cuda_native_lib=gguf_cuda_native_lib,
        gguf_cuda_native_manifest=gguf_cuda_native_manifest,
        runtime_include=repo_root / "runtime" / "include",
        common_include=repo_root / "kernels" / "common" / "include",
        kernels_include=repo_root / "kernels" / "cuda" / "include",
    )


def _requires_kernel_library(kernel_manifest: Mapping[str, Any] | None, library: str) -> bool:
    if kernel_manifest is None:
        return False
    return any(item.get("kernel_library") == library for item in kernel_manifest.get("required_kernels", []))


def _ensure_cmake_cutlass_gemm_archives(arch: str, kernel_manifest: Mapping[str, Any]) -> tuple[Path, ...]:
    require_cuda_library("cutlass")
    repo_root = _repo_root()
    cache_root = Path(os.environ.get("DINOML_CACHE_DIR", Path.home() / ".cache" / "dinoml_v2"))
    arch_num = _cmake_arch(arch)
    support_root = cache_root / "support" / f"cuda-{arch_num}" / "cutlass-gemm" / "cmake-full"
    build_dir = support_root / "build"
    lib_dir = support_root / "lib"
    modules = _required_cutlass_gemm_modules(kernel_manifest)
    archives = tuple(lib_dir / module["archive"] for module in modules)
    lib_dir.mkdir(parents=True, exist_ok=True)
    _prepare_cmake_build_dir(build_dir)
    if any(not archive.exists() for archive in archives) or not build_dir.exists():
        _run_cmake(
            [
                "cmake",
                "-S",
                str(repo_root),
                "-B",
                str(build_dir),
                "-DCMAKE_BUILD_TYPE=Release",
                "-DDINOML_ENABLE_CUDA=ON",
                "-DDINOML_ENABLE_CUTLASS_GEMM=ON",
                "-DDINOML_ENABLE_CUTLASS_BMM=OFF",
                "-DDINOML_ENABLE_CUTLASS_CONV=OFF",
                "-DDINOML_ENABLE_LIBGGUF_CUDA=OFF",
                f"-DDINOML_CUTLASS_GEMM_CHUNK_SIZE={CUTLASS_GEMM_CMAKE_CHUNK_SIZE}",
                f"-DCMAKE_CUDA_ARCHITECTURES={arch_num}",
                f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={lib_dir}",
                f"-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY={lib_dir}",
                f"-DCMAKE_RUNTIME_OUTPUT_DIRECTORY={lib_dir}",
            ],
            cwd=repo_root,
        )
    targets = [module["target"] for module in modules]
    _run_cmake(
        [
            "cmake",
            "--build",
            str(build_dir),
            "--target",
            *targets,
            "--parallel",
        ],
        cwd=repo_root,
    )
    missing = [str(archive) for archive in archives if not archive.exists()]
    if missing:
        raise RuntimeError(f"Expected CMake-built CUTLASS GEMM static archives, but these were not produced: {missing}")
    write_json(
        lib_dir / "cutlass_gemm_manifest.json",
        {
            "schema_version": 3,
            "target": {"name": "cuda", "arch": f"sm_{arch_num}"},
            "provider": "cutlass",
            "library_name": "cutlass_gemm",
            "family": "gemm_universal",
            "build_mode": "cmake_op_dtype_static_archives",
            "modules": [
                {
                    **module,
                    "archive_sha256": file_sha256(lib_dir / module["archive"]),
                }
                for module in modules
            ],
            "source": "kernels/cuda/src/cutlass_gemm.cu",
            "source_sha256": _cutlass_gemm_source_sha256(repo_root),
            "compile": {
                "system": "cmake",
                "targets": targets,
                "build_dir": str(build_dir),
                "gemm_chunk_size": CUTLASS_GEMM_CMAKE_CHUNK_SIZE,
            },
            "cache_key": "cmake-full",
        },
    )
    return archives


def _required_cutlass_gemm_modules(kernel_manifest: Mapping[str, Any]) -> tuple[dict[str, str], ...]:
    modules = {}
    for item in kernel_manifest.get("required_kernels", []):
        if item.get("kernel_library") != "cutlass_gemm":
            continue
        op_name = str(item["op"])
        dtype = str(item.get("dtype") or item.get("candidate_set", {}).get("dtype"))
        archive = str(item.get("support_archive") or cutlass_gemm_static_library_name(op_name, dtype))
        modules[archive] = {
            "op": op_name,
            "dtype": dtype,
            "archive": archive,
            "target": cutlass_gemm_cmake_target(op_name, dtype),
        }
    return tuple(modules[key] for key in sorted(modules))


def _ensure_cmake_cutlass_bmm_archives(arch: str, kernel_manifest: Mapping[str, Any]) -> tuple[Path, ...]:
    require_cuda_library("cutlass")
    repo_root = _repo_root()
    cache_root = Path(os.environ.get("DINOML_CACHE_DIR", Path.home() / ".cache" / "dinoml_v2"))
    arch_num = _cmake_arch(arch)
    support_root = cache_root / "support" / f"cuda-{arch_num}" / "cutlass-bmm" / "cmake-full"
    build_dir = support_root / "build"
    lib_dir = support_root / "lib"
    modules = _required_cutlass_bmm_modules(kernel_manifest)
    archives = tuple(lib_dir / module["archive"] for module in modules)
    lib_dir.mkdir(parents=True, exist_ok=True)
    _prepare_cmake_build_dir(build_dir)
    if any(not archive.exists() for archive in archives) or not build_dir.exists():
        _run_cmake(
            [
                "cmake",
                "-S",
                str(repo_root),
                "-B",
                str(build_dir),
                "-DCMAKE_BUILD_TYPE=Release",
                "-DDINOML_ENABLE_CUDA=ON",
                "-DDINOML_ENABLE_CUTLASS_GEMM=OFF",
                "-DDINOML_ENABLE_CUTLASS_BMM=ON",
                "-DDINOML_ENABLE_CUTLASS_CONV=OFF",
                "-DDINOML_ENABLE_LIBGGUF_CUDA=OFF",
                f"-DCMAKE_CUDA_ARCHITECTURES={arch_num}",
                f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={lib_dir}",
                f"-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY={lib_dir}",
                f"-DCMAKE_RUNTIME_OUTPUT_DIRECTORY={lib_dir}",
            ],
            cwd=repo_root,
        )
    targets = [module["target"] for module in modules]
    _run_cmake(
        [
            "cmake",
            "--build",
            str(build_dir),
            "--target",
            *targets,
            "--parallel",
        ],
        cwd=repo_root,
    )
    missing = [str(archive) for archive in archives if not archive.exists()]
    if missing:
        raise RuntimeError(f"Expected CMake-built CUTLASS BMM static archives, but these were not produced: {missing}")
    write_json(
        lib_dir / "cutlass_bmm_manifest.json",
        {
            "schema_version": 1,
            "target": {"name": "cuda", "arch": f"sm_{arch_num}"},
            "provider": "cutlass",
            "library_name": "cutlass_bmm",
            "family": "bmm_strided",
            "build_mode": "cmake_op_dtype_static_archives",
            "modules": [
                {
                    **module,
                    "archive_sha256": file_sha256(lib_dir / module["archive"]),
                }
                for module in modules
            ],
            "source": "kernels/cuda/src/cutlass_bmm.cu",
            "source_sha256": file_sha256(repo_root / "kernels" / "cuda" / "src" / "cutlass_bmm.cu"),
            "compile": {
                "system": "cmake",
                "targets": targets,
                "build_dir": str(build_dir),
            },
            "cache_key": "cmake-full",
        },
    )
    return archives


def _required_cutlass_bmm_modules(kernel_manifest: Mapping[str, Any]) -> tuple[dict[str, str], ...]:
    modules = {}
    for item in kernel_manifest.get("required_kernels", []):
        if item.get("kernel_library") != "cutlass_bmm":
            continue
        op_name = str(item["op"])
        dtype = str(item.get("dtype") or item.get("candidate_set", {}).get("dtype"))
        archive = cutlass_bmm_static_library_name(op_name, dtype)
        modules[archive] = {
            "op": op_name,
            "dtype": dtype,
            "archive": archive,
            "target": cutlass_bmm_cmake_target(op_name, dtype),
        }
    return tuple(modules[key] for key in sorted(modules))


def _ensure_cmake_cutlass_conv_archives(arch: str, kernel_manifest: Mapping[str, Any]) -> tuple[Path, ...]:
    require_cuda_library("cutlass")
    repo_root = _repo_root()
    cache_root = Path(os.environ.get("DINOML_CACHE_DIR", Path.home() / ".cache" / "dinoml_v2"))
    arch_num = _cmake_arch(arch)
    support_root = cache_root / "support" / f"cuda-{arch_num}" / "cutlass-conv" / "cmake-full"
    build_dir = support_root / "build"
    lib_dir = support_root / "lib"
    modules = _required_cutlass_conv_modules(kernel_manifest)
    archives = tuple(lib_dir / module["archive"] for module in modules)
    lib_dir.mkdir(parents=True, exist_ok=True)
    _prepare_cmake_build_dir(build_dir)
    if any(not archive.exists() for archive in archives) or not build_dir.exists():
        _run_cmake(
            [
                "cmake",
                "-S",
                str(repo_root),
                "-B",
                str(build_dir),
                "-DCMAKE_BUILD_TYPE=Release",
                "-DDINOML_ENABLE_CUDA=ON",
                "-DDINOML_ENABLE_CUTLASS_GEMM=OFF",
                "-DDINOML_ENABLE_CUTLASS_BMM=OFF",
                "-DDINOML_ENABLE_CUTLASS_CONV=ON",
                "-DDINOML_ENABLE_LIBGGUF_CUDA=OFF",
                f"-DCMAKE_CUDA_ARCHITECTURES={arch_num}",
                f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={lib_dir}",
                f"-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY={lib_dir}",
                f"-DCMAKE_RUNTIME_OUTPUT_DIRECTORY={lib_dir}",
            ],
            cwd=repo_root,
        )
    targets = [module["target"] for module in modules]
    _run_cmake(
        [
            "cmake",
            "--build",
            str(build_dir),
            "--target",
            *targets,
            "--parallel",
        ],
        cwd=repo_root,
    )
    missing = [str(archive) for archive in archives if not archive.exists()]
    if missing:
        raise RuntimeError(f"Expected CMake-built CUTLASS Conv static archives, but these were not produced: {missing}")
    write_json(
        lib_dir / "cutlass_conv_manifest.json",
        {
            "schema_version": 3,
            "target": {"name": "cuda", "arch": f"sm_{arch_num}"},
            "provider": "cutlass",
            "library_name": "cutlass_conv",
            "family": "conv2d_fprop",
            "build_mode": "cmake_op_dtype_static_archives",
            "modules": [
                {
                    **module,
                    "archive_sha256": file_sha256(lib_dir / module["archive"]),
                }
                for module in modules
            ],
            "source": "kernels/cuda/src/cutlass_conv.cu",
            "source_sha256": file_sha256(repo_root / "kernels" / "cuda" / "src" / "cutlass_conv.cu"),
            "compile": {
                "system": "cmake",
                "targets": targets,
                "build_dir": str(build_dir),
            },
            "cache_key": "cmake-full",
        },
    )
    return archives


def _required_cutlass_conv_modules(kernel_manifest: Mapping[str, Any]) -> tuple[dict[str, str], ...]:
    modules = {}
    for item in kernel_manifest.get("required_kernels", []):
        if item.get("kernel_library") != "cutlass_conv":
            continue
        op_name = str(item["op"])
        dtype = str(item.get("dtype") or item.get("candidate_set", {}).get("dtype"))
        archive = cutlass_conv_static_library_name(op_name, dtype)
        modules[archive] = {
            "op": op_name,
            "dtype": dtype,
            "archive": archive,
            "target": cutlass_conv_cmake_target(op_name, dtype),
        }
    return tuple(modules[key] for key in sorted(modules))


def _cutlass_gemm_source_sha256(repo_root: Path) -> str:
    source_paths = [
        repo_root / "kernels" / "cuda" / "src" / "cutlass_common.cuh",
        repo_root / "kernels" / "cuda" / "src" / "cutlass_gemm.cu",
        repo_root / "tools" / "generate_cutlass_gemm_unit.py",
        repo_root / "src" / "dinoml" / "kernels" / "families" / "gemm.py",
        repo_root / "src" / "dinoml" / "kernels" / "providers" / "cutlass" / "gemm.py",
    ]
    digest = hashlib.sha256()
    for path in source_paths:
        digest.update(str(path.relative_to(repo_root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _requires_gguf_cuda_native_library(kernel_manifest: Mapping[str, Any] | None) -> bool:
    if kernel_manifest is None:
        return False
    for item in kernel_manifest.get("required_kernels", []):
        if not isinstance(item, Mapping):
            continue
        plan = item.get("gguf_runtime_dequant")
        if not isinstance(plan, Mapping):
            continue
        if str(plan.get("status")) == "lowered_runtime_dequant_scratch":
            return True
    return False


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


def _prepare_cmake_build_dir(build_dir: Path) -> None:
    if shutil.which("ninja") is None:
        return
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


def _cmake_library_kind(path: Path) -> str:
    return "STATIC" if path.suffix == ".a" else "SHARED"


def _ensure_cmake_libgguf_cuda_native_archive(
    arch: str,
    *,
    cache_root: Path,
    cache_key: str,
    repo_root: Path,
) -> tuple[Path | None, Path | None]:
    source_root = repo_root / "third_party" / "libgguf"
    if not (source_root / "src" / "libgguf" / "libgguf_cuda" / "csrc" / "libgguf_cuda_native.cu").exists():
        raise RuntimeError(
            "GGUF runtime dequant requires vendored libgguf CUDA sources under third_party/libgguf"
        )
    source_provenance = libgguf_source_provenance(source_root)
    source_key = libgguf_provenance_key(source_provenance)
    support_root = (
        cache_root
        / "support"
        / f"cuda-{_cmake_arch(arch)}"
        / "libgguf-cuda-native"
        / cache_key
        / source_key[:16]
    )
    build_dir = support_root / "build"
    lib_dir = support_root / "lib"
    manifest_path = lib_dir / "libgguf_cuda_native_manifest.json"
    lib_dir.mkdir(parents=True, exist_ok=True)
    archive = lib_dir / "libgguf_cuda_native.a"
    if archive.exists() and _libgguf_native_cache_manifest_valid(
        manifest_path,
        library=archive,
        source_provenance=source_provenance,
        source_key=source_key,
    ):
        return archive, manifest_path
    _prepare_cmake_build_dir(build_dir)
    if not archive.exists() or not build_dir.exists():
        _run_cmake(
            [
                "cmake",
                "-S",
                str(repo_root),
                "-B",
                str(build_dir),
                "-DCMAKE_BUILD_TYPE=Release",
                "-DDINOML_ENABLE_CUDA=ON",
                "-DDINOML_ENABLE_CUTLASS_GEMM=OFF",
                "-DDINOML_ENABLE_CUTLASS_BMM=OFF",
                "-DDINOML_ENABLE_CUTLASS_CONV=OFF",
                "-DDINOML_ENABLE_LIBGGUF_CUDA=ON",
                f"-DCMAKE_CUDA_ARCHITECTURES={_cmake_arch(arch)}",
                f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={lib_dir}",
                f"-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY={lib_dir}",
                f"-DCMAKE_RUNTIME_OUTPUT_DIRECTORY={lib_dir}",
            ],
            cwd=repo_root,
        )
    _run_cmake(
        [
            "cmake",
            "--build",
            str(build_dir),
            "--target",
            "libgguf_cuda_native",
            "--parallel",
        ],
        cwd=repo_root,
    )
    if not archive.exists():
        raise RuntimeError(f"Expected libgguf_cuda_native under {lib_dir}, but it was not produced")
    write_json(
        manifest_path,
        {
            "schema_version": 1,
            "name": "gguf_cuda_native",
            "link_mode": "direct",
            "build_mode": "cmake_static_archive",
            "library": archive.name,
            "library_path": str(archive),
            "library_kind": "static",
            "library_sha256": file_sha256(archive),
            "source_provenance_key": source_key,
            "source_provenance": source_provenance,
            "compile": {
                "system": "cmake",
                "targets": ["libgguf_cuda_native"],
                "build_dir": str(build_dir),
            },
        },
    )
    return archive, manifest_path


def _libgguf_native_cache_manifest_valid(
    manifest_path: Path,
    *,
    library: Path,
    source_provenance: dict[str, object],
    source_key: str,
) -> bool:
    if not manifest_path.exists():
        return False
    try:
        manifest = read_json(manifest_path)
    except Exception:
        return False
    if str(manifest.get("source_provenance_key", "")) != source_key:
        return False
    if manifest.get("source_provenance") != source_provenance:
        return False
    if str(manifest.get("library", "")) != library.name:
        return False
    if str(manifest.get("library_sha256", "")) != file_sha256(library):
        return False
    return True


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _cmake_arch(arch: str) -> str:
    match = re.fullmatch(r"sm_(\d+)", arch)
    if match:
        return match.group(1)
    if re.fullmatch(r"\d+", arch):
        return arch
    raise ValueError(f"Expected CUDA arch like 'sm_86' or '86', got {arch!r}")
