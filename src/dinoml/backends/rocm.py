from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from dinoml.backends.registry import _shared_library_name
from dinoml.ir import write_json
from dinoml.kernels.manifest import build_support_manifest
from dinoml.kernels.providers.ck.gemm import ck_gemm_cmake_target, ck_gemm_static_library_name
from dinoml.kernels.providers.ck.bmm import ck_bmm_cmake_target, ck_bmm_static_library_name
from dinoml.kernels.providers.ck.conv import ck_conv_cmake_target, ck_conv_static_library_name
from dinoml.kernels.providers.ck.flash_attention import (
    FLASH_ATTN_CK_LIBRARY,
    flash_attn_ck_cmake_target,
    flash_attn_ck_static_library_name,
)
from dinoml.libgguf_cuda import file_sha256
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
    ck_gemm_archives: tuple[Path, ...]
    ck_bmm_archives: tuple[Path, ...]
    ck_conv_archives: tuple[Path, ...]
    flash_attn_ck_archives: tuple[Path, ...]
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
    ck_gemm_archives = tuple(artifact_lib_dir / archive.name for archive in support_libs.ck_gemm_archives)
    ck_bmm_archives = tuple(artifact_lib_dir / archive.name for archive in support_libs.ck_bmm_archives)
    ck_conv_archives = tuple(artifact_lib_dir / archive.name for archive in support_libs.ck_conv_archives)
    flash_attn_ck_archives = tuple(artifact_lib_dir / archive.name for archive in support_libs.flash_attn_ck_archives)
    shutil.copy2(support_libs.runtime_lib, runtime_lib)
    shutil.copy2(support_libs.rocm_runtime_lib, rocm_runtime_lib)
    shutil.copy2(support_libs.kernels_lib, kernels_lib)
    for source_archive, artifact_archive in zip(support_libs.ck_gemm_archives, ck_gemm_archives):
        shutil.copy2(source_archive, artifact_archive)
    for source_archive, artifact_archive in zip(support_libs.ck_bmm_archives, ck_bmm_archives):
        shutil.copy2(source_archive, artifact_archive)
    for source_archive, artifact_archive in zip(support_libs.ck_conv_archives, ck_conv_archives):
        shutil.copy2(source_archive, artifact_archive)
    for source_archive, artifact_archive in zip(support_libs.flash_attn_ck_archives, flash_attn_ck_archives):
        shutil.copy2(source_archive, artifact_archive)

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
                "ck_gemm_archives": [_cmake_path(path) for path in ck_gemm_archives],
                "ck_bmm_archives": [_cmake_path(path) for path in ck_bmm_archives],
                "ck_conv_archives": [_cmake_path(path) for path in ck_conv_archives],
                "flash_attn_ck_archives": [_cmake_path(path) for path in flash_attn_ck_archives],
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
    support_root = cache_root / "support" / _rocm_support_cache_dir_name(arch) / manifest_key
    build_dir = support_root / "build"
    lib_dir = support_root / "lib"
    runtime_lib = lib_dir / _shared_library_name("dinoml_runtime")
    rocm_runtime_lib = lib_dir / _shared_library_name("dinoml_rocm_runtime")
    kernels_lib = lib_dir / _shared_library_name("dinoml_rocm_kernels")
    ck_gemm_archives: tuple[Path, ...] = ()
    ck_bmm_archives: tuple[Path, ...] = ()
    ck_conv_archives: tuple[Path, ...] = ()
    flash_attn_ck_archives: tuple[Path, ...] = ()
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
        "-DDINOML_ENABLE_CK_GEMM=OFF",
        "-DDINOML_ENABLE_CK_BMM=OFF",
        "-DDINOML_ENABLE_CK_CONV=OFF",
        "-DDINOML_ENABLE_FLASH_ATTN_CK=OFF",
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
    if _requires_kernel_library(kernel_manifest, "ck_gemm"):
        ck_gemm_archives = _ensure_cmake_ck_gemm_archives(arch, kernel_manifest)
    if _requires_kernel_library(kernel_manifest, "ck_bmm"):
        ck_bmm_archives = _ensure_cmake_ck_bmm_archives(arch, kernel_manifest)
    if _requires_kernel_library(kernel_manifest, "ck_conv"):
        ck_conv_archives = _ensure_cmake_ck_conv_archives(arch, kernel_manifest)
    if _requires_kernel_library(kernel_manifest, FLASH_ATTN_CK_LIBRARY):
        flash_attn_ck_archives = _ensure_cmake_flash_attn_ck_archives(arch, kernel_manifest)
    support_target = (
        dict(kernel_manifest.get("target", {"name": "rocm", "arch": _cmake_arch(arch)}))
        if kernel_manifest is not None
        else {"name": "rocm", "arch": _cmake_arch(arch)}
    )
    libraries: dict[str, Any] = {
        "runtime": runtime_lib.name,
        "rocm_runtime": rocm_runtime_lib.name,
        "kernels": kernels_lib.name,
    }
    if ck_gemm_archives:
        libraries["ck_gemm_static"] = [archive.name for archive in ck_gemm_archives]
    if ck_bmm_archives:
        libraries["ck_bmm_static"] = [archive.name for archive in ck_bmm_archives]
    if ck_conv_archives:
        libraries["ck_conv_static"] = [archive.name for archive in ck_conv_archives]
    if flash_attn_ck_archives:
        libraries["flash_attn_ck_static"] = [archive.name for archive in flash_attn_ck_archives]
    write_json(
        lib_dir / "support_manifest.json",
        build_support_manifest(
            target=support_target,
            libraries=libraries,
            required_kernel_cache_key=None if kernel_manifest is None else str(kernel_manifest.get("support_cache_key", kernel_manifest["cache_key"])),
        ),
    )
    return RocmSupportLibs(
        runtime_lib=runtime_lib,
        rocm_runtime_lib=rocm_runtime_lib,
        kernels_lib=kernels_lib,
        ck_gemm_archives=ck_gemm_archives,
        ck_bmm_archives=ck_bmm_archives,
        ck_conv_archives=ck_conv_archives,
        flash_attn_ck_archives=flash_attn_ck_archives,
        runtime_include=repo_root / "runtime" / "include",
        common_include=repo_root / "kernels" / "common" / "include",
        kernels_include=repo_root / "kernels" / "rocm" / "include",
    )


def _requires_kernel_library(kernel_manifest: Mapping[str, Any] | None, library: str) -> bool:
    if kernel_manifest is None:
        return False
    return any(item.get("kernel_library") == library for item in kernel_manifest.get("required_kernels", []))


def _ensure_cmake_ck_gemm_archives(arch: str, kernel_manifest: Mapping[str, Any]) -> tuple[Path, ...]:
    repo_root = _repo_root()
    cache_root = Path(os.environ.get("DINOML_CACHE_DIR", Path.home() / ".cache" / "dinoml_v2"))
    arch_name = _cmake_arch(arch)
    support_root = cache_root / "support" / _rocm_support_cache_dir_name(arch_name) / "ck-gemm" / "cmake-full"
    build_dir = support_root / "build"
    lib_dir = support_root / "lib"
    modules = _required_ck_gemm_modules(kernel_manifest)
    archives = tuple(lib_dir / module["archive"] for module in modules)
    ops = _cmake_cache_list(module["op"] for module in modules)
    dtypes = _cmake_cache_list(module["dtype"] for module in modules)
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
                "-DDINOML_ENABLE_CUDA=OFF",
                "-DDINOML_ENABLE_ROCM=ON",
                "-DDINOML_ENABLE_CK_GEMM=ON",
                f"-DDINOML_CK_GEMM_OPS={ops}",
                f"-DDINOML_CK_GEMM_DTYPES={dtypes}",
                f"-DCMAKE_HIP_ARCHITECTURES={arch_name}",
                f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={lib_dir}",
                f"-DCMAKE_RUNTIME_OUTPUT_DIRECTORY={lib_dir}",
                f"-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY={lib_dir}",
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
        raise RuntimeError(f"Expected CMake-built CK GEMM static archives, but these were not produced: {missing}")
    write_json(
        lib_dir / "ck_gemm_manifest.json",
        {
            "schema_version": 1,
            "target": {"name": "rocm", "arch": arch_name},
            "provider": "ck",
            "library_name": "ck_gemm",
            "family": "gemm_universal",
            "build_mode": "cmake_op_dtype_static_archives",
            "modules": [
                {
                    **module,
                    "archive_sha256": file_sha256(lib_dir / module["archive"]),
                }
                for module in modules
            ],
            "source": "kernels/rocm/src/ck_gemm.hip",
            "source_sha256": _ck_gemm_source_sha256(repo_root),
            "compile": {
                "system": "cmake",
                "targets": targets,
                "build_dir": str(build_dir),
            },
            "cache_key": "cmake-full",
        },
    )
    return archives


def _ensure_cmake_ck_bmm_archives(arch: str, kernel_manifest: Mapping[str, Any]) -> tuple[Path, ...]:
    repo_root = _repo_root()
    cache_root = Path(os.environ.get("DINOML_CACHE_DIR", Path.home() / ".cache" / "dinoml_v2"))
    arch_name = _cmake_arch(arch)
    support_root = cache_root / "support" / _rocm_support_cache_dir_name(arch_name) / "ck-bmm" / "cmake-full"
    build_dir = support_root / "build"
    lib_dir = support_root / "lib"
    modules = _required_ck_bmm_modules(kernel_manifest)
    archives = tuple(lib_dir / module["archive"] for module in modules)
    ops = _cmake_cache_list(module["op"] for module in modules)
    dtypes = _cmake_cache_list(module["dtype"] for module in modules)
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
                "-DDINOML_ENABLE_CUDA=OFF",
                "-DDINOML_ENABLE_ROCM=ON",
                "-DDINOML_ENABLE_CK_GEMM=OFF",
                "-DDINOML_ENABLE_CK_BMM=ON",
                f"-DDINOML_CK_BMM_OPS={ops}",
                f"-DDINOML_CK_BMM_DTYPES={dtypes}",
                f"-DCMAKE_HIP_ARCHITECTURES={arch_name}",
                f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={lib_dir}",
                f"-DCMAKE_RUNTIME_OUTPUT_DIRECTORY={lib_dir}",
                f"-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY={lib_dir}",
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
        raise RuntimeError(f"Expected CMake-built CK BMM static archives, but these were not produced: {missing}")
    write_json(
        lib_dir / "ck_bmm_manifest.json",
        {
            "schema_version": 1,
            "target": {"name": "rocm", "arch": arch_name},
            "provider": "ck",
            "library_name": "ck_bmm",
            "family": "bmm_strided",
            "build_mode": "cmake_op_dtype_static_archives",
            "modules": [
                {
                    **module,
                    "archive_sha256": file_sha256(lib_dir / module["archive"]),
                }
                for module in modules
            ],
            "source": "kernels/rocm/src/ck_bmm.hip",
            "source_sha256": _ck_bmm_source_sha256(repo_root),
            "compile": {
                "system": "cmake",
                "targets": targets,
                "build_dir": str(build_dir),
            },
            "cache_key": "cmake-full",
        },
    )
    return archives


def _ensure_cmake_ck_conv_archives(arch: str, kernel_manifest: Mapping[str, Any]) -> tuple[Path, ...]:
    repo_root = _repo_root()
    cache_root = Path(os.environ.get("DINOML_CACHE_DIR", Path.home() / ".cache" / "dinoml_v2"))
    arch_name = _cmake_arch(arch)
    support_root = cache_root / "support" / _rocm_support_cache_dir_name(arch_name) / "ck-conv" / "cmake-full"
    build_dir = support_root / "build"
    lib_dir = support_root / "lib"
    modules = _required_ck_conv_modules(kernel_manifest)
    archives = tuple(lib_dir / module["archive"] for module in modules)
    ops = _cmake_cache_list(module["op"] for module in modules)
    dtypes = _cmake_cache_list(module["dtype"] for module in modules)
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
                "-DDINOML_ENABLE_CUDA=OFF",
                "-DDINOML_ENABLE_ROCM=ON",
                "-DDINOML_ENABLE_CK_GEMM=OFF",
                "-DDINOML_ENABLE_CK_BMM=OFF",
                "-DDINOML_ENABLE_CK_CONV=ON",
                f"-DDINOML_CK_CONV_OPS={ops}",
                f"-DDINOML_CK_CONV_DTYPES={dtypes}",
                f"-DCMAKE_HIP_ARCHITECTURES={arch_name}",
                f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={lib_dir}",
                f"-DCMAKE_RUNTIME_OUTPUT_DIRECTORY={lib_dir}",
                f"-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY={lib_dir}",
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
        raise RuntimeError(f"Expected CMake-built CK Conv static archives, but these were not produced: {missing}")
    write_json(
        lib_dir / "ck_conv_manifest.json",
        {
            "schema_version": 1,
            "target": {"name": "rocm", "arch": arch_name},
            "provider": "ck",
            "library_name": "ck_conv",
            "family": "conv2d_fprop",
            "build_mode": "cmake_op_dtype_static_archives",
            "modules": [
                {
                    **module,
                    "archive_sha256": file_sha256(lib_dir / module["archive"]),
                }
                for module in modules
            ],
            "source": "kernels/rocm/src/ck_conv.hip",
            "source_sha256": _ck_conv_source_sha256(repo_root),
            "compile": {
                "system": "cmake",
                "targets": targets,
                "build_dir": str(build_dir),
            },
            "cache_key": "cmake-full",
        },
    )
    return archives


def _ensure_cmake_flash_attn_ck_archives(arch: str, kernel_manifest: Mapping[str, Any]) -> tuple[Path, ...]:
    del kernel_manifest
    repo_root = _repo_root()
    cache_root = Path(os.environ.get("DINOML_CACHE_DIR", Path.home() / ".cache" / "dinoml_v2"))
    arch_name = _cmake_arch(arch)
    support_root = cache_root / "support" / _rocm_support_cache_dir_name(arch_name) / "flash-attn-ck" / "cmake-full"
    build_dir = support_root / "build"
    lib_dir = support_root / "lib"
    archive = lib_dir / flash_attn_ck_static_library_name("float16")
    lib_dir.mkdir(parents=True, exist_ok=True)
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
                "-DDINOML_ENABLE_CUDA=OFF",
                "-DDINOML_ENABLE_ROCM=ON",
                "-DDINOML_ENABLE_CK_GEMM=OFF",
                "-DDINOML_ENABLE_CK_BMM=OFF",
                "-DDINOML_ENABLE_CK_CONV=OFF",
                "-DDINOML_ENABLE_FLASH_ATTN_CK=ON",
                f"-DCMAKE_HIP_ARCHITECTURES={arch_name}",
                f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={lib_dir}",
                f"-DCMAKE_RUNTIME_OUTPUT_DIRECTORY={lib_dir}",
                f"-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY={lib_dir}",
            ],
            cwd=repo_root,
        )
    target = flash_attn_ck_cmake_target()
    _run_cmake(
        [
            "cmake",
            "--build",
            str(build_dir),
            "--target",
            target,
            "--parallel",
        ],
        cwd=repo_root,
    )
    if not archive.exists():
        raise RuntimeError(f"Expected CMake-built CK FlashAttention static archive, but it was not produced: {archive}")
    write_json(
        lib_dir / "flash_attn_ck_manifest.json",
        {
            "schema_version": 1,
            "target": {"name": "rocm", "arch": arch_name},
            "provider": "ck",
            "library_name": FLASH_ATTN_CK_LIBRARY,
            "family": "flash_attention_fwd",
            "build_mode": "cmake_static_archive",
            "modules": [
                {
                    "op": "flash_attention",
                    "dtype": "float16",
                    "archive": archive.name,
                    "target": target,
                    "archive_sha256": file_sha256(archive),
                }
            ],
            "source_sha256": _flash_attn_ck_source_sha256(repo_root),
            "compile": {
                "system": "cmake",
                "targets": [target],
                "build_dir": str(build_dir),
            },
            "cache_key": "cmake-full",
        },
    )
    return (archive,)


def _required_ck_gemm_modules(kernel_manifest: Mapping[str, Any]) -> tuple[dict[str, str], ...]:
    modules = {}
    for item in kernel_manifest.get("required_kernels", []):
        if item.get("kernel_library") != "ck_gemm":
            continue
        op_name = str(item["op"])
        dtype = str(item.get("dtype") or item.get("candidate_set", {}).get("dtype"))
        archive = str(item.get("support_archive") or ck_gemm_static_library_name(op_name, dtype))
        modules[archive] = {
            "op": op_name,
            "dtype": dtype,
            "archive": archive,
            "target": ck_gemm_cmake_target(op_name, dtype),
        }
    return tuple(modules[key] for key in sorted(modules))


def _required_ck_bmm_modules(kernel_manifest: Mapping[str, Any]) -> tuple[dict[str, str], ...]:
    modules = {}
    for item in kernel_manifest.get("required_kernels", []):
        if item.get("kernel_library") != "ck_bmm":
            continue
        op_name = str(item["op"])
        dtype = str(item.get("dtype") or item.get("candidate_set", {}).get("dtype"))
        archive = str(item.get("support_archive") or ck_bmm_static_library_name(op_name, dtype))
        modules[archive] = {
            "op": op_name,
            "dtype": dtype,
            "archive": archive,
            "target": ck_bmm_cmake_target(op_name, dtype),
        }
    return tuple(modules[key] for key in sorted(modules))


def _required_ck_conv_modules(kernel_manifest: Mapping[str, Any]) -> tuple[dict[str, str], ...]:
    modules = {}
    for item in kernel_manifest.get("required_kernels", []):
        if item.get("kernel_library") != "ck_conv":
            continue
        op_name = str(item["op"])
        dtype = str(item.get("dtype") or item.get("candidate_set", {}).get("dtype"))
        archive = str(item.get("support_archive") or ck_conv_static_library_name(op_name, dtype))
        modules[archive] = {
            "op": op_name,
            "dtype": dtype,
            "archive": archive,
            "target": ck_conv_cmake_target(op_name, dtype),
        }
    return tuple(modules[key] for key in sorted(modules))


def _ck_gemm_source_sha256(repo_root: Path) -> str:
    source_paths = [
        repo_root / "kernels" / "rocm" / "src" / "ck_gemm.hip",
        repo_root / "tools" / "generate_ck_gemm_unit.py",
        repo_root / "src" / "dinoml" / "kernels" / "families" / "gemm.py",
        repo_root / "src" / "dinoml" / "kernels" / "providers" / "ck" / "gemm.py",
    ]
    digest = hashlib.sha256()
    for path in source_paths:
        digest.update(str(path.relative_to(repo_root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _ck_bmm_source_sha256(repo_root: Path) -> str:
    source_paths = [
        repo_root / "kernels" / "rocm" / "src" / "ck_bmm.hip",
        repo_root / "tools" / "generate_ck_bmm_unit.py",
        repo_root / "src" / "dinoml" / "kernels" / "families" / "bmm.py",
        repo_root / "src" / "dinoml" / "kernels" / "providers" / "ck" / "bmm.py",
    ]
    digest = hashlib.sha256()
    for path in source_paths:
        digest.update(str(path.relative_to(repo_root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _ck_conv_source_sha256(repo_root: Path) -> str:
    source_paths = [
        repo_root / "kernels" / "rocm" / "src" / "ck_conv.hip",
        repo_root / "tools" / "generate_ck_conv_unit.py",
        repo_root / "src" / "dinoml" / "kernels" / "providers" / "ck" / "conv.py",
    ]
    digest = hashlib.sha256()
    for path in source_paths:
        digest.update(str(path.relative_to(repo_root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _flash_attn_ck_source_sha256(repo_root: Path) -> str:
    generator_dir = repo_root / "third_party" / "composable_kernel" / "example" / "ck_tile" / "01_fmha"
    source_paths = [
        repo_root / "CMakeLists.txt",
        repo_root / "kernels" / "rocm" / "src" / "flash_attn_ck_wrapper.cpp",
        repo_root / "kernels" / "rocm" / "include" / "dinoml" / "rocm_kernels.h",
        repo_root / "third_party" / "flash_attn_ck" / "flash_attn_dinoml.h",
        repo_root / "third_party" / "flash_attn_ck" / "interface_src" / "flash_attn_dinoml.cpp",
        generator_dir / "bias.hpp",
        generator_dir / "fmha_fwd.hpp",
        generator_dir / "generate.py",
        generator_dir / "mask.hpp",
        generator_dir / "rotary.hpp",
        *sorted((generator_dir / "codegen").rglob("*.py")),
    ]
    digest = hashlib.sha256()
    for path in source_paths:
        digest.update(str(path.relative_to(repo_root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


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
    if arch is None:
        raise ValueError("Expected ROCm arch like 'gfx1201'")
    value = str(arch).strip()
    if not value:
        raise ValueError("Expected ROCm arch like 'gfx1201'")
    if not value.startswith("gfx"):
        raise ValueError(f"Expected ROCm arch like 'gfx1201', got {arch!r}")
    return value


def _rocm_support_cache_dir_name(arch: str) -> str:
    segment = re.sub(r"[^A-Za-z0-9_.-]+", "_", _cmake_arch(arch))
    return f"rocm-{segment}"


def _cmake_cache_list(values) -> str:
    return ";".join(dict.fromkeys(str(value) for value in values))


def _run_cmake(cmd: list[str], *, cwd: Path) -> None:
    cmd = _with_rocm_cmake_cache_args(cmd)
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


def _with_rocm_cmake_cache_args(cmd: list[str]) -> list[str]:
    if len(cmd) < 2 or Path(cmd[0]).stem != "cmake":
        return cmd
    if "--build" in cmd or "-S" not in cmd:
        return cmd
    args = _rocm_cmake_cache_args()
    if not args:
        return cmd
    return [cmd[0], *args, *cmd[1:]]


def _rocm_cmake_cache_args() -> list[str]:
    root = os.environ.get("DINOML_ROCM_ROOT") or _run_rocm_sdk_path("--root") or _rocm_sdk_devel_root()
    if not root:
        return []
    cmake_prefix = os.environ.get("DINOML_ROCM_CMAKE_PREFIX") or _run_rocm_sdk_path("--cmake")
    bin_dir = os.environ.get("DINOML_ROCM_BIN") or _run_rocm_sdk_path("--bin")
    llvm_bin = Path(root) / "lib" / "llvm" / "bin"
    args = [
        f"-DDINOML_ROCM_ROOT={root}",
        f"-DDINOML_ROCM_CMAKE_PREFIX={cmake_prefix or Path(root) / 'lib' / 'cmake'}",
        f"-DDINOML_ROCM_BIN={bin_dir or Path(root) / 'bin'}",
    ]
    if llvm_bin.exists():
        args.append(f"-DDINOML_ROCM_LLVM_BIN={llvm_bin}")
    return args


def _rocm_sdk_devel_root() -> str | None:
    candidates = [
        Path(sys.prefix) / "Lib" / "site-packages" / "_rocm_sdk_devel",
        Path(sys.prefix) / "lib" / "site-packages" / "_rocm_sdk_devel",
    ]
    for candidate in candidates:
        if (candidate / "lib" / "cmake" / "hip-lang" / "hip-lang-config.cmake").exists():
            return str(candidate)
    return None


def _with_default_cmake_generator(cmd: list[str]) -> list[str]:
    if len(cmd) < 2 or Path(cmd[0]).stem != "cmake":
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
        original_path = env.get("PATH", "")
        vs_env = _visual_studio_environment()
        env.update(vs_env)
        if original_path and vs_env.get("PATH"):
            env["PATH"] = os.pathsep.join([vs_env["PATH"], original_path])
        _prepend_paths(env, _rocm_runtime_paths())
    _CMAKE_ENV = env
    return dict(env)


def _rocm_runtime_paths() -> list[str]:
    paths = []
    root = _run_rocm_sdk_path("--root")
    bin_dir = _run_rocm_sdk_path("--bin")
    if bin_dir:
        paths.append(bin_dir)
    elif root:
        root_bin = Path(root) / "bin"
        if root_bin.exists():
            paths.append(str(root_bin))
    if root:
        paths.append(str(Path(root) / "lib" / "llvm" / "bin"))
    if not root and not bin_dir:
        for env_root in _rocm_environment_roots():
            _append_rocm_root_runtime_paths(paths, env_root)
    return _unique_rocm_runtime_paths(paths)


def _rocm_environment_roots() -> list[Path]:
    roots = []
    seen = set()
    for key in ("HIP_PATH", "ROCM_PATH"):
        value = os.environ.get(key)
        if not value:
            continue
        value = value.strip().strip("\"'")
        if not value:
            continue
        root = Path(value)
        normalized = os.path.normcase(os.path.abspath(root))
        if normalized in seen:
            continue
        seen.add(normalized)
        roots.append(root)
    return roots


def _append_rocm_root_runtime_paths(paths: list[str], root: Path) -> None:
    bin_dir = root / "bin"
    llvm_bin = root / "lib" / "llvm" / "bin"
    if bin_dir.exists():
        paths.append(str(bin_dir))
    if llvm_bin.exists():
        paths.append(str(llvm_bin))


def _unique_rocm_runtime_paths(paths: list[str]) -> list[str]:
    seen = set()
    unique = []
    for path in paths:
        normalized = os.path.normcase(os.path.abspath(path))
        if normalized in seen:
            continue
        seen.add(normalized)
        unique.append(path)
    return unique


def _run_rocm_sdk_path(arg: str) -> str | None:
    rocm_sdk = _rocm_sdk_command()
    if rocm_sdk is None:
        return None
    try:
        subprocess.run(
            [*rocm_sdk, "init"],
            text=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        proc = subprocess.run(
            [*rocm_sdk, "path", arg],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return None
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
    try:
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
    except OSError:
        return False
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
