from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from dinoml.backends.cutlass import (
    ensure_cutlass_bmm_support_lib,
    ensure_cutlass_conv_support_scaffold,
    ensure_cutlass_gemm_support_lib,
)
from dinoml.ir import read_json, write_json
from dinoml.kernels.providers.cutlass.bmm import cutlass_bmm_used_candidate_plan
from dinoml.kernels.providers.cutlass.conv import cutlass_conv_used_candidate_plan
from dinoml.kernels.providers.cutlass.gemm import cutlass_gemm_used_candidate_plan
from dinoml.kernels.manifest import build_support_manifest
from dinoml.libgguf_cuda import (
    file_sha256,
    libgguf_provenance_key,
    libgguf_source_provenance,
    libgguf_submodule_source_root,
    resolve_libgguf_cuda_direct_link_library,
)
from dinoml.lowering.cuda import render_cuda_module, render_template
from dinoml.lowering.ops import collect_generated_sources


@dataclass(frozen=True)
class SupportLibs:
    runtime_lib: Path
    cuda_runtime_lib: Path
    kernels_lib: Path
    cutlass_gemm_lib: Path | None
    cutlass_bmm_lib: Path | None
    cutlass_conv_lib: Path | None
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
    cutlass_gemm_lib = None if support_libs.cutlass_gemm_lib is None else artifact_lib_dir / support_libs.cutlass_gemm_lib.name
    cutlass_bmm_lib = None if support_libs.cutlass_bmm_lib is None else artifact_lib_dir / support_libs.cutlass_bmm_lib.name
    cutlass_conv_lib = None if support_libs.cutlass_conv_lib is None else artifact_lib_dir / support_libs.cutlass_conv_lib.name
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
    if support_libs.cutlass_gemm_lib is not None and cutlass_gemm_lib is not None:
        shutil.copy2(support_libs.cutlass_gemm_lib, cutlass_gemm_lib)
    if support_libs.cutlass_bmm_lib is not None and cutlass_bmm_lib is not None:
        shutil.copy2(support_libs.cutlass_bmm_lib, cutlass_bmm_lib)
    if support_libs.cutlass_conv_lib is not None and cutlass_conv_lib is not None:
        shutil.copy2(support_libs.cutlass_conv_lib, cutlass_conv_lib)
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
            "cuda_module_cmake.txt.j2",
            {
                "runtime_lib": str(runtime_lib),
                "cuda_runtime_lib": str(cuda_runtime_lib),
                "kernels_lib": str(kernels_lib),
                "cutlass_gemm_lib": "" if cutlass_gemm_lib is None else str(cutlass_gemm_lib),
                "cutlass_bmm_lib": "" if cutlass_bmm_lib is None else str(cutlass_bmm_lib),
                "cutlass_conv_lib": "" if cutlass_conv_lib is None else str(cutlass_conv_lib),
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
    cutlass_gemm_lib = None
    cutlass_bmm_lib = None
    cutlass_conv_lib = None
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
            f"-DCMAKE_CUDA_ARCHITECTURES={_cmake_arch(arch)}",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={lib_dir}",
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
        cutlass_support = ensure_cutlass_gemm_support_lib(
            arch,
            cache_key=kernel_manifest.get("support_cache_key", kernel_manifest["cache_key"])[:16],
            used_candidate_plan=cutlass_gemm_used_candidate_plan(kernel_manifest),
        )
        cutlass_gemm_lib = cutlass_support.library
    if _requires_kernel_library(kernel_manifest, "cutlass_bmm"):
        cutlass_support = ensure_cutlass_bmm_support_lib(
            arch,
            cache_key=kernel_manifest.get("support_cache_key", kernel_manifest["cache_key"])[:16],
            used_candidate_plan=cutlass_bmm_used_candidate_plan(kernel_manifest),
        )
        cutlass_bmm_lib = cutlass_support.library
    if _requires_kernel_library(kernel_manifest, "cutlass_conv"):
        cutlass_support = ensure_cutlass_conv_support_scaffold(
            arch,
            cache_key=kernel_manifest.get("support_cache_key", kernel_manifest["cache_key"])[:16],
            used_candidate_plan=cutlass_conv_used_candidate_plan(kernel_manifest),
        )
        cutlass_conv_lib = cutlass_support.manifest.parent / "libdinoml_cutlass_conv.so"
        if not cutlass_conv_lib.exists():
            raise NotImplementedError(
                "CUTLASS Conv support scaffold did not produce a compiled support library; "
                "nvcc is required before generated CUDA module build can link the guarded Conv wrapper"
            )
    if _requires_gguf_cuda_native_library(kernel_manifest):
        gguf_cuda_native_lib, gguf_cuda_native_manifest = _ensure_libgguf_cuda_native_library(
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
    if cutlass_gemm_lib is not None:
        libraries["cutlass_gemm"] = cutlass_gemm_lib.name
    if cutlass_bmm_lib is not None:
        libraries["cutlass_bmm"] = cutlass_bmm_lib.name
    if cutlass_conv_lib is not None:
        libraries["cutlass_conv"] = cutlass_conv_lib.name
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
        cutlass_gemm_lib=cutlass_gemm_lib,
        cutlass_bmm_lib=cutlass_bmm_lib,
        cutlass_conv_lib=cutlass_conv_lib,
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
    proc = subprocess.run(cmd, cwd=str(cwd), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(
            "CMake command failed\n"
            f"Command: {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )


def _cmake_library_kind(path: Path) -> str:
    return "STATIC" if path.suffix == ".a" else "SHARED"


def _first_existing(paths: tuple[Path, ...]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _ensure_libgguf_cuda_native_library(
    arch: str,
    *,
    cache_root: Path,
    cache_key: str,
    repo_root: Path,
) -> tuple[Path | None, Path | None]:
    explicit_library = resolve_libgguf_cuda_direct_link_library()
    if explicit_library is not None:
        return explicit_library, None
    source_root = libgguf_submodule_source_root(repo_root)
    if source_root is None:
        return None, None
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
    cached = _first_existing(
        (
            lib_dir / "libgguf_cuda_native.a",
            lib_dir / "gguf_cuda_native.so",
            lib_dir / "libgguf_cuda_native.so",
        )
    )
    if cached is not None and _libgguf_native_cache_manifest_valid(
        manifest_path,
        library=cached,
        source_provenance=source_provenance,
        source_key=source_key,
    ):
        return cached, manifest_path
    _run_cmake(
        [
            "cmake",
            "-S",
            str(source_root),
            "-B",
            str(build_dir),
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_CUDA_ARCHITECTURES={_cmake_arch(arch)}",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={lib_dir}",
            f"-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY={lib_dir}",
        ],
        cwd=source_root,
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
        cwd=source_root,
    )
    built = _first_existing(
        (
            lib_dir / "libgguf_cuda_native.a",
            lib_dir / "gguf_cuda_native.so",
            lib_dir / "libgguf_cuda_native.so",
        )
    )
    if built is None:
        return None, None
    write_json(
        manifest_path,
        {
            "schema_version": 1,
            "name": "gguf_cuda_native",
            "link_mode": "direct",
            "library": built.name,
            "library_path": str(built),
            "library_kind": "static" if built.suffix == ".a" else "shared",
            "library_sha256": file_sha256(built),
            "source_provenance_key": source_key,
            "source_provenance": source_provenance,
        },
    )
    return built, manifest_path


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
