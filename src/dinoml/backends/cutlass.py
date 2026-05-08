from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from dinoml.backends.cuda_libraries import require_cuda_library
from dinoml.ir import write_json
from dinoml.kernels.external import external_kernel_families
from dinoml.kernels.manifest import build_external_kernel_plan


@dataclass(frozen=True)
class CutlassSupportLib:
    library: Path
    include_roots: tuple[Path, ...]
    source: Path
    manifest: Path


def ensure_cutlass_gemm_support_lib(arch: str, *, cache_key: str | None = None) -> CutlassSupportLib:
    cutlass = require_cuda_library("cutlass")
    require_cuda_library("cublaslt")
    cache_root = Path(os.environ.get("DINOML_CACHE_DIR", Path.home() / ".cache" / "dinoml_v2"))
    arch_num = _cmake_arch(arch)
    plan = build_external_kernel_plan({"name": "cuda", "arch": f"sm_{arch_num}"})
    manifest_key = cache_key or plan["cache_key"][:16]
    support_root = cache_root / "support" / f"cuda-{arch_num}" / "cutlass-gemm" / manifest_key
    src_dir = support_root / "src"
    lib_dir = support_root / "lib"
    src_dir.mkdir(parents=True, exist_ok=True)
    lib_dir.mkdir(parents=True, exist_ok=True)
    source = src_dir / "dinoml_cutlass_gemm.cu"
    library = lib_dir / "libdinoml_cutlass_gemm.so"
    manifest = lib_dir / "cutlass_gemm_manifest.json"
    repo_source = _repo_cutlass_gemm_source()
    include_roots = (
        *cutlass.include_roots,
        *(root.parent / "tools" / "util" / "include" for root in cutlass.include_roots if root.name == "include"),
    )
    source_hash = _file_sha256(repo_source)
    if (
        library.exists()
        and manifest.exists()
        and source.exists()
        and _file_sha256(source) == source_hash
        and _cached_manifest_matches(manifest, source_hash, plan["cache_key"])
    ):
        return CutlassSupportLib(
            library=library,
            include_roots=tuple(root for root in include_roots if root.exists()),
            source=source,
            manifest=manifest,
        )
    shutil.copy2(repo_source, source)

    include_args = []
    for root in include_roots:
        if root.exists():
            include_args.append(f"-I{root}")
    _run_nvcc(
        [
            "nvcc",
            "-std=c++17",
            "-O3",
            "--use_fast_math",
            "--expt-relaxed-constexpr",
            "-shared",
            "-Xcompiler=-fPIC",
            f"-arch=sm_{arch_num}",
            *include_args,
            str(source),
            "-o",
            str(library),
        ],
        cwd=support_root,
    )
    write_json(
        manifest,
        {
            "schema_version": 1,
            "target": {"name": "cuda", "arch": f"sm_{arch_num}"},
            "provider": "cutlass",
            "families": [family.to_json() for family in external_kernel_families(provider="cutlass", backend="cuda")],
            "library": library.name,
            "source": source.name,
            "source_sha256": source_hash,
            "external_kernel_plan_cache_key": plan["cache_key"],
            "cache_key": manifest_key,
        },
    )
    return CutlassSupportLib(
        library=library,
        include_roots=tuple(root for root in include_roots if root.exists()),
        source=source,
        manifest=manifest,
    )


def _repo_cutlass_gemm_source() -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    source = repo_root / "kernels" / "cuda" / "src" / "cutlass_gemm.cu"
    if not source.exists():
        raise FileNotFoundError(f"Missing CUTLASS GEMM source: {source}")
    return source


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cached_manifest_matches(path: Path, source_hash: str, plan_cache_key: str) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return payload.get("source_sha256") == source_hash and payload.get("external_kernel_plan_cache_key") == plan_cache_key


def _run_nvcc(cmd: list[str], *, cwd: Path) -> None:
    proc = subprocess.run(cmd, cwd=str(cwd), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(
            "CUTLASS support build failed\n"
            f"Command: {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )


def _cmake_arch(arch: str) -> str:
    match = re.fullmatch(r"sm_(\d+)", arch)
    if match:
        return match.group(1)
    if re.fullmatch(r"\d+", arch):
        return arch
    raise ValueError(f"Expected CUDA arch like 'sm_86' or '86', got {arch!r}")
