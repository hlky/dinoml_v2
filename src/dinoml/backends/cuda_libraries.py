from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class CudaLibrary:
    name: str
    available: bool
    include_roots: tuple[Path, ...] = ()
    library_roots: tuple[Path, ...] = ()
    headers: tuple[str, ...] = ()
    notes: str = ""

    def to_json(self) -> dict[str, object]:
        return {
            "name": self.name,
            "available": self.available,
            "include_roots": [str(path) for path in self.include_roots],
            "library_roots": [str(path) for path in self.library_roots],
            "headers": list(self.headers),
            "notes": self.notes,
        }


def discover_cuda_libraries() -> dict[str, CudaLibrary]:
    cuda_root = _cuda_root()
    cuda_include = cuda_root / "include" if cuda_root is not None else None
    cuda_lib = cuda_root / "lib64" if cuda_root is not None else None
    target_include = _first_existing(
        [
            Path("/usr/local/cuda/include"),
            Path("/usr/local/cuda-12.8/targets/x86_64-linux/include"),
            Path("/usr/local/cuda-12/targets/x86_64-linux/include"),
        ]
    )
    if cuda_include is None and target_include is not None:
        cuda_include = target_include

    cub_header = "cub/cub.cuh"
    cutlass_header = "cutlass/gemm/device/gemm_universal.h"
    cudnn_header = "cudnn.h"
    cublaslt_header = "cublasLt.h"
    cutlass_roots = tuple(_cutlass_include_roots(_cutlass_candidates()))
    cuda_includes = tuple(path for path in (cuda_include,) if path is not None and path.exists())
    cuda_libs = tuple(path for path in (cuda_lib,) if path is not None and path.exists())

    return {
        "cuda": CudaLibrary(
            name="cuda",
            available=bool(cuda_includes),
            include_roots=cuda_includes,
            library_roots=cuda_libs,
            headers=("cuda_runtime.h",),
            notes="CUDA toolkit include/lib roots used by generated support builds.",
        ),
        "cub": CudaLibrary(
            name="cub",
            available=_header_exists(cuda_includes, cub_header),
            include_roots=cuda_includes,
            headers=(cub_header,),
            notes="CUB reductions/scans/sort helpers; usually bundled with CUDA.",
        ),
        "cublaslt": CudaLibrary(
            name="cublaslt",
            available=_header_exists(cuda_includes, cublaslt_header),
            include_roots=cuda_includes,
            library_roots=cuda_libs,
            headers=(cublaslt_header,),
            notes="cuBLASLt fallback/reference path for GEMM benchmarking and validation.",
        ),
        "cudnn": CudaLibrary(
            name="cudnn",
            available=_header_exists(cuda_includes, cudnn_header),
            include_roots=cuda_includes,
            library_roots=cuda_libs,
            headers=(cudnn_header,),
            notes="cuDNN path for future conv/pool/norm library-backed ports.",
        ),
        "cutlass": CudaLibrary(
            name="cutlass",
            available=_header_exists(cutlass_roots, cutlass_header),
            include_roots=cutlass_roots,
            headers=(cutlass_header,),
            notes="Header-only CUTLASS candidate source for GEMM/conv codegen and profilers.",
        ),
    }


def require_cuda_library(name: str) -> CudaLibrary:
    libraries = discover_cuda_libraries()
    try:
        library = libraries[name]
    except KeyError as exc:
        raise ValueError(f"Unknown CUDA library requirement: {name}") from exc
    if not library.available:
        raise RuntimeError(f"Required CUDA library {name!r} is not available")
    return library


def _cuda_root() -> Path | None:
    for env_name in ("CUDA_HOME", "CUDA_PATH"):
        value = os.environ.get(env_name)
        if value:
            path = Path(value)
            if path.exists():
                return path
    for candidate in (Path("/usr/local/cuda"), Path("/usr/local/cuda-12.8"), Path("/usr/local/cuda-12")):
        if candidate.exists():
            return candidate
    return None


def _cutlass_candidates() -> list[Path]:
    candidates: list[Path] = []
    for env_name in ("DINOML_CUTLASS_ROOT", "CUTLASS_ROOT"):
        value = os.environ.get(env_name)
        if value:
            candidates.append(Path(value))
    candidates.extend(
        [
            Path("/workspace/dinoml_v2/third_party/cutlass"),
            Path("/workspace/dinoml_v2/3rdparty/cutlass"),
            Path("/workspace/dinoml/3rdparty/cutlass"),
        ]
    )
    return candidates


def _existing_paths(paths: Iterable[Path]) -> list[Path]:
    return [path for path in paths if path.exists()]


def _cutlass_include_roots(paths: Iterable[Path]) -> list[Path]:
    roots: list[Path] = []
    for path in paths:
        include = path / "include"
        if include.exists():
            roots.append(include)
        elif path.exists():
            roots.append(path)
    return roots


def _first_existing(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _header_exists(include_roots: Iterable[Path], header: str) -> bool:
    return any((root / header).exists() for root in include_roots)
