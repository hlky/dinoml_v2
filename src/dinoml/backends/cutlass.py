from __future__ import annotations

import os
import re
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
    source.write_text(_cutlass_gemm_source(), encoding="utf-8")

    include_args = []
    include_roots = (
        *cutlass.include_roots,
        *(root.parent / "tools" / "util" / "include" for root in cutlass.include_roots if root.name == "include"),
    )
    for root in include_roots:
        if root.exists():
            include_args.append(f"-I{root}")
    _run_nvcc(
        [
            "nvcc",
            "-std=c++17",
            "-O3",
            "--use_fast_math",
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
            "cache_key": manifest_key,
        },
    )
    return CutlassSupportLib(
        library=library,
        include_roots=tuple(root for root in include_roots if root.exists()),
        source=source,
        manifest=manifest,
    )


def _cutlass_gemm_source() -> str:
    return r'''
#include <cuda_runtime.h>

#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/layout/matrix.h>

namespace {

template <typename LayoutB>
int launch_gemm(
    const float* a,
    const float* b,
    float* c,
    int m,
    int n,
    int k,
    int ldb,
    cudaStream_t stream) {
  if (a == nullptr || b == nullptr || c == nullptr) {
    return 1;
  }
  if (m <= 0 || n <= 0 || k <= 0) {
    return 2;
  }
  using Gemm = cutlass::gemm::device::Gemm<
      float,
      cutlass::layout::RowMajor,
      float,
      LayoutB,
      float,
      cutlass::layout::RowMajor>;
  Gemm gemm;
  typename Gemm::Arguments args(
      {m, n, k},
      {a, k},
      {b, ldb},
      {c, n},
      {c, n},
      {1.0f, 0.0f});
  cutlass::Status status = gemm(args, nullptr, stream);
  return status == cutlass::Status::kSuccess ? 0 : 3;
}

template <typename LayoutB>
float profile_gemm(
    const float* a,
    const float* b,
    float* c,
    int m,
    int n,
    int k,
    int ldb,
    int iterations,
    cudaStream_t stream) {
  if (iterations <= 0) {
    iterations = 20;
  }
  cudaEvent_t start;
  cudaEvent_t end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  launch_gemm<LayoutB>(a, b, c, m, n, k, ldb, stream);
  cudaEventRecord(start, stream);
  for (int i = 0; i < iterations; ++i) {
    launch_gemm<LayoutB>(a, b, c, m, n, k, ldb, stream);
  }
  cudaEventRecord(end, stream);
  cudaEventSynchronize(end);
  float ms = 0.0f;
  cudaEventElapsedTime(&ms, start, end);
  cudaEventDestroy(start);
  cudaEventDestroy(end);
  return ms / static_cast<float>(iterations);
}

}  // namespace

extern "C" int dinoml_cutlass_gemm_rrr_f32(
    const float* a,
    const float* b,
    float* c,
    int m,
    int n,
    int k,
    cudaStream_t stream) {
  return launch_gemm<cutlass::layout::RowMajor>(a, b, c, m, n, k, n, stream);
}

extern "C" int dinoml_cutlass_gemm_rcr_f32(
    const float* a,
    const float* b,
    float* c,
    int m,
    int n,
    int k,
    cudaStream_t stream) {
  return launch_gemm<cutlass::layout::ColumnMajor>(a, b, c, m, n, k, k, stream);
}

extern "C" float dinoml_profile_cutlass_gemm_rrr_f32(
    const float* a,
    const float* b,
    float* c,
    int m,
    int n,
    int k,
    int iterations,
    cudaStream_t stream) {
  return profile_gemm<cutlass::layout::RowMajor>(a, b, c, m, n, k, n, iterations, stream);
}

extern "C" float dinoml_profile_cutlass_gemm_rcr_f32(
    const float* a,
    const float* b,
    float* c,
    int m,
    int n,
    int k,
    int iterations,
    cudaStream_t stream) {
  return profile_gemm<cutlass::layout::ColumnMajor>(a, b, c, m, n, k, k, iterations, stream);
}
'''


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
