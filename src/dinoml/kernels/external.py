from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from dinoml.kernels.gemm import (
    GEMM_SUPPORTED_DTYPES,
    cutlass_gemm_candidate_set,
    cutlass_gemm_candidates,
    cutlass_gemm_profiler_symbol,
    cutlass_gemm_symbol,
)


@dataclass(frozen=True)
class ExternalKernelFamily:
    op_name: str
    backend: str
    provider: str
    family: str
    required_libraries: tuple[str, ...]
    profiler_symbol: str
    kernel_symbol: str
    attrs: Mapping[str, Any]
    kernel_symbols_by_dtype: Mapping[str, str] | None = None
    profiler_symbols_by_dtype: Mapping[str, str] | None = None
    candidates_by_dtype: Mapping[str, Sequence[Mapping[str, Any]]] | None = None
    candidate_sets_by_dtype: Mapping[str, Mapping[str, Any]] | None = None

    def to_json(self) -> dict[str, Any]:
        candidates_by_dtype = {
            dtype: [dict(candidate) for candidate in candidates]
            for dtype, candidates in (self.candidates_by_dtype or {}).items()
        }
        candidate_sets_by_dtype = {
            dtype: dict(candidate_set)
            for dtype, candidate_set in (self.candidate_sets_by_dtype or {}).items()
        }
        return {
            "op_name": self.op_name,
            "backend": self.backend,
            "provider": self.provider,
            "family": self.family,
            "required_libraries": list(self.required_libraries),
            "profiler_symbol": self.profiler_symbol,
            "kernel_symbol": self.kernel_symbol,
            "kernel_symbols_by_dtype": dict(self.kernel_symbols_by_dtype or {}),
            "profiler_symbols_by_dtype": dict(self.profiler_symbols_by_dtype or {}),
            "candidates_by_dtype": candidates_by_dtype,
            "candidates": candidates_by_dtype.get("float32", []),
            "candidate_sets_by_dtype": candidate_sets_by_dtype,
            "candidate_set": candidate_sets_by_dtype.get("float32", {}),
            "attrs": dict(self.attrs),
        }


CUTLASS_GEMM_FAMILIES = (
    ExternalKernelFamily(
        op_name="gemm_rcr",
        backend="cuda",
        provider="cutlass",
        family="gemm_universal",
        required_libraries=("cutlass", "cublaslt"),
        profiler_symbol=cutlass_gemm_profiler_symbol("gemm_rcr", "float32"),
        kernel_symbol=cutlass_gemm_symbol("gemm_rcr", "float32"),
        kernel_symbols_by_dtype={dtype: cutlass_gemm_symbol("gemm_rcr", dtype) for dtype in GEMM_SUPPORTED_DTYPES},
        profiler_symbols_by_dtype={dtype: cutlass_gemm_profiler_symbol("gemm_rcr", dtype) for dtype in GEMM_SUPPORTED_DTYPES},
        candidates_by_dtype={dtype: cutlass_gemm_candidates("gemm_rcr", dtype) for dtype in GEMM_SUPPORTED_DTYPES},
        candidate_sets_by_dtype={dtype: cutlass_gemm_candidate_set("gemm_rcr", dtype) for dtype in GEMM_SUPPORTED_DTYPES},
        attrs={
            "a_layout": "row",
            "b_layout": "column",
            "c_layout": "row",
            "epilogue": "linear_combination",
            "supported_dtypes": list(GEMM_SUPPORTED_DTYPES),
        },
    ),
    ExternalKernelFamily(
        op_name="gemm_rrr",
        backend="cuda",
        provider="cutlass",
        family="gemm_universal",
        required_libraries=("cutlass", "cublaslt"),
        profiler_symbol=cutlass_gemm_profiler_symbol("gemm_rrr", "float32"),
        kernel_symbol=cutlass_gemm_symbol("gemm_rrr", "float32"),
        kernel_symbols_by_dtype={dtype: cutlass_gemm_symbol("gemm_rrr", dtype) for dtype in GEMM_SUPPORTED_DTYPES},
        profiler_symbols_by_dtype={dtype: cutlass_gemm_profiler_symbol("gemm_rrr", dtype) for dtype in GEMM_SUPPORTED_DTYPES},
        candidates_by_dtype={dtype: cutlass_gemm_candidates("gemm_rrr", dtype) for dtype in GEMM_SUPPORTED_DTYPES},
        candidate_sets_by_dtype={dtype: cutlass_gemm_candidate_set("gemm_rrr", dtype) for dtype in GEMM_SUPPORTED_DTYPES},
        attrs={
            "a_layout": "row",
            "b_layout": "row",
            "c_layout": "row",
            "epilogue": "linear_combination",
            "supported_dtypes": list(GEMM_SUPPORTED_DTYPES),
        },
    ),
)


def external_kernel_families(provider: str | None = None, backend: str | None = None) -> tuple[ExternalKernelFamily, ...]:
    families = CUTLASS_GEMM_FAMILIES
    if provider is not None:
        families = tuple(family for family in families if family.provider == provider)
    if backend is not None:
        families = tuple(family for family in families if family.backend == backend)
    return families
