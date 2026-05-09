from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from dinoml.kernels.bmm import (
    BMM_BASE_OPS,
    BMM_SUPPORTED_DTYPES,
    bmm_op_spec,
)
from dinoml.kernels.gemm import (
    GEMM_OPS,
    GEMM_SUPPORTED_DTYPES,
    cutlass_gemm_candidate_set,
    cutlass_gemm_candidates,
    cutlass_gemm_profiler_symbol,
    cutlass_gemm_symbol,
    gemm_op_spec,
)
from dinoml.kernels.providers.cutlass.bmm import (
    cutlass_bmm_candidate_set,
    cutlass_bmm_candidates,
    cutlass_bmm_profiler_symbol,
    cutlass_bmm_symbol,
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


def _cutlass_gemm_family(op_name: str) -> ExternalKernelFamily:
    spec = gemm_op_spec(op_name)
    return ExternalKernelFamily(
        op_name=op_name,
        backend="cuda",
        provider="cutlass",
        family="gemm_universal",
        required_libraries=("cutlass", "cublaslt"),
        profiler_symbol=cutlass_gemm_profiler_symbol(op_name, "float32"),
        kernel_symbol=cutlass_gemm_symbol(op_name, "float32"),
        kernel_symbols_by_dtype={dtype: cutlass_gemm_symbol(op_name, dtype) for dtype in GEMM_SUPPORTED_DTYPES},
        profiler_symbols_by_dtype={dtype: cutlass_gemm_profiler_symbol(op_name, dtype) for dtype in GEMM_SUPPORTED_DTYPES},
        candidates_by_dtype={dtype: cutlass_gemm_candidates(op_name, dtype) for dtype in GEMM_SUPPORTED_DTYPES},
        candidate_sets_by_dtype={dtype: cutlass_gemm_candidate_set(op_name, dtype) for dtype in GEMM_SUPPORTED_DTYPES},
        attrs={
            "a_layout": spec.layouts["a"],
            "b_layout": spec.layouts["b"],
            "c_layout": spec.layouts["c"],
            "epilogue": spec.epilogue.name,
            "epilogue_config": spec.epilogue.to_json(),
            "supported_dtypes": list(GEMM_SUPPORTED_DTYPES),
        },
    )


CUTLASS_GEMM_FAMILIES = tuple(_cutlass_gemm_family(op_name) for op_name in GEMM_OPS)


def _cutlass_bmm_family(op_name: str) -> ExternalKernelFamily:
    spec = bmm_op_spec(op_name)
    return ExternalKernelFamily(
        op_name=op_name,
        backend="cuda",
        provider="cutlass",
        family="bmm_strided",
        required_libraries=("cutlass", "cublaslt"),
        profiler_symbol=cutlass_bmm_profiler_symbol(op_name, "float32"),
        kernel_symbol=cutlass_bmm_symbol(op_name, "float32"),
        kernel_symbols_by_dtype={dtype: cutlass_bmm_symbol(op_name, dtype) for dtype in BMM_SUPPORTED_DTYPES},
        profiler_symbols_by_dtype={dtype: cutlass_bmm_profiler_symbol(op_name, dtype) for dtype in BMM_SUPPORTED_DTYPES},
        candidates_by_dtype={dtype: cutlass_bmm_candidates(op_name, dtype) for dtype in BMM_SUPPORTED_DTYPES},
        candidate_sets_by_dtype={dtype: cutlass_bmm_candidate_set(op_name, dtype) for dtype in BMM_SUPPORTED_DTYPES},
        attrs={
            "a_layout": spec.layouts["a"],
            "b_layout": spec.layouts["b"],
            "c_layout": spec.layouts["c"],
            "epilogue": spec.epilogue,
            "supported_dtypes": list(BMM_SUPPORTED_DTYPES),
        },
    )


CUTLASS_BMM_FAMILIES = tuple(_cutlass_bmm_family(op_name) for op_name in BMM_BASE_OPS)


def external_kernel_families(provider: str | None = None, backend: str | None = None) -> tuple[ExternalKernelFamily, ...]:
    families = (*CUTLASS_GEMM_FAMILIES, *CUTLASS_BMM_FAMILIES)
    if provider is not None:
        families = tuple(family for family in families if family.provider == provider)
    if backend is not None:
        families = tuple(family for family in families if family.backend == backend)
    return families
