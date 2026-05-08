from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


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

    def to_json(self) -> dict[str, Any]:
        return {
            "op_name": self.op_name,
            "backend": self.backend,
            "provider": self.provider,
            "family": self.family,
            "required_libraries": list(self.required_libraries),
            "profiler_symbol": self.profiler_symbol,
            "kernel_symbol": self.kernel_symbol,
            "attrs": dict(self.attrs),
        }


CUTLASS_GEMM_FAMILIES = (
    ExternalKernelFamily(
        op_name="gemm_rcr",
        backend="cuda",
        provider="cutlass",
        family="gemm_universal",
        required_libraries=("cutlass", "cublaslt"),
        profiler_symbol="dinoml_profile_cutlass_gemm_rcr_f32",
        kernel_symbol="dinoml_cutlass_gemm_rcr_f32",
        attrs={"a_layout": "row", "b_layout": "column", "c_layout": "row", "epilogue": "linear_combination"},
    ),
    ExternalKernelFamily(
        op_name="gemm_rrr",
        backend="cuda",
        provider="cutlass",
        family="gemm_universal",
        required_libraries=("cutlass", "cublaslt"),
        profiler_symbol="dinoml_profile_cutlass_gemm_rrr_f32",
        kernel_symbol="dinoml_cutlass_gemm_rrr_f32",
        attrs={"a_layout": "row", "b_layout": "row", "c_layout": "row", "epilogue": "linear_combination"},
    ),
)


def external_kernel_families(provider: str | None = None, backend: str | None = None) -> tuple[ExternalKernelFamily, ...]:
    families = CUTLASS_GEMM_FAMILIES
    if provider is not None:
        families = tuple(family for family in families if family.provider == provider)
    if backend is not None:
        families = tuple(family for family in families if family.backend == backend)
    return families
