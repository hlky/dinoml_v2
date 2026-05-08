from __future__ import annotations

from typing import Sequence

from dinoml.ops.registry import FrontendBinding, KernelBinding, OpDef, OpRegistry, OpSchema


def infer_gemm_rrr(shapes: Sequence[Sequence[int]]) -> list[int]:
    a_shape, b_shape = _validate_rank2_shapes("gemm_rrr", shapes)
    if int(a_shape[1]) != int(b_shape[0]):
        raise ValueError(f"gemm_rrr expected A[M,K] and B[K,N], got {list(a_shape)} and {list(b_shape)}")
    return [int(a_shape[0]), int(b_shape[1])]


def infer_gemm_rcr(shapes: Sequence[Sequence[int]]) -> list[int]:
    a_shape, b_shape = _validate_rank2_shapes("gemm_rcr", shapes)
    if int(a_shape[1]) != int(b_shape[1]):
        raise ValueError(f"gemm_rcr expected A[M,K] and B[N,K], got {list(a_shape)} and {list(b_shape)}")
    return [int(a_shape[0]), int(b_shape[0])]


def register_gemm_ops(registry: OpRegistry) -> None:
    registry.register(
        OpDef(
            name="gemm_rrr",
            schema=OpSchema(inputs=("a", "b")),
            infer_shape=infer_gemm_rrr,
            backend_kernels={
                "cuda": KernelBinding(
                    "dinoml_cutlass_gemm_rrr_f32",
                    "cutlass_gemm",
                    profiler_symbol="dinoml_profile_cutlass_gemm_rrr_f32",
                ),
            },
            frontend=FrontendBinding("gemm_rrr"),
            allowed_dtypes=("float32",),
            profiler=True,
            description="CUTLASS-backed rank-2 GEMM: row-major A[M,K], row-major B[K,N], row-major C[M,N].",
        )
    )
    registry.register(
        OpDef(
            name="gemm_rcr",
            schema=OpSchema(inputs=("a", "b")),
            infer_shape=infer_gemm_rcr,
            backend_kernels={
                "cuda": KernelBinding(
                    "dinoml_cutlass_gemm_rcr_f32",
                    "cutlass_gemm",
                    profiler_symbol="dinoml_profile_cutlass_gemm_rcr_f32",
                ),
            },
            frontend=FrontendBinding("gemm_rcr"),
            allowed_dtypes=("float32",),
            profiler=True,
            description="CUTLASS-backed rank-2 GEMM: row-major A[M,K], column-major-logical B[N,K], row-major C[M,N].",
        )
    )


def _validate_rank2_shapes(op_name: str, shapes: Sequence[Sequence[int]]) -> tuple[Sequence[int], Sequence[int]]:
    if len(shapes) != 2:
        raise ValueError(f"{op_name} expects exactly two inputs")
    a_shape, b_shape = shapes
    if len(a_shape) != 2 or len(b_shape) != 2:
        raise ValueError(f"{op_name} currently supports rank-2 tensors only")
    if int(a_shape[0]) <= 0 or int(a_shape[1]) <= 0 or int(b_shape[0]) <= 0 or int(b_shape[1]) <= 0:
        raise ValueError(f"{op_name} dimensions must be positive")
    return a_shape, b_shape
