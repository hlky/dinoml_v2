from __future__ import annotations

from typing import Any, Sequence

from dinoml.frontend import Tensor, as_tensor
from dinoml.kernels.gemm import (
    GEMM_SUPPORTED_DTYPES,
    cutlass_gemm_candidate_set,
    cutlass_gemm_candidates,
    cutlass_gemm_profiler_symbol,
    cutlass_gemm_symbol,
)
from dinoml.ops.registry import FrontendBinding, KernelBinding, KernelVariant, OpDef, OpRegistry, OpSchema


def gemm_rrr(a: object, b: object) -> Tensor:
    return _gemm("gemm_rrr", a, b)


def gemm_rcr(a: object, b: object) -> Tensor:
    return _gemm("gemm_rcr", a, b)


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
                    cutlass_gemm_symbol("gemm_rrr", "float32"),
                    "cutlass_gemm",
                    profiler_symbol=cutlass_gemm_profiler_symbol("gemm_rrr", "float32"),
                    dtype_variants=_cutlass_dtype_variants("gemm_rrr"),
                ),
            },
            frontend=FrontendBinding("gemm_rrr"),
            allowed_dtypes=GEMM_SUPPORTED_DTYPES,
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
                    cutlass_gemm_symbol("gemm_rcr", "float32"),
                    "cutlass_gemm",
                    profiler_symbol=cutlass_gemm_profiler_symbol("gemm_rcr", "float32"),
                    dtype_variants=_cutlass_dtype_variants("gemm_rcr"),
                ),
            },
            frontend=FrontendBinding("gemm_rcr"),
            allowed_dtypes=GEMM_SUPPORTED_DTYPES,
            profiler=True,
            description="CUTLASS-backed rank-2 GEMM: row-major A[M,K], column-major-logical B[N,K], row-major C[M,N].",
        )
    )


def _gemm(op_name: str, a: object, b: object) -> Tensor:
    a_tensor = as_tensor(a, dtype_hint=b.dtype if isinstance(b, Tensor) else "float32")
    b_tensor = as_tensor(b, dtype_hint=a_tensor.dtype)
    if a_tensor.builder is not b_tensor.builder:
        raise ValueError("Cannot combine tensors from different DinoML traces")
    if a_tensor.dtype != b_tensor.dtype:
        raise ValueError(f"{op_name} dtype mismatch: {a_tensor.dtype} vs {b_tensor.dtype}")
    if a_tensor.dtype not in GEMM_SUPPORTED_DTYPES:
        raise ValueError(f"{op_name} does not support dtype {a_tensor.dtype}")
    infer_shape = infer_gemm_rrr if op_name == "gemm_rrr" else infer_gemm_rcr
    out_shape = infer_shape([a_tensor.shape, b_tensor.shape])
    out_shape_spec = _output_shape_spec(op_name, a_tensor.shape_spec, b_tensor.shape_spec)
    return a_tensor.builder.emit(op_name, [a_tensor, b_tensor], out_shape, a_tensor.dtype, {}, shape_spec=out_shape_spec)


def _cutlass_dtype_variants(op_name: str) -> dict[str, KernelVariant]:
    return {
        dtype: KernelVariant(
            cutlass_gemm_symbol(op_name, dtype),
            profiler_symbol=cutlass_gemm_profiler_symbol(op_name, dtype),
            candidates=cutlass_gemm_candidates(op_name, dtype),
            candidate_set=cutlass_gemm_candidate_set(op_name, dtype),
        )
        for dtype in GEMM_SUPPORTED_DTYPES
    }


def _output_shape_spec(op_name: str, a_shape_spec: Sequence[Any], b_shape_spec: Sequence[Any]) -> list[Any]:
    if op_name == "gemm_rrr":
        return [a_shape_spec[0], b_shape_spec[1]]
    if op_name == "gemm_rcr":
        return [a_shape_spec[0], b_shape_spec[0]]
    raise ValueError(f"Unsupported GEMM op: {op_name}")


def _validate_rank2_shapes(op_name: str, shapes: Sequence[Sequence[int]]) -> tuple[Sequence[int], Sequence[int]]:
    if len(shapes) != 2:
        raise ValueError(f"{op_name} expects exactly two inputs")
    a_shape, b_shape = shapes
    if len(a_shape) != 2 or len(b_shape) != 2:
        raise ValueError(f"{op_name} currently supports rank-2 tensors only")
    if int(a_shape[0]) <= 0 or int(a_shape[1]) <= 0 or int(b_shape[0]) <= 0 or int(b_shape[1]) <= 0:
        raise ValueError(f"{op_name} dimensions must be positive")
    return a_shape, b_shape
