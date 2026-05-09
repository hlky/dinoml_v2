from __future__ import annotations

from typing import Sequence

from dinoml.frontend import Tensor, as_tensor
from dinoml.kernels.gemm import (
    GEMM_OPS,
    GEMM_SUPPORTED_DTYPES,
    cutlass_gemm_candidate_set,
    cutlass_gemm_candidates,
    cutlass_gemm_profiler_symbol,
    cutlass_gemm_symbol,
    gemm_op_spec,
)
from dinoml.ops.registry import FrontendBinding, KernelBinding, KernelVariant, OpDef, OpRegistry, OpSchema


def infer_gemm_rrr(shapes: Sequence[Sequence[int]]) -> list[int]:
    return gemm_op_spec("gemm_rrr").validate_shapes(shapes)


def infer_gemm_rcr(shapes: Sequence[Sequence[int]]) -> list[int]:
    return gemm_op_spec("gemm_rcr").validate_shapes(shapes)


def infer_gemm_rrr_bias(shapes: Sequence[Sequence[int]]) -> list[int]:
    return gemm_op_spec("gemm_rrr_bias").validate_shapes(shapes)


def infer_gemm_rcr_bias(shapes: Sequence[Sequence[int]]) -> list[int]:
    return gemm_op_spec("gemm_rcr_bias").validate_shapes(shapes)


def infer_gemm_rrr_bias_relu(shapes: Sequence[Sequence[int]]) -> list[int]:
    return gemm_op_spec("gemm_rrr_bias_relu").validate_shapes(shapes)


def infer_gemm_rcr_bias_relu(shapes: Sequence[Sequence[int]]) -> list[int]:
    return gemm_op_spec("gemm_rcr_bias_relu").validate_shapes(shapes)


def register_gemm_ops(registry: OpRegistry) -> None:
    for op_name in GEMM_OPS:
        spec = gemm_op_spec(op_name)
        registry.register(
            OpDef(
                name=op_name,
                schema=OpSchema(inputs=_schema_inputs(spec)),
                infer_shape=_infer_shape_fn(op_name),
                backend_kernels={
                    "cuda": KernelBinding(
                        cutlass_gemm_symbol(op_name, "float32"),
                        "cutlass_gemm",
                        profiler_symbol=cutlass_gemm_profiler_symbol(op_name, "float32"),
                        dtype_variants=_cutlass_dtype_variants(op_name),
                    ),
                },
                frontend=FrontendBinding(op_name),
                allowed_dtypes=GEMM_SUPPORTED_DTYPES,
                profiler=True,
                description=_description(op_name),
            )
        )


def _gemm(op_name: str, a: object, b: object, *epilogue_inputs: object) -> Tensor:
    a_tensor = as_tensor(a, dtype_hint=b.dtype if isinstance(b, Tensor) else "float32")
    b_tensor = as_tensor(b, dtype_hint=a_tensor.dtype)
    tensors = [a_tensor, b_tensor, *(as_tensor(value, dtype_hint=a_tensor.dtype) for value in epilogue_inputs)]
    for tensor in tensors[1:]:
        if a_tensor.builder is not tensor.builder:
            raise ValueError("Cannot combine tensors from different DinoML traces")
        if a_tensor.dtype != tensor.dtype:
            raise ValueError(f"{op_name} dtype mismatch: {a_tensor.dtype} vs {tensor.dtype}")
    if a_tensor.dtype not in GEMM_SUPPORTED_DTYPES:
        raise ValueError(f"{op_name} does not support dtype {a_tensor.dtype}")
    spec = gemm_op_spec(op_name)
    out_shape = spec.validate_shapes([tensor.shape for tensor in tensors])
    out_shape_spec = spec.output_shape_spec([tensor.shape_spec for tensor in tensors])
    return a_tensor.builder.emit(op_name, tensors, out_shape, a_tensor.dtype, {}, shape_spec=out_shape_spec)


def _make_gemm_frontend(op_name: str):
    spec = gemm_op_spec(op_name)

    def _frontend(a: object, b: object, *epilogue_inputs: object) -> Tensor:
        expected_epilogue_inputs = spec.input_count - 2
        if len(epilogue_inputs) != expected_epilogue_inputs:
            raise ValueError(f"{op_name} expects {spec.input_count} inputs, got {2 + len(epilogue_inputs)}")
        return _gemm(op_name, a, b, *epilogue_inputs)

    _frontend.__name__ = op_name
    _frontend.__qualname__ = op_name
    return _frontend


GEMM_FRONTEND_OPS = {op_name: _make_gemm_frontend(op_name) for op_name in GEMM_OPS}
globals().update(GEMM_FRONTEND_OPS)


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


def _schema_inputs(spec) -> tuple[str, ...]:
    return ("a", "b", *spec.epilogue.inputs)


def _infer_shape_fn(op_name: str):
    return lambda shapes: gemm_op_spec(op_name).validate_shapes(shapes)


def _description(op_name: str) -> str:
    spec = gemm_op_spec(op_name)
    rhs = "row-major B[K,N]" if spec.base_layout == "rrr" else "column-major-logical B[N,K]"
    if spec.epilogue.activation:
        epilogue = f"with fused bias+{spec.epilogue.activation} epilogue"
    elif spec.epilogue.has_bias:
        epilogue = "with fused bias epilogue"
    else:
        epilogue = "with linear-combination epilogue"
    return f"CUTLASS-backed rank-2 GEMM: row-major A[M,K], {rhs}, row-major C[M,N], {epilogue}."
