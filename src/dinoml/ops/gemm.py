from __future__ import annotations

from typing import Sequence

from dinoml.frontend import Tensor, as_tensor
from dinoml.kernels.gemm import (
    GEMM_SUPPORTED_DTYPES,
    ck_gemm_candidate_set,
    ck_gemm_candidates,
    ck_gemm_profiler_symbol,
    ck_gemm_symbol,
    cutlass_gemm_candidate_set,
    cutlass_gemm_candidates,
    cutlass_gemm_profiler_symbol,
    cutlass_gemm_symbol,
    gemm_op_spec,
)
from dinoml.ops.registry import FrontendBinding, KernelBinding, KernelVariant, OpDef, OpSchema, op_def


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


def _ck_dtype_variants(op_name: str) -> dict[str, KernelVariant]:
    return {
        dtype: KernelVariant(
            ck_gemm_symbol(op_name, dtype),
            profiler_symbol=ck_gemm_profiler_symbol(op_name, dtype),
            candidates=ck_gemm_candidates(op_name, dtype),
            candidate_set=ck_gemm_candidate_set(op_name, dtype),
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
    return f"CUTLASS-backed folded GEMM: row-major A[...,K], {rhs}, row-major C[...,N], {epilogue}."


def _backend_kernels(op_name: str) -> dict[str, KernelBinding]:
    backend_kernels = {
        "cuda": KernelBinding(
            cutlass_gemm_symbol(op_name, "float32"),
            "cutlass_gemm",
            profiler_symbol=cutlass_gemm_profiler_symbol(op_name, "float32"),
            dtype_variants=_cutlass_dtype_variants(op_name),
        ),
        "rocm": KernelBinding(
            ck_gemm_symbol(op_name, "float32"),
            "ck_gemm",
            profiler_symbol=ck_gemm_profiler_symbol(op_name, "float32"),
            dtype_variants=_ck_dtype_variants(op_name),
        ),
    }
    if op_name in {"gemm_rcr", "gemm_rcr_bias", "gemm_rcr_bias_fast_gelu", "gemm_rcr_bias_quick_gelu"}:
        backend_kernels["cpu"] = KernelBinding(
            symbol="generated_gemm",
            library="model",
            source_template="gemm_cpu.cpp.j2",
        )
    return backend_kernels


def _gemm_schema(op_name: str) -> OpSchema:
    return OpSchema(inputs=_schema_inputs(gemm_op_spec(op_name)))


class _GemmOp(OpDef):
    allowed_dtypes = GEMM_SUPPORTED_DTYPES
    profiler = True

    @classmethod
    def forward(cls, a: object, b: object, *epilogue_inputs: object) -> Tensor:
        spec = gemm_op_spec(cls.name)
        expected_epilogue_inputs = spec.input_count - 2
        if len(epilogue_inputs) != expected_epilogue_inputs:
            raise ValueError(f"{cls.name} expects {spec.input_count} inputs, got {2 + len(epilogue_inputs)}")
        return _gemm(cls.name, a, b, *epilogue_inputs)


@op_def
class GemmRcr(_GemmOp):
    name = "gemm_rcr"
    schema = _gemm_schema(name)
    infer_shape = _infer_shape_fn(name)
    backend_kernels = _backend_kernels(name)
    frontend = FrontendBinding(name)
    description = _description(name)


@op_def
class GemmRrr(_GemmOp):
    name = "gemm_rrr"
    schema = _gemm_schema(name)
    infer_shape = _infer_shape_fn(name)
    backend_kernels = _backend_kernels(name)
    frontend = FrontendBinding(name)
    description = _description(name)


@op_def
class GemmRcrBias(_GemmOp):
    name = "gemm_rcr_bias"
    schema = _gemm_schema(name)
    infer_shape = _infer_shape_fn(name)
    backend_kernels = _backend_kernels(name)
    frontend = FrontendBinding(name)
    description = _description(name)


@op_def
class GemmRrrBias(_GemmOp):
    name = "gemm_rrr_bias"
    schema = _gemm_schema(name)
    infer_shape = _infer_shape_fn(name)
    backend_kernels = _backend_kernels(name)
    frontend = FrontendBinding(name)
    description = _description(name)


@op_def
class GemmRcrBiasRelu(_GemmOp):
    name = "gemm_rcr_bias_relu"
    schema = _gemm_schema(name)
    infer_shape = _infer_shape_fn(name)
    backend_kernels = _backend_kernels(name)
    frontend = FrontendBinding(name)
    description = _description(name)


@op_def
class GemmRrrBiasRelu(_GemmOp):
    name = "gemm_rrr_bias_relu"
    schema = _gemm_schema(name)
    infer_shape = _infer_shape_fn(name)
    backend_kernels = _backend_kernels(name)
    frontend = FrontendBinding(name)
    description = _description(name)


@op_def
class GemmRcrBiasGelu(_GemmOp):
    name = "gemm_rcr_bias_gelu"
    schema = _gemm_schema(name)
    infer_shape = _infer_shape_fn(name)
    backend_kernels = _backend_kernels(name)
    frontend = FrontendBinding(name)
    description = _description(name)


@op_def
class GemmRrrBiasGelu(_GemmOp):
    name = "gemm_rrr_bias_gelu"
    schema = _gemm_schema(name)
    infer_shape = _infer_shape_fn(name)
    backend_kernels = _backend_kernels(name)
    frontend = FrontendBinding(name)
    description = _description(name)


@op_def
class GemmRcrBiasFastGelu(_GemmOp):
    name = "gemm_rcr_bias_fast_gelu"
    schema = _gemm_schema(name)
    infer_shape = _infer_shape_fn(name)
    backend_kernels = _backend_kernels(name)
    frontend = FrontendBinding(name)
    description = _description(name)


@op_def
class GemmRrrBiasFastGelu(_GemmOp):
    name = "gemm_rrr_bias_fast_gelu"
    schema = _gemm_schema(name)
    infer_shape = _infer_shape_fn(name)
    backend_kernels = _backend_kernels(name)
    frontend = FrontendBinding(name)
    description = _description(name)


@op_def
class GemmRcrBiasQuickGelu(_GemmOp):
    name = "gemm_rcr_bias_quick_gelu"
    schema = _gemm_schema(name)
    infer_shape = _infer_shape_fn(name)
    backend_kernels = _backend_kernels(name)
    frontend = FrontendBinding(name)
    description = _description(name)


@op_def
class GemmRcrBiasSigmoid(_GemmOp):
    name = "gemm_rcr_bias_sigmoid"
    schema = _gemm_schema(name)
    infer_shape = _infer_shape_fn(name)
    backend_kernels = _backend_kernels(name)
    frontend = FrontendBinding(name)
    description = _description(name)


@op_def
class GemmRrrBiasSigmoid(_GemmOp):
    name = "gemm_rrr_bias_sigmoid"
    schema = _gemm_schema(name)
    infer_shape = _infer_shape_fn(name)
    backend_kernels = _backend_kernels(name)
    frontend = FrontendBinding(name)
    description = _description(name)


@op_def
class GemmRcrBiasTanh(_GemmOp):
    name = "gemm_rcr_bias_tanh"
    schema = _gemm_schema(name)
    infer_shape = _infer_shape_fn(name)
    backend_kernels = _backend_kernels(name)
    frontend = FrontendBinding(name)
    description = _description(name)


@op_def
class GemmRrrBiasTanh(_GemmOp):
    name = "gemm_rrr_bias_tanh"
    schema = _gemm_schema(name)
    infer_shape = _infer_shape_fn(name)
    backend_kernels = _backend_kernels(name)
    frontend = FrontendBinding(name)
    description = _description(name)


@op_def
class GemmRcrBiasSwish(_GemmOp):
    name = "gemm_rcr_bias_swish"
    schema = _gemm_schema(name)
    infer_shape = _infer_shape_fn(name)
    backend_kernels = _backend_kernels(name)
    frontend = FrontendBinding(name)
    description = _description(name)


@op_def
class GemmRrrBiasSwish(_GemmOp):
    name = "gemm_rrr_bias_swish"
    schema = _gemm_schema(name)
    infer_shape = _infer_shape_fn(name)
    backend_kernels = _backend_kernels(name)
    frontend = FrontendBinding(name)
    description = _description(name)


@op_def
class GemmRcrBiasHardswish(_GemmOp):
    name = "gemm_rcr_bias_hardswish"
    schema = _gemm_schema(name)
    infer_shape = _infer_shape_fn(name)
    backend_kernels = _backend_kernels(name)
    frontend = FrontendBinding(name)
    description = _description(name)


@op_def
class GemmRrrBiasHardswish(_GemmOp):
    name = "gemm_rrr_bias_hardswish"
    schema = _gemm_schema(name)
    infer_shape = _infer_shape_fn(name)
    backend_kernels = _backend_kernels(name)
    frontend = FrontendBinding(name)
    description = _description(name)


@op_def
class GemmRcrBiasElup1(_GemmOp):
    name = "gemm_rcr_bias_elup1"
    schema = _gemm_schema(name)
    infer_shape = _infer_shape_fn(name)
    backend_kernels = _backend_kernels(name)
    frontend = FrontendBinding(name)
    description = _description(name)


@op_def
class GemmRrrBiasElup1(_GemmOp):
    name = "gemm_rrr_bias_elup1"
    schema = _gemm_schema(name)
    infer_shape = _infer_shape_fn(name)
    backend_kernels = _backend_kernels(name)
    frontend = FrontendBinding(name)
    description = _description(name)


@op_def
class GemmRcrBiasAdd(_GemmOp):
    name = "gemm_rcr_bias_add"
    schema = _gemm_schema(name)
    infer_shape = _infer_shape_fn(name)
    backend_kernels = _backend_kernels(name)
    frontend = FrontendBinding(name)
    description = _description(name)


@op_def
class GemmRrrBiasAdd(_GemmOp):
    name = "gemm_rrr_bias_add"
    schema = _gemm_schema(name)
    infer_shape = _infer_shape_fn(name)
    backend_kernels = _backend_kernels(name)
    frontend = FrontendBinding(name)
    description = _description(name)


@op_def
class GemmRcrBiasAddAdd(_GemmOp):
    name = "gemm_rcr_bias_add_add"
    schema = _gemm_schema(name)
    infer_shape = _infer_shape_fn(name)
    backend_kernels = _backend_kernels(name)
    frontend = FrontendBinding(name)
    description = _description(name)


@op_def
class GemmRrrBiasAddAdd(_GemmOp):
    name = "gemm_rrr_bias_add_add"
    schema = _gemm_schema(name)
    infer_shape = _infer_shape_fn(name)
    backend_kernels = _backend_kernels(name)
    frontend = FrontendBinding(name)
    description = _description(name)


@op_def
class GemmRcrBiasMul(_GemmOp):
    name = "gemm_rcr_bias_mul"
    schema = _gemm_schema(name)
    infer_shape = _infer_shape_fn(name)
    backend_kernels = _backend_kernels(name)
    frontend = FrontendBinding(name)
    description = _description(name)


@op_def
class GemmRrrBiasMul(_GemmOp):
    name = "gemm_rrr_bias_mul"
    schema = _gemm_schema(name)
    infer_shape = _infer_shape_fn(name)
    backend_kernels = _backend_kernels(name)
    frontend = FrontendBinding(name)
    description = _description(name)


@op_def
class GemmRcrBiasMulAdd(_GemmOp):
    name = "gemm_rcr_bias_mul_add"
    schema = _gemm_schema(name)
    infer_shape = _infer_shape_fn(name)
    backend_kernels = _backend_kernels(name)
    frontend = FrontendBinding(name)
    description = _description(name)


@op_def
class GemmRrrBiasMulAdd(_GemmOp):
    name = "gemm_rrr_bias_mul_add"
    schema = _gemm_schema(name)
    infer_shape = _infer_shape_fn(name)
    backend_kernels = _backend_kernels(name)
    frontend = FrontendBinding(name)
    description = _description(name)


@op_def
class GemmRcrBiasAddRelu(_GemmOp):
    name = "gemm_rcr_bias_add_relu"
    schema = _gemm_schema(name)
    infer_shape = _infer_shape_fn(name)
    backend_kernels = _backend_kernels(name)
    frontend = FrontendBinding(name)
    description = _description(name)


@op_def
class GemmRrrBiasAddRelu(_GemmOp):
    name = "gemm_rrr_bias_add_relu"
    schema = _gemm_schema(name)
    infer_shape = _infer_shape_fn(name)
    backend_kernels = _backend_kernels(name)
    frontend = FrontendBinding(name)
    description = _description(name)


@op_def
class GemmRcrBiasAddAddRelu(_GemmOp):
    name = "gemm_rcr_bias_add_add_relu"
    schema = _gemm_schema(name)
    infer_shape = _infer_shape_fn(name)
    backend_kernels = _backend_kernels(name)
    frontend = FrontendBinding(name)
    description = _description(name)


@op_def
class GemmRrrBiasAddAddRelu(_GemmOp):
    name = "gemm_rrr_bias_add_add_relu"
    schema = _gemm_schema(name)
    infer_shape = _infer_shape_fn(name)
    backend_kernels = _backend_kernels(name)
    frontend = FrontendBinding(name)
    description = _description(name)


@op_def
class GemmRcrBiasSigmoidMul(_GemmOp):
    name = "gemm_rcr_bias_sigmoid_mul"
    schema = _gemm_schema(name)
    infer_shape = _infer_shape_fn(name)
    backend_kernels = _backend_kernels(name)
    frontend = FrontendBinding(name)
    description = _description(name)


@op_def
class GemmRrrBiasSigmoidMul(_GemmOp):
    name = "gemm_rrr_bias_sigmoid_mul"
    schema = _gemm_schema(name)
    infer_shape = _infer_shape_fn(name)
    backend_kernels = _backend_kernels(name)
    frontend = FrontendBinding(name)
    description = _description(name)


@op_def
class GemmRcrBiasSigmoidMulTanh(_GemmOp):
    name = "gemm_rcr_bias_sigmoid_mul_tanh"
    schema = _gemm_schema(name)
    infer_shape = _infer_shape_fn(name)
    backend_kernels = _backend_kernels(name)
    frontend = FrontendBinding(name)
    description = _description(name)


@op_def
class GemmRrrBiasSigmoidMulTanh(_GemmOp):
    name = "gemm_rrr_bias_sigmoid_mul_tanh"
    schema = _gemm_schema(name)
    infer_shape = _infer_shape_fn(name)
    backend_kernels = _backend_kernels(name)
    frontend = FrontendBinding(name)
    description = _description(name)


@op_def
class GemmRcrBiasMulTanh(_GemmOp):
    name = "gemm_rcr_bias_mul_tanh"
    schema = _gemm_schema(name)
    infer_shape = _infer_shape_fn(name)
    backend_kernels = _backend_kernels(name)
    frontend = FrontendBinding(name)
    description = _description(name)


@op_def
class GemmRrrBiasMulTanh(_GemmOp):
    name = "gemm_rrr_bias_mul_tanh"
    schema = _gemm_schema(name)
    infer_shape = _infer_shape_fn(name)
    backend_kernels = _backend_kernels(name)
    frontend = FrontendBinding(name)
    description = _description(name)


def gemm_rcr(a: object, b: object, *epilogue_inputs: object) -> Tensor:
    return GemmRcr.forward(a, b, *epilogue_inputs)


def gemm_rrr(a: object, b: object, *epilogue_inputs: object) -> Tensor:
    return GemmRrr.forward(a, b, *epilogue_inputs)


def gemm_rcr_bias(a: object, b: object, *epilogue_inputs: object) -> Tensor:
    return GemmRcrBias.forward(a, b, *epilogue_inputs)


def gemm_rrr_bias(a: object, b: object, *epilogue_inputs: object) -> Tensor:
    return GemmRrrBias.forward(a, b, *epilogue_inputs)


def gemm_rcr_bias_relu(a: object, b: object, *epilogue_inputs: object) -> Tensor:
    return GemmRcrBiasRelu.forward(a, b, *epilogue_inputs)


def gemm_rrr_bias_relu(a: object, b: object, *epilogue_inputs: object) -> Tensor:
    return GemmRrrBiasRelu.forward(a, b, *epilogue_inputs)


def gemm_rcr_bias_gelu(a: object, b: object, *epilogue_inputs: object) -> Tensor:
    return GemmRcrBiasGelu.forward(a, b, *epilogue_inputs)


def gemm_rrr_bias_gelu(a: object, b: object, *epilogue_inputs: object) -> Tensor:
    return GemmRrrBiasGelu.forward(a, b, *epilogue_inputs)


def gemm_rcr_bias_fast_gelu(a: object, b: object, *epilogue_inputs: object) -> Tensor:
    return GemmRcrBiasFastGelu.forward(a, b, *epilogue_inputs)


def gemm_rrr_bias_fast_gelu(a: object, b: object, *epilogue_inputs: object) -> Tensor:
    return GemmRrrBiasFastGelu.forward(a, b, *epilogue_inputs)


def gemm_rcr_bias_quick_gelu(a: object, b: object, *epilogue_inputs: object) -> Tensor:
    return GemmRcrBiasQuickGelu.forward(a, b, *epilogue_inputs)


def gemm_rcr_bias_sigmoid(a: object, b: object, *epilogue_inputs: object) -> Tensor:
    return GemmRcrBiasSigmoid.forward(a, b, *epilogue_inputs)


def gemm_rrr_bias_sigmoid(a: object, b: object, *epilogue_inputs: object) -> Tensor:
    return GemmRrrBiasSigmoid.forward(a, b, *epilogue_inputs)


def gemm_rcr_bias_tanh(a: object, b: object, *epilogue_inputs: object) -> Tensor:
    return GemmRcrBiasTanh.forward(a, b, *epilogue_inputs)


def gemm_rrr_bias_tanh(a: object, b: object, *epilogue_inputs: object) -> Tensor:
    return GemmRrrBiasTanh.forward(a, b, *epilogue_inputs)


def gemm_rcr_bias_swish(a: object, b: object, *epilogue_inputs: object) -> Tensor:
    return GemmRcrBiasSwish.forward(a, b, *epilogue_inputs)


def gemm_rrr_bias_swish(a: object, b: object, *epilogue_inputs: object) -> Tensor:
    return GemmRrrBiasSwish.forward(a, b, *epilogue_inputs)


def gemm_rcr_bias_hardswish(a: object, b: object, *epilogue_inputs: object) -> Tensor:
    return GemmRcrBiasHardswish.forward(a, b, *epilogue_inputs)


def gemm_rrr_bias_hardswish(a: object, b: object, *epilogue_inputs: object) -> Tensor:
    return GemmRrrBiasHardswish.forward(a, b, *epilogue_inputs)


def gemm_rcr_bias_elup1(a: object, b: object, *epilogue_inputs: object) -> Tensor:
    return GemmRcrBiasElup1.forward(a, b, *epilogue_inputs)


def gemm_rrr_bias_elup1(a: object, b: object, *epilogue_inputs: object) -> Tensor:
    return GemmRrrBiasElup1.forward(a, b, *epilogue_inputs)


def gemm_rcr_bias_add(a: object, b: object, *epilogue_inputs: object) -> Tensor:
    return GemmRcrBiasAdd.forward(a, b, *epilogue_inputs)


def gemm_rrr_bias_add(a: object, b: object, *epilogue_inputs: object) -> Tensor:
    return GemmRrrBiasAdd.forward(a, b, *epilogue_inputs)


def gemm_rcr_bias_add_add(a: object, b: object, *epilogue_inputs: object) -> Tensor:
    return GemmRcrBiasAddAdd.forward(a, b, *epilogue_inputs)


def gemm_rrr_bias_add_add(a: object, b: object, *epilogue_inputs: object) -> Tensor:
    return GemmRrrBiasAddAdd.forward(a, b, *epilogue_inputs)


def gemm_rcr_bias_mul(a: object, b: object, *epilogue_inputs: object) -> Tensor:
    return GemmRcrBiasMul.forward(a, b, *epilogue_inputs)


def gemm_rrr_bias_mul(a: object, b: object, *epilogue_inputs: object) -> Tensor:
    return GemmRrrBiasMul.forward(a, b, *epilogue_inputs)


def gemm_rcr_bias_mul_add(a: object, b: object, *epilogue_inputs: object) -> Tensor:
    return GemmRcrBiasMulAdd.forward(a, b, *epilogue_inputs)


def gemm_rrr_bias_mul_add(a: object, b: object, *epilogue_inputs: object) -> Tensor:
    return GemmRrrBiasMulAdd.forward(a, b, *epilogue_inputs)


def gemm_rcr_bias_add_relu(a: object, b: object, *epilogue_inputs: object) -> Tensor:
    return GemmRcrBiasAddRelu.forward(a, b, *epilogue_inputs)


def gemm_rrr_bias_add_relu(a: object, b: object, *epilogue_inputs: object) -> Tensor:
    return GemmRrrBiasAddRelu.forward(a, b, *epilogue_inputs)


def gemm_rcr_bias_add_add_relu(a: object, b: object, *epilogue_inputs: object) -> Tensor:
    return GemmRcrBiasAddAddRelu.forward(a, b, *epilogue_inputs)


def gemm_rrr_bias_add_add_relu(a: object, b: object, *epilogue_inputs: object) -> Tensor:
    return GemmRrrBiasAddAddRelu.forward(a, b, *epilogue_inputs)


def gemm_rcr_bias_sigmoid_mul(a: object, b: object, *epilogue_inputs: object) -> Tensor:
    return GemmRcrBiasSigmoidMul.forward(a, b, *epilogue_inputs)


def gemm_rrr_bias_sigmoid_mul(a: object, b: object, *epilogue_inputs: object) -> Tensor:
    return GemmRrrBiasSigmoidMul.forward(a, b, *epilogue_inputs)


def gemm_rcr_bias_sigmoid_mul_tanh(a: object, b: object, *epilogue_inputs: object) -> Tensor:
    return GemmRcrBiasSigmoidMulTanh.forward(a, b, *epilogue_inputs)


def gemm_rrr_bias_sigmoid_mul_tanh(a: object, b: object, *epilogue_inputs: object) -> Tensor:
    return GemmRrrBiasSigmoidMulTanh.forward(a, b, *epilogue_inputs)


def gemm_rcr_bias_mul_tanh(a: object, b: object, *epilogue_inputs: object) -> Tensor:
    return GemmRcrBiasMulTanh.forward(a, b, *epilogue_inputs)


def gemm_rrr_bias_mul_tanh(a: object, b: object, *epilogue_inputs: object) -> Tensor:
    return GemmRrrBiasMulTanh.forward(a, b, *epilogue_inputs)


__all__ = [
    "gemm_rcr",
    "gemm_rcr_bias",
    "gemm_rcr_bias_add",
    "gemm_rcr_bias_add_add",
    "gemm_rcr_bias_add_add_relu",
    "gemm_rcr_bias_add_relu",
    "gemm_rcr_bias_elup1",
    "gemm_rcr_bias_fast_gelu",
    "gemm_rcr_bias_gelu",
    "gemm_rcr_bias_hardswish",
    "gemm_rcr_bias_mul",
    "gemm_rcr_bias_mul_add",
    "gemm_rcr_bias_mul_tanh",
    "gemm_rcr_bias_quick_gelu",
    "gemm_rcr_bias_relu",
    "gemm_rcr_bias_sigmoid",
    "gemm_rcr_bias_sigmoid_mul",
    "gemm_rcr_bias_sigmoid_mul_tanh",
    "gemm_rcr_bias_swish",
    "gemm_rcr_bias_tanh",
    "gemm_rrr",
    "gemm_rrr_bias",
    "gemm_rrr_bias_add",
    "gemm_rrr_bias_add_add",
    "gemm_rrr_bias_add_add_relu",
    "gemm_rrr_bias_add_relu",
    "gemm_rrr_bias_elup1",
    "gemm_rrr_bias_fast_gelu",
    "gemm_rrr_bias_gelu",
    "gemm_rrr_bias_hardswish",
    "gemm_rrr_bias_mul",
    "gemm_rrr_bias_mul_add",
    "gemm_rrr_bias_mul_tanh",
    "gemm_rrr_bias_relu",
    "gemm_rrr_bias_sigmoid",
    "gemm_rrr_bias_sigmoid_mul",
    "gemm_rrr_bias_sigmoid_mul_tanh",
    "gemm_rrr_bias_swish",
    "gemm_rrr_bias_tanh",
]
