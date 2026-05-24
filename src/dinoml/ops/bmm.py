from __future__ import annotations

from dinoml.frontend import Tensor, as_tensor
from dinoml.kernels.bmm import BMM_BASE_OPS, BMM_SUPPORTED_DTYPES, bmm_op_spec
from dinoml.kernels.providers.cutlass.bmm import (
    cutlass_bmm_candidate_set,
    cutlass_bmm_candidates,
    cutlass_bmm_profiler_symbol,
    cutlass_bmm_symbol,
)
from dinoml.kernels.providers.ck.bmm import (
    ck_bmm_candidate_set,
    ck_bmm_candidates,
    ck_bmm_profiler_symbol,
    ck_bmm_symbol,
)
from dinoml.ops.registry import FrontendBinding, KernelBinding, KernelVariant, OpDef, OpSchema, op_def


def _bmm(op_name: str, a: object, b: object, *epilogue_inputs: object) -> Tensor:
    a_tensor = as_tensor(a, dtype_hint=b.dtype if isinstance(b, Tensor) else "float32")
    b_tensor = as_tensor(b, dtype_hint=a_tensor.dtype)
    tensors = [a_tensor, b_tensor, *(as_tensor(value, dtype_hint=a_tensor.dtype) for value in epilogue_inputs)]
    for tensor in tensors[1:]:
        if a_tensor.builder is not tensor.builder:
            raise ValueError("Cannot combine tensors from different DinoML traces")
        if a_tensor.dtype != tensor.dtype:
            raise ValueError(f"{op_name} dtype mismatch: {a_tensor.dtype} vs {tensor.dtype}")
    if a_tensor.dtype not in BMM_SUPPORTED_DTYPES:
        raise ValueError(f"{op_name} does not support dtype {a_tensor.dtype}")
    spec = bmm_op_spec(op_name)
    out_shape = spec.validate_shapes([tensor.shape for tensor in tensors])
    out_shape_spec = spec.output_shape_spec([tensor.shape_spec for tensor in tensors])
    return a_tensor.builder.emit(op_name, tensors, out_shape, a_tensor.dtype, {}, shape_spec=out_shape_spec)


def bmm(a: object, b: object) -> Tensor:
    return _bmm("bmm_rrr", a, b)


def bmm_xxx(
    a: object,
    b: object,
    layout: str = "rrr",
    *,
    a_layout: str | None = None,
    b_layout: str | None = None,
    c_layout: str | None = None,
) -> Tensor:
    resolved_layout = _resolve_layout(
        "bmm_xxx",
        layout,
        a_layout=a_layout,
        b_layout=b_layout,
        c_layout=c_layout,
    )
    return _bmm(f"bmm_{resolved_layout}", a, b)


def bmm_xxx_add(
    a: object,
    b: object,
    d0: object,
    layout: str = "rrr",
    *,
    a_layout: str | None = None,
    b_layout: str | None = None,
    c_layout: str | None = None,
) -> Tensor:
    resolved_layout = _resolve_layout(
        "bmm_xxx_add",
        layout,
        a_layout=a_layout,
        b_layout=b_layout,
        c_layout=c_layout,
    )
    return _bmm(f"bmm_{resolved_layout}_add", a, b, d0)


def _resolve_layout(
    helper_name: str,
    layout: str,
    *,
    a_layout: str | None = None,
    b_layout: str | None = None,
    c_layout: str | None = None,
) -> str:
    if any(value is not None for value in (a_layout, b_layout, c_layout)):
        if layout != "rrr":
            raise ValueError(
                f"{helper_name} accepts either layout or per-input layout keywords, not both"
            )
        return "".join(
            (
                _normalize_layout_marker(helper_name, "a_layout", a_layout or "r"),
                _normalize_layout_marker(helper_name, "b_layout", b_layout or "r"),
                _normalize_layout_marker(helper_name, "c_layout", c_layout or "r"),
            )
        )
    normalized = layout.lower()
    if len(normalized) != 3 or any(marker not in {"c", "r"} for marker in normalized):
        supported = ", ".join(op_name.removeprefix("bmm_") for op_name in BMM_BASE_OPS)
        raise ValueError(f"{helper_name} expected layout to be one of {supported}")
    return normalized


def _normalize_layout_marker(helper_name: str, arg_name: str, value: str) -> str:
    normalized = value.lower()
    if normalized in {"c", "col", "column", "column_major"}:
        return "c"
    if normalized in {"r", "row", "row_major"}:
        return "r"
    raise ValueError(f"{helper_name} expected {arg_name} to be row or column, got {value!r}")


def _infer_shape_fn(op_name: str):
    return lambda shapes: bmm_op_spec(op_name).validate_shapes(shapes)


def _backend_kernels(op_name: str) -> dict[str, KernelBinding]:
    backend_kernels = {
        "cuda": KernelBinding(
            cutlass_bmm_symbol(op_name, "float32"),
            "cutlass_bmm",
            profiler_symbol=cutlass_bmm_profiler_symbol(op_name, "float32"),
            dtype_variants={
                dtype: KernelVariant(
                    cutlass_bmm_symbol(op_name, dtype),
                    profiler_symbol=cutlass_bmm_profiler_symbol(op_name, dtype),
                    candidates=cutlass_bmm_candidates(op_name, dtype),
                    candidate_set=cutlass_bmm_candidate_set(op_name, dtype),
                )
                for dtype in BMM_SUPPORTED_DTYPES
            },
        ),
        "rocm": KernelBinding(
            ck_bmm_symbol(op_name, "float32"),
            "ck_bmm",
            profiler_symbol=ck_bmm_profiler_symbol(op_name, "float32"),
            dtype_variants={
                dtype: KernelVariant(
                    ck_bmm_symbol(op_name, dtype),
                    profiler_symbol=ck_bmm_profiler_symbol(op_name, dtype),
                    candidates=ck_bmm_candidates(op_name, dtype),
                    candidate_set=ck_bmm_candidate_set(op_name, dtype),
                )
                for dtype in BMM_SUPPORTED_DTYPES
            },
        ),
    }
    if op_name in {"bmm_rcr", "bmm_rrr"}:
        backend_kernels["cpu"] = KernelBinding(
            symbol="generated_bmm",
            library="model",
            source_template="bmm_cpu.cpp.j2",
        )
    return backend_kernels


def _description(op_name: str) -> str:
    spec = bmm_op_spec(op_name)
    output = "C[B,N,M]" if spec.c_layout == "c" else "C[B,M,N]"
    epilogue = " with fused add epilogue" if spec.epilogue == "add" else ""
    return (
        "Batched matrix multiply frontend op: "
        f"A {spec.layouts['a']}-major, B {spec.layouts['b']}-major, {spec.layouts['c']}-major {output}{epilogue}."
    )


def _bmm_schema(op_name: str) -> OpSchema:
    spec = bmm_op_spec(op_name)
    return OpSchema(inputs=("a", "b", *spec.inputs))


class _BmmOp(OpDef):
    allowed_dtypes = BMM_SUPPORTED_DTYPES
    profiler = True

    @classmethod
    def forward(cls, a: object, b: object, *epilogue_inputs: object) -> Tensor:
        spec = bmm_op_spec(cls.name)
        expected_epilogue_inputs = spec.input_count - 2
        if len(epilogue_inputs) != expected_epilogue_inputs:
            raise ValueError(f"{cls.name} expects {spec.input_count} inputs, got {2 + len(epilogue_inputs)}")
        return _bmm(cls.name, a, b, *epilogue_inputs)


@op_def
class BmmCcc(_BmmOp):
    name = "bmm_ccc"
    schema = _bmm_schema(name)
    infer_shape = _infer_shape_fn(name)
    backend_kernels = _backend_kernels(name)
    frontend = FrontendBinding(name)
    description = _description(name)


@op_def
class BmmCcr(_BmmOp):
    name = "bmm_ccr"
    schema = _bmm_schema(name)
    infer_shape = _infer_shape_fn(name)
    backend_kernels = _backend_kernels(name)
    frontend = FrontendBinding(name)
    description = _description(name)


@op_def
class BmmCrc(_BmmOp):
    name = "bmm_crc"
    schema = _bmm_schema(name)
    infer_shape = _infer_shape_fn(name)
    backend_kernels = _backend_kernels(name)
    frontend = FrontendBinding(name)
    description = _description(name)


@op_def
class BmmCrr(_BmmOp):
    name = "bmm_crr"
    schema = _bmm_schema(name)
    infer_shape = _infer_shape_fn(name)
    backend_kernels = _backend_kernels(name)
    frontend = FrontendBinding(name)
    description = _description(name)


@op_def
class BmmRcc(_BmmOp):
    name = "bmm_rcc"
    schema = _bmm_schema(name)
    infer_shape = _infer_shape_fn(name)
    backend_kernels = _backend_kernels(name)
    frontend = FrontendBinding(name)
    description = _description(name)


@op_def
class BmmRcr(_BmmOp):
    name = "bmm_rcr"
    schema = _bmm_schema(name)
    infer_shape = _infer_shape_fn(name)
    backend_kernels = _backend_kernels(name)
    frontend = FrontendBinding(name)
    description = _description(name)


@op_def
class BmmRrc(_BmmOp):
    name = "bmm_rrc"
    schema = _bmm_schema(name)
    infer_shape = _infer_shape_fn(name)
    backend_kernels = _backend_kernels(name)
    frontend = FrontendBinding(name)
    description = _description(name)


@op_def
class BmmRrr(_BmmOp):
    name = "bmm_rrr"
    schema = _bmm_schema(name)
    infer_shape = _infer_shape_fn(name)
    backend_kernels = _backend_kernels(name)
    frontend = FrontendBinding(name)
    description = _description(name)


@op_def
class BmmCccAdd(_BmmOp):
    name = "bmm_ccc_add"
    schema = _bmm_schema(name)
    infer_shape = _infer_shape_fn(name)
    backend_kernels = _backend_kernels(name)
    frontend = FrontendBinding(name)
    description = _description(name)


@op_def
class BmmCcrAdd(_BmmOp):
    name = "bmm_ccr_add"
    schema = _bmm_schema(name)
    infer_shape = _infer_shape_fn(name)
    backend_kernels = _backend_kernels(name)
    frontend = FrontendBinding(name)
    description = _description(name)


@op_def
class BmmCrcAdd(_BmmOp):
    name = "bmm_crc_add"
    schema = _bmm_schema(name)
    infer_shape = _infer_shape_fn(name)
    backend_kernels = _backend_kernels(name)
    frontend = FrontendBinding(name)
    description = _description(name)


@op_def
class BmmCrrAdd(_BmmOp):
    name = "bmm_crr_add"
    schema = _bmm_schema(name)
    infer_shape = _infer_shape_fn(name)
    backend_kernels = _backend_kernels(name)
    frontend = FrontendBinding(name)
    description = _description(name)


@op_def
class BmmRccAdd(_BmmOp):
    name = "bmm_rcc_add"
    schema = _bmm_schema(name)
    infer_shape = _infer_shape_fn(name)
    backend_kernels = _backend_kernels(name)
    frontend = FrontendBinding(name)
    description = _description(name)


@op_def
class BmmRcrAdd(_BmmOp):
    name = "bmm_rcr_add"
    schema = _bmm_schema(name)
    infer_shape = _infer_shape_fn(name)
    backend_kernels = _backend_kernels(name)
    frontend = FrontendBinding(name)
    description = _description(name)


@op_def
class BmmRrcAdd(_BmmOp):
    name = "bmm_rrc_add"
    schema = _bmm_schema(name)
    infer_shape = _infer_shape_fn(name)
    backend_kernels = _backend_kernels(name)
    frontend = FrontendBinding(name)
    description = _description(name)


@op_def
class BmmRrrAdd(_BmmOp):
    name = "bmm_rrr_add"
    schema = _bmm_schema(name)
    infer_shape = _infer_shape_fn(name)
    backend_kernels = _backend_kernels(name)
    frontend = FrontendBinding(name)
    description = _description(name)


def bmm_ccc(a: object, b: object) -> Tensor:
    return BmmCcc.forward(a, b)


def bmm_ccr(a: object, b: object) -> Tensor:
    return BmmCcr.forward(a, b)


def bmm_crc(a: object, b: object) -> Tensor:
    return BmmCrc.forward(a, b)


def bmm_crr(a: object, b: object) -> Tensor:
    return BmmCrr.forward(a, b)


def bmm_rcc(a: object, b: object) -> Tensor:
    return BmmRcc.forward(a, b)


def bmm_rcr(a: object, b: object) -> Tensor:
    return BmmRcr.forward(a, b)


def bmm_rrc(a: object, b: object) -> Tensor:
    return BmmRrc.forward(a, b)


def bmm_rrr(a: object, b: object) -> Tensor:
    return BmmRrr.forward(a, b)


def bmm_ccc_add(a: object, b: object, d0: object) -> Tensor:
    return BmmCccAdd.forward(a, b, d0)


def bmm_ccr_add(a: object, b: object, d0: object) -> Tensor:
    return BmmCcrAdd.forward(a, b, d0)


def bmm_crc_add(a: object, b: object, d0: object) -> Tensor:
    return BmmCrcAdd.forward(a, b, d0)


def bmm_crr_add(a: object, b: object, d0: object) -> Tensor:
    return BmmCrrAdd.forward(a, b, d0)


def bmm_rcc_add(a: object, b: object, d0: object) -> Tensor:
    return BmmRccAdd.forward(a, b, d0)


def bmm_rcr_add(a: object, b: object, d0: object) -> Tensor:
    return BmmRcrAdd.forward(a, b, d0)


def bmm_rrc_add(a: object, b: object, d0: object) -> Tensor:
    return BmmRrcAdd.forward(a, b, d0)


def bmm_rrr_add(a: object, b: object, d0: object) -> Tensor:
    return BmmRrrAdd.forward(a, b, d0)


__all__ = [
    "bmm",
    "bmm_ccc",
    "bmm_ccc_add",
    "bmm_ccr",
    "bmm_ccr_add",
    "bmm_crc",
    "bmm_crc_add",
    "bmm_crr",
    "bmm_crr_add",
    "bmm_rcc",
    "bmm_rcc_add",
    "bmm_rcr",
    "bmm_rcr_add",
    "bmm_rrc",
    "bmm_rrc_add",
    "bmm_rrr",
    "bmm_rrr_add",
    "bmm_xxx",
    "bmm_xxx_add",
]
