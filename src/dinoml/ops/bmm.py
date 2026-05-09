from __future__ import annotations

from dinoml.frontend import Tensor, as_tensor
from dinoml.kernels.bmm import BMM_OPS, BMM_SUPPORTED_DTYPES, bmm_op_spec
from dinoml.ops.registry import FrontendBinding, OpDef, OpRegistry, OpSchema


def register_bmm_ops(registry: OpRegistry) -> None:
    for op_name in BMM_OPS:
        spec = bmm_op_spec(op_name)
        registry.register(
            OpDef(
                name=op_name,
                schema=OpSchema(inputs=("a", "b", *spec.inputs)),
                infer_shape=_infer_shape_fn(op_name),
                frontend=FrontendBinding(op_name),
                allowed_dtypes=BMM_SUPPORTED_DTYPES,
                description=_description(op_name),
            )
        )


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


def _make_bmm_frontend(op_name: str):
    spec = bmm_op_spec(op_name)

    def _frontend(a: object, b: object, *epilogue_inputs: object) -> Tensor:
        expected_epilogue_inputs = spec.input_count - 2
        if len(epilogue_inputs) != expected_epilogue_inputs:
            raise ValueError(f"{op_name} expects {spec.input_count} inputs, got {2 + len(epilogue_inputs)}")
        return _bmm(op_name, a, b, *epilogue_inputs)

    _frontend.__name__ = op_name
    _frontend.__qualname__ = op_name
    return _frontend


BMM_FRONTEND_OPS = {op_name: _make_bmm_frontend(op_name) for op_name in BMM_OPS}
globals().update(BMM_FRONTEND_OPS)


def _infer_shape_fn(op_name: str):
    return lambda shapes: bmm_op_spec(op_name).validate_shapes(shapes)


def _description(op_name: str) -> str:
    spec = bmm_op_spec(op_name)
    output = "C[B,N,M]" if spec.c_layout == "c" else "C[B,M,N]"
    epilogue = " with fused add epilogue" if spec.epilogue == "add" else ""
    return (
        "Batched matrix multiply frontend op: "
        f"A {spec.layouts['a']}-major, B {spec.layouts['b']}-major, {spec.layouts['c']}-major {output}{epilogue}."
    )
