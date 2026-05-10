from __future__ import annotations

import builtins
from typing import Any, Callable, Mapping

from dinoml.frontend import GraphBuilder, Parameter, Tensor, as_tensor
from dinoml.ir import normalize_dtype
from dinoml.shapes import Shape
from dinoml.ops.definitions import OP_REGISTRY, OpDef, get_op_def
from dinoml.ops.creation import ARANGE_DTYPES, CREATION_DTYPES, RANDN_DTYPES
from dinoml.ops.bmm import BMM_FRONTEND_OPS, BMM_HELPER_OPS
from dinoml.ops.elementwise import CAST_ELEMENTWISE_DTYPES, ELEMENTWISE_BY_NAME, FLOAT_ELEMENTWISE_DTYPES, elementwise_output_dtype
from dinoml.ops.gemm import GEMM_FRONTEND_OPS
from dinoml.ops.reductions import reduce_max, reduce_mean, reduce_min, reduce_sum, var, vector_norm
from dinoml.ops.shape_views import flatten, identity, reshape, squeeze, unsqueeze
from dinoml.ops.softmax import softmax


def emit_registered_op(op_name: str, *args: Any, attrs: Mapping[str, Any] | None = None) -> Tensor:
    op_def = get_op_def(op_name)
    if not op_def.accepts_input_count(len(args)):
        raise ValueError(f"{op_name} expects {op_def.input_count} inputs, got {len(args)}")
    dtype_hint = _dtype_hint(args, op_def)
    tensors = [as_tensor(arg, dtype_hint=dtype_hint) for arg in args]
    builder, dtype = _resolve_builder_and_dtype(op_def, tensors)
    op_attrs = dict(op_def.frontend.default_attrs if op_def.frontend is not None else {})
    if attrs is not None:
        op_attrs.update(attrs)
    out_shape = op_def.infer_shape_for([tensor.shape for tensor in tensors], op_attrs)
    out_shape_spec = _infer_shape_spec([tensor.shape_spec for tensor in tensors], out_shape)
    out_dtype = elementwise_output_dtype(op_name, dtype, op_attrs) if op_name in ELEMENTWISE_BY_NAME else dtype
    return builder.emit(op_name, tensors, out_shape, out_dtype, op_attrs, shape_spec=out_shape_spec)


def make_frontend_op(op_name: str) -> Callable[..., Tensor]:
    op_def = get_op_def(op_name)
    frontend_name = op_def.frontend.name if op_def.frontend is not None else op_name

    def _frontend(*args: Any, **attrs: Any) -> Tensor:
        return emit_registered_op(op_name, *args, attrs=attrs or None)

    _frontend.__name__ = frontend_name
    _frontend.__qualname__ = frontend_name
    _frontend.__doc__ = op_def.description
    return _frontend


def _where_frontend(condition: Any, x: Any, y: Any) -> Tensor:
    op_def = get_op_def("where")
    condition_tensor = as_tensor(condition)
    x_tensor = as_tensor(x)
    y_tensor = as_tensor(y, dtype_hint=x_tensor.dtype)
    tensors = [condition_tensor, x_tensor, y_tensor]
    builder = condition_tensor.builder
    for tensor in tensors[1:]:
        if tensor.builder is not builder:
            raise ValueError("Cannot combine tensors from different DinoML traces")
    if condition_tensor.dtype != "bool":
        raise ValueError(f"where condition must have dtype bool, got {condition_tensor.dtype}")
    if x_tensor.dtype != y_tensor.dtype:
        raise ValueError(f"where x/y dtype mismatch: {x_tensor.dtype} vs {y_tensor.dtype}")
    if x_tensor.dtype not in FLOAT_ELEMENTWISE_DTYPES:
        raise ValueError(f"where does not support dtype {x_tensor.dtype}")
    out_shape = op_def.infer_shape([tensor.shape for tensor in tensors])
    out_shape_spec = _infer_shape_spec([tensor.shape_spec for tensor in tensors], out_shape)
    return builder.emit("where", tensors, out_shape, x_tensor.dtype, {}, shape_spec=out_shape_spec)


def _cast_frontend(x: Any, dtype: str) -> Tensor:
    dtype = normalize_dtype(dtype)
    if dtype not in CAST_ELEMENTWISE_DTYPES:
        raise ValueError(f"cast does not support dtype {dtype}")
    input_tensor = as_tensor(x)
    if input_tensor.dtype not in CAST_ELEMENTWISE_DTYPES:
        raise ValueError(f"cast does not support input dtype {input_tensor.dtype}")
    return input_tensor.builder.emit(
        "cast",
        [input_tensor],
        input_tensor.shape,
        dtype,
        {"dtype": dtype},
        shape_spec=input_tensor.shape_spec,
    )


def _full_frontend(shape: Any, fill_value: Any, dtype: str = "float32") -> Tensor:
    dtype = normalize_dtype(dtype)
    if dtype not in CREATION_DTYPES:
        raise ValueError(f"full does not support dtype {dtype}")
    shape_obj = Shape(shape)
    if len(shape_obj) == 0:
        raise ValueError("full shape must not be empty")
    if shape_obj.dynamic:
        raise ValueError("full currently supports only static shapes")
    if dtype == "bool":
        normalized_fill: bool | float = bool(fill_value)
    else:
        normalized_fill = float(fill_value)
    attrs = {"shape": shape_obj.max_shape, "fill_value": normalized_fill, "dtype": dtype}
    return GraphBuilder.current().emit(
        "full",
        [],
        shape_obj.max_shape,
        dtype,
        attrs,
        shape_spec=shape_obj.to_json(),
    )


def _arange_frontend(start: Any, end: Any | None = None, step: Any = 1, dtype: str = "float32") -> Tensor:
    dtype = normalize_dtype(dtype)
    if dtype not in ARANGE_DTYPES:
        raise ValueError(f"arange does not support dtype {dtype}")
    if end is None:
        normalized_start = 0.0
        normalized_end = _creation_number(start, "end")
    else:
        normalized_start = _creation_number(start, "start")
        normalized_end = _creation_number(end, "end")
    normalized_step = _creation_number(step, "step")
    attrs = {"start": normalized_start, "end": normalized_end, "step": normalized_step, "dtype": dtype}
    op_def = get_op_def("arange")
    out_shape = op_def.infer_shape_for([], attrs)
    return GraphBuilder.current().emit(
        "arange",
        [],
        out_shape,
        dtype,
        attrs,
        shape_spec=out_shape,
    )


def _randn_frontend(shape: Any, dtype: str = "float32", seed: int = 0) -> Tensor:
    dtype = normalize_dtype(dtype)
    if dtype not in RANDN_DTYPES:
        raise ValueError(f"randn does not support dtype {dtype}")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("randn requires integer seed")
    if seed < 0 or seed > 0xFFFFFFFFFFFFFFFF:
        raise ValueError("randn seed must fit in uint64")
    shape_obj = Shape(shape)
    if len(shape_obj) == 0:
        raise ValueError("randn shape must not be empty")
    if shape_obj.dynamic:
        raise ValueError("randn currently supports only static shapes")
    attrs = {"shape": shape_obj.max_shape, "dtype": dtype, "seed": int(seed)}
    return GraphBuilder.current().emit(
        "randn",
        [],
        shape_obj.max_shape,
        dtype,
        attrs,
        shape_spec=shape_obj.to_json(),
    )


def _creation_number(value: Any, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"arange requires numeric {name}")
    return float(value)


def output(x: Any, name: str = "output_0") -> Tensor:
    tensor = as_tensor(x)
    tensor.output_name = name
    return tensor


def _resolve_builder_and_dtype(op_def: OpDef, tensors: list[Tensor]) -> tuple[GraphBuilder, str]:
    if not tensors:
        return GraphBuilder.current(), op_def.allowed_dtypes[0]
    first = tensors[0]
    for tensor in tensors[1:]:
        if tensor.builder is not first.builder:
            raise ValueError("Cannot combine tensors from different DinoML traces")
        if tensor.dtype != first.dtype:
            raise ValueError(f"{op_def.name} dtype mismatch: {first.dtype} vs {tensor.dtype}")
    if first.dtype not in op_def.allowed_dtypes:
        raise ValueError(f"{op_def.name} does not support dtype {first.dtype}")
    return first.builder, first.dtype


def _dtype_hint(args: tuple[Any, ...], op_def: OpDef) -> str:
    for arg in args:
        if isinstance(arg, (Tensor, Parameter)):
            return arg.dtype
    return op_def.allowed_dtypes[0]


def _infer_shape_spec(shape_specs: list[list[Any]], out_shape: list[int]) -> list[Any]:
    if not shape_specs:
        return list(out_shape)
    result: list[Any] = []
    max_rank = builtins.max(len(shape) for shape in shape_specs)
    aligned = [[1] * (max_rank - len(shape)) + list(shape) for shape in shape_specs]
    for dims in zip(*aligned):
        chosen = dims[0]
        for dim in dims[1:]:
            if _dim_is_one(chosen):
                chosen = dim
            elif _dim_is_one(dim):
                continue
            elif dim == chosen:
                continue
            else:
                return list(out_shape)
        result.append(chosen)
    return result


def _dim_is_one(dim: Any) -> bool:
    return isinstance(dim, int) and int(dim) == 1


for _frontend_name in OP_REGISTRY.frontend_names():
    _op_def = OP_REGISTRY.get_frontend(_frontend_name)
    globals()[_frontend_name] = make_frontend_op(_op_def.name)

globals()["where"] = _where_frontend
globals()["cast"] = _cast_frontend
globals()["full"] = _full_frontend
globals()["arange"] = _arange_frontend
globals()["randn"] = _randn_frontend
globals().update(GEMM_FRONTEND_OPS)
globals().update(BMM_FRONTEND_OPS)
globals().update(BMM_HELPER_OPS)


__all__ = list(dict.fromkeys([
    *OP_REGISTRY.frontend_names(),
    *BMM_HELPER_OPS,
    "emit_registered_op",
    "flatten",
    "identity",
    "make_frontend_op",
    "output",
    "reshape",
    "reduce_max",
    "reduce_mean",
    "reduce_min",
    "reduce_sum",
    "randn",
    "softmax",
    "squeeze",
    "unsqueeze",
    "var",
    "vector_norm",
    "arange",
    "cast",
    "full",
    "where",
]))
