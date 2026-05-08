from __future__ import annotations

import builtins
from typing import Any, Callable, Mapping

from dinoml.frontend import GraphBuilder, Parameter, Tensor, as_tensor
from dinoml.ops.definitions import OP_REGISTRY, OpDef, get_op_def
from dinoml.ops.shape_views import flatten, identity, reshape, squeeze, unsqueeze


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
    out_shape = op_def.infer_shape([tensor.shape for tensor in tensors])
    out_shape_spec = _infer_shape_spec([tensor.shape_spec for tensor in tensors], out_shape)
    out = builder.emit(op_name, tensors, out_shape, dtype, op_attrs)
    out.shape_spec = out_shape_spec
    builder.tensors[out.name]["shape_spec"] = out_shape_spec
    return out


def make_frontend_op(op_name: str) -> Callable[..., Tensor]:
    op_def = get_op_def(op_name)
    frontend_name = op_def.frontend.name if op_def.frontend is not None else op_name

    def _frontend(*args: Any, **attrs: Any) -> Tensor:
        return emit_registered_op(op_name, *args, attrs=attrs or None)

    _frontend.__name__ = frontend_name
    _frontend.__qualname__ = frontend_name
    _frontend.__doc__ = op_def.description
    return _frontend


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


__all__ = [
    *OP_REGISTRY.frontend_names(),
    "emit_registered_op",
    "flatten",
    "identity",
    "make_frontend_op",
    "output",
    "reshape",
    "squeeze",
    "unsqueeze",
]
