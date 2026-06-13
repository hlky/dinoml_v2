from __future__ import annotations

from typing import Any

from dinoml.frontend import Parameter, Tensor, as_tensor
from dinoml.ops._frontend_utils import infer_shape_spec
from dinoml.ops.definitions import get_op_def
from dinoml.ops.elementwise import ELEMENTWISE_OUTPUT_DTYPES


def where(condition: Any, x: Any, y: Any) -> Tensor:
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
    if x_tensor.dtype not in ELEMENTWISE_OUTPUT_DTYPES:
        raise ValueError(f"where does not support dtype {x_tensor.dtype}")
    out_shape = op_def.infer_shape([tensor.shape for tensor in tensors])
    out_shape_spec = infer_shape_spec([tensor.shape_spec for tensor in tensors], out_shape)
    return builder.emit("where", tensors, out_shape, x_tensor.dtype, {}, shape_spec=out_shape_spec)


def masked_fill(x: Any, mask: Any, value: Any) -> Tensor:
    x_tensor = as_tensor(x)
    mask_tensor = as_tensor(mask)
    if x_tensor.builder is not mask_tensor.builder:
        raise ValueError("Cannot combine tensors from different DinoML traces")
    if mask_tensor.dtype != "bool":
        raise ValueError(f"masked_fill mask must have dtype bool, got {mask_tensor.dtype}")
    if x_tensor.dtype not in ELEMENTWISE_OUTPUT_DTYPES or x_tensor.dtype == "bool":
        raise ValueError(f"masked_fill does not support dtype {x_tensor.dtype}")
    fill_tensor = _masked_fill_scalar_tensor(x_tensor, value)
    return where(mask_tensor, fill_tensor, x_tensor)


def _masked_fill_scalar_tensor(x_tensor: Tensor, value: Any) -> Tensor:
    if isinstance(value, bool):
        if x_tensor.dtype == "bool":
            normalized_value: bool | float = value
        else:
            normalized_value = float(value)
    elif isinstance(value, (int, float)):
        if x_tensor.dtype == "bool":
            normalized_value = bool(value)
        else:
            normalized_value = float(value)
    else:
        raise TypeError(f"masked_fill value must be a scalar bool/int/float, got {type(value).__name__}")
    return x_tensor.builder.constant(Parameter([], dtype=x_tensor.dtype, value=normalized_value))
