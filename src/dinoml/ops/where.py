from __future__ import annotations

from typing import Any

from dinoml.frontend import Tensor, as_tensor
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
