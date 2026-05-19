from __future__ import annotations

from typing import Any

from dinoml.frontend import Tensor, as_tensor
from dinoml.ir import normalize_dtype
from dinoml.ops.elementwise import CAST_ELEMENTWISE_DTYPES, infer_elementwise
from dinoml.ops.registry import AttrDef, FrontendBinding, OpDef, OpSchema, op_def

arity = 1

attr_defaults = (("dtype", "float32"),)

op_name = "cast"


@op_def
class Cast(OpDef):
    name = op_name
    schema = OpSchema(
        inputs=tuple(f"x{idx}" for idx in range(arity)),
        attrs=tuple(
            AttrDef(name=name, type_name=type(default).__name__, default=default)
            for name, default in attr_defaults
        ),
    )
    infer_shape = infer_elementwise
    frontend = FrontendBinding(
        name=op_name,
        default_attrs={name: default for name, default in attr_defaults},
    )
    allowed_dtypes = CAST_ELEMENTWISE_DTYPES
    description = f"Elementwise {op_name}. Lowered through fused_elementwise."

    @classmethod
    def forward(cls, x: Any, dtype: str) -> Tensor:
        dtype = normalize_dtype(dtype)
        if dtype not in CAST_ELEMENTWISE_DTYPES:
            raise ValueError(f"cast does not support dtype {dtype}")
        input_tensor = as_tensor(x)
        if input_tensor.dtype not in CAST_ELEMENTWISE_DTYPES:
            raise ValueError(f"cast does not support input dtype {input_tensor.dtype}")
        return input_tensor.builder.emit(
            op_name,
            [input_tensor],
            input_tensor.shape,
            dtype,
            {"dtype": dtype},
            shape_spec=input_tensor.shape_spec,
        )


def cast(x: Any, dtype: str) -> Tensor:
    return Cast.forward(x, dtype)
