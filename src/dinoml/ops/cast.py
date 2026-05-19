from typing import Any, Sequence

from dinoml.frontend import Tensor, as_tensor
from dinoml.ir import normalize_dtype
from dinoml.ops import OpDef
from dinoml.ops.registry import AttrDef, FrontendBinding, OpSchema

arity = 1

attr_defaults = (("dtype", "float32"),)

op_name = "cast"

# common
FLOAT_ELEMENTWISE_DTYPES = ("float16", "float32", "bfloat16")
ELEMENTWISE_OUTPUT_DTYPES = (*FLOAT_ELEMENTWISE_DTYPES, "bool")
CAST_ELEMENTWISE_DTYPES = ELEMENTWISE_OUTPUT_DTYPES


# common
def broadcast_shape(a_shape: Sequence[int], b_shape: Sequence[int]) -> list[int]:
    result = []
    for a_dim, b_dim in zip(reversed(a_shape), reversed(b_shape)):
        if a_dim == b_dim:
            result.append(a_dim)
        elif a_dim == 1:
            result.append(b_dim)
        elif b_dim == 1:
            result.append(a_dim)
        else:
            raise ValueError(
                f"Shapes are not broadcastable: {list(a_shape)} and {list(b_shape)}"
            )
    longer = list(a_shape) if len(a_shape) > len(b_shape) else list(b_shape)
    prefix = longer[: abs(len(a_shape) - len(b_shape))]
    return [*prefix, *reversed(result)]


# common
def infer_elementwise(shapes: Sequence[Sequence[int]]) -> list[int]:
    if not shapes:
        raise ValueError("elementwise ops require at least one input")
    shape = list(shapes[0])
    for next_shape in shapes[1:]:
        shape = broadcast_shape(shape, next_shape)
    return shape


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
    frontend = (
        FrontendBinding(
            name=op_name,
            default_attrs={name: default for name, default in attr_defaults},
        ),
    )
    allowed_dtypes = CAST_ELEMENTWISE_DTYPES
    description = f"Elementwise {op_name}. Lowered through fused_elementwise."

    @classmethod
    def forward(self, x: Tensor, dtype: str):
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


def cast(x: Tensor, dtype: str) -> Tensor:
    return Cast.forward(x, dtype)
