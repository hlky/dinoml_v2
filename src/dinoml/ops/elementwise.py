from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from dinoml.ops.registry import AttrDef, FrontendBinding, KernelBinding, OpDef, OpRegistry, OpSchema


@dataclass(frozen=True)
class ElementwiseSpec:
    name: str
    arity: int
    math_func: str
    attr_defaults: tuple[tuple[str, Any], ...] = ()
    output_dtype: str | None = None
    allowed_dtypes: tuple[str, ...] | None = None


ELEMENTWISE_SPECS: tuple[ElementwiseSpec, ...] = (
    ElementwiseSpec("add", 2, "add"),
    ElementwiseSpec("sub", 2, "sub"),
    ElementwiseSpec("mul", 2, "mul"),
    ElementwiseSpec("div", 2, "div"),
    ElementwiseSpec("tanh", 1, "tanh"),
    ElementwiseSpec("cos", 1, "cos"),
    ElementwiseSpec("sin", 1, "sin"),
    ElementwiseSpec("sign", 1, "sign"),
    ElementwiseSpec("abs", 1, "abs"),
    ElementwiseSpec("log", 1, "log"),
    ElementwiseSpec("log1p", 1, "log1p"),
    ElementwiseSpec("exp", 1, "exp"),
    ElementwiseSpec("sqrt", 1, "sqrt"),
    ElementwiseSpec("max", 2, "max"),
    ElementwiseSpec("min", 2, "min"),
    ElementwiseSpec("sigmoid", 1, "sigmoid"),
    ElementwiseSpec("leaky_relu", 1, "leaky_relu", (("negative_slope", 0.01),)),
    ElementwiseSpec("hardtanh", 1, "hardtanh", (("min_value", -1.0), ("max_value", 1.0))),
    ElementwiseSpec("relu", 1, "relu"),
    ElementwiseSpec(
        "nan_to_num",
        1,
        "nan_to_num",
        (("nan_replacement", 0.0), ("posinf_replacement", 0.0), ("neginf_replacement", 0.0)),
    ),
    ElementwiseSpec(
        "clamp_nan_to_num",
        1,
        "clamp_nan_to_num",
        (("clamp_min", -3.4028234663852886e38), ("clamp_max", 3.4028234663852886e38), ("nan_replacement", 0.0)),
    ),
    ElementwiseSpec("silu", 1, "silu"),
    ElementwiseSpec("pow", 2, "pow"),
    ElementwiseSpec("gelu", 1, "gelu", (("approximation", "tanh"),)),
    ElementwiseSpec("fast_gelu", 1, "fast_gelu"),
    ElementwiseSpec("softplus", 1, "softplus"),
    ElementwiseSpec("elu", 1, "elu", (("alpha", 1.0),)),
    ElementwiseSpec("softsign", 1, "softsign"),
    ElementwiseSpec("floor_div", 2, "floor_div"),
    ElementwiseSpec("celu", 1, "celu", (("alpha", 1.0),)),
    ElementwiseSpec("floor", 1, "floor"),
    ElementwiseSpec("eq", 2, "eq", output_dtype="bool"),
    ElementwiseSpec("ge", 2, "ge", output_dtype="bool"),
    ElementwiseSpec("gt", 2, "gt", output_dtype="bool"),
    ElementwiseSpec("le", 2, "le", output_dtype="bool"),
    ElementwiseSpec("lt", 2, "lt", output_dtype="bool"),
    ElementwiseSpec("ne", 2, "ne", output_dtype="bool"),
    ElementwiseSpec("where", 3, "where"),
    ElementwiseSpec("cast", 1, "cast", (("dtype", "float32"),)),
)

ELEMENTWISE_BY_NAME = {spec.name: spec for spec in ELEMENTWISE_SPECS}
FUSABLE_ELEMENTWISE_OPS = frozenset(ELEMENTWISE_BY_NAME)
FLOAT_ELEMENTWISE_DTYPES = ("float16", "float32", "bfloat16")
EQ_ELEMENTWISE_DTYPES = (*FLOAT_ELEMENTWISE_DTYPES, "int32", "int64")
ELEMENTWISE_OUTPUT_DTYPES = (*FLOAT_ELEMENTWISE_DTYPES, "bool")
CAST_ELEMENTWISE_DTYPES = ELEMENTWISE_OUTPUT_DTYPES


def elementwise_output_dtype(op_name: str, input_dtype: str, attrs: Mapping[str, Any] | None = None) -> str:
    spec = ELEMENTWISE_BY_NAME[op_name]
    if op_name == "cast":
        dtype = str((attrs or {}).get("dtype", spec.attr_defaults[0][1]))
        if dtype not in CAST_ELEMENTWISE_DTYPES:
            raise ValueError(f"cast does not support dtype {dtype}")
        return dtype
    return spec.output_dtype or input_dtype


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
            raise ValueError(f"Shapes are not broadcastable: {list(a_shape)} and {list(b_shape)}")
    longer = list(a_shape) if len(a_shape) > len(b_shape) else list(b_shape)
    prefix = longer[: abs(len(a_shape) - len(b_shape))]
    return [*prefix, *reversed(result)]


def infer_elementwise(shapes: Sequence[Sequence[int]]) -> list[int]:
    if not shapes:
        raise ValueError("elementwise ops require at least one input")
    shape = list(shapes[0])
    for next_shape in shapes[1:]:
        shape = broadcast_shape(shape, next_shape)
    return shape


def register_elementwise_ops(registry: OpRegistry) -> None:
    for spec in ELEMENTWISE_SPECS:
        registry.register(
            OpDef(
                name=spec.name,
                schema=OpSchema(
                    inputs=tuple(f"x{idx}" for idx in range(spec.arity)),
                    attrs=tuple(AttrDef(name, type(default).__name__, default) for name, default in spec.attr_defaults),
                ),
                infer_shape=infer_elementwise,
                frontend=FrontendBinding(
                    spec.name,
                    default_attrs={name: default for name, default in spec.attr_defaults},
                ),
                allowed_dtypes=spec.allowed_dtypes
                or (
                    CAST_ELEMENTWISE_DTYPES
                    if spec.name == "cast"
                    else EQ_ELEMENTWISE_DTYPES if spec.name == "eq" else FLOAT_ELEMENTWISE_DTYPES
                ),
                description=f"Elementwise {spec.name}. Lowered through fused_elementwise.",
            )
        )
    registry.register(
        OpDef(
            name="fused_elementwise",
            schema=OpSchema(attrs=(AttrDef("sub_ops", "list[dict]", required=True),)),
            infer_shape=infer_elementwise,
            backend_kernels={
                "cuda": KernelBinding("generated_fused_elementwise", "model", source_template="fused_elementwise_cuda"),
                "cpu": KernelBinding("generated_fused_elementwise", "model", source_template="fused_elementwise_cpu"),
            },
            variadic_inputs=True,
            allowed_dtypes=ELEMENTWISE_OUTPUT_DTYPES,
            description="Internal fused elementwise subgraph generated into the model module.",
        )
    )
