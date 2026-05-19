from __future__ import annotations

import builtins
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from dinoml.frontend import Parameter, Tensor, as_tensor
from dinoml.ops._frontend_utils import infer_shape_spec
from dinoml.ops.registry import AttrDef, FrontendBinding, KernelBinding, OpDef, OpSchema, op_def


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
)

FLOAT_ELEMENTWISE_DTYPES = ("float16", "float32", "bfloat16")
EQ_ELEMENTWISE_DTYPES = (*FLOAT_ELEMENTWISE_DTYPES, "int32", "int64")
ELEMENTWISE_OUTPUT_DTYPES = (*FLOAT_ELEMENTWISE_DTYPES, "bool")
FUSED_ELEMENTWISE_RUNTIME_DTYPES = (*EQ_ELEMENTWISE_DTYPES, "bool")
CAST_ELEMENTWISE_DTYPES = ELEMENTWISE_OUTPUT_DTYPES
FUSION_ONLY_SPECS: tuple[ElementwiseSpec, ...] = (
    ElementwiseSpec(name="cast", arity=1, math_func="cast", attr_defaults=(("dtype", "float32"),)),
)
ELEMENTWISE_BY_NAME = {spec.name: spec for spec in (*ELEMENTWISE_SPECS, *FUSION_ONLY_SPECS)}
FUSABLE_ELEMENTWISE_OPS = frozenset(ELEMENTWISE_BY_NAME)


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
    prefix = longer[: builtins.abs(len(a_shape) - len(b_shape))]
    return [*prefix, *reversed(result)]


def infer_elementwise(shapes: Sequence[Sequence[int]]) -> list[int]:
    if not shapes:
        raise ValueError("elementwise ops require at least one input")
    shape = list(shapes[0])
    for next_shape in shapes[1:]:
        shape = broadcast_shape(shape, next_shape)
    return shape


def _elementwise_schema(spec: ElementwiseSpec) -> OpSchema:
    return OpSchema(
        inputs=tuple(f"x{idx}" for idx in range(spec.arity)),
        attrs=tuple(AttrDef(name, type(default).__name__, default) for name, default in spec.attr_defaults),
    )


def _elementwise_binding(spec: ElementwiseSpec) -> FrontendBinding:
    return FrontendBinding(spec.name, default_attrs={name: default for name, default in spec.attr_defaults})


def _elementwise_allowed_dtypes(spec: ElementwiseSpec) -> tuple[str, ...]:
    return spec.allowed_dtypes or (EQ_ELEMENTWISE_DTYPES if spec.name == "eq" else FLOAT_ELEMENTWISE_DTYPES)


def _elementwise_description(spec: ElementwiseSpec) -> str:
    return f"Elementwise {spec.name}. Lowered through fused_elementwise."


class _ElementwiseOp(OpDef):
    infer_shape = infer_elementwise

    @classmethod
    def forward(cls, *args: Any, **attrs: Any) -> Tensor:
        return _emit_elementwise(cls.name, *args, attrs=attrs or None)


def _emit_elementwise(op_name: str, *args: Any, attrs: Mapping[str, Any] | None = None) -> Tensor:
    spec = ELEMENTWISE_BY_NAME[op_name]
    if len(args) != spec.arity:
        raise ValueError(f"{op_name} expects {spec.arity} inputs, got {len(args)}")
    allowed_dtypes = _elementwise_allowed_dtypes(spec)
    dtype_hint = _dtype_hint(args, allowed_dtypes)
    tensors = [as_tensor(arg, dtype_hint=dtype_hint) for arg in args]
    if not tensors:
        raise ValueError(f"{op_name} requires at least one input")
    first = tensors[0]
    for tensor in tensors[1:]:
        if tensor.builder is not first.builder:
            raise ValueError("Cannot combine tensors from different DinoML traces")
        if tensor.dtype != first.dtype:
            raise ValueError(f"{op_name} dtype mismatch: {first.dtype} vs {tensor.dtype}")
    if first.dtype not in allowed_dtypes:
        raise ValueError(f"{op_name} does not support dtype {first.dtype}")
    op_attrs = {name: default for name, default in spec.attr_defaults}
    if attrs is not None:
        op_attrs.update(attrs)
    out_shape = infer_elementwise([tensor.shape for tensor in tensors])
    out_shape_spec = infer_shape_spec([tensor.shape_spec for tensor in tensors], out_shape)
    out_dtype = elementwise_output_dtype(op_name, first.dtype, op_attrs)
    return first.builder.emit(op_name, tensors, out_shape, out_dtype, op_attrs, shape_spec=out_shape_spec)


def _dtype_hint(args: tuple[Any, ...], allowed_dtypes: tuple[str, ...]) -> str:
    for arg in args:
        if isinstance(arg, (Tensor, Parameter)):
            return arg.dtype
    return allowed_dtypes[0]


@op_def
class Add(_ElementwiseOp):
    _spec = ELEMENTWISE_BY_NAME["add"]
    name = _spec.name
    schema = _elementwise_schema(_spec)
    frontend = _elementwise_binding(_spec)
    allowed_dtypes = _elementwise_allowed_dtypes(_spec)
    description = _elementwise_description(_spec)


@op_def
class Sub(_ElementwiseOp):
    _spec = ELEMENTWISE_BY_NAME["sub"]
    name = _spec.name
    schema = _elementwise_schema(_spec)
    frontend = _elementwise_binding(_spec)
    allowed_dtypes = _elementwise_allowed_dtypes(_spec)
    description = _elementwise_description(_spec)


@op_def
class Mul(_ElementwiseOp):
    _spec = ELEMENTWISE_BY_NAME["mul"]
    name = _spec.name
    schema = _elementwise_schema(_spec)
    frontend = _elementwise_binding(_spec)
    allowed_dtypes = _elementwise_allowed_dtypes(_spec)
    description = _elementwise_description(_spec)


@op_def
class Div(_ElementwiseOp):
    _spec = ELEMENTWISE_BY_NAME["div"]
    name = _spec.name
    schema = _elementwise_schema(_spec)
    frontend = _elementwise_binding(_spec)
    allowed_dtypes = _elementwise_allowed_dtypes(_spec)
    description = _elementwise_description(_spec)


@op_def
class Tanh(_ElementwiseOp):
    _spec = ELEMENTWISE_BY_NAME["tanh"]
    name = _spec.name
    schema = _elementwise_schema(_spec)
    frontend = _elementwise_binding(_spec)
    allowed_dtypes = _elementwise_allowed_dtypes(_spec)
    description = _elementwise_description(_spec)


@op_def
class Cos(_ElementwiseOp):
    _spec = ELEMENTWISE_BY_NAME["cos"]
    name = _spec.name
    schema = _elementwise_schema(_spec)
    frontend = _elementwise_binding(_spec)
    allowed_dtypes = _elementwise_allowed_dtypes(_spec)
    description = _elementwise_description(_spec)


@op_def
class Sin(_ElementwiseOp):
    _spec = ELEMENTWISE_BY_NAME["sin"]
    name = _spec.name
    schema = _elementwise_schema(_spec)
    frontend = _elementwise_binding(_spec)
    allowed_dtypes = _elementwise_allowed_dtypes(_spec)
    description = _elementwise_description(_spec)


@op_def
class Sign(_ElementwiseOp):
    _spec = ELEMENTWISE_BY_NAME["sign"]
    name = _spec.name
    schema = _elementwise_schema(_spec)
    frontend = _elementwise_binding(_spec)
    allowed_dtypes = _elementwise_allowed_dtypes(_spec)
    description = _elementwise_description(_spec)


@op_def
class Abs(_ElementwiseOp):
    _spec = ELEMENTWISE_BY_NAME["abs"]
    name = _spec.name
    schema = _elementwise_schema(_spec)
    frontend = _elementwise_binding(_spec)
    allowed_dtypes = _elementwise_allowed_dtypes(_spec)
    description = _elementwise_description(_spec)


@op_def
class Log(_ElementwiseOp):
    _spec = ELEMENTWISE_BY_NAME["log"]
    name = _spec.name
    schema = _elementwise_schema(_spec)
    frontend = _elementwise_binding(_spec)
    allowed_dtypes = _elementwise_allowed_dtypes(_spec)
    description = _elementwise_description(_spec)


@op_def
class Log1p(_ElementwiseOp):
    _spec = ELEMENTWISE_BY_NAME["log1p"]
    name = _spec.name
    schema = _elementwise_schema(_spec)
    frontend = _elementwise_binding(_spec)
    allowed_dtypes = _elementwise_allowed_dtypes(_spec)
    description = _elementwise_description(_spec)


@op_def
class Exp(_ElementwiseOp):
    _spec = ELEMENTWISE_BY_NAME["exp"]
    name = _spec.name
    schema = _elementwise_schema(_spec)
    frontend = _elementwise_binding(_spec)
    allowed_dtypes = _elementwise_allowed_dtypes(_spec)
    description = _elementwise_description(_spec)


@op_def
class Sqrt(_ElementwiseOp):
    _spec = ELEMENTWISE_BY_NAME["sqrt"]
    name = _spec.name
    schema = _elementwise_schema(_spec)
    frontend = _elementwise_binding(_spec)
    allowed_dtypes = _elementwise_allowed_dtypes(_spec)
    description = _elementwise_description(_spec)


@op_def
class Max(_ElementwiseOp):
    _spec = ELEMENTWISE_BY_NAME["max"]
    name = _spec.name
    schema = _elementwise_schema(_spec)
    frontend = _elementwise_binding(_spec)
    allowed_dtypes = _elementwise_allowed_dtypes(_spec)
    description = _elementwise_description(_spec)


@op_def
class Min(_ElementwiseOp):
    _spec = ELEMENTWISE_BY_NAME["min"]
    name = _spec.name
    schema = _elementwise_schema(_spec)
    frontend = _elementwise_binding(_spec)
    allowed_dtypes = _elementwise_allowed_dtypes(_spec)
    description = _elementwise_description(_spec)


@op_def
class Sigmoid(_ElementwiseOp):
    _spec = ELEMENTWISE_BY_NAME["sigmoid"]
    name = _spec.name
    schema = _elementwise_schema(_spec)
    frontend = _elementwise_binding(_spec)
    allowed_dtypes = _elementwise_allowed_dtypes(_spec)
    description = _elementwise_description(_spec)


@op_def
class LeakyRelu(_ElementwiseOp):
    _spec = ELEMENTWISE_BY_NAME["leaky_relu"]
    name = _spec.name
    schema = _elementwise_schema(_spec)
    frontend = _elementwise_binding(_spec)
    allowed_dtypes = _elementwise_allowed_dtypes(_spec)
    description = _elementwise_description(_spec)


@op_def
class Hardtanh(_ElementwiseOp):
    _spec = ELEMENTWISE_BY_NAME["hardtanh"]
    name = _spec.name
    schema = _elementwise_schema(_spec)
    frontend = _elementwise_binding(_spec)
    allowed_dtypes = _elementwise_allowed_dtypes(_spec)
    description = _elementwise_description(_spec)


@op_def
class Relu(_ElementwiseOp):
    _spec = ELEMENTWISE_BY_NAME["relu"]
    name = _spec.name
    schema = _elementwise_schema(_spec)
    frontend = _elementwise_binding(_spec)
    allowed_dtypes = _elementwise_allowed_dtypes(_spec)
    description = _elementwise_description(_spec)


@op_def
class NanToNum(_ElementwiseOp):
    _spec = ELEMENTWISE_BY_NAME["nan_to_num"]
    name = _spec.name
    schema = _elementwise_schema(_spec)
    frontend = _elementwise_binding(_spec)
    allowed_dtypes = _elementwise_allowed_dtypes(_spec)
    description = _elementwise_description(_spec)


@op_def
class ClampNanToNum(_ElementwiseOp):
    _spec = ELEMENTWISE_BY_NAME["clamp_nan_to_num"]
    name = _spec.name
    schema = _elementwise_schema(_spec)
    frontend = _elementwise_binding(_spec)
    allowed_dtypes = _elementwise_allowed_dtypes(_spec)
    description = _elementwise_description(_spec)


@op_def
class Silu(_ElementwiseOp):
    _spec = ELEMENTWISE_BY_NAME["silu"]
    name = _spec.name
    schema = _elementwise_schema(_spec)
    frontend = _elementwise_binding(_spec)
    allowed_dtypes = _elementwise_allowed_dtypes(_spec)
    description = _elementwise_description(_spec)


@op_def
class Pow(_ElementwiseOp):
    _spec = ELEMENTWISE_BY_NAME["pow"]
    name = _spec.name
    schema = _elementwise_schema(_spec)
    frontend = _elementwise_binding(_spec)
    allowed_dtypes = _elementwise_allowed_dtypes(_spec)
    description = _elementwise_description(_spec)


@op_def
class Gelu(_ElementwiseOp):
    _spec = ELEMENTWISE_BY_NAME["gelu"]
    name = _spec.name
    schema = _elementwise_schema(_spec)
    frontend = _elementwise_binding(_spec)
    allowed_dtypes = _elementwise_allowed_dtypes(_spec)
    description = _elementwise_description(_spec)


@op_def
class FastGelu(_ElementwiseOp):
    _spec = ELEMENTWISE_BY_NAME["fast_gelu"]
    name = _spec.name
    schema = _elementwise_schema(_spec)
    frontend = _elementwise_binding(_spec)
    allowed_dtypes = _elementwise_allowed_dtypes(_spec)
    description = _elementwise_description(_spec)


@op_def
class Softplus(_ElementwiseOp):
    _spec = ELEMENTWISE_BY_NAME["softplus"]
    name = _spec.name
    schema = _elementwise_schema(_spec)
    frontend = _elementwise_binding(_spec)
    allowed_dtypes = _elementwise_allowed_dtypes(_spec)
    description = _elementwise_description(_spec)


@op_def
class Elu(_ElementwiseOp):
    _spec = ELEMENTWISE_BY_NAME["elu"]
    name = _spec.name
    schema = _elementwise_schema(_spec)
    frontend = _elementwise_binding(_spec)
    allowed_dtypes = _elementwise_allowed_dtypes(_spec)
    description = _elementwise_description(_spec)


@op_def
class Softsign(_ElementwiseOp):
    _spec = ELEMENTWISE_BY_NAME["softsign"]
    name = _spec.name
    schema = _elementwise_schema(_spec)
    frontend = _elementwise_binding(_spec)
    allowed_dtypes = _elementwise_allowed_dtypes(_spec)
    description = _elementwise_description(_spec)


@op_def
class FloorDiv(_ElementwiseOp):
    _spec = ELEMENTWISE_BY_NAME["floor_div"]
    name = _spec.name
    schema = _elementwise_schema(_spec)
    frontend = _elementwise_binding(_spec)
    allowed_dtypes = _elementwise_allowed_dtypes(_spec)
    description = _elementwise_description(_spec)


@op_def
class Celu(_ElementwiseOp):
    _spec = ELEMENTWISE_BY_NAME["celu"]
    name = _spec.name
    schema = _elementwise_schema(_spec)
    frontend = _elementwise_binding(_spec)
    allowed_dtypes = _elementwise_allowed_dtypes(_spec)
    description = _elementwise_description(_spec)


@op_def
class Floor(_ElementwiseOp):
    _spec = ELEMENTWISE_BY_NAME["floor"]
    name = _spec.name
    schema = _elementwise_schema(_spec)
    frontend = _elementwise_binding(_spec)
    allowed_dtypes = _elementwise_allowed_dtypes(_spec)
    description = _elementwise_description(_spec)


@op_def
class Eq(_ElementwiseOp):
    _spec = ELEMENTWISE_BY_NAME["eq"]
    name = _spec.name
    schema = _elementwise_schema(_spec)
    frontend = _elementwise_binding(_spec)
    allowed_dtypes = _elementwise_allowed_dtypes(_spec)
    description = _elementwise_description(_spec)


@op_def
class Ge(_ElementwiseOp):
    _spec = ELEMENTWISE_BY_NAME["ge"]
    name = _spec.name
    schema = _elementwise_schema(_spec)
    frontend = _elementwise_binding(_spec)
    allowed_dtypes = _elementwise_allowed_dtypes(_spec)
    description = _elementwise_description(_spec)


@op_def
class Gt(_ElementwiseOp):
    _spec = ELEMENTWISE_BY_NAME["gt"]
    name = _spec.name
    schema = _elementwise_schema(_spec)
    frontend = _elementwise_binding(_spec)
    allowed_dtypes = _elementwise_allowed_dtypes(_spec)
    description = _elementwise_description(_spec)


@op_def
class Le(_ElementwiseOp):
    _spec = ELEMENTWISE_BY_NAME["le"]
    name = _spec.name
    schema = _elementwise_schema(_spec)
    frontend = _elementwise_binding(_spec)
    allowed_dtypes = _elementwise_allowed_dtypes(_spec)
    description = _elementwise_description(_spec)


@op_def
class Lt(_ElementwiseOp):
    _spec = ELEMENTWISE_BY_NAME["lt"]
    name = _spec.name
    schema = _elementwise_schema(_spec)
    frontend = _elementwise_binding(_spec)
    allowed_dtypes = _elementwise_allowed_dtypes(_spec)
    description = _elementwise_description(_spec)


@op_def
class Ne(_ElementwiseOp):
    _spec = ELEMENTWISE_BY_NAME["ne"]
    name = _spec.name
    schema = _elementwise_schema(_spec)
    frontend = _elementwise_binding(_spec)
    allowed_dtypes = _elementwise_allowed_dtypes(_spec)
    description = _elementwise_description(_spec)


@op_def
class Where(_ElementwiseOp):
    _spec = ELEMENTWISE_BY_NAME["where"]
    name = _spec.name
    schema = _elementwise_schema(_spec)
    frontend = _elementwise_binding(_spec)
    allowed_dtypes = _elementwise_allowed_dtypes(_spec)
    description = _elementwise_description(_spec)


@op_def
class FusedElementwise(OpDef):
    name = "fused_elementwise"
    schema = OpSchema(attrs=(AttrDef("sub_ops", "list[dict]", required=True),))
    infer_shape = infer_elementwise
    backend_kernels = {
        "cuda": KernelBinding("generated_fused_elementwise", "model", source_template="fused_elementwise_gpu"),
        "cpu": KernelBinding("generated_fused_elementwise", "model", source_template="fused_elementwise_cpu"),
        "rocm": KernelBinding("generated_fused_elementwise", "model", source_template="fused_elementwise_gpu"),
    }
    variadic_inputs = True
    allowed_dtypes = FUSED_ELEMENTWISE_RUNTIME_DTYPES
    description = "Internal fused elementwise subgraph generated into the model module."


def add(x0: Any, x1: Any) -> Tensor:
    return Add.forward(x0, x1)


def sub(x0: Any, x1: Any) -> Tensor:
    return Sub.forward(x0, x1)


def mul(x0: Any, x1: Any) -> Tensor:
    return Mul.forward(x0, x1)


def div(x0: Any, x1: Any) -> Tensor:
    return Div.forward(x0, x1)


def tanh(x0: Any) -> Tensor:
    return Tanh.forward(x0)


def cos(x0: Any) -> Tensor:
    return Cos.forward(x0)


def sin(x0: Any) -> Tensor:
    return Sin.forward(x0)


def sign(x0: Any) -> Tensor:
    return Sign.forward(x0)


def abs(x0: Any) -> Tensor:
    return Abs.forward(x0)


def log(x0: Any) -> Tensor:
    return Log.forward(x0)


def log1p(x0: Any) -> Tensor:
    return Log1p.forward(x0)


def exp(x0: Any) -> Tensor:
    return Exp.forward(x0)


def sqrt(x0: Any) -> Tensor:
    return Sqrt.forward(x0)


def max(x0: Any, x1: Any) -> Tensor:
    return Max.forward(x0, x1)


def min(x0: Any, x1: Any) -> Tensor:
    return Min.forward(x0, x1)


def sigmoid(x0: Any) -> Tensor:
    return Sigmoid.forward(x0)


def leaky_relu(x0: Any, negative_slope: float = 0.01) -> Tensor:
    return LeakyRelu.forward(x0, negative_slope=negative_slope)


def hardtanh(x0: Any, min_value: float = -1.0, max_value: float = 1.0) -> Tensor:
    return Hardtanh.forward(x0, min_value=min_value, max_value=max_value)


def relu(x0: Any) -> Tensor:
    return Relu.forward(x0)


def nan_to_num(
    x0: Any,
    nan_replacement: float = 0.0,
    posinf_replacement: float = 0.0,
    neginf_replacement: float = 0.0,
) -> Tensor:
    return NanToNum.forward(
        x0,
        nan_replacement=nan_replacement,
        posinf_replacement=posinf_replacement,
        neginf_replacement=neginf_replacement,
    )


def clamp_nan_to_num(
    x0: Any,
    clamp_min: float = -3.4028234663852886e38,
    clamp_max: float = 3.4028234663852886e38,
    nan_replacement: float = 0.0,
) -> Tensor:
    return ClampNanToNum.forward(
        x0,
        clamp_min=clamp_min,
        clamp_max=clamp_max,
        nan_replacement=nan_replacement,
    )


def silu(x0: Any) -> Tensor:
    return Silu.forward(x0)


def pow(x0: Any, x1: Any) -> Tensor:
    return Pow.forward(x0, x1)


def gelu(x0: Any, approximation: str = "tanh") -> Tensor:
    return Gelu.forward(x0, approximation=approximation)


def gelu_new(x0: Any) -> Tensor:
    return Gelu.forward(x0)


def fast_gelu(x0: Any) -> Tensor:
    return FastGelu.forward(x0)


def softplus(x0: Any) -> Tensor:
    return Softplus.forward(x0)


def elu(x0: Any, alpha: float = 1.0) -> Tensor:
    return Elu.forward(x0, alpha=alpha)


def softsign(x0: Any) -> Tensor:
    return Softsign.forward(x0)


def floor_div(x0: Any, x1: Any) -> Tensor:
    return FloorDiv.forward(x0, x1)


def celu(x0: Any, alpha: float = 1.0) -> Tensor:
    return Celu.forward(x0, alpha=alpha)


def floor(x0: Any) -> Tensor:
    return Floor.forward(x0)


def eq(x0: Any, x1: Any) -> Tensor:
    return Eq.forward(x0, x1)


def ge(x0: Any, x1: Any) -> Tensor:
    return Ge.forward(x0, x1)


def gt(x0: Any, x1: Any) -> Tensor:
    return Gt.forward(x0, x1)


def le(x0: Any, x1: Any) -> Tensor:
    return Le.forward(x0, x1)


def lt(x0: Any, x1: Any) -> Tensor:
    return Lt.forward(x0, x1)


def ne(x0: Any, x1: Any) -> Tensor:
    return Ne.forward(x0, x1)


__all__ = [
    "CAST_ELEMENTWISE_DTYPES",
    "ELEMENTWISE_BY_NAME",
    "ELEMENTWISE_OUTPUT_DTYPES",
    "ELEMENTWISE_SPECS",
    "EQ_ELEMENTWISE_DTYPES",
    "FLOAT_ELEMENTWISE_DTYPES",
    "FUSABLE_ELEMENTWISE_OPS",
    "FUSED_ELEMENTWISE_RUNTIME_DTYPES",
    "FUSION_ONLY_SPECS",
    "ElementwiseSpec",
    "FusedElementwise",
    "add",
    "sub",
    "mul",
    "div",
    "tanh",
    "cos",
    "sin",
    "sign",
    "abs",
    "log",
    "log1p",
    "exp",
    "sqrt",
    "max",
    "min",
    "sigmoid",
    "leaky_relu",
    "hardtanh",
    "relu",
    "nan_to_num",
    "clamp_nan_to_num",
    "silu",
    "pow",
    "gelu",
    "gelu_new",
    "fast_gelu",
    "softplus",
    "elu",
    "softsign",
    "floor_div",
    "celu",
    "floor",
    "eq",
    "ge",
    "gt",
    "le",
    "lt",
    "ne",
    "broadcast_shape",
    "elementwise_output_dtype",
    "infer_elementwise",
]
