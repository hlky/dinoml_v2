from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Any, Mapping, Sequence

from dinoml.ir import normalize_dtype


GEMM_SUPPORTED_DTYPES = ("float16", "float32", "bfloat16")


@dataclass(frozen=True)
class GemmEpilogue:
    name: str
    cutlass_functor: str
    inputs: tuple[str, ...] = ()
    activation: str | None = None
    pre_residual_activation: str | None = None
    bias_axis: str | None = None
    accumulator_dtype: str = "float32"
    output_dtype: str = "same"
    launch_abi: str = "dinoml_cutlass_gemm_v1"

    @property
    def has_bias(self) -> bool:
        return "bias" in self.inputs

    @property
    def residual_inputs(self) -> tuple[str, ...]:
        return tuple(name for name in self.inputs if name.startswith("d"))

    def to_json(self) -> dict[str, Any]:
        payload = {
            "name": self.name,
            "cutlass_functor": self.cutlass_functor,
            "inputs": list(self.inputs),
            "activation": self.activation,
            "bias_axis": self.bias_axis,
            "accumulator_dtype": self.accumulator_dtype,
            "output_dtype": self.output_dtype,
            "launch_abi": self.launch_abi,
        }
        if self.pre_residual_activation is not None:
            payload["pre_residual_activation"] = self.pre_residual_activation
        return payload


@dataclass(frozen=True)
class GemmOpSpec:
    name: str
    base_layout: str
    layouts: Mapping[str, str]
    epilogue: GemmEpilogue

    @property
    def input_count(self) -> int:
        return 2 + len(self.epilogue.inputs)

    def n_dim_from_rhs(self, b_shape: Sequence[int]) -> int:
        if self.base_layout == "rrr":
            return int(b_shape[1])
        if self.base_layout == "rcr":
            return int(b_shape[0])
        raise ValueError(f"Unsupported GEMM layout: {self.base_layout}")

    def validate_shapes(self, shapes: Sequence[Sequence[int]]) -> list[int]:
        if len(shapes) != self.input_count:
            raise ValueError(f"{self.name} expects exactly {self.input_count} inputs")
        a_shape, b_shape = shapes[0], shapes[1]
        if len(a_shape) < 2 or len(b_shape) != 2:
            raise ValueError(f"{self.name} expects A[...,K] and rank-2 B")
        if any(int(dim) <= 0 for shape in (a_shape, b_shape) for dim in shape):
            raise ValueError(f"{self.name} dimensions must be positive")
        k = int(a_shape[-1])
        if self.base_layout == "rrr":
            if k != int(b_shape[0]):
                raise ValueError(f"{self.name} expected A[...,K] and B[K,N], got {list(a_shape)} and {list(b_shape)}")
            n = int(b_shape[1])
        elif self.base_layout == "rcr":
            if k != int(b_shape[1]):
                raise ValueError(f"{self.name} expected A[...,K] and B[N,K], got {list(a_shape)} and {list(b_shape)}")
            n = int(b_shape[0])
        else:
            raise ValueError(f"Unsupported GEMM layout: {self.base_layout}")
        output_shape = [*(int(dim) for dim in a_shape[:-1]), n]
        for input_name, shape in zip(self.epilogue.inputs, shapes[2:]):
            if input_name == "bias":
                _validate_bias_shape(self.name, shape, n)
            elif input_name.startswith("d"):
                _validate_residual_shape(self.name, input_name, shape, output_shape)
        return output_shape

    def output_shape_spec(self, shape_specs: Sequence[Sequence[Any]]) -> list[Any]:
        if self.base_layout == "rrr":
            return [*shape_specs[0][:-1], shape_specs[1][1]]
        if self.base_layout == "rcr":
            return [*shape_specs[0][:-1], shape_specs[1][0]]
        raise ValueError(f"Unsupported GEMM layout: {self.base_layout}")

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "base_layout": self.base_layout,
            "layouts": dict(self.layouts),
            "input_count": self.input_count,
            "epilogue": self.epilogue.to_json(),
        }


LINEAR_COMBINATION_EPILOGUE = GemmEpilogue(
    name="linear_combination",
    cutlass_functor="cutlass::epilogue::thread::LinearCombination",
)
BIAS_EPILOGUE = GemmEpilogue(
    name="bias",
    cutlass_functor="cutlass::epilogue::thread::LinearCombination",
    inputs=("bias",),
    bias_axis="n",
    launch_abi="dinoml_cutlass_gemm_bias_v1",
)
BIAS_RELU_EPILOGUE = GemmEpilogue(
    name="bias_relu",
    cutlass_functor="cutlass::epilogue::thread::LinearCombinationRelu",
    inputs=("bias",),
    activation="relu",
    bias_axis="n",
    launch_abi="dinoml_cutlass_gemm_bias_v1",
)

BIAS_ACTIVATION_EPILOGUES: dict[str, GemmEpilogue] = {
    "gelu": GemmEpilogue(
        name="bias_gelu",
        cutlass_functor="cutlass::epilogue::thread::LinearCombinationGELU",
        inputs=("bias",),
        activation="gelu",
        bias_axis="n",
        launch_abi="dinoml_cutlass_gemm_bias_v1",
    ),
    "fast_gelu": GemmEpilogue(
        name="bias_fast_gelu",
        cutlass_functor="cutlass::epilogue::thread::LinearCombinationFastGELU",
        inputs=("bias",),
        activation="fast_gelu",
        bias_axis="n",
        launch_abi="dinoml_cutlass_gemm_bias_v1",
    ),
    "quick_gelu": GemmEpilogue(
        name="bias_quick_gelu",
        cutlass_functor="cutlass::epilogue::thread::LinearCombinationQuickGELU",
        inputs=("bias",),
        activation="quick_gelu",
        bias_axis="n",
        launch_abi="dinoml_cutlass_gemm_bias_v1",
    ),
    "sigmoid": GemmEpilogue(
        name="bias_sigmoid",
        cutlass_functor="cutlass::epilogue::thread::LinearCombinationSigmoid",
        inputs=("bias",),
        activation="sigmoid",
        bias_axis="n",
        launch_abi="dinoml_cutlass_gemm_bias_v1",
    ),
    "tanh": GemmEpilogue(
        name="bias_tanh",
        cutlass_functor="cutlass::epilogue::thread::LinearCombinationTanh",
        inputs=("bias",),
        activation="tanh",
        bias_axis="n",
        launch_abi="dinoml_cutlass_gemm_bias_v1",
    ),
    "swish": GemmEpilogue(
        name="bias_swish",
        cutlass_functor="cutlass::epilogue::thread::LinearCombinationSilu",
        inputs=("bias",),
        activation="swish",
        bias_axis="n",
        launch_abi="dinoml_cutlass_gemm_bias_v1",
    ),
    "hardswish": GemmEpilogue(
        name="bias_hardswish",
        cutlass_functor="cutlass::epilogue::thread::LinearCombinationHardSwish",
        inputs=("bias",),
        activation="hardswish",
        bias_axis="n",
        launch_abi="dinoml_cutlass_gemm_bias_v1",
    ),
    "elup1": GemmEpilogue(
        name="bias_elup1",
        cutlass_functor="cutlass::epilogue::thread::LinearCombinationELUp1",
        inputs=("bias",),
        activation="elup1",
        bias_axis="n",
        launch_abi="dinoml_cutlass_gemm_bias_v1",
    ),
}

BIAS_RESIDUAL_EPILOGUES: dict[str, GemmEpilogue] = {
    "add": GemmEpilogue(
        name="bias_add",
        cutlass_functor="dinoml::cutlass_epilogue::BiasAdd",
        inputs=("bias", "d0"),
        launch_abi="dinoml_cutlass_gemm_bias_residual_v1",
    ),
    "add_add": GemmEpilogue(
        name="bias_add_add",
        cutlass_functor="dinoml::cutlass_epilogue::BiasAddAdd",
        inputs=("bias", "d0", "d1"),
        launch_abi="dinoml_cutlass_gemm_bias_residual2_v1",
    ),
    "mul": GemmEpilogue(
        name="bias_mul",
        cutlass_functor="dinoml::cutlass_epilogue::BiasMul",
        inputs=("bias", "d0"),
        launch_abi="dinoml_cutlass_gemm_bias_residual_v1",
    ),
    "mul_add": GemmEpilogue(
        name="bias_mul_add",
        cutlass_functor="dinoml::cutlass_epilogue::BiasMulAdd",
        inputs=("bias", "d0", "d1"),
        launch_abi="dinoml_cutlass_gemm_bias_residual2_v1",
    ),
}

BIAS_RESIDUAL_RELU_EPILOGUES: dict[str, GemmEpilogue] = {
    "add_relu": GemmEpilogue(
        name="bias_add_relu",
        cutlass_functor="dinoml::cutlass_epilogue::BiasAddRelu",
        inputs=("bias", "d0"),
        activation="relu",
        launch_abi="dinoml_cutlass_gemm_bias_residual_v1",
    ),
    "add_add_relu": GemmEpilogue(
        name="bias_add_add_relu",
        cutlass_functor="dinoml::cutlass_epilogue::BiasAddAddRelu",
        inputs=("bias", "d0", "d1"),
        activation="relu",
        launch_abi="dinoml_cutlass_gemm_bias_residual2_v1",
    ),
}

BIAS_RESIDUAL_COMPOUND_EPILOGUES: dict[str, GemmEpilogue] = {
    "sigmoid_mul": GemmEpilogue(
        name="bias_sigmoid_mul",
        cutlass_functor="dinoml::cutlass_epilogue::BiasSigmoidMul",
        inputs=("bias", "d0"),
        pre_residual_activation="sigmoid",
        launch_abi="dinoml_cutlass_gemm_bias_residual_v1",
    ),
    "sigmoid_mul_tanh": GemmEpilogue(
        name="bias_sigmoid_mul_tanh",
        cutlass_functor="dinoml::cutlass_epilogue::BiasSigmoidMulTanh",
        inputs=("bias", "d0"),
        activation="tanh",
        pre_residual_activation="sigmoid",
        launch_abi="dinoml_cutlass_gemm_bias_residual_v1",
    ),
    "mul_tanh": GemmEpilogue(
        name="bias_mul_tanh",
        cutlass_functor="dinoml::cutlass_epilogue::BiasMulTanh",
        inputs=("bias", "d0"),
        activation="tanh",
        launch_abi="dinoml_cutlass_gemm_bias_residual_v1",
    ),
}


def _gemm_op_spec(name: str, base_layout: str, epilogue: GemmEpilogue) -> GemmOpSpec:
    return GemmOpSpec(
        name=name,
        base_layout=base_layout,
        layouts={"a": "row", "b": "row" if base_layout == "rrr" else "column", "c": "row"},
        epilogue=epilogue,
    )


GEMM_OP_SPECS: dict[str, GemmOpSpec] = {
    "gemm_rcr": _gemm_op_spec("gemm_rcr", "rcr", LINEAR_COMBINATION_EPILOGUE),
    "gemm_rrr": _gemm_op_spec("gemm_rrr", "rrr", LINEAR_COMBINATION_EPILOGUE),
    "gemm_rcr_bias": _gemm_op_spec("gemm_rcr_bias", "rcr", BIAS_EPILOGUE),
    "gemm_rrr_bias": _gemm_op_spec("gemm_rrr_bias", "rrr", BIAS_EPILOGUE),
    "gemm_rcr_bias_relu": _gemm_op_spec("gemm_rcr_bias_relu", "rcr", BIAS_RELU_EPILOGUE),
    "gemm_rrr_bias_relu": _gemm_op_spec("gemm_rrr_bias_relu", "rrr", BIAS_RELU_EPILOGUE),
    **{
        f"gemm_{layout}_bias_{activation}": _gemm_op_spec(f"gemm_{layout}_bias_{activation}", layout, epilogue)
        for activation, epilogue in BIAS_ACTIVATION_EPILOGUES.items()
        for layout in ("rcr", "rrr")
        if activation != "quick_gelu" or layout == "rcr"
    },
    **{
        f"gemm_{layout}_bias_{name}": _gemm_op_spec(f"gemm_{layout}_bias_{name}", layout, epilogue)
        for name, epilogue in BIAS_RESIDUAL_EPILOGUES.items()
        for layout in ("rcr", "rrr")
    },
    **{
        f"gemm_{layout}_bias_{name}": _gemm_op_spec(f"gemm_{layout}_bias_{name}", layout, epilogue)
        for name, epilogue in BIAS_RESIDUAL_RELU_EPILOGUES.items()
        for layout in ("rcr", "rrr")
    },
    **{
        f"gemm_{layout}_bias_{name}": _gemm_op_spec(f"gemm_{layout}_bias_{name}", layout, epilogue)
        for name, epilogue in BIAS_RESIDUAL_COMPOUND_EPILOGUES.items()
        for layout in ("rcr", "rrr")
    },
}
GEMM_OPS = tuple(GEMM_OP_SPECS)


def gemm_op_spec(op_name: str) -> GemmOpSpec:
    try:
        return GEMM_OP_SPECS[op_name]
    except KeyError as exc:
        supported = ", ".join(GEMM_OPS)
        raise ValueError(f"Unsupported GEMM op {op_name!r}; supported ops: {supported}") from exc


def gemm_problem(op_name: str, shapes: Sequence[Sequence[int]]) -> tuple[int, int, int, tuple[int, ...]]:
    spec = gemm_op_spec(op_name)
    output = spec.validate_shapes(shapes)
    m = int(prod(int(dim) for dim in shapes[0][:-1]))
    n = int(output[-1])
    k = int(shapes[0][-1])
    return m, n, k, tuple(int(dim) for dim in output)


def _validate_bias_shape(op_name: str, bias_shape: Sequence[int], n: int) -> None:
    if len(bias_shape) == 1 and int(bias_shape[0]) == n:
        return
    if len(bias_shape) == 2 and int(bias_shape[0]) == 1 and int(bias_shape[1]) == n:
        return
    raise ValueError(f"{op_name} expected bias shape [N] or [1, N] with N={n}, got {list(bias_shape)}")


def _validate_residual_shape(op_name: str, input_name: str, shape: Sequence[int], output_shape: Sequence[int]) -> None:
    if tuple(int(dim) for dim in shape) == tuple(int(dim) for dim in output_shape):
        return
    raise ValueError(f"{op_name} expected {input_name} shape {list(output_shape)}, got {list(shape)}")


def normalize_gemm_dtype(dtype: str) -> str:
    normalized = normalize_dtype(dtype)
    if normalized not in GEMM_SUPPORTED_DTYPES:
        supported = ", ".join(GEMM_SUPPORTED_DTYPES)
        raise ValueError(f"Unsupported GEMM dtype {dtype!r}; supported dtypes: {supported}")
    return normalized
