from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Any, Mapping, Sequence

from dinoml.kernels.families.gemm import GEMM_SUPPORTED_DTYPES


@dataclass(frozen=True)
class DualGemmEpilogue:
    name: str
    activation: str
    inputs: tuple[str, ...] = ()
    launch_abi: str = "dinoml_cutlass_dual_gemm_v1"

    @property
    def has_bias(self) -> bool:
        return self.inputs == ("bias0", "bias1")

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "activation": self.activation,
            "inputs": list(self.inputs),
            "launch_abi": self.launch_abi,
        }


@dataclass(frozen=True)
class DualGemmOpSpec:
    name: str
    base_layout: str
    layouts: Mapping[str, str]
    epilogue: DualGemmEpilogue
    allow_rhs_broadcast: bool = True

    @property
    def input_count(self) -> int:
        return 3 + len(self.epilogue.inputs)

    def validate_shapes(self, shapes: Sequence[Sequence[int]]) -> list[int]:
        if len(shapes) != self.input_count:
            raise ValueError(f"{self.name} expects exactly {self.input_count} inputs")
        if self.base_layout != "rcr":
            raise ValueError(f"Unsupported DualGEMM layout: {self.base_layout}")

        a_shape, b0_shape, b1_shape = shapes[:3]
        if len(a_shape) < 2 or len(b0_shape) != 2 or len(b1_shape) != 2:
            raise ValueError(f"{self.name} expects A[...,K] and rank-2 B0/B1 tensors")
        if any(int(dim) <= 0 for shape in (a_shape, b0_shape, b1_shape) for dim in shape):
            raise ValueError(f"{self.name} dimensions must be positive")

        k = int(a_shape[-1])
        if k != int(b0_shape[1]):
            raise ValueError(f"{self.name} expected B0[N,K] with K={k}, got {list(b0_shape)}")
        if k != int(b1_shape[1]):
            raise ValueError(f"{self.name} expected B1[N,K] with K={k}, got {list(b1_shape)}")

        n = int(b0_shape[0])
        b1_n = int(b1_shape[0])
        if b1_n != n:
            if not self.allow_rhs_broadcast or b1_n != 1:
                raise ValueError(
                    f"{self.name} expected B1[N,K] or B1[1,K] matching N={n}, got {list(b1_shape)}"
                )

        if self.epilogue.has_bias:
            bias0_shape, bias1_shape = shapes[3], shapes[4]
            _validate_bias_shape(self.name, "bias0", bias0_shape, n)
            _validate_bias_shape(self.name, "bias1", bias1_shape, b1_n)

        return [*(int(dim) for dim in a_shape[:-1]), n]

    def output_shape_spec(self, shape_specs: Sequence[Sequence[Any]]) -> list[Any]:
        if self.base_layout != "rcr":
            raise ValueError(f"Unsupported DualGEMM layout: {self.base_layout}")
        return [*shape_specs[0][:-1], shape_specs[1][0]]

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "base_layout": self.base_layout,
            "layouts": dict(self.layouts),
            "input_count": self.input_count,
            "epilogue": self.epilogue.to_json(),
            "allow_rhs_broadcast": self.allow_rhs_broadcast,
        }


DUAL_GEMM_RCR_EPILOGUES: dict[str, DualGemmEpilogue] = {
    "relu": DualGemmEpilogue(name="left_relu_and_mul", activation="relu"),
    "gelu": DualGemmEpilogue(name="left_gelu_and_mul", activation="gelu"),
    "fast_gelu": DualGemmEpilogue(name="left_fast_gelu_and_mul", activation="fast_gelu"),
    "quick_gelu": DualGemmEpilogue(name="left_quick_gelu_and_mul", activation="quick_gelu"),
    "sigmoid": DualGemmEpilogue(name="left_sigmoid_and_mul", activation="sigmoid"),
    "tanh": DualGemmEpilogue(name="left_tanh_and_mul", activation="tanh"),
    "silu": DualGemmEpilogue(name="left_silu_and_mul", activation="silu"),
    "hardswish": DualGemmEpilogue(name="left_hardswish_and_mul", activation="hardswish"),
    "elup1": DualGemmEpilogue(name="left_elup1_and_mul", activation="elup1"),
}

DUAL_GEMM_RCR_BIAS_EPILOGUES: dict[str, DualGemmEpilogue] = {
    "relu": DualGemmEpilogue(
        name="left_bias_relu_and_mul",
        activation="relu",
        inputs=("bias0", "bias1"),
        launch_abi="dinoml_cutlass_dual_gemm_bias_v1",
    ),
    "gelu": DualGemmEpilogue(
        name="left_bias_gelu_and_mul",
        activation="gelu",
        inputs=("bias0", "bias1"),
        launch_abi="dinoml_cutlass_dual_gemm_bias_v1",
    ),
    "fast_gelu": DualGemmEpilogue(
        name="left_bias_fast_gelu_and_mul",
        activation="fast_gelu",
        inputs=("bias0", "bias1"),
        launch_abi="dinoml_cutlass_dual_gemm_bias_v1",
    ),
    "quick_gelu": DualGemmEpilogue(
        name="left_bias_quick_gelu_and_mul",
        activation="quick_gelu",
        inputs=("bias0", "bias1"),
        launch_abi="dinoml_cutlass_dual_gemm_bias_v1",
    ),
    "sigmoid": DualGemmEpilogue(
        name="left_bias_sigmoid_and_mul",
        activation="sigmoid",
        inputs=("bias0", "bias1"),
        launch_abi="dinoml_cutlass_dual_gemm_bias_v1",
    ),
    "tanh": DualGemmEpilogue(
        name="left_bias_tanh_and_mul",
        activation="tanh",
        inputs=("bias0", "bias1"),
        launch_abi="dinoml_cutlass_dual_gemm_bias_v1",
    ),
    "swish": DualGemmEpilogue(
        name="left_bias_swish_and_mul",
        activation="swish",
        inputs=("bias0", "bias1"),
        launch_abi="dinoml_cutlass_dual_gemm_bias_v1",
    ),
    "hardswish": DualGemmEpilogue(
        name="left_bias_hardswish_and_mul",
        activation="hardswish",
        inputs=("bias0", "bias1"),
        launch_abi="dinoml_cutlass_dual_gemm_bias_v1",
    ),
    "elup1": DualGemmEpilogue(
        name="left_bias_elup1_and_mul",
        activation="elup1",
        inputs=("bias0", "bias1"),
        launch_abi="dinoml_cutlass_dual_gemm_bias_v1",
    ),
}


def _dual_gemm_op_spec(name: str, epilogue: DualGemmEpilogue) -> DualGemmOpSpec:
    return DualGemmOpSpec(
        name=name,
        base_layout="rcr",
        layouts={"a": "row", "b0": "column", "b1": "column", "c": "row"},
        epilogue=epilogue,
    )


DUAL_GEMM_OP_SPECS: dict[str, DualGemmOpSpec] = {
    **{
        f"dual_gemm_rcr_{activation}": _dual_gemm_op_spec(f"dual_gemm_rcr_{activation}", epilogue)
        for activation, epilogue in DUAL_GEMM_RCR_EPILOGUES.items()
    },
    **{
        f"dual_gemm_rcr_bias_{activation}": _dual_gemm_op_spec(f"dual_gemm_rcr_bias_{activation}", epilogue)
        for activation, epilogue in DUAL_GEMM_RCR_BIAS_EPILOGUES.items()
    },
}
DUAL_GEMM_OPS = tuple(DUAL_GEMM_OP_SPECS)


def dual_gemm_op_spec(op_name: str) -> DualGemmOpSpec:
    try:
        return DUAL_GEMM_OP_SPECS[op_name]
    except KeyError as exc:
        supported = ", ".join(DUAL_GEMM_OPS)
        raise ValueError(f"Unsupported DualGEMM op {op_name!r}; supported ops: {supported}") from exc


def dual_gemm_problem(op_name: str, shapes: Sequence[Sequence[int]]) -> tuple[int, int, int, tuple[int, ...]]:
    spec = dual_gemm_op_spec(op_name)
    output = spec.validate_shapes(shapes)
    m = int(prod(int(dim) for dim in shapes[0][:-1]))
    n = int(output[-1])
    k = int(shapes[0][-1])
    return m, n, k, tuple(int(dim) for dim in output)


def _validate_bias_shape(op_name: str, bias_name: str, bias_shape: Sequence[int], n: int) -> None:
    if len(bias_shape) == 1 and int(bias_shape[0]) == n:
        return
    if len(bias_shape) == 2 and int(bias_shape[0]) == 1 and int(bias_shape[1]) == n:
        return
    raise ValueError(f"{op_name} expected {bias_name} shape [N] or [1, N] with N={n}, got {list(bias_shape)}")


__all__ = [
    "DUAL_GEMM_RCR_BIAS_EPILOGUES",
    "DUAL_GEMM_RCR_EPILOGUES",
    "DUAL_GEMM_OPS",
    "DUAL_GEMM_OP_SPECS",
    "DualGemmEpilogue",
    "DualGemmOpSpec",
    "GEMM_SUPPORTED_DTYPES",
    "dual_gemm_op_spec",
    "dual_gemm_problem",
]
