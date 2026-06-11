from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

import dinoml as dml


ATOL_BY_DTYPE = {"float16": 0.003, "float32": 1e-5, "bfloat16": 0.02}
RTOL_BY_DTYPE = {"float16": 0.002, "float32": 1e-5, "bfloat16": 0.02}
NORMALIZATION_SPECIALIZATION_DTYPES = ("float16", "float32", "bfloat16")

LAYER_NORM_CASES = (
    {"tag": "last_dim", "shape": [2, 5, 16], "normalized_shape": [16]},
    {"tag": "nd_suffix", "shape": [2, 3, 4, 8], "normalized_shape": [4, 8]},
)

LAYERNORM_SIGMOID_MUL_CASES = (
    {"tag": "last_dim_affine", "shape": [2, 5, 16], "normalized_shape": [16], "use_affine": True},
    {"tag": "nd_suffix_default_affine", "shape": [2, 3, 4, 8], "normalized_shape": [4, 8], "use_affine": False},
)

BATCH_LAYERNORM_SIGMOID_MUL_CASES = (
    {"tag": "rank3_affine", "shape": [3, 4, 16], "normalized_shape": [16], "use_affine": True},
    {"tag": "rank3_default_affine", "shape": [2, 5, 33], "normalized_shape": [33], "use_affine": False},
)

GROUP_LAYERNORM_CASES = (
    {
        "tag": "mixed_suffix_affine",
        "shapes": ([2, 5, 16], [2, 5, 4, 8]),
        "normalized_shapes": ([16], [4, 8]),
        "use_affine": True,
    },
    {
        "tag": "three_group_default_affine",
        "shapes": ([3, 4, 32], [3, 4, 8, 4], [3, 4, 2, 2, 4]),
        "normalized_shapes": ([32], [8, 4], [2, 2, 4]),
        "use_affine": False,
    },
)


class _LayernormSigmoidMulModule(dml.Module):
    def __init__(self, normalized_shape: list[int], use_affine: bool):
        self._normalized_shape = list(normalized_shape)
        self._use_affine = bool(use_affine)

    def forward(self, x, weight=None, bias=None):
        y = dml.ops.layernorm_sigmoid_mul(
            x,
            weight if self._use_affine else None,
            bias if self._use_affine else None,
            normalized_shape=self._normalized_shape,
            eps=1e-5,
        )
        return dml.ops.output(y, "y")


class _LayerNormModule(dml.Module):
    def __init__(self, normalized_shape: list[int]):
        self._normalized_shape = list(normalized_shape)

    def forward(self, x, weight, bias):
        y = dml.ops.layer_norm(x, weight, bias, eps=1e-5, normalized_shape=self._normalized_shape)
        return dml.ops.output(y, "y")


class _BatchLayernormSigmoidMulModule(dml.Module):
    def __init__(self, normalized_shape: list[int], use_affine: bool):
        self._normalized_shape = list(normalized_shape)
        self._use_affine = bool(use_affine)

    def forward(self, x, weight=None, bias=None):
        y = dml.ops.batch_layernorm_sigmoid_mul(
            x,
            weight if self._use_affine else None,
            bias if self._use_affine else None,
            normalized_shape=self._normalized_shape,
            eps=1e-5,
        )
        return dml.ops.output(y, "y")


class _GroupLayernormModule(dml.Module):
    def __init__(self, op_name: str, normalized_shapes: list[list[int]], use_affine: bool):
        self._op_name = op_name
        self._normalized_shapes = [list(shape) for shape in normalized_shapes]
        self._use_affine = bool(use_affine)

    def forward(self, **inputs):
        values = [inputs[f"x{index}"] for index in range(len(self._normalized_shapes))]
        if self._use_affine:
            weights = [inputs[f"weight{index}"] for index in range(len(self._normalized_shapes))]
            biases = [inputs[f"bias{index}"] for index in range(len(self._normalized_shapes))]
        else:
            weights = None
            biases = None
        op = getattr(dml.ops, self._op_name)
        outputs = op(
            values,
            weights=weights,
            biases=biases,
            normalized_shapes=self._normalized_shapes,
            eps=1e-5,
        )
        return tuple(dml.ops.output(value, f"y{index}") for index, value in enumerate(outputs))


def normalization_case_id(op_name: str, dtype: str, case: dict[str, object]) -> str:
    return f"{op_name}_{dtype}_{case['tag']}"


def artifact_stem(op_name: str, dtype: str, case: dict[str, object]) -> str:
    digest = hashlib.sha256(normalization_case_id(op_name, dtype, case).encode("utf-8")).hexdigest()[:10]
    return f"nsp_{digest}"


def cpu_artifact_path(op_name: str, dtype: str, case: dict[str, object]) -> Path:
    root = Path(__file__).resolve().parents[1] / ".pa" / "norm"
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{artifact_stem(op_name, dtype, case)}.dinoml"


def trace_single_output_spec(op_name: str, dtype: str, case: dict[str, object], *, name_prefix: str = ""):
    shape = list(case["shape"])
    normalized_shape = list(case["normalized_shape"])
    use_affine = bool(case.get("use_affine", True))
    if op_name == "layer_norm":
        model = _LayerNormModule(normalized_shape)
        inputs = {
            "x": dml.TensorSpec(shape, dtype),
            "weight": dml.TensorSpec(normalized_shape, dtype),
            "bias": dml.TensorSpec(normalized_shape, dtype),
        }
    elif op_name == "layernorm_sigmoid_mul":
        model = _LayernormSigmoidMulModule(normalized_shape, use_affine)
        inputs = {"x": dml.TensorSpec(shape, dtype)}
        if use_affine:
            inputs["weight"] = dml.TensorSpec(normalized_shape, dtype)
            inputs["bias"] = dml.TensorSpec(normalized_shape, dtype)
    elif op_name == "batch_layernorm_sigmoid_mul":
        model = _BatchLayernormSigmoidMulModule(normalized_shape, use_affine)
        inputs = {"x": dml.TensorSpec(shape, dtype)}
        if use_affine:
            affine_shape = [int(shape[0]), int(normalized_shape[0])]
            inputs["weight"] = dml.TensorSpec(affine_shape, dtype)
            inputs["bias"] = dml.TensorSpec(affine_shape, dtype)
    else:
        raise ValueError(f"Unsupported single-output op: {op_name}")
    return dml.trace(model, inputs=inputs, name=f"{name_prefix}{normalization_case_id(op_name, dtype, case)}")


def trace_group_spec(op_name: str, dtype: str, case: dict[str, object], *, name_prefix: str = ""):
    shapes = [list(shape) for shape in case["shapes"]]
    normalized_shapes = [list(shape) for shape in case["normalized_shapes"]]
    use_affine = bool(case["use_affine"])
    model = _GroupLayernormModule(op_name, normalized_shapes, use_affine)
    inputs: dict[str, dml.TensorSpec] = {}
    for index, shape in enumerate(shapes):
        inputs[f"x{index}"] = dml.TensorSpec(shape, dtype)
        if use_affine:
            inputs[f"weight{index}"] = dml.TensorSpec(normalized_shapes[index], dtype)
            inputs[f"bias{index}"] = dml.TensorSpec(normalized_shapes[index], dtype)
    return dml.trace(model, inputs=inputs, name=f"{name_prefix}{normalization_case_id(op_name, dtype, case)}")


def random_single_output_inputs(op_name: str, dtype: str, case: dict[str, object]) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(7)
    inputs: dict[str, np.ndarray] = {
        "x": rng.standard_normal(list(case["shape"]), dtype=np.float32).astype(np.float32),
    }
    if bool(case.get("use_affine", True)):
        if op_name == "batch_layernorm_sigmoid_mul":
            inputs["weight"] = rng.standard_normal([int(case["shape"][0]), int(case["normalized_shape"][0])], dtype=np.float32).astype(np.float32)
            inputs["bias"] = rng.standard_normal([int(case["shape"][0]), int(case["normalized_shape"][0])], dtype=np.float32).astype(np.float32)
        else:
            inputs["weight"] = rng.standard_normal(list(case["normalized_shape"]), dtype=np.float32).astype(np.float32)
            inputs["bias"] = rng.standard_normal(list(case["normalized_shape"]), dtype=np.float32).astype(np.float32)
    return _coerce_inputs_dtype(inputs, dtype)


def random_group_inputs(dtype: str, case: dict[str, object]) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(7)
    inputs: dict[str, np.ndarray] = {}
    for index, shape in enumerate(case["shapes"]):
        inputs[f"x{index}"] = rng.standard_normal(list(shape), dtype=np.float32).astype(np.float32)
        if bool(case["use_affine"]):
            affine_shape = list(case["normalized_shapes"][index])
            inputs[f"weight{index}"] = rng.standard_normal(affine_shape, dtype=np.float32).astype(np.float32)
            inputs[f"bias{index}"] = rng.standard_normal(affine_shape, dtype=np.float32).astype(np.float32)
    return _coerce_inputs_dtype(inputs, dtype)


def torch_single_output_oracle(
    torch,
    op_name: str,
    inputs: dict[str, np.ndarray],
    case: dict[str, object],
    *,
    dtype: str,
    device: str,
) -> np.ndarray:
    torch_dtype = _torch_dtype(torch, dtype)
    x = torch.from_numpy(inputs["x"]).to(device=device, dtype=torch_dtype)
    if op_name == "layer_norm":
        weight = torch.from_numpy(inputs["weight"]).to(device=device, dtype=torch_dtype)
        bias = torch.from_numpy(inputs["bias"]).to(device=device, dtype=torch_dtype)
        return torch.nn.functional.layer_norm(
            x,
            list(case["normalized_shape"]),
            weight,
            bias,
            eps=1e-5,
        ).float().cpu().numpy()
    if op_name == "layernorm_sigmoid_mul":
        weight = None if not bool(case.get("use_affine", True)) else torch.from_numpy(inputs["weight"]).to(device=device, dtype=torch_dtype)
        bias = None if not bool(case.get("use_affine", True)) else torch.from_numpy(inputs["bias"]).to(device=device, dtype=torch_dtype)
        normalized = torch.nn.functional.layer_norm(
            x,
            list(case["normalized_shape"]),
            weight,
            bias,
            eps=1e-5,
        )
        return (x * torch.sigmoid(normalized)).float().cpu().numpy()
    if op_name == "batch_layernorm_sigmoid_mul":
        weight = None if not bool(case.get("use_affine", True)) else torch.from_numpy(inputs["weight"]).to(device=device, dtype=torch_dtype)
        bias = None if not bool(case.get("use_affine", True)) else torch.from_numpy(inputs["bias"]).to(device=device, dtype=torch_dtype)
        outputs = []
        for batch_index in range(int(x.shape[0])):
            normalized = torch.nn.functional.layer_norm(
                x[batch_index],
                list(case["normalized_shape"]),
                None if weight is None else weight[batch_index],
                None if bias is None else bias[batch_index],
                eps=1e-5,
            )
            outputs.append(x[batch_index] * torch.sigmoid(normalized))
        return torch.stack(outputs, dim=0).float().cpu().numpy()
    raise ValueError(f"Unsupported single-output op: {op_name}")


def torch_group_oracle(
    torch,
    op_name: str,
    inputs: dict[str, np.ndarray],
    case: dict[str, object],
    *,
    dtype: str,
    device: str,
) -> dict[str, np.ndarray]:
    torch_dtype = _torch_dtype(torch, dtype)
    expected: dict[str, np.ndarray] = {}
    for index, normalized_shape in enumerate(case["normalized_shapes"]):
        x = torch.from_numpy(inputs[f"x{index}"]).to(device=device, dtype=torch_dtype)
        weight = None
        bias = None
        if bool(case["use_affine"]):
            weight = torch.from_numpy(inputs[f"weight{index}"]).to(device=device, dtype=torch_dtype)
            bias = torch.from_numpy(inputs[f"bias{index}"]).to(device=device, dtype=torch_dtype)
        normalized = torch.nn.functional.layer_norm(x, list(normalized_shape), weight, bias, eps=1e-5)
        if op_name == "group_layernorm_sigmoid_mul":
            normalized = x * torch.sigmoid(normalized)
        expected[f"y{index}"] = normalized.float().cpu().numpy()
    return expected


def output_names(case: dict[str, object]) -> list[str]:
    return [f"y{index}" for index in range(len(case["shapes"]))]


def _coerce_inputs_dtype(inputs: dict[str, np.ndarray], dtype: str) -> dict[str, np.ndarray]:
    if dtype == "float16":
        return {name: value.astype(np.float16) for name, value in inputs.items()}
    if dtype == "bfloat16":
        return inputs
    return inputs


def _torch_dtype(torch, dtype: str):
    return {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}[dtype]
