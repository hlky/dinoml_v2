from __future__ import annotations

import numpy as np

import dinoml as dml


IN_FEATURES = 8
OUT_FEATURES = 6
MAX_BATCH = 4
VALIDATION_BATCH = 3


class CudaLinear(dml.Module):
    def __init__(self):
        self.weight = dml.Parameter([IN_FEATURES, OUT_FEATURES], dtype="float32", name="weight")
        self.bias = dml.Parameter([OUT_FEATURES], dtype="float32", name="bias")

    def forward(self, x):
        y = dml.ops.gemm_rrr_bias(x, self.weight, self.bias)
        return dml.ops.output(y, "y")


def build_constants() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(2025)
    return {
        "weight": rng.standard_normal((IN_FEATURES, OUT_FEATURES)).astype(np.float32) * 0.25,
        "bias": rng.standard_normal((OUT_FEATURES,)).astype(np.float32) * 0.1,
    }


def build_spec() -> dml.ir.ModelSpec:
    batch = dml.Dim("batch", min=1, max=MAX_BATCH, typical=VALIDATION_BATCH, buckets=(1, VALIDATION_BATCH, MAX_BATCH))
    return dml.trace(
        CudaLinear(),
        inputs={"x": dml.TensorSpec([batch, IN_FEATURES], "float32")},
        constants=build_constants(),
        name="cuda_linear",
    )


def build_validation_inputs() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(2026)
    return {"x": rng.standard_normal((VALIDATION_BATCH, IN_FEATURES)).astype(np.float32)}


def numpy_reference(inputs: dict[str, np.ndarray], constants: dict[str, np.ndarray] | None = None) -> dict[str, np.ndarray]:
    constants = build_constants() if constants is None else constants
    x = inputs["x"].astype(np.float32)
    y = x @ constants["weight"].astype(np.float32) + constants["bias"].astype(np.float32)
    return {"y": y.astype(np.float32)}


def torch_reference(inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    import torch

    constants = build_constants()
    x = torch.from_numpy(inputs["x"])
    weight = torch.from_numpy(constants["weight"])
    bias = torch.from_numpy(constants["bias"])
    y = x @ weight + bias
    return {"y": y.numpy().astype(np.float32)}
