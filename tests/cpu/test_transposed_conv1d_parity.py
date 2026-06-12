from __future__ import annotations

import numpy as np
import pytest

import dinoml as dml
from dinoml.reference import reference_numpy
from dinoml.runtime import load


_DTYPES = ("float16", "float32", "bfloat16")
_ATOL_BY_DTYPE = {"float16": 0.003, "float32": 1e-5, "bfloat16": 0.02}
_RTOL_BY_DTYPE = {"float16": 0.002, "float32": 1e-5, "bfloat16": 0.02}


class _TransposeConv1dParityModule(dml.Module):
    def forward(self, x, weight):
        y = dml.ops.transposed_conv1d(x, weight, stride=2, padding=1, output_padding=1, dilation=1)
        return dml.ops.output(y, "y")


def _trace_spec(dtype: str):
    spec = dml.trace(
        _TransposeConv1dParityModule(),
        inputs={
            "x": dml.TensorSpec([1, 3, 4], dtype),
            "weight": dml.TensorSpec([3, 5, 3], dtype),
        },
        name=f"transposed_conv1d_{dtype}_parity",
    )
    return spec


def _random_inputs(dtype: str) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(7)
    inputs = {
        "x": rng.standard_normal([1, 3, 4], dtype=np.float32).astype(np.float32),
        "weight": rng.standard_normal([3, 5, 3], dtype=np.float32).astype(np.float32),
    }
    if dtype == "float16":
        return {name: value.astype(np.float16) for name, value in inputs.items()}
    return inputs


def _torch_oracle(torch, inputs: dict[str, np.ndarray], *, dtype: str) -> np.ndarray:
    torch_dtype = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}[dtype]
    x = torch.from_numpy(inputs["x"]).to(dtype=torch_dtype)
    weight = torch.from_numpy(inputs["weight"]).to(dtype=torch_dtype)
    result = torch.nn.functional.conv_transpose1d(
        x,
        weight,
        bias=None,
        stride=2,
        padding=1,
        output_padding=1,
        dilation=1,
        groups=1,
    )
    return result.float().cpu().numpy()


@pytest.mark.parametrize("dtype", _DTYPES)
def test_transposed_conv1d_reference_matches_torch_cpu(dtype: str):
    torch = pytest.importorskip("torch")
    spec = _trace_spec(dtype)
    inputs = _random_inputs(dtype)

    actual = reference_numpy(spec, inputs)["y"]
    expected = _torch_oracle(torch, inputs, dtype=dtype)
    np.testing.assert_allclose(actual, expected, atol=_ATOL_BY_DTYPE[dtype], rtol=_RTOL_BY_DTYPE[dtype])


@pytest.mark.parametrize("dtype", _DTYPES)
def test_cpu_transposed_conv1d_parity_matches_torch(dtype: str, tmp_path):
    torch = pytest.importorskip("torch")
    spec = _trace_spec(dtype)
    inputs = _random_inputs(dtype)

    artifact = dml.compile(spec, dml.Target("cpu"), tmp_path / f"transposed_conv1d_{dtype}_cpu.dinoml")
    module = load(artifact.path)
    session = module.create_session()
    try:
        actual = session.run_numpy(inputs)["y"]
    finally:
        session.close()
        module.close()

    expected = _torch_oracle(torch, inputs, dtype=dtype)
    np.testing.assert_allclose(actual, expected, atol=_ATOL_BY_DTYPE[dtype], rtol=_RTOL_BY_DTYPE[dtype])
