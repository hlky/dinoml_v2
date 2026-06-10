from __future__ import annotations

import numpy as np
import pytest

import dinoml as dml
from dinoml.reference import reference_numpy
from dinoml.shapes import Dim


class SymbolicHelperModule(dml.Module):
    def forward(self, x):
        n = dml.ops.size(x, 0)
        m = dml.ops.int_add(dml.ops.int_mul(n, 2), 1)
        assert dml.ops.getitem((n, m), 0) == n
        assert dml.ops.tuple_construct(n, m) == (n, m)
        assert dml.ops.list_construct(n, m) == [n, m]
        return dml.ops.output(x, "out")


def test_symbolic_container_helpers_do_not_emit_compute_nodes():
    spec = dml.trace(
        SymbolicHelperModule(),
        inputs={"x": dml.TensorSpec([Dim("n", 2, 4), 3], "float32")},
        name="fresh_symbolic_helpers",
    )

    assert spec.ir["nodes"] == []
    assert spec.ir["outputs"][0]["shape_spec"][0]["name"] == "n"


def test_frontend_rejects_invalid_cast_dtype():
    class BadCast(dml.Module):
        def forward(self, x):
            return dml.ops.output(dml.ops.cast(x, "int32"), "out")

    with pytest.raises(ValueError, match="cast does not support dtype int32"):
        dml.trace(BadCast(), inputs={"x": dml.TensorSpec([2, 3], "float32")})


def test_clamp_requires_at_least_one_bound():
    class NoBoundClamp(dml.Module):
        def forward(self, x):
            return dml.ops.output(dml.ops.clamp(x), "out")

    with pytest.raises(ValueError, match="clamp requires at least one of min or max"):
        dml.trace(NoBoundClamp(), inputs={"x": dml.TensorSpec([2, 3], "float32")})


@pytest.mark.parametrize(
    ("kwargs", "expected_ops"),
    [
        ({"min": -0.5}, ["max"]),
        ({"max": 0.75}, ["min"]),
        ({"min": -0.5, "max": 0.75}, ["max", "min"]),
    ],
)
def test_clamp_traces_through_existing_elementwise_ops(kwargs, expected_ops):
    class ClampModule(dml.Module):
        def __init__(self, clamp_kwargs):
            self._clamp_kwargs = clamp_kwargs

        def forward(self, x):
            return dml.ops.output(dml.ops.clamp(x, **self._clamp_kwargs), "out")

    spec = dml.trace(
        ClampModule(kwargs),
        inputs={"x": dml.TensorSpec([2, 3], "float32")},
        name="clamp_frontend_lowering_contract",
    )

    assert [node["op"] for node in spec.ir["nodes"]] == expected_ops


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min": -0.5},
        {"max": 0.75},
        {"min": -0.5, "max": 0.75},
        {"min": 1.0, "max": 0.0},
    ],
)
def test_clamp_reference_matches_torch(kwargs):
    torch = pytest.importorskip("torch")

    class ClampModule(dml.Module):
        def __init__(self, clamp_kwargs):
            self._clamp_kwargs = clamp_kwargs

        def forward(self, x):
            return dml.ops.output(dml.ops.clamp(x, **self._clamp_kwargs), "out")

    values = np.array([[-2.0, -0.25, 0.5], [np.nan, 1.0, 2.0]], dtype=np.float32)
    spec = dml.trace(
        ClampModule(kwargs),
        inputs={"x": dml.TensorSpec([2, 3], "float32")},
        name="clamp_torch_parity_contract",
    )

    actual = reference_numpy(spec, {"x": values})["out"]
    expected = torch.clamp(torch.from_numpy(values.copy()), **kwargs).numpy()

    np.testing.assert_allclose(actual, expected, atol=0.0, rtol=0.0, equal_nan=True)


def test_clamp_preserves_nan_while_clamp_nan_to_num_replaces_it():
    torch = pytest.importorskip("torch")

    class ClampNanModule(dml.Module):
        def forward(self, x):
            return {
                "clamp": dml.ops.output(dml.ops.clamp(x, min=-0.5, max=0.25), "clamp"),
                "clamp_nan_to_num": dml.ops.output(
                    dml.ops.clamp_nan_to_num(x, clamp_min=-0.5, clamp_max=0.25, nan_replacement=0.0),
                    "clamp_nan_to_num",
                ),
            }

    values = np.array([[np.nan, -2.0, 0.5]], dtype=np.float32)
    spec = dml.trace(
        ClampNanModule(),
        inputs={"x": dml.TensorSpec([1, 3], "float32")},
        name="clamp_nan_behavior_contract",
    )

    outputs = reference_numpy(spec, {"x": values})
    expected_clamp = torch.clamp(torch.from_numpy(values.copy()), min=-0.5, max=0.25).numpy()
    expected_clamp_nan_to_num = torch.nan_to_num(torch.from_numpy(values.copy()), nan=0.0).clamp(-0.5, 0.25).numpy()

    np.testing.assert_allclose(outputs["clamp"], expected_clamp, atol=0.0, rtol=0.0, equal_nan=True)
    np.testing.assert_allclose(outputs["clamp_nan_to_num"], expected_clamp_nan_to_num, atol=0.0, rtol=0.0, equal_nan=True)
    assert np.isnan(outputs["clamp"][0, 0])
    assert outputs["clamp_nan_to_num"][0, 0] == np.float32(0.0)
