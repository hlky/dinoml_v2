from __future__ import annotations

import pytest

import dinoml as dml
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
