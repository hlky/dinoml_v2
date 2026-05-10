import pytest

import dinoml as dml


class StaticSymbolicHelpers(dml.Module):
    def forward(self, x):
        assert x.builder.nodes == []
        assert x.builder.views == []
        shape = dml.ops.size(x)
        assert shape == (2, 3, 4)
        assert dml.ops.size(x, 0) == 2
        assert dml.ops.size(x, -1) == 4

        values = dml.ops.tuple_construct("prefix", x)
        listed = dml.ops.list_construct(values[0], dml.ops.getitem(values, -1))
        assert x.builder.nodes == []
        assert x.builder.views == []
        return dml.ops.getitem(listed, -1)


class DynamicSymbolicHelpers(dml.Module):
    def forward(self, x):
        assert x.builder.nodes == []
        assert x.builder.views == []
        shape = dml.ops.size(x)
        assert shape[0]["kind"] == "dim"
        assert shape[0]["name"] == "batch"
        assert shape[0]["max"] == 4
        assert dml.ops.size(x, 0)["name"] == "batch"
        assert dml.ops.size(x, -1) == 16

        values = dml.ops.list_construct("ignored", x)
        assert x.builder.nodes == []
        assert x.builder.views == []
        return dml.ops.getitem(values, -1)


class BadSizeDim(dml.Module):
    def __init__(self, dim):
        self.dim = dim

    def forward(self, x):
        dml.ops.size(x, self.dim)
        return x


class BadGetitemIndex(dml.Module):
    def forward(self, x):
        dml.ops.getitem([x], True)
        return x


def test_symbolic_helpers_are_exported_through_ops():
    assert {"size", "getitem", "tuple_construct", "list_construct"}.issubset(dml.ops.__all__)
    assert dml.ops.size is not None
    assert dml.ops.getitem is not None
    assert dml.ops.tuple_construct is not None
    assert dml.ops.list_construct is not None


def test_static_symbolic_helpers_emit_no_nodes_inside_trace():
    spec = dml.trace(StaticSymbolicHelpers(), inputs={"x": dml.TensorSpec([2, 3, 4])}, name="static_symbolic_helpers")

    assert spec.ir["nodes"] == []
    assert spec.ir["outputs"][0]["shape_spec"] == [2, 3, 4]
    assert len(spec.ir["metadata"]["views"]["views"]) == 1
    assert spec.ir["metadata"]["views"]["views"][0]["transform"] == "identity"


def test_dynamic_symbolic_helpers_return_shape_spec_entries_without_nodes():
    batch = dml.Dim("batch", min=1, max=4, typical=2)
    spec = dml.trace(
        DynamicSymbolicHelpers(),
        inputs={"x": dml.TensorSpec([batch, 16])},
        name="dynamic_symbolic_helpers",
    )

    assert spec.ir["nodes"] == []
    assert spec.ir["outputs"][0]["shape"] == [4, 16]
    assert spec.ir["outputs"][0]["shape_spec"][0]["name"] == "batch"


@pytest.mark.parametrize("dim", [True, 1.0, "0"])
def test_size_rejects_non_integer_and_bool_dim(dim):
    with pytest.raises(TypeError, match="size dim must be an integer"):
        dml.trace(BadSizeDim(dim), inputs={"x": dml.TensorSpec([2, 3])})


@pytest.mark.parametrize("dim", [2, -3])
def test_size_rejects_out_of_range_dim(dim):
    with pytest.raises(IndexError, match="out of range"):
        dml.trace(BadSizeDim(dim), inputs={"x": dml.TensorSpec([2, 3])})


def test_getitem_rejects_bool_index_inside_trace():
    with pytest.raises(TypeError, match="getitem index must not be bool"):
        dml.trace(BadGetitemIndex(), inputs={"x": dml.TensorSpec([2, 3])})


def test_getitem_preserves_python_indexing_errors():
    with pytest.raises(IndexError):
        dml.ops.getitem(["a"], 1)

    with pytest.raises(TypeError):
        dml.ops.getitem(["a"], "bad")
