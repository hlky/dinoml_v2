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


class SymbolicIntHelpers(dml.Module):
    def forward(self, x):
        assert x.builder.nodes == []
        assert x.builder.views == []
        n = dml.ops.size(x, 0)
        expr = dml.ops.int_div(dml.ops.int_mul(dml.ops.int_add(n, 3), 2), 5)
        assert expr == {
            "kind": "int_expr",
            "op": "div",
            "lhs": {
                "kind": "int_expr",
                "op": "mul",
                "lhs": {
                    "kind": "int_expr",
                    "op": "add",
                    "lhs": {
                        "kind": "dim",
                        "name": "batch",
                        "min": 1,
                        "max": 8,
                        "divisible_by": 1,
                        "typical": 4,
                    },
                    "rhs": 3,
                },
                "rhs": 2,
            },
            "rhs": 5,
        }
        assert dml.ops.int_sub(10, 4) == 6
        assert x.builder.nodes == []
        assert x.builder.views == []
        return x


def test_symbolic_helpers_are_exported_through_ops():
    assert {
        "size",
        "getitem",
        "tuple_construct",
        "list_construct",
        "int_add",
        "int_sub",
        "int_mul",
        "int_div",
    }.issubset(dml.ops.__all__)
    assert dml.ops.size is not None
    assert dml.ops.getitem is not None
    assert dml.ops.tuple_construct is not None
    assert dml.ops.list_construct is not None
    assert dml.ops.int_add is not None
    assert dml.ops.int_sub is not None
    assert dml.ops.int_mul is not None
    assert dml.ops.int_div is not None


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


def test_dynamic_symbolic_int_expressions_are_json_compatible_and_nested():
    batch = dml.Dim("batch", min=1, max=8, typical=4)
    dim = dml.TensorSpec([batch, 16]).shape_spec[0]

    expr = dml.ops.int_sub(dml.ops.int_add(dim, 2), dml.ops.int_div(dim, 3))

    assert expr == {
        "kind": "int_expr",
        "op": "sub",
        "lhs": {
            "kind": "int_expr",
            "op": "add",
            "lhs": {"kind": "dim", "name": "batch", "min": 1, "max": 8, "divisible_by": 1, "typical": 4},
            "rhs": 2,
        },
        "rhs": {
            "kind": "int_expr",
            "op": "div",
            "lhs": {"kind": "dim", "name": "batch", "min": 1, "max": 8, "divisible_by": 1, "typical": 4},
            "rhs": 3,
        },
    }


def test_static_symbolic_int_expressions_constant_fold():
    assert dml.ops.int_add(4, 5) == 9
    assert dml.ops.int_sub(4, 5) == -1
    assert dml.ops.int_mul(4, 5) == 20
    assert dml.ops.int_div(7, 3) == 2
    assert dml.ops.int_div(-7, 3) == -3


@pytest.mark.parametrize("bad", [True, 1.0, "1", None])
def test_symbolic_int_helpers_reject_unsupported_values(bad):
    with pytest.raises(TypeError, match="must be an integer or symbolic dimension"):
        dml.ops.int_add(bad, 1)


@pytest.mark.parametrize(
    "bad",
    [
        {"kind": "dim", "name": "batch"},
        {"kind": "not_dim", "name": "batch", "min": 1, "max": 4},
        {"kind": "int_expr", "op": "pow", "lhs": 2, "rhs": 3},
    ],
)
def test_symbolic_int_helpers_reject_unsupported_mappings(bad):
    with pytest.raises((TypeError, ValueError, KeyError), match="Unsupported|missing|required|'min'"):
        dml.ops.int_add(bad, 1)


def test_symbolic_int_div_rejects_static_zero_rhs():
    with pytest.raises(ZeroDivisionError, match="div by zero"):
        dml.ops.int_div(4, 0)

    dim = dml.TensorSpec([dml.Dim("batch", 1, 4)]).shape_spec[0]
    with pytest.raises(ZeroDivisionError, match="div by zero"):
        dml.ops.int_div(dim, 0)

    with pytest.raises(ZeroDivisionError, match="div by zero"):
        dml.ops.int_add({"kind": "int_expr", "op": "div", "lhs": dim, "rhs": 0}, 1)


def test_symbolic_int_expressions_are_supported_shape_dims():
    dim = dml.TensorSpec([dml.Dim("batch", 1, 4)]).shape_spec[0]
    expr = dml.ops.int_add(dim, 1)

    spec = dml.TensorSpec([expr])

    assert spec.max_shape == [5]
    assert spec.shape_spec == [
        {
            "kind": "int_expr",
            "op": "add",
            "lhs": {"kind": "dim", "name": "batch", "min": 1, "max": 4, "divisible_by": 1},
            "rhs": 1,
        }
    ]


def test_symbolic_shape_dim_interval_handles_nested_add_mul_div():
    dim = dml.TensorSpec([dml.Dim("batch", 1, 8)]).shape_spec[0]
    expr = dml.ops.int_div(dml.ops.int_mul(dml.ops.int_add(dim, 3), 2), 5)
    spec = dml.TensorSpec([expr])

    assert spec.max_shape == [4]
    assert spec.shape_spec[0]["kind"] == "int_expr"


def test_symbolic_shape_dim_rejects_invalid_bounds_and_divisor_interval():
    dim = dml.TensorSpec([dml.Dim("batch", 1, 4)]).shape_spec[0]

    with pytest.raises(ValueError, match="minimum must be positive"):
        dml.TensorSpec([dml.ops.int_sub(dim, 5)])

    with pytest.raises(ZeroDivisionError, match="denominator interval contains zero"):
        dml.TensorSpec([dml.ops.int_div(dim, dml.ops.int_sub(dim, 2))])


def test_symbolic_int_helpers_emit_no_nodes_inside_trace():
    batch = dml.Dim("batch", min=1, max=8, typical=4)
    spec = dml.trace(SymbolicIntHelpers(), inputs={"x": dml.TensorSpec([batch, 16])}, name="symbolic_int_helpers")

    assert spec.ir["nodes"] == []
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
