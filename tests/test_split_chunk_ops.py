import numpy as np
import pytest

import dinoml as dml
from dinoml.backends.cpu import execute_cpu
from dinoml.passes import PassManager, validate_ir
from dinoml.runtime import load
from dinoml.shapes import Dim


class SplitModule(dml.Module):
    def __init__(self, split_size_or_sections, dim=0, named=False):
        self.split_size_or_sections = split_size_or_sections
        self.dim = dim
        self.named = named

    def forward(self, x):
        parts = dml.ops.split(x, self.split_size_or_sections, dim=self.dim)
        if self.named:
            return tuple(dml.ops.output(part, f"part_{idx}") for idx, part in enumerate(parts))
        return parts


class ChunkModule(dml.Module):
    def __init__(self, chunks, dim=0, named=False):
        self.chunks = chunks
        self.dim = dim
        self.named = named

    def forward(self, x):
        parts = dml.ops.chunk(x, self.chunks, dim=self.dim)
        if self.named:
            return tuple(dml.ops.output(part, f"chunk_{idx}") for idx, part in enumerate(parts))
        return parts


def _trace_split(split_size_or_sections, dim=0, shape=(2, 5, 3), dtype="float32", named=False):
    return dml.trace(
        SplitModule(split_size_or_sections, dim=dim, named=named),
        inputs={"x": dml.TensorSpec(shape, dtype)},
        name="split",
    )


def _trace_chunk(chunks, dim=0, shape=(2, 5, 3), dtype="float32", named=False):
    return dml.trace(
        ChunkModule(chunks, dim=dim, named=named),
        inputs={"x": dml.TensorSpec(shape, dtype)},
        name="chunk",
    )


def test_split_frontend_ir_emits_dynamic_slices_and_flattens_tuple_outputs():
    spec = _trace_split(2, dim=1)

    assert [node["op"] for node in spec.ir["nodes"]] == ["dynamic_slice", "dynamic_slice", "dynamic_slice"]
    assert [output["name"] for output in spec.ir["outputs"]] == ["output_0", "output_1", "output_2"]
    assert [output["shape"] for output in spec.ir["outputs"]] == [[2, 2, 3], [2, 2, 3], [2, 1, 3]]
    assert spec.ir["nodes"][0]["attrs"] == {"start_indices": [0, 0, 0], "slice_sizes": [2, 2, 3]}
    assert spec.ir["nodes"][1]["attrs"] == {"start_indices": [0, 2, 0], "slice_sizes": [2, 2, 3]}
    assert spec.ir["nodes"][2]["attrs"] == {"start_indices": [0, 4, 0], "slice_sizes": [2, 1, 3]}


def test_split_normalizes_negative_dim():
    spec = _trace_split(1, dim=-1, shape=(2, 3, 2))

    assert [output["shape"] for output in spec.ir["outputs"]] == [[2, 3, 1], [2, 3, 1]]
    assert spec.ir["nodes"][1]["attrs"] == {"start_indices": [0, 0, 1], "slice_sizes": [2, 3, 1]}


def test_split_sections():
    spec = _trace_split([1, 3, 1], dim=1)

    assert [output["shape"] for output in spec.ir["outputs"]] == [[2, 1, 3], [2, 3, 3], [2, 1, 3]]
    assert spec.ir["nodes"][1]["attrs"] == {"start_indices": [0, 1, 0], "slice_sizes": [2, 3, 3]}


def test_chunk_uneven_and_chunks_greater_than_dim():
    uneven = _trace_chunk(3, dim=1, shape=(2, 10, 3))
    too_many = _trace_chunk(8, dim=1, shape=(2, 3, 3))

    assert [output["shape"] for output in uneven.ir["outputs"]] == [[2, 4, 3], [2, 4, 3], [2, 2, 3]]
    assert [output["shape"] for output in too_many.ir["outputs"]] == [[2, 1, 3], [2, 1, 3], [2, 1, 3]]
    assert len(too_many.ir["outputs"]) == 3


def test_cpu_reference_split_and_chunk():
    x = np.arange(30, dtype=np.float32).reshape(2, 5, 3)
    split_spec = _trace_split([2, 3], dim=1, named=True)
    chunk_spec = _trace_chunk(3, dim=1, shape=(2, 5, 3), named=True)

    split_actual = execute_cpu(split_spec, {"x": x})
    chunk_actual = execute_cpu(chunk_spec, {"x": x})

    np.testing.assert_array_equal(split_actual["part_0"], x[:, :2, :])
    np.testing.assert_array_equal(split_actual["part_1"], x[:, 2:, :])
    np.testing.assert_array_equal(chunk_actual["chunk_0"], x[:, :2, :])
    np.testing.assert_array_equal(chunk_actual["chunk_1"], x[:, 2:4, :])
    np.testing.assert_array_equal(chunk_actual["chunk_2"], x[:, 4:, :])


@pytest.mark.parametrize(
    ("dtype", "input_value"),
    [
        ("float32", np.arange(30, dtype=np.float32).reshape(2, 5, 3)),
        ("bool", (np.arange(30).reshape(2, 5, 3) % 2) == 0),
    ],
)
def test_split_generated_cpu_runtime_returns_multiple_outputs(tmp_path, dtype, input_value):
    spec = _trace_split(2, dim=1, dtype=dtype, named=True)
    lowered, _ = PassManager().run(spec.ir)
    validate_ir(lowered)
    assert [node["op"] for node in lowered["nodes"]] == ["dynamic_slice", "dynamic_slice", "dynamic_slice"]

    artifact = dml.compile(spec, dml.Target("cpu"), tmp_path / f"split_{dtype}_cpu.dinoml")
    session = load(artifact.path).create_session()
    try:
        actual = session.run_numpy({"x": input_value})
    finally:
        session.close()

    np.testing.assert_array_equal(actual["part_0"], input_value[:, :2, :])
    np.testing.assert_array_equal(actual["part_1"], input_value[:, 2:4, :])
    np.testing.assert_array_equal(actual["part_2"], input_value[:, 4:, :])


def test_split_rejects_bad_dims_sections_and_dynamic_shapes():
    with pytest.raises(ValueError, match="out of range"):
        _trace_split(2, dim=3)
    with pytest.raises(ValueError, match="positive"):
        _trace_split(0, dim=1)
    with pytest.raises(ValueError, match="positive"):
        _trace_split(-1, dim=1)
    with pytest.raises(ValueError, match="positive integer"):
        _trace_split(True, dim=1)
    with pytest.raises(ValueError, match="positive integer"):
        _trace_split(1.5, dim=1)
    with pytest.raises(ValueError, match="positive"):
        _trace_split([2, 0, 3], dim=1)
    with pytest.raises(ValueError, match="positive"):
        _trace_split([2, -1, 4], dim=1)
    with pytest.raises(ValueError, match="positive integers"):
        _trace_split([2, True, 3], dim=1)
    with pytest.raises(ValueError, match="sum to dim size 5"):
        _trace_split([2, 2], dim=1)
    with pytest.raises(ValueError, match="non-empty"):
        _trace_split([], dim=1)
    with pytest.raises(ValueError, match="only static input shapes"):
        dml.trace(
            SplitModule(2, dim=0),
            inputs={"x": dml.TensorSpec([Dim("n", 1, 4), 3])},
        )


def test_chunk_rejects_bad_chunks_dim_and_dynamic_shapes():
    with pytest.raises(ValueError, match="positive"):
        _trace_chunk(0, dim=1)
    with pytest.raises(ValueError, match="positive integer"):
        _trace_chunk(True, dim=1)
    with pytest.raises(ValueError, match="out of range"):
        _trace_chunk(2, dim=-4)
    with pytest.raises(ValueError, match="only static input shapes"):
        dml.trace(
            ChunkModule(2, dim=0),
            inputs={"x": dml.TensorSpec([Dim("n", 1, 4), 3])},
        )
