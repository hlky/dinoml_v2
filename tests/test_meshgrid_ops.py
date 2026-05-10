import numpy as np
import pytest

import dinoml as dml
from dinoml.passes import PassManager, validate_ir
from dinoml.runtime import load
from dinoml.shapes import Dim


class MeshgridModule(dml.Module):
    def __init__(self, indexing="ij", named=False):
        self.indexing = indexing
        self.named = named

    def forward(self, **inputs):
        values = tuple(inputs[name] for name in sorted(inputs))
        grids = dml.ops.meshgrid(values, indexing=self.indexing)
        if self.named:
            return tuple(dml.ops.output(grid, f"grid_{idx}") for idx, grid in enumerate(grids))
        return grids


def _trace_meshgrid(shapes=((2,), (3,), (4,)), dtype="float32", indexing="ij", named=False):
    inputs = {f"x{idx}": dml.TensorSpec(shape, dtype) for idx, shape in enumerate(shapes)}
    return dml.trace(MeshgridModule(indexing=indexing, named=named), inputs=inputs, name=f"meshgrid_{dtype}")


def _meshgrid_inputs(dtype, shapes=((2,), (3,), (4,))):
    if dtype == "bool":
        return {
            f"x{idx}": (np.arange(shape[0]) % 2 == 0).astype(np.bool_)
            for idx, shape in enumerate(shapes)
        }
    return {f"x{idx}": (idx * 10 + np.arange(shape[0], dtype=np.float32)) for idx, shape in enumerate(shapes)}


def test_meshgrid_frontend_emits_tuple_outputs_with_views_and_expands():
    spec = _trace_meshgrid()

    assert [output["shape"] for output in spec.ir["outputs"]] == [[2, 3, 4], [2, 3, 4], [2, 3, 4]]
    assert [output["dtype"] for output in spec.ir["outputs"]] == ["float32", "float32", "float32"]
    assert [node["op"] for node in spec.ir["nodes"]] == ["expand", "expand", "expand"]
    assert [node["attrs"] for node in spec.ir["nodes"]] == [
        {"shape": [2, 3, 4]},
        {"shape": [2, 3, 4]},
        {"shape": [2, 3, 4]},
    ]
    views = spec.ir["metadata"]["views"]["views"]
    assert [view["transform"] for view in views] == ["reshape", "reshape", "reshape"]
    assert [view["shape"] for view in views] == [[2, 1, 1], [1, 3, 1], [1, 1, 4]]
    assert [node["inputs"][0] for node in spec.ir["nodes"]] == [view["tensor"] for view in views]


@pytest.mark.parametrize("dtype", ["float32", "bool"])
def test_meshgrid_generated_cpu_runtime_returns_multiple_outputs(tmp_path, dtype):
    shapes = ((2,), (3,), (4,)) if dtype == "float32" else ((2,), (3,))
    spec = _trace_meshgrid(shapes, dtype=dtype, named=True)
    lowered, _ = PassManager().run(spec.ir)
    validate_ir(lowered)
    assert [node["op"] for node in lowered["nodes"]] == ["expand"] * len(shapes)
    assert lowered["metadata"]["memory_plan"]["views"]["views"]

    artifact = dml.compile(spec, dml.Target("cpu"), tmp_path / f"meshgrid_{dtype}_cpu.dinoml")
    session = load(artifact.path).create_session()
    inputs = _meshgrid_inputs(dtype, shapes)
    try:
        actual = session.run_numpy(inputs)
    finally:
        session.close()

    expected = np.meshgrid(*(inputs[f"x{idx}"] for idx in range(len(shapes))), indexing="ij")
    for idx, expected_grid in enumerate(expected):
        np.testing.assert_array_equal(actual[f"grid_{idx}"], expected_grid)


def test_meshgrid_rejects_empty_inputs():
    class EmptyMeshgrid(dml.Module):
        def forward(self):
            return dml.ops.meshgrid([])

    with pytest.raises(ValueError, match="non-empty sequence"):
        dml.trace(EmptyMeshgrid(), inputs={})


def test_meshgrid_rejects_non_rank_1_dtype_mismatch_dynamic_and_bad_indexing():
    with pytest.raises(ValueError, match="rank-1"):
        _trace_meshgrid(shapes=((2, 1), (3,)))

    with pytest.raises(ValueError, match="dtype mismatch"):
        dml.trace(
            MeshgridModule(),
            inputs={"x0": dml.TensorSpec([2], "float32"), "x1": dml.TensorSpec([3], "bool")},
        )

    with pytest.raises(ValueError, match="only static input shapes"):
        dml.trace(
            MeshgridModule(),
            inputs={"x0": dml.TensorSpec([Dim("n", 1, 4)]), "x1": dml.TensorSpec([3])},
        )

    with pytest.raises(NotImplementedError, match='indexing="ij" only'):
        _trace_meshgrid(indexing="xy")

    with pytest.raises(ValueError, match="does not support dtype int64"):
        _trace_meshgrid(dtype="int64")
