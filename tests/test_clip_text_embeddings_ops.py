import shutil

import numpy as np
import pytest

import dinoml as dml
from dinoml import runtime
from dinoml.backends.cpu import execute_cpu
from dinoml.lowering.ops import collect_generated_sources
from dinoml.passes import PassManager, validate_ir


class ClipTextEmbeddingsModule(dml.Module):
    def forward(self, token_table, position_table, input_ids, position_ids):
        token_embeddings = dml.ops.embedding(token_table, input_ids)
        position_embeddings = dml.ops.embedding(position_table, position_ids)
        hidden_states = dml.ops.add(token_embeddings, position_embeddings)
        return dml.ops.output(hidden_states, "out")


def _trace(*, position_ids_shape, position_ids_dtype="int64"):
    batch = 2
    seq_len = 6
    hidden = 4
    return dml.trace(
        ClipTextEmbeddingsModule(),
        inputs={
            "token_table": dml.TensorSpec([8, hidden], "float32"),
            "position_table": dml.TensorSpec([seq_len, hidden], "float32"),
            "input_ids": dml.TensorSpec([batch, seq_len], "int64"),
            "position_ids": dml.TensorSpec(position_ids_shape, position_ids_dtype),
        },
        name=f"clip_text_embeddings_{position_ids_dtype}_{'x'.join(str(dim) for dim in position_ids_shape)}",
    )


def _token_table():
    values = np.arange(8 * 4, dtype=np.float32).reshape(8, 4) * 0.25 - 1.5
    return np.asarray(values, dtype=np.float32)


def _position_table():
    values = np.arange(6 * 4, dtype=np.float32).reshape(6, 4) * 0.5 + 0.25
    return np.asarray(values, dtype=np.float32)


def _input_ids():
    return np.array([[3, 0, 1, 6, 2, 4], [1, 5, 3, 0, 6, 2]], dtype=np.int64)


def _position_ids_1d():
    return np.array([0, 1, 2, 3, 4, 5], dtype=np.int64)


def _position_ids_2d():
    return np.array([[0, 1, 2, 3, 4, 5], [5, 4, 3, 2, 1, 0]], dtype=np.int32)


def _reference_clip_text_embeddings(token_table, position_table, input_ids, position_ids):
    token_embeddings = np.asarray(token_table, dtype=np.float32)[np.asarray(input_ids, dtype=np.int64)]
    position_embeddings = np.asarray(position_table, dtype=np.float32)[np.asarray(position_ids, dtype=np.int64)]
    return token_embeddings + position_embeddings


@pytest.mark.parametrize(
    ("position_ids_shape", "position_ids_dtype", "position_ids"),
    [
        ((6,), "int64", _position_ids_1d()),
        ((2, 6), "int32", _position_ids_2d()),
    ],
)
def test_clip_text_embeddings_frontend_ir_and_cpu_reference_match_numpy(
    position_ids_shape,
    position_ids_dtype,
    position_ids,
):
    spec = _trace(position_ids_shape=position_ids_shape, position_ids_dtype=position_ids_dtype)
    token_table = _token_table()
    position_table = _position_table()
    input_ids = _input_ids()

    assert [node["op"] for node in spec.ir["nodes"]] == ["embedding", "embedding", "add"]
    assert spec.ir["outputs"][0]["shape"] == [2, 6, 4]
    assert spec.ir["outputs"][0]["shape_spec"] == [2, 6, 4]
    assert spec.ir["outputs"][0]["dtype"] == "float32"

    actual = execute_cpu(
        spec,
        {
            "token_table": token_table,
            "position_table": position_table,
            "input_ids": input_ids,
            "position_ids": position_ids,
        },
    )["out"]
    expected = _reference_clip_text_embeddings(token_table, position_table, input_ids, position_ids)

    np.testing.assert_array_equal(actual, expected)


def test_clip_text_embeddings_generated_cpu_runtime_supports_broadcast_position_ids(tmp_path, monkeypatch):
    spec = _trace(position_ids_shape=(6,), position_ids_dtype="int64")
    lowered, _ = PassManager().run(spec.ir)
    validate_ir(lowered)
    assert [node["op"] for node in lowered["nodes"]] == ["embedding", "embedding", "fused_elementwise"]

    tensor_map = {tensor["name"]: tensor for tensor in lowered["tensors"]}
    sources = collect_generated_sources("cpu", lowered["nodes"], tensor_map)
    assert len(sources["kernels"]) == 3
    assert sum("embedding_" in source for source in sources["kernels"]) == 2
    assert any("dinoml::math::add" in source for source in sources["kernels"])

    artifact = dml.compile(spec, dml.Target("cpu"), tmp_path / "clip_text_embeddings_cpu.dinoml")

    token_table = _token_table()
    position_table = _position_table()
    input_ids = _input_ids()
    position_ids = _position_ids_1d()
    expected = _reference_clip_text_embeddings(token_table, position_table, input_ids, position_ids)

    module = runtime.load(artifact.path)
    session = module.create_session()
    try:
        actual = session.run_numpy(
            {
                "token_table": token_table,
                "position_table": position_table,
                "input_ids": input_ids,
                "position_ids": position_ids,
            }
        )["out"]
    finally:
        session.close()
        module.close()

    np.testing.assert_array_equal(actual, expected)


@pytest.mark.skipif(shutil.which("nvcc") is None, reason="nvcc is required")
def test_clip_text_embeddings_generated_cuda_runtime_supports_batched_position_ids(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device is required")
    spec = _trace(position_ids_shape=(2, 6), position_ids_dtype="int32")
    lowered, _ = PassManager().run(spec.ir)
    validate_ir(lowered)
    assert [node["op"] for node in lowered["nodes"]] == ["embedding", "embedding", "fused_elementwise"]

    tensor_map = {tensor["name"]: tensor for tensor in lowered["tensors"]}
    sources = collect_generated_sources("cuda", lowered["nodes"], tensor_map)
    assert len(sources["kernels"]) == 3
    assert sum("embedding_" in source for source in sources["kernels"]) == 2
    assert any("dinoml::math::add" in source for source in sources["kernels"])

    artifact = dml.compile(spec, dml.Target("cuda", arch="sm_86"), tmp_path / "clip_text_embeddings_cuda.dinoml")

    token_table = _token_table()
    position_table = _position_table()
    input_ids = _input_ids()
    position_ids = _position_ids_2d()
    expected = _reference_clip_text_embeddings(token_table, position_table, input_ids, position_ids)

    module = runtime.load(artifact.path)
    session = module.create_session()
    try:
        actual = session.run_numpy(
            {
                "token_table": token_table,
                "position_table": position_table,
                "input_ids": input_ids,
                "position_ids": position_ids,
            }
        )["out"]
    finally:
        session.close()
        module.close()

    np.testing.assert_array_equal(actual, expected)
