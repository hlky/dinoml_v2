import shutil

import numpy as np
import pytest

import dinoml as dml
from dinoml.backends.cpu import execute_cpu
from dinoml.lowering.ops import render_generated_kernels
from dinoml.passes import PassManager, validate_ir
from dinoml.runtime import load


class ClipLegacyTextPoolingModule(dml.Module):
    def forward(self, input_ids, hidden_states):
        indices = dml.ops.argmax(input_ids, dim=-1, keepdim=True)
        pooled = dml.ops.batch_gather(hidden_states, indices)
        pooled = dml.ops.squeeze(pooled, 1)
        return dml.ops.output(pooled, "out")


def _trace(hidden_dtype: str = "float32"):
    return dml.trace(
        ClipLegacyTextPoolingModule(),
        inputs={
            "input_ids": dml.TensorSpec([3, 6], "int64"),
            "hidden_states": dml.TensorSpec([3, 6, 4], hidden_dtype),
        },
        name=f"clip_legacy_text_pooling_{hidden_dtype}",
    )


def _input_ids():
    return np.array(
        [
            [49406, 120, 49407, 17, 0, 0],
            [49406, 49407, 42, 49407, 12, 0],
            [49406, 1, 2, 3, 4, 49407],
        ],
        dtype=np.int64,
    )


def _hidden_states(dtype: str = "float32"):
    values = np.arange(3 * 6 * 4, dtype=np.float32).reshape(3, 6, 4)
    if dtype == "float16":
        return values.astype(np.float16)
    return values


def _expected_pooled_hidden_states(input_ids, hidden_states):
    indices = np.argmax(input_ids, axis=-1)
    gathered = np.take_along_axis(hidden_states, indices[:, None, None], axis=1)
    return gathered.squeeze(axis=1)


def test_clip_legacy_text_pooling_frontend_ir_and_cpu_reference():
    spec = _trace()
    input_ids = _input_ids()
    hidden_states = _hidden_states()

    assert [node["op"] for node in spec.ir["nodes"]] == ["argmax", "batch_gather"]
    assert spec.ir["outputs"][0]["shape"] == [3, 4]
    assert spec.ir["outputs"][0]["dtype"] == "float32"
    assert spec.ir["metadata"]["views"]["views"][0]["transform"] == "squeeze"
    assert spec.ir["metadata"]["views"]["views"][0]["source"] == spec.ir["nodes"][1]["outputs"][0]
    np.testing.assert_array_equal(np.argmax(input_ids, axis=-1), np.array([2, 1, 5], dtype=np.int64))

    actual = execute_cpu(spec, {"input_ids": input_ids, "hidden_states": hidden_states})["out"]

    assert actual.dtype == np.float32
    np.testing.assert_array_equal(actual, _expected_pooled_hidden_states(input_ids, hidden_states))


def test_clip_legacy_text_pooling_generated_cpu_source_and_runtime(tmp_path, monkeypatch):
    spec = _trace()
    lowered, _ = PassManager().run(spec.ir)
    validate_ir(lowered)
    assert [node["op"] for node in lowered["nodes"]] == ["argmax", "batch_gather"]
    tensor_map = {tensor["name"]: tensor for tensor in lowered["tensors"]}

    cpu_sources = render_generated_kernels("cpu", lowered["nodes"], tensor_map)

    assert len(cpu_sources) == 2
    assert "static int argmax_" in cpu_sources[0]
    assert "int64_t* DINO_RESTRICT y" in cpu_sources[0]
    assert "static int batch_gather_" in cpu_sources[1]
    assert "const int64_t* DINO_RESTRICT index" in cpu_sources[1]
    assert "selected_index = static_cast<int64_t>(index[batch * 1 + k]);" in cpu_sources[1]
    assert "const int64_t input_idx = batch * 24 + selected_index * 4 + slice_offset;" in cpu_sources[1]
    assert 'return dino_runtime_fail("batch_gather index out of bounds");' in cpu_sources[1]

    artifact = dml.compile(spec, dml.Target("cpu"), tmp_path / "clip_legacy_text_pooling_cpu.dinoml")
    session = load(artifact.path).create_session()
    input_ids = _input_ids()
    hidden_states = _hidden_states()
    try:
        actual = session.run_numpy({"input_ids": input_ids, "hidden_states": hidden_states})["out"]
    finally:
        session.close()

    np.testing.assert_array_equal(actual, _expected_pooled_hidden_states(input_ids, hidden_states))


@pytest.mark.skipif(shutil.which("nvcc") is None, reason="nvcc is required")
def test_clip_legacy_text_pooling_generated_cuda_source_and_runtime(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device is required")
    spec = _trace()
    lowered, _ = PassManager().run(spec.ir)
    validate_ir(lowered)
    tensor_map = {tensor["name"]: tensor for tensor in lowered["tensors"]}

    cuda_sources = render_generated_kernels("cuda", lowered["nodes"], tensor_map)

    assert len(cuda_sources) == 2
    assert "static int argmax_" in cuda_sources[0]
    assert "int64_t* DINO_RESTRICT y" in cuda_sources[0]
    assert "static int batch_gather_" in cuda_sources[1]
    assert "const int64_t* DINO_RESTRICT index" in cuda_sources[1]
    assert "#include <assert.h>" in cuda_sources[1]

    artifact = dml.compile(spec, dml.Target("cuda", arch="sm_86"), tmp_path / "clip_legacy_text_pooling_cuda.dinoml")
    session = load(artifact.path).create_session()
    input_ids = _input_ids()
    hidden_states = _hidden_states()
    try:
        actual = session.run_numpy({"input_ids": input_ids, "hidden_states": hidden_states})["out"]
    finally:
        session.close()

    np.testing.assert_array_equal(actual, _expected_pooled_hidden_states(input_ids, hidden_states))
