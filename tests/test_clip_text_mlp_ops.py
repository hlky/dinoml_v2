import json
import shutil

import numpy as np
import pytest

import dinoml as dml
from dinoml.backends.cpu import execute_cpu
from dinoml.kernels.manifest import build_kernel_manifest
from dinoml.passes import PassManager, validate_ir


BATCH = 2
SEQ_LEN = 3
HIDDEN = 4
INTERMEDIATE = 6


class ClipTextMlpModule(dml.Module):
    def __init__(self):
        self.fc1_weight = dml.Parameter([INTERMEDIATE, HIDDEN], dtype="float32", value=_fc1_weight())
        self.fc1_bias = dml.Parameter([INTERMEDIATE], dtype="float32", value=_fc1_bias())
        self.fc2_weight = dml.Parameter([HIDDEN, INTERMEDIATE], dtype="float32", value=_fc2_weight())
        self.fc2_bias = dml.Parameter([HIDDEN], dtype="float32", value=_fc2_bias())

    def forward(self, hidden_states):
        hidden_states = dml.ops.gemm_rcr_bias_quick_gelu(hidden_states, self.fc1_weight, self.fc1_bias)
        hidden_states = dml.ops.gemm_rcr_bias(hidden_states, self.fc2_weight, self.fc2_bias)
        return dml.ops.output(hidden_states, "out")


def _trace():
    return dml.trace(
        ClipTextMlpModule(),
        inputs={"hidden_states": dml.TensorSpec([BATCH, SEQ_LEN, HIDDEN], "float32")},
        name="clip_text_mlp_float32",
    )


def _hidden_states():
    return np.array(
        [
            [[0.25, -0.50, 0.75, 1.00], [1.50, 0.25, -0.75, 0.50], [-0.25, 0.75, 0.50, -1.25]],
            [[-1.00, 0.50, 0.25, -0.75], [0.75, -0.25, -1.25, 1.00], [1.25, 1.50, -0.50, 0.25]],
        ],
        dtype=np.float32,
    )


def _fc1_weight():
    values = np.arange(INTERMEDIATE * HIDDEN, dtype=np.float32).reshape(INTERMEDIATE, HIDDEN)
    return values * 0.125 - 0.75


def _fc1_bias():
    return np.linspace(-0.4, 0.45, INTERMEDIATE, dtype=np.float32)


def _fc2_weight():
    values = np.arange(HIDDEN * INTERMEDIATE, dtype=np.float32).reshape(HIDDEN, INTERMEDIATE)
    return values * 0.0625 - 0.5


def _fc2_bias():
    return np.linspace(0.3, -0.15, HIDDEN, dtype=np.float32)


def _reference_clip_text_mlp(hidden_states):
    hidden_states = np.asarray(hidden_states, dtype=np.float32)
    intermediate = hidden_states @ _fc1_weight().T + _fc1_bias()
    intermediate = intermediate / (1.0 + np.exp(-1.702 * intermediate))
    return intermediate @ _fc2_weight().T + _fc2_bias()


def test_clip_text_mlp_frontend_ir_and_cpu_reference_match_numpy():
    spec = _trace()
    hidden_states = _hidden_states()

    assert [node["op"] for node in spec.ir["nodes"]] == ["gemm_rcr_bias_quick_gelu", "gemm_rcr_bias"]
    assert spec.ir["outputs"][0]["shape"] == [BATCH, SEQ_LEN, HIDDEN]
    assert spec.ir["outputs"][0]["dtype"] == "float32"

    actual = execute_cpu(spec, {"hidden_states": hidden_states})["out"]
    expected = _reference_clip_text_mlp(hidden_states)

    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_clip_text_mlp_manifest_and_lowering_use_quick_gelu_then_bias_gemm():
    spec = _trace()
    lowered, _ = PassManager().run(spec.ir)
    validate_ir(lowered)

    assert [node["op"] for node in lowered["nodes"]] == ["gemm_rcr_bias_quick_gelu", "gemm_rcr_bias"]

    manifest = build_kernel_manifest(lowered, {"name": "cuda", "arch": "sm_86"})
    required = manifest["required_kernels"]
    assert [entry["op"] for entry in required] == ["gemm_rcr_bias_quick_gelu", "gemm_rcr_bias"]

    quick_gelu_required = required[0]
    bias_required = required[1]
    assert quick_gelu_required["candidate_set_id"] == "cutlass_gemm_rcr_bias_quick_gelu_float32_bias_quick_gelu_v1"
    assert quick_gelu_required["kernel_library"] == "cutlass_gemm"
    assert quick_gelu_required["kernel_symbol"].startswith("dinoml_cutlass_gemm_rcr_bias_quick_gelu_float32_")
    assert quick_gelu_required["candidate_set"]["epilogue_config"]["activation"] == "quick_gelu"
    assert quick_gelu_required["candidate_set"]["epilogue_config"]["launch_abi"] == "dinoml_cutlass_gemm_bias_v1"
    assert bias_required["candidate_set_id"] == "cutlass_gemm_rcr_bias_float32_bias_v1"
    assert bias_required["kernel_library"] == "cutlass_gemm"
    assert bias_required["kernel_symbol"].startswith("dinoml_cutlass_gemm_rcr_bias_float32_")
    assert bias_required["candidate_set"]["epilogue_config"]["inputs"] == ["bias"]
    assert bias_required["candidate_set"]["epilogue_config"]["launch_abi"] == "dinoml_cutlass_gemm_bias_v1"


def test_fast_gelu_and_quick_gelu_gemm_surfaces_remain_distinct():
    class DistinctGeluModule(dml.Module):
        def forward(self, x, fast_weight, fast_bias, quick_weight, quick_bias):
            fast = dml.ops.gemm_rcr_bias_fast_gelu(x, fast_weight, fast_bias)
            quick = dml.ops.gemm_rcr_bias_quick_gelu(x, quick_weight, quick_bias)
            return dml.ops.output(dml.ops.add(fast, quick), "out")

    spec = dml.trace(
        DistinctGeluModule(),
        inputs={
            "x": dml.TensorSpec([BATCH, SEQ_LEN, HIDDEN], "float32"),
            "fast_weight": dml.TensorSpec([INTERMEDIATE, HIDDEN], "float32"),
            "fast_bias": dml.TensorSpec([INTERMEDIATE], "float32"),
            "quick_weight": dml.TensorSpec([INTERMEDIATE, HIDDEN], "float32"),
            "quick_bias": dml.TensorSpec([INTERMEDIATE], "float32"),
        },
        name="distinct_fast_and_quick_gelu_surface",
    )
    lowered, _ = PassManager().run(spec.ir)
    validate_ir(lowered)

    manifest = build_kernel_manifest(lowered, {"name": "cuda", "arch": "sm_86"})
    provider_entries = [
        entry
        for entry in manifest["required_kernels"]
        if entry["op"] in {"gemm_rcr_bias_fast_gelu", "gemm_rcr_bias_quick_gelu"}
    ]

    assert [entry["op"] for entry in provider_entries] == ["gemm_rcr_bias_fast_gelu", "gemm_rcr_bias_quick_gelu"]
    assert provider_entries[0]["candidate_set"]["epilogue_config"]["activation"] == "fast_gelu"
    assert provider_entries[1]["candidate_set"]["epilogue_config"]["activation"] == "quick_gelu"
    assert provider_entries[0]["kernel_symbol"].startswith("dinoml_cutlass_gemm_rcr_bias_fast_gelu_float32_")
    assert provider_entries[1]["kernel_symbol"].startswith("dinoml_cutlass_gemm_rcr_bias_quick_gelu_float32_")


@pytest.mark.skipif(shutil.which("nvcc") is None, reason="nvcc is required")
def test_clip_text_mlp_quick_gelu_generated_cuda_runtime_matches_reference(tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device is required")

    from dinoml import runtime

    spec = _trace()
    hidden_states = _hidden_states()
    expected = _reference_clip_text_mlp(hidden_states)

    artifact = dml.compile(
        spec,
        dml.Target("cuda", arch="sm_86", no_tf32=True),
        tmp_path / "clip_text_mlp_quick_gelu_cuda.dinoml",
    )

    generated = (artifact.path / "debug" / "generated_src" / "module.cu").read_text(encoding="utf-8")
    assert "dinoml_cutlass_gemm_rcr_bias_quick_gelu" in generated

    kernel_manifest = json.loads((artifact.path / "kernel_manifest.json").read_text(encoding="utf-8"))
    assert [entry["op"] for entry in kernel_manifest["required_kernels"]] == ["gemm_rcr_bias_quick_gelu", "gemm_rcr_bias"]

    module = runtime.load(artifact.path)
    session = module.create_session()
    try:
        actual = session.run_numpy({"hidden_states": hidden_states})["out"]
    finally:
        session.close()
        module.close()

    np.testing.assert_allclose(actual, expected, atol=5e-4, rtol=5e-4)
