import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_SRC = REPO_ROOT / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

import dinoml as dml
from dinoml.backends.cpu import execute_cpu
from dinoml.kernels.manifest import build_kernel_manifest
from dinoml.lowering.ops import collect_generated_sources
from dinoml.models.clip import LegacyCLIPTextConfig, LegacyCLIPTextModelWithProjection
from dinoml.passes import PassManager, validate_ir


LOCAL_TRANSFORMERS_SRC = Path("/workspace/transformers/src")
BATCH = 2
SEQ_LEN = 4
VOCAB_SIZE = 16
HIDDEN = 6
NUM_HEADS = 2
INTERMEDIATE = 8
PROJECTION = 5
EPS = 1.0e-5
MAX_POSITION_EMBEDDINGS = 6


def _config(*, eos_token_id: int = 2, num_hidden_layers: int = 2):
    return LegacyCLIPTextConfig(
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=MAX_POSITION_EMBEDDINGS,
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        num_attention_heads=NUM_HEADS,
        num_hidden_layers=num_hidden_layers,
        projection_dim=PROJECTION,
        layer_norm_eps=EPS,
        eos_token_id=eos_token_id,
    )


def _weights():
    rng = np.random.default_rng(2031)

    def _normal(shape, scale):
        return (rng.standard_normal(shape).astype(np.float32) / scale).astype(np.float32)

    config = _config()
    weights = {
        "text_model.embeddings.token_embedding.weight": _normal((config.vocab_size, config.hidden_size), 3.5),
        "text_model.embeddings.position_embedding.weight": _normal(
            (config.max_position_embeddings, config.hidden_size), 4.0
        ),
        "text_model.final_layer_norm.weight": _normal((config.hidden_size,), 4.0),
        "text_model.final_layer_norm.bias": _normal((config.hidden_size,), 6.0),
        "text_projection.weight": _normal((config.projection_dim, config.hidden_size), 4.0),
    }
    for layer_idx in range(config.num_hidden_layers):
        prefix = f"text_model.encoder.layers.{layer_idx}"
        weights[f"{prefix}.self_attn.q_proj.weight"] = _normal((config.hidden_size, config.hidden_size), 5.0)
        weights[f"{prefix}.self_attn.q_proj.bias"] = _normal((config.hidden_size,), 7.0)
        weights[f"{prefix}.self_attn.k_proj.weight"] = _normal((config.hidden_size, config.hidden_size), 5.0)
        weights[f"{prefix}.self_attn.k_proj.bias"] = _normal((config.hidden_size,), 7.0)
        weights[f"{prefix}.self_attn.v_proj.weight"] = _normal((config.hidden_size, config.hidden_size), 5.0)
        weights[f"{prefix}.self_attn.v_proj.bias"] = _normal((config.hidden_size,), 7.0)
        weights[f"{prefix}.self_attn.out_proj.weight"] = _normal((config.hidden_size, config.hidden_size), 5.0)
        weights[f"{prefix}.self_attn.out_proj.bias"] = _normal((config.hidden_size,), 7.0)
        weights[f"{prefix}.layer_norm1.weight"] = _normal((config.hidden_size,), 4.0)
        weights[f"{prefix}.layer_norm1.bias"] = _normal((config.hidden_size,), 6.0)
        weights[f"{prefix}.mlp.fc1.weight"] = _normal((config.intermediate_size, config.hidden_size), 4.5)
        weights[f"{prefix}.mlp.fc1.bias"] = _normal((config.intermediate_size,), 6.5)
        weights[f"{prefix}.mlp.fc2.weight"] = _normal((config.hidden_size, config.intermediate_size), 4.5)
        weights[f"{prefix}.mlp.fc2.bias"] = _normal((config.hidden_size,), 6.5)
        weights[f"{prefix}.layer_norm2.weight"] = _normal((config.hidden_size,), 4.0)
        weights[f"{prefix}.layer_norm2.bias"] = _normal((config.hidden_size,), 6.0)
    return weights


WEIGHTS = _weights()


def _input_ids(*, eos_token_id: int = 2):
    if eos_token_id == 2:
        return np.array(
            [
                [0, 5, 15, 1],
                [0, 15, 4, 1],
            ],
            dtype=np.int64,
        )
    return np.array(
        [
            [0, eos_token_id, 3, eos_token_id],
            [5, 4, eos_token_id, eos_token_id],
        ],
        dtype=np.int64,
    )


def _attention_mask():
    return np.array(
        [
            [True, True, True, False],
            [True, True, True, False],
        ],
        dtype=np.bool_,
    )


def _position_ids():
    return np.array([0, 1, 2, 3], dtype=np.int64)


def _trace(*, eos_token_id: int = 2, include_position_ids: bool = True, num_hidden_layers: int = 2):
    inputs = {
        "input_ids": dml.TensorSpec([BATCH, SEQ_LEN], "int64"),
        "attention_mask": dml.TensorSpec([BATCH, SEQ_LEN], "bool"),
    }
    if include_position_ids:
        inputs["position_ids"] = dml.TensorSpec([SEQ_LEN], "int64")
    return dml.trace(
        LegacyCLIPTextModelWithProjection(
            _config(eos_token_id=eos_token_id, num_hidden_layers=num_hidden_layers),
            WEIGHTS,
        ),
        inputs=inputs,
        name=f"clip_text_model_with_projection_eos_{eos_token_id}_{num_hidden_layers}_layer",
    )


def _import_local_transformers():
    if str(LOCAL_TRANSFORMERS_SRC) not in sys.path:
        sys.path.insert(0, str(LOCAL_TRANSFORMERS_SRC))
    transformers = pytest.importorskip("transformers")
    resolved = Path(transformers.__file__).resolve()
    assert resolved.is_relative_to(LOCAL_TRANSFORMERS_SRC.resolve()), (
        f"expected local /workspace/transformers import, got {resolved}"
    )
    return transformers


def _reference_outputs(*, eos_token_id: int = 2, include_position_ids: bool = True, num_hidden_layers: int = 2):
    torch = pytest.importorskip("torch")
    transformers = _import_local_transformers()

    text_config = transformers.CLIPTextConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        projection_dim=PROJECTION,
        num_attention_heads=NUM_HEADS,
        num_hidden_layers=num_hidden_layers,
        max_position_embeddings=MAX_POSITION_EMBEDDINGS,
        hidden_act="quick_gelu",
        attention_dropout=0.0,
        layer_norm_eps=EPS,
        bos_token_id=0,
        eos_token_id=eos_token_id,
        pad_token_id=1 if eos_token_id == 2 else eos_token_id,
    )
    vision_config = transformers.CLIPVisionConfig(
        hidden_size=8,
        intermediate_size=16,
        projection_dim=PROJECTION,
        num_attention_heads=2,
        num_hidden_layers=1,
        image_size=4,
        patch_size=2,
    )
    config = transformers.CLIPConfig(
        text_config=text_config.to_dict(),
        vision_config=vision_config.to_dict(),
        projection_dim=PROJECTION,
    )
    model = transformers.CLIPModel(config)
    state_dict = model.state_dict()
    for name, value in WEIGHTS.items():
        if name in state_dict:
            state_dict[name] = torch.from_numpy(np.asarray(value, dtype=np.float32))
    model.load_state_dict(state_dict)
    model.eval()

    input_ids = torch.from_numpy(_input_ids(eos_token_id=eos_token_id))
    attention_mask = torch.from_numpy(_attention_mask())
    kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
    if include_position_ids:
        kwargs["position_ids"] = torch.from_numpy(_position_ids())
    with torch.inference_mode():
        text_features = model.get_text_features(**kwargs)
    return text_features.pooler_output.detach().cpu().numpy().astype(np.float32)


@pytest.mark.parametrize(
    ("eos_token_id", "expected_counts", "include_position_ids"),
    [
        (2, {"argmax": 1, "batch_gather": 1, "eq": 0}, True),
        (2, {"argmax": 1, "batch_gather": 1, "eq": 0}, False),
        (7, {"argmax": 1, "batch_gather": 1, "eq": 1}, True),
        (7, {"argmax": 1, "batch_gather": 1, "eq": 1}, False),
    ],
)
def test_clip_text_wrapper_get_text_features_matches_local_transformers(
    eos_token_id, expected_counts, include_position_ids
):
    num_hidden_layers = 2
    spec = _trace(
        eos_token_id=eos_token_id,
        include_position_ids=include_position_ids,
        num_hidden_layers=num_hidden_layers,
    )
    node_ops = [node["op"] for node in spec.ir["nodes"]]

    assert node_ops.count("embedding") == 2
    assert node_ops.count("dynamic_slice") == 1
    assert node_ops.count("layer_norm") == (2 * num_hidden_layers) + 1
    assert node_ops.count("gemm_rcr_bias") == 5 * num_hidden_layers
    assert node_ops.count("gemm_rcr_bias_fast_gelu") == num_hidden_layers
    assert node_ops.count("gemm_rcr") == 1
    assert node_ops.count("bmm_rcr") == num_hidden_layers
    assert node_ops.count("bmm_rrr") == num_hidden_layers
    for op_name, expected_count in expected_counts.items():
        assert node_ops.count(op_name) == expected_count
    dynamic_slice_node = next(node for node in spec.ir["nodes"] if node["op"] == "dynamic_slice")
    assert dynamic_slice_node["attrs"] == {"start_indices": [0, 0, 0], "slice_sizes": [1, SEQ_LEN, SEQ_LEN]}
    assert spec.ir["outputs"][0]["name"] == "text_features"
    assert spec.ir["outputs"][0]["shape"] == [BATCH, PROJECTION]
    assert [entry["name"] for entry in spec.ir["inputs"]] == ["input_ids", "attention_mask"] + (
        ["position_ids"] if include_position_ids else []
    )

    inputs = {
        "input_ids": _input_ids(eos_token_id=eos_token_id),
        "attention_mask": _attention_mask(),
    }
    if include_position_ids:
        inputs["position_ids"] = _position_ids()

    actual = execute_cpu(spec, inputs)["text_features"]
    expected = _reference_outputs(
        eos_token_id=eos_token_id,
        include_position_ids=include_position_ids,
        num_hidden_layers=num_hidden_layers,
    )

    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(
    ("eos_token_id", "expected_counts", "include_position_ids"),
    [
        (2, {"argmax": 1, "batch_gather": 1, "eq": 0}, True),
        (2, {"argmax": 1, "batch_gather": 1, "eq": 0}, False),
        (7, {"argmax": 1, "batch_gather": 1, "eq": 1}, True),
        (7, {"argmax": 1, "batch_gather": 1, "eq": 1}, False),
    ],
)
def test_clip_text_wrapper_zero_layer_matches_local_transformers(
    eos_token_id, expected_counts, include_position_ids
):
    spec = _trace(
        eos_token_id=eos_token_id,
        include_position_ids=include_position_ids,
        num_hidden_layers=0,
    )
    node_ops = [node["op"] for node in spec.ir["nodes"]]

    assert node_ops.count("embedding") == 2
    assert node_ops.count("dynamic_slice") == 1
    assert node_ops.count("layer_norm") == 1
    assert node_ops.count("gemm_rcr") == 1
    assert node_ops.count("gemm_rcr_bias") == 0
    assert node_ops.count("gemm_rcr_bias_fast_gelu") == 0
    assert node_ops.count("bmm_rcr") == 0
    assert node_ops.count("bmm_rrr") == 0
    for op_name, expected_count in expected_counts.items():
        assert node_ops.count(op_name) == expected_count

    inputs = {
        "input_ids": _input_ids(eos_token_id=eos_token_id),
        "attention_mask": _attention_mask(),
    }
    if include_position_ids:
        inputs["position_ids"] = _position_ids()

    actual = execute_cpu(spec, inputs)["text_features"]
    expected = _reference_outputs(
        eos_token_id=eos_token_id,
        include_position_ids=include_position_ids,
        num_hidden_layers=0,
    )

    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_clip_text_wrapper_zero_layer_cpu_artifact_matches_local_transformers(tmp_path, monkeypatch):
    from dinoml import runtime

    monkeypatch.setenv("DINOML_CACHE_DIR", str(tmp_path / "cache"))
    spec = _trace(num_hidden_layers=0)
    artifact = dml.compile(spec, dml.Target("cpu"), tmp_path / "clip_text_wrapper_zero_layer_cpu.dinoml")

    module = runtime.load(artifact.path)
    session = module.create_session()
    try:
        actual = session.run_numpy(
            {
                "input_ids": _input_ids(),
                "attention_mask": _attention_mask(),
                "position_ids": _position_ids(),
            }
        )["text_features"]
    finally:
        session.close()
        module.close()

    expected = _reference_outputs(num_hidden_layers=0)
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_clip_text_wrapper_cpu_compile_boundary_stays_honest(tmp_path, monkeypatch):
    monkeypatch.setenv("DINOML_CACHE_DIR", str(tmp_path / "cache"))
    spec = _trace()
    with pytest.raises(NotImplementedError, match="cpu backend does not support op gemm_rcr_bias_fast_gelu"):
        dml.compile(spec, dml.Target("cpu"), tmp_path / "clip_text_wrapper_cpu.dinoml")


@pytest.mark.parametrize("eos_token_id", [2, 7])
def test_clip_text_wrapper_manifest_keeps_provider_and_model_kernels_honest(eos_token_id):
    spec = _trace(eos_token_id=eos_token_id, num_hidden_layers=2)
    lowered, _ = PassManager().run(spec.ir)
    validate_ir(lowered)
    tensor_map = {tensor["name"]: tensor for tensor in lowered["tensors"]}
    cuda_sources = collect_generated_sources("cuda", lowered["nodes"], tensor_map)

    manifest = build_kernel_manifest(lowered, {"name": "cuda", "arch": "sm_86"})
    required = manifest["required_kernels"]
    ops = [entry["op"] for entry in required]

    assert "embedding" in ops
    assert "layer_norm" in ops
    assert "softmax" in ops
    assert "argmax" in ops
    assert "batch_gather" in ops
    if eos_token_id != 2:
        assert "fused_elementwise" in ops
    assert "gemm_rcr_bias" in ops
    assert "gemm_rcr_bias_fast_gelu" in ops
    assert "gemm_rcr" in ops
    assert "bmm_rcr" in ops
    assert "bmm_rrr" in ops

    provider_ops = {"gemm_rcr_bias", "gemm_rcr_bias_fast_gelu", "gemm_rcr", "bmm_rcr", "bmm_rrr"}
    provider_entries = [entry for entry in required if entry["op"] in provider_ops]
    model_entries = [entry for entry in required if entry["op"] not in provider_ops]

    assert provider_entries
    assert all(entry["kernel_library"] in {"cutlass_gemm", "cutlass_bmm"} for entry in provider_entries)
    assert any(entry["op"] == "gemm_rcr" for entry in provider_entries)
    assert any(entry["op"] == "gemm_rcr_bias_fast_gelu" for entry in provider_entries)
    assert model_entries
    assert all(entry["kernel_library"] == "model" for entry in model_entries)
    assert len(cuda_sources["kernels"]) >= 7
    assert any("static int dynamic_slice_" in source for source in cuda_sources["kernels"])
    assert any("static int embedding_" in source for source in cuda_sources["kernels"])
    assert any("static int layer_norm_" in source for source in cuda_sources["kernels"])
    assert any("static int softmax_" in source for source in cuda_sources["kernels"])
    assert any("static int argmax_" in source for source in cuda_sources["kernels"])
    assert any("static int batch_gather_" in source for source in cuda_sources["kernels"])
    if eos_token_id != 2:
        assert any("const int64_t* DINO_RESTRICT ptr_input_ids" in source for source in cuda_sources["kernels"])
        assert any("dinoml::math::eq(" in source for source in cuda_sources["kernels"])


@pytest.mark.skipif(shutil.which("nvcc") is None, reason="nvcc is required")
@pytest.mark.skipif(
    os.environ.get("DINOML_RUN_EXPENSIVE_CUDA_CLIP_WRAPPER") != "1",
    reason="set DINOML_RUN_EXPENSIVE_CUDA_CLIP_WRAPPER=1 to run the expensive CUDA wrapper runtime smoke",
)
def test_clip_text_wrapper_generated_cuda_runtime_matches_local_transformers(tmp_path, monkeypatch):
    from dinoml import runtime

    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device is required")

    monkeypatch.setenv("DINOML_CACHE_DIR", str(tmp_path / "cache"))
    spec = _trace()
    artifact = dml.compile(
        spec,
        dml.Target("cuda", arch="sm_86", no_tf32=True),
        tmp_path / "clip_text_wrapper_cuda.dinoml",
    )

    module = runtime.load(artifact.path)
    session = module.create_session()
    try:
        actual = session.run_numpy(
            {
                "input_ids": _input_ids(),
                "attention_mask": _attention_mask(),
                "position_ids": _position_ids(),
            }
        )["text_features"]
    finally:
        session.close()
        module.close()

    expected = _reference_outputs()
    np.testing.assert_allclose(actual, expected, atol=5e-4, rtol=2e-3)
