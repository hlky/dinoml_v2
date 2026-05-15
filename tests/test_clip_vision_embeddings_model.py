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
from dinoml import runtime
from dinoml.backends.cpu import execute_cpu
from dinoml.kernels.manifest import build_kernel_manifest
from dinoml.lowering.ops import collect_generated_sources
from dinoml.models.clip import LegacyCLIPVisionEmbeddings, LegacyCLIPVisionEmbeddingsConfig
from dinoml.passes import PassManager, validate_ir


BATCH = 2
NUM_CHANNELS = 3
IMAGE_SIZE = 4
PATCH_SIZE = 2
HIDDEN = 6
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2


def _config():
    return LegacyCLIPVisionEmbeddingsConfig(
        hidden_size=HIDDEN,
        image_size=IMAGE_SIZE,
        patch_size=PATCH_SIZE,
        num_channels=NUM_CHANNELS,
    )


def _weights():
    rng = np.random.default_rng(2047)

    def _normal(shape, scale):
        return (rng.standard_normal(shape).astype(np.float32) / scale).astype(np.float32)

    config = _config()
    return {
        "vision_model.embeddings.class_embedding": _normal((config.hidden_size,), 4.0),
        "vision_model.embeddings.patch_embedding.weight": _normal(
            (config.hidden_size, config.num_channels, config.patch_size, config.patch_size),
            5.0,
        ),
        "vision_model.embeddings.position_embedding.weight": _normal(
            (config.num_positions, config.hidden_size),
            4.5,
        ),
    }


WEIGHTS = _weights()


def _pixel_values():
    values = np.linspace(
        -1.25,
        1.75,
        num=BATCH * NUM_CHANNELS * IMAGE_SIZE * IMAGE_SIZE,
        dtype=np.float32,
    )
    return values.reshape(BATCH, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)


def _trace():
    return dml.trace(
        LegacyCLIPVisionEmbeddings(_config(), WEIGHTS),
        inputs={"pixel_values": dml.TensorSpec([BATCH, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE], "float32")},
        name="clip_vision_embeddings",
    )


class _ZeroBiasPatchProjection(dml.Module):
    def __init__(self, config: LegacyCLIPVisionEmbeddingsConfig, weights):
        self.weight = dml.Parameter(
            [config.hidden_size, config.num_channels, config.patch_size, config.patch_size],
            dtype="float32",
            value=np.asarray(weights["vision_model.embeddings.patch_embedding.weight"], dtype=np.float32),
        )
        self.zero_bias = dml.Parameter(
            [config.hidden_size],
            dtype="float32",
            value=np.zeros((config.hidden_size,), dtype=np.float32),
        )
        self.patch_size = config.patch_size

    def forward(self, pixel_values):
        patches = dml.ops.conv2d_bias(
            pixel_values,
            self.weight,
            self.zero_bias,
            stride=(self.patch_size, self.patch_size),
            padding=(0, 0),
            dilation=(1, 1),
            groups=1,
        )
        return dml.ops.output(patches, "patches")


def _trace_patch_projection():
    return dml.trace(
        _ZeroBiasPatchProjection(_config(), WEIGHTS),
        inputs={"pixel_values": dml.TensorSpec([BATCH, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE], "float32")},
        name="clip_vision_patch_projection_zero_bias",
    )


def _reference_embeddings():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    config = transformers.CLIPVisionConfig(
        hidden_size=HIDDEN,
        intermediate_size=16,
        num_attention_heads=2,
        num_hidden_layers=1,
        image_size=IMAGE_SIZE,
        patch_size=PATCH_SIZE,
        num_channels=NUM_CHANNELS,
    )
    module = transformers.models.clip.modeling_clip.CLIPVisionEmbeddings(config)
    with torch.no_grad():
        module.class_embedding.copy_(
            torch.from_numpy(np.asarray(WEIGHTS["vision_model.embeddings.class_embedding"], dtype=np.float32))
        )
        module.patch_embedding.weight.copy_(
            torch.from_numpy(np.asarray(WEIGHTS["vision_model.embeddings.patch_embedding.weight"], dtype=np.float32))
        )
        module.position_embedding.weight.copy_(
            torch.from_numpy(np.asarray(WEIGHTS["vision_model.embeddings.position_embedding.weight"], dtype=np.float32))
        )
    module.eval()
    with torch.inference_mode():
        return module(torch.from_numpy(_pixel_values())).detach().cpu().numpy().astype(np.float32)


def _reference_patch_projection():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    config = transformers.CLIPVisionConfig(
        hidden_size=HIDDEN,
        intermediate_size=16,
        num_attention_heads=2,
        num_hidden_layers=1,
        image_size=IMAGE_SIZE,
        patch_size=PATCH_SIZE,
        num_channels=NUM_CHANNELS,
    )
    module = transformers.models.clip.modeling_clip.CLIPVisionEmbeddings(config)
    with torch.no_grad():
        module.patch_embedding.weight.copy_(
            torch.from_numpy(np.asarray(WEIGHTS["vision_model.embeddings.patch_embedding.weight"], dtype=np.float32))
        )
    module.eval()
    with torch.inference_mode():
        return module.patch_embedding(torch.from_numpy(_pixel_values())).detach().cpu().numpy().astype(np.float32)


def test_clip_vision_embeddings_wrapper_matches_local_transformers():
    spec = _trace()
    node_ops = [node["op"] for node in spec.ir["nodes"]]

    assert node_ops == ["conv2d_bias", "permute021", "expand", "concatenate", "embedding", "add"]
    assert spec.ir["outputs"][0]["name"] == "embeddings"
    assert spec.ir["outputs"][0]["shape"] == [BATCH, NUM_PATCHES + 1, HIDDEN]
    assert spec.ir["outputs"][0]["shape_spec"] == [BATCH, NUM_PATCHES + 1, HIDDEN]
    assert spec.ir["outputs"][0]["dtype"] == "float32"

    actual = execute_cpu(spec, {"pixel_values": _pixel_values()})["embeddings"]
    expected = _reference_embeddings()

    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_clip_vision_patch_projection_zero_bias_matches_transformers_bias_free_conv():
    spec = _trace_patch_projection()
    node_ops = [node["op"] for node in spec.ir["nodes"]]

    assert node_ops == ["conv2d_bias"]
    assert spec.ir["outputs"][0]["shape"] == [BATCH, HIDDEN, IMAGE_SIZE // PATCH_SIZE, IMAGE_SIZE // PATCH_SIZE]

    actual = execute_cpu(spec, {"pixel_values": _pixel_values()})["patches"]
    expected = _reference_patch_projection()

    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_clip_vision_embeddings_cpu_artifact_matches_local_transformers(tmp_path, monkeypatch):
    monkeypatch.setenv("DINOML_CACHE_DIR", str(tmp_path / "cache"))
    spec = _trace()
    artifact = dml.compile(spec, dml.Target("cpu"), tmp_path / "clip_vision_embeddings_cpu.dinoml")

    generated = (artifact.path / "debug" / "generated_src" / "module.cpp").read_text(encoding="utf-8")
    assert "static int conv2d_bias_" in generated

    module = runtime.load(artifact.path)
    session = module.create_session()
    try:
        actual = session.run_numpy({"pixel_values": _pixel_values()})["embeddings"]
    finally:
        session.close()
        module.close()

    expected = _reference_embeddings()
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_clip_vision_embeddings_manifest_keeps_conv_provider_and_model_kernels_honest():
    spec = _trace()
    lowered, _ = PassManager().run(spec.ir)
    validate_ir(lowered)
    tensor_map = {tensor["name"]: tensor for tensor in lowered["tensors"]}
    cuda_sources = collect_generated_sources("cuda", lowered["nodes"], tensor_map)

    manifest = build_kernel_manifest(lowered, {"name": "cuda", "arch": "sm_86"})
    required = manifest["required_kernels"]
    ops = [entry["op"] for entry in required]

    assert "conv2d_bias" in ops
    assert "permute021" in ops
    assert "expand" in ops
    assert "concatenate" in ops
    assert "embedding" in ops
    assert "fused_elementwise" in ops

    provider_entries = [entry for entry in required if entry["op"] == "conv2d_bias"]
    model_entries = [entry for entry in required if entry["op"] != "conv2d_bias"]

    assert len(provider_entries) == 1
    assert provider_entries[0]["kernel_library"] == "cutlass_conv"
    assert provider_entries[0]["cutlass_conv_plan"]["selected_candidate"]["kernel_symbol"] == provider_entries[0]["kernel_symbol"]
    assert provider_entries[0]["cutlass_conv_plan"]["status"] == "bounded_runtime"
    assert provider_entries[0]["cutlass_conv_plan"]["selected_candidate"]["opclass"] == "simt"
    assert provider_entries[0]["cutlass_conv_plan"]["weight_transform"]["padded_input_channels"] == NUM_CHANNELS

    assert model_entries
    assert all(entry["kernel_library"] == "model" for entry in model_entries)

    assert len(cuda_sources["kernels"]) >= 5
    assert any("embedding_" in source for source in cuda_sources["kernels"])
    assert any("permute021_" in source for source in cuda_sources["kernels"])
    assert any("expand_" in source for source in cuda_sources["kernels"])
    assert any("concatenate_" in source for source in cuda_sources["kernels"])
    assert any("dinoml::math::add" in source for source in cuda_sources["kernels"])


def test_clip_vision_patch_projection_exact_float32_cuda_runtime_boundary_matches_transformers(
    tmp_path, use_shared_dinoml_cuda_cache
):
    torch = pytest.importorskip("torch")

    spec = _trace_patch_projection()

    manifest = build_kernel_manifest(spec.ir, {"name": "cuda", "arch": "sm_86"})
    [required] = manifest["required_kernels"]

    assert required["op"] == "conv2d_bias"
    assert required["candidate_set"]["status"] == "bounded_runtime"
    assert required["candidate_set"]["profiler_status"] == "bounded_runtime_profiler"
    assert "profiler_blocked_reason" not in required["candidate_set"]
    assert required["cutlass_conv_plan"]["status"] == "bounded_runtime"
    assert required["cutlass_conv_plan"]["profiler_status"] == "bounded_runtime_profiler"
    assert required["selected_candidate_id"].endswith("simt_sm80_nhwc_ohwi_bias")
    assert required["cutlass_conv_plan"]["selected_candidate"]["opclass"] == "simt"
    assert required["cutlass_conv_plan"]["selected_candidate"]["iterator_algorithm"] == "analytic"
    assert required["cutlass_conv_plan"]["runtime"]["launcher"] == "cutlass_implicit_gemm_conv2d_fprop_bias"
    assert "blocked_reason" not in required["cutlass_conv_plan"]
    assert required["cutlass_conv_plan"]["input_shape"] == [BATCH, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE]
    assert required["cutlass_conv_plan"]["weight_shape"] == [HIDDEN, NUM_CHANNELS, PATCH_SIZE, PATCH_SIZE]
    assert required["cutlass_conv_plan"]["output_shape"] == [BATCH, HIDDEN, IMAGE_SIZE // PATCH_SIZE, IMAGE_SIZE // PATCH_SIZE]
    assert required["cutlass_conv_plan"]["conv_config"] == {
        "stride": [PATCH_SIZE, PATCH_SIZE],
        "padding": [0, 0],
        "dilation": [1, 1],
        "groups": 1,
    }

    if shutil.which("nvcc") is None or not torch.cuda.is_available():
        return

    artifact = dml.compile(spec, dml.Target("cuda", arch="sm_86"), tmp_path / "clip_vision_patch_projection_cuda.dinoml")
    module = runtime.load(artifact.path)
    session = module.create_session()
    try:
        actual = session.run_numpy({"pixel_values": _pixel_values()})["patches"]
    finally:
        session.close()
        module.close()

    expected = _reference_patch_projection()
    np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-4)
