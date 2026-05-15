import numpy as np
import pytest

import dinoml as dml
from dinoml.backends.cpu import execute_cpu
from dinoml.kernels.manifest import build_kernel_manifest
from dinoml.lowering.ops import collect_generated_sources
from dinoml.models.clip import LegacyCLIPVisionConfig, LegacyCLIPVisionModelWithProjection
from dinoml.passes import PassManager, validate_ir


BATCH = 2
NUM_CHANNELS = 3
IMAGE_SIZE = 4
PATCH_SIZE = 2
HIDDEN = 6
NUM_HEADS = 2
INTERMEDIATE = 10
PROJECTION = 5
EPS = 1.0e-5
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2


def _config():
    return LegacyCLIPVisionConfig(
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        num_attention_heads=NUM_HEADS,
        num_hidden_layers=0,
        projection_dim=PROJECTION,
        image_size=IMAGE_SIZE,
        patch_size=PATCH_SIZE,
        num_channels=NUM_CHANNELS,
        layer_norm_eps=EPS,
    )


def _weights():
    rng = np.random.default_rng(2053)

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
        "vision_model.pre_layrnorm.weight": _normal((config.hidden_size,), 3.5),
        "vision_model.pre_layrnorm.bias": _normal((config.hidden_size,), 5.5),
        "vision_model.post_layernorm.weight": _normal((config.hidden_size,), 3.5),
        "vision_model.post_layernorm.bias": _normal((config.hidden_size,), 5.5),
        "visual_projection.weight": _normal((config.projection_dim, config.hidden_size), 4.0),
    }


WEIGHTS = _weights()


def _pixel_values():
    values = np.linspace(
        -1.5,
        1.5,
        num=BATCH * NUM_CHANNELS * IMAGE_SIZE * IMAGE_SIZE,
        dtype=np.float32,
    )
    return values.reshape(BATCH, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)


def _trace():
    return dml.trace(
        LegacyCLIPVisionModelWithProjection(_config(), WEIGHTS),
        inputs={"pixel_values": dml.TensorSpec([BATCH, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE], "float32")},
        name="clip_vision_model_with_projection_zero_layer",
    )


def _reference_outputs():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    config = transformers.CLIPVisionConfig(
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        projection_dim=PROJECTION,
        num_attention_heads=NUM_HEADS,
        num_hidden_layers=0,
        image_size=IMAGE_SIZE,
        patch_size=PATCH_SIZE,
        num_channels=NUM_CHANNELS,
        hidden_act="quick_gelu",
        attention_dropout=0.0,
        layer_norm_eps=EPS,
    )
    model = transformers.CLIPVisionModelWithProjection(config)
    state_dict = model.state_dict()
    for name, value in WEIGHTS.items():
        state_dict[name] = torch.from_numpy(np.asarray(value, dtype=np.float32))
    model.load_state_dict(state_dict)
    model.eval()

    pixel_values = torch.from_numpy(_pixel_values())
    with torch.inference_mode():
        outputs = model(pixel_values=pixel_values)
        pooled = model.vision_model(pixel_values=pixel_values).pooler_output
    return {
        "last_hidden_state": outputs.last_hidden_state.detach().cpu().numpy().astype(np.float32),
        "pooler_output": pooled.detach().cpu().numpy().astype(np.float32),
        "image_features": outputs.image_embeds.detach().cpu().numpy().astype(np.float32),
    }


def test_clip_vision_wrapper_zero_layer_matches_local_transformers():
    spec = _trace()
    node_ops = [node["op"] for node in spec.ir["nodes"]]

    assert node_ops.count("conv2d_bias") == 1
    assert node_ops.count("permute021") == 1
    assert node_ops.count("expand") == 1
    assert node_ops.count("concatenate") == 1
    assert node_ops.count("embedding") == 1
    assert node_ops.count("add") == 1
    assert node_ops.count("layer_norm") == 2
    assert node_ops.count("dynamic_slice") == 1
    assert node_ops.count("gemm_rcr") == 1
    dynamic_slice_node = next(node for node in spec.ir["nodes"] if node["op"] == "dynamic_slice")
    assert dynamic_slice_node["attrs"] == {
        "start_indices": [0, 0, 0],
        "slice_sizes": [BATCH, 1, HIDDEN],
    }
    assert [output["name"] for output in spec.ir["outputs"]] == [
        "last_hidden_state",
        "pooler_output",
        "image_features",
    ]
    assert spec.ir["outputs"][0]["shape"] == [BATCH, NUM_PATCHES + 1, HIDDEN]
    assert spec.ir["outputs"][1]["shape"] == [BATCH, HIDDEN]
    assert spec.ir["outputs"][2]["shape"] == [BATCH, PROJECTION]

    actual = execute_cpu(spec, {"pixel_values": _pixel_values()})
    expected = _reference_outputs()

    np.testing.assert_allclose(actual["last_hidden_state"], expected["last_hidden_state"], atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(actual["pooler_output"], expected["pooler_output"], atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(actual["image_features"], expected["image_features"], atol=1e-5, rtol=1e-5)


def test_clip_vision_wrapper_cpu_compile_boundary_stays_honest(tmp_path, monkeypatch):
    monkeypatch.setenv("DINOML_CACHE_DIR", str(tmp_path / "cache"))
    spec = _trace()
    with pytest.raises(NotImplementedError, match="cpu backend does not support op conv2d_bias"):
        dml.compile(spec, dml.Target("cpu"), tmp_path / "clip_vision_model_with_projection_cpu.dinoml")


def test_clip_vision_wrapper_manifest_keeps_provider_and_model_kernels_honest():
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
    assert "layer_norm" in ops
    assert "dynamic_slice" in ops
    assert "gemm_rcr" in ops

    provider_ops = {"conv2d_bias", "gemm_rcr"}
    provider_entries = [entry for entry in required if entry["op"] in provider_ops]
    model_entries = [entry for entry in required if entry["op"] not in provider_ops]

    assert len(provider_entries) == 2
    conv_entry = next(entry for entry in provider_entries if entry["op"] == "conv2d_bias")
    gemm_entry = next(entry for entry in provider_entries if entry["op"] == "gemm_rcr")
    assert conv_entry["kernel_library"] == "cutlass_conv"
    assert conv_entry["cutlass_conv_plan"]["selected_candidate"]["kernel_symbol"] == conv_entry["kernel_symbol"]
    assert conv_entry["cutlass_conv_plan"]["status"] == "manifest_scaffold_only"
    assert gemm_entry["kernel_library"] == "cutlass_gemm"

    assert model_entries
    assert all(entry["kernel_library"] == "model" for entry in model_entries)

    assert any("embedding_" in source for source in cuda_sources["kernels"])
    assert any("permute021_" in source for source in cuda_sources["kernels"])
    assert any("dynamic_slice_" in source for source in cuda_sources["kernels"])
    assert any("layer_norm_" in source for source in cuda_sources["kernels"])
    assert any("dinoml::math::add" in source for source in cuda_sources["kernels"])
