from __future__ import annotations

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
from dinoml.kernels.codegen import create_codegen_plan
from dinoml.kernels.manifest import build_kernel_manifest
from dinoml.models.clip import LegacyCLIPModel, LegacyCLIPTextConfig, LegacyCLIPVisionConfig
from dinoml.passes import PassManager, validate_ir


LOCAL_TRANSFORMERS_SRC = Path("/workspace/transformers/src")
TEXT_BATCH = 2
IMAGE_BATCH = 3
SEQ_LEN = 4
VOCAB_SIZE = 16
TEXT_HIDDEN = 6
VISION_HIDDEN = 6
NUM_HEADS = 2
TEXT_INTERMEDIATE = 8
VISION_INTERMEDIATE = 10
PROJECTION = 5
EPS = 1.0e-5
MAX_POSITION_EMBEDDINGS = 6
NUM_CHANNELS = 3
IMAGE_SIZE = 4
PATCH_SIZE = 2


def _text_config(*, num_hidden_layers: int = 2):
    return LegacyCLIPTextConfig(
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=MAX_POSITION_EMBEDDINGS,
        hidden_size=TEXT_HIDDEN,
        intermediate_size=TEXT_INTERMEDIATE,
        num_attention_heads=NUM_HEADS,
        num_hidden_layers=num_hidden_layers,
        projection_dim=PROJECTION,
        layer_norm_eps=EPS,
        eos_token_id=2,
    )


def _vision_config(*, num_hidden_layers: int = 2):
    return LegacyCLIPVisionConfig(
        hidden_size=VISION_HIDDEN,
        intermediate_size=VISION_INTERMEDIATE,
        num_attention_heads=NUM_HEADS,
        num_hidden_layers=num_hidden_layers,
        projection_dim=PROJECTION,
        image_size=IMAGE_SIZE,
        patch_size=PATCH_SIZE,
        num_channels=NUM_CHANNELS,
        layer_norm_eps=EPS,
    )


def _weights():
    rng = np.random.default_rng(2089)

    def _normal(shape, scale):
        return (rng.standard_normal(shape).astype(np.float32) / scale).astype(np.float32)

    text_config = _text_config()
    vision_config = _vision_config()
    weights = {
        "text_model.embeddings.token_embedding.weight": _normal((text_config.vocab_size, text_config.hidden_size), 3.5),
        "text_model.embeddings.position_embedding.weight": _normal(
            (text_config.max_position_embeddings, text_config.hidden_size), 4.0
        ),
        "text_model.final_layer_norm.weight": _normal((text_config.hidden_size,), 4.0),
        "text_model.final_layer_norm.bias": _normal((text_config.hidden_size,), 6.0),
        "text_projection.weight": _normal((text_config.projection_dim, text_config.hidden_size), 4.0),
        "vision_model.embeddings.class_embedding": _normal((vision_config.hidden_size,), 4.0),
        "vision_model.embeddings.patch_embedding.weight": _normal(
            (
                vision_config.hidden_size,
                vision_config.num_channels,
                vision_config.patch_size,
                vision_config.patch_size,
            ),
            5.0,
        ),
        "vision_model.embeddings.position_embedding.weight": _normal(
            (vision_config.num_positions, vision_config.hidden_size),
            4.5,
        ),
        "vision_model.pre_layrnorm.weight": _normal((vision_config.hidden_size,), 3.5),
        "vision_model.pre_layrnorm.bias": _normal((vision_config.hidden_size,), 5.5),
        "vision_model.post_layernorm.weight": _normal((vision_config.hidden_size,), 3.5),
        "vision_model.post_layernorm.bias": _normal((vision_config.hidden_size,), 5.5),
        "visual_projection.weight": _normal((vision_config.projection_dim, vision_config.hidden_size), 4.0),
        "logit_scale": np.array(np.log(1.7), dtype=np.float32),
    }
    for layer_idx in range(text_config.num_hidden_layers):
        prefix = f"text_model.encoder.layers.{layer_idx}"
        weights[f"{prefix}.self_attn.q_proj.weight"] = _normal((text_config.hidden_size, text_config.hidden_size), 5.0)
        weights[f"{prefix}.self_attn.q_proj.bias"] = _normal((text_config.hidden_size,), 7.0)
        weights[f"{prefix}.self_attn.k_proj.weight"] = _normal((text_config.hidden_size, text_config.hidden_size), 5.0)
        weights[f"{prefix}.self_attn.k_proj.bias"] = _normal((text_config.hidden_size,), 7.0)
        weights[f"{prefix}.self_attn.v_proj.weight"] = _normal((text_config.hidden_size, text_config.hidden_size), 5.0)
        weights[f"{prefix}.self_attn.v_proj.bias"] = _normal((text_config.hidden_size,), 7.0)
        weights[f"{prefix}.self_attn.out_proj.weight"] = _normal((text_config.hidden_size, text_config.hidden_size), 5.0)
        weights[f"{prefix}.self_attn.out_proj.bias"] = _normal((text_config.hidden_size,), 7.0)
        weights[f"{prefix}.layer_norm1.weight"] = _normal((text_config.hidden_size,), 4.0)
        weights[f"{prefix}.layer_norm1.bias"] = _normal((text_config.hidden_size,), 6.0)
        weights[f"{prefix}.mlp.fc1.weight"] = _normal((text_config.intermediate_size, text_config.hidden_size), 4.5)
        weights[f"{prefix}.mlp.fc1.bias"] = _normal((text_config.intermediate_size,), 6.5)
        weights[f"{prefix}.mlp.fc2.weight"] = _normal((text_config.hidden_size, text_config.intermediate_size), 4.5)
        weights[f"{prefix}.mlp.fc2.bias"] = _normal((text_config.hidden_size,), 6.5)
        weights[f"{prefix}.layer_norm2.weight"] = _normal((text_config.hidden_size,), 4.0)
        weights[f"{prefix}.layer_norm2.bias"] = _normal((text_config.hidden_size,), 6.0)
    for layer_idx in range(vision_config.num_hidden_layers):
        prefix = f"vision_model.encoder.layers.{layer_idx}"
        weights[f"{prefix}.layer_norm1.weight"] = _normal((vision_config.hidden_size,), 4.0)
        weights[f"{prefix}.layer_norm1.bias"] = _normal((vision_config.hidden_size,), 6.0)
        weights[f"{prefix}.self_attn.q_proj.weight"] = _normal((vision_config.hidden_size, vision_config.hidden_size), 5.0)
        weights[f"{prefix}.self_attn.q_proj.bias"] = _normal((vision_config.hidden_size,), 7.0)
        weights[f"{prefix}.self_attn.k_proj.weight"] = _normal((vision_config.hidden_size, vision_config.hidden_size), 5.0)
        weights[f"{prefix}.self_attn.k_proj.bias"] = _normal((vision_config.hidden_size,), 7.0)
        weights[f"{prefix}.self_attn.v_proj.weight"] = _normal((vision_config.hidden_size, vision_config.hidden_size), 5.0)
        weights[f"{prefix}.self_attn.v_proj.bias"] = _normal((vision_config.hidden_size,), 7.0)
        weights[f"{prefix}.self_attn.out_proj.weight"] = _normal((vision_config.hidden_size, vision_config.hidden_size), 5.0)
        weights[f"{prefix}.self_attn.out_proj.bias"] = _normal((vision_config.hidden_size,), 7.0)
        weights[f"{prefix}.layer_norm2.weight"] = _normal((vision_config.hidden_size,), 4.0)
        weights[f"{prefix}.layer_norm2.bias"] = _normal((vision_config.hidden_size,), 6.0)
        weights[f"{prefix}.mlp.fc1.weight"] = _normal((vision_config.intermediate_size, vision_config.hidden_size), 4.0)
        weights[f"{prefix}.mlp.fc1.bias"] = _normal((vision_config.intermediate_size,), 6.0)
        weights[f"{prefix}.mlp.fc2.weight"] = _normal((vision_config.hidden_size, vision_config.intermediate_size), 4.0)
        weights[f"{prefix}.mlp.fc2.bias"] = _normal((vision_config.hidden_size,), 6.0)
    return weights


WEIGHTS = _weights()


def _input_ids():
    return np.array(
        [
            [0, 5, 15, 1],
            [0, 15, 4, 1],
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


def _pixel_values():
    values = np.linspace(
        -1.5,
        1.5,
        num=IMAGE_BATCH * NUM_CHANNELS * IMAGE_SIZE * IMAGE_SIZE,
        dtype=np.float32,
    )
    return values.reshape(IMAGE_BATCH, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)


class _CLIPTextFeaturesModule(dml.Module):
    def __init__(self):
        self.model = LegacyCLIPModel(_text_config(), _vision_config(), WEIGHTS)

    def forward(self, input_ids, attention_mask):
        return dml.ops.output(self.model.get_text_features(input_ids, attention_mask), "text_features")


class _CLIPImageFeaturesModule(dml.Module):
    def __init__(self):
        self.model = LegacyCLIPModel(_text_config(), _vision_config(), WEIGHTS)

    def forward(self, pixel_values):
        return dml.ops.output(self.model.get_image_features(pixel_values), "image_features")


def _trace_text_features():
    return dml.trace(
        _CLIPTextFeaturesModule(),
        inputs={
            "input_ids": dml.TensorSpec([TEXT_BATCH, SEQ_LEN], "int64"),
            "attention_mask": dml.TensorSpec([TEXT_BATCH, SEQ_LEN], "bool"),
        },
        name="clip_model_text_features",
    )


def _trace_image_features():
    return dml.trace(
        _CLIPImageFeaturesModule(),
        inputs={"pixel_values": dml.TensorSpec([IMAGE_BATCH, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE], "float32")},
        name="clip_model_image_features",
    )


def _trace_model():
    return dml.trace(
        LegacyCLIPModel(_text_config(), _vision_config(), WEIGHTS),
        inputs={
            "input_ids": dml.TensorSpec([TEXT_BATCH, SEQ_LEN], "int64"),
            "pixel_values": dml.TensorSpec([IMAGE_BATCH, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE], "float32"),
            "attention_mask": dml.TensorSpec([TEXT_BATCH, SEQ_LEN], "bool"),
        },
        name="clip_model_two_tower",
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


def _reference_outputs(*, text_num_hidden_layers: int = 2, vision_num_hidden_layers: int = 2):
    torch = pytest.importorskip("torch")
    transformers = _import_local_transformers()

    text_config = transformers.CLIPTextConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=TEXT_HIDDEN,
        intermediate_size=TEXT_INTERMEDIATE,
        projection_dim=PROJECTION,
        num_attention_heads=NUM_HEADS,
        num_hidden_layers=text_num_hidden_layers,
        max_position_embeddings=MAX_POSITION_EMBEDDINGS,
        hidden_act="quick_gelu",
        attention_dropout=0.0,
        layer_norm_eps=EPS,
        bos_token_id=0,
        eos_token_id=2,
        pad_token_id=1,
    )
    vision_config = transformers.CLIPVisionConfig(
        hidden_size=VISION_HIDDEN,
        intermediate_size=VISION_INTERMEDIATE,
        projection_dim=PROJECTION,
        num_attention_heads=NUM_HEADS,
        num_hidden_layers=vision_num_hidden_layers,
        image_size=IMAGE_SIZE,
        patch_size=PATCH_SIZE,
        num_channels=NUM_CHANNELS,
        hidden_act="quick_gelu",
        attention_dropout=0.0,
        layer_norm_eps=EPS,
    )
    clip_config = transformers.CLIPConfig(
        text_config=text_config.to_dict(),
        vision_config=vision_config.to_dict(),
        projection_dim=PROJECTION,
        logit_scale_init_value=float(np.asarray(WEIGHTS["logit_scale"], dtype=np.float32)),
    )
    clip_model = transformers.CLIPModel(clip_config)

    def _load(model):
        state_dict = model.state_dict()
        for name, value in WEIGHTS.items():
            if name == "logit_scale" and name in state_dict:
                state_dict[name] = torch.tensor(float(np.asarray(value, dtype=np.float32)), dtype=torch.float32)
            elif name in state_dict:
                state_dict[name] = torch.from_numpy(np.asarray(value, dtype=np.float32))
        model.load_state_dict(state_dict)
        model.eval()
        return model

    clip_model = _load(clip_model)

    text_inputs = {
        "input_ids": torch.from_numpy(_input_ids()),
        "attention_mask": torch.from_numpy(_attention_mask()),
    }
    image_inputs = {
        "pixel_values": torch.from_numpy(_pixel_values()),
    }

    with torch.inference_mode():
        text_features = clip_model.get_text_features(**text_inputs).pooler_output
        image_features = clip_model.get_image_features(**image_inputs).pooler_output
        outputs = clip_model(**text_inputs, **image_inputs)
    return {
        "text_features": text_features.detach().cpu().numpy().astype(np.float32),
        "image_features": image_features.detach().cpu().numpy().astype(np.float32),
        "logits_per_image": outputs.logits_per_image.detach().cpu().numpy().astype(np.float32),
        "logits_per_text": outputs.logits_per_text.detach().cpu().numpy().astype(np.float32),
        "text_embeds": outputs.text_embeds.detach().cpu().numpy().astype(np.float32),
        "image_embeds": outputs.image_embeds.detach().cpu().numpy().astype(np.float32),
    }


def test_clip_model_get_text_and_image_features_match_local_transformers():
    text_spec = _trace_text_features()
    image_spec = _trace_image_features()

    assert text_spec.ir["outputs"][0]["name"] == "text_features"
    assert text_spec.ir["outputs"][0]["shape"] == [TEXT_BATCH, PROJECTION]
    assert image_spec.ir["outputs"][0]["name"] == "image_features"
    assert image_spec.ir["outputs"][0]["shape"] == [IMAGE_BATCH, PROJECTION]

    expected = _reference_outputs()
    actual_text = execute_cpu(
        text_spec,
        {
            "input_ids": _input_ids(),
            "attention_mask": _attention_mask(),
        },
    )["text_features"]
    actual_image = execute_cpu(image_spec, {"pixel_values": _pixel_values()})["image_features"]

    np.testing.assert_allclose(actual_text, expected["text_features"], atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(actual_image, expected["image_features"], atol=1e-5, rtol=1e-5)


def test_clip_model_two_tower_logits_and_normalized_embeds_match_local_transformers():
    spec = _trace_model()
    node_ops = [node["op"] for node in spec.ir["nodes"]]

    assert node_ops.count("conv2d_bias") == 1
    assert node_ops.count("embedding") == 3
    assert node_ops.count("vector_norm") == 2
    assert node_ops.count("div") == 2
    assert node_ops.count("gemm_rcr") == 3
    assert node_ops.count("gemm_rcr_bias") == 5 * (_text_config().num_hidden_layers + _vision_config().num_hidden_layers)
    assert node_ops.count("gemm_rcr_bias_fast_gelu") == _text_config().num_hidden_layers + _vision_config().num_hidden_layers
    assert node_ops.count("bmm_rcr") == _text_config().num_hidden_layers + _vision_config().num_hidden_layers
    assert node_ops.count("bmm_rrr") == _text_config().num_hidden_layers + _vision_config().num_hidden_layers
    assert node_ops.count("exp") == 1
    assert [output["name"] for output in spec.ir["outputs"]] == [
        "logits_per_image",
        "logits_per_text",
        "text_embeds",
        "image_embeds",
    ]
    assert spec.ir["outputs"][0]["shape"] == [IMAGE_BATCH, TEXT_BATCH]
    assert spec.ir["outputs"][1]["shape"] == [TEXT_BATCH, IMAGE_BATCH]
    assert spec.ir["outputs"][2]["shape"] == [TEXT_BATCH, PROJECTION]
    assert spec.ir["outputs"][3]["shape"] == [IMAGE_BATCH, PROJECTION]

    actual = execute_cpu(
        spec,
        {
            "input_ids": _input_ids(),
            "pixel_values": _pixel_values(),
            "attention_mask": _attention_mask(),
        },
    )
    expected = _reference_outputs()

    np.testing.assert_allclose(actual["logits_per_image"], expected["logits_per_image"], atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(actual["logits_per_text"], expected["logits_per_text"], atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(actual["text_embeds"], expected["text_embeds"], atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(actual["image_embeds"], expected["image_embeds"], atol=1e-5, rtol=1e-5)


def test_clip_model_zero_layer_text_tower_matches_local_transformers():
    text_config = _text_config(num_hidden_layers=0)
    vision_config = _vision_config(num_hidden_layers=0)
    spec = dml.trace(
        LegacyCLIPModel(text_config, vision_config, WEIGHTS),
        inputs={
            "input_ids": dml.TensorSpec([TEXT_BATCH, SEQ_LEN], "int64"),
            "pixel_values": dml.TensorSpec([IMAGE_BATCH, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE], "float32"),
            "attention_mask": dml.TensorSpec([TEXT_BATCH, SEQ_LEN], "bool"),
        },
        name="clip_model_two_tower_zero_text_zero_vision",
    )
    node_ops = [node["op"] for node in spec.ir["nodes"]]

    assert node_ops.count("conv2d_bias") == 1
    assert node_ops.count("embedding") == 3
    assert node_ops.count("layer_norm") == 3
    assert node_ops.count("vector_norm") == 2
    assert node_ops.count("div") == 2
    assert node_ops.count("gemm_rcr") == 3
    assert node_ops.count("gemm_rcr_bias") == 0
    assert node_ops.count("gemm_rcr_bias_fast_gelu") == 0
    assert node_ops.count("bmm_rcr") == 0
    assert node_ops.count("bmm_rrr") == 0
    assert node_ops.count("exp") == 1

    actual = execute_cpu(
        spec,
        {
            "input_ids": _input_ids(),
            "pixel_values": _pixel_values(),
            "attention_mask": _attention_mask(),
        },
    )
    expected = _reference_outputs(text_num_hidden_layers=0, vision_num_hidden_layers=0)

    np.testing.assert_allclose(actual["logits_per_image"], expected["logits_per_image"], atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(actual["logits_per_text"], expected["logits_per_text"], atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(actual["text_embeds"], expected["text_embeds"], atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(actual["image_embeds"], expected["image_embeds"], atol=1e-5, rtol=1e-5)


def test_clip_model_two_tower_zero_text_zero_vision_cpu_compile_boundary_stays_honest(tmp_path, monkeypatch):
    monkeypatch.setenv("DINOML_CACHE_DIR", str(tmp_path / "cache"))
    spec = dml.trace(
        LegacyCLIPModel(_text_config(num_hidden_layers=0), _vision_config(num_hidden_layers=0), WEIGHTS),
        inputs={
            "input_ids": dml.TensorSpec([TEXT_BATCH, SEQ_LEN], "int64"),
            "pixel_values": dml.TensorSpec([IMAGE_BATCH, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE], "float32"),
            "attention_mask": dml.TensorSpec([TEXT_BATCH, SEQ_LEN], "bool"),
        },
        name="clip_model_two_tower_zero_text_zero_vision",
    )
    with pytest.raises(NotImplementedError, match="cpu backend does not support op conv2d_bias"):
        dml.compile(spec, dml.Target("cpu"), tmp_path / "clip_model_two_tower_zero_cpu.dinoml")


def test_clip_model_two_tower_cpu_compile_boundary_stays_honest(tmp_path, monkeypatch):
    monkeypatch.setenv("DINOML_CACHE_DIR", str(tmp_path / "cache"))
    spec = _trace_model()
    with pytest.raises(NotImplementedError, match="cpu backend does not support op conv2d_bias"):
        dml.compile(spec, dml.Target("cpu"), tmp_path / "clip_model_two_tower_cpu.dinoml")


def test_clip_model_manifest_keeps_provider_and_model_kernels_honest():
    spec = _trace_model()
    lowered, _ = PassManager().run(spec.ir)
    validate_ir(lowered)

    manifest = build_kernel_manifest(lowered, {"name": "cuda", "arch": "sm_86"})
    required = manifest["required_kernels"]
    ops = [entry["op"] for entry in required]

    assert "conv2d_bias" in ops
    assert "gemm_rcr_bias" in ops
    assert "gemm_rcr_bias_fast_gelu" in ops
    assert "bmm_rcr" in ops
    assert "bmm_rrr" in ops
    assert "gemm_rcr" in ops
    assert "vector_norm" in ops
    assert "softmax" in ops
    assert "layer_norm" in ops
    assert "embedding" in ops
    assert "permute" in ops

    provider_ops = {"conv2d_bias", "gemm_rcr_bias", "gemm_rcr_bias_fast_gelu", "bmm_rcr", "bmm_rrr", "gemm_rcr"}
    provider_entries = [entry for entry in required if entry["op"] in provider_ops]
    model_entries = [entry for entry in required if entry["op"] not in provider_ops]

    conv_entries = [entry for entry in provider_entries if entry["op"] == "conv2d_bias"]
    gemm_entries = [entry for entry in provider_entries if entry["op"] == "gemm_rcr"]
    assert len(conv_entries) == 1
    assert len(gemm_entries) >= 1
    assert conv_entries[0]["kernel_library"] == "cutlass_conv"
    assert conv_entries[0]["cutlass_conv_plan"]["status"] == "manifest_scaffold_only"
    assert all(entry["kernel_library"] == "cutlass_gemm" for entry in provider_entries if entry["op"] in {"gemm_rcr_bias", "gemm_rcr_bias_fast_gelu", "gemm_rcr"})
    assert all(entry["kernel_library"] == "cutlass_bmm" for entry in provider_entries if entry["op"] in {"bmm_rcr", "bmm_rrr"})
    assert model_entries
    assert all(entry["kernel_library"] == "model" for entry in model_entries)


def test_clip_model_codegen_plan_keeps_conv_scaffold_artifact_visible(tmp_path):
    spec = _trace_model()
    lowered, _ = PassManager().run(spec.ir)
    validate_ir(lowered)

    manifest = build_kernel_manifest(lowered, {"name": "cuda", "arch": "sm_86", "no_tf32": True})
    codegen_plan = create_codegen_plan(manifest, tmp_path / "cache")

    assert [entry["name"] for entry in codegen_plan.external_support_libraries] == [
        "cutlass_bmm",
        "cutlass_conv",
        "cutlass_gemm",
    ]

    conv_manifest_entry = next(entry for entry in manifest["required_kernels"] if entry["op"] == "conv2d_bias")
    conv_support_lib = next(
        entry for entry in codegen_plan.external_support_libraries if entry["name"] == "cutlass_conv"
    )

    assert conv_manifest_entry["cutlass_conv_plan"]["status"] == "manifest_scaffold_only"
    assert conv_support_lib["kernel_symbols"] == [conv_manifest_entry["kernel_symbol"]]
    assert conv_support_lib["profiler_symbols"] == [conv_manifest_entry["profiler_symbol"]]
    assert conv_support_lib["transform_helper_symbols"] == [
        "dinoml_cutlass_conv_input_pack_nchw_to_nhwc_float32_v1",
        "dinoml_cutlass_conv_output_unpack_nhwc_to_nchw_float32_v1",
        "dinoml_cutlass_conv_weight_pack_oihw_to_ohwi_float32_v1",
    ]
    assert [stage["stage_name"] for stage in codegen_plan.wrapper_stages] == [
        "activation_pack",
        "weight_pack",
        "provider_launch",
        "output_unpack",
    ]
    assert codegen_plan.wrapper_stages[2]["stage_kind"] == "provider_launcher"
    assert codegen_plan.wrapper_stages[2]["symbol"] == conv_manifest_entry["kernel_symbol"]
