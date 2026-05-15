from __future__ import annotations

import json
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
from dinoml import runtime
from dinoml.backends.cpu import execute_cpu
from dinoml.kernels.codegen import create_codegen_plan
from dinoml.kernels.manifest import build_kernel_manifest
from dinoml.models.clip import (
    LegacyCLIPModel,
    LegacyCLIPTextConfig,
    LegacyCLIPVisionConfig,
    legacy_clip_configs_from_transformers_clip_config,
    legacy_clip_model_from_transformers_clip_model,
    legacy_clip_weights_from_transformers_state_dict,
)
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


def _text_config(*, num_hidden_layers: int = 2, eos_token_id: int = 2):
    return LegacyCLIPTextConfig(
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=MAX_POSITION_EMBEDDINGS,
        hidden_size=TEXT_HIDDEN,
        intermediate_size=TEXT_INTERMEDIATE,
        num_attention_heads=NUM_HEADS,
        num_hidden_layers=num_hidden_layers,
        projection_dim=PROJECTION,
        layer_norm_eps=EPS,
        eos_token_id=eos_token_id,
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


def _input_ids_for_eos(eos_token_id: int):
    if eos_token_id == 2:
        return _input_ids()
    return np.array(
        [
            [0, eos_token_id, 5, eos_token_id],
            [0, 4, eos_token_id, eos_token_id],
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


def _load_cached_transformers_clip_checkpoint(*, default_checkpoint_id: str):
    checkpoint_id = os.environ.get("DINOML_CLIP_CHECKPOINT_ID", default_checkpoint_id)
    transformers = _import_local_transformers()
    try:
        clip_model = transformers.CLIPModel.from_pretrained(checkpoint_id, local_files_only=True)
    except Exception as exc:
        pytest.skip(f"checkpoint model {checkpoint_id!r} not available in local cache: {exc}")
    clip_model.eval()
    return checkpoint_id, clip_model


def _cached_checkpoint_runtime_inputs(
    *,
    text_config,
    vision_config,
):
    seq_len = min(4, int(text_config.max_position_embeddings))
    eos_token_id = int(text_config.eos_token_id)
    vocab_size = int(text_config.vocab_size)

    token_ids = []
    candidate = 0
    while len(token_ids) < max(seq_len - 1, 0):
        if candidate != eos_token_id:
            token_ids.append(candidate)
        candidate += 1
    token_ids.append(vocab_size - 1 if eos_token_id == 2 else eos_token_id)

    pixel_values = np.linspace(
        -1.0,
        1.0,
        num=int(vision_config.num_channels) * int(vision_config.image_size) * int(vision_config.image_size),
        dtype=np.float32,
    ).reshape(1, int(vision_config.num_channels), int(vision_config.image_size), int(vision_config.image_size))

    return seq_len, {
        "input_ids": np.asarray([token_ids], dtype=np.int64),
        "attention_mask": np.ones((1, seq_len), dtype=np.bool_),
        "pixel_values": pixel_values,
    }


def _trace_cached_checkpoint_two_tower_spec(clip_model):
    text_config, vision_config = legacy_clip_configs_from_transformers_clip_config(clip_model.config)
    adapted_model = legacy_clip_model_from_transformers_clip_model(clip_model)
    seq_len, inputs = _cached_checkpoint_runtime_inputs(text_config=text_config, vision_config=vision_config)
    spec = dml.trace(
        adapted_model,
        inputs={
            "input_ids": dml.TensorSpec([1, seq_len], "int64"),
            "pixel_values": dml.TensorSpec(
                [1, int(vision_config.num_channels), int(vision_config.image_size), int(vision_config.image_size)],
                "float32",
            ),
            "attention_mask": dml.TensorSpec([1, seq_len], "bool"),
        },
        name="clip_model_two_tower_transformers_cached_checkpoint_runtime_smoke",
    )
    return text_config, vision_config, spec, inputs


def _cached_checkpoint_expected_outputs(clip_model, inputs):
    torch = pytest.importorskip("torch")
    with torch.inference_mode():
        expected = clip_model(
            input_ids=torch.from_numpy(inputs["input_ids"]),
            attention_mask=torch.from_numpy(inputs["attention_mask"]),
            pixel_values=torch.from_numpy(inputs["pixel_values"]),
        )
    return {
        "logits_per_image": expected.logits_per_image.detach().cpu().numpy().astype(np.float32),
        "logits_per_text": expected.logits_per_text.detach().cpu().numpy().astype(np.float32),
        "text_embeds": expected.text_embeds.detach().cpu().numpy().astype(np.float32),
        "image_embeds": expected.image_embeds.detach().cpu().numpy().astype(np.float32),
    }


def _reference_outputs(*, text_num_hidden_layers: int = 2, vision_num_hidden_layers: int = 2):
    torch = pytest.importorskip("torch")
    clip_model = _build_local_transformers_clip_model(
        text_num_hidden_layers=text_num_hidden_layers,
        vision_num_hidden_layers=vision_num_hidden_layers,
    )
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


def _build_local_transformers_clip_model(
    *,
    text_num_hidden_layers: int = 2,
    vision_num_hidden_layers: int = 2,
    eos_token_id: int = 2,
):
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
        eos_token_id=eos_token_id,
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
    return clip_model


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


def test_clip_model_transformers_adapter_matches_local_transformers_cpu_reference():
    torch = pytest.importorskip("torch")
    clip_model = _build_local_transformers_clip_model()
    text_config, vision_config = legacy_clip_configs_from_transformers_clip_config(clip_model.config)

    assert text_config == _text_config()
    assert vision_config == _vision_config()

    adapted_model = legacy_clip_model_from_transformers_clip_model(clip_model)
    spec = dml.trace(
        adapted_model,
        inputs={
            "input_ids": dml.TensorSpec([TEXT_BATCH, SEQ_LEN], "int64"),
            "pixel_values": dml.TensorSpec([IMAGE_BATCH, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE], "float32"),
            "attention_mask": dml.TensorSpec([TEXT_BATCH, SEQ_LEN], "bool"),
        },
        name="clip_model_two_tower_transformers_adapter",
    )
    actual = execute_cpu(
        spec,
        {
            "input_ids": _input_ids(),
            "pixel_values": _pixel_values(),
            "attention_mask": _attention_mask(),
        },
    )

    text_inputs = {
        "input_ids": torch.from_numpy(_input_ids()),
        "attention_mask": torch.from_numpy(_attention_mask()),
    }
    image_inputs = {
        "pixel_values": torch.from_numpy(_pixel_values()),
    }
    with torch.inference_mode():
        expected = clip_model(**text_inputs, **image_inputs)

    np.testing.assert_allclose(
        actual["logits_per_image"],
        expected.logits_per_image.detach().cpu().numpy().astype(np.float32),
        atol=1e-5,
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        actual["logits_per_text"],
        expected.logits_per_text.detach().cpu().numpy().astype(np.float32),
        atol=1e-5,
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        actual["text_embeds"],
        expected.text_embeds.detach().cpu().numpy().astype(np.float32),
        atol=1e-5,
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        actual["image_embeds"],
        expected.image_embeds.detach().cpu().numpy().astype(np.float32),
        atol=1e-5,
        rtol=1e-5,
    )


def test_clip_model_transformers_adapter_non_2_eos_matches_local_transformers_cpu_reference():
    torch = pytest.importorskip("torch")
    eos_token_id = 7
    clip_model = _build_local_transformers_clip_model(eos_token_id=eos_token_id)
    text_config, vision_config = legacy_clip_configs_from_transformers_clip_config(clip_model.config)

    assert text_config == _text_config(eos_token_id=eos_token_id)
    assert vision_config == _vision_config()

    adapted_model = legacy_clip_model_from_transformers_clip_model(clip_model)
    spec = dml.trace(
        adapted_model,
        inputs={
            "input_ids": dml.TensorSpec([TEXT_BATCH, SEQ_LEN], "int64"),
            "pixel_values": dml.TensorSpec([IMAGE_BATCH, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE], "float32"),
            "attention_mask": dml.TensorSpec([TEXT_BATCH, SEQ_LEN], "bool"),
        },
        name="clip_model_two_tower_transformers_adapter_non_2_eos",
    )
    inputs = {
        "input_ids": _input_ids_for_eos(eos_token_id),
        "pixel_values": _pixel_values(),
        "attention_mask": _attention_mask(),
    }
    actual = execute_cpu(spec, inputs)

    text_inputs = {
        "input_ids": torch.from_numpy(inputs["input_ids"]),
        "attention_mask": torch.from_numpy(inputs["attention_mask"]),
    }
    image_inputs = {
        "pixel_values": torch.from_numpy(inputs["pixel_values"]),
    }
    with torch.inference_mode():
        expected = clip_model(**text_inputs, **image_inputs)

    np.testing.assert_allclose(
        actual["logits_per_image"],
        expected.logits_per_image.detach().cpu().numpy().astype(np.float32),
        atol=1e-5,
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        actual["logits_per_text"],
        expected.logits_per_text.detach().cpu().numpy().astype(np.float32),
        atol=1e-5,
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        actual["text_embeds"],
        expected.text_embeds.detach().cpu().numpy().astype(np.float32),
        atol=1e-5,
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        actual["image_embeds"],
        expected.image_embeds.detach().cpu().numpy().astype(np.float32),
        atol=1e-5,
        rtol=1e-5,
    )


def test_clip_model_transformers_state_dict_adapter_rejects_missing_required_weight():
    clip_model = _build_local_transformers_clip_model()
    text_config, vision_config = legacy_clip_configs_from_transformers_clip_config(clip_model.config)
    state_dict = dict(clip_model.state_dict())
    state_dict.pop("text_model.embeddings.token_embedding.weight")

    with pytest.raises(KeyError, match="text_model.embeddings.token_embedding.weight"):
        legacy_clip_weights_from_transformers_state_dict(state_dict, text_config, vision_config)


def test_clip_model_transformers_config_adapter_rejects_non_quick_gelu():
    transformers = _import_local_transformers()
    clip_config = transformers.CLIPConfig(
        text_config=transformers.CLIPTextConfig(
            vocab_size=VOCAB_SIZE,
            hidden_size=TEXT_HIDDEN,
            intermediate_size=TEXT_INTERMEDIATE,
            projection_dim=PROJECTION,
            num_attention_heads=NUM_HEADS,
            num_hidden_layers=1,
            max_position_embeddings=MAX_POSITION_EMBEDDINGS,
            hidden_act="gelu",
            attention_dropout=0.0,
            layer_norm_eps=EPS,
            bos_token_id=0,
            eos_token_id=2,
            pad_token_id=1,
        ).to_dict(),
        vision_config=transformers.CLIPVisionConfig(
            hidden_size=VISION_HIDDEN,
            intermediate_size=VISION_INTERMEDIATE,
            projection_dim=PROJECTION,
            num_attention_heads=NUM_HEADS,
            num_hidden_layers=1,
            image_size=IMAGE_SIZE,
            patch_size=PATCH_SIZE,
            num_channels=NUM_CHANNELS,
            hidden_act="quick_gelu",
            attention_dropout=0.0,
            layer_norm_eps=EPS,
        ).to_dict(),
        projection_dim=PROJECTION,
    )

    with pytest.raises(ValueError, match="hidden_act='quick_gelu'"):
        legacy_clip_configs_from_transformers_clip_config(clip_config)


def test_clip_model_transformers_checkpoint_adapter_state_smoke_local_cache_only(tmp_path):
    if os.environ.get("DINOML_RUN_CLIP_CHECKPOINT_ADAPTER_STATE_SMOKE") != "1":
        pytest.skip(
            "set DINOML_RUN_CLIP_CHECKPOINT_ADAPTER_STATE_SMOKE=1 to validate a cached Transformers CLIP checkpoint state import plus trace/admission smoke"
        )
    _, clip_model = _load_cached_transformers_clip_checkpoint(default_checkpoint_id="openai/clip-vit-large-patch14")

    text_config, vision_config = legacy_clip_configs_from_transformers_clip_config(clip_model.config)
    weights = legacy_clip_weights_from_transformers_state_dict(
        clip_model.state_dict(),
        text_config,
        vision_config,
    )
    adapted_model = legacy_clip_model_from_transformers_clip_model(clip_model)

    seq_len = min(4, text_config.max_position_embeddings)
    spec = dml.trace(
        adapted_model,
        inputs={
            "input_ids": dml.TensorSpec([1, seq_len], "int64"),
            "pixel_values": dml.TensorSpec(
                [1, vision_config.num_channels, vision_config.image_size, vision_config.image_size],
                "float32",
            ),
            "attention_mask": dml.TensorSpec([1, seq_len], "bool"),
        },
        name="clip_model_two_tower_transformers_cached_checkpoint_adapter_smoke",
    )
    lowered, _ = PassManager().run(spec.ir)
    validate_ir(lowered)
    manifest = build_kernel_manifest(lowered, {"name": "cuda", "arch": "sm_86", "no_tf32": True})
    codegen_plan = create_codegen_plan(manifest, tmp_path / "cache")

    assert text_config.hidden_size == int(clip_model.config.text_config.hidden_size)
    assert text_config.projection_dim == int(clip_model.config.projection_dim)
    assert vision_config.hidden_size == int(clip_model.config.vision_config.hidden_size)
    assert vision_config.num_hidden_layers == int(clip_model.config.vision_config.num_hidden_layers)
    assert vision_config.projection_dim == int(clip_model.config.projection_dim)
    assert weights["text_projection.weight"].shape == (text_config.projection_dim, text_config.hidden_size)
    assert weights["vision_model.embeddings.patch_embedding.weight"].shape == (
        vision_config.hidden_size,
        vision_config.num_channels,
        vision_config.patch_size,
        vision_config.patch_size,
    )
    assert spec.ir["outputs"][0]["shape"] == [1, 1]
    assert spec.ir["outputs"][1]["shape"] == [1, 1]
    assert spec.ir["outputs"][2]["shape"] == [1, text_config.projection_dim]
    assert spec.ir["outputs"][3]["shape"] == [1, vision_config.projection_dim]

    required_ops = {entry["op"] for entry in manifest["required_kernels"]}
    assert {"conv2d_bias", "gemm_rcr_bias", "gemm_rcr_bias_fast_gelu", "bmm_rcr", "bmm_rrr", "gemm_rcr"} <= required_ops
    assert {entry["name"] for entry in codegen_plan.external_support_libraries} == {
        "cutlass_bmm",
        "cutlass_conv",
        "cutlass_gemm",
    }


@pytest.mark.filterwarnings("ignore:overflow encountered in exp:RuntimeWarning")
def test_clip_model_transformers_checkpoint_runtime_smoke_local_cache_only():
    if os.environ.get("DINOML_RUN_CLIP_CHECKPOINT_RUNTIME_SMOKE") != "1":
        pytest.skip(
            "set DINOML_RUN_CLIP_CHECKPOINT_RUNTIME_SMOKE=1 to validate cached openai/clip-vit-base-patch32 CPU runtime parity against local Transformers"
        )
    _, clip_model = _load_cached_transformers_clip_checkpoint(default_checkpoint_id="openai/clip-vit-base-patch32")
    text_config, vision_config, spec, inputs = _trace_cached_checkpoint_two_tower_spec(clip_model)

    with np.errstate(over="ignore"):
        actual = execute_cpu(spec, inputs)
    expected = _cached_checkpoint_expected_outputs(clip_model, inputs)

    assert spec.ir["outputs"][0]["shape"] == [1, 1]
    assert spec.ir["outputs"][1]["shape"] == [1, 1]
    assert spec.ir["outputs"][2]["shape"] == [1, text_config.projection_dim]
    assert spec.ir["outputs"][3]["shape"] == [1, vision_config.projection_dim]

    np.testing.assert_allclose(
        actual["logits_per_image"],
        expected["logits_per_image"],
        atol=2e-5,
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        actual["logits_per_text"],
        expected["logits_per_text"],
        atol=2e-5,
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        actual["text_embeds"],
        expected["text_embeds"],
        atol=2e-5,
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        actual["image_embeds"],
        expected["image_embeds"],
        atol=2e-5,
        rtol=1e-5,
    )


@pytest.mark.filterwarnings("ignore:overflow encountered in exp:RuntimeWarning")
def test_clip_model_transformers_checkpoint_compiled_cpu_smoke_local_cache_only(tmp_path, monkeypatch):
    if os.environ.get("DINOML_RUN_CLIP_CHECKPOINT_COMPILED_CPU_SMOKE") != "1":
        pytest.skip(
            "set DINOML_RUN_CLIP_CHECKPOINT_COMPILED_CPU_SMOKE=1 to validate cached openai/clip-vit-base-patch32 compiled CPU parity against local Transformers"
        )
    monkeypatch.setenv("HF_HOME", "/workspace/.cache/huggingface")
    monkeypatch.setenv("DINOML_CACHE_DIR", str(tmp_path / "cache"))

    _, clip_model = _load_cached_transformers_clip_checkpoint(default_checkpoint_id="openai/clip-vit-base-patch32")
    text_config, vision_config, spec, inputs = _trace_cached_checkpoint_two_tower_spec(clip_model)
    expected = _cached_checkpoint_expected_outputs(clip_model, inputs)

    artifact = dml.compile(
        spec,
        dml.Target("cpu"),
        tmp_path / "clip_model_two_tower_transformers_cached_checkpoint_cpu.dinoml",
    )

    generated = (artifact.path / "debug" / "generated_src" / "module.cpp").read_text(encoding="utf-8")
    assert "static int conv2d_bias_" in generated
    assert "static int gemm_rcr_bias_fast_gelu_" in generated
    assert "static int bmm_rcr_" in generated
    assert "static int bmm_rrr_" in generated

    module = runtime.load(artifact.path)
    session = module.create_session()
    try:
        actual = session.run_numpy(inputs)
    finally:
        session.close()
        module.close()

    assert spec.ir["outputs"][0]["shape"] == [1, 1]
    assert spec.ir["outputs"][1]["shape"] == [1, 1]
    assert spec.ir["outputs"][2]["shape"] == [1, text_config.projection_dim]
    assert spec.ir["outputs"][3]["shape"] == [1, vision_config.projection_dim]

    np.testing.assert_allclose(actual["logits_per_image"], expected["logits_per_image"], atol=3e-5, rtol=1e-5)
    np.testing.assert_allclose(actual["logits_per_text"], expected["logits_per_text"], atol=3e-5, rtol=1e-5)
    np.testing.assert_allclose(actual["text_embeds"], expected["text_embeds"], atol=3e-5, rtol=1e-5)
    np.testing.assert_allclose(actual["image_embeds"], expected["image_embeds"], atol=3e-5, rtol=1e-5)


@pytest.mark.filterwarnings("ignore:overflow encountered in exp:RuntimeWarning")
@pytest.mark.skipif(shutil.which("nvcc") is None, reason="nvcc is required")
def test_clip_model_transformers_checkpoint_compiled_cuda_smoke_local_cache_only(
    tmp_path,
    monkeypatch,
    use_shared_dinoml_cuda_cache,
):
    if os.environ.get("DINOML_RUN_CLIP_CHECKPOINT_COMPILED_CUDA_SMOKE") != "1":
        pytest.skip(
            "set DINOML_RUN_CLIP_CHECKPOINT_COMPILED_CUDA_SMOKE=1 to validate cached openai/clip-vit-base-patch32 CUDA compile/load/run tractability and current drift bounds against local Transformers"
        )
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device is required")

    monkeypatch.setenv("HF_HOME", "/workspace/.cache/huggingface")

    _, clip_model = _load_cached_transformers_clip_checkpoint(default_checkpoint_id="openai/clip-vit-base-patch32")
    text_config, vision_config, spec, inputs = _trace_cached_checkpoint_two_tower_spec(clip_model)

    artifact = dml.compile(
        spec,
        dml.Target("cuda", arch="sm_86", no_tf32=True),
        tmp_path / "clip_model_two_tower_transformers_cached_checkpoint_cuda.dinoml",
    )

    generated = (artifact.path / "debug" / "generated_src" / "module.cu").read_text(encoding="utf-8")
    assert "dinoml_cutlass_conv" in generated
    assert "dinoml_cutlass_gemm" in generated
    assert "dinoml_cutlass_bmm" in generated

    kernel_manifest = json.loads((artifact.path / "kernel_manifest.json").read_text(encoding="utf-8"))
    required_ops = {entry["op"] for entry in kernel_manifest["required_kernels"]}
    assert {"conv2d_bias", "gemm_rcr_bias", "gemm_rcr_bias_fast_gelu", "bmm_rcr", "bmm_rrr", "gemm_rcr"} <= required_ops

    module = runtime.load(artifact.path)
    session = module.create_session()
    try:
        actual = session.run_numpy(inputs)
    finally:
        session.close()
        module.close()

    expected = _cached_checkpoint_expected_outputs(clip_model, inputs)

    assert spec.ir["outputs"][0]["shape"] == [1, 1]
    assert spec.ir["outputs"][1]["shape"] == [1, 1]
    assert spec.ir["outputs"][2]["shape"] == [1, text_config.projection_dim]
    assert spec.ir["outputs"][3]["shape"] == [1, vision_config.projection_dim]

    for name in ("logits_per_image", "logits_per_text", "text_embeds", "image_embeds"):
        assert np.isfinite(actual[name]).all()

    assert np.max(np.abs(actual["logits_per_image"] - expected["logits_per_image"])) < 0.9
    assert np.max(np.abs(actual["logits_per_text"] - expected["logits_per_text"])) < 0.9
    assert np.max(np.abs(actual["text_embeds"] - expected["text_embeds"])) < 0.05
    assert np.max(np.abs(actual["image_embeds"] - expected["image_embeds"])) < 0.1


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


def test_clip_model_two_tower_zero_text_zero_vision_cpu_artifact_matches_local_transformers(tmp_path, monkeypatch):
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
    artifact = dml.compile(spec, dml.Target("cpu"), tmp_path / "clip_model_two_tower_zero_cpu.dinoml")

    generated = (artifact.path / "debug" / "generated_src" / "module.cpp").read_text(encoding="utf-8")
    assert "static int conv2d_bias_" in generated

    module = runtime.load(artifact.path)
    session = module.create_session()
    try:
        actual = session.run_numpy(
            {
                "input_ids": _input_ids(),
                "pixel_values": _pixel_values(),
                "attention_mask": _attention_mask(),
            }
        )
    finally:
        session.close()
        module.close()

    expected = _reference_outputs(text_num_hidden_layers=0, vision_num_hidden_layers=0)
    np.testing.assert_allclose(actual["logits_per_image"], expected["logits_per_image"], atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(actual["logits_per_text"], expected["logits_per_text"], atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(actual["text_embeds"], expected["text_embeds"], atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(actual["image_embeds"], expected["image_embeds"], atol=1e-5, rtol=1e-5)


def test_clip_model_two_tower_cpu_artifact_matches_local_transformers(tmp_path, monkeypatch):
    monkeypatch.setenv("DINOML_CACHE_DIR", str(tmp_path / "cache"))
    spec = _trace_model()
    artifact = dml.compile(spec, dml.Target("cpu"), tmp_path / "clip_model_two_tower_cpu.dinoml")

    generated = (artifact.path / "debug" / "generated_src" / "module.cpp").read_text(encoding="utf-8")
    assert "static int conv2d_bias_" in generated
    assert "static int gemm_rcr_bias_fast_gelu_" in generated
    assert "static int bmm_rcr_" in generated
    assert "static int bmm_rrr_" in generated

    module = runtime.load(artifact.path)
    session = module.create_session()
    try:
        actual = session.run_numpy(
            {
                "input_ids": _input_ids(),
                "pixel_values": _pixel_values(),
                "attention_mask": _attention_mask(),
            }
        )
    finally:
        session.close()
        module.close()

    expected = _reference_outputs()
    np.testing.assert_allclose(actual["logits_per_image"], expected["logits_per_image"], atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(actual["logits_per_text"], expected["logits_per_text"], atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(actual["text_embeds"], expected["text_embeds"], atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(actual["image_embeds"], expected["image_embeds"], atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(shutil.which("nvcc") is None, reason="nvcc is required")
@pytest.mark.skipif(
    os.environ.get("DINOML_RUN_EXPENSIVE_CUDA_CLIP_MODEL") != "1",
    reason="set DINOML_RUN_EXPENSIVE_CUDA_CLIP_MODEL=1 to run the expensive CUDA CLIP full-model smoke",
)
def test_clip_model_two_tower_generated_cuda_runtime_matches_transformers(tmp_path, use_shared_dinoml_cuda_cache):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device is required")

    expected = _reference_outputs()
    full_spec = _trace_model()
    full_artifact = dml.compile(
        full_spec,
        dml.Target("cuda", arch="sm_86", no_tf32=True),
        tmp_path / "clip_model_two_tower_cuda.dinoml",
    )
    kernel_manifest = json.loads((full_artifact.path / "kernel_manifest.json").read_text(encoding="utf-8"))
    conv_entry = next(entry for entry in kernel_manifest["required_kernels"] if entry["op"] == "conv2d_bias")
    assert conv_entry["cutlass_conv_plan"]["status"] == "bounded_runtime"
    assert conv_entry["cutlass_conv_plan"]["selected_candidate"]["opclass"] == "simt"

    module = runtime.load(full_artifact.path)
    session = module.create_session()
    try:
        full_actual = session.run_numpy(
            {
                "input_ids": _input_ids(),
                "pixel_values": _pixel_values(),
                "attention_mask": _attention_mask(),
            }
        )
    finally:
        session.close()
        module.close()

    for name in ("text_embeds", "image_embeds", "logits_per_text", "logits_per_image"):
        np.testing.assert_allclose(full_actual[name], expected[name], atol=5.0e-4, rtol=5.0e-4)


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
    assert conv_entries[0]["cutlass_conv_plan"]["status"] == "bounded_runtime"
    assert conv_entries[0]["cutlass_conv_plan"]["selected_candidate"]["opclass"] == "simt"
    assert all(entry["kernel_library"] == "cutlass_gemm" for entry in provider_entries if entry["op"] in {"gemm_rcr_bias", "gemm_rcr_bias_fast_gelu", "gemm_rcr"})
    assert all(entry["kernel_library"] == "cutlass_bmm" for entry in provider_entries if entry["op"] in {"bmm_rcr", "bmm_rrr"})
    assert model_entries
    assert all(entry["kernel_library"] == "model" for entry in model_entries)


def test_clip_model_codegen_plan_keeps_conv_runtime_artifact_visible(tmp_path):
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

    assert conv_manifest_entry["cutlass_conv_plan"]["status"] == "bounded_runtime"
    assert conv_support_lib["kernel_symbols"] == [conv_manifest_entry["kernel_symbol"]]
    assert conv_manifest_entry["profiler_symbol"] in conv_support_lib["profiler_symbols"]
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
