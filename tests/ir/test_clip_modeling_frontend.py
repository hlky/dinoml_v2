from __future__ import annotations

from collections import Counter
from dataclasses import replace
from types import SimpleNamespace

import numpy as np

import dinoml as dml
from dinoml.models.clip import (
    CLIPConfig,
    CLIPModel,
    CLIPTextConfig,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPVisionConfig,
    clip_model_from_transformers_clip_model,
)
from dinoml.reference import reference_numpy


BATCH = 2
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


def build_text_config() -> CLIPTextConfig:
    return CLIPTextConfig(
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=MAX_POSITION_EMBEDDINGS,
        hidden_size=TEXT_HIDDEN,
        intermediate_size=TEXT_INTERMEDIATE,
        num_attention_heads=NUM_HEADS,
        num_hidden_layers=1,
        projection_dim=PROJECTION,
        layer_norm_eps=EPS,
        eos_token_id=2,
    )


def build_vision_config() -> CLIPVisionConfig:
    return CLIPVisionConfig(
        hidden_size=VISION_HIDDEN,
        intermediate_size=VISION_INTERMEDIATE,
        num_attention_heads=NUM_HEADS,
        num_hidden_layers=1,
        projection_dim=PROJECTION,
        image_size=IMAGE_SIZE,
        patch_size=PATCH_SIZE,
        num_channels=NUM_CHANNELS,
        layer_norm_eps=EPS,
    )


def build_weights() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(2089)

    def _normal(shape, scale):
        return (rng.standard_normal(shape).astype(np.float32) / scale).astype(np.float32)

    text_config = build_text_config()
    vision_config = build_vision_config()
    return {
        "text_model.embeddings.token_embedding.weight": _normal((text_config.vocab_size, text_config.hidden_size), 3.5),
        "text_model.embeddings.position_embedding.weight": _normal(
            (text_config.max_position_embeddings, text_config.hidden_size), 4.0
        ),
        "text_model.encoder.layers.0.self_attn.q_proj.weight": _normal((text_config.hidden_size, text_config.hidden_size), 5.0),
        "text_model.encoder.layers.0.self_attn.q_proj.bias": _normal((text_config.hidden_size,), 7.0),
        "text_model.encoder.layers.0.self_attn.k_proj.weight": _normal((text_config.hidden_size, text_config.hidden_size), 5.0),
        "text_model.encoder.layers.0.self_attn.k_proj.bias": _normal((text_config.hidden_size,), 7.0),
        "text_model.encoder.layers.0.self_attn.v_proj.weight": _normal((text_config.hidden_size, text_config.hidden_size), 5.0),
        "text_model.encoder.layers.0.self_attn.v_proj.bias": _normal((text_config.hidden_size,), 7.0),
        "text_model.encoder.layers.0.self_attn.out_proj.weight": _normal((text_config.hidden_size, text_config.hidden_size), 5.0),
        "text_model.encoder.layers.0.self_attn.out_proj.bias": _normal((text_config.hidden_size,), 7.0),
        "text_model.encoder.layers.0.layer_norm1.weight": _normal((text_config.hidden_size,), 4.0),
        "text_model.encoder.layers.0.layer_norm1.bias": _normal((text_config.hidden_size,), 6.0),
        "text_model.encoder.layers.0.mlp.fc1.weight": _normal((text_config.intermediate_size, text_config.hidden_size), 4.5),
        "text_model.encoder.layers.0.mlp.fc1.bias": _normal((text_config.intermediate_size,), 6.5),
        "text_model.encoder.layers.0.mlp.fc2.weight": _normal((text_config.hidden_size, text_config.intermediate_size), 4.5),
        "text_model.encoder.layers.0.mlp.fc2.bias": _normal((text_config.hidden_size,), 6.5),
        "text_model.encoder.layers.0.layer_norm2.weight": _normal((text_config.hidden_size,), 4.0),
        "text_model.encoder.layers.0.layer_norm2.bias": _normal((text_config.hidden_size,), 6.0),
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
        "vision_model.encoder.layers.0.layer_norm1.weight": _normal((vision_config.hidden_size,), 4.0),
        "vision_model.encoder.layers.0.layer_norm1.bias": _normal((vision_config.hidden_size,), 6.0),
        "vision_model.encoder.layers.0.self_attn.q_proj.weight": _normal((vision_config.hidden_size, vision_config.hidden_size), 5.0),
        "vision_model.encoder.layers.0.self_attn.q_proj.bias": _normal((vision_config.hidden_size,), 7.0),
        "vision_model.encoder.layers.0.self_attn.k_proj.weight": _normal((vision_config.hidden_size, vision_config.hidden_size), 5.0),
        "vision_model.encoder.layers.0.self_attn.k_proj.bias": _normal((vision_config.hidden_size,), 7.0),
        "vision_model.encoder.layers.0.self_attn.v_proj.weight": _normal((vision_config.hidden_size, vision_config.hidden_size), 5.0),
        "vision_model.encoder.layers.0.self_attn.v_proj.bias": _normal((vision_config.hidden_size,), 7.0),
        "vision_model.encoder.layers.0.self_attn.out_proj.weight": _normal((vision_config.hidden_size, vision_config.hidden_size), 5.0),
        "vision_model.encoder.layers.0.self_attn.out_proj.bias": _normal((vision_config.hidden_size,), 7.0),
        "vision_model.encoder.layers.0.layer_norm2.weight": _normal((vision_config.hidden_size,), 4.0),
        "vision_model.encoder.layers.0.layer_norm2.bias": _normal((vision_config.hidden_size,), 6.0),
        "vision_model.encoder.layers.0.mlp.fc1.weight": _normal((vision_config.intermediate_size, vision_config.hidden_size), 4.0),
        "vision_model.encoder.layers.0.mlp.fc1.bias": _normal((vision_config.intermediate_size,), 6.0),
        "vision_model.encoder.layers.0.mlp.fc2.weight": _normal((vision_config.hidden_size, vision_config.intermediate_size), 4.0),
        "vision_model.encoder.layers.0.mlp.fc2.bias": _normal((vision_config.hidden_size,), 6.0),
        "visual_projection.weight": _normal((vision_config.projection_dim, vision_config.hidden_size), 4.0),
        "logit_scale": np.array(np.log(1.7), dtype=np.float32),
    }


WEIGHTS = build_weights()


def build_text_inputs() -> dict[str, np.ndarray]:
    return {
        "input_ids": np.array([[0, 5, 15, 1], [0, 15, 4, 1]], dtype=np.int64),
        "attention_mask": np.array([[True, True, True, False], [True, True, True, False]], dtype=np.bool_),
    }


def build_model_inputs() -> dict[str, np.ndarray]:
    pixel_values = np.linspace(
        -1.5,
        1.5,
        num=IMAGE_BATCH * NUM_CHANNELS * IMAGE_SIZE * IMAGE_SIZE,
        dtype=np.float32,
    ).reshape(IMAGE_BATCH, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
    return {
        **build_text_inputs(),
        "pixel_values": pixel_values,
    }


def test_clip_text_model_with_projection_uses_expected_trace_contract():
    model = CLIPTextModelWithProjection(build_text_config(), WEIGHTS)
    spec = dml.trace(
        model,
        inputs={
            "input_ids": dml.TensorSpec([BATCH, SEQ_LEN], "int64"),
            "attention_mask": dml.TensorSpec([BATCH, SEQ_LEN], "bool"),
        },
        name="clip_text_with_projection",
    )
    op_counts = Counter(node["op"] for node in spec.ir["nodes"])

    assert spec.ir["outputs"][0]["name"] == "text_features"
    assert spec.ir["outputs"][0]["shape"] == [BATCH, PROJECTION]
    assert op_counts["embedding"] == 2
    assert op_counts["gemm_rcr_bias"] == 3
    assert op_counts["gemm_rcr_bias_quick_gelu"] == 1
    assert op_counts["layer_norm"] == 2
    assert op_counts["add_layer_norm"] == 1

    outputs = reference_numpy(spec, build_text_inputs())
    assert outputs["text_features"].shape == (BATCH, PROJECTION)


def test_clip_text_model_without_projection_uses_expected_trace_contract():
    model = CLIPTextModel(build_text_config(), WEIGHTS)
    spec = dml.trace(
        model,
        inputs={
            "input_ids": dml.TensorSpec([BATCH, SEQ_LEN], "int64"),
            "attention_mask": dml.TensorSpec([BATCH, SEQ_LEN], "bool"),
        },
        name="clip_text_without_projection",
    )
    op_counts = Counter(node["op"] for node in spec.ir["nodes"])
    output_shapes = {output["name"]: output["shape"] for output in spec.ir["outputs"]}

    assert output_shapes == {
        "last_hidden_state": [BATCH, SEQ_LEN, TEXT_HIDDEN],
        "pooler_output": [BATCH, TEXT_HIDDEN],
    }
    assert op_counts["embedding"] == 2
    assert op_counts["gemm_rcr_bias"] == 3
    assert op_counts["gemm_rcr_bias_quick_gelu"] == 1
    assert op_counts["layer_norm"] == 2
    assert op_counts["add_layer_norm"] == 1


def test_clip_model_uses_expected_trace_contract():
    model = CLIPModel(CLIPConfig(build_text_config(), build_vision_config()), WEIGHTS)
    spec = dml.trace(
        model,
        inputs={
            "input_ids": dml.TensorSpec([BATCH, SEQ_LEN], "int64"),
            "pixel_values": dml.TensorSpec([IMAGE_BATCH, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE], "float32"),
            "attention_mask": dml.TensorSpec([BATCH, SEQ_LEN], "bool"),
        },
        name="clip_model",
    )
    op_counts = Counter(node["op"] for node in spec.ir["nodes"])
    output_shapes = {output["name"]: output["shape"] for output in spec.ir["outputs"]}

    assert output_shapes == {
        "logits_per_image": [IMAGE_BATCH, BATCH],
        "logits_per_text": [BATCH, IMAGE_BATCH],
        "text_embeds": [BATCH, PROJECTION],
        "image_embeds": [IMAGE_BATCH, PROJECTION],
    }
    assert op_counts["conv2d_bias"] == 1
    assert op_counts["embedding"] == 3
    assert op_counts["gemm_rcr_bias"] == 6
    assert op_counts["gemm_rcr_bias_quick_gelu"] == 2
    assert op_counts["layer_norm"] == 5
    assert op_counts["add_layer_norm"] == 2

    outputs = reference_numpy(spec, build_model_inputs())
    assert outputs["logits_per_image"].shape == (IMAGE_BATCH, BATCH)


def test_clip_text_model_can_opt_into_flash_attention_for_unpadded_fp16_trace():
    config = replace(
        build_text_config(),
        use_flash_attention=True,
        dtype="float16",
    )
    model = CLIPTextModelWithProjection(config, WEIGHTS)
    spec = dml.trace(
        model,
        inputs={
            "input_ids": dml.TensorSpec([BATCH, SEQ_LEN], "int64"),
        },
        name="clip_text_flash_attention_trace",
    )
    op_counts = Counter(node["op"] for node in spec.ir["nodes"])

    assert op_counts["flash_attention_qkv"] == config.num_hidden_layers
    assert op_counts["softmax"] == 0
    assert op_counts["bmm_rcr"] == 0
    assert op_counts["bmm_rrr"] == 0
    assert op_counts["dynamic_slice"] == 0
    assert [tensor["name"] for tensor in spec.ir["inputs"]] == ["input_ids"]


def test_clip_text_model_float16_config_builds_float16_parameters_directly():
    config = replace(build_text_config(), dtype="float16")
    model = CLIPTextModelWithProjection(config, WEIGHTS)
    _assert_float_parameters_are_float16(model)


def test_clip_full_model_float16_config_builds_float16_parameters_directly():
    config = CLIPConfig(
        replace(build_text_config(), dtype="float16"),
        replace(build_vision_config(), dtype="float16"),
    )
    model = CLIPModel(config, WEIGHTS)
    _assert_float_parameters_are_float16(model)


def test_transformers_clip_adapter_constructs_requested_dtype_directly():
    clip_model = SimpleNamespace(
        config=_fake_transformers_clip_config(),
        state_dict=lambda: WEIGHTS,
    )
    model = clip_model_from_transformers_clip_model(clip_model, dtype="float16")
    _assert_float_parameters_are_float16(model)


def _assert_float_parameters_are_float16(model):
    floating_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.dtype in {"float16", "float32"}
    ]
    assert floating_parameters
    assert {parameter.dtype for _, parameter in floating_parameters} == {"float16"}
    assert {
        np.asarray(parameter.value).dtype
        for _, parameter in floating_parameters
        if parameter.value is not None
    } == {np.dtype("float16")}


def _fake_transformers_clip_config():
    text_config = build_text_config()
    vision_config = build_vision_config()
    return SimpleNamespace(
        projection_dim=PROJECTION,
        text_config=SimpleNamespace(
            vocab_size=text_config.vocab_size,
            max_position_embeddings=text_config.max_position_embeddings,
            hidden_size=text_config.hidden_size,
            intermediate_size=text_config.intermediate_size,
            num_attention_heads=text_config.num_attention_heads,
            num_hidden_layers=text_config.num_hidden_layers,
            layer_norm_eps=text_config.layer_norm_eps,
            eos_token_id=text_config.eos_token_id,
            hidden_act="quick_gelu",
            attention_dropout=0.0,
        ),
        vision_config=SimpleNamespace(
            hidden_size=vision_config.hidden_size,
            intermediate_size=vision_config.intermediate_size,
            num_attention_heads=vision_config.num_attention_heads,
            num_hidden_layers=vision_config.num_hidden_layers,
            image_size=vision_config.image_size,
            patch_size=vision_config.patch_size,
            num_channels=vision_config.num_channels,
            layer_norm_eps=vision_config.layer_norm_eps,
            hidden_act="quick_gelu",
            attention_dropout=0.0,
        ),
    )
