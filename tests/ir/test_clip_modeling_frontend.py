from __future__ import annotations

from collections import Counter
from dataclasses import replace
from types import SimpleNamespace

import numpy as np

import dinoml as dml
from dinoml.reference import reference_numpy
from dinoml.models.clip import (
    LegacyCLIPModel,
    LegacyCLIPTextModelWithProjection,
    legacy_clip_model_from_transformers_clip_model,
)
from examples import clip_model_workflow, clip_text_workflow


def test_clip_text_model_uses_nn_wrappers_without_changing_trace_contract():
    spec = clip_text_workflow.build_spec()
    op_counts = Counter(node["op"] for node in spec.ir["nodes"])

    assert spec.ir["outputs"][0]["name"] == "text_features"
    assert spec.ir["outputs"][0]["shape"] == [clip_text_workflow.BATCH, clip_text_workflow.PROJECTION]
    assert op_counts["embedding"] == 2
    assert op_counts["gemm_rcr_bias"] == 3
    assert op_counts["gemm_rcr_bias_quick_gelu"] == 1
    assert op_counts["layer_norm"] == 2
    assert op_counts["add_layer_norm"] == 1

    outputs = reference_numpy(spec, clip_text_workflow.build_validation_inputs())
    assert outputs["text_features"].shape == (clip_text_workflow.BATCH, clip_text_workflow.PROJECTION)


def test_clip_model_uses_nn_wrappers_without_changing_trace_contract():
    spec = clip_model_workflow.build_spec()
    op_counts = Counter(node["op"] for node in spec.ir["nodes"])
    output_shapes = {output["name"]: output["shape"] for output in spec.ir["outputs"]}

    assert output_shapes == {
        "logits_per_image": [clip_model_workflow.IMAGE_BATCH, clip_model_workflow.TEXT_BATCH],
        "logits_per_text": [clip_model_workflow.TEXT_BATCH, clip_model_workflow.IMAGE_BATCH],
        "text_embeds": [clip_model_workflow.TEXT_BATCH, clip_model_workflow.PROJECTION],
        "image_embeds": [clip_model_workflow.IMAGE_BATCH, clip_model_workflow.PROJECTION],
    }
    assert op_counts["conv2d_bias"] == 1
    assert op_counts["embedding"] == 3
    assert op_counts["gemm_rcr_bias"] == 6
    assert op_counts["gemm_rcr_bias_quick_gelu"] == 2
    assert op_counts["layer_norm"] == 5
    assert op_counts["add_layer_norm"] == 2

    outputs = reference_numpy(spec, clip_model_workflow.build_validation_inputs())
    assert outputs["logits_per_image"].shape == (
        clip_model_workflow.IMAGE_BATCH,
        clip_model_workflow.TEXT_BATCH,
    )


def test_clip_text_model_can_opt_into_flash_attention_for_unpadded_fp16_trace():
    config = replace(
        clip_text_workflow.build_config(),
        use_flash_attention=True,
        assume_unpadded_attention_mask=True,
        dtype="float16",
    )
    model = LegacyCLIPTextModelWithProjection(config, clip_text_workflow.WEIGHTS)

    spec = dml.trace(
        model,
        inputs={
            "input_ids": dml.TensorSpec([clip_text_workflow.BATCH, clip_text_workflow.SEQ_LEN], "int64"),
            "attention_mask": dml.TensorSpec([clip_text_workflow.BATCH, clip_text_workflow.SEQ_LEN], "bool"),
        },
        name="clip_text_flash_attention_trace",
    )
    op_counts = Counter(node["op"] for node in spec.ir["nodes"])

    assert op_counts["flash_attention_qkv"] == config.num_hidden_layers
    assert op_counts["softmax"] == 0
    assert op_counts["bmm_rcr"] == 0
    assert op_counts["bmm_rrr"] == 0
    assert op_counts["dynamic_slice"] == 0


def test_clip_text_model_float16_config_builds_float16_parameters_directly():
    config = replace(clip_text_workflow.build_config(), dtype="float16")
    model = LegacyCLIPTextModelWithProjection(config, clip_text_workflow.WEIGHTS)

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


def test_clip_full_model_float16_config_builds_float16_parameters_directly():
    text_config = replace(clip_model_workflow.build_text_config(), dtype="float16")
    vision_config = replace(clip_model_workflow.build_vision_config(), dtype="float16")
    model = LegacyCLIPModel(text_config, vision_config, clip_model_workflow.WEIGHTS)

    _assert_float_parameters_are_float16(model)


def test_transformers_clip_adapter_constructs_requested_dtype_directly():
    clip_model = SimpleNamespace(
        config=_fake_transformers_clip_config(),
        state_dict=lambda: clip_model_workflow.WEIGHTS,
    )

    model = legacy_clip_model_from_transformers_clip_model(clip_model, dtype="float16")

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
    text_config = clip_model_workflow.build_text_config()
    vision_config = clip_model_workflow.build_vision_config()
    return SimpleNamespace(
        projection_dim=clip_model_workflow.PROJECTION,
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
        ),
    )
